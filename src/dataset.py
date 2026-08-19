"""
FAZ 3 (1/2) -- Veri Yukleme ve Artirma (PLAN Bolum 7.1-7.2).

metadata.csv + outputs/folds/fold_{i}.json okur, PyTorch DataLoader'lari uretir.

ONBELLEK
--------
959 PNG'yi her epoch'ta yeniden decode etmek yerine, bir kez RAM'e uint8 dizi
olarak aciyoruz: (959, 400, 400, 3) = ~439 MB. Colab'in 12 GB RAM'ine rahat
sigar, PNG decode maliyeti tamamen kalkar. Diske YAZILMIYOR -- 439 MB'lik bir
dosyanin repoya girmesini istemiyoruz, yeniden kurmasi zaten ~20-30 saniye.

Onbellek 400x400 TAM COZUNURLUKTE tutuluyor, 224'e kucultulmus halde degil.
Sebep: PLAN 7.2'nin RandomResizedCrop(224, scale=(0.85,1.0)) donusumu, kaynak
goruntu 224'ten buyuk oldugunda anlamli calisir. Onceden 224'e kucultursek
kirpma islemi upsampling yapmak zorunda kalir ve bulaniklik ekler.
Boylece hem PLAN 7.1 (egitim disi: 400 -> 224 dogrudan resize) hem PLAN 7.2
(egitim: 400'den kirp) harfiyen uygulanabiliyor.

ARTIRMA -- neden bunlar, neden flip YOK
----------------------------------------
Spektrogramda x ekseni ZAMAN, y ekseni FREKANS'tir. Bu yuzden:
  - zaman maskeleme  = DIKEY seritler (sutunlari sifirla)
  - frekans maskeleme = YATAY seritler (satirlari sifirla)

PLAN 7.2 yatay/dikey cevirmeyi acikca YASAKLIYOR: zaman eksenini ters cevirmek
sesi geri sarmak demektir, frekans eksenini ters cevirmek ise tiz ile pesi
takas etmek -- ikisi de fiziksel olarak anlamsizdir ve modele yanlis bir
degismezlik ogretir.

Maskeleme NORMALIZASYONDAN SONRA uygulaniyor ve maskelenen bolge 0 yapiliyor.
Normalize edilmis uzayda 0, veri setinin ortalamasina karsilik gelir -- yani
"bilgi yok" demenin dogru yolu budur, siyah (=-mean/std) demek degil.

Kullanim (kendi kendine test + artirma onizlemesi):
    python src/dataset.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import config as cfg


# ---------------------------------------------------------------
# ONBELLEK
# ---------------------------------------------------------------
def resolve_path(row):
    """
    metadata.csv satirindan gercek dosya yolunu cozer.

    filepath sutunu repo koku'ne GORELI tutulur (orn.
    'synthetic_dataset/fence_cutting/x_spectrogram.png') -- boylece ayni CSV
    hem Windows'ta hem Colab'da calisir.

    Geriye donuk uyum: eski CSV'lerde mutlak Windows yolu olabilir. O durumda
    label + filename sutunlarindan yeniden kurariz; bu ikisi platformdan
    bagimsizdir.
    """
    p = Path(str(row.filepath))
    if not p.is_absolute():
        p = cfg.ROOT / p
    if p.exists():
        return p
    fallback = cfg.DATA_DIR / row.label / row.filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Spektrogram bulunamadi: {row.filename}\n"
        f"  denenen: {p}\n"
        f"  denenen: {fallback}\n"
        f"metadata.csv baska bir makinede uretilmis olabilir -- "
        f"'python src/metadata.py' ile yeniden uret."
    )


def build_cache(df, verbose=True):
    """
    Tum spektrogramlari tek bir uint8 dizisine ac: (N, 400, 400, 3).

    metadata.csv'deki SATIR SIRASI korunur -- fold JSON'larindaki indeksler
    dogrudan bu diziye isaret eder.
    """
    n = len(df)
    h, w = cfg.EXPECTED_PNG_SIZE
    cache = np.empty((n, h, w, 3), dtype=np.uint8)

    if verbose:
        print(f"  {n} PNG onbellege aliniyor ({n * h * w * 3 / 1e6:.0f} MB)...",
              flush=True)
    for i, row in enumerate(df.itertuples(index=False)):
        with Image.open(resolve_path(row)) as im:
            arr = np.asarray(im.convert("RGB"))
        if arr.shape != (h, w, 3):
            raise ValueError(f"beklenmeyen boyut {arr.shape}: {row.filename}")
        cache[i] = arr
    if verbose:
        print(f"  onbellek hazir: {cache.shape} {cache.dtype}", flush=True)
    return cache


# ---------------------------------------------------------------
# SPECAUGMENT TARZI MASKELEME (PLAN 7.2)
# ---------------------------------------------------------------
class SpecMasking(torch.nn.Module):
    """
    Zaman (dikey serit) ve frekans (yatay serit) maskelemesi.

    Normalize edilmis tensore uygulanir; maskelenen bolge 0 yapilir
    (= normalize uzayinda veri seti ortalamasi).

    max_frac: serit genisligi/yuksekligi, ilgili eksenin en fazla bu orani
              kadar olur (PLAN: <= %10)
    n_masks:  serit sayisi araligi (PLAN: 1-2)
    """

    def __init__(self, max_frac=0.10, n_masks=(1, 2), p=0.5):
        super().__init__()
        self.max_frac = max_frac
        self.n_min, self.n_max = n_masks
        self.p = p

    def _apply(self, img, axis):
        """axis=2 -> yukseklik (frekans, yatay serit); axis=3 -> genislik (zaman, dikey serit)"""
        size = img.shape[axis - 1]          # img: (C,H,W) -> axis 2->H(1), 3->W(2)
        max_len = max(1, int(size * self.max_frac))
        n = int(torch.randint(self.n_min, self.n_max + 1, (1,)).item())
        for _ in range(n):
            length = int(torch.randint(1, max_len + 1, (1,)).item())
            start = int(torch.randint(0, size - length + 1, (1,)).item())
            if axis == 2:
                img[:, start:start + length, :] = 0.0
            else:
                img[:, :, start:start + length] = 0.0
        return img

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            img = self._apply(img, axis=3)   # zaman -> dikey serit
        if torch.rand(1).item() < self.p:
            img = self._apply(img, axis=2)   # frekans -> yatay serit
        return img


# ---------------------------------------------------------------
# DONUSUMLER (PLAN 7.1 / 7.2)
# ---------------------------------------------------------------
def build_transforms(train):
    """train=True -> PLAN 7.2 artirmalari; train=False -> sadece PLAN 7.1."""
    norm = v2.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD)
    to_float = v2.ToDtype(torch.float32, scale=True)     # uint8 -> [0,1]

    if not train:
        return v2.Compose([
            v2.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE), antialias=True),
            to_float,
            norm,
        ])

    return v2.Compose([
        # NOT: yatay/dikey cevirme YOK -- spektrogramda fiziksel olarak anlamsiz
        v2.RandomResizedCrop(cfg.INPUT_SIZE, scale=cfg.CROP_SCALE,
                             ratio=(1.0, 1.0), antialias=True),
        v2.ColorJitter(brightness=cfg.JITTER, contrast=cfg.JITTER),
        to_float,
        norm,
        SpecMasking(max_frac=cfg.MASK_MAX_FRAC, n_masks=cfg.MASK_N,
                    p=cfg.MASK_P),
    ])


# ---------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------
class SpectrogramDataset(Dataset):
    """Onbellekteki uint8 diziden, verilen indeksler uzerinde calisan Dataset."""

    def __init__(self, cache, labels, indices, train):
        self.cache = cache
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = build_transforms(train)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        # HWC uint8 -> CHW uint8 tensor (kopya: transform yerinde degistirebilir)
        img = torch.from_numpy(self.cache[j]).permute(2, 0, 1).contiguous()
        return self.transform(img), int(self.labels[j])


# ---------------------------------------------------------------
# FOLD -> DATALOADER
# ---------------------------------------------------------------
def load_fold(fold_idx):
    """outputs/folds/fold_{i}.json oku."""
    path = cfg.FOLDS_DIR / f"fold_{fold_idx}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok -- once 'python src/splits.py' calistir (Faz 1).")
    return json.loads(path.read_text(encoding="utf-8"))


def make_loaders(fold_idx, df=None, cache=None, batch_size=cfg.BATCH_SIZE,
                 num_workers=0, verbose=True):
    """
    Bir katman icin (train, val, test) DataLoader uclusu.

    num_workers=0 varsayilan: goruntuler zaten RAM'de, disk I/O yok, dolayisiyla
    worker surecleri fayda saglamaz -- ustelik Windows'ta 439 MB'lik onbellegin
    her worker'a kopyalanmasi anlamina gelirdi.
    """
    if df is None:
        df = pd.read_csv(cfg.METADATA_CSV)
    if cache is None:
        cache = build_cache(df, verbose=verbose)

    fold = load_fold(fold_idx)
    labels = df.label_idx.to_numpy()

    # Faz 1'in sizinti garantilerinin hala gecerli oldugunu burada da dogrula:
    # fold JSON'u ile metadata.csv arasinda bir kayma olursa sessizce yanlis
    # egitim yapmaktansa durmak dogrudur.
    tr, va, te = fold["train_idx"], fold["val_idx"], fold["test_idx"]
    n_expected = len(tr) + len(va) + len(te) + len(fold["dropped_idx"])
    if n_expected != len(df):
        raise ValueError(
            f"fold_{fold_idx}.json {n_expected} indeks tasiyor ama metadata.csv "
            f"{len(df)} satir -- bolmeler guncel degil, 'python src/splits.py' calistir.")
    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
        raise ValueError(f"fold_{fold_idx}: bolmeler ortusuyor")
    if df.is_mixup.to_numpy()[te].any():
        raise ValueError(f"fold_{fold_idx}: test setinde MixUp ornegi var")
    if df.is_mixup.to_numpy()[va].any():
        raise ValueError(f"fold_{fold_idx}: dogrulama setinde MixUp ornegi var")

    def mk(indices, train, shuffle):
        return DataLoader(
            SpectrogramDataset(cache, labels, indices, train=train),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            drop_last=False, pin_memory=torch.cuda.is_available(),
        )

    return (mk(tr, train=True, shuffle=True),
            mk(va, train=False, shuffle=False),
            mk(te, train=False, shuffle=False)), fold, df, cache


# ---------------------------------------------------------------
# KENDI KENDINE TEST + ARTIRMA ONIZLEMESI
# ---------------------------------------------------------------
def preview_augmentation(cache, df, out_path, n=6, seed=cfg.SEED):
    """
    Artirmanin spektrograma ne yaptigini gozle gorulebilir kilar.

    Ust satir: orijinal (sadece resize). Alt satirlar: ayni goruntunun
    artirilmis varyantlari. Maskeleme seritleri ve kirpma burada gorulmeli.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    picks = [int(rng.choice(np.flatnonzero(df.label_idx.to_numpy() == c)))
             for c in range(cfg.N_CLASSES)]

    t_eval = build_transforms(train=False)
    t_train = build_transforms(train=True)
    mean = torch.tensor(cfg.NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.NORM_STD).view(3, 1, 1)

    def show(ax, t, title=None):
        img = (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=8)

    rows = 1 + (n - 1)
    fig, axes = plt.subplots(rows, cfg.N_CLASSES,
                             figsize=(2.2 * cfg.N_CLASSES, 2.2 * rows))
    for col, j in enumerate(picks):
        base = torch.from_numpy(cache[j]).permute(2, 0, 1).contiguous()
        show(axes[0, col], t_eval(base), cfg.IDX_TO_LABEL[col])
        for r in range(1, rows):
            show(axes[r, col], t_train(base))
    axes[0, 0].set_ylabel("orijinal", fontsize=8)
    for r in range(1, rows):
        axes[r, 0].set_ylabel(f"artirma {r}", fontsize=8)
    fig.suptitle("Egitim-zamani veri artirma (ust satir: artirmasiz)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def self_test():
    cfg.ensure_dirs()
    print("=" * 74)
    print("FAZ 3 (1/2) -- VERI YUKLEME TESTI")
    print("=" * 74)

    df = pd.read_csv(cfg.METADATA_CSV)
    cache = build_cache(df)

    print(f"\n[1] Katman 0 DataLoader'lari")
    print("-" * 74)
    (tr, va, te), fold, df, cache = make_loaders(0, df=df, cache=cache,
                                                 verbose=False)
    for name, dl in [("egitim", tr), ("dogrulama", va), ("test", te)]:
        print(f"  {name:<11} {len(dl.dataset):>4} ornek, {len(dl):>3} batch, "
              f"artirma={'VAR' if dl.dataset.train else 'yok'}")

    print(f"\n[2] Batch sekilleri ve deger araliklari")
    print("-" * 74)
    for name, dl in [("egitim", tr), ("test", te)]:
        x, y = next(iter(dl))
        print(f"  {name:<11} x={tuple(x.shape)} {x.dtype}  "
              f"[{x.min():.2f}, {x.max():.2f}]   y={tuple(y.shape)} {y.dtype}")
        assert x.shape[1:] == (3, cfg.INPUT_SIZE, cfg.INPUT_SIZE), \
            f"{name}: beklenmeyen girdi sekli {tuple(x.shape)}"
        assert y.min() >= 0 and y.max() < cfg.N_CLASSES, f"{name}: etiket araligi bozuk"

    print(f"\n[3] Artirma gercekten rastgele mi (ayni ornek, iki cagri)")
    print("-" * 74)
    ds = tr.dataset
    torch.manual_seed(1)
    a, _ = ds[0]
    torch.manual_seed(2)
    b, _ = ds[0]
    diff = (a - b).abs().mean().item()
    print(f"  ayni indeksin iki artirmasi arasindaki ort. fark: {diff:.4f}")
    assert diff > 1e-3, "artirma rastgelelik uretmiyor"
    print(f"  [x] Artirma rastgele")

    ds_eval = te.dataset
    c, _ = ds_eval[0]
    d, _ = ds_eval[0]
    assert torch.allclose(c, d), "degerlendirme donusumu deterministik degil"
    print(f"  [x] Degerlendirme donusumu deterministik (artirma yok)")

    print(f"\n[4] Maskeleme calisiyor mu")
    print("-" * 74)
    torch.manual_seed(0)
    m = SpecMasking(max_frac=cfg.MASK_MAX_FRAC, n_masks=cfg.MASK_N, p=1.0)
    probe = torch.ones(3, 224, 224)
    out = m(probe.clone())
    zero_frac = (out == 0).float().mean().item()
    print(f"  maskelenen piksel orani: {zero_frac:.1%} "
          f"(1-2 serit x <={cfg.MASK_MAX_FRAC:.0%} iki eksende)")
    assert 0 < zero_frac < 0.5, f"maskeleme orani makul degil: {zero_frac}"
    print(f"  [x] Zaman (dikey) ve frekans (yatay) seritleri uygulaniyor")

    print(f"\n[5] Sinif dagilimi fold JSON'u ile tutuyor mu")
    print("-" * 74)
    for name, dl, key in [("egitim", tr, "train"), ("dogrulama", va, "val"),
                          ("test", te, "test")]:
        idx = dl.dataset.indices
        got = {c: int((df.label_idx.to_numpy()[idx] == i).sum())
               for i, c in cfg.IDX_TO_LABEL.items()}
        exp = fold["class_counts"][key]
        ok = got == exp
        print(f"  {name:<11} {got}   {'OK' if ok else '*** ' + str(exp) + ' bekleniyordu ***'}")
        assert ok, f"{name}: sinif dagilimi fold JSON'u ile uyusmuyor"

    out_path = cfg.FIG_DIR / "augmentation_preview.png"
    preview_augmentation(cache, df, out_path)
    print(f"\n[6] Artirma onizlemesi kaydedildi: {out_path}")
    print(f"    Bunu GOZLE INCELE: seritler gorunuyor mu, kirpma olayi")
    print(f"    kadraj disinda birakmis mi, renkler bozulmus mu?")

    print(f"\n{'=' * 74}")
    print("FAZ 3 (1/2) TAMAM -- dataset.py hazir.")
    return 0


if __name__ == "__main__":
    sys.exit(self_test())
