"""
FAZ 2 -- Model Mimarisi (PLAN Bolum 6).

2D-CNN omurgasi + SK-Attention. Makale 4'ten (You ve ark., 2025, IEEE Sensors
Journal 25(22)) uyarlanmistir: makalenin girdisi (B,1,4000,10) zaman-uzay
matrisiydi, bizimki spektrogram goruntusu. Omurga ayni, sadece girdi kanali
ve boyutu degisti.

    Girdi (B,3,224,224)
      -> Conv(3->16,3x3) + ReLU + MaxPool2  -> (B,16,112,112)
      -> Conv(16->32,3x3) + ReLU + MaxPool2 -> (B,32,56,56)
      -> Conv(32->64,3x3) + ReLU + MaxPool2 -> (B,64,28,28)
      -> Dikkat modulu (SK / SE / CBAM / yok) -> (B,64,28,28)
      -> AdaptiveAvgPool2d(1) -> Flatten     -> (B,64)      [t-SNE ozelligi]
      -> Dropout(0.5) -> Linear(64->3)       -> (B,3)       [logit]

PLANDA BELIRTILMEYEN, BURADA VERILEN KARARLAR
---------------------------------------------
1) SK dallarindaki grup konvolusyonunun grup sayisi (G): PLAN "grup
   konvolusyonu" diyor ama G'yi vermiyor. Orijinal SKNet (Li ve ark., CVPR
   2019) G=32 kullaniyor; C=64 icin grup basina 2 kanal demek. G=32 secildi.
   Bu ayni zamanda parametre butcesi icin zorunlu: gruplamasiz 5x5 dali tek
   basina 64*64*25 = 102.400 parametre eder ve PLAN'in ~50-100k toplam hedefini
   tek basina asardi.

2) Omurgada BatchNorm YOK. PLAN 6.1 blogu acikca "Conv2d + ReLU + MaxPool2d"
   diye tarif ediyor, BN'den soz etmiyor; harfiyen uygulandi. (BN yalnizca SK
   modulunun kendi icinde var, PLAN 6.2 oyle tarif ediyor.) Faz 3'te egitim
   kararsiz olursa omurgaya BN eklemek ilk denenecek ablasyon olmali.

3) SE ve CBAM varyantlari da uygulandi. PLAN 6.3 bunlari "istege bagli" diye
   isaretlemis ama ablasyon tablosunda (8.5) satirlari var; maliyeti dusuk
   oldugu icin Faz 4'te tablo eksiksiz doldurulabilsin diye eklendi.

Kullanim (izole birim testi -- PLAN Bolum 12 adim 3):
    python src/model.py
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ---------------------------------------------------------------
# SK-ATTENTION (PLAN Bolum 6.2)
# ---------------------------------------------------------------
class SKAttention(nn.Module):
    """
    Selective Kernel Attention -- Split / Fuse / Select.

    PLAN 6.2 tablosu: M=2, kernel 3x3 & 5x5, r=16, L=32, C=64.
    d = max(C/r, L) = max(64/16, 32) = 32

    Kanal basina yumusak dikkat: her kanal icin dal agirliklari toplami 1
    (softmax garantisi). Bu, modelin her kanalda hangi alici alanin (3x3 mi
    5x5 mi) daha bilgilendirici oldugunu VERIYE GORE secmesini saglar --
    Makale 4'un SE/CBAM'e ustunlugunu acikladigi mekanizma bu.
    """

    def __init__(self, channels=64, kernels=cfg.SK_KERNELS, r=cfg.SK_R,
                 L=cfg.SK_L, groups=cfg.SK_GROUPS):
        super().__init__()
        self.channels = channels
        self.kernels = tuple(kernels)
        self.M = len(self.kernels)
        if channels % groups != 0:
            raise ValueError(
                f"kanal sayisi ({channels}) grup sayisina ({groups}) bolunemiyor")

        # --- SPLIT: M paralel dal, farkli kernel boyutlariyla ---
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k, stride=1,
                          padding=k // 2, groups=groups, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for k in self.kernels
        ])

        # --- FUSE: uzamsal bilgiyi ozetle, d boyutuna sikistir ---
        d = max(channels // r, L)
        self.d = d
        self.fuse = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )

        # --- SELECT: dal basina bir FC (PLAN'daki FC_A / FC_B) ---
        self.selectors = nn.ModuleList(
            [nn.Linear(d, channels) for _ in range(self.M)])

    def forward(self, x, return_weights=False):
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"beklenen {self.channels} kanal, gelen {C}")

        # SPLIT -> her dal (B, C, H, W)
        feats = torch.stack([br(x) for br in self.branches], dim=1)  # (B,M,C,H,W)

        # FUSE: dallari topla, global ortalama havuzla, sikistir
        u = feats.sum(dim=1)                       # (B,C,H,W)
        s = u.mean(dim=(2, 3))                     # (B,C)  global avg pool
        z = self.fuse(s)                           # (B,d)

        # SELECT: kanal basina dal agirliklari, dal ekseninde softmax
        logits = torch.stack([sel(z) for sel in self.selectors], dim=1)  # (B,M,C)
        weights = torch.softmax(logits, dim=1)     # her kanal icin toplam = 1

        v = (feats * weights[:, :, :, None, None]).sum(dim=1)  # (B,C,H,W)
        if return_weights:
            return v, weights
        return v


# ---------------------------------------------------------------
# KARSILASTIRMA ICIN DIGER DIKKAT MODULLERI (PLAN 6.3, istege bagli)
# ---------------------------------------------------------------
class SEAttention(nn.Module):
    """Squeeze-and-Excitation (Hu ve ark., 2018). Tek dal, kanal bazli kapi."""

    def __init__(self, channels=64, r=cfg.SK_R):
        super().__init__()
        d = max(channels // r, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))            # (B,C)
        return x * w[:, :, None, None]


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo ve ark., 2018): kanal + uzamsal."""

    def __init__(self, channels=64, r=cfg.SK_R, spatial_kernel=7):
        super().__init__()
        d = max(channels // r, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, channels, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x):
        # kanal dikkati: avg + max havuzlarinin paylasimli MLP'si
        ca = torch.sigmoid(self.mlp(x.mean(dim=(2, 3)))
                           + self.mlp(x.amax(dim=(2, 3))))
        x = x * ca[:, :, None, None]
        # uzamsal dikkat
        sa = torch.sigmoid(self.spatial(torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)))
        return x * sa


ATTENTIONS = {
    "none": None,
    "sk": SKAttention,
    "se": SEAttention,
    "cbam": CBAM,
}


# ---------------------------------------------------------------
# ANA MODEL (PLAN Bolum 6.1)
# ---------------------------------------------------------------
class DASNet(nn.Module):
    """
    PLAN 6.1 omurgasi + secilebilir dikkat modulu.

    attention='sk'   -> ana model
    attention='none' -> ablasyon baseline'i (duz 2D-CNN)
    """

    def __init__(self, n_classes=cfg.N_CLASSES, in_channels=cfg.IN_CHANNELS,
                 channels=cfg.CONV_CHANNELS, attention="sk",
                 dropout=cfg.DROPOUT):
        super().__init__()
        if attention not in ATTENTIONS:
            raise ValueError(f"bilinmeyen dikkat modulu: {attention!r} "
                             f"(secenekler: {sorted(ATTENTIONS)})")
        self.attention_name = attention

        # PLAN 6.1: Conv + ReLU + MaxPool x3. Omurgada BatchNorm YOK (bkz. modul basi).
        blocks, c_in = [], in_channels
        for c_out in channels:
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.out_channels = c_in

        att_cls = ATTENTIONS[attention]
        self.attention = att_cls(c_in) if att_cls is not None else nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c_in, n_classes)

    def forward_features(self, x):
        """
        AdaptiveAvgPool sonrasi 64 boyutlu ozellik vektoru.
        PLAN 8.4'teki t-SNE gorsellestirmesi bunu kullanacak.
        """
        x = self.features(x)
        x = self.attention(x)
        return torch.flatten(self.pool(x), 1)

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------
# IZOLE BIRIM TESTI (PLAN Bolum 12, adim 3)
# ---------------------------------------------------------------
def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def self_test():
    torch.manual_seed(cfg.SEED)
    line = "-" * 70
    print("=" * 70)
    print("FAZ 2 -- MODEL BIRIM TESTI")
    print("=" * 70)

    # === 1) SK modulu izole test (PLAN 12.3) ===
    print(f"\n[1] SK-Attention izole testi -- girdi (2, 64, 28, 28)")
    print(line)
    sk = SKAttention(channels=64).eval()
    x = torch.randn(2, 64, 28, 28)
    with torch.no_grad():
        y, w = sk(x, return_weights=True)

    print(f"  M (dal sayisi)     : {sk.M}   kernel: {sk.kernels}")
    print(f"  d = max(C/r, L)    : {sk.d}   (C=64, r={cfg.SK_R}, L={cfg.SK_L})")
    print(f"  grup konvolusyonu G: {cfg.SK_GROUPS}")
    print(f"  girdi  sekli       : {tuple(x.shape)}")
    print(f"  cikti  sekli       : {tuple(y.shape)}")
    print(f"  agirlik sekli      : {tuple(w.shape)}  (B, M, C)")

    _check(y.shape == x.shape,
           f"cikti sekli girdiden farkli: {tuple(y.shape)} != {tuple(x.shape)}")
    print(f"  [x] Cikti sekli girdiyle ayni")

    # Softmax kisiti: her ornek ve her kanal icin a_c + b_c = 1
    sums = w.sum(dim=1)                              # (B, C)
    max_err = (sums - 1.0).abs().max().item()
    print(f"  dal agirliklari toplami: min={sums.min():.6f} maks={sums.max():.6f} "
          f"(hedef 1.0, maks sapma {max_err:.2e})")
    _check(torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
           f"softmax kisiti saglanmiyor: a+b != 1 (maks sapma {max_err:.2e})")
    print(f"  [x] Softmax kisiti a_c + b_c = 1 saglandi (tum B x C icin)")

    _check((w >= 0).all() and (w <= 1).all(), "dal agirliklari [0,1] disinda")
    print(f"  [x] Tum agirliklar [0, 1] araliginda")

    # Agirliklar gercekten kanal basina degisiyor mu (sabit degil)?
    spread = w[:, 0, :].std().item()
    print(f"  a agirliginin kanallar arasi std: {spread:.4f}")
    _check(spread > 1e-4, "dal agirliklari kanallar arasinda degismiyor -- "
                          "secim mekanizmasi calismiyor olabilir")
    print(f"  [x] Agirliklar kanal basina farklilasiyor (dinamik secim aktif)")

    # === 2) Tam model, ileri gecis ===
    print(f"\n[2] Tam model ileri gecisi -- girdi (2, 3, 224, 224)")
    print(line)
    img = torch.randn(2, cfg.IN_CHANNELS, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
    for name in ["none", "sk", "se", "cbam"]:
        m = DASNet(attention=name).eval()
        with torch.no_grad():
            out = m(img)
            feat = m.forward_features(img)
        _check(out.shape == (2, cfg.N_CLASSES),
               f"[{name}] logit sekli {tuple(out.shape)}, (2,{cfg.N_CLASSES}) bekleniyordu")
        _check(feat.shape == (2, 64),
               f"[{name}] ozellik sekli {tuple(feat.shape)}, (2,64) bekleniyordu")
        _check(torch.isfinite(out).all(), f"[{name}] logitlerde NaN/Inf var")
        tag = "  <- ana model" if name == "sk" else (
            "  <- ablasyon baseline'i" if name == "none" else "")
        print(f"  {name:<5} logit={tuple(out.shape)} ozellik={tuple(feat.shape)} "
              f"parametre={count_parameters(m):>7,}{tag}")

    # === 3) Ara katman sekilleri (PLAN 6.1 blogu ile karsilastirma) ===
    print(f"\n[3] Omurga ara katman sekilleri")
    print(line)
    m = DASNet(attention="sk").eval()
    h = img
    print(f"  {'girdi':<28} {tuple(h.shape)}")
    with torch.no_grad():
        for i, layer in enumerate(m.features):
            h = layer(h)
            if isinstance(layer, nn.MaxPool2d):
                print(f"  {'blok ' + str(i // 3 + 1) + ' sonrasi':<28} {tuple(h.shape)}")
        expected = (2, 64, 28, 28)
        _check(tuple(h.shape) == expected,
               f"omurga cikti sekli {tuple(h.shape)}, {expected} bekleniyordu "
               f"(PLAN 6.1 blogu)")
        print(f"  {'-> SK modulune giren':<28} {tuple(h.shape)}   PLAN 6.1 ile uyumlu")

    # === 4) Parametre butcesi ===
    print(f"\n[4] Parametre sayisi")
    print(line)
    base = count_parameters(DASNet(attention="none"))
    skm = count_parameters(DASNet(attention="sk"))
    print(f"  duz 2D-CNN (SK'siz) : {base:>8,}")
    print(f"  2D-CNN + SK         : {skm:>8,}   (SK modulunun ektisi: +{skm - base:,})")
    print(f"  PLAN 6.1 beklentisi : ~50.000-100.000")
    if skm < 50_000:
        print(f"  NOT: {skm:,} parametre PLAN'in verdigi araligin ALTINDA.")
        print(f"       Sebep: SK dallarinda grup konvolusyonu (G={cfg.SK_GROUPS}).")
        print(f"       PLAN 6.1 kucuklugu 19 etkin ornek icin AVANTAJ sayiyor,")
        print(f"       dolayisiyla bu bir sorun degil -- ama rapora not dusulmeli.")

    # === 5) Geri yayilim calisiyor mu ===
    print(f"\n[5] Geri yayilim kontrolu")
    print(line)
    m = DASNet(attention="sk").train()
    out = m(img)
    loss = nn.CrossEntropyLoss()(out, torch.tensor([0, 2]))
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    _check(not missing, f"gradyan almayan/bozuk parametreler: {missing[:5]}")
    print(f"  kayip = {loss.item():.4f}")
    print(f"  [x] Tum {sum(1 for _ in m.parameters())} parametre tensoru sonlu gradyan aldi")

    print(f"\n{'=' * 70}")
    print("KABUL KRITERI (PLAN Bolum 12, adim 3)")
    print("-" * 70)
    print("  [x] SK modulune (2,64,28,28) verildi, cikti ayni sekilde dondu")
    print("  [x] Softmax kisiti a+b=1 dogrulandi")
    print("  [x] Tam model (B,3,224,224) -> (B,3) logit uretiyor")
    print("  [x] Baseline / SE / CBAM / SK varyantlarinin hepsi calisiyor")
    print("  [x] Geri yayilim tum parametrelere ulasiyor")
    print("\nFAZ 2 TAMAM.")
    return 0


if __name__ == "__main__":
    sys.exit(self_test())
