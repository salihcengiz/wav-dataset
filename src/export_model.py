"""
NIHAI MODELI DISA AKTAR -- gercek saha verisinde onceden-egitilmis (pretrained)
baslangic noktasi olarak kullanilmak uzere.

NEDEN AYRI BIR EGITIM GEREKIYOR
-------------------------------
Capraz dogrulama 4 ayri model uretir (fold_0..3). Bunlardan birini secmek
YANLIS olur:
  - En yuksek test skorlu olani secmek = test setine bakarak model secmek.
    Faz 1'den beri kacindigimiz seyin ta kendisi.
  - Rastgele birini secmek = elimizdeki verinin sadece ~%75'ini kullanmak.

Standart pratik: capraz dogrulama performansi TAHMIN eder, nihai model TUM
veriyle yeniden egitilir. Raporlanan skor (macro-F1 0.622 +- 0.166) capraz
dogrulamadan gelir ve gecerliligini korur; bu dosyanin urettigi model ise
DAGITIM/AKTARIM icin bir yapaydir.

VERILEN KARARLAR
----------------
E1) Egitim verisi: TUM 959 ornek. Test/dogrulama ayrilmadigi icin MixUp
    ornekleri de tamamen kullanilabilir (sizacak bir yer yok). Karsilastirma:
    en genis katman 503 ornek goruyordu, bu model 959 goruyor.

E2) Epoch sayisi: capraz dogrulamadaki "en iyi epoch" degerlerinin MEDYANI.
    Dogrulama seti olmadigi icin erken durdurma calistirilamaz; ne zaman
    duracagimizi CV soyluyor. (Paket 2 / SK: 15, 25, 42, 48 -> medyan 34)

E3) Bu modelin KENDI basina durust bir performans sayisi YOKTUR. Beklenen
    performansi capraz dogrulama tahminidir. Ayni veri uzerinde olculurse
    anlamsiz derecede yuksek cikar -- model o veriyi gormustur.

CIKTI
-----
outputs/pretrained/
    das_2dcnn_{attention}_v1.pt   kendi kendini tarif eden paket
    MODEL_CARD.md                 ne oldugu, nasil yuklenecegi, sinirlari

Kullanim:
    python src/export_model.py                     # SK, CV medyan epoch
    python src/export_model.py --attention none    # baseline
    python src/export_model.py --epochs 40         # epoch sayisini elle ver
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import datetime
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from dataset import SpectrogramDataset, build_cache
from model import DASNet, count_parameters
from train import get_device, set_deterministic, train_one_epoch

BUNDLE_FORMAT_VERSION = 1


def git_commit():
    """Hangi kod surumuyle uretildigi -- izlenebilirlik icin."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def read_cv_results(attention):
    """
    outputs/results/ altindaki katman gecmislerini oku.

    Donen: (cv_ozet_sozlugu | None, medyan_en_iyi_epoch | None)
    """
    histories = []
    for i in range(cfg.N_SPLITS):
        p = cfg.RESULTS_DIR / f"fold_{i}_{attention}_history.json"
        if p.exists():
            histories.append(json.loads(p.read_text(encoding="utf-8")))
    if not histories:
        return None, None

    f1 = np.array([h["test_macro_f1"] for h in histories])
    acc = np.array([h["test_acc"] for h in histories])
    best_epochs = [h["best_epoch"] for h in histories]

    summary = {
        "n_folds": len(histories),
        "macro_f1_mean": round(float(f1.mean()), 4),
        "macro_f1_std": round(float(f1.std()), 4),
        "accuracy_mean": round(float(acc.mean()), 4),
        "accuracy_std": round(float(acc.std()), 4),
        "per_fold_macro_f1": [round(float(x), 4) for x in f1],
        "per_fold_best_epoch": best_epochs,
        "monitor": histories[0].get("monitor"),
    }
    return summary, int(round(float(np.median(best_epochs))))


def train_on_all_data(attention, epochs, device, batch_size, verbose=True):
    """E1 + E2: tum veriyle, sabit epoch sayisiyla, sifirdan egit."""
    set_deterministic(cfg.SEED)

    df = pd.read_csv(cfg.METADATA_CSV)
    cache = build_cache(df, verbose=verbose)
    labels = df.label_idx.to_numpy()
    all_idx = np.arange(len(df))

    ds = SpectrogramDataset(cache, labels, all_idx, train=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    model = DASNet(attention=attention).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                                 weight_decay=cfg.WEIGHT_DECAY)

    if verbose:
        print(f"\n{'=' * 74}")
        print(f"NIHAI EGITIM -- tum veri  |  dikkat={attention}  |  cihaz={device}")
        print("=" * 74)
        print(f"  ornek {len(ds)} (tumu, MixUp dahil) | batch {batch_size} "
              f"| epoch {epochs} (sabit, erken durdurma YOK)")
        print(f"  parametre {count_parameters(model):,} | girdi "
              f"{cfg.INPUT_H}x{cfg.INPUT_W} | BatchNorm "
              f"{'acik' if cfg.BACKBONE_BATCHNORM else 'kapali'} "
              f"| label smoothing {cfg.LABEL_SMOOTHING}")
        print(f"  {'-' * 40}")
        print(f"  {'ep':>4} {'loss':>10} {'acc':>9}")

    history = []
    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        history.append({"epoch": epoch, "loss": round(loss, 5),
                        "acc": round(acc, 5)})
        if verbose and (epoch <= 3 or epoch % 5 == 0 or epoch == epochs):
            print(f"  {epoch:>4} {loss:>10.4f} {acc:>9.3f}")

    return model, history, len(ds)


def build_bundle(model, attention, epochs, history, n_samples, cv_summary):
    """Kendi kendini tarif eden paket -- baska bir projede yeniden kurulabilsin."""
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,

        "architecture": {
            "class": "DASNet",
            "attention": attention,
            "in_channels": cfg.IN_CHANNELS,
            "conv_channels": list(cfg.CONV_CHANNELS),
            "batchnorm": cfg.BACKBONE_BATCHNORM,
            "dropout": cfg.DROPOUT,
            "n_classes": cfg.N_CLASSES,
            "sk": {"M": cfg.SK_M, "kernels": list(cfg.SK_KERNELS),
                   "r": cfg.SK_R, "L": cfg.SK_L, "groups": cfg.SK_GROUPS},
            "n_parameters": count_parameters(model),
            "feature_dim": cfg.CONV_CHANNELS[-1],
        },

        "input": {
            "height": cfg.INPUT_H, "width": cfg.INPUT_W,
            "layout": "dikey=frekans, yatay=zaman",
            "source_png": list(cfg.EXPECTED_PNG_SIZE),
            "norm_mean": list(cfg.NORM_MEAN), "norm_std": list(cfg.NORM_STD),
            "note": "uint8 RGB -> [0,1] -> Normalize(mean,std). Egitimde "
                    "RandomResizedCrop + jitter + zaman/frekans maskeleme; "
                    "degerlendirmede sadece Resize.",
        },

        "classes": list(cfg.CLASSES),
        "class_to_idx": dict(cfg.LABEL_TO_IDX),

        "training": {
            "data": "sentetik DAS spektrogramlari (synth_das_pipeline.py)",
            "n_samples": n_samples,
            "n_effective_recordings": cfg.N_EFFECTIVE_GROUPS,
            "epochs": epochs,
            "epochs_source": "capraz dogrulamadaki en iyi epoch'larin medyani",
            "optimizer": "adam", "lr": cfg.LR,
            "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE,
            "label_smoothing": cfg.LABEL_SMOOTHING,
            "seed": cfg.SEED,
            "holdout": None,
            "loss_history": history,
        },

        "cv_performance": cv_summary,

        "caveats": [
            "Bu model TUM veriyle egitildi; kendi basina durust bir test skoru YOKTUR.",
            "Beklenen performans capraz dogrulama tahminidir (cv_performance).",
            "Egitim verisi ~959 spektrogram ama yalnizca 19 BAGIMSIZ kayittan turetilmistir.",
            "Sentetik veri gercek DAS verisinin yerini tutmaz; saha verisiyle "
            "fine-tuning sarttir.",
            "'chain_link_climbing' sinifi akustik olarak tutarli bir kume degildir; "
            "orneklerinin ~%38'i 'metal_bending' ile karistirilmaktadir.",
            "'Olay yok / normal' sinifi YOKTUR -- model 'hangi tehdit' sorusunu "
            "cevaplar, 'tehdit var mi' sorusunu cevaplayamaz.",
        ],

        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


MODEL_CARD = """# DAS 2D-CNN + {attention_upper} — Önceden Eğitilmiş Model

**Dosya:** `{filename}`
**Üretim tarihi:** {created}
**Kod sürümü:** `{commit}`

---

## Bu nedir?

Sentetik DAS spektrogramlarıyla eğitilmiş, üç çit-ihlali olayını sınıflandıran
bir 2D-CNN. Mimari You ve ark. (2025, IEEE Sensors Journal 25(22), 41320–41328)
makalesinden uyarlanmıştır.

**Amacı:** gerçek saha verisiyle çalışmaya başlarken **sıfırdan başlamamak.**
Omurga ağırlıkları, spektrogramlarda kenar/darbe/bant örüntülerini tanımayı
zaten öğrenmiş durumda.

| | |
|---|---|
| Parametre sayısı | {n_params:,} |
| Girdi | {h}×{w} (dikey=frekans, yatay=zaman), 3 kanal |
| Özellik boyutu | {feat} (sınıflandırıcı öncesi) |
| Sınıflar | {classes} |
| Eğitim örneği | {n_samples} |
| **Etkin bağımsız kayıt** | **{n_rec}** |

## Beklenen performans

**macro-F1 {f1_mean} ± {f1_std}** ({n_folds} katlı, kaynak-gruplu çapraz doğrulama)

Katman katman: {per_fold}

> ⚠️ **Bu modelin kendi başına dürüst bir test skoru YOKTUR.** Tüm veriyle
> eğitildi. Yukarıdaki sayı, aynı konfigürasyonun çapraz doğrulamadaki
> tahminidir. Bu modeli eğitildiği veri üzerinde ölçerseniz anlamsız derecede
> yüksek bir sonuç alırsınız.

## Nasıl yüklenir

```python
import torch
from model import DASNet, load_pretrained

bundle = torch.load('{filename}', map_location='cpu', weights_only=False)

# 1) Aynı 3 sınıfla kullanmak
model = DASNet(attention='{attention}')
model.load_state_dict(bundle['state_dict'])
model.eval()

# 2) Gerçek veride FARKLI sayıda sınıfla (transfer öğrenme)
model = DASNet(attention='{attention}', n_classes=YENI_SINIF_SAYISI)
load_pretrained(model, bundle)
# -> omurga + SK-Attention yüklenir, classifier sıfırdan başlar

# 3) Çok az saha verisi varsa: omurgayı dondur, sadece sınıf katmanını eğit
load_pretrained(model, bundle, freeze_backbone=True)
```

> ⚠️ **`load_state_dict(..., strict=False)` tek başına YETMEZ.** `strict=False`
> yalnızca eksik/fazla anahtarları tolere eder; sınıf sayısı değiştiğinde
> `classifier` katmanında **boyut uyuşmazlığı** hatası verir. `load_pretrained()`
> bu tensörleri atlayıp omurgayı yükler.

**Ön işleme aynı olmalı:**
```python
from torchvision.transforms import v2
tf = v2.Compose([
    v2.Resize(({h}, {w}), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean={norm_mean}, std={norm_std}),
])
```

## Bilinmesi gerekenler

{caveats}

## Paketin içindekiler

`bundle` sözlüğü şunları taşır: `architecture`, `input`, `classes`,
`training` (hiperparametreler + epoch kayıp geçmişi), `cv_performance`,
`caveats`, `state_dict`. Yani model başka bir projede **bu dosyaya bakarak**
yeniden kurulabilir.

## Katman modelleri

Çapraz doğrulamadaki {n_folds} model ayrıca `outputs/checkpoints/` altında
duruyor (`fold_i_{attention}_best.pt`). Topluluk (ensemble) denemek veya
katman varyansını incelemek için kullanılabilir.
"""


def write_model_card(bundle, path, filename):
    cv = bundle["cv_performance"] or {}
    arch = bundle["architecture"]
    inp = bundle["input"]
    per_fold = (", ".join(str(x) for x in cv.get("per_fold_macro_f1", []))
                or "yok")
    path.write_text(MODEL_CARD.format(
        attention=arch["attention"],
        attention_upper=arch["attention"].upper(),
        filename=filename,
        created=bundle["created_utc"][:19].replace("T", " ") + " UTC",
        commit=(bundle["git_commit"] or "bilinmiyor")[:12],
        n_params=arch["n_parameters"],
        h=inp["height"], w=inp["width"],
        feat=arch["feature_dim"],
        classes=", ".join(f"`{c}`" for c in bundle["classes"]),
        n_samples=bundle["training"]["n_samples"],
        n_rec=bundle["training"]["n_effective_recordings"],
        f1_mean=cv.get("macro_f1_mean", "?"),
        f1_std=cv.get("macro_f1_std", "?"),
        n_folds=cv.get("n_folds", cfg.N_SPLITS),
        per_fold=per_fold,
        norm_mean=list(inp["norm_mean"]), norm_std=list(inp["norm_std"]),
        caveats="\n".join(f"- {c}" for c in bundle["caveats"]),
    ), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="Nihai modeli tum veriyle egit ve disa aktar")
    ap.add_argument("--attention", default="sk",
                    choices=["sk", "none", "se", "cbam"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="varsayilan: CV'deki en iyi epoch'larin medyani")
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg.ensure_dirs()
    out_dir = cfg.OUT_DIR / "pretrained"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    print("=" * 74)
    print(f"NIHAI MODEL DISA AKTARIMI  |  dikkat={args.attention}")
    print("=" * 74)

    cv_summary, median_epoch = read_cv_results(args.attention)
    if cv_summary:
        print(f"  Capraz dogrulama bulundu ({cv_summary['n_folds']} katman)")
        print(f"    macro-F1 : {cv_summary['macro_f1_mean']} "
              f"+- {cv_summary['macro_f1_std']}")
        print(f"    dogruluk : {cv_summary['accuracy_mean']} "
              f"+- {cv_summary['accuracy_std']}")
        print(f"    en iyi epoch'lar: {cv_summary['per_fold_best_epoch']} "
              f"-> medyan {median_epoch}")
    else:
        print(f"  UYARI: outputs/results/ altinda "
              f"fold_*_{args.attention}_history.json bulunamadi.")
        print(f"         Paket sadece agirliklari tasiyacak, performans "
              f"tahmini olmayacak.")
        print(f"         Once 'python src/train.py --fold all "
              f"--attention {args.attention}' calistirmak onerilir.")

    epochs = args.epochs or median_epoch
    if epochs is None:
        raise SystemExit(
            "Epoch sayisi belirlenemedi: ne CV gecmisi var ne --epochs verildi.")

    model, history, n_samples = train_on_all_data(
        args.attention, epochs, device, args.batch_size)

    bundle = build_bundle(model, args.attention, epochs, history,
                          n_samples, cv_summary)
    filename = f"das_2dcnn_{args.attention}_v1.pt"
    ckpt_path = out_dir / filename
    torch.save(bundle, ckpt_path)

    card_path = out_dir / "MODEL_CARD.md"
    write_model_card(bundle, card_path, filename)

    size_mb = ckpt_path.stat().st_size / 1e6
    print(f"\n{'=' * 74}")
    print("YAZILDI")
    print("=" * 74)
    print(f"  {ckpt_path}   ({size_mb:.2f} MB)")
    print(f"  {card_path}")
    print(f"\n  Paket icerigi: architecture, input, classes, training, "
          f"cv_performance, caveats, state_dict")
    print(f"  Son epoch egitim kaybi: {history[-1]['loss']:.4f} "
          f"| dogruluk: {history[-1]['acc']:.3f}")
    print(f"\n  HATIRLATMA: bu model tum veriyle egitildi, kendi basina "
          f"durust bir test")
    print(f"  skoru yoktur. Raporlanacak sayi capraz dogrulamadan gelir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
