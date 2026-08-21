"""
FAZ 3 (2/2) -- Egitim Dongusu (PLAN Bolum 7.3-7.5).

Kullanim:
    python src/train.py --fold 0                  # tek katman (PLAN 12.4: ONCE BUNU)
    python src/train.py --fold all                # tum katmanlar
    python src/train.py --fold all --attention none   # ablasyon baseline'i
    python src/train.py --fold 0 --epochs 3       # hizli smoke test

PLAN 7.5 KURALLARI
------------------
- Her katman icin SIFIRDAN yeni model; agirliklar katmanlar arasi tasinmaz.
- Tohum sabit ve HER KATMANDA AYNI (PLAN: "her katman icin farkli tohum
  kullanma -- katman varyansi gercek olsun"). Boylece katmanlar arasi fark
  veriden gelir, rastgelelikten degil.
- Katman basina en iyi checkpoint + kayip/dogruluk egrisi kaydedilir.

PLANDA BELIRTILMEYEN, BURADA VERILEN KARARLAR
---------------------------------------------
D1) Izlenen metrik: **val macro-F1** (PAKET 1 / A1 -- eskiden val_loss'tu).

    PLAN 7.3 ReduceLROnPlateau icin val_loss diyor, erken durdurmanin neyi
    izleyecegini ise soylemiyor. Ilk uygulamada ucu de (erken durdurma,
    checkpoint, LR zamanlayici) val_loss'a baglanmisti. OLCULEN SONUC: bu
    secim baseline modelini iki katmanda TAMAMEN sabote etti --

        baseline katman 2:  epoch  1 -> val_acc 0.182, val_loss 1.30
                            epoch 11 -> val_acc 0.536, val_loss 3.48

    Model dogrulukta 3 katina cikarken, birkac ornekte asiri kendine guvendigi
    icin kayip yukseldi. Kural "1. epoch'tan beri iyilesme yok" deyip modeli
    1. epoch'ta dondurdu; test macro-F1 0.162 cikti (rastgele tahmin 0.333).

    Bu yuzden ucu de artik val macro-F1 izliyor: PLAN 8.1 zaten macro-F1'i
    BIRINCIL METRIK ilan ediyor, model secimini ona gore yapmak tutarli olan.
    `--monitor loss` ile eski davranisa donulebilir (karsilastirma icin).

D2) Esitlikte EN ERKEN epoch secilir (A2). Iyilesme kontrolu KATI esitsizlik
    kullanir, dolayisiyla ayni skora ulasan ilk epoch korunur. Kanit: SK
    katman 0'da dogrulama 10. epoch'ta tavana vurmustu ama 51. epoch secildi --
    aradaki 41 epoch modeli sadece daha "emin" (ve daha ezberci) yapti.

D3) Determinizm (A3). Ilk kosuda ayni tohumla iki farkli sonuc alindi
    (test F1 0.558 ve 0.572) cunku GPU'da konvolusyon geri yayilimi varsayilan
    olarak deterministik degil. Artik cuDNN deterministik moda aliniyor.
    Kucuk bir hiz bedeli var, karsiliginda sonuclar tekrar uretilebilir.

D4) Checkpoint adi: fold_{i}_{attention}_best.pt. PLAN 7.5 "fold_{i}_best.pt"
    diyor ama ablasyon icin birden fazla varyant egitecegiz (none/sk/se/cbam);
    dikkat modulu adi eklenmezse dosyalar birbirini ezerdi.
"""
import os

# CUDA deterministik matris carpimi icin -- torch import'undan ONCE ayarlanmali.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import config as cfg
from dataset import build_cache, make_loaders
from model import DASNet, count_parameters

import pandas as pd


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- IZLENEN METRIK (A1) -------------------------------------------------
# name -> (history anahtari, "buyuk mu iyi", scheduler modu)
MONITORS = {
    "macro_f1": ("val_macro_f1", True, "max"),
    "loss": ("val_loss", False, "min"),
}


def set_deterministic(seed):
    """
    A3 -- ayni tohum ayni sonucu versin.

    GPU'da konvolusyon geri yayilimi varsayilan olarak deterministik degildir
    (cuDNN algoritma secimi + atomik toplamalar). Bunu kapatmadan "ayni tohumla
    tekrar uretilebilir" diyemeyiz -- nitekim ilk kosuda katman 0 iki kez
    calistirildi ve 0.558 / 0.572 gibi iki farkli sonuc verdi.

    warn_only=True: deterministik uygulamasi olmayan bir islem cikarsa hata
    firlatmak yerine uyarir. Sert bir cokme, egitimi tamamen durdurmaktan iyidir.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Bir veri kumesi uzerinde kayip / dogruluk / macro-F1."""
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * y.size(0)
        n += y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "loss": total_loss / max(n, 1),
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(cfg.N_CLASSES)),
                                   zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += int((out.argmax(1) == y).sum())
        n += y.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep, history["train_loss"], label="egitim")
    axes[0].plot(ep, history["val_loss"], label="dogrulama")
    if history.get("best_epoch"):
        axes[0].axvline(history["best_epoch"], ls="--", c="gray", lw=1,
                        label=f"en iyi (ep {history['best_epoch']})")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("kayip")
    axes[0].set_title("Kayip"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(ep, history["train_acc"], label="egitim")
    axes[1].plot(ep, history["val_acc"], label="dogrulama")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("dogruluk")
    axes[1].set_title("Dogruluk"); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[1].set_ylim(0, 1.02)

    axes[2].plot(ep, history["val_macro_f1"], c="tab:green", label="dogrulama macro-F1")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("macro-F1")
    axes[2].set_title("Dogrulama macro-F1"); axes[2].legend(); axes[2].grid(alpha=.3)
    axes[2].set_ylim(0, 1.02)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def train_fold(fold_idx, attention, df, cache, device, max_epochs,
               batch_size=cfg.BATCH_SIZE, patience=cfg.EARLY_STOP_PATIENCE,
               monitor="macro_f1", batchnorm=cfg.BACKBONE_BATCHNORM,
               label_smoothing=cfg.LABEL_SMOOTHING, verbose=True):
    """Tek bir katmani sifirdan egitir. Donen: history sozlugu."""
    # PLAN 7.5: tohum her katmanda AYNI -- katman varyansi veriden gelsin
    set_deterministic(cfg.SEED)

    mon_key, mon_higher_better, mon_mode = MONITORS[monitor]

    (tr, va, te), fold, df, cache = make_loaders(
        fold_idx, df=df, cache=cache, batch_size=batch_size, verbose=False)

    model = DASNet(attention=attention, batchnorm=batchnorm).to(device)
    # PLAN 7.3: siniflar dengeli, agirlik yok.
    # PAKET 2 / B2: label smoothing -- "emin ve yanlis" tahminleri cezalandirir.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                                 weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mon_mode, factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE)

    if verbose:
        print(f"\n{'=' * 78}")
        print(f"KATMAN {fold_idx}  |  dikkat={attention}  |  cihaz={device}")
        print(f"{'=' * 78}")
        print(f"  egitim {len(tr.dataset):>4} | dogrulama {len(va.dataset):>4} "
              f"| test {len(te.dataset):>4} | atilan mixup {fold['n_dropped_mixup']}")
        print(f"  parametre {count_parameters(model):,} | batch {batch_size} "
              f"| maks epoch {max_epochs} | erken durdurma sabri {patience}")
        print(f"  girdi {cfg.INPUT_H}x{cfg.INPUT_W} | BatchNorm "
              f"{'acik' if batchnorm else 'KAPALI'} | label smoothing {label_smoothing}")
        print(f"  izlenen metrik: {mon_key} "
              f"({'buyugu' if mon_higher_better else 'kucugu'} iyi)")
        print(f"  {'ep':>4} {'lr':>9} {'tr_loss':>9} {'tr_acc':>8} "
              f"{'val_loss':>9} {'val_acc':>8} {'val_F1':>8}  ")
        print(f"  {'-' * 62}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_macro_f1": [], "lr": []}
    best_score = -float("inf") if mon_higher_better else float("inf")
    best_epoch, best_state, bad = 0, None, 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, tr, criterion, optimizer, device)
        val = evaluate(model, va, criterion, device)
        score = val["macro_f1"] if mon_higher_better else val["loss"]
        scheduler.step(score)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["lr"].append(lr_now)

        # A1 + A2: izlenen metrik (varsayilan val macro-F1) hem scheduler hem
        # erken durdurma hem checkpoint icin. KATI esitsizlik kullaniliyor --
        # boylece ayni skora ulasan EN ERKEN epoch korunur (A2).
        improved = (score > best_score + 1e-5 if mon_higher_better
                    else score < best_score - 1e-5)
        if improved:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1

        if verbose:
            mark = " *" if improved else ""
            print(f"  {epoch:>4} {lr_now:>9.2e} {tr_loss:>9.4f} {tr_acc:>8.3f} "
                  f"{val['loss']:>9.4f} {val['acc']:>8.3f} {val['macro_f1']:>8.3f}{mark}")

        if bad >= patience:
            if verbose:
                print(f"  -> erken durdurma: {patience} epoch boyunca iyilesme yok")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    test = evaluate(model, te, criterion, device)

    history.update({
        "fold": fold_idx, "attention": attention,
        "monitor": mon_key,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch, "best_score": best_score,
        "best_val_loss": history["val_loss"][best_epoch - 1],
        "best_val_macro_f1": history["val_macro_f1"][best_epoch - 1],
        "test_loss": test["loss"], "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
        "test_y_true": test["y_true"].tolist(),
        "test_y_pred": test["y_pred"].tolist(),
        "test_idx": fold["test_idx"],
        "n_train": len(tr.dataset), "n_val": len(va.dataset),
        "n_test": len(te.dataset),
        "n_parameters": count_parameters(model),
        "seconds": round(elapsed, 1),
    })

    ckpt = cfg.CKPT_DIR / f"fold_{fold_idx}_{attention}_best.pt"
    torch.save({"state_dict": best_state, "fold": fold_idx,
                "attention": attention, "best_epoch": best_epoch,
                "monitor": mon_key, "best_score": best_score}, ckpt)

    fig = cfg.FIG_DIR / f"fold_{fold_idx}_{attention}_curves.png"
    plot_curves(history, fig,
                f"Katman {fold_idx} -- {attention} "
                f"(en iyi epoch {best_epoch}, test macro-F1 {test['macro_f1']:.3f})")

    hist_path = cfg.RESULTS_DIR / f"fold_{fold_idx}_{attention}_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if verbose:
        gap = history["train_acc"][best_epoch - 1] - history["val_acc"][best_epoch - 1]
        vf1 = history["val_macro_f1"][best_epoch - 1]
        print(f"  {'-' * 62}")
        print(f"  sure {elapsed:.0f}s | {history['epochs_run']} epoch | "
              f"en iyi epoch {best_epoch}")
        print(f"  TEST  kayip {test['loss']:.4f} | dogruluk {test['acc']:.3f} | "
              f"macro-F1 {test['macro_f1']:.3f}")
        print(f"  dogrulama F1 {vf1:.3f} -> test F1 {test['macro_f1']:.3f}"
              f"   (fark {test['macro_f1'] - vf1:+.3f})")
        print(f"  en iyi epoch'ta egitim-dogrulama dogruluk farki: {gap:+.3f}"
              f"   {'(asiri ogrenme suphesi)' if gap > 0.25 else ''}")
        print(f"  checkpoint: {ckpt.name} | egri: {fig.name}")

    return history


def main():
    ap = argparse.ArgumentParser(description="DAS 2D-CNN + SK-Attention egitimi")
    ap.add_argument("--fold", default="0",
                    help="katman indeksi (0-3) veya 'all'")
    ap.add_argument("--attention", default="sk",
                    choices=["sk", "none", "se", "cbam"],
                    help="dikkat modulu ('none' = ablasyon baseline'i)")
    ap.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=cfg.EARLY_STOP_PATIENCE)
    ap.add_argument("--monitor", default="macro_f1", choices=list(MONITORS),
                    help="erken durdurma/checkpoint neyi izlesin "
                         "(varsayilan macro_f1; 'loss' eski davranis)")
    ap.add_argument("--no-batchnorm", action="store_true",
                    help="omurgadaki BatchNorm'u kapat (PLAN 6.1 harfi, ablasyon)")
    ap.add_argument("--label-smoothing", type=float, default=cfg.LABEL_SMOOTHING)
    ap.add_argument("--device", default=None, help="cuda / cpu (varsayilan: otomatik)")
    args = ap.parse_args()

    cfg.ensure_dirs()
    device = get_device(args.device)
    folds = list(range(cfg.N_SPLITS)) if args.fold == "all" else [int(args.fold)]

    batchnorm = not args.no_batchnorm
    print("=" * 78)
    print(f"FAZ 3 -- EGITIM  |  dikkat={args.attention}  |  cihaz={device}")
    print(f"  izlenen metrik : {MONITORS[args.monitor][0]}   (PAKET 1 / A1)")
    print(f"  determinizm    : acik   (PAKET 1 / A3)")
    print(f"  BatchNorm      : {'acik' if batchnorm else 'KAPALI'}   (PAKET 2 / B1)")
    print(f"  label smoothing: {args.label_smoothing}   (PAKET 2 / B2)")
    print(f"  girdi boyutu   : {cfg.INPUT_H}x{cfg.INPUT_W} "
          f"(frekans x zaman)   (PAKET 2 / C1)")
    print("=" * 78)
    if device.type == "cpu":
        print("  NOT: CPU'da calisiyor. Tam egitim icin GPU (Colab) onerilir.")

    df = pd.read_csv(cfg.METADATA_CSV)
    cache = build_cache(df)

    results = []
    for i in folds:
        results.append(train_fold(i, args.attention, df, cache, device,
                                  max_epochs=args.epochs,
                                  batch_size=args.batch_size,
                                  patience=args.patience,
                                  monitor=args.monitor,
                                  batchnorm=batchnorm,
                                  label_smoothing=args.label_smoothing))

    if len(results) > 1:
        print(f"\n{'=' * 78}")
        print(f"OZET -- {args.attention}")
        print("=" * 78)
        print(f"  {'katman':<8}{'epoch':>7}{'en_iyi':>8}{'val_F1':>9}"
              f"{'test_acc':>10}{'test_F1':>9}{'fark':>8}{'sure':>8}")
        print("  " + "-" * 67)
        for r in results:
            d = r["test_macro_f1"] - r["best_val_macro_f1"]
            print(f"  {r['fold']:<8}{r['epochs_run']:>7}{r['best_epoch']:>8}"
                  f"{r['best_val_macro_f1']:>9.3f}{r['test_acc']:>10.3f}"
                  f"{r['test_macro_f1']:>9.3f}{d:>+8.3f}{r['seconds']:>7.0f}s")
        f1 = np.array([r["test_macro_f1"] for r in results])
        acc = np.array([r["test_acc"] for r in results])
        vf1 = np.array([r["best_val_macro_f1"] for r in results])
        print("  " + "-" * 67)
        print(f"  macro-F1 : {f1.mean():.3f} +- {f1.std():.3f}   <- BIRINCIL METRIK")
        print(f"  dogruluk : {acc.mean():.3f} +- {acc.std():.3f}")
        print(f"\n  PLAN 8.2: ortalama tek basina raporlanmaz, std ile birlikte.")
        # Dogrulama seti testi tahmin edebiliyor mu? (Ilk kosuda edemiyordu.)
        if len(f1) > 2 and vf1.std() > 1e-6 and f1.std() > 1e-6:
            corr = float(np.corrcoef(vf1, f1)[0, 1])
            print(f"  dogrulama-test korelasyonu: {corr:+.3f}  "
                  f"({'dogrulama testi tahmin ediyor' if corr > 0.5 else 'dogrulama GUVENILMEZ'})")

    print(f"\nFAZ 3 TAMAM. Sonuclar: {cfg.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
