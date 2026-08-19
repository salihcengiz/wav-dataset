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
D1) Izlenen metrik: val_loss. PLAN 7.3 ReduceLROnPlateau icin acikca val_loss
    diyor ama erken durdurmanin ve "en iyi checkpoint"in neye gore secilecegini
    soylemiyor. Ucu de val_loss'a baglandi -- tek ve tutarli bir olcut olsun
    diye. Bilgi amacli val macro-F1 de her epoch loglaniyor.
    NOT: dogrulama seti kucuk (110 ornek), bu yuzden val_loss gurultuludur;
    erken durdurma sabri 10 epoch bu gurultuyu tolere edecek kadar genis.

D2) Checkpoint adi: fold_{i}_{attention}_best.pt. PLAN 7.5 "fold_{i}_best.pt"
    diyor ama ablasyon icin birden fazla varyant egitecegiz (none/sk/se/cbam);
    dikkat modulu adi eklenmezse dosyalar birbirini ezerdi.
"""
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
               verbose=True):
    """Tek bir katmani sifirdan egitir. Donen: history sozlugu."""
    # PLAN 7.5: tohum her katmanda AYNI -- katman varyansi veriden gelsin
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    (tr, va, te), fold, df, cache = make_loaders(
        fold_idx, df=df, cache=cache, batch_size=batch_size, verbose=False)

    model = DASNet(attention=attention).to(device)
    criterion = nn.CrossEntropyLoss()          # PLAN 7.3: siniflar dengeli, agirlik yok
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR,
                                 weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE)

    if verbose:
        print(f"\n{'=' * 78}")
        print(f"KATMAN {fold_idx}  |  dikkat={attention}  |  cihaz={device}")
        print(f"{'=' * 78}")
        print(f"  egitim {len(tr.dataset):>4} | dogrulama {len(va.dataset):>4} "
              f"| test {len(te.dataset):>4} | atilan mixup {fold['n_dropped_mixup']}")
        print(f"  parametre {count_parameters(model):,} | batch {batch_size} "
              f"| maks epoch {max_epochs} | erken durdurma sabri {patience}")
        print(f"  {'ep':>4} {'lr':>9} {'tr_loss':>9} {'tr_acc':>8} "
              f"{'val_loss':>9} {'val_acc':>8} {'val_F1':>8}  ")
        print(f"  {'-' * 62}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_macro_f1": [], "lr": []}
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, tr, criterion, optimizer, device)
        val = evaluate(model, va, criterion, device)
        scheduler.step(val["loss"])

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["lr"].append(lr_now)

        # D1: val_loss hem scheduler hem erken durdurma hem checkpoint icin
        improved = val["loss"] < best_loss - 1e-5
        if improved:
            best_loss, best_epoch, bad = val["loss"], epoch, 0
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
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch, "best_val_loss": best_loss,
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
                "best_val_loss": best_loss}, ckpt)

    fig = cfg.FIG_DIR / f"fold_{fold_idx}_{attention}_curves.png"
    plot_curves(history, fig,
                f"Katman {fold_idx} -- {attention} "
                f"(en iyi epoch {best_epoch}, test macro-F1 {test['macro_f1']:.3f})")

    hist_path = cfg.RESULTS_DIR / f"fold_{fold_idx}_{attention}_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if verbose:
        gap = history["train_acc"][best_epoch - 1] - history["val_acc"][best_epoch - 1]
        print(f"  {'-' * 62}")
        print(f"  sure {elapsed:.0f}s | {history['epochs_run']} epoch | "
              f"en iyi epoch {best_epoch}")
        print(f"  TEST  kayip {test['loss']:.4f} | dogruluk {test['acc']:.3f} | "
              f"macro-F1 {test['macro_f1']:.3f}")
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
    ap.add_argument("--device", default=None, help="cuda / cpu (varsayilan: otomatik)")
    args = ap.parse_args()

    cfg.ensure_dirs()
    device = get_device(args.device)
    folds = list(range(cfg.N_SPLITS)) if args.fold == "all" else [int(args.fold)]

    print("=" * 78)
    print(f"FAZ 3 -- EGITIM  |  dikkat={args.attention}  |  cihaz={device}")
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
                                  patience=args.patience))

    if len(results) > 1:
        print(f"\n{'=' * 78}")
        print(f"OZET -- {args.attention}")
        print("=" * 78)
        print(f"  {'katman':<8}{'epoch':>7}{'en_iyi':>8}{'test_acc':>10}"
              f"{'test_F1':>9}{'sure':>8}")
        print("  " + "-" * 48)
        for r in results:
            print(f"  {r['fold']:<8}{r['epochs_run']:>7}{r['best_epoch']:>8}"
                  f"{r['test_acc']:>10.3f}{r['test_macro_f1']:>9.3f}"
                  f"{r['seconds']:>7.0f}s")
        f1 = np.array([r["test_macro_f1"] for r in results])
        acc = np.array([r["test_acc"] for r in results])
        print("  " + "-" * 48)
        print(f"  macro-F1 : {f1.mean():.3f} +- {f1.std():.3f}   <- BIRINCIL METRIK")
        print(f"  dogruluk : {acc.mean():.3f} +- {acc.std():.3f}")
        print(f"\n  PLAN 8.2: ortalama tek basina raporlanmaz, std ile birlikte.")

    print(f"\nFAZ 3 TAMAM. Sonuclar: {cfg.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
