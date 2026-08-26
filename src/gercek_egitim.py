"""
GERCEK VERI -- ADIM 6: EGITIM

Onbellekten okur, DASNet'i egitir, test setinde bir kez olcer.

=== SENTETIK EGITIMDEN (train.py) FARKLARI ===

1. CAPRAZ DOGRULAMA YOK. Sentetikte 19 bagimsiz kayit vardi, tek bir
   train/test bolmesi cok gurultulu olurdu, o yuzden 4 katli gruplu CV
   yapmistik. Burada ekibin hazir bolmeleri var ve dosya/oturum/tarih
   duzeyinde temiz oldugu OLCULDU (Rapor Bolum 3). 21.101 bagimsiz dosya
   ile tek bolme yeterince kararli.

2. DOGRULAMA SETI GUVENILIR. Sentetikte dogrulama setinde sinif basina
   1 kayit vardi ve testle korelasyonu ~0'di (SENTETIK_VERI_SONUCLARI
   Bolum 5). Burada 3.287 dosyadan 37.517 pencere var; model secimi artik
   anlamli.

3. VERI 230 KAT BUYUK. 959 -> 220.834 pencere.

=== KORUNAN DERSLER (DURUM.md Bolum 6) ===

A1. Model secimi ve erken durdurma **val macro-F1** izler, val_loss DEGIL.
    Sentetikte val_loss izlemek baseline'i iki katmanda 1. epoch'ta
    dondurmustu.
A2. Esitlikte EN ERKEN epoch secilir (kati esitsizlik).
A3. set_deterministic() -- ayni tohum ayni sonuc.
+   Egitim ve cikarim TEK fonksiyondan gecer (gercek_veri_kumesi.hazirla).

=== UC KOSU (onceden tasarlandi, test setine bakmadan) ===

    #  girdi     baslangic   neyi olcer
    1  viridis   sifirdan    referans
    2  viridis   aktarim     1'e karsi: sentetik on-egitim ise yariyor mu
    3  gri       sifirdan    1'e karsi: viridis zarar veriyor mu

Her koşuda tek degisken degisiyor. Ikisini birden degistirmek (orn.
aktarim+viridis vs sifirdan+gri) hangi etkinin fark yarattigini
belirsizlestirirdi -- bu projenin daha once dustugu hata turu.

Aktarim varsayimi artik tartismali: onceden egitilmis paket 19 bagimsiz
sentetik kayittan uretildi, burada 21.101 gercek dosya var. Sonuc ne
cikarsa ciksin raporlanacak.

=== KULLANIM ===

    python gercek_egitim.py --kosu 1              # viridis + sifirdan
    python gercek_egitim.py --kosu 2              # viridis + aktarim
    python gercek_egitim.py --kosu 3              # gri + sifirdan
    python gercek_egitim.py --kosu 1 --hizli      # 200 batch'lik duman testi

JupyterLab:
    import gercek_egitim
    gercek_egitim.kos(1)
"""
import json
import os
import sys
import time
from pathlib import Path

# CUDA deterministik matris carpimi -- torch import'undan ONCE
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

_burada = str(Path(__file__).resolve().parent)
if _burada not in sys.path:
    sys.path.insert(0, _burada)

from gercek_veri_kumesi import OnbellekKumesi, hazirla, yukleyici
from model import DASNet, count_parameters, load_pretrained

VERI = Path("/tf/start_training/RELATIONNET/FENCE_DATA_NEW")
CIKTI = VERI / "egitim_ciktilari"
ONCEDEN = "das_2dcnn_sk_v1.pt"

TOHUM = 42
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAKS_EPOCH = 40
SABIR = 6                # erken durdurma
LR_SABIR = 3
LABEL_SMOOTHING = 0.1

KOSULAR = {
    1: {"ad": "viridis_sifirdan", "renk": "viridis", "aktarim": False},
    2: {"ad": "viridis_aktarim", "renk": "viridis", "aktarim": True},
    3: {"ad": "gri_sifirdan", "renk": "gri", "aktarim": False},
}


def set_deterministic(tohum=TOHUM):
    """
    A3 -- ayni tohum ayni sonucu versin.

    GPU'da konvolusyon geri yayilimi varsayilan olarak deterministik
    degildir (cuDNN algoritma secimi + atomik toplamalar). Sentetik
    asamada ayni tohumla iki farkli sonuc alinmisti (0.558 / 0.572).

    warn_only=True: deterministik uygulamasi olmayan bir islem cikarsa
    hata firlatmak yerine uyarir -- egitimi tamamen durdurmaktan iyidir.
    """
    torch.manual_seed(tohum)
    np.random.seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def degerlendir(model, yukl, kayip_fn, cihaz, n_sinif=3, maks_batch=None):
    """Bir veri kumesinde kayip / dogruluk / macro-F1 + tahminler."""
    model.eval()
    toplam, n = 0.0, 0
    ys, ps = [], []
    for i, (x, y) in enumerate(yukl):
        if maks_batch and i >= maks_batch:
            break
        x = hazirla(x.to(cihaz, non_blocking=True), egitim=False)
        y = y.to(cihaz, non_blocking=True)
        cikti = model(x)
        toplam += kayip_fn(cikti, y).item() * y.size(0)
        n += y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(cikti.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return {
        "kayip": toplam / max(n, 1),
        "dogruluk": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(n_sinif)),
                                   zero_division=0)),
        "y_true": y_true, "y_pred": y_pred,
    }


def bir_epoch(model, yukl, kayip_fn, optim, cihaz, maks_batch=None,
              her=200, t0=None):
    model.train()
    toplam, n, dogru = 0.0, 0, 0
    for i, (x, y) in enumerate(yukl):
        if maks_batch and i >= maks_batch:
            break
        x = hazirla(x.to(cihaz, non_blocking=True), egitim=True)
        y = y.to(cihaz, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        cikti = model(x)
        kayip = kayip_fn(cikti, y)
        kayip.backward()
        optim.step()
        toplam += kayip.item() * y.size(0)
        dogru += int((cikti.argmax(1) == y).sum())
        n += y.size(0)
        if her and (i + 1) % her == 0:
            hiz = n / (time.perf_counter() - t0) if t0 else 0
            print(f"      batch {i+1:>5}  kayip {toplam/n:.4f}  "
                  f"dogruluk {dogru/n:.3f}  {hiz:,.0f} ornek/s", flush=True)
    return toplam / max(n, 1), dogru / max(n, 1)


def kos(kosu=1, veri=VERI, cikti=CIKTI, epoch=MAKS_EPOCH, batch=BATCH,
        isci=6, sinif_agirligi=False, hizli=False, onceden=None):
    """
    Bir konfigurasyonu bastan sona egitir ve test setinde OLCER.

    hizli=True: 200 egitim + 50 dogrulama batch'i. Hattin calistigini
    dogrulamak icin; sonuc raporlanmaz.
    """
    if kosu not in KOSULAR:
        raise ValueError(f"kosu 1, 2 veya 3 olmali ({kosu} verildi)")
    ayar = KOSULAR[kosu]
    veri, cikti = Path(veri), Path(cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    set_deterministic(TOHUM)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"KOSU {kosu} -- {ayar['ad']}")
    print("=" * 78)
    print(f"  girdi temsili : {ayar['renk']}")
    print(f"  baslangic     : {'AKTARIM (' + ONCEDEN + ')' if ayar['aktarim'] else 'SIFIRDAN'}")
    print(f"  cihaz         : {cihaz}"
          + (f"  ({torch.cuda.get_device_name(0)})" if cihaz.type == "cuda" else ""))
    print(f"  tohum {TOHUM} | batch {batch} | lr {LR} | maks epoch {epoch} "
          f"| sabir {SABIR}")

    # --- veri ---
    kumeler, yukleyiciler = {}, {}
    for ad, dosya in (("train", "onbellek_train_final_k0.h5"),
                      ("val", "onbellek_val_final_k0.h5"),
                      ("test", "onbellek_test_final_k0.h5")):
        k = OnbellekKumesi(veri / dosya, egitim=(ad == "train"),
                           renk=ayar["renk"])
        kumeler[ad] = k
        yukleyiciler[ad] = yukleyici(k, batch=batch, isci=isci)
    siniflar = kumeler["train"].siniflar
    print(f"  train {len(kumeler['train']):,} | val {len(kumeler['val']):,} "
          f"| test {len(kumeler['test']):,} | siniflar {siniflar}")

    # --- model ---
    model = DASNet(attention="sk", n_classes=len(siniflar)).to(cihaz)
    if ayar["aktarim"]:
        yol = Path(onceden or (Path(_burada).parent / "outputs" / "pretrained" / ONCEDEN))
        if not yol.exists():
            raise FileNotFoundError(
                f"Onceden egitilmis paket bulunamadi: {yol}\n"
                f"Kosu 2 icin gerekli. Depodan kopyalanmali.")
        print(f"  aktarim paketi: {yol}")
        # classifier BILEREK atlaniyor: sentetik siniflar
        # [chain_link_climbing, fence_cutting, metal_bending], gercek siniflar
        # [cutting, climbing, noise]. Sayi ayni (3) oldugu icin sekil kontrolu
        # bunu yakalayamaz ve sessizce yuklerdi -- ama farkli kavramlar, sira
        # farkli, 'noise'un sentetikte karsiligi yok. MODEL_CARD da
        # "classifier sifirdan baslayacak" diyor.
        load_pretrained(model, str(yol), verbose=True, atla=("classifier",))
    print(f"  parametre     : {count_parameters(model):,}")

    agirlik = None
    if sinif_agirligi:
        agirlik = kumeler["train"].sinif_agirliklari().to(cihaz)
        print(f"  sinif agirligi: {[round(float(v),3) for v in agirlik]}")
    kayip_fn = nn.CrossEntropyLoss(weight=agirlik,
                                   label_smoothing=LABEL_SMOOTHING)
    optim = torch.optim.Adam(model.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY)
    zamanlayici = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=LR_SABIR)

    mb_tr, mb_va = (200, 50) if hizli else (None, None)
    if hizli:
        epoch = min(epoch, 2)
        print(f"  *** HIZLI MOD: {mb_tr} egitim / {mb_va} dogrulama batch, "
              f"{epoch} epoch. Sonuc RAPORLANMAZ. ***")

    # --- egitim dongusu ---
    gecmis = {"train_kayip": [], "train_dogruluk": [], "val_kayip": [],
              "val_dogruluk": [], "val_macro_f1": [], "lr": [], "saniye": []}
    en_iyi_skor, en_iyi_epoch, en_iyi_durum, kotu = -float("inf"), 0, None, 0
    t_bas = time.perf_counter()

    for ep in range(1, epoch + 1):
        lr_su_an = optim.param_groups[0]["lr"]
        t0 = time.perf_counter()
        print(f"\n  --- epoch {ep}/{epoch}  (lr {lr_su_an:.2e}) ---", flush=True)
        tr_kayip, tr_dog = bir_epoch(model, yukleyiciler["train"], kayip_fn,
                                     optim, cihaz, maks_batch=mb_tr, t0=t0)
        val = degerlendir(model, yukleyiciler["val"], kayip_fn, cihaz,
                          len(siniflar), maks_batch=mb_va)
        sure = time.perf_counter() - t0
        zamanlayici.step(val["macro_f1"])

        gecmis["train_kayip"].append(tr_kayip)
        gecmis["train_dogruluk"].append(tr_dog)
        gecmis["val_kayip"].append(val["kayip"])
        gecmis["val_dogruluk"].append(val["dogruluk"])
        gecmis["val_macro_f1"].append(val["macro_f1"])
        gecmis["lr"].append(lr_su_an)
        gecmis["saniye"].append(round(sure, 1))

        # A1 + A2: val macro-F1 izlenir, KATI esitsizlik -> en erken epoch
        iyilesti = val["macro_f1"] > en_iyi_skor + 1e-5
        if iyilesti:
            en_iyi_skor, en_iyi_epoch, kotu = val["macro_f1"], ep, 0
            en_iyi_durum = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
        else:
            kotu += 1

        print(f"    egitim  kayip {tr_kayip:.4f}  dogruluk {tr_dog:.3f}")
        print(f"    dogrul. kayip {val['kayip']:.4f}  dogruluk "
              f"{val['dogruluk']:.3f}  macro-F1 {val['macro_f1']:.4f}"
              f"{'  *' if iyilesti else ''}")
        print(f"    sure {sure/60:.1f} dk"
              + (f"  |  {SABIR - kotu} epoch sabir kaldi" if kotu else ""))

        if kotu >= SABIR:
            print(f"\n  -> erken durdurma: {SABIR} epoch iyilesme yok")
            break

    toplam_sure = time.perf_counter() - t_bas
    model.load_state_dict(en_iyi_durum)

    # --- TEST: yalnizca BURADA, bir kez ---
    print(f"\n  {'-' * 70}")
    print(f"  TEST SETI  (en iyi epoch {en_iyi_epoch}, val macro-F1 "
          f"{en_iyi_skor:.4f})")
    print(f"  {'-' * 70}")
    test = degerlendir(model, yukleyiciler["test"], kayip_fn, cihaz,
                       len(siniflar), maks_batch=mb_va)
    print(f"  kayip {test['kayip']:.4f} | dogruluk {test['dogruluk']:.4f} | "
          f"macro-F1 {test['macro_f1']:.4f}")
    print()
    print(classification_report(test["y_true"], test["y_pred"],
                                labels=list(range(len(siniflar))),
                                target_names=siniflar, digits=3,
                                zero_division=0))
    km = confusion_matrix(test["y_true"], test["y_pred"],
                          labels=list(range(len(siniflar))))
    print("  Karisiklik matrisi (satir = gercek):")
    print(f"  {'':<10}" + "".join(f"{s[:8]:>10}" for s in siniflar))
    for i, s in enumerate(siniflar):
        print(f"  {s:<10}" + "".join(f"{v:>10,}" for v in km[i]))

    print(f"\n  TABAN CIZGISI (dogrusal, 26 ozellik): macro-F1 0.771")
    fark = test["macro_f1"] - 0.771
    print(f"  BU KOSU                            : macro-F1 "
          f"{test['macro_f1']:.3f}   ({fark:+.3f})")

    # --- kaydet ---
    if not hizli:
        etiket = f"kosu{kosu}_{ayar['ad']}"
        torch.save({"state_dict": en_iyi_durum, "kosu": kosu, "ayar": ayar,
                    "en_iyi_epoch": en_iyi_epoch, "val_macro_f1": en_iyi_skor,
                    "siniflar": siniflar},
                   cikti / f"{etiket}.pt")
        gecmis.update({
            "kosu": kosu, "ayar": ayar, "siniflar": siniflar,
            "en_iyi_epoch": en_iyi_epoch, "en_iyi_val_macro_f1": en_iyi_skor,
            "test_kayip": test["kayip"], "test_dogruluk": test["dogruluk"],
            "test_macro_f1": test["macro_f1"],
            "karisiklik": km.tolist(),
            "n_train": len(kumeler["train"]), "n_val": len(kumeler["val"]),
            "n_test": len(kumeler["test"]),
            "parametre": count_parameters(model),
            "toplam_dakika": round(toplam_sure / 60, 1),
            "tohum": TOHUM, "batch": batch, "lr": LR,
            "label_smoothing": LABEL_SMOOTHING,
            "sinif_agirligi": bool(sinif_agirligi),
        })
        (cikti / f"{etiket}_gecmis.json").write_text(
            json.dumps(gecmis, indent=2), encoding="utf-8")
        print(f"\n  kaydedildi: {cikti / (etiket + '.pt')}")
        print(f"              {cikti / (etiket + '_gecmis.json')}")

    print(f"\n  toplam sure: {toplam_sure/60:.1f} dk")
    return gecmis


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gercek veri egitimi")
    ap.add_argument("--kosu", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--veri", default=str(VERI))
    ap.add_argument("--cikti", default=str(CIKTI))
    ap.add_argument("--epoch", type=int, default=MAKS_EPOCH)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--isci", type=int, default=6)
    ap.add_argument("--sinif-agirligi", action="store_true",
                    help="dengesiz siniflar icin agirlikli kayip")
    ap.add_argument("--hizli", action="store_true",
                    help="kisa duman testi, sonuc raporlanmaz")
    ap.add_argument("--onceden", default=None, help="aktarim paketi yolu")
    a = ap.parse_args()
    kos(a.kosu, veri=a.veri, cikti=a.cikti, epoch=a.epoch, batch=a.batch,
        isci=a.isci, sinif_agirligi=a.sinif_agirligi, hizli=a.hizli,
        onceden=a.onceden)
