"""
BOS PENCERE TESTI -- model "hicbir sey yok" diyebiliyor mu?

=== NEDEN BU TEST ===

Saha testinde (MLflow waterfall) modelin, olay olmayan kanallarda emin
bir sekilde saldiri sinifi ilan ettigi gorulduL kanal 0 boyunca 30 saniye
kesintisiz `cutting`. Detailed Test'te tek pencere incelendiginde sebep
goruldu: BOS bir pencerede model `noise` secenegini AKTIF OLARAK
REDDEDIYOR (logit -1.110).

Teshis hipotezi: `noise` bizim veri setimizde ETIKETLENMIS BIR OLAY TURU
(ortam gurultusu, arac). "Pencerede sinyal yok" demek degil. Bos
pencereler egitimden SILINDI (bos_mu, train'in %23'u), `noise` diye
etiketlenmedi. Dolayisiyla modelin "hicbir sey yok" cevabi yok -- her
pencerede uc olay sinifindan birini secmek zorunda.

Bu script hipotezi SAYIYA cevirir:

    "Egitimde eledigimiz pencerelerin %X'ine model, logit > esik ile
     cutting ya da climbing diyor."

X yuksekse teshis kapanir ve cozum bellidir: bosluk_orani > BOS_ESIK olan
pencereler modele hic sorulmamali (ONNX bu degeri ikinci cikti olarak
zaten uretiyor). X dusukse teshis yanlistir ve baska yere bakilir.

=== NEDEN SAHA VERISINE IHTIYAC YOK ===

Test arayuzunun hangi dosyalari okudugu belirsiz (diskteki .hdf5
kopyalarinda 6-14 kanal var, arayuz 30 kanal sunuyor) ve .sdf ailesinde
ornekleme hizi tartismali. Bu test kendi egitim verimizle calisir --
okudugumuz format, bildigimiz ornekleme hizi, olctugumuz bosluk.

=== KULLANIM ===

    python bos_pencere_testi.py                     # varsayilan 3000 pencere
    python bos_pencere_testi.py --n 8000 --csv train_final.csv
"""
import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_KOK = Path(__file__).resolve().parent.parent
for _y in (str(_KOK / "src"), str(_KOK / "CNN-BiLSTM")):
    if _y not in sys.path:
        sys.path.insert(0, _y)

import real_data as rd                                    # noqa: E402

VERI = Path("/tf/start_training/RELATIONNET/FENCE_DATA_NEW")
CIKTI = VERI / "egitim_ciktilari"
SINIFLAR = ["cutting", "climbing", "noise"]
SALDIRI = {"cutting", "climbing"}


def csv_bul(veri, desen):
    aday = sorted(glob.glob(str(Path(veri) / desen)))
    if not aday:
        aday = sorted(glob.glob(str(Path(veri) / "*.csv")))
    return aday


def ornekle(csv_yol, n, tohum=42):
    """
    CSV'den n satir sec -- DOSYA BASINA gruplanmis halde.

    Rastgele satir secip her biri icin HDF5 acmak cok pahali (olculdu:
    3.87 ms/satir, buyuk kismi dosya acma). Once dosya seciyoruz, sonra o
    dosyadan cok satir okuyoruz; dosya bir kez aciliyor.
    """
    df = pd.read_csv(csv_yol)
    rng = np.random.default_rng(tohum)
    # object dtype: pandas StringArray'i dogrudan shuffle etmek uyari veriyor
    dosyalar = np.asarray(df["file"].astype(str).unique(), dtype=object)
    rng.shuffle(dosyalar)

    secilen, kalan = [], n
    for d in dosyalar:
        if kalan <= 0:
            break
        grup = df[df["file"].astype(str) == d]
        k = min(len(grup), max(1, n // 40))       # dosya basina ~n/40 satir
        secilen.append(grup.sample(n=min(k, kalan), random_state=tohum))
        kalan -= len(secilen[-1])
    return pd.concat(secilen, ignore_index=True), len(df)


def pencereleri_yukle(df):
    """
    Ham (normalize EDILMEMIS) pencereleri ve bosluk olcumlerini toplar.

    bos_ele=False: normalde pencere_yukle bos pencerelerde None dondurur.
    Burada tam da onlari istiyoruz.
    normalize=False: ONNX sarmalayicisi normalizasyonu KENDI yapiyor;
    iki kez uygulamak egitimden farkli bir girdi uretirdi.
    """
    import h5py
    pencereler, bosluklar, etiketler, kaynak = [], [], [], []
    hata = 0
    for dosya, grup in df.groupby(df["file"].astype(str)):
        if not os.path.exists(dosya):
            hata += len(grup)
            continue
        try:
            with h5py.File(dosya, "r") as f:
                for r in grup.itertuples():
                    try:
                        s = rd.pencere_yukle(
                            dosya, int(r.channel),
                            int(r.window_start), int(r.window_end),
                            normalize=False, bos_ele=False, dosya_nesnesi=f)
                    except Exception:       # noqa: BLE001
                        s = None
                    if s is None or len(s) != rd.PENCERE:
                        hata += 1
                        continue
                    pencereler.append(s.astype(np.float32))
                    bosluklar.append(rd.bosluk_orani(s))
                    etiketler.append(str(r.event))
                    kaynak.append((os.path.basename(dosya), int(r.channel)))
        except OSError:
            hata += len(grup)
    return (np.asarray(pencereler), np.asarray(bosluklar),
            etiketler, kaynak, hata)


def tahmin_et(sarmal, X, batch=64, cihaz="cpu"):
    import torch
    sarmal = sarmal.to(cihaz).eval()
    logitler, bosluklar = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(cihaz)
            lg, bo = sarmal(x)
            logitler.append(lg.cpu().numpy())
            bosluklar.append(bo.cpu().numpy())
    return np.concatenate(logitler), np.concatenate(bosluklar)


ESIKLER = (0.0, 0.5, 0.9, 1.5)


def rapor(ad, logit, maske, esik):
    """
    Bir grup pencere icin tahmin dagilimini yazdirir.

    Esik TARANIYOR: saha arayuzundeki CLASS THRESHOLD ham logite mi
    olasiliga mi uygulaniyor bilmiyoruz, ve hoca 0.7 ile 0.9 arasinda
    gidip geldi. Tek sayi yerine birkac esikte gostermek, hangi esikte
    ne kadar yanlis alarm kaldigini dogrudan okutuyor.
    """
    if maske.sum() == 0:
        print(f"  {ad}: ornek yok")
        return None
    L = logit[maske]
    tahmin = L.argmax(1)
    en_buyuk = L.max(1)
    sayim = Counter(SINIFLAR[t] for t in tahmin)
    n = len(L)
    saldiri = np.array([SINIFLAR[t] in SALDIRI for t in tahmin])

    print(f"  {ad:5s} n={n:6,}   "
          + "  ".join(f"{s} {100*sayim.get(s,0)/n:5.1f}%" for s in SINIFLAR))
    print(f"  {'':5s} en buyuk logit: ortalama {en_buyuk.mean():+.3f}  "
          f"medyan {np.median(en_buyuk):+.3f}  maks {en_buyuk.max():+.3f}")
    print(f"  {'':5s} esige gore SALDIRI orani: "
          + "   ".join(f">{e}: {100*((en_buyuk > e) & saldiri).mean():5.1f}%"
                       for e in ESIKLER))
    return {"n": n,
            "saldiri": float(saldiri.mean()),
            "ort_logit": float(en_buyuk.mean()),
            "maks_logit": float(en_buyuk.max()),
            "esikli": {e: float(((en_buyuk > e) & saldiri).mean())
                       for e in ESIKLER}}


def main():
    ap = argparse.ArgumentParser(description="Bos pencere testi")
    ap.add_argument("--veri", default=str(VERI))
    ap.add_argument("--cikti", default=str(CIKTI))
    ap.add_argument("--csv", default="*train*.csv")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--esik", type=float, default=0.9,
                    help="saha arayuzundeki CLASS THRESHOLD (ham logit); "
                         "rapor ayrica ESIKLER'i de tariyor")
    # BOS_ESIK'i ezmek: sahte veride hic bos pencere yok, o kod yolunu
    # yerelde sinamak icin gerekli. Sunucuda ayrica esik duyarliligini
    # olcmeye de yarar.
    ap.add_argument("--bos-esik", type=float, default=rd.BOS_ESIK,
                    help=f"bosluk_orani esigi (varsayilan {rd.BOS_ESIK})")
    ap.add_argument("--cihaz", default=None)
    a = ap.parse_args()

    import torch
    from onnx_disa_aktar import sarmalayici_kur
    cihaz = a.cihaz or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("BOS PENCERE TESTI -- model 'hicbir sey yok' diyebiliyor mu?")
    print("=" * 78)

    csvler = csv_bul(a.veri, a.csv)
    if not csvler:
        sys.exit(f"CSV bulunamadi: {a.veri}/{a.csv}")
    print(f"CSV      : {csvler[0]}")

    df, n_toplam = ornekle(csvler[0], a.n)
    print(f"ornek    : {len(df):,} satir  (CSV'de toplam {n_toplam:,})")
    print(f"cihaz    : {cihaz}   esik: logit > {a.esik}")

    X, bosluk, etiket, kaynak, hata = pencereleri_yukle(df)
    print(f"yuklenen : {len(X):,} pencere   (okunamayan {hata:,})")
    if len(X) == 0:
        sys.exit("hic pencere yuklenemedi -- CSV'deki 'file' yollari dogru mu?")

    bos = bosluk > a.bos_esik
    ek = "" if a.bos_esik == rd.BOS_ESIK else f"  (varsayilan {rd.BOS_ESIK} EZILDI)"
    print(f"\nbosluk_orani > {a.bos_esik}{ek}  ->  BOS: {bos.sum():,} "
          f"({100*bos.mean():.1f}%)   DOLU: {(~bos).sum():,}")
    print(f"bosluk_orani dagilimi: min {bosluk.min():.3f}  "
          f"medyan {np.median(bosluk):.3f}  maks {bosluk.max():.3f}")

    # Kosu 3 (gri+SK) ile kosu 4 (viridis+BiLSTM) karsilastirmasi MIMARI ve
    # RENGI birlikte degistiriyor. Kosu 1 (viridis+SK) ikisini ayiriyor:
    #   kosu 1 de susuyorsa  -> fark MIMARIDEN
    #   kosu 1 de bagiriyorsa -> fark VIRIDIS'ten
    KOSULAR = [("kosu1-SK-viridis", "kosu1_viridis_sifirdan.pt"),
               ("kosu3-SK-gri", "kosu3_gri_sifirdan.pt"),
               ("kosu5-SK-gri-yeni", "kosu5_gri_sifirdan_yeni_rejim.pt"),
               ("kosu4-BiLSTM", "kosu4_bilstm_yeni_rejim.pt")]
    modeller = {}
    for ad, ck in KOSULAR:
        yol = Path(a.cikti) / ck
        if not yol.exists():
            print(f"  {ad}: checkpoint yok ({ck}) -- atlaniyor")
            continue
        modeller[ad] = sarmalayici_kur(ckpt=str(yol))

    ozet = {}
    for ad, m in modeller.items():
        logit, _ = tahmin_et(m, X, cihaz=cihaz)
        print("\n" + "-" * 78)
        print(f"{ad}")
        print("-" * 78)
        print(f"  *** BOS pencereler (model bunlar icin EGITILMEDI) ***")
        b = rapor("BOS", logit, bos, a.esik)
        print(f"  --- DOLU pencereler (kontrast; egitimde goruldu) ---")
        d = rapor("DOLU", logit, ~bos, a.esik)
        if b and d:
            b["ayrim"] = d["ort_logit"] / max(b["ort_logit"], 1e-9)
            print(f"  {'':5s} >>> AYRIM: dolu/bos ortalama logit orani = "
                  f"{b['ayrim']:.2f}x   "
                  f"(bos maks {b['maks_logit']:+.3f} vs esik {a.esik})")
        ozet[ad] = b

    print("\n" + "=" * 78)
    print("SONUC")
    print("=" * 78)
    if not any(ozet.values()):
        print("  BOS pencere bulunamadi -- bu ornekte hicbiri esigi asmadi.")
        print(f"  (--bos-esik ile esik dusurulebilir; su an {a.bos_esik})")
    print(f"  {'model':<20s} {'bos maks':>9s} {'ayrim':>7s} "
          + "".join(f"{'>'+str(e):>9s}" for e in ESIKLER))
    for ad, o in ozet.items():
        if not o:
            continue
        print(f"  {ad:<20s} {o['maks_logit']:>+9.3f} "
              f"{o.get('ayrim', float('nan')):>6.2f}x "
              + "".join(f"{100*o['esikli'][e]:>8.1f}%" for e in ESIKLER))
    print(f"\n  'bos maks' = bos pencerelerde gorulen EN YUKSEK logit.")
    print(f"  Esikten KUCUKSE model bos pencerede hic alarm uretmez.")
    print(f"  'ayrim'   = dolu/bos ortalama logit orani. 1'e yakinsa model")
    print(f"  bosluk ile olayi AYIRT EDEMIYOR demektir.")
    print("\n  Bu oran yuksekse: bosluk_orani filtresi saha hattinda")
    print("  uygulanmali. ONNX bu degeri ikinci cikti olarak zaten veriyor.")
    print("  Oran, filtrenin ONLEYECEGI yanlis alarmin ust siniridir.")


if __name__ == "__main__":
    main()
