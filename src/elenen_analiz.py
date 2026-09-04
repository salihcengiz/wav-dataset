"""
ELENEN PENCERE ANALIZI -- sildigimiz veri, simdi kacirdigimiz veri mi?

=== NEDEN ===

Faz 0'da `bos_mu` ile egitim setinin %23'unu eledik. O zaman yazilan
docstring (real_data.bos_mu) sunu diyordu:

    "olay etiketli pencerelerin ~%25'i (climbing %20, cutting %28)
     spektral olarak bos. Bunlar MUHTEMELEN OLAYDAN UZAK KANALLAR --
     labels tablosu kanal ARALIGI veriyor, aralik kenarindaki
     kanallarda sinyal zayiflamis olabilir."

2026-09-04'te benchmark verisinde olculdu (`kacirma_analiz.py`):

    kacirma olayin KANAL KENARLARINDA merkezin 3 KATI
    (%37.5 / %31.7 kenar,  %11.7 merkez)
    kacirilan pencereler dusuk frekans baskin: 0.802
    yakalanan pencereler                     : 0.587

Yani aylar once hipotezi yazmisiz, sonra o pencereleri elemisiz, simdi
tam onlarin uzerine dusuyoruz. Bu script zinciri KAPATIYOR:

    Eledigimiz egitim pencereleri, simdi kacirdigimiz benchmark
    pencereleriyle AYNI spektral imzaya ve AYNI konumsal desene
    sahip mi?

Evetse nedensellik tamamlanir ve cozum bellidir: o pencereleri
egitime geri koymak.

=== NEDEN SAYIM YETMEZ ===

"Elenen pencerelerin kaci etiketli olayin icinde" sorusunun cevabi
tanim geregi %100 -- CSV'nin her satiri zaten etiketli bir pencere.
Anlamli soru, elenenlerin NE OLDUGU.

=== ORNEKLEME ===

Rastgele SATIR degil, rastgele OLAY secililiyor: ayni (dosya, event,
window_start, window_end) degerlerini paylasan satirlar bir olayin
farkli KANALLARI. Kanal kenari analizi ancak olayin tum kanallari
elde olunca yapilabilir.

=== KULLANIM ===

    python3 elenen_analiz.py
    python3 elenen_analiz.py --olay 400 --csv train_final.csv
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

# kacirma_analiz.py'nin benchmark verisinde olctugu referans degerler
REF_YAKALANAN = 0.587      # tespit edilen gercek olay pencereleri
REF_KACAN = 0.802          # KACIRILAN gercek olay pencereleri
REF_ARTEFAKT = 0.956       # olu/suruklenmeli kanallar


def olay_sec(csv_yol, n_olay, tohum=42):
    """
    Rastgele OLAY sec, satir degil.

    Ayni (file, event, window_start, window_end) degerlerini paylasan
    satirlar bir olayin farkli kanallari. Kanal kenari analizi icin
    olayin TUM kanallari gerekiyor.
    """
    df = pd.read_csv(csv_yol)
    anahtar = ["file", "event", "window_start", "window_end"]
    gruplar = df.groupby(anahtar, sort=False)
    anahtarlar = list(gruplar.groups.keys())
    rng = np.random.default_rng(tohum)
    idx = rng.permutation(len(anahtarlar))[:n_olay]
    secilen = pd.concat([gruplar.get_group(anahtarlar[i]) for i in idx])
    return secilen.reset_index(drop=True), len(df), len(anahtarlar)


def olcutler(s):
    x = s - s.mean()
    G = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / rd.FS)
    top = G.sum()
    med = np.median(s)
    return {
        "bosluk": rd.bosluk_orani(s),
        "dusuk_frek": float(G[f < 100].sum() / top) if top > 0 else 1.0,
        "mad": float(np.median(np.abs(s - med))),
        "std": float(s.std()),
    }


def yukle(df, bos_esik=None):
    """
    bos_esik: rd.BOS_ESIK'i ezer. Iki ise yariyor --
      (a) sahte veride hic bos pencere yok, `elendi` kod yolu ancak
          esik dusurulerek sinanabiliyor
      (b) sunucuda esik duyarliligi: 0.45 yerine 0.35/0.55 ile ne kadar
          farkli bir populasyon elenirdi
    """
    import h5py
    esik = rd.BOS_ESIK if bos_esik is None else float(bos_esik)
    kayitlar, hata = [], 0
    for dosya, grup in df.groupby(df["file"].astype(str), sort=False):
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
                    except Exception:               # noqa: BLE001
                        s = None
                    if s is None or len(s) != rd.PENCERE:
                        hata += 1
                        continue
                    o = olcutler(s)
                    o.update(dosya=os.path.basename(dosya),
                             kanal=int(r.channel), event=str(r.event),
                             bas=int(r.window_start), son=int(r.window_end),
                             elendi=bool(o["bosluk"] > esik))
                    kayitlar.append(o)
        except OSError:
            hata += len(grup)
    return kayitlar, hata


def kanal_konumu(kayitlar):
    """Her pencereye, kendi olayinin kanal araligindaki konumunu yazar."""
    from collections import defaultdict
    olaylar = defaultdict(list)
    for k in kayitlar:
        olaylar[(k["dosya"], k["event"], k["bas"], k["son"])].append(k)
    for grup in olaylar.values():
        kn = [g["kanal"] for g in grup]
        merkez, yari = (min(kn) + max(kn)) / 2, max((max(kn) - min(kn)) / 2, 0.5)
        for g in grup:
            g["kanal_konum"] = (g["kanal"] - merkez) / yari
            g["olay_genisligi"] = max(kn) - min(kn) + 1
    return olaylar


def rapor(kayitlar, olaylar):
    n = len(kayitlar)
    elenen = [k for k in kayitlar if k["elendi"]]
    kalan = [k for k in kayitlar if not k["elendi"]]
    print("\n" + "=" * 86)
    print(f"PENCERE: {n:,}   ELENEN: {len(elenen):,} (%{100*len(elenen)/n:.1f})"
          f"   KALAN: {len(kalan):,}   OLAY: {len(olaylar):,}")
    print("=" * 86)

    # --- 1) SINIF BAZINDA ELEME ORANI ---
    print("\n  1) SINIF BAZINDA ELEME  (Faz 0: climbing %20, cutting %28, noise %0)")
    print("  " + "-" * 76)
    print(f"  {'sinif':<12s} {'toplam':>8s} {'elenen':>8s} {'oran':>8s}")
    for s in sorted({k["event"] for k in kayitlar}):
        g = [k for k in kayitlar if k["event"] == s]
        e = sum(1 for k in g if k["elendi"])
        print(f"  {s:<12s} {len(g):>8,} {e:>8,} {100*e/len(g):>7.1f}%")

    if not elenen:
        print("\n  hic elenen pencere yok -- karsilastirma yapilamiyor")
        return

    # --- 2) SPEKTRAL IMZA: elenen vs kalan vs BENCHMARK referanslari ---
    print("\n  2) SPEKTRAL IMZA -- eledigimiz veri neye benziyor?")
    print("  " + "-" * 76)
    print(f"  {'olcut':<14s} {'KALAN':>10s} {'ELENEN':>10s}")
    for ad in ("dusuk_frek", "bosluk", "mad", "std"):
        print(f"  {ad:<14s} {np.mean([k[ad] for k in kalan]):>10.4g} "
              f"{np.mean([k[ad] for k in elenen]):>10.4g}")

    df_elenen = np.mean([k["dusuk_frek"] for k in elenen])
    df_kalan = np.mean([k["dusuk_frek"] for k in kalan])
    print(f"\n  dusuk_frek ekseninde karsilastirma:")
    print(f"    egitimde KALAN            {df_kalan:.3f}")
    print(f"    benchmark'ta YAKALANAN    {REF_YAKALANAN:.3f}   <- referans")
    print(f"    benchmark'ta KACIRILAN    {REF_KACAN:.3f}   <- referans")
    print(f"    egitimde ELENEN           {df_elenen:.3f}")
    print(f"    benchmark ARTEFAKT        {REF_ARTEFAKT:.3f}   <- referans")
    yakin = min([("yakalanan", REF_YAKALANAN), ("kacirilan", REF_KACAN),
                 ("artefakt", REF_ARTEFAKT)],
                key=lambda t: abs(t[1] - df_elenen))
    print(f"\n  -> Eledigimiz pencereler en cok '{yakin[0]}' grubuna benziyor.")
    if yakin[0] == "kacirilan":
        print("     ZINCIR KAPANDI: sildigimiz populasyonun uzerine dusuyoruz.")

    # --- 3) KANAL KONUMU: elenenler kenarda mi? ---
    print("\n  3) ELENENLER OLAYIN NERESINDE?")
    print("  " + "-" * 76)
    cok_kanalli = [k for k in kayitlar if k["olay_genisligi"] >= 3]
    if not cok_kanalli:
        print("    3+ kanalli olay yok -- konum analizi yapilamiyor")
    else:
        kenarlar = np.linspace(-1, 1, 5)
        for i in range(len(kenarlar) - 1):
            a, b = kenarlar[i], kenarlar[i + 1]
            dilim = [k for k in cok_kanalli
                     if a <= k["kanal_konum"] < b
                     or (i == len(kenarlar) - 2 and k["kanal_konum"] == b)]
            if not dilim:
                print(f"    [{a:+.1f}, {b:+.1f}) : ornek yok")
                continue
            e = sum(1 for k in dilim if k["elendi"])
            etiket = " kenar" if abs((a + b) / 2) > 0.5 else " merkez"
            print(f"    [{a:+.1f}, {b:+.1f}){etiket:>8s} : n={len(dilim):>6}  "
                  f"eleme %{100*e/len(dilim):>5.1f}")
        print("\n  Benchmark'ta KACIRMA: kenar %37.5/%31.7, merkez %11.7")
        print("  Ayni desen cikarsa iki olgu ayni populasyonun iki yuzu.")

    # --- 4) OLAY GENISLIGI ---
    print("\n  4) OLAY GENISLIGINE GORE ELEME")
    print("  " + "-" * 76)
    for gen in sorted({k["olay_genisligi"] for k in kayitlar}):
        d = [k for k in kayitlar if k["olay_genisligi"] == gen]
        if len(d) < 20:
            continue
        e = sum(1 for k in d if k["elendi"])
        print(f"    {gen:>2} kanal : n={len(d):>6}  eleme %{100*e/len(d):>5.1f}"
              f"   dusuk_frek {np.mean([k['dusuk_frek'] for k in d]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Elenen pencere analizi")
    ap.add_argument("--veri", default=str(VERI))
    ap.add_argument("--csv", default="*train*.csv")
    ap.add_argument("--olay", type=int, default=400,
                    help="kac OLAY orneklensin (satir degil)")
    ap.add_argument("--bos-esik", type=float, default=None,
                    help=f"bosluk_orani esigini ez (varsayilan {rd.BOS_ESIK})")
    a = ap.parse_args()

    csvler = sorted(glob.glob(str(Path(a.veri) / a.csv)))
    if not csvler:
        sys.exit(f"CSV bulunamadi: {a.veri}/{a.csv}")
    print(f"CSV: {csvler[0]}")

    df, n_satir, n_olay = olay_sec(csvler[0], a.olay)
    print(f"CSV'de {n_satir:,} satir, {n_olay:,} olay. "
          f"Ornek: {a.olay} olay -> {len(df):,} satir")

    kayitlar, hata = yukle(df, a.bos_esik)
    esik = rd.BOS_ESIK if a.bos_esik is None else a.bos_esik
    ek = "" if a.bos_esik is None else "  (varsayilan EZILDI)"
    print(f"yuklenen: {len(kayitlar):,} pencere  (okunamayan {hata:,})   "
          f"bos_esik={esik}{ek}")
    if not kayitlar:
        sys.exit("hic pencere yuklenemedi")
    olaylar = kanal_konumu(kayitlar)
    rapor(kayitlar, olaylar)
