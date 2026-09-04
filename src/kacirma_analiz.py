"""
KACIRMA ANALIZI -- %31.2 kacirmanin ne kadari GERCEK kacirma?

=== NEDEN ===

Saha olcumunde model, GT kutularinin icindeki pencerelerin %31.2'sinde
alarm uretmiyor (kosu 4 + bastirma, esik 0.90). Ve dosyadan dosyaya
cok degisiyor: record_49'da %10.0, record_26'da %73.7 -- ustelik
record_26'daki oran IKI mimaride de ayni, yani modele ozgu degil.

Ama "%31.2 kacirma" demeden once bir soru var:

    GT bir ZAMAN ARALIGI isaretliyor. Bir tirmanma surekli degil --
    kisi duruyor, tutunuyor, yeniden basliyor. O araliktaki her 7.5
    saniyelik pencerede hareket olmak zorunda DEGIL.

Sessiz anlarda model dogru davranip "olay yok" diyorsa, biz bunu
kacirma sayiyoruz. Yani gercek kacirma orani %31.2'den DUSUK olabilir.

=== NE OLCULUYOR ===

GT icindeki pencereler ikiye ayriliyor (alarm ureten / uretmeyen) ve
uc acidan karsilastiriliyor:

  1. SPEKTRAL -- kacirilan pencereler bos mu? Kendi bosluk olcutlerimiz
     (bosluk_orani, dusuk_frek) ne diyor. Yuksek oranda bos cikiyorsa
     bunlar gercek kacirma degil, GT'nin zaman cozunurlugu.

  2. KONUM -- kacirmalar GT kutusunun KANAL kenarlarinda mi toplaniyor?
     Veri setinin kendisi "weak climbing" diye ayri bir etiket tasiyor,
     yani hazirlayanlar da kenarlarda sinyalin zayifladigini biliyor.

  3. ZAMAN -- kacirmalar kutunun basinda/sonunda mi? Olayin baslama ve
     bitis anlari GT'de genis isaretlenmis olabilir.

=== KULLANIM ===

    python3 kacirma_analiz.py
    python3 kacirma_analiz.py --dosya record_26 --adim 4000
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

_KOK = Path(__file__).resolve().parent.parent
for _y in (str(_KOK / "src"), str(_KOK / "CNN-BiLSTM")):
    if _y not in sys.path:
        sys.path.insert(0, _y)

import real_data as rd                                    # noqa: E402
from ham_bin import ham_ac, kanal_oku                     # noqa: E402

HAM_DIZIN = "/tf/rawData/2026_newdata/IGA_RECORDS"
KOPYA_DIZIN = "/tf/segment/Fence Benchmark Data"
CKPT = ("/tf/start_training/RELATIONNET/FENCE_DATA_NEW/"
        "egitim_ciktilari/kosu4_bilstm_yeni_rejim.pt")
SINIF = ["cutting", "climbing", "noise"]
SALDIRI = ("cutting", "climbing")


def gt_kutulari(kopya_yol):
    import h5py
    with h5py.File(kopya_yol, "r") as f:
        if "labels" not in f:
            return []
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                for r in f["labels"][:]]


def gt_gruplari(kutular, kanal_bosluk=2):
    """
    Tek kanalli GT etiketlerini FIZIKSEL OLAYLARA gruplar.

    ⚠ `labels` her satirda TEK kanal veriyor (kanal_bas == kanal_son).
    Waterfall'da cok kanalli gorunen kutular aslinda yan yana cizilmis
    AYRI etiketler. Bu fark edilmeden yazilan ilk surumde "kutu icinde
    kanal konumu" hep 0 cikti ve tum pencereler tek dilime dustu --
    analiz sessizce anlamsizdi.

    Bir olayin kanal KENARINDA mi MERKEZINDE mi oldugunu sorabilmek icin
    once gruplamak gerekiyor.

    Kural: zaman araliklari ORTUSEN ve kanallari en fazla `kanal_bosluk`
    uzaklikta olan etiketler ayni gruba girer. Zincirleme birlesmeler
    icin kararli hale gelene kadar tekrarlaniyor.
    """
    gruplar = []
    for ob, os_, kb, ks in sorted(kutular, key=lambda r: (r[2], r[0])):
        gruplar.append({"bas": ob, "son": os_, "kanal_bas": kb,
                        "kanal_son": ks, "etiket": [(ob, os_, kb, ks)]})

    degisti = True
    while degisti:
        degisti = False
        for i in range(len(gruplar)):
            for j in range(i + 1, len(gruplar)):
                a, b = gruplar[i], gruplar[j]
                zaman = a["bas"] < b["son"] and a["son"] > b["bas"]
                kanal = (a["kanal_bas"] <= b["kanal_son"] + kanal_bosluk
                         and a["kanal_son"] >= b["kanal_bas"] - kanal_bosluk)
                if zaman and kanal:
                    a["bas"] = min(a["bas"], b["bas"])
                    a["son"] = max(a["son"], b["son"])
                    a["kanal_bas"] = min(a["kanal_bas"], b["kanal_bas"])
                    a["kanal_son"] = max(a["kanal_son"], b["kanal_son"])
                    a["etiket"] += b["etiket"]
                    gruplar.pop(j)
                    degisti = True
                    break
            if degisti:
                break
    return gruplar


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


def topla(ham_dizin, kopya_dizin, adim, sarmal, cihaz, dosya_filtre=None,
          batch=64):
    """YALNIZCA GT icindeki pencereleri toplar -- kacirma analizi bu."""
    import torch
    kayitlar = []
    for hy in sorted(glob.glob(os.path.join(ham_dizin, "*.bin"))):
        ad = os.path.basename(hy)
        if dosya_filtre and dosya_filtre not in ad:
            continue
        ky = os.path.join(kopya_dizin, ad + ".hdf5")
        if not os.path.exists(ky):
            continue
        A, b = ham_ac(hy, ky)
        kutular = gt_kutulari(ky)
        if not kutular:
            continue
        gruplar = gt_gruplari(kutular)
        genislik = [g["kanal_son"] - g["kanal_bas"] + 1 for g in gruplar]
        print(f"  {ad[:44]:46s} {len(kutular):>3} etiket -> "
              f"{len(gruplar):>2} olay, kanal genisligi {genislik}", flush=True)

        tampon, meta = [], []
        for gi, g in enumerate(gruplar):
            # konum GRUBA gore -- etikete gore degil (etiketler tek kanalli)
            kmerkez = (g["kanal_bas"] + g["kanal_son"]) / 2
            kyari = max((g["kanal_son"] - g["kanal_bas"]) / 2, 0.5)
            for ob, os_, kb, ks in g["etiket"]:
                tmerkez = (ob + os_) / 2
                tyari = max((os_ - ob) / 2, 1.0)
                for kanal in range(kb, ks + 1):
                    if kanal >= b["n_kanal"]:
                        continue
                    ilk = max(0, ob - rd.PENCERE + 1)
                    for bas in range(ilk,
                                     min(os_, b["n_ornek"] - rd.PENCERE) + 1,
                                     adim):
                        son = bas + rd.PENCERE
                        if not (bas < os_ and son > ob):
                            continue
                        s = kanal_oku(A, kanal, bas, son)
                        if len(s) != rd.PENCERE:
                            continue
                        o = olcutler(s)
                        o.update(
                            dosya=ad, kanal=kanal, bas=bas, olay=gi,
                            olay_genisligi=g["kanal_son"] - g["kanal_bas"] + 1,
                            kanal_konum=(kanal - kmerkez) / kyari,
                            zaman_konum=((bas + rd.PENCERE / 2) - tmerkez) / tyari)
                        tampon.append(s.astype(np.float32))
                        meta.append(o)
                        if len(tampon) == batch:
                            _tahmin(tampon, meta, sarmal, cihaz, kayitlar)
                            tampon, meta = [], []
        if tampon:
            _tahmin(tampon, meta, sarmal, cihaz, kayitlar)
        del A
    return kayitlar


def _tahmin(tampon, meta, sarmal, cihaz, kayitlar):
    import torch
    with torch.no_grad():
        lg, _ = sarmal(torch.from_numpy(np.stack(tampon)).to(cihaz))
    lg = lg.cpu().numpy()
    for o, l in zip(meta, lg):
        o["tahmin"] = SINIF[int(l.argmax())]
        o["logit"] = float(l.max())
        kayitlar.append(o)


def rapor(kayitlar, esik):
    # Kaydin uzerine bayrak yaziliyor; `k in kacan` gibi liste uyeligi
    # sozlukleri DEGER bazinda karsilastirir -- hem O(n^2) hem kirilgan.
    for k in kayitlar:
        k["yakalandi"] = (k["tahmin"] in SALDIRI) and (k["logit"] > esik)
        k["bos_sayilir"] = (k["bosluk"] > rd.BOS_ESIK
                            or k["dusuk_frek"] > 0.9084)
    yakalanan = [k for k in kayitlar if k["yakalandi"]]
    kacan = [k for k in kayitlar if not k["yakalandi"]]
    n = len(kayitlar)
    print("\n" + "=" * 86)
    print(f"GT ICI PENCERELER: {n:,}   yakalanan {len(yakalanan):,}   "
          f"kacan {len(kacan):,}  (%{100*len(kacan)/n:.1f})")
    print("=" * 86)
    if not kacan:
        return

    # --- 1) SPEKTRAL: kacanlar bos mu? ---
    print("\n  1) KACANLAR SPEKTRAL OLARAK BOS MU?")
    print("  " + "-" * 76)
    print(f"  {'olcut':<14s} {'yakalanan':>12s} {'kacan':>12s}")
    for ad in ("bosluk", "dusuk_frek", "mad", "std"):
        y = np.mean([k[ad] for k in yakalanan]) if yakalanan else float("nan")
        c = np.mean([k[ad] for k in kacan])
        print(f"  {ad:<14s} {y:>12.4g} {c:>12.4g}")

    # asil soru: kacanlarin kaci KENDI bosluk olcutlerimize gore bos
    bos_sayilan = [k for k in kacan if k["bos_sayilir"]]
    print(f"\n  Kendi bosluk olcutlerimize gore BOS sayilan kacanlar: "
          f"{len(bos_sayilan):,} / {len(kacan):,}  "
          f"(%{100*len(bos_sayilan)/len(kacan):.1f})")
    print("  -> Bunlar GERCEK kacirma DEGIL: GT bir zaman araligi isaretliyor,")
    print("     o aralikta hareketin durdugu anlar var.")
    gercek = len(kacan) - len(bos_sayilan)
    print(f"\n  DUZELTILMIS KACIRMA: {gercek:,} / {n:,}  "
          f"(%{100*gercek/n:.1f})   [ham: %{100*len(kacan)/n:.1f}]")

    # --- 2) KONUM: kanal kenarlarinda mi? ---
    print("\n  2) KACIRMALAR GT KUTUSUNUN NERESINDE?")
    print("  " + "-" * 76)
    for ad, etiket in (("kanal_konum", "kanal ekseni"),
                       ("zaman_konum", "zaman ekseni")):
        print(f"\n  {etiket} (0 = kutu merkezi, +-1 = kenar):")
        kenarlar = np.linspace(-1, 1, 5)
        for i in range(len(kenarlar) - 1):
            a, b = kenarlar[i], kenarlar[i + 1]
            dilim = [k for k in kayitlar if a <= k[ad] < b or
                     (i == len(kenarlar) - 2 and k[ad] == b)]
            if not dilim:
                print(f"    [{a:+.1f}, {b:+.1f}) : ornek yok")
                continue
            kc = sum(1 for k in dilim if not k["yakalandi"])
            bs = sum(1 for k in dilim
                     if not k["yakalandi"] and k["bos_sayilir"])
            print(f"    [{a:+.1f}, {b:+.1f}) : n={len(dilim):>5}  "
                  f"ham kacirma %{100*kc/len(dilim):>5.1f}  "
                  f"duzeltilmis %{100*(kc-bs)/len(dilim):>5.1f}")

    # --- 3) DOSYA BAZINDA + SPEKTRAL KIRILIM ---
    #
    # record_26 iki mimaride de digerlerinin 2-3 kati kaciriyor, yani
    # sebep modelde degil o kayitta. Kacirmanin spektral imzasi
    # (dusuk_frek yuksek = zayif/sonumlenmis olay) dosya bazinda da
    # goruluyorsa, record_26'nin olaylari daha zayif demektir.
    print("\n  3) DOSYA BAZINDA -- kacirma ve GT pencerelerinin spektrumu")
    print("  " + "-" * 84)
    print(f"  {'dosya':<34s} {'n':>5s} {'ham':>7s} {'duzelt':>8s} "
          f"{'dusuk_frek':>11s} {'mad':>8s} {'olay':>5s} {'genislik':>9s}")
    for d in sorted({k["dosya"] for k in kayitlar}):
        g = [k for k in kayitlar if k["dosya"] == d]
        gk = sum(1 for k in g if not k["yakalandi"])
        gb = sum(1 for k in g if not k["yakalandi"] and k["bos_sayilir"])
        gen = sorted({k["olay_genisligi"] for k in g})
        print(f"  {d[:34]:<34s} {len(g):>5} "
              f"{100*gk/len(g):>6.1f}% {100*(gk-gb)/len(g):>7.1f}% "
              f"{np.mean([k['dusuk_frek'] for k in g]):>11.3f} "
              f"{np.mean([k['mad'] for k in g]):>8.1f} "
              f"{len({k['olay'] for k in g}):>5} "
              f"{str(gen)[:9]:>9s}")
    print("\n  dusuk_frek: 100 Hz alti enerji payi. Yuksek = zayif/sonumlenmis")
    print("  olay: gruplanmis GT olayi sayisi   genislik: kac kanala yayilmis")

    # --- 4) OLAY GENISLIGI ---
    #
    # Bir olay kac kanala yayiliyorsa o kadar guclu baglanmis demektir.
    # Dar olaylar (1-2 kanal) zayif kuplaj isareti olabilir.
    print("\n  4) OLAY KANAL GENISLIGINE GORE")
    print("  " + "-" * 76)
    for gen in sorted({k["olay_genisligi"] for k in kayitlar}):
        d = [k for k in kayitlar if k["olay_genisligi"] == gen]
        kc = sum(1 for k in d if not k["yakalandi"])
        print(f"    {gen:>2} kanal : n={len(d):>5}  kacirma %{100*kc/len(d):>5.1f}"
              f"   dusuk_frek {np.mean([k['dusuk_frek'] for k in d]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kacirma analizi")
    ap.add_argument("--ham-dizin", default=HAM_DIZIN)
    ap.add_argument("--kopya-dizin", default=KOPYA_DIZIN)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--adim", type=int, default=4000)
    ap.add_argument("--esik", type=float, default=0.9)
    ap.add_argument("--dosya", default=None, help="ad parcasi ile filtrele")
    ap.add_argument("--cihaz", default=None)
    a = ap.parse_args()

    import torch
    from onnx_disa_aktar import sarmalayici_kur
    cihaz = a.cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
    sarmal = sarmalayici_kur(ckpt=a.ckpt, bos_bastir=True).to(cihaz).eval()

    kayitlar = topla(a.ham_dizin, a.kopya_dizin, a.adim, sarmal, cihaz,
                     a.dosya)
    if not kayitlar:
        sys.exit("hic GT penceresi toplanamadi")
    rapor(kayitlar, a.esik)
