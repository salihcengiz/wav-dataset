"""
HAM ANALIZ -- yanlis alarm veren kanallari SONUNDA olcebiliyoruz

=== NEDEN BU SCRIPT VAR ===

Saha waterfall'inda tek kanalda, dakikalarca suren, sabit sinifli
sutunlar var (kanal 0'da kesintisiz `cutting` gibi). Bunlari haftalardir
olcemedik cunku `.hdf5` kopyalari olay cevresine kirpilmis: record_26'da
201 kanaldan 11'i. Ve o 11 kanalda model KUSURSUZ -- GT disi pencerelerde
yanlis alarm %0.0. Yani belirtiyi goruyorduk, sebebini olcemiyorduk.

2026-09-02'de ham format cozuldu (`ham_bin.py`, kopyayla dogrulandi,
maks fark 0). Artik 201 kanalin hepsini okuyabiliyoruz.

=== NE OLCULUYOR ===

Her (kanal, pencere) icin: model tahmini + logit, bosluk_orani,
dusuk_frek, MAD. GT `.hdf5` kopyasindaki `labels`'tan.

Uc rapor:

  1. KANAL BAZINDA -- hangi kanallar GT disinda alarm uretiyor, ve
     tahminleri ne kadar TUTARLI. Yuksek alarm + yuksek tutarlilik =
     artefakt imzasi (gercek gurultu cesitlenir, olu kanal takilir).

  2. OLCUT KARSILASTIRMASI -- artefakt kanallarini hangi buyukluk
     ayiriyor. Bu sefer GT-disi ornegi 53 degil binlerce.

  3. TAKAS -- esik egrisi, tum kanallarda.

=== KULLANIM ===

    python3 ham_analiz.py
    python3 ham_analiz.py --adim 8000 --kanal-adim 2   # hizli gecis
"""
import argparse
import glob
import os
import sys
from collections import Counter
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
        "egitim_ciktilari/kosu3_gri_sifirdan.pt")
SINIF = ["cutting", "climbing", "noise"]
SALDIRI = ("cutting", "climbing")


def gt_kutulari(kopya_yol):
    import h5py
    with h5py.File(kopya_yol, "r") as f:
        if "labels" not in f:
            return []
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                for r in f["labels"][:]]


def gt_mi(bas, son, kanal, kutular):
    return any(bas < os_ and son > ob and kb <= kanal <= ks
               for ob, os_, kb, ks in kutular)


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


def topla(ham_dizin, kopya_dizin, adim, kanal_adim, sarmal, cihaz,
          n_dosya=None, batch=64):
    import torch
    kayitlar = []
    hamlar = sorted(glob.glob(os.path.join(ham_dizin, "*.bin")))[:n_dosya]
    print(f"ham dosya: {len(hamlar)}")

    for hy in hamlar:
        ky = os.path.join(kopya_dizin, os.path.basename(hy) + ".hdf5")
        if not os.path.exists(ky):
            print(f"  {os.path.basename(hy)}: kopya yok, atlandi")
            continue
        A, b = ham_ac(hy, ky)
        kutular = gt_kutulari(ky)
        kanallar = range(0, b["n_kanal"], kanal_adim)
        print(f"  {os.path.basename(hy)[:44]:46s} "
              f"{b['n_kanal']} kanal x {b['n_ornek']} ornek, "
              f"{len(kutular)} GT kutusu", flush=True)

        tampon, meta = [], []
        for kanal in kanallar:
            for bas in range(0, b["n_ornek"] - rd.PENCERE + 1, adim):
                s = kanal_oku(A, kanal, bas, bas + rd.PENCERE)
                o = olcutler(s)
                o.update(dosya=os.path.basename(hy), kanal=kanal, bas=bas,
                         gt=gt_mi(bas, bas + rd.PENCERE, kanal, kutular))
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


def alarm(ks, esik):
    return np.array([(k["tahmin"] in SALDIRI) and (k["logit"] > esik)
                     for k in ks])


def kanal_raporu(kayitlar, esik, en_fazla=40):
    print("\n" + "=" * 92)
    print("KANAL BAZINDA -- GT DISI PENCERELER  (alarm oranina gore sirali)")
    print("=" * 92)
    satirlar = []
    for kn in sorted({k["kanal"] for k in kayitlar}):
        d = [k for k in kayitlar if k["kanal"] == kn and not k["gt"]]
        if len(d) < 3:
            continue
        a = alarm(d, esik)
        t = [k["tahmin"] for k in d]
        baskin, nb = Counter(t).most_common(1)[0]
        satirlar.append((a.mean(), nb / len(t), kn, len(d), baskin,
                         np.mean([k["bosluk"] for k in d]),
                         np.mean([k["dusuk_frek"] for k in d]),
                         np.mean([k["mad"] for k in d])))
    satirlar.sort(reverse=True)
    print(f"  {'kanal':>5} {'n':>5} {'alarm%':>8} {'tutarli%':>9} "
          f"{'baskin':>9} {'bosluk':>8} {'dusuk_fr':>9} {'mad':>9}")
    print("  " + "-" * 86)
    for al, tut, kn, n, bas, bo, df, md in satirlar[:en_fazla]:
        isaret = "  <-- ARTEFAKT" if al > 0.3 and tut > 0.8 else ""
        print(f"  {kn:>5} {n:>5} {100*al:>7.1f}% {100*tut:>8.1f}% "
              f"{bas:>9} {bo:>8.3f} {df:>9.3f} {md:>9.1f}{isaret}")
    supheli = [s for s in satirlar if s[0] > 0.3 and s[1] > 0.8]
    temiz = [s for s in satirlar if s[0] == 0.0]
    print(f"\n  toplam {len(satirlar)} kanal | ARTEFAKT supheli "
          f"{len(supheli)} | hic alarm uretmeyen {len(temiz)}")
    return [s[2] for s in supheli]


def olcut_karsilastir(kayitlar, supheli_kanallar):
    """Artefakt kanallarini digerlerinden hangi buyukluk ayiriyor?"""
    if not supheli_kanallar:
        print("\n  artefakt kanal yok -- olcut karsilastirmasi atlandi")
        return
    print("\n" + "=" * 92)
    print("OLCUT -- artefakt kanallari digerlerinden ayirabiliyor mu?")
    print("=" * 92)
    A = [k for k in kayitlar if k["kanal"] in set(supheli_kanallar) and not k["gt"]]
    B = [k for k in kayitlar if k["gt"]]
    print(f"  artefakt pencereleri: {len(A):,}   gercek olay: {len(B):,}\n")
    print(f"  {'olcut':<14s} {'artefakt':>12s} {'gercek olay':>13s} "
          f"{'esik':>10s} {'ELENEN':>9s}")
    print("  " + "-" * 62)
    for ad, yon in [("bosluk", "yuksek"), ("dusuk_frek", "yuksek"),
                    ("mad", "dusuk"), ("std", "dusuk")]:
        a = np.array([k[ad] for k in A]); b = np.array([k[ad] for k in B])
        if len(a) == 0 or len(b) == 0:
            continue
        if yon == "dusuk":
            esik = np.quantile(b, 0.05); elenen = (a < esik).mean()
        else:
            esik = np.quantile(b, 0.95); elenen = (a > esik).mean()
        print(f"  {ad:<14s} {a.mean():>12.4g} {b.mean():>13.4g} "
              f"{esik:>10.4g} {100*elenen:>8.1f}%")
    print("\n  ELENEN = gercek olaylarin %95'ini koruyan esikte,")
    print("  artefakt pencerelerinin yuzde kaci susturulur.")


def takas(kayitlar, esikler=(0.0, 0.5, 0.75, 0.9, 1.1, 1.3, 1.5, 2.0)):
    ici = [k for k in kayitlar if k["gt"]]
    disi = [k for k in kayitlar if not k["gt"]]
    if not ici or not disi:
        return
    print("\n" + "=" * 92)
    print(f"TAKAS  (GT-ici {len(ici):,}  GT-disi {len(disi):,})")
    print("=" * 92)
    print(f"  {'esik':>6} {'KACIRMA':>9} {'Y.ALARM':>9}")
    print("  " + "-" * 30)
    for e in esikler:
        print(f"  {e:>6.2f} {100*(1-alarm(ici, e).mean()):>8.1f}% "
              f"{100*alarm(disi, e).mean():>8.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ham dosyalarda tam kanal analizi")
    ap.add_argument("--ham-dizin", default=HAM_DIZIN)
    ap.add_argument("--kopya-dizin", default=KOPYA_DIZIN)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--adim", type=int, default=8000)
    ap.add_argument("--kanal-adim", type=int, default=1)
    ap.add_argument("--esik", type=float, default=0.9)
    ap.add_argument("--dosya", type=int, default=None, help="ilk N dosya")
    ap.add_argument("--cihaz", default=None)
    a = ap.parse_args()

    import torch
    from onnx_disa_aktar import sarmalayici_kur
    cihaz = a.cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
    # bastirma KAPALI -- modelin ham davranisini olcuyoruz
    sarmal = sarmalayici_kur(ckpt=a.ckpt, bos_bastir=False).to(cihaz).eval()

    kayitlar = topla(a.ham_dizin, a.kopya_dizin, a.adim, a.kanal_adim,
                     sarmal, cihaz, a.dosya)
    if not kayitlar:
        sys.exit("hic pencere toplanamadi")
    gt = sum(k["gt"] for k in kayitlar)
    print(f"\nPENCERE: {len(kayitlar):,}   GT-ici: {gt:,}   "
          f"GT-disi: {len(kayitlar)-gt:,}")

    supheli = kanal_raporu(kayitlar, a.esik)
    olcut_karsilastir(kayitlar, supheli)
    takas(kayitlar)
