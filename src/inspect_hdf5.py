"""
GERCEK VERI -- ADIM 2b: BIR .sdf.hdf5 DOSYASINI AC

Ureticinin bicimini bilmemize GEREK YOK. HDF5 kendi kendini tarif eder:
icindeki her veri kumesinin adi, boyutu, tipi ve OZNITELIKLERI (attributes)
dosyanin icinde yazilidir. Ornekleme frekansi, olcum aralig (gauge length),
baslangic zamani gibi bilgiler neredeyse her zaman oznitelik olarak durur.

Bu script bir dosyayi acar ve icindeki HER SEYI listeler. Tek bir dosya
yeter -- yapiyi gorunce digerleri icin okuyucu yazariz.

DIKKAT: DAS dosyalari cok buyuk olabilir (GB'larca). Bu script veriyi
BELLEGE YUKLEMEZ; sadece basliklari ve cok kucuk bir kesiti okur.

JupyterLab'e yapistirmak icin yazilmistir. Gereken: h5py, numpy.
HICBIR SEY YAZMAZ, sadece okur.
"""
from pathlib import Path

import h5py
import numpy as np

MAKS_OZNITELIK = 40      # bir dugumde en fazla kac oznitelik yazdirilsin
MAKS_DUGUM = 200         # toplam kac dugum gezilsin


def bicimle(v):
    """Oznitelik degerini okunabilir, kisa bir dizeye cevir."""
    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return f"<{len(v)} bayt>"
    if isinstance(v, np.ndarray):
        if v.size <= 8:
            return np.array2string(v, precision=6, threshold=8)
        return (f"dizi{v.shape} {v.dtype}  "
                f"[{np.array2string(v.ravel()[:4], precision=4)} ...]")
    s = str(v)
    return s if len(s) <= 90 else s[:87] + "..."


def insan_boyut(n):
    for b in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or b == "TB":
            return f"{n:,.1f} {b}"
        n /= 1024


def rapor(yol, ornek_goster=True):
    yol = Path(yol)
    cizgi = "-" * 76
    print("=" * 76)
    print(f"HDF5 YAPISI  |  {yol.name}")
    print("=" * 76)

    if not yol.exists():
        print(f"  !!! Dosya bulunamadi: {yol}")
        return
    print(f"  tam yol : {yol}")
    print(f"  boyut   : {insan_boyut(yol.stat().st_size)}")

    with h5py.File(yol, "r") as f:
        # --- Kok oznitelikleri: en degerli meta veri genelde burada ---
        print(f"\n[1] KOK OZNITELIKLERI  <-- ornekleme frekansi, zaman, birim")
        print(cizgi)
        if not f.attrs:
            print("  (yok)")
        for i, (k, v) in enumerate(f.attrs.items()):
            if i >= MAKS_OZNITELIK:
                print(f"  ... ve {len(f.attrs)-MAKS_OZNITELIK} oznitelik daha")
                break
            print(f"  {str(k)[:30]:<32}{bicimle(v)}")

        # --- Tum agaci gez ---
        print(f"\n[2] ICERIK AGACI")
        print(cizgi)
        veri_kumeleri = []
        sayac = {"n": 0}

        def gez(ad, nesne):
            if sayac["n"] >= MAKS_DUGUM:
                return None
            sayac["n"] += 1
            derinlik = ad.count("/")
            girinti = "  " + "   " * derinlik
            kisa = ad.rsplit("/", 1)[-1]

            if isinstance(nesne, h5py.Group):
                print(f"{girinti}[G] {kisa}/")
            else:
                sikistirma = f" {nesne.compression}" if nesne.compression else ""
                print(f"{girinti}[D] {kisa:<26} {str(nesne.shape):>20} "
                      f"{str(nesne.dtype):>12}{sikistirma}")
                veri_kumeleri.append((ad, nesne.shape, nesne.dtype, nesne.size))

            for k, v in list(nesne.attrs.items())[:MAKS_OZNITELIK]:
                print(f"{girinti}     . {str(k)[:26]:<28}{bicimle(v)}")
            return None

        f.visititems(gez)
        if sayac["n"] >= MAKS_DUGUM:
            print(f"  ... {MAKS_DUGUM} dugum siniri asildi, kalani gosterilmedi")

        if not veri_kumeleri:
            print("\n  Hicbir veri kumesi (dataset) yok -- sadece gruplar?")
            return

        # --- En buyuk veri kumesi: asil sinyal muhtemelen bu ---
        print(f"\n[3] EN BUYUK VERI KUMELERI  <-- asil sinyal hangisi?")
        print(cizgi)
        for ad, sekil, tip, boy in sorted(veri_kumeleri,
                                          key=lambda t: -t[3])[:8]:
            bayt = boy * np.dtype(tip).itemsize
            print(f"  {insan_boyut(bayt):>12}  {str(sekil):>22} {str(tip):>10}  {ad[:34]}")

        ana_ad, ana_sekil, ana_tip, _ = max(veri_kumeleri, key=lambda t: t[3])

        # --- Ana veri kumesinden kucuk bir kesit ---
        if ornek_goster:
            print(f"\n[4] ANA VERI KUMESI: '{ana_ad}'")
            print(cizgi)
            d = f[ana_ad]
            print(f"  sekil {d.shape}   tip {d.dtype}")
            if len(d.shape) == 2:
                print(f"  -> iki boyutlu. DAS'ta bu genelde")
                print(f"     (zaman_ornegi, fiber_konumu) veya tersi demektir.")
                print(f"     {d.shape[0]:,} x {d.shape[1]:,}")
            try:
                dilim = tuple(slice(0, min(4, s)) for s in d.shape)
                kesit = np.asarray(d[dilim])
                print(f"\n  Sol ust kose {kesit.shape}:")
                print("   " + np.array2string(kesit, precision=5,
                                              max_line_width=72).replace("\n", "\n   "))
                # Istatistik icin daha genis ama yine kucuk bir kesit
                dilim2 = tuple(slice(0, min(2000, s)) for s in d.shape)
                orn = np.asarray(d[dilim2]).astype(np.float64)
                print(f"\n  Ilk {orn.shape} kesitin istatistigi:")
                print(f"     min {orn.min():+.6g}   maks {orn.max():+.6g}")
                print(f"     ort {orn.mean():+.6g}   std {orn.std():.6g}")
                sifir = float((orn == 0).mean())
                print(f"     sifir orani {sifir:.1%}")
            except Exception as e:  # noqa: BLE001
                print(f"  (kesit okunamadi: {e})")

    print(f"\n{'=' * 76}")
    print("BU CIKTIYI PAYLAS -- yapiyi gorunce okuyucu ve on isleme yazacagiz.")
    print("Oznitelikler hassas bilgi tasiyorsa (saha adi, koordinat) duzenle.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Kullanim: python inspect_hdf5.py <dosya.sdf.hdf5>")
        raise SystemExit(1)
    rapor(sys.argv[1])
