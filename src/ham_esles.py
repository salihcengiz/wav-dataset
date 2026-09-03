"""
HAM ESLESTIRME -- benchmark .hdf5 kopyalarinin ham karsiliklarini bul

=== NEDEN ===

Waterfall testinin kullandigi 17 dosyanin `/tf/segment/Fence Benchmark
Data/` altindaki `.hdf5` kopyalari OLAY CEVRESINE KIRPILMIS: her dosyada
201 (ya da 30) kanaldan yalnizca 6-14 tanesi var. Yanlis alarm ureten
kanallar tam da eksik olanlar, o yuzden sorunu bu kopyalarla olcemiyoruz.

Ama arayuz tum kanallari gosterebiliyor -- demek ki KIRPILMAMIS bir kopya
okuyor. Bir tanesini biliyoruz:

    /tf/segment/Fence Benchmark Data/record_0401143439_..._record.sdf.hdf5
    /tf/rawData/2026_newdata/Turk_Telekom_Umitkoy/ttdata/
                             record_0401143439_..._record.sdf

Yani .hdf5 uzantisi atilinca ham dosyanin adi cikiyor. Bu script o
eslemeyi 17 dosyanin hepsi icin yapiyor.

=== NASIL ===

/tf tek gecis os.walk ile taraniyor ve YALNIZCA aranan 17 adin yollari
toplaniyor. os.walk adlari bedava veriyor; pahali olan stat, o yuzden
stat sadece ESLESEN dosyalara uygulaniyor. (Ilk surum her dosyayi stat
ediyordu -- 447.606 cagri, dakikalarca surdu.)

Boyut orani onemli: ham dosya kac kat buyukse kabaca o kadar cok kanal
tasiyor demektir. 20 kat buyukse ~20 kat kanal.

=== KULLANIM ===

    python3 ham_esles.py
    python3 ham_esles.py --kaynak "/tf/segment/Fence Benchmark Data"
"""
import argparse
import os
import time
from collections import defaultdict

KAYNAK = "/tf/segment/Fence Benchmark Data"
ATLA = {"proc", "sys", "dev", "__pycache__", ".git", "site-packages",
        "node_modules", ".ipynb_checkpoints"}


def ara(kok, aranan, sure_butcesi=900, ilerleme=100000):
    """
    Yalnizca `aranan` kumesindeki ADLARI ariyor. ad -> [yol, ...]

    ⚠ Ilk surum her dosya icin os.path.getsize cagiriyordu -- 447.606
    stat islemi, dakikalarca surdu. os.walk zaten adlari BEDAVA veriyor;
    pahali olan stat. Simdi stat yalnizca ESLESEN dosyalara uygulaniyor
    (17 tane), tarama saniyeler suruyor.
    """
    basla = time.time()
    bulunan = defaultdict(list)
    n, kesildi, son_bildirim = 0, False, 0
    for dizin, altlar, adlar in os.walk(kok, followlinks=False):
        altlar[:] = [d for d in altlar if d not in ATLA and not d.startswith(".")]
        if time.time() - basla > sure_butcesi:
            kesildi = True
            break
        for ad in adlar:
            if ad in aranan:
                bulunan[ad].append(os.path.join(dizin, ad))
        n += len(adlar)
        if n - son_bildirim >= ilerleme:
            son_bildirim = n
            print(f"    {n:,} dosya... ({time.time()-basla:.0f} sn, "
                  f"{sum(len(v) for v in bulunan.values())} eslesme)",
                  flush=True)
    return bulunan, n, kesildi, time.time() - basla


def ham_adaylari(hdf5_ad):
    """
    `record_X.bin.hdf5` -> aranacak ham adlar.

    Once .hdf5 atilir (`record_X.bin`). Bazi kaynaklar farkli
    uzantiyla saklayabilir diye govde de aday olarak dondurulur.
    """
    adaylar = []
    if hdf5_ad.endswith(".hdf5"):
        adaylar.append(hdf5_ad[:-5])
    if hdf5_ad.endswith(".h5"):
        adaylar.append(hdf5_ad[:-3])
    govde = hdf5_ad
    for u in (".hdf5", ".h5", ".bin", ".sdf"):
        if govde.endswith(u):
            govde = govde[: -len(u)]
    if govde not in adaylar:
        adaylar.append(govde)
    return adaylar


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark dosyalarinin hamini bul")
    ap.add_argument("--kaynak", default=KAYNAK)
    ap.add_argument("--kok", default="/tf")
    ap.add_argument("--sure", type=int, default=900)
    a = ap.parse_args()

    hedefler = sorted(os.listdir(a.kaynak))
    print(f"benchmark dosyasi: {len(hedefler)}  ({a.kaynak})")

    # aranacak ad kumesi -- walk sirasinda O(1) kontrol
    aranan = set()
    for h in hedefler:
        aranan.update(ham_adaylari(h))
    print(f"aranan ad: {len(aranan)}")

    print(f"taraniyor: {a.kok} ...", flush=True)
    ix, n, kesildi, gecen = ara(a.kok, aranan, a.sure)
    print(f"  {n:,} dosya tarandi, {gecen:.0f} sn, "
          f"{sum(len(v) for v in ix.values())} eslesme"
          + ("   *** SURE DOLDU, EKSIK ***" if kesildi else ""))

    print("\n" + "=" * 100)
    print("ESLESME")
    print("=" * 100)
    print(f"  {'benchmark dosyasi':<46s} {'kopya':>9s} {'ham':>10s} "
          f"{'kat':>6s}  ham yol")
    print("  " + "-" * 96)

    bulunan, eksik = [], []
    for h in hedefler:
        yol_h = os.path.join(a.kaynak, h)
        try:
            b_h = os.path.getsize(yol_h)
        except OSError:
            continue
        vurus = []
        for aday in ham_adaylari(h):
            for y in ix.get(aday, []):
                if os.path.abspath(y) == os.path.abspath(yol_h):
                    continue
                try:                      # stat YALNIZCA eslesenlere
                    vurus.append((os.path.getsize(y), y))
                except OSError:
                    vurus.append((-1, y))
        if vurus:
            b, y = max(vurus)
            kat = b / b_h if b_h else 0
            bulunan.append((h, b_h, b, y))
            print(f"  {h[:46]:<46s} {b_h/1e6:>8.1f}M {b/1e6:>9.1f}M "
                  f"{kat:>6.1f}x  {y}")
            for b2, y2 in sorted(vurus, reverse=True)[1:]:
                print(f"  {'':<46s} {'':>8s}  {b2/1e6:>9.1f}M "
                      f"{'':>6s}  {y2}")
        else:
            eksik.append(h)
            print(f"  {h[:46]:<46s} {b_h/1e6:>8.1f}M {'--':>9s} "
                  f"{'--':>6s}  BULUNAMADI")

    print("\n" + "=" * 100)
    print(f"BULUNAN: {len(bulunan)}/{len(hedefler)}   EKSIK: {len(eksik)}")
    if bulunan:
        dizinler = sorted({os.path.dirname(y) for _, _, _, y in bulunan})
        print("\n  ham dosyalarin bulundugu dizinler:")
        for d in dizinler:
            print(f"    {d}")
        ort = sum(b / bh for _, bh, b, _ in bulunan) / len(bulunan)
        print(f"\n  ortalama boyut orani: {ort:.1f}x")
        print("  -> ham dosyalar kabaca bu kat kadar cok kanal tasiyor.")
        print("     Kirpilmis kopyalarda olmayan kanallar burada.")
    if eksik:
        print(f"\n  eslesmeyenler ({len(eksik)}):")
        for h in eksik:
            print(f"    {h}")
