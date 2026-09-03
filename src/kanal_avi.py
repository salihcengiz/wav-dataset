"""
KANAL AVI -- /tf altinda COK KANALLI kayit dosyasi var mi?

=== NEDEN ===

Waterfall'daki yanlis alarmlar, olaydan uzak / olu kanallarda. Ama
elimizdeki HICBIR dosyada o kanallar yok:

    benchmark  record_26...bin.hdf5   attr channels=201, DISKTE 11
    egitim     record_lokasyon2...    attr channels=1956, DISKTE 16

Ikisi de olay cevresine kirpilmis. Yani modele "bu tur pencereler arka
plandir" diye ogretmek isteseydik bile OGRETECEK ORNEK YOK. Test arayuzu
ise 201 kanalin hepsini gosterebiliyor -- demek ki bir yerde tam kanalli
kopya var.

Bu script onu ariyor.

=== NASIL ===

Iki asama, cunku her HDF5'i acmak pahali:

  1. os.walk ile dosyalari ve BOYUTLARINI topla (hizli). Boyut kanal
     sayisiyla dogru orantili: 200 kanal x 56k ornek x 4 bayt ~ 45 MB,
     11 kanal ~ 2.5 MB. Yani buyuk dosyalar aday.
  2. En buyuk N dosyayi ac, DISKTEKI kanal veri kumesi sayisini attr'daki
     `channels` ile karsilastir. Oran 1'e yakinsa TAM KANALLI kopyadir.

=== KULLANIM ===

    python3 kanal_avi.py
    python3 kanal_avi.py --kok /tf --ac 40 --sure 600
"""
import argparse
import os
import sys
import time
from collections import defaultdict

UZANTILAR = (".hdf5", ".h5", ".sdf", ".bin")
ATLA = {"proc", "sys", "dev", "__pycache__", ".git", "site-packages",
        "node_modules", ".ipynb_checkpoints"}


def tara(kok, sure_butcesi):
    """os.walk -- dosya yolu + boyut. Acmiyor, sadece listeliyor."""
    basla = time.time()
    dosyalar = []
    kesildi = False
    for dizin, altlar, adlar in os.walk(kok, followlinks=False):
        altlar[:] = [d for d in altlar if d not in ATLA and not d.startswith(".")]
        if time.time() - basla > sure_butcesi:
            kesildi = True
            break
        for ad in adlar:
            if ad.lower().endswith(UZANTILAR):
                y = os.path.join(dizin, ad)
                try:
                    dosyalar.append((os.path.getsize(y), y))
                except OSError:
                    pass
    return dosyalar, kesildi, time.time() - basla


def dizin_ozeti(dosyalar, en_fazla=25):
    grup = defaultdict(lambda: [0, 0, 0])       # n, toplam, maks
    for boyut, y in dosyalar:
        g = grup[os.path.dirname(y)]
        g[0] += 1
        g[1] += boyut
        g[2] = max(g[2], boyut)
    print("\n" + "=" * 96)
    print("DIZIN OZETI  (en buyuk dosyaya gore sirali)")
    print("=" * 96)
    print(f"  {'dizin':<62s} {'dosya':>6s} {'toplam':>10s} {'en buyuk':>10s}")
    print("  " + "-" * 92)
    for d, (n, top, mak) in sorted(grup.items(), key=lambda x: -x[1][2])[:en_fazla]:
        print(f"  {d[-62:]:<62s} {n:>6} {top/1e6:>9.0f}M {mak/1e6:>9.1f}M")
    return grup


def kanal_say(yol):
    """Diskteki kanal veri kumesi sayisi + attr'daki channels."""
    import h5py
    with h5py.File(yol, "r") as f:
        a = dict(f.attrs)
        kn = sorted(int(k) for k in f.keys() if str(k).isdigit())
        d = f[str(kn[0])] if kn else None
        n_ornek = d.shape[0] if d is not None else 0
        return {
            "attr_kanal": int(a.get("channels", 0)),
            "diskte": len(kn),
            "aralik": (kn[0], kn[-1]) if kn else (0, 0),
            "ornek": n_ornek,
            "prf": a.get("prf"),
            "duration": a.get("duration"),
            "dtype": str(d.dtype) if d is not None else "?",
        }


def ac_ve_bak(dosyalar, kac, esik_kanal=30):
    """
    En buyuk `kac` dosyayi acip kanal dokumunu cikarir.

    Isaretleme yalnizca DISKTEKI kanal sayisina bakiyor; `channels`
    ozniteligine degil. Oran bilgi amacli -- bazi dosyalarda o oznitelik
    hic yok ve orana bagli bir kosul o dosyalari sessizce atlardi.
    """
    print("\n" + "=" * 96)
    print(f"EN BUYUK {kac} DOSYA -- diskteki kanal sayisi")
    print("=" * 96)
    print(f"  {'dosya':<50s} {'MB':>7s} {'attr':>6s} {'diskte':>7s} "
          f"{'oran':>6s} {'aralik':>14s} {'fs':>7s}")
    print("  " + "-" * 92)
    tam_kanalli, acilamayan = [], []
    for boyut, y in sorted(dosyalar, reverse=True)[:kac]:
        try:
            o = kanal_say(y)
        except Exception as e:              # noqa: BLE001
            # ACILAMAYAN DOSYA HAKKINDA HICBIR SEY BILEMEYIZ.
            # Ilk surumde bunlar sessizce atlaniyordu ve script "cok
            # kanalli dosya yok" diye sonuc basiyordu -- oysa en buyuk
            # 40 dosyanin 38'i acilamamisti. Ayri sayiliyor.
            acilamayan.append((boyut, y, type(e).__name__))
            print(f"  {os.path.basename(y)[:50]:<50s} {boyut/1e6:>6.1f}M  "
                  f"ACILAMADI ({type(e).__name__}) -- HDF5 degil?")
            continue
        oran = o["diskte"] / o["attr_kanal"] if o["attr_kanal"] else 0
        fs = (o["ornek"] / float(o["duration"])) if o["duration"] else 0
        isaret = ""
        if o["diskte"] >= esik_kanal:
            isaret = "  <-- COK KANALLI"
            tam_kanalli.append((y, o))
        print(f"  {os.path.basename(y)[:50]:<50s} {boyut/1e6:>6.1f}M "
              f"{o['attr_kanal']:>6} {o['diskte']:>7} {oran:>6.2f} "
              f"{str(o['aralik']):>14s} {fs:>7.0f}{isaret}")
    return tam_kanalli, acilamayan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cok kanalli kayit dosyasi ara")
    ap.add_argument("--kok", default="/tf")
    ap.add_argument("--ac", type=int, default=30, help="kac dosya acilsin")
    ap.add_argument("--sure", type=int, default=300, help="tarama sn butcesi")
    ap.add_argument("--dizin-sayisi", type=int, default=25)
    ap.add_argument("--esik-kanal", type=int, default=30,
                    help="kac kanaldan fazlasi 'cok kanalli' sayilsin")
    a = ap.parse_args()

    print(f"taraniyor: {a.kok}   (en fazla {a.sure} sn)")
    dosyalar, kesildi, gecen = tara(a.kok, a.sure)
    print(f"bulunan: {len(dosyalar):,} dosya   sure {gecen:.0f} sn"
          + ("   *** SURE DOLDU, tarama EKSIK ***" if kesildi else ""))
    if not dosyalar:
        sys.exit("hic dosya bulunamadi")

    dizin_ozeti(dosyalar, a.dizin_sayisi)
    tam, acilamayan = ac_ve_bak(dosyalar, a.ac, a.esik_kanal)

    print("\n" + "=" * 96)
    print("SONUC")
    print("=" * 96)
    if tam:
        print(f"{len(tam)} COK KANALLI (ve okunabilir) dosya bulundu:")
        for y, o in tam[:10]:
            print(f"  {y}")
            print(f"     diskte {o['diskte']} kanal "
                  f"({o['aralik'][0]}..{o['aralik'][1]}), "
                  f"{o['ornek']} ornek, dtype {o['dtype']}")
        print("\n  Bunlar arka plan / olu kanal ornegi icerebilir --")
        print("  yanlis alarm populasyonunu bunlarda olcebiliriz.")

    if acilamayan:
        print(f"\n  ⚠ {len(acilamayan)} dosya ACILAMADI (incelenen "
              f"{min(a.ac, len(dosyalar))} dosyanin buyuk kismi).")
        print("  Bunlar HAM .bin/.sdf -- HDF5 degil, h5py okuyamiyor.")
        print("  ACILAMAYAN DOSYA HAKKINDA HICBIR SEY BILINMEZ:")
        print("  kac kanal tasidiklari, arka plan icerip icermedikleri")
        print("  BILINMIYOR. 'Cok kanalli dosya yok' SONUCU CIKARILAMAZ.")
        print(f"\n  Toplam {sum(b for b, _, _ in acilamayan)/1e9:.0f} GB ham veri.")
        print("  Sonraki adim: bu formati okumak. Ekibin .hdf5 kopyalari")
        print("  bunlardan uretilmis, yani bir donusturucu VAR:")
        print("    grep -rl 'sdf\\|RealUInt16\\|pulsewidth' /tf --include=*.py")

    if not tam and not acilamayan:
        print("Ne cok kanalli ne de acilamayan dosya var.")
