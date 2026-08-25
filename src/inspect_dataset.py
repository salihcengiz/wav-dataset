"""
GERCEK VERI -- ADIM 1: ENVANTER (kesif)

Uzak sunucuda, JupyterLab hucresine YAPISTIRILARAK calistirilmak uzere
yazilmistir. Bu yuzden:
  - HICBIR harici kutuphane kullanmaz (os, pathlib, collections -- hepsi standart)
  - Bu depodaki hicbir modulu import etmez (uzak sunucuda depo yok)
  - HICBIR SEY YAZMAZ, silmez, degistirmez -- sadece okur
  - Ciktisi kasten kisa tutulur, sohbete yapistirilabilsin diye

Amaci "veriyi analiz etmek" DEGIL, "orada ne oldugunu ogrenmek". Bicimi
gordukten sonra ona ozel bir okuyucu yazilacak (ADIM 2).

Kullanim:
    KOK = "/veri/klasoru/yolu"      <-- burayi degistir
    ...sonra hucreyi calistir
"""
import os
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------
# AYAR -- sadece burayi degistir
# ---------------------------------------------------------------
KOK = "."          # incelenecek klasorun yolu

MAKS_DOSYA = 300_000    # guvenlik siniri: bundan fazlasini tarama
AGAC_DERINLIK = 3       # klasor agacinda kac seviye gosterilsin
ORNEK_AD = 3            # her uzanti icin kac ornek dosya adi


def insan_boyut(n):
    for birim in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or birim == "TB":
            return f"{n:,.1f} {birim}"
        n /= 1024


def tara(kok):
    """Klasoru gez, dosyalari topla. Erisim hatalarini yut."""
    kok = Path(kok)
    dosyalar = []
    hatalar = []
    kesildi = False

    for dizin, alt_dizinler, adlar in os.walk(kok, onerror=hatalar.append):
        alt_dizinler[:] = [d for d in alt_dizinler
                           if not d.startswith(".") and d != "__pycache__"]
        for ad in adlar:
            if ad.startswith("."):
                continue
            p = Path(dizin) / ad
            try:
                boyut = p.stat().st_size
            except OSError as e:
                hatalar.append(e)
                continue
            dosyalar.append((p, boyut))
            if len(dosyalar) >= MAKS_DOSYA:
                kesildi = True
                break
        if kesildi:
            break
    return kok, dosyalar, hatalar, kesildi


def rapor(kok_yolu=KOK):
    kok, dosyalar, hatalar, kesildi = tara(kok_yolu)
    cizgi = "-" * 76

    print("=" * 76)
    print(f"ENVANTER  |  {kok.resolve() if kok.exists() else kok}")
    print("=" * 76)

    if not kok.exists():
        print("  !!! BU YOL BULUNAMADI.")
        print("  Dogru yolu bulmak icin:")
        print("     import os; print(os.getcwd()); print(os.listdir('.'))")
        return
    if not dosyalar:
        print("  Klasor bos gorunuyor (ya da tum dosyalar gizli/erisilemez).")
        return

    toplam = sum(b for _, b in dosyalar)
    print(f"\n[1] Genel")
    print(cizgi)
    print(f"  dosya sayisi : {len(dosyalar):,}{'  (SINIRA TAKILDI)' if kesildi else ''}")
    print(f"  toplam boyut : {insan_boyut(toplam)}")
    if hatalar:
        print(f"  erisilemeyen : {len(hatalar)} oge (izin/baglanti)")

    # --- 2) Uzantilara gore dagilim: bicimi burada goruruz ---
    print(f"\n[2] Uzantilara gore  <-- BICIMI BU SOYLUYOR")
    print(cizgi)
    say = Counter()
    boyut_top = defaultdict(int)
    ornekler = defaultdict(list)
    for p, b in dosyalar:
        u = p.suffix.lower() or "(uzantisiz)"
        say[u] += 1
        boyut_top[u] += b
        if len(ornekler[u]) < ORNEK_AD:
            ornekler[u].append(p.name)
    print(f"  {'uzanti':<14}{'adet':>9}{'toplam':>13}{'ortalama':>13}")
    for u, n in say.most_common(20):
        print(f"  {u:<14}{n:>9,}{insan_boyut(boyut_top[u]):>13}"
              f"{insan_boyut(boyut_top[u]/n):>13}")
    if len(say) > 20:
        print(f"  ... ve {len(say)-20} uzanti daha")

    # --- 3) Ornek dosya adlari: adlandirma semasi meta veri tasiyor mu? ---
    print(f"\n[3] Ornek dosya adlari  <-- adlandirma semasi meta veri tasiyor mu?")
    print(cizgi)
    for u, _ in say.most_common(8):
        print(f"  [{u}]")
        for ad in ornekler[u]:
            print(f"     {ad[:70]}")

    # --- 4) Klasor yapisi: siniflar klasor olarak mi ayrilmis? ---
    print(f"\n[4] Klasor yapisi ({AGAC_DERINLIK} seviye)  <-- siniflar klasor mu?")
    print(cizgi)
    dizin_say = Counter()
    for p, _ in dosyalar:
        try:
            bagil = p.parent.relative_to(kok)
        except ValueError:
            continue
        parcalar = bagil.parts[:AGAC_DERINLIK]
        dizin_say["/".join(parcalar) if parcalar else "."] += 1
    for d, n in sorted(dizin_say.items())[:40]:
        girinti = "  " * (d.count("/") + 1)
        print(f"  {girinti}{d.split('/')[-1] if d != '.' else '(kok)':<40} {n:>8,} dosya")
    if len(dizin_say) > 40:
        print(f"  ... ve {len(dizin_say)-40} klasor daha")

    # --- 5) En buyuk dosyalar: ana veri hangisi? ---
    print(f"\n[5] En buyuk 10 dosya")
    print(cizgi)
    for p, b in sorted(dosyalar, key=lambda t: -t[1])[:10]:
        try:
            gosterim = str(p.relative_to(kok))
        except ValueError:
            gosterim = p.name
        print(f"  {insan_boyut(b):>12}   {gosterim[:60]}")

    # --- 6) Etiket/meta veri adayi dosyalar ---
    print(f"\n[6] Etiket / meta veri adaylari")
    print(cizgi)
    meta_uzanti = {".csv", ".json", ".txt", ".xml", ".yaml", ".yml",
                   ".xlsx", ".xls", ".md", ".log", ".ini", ".cfg"}
    adaylar = [(p, b) for p, b in dosyalar if p.suffix.lower() in meta_uzanti]
    if not adaylar:
        print("  Yok. Etiketler klasor adlarinda veya dosya adlarinda olabilir.")
    else:
        for p, b in sorted(adaylar, key=lambda t: -t[1])[:15]:
            try:
                gosterim = str(p.relative_to(kok))
            except ValueError:
                gosterim = p.name
            print(f"  {insan_boyut(b):>10}   {gosterim[:62]}")
        if len(adaylar) > 15:
            print(f"  ... ve {len(adaylar)-15} dosya daha")

    print(f"\n{'=' * 76}")
    print("BU CIKTIYI PAYLAS -- bicime gore ADIM 2 okuyucusunu yazacagiz.")
    print("Dosya adlari hassas bilgi tasiyorsa duzenleyip oyle paylas.")


if __name__ == "__main__":
    import sys
    rapor(sys.argv[1] if len(sys.argv) > 1 else KOK)
