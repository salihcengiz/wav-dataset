"""
GERCEK VERI -- ADIM 2a: CSV INDEKSINI COZ

Sorumlunun verdigi CSV'ler, karmasik ana depodaki hangi kayitlarin bizim
isimize yaradigini gosteren bir HARITA. Once haritayi okuyacagiz, sonra
isaret ettigi dosyalari.

Bu script'in cevaplamaya calistigi sorular:
  - CSV'de hangi sutunlar var, ne tutuyorlar?
  - Hangi sutun DOSYA YOLU? (isaret ettigi dosyalar gercekten var mi?)
  - Hangi sutun ETIKET/SINIF? Kac sinif, dagilim nasil?
  - Kac BAGIMSIZ kayit var? (sentetikte "1000 ornek" aslinda 19'du --
    ayni tuzak burada da olabilir)
  - Zaman, konum, kanal gibi gruplama adayi sutunlar var mi?

JupyterLab'e yapistirmak icin yazilmistir: harici bagimliligi sadece pandas.
HICBIR SEY YAZMAZ, sadece okur.
"""
from collections import Counter
from pathlib import Path

import pandas as pd

# Sutun adlarinda bu kelimeler gecerse ne olduguna dair ipucu sayilir.
# SIRA ONEMLI: ilk eslesen kazanir. 'channel_start' gibi adlar hem 'channel'
# hem 'start' iceriyor -- konum, zamandan once gelmeli.
IPUCU = {
    "yol": ("path", "file", "filename", "filepath", "uri", "url"),
    "etiket": ("label", "class", "event", "type", "category", "target",
               "annotation", "activity"),
    "konum": ("channel", "chan", "position", "distance", "offset", "depth",
              "locus", "loci", "zone", "location"),
    "sure": ("duration", "length", "samples", "seconds", "nsamples"),
    "kimlik": ("id", "uuid", "session", "run", "experiment", "site", "trial"),
    "zaman": ("time", "date", "timestamp", "start", "end", "utc", "epoch"),
}


def sutun_tipi(ad):
    a = ad.lower()
    for tip, kelimeler in IPUCU.items():
        if any(k in a for k in kelimeler):
            return tip
    return ""


def csv_oku(yol):
    """Ayraci ve kodlamayi tahmin ederek oku."""
    for kwargs in ({}, {"sep": ";"}, {"sep": "\t"},
                   {"encoding": "latin-1"}, {"sep": ";", "encoding": "latin-1"}):
        try:
            df = pd.read_csv(yol, **kwargs)
            if df.shape[1] > 1 or not kwargs:
                return df, kwargs
        except Exception:  # noqa: BLE001
            continue
    return None, None


def rapor(kok=".", maks_csv=10, ornek_satir=5):
    kok = Path(kok)
    cizgi = "-" * 76
    csvler = sorted(kok.rglob("*.csv")) if kok.is_dir() else [kok]

    print("=" * 76)
    print(f"CSV INDEKS INCELEMESI  |  {kok}")
    print("=" * 76)
    if not csvler:
        print("  Bu yol altinda .csv bulunamadi.")
        return
    print(f"  {len(csvler)} CSV bulundu:")
    for c in csvler[:maks_csv]:
        try:
            g = c.relative_to(kok) if kok.is_dir() else c.name
        except ValueError:
            g = c.name
        print(f"     {str(g)[:64]}   ({c.stat().st_size/1024:,.0f} KB)")
    if len(csvler) > maks_csv:
        print(f"     ... ve {len(csvler)-maks_csv} tane daha")

    for c in csvler[:maks_csv]:
        df, kw = csv_oku(c)
        print(f"\n{'=' * 76}")
        print(f"DOSYA: {c.name}")
        print("=" * 76)
        if df is None:
            print("  !!! Okunamadi (ayrac/kodlama cozulemedi).")
            continue
        if kw:
            print(f"  (okuma ayari: {kw})")
        print(f"  {len(df):,} satir  x  {len(df.columns)} sutun")

        # --- Sutunlar ---
        print(f"\n  [A] Sutunlar")
        print("  " + cizgi)
        print(f"  {'sutun':<28}{'tip':>12}{'benzersiz':>11}{'bos':>7}  ipucu")
        for s in df.columns:
            k = df[s]
            n_uniq = k.nunique(dropna=True)
            n_bos = int(k.isna().sum())
            print(f"  {str(s)[:27]:<28}{str(k.dtype):>12}{n_uniq:>11,}"
                  f"{n_bos:>7}  {sutun_tipi(str(s))}")

        # --- Ilk satirlar ---
        print(f"\n  [B] Ilk {ornek_satir} satir")
        print("  " + cizgi)
        with pd.option_context("display.max_columns", 40,
                               "display.width", 200,
                               "display.max_colwidth", 40):
            print("  " + df.head(ornek_satir).to_string().replace("\n", "\n  "))

        # --- Az benzersiz degerli sutunlar = etiket adayi ---
        print(f"\n  [C] Etiket adaylari (az sayida benzersiz deger)")
        print("  " + cizgi)
        bulundu = False
        for s in df.columns:
            n = df[s].nunique(dropna=True)
            # Etiket olmasi icin: az sayida deger VE degerler TEKRARLIYOR olmali.
            # Yol/zaman sutunlari satir basina benzersizdir, etiket olamazlar.
            tekrarliyor = n < len(df)
            if 1 < n <= 30 and tekrarliyor and sutun_tipi(str(s)) not in ("yol", "zaman"):
                bulundu = True
                print(f"  {s}  ({n} deger)")
                for d, adet in df[s].value_counts(dropna=False).head(12).items():
                    print(f"     {str(d)[:46]:<48} {adet:>7,}")
                if n > 12:
                    print(f"     ... ve {n-12} deger daha")
        if not bulundu:
            print("  Yok -- etiketler baska bir dosyada veya dosya adinda olabilir.")

        # --- Dosya yolu sutunlari: isaret ettikleri gercekten var mi? ---
        print(f"\n  [D] Dosya yolu sutunlari  <-- .sdf.hdf5 buradan gelecek")
        print("  " + cizgi)
        yol_sut = [s for s in df.columns
                   if sutun_tipi(str(s)) == "yol" or
                   (df[s].dtype == object and
                    df[s].astype(str).str.contains(r"\.(h5|hdf5|sdf|tdms|npy|wav|segy)",
                                                   case=False, regex=True, na=False).any())]
        if not yol_sut:
            print("  Yol gibi gorunen sutun yok.")
        for s in yol_sut:
            ornek = df[s].dropna().astype(str)
            print(f"  [{s}]  {len(ornek):,} deger")
            for v in ornek.head(3):
                print(f"     {v[:68]}")
            # uzanti dagilimi
            uz = Counter()
            for v in ornek:
                p = str(v).replace("\\", "/").rstrip("/")
                ad = p.rsplit("/", 1)[-1]
                nokta = ad.find(".")
                uz[ad[nokta:].lower() if nokta > 0 else "(uzantisiz)"] += 1
            print(f"     uzantilar: {dict(uz.most_common(6))}")
            # var mi?
            var = yok = 0
            for v in ornek.head(200):
                p = Path(str(v))
                aday = p if p.is_absolute() else (c.parent / p)
                if aday.exists() or p.exists():
                    var += 1
                else:
                    yok += 1
            print(f"     ilk 200 yolun {var}'i diskte BULUNDU, {yok}'i bulunamadi")
            if yok and not var:
                print(f"     -> yollar bu CSV'ye gore goreli degil. Kok klasoru")
                print(f"        bulup basina eklememiz gerekecek.")

        # --- Gruplama adaylari: sizintiyi onleyecek grup kimligi ne olacak? ---
        print(f"\n  [E] Gruplama adaylari  <-- SIZINTIYI BU BELIRLEYECEK")
        print("  " + cizgi)
        print(f"  Sentetik veride 'grup' = kaynak ses kaydiydi. Burada ne olmali?")
        for s in df.columns:
            tip = sutun_tipi(str(s))
            if tip in ("kimlik", "zaman", "konum"):
                n = df[s].nunique(dropna=True)
                print(f"  {s:<28} {tip:<8} {n:>8,} benzersiz  "
                      f"(satir/grup ~{len(df)/max(n,1):.1f})")

    print(f"\n{'=' * 76}")
    print("BU CIKTIYI PAYLAS. Sonra bir .sdf.hdf5 dosyasini acacagiz (ADIM 2b).")


if __name__ == "__main__":
    import sys
    rapor(sys.argv[1] if len(sys.argv) > 1 else ".")
