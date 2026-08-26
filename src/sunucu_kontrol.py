"""
GERCEK VERI -- ADIM 3: SUNUCU ORTAMI VE YUKLEME HIZI OLCUMU

DURUM.md Bolum 5, gercek egitim hattinin onunde iki pratik soru oldugunu
soyluyor. Bu script ikisini de OLCEREK cevaplar (tahminle degil):

  SORU 1 -- Sunucuda PyTorch ve GPU var mi?
     Yol /tf/ bir TensorFlow imajina isaret ediyor, ekibin Keras modelleri
     var, bizim modelimiz PyTorch. Cevap "hayir" ise egitim hatti bastan
     farkli kurulmali (pip install, ya da conda ortami, ya da CPU'da kucuk
     alt kume).

  SORU 2 -- 293.469 satir nasil islenecek?
     Her satir bir HDF5 acma islemi. Satirlar fazlasiyla yedekli: ~21.318
     dosya x ~14 bitisik kanal, hepsi ayni olayi goruyor. Alt orneklem
     sart ama ORANI olcume dayanmali.

=== NEDEN AYRI BIR SCRIPT ===

Sentetik asamada 959 PNG'yi RAM'e alip gectik gittik (dataset.py). Burada
373.908 pencere var ve her biri bir HDF5 dilimi. Onbellek stratejisi,
alt orneklem orani ve num_workers sayisi ancak SATIR BASINA MALIYET
bilinince secilebilir. Yanlis secim, farki dakikalarla degil SAATLERLE
olcuulen bir egitim demek.

Olculen sey uc parcaya ayriliyor cunku cozumleri farkli:
     dosya acma    -> cozumu: ayni dosyanin kanallarini birlikte oku
     dilim okuma   -> cozumu: alt orneklem / onbellek
     CPU (STFT)    -> cozumu: num_workers

=== CALISTIRMA (JupyterLab) ===

Sol panelden yukari ok (^) ile su iki dosyayi ayni klasore yukle:
     real_data.py
     sunucu_kontrol.py

Sonra bir hucrede:

     import sunucu_kontrol
     sunucu_kontrol.tam_rapor()

veya terminalde:

     python sunucu_kontrol.py

Farkli bir kok klasor icin:

     sunucu_kontrol.tam_rapor(kok="/tf/start_training/RELATIONNET/FENCE_DATA_NEW")

=== GUVENLIK ===

HICBIR SEY YAZMAZ, HICBIR SEY SILMEZ. Sadece okur ve ekrana yazar.
Ciktida dosya yollari ve sayilar var, veri yok -- oldugu gibi paylasilabilir.

=== BAGIMLILIK ===

numpy + pandas + h5py + real_data.py. torch VARSA raporlanir, YOKSA
script yine calisir (zaten sorulan soru "torch var mi").
"""
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None

# real_data.py ayni klasorde olmali (JupyterLab'e beraber yuklenir).
# Yerelde src/ icinden calistirildiginda da bulunur.
try:
    import real_data as rd
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import real_data as rd
    except ImportError:
        rd = None

# Satir secimi onbellek_kur.py'de tanimli -- orasi kalici hattin sahibi,
# burasi bir kerelik teshis araci. Tek tanim, iki kopya yok.
from onbellek_kur import KOK, _bicim, alt_orneklem  # noqa: E402


# ---------------------------------------------------------------
# VARSAYILANLAR
# ---------------------------------------------------------------
CSV_ADLARI = ("train_final.csv", "val_final.csv", "test_final.csv")

CIZGI = "-" * 76
CIFT = "=" * 76


def _baslik(metin):
    print(f"\n{CIFT}")
    print(metin)
    print(CIFT)


def _alt(metin):
    print(f"\n  {metin}")
    print("  " + CIZGI)


# ---------------------------------------------------------------
# A) ORTAM  <-- SORU 1
# ---------------------------------------------------------------
def ortam_raporu(kok=KOK):
    """
    PyTorch var mi, GPU var mi, ne kadar disk/RAM/CPU var?

    Karar agaci (cikti buna gore yorumlanacak):

      torch YOK              -> pip install torch (surum uyumu kontrol edilmeli)
                                ya da modeli Keras'a cevirmek (istenmez: onceden
                                egitilmis paketimiz PyTorch state_dict'i)
      torch VAR, GPU YOK     -> CPU egitimi. Alt orneklem orani cok daha agresif
                                olmali; 34.835 parametrelik model CPU'da da
                                egitilebilir ama epoch suresi 10-20 kat artar
      torch VAR, GPU VAR     -> ideal durum, alt orneklemi veri yedekliligine
                                gore secebiliriz (hesap kisiti degil)
    """
    _baslik("A) ORTAM  --  SORU 1: PyTorch ve GPU var mi?")

    _alt("Python ve isletim sistemi")
    print(f"  python         : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  platform       : {platform.platform()}")

    # CPU: sched_getaffinity gercekten KULLANILABILIR cekirdegi verir.
    # Konteynerlerde cpu_count() ana makinenin cekirdek sayisini gosterip
    # yaniltabilir -- num_workers'i ona gore secersek surec bogulur.
    try:
        n_kullanilabilir = len(os.sched_getaffinity(0))
    except AttributeError:
        n_kullanilabilir = None
    print(f"  cpu_count      : {os.cpu_count()}"
          + (f"   (kullanilabilir: {n_kullanilabilir})" if n_kullanilabilir else ""))

    ram = _ram_gb()
    if ram:
        print(f"  RAM            : {ram:.1f} GB")

    _alt("PyTorch")
    try:
        import torch
        print(f"  torch          : {torch.__version__}")
        try:
            import torchvision
            print(f"  torchvision    : {torchvision.__version__}")
        except ImportError:
            print(f"  torchvision    : YOK  <-- Resize/Normalize icin gerekli")
            print(f"                   (alternatif: numpy ile kendi resize'imizi yazariz)")

        cuda_var = torch.cuda.is_available()
        print(f"  CUDA derlemesi : {torch.version.cuda or 'CPU-only build'}")
        print(f"  cuda.is_available(): {cuda_var}")
        if cuda_var:
            for i in range(torch.cuda.device_count()):
                ozellik = torch.cuda.get_device_properties(i)
                print(f"     GPU {i}: {ozellik.name}  "
                      f"{ozellik.total_memory / 1e9:.1f} GB  "
                      f"(cc {ozellik.major}.{ozellik.minor})")
            print(f"  -> EGITIM GPU'DA YAPILABILIR.")
        else:
            print(f"  -> GPU YOK veya gorunmuyor. Egitim CPU'da olur.")
            print(f"     nvidia-smi ciktisini da paylas: GPU fiziksel olarak var")
            print(f"     ama torch CPU-only kurulmus olabilir (bizim yereldeki gibi).")
    except ImportError:
        print(f"  torch          : *** KURULU DEGIL ***")
        print(f"  -> Sunucudaki imaj TensorFlow. Secenekler:")
        print(f"     1) pip install torch --index-url .../cpu  (veya cu11x)")
        print(f"     2) egitimi CPU'da, kucuk alt kumede yapmak")
        print(f"     Sorumluya sorulacak: pip install serbest mi, internet var mi?")

    _alt("Diger kutuphaneler")
    for ad in ("numpy", "pandas", "h5py", "sklearn", "scipy", "tensorflow"):
        try:
            m = __import__(ad)
            print(f"  {ad:<14} : {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  {ad:<14} : yok")

    _alt("Disk")
    for yol in (kok, "/tf", "/tmp", os.getcwd()):
        try:
            k = shutil.disk_usage(yol)
            print(f"  {str(yol):<40} bos {k.free / 1e9:>7.1f} GB / "
                  f"toplam {k.total / 1e9:.1f} GB")
        except OSError as e:
            print(f"  {str(yol):<40} okunamadi ({e.__class__.__name__})")
    print(f"\n  NOT: bos alan, on-hesaplanmis spektrogram onbellegi icin onemli")
    print(f"       (Bolum D'de boyut tahmini var).")


def _ram_gb():
    """Toplam RAM (GB). Linux'ta /proc, digerlerinde sessizce None."""
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
    except (AttributeError, ValueError, OSError):
        return None


# ---------------------------------------------------------------
# B) CSV YEDEKLILIGI  <-- SORU 2'nin birinci yarisi
# ---------------------------------------------------------------
def csv_raporu(kok=KOK, csv_adlari=CSV_ADLARI):
    """
    Satirlar ne kadar yedekli, alt orneklem ne kadar veri birakir?

    CSV'nin bir satiri = (kayit dosyasi, fiber kanali, zaman penceresi).
    Ayni dosyanin bitisik kanallari AYNI fiziksel olayi, birkac metre
    otesinden goruyor. 14 kanalin 14'unu de egitime koymak, ayni olayi 14
    kez gostermek demek -- sentetik asamadaki "1000 ornek ama 19 kayit"
    tuzaginin bu veri setindeki karsiligi.

    Fark su: orada yedeklilik SIZINTI uretiyordu (ayni kayit hem egitimde
    hem testte). Burada bolmeler dosya duzeyinde temiz (Rapor Bolum 3),
    yani yedeklilik sizinti degil sadece BOSA HESAP. Bu yuzden alt orneklem
    bir metodoloji zorunlulugu degil, hiz karari.

    Bu fonksiyon "dosya basina k kanal" stratejisinin her k icin kac satir
    ve nasil bir sinif dagilimi birakacagini hesaplar.

    Donen: {csv_adi: DataFrame} -- hiz olcumu ayni df'leri kullanir.
    """
    _baslik("B) CSV YEDEKLILIGI  --  SORU 2: 293.469 satir nasil islenecek?")
    kok = Path(kok)
    tablolar = {}

    for ad in csv_adlari:
        yol = kok / ad
        if not yol.exists():
            print(f"\n  {ad}: BULUNAMADI ({yol})")
            continue
        df = pd.read_csv(yol)
        tablolar[ad] = df

        _alt(f"{ad}  --  {len(df):,} satir")

        if "file" not in df.columns:
            print(f"  !!! 'file' sutunu yok. Sutunlar: {list(df.columns)}")
            continue

        n_dosya = df["file"].nunique()
        print(f"  benzersiz dosya          : {n_dosya:,}")
        print(f"  dosya basina satir       : ortalama {len(df) / n_dosya:.1f}")

        # Dosya bicimi: karar '.bin.hdf5' (Rapor 1.5). '.sdf' satirlari
        # elenecek, kac tane oldugunu bilmemiz gerekiyor.
        bicimler = df["file"].astype(str).map(_bicim)
        print(f"  dosya bicimi (satir)     : "
              + "  ".join(f"{b} {n:,} (%{100*n/len(df):.1f})"
                          for b, n in bicimler.value_counts().items()))

        if "channel" in df.columns:
            n_kanal = df["channel"].nunique()
            cift = df.groupby("file")["channel"].nunique()
            print(f"  benzersiz kanal (genel)  : {n_kanal:,}")
            print(f"  dosya basina KANAL       : ortalama {cift.mean():.1f}  "
                  f"medyan {cift.median():.0f}  min {cift.min()}  maks {cift.max()}")

            # Kanallar bitisik mi? Bitisikse "her n'inci kanali al" mantikli;
            # dagiInikSA rastgele secim gerekir.
            ornek = df[df["file"] == df["file"].iloc[0]]["channel"].sort_values().unique()
            if len(ornek) > 1:
                fark = np.diff(ornek).tolist()
                print(f"  ornek dosyanin kanallari : {ornek[:12].tolist()}"
                      f"{' ...' if len(ornek) > 12 else ''}")
                print(f"  kanal araliklari         : "
                      f"{sorted(Counter(fark).items())[:5]}"
                      f"   (hepsi (1, n) ise kanallar bitisik)")

        if "event" in df.columns:
            print(f"\n  sinif dagilimi (satir bazinda):")
            for sinif, adet in df["event"].value_counts().items():
                print(f"     {str(sinif):<14} {adet:>9,}  (%{100 * adet / len(df):.1f})")
            print(f"  sinif dagilimi (DOSYA bazinda -- gercek bagimsizlik):")
            dosya_sinif = df.groupby("file")["event"].first().value_counts()
            for sinif, adet in dosya_sinif.items():
                print(f"     {str(sinif):<14} {adet:>9,}  (%{100 * adet / n_dosya:.1f})")

        # Pencere uzunlugu dagilimi -- yansitmali doldurma ne siklikta devreye girecek?
        if {"window_start", "window_end"} <= set(df.columns):
            uzunluk = (df["window_end"] - df["window_start"]).astype(int)
            print(f"\n  pencere uzunlugu         : {uzunluk.nunique():,} farkli deger")
            print(f"     min {uzunluk.min():,}  medyan {int(uzunluk.median()):,}  "
                  f"maks {uzunluk.max():,}")
            kisa = int((uzunluk < rd.PENCERE).sum()) if rd else 0
            uzun = int((uzunluk > rd.PENCERE).sum()) if rd else 0
            if rd:
                print(f"     {rd.PENCERE:,}'den kisa: {kisa:>8,} (%{100*kisa/len(df):.1f})"
                      f"  -> yansitmali doldurulacak")
                print(f"     {rd.PENCERE:,}'den uzun: {uzun:>8,} (%{100*uzun/len(df):.1f})"
                      f"  -> enerji merkezine kirpilacak")

        # --- ALT ORNEKLEM PROJEKSIYONU ---
        if "channel" in df.columns:
            print(f"\n  ALT ORNEKLEM PROJEKSIYONU")
            print(f"  (once .sdf elenir, sonra dosya basina k kanal secilir;"
                  f" oran ham {len(df):,} satira gore)")
            print(f"  {'k':>4}{'satir':>12}{'oran':>9}   sinif dagilimi")
            for k in (1, 2, 4, 8, None):
                alt = alt_orneklem(df, kanal_basina=k)
                oran = len(alt) / len(df)
                if "event" in alt.columns:
                    dag = "  ".join(
                        f"{str(s)[:8]} {n:,}"
                        for s, n in alt["event"].value_counts().items())
                else:
                    dag = ""
                etiket = "hepsi" if k is None else str(k)
                print(f"  {etiket:>4}{len(alt):>12,}{oran:>8.1%}   {dag}")

    return tablolar


# _bicim ve alt_orneklem ARTIK BURADA DEGIL -- onbellek_kur.py'de
# (dosyanin basinda import ediliyor).


# ---------------------------------------------------------------
# C) YUKLEME HIZI  <-- SORU 2'nin ikinci yarisi
# ---------------------------------------------------------------
def hiz_raporu(df, n=200, tohum=42, pencere_basi_spektrogram=True):
    """
    Satir basina gercek maliyeti olcer ve epoch suresine cevirir.

    UC AYRI OLCUM, cunku uc ayri cozumu var:

      1. DAGINIK okuma  -- her satir farkli bir dosya, her seferinde ac-kapa.
         DataLoader'in rastgele karistirilmis (shuffle=True) hali budur.
         Gercekci en KOTU durum.

      2. GRUPLU okuma   -- ayni dosyanin tum kanallari tek acilista.
         real_data.pencere_yukle'nin `dosya_nesnesi` parametresi bunun icin
         var. Egitimden ONCE tek seferlik on-hesaplama yapacaksak hiz budur.

      3. ASAMA KIRILIMI -- acma / okuma / hat / STFT ayri ayri.
         I/O mu bagliyor CPU mu? I/O ise onbellek, CPU ise num_workers.

    Cikti "1 epoch kac dakika" tahminine kadar goturulur.
    """
    _baslik("C) YUKLEME HIZI  --  satir basina maliyet ve epoch tahmini")

    if h5py is None:
        print("  h5py yok, olcum yapilamiyor.")
        return
    if rd is None:
        print("  real_data.py bulunamadi -- ayni klasore yuklenmeli.")
        return

    # Olcum yalnizca .bin.hdf5 uzerinde: egitim hatti da yalnizca onu okuyacak
    # (Rapor 1.5). .sdf'yi olcuye katmak, hic kullanmayacagimiz bir bicimin
    # maliyetini ortalamaya karistirirdi.
    ham = len(df)
    df = alt_orneklem(df, kanal_basina=None, yalniz_bin=True)
    if len(df) < ham:
        print(f"  ({ham - len(df):,} .sdf satiri olcum disi birakildi, "
              f"kalan {len(df):,})")
    if df.empty:
        print("  .bin.hdf5 satiri yok, olcum yapilamiyor.")
        return

    # --- 1) DAGINIK: rastgele n satir, her biri kendi dosyasini aciyor ---
    _alt(f"1) DAGINIK okuma  ({n} rastgele satir, her satirda ac-kapa)")
    ornek = df.sample(n=min(n, len(df)), random_state=tohum)
    sonuc = _olc(ornek, dosya_nesnesi_paylas=False,
                 spektrogram_da=pencere_basi_spektrogram)
    _yaz_olcum(sonuc)
    daginik = sonuc["satir_basi_ms"]

    # --- 2) GRUPLU: az sayida dosya, hepsinin tum kanallari ---
    _alt(f"2) GRUPLU okuma  (ayni dosyanin kanallari tek acilista)")
    dosyalar = df["file"].drop_duplicates()
    n_dosya = max(3, min(20, len(dosyalar)))
    secilen = dosyalar.sample(n=n_dosya, random_state=tohum)
    grup_df = df[df["file"].isin(secilen)]
    print(f"  {n_dosya} dosya, toplam {len(grup_df):,} satir")
    sonuc_g = _olc(grup_df, dosya_nesnesi_paylas=True,
                   spektrogram_da=pencere_basi_spektrogram)
    _yaz_olcum(sonuc_g)
    gruplu = sonuc_g["satir_basi_ms"]

    if daginik and gruplu:
        print(f"\n  GRUPLU / DAGINIK hiz orani: {daginik / gruplu:.1f}x")
        if daginik / gruplu > 1.5:
            print(f"  -> Dosya acma maliyeti onemli. On-hesaplama dosya dosya")
            print(f"     yapilmali; DataLoader'a rastgele HDF5 okutmak israf.")
        else:
            print(f"  -> Dosya acma maliyeti onemsiz. Dogrudan DataLoader'dan")
            print(f"     okumak da makul.")

    # --- 3) EPOCH TAHMINI ---
    _alt("3) EPOCH SURESI TAHMINI  (tek surec, num_workers=0)")
    print(f"  {'kanal/dosya':>12}{'satir':>12}{'daginik':>12}{'gruplu':>12}")
    for k in (1, 2, 4, 8, None):
        alt = alt_orneklem(df, kanal_basina=k)
        etiket = "hepsi" if k is None else str(k)
        t_d = len(alt) * daginik / 1000 / 60 if daginik else float("nan")
        t_g = len(alt) * gruplu / 1000 / 60 if gruplu else float("nan")
        print(f"  {etiket:>12}{len(alt):>12,}{t_d:>10.1f} dk{t_g:>10.1f} dk")
    print(f"\n  Bunlar TEK SURECLI sureler. DataLoader(num_workers=W) ile")
    print(f"  kabaca W'ye bolunur (I/O bagliysa daha az). GPU'daki ileri/geri")
    print(f"  gecis bu surenin yaninda ihmal edilebilir -- model 34.835")
    print(f"  parametre, yani hat VERI TARAFINDAN baglanacak.")
    print(f"\n  Karar notu: on-hesaplama yapilirsa bu maliyet BIR KEZ odenir,")
    print(f"  sonraki tum epoch'lar onbellekten okur (bkz. Bolum D).")


def _olc(df, dosya_nesnesi_paylas, spektrogram_da=True):
    """
    Satirlari gercekten yukler, asama asama sure toplar.

    Toplananlar:
        t_ac      : h5py.File acma (+ kapatma)
        t_oku     : dilim okuma + hypot   (I/O + biraz CPU)
        t_hat     : pencereye oturt + bos kontrol + normalize  (CPU)
        t_stft    : spektrogram  (CPU)
    """
    t_ac = t_oku = t_hat = t_stft = 0.0
    n_ok = n_bos = n_hata = 0
    t0 = time.perf_counter()

    if dosya_nesnesi_paylas:
        gruplar = df.groupby("file", sort=False)
    else:
        # her satir kendi "grubu" -> her satirda ayri acilis
        gruplar = ((r.file, df.iloc[[i]]) for i, r in enumerate(df.itertuples()))

    for dosya, grup in gruplar:
        try:
            ta = time.perf_counter()
            f = h5py.File(str(dosya), "r")
            t_ac += time.perf_counter() - ta
        except Exception:  # noqa: BLE001  -- dosya yok / bozuk / izin
            n_hata += len(grup)
            continue
        try:
            for r in grup.itertuples():
                ta = time.perf_counter()
                s = rd.genlik_oku(f, r.channel, r.window_start, r.window_end)
                t_oku += time.perf_counter() - ta
                if s is None or s.size == 0:
                    n_hata += 1
                    continue

                ta = time.perf_counter()
                s = rd.pencereye_oturt(s, rd.PENCERE)
                bos = rd.bos_mu(s)
                if not bos:
                    s = rd.normalize_et(s)
                t_hat += time.perf_counter() - ta
                if bos:
                    n_bos += 1
                    continue

                if spektrogram_da:
                    ta = time.perf_counter()
                    rd.spektrogram(s)
                    t_stft += time.perf_counter() - ta
                n_ok += 1
        finally:
            f.close()

    toplam = time.perf_counter() - t0
    n = max(n_ok + n_bos + n_hata, 1)
    return {
        "n": n, "n_ok": n_ok, "n_bos": n_bos, "n_hata": n_hata,
        "toplam_s": toplam, "satir_basi_ms": 1000 * toplam / n,
        "t_ac": t_ac, "t_oku": t_oku, "t_hat": t_hat, "t_stft": t_stft,
    }


def _yaz_olcum(s):
    if s["n_ok"] == 0 and s["n_hata"] == s["n"]:
        print(f"  !!! {s['n']} satirin hicbiri okunamadi.")
        print(f"      'file' sutunundaki yollar bu makineden erisilebilir mi?")
        return
    print(f"  toplam {s['toplam_s']:.2f} s  /  {s['n']:,} satir  "
          f"->  SATIR BASINA {s['satir_basi_ms']:.2f} ms")
    print(f"  islenen {s['n_ok']:,}   bos (elendi) {s['n_bos']:,} "
          f"(%{100 * s['n_bos'] / max(s['n'], 1):.1f})   hata {s['n_hata']:,}")
    pay = max(s["toplam_s"], 1e-9)
    print(f"  asama kirilimi:")
    for ad, sure in (("dosya acma", s["t_ac"]), ("dilim okuma + hypot", s["t_oku"]),
                     ("pencere + bos + norm", s["t_hat"]), ("STFT", s["t_stft"])):
        print(f"     {ad:<22} {sure:>7.2f} s  (%{100 * sure / pay:>4.1f})")
    olculen = s["t_ac"] + s["t_oku"] + s["t_hat"] + s["t_stft"]
    print(f"     {'(olculmeyen ek yuk)':<22} {pay - olculen:>7.2f} s  "
          f"(%{100 * (pay - olculen) / pay:>4.1f})")


# ---------------------------------------------------------------
# D) ONBELLEK BOYUTU
# ---------------------------------------------------------------
def onbellek_tahmini(df, kanallar=(1, 2, 4, None)):
    """
    On-hesaplanmis spektrogram onbellegi ne kadar yer kaplar?

    Egitim hatti icin iki secenek var ve secim boyuta bagli:

      A) Her epoch'ta HDF5'ten oku      -> disk gerekmez, her epoch yavas
      B) Bir kez hesapla, diske yaz     -> bir kerelik maliyet, sonra hizli

    Sentetik asamada (B)'nin RAM'deki hali kullanilmisti (959 PNG = 439 MB).
    Burada boyut cok daha buyuk, o yuzden uc format karsilastiriliyor:

      float32 (129 x 234)   : tam hassasiyet, 121 KB/pencere
      float16               : yarisi, dB degerleri icin fazlasiyla yeterli
      uint8                 : dB [-80, 0] araligini 256 kademeye kuantala
                              -> 0.31 dB adim. Sentetik asamada model zaten
                              8-bit PNG goruyordu (viridis), yani bu format
                              onceden egitilmis modelin gordugu hassasiyetle
                              BIREBIR ayni.
    """
    _baslik("D) ONBELLEK BOYUTU  --  on-hesaplama fizibilitesi")

    if rd is None:
        print("  real_data.py yok, tahmin yapilamiyor.")
        return

    n_frek = rd.N_FFT // 2 + 1
    n_cerceve = 1 + (rd.PENCERE - rd.N_FFT) // rd.HOP
    hucre = n_frek * n_cerceve
    print(f"  bir spektrogram: {n_frek} frekans x {n_cerceve} cerceve "
          f"= {hucre:,} hucre")
    print(f"  (pencere {rd.PENCERE:,} ornek, n_fft={rd.N_FFT}, hop={rd.HOP})")

    print(f"\n  {'kanal/dosya':>12}{'pencere':>12}{'float32':>12}"
          f"{'float16':>12}{'uint8':>12}")
    for k in kanallar:
        alt = alt_orneklem(df, kanal_basina=k)
        # bos pencerelerin ~%27'si elenecek (Rapor 5.4)
        n = int(len(alt) * 0.73)
        etiket = "hepsi" if k is None else str(k)
        print(f"  {etiket:>12}{n:>12,}"
              f"{n * hucre * 4 / 1e9:>10.1f} GB"
              f"{n * hucre * 2 / 1e9:>10.1f} GB"
              f"{n * hucre * 1 / 1e9:>10.1f} GB")
    print(f"\n  (bos pencere elemesi icin %27 dusuldu -- Rapor Bolum 5.4)")
    print(f"\n  uint8 onerilir: onceden egitilmis model zaten 8-bit PNG ile")
    print(f"  egitildi, kuantalama yeni bir kayip getirmiyor.")


# ---------------------------------------------------------------
# TAM RAPOR
# ---------------------------------------------------------------
def tam_rapor(kok=KOK, csv_adlari=CSV_ADLARI, hiz_csv="train_final.csv", n=200):
    """
    Dort bolumu sirayla calistirir. JupyterLab'de bu tek cagri yeterli.
    """
    print(CIFT)
    print("SUNUCU ORTAM VE YUKLEME HIZI RAPORU")
    print(f"kok: {kok}")
    print(CIFT)

    ortam_raporu(kok)
    tablolar = csv_raporu(kok, csv_adlari)

    df = tablolar.get(hiz_csv)
    if df is None and tablolar:
        hiz_csv, df = next(iter(tablolar.items()))
    if df is not None:
        print(f"\n  (hiz ve onbellek olcumleri {hiz_csv} uzerinde)")
        hiz_raporu(df, n=n)
        onbellek_tahmini(df)
    else:
        print("\n  CSV okunamadigi icin hiz olcumu atlandi.")

    _baslik("BU CIKTIYI PAYLAS")
    print("  Cevabi beklenen sorular:")
    print("    1. torch var mi, GPU gorunuyor mu?          -> Bolum A")
    print("    2. satir basina kac ms, epoch kac dakika?   -> Bolum C")
    print("    3. hangi alt orneklem orani secilecek?      -> Bolum B + C")
    print("    4. on-hesaplama diske sigar mi?             -> Bolum D")
    print("\n  Sonraki adim: PyTorch Dataset + egitim hatti (DURUM.md Bolum 5).")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sunucu ortam ve yukleme hizi olcumu")
    ap.add_argument("--kok", default=KOK, help="CSV'lerin bulundugu klasor")
    ap.add_argument("--csv", default="train_final.csv",
                    help="hiz olcumunun yapilacagi CSV")
    ap.add_argument("-n", type=int, default=200,
                    help="daginik okuma olcumunde kac satir")
    ap.add_argument("--yalniz-ortam", action="store_true",
                    help="sadece Bolum A (veri gerektirmez)")
    a = ap.parse_args()

    if a.yalniz_ortam:
        ortam_raporu(a.kok)
    else:
        tam_rapor(kok=a.kok, hiz_csv=a.csv, n=a.n)
