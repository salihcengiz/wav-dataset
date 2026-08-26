"""
GERCEK VERI -- ADIM 4: SPEKTROGRAM ONBELLEGI

Egitimden ONCE, bir kez, tum pencereleri spektrograma cevirip uint8 olarak
diske yazar. Egitim sonra bu dosyadan okur.

=== NEDEN ON-HESAPLAMA (olcume dayali) ===

sunucu_kontrol.py 2026-08-26'da sunucuda su sonuclari verdi:

    satir basina    : 3.87 ms (gruplu okuma) / 6.50 ms (daginik)
    darbogaz        : dilim okuma + hypot (%49), STFT yalnizca %22
    tam egitim seti : 18.5 dk tek surecte
    RAM             : 16.5 GB,  disk bos: 646 GB

Her epoch'ta HDF5'ten okumak, her epoch'a 4-19 dakika eklerdi. Ustelik
DataLoader rastgele karistirdigi icin GRUPLU degil DAGINIK okuma olurdu
(6.50 ms, 1.7 kat yavas). On-hesaplama bu maliyeti BIR KEZ odetir; sonraki
epoch'lar RAM'e sigan bir uint8 diziden okur.

=== NEDEN uint8 ===

Spektrogram dB olarak [-80, 0] araliginda. uint8'e kuantalama 0.31 dB adim
demek. Sentetik asamada model zaten 8-bit viridis PNG goruyordu -- yani
onceden egitilmis modelin egitildigi hassasiyet TAM OLARAK bu. Kuantalama
yeni bir kayip getirmiyor, sadece boyutu 4'e boluyor.

    k=2 icin : 1.2 GB   <- RAM'e rahat sigar
    k=4 icin : 2.2 GB
    hepsi    : 6.2 GB

=== NEDEN 129x231 (224x320 DEGIL) ===

Onbellek STFT'nin DOGAL cozunurlugunde tutuluyor. 224x320'ye buyutup
saklamak boyutu 2.4 kat artirirdi (71.680 hucre vs 29.799) ve hicbir bilgi
eklemezdi -- buyutme zaten interpolasyon. Olceklendirme egitim aninda,
GPU'da (ya da transform'da) yapilir; orada bedava.

=== NEDEN torch YOK ===

Sunucuda torch KURULU DEGIL (olculdu, 2026-08-26; GPU var -- RTX 3090 --
ama torch yok). Bu script yalnizca numpy + h5py kullanir, dolayisiyla
torch kurulumu beklenmeden calistirilabilir. Uretilen onbellek
cerceve-bagimsizdir: PyTorch da Keras da okuyabilir.

=== KULLANIM ===

    # 1) ONCE KESIF -- k'yi olcumle sec (~1 dk)
    python onbellek_kur.py --kesif

    # 2) ONBELLEGI KUR
    python onbellek_kur.py --csv train_final.csv --k 2
    python onbellek_kur.py --csv val_final.csv   --k 2
    python onbellek_kur.py --csv test_final.csv  --k 0     # 0 = hepsi

    # 3) DOGRULA -- onbellek gercekten hattin urettigiyle ayni mi
    python onbellek_kur.py --dogrula onbellek_train_final_k2.h5

JupyterLab'de:

    import onbellek_kur
    onbellek_kur.kesif()
    onbellek_kur.kur(csv="train_final.csv", k=2)

=== BAGIMLILIK ===

numpy + pandas + h5py + real_data.py. Baska hicbir sey -- ozellikle torch
degil. Bu modul satir seciminin (alt_orneklem) de sahibidir;
sunucu_kontrol.py onu buradan import eder.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None

_burada = str(Path(__file__).resolve().parent)
if _burada not in sys.path:
    sys.path.insert(0, _burada)

import real_data as rd

KOK = "/tf/start_training/RELATIONNET/FENCE_DATA_NEW"

# Sinif -> indeks. support_set_creator.py'deki resmi harita (Rapor 6.4).
SINIFLAR = {"cutting": 0, "climbing": 1, "noise": 2}
IDX_SINIF = {v: k for k, v in SINIFLAR.items()}

CIZGI = "-" * 76
CIFT = "=" * 76


# ---------------------------------------------------------------
# SATIR SECIMI  (once sunucu_kontrol.py'deydi, 2026-08-26'da buraya alindi)
#
# Neden tasindi: alt_orneklem KALICI hattin parcasi -- egitimin hangi
# satirlari gordugunu o belirliyor. sunucu_kontrol.py ise bir kerelik
# teshis araci. Kalici hattin gecici araca bagimli olmasi ters bir
# bagimlilik yonuydu. Artik sunucu_kontrol.py bunlari BURADAN import
# ediyor -- iki kopya YOK, tek tanim var.
# ---------------------------------------------------------------
def _bicim(yol):
    """Dosya adindan bicimi cikarir: .bin.hdf5 / .sdf.hdf5 / diger."""
    ad = str(yol).replace("\\", "/").rsplit("/", 1)[-1].lower()
    for b in (".bin.hdf5", ".sdf.hdf5"):
        if ad.endswith(b):
            return b
    nokta = ad.find(".")
    return ad[nokta:] if nokta > 0 else "(uzantisiz)"


def alt_orneklem(df, kanal_basina=2, yalniz_bin=True):
    """
    Her dosyadan en fazla `kanal_basina` kanal secer -- kanal araliginin
    ICINE esit araliklarla yayarak, UC KANALLARI DISLAYARAK. `yalniz_bin`
    ile .bin.hdf5 disindaki satirlar elenir (Rapor 1.5: val/test'in %100'u
    bu bicimde, .sdf'nin olculebilir bir karsiligi yok).

    NEDEN ALT ORNEKLEM (Rapor 8.5)
    ------------------------------
    Satirlar fazlasiyla yedekli: 21.318 dosya, dosya basina ortalama 10.9
    bitisik kanal, hepsi AYNI olayi birkac metre oteden goruyor.

    Kritik nokta: bu fonksiyon her dosyadan EN AZ BIR kanal alir, yani k ne
    olursa olsun 21.318 dosyanin hepsi kalir. **k cesitliligi degil
    YEDEKLILIGI keser.** Bagimsiz birim dosya oldugu icin k=2 bilgi kaybi
    olmadan maliyeti %19'a indiriyor.

    Olculen yan fayda: noise kayitlari kanal basina daha cok pencere
    tasidigi icin (2.4 vs climbing 1.2), kanal budandikca noise'un payi
    %6.8'den %17.8'e cikiyor -- sinif dengesizligi kendiliginden duzeliyor.

    NEDEN UCLAR DISLANIYOR (2026-08-26'da duzeltildi)
    ------------------------------------------------
    Ilk surum `linspace(0, n-1, k)` kullaniyordu, yani k=2 icin ILK ve SON
    kanali seciyordu. Gerekcesi "hem guclu hem zayif kanaldan ornek al"
    idi -- ama bu gerekce kendi raporumuzla celisiyor:

        Rapor 6.1: bos pencerelerin muhtemel sebebi, labels tablosunun
        kanal ARALIGI vermesi ve "araligin kenarindaki kanallarda sinyal
        zayiflamis" olmasi.

    Yani kenar kanallar sistematik olarak en zayif olanlar. Ustelik zayif
    pencereler zaten `bos_mu` ile ELENIYOR -- onlari secmek modele "kenar
    durumu ogretmek" degil, sadece butceyi cope atmak. k=2'de tam olarak
    en kotu iki kanali seciyorduk.

    Yeni kural: linspace(0, n-1, k+2)[1:-1] -- uclar disarida kalir,
    secilenler aralik icine esit dagilir.
        n=14, k=1 -> [7]        (merkez)
        n=14, k=2 -> [4, 9]     (ic bolgede iki nokta)
        n=14, k=4 -> [2,5,8,11]

    Bu kuralin gerekcesi hala bir TAHMIN. kesif() onu olcup dogruluyor.

    Rastgele secim yerine esit aralik tercih ediliyor cunku rastgele secim
    bazi dosyalarda iki KOMSU kanali secip birbirinin neredeyse kopyasi iki
    ornek uretebilir -- alt orneklemin amaci tam da bunu onlemek.

    kanal_basina=None -> kanal elemesi yapilmaz (referans satiri icin).

    Secim RASTGELE DEGIL (deterministik siralama), yani ayni CSV her zaman
    ayni alt kumeyi verir -- tekrar uretilebilirlik icin.
    """
    if yalniz_bin and "file" in df.columns:
        df = df[df["file"].astype(str).map(_bicim) == ".bin.hdf5"]

    if kanal_basina is None or "channel" not in df.columns:
        return df

    parcalar = []
    for _, grup in df.groupby("file", sort=False):
        kanallar = np.sort(grup["channel"].unique())
        if len(kanallar) <= kanal_basina:
            secilen = kanallar
        else:
            # k+2 nokta uret, ilk ve sonuncuyu at -> uc kanallar dislanir
            idx = np.linspace(0, len(kanallar) - 1, kanal_basina + 2)[1:-1]
            secilen = kanallar[np.unique(np.round(idx).astype(int))]
        parcalar.append(grup[grup["channel"].isin(secilen)])
    if not parcalar:
        return df.iloc[:0]
    return pd.concat(parcalar, ignore_index=True)


# ---------------------------------------------------------------
# uint8 <-> dB
# ---------------------------------------------------------------
def db_to_uint8(db, top_db=rd.TOP_DB):
    """
    dB [-top_db, 0]  ->  uint8 [0, 255].  Adim = top_db/255 = 0.31 dB.

    spektrogram() ciktisi ref=max ile normalize edildigi icin ust sinir
    her zaman 0.0, alt sinir -top_db. Yani sabit bir aralik; her pencere
    icin ayri olcek saklamaya gerek yok.
    """
    x = np.clip(db, -top_db, 0.0)
    return np.round((x + top_db) * (255.0 / top_db)).astype(np.uint8)


def uint8_to_db(u, top_db=rd.TOP_DB):
    """Tersi. Egitim tarafi bunu cagirir."""
    return u.astype(np.float32) * (top_db / 255.0) - top_db


# ---------------------------------------------------------------
# 1) KESIF -- kenar kanallar gercekten daha mi bos?
# ---------------------------------------------------------------
def kesif(kok=KOK, csv="train_final.csv", n_dosya=800, tohum=42):
    """
    Bos pencere oraninin KANAL KONUMUNA gore degisip degismedigini olcer.

    NEDEN OLCULUYOR
    ---------------
    Rapor 6.1 bos pencereleri "araligin kenarindaki kanallarda sinyal
    zayiflamis olabilir" diye acikliyordu -- ama bu bir TAHMINDI, olculmemisti.
    alt_orneklem'in hangi kanallari secmesi gerektigi tam da buna bagli:

        tahmin dogruysa -> uc kanallar dislanmali (mevcut kural)
        tahmin yanlissa -> uclari dislamanin bir faydasi yok, kural
                           gereksiz yere karmasiklastiriyor demektir

    Projede kural: olcmeden karar verilmez. Bu fonksiyon o olcumu yapar.

    YONTEM -- VE NEDEN MARJINAL TABLO YETMEZ
    ----------------------------------------
    En az 3 kanali olan dosyalardan rastgele n_dosya secilir, HER kanali
    okunur (alt orneklem YOK -- olcecegimiz sey zaten kanal secimi).
    Her satir icin kanalin dosya icindeki goreli konumu hesaplanir:

        konum = sira / (kanal_sayisi - 1)      0.0 = ilk, 1.0 = son

    Ilk surum burada durup "uc kanallarin bos orani vs ic kanallarin bos
    orani" diye tek bir marjinal karsilastirma yapiyordu. BU YANLIS, cunki
    iki karistirici var:

      1. KANAL SAYISI. 3 kanalli bir dosya yalnizca UC kovalarina katki
         verir, ic kovalara giremez. Yani kovalar ayni dosyalari
         ornekleMIyor.
      2. SINIF. noise'da bos orani ~%1, cutting'de ~%25. Az kanalli
         dosyalar agirlikli noise ise, "uclar daha az bos" sonucu
         tamamen bundan dogar -- konumla ilgisi olmaz.

    Bu, projenin bir kez dustugu hatanin aynisi (DURUM.md Bolum 6, "tek
    degiskenli F testine guvenmek"): kontrolsuz marjinal karsilastirmadan
    sonuc cikarmak.

    COZUM -- ESLESTIRILMIS TEST (Bolum B):
        Her dosyada KENDI uc kanallarinin bos orani ile KENDI ic
        kanallarinin bos orani ayri hesaplanip farki alinir. Ayni dosya =
        ayni sinif, ayni kanal sayisi, ayni kayit kosullari. Iki
        karistirici da tanim geregi sabitlenir. Sonra bu farklarin
        ortalamasi sifirdan anlamli olcude farkli mi diye bakilir.

    Bolum C (sinif icinde ayri ayri) ve Bolum D (yonlu degisim var mi)
    ikinci ve ucuncu kontrollerdir.
    """
    _baslik(f"KESIF -- bos pencere orani kanal konumuna gore degisiyor mu?")
    if h5py is None:
        print("  h5py yok."); return

    df = pd.read_csv(Path(kok) / csv)
    df = alt_orneklem(df, kanal_basina=None, yalniz_bin=True)

    kanal_sayisi = df.groupby("file")["channel"].transform("nunique")
    df = df[kanal_sayisi >= 3]
    dosyalar = df["file"].drop_duplicates()
    n_dosya = min(n_dosya, len(dosyalar))
    secilen = dosyalar.sample(n=n_dosya, random_state=tohum)
    alt = df[df["file"].isin(secilen)]
    print(f"  {n_dosya:,} dosya, {len(alt):,} satir okunacak "
          f"(tahmini {len(alt) * 3.87 / 1000:.0f} s)")

    kayitlar = []
    t0 = time.perf_counter()
    for dosya, grup in alt.groupby("file", sort=False):
        kanallar = np.sort(grup["channel"].unique())
        n_k = len(kanallar)
        sira = {c: i for i, c in enumerate(kanallar)}
        try:
            f = h5py.File(str(dosya), "r")
        except Exception:  # noqa: BLE001
            continue
        try:
            for r in grup.itertuples():
                s = rd.genlik_oku(f, r.channel, r.window_start, r.window_end)
                if s is None or s.size == 0:
                    continue
                s = rd.pencereye_oturt(s, rd.PENCERE)
                kayitlar.append({
                    "dosya": dosya,
                    "konum": sira[r.channel] / (n_k - 1),
                    "sira": sira[r.channel],          # bastan kacinci
                    "ters_sira": n_k - 1 - sira[r.channel],   # sondan
                    "n_kanal": n_k,
                    "sinif": r.event,
                    "bos": bool(rd.bos_mu(s)),
                })
        finally:
            f.close()
    sure = time.perf_counter() - t0

    k = pd.DataFrame(kayitlar)
    if k.empty:
        print("  Hicbir satir okunamadi."); return
    print(f"  okundu: {len(k):,} pencere, {sure:.0f} s "
          f"({1000 * sure / len(k):.2f} ms/satir)")

    k["uc_mu"] = (k["konum"] <= 0.001) | (k["konum"] >= 0.999)

    # --- A) HAM (marjinal) tablo -- karistiricilar kontrol EDILMEDI ---
    _alt("A) HAM tablo -- bos orani konuma gore  (KARISTIRICI VAR, altta)")
    kova = pd.cut(k["konum"], bins=[-.01, .001, .25, .5, .75, .999, 1.01],
                  labels=["ILK (uc)", "0-25%", "25-50%", "50-75%",
                          "75-100%", "SON (uc)"])
    tablo = k.groupby(kova, observed=False).agg(
        pencere=("bos", "size"), bos=("bos", "mean"),
        ort_kanal=("n_kanal", "mean"))
    sinif_pay = pd.crosstab(kova, k["sinif"], normalize="index")
    print(f"  {'konum':<12}{'pencere':>9}{'bos':>8}{'ort_n_kanal':>13}"
          f"   sinif karisimi")
    for ad, sat in tablo.iterrows():
        if sat["pencere"] == 0:
            continue
        pay = "  ".join(f"{s[:4]} {sinif_pay.loc[ad, s]:.0%}"
                        for s in sinif_pay.columns)
        print(f"  {str(ad):<12}{int(sat['pencere']):>9,}{sat['bos']:>8.1%}"
              f"{sat['ort_kanal']:>13.1f}   {pay}")

    print(f"\n  ^ 'ort_n_kanal' ve 'sinif karisimi' sutunlari KOVADAN KOVAYA")
    print(f"    degisiyorsa bu tablo yaniltir: 3 kanalli bir dosya yalnizca")
    print(f"    UC kovalarina girer, ic kovalara HIC giremez. Ayrica noise'un")
    print(f"    bos orani ~%1, cutting'inki ~%25 -- kovalarin sinif karisimi")
    print(f"    farkliysa olculen sey konum degil SINIF olur.")

    # --- B) ESLESTIRILMIS dosya-ici karsilastirma  <-- ASIL TEST ---
    _alt("B) ESLESTIRILMIS test -- her dosya KENDI ucu ile KENDI ici")
    print(f"  Her dosyada uc kanallarin bos orani ile ic kanallarin bos orani")
    print(f"  ayri hesaplanip FARKI aliniyor. Ayni dosya = ayni sinif, ayni")
    print(f"  kanal sayisi, ayni kayit kosullari -> karistiricilar sabit.")

    genis = k[k["n_kanal"] >= 5]
    farklar = []
    for dosya, g in genis.groupby("dosya", sort=False):
        u, i = g[g["uc_mu"]]["bos"], g[~g["uc_mu"]]["bos"]
        if len(u) and len(i):
            farklar.append(u.mean() - i.mean())
    farklar = np.array(farklar)

    if len(farklar) < 20:
        print(f"\n  Yeterli dosya yok ({len(farklar)}). --n-dosya artirilmali.")
    else:
        ort = float(farklar.mean())
        se = float(farklar.std(ddof=1) / np.sqrt(len(farklar)))
        t = ort / se if se else 0.0
        arti = int((farklar > 0).sum())
        eksi = int((farklar < 0).sum())
        print(f"\n  degerlendirilen dosya : {len(farklar):,} "
              f"(>=5 kanalli, hem uc hem ic satiri olan)")
        print(f"  ortalama fark (uc-ic) : {ort:+.1%}  "
              f"(standart hata {se:.1%}, t = {t:+.1f})")
        print(f"  isaret sayimi         : uc daha bos {arti:,} dosya  |  "
              f"ic daha bos {eksi:,} dosya  |  esit {len(farklar)-arti-eksi:,}")

        if abs(t) < 2:
            karar = ("FARK YOK. Uc kanallar ile ic kanallar arasinda bos "
                     "orani farki\n     yok. Rapor 6.1'in tahmini "
                     "DOGRULANMADI -- uc dislama kurali\n     gereksiz.")
        elif t > 0:
            karar = ("UC KANALLAR DAHA BOS. Rapor 6.1'in tahmini dogrulandi,\n"
                     "     alt_orneklem'in uclari dislamasi yerinde.")
        else:
            karar = ("UC KANALLAR DAHA AZ BOS. Tahminin TERSI cikti;\n"
                     "     uc dislama kurali ZARARLI, kaldirilmali.")
        print(f"\n  -> {karar}")

    # --- C) Sinif icinde ayri ayri (ikinci kontrol) ---
    _alt("C) Sinif icinde uc vs ic  (>=5 kanalli dosyalar)")
    print(f"  {'sinif':<12}{'uc n':>8}{'uc bos':>9}{'ic n':>8}{'ic bos':>9}"
          f"{'fark':>9}")
    for sinif, g in genis.groupby("sinif"):
        u, i = g[g["uc_mu"]]["bos"], g[~g["uc_mu"]]["bos"]
        if len(u) < 20 or len(i) < 20:
            continue
        print(f"  {str(sinif):<12}{len(u):>8,}{u.mean():>9.1%}"
              f"{len(i):>8,}{i.mean():>9.1%}{u.mean()-i.mean():>+9.1%}")

    # --- D) Yon var mi? Bastan/sondan mutlak sira ---
    _alt("D) Mutlak sira -- bos orani kanal ekseninde YONLU mu degisiyor?")
    print(f"  (A'daki tablo tek yonlu artiyorsa, mesele 'uc vs ic' degil")
    print(f"   'dusuk kanal vs yuksek kanal' olabilir)")
    print(f"  {'sira':>6}{'bastan n':>10}{'bastan bos':>12}"
          f"{'sondan n':>10}{'sondan bos':>12}")
    for s in range(6):
        b = genis[genis["sira"] == s]["bos"]
        t_ = genis[genis["ters_sira"] == s]["bos"]
        if len(b) < 20 and len(t_) < 20:
            continue
        print(f"  {s:>6}{len(b):>10,}{b.mean():>12.1%}"
              f"{len(t_):>10,}{t_.mean():>12.1%}")

    _alt("E) Bos pencere orani -- sinifa gore  (Rapor 5.4 ile karsilastir)")
    for sinif, sat in k.groupby("sinif")["bos"].agg(["size", "mean"]).iterrows():
        print(f"  {str(sinif):<12}{int(sat['size']):>10,}{sat['mean']:>11.1%}")
    print(f"  {'TOPLAM':<12}{len(k):>10,}{k['bos'].mean():>11.1%}")
    return k


# ---------------------------------------------------------------
# 2) ONBELLEK KURULUMU
# ---------------------------------------------------------------
def kur(kok=KOK, csv="train_final.csv", k=2, cikti=None, ustune_yaz=False,
        blok=512, ilerleme=1000):
    """
    Secilen satirlari spektrograma cevirip uint8 olarak tek bir HDF5'e yazar.

    k=0 veya None -> kanal elemesi yok (tum satirlar).

    Cikti dosyasinin icerigi:
        spektrogram : uint8 (N, 129, 231)   dB [-80,0] kuantalanmis
        etiket      : uint8 (N,)            0=cutting 1=climbing 2=noise
        dosya_idx   : int32 (N,)            `dosyalar` dizisine indeks
        kanal       : int32 (N,)
        pencere_bas : int64 (N,)            CSV'deki window_start
        pencere_son : int64 (N,)            CSV'deki window_end
        dosyalar    : string (M,)           izlenebilirlik icin
        + attrs     : fs, pencere, n_fft, hop, top_db, alan, k, kaynak csv...

    Bos pencereler YAZILMAZ (real_data.pencere_yukle None dondurur).
    Kac tanesinin elendigi rapor edilir ve attrs'a yazilir.

    Satirlar DOSYA DOSYA islenir (gruplu okuma, 1.7 kat hizli) ama cikti
    sirasi CSV sirasini korumaz -- egitim zaten karistiriyor, val/test
    icin de indeks sutunlari izlenebilirligi sagliyor.
    """
    _baslik(f"ONBELLEK KURULUMU -- {csv}, k={k if k else 'hepsi'}")
    if h5py is None:
        print("  h5py yok."); return

    kok = Path(kok)
    df = pd.read_csv(kok / csv)
    ham = len(df)
    df = alt_orneklem(df, kanal_basina=(k or None), yalniz_bin=True)
    print(f"  {ham:,} satir -> {len(df):,} satir secildi "
          f"(%{100 * len(df) / ham:.1f})")

    if cikti is None:
        cikti = kok / f"onbellek_{Path(csv).stem}_k{k or 0}.h5"
    cikti = Path(cikti)
    if cikti.exists() and not ustune_yaz:
        print(f"  !!! {cikti} ZATEN VAR. Ustune yazmak icin --ustune-yaz.")
        print(f"      (kazara uzerine yazip saatlerce yeniden kurmamak icin)")
        return

    dosyalar = df["file"].astype(str).drop_duplicates().tolist()
    dosya_idx = {d: i for i, d in enumerate(dosyalar)}

    n_frek = rd.N_FFT // 2 + 1
    n_cerceve = 1 + (rd.PENCERE - rd.N_FFT) // rd.HOP
    print(f"  spektrogram sekli: ({n_frek}, {n_cerceve})   "
          f"tahmini boyut {len(df) * 0.73 * n_frek * n_cerceve / 1e9:.1f} GB")
    print(f"  cikti: {cikti}")

    bilinmeyen_sinif = set()
    n_yaz = n_bos = n_hata = 0
    t0 = time.perf_counter()

    with h5py.File(cikti, "w") as out:
        dS = out.create_dataset(
            "spektrogram", shape=(0, n_frek, n_cerceve),
            maxshape=(None, n_frek, n_cerceve), dtype=np.uint8,
            chunks=(min(64, blok), n_frek, n_cerceve), compression="lzf")
        dY = out.create_dataset("etiket", shape=(0,), maxshape=(None,),
                                dtype=np.uint8)
        dF = out.create_dataset("dosya_idx", shape=(0,), maxshape=(None,),
                                dtype=np.int32)
        dC = out.create_dataset("kanal", shape=(0,), maxshape=(None,),
                                dtype=np.int32)
        dW = out.create_dataset("pencere_bas", shape=(0,), maxshape=(None,),
                                dtype=np.int64)
        # window_end de saklaniyor: onbellek KENDINI TARIF ETSIN. Bu olmadan
        # dogrula() bir kaydi kaynaktan yeniden hesaplayamaz -- pencere
        # uzunlugu CSV'de 1.727 farkli deger aliyor, varsayilamaz.
        dE = out.create_dataset("pencere_son", shape=(0,), maxshape=(None,),
                                dtype=np.int64)

        tS, tY, tF, tC, tW, tE = [], [], [], [], [], []

        def bosalt():
            """Tamponu diske yaz."""
            nonlocal tS, tY, tF, tC, tW, tE
            if not tS:
                return
            n0, n1 = dS.shape[0], dS.shape[0] + len(tS)
            for d, t, dt in ((dS, tS, np.uint8), (dY, tY, np.uint8),
                             (dF, tF, np.int32), (dC, tC, np.int32),
                             (dW, tW, np.int64), (dE, tE, np.int64)):
                d.resize(n1, axis=0)
                d[n0:n1] = np.asarray(t, dtype=dt)
            tS, tY, tF, tC, tW, tE = [], [], [], [], [], []

        n_dosya = len(dosyalar)
        for i, (dosya, grup) in enumerate(df.groupby("file", sort=False), 1):
            try:
                f = h5py.File(str(dosya), "r")
            except Exception:  # noqa: BLE001
                n_hata += len(grup)
                continue
            try:
                for r in grup.itertuples():
                    sinif = SINIFLAR.get(str(r.event))
                    if sinif is None:
                        bilinmeyen_sinif.add(str(r.event))
                        n_hata += 1
                        continue
                    s = rd.pencere_yukle(None, r.channel, r.window_start,
                                         r.window_end, dosya_nesnesi=f)
                    if s is None:
                        n_bos += 1
                        continue
                    tS.append(db_to_uint8(rd.spektrogram(s)))
                    tY.append(sinif)
                    tF.append(dosya_idx[str(dosya)])
                    tC.append(int(r.channel))
                    tW.append(int(r.window_start))
                    tE.append(int(r.window_end))
                    n_yaz += 1
            finally:
                f.close()

            if len(tS) >= blok:
                bosalt()
            if i % ilerleme == 0:
                gecen = time.perf_counter() - t0
                kalan = gecen * (n_dosya - i) / i
                print(f"    {i:>7,}/{n_dosya:,} dosya  yazilan {n_yaz:>8,}  "
                      f"bos {n_bos:>7,}  gecen {gecen/60:>5.1f} dk  "
                      f"kalan ~{kalan/60:.1f} dk", flush=True)
        bosalt()

        out.create_dataset("dosyalar", data=np.array(dosyalar, dtype=object),
                           dtype=h5py.string_dtype())
        out.attrs.update({
            "kaynak_csv": csv, "kok": str(kok), "k": k or 0,
            "fs": rd.FS, "pencere": rd.PENCERE, "alan": rd.ALAN,
            "n_fft": rd.N_FFT, "hop": rd.HOP, "top_db": rd.TOP_DB,
            "bos_esik": rd.BOS_ESIK, "bos_frekans": rd.BOS_FREKANS,
            "siniflar": [IDX_SINIF[i] for i in range(len(IDX_SINIF))],
            "ham_satir": ham, "secilen_satir": len(df),
            "yazilan": n_yaz, "elenen_bos": n_bos, "hata": n_hata,
            "uretim": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    sure = time.perf_counter() - t0
    _alt("SONUC")
    print(f"  yazilan pencere : {n_yaz:,}")
    print(f"  elenen (bos)    : {n_bos:,}  (%{100*n_bos/max(len(df),1):.1f})")
    print(f"  hata            : {n_hata:,}")
    if bilinmeyen_sinif:
        print(f"  !!! bilinmeyen sinif etiketleri: {sorted(bilinmeyen_sinif)}")
    print(f"  sure            : {sure/60:.1f} dk "
          f"({1000*sure/max(len(df),1):.2f} ms/satir)")
    print(f"  dosya boyutu    : {cikti.stat().st_size/1e9:.2f} GB")
    print(f"\n  Dogrulamak icin: python onbellek_kur.py --dogrula {cikti}")
    return cikti


# ---------------------------------------------------------------
# 3) DOGRULAMA
# ---------------------------------------------------------------
def dogrula(onbellek, n=20, tohum=0):
    """
    Onbellek gercekten hattin urettigiyle ayni mi?

    NEDEN GEREKLI: Onbellek, egitimin gordugu TEK sey olacak. Kurulumda bir
    hata olursa (yanlis kanal, kaydirilmis pencere, ters etiket) egitim
    sessizce yanlis veriyle calisir ve bunu hicbir metrik gostermez --
    model sadece "ogrenemiyor" gorunur.

    Bu yuzden rastgele n kayit secilip KAYNAK DOSYADAN yeniden hesaplanir
    ve onbellektekiyle karsilastirilir. Onbellek (dosya, kanal, pencere_bas,
    pencere_son) dortlusunu sakladigi icin yeniden hesaplama BIREBIR ayni
    girdiyi kullanir -- hat deterministik oldugundan beklenen fark TAM 0.0.

    Sifirdan buyuk herhangi bir fark GERCEK bir hatadir (yanlis kanal,
    kaydirilmis pencere, bozuk yazim). Esik gevsek tutulmuyor: bir
    dogrulamanin degeri, BASARISIZ OLABILMESINDEN gelir.
    """
    _baslik(f"DOGRULAMA -- {onbellek}")
    if h5py is None:
        print("  h5py yok."); return

    with h5py.File(str(onbellek), "r") as f:
        N = f["spektrogram"].shape[0]
        print(f"  {N:,} pencere, sekil {f['spektrogram'].shape[1:]}")
        print(f"  kaynak: {f.attrs.get('kaynak_csv')}  k={f.attrs.get('k')}")
        print(f"  elenen bos: {f.attrs.get('elenen_bos'):,} / "
              f"secilen {f.attrs.get('secilen_satir'):,}")

        y = f["etiket"][:]
        siniflar = list(f.attrs.get("siniflar", []))
        _alt("Sinif dagilimi")
        for i, ad in enumerate(siniflar):
            adet = int((y == i).sum())
            print(f"  {ad:<12}{adet:>10,}  (%{100*adet/max(N,1):.1f})")

        dosyalar = [d.decode() if isinstance(d, bytes) else str(d)
                    for d in f["dosyalar"][:]]
        rng = np.random.default_rng(tohum)
        idx = np.sort(rng.choice(N, size=min(n, N), replace=False))

        if "pencere_son" not in f:
            print("\n  !!! Bu onbellek 'pencere_son' tasimayan eski bir surum.")
            print("      Birebir dogrulama yapilamaz, yeniden kurulmali.")
            return

        _alt(f"{len(idx)} rastgele kayit kaynaktan yeniden hesaplaniyor")
        farklar, hatali = [], 0
        for i in idx:
            u = f["spektrogram"][i]
            dosya = dosyalar[int(f["dosya_idx"][i])]
            s = rd.pencere_yukle(dosya, int(f["kanal"][i]),
                                 int(f["pencere_bas"][i]),
                                 int(f["pencere_son"][i]))
            if s is None:
                # Onbellekte var ama simdi 'bos' cikiyorsa hattin kendisi
                # tutarsiz demektir -- bu da bir HATA, sessizce gecilemez.
                hatali += 1
                continue
            yeniden = db_to_uint8(rd.spektrogram(s))
            farklar.append(int(np.abs(u.astype(np.int16)
                                      - yeniden.astype(np.int16)).max()))

        gecti = True
        if farklar:
            farklar = np.array(farklar)
            n_ayni = int((farklar == 0).sum())
            print(f"  karsilastirilan     : {len(farklar)}")
            print(f"  BIREBIR ayni        : {n_ayni}/{len(farklar)}")
            print(f"  maks uint8 sapmasi  : {farklar.max()} "
                  f"({uint8_to_db(farklar.max()) + rd.TOP_DB:.2f} dB)")
            if n_ayni != len(farklar):
                gecti = False
                print(f"  !!! BASARISIZ: {len(farklar)-n_ayni} kayit kaynaktan")
                print(f"      yeniden hesaplandiginda FARKLI cikti. Onbellek")
                print(f"      bozuk -- egitimde kullanilmamali.")
        if hatali:
            gecti = False
            print(f"  !!! {hatali} kayit yeniden okunamadi veya 'bos' cikti.")

        print(f"\n  {'[x] DOGRULAMA GECTI' if gecti else '[!] DOGRULAMA BASARISIZ'}")
        return gecti


# ---------------------------------------------------------------
def _baslik(m):
    print(f"\n{CIFT}\n{m}\n{CIFT}")


def _alt(m):
    print(f"\n  {m}\n  {CIZGI}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Spektrogram onbellegi")
    ap.add_argument("--kok", default=KOK)
    ap.add_argument("--csv", default="train_final.csv")
    ap.add_argument("--k", type=int, default=2,
                    help="dosya basina kanal (0 = hepsi)")
    ap.add_argument("--cikti", default=None)
    ap.add_argument("--ustune-yaz", action="store_true")
    ap.add_argument("--kesif", action="store_true",
                    help="once bunu calistir: kanal konumu / bos oran olcumu")
    ap.add_argument("--n-dosya", type=int, default=800, help="kesif ornek boyu")
    ap.add_argument("--dogrula", default=None, help="kurulmus onbellegi dogrula")
    a = ap.parse_args()

    if a.dogrula:
        dogrula(a.dogrula)
    elif a.kesif:
        kesif(kok=a.kok, csv=a.csv, n_dosya=a.n_dosya)
    else:
        kur(kok=a.kok, csv=a.csv, k=a.k, cikti=a.cikti,
            ustune_yaz=a.ustune_yaz)
