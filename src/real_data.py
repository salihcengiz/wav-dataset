"""
GERCEK SAHA VERISI -- ON ISLEME HATTI

Bu modul, .bin.hdf5 dosyalarindan model girdisi uretmenin TEK kaynagidir.

=== NEDEN TEK FONKSIYON ===

Ekibin mevcut kodunda egitim ve cikarim ayri yollardan gidiyor ve DORT
noktada birbirinden ayrilmislar (bkz. GERCEK_VERI_FAZ0_RAPORU.md Bolum 6.3):

    olcek katsayisi : cikarimda /16384 var, egitimde yok
    pencere boyutu  : 15000 (support_set_creator) vs 20000 (test.py)
    P / S bileseni  : kod 'P' aliyor, yorum 'S' diyor
    sessiz pencere  : cikarimda eleniyor, egitimde elenmiyor

Model, egitimde gordugu bicimi bekler. Cikarimda farkli bir sey verirsen
tahminleri coper. Bu yuzden burada egitim ve cikarim AYNI fonksiyonu
cagirir -- iki ayri kod yolu yoktur, dolayisiyla tutarsizlik imkansizdir.

=== KARARLAR (GERCEK_VERI_FAZ0_RAPORU.md Bolum 1.5) ===

    bicim       : .bin.hdf5      (val/test'in %100'u)
    sinyal      : hypot(re, im)  yalnizca P alanindan
    pencere     : 15.000 ornek = 7.5 s @ 2000 Hz
    standart    : uzunsa enerji merkezine kirp, kisaysa yansitmali doldur
    bos pencere : elenir (olay pencerelerinin ~%25'i bos)
    normalizasyon: pencere-ici (medyan / MAD)  -- ZORUNLU
    olcek kats. : uygulanmaz (pencere-ici normalizasyon zaten sadelestirir)

=== BAGIMLILIK ===

Yalnizca numpy + h5py. Uzak sunucudaki JupyterLab hucresine yapistirilarak
da calisabilsin diye bu depodaki hicbir modulu import etmez.
"""
import numpy as np

try:
    import h5py
except ImportError:  # yalnizca spektrogram fonksiyonlari kullanilacaksa gerekmez
    h5py = None


# ---------------------------------------------------------------
# VARSAYILANLAR
# ---------------------------------------------------------------
FS = 2000               # Hz -- .bin dosyalarinda dogrulandi (duration = n/prf)
PENCERE = 15_000        # ornek = 7.5 s
ALAN = "P"              # P veya S
N_FFT = 256             # sentetik veri setiyle ayni
HOP = 64                # sentetik veri setiyle ayni
TOP_DB = 80.0           # dB tabani (librosa.amplitude_to_db varsayilani)

# Bos pencere esigi: 500 Hz ustundeki enerji payi. Duz (beyaz) bir spektrumda
# bu deger 0.5'tir; olculen degerler climbing %20, cutting %28, noise %0.
BOS_ESIK = 0.45
BOS_FREKANS = 500.0


# ---------------------------------------------------------------
# 1) OKUMA
# ---------------------------------------------------------------
def genlik_oku(f, kanal, ws, we, alan=ALAN):
    """
    Acik bir HDF5 dosyasindan bir kanalin bir dilimini GENLIK olarak oku.

    Ekibin yontemi: karmasik I/Q'nun buyuklugu.
        genlik = |z| = sqrt(re^2 + im^2)

    Faz KULLANILMAZ. Olculdu: unwrap(angle(z)) sonrasi diff, her kanalda
    std = 1.814 veriyor -- [-pi,pi] duzgun dagilimin std'si tam olarak
    pi/sqrt(3) = 1.8138. Yani faz saf rastgele. Sebebi ham re/im
    degerlerinin ±11 araliginda olmasi (22 benzersiz deger), faz
    kuantalanmasi ~19 derece.

    .sdf dosyalarinda P alani duz float'tir, o durumda dogrudan kullanilir.

    Donen: float64 dizi, veya kanal yoksa None
    """
    ch = str(kanal)
    if ch not in f:
        return None
    ham = f[ch][int(ws):int(we)]          # once dilimle, sonra alan sec (hizli)

    if ham.dtype.names and alan in ham.dtype.names:
        ham = ham[alan]
    if ham.dtype.names and "re" in ham.dtype.names:
        return np.hypot(ham["re"].astype(np.float64),
                        ham["im"].astype(np.float64))
    return np.asarray(ham, dtype=np.float64)


# ---------------------------------------------------------------
# 2) PENCEREYE OTURTMA  <-- 7.5 SANIYELIK PENCERE BURADA URETILIYOR
# ---------------------------------------------------------------
def pencereye_oturt(s, pencere=PENCERE):
    """
    Degisken uzunluktaki bir dilimi tam olarak `pencere` ornege getirir.

    ***  7.5 SANIYELIK PENCERE TAM OLARAK BURADA OLUSUYOR.  ***
    pencere=15000 ve FS=2000 oldugu icin 15000/2000 = 7.5 saniye.

    CSV'de hazir 15.000'lik pencere YOKTUR; CSV degisken uzunluk verir
    (val/test'te 7.500 / 10.000 / 15.000 / 20.000). Standartlastirma
    yukleme aninda, burada yapilir.

    Kural ekibin support_set_creator.py dosyasindan alinmistir (kendi
    kuralimizi uydurmuyoruz):

      UZUNSA  -> ENERJI AGIRLIK MERKEZINE ortalayarak kirp.
                 Agirlik merkezi c = sum(i * s[i]^2) / sum(s[i]^2), yani
                 sinyalin en yogun oldugu nokta. Olayin ortasi kadrajda
                 kalsin diye baslangica veya sona gore degil, enerjiye
                 gore ortalanir.

      KISAYSA -> YANSITMALI (reflect) doldur, simetrik olarak iki yana.
                 Sifirla doldurmak spektrograma yapay bir kesinti
                 ekleyecegi icin yansitma tercih edilir.
    """
    s = np.asarray(s, dtype=np.float64)
    n = s.shape[0]
    if n == pencere:
        return s

    if n > pencere:
        p = s ** 2
        toplam = p.sum()
        if toplam > 0:
            c = int(round(float((np.arange(n) * p).sum() / toplam)))
        else:
            c = n // 2
        bas = max(0, min(c - pencere // 2, n - pencere))
        return s[bas:bas + pencere]

    eksik = pencere - n
    sol, sag = eksik // 2, eksik - eksik // 2
    mod = "reflect" if n > 1 else "edge"
    return np.pad(s, (sol, sag), mode=mod)


# ---------------------------------------------------------------
# 3) BOS PENCERE KONTROLU
# ---------------------------------------------------------------
def bosluk_orani(s, fs=FS, frekans=BOS_FREKANS):
    """
    Sinyalin ne kadar "duz" (beyaz gurultu benzeri) oldugunu olcer.

    `frekans` Hz ustundeki enerji payini dondurur. Tam duz bir spektrumda
    (0-1000 Hz) bu deger 0.5'e yakindir; yapisi olan bir sinyalde daha
    dusuktur cunku enerji dusuk frekanslarda toplanir.
    """
    x = np.asarray(s, dtype=np.float64)
    x = x - x.mean()
    G = np.abs(np.rfft(x)) ** 2 if hasattr(np, "rfft") else np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    top = G.sum()
    if top <= 0:
        return 1.0
    return float(G[f >= frekans].sum() / top)


def bos_mu(s, esik=BOS_ESIK, fs=FS):
    """
    Pencerede tespit edilebilir bir olay var mi?

    Olculdu: olay etiketli pencerelerin ~%25'i (climbing %20, cutting %28)
    spektral olarak bos. noise sinifinda %0. Bunlar muhtemelen olaydan uzak
    kanallar -- labels tablosu kanal ARALIGI veriyor, aralik kenarindaki
    kanallarda sinyal zayiflamis olabilir.

    Ekibin test.py'si de cikarimda sessiz pencereleri eliyor
    (np.mean(window**2) < 1e-4) ama egitimde elemiyordu. Biz iki tarafta
    da eliyoruz.
    """
    return bosluk_orani(s, fs=fs) > esik


# ---------------------------------------------------------------
# 4) PENCERE-ICI NORMALIZASYON
# ---------------------------------------------------------------
def normalize_et(s):
    """
    (s - medyan) / MAD

    ZORUNLU. Uc sorunu birden cozer:

      1. Kanaldan kanala genlik seviyesi 36 kata kadar degisiyor
         (olculdu: 1.6 ile 58.4 arasi). Seviye sinifla ilgili DEGIL
         (ham_ort F = 0.164), yani model icin sadece gurultu.
      2. /16384 olcek katsayisi meselesini gereksiz kilar -- sabit bir
         carpan medyana bolununce sadelesir.
      3. P ile S arasindaki genlik farkini (2 kat) etkisiz kilar.

    Medyan ve MAD kullaniliyor (ortalama/std degil) cunku darbeli
    sinyallerde birkac buyuk tepe ortalamayi ve std'yi cekiyor; medyan
    bunlara dayaniklidir.
    """
    x = np.asarray(s, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + 1e-9)


# ---------------------------------------------------------------
# 5) TEK GIRIS NOKTASI
# ---------------------------------------------------------------
def pencere_yukle(dosya, kanal, ws, we, *, alan=ALAN, pencere=PENCERE,
                  normalize=True, bos_ele=True, fs=FS, dosya_nesnesi=None):
    """
    Bir CSV satirindan modele hazir 1B sinyali uretir.

    *** EGITIM DE CIKARIM DA BU FONKSIYONU CAGIRIR. ***

    Adimlar:
        1. HDF5'ten dilimi oku, genlige cevir       (genlik_oku)
        2. `pencere` ornege standartlastir          (pencereye_oturt)
        3. Bos mu diye bak, bossa None dondur       (bos_mu)
        4. Pencere-ici normalize et                 (normalize_et)

    dosya_nesnesi: acik bir h5py.File verilirse tekrar acilmaz (toplu
        yuklemede ayni dosyadan cok satir okunurken hizlandirir).

    Donen: (pencere,) float64 dizi, veya elenmisse None
    """
    if h5py is None:
        raise ImportError("h5py gerekli")

    if dosya_nesnesi is not None:
        s = genlik_oku(dosya_nesnesi, kanal, ws, we, alan=alan)
    else:
        with h5py.File(str(dosya), "r") as f:
            s = genlik_oku(f, kanal, ws, we, alan=alan)

    if s is None or s.size == 0:
        return None

    s = pencereye_oturt(s, pencere)

    if bos_ele and bos_mu(s, fs=fs):
        return None

    return normalize_et(s) if normalize else s


# ---------------------------------------------------------------
# 6) SPEKTROGRAM
# ---------------------------------------------------------------
def _hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def spektrogram(s, n_fft=N_FFT, hop=HOP, fs=FS, top_db=TOP_DB):
    """
    STFT -> dB spektrogram. numpy disinda bagimliligi yoktur.

    Parametreler sentetik veri setiyle AYNI (n_fft=256, hop=64 @ 2000 Hz),
    boylece onceden egitilmis modelin gordugu temsille ortusur:
        129 frekans bini x ~234 zaman cercevesi  (15.000 ornek icin)

    dB donusumu librosa.amplitude_to_db(ref=np.max) ile ayni:
    en yuksek degere gore normalize edilir, top_db altindaki degerler kirpilir.

    Donen: (129, cerceve) float64, dikey=frekans, yatay=zaman
    """
    x = np.asarray(s, dtype=np.float64)
    x = x - x.mean()                    # DC cikar -- genlik hep pozitif, ofsetli

    n = len(x)
    n_cerceve = 1 + max(0, (n - n_fft) // hop)
    if n_cerceve <= 0:
        raise ValueError(f"sinyal cok kisa: {n} ornek, n_fft={n_fft}")

    pencere = _hann(n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_cerceve)[:, None]
    cerceveler = x[idx] * pencere[None, :]
    S = np.abs(np.fft.rfft(cerceveler, axis=1)).T      # (frekans, zaman)

    ref = S.max()
    if ref <= 0:
        return np.full_like(S, -top_db)
    db = 20.0 * np.log10(np.maximum(S, 1e-10) / ref)
    return np.maximum(db, -top_db)


def hazirla(dosya, kanal, ws, we, **kw):
    """pencere_yukle + spektrogram. Elenirse None."""
    s = pencere_yukle(dosya, kanal, ws, we, **kw)
    return None if s is None else spektrogram(s)


# ---------------------------------------------------------------
# KENDI KENDINE TEST (veri gerektirmez)
# ---------------------------------------------------------------
def self_test():
    cizgi = "-" * 70
    print("=" * 70)
    print("GERCEK VERI ON ISLEME HATTI -- BIRIM TESTI")
    print("=" * 70)
    rng = np.random.default_rng(0)

    print(f"\n[1] pencereye_oturt -- 7.5 SANIYELIK PENCERE URETIMI")
    print(cizgi)
    for n in (7_500, 10_000, 15_000, 20_000, 35_634, 80_000):
        s = rng.normal(5, 1, n)
        o = pencereye_oturt(s, PENCERE)
        islem = "kirpildi" if n > PENCERE else ("dolduruldu" if n < PENCERE else "aynen")
        print(f"  {n:>7,} ornek ({n/FS:>5.2f} s)  ->  {len(o):>6,} ornek "
              f"({len(o)/FS:.2f} s)   {islem}")
        assert len(o) == PENCERE
    print(f"  [x] Hepsi {PENCERE:,} ornek = {PENCERE/FS} saniye")

    print(f"\n[2] Enerji merkezine ortalama gercekten calisiyor mu")
    print(cizgi)
    s = np.full(40_000, 0.1)
    s[30_000:31_000] = 10.0                      # olay 30.000. ornekte
    o = pencereye_oturt(s, PENCERE)
    tepe = int(np.argmax(o))
    print(f"  olay 30.000'de, pencere 15.000  ->  kirpilan pencerede tepe: {tepe:,}")
    print(f"  merkeze uzaklik: {abs(tepe - PENCERE//2):,} ornek")
    assert abs(tepe - PENCERE // 2) < 1500, "olay merkeze oturmadi"
    print(f"  [x] Olay pencerenin ortasina geldi")

    print(f"\n[3] normalize_et -- olcek bagimsizligi")
    print(cizgi)
    taban = rng.normal(0, 1, PENCERE) + 50
    for k in (1.0, 16384.0, 0.001):
        a = normalize_et(taban * k)
        print(f"  x{k:<10,.3f} -> medyan {np.median(a):+.3e}  std {a.std():.4f}")
    a1, a2 = normalize_et(taban), normalize_et(taban * 16384.0)
    print(f"  x1 ile x16384 arasindaki maks fark: {np.abs(a1-a2).max():.2e}")
    assert np.allclose(a1, a2), "normalizasyon olcekten bagimsiz degil"
    print(f"  [x] /16384 olcek katsayisi sonucu DEGISTIRMIYOR")

    print(f"\n[4] bos_mu -- beyaz gurultu vs yapili sinyal")
    print(cizgi)
    beyaz = rng.normal(0, 1, PENCERE)
    t = np.arange(PENCERE) / FS
    yapili = np.sin(2*np.pi*30*t) + 0.3*rng.normal(0, 1, PENCERE)
    for ad, x in (("beyaz gurultu", beyaz), ("30 Hz + gurultu", yapili)):
        print(f"  {ad:<18} bosluk orani {bosluk_orani(x):.3f}  "
              f"-> {'BOS' if bos_mu(x) else 'dolu'}")
    assert bos_mu(beyaz) and not bos_mu(yapili)
    print(f"  [x] Bos pencere tespiti calisiyor (esik {BOS_ESIK})")

    print(f"\n[5] spektrogram -- sekil ve deger araligi")
    print(cizgi)
    S = spektrogram(normalize_et(yapili))
    print(f"  girdi {PENCERE:,} ornek  ->  cikti {S.shape}  (frekans x zaman)")
    print(f"  beklenen frekans bini: {N_FFT//2 + 1}")
    print(f"  deger araligi: [{S.min():.1f}, {S.max():.1f}] dB")
    assert S.shape[0] == N_FFT // 2 + 1
    assert S.max() <= 0.001 and S.min() >= -TOP_DB - 0.001
    frk = np.fft.rfftfreq(N_FFT, 1.0/FS)
    tepe_bin = int(np.argmax(S.mean(axis=1)))
    print(f"  ortalama spektrumun tepesi: {frk[tepe_bin]:.0f} Hz  (30 Hz bekleniyor)")
    assert abs(frk[tepe_bin] - 30) < 20
    print(f"  [x] Spektrogram dogru frekansi buluyor")

    print(f"\n{'=' * 70}")
    print("TUM TESTLER GECTI.")
    print(f"  pencere {PENCERE:,} ornek = {PENCERE/FS} s @ {FS} Hz")
    print(f"  spektrogram {N_FFT//2+1} frekans x {S.shape[1]} cerceve")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
