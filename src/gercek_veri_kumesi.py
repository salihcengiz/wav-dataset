"""
GERCEK VERI -- ADIM 5: PYTORCH VERI KUMESI

onbellek_kur.py'nin urettigi uint8 spektrogram onbelleginden model girdisi
uretir. Egitim ve degerlendirme AYNI sinifi kullanir.

    onbellek (129 x 231 uint8)
      -> renklendir (viridis 3 kanal, veya gri)        [Dataset, CPU]
      -> (3, 129, 231) uint8 tensor                    [DataLoader ciktisi]
                              |
                              |  .to(cuda)             <- 1.8 MB/batch
                              v
      -> [0,1] -> 224x320'ye olcekle -> ImageNet norm  [hazirla(), GPU]
      -> (istege bagli) zaman/frekans maskeleme        [yalnizca egitimde]
      -> (B, 3, 224, 320) float32

=== NEDEN DONUSUM GPU'DA ===

Ilk surum her ornegi CPU'da 224x320 float32'ye ceviriyordu. Batch basina
PCIe'den gecen veri 55 MB (64 x 3 x 224 x 320 x 4 bayt). uint8 gonderip
olceklemeyi GPU'da yapinca 1.8 MB -- 30 kat az.

Ustelik en pahali islem olcekleme (interpolasyon) ve GPU bosta: model
34.835 parametre, ileri/geri gecis birkac milisaniye suruyor. Darbogaz
VERI tarafinda (bu zaten olculmustu: dilim okuma %49, STFT %22).

TEK KOD YOLU korunuyor: hazirla() hem egitimde hem cikarimda cagrilir,
tensor hangi cihazdaysa orada calisir. Iki ayri donusum yolu YOK.

=== NEDEN VIRIDIS ===

Onceden egitilmis model (das_2dcnn_sk_v1.pt) sentetik asamada VIRIDIS
renkli PNG goruyordu: dB degeri matplotlib viridis ile renklendirilip
3 kanala yayilmisti. Omurganin ilk katmani (Conv 3->16) o temsile gore
ogrendi.

Onbellekte ise ham dB var. Gri tonu 3 kanala kopyalarsak modelin ilk
katmani egitimde gordugunden FARKLI bir girdi dagilimi gorur ve aktarilan
agirliklarin bir kismi bosa gider. viridis LUT'u uygulamak parite saglar.

viridis tablosu (256 x 3) modulun icine GOMULU -- matplotlib bagimliligi
yok, sunucuda kurulu olup olmadigi onemsiz.

    renk="viridis"  -> aktarim icin dogru secim (varsayilan)
    renk="gri"      -> sifirdan egitim icin yeterli, biraz daha hizli

=== NEDEN AKTARIM ARTIK TARTISMALI ===

Onceden egitilmis paket, "cok az saha verisi varken sifirdan baslamamak"
icin uretilmisti (MODEL_CARD). O varsayim ARTIK GECERLI DEGIL:

    sentetik : 959 spektrogram, 19 BAGIMSIZ kayit
    gercek   : 220.834 pencere, 21.101 BAGIMSIZ dosya

1.100 kat daha fazla bagimsiz kaynak. Bu olcekte sifirdan egitim muhtemelen
en az aktarim kadar iyi, hatta daha iyi olur -- sentetik veri gercek DAS
verisinin yerini tutmuyor (MODEL_CARD'in kendi uyarisi).

Karar OLCUMLE verilmeli: iki kosu (sifirdan / aktarimli), ayni bolmeler,
ayni tohum. Ikisi de raporlanir. Bu, raporda "sentetik on-egitim ise
yariyor mu" diye somut bir bolum acar.

=== ONBELLEK NEDEN HER ISCIDE AYRI ACILIYOR ===

h5py dosya tutamaclari fork() sonrasi PAYLASILAMAZ. DataLoader(num_workers>0)
surecleri fork ile kopyalar; ana surecte acilmis bir tutamac cocuk
sureclerde sessizce bozuk veri dondurebilir. Bu yuzden dosya __init__'te
DEGIL, her iscide ilk erisimde aciliyor.

=== BAGIMLILIK ===

numpy + h5py + torch. torchvision GEREKMEZ (olcekleme F.interpolate ile).
"""
import base64
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except ImportError:
    h5py = None

_burada = str(Path(__file__).resolve().parent)
if _burada not in sys.path:
    sys.path.insert(0, _burada)


# ---------------------------------------------------------------
# SABITLER
# ---------------------------------------------------------------
GIRDI_H, GIRDI_W = 224, 320          # config.INPUT_H / INPUT_W ile ayni
TOP_DB = 80.0                        # real_data.TOP_DB

# ImageNet istatistikleri -- onceden egitilmis model bunlarla egitildi
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# matplotlib viridis, 256 x 3 uint8. zlib + base64 ile gomulu.
_VIRIDIS_B64 = (
    "eNoBAAP//EQBVEQCVkUEV0UFWUYHWkYIXEYKXUYLXkcNYEcOYUcQY0cRZEcTZUgUZ0gWaEgXaUgY"
    "akgabEgbbUgcbkgdb0gfcEggcUghc0gjdEgkdUgldkgmd0goeEgpeUcqekcsekcte0cufEcvfUYw"
    "fkYyfkYzf0Y0gEU1gUU3gUU4gkQ5g0Q6g0Q7hEM9hEM+hUI/hUJAhkJBhkFCh0FEh0BFiEBGiD9H"
    "iD9IiT5JiT5KiT5Mij1Nij1OijxPijxQiztRiztSizpTizpUjDlVjDlWjDhYjDhZjDdajDdbjTZc"
    "jTZdjTVejTVfjTRgjTRhjTNijTNjjTJkjjJljjFmjjFnjjFojjBpjjBqji9rji9sji5tji5uji5v"
    "ji1wji1xjixxjixyjixzjit0jit1jip2jip3jip4jil5jil6jil7jih8jih9jid+jid/jieAjiaB"
    "jiaCjiaCjiWDjiWEjiWFjiSGjiSHjiOIjiOJjiOKjSKLjSKMjSKNjSGOjSGPjSGQjSGRjCCSjCCS"
    "jCCTjB+UjB+Vix+Wix+Xix+Yix+Zih+aih6bih6ciR6diR+eiR+fiB+giB+hiB+hhx+ihyCjhiCk"
    "hiGlhSGmhSKnhSKohCOpgySqgyWrgiWsgiatgSetgSiugCmvfyqwfyyxfi2yfS6zfC+0fDG1ezK2"
    "ejS2eTW3eTe4eDi5dzq6dju7dT28dD+8c0C9ckK+cUS/cEbAb0jBbkrBbUzCbE7Da1DEalLFaVTF"
    "aFbGZ1jHZVrIZFzIY17JYmDKYGPLX2XLXmfMXGnNW2zNWm7OWHDPV3PQVnXQVHfRU3rRUXzSUH/T"
    "ToHTTYTUS4bVSYnVSIvWRo7WRZDXQ5PXQZXYQJjYPpvZPJ3ZO6DaOaLaN6XbNqjbNKrcMq3cMLDd"
    "L7LdLbXeK7jeKbreKL3fJsDfJcLfI8XgIcjgIMrhH83hHdDhHNLiG9XiGtjiGdrjGd3jGN/jGOLk"
    "GOXkGefkGerlGuzlG+/lHPHlHfTmHvbmIPjmIfvnI/3nJRfoSno="
)


def viridis_lut():
    """256 x 3 uint8 viridis tablosu. matplotlib gerekmez."""
    ham = zlib.decompress(base64.b64decode(_VIRIDIS_B64))
    return np.frombuffer(ham, dtype=np.uint8).reshape(256, 3)


# ---------------------------------------------------------------
# VERI KUMESI
# ---------------------------------------------------------------
class OnbellekKumesi(Dataset):
    """
    onbellek_kur.py'nin urettigi .h5 dosyasindan okur.

    Parametreler
    ------------
    yol : str | Path
        onbellek_{split}_final_k0.h5
    indeksler : dizi | None
        Kullanilacak satirlarin indeksleri. None -> hepsi.
        Alt orneklem icin onbellek_kur.onbellek_alt_kume() ile uretilir --
        onbellegi yeniden kurmaya gerek YOK.
    egitim : bool
        Yalnizca yukleyici()'nin shuffle varsayilanini belirler. Maskeleme
        artik burada DEGIL, hazirla(egitim=True) icinde.
    renk : "viridis" | "gri"
    bellege_al : bool
        True ise TUM onbellek bir kez okunup RAM'de tutulur.

        *** KARISTIRILMIS OKUMA ICIN ZORUNLU. ***

        Onbellek chunks=(64,129,231) + LZF ile yazildi. shuffle=True iken
        her ornek rastgele bir yerden istenir ve HDF5 tek ornek icin 64
        orneklik BUTUN blogu acmak zorunda kalir. Ustelik h5py'nin
        varsayilan blok onbellegi 1 MB, blok 1.9 MB -- hicbir zaman
        tutmaz, her okuma bastan acar. Olculen sonuc: 900 ornek/s
        (RTX 3090'da 34.835 parametrelik model icin absurt derecede yavas).

        RAM'e alinca rastgele erisim bedava olur. Maliyet: train icin
        6.58 GB (129*231 bayt x 220.834). Yukleme SIRALI oldugu icin her
        blok yalnizca BIR kez aciliyor.

        val/test icin gerekmez: sirali okunuyorlar ve batch=64 tam olarak
        bir bloga denk geliyor, yani blok basina tek acilis.
    """

    def __init__(self, yol, indeksler=None, egitim=False, renk="viridis",
                 bellege_al=False, rdcc_mb=64, sessiz=False):
        if h5py is None:
            raise ImportError("h5py gerekli")
        if renk not in ("viridis", "gri"):
            raise ValueError(f"renk 'viridis' veya 'gri' olmali, {renk!r} degil")

        self.yol = str(yol)
        self.egitim = egitim
        self.renk = renk
        self.rdcc = int(rdcc_mb * 1024 * 1024)
        self._f = None                       # her iscide ayri acilacak
        self._ram = None

        with h5py.File(self.yol, "r") as f:
            self.n_toplam = int(f["spektrogram"].shape[0])
            self.sekil = tuple(f["spektrogram"].shape[1:])
            self.etiketler_tum = f["etiket"][:]
            self.siniflar = [s.decode() if isinstance(s, bytes) else str(s)
                             for s in f.attrs.get("siniflar", [])]
            self.kaynak_csv = str(f.attrs.get("kaynak_csv", "?"))

        # SIRALI tutuluyor: h5py artan sirali indeks listesi ister ve
        # bellege alma bloklari sirayla gezebilsin diye. Sira onemsiz --
        # egitimde DataLoader zaten karistiriyor, degerlendirmede metrikler
        # siradan bagimsiz.
        self.indeksler = np.sort(
            np.arange(self.n_toplam) if indeksler is None
            else np.asarray(indeksler, dtype=np.int64))
        self.etiketler = self.etiketler_tum[self.indeksler]

        self._lut = viridis_lut() if renk == "viridis" else None

        if bellege_al:
            self._bellege_al(sessiz)

    def _bellege_al(self, sessiz=False):
        """
        Kullanilacak satirlari tek seferde RAM'e okur.

        SIRALI okuma: bloklar artan sirada gezildigi icin her blok tam
        olarak bir kez aciliyor. Rastgele okumadaki 64 kat buyutme yok.
        """
        n = len(self.indeksler)
        bayt = n * int(np.prod(self.sekil))
        if not sessiz:
            print(f"    onbellek RAM'e aliniyor: {n:,} pencere, "
                  f"{bayt / 1e9:.2f} GB ...", end="", flush=True)
        t0 = time.perf_counter()
        self._ram = np.empty((n,) + self.sekil, dtype=np.uint8)
        with h5py.File(self.yol, "r", rdcc_nbytes=self.rdcc) as f:
            d = f["spektrogram"]
            adim = 4096
            for i in range(0, n, adim):
                idx = self.indeksler[i:i + adim]   # zaten sirali (bkz. __init__)
                self._ram[i:i + len(idx)] = d[idx]
        if not sessiz:
            print(f" {time.perf_counter() - t0:.0f} s")

    # --- h5py tutamaci: her iscide ayri ---
    @property
    def dosya(self):
        if self._f is None:
            # rdcc_nbytes: varsayilan 1 MB, bizim blok 1.9 MB -- varsayilanla
            # blok onbellegi HIC tutmaz, her okuma bastan acar.
            self._f = h5py.File(self.yol, "r", rdcc_nbytes=self.rdcc)
        return self._f

    def __getstate__(self):
        """
        DataLoader isci surecine kopyalanirken acik h5py tutamacini AT.

        Iki sorunu birden cozer:
          1. h5py.File picklable degil -- Windows/macOS'ta DataLoader
             'spawn' kullanir ve nesneyi pickle eder; tutamac kalsaydi
             num_workers>0 dogrudan cokerdi.
          2. Linux'ta 'fork' kullanilir; ana surecte acilmis bir tutamac
             cocuklara kopyalanir ve AYNI dosya ofsetini paylasirlar --
             bu, hata vermeden YANLIS veri okunmasina yol acabilir.
             En sinsi hata turu: egitim calisir, sonuc bozuktur.

        Tutamaci burada dusurunce her isci kendi dosyasini acar.

        NOT: `_ram` (bellege alinmis dizi) BILEREK dusurulmuyor. Linux'ta
        DataLoader 'fork' kullanir, pickle devreye girmez ve dizi
        kopyalanmadan (copy-on-write) paylasilir. 'spawn' kullanan bir
        platformda isci basina kopyalanirdi -- sunucu Linux, sorun degil,
        ama bellege_al=True + Windows + isci>0 birlesimi kullanilmamali.
        """
        durum = self.__dict__.copy()
        durum["_f"] = None
        return durum

    def __len__(self):
        return len(self.indeksler)

    def __getitem__(self, i):
        """
        Donen: (3, 129, 231) uint8 tensor ve etiket.

        Olcekleme/normalizasyon BURADA YAPILMIYOR -- hazirla() ile GPU'da
        yapiliyor (modul docstring'ine bak). Burada yalnizca renklendirme
        var, cunku o uint8 uzerinde bedava.
        """
        if self._ram is not None:
            u = self._ram[i]                        # (129, 231) uint8, RAM
        else:
            u = self.dosya["spektrogram"][int(self.indeksler[i])]
        y = int(self.etiketler[i])

        if self._lut is not None:
            rgb = self._lut[u]                      # (129, 231, 3) uint8
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        else:
            x = torch.from_numpy(np.ascontiguousarray(u))[None].repeat(3, 1, 1)
        return x, y

    # --- yardimcilar ---
    def sinif_sayilari(self):
        return np.bincount(self.etiketler, minlength=max(len(self.siniflar), 1))

    def sinif_agirliklari(self):
        """
        Dengesiz siniflar icin CrossEntropyLoss(weight=...) agirliklari.

        noise train'in yalnizca %8.8'i. Agirlik = N / (K * n_k), yani az
        gorulen sinif daha cok agirlik alir. Ortalamasi ~1 olacak sekilde
        olceklenir ki kayip buyuklugu karsilastirilabilir kalsin.
        """
        n = self.sinif_sayilari().astype(np.float64)
        n[n == 0] = 1.0
        w = n.sum() / (len(n) * n)
        return torch.tensor(w / w.mean(), dtype=torch.float32)


# ---------------------------------------------------------------
# DONUSUM -- EGITIM DE CIKARIM DA BUNU CAGIRIR
# ---------------------------------------------------------------
_MEAN = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(NORM_STD).view(1, 3, 1, 1)


def hazirla(x, egitim=False, girdi=(GIRDI_H, GIRDI_W),
            maske_p=0.5, maske_frac=0.10):
    """
    (B, 3, 129, 231) uint8  ->  (B, 3, 224, 320) float32, normalize edilmis.

    Tensor hangi cihazdaysa orada calisir -- GPU'ya tasinmis bir batch
    verilirse tum is GPU'da yapilir.

    Sira, onceden egitilmis modelin gordugu sirayla AYNI (MODEL_CARD):
        Resize -> [0,1] -> Normalize(ImageNet)

    egitim=True ise sonrasinda zaman/frekans maskelemesi uygulanir.
    Degerlendirmede MUTLAKA egitim=False -- yoksa olculen sey rastgele
    bozulmus girdiler uzerindeki performans olur.
    """
    if x.dtype == torch.uint8:
        x = x.float()
    x = F.interpolate(x, size=tuple(girdi), mode="bilinear",
                      align_corners=False, antialias=True) / 255.0
    x = (x - _MEAN.to(x.device, x.dtype)) / _STD.to(x.device, x.dtype)
    return _maskele(x, maske_p, maske_frac) if egitim else x


def _maskele(x, p=0.5, frac=0.10):
    """
    SpecAugment ruhunda zaman (dikey serit) ve frekans (yatay serit)
    maskelemesi. Batch'teki her ornege AYRI uygulanir.

    Maskelenen bolge 0 yapiliyor -- normalize edilmis uzayda 0, veri
    setinin ortalamasina karsilik gelir. "Bilgi yok" demenin dogru yolu
    budur; siyah (= -mean/std) demek degil.

    CEVIRME YOK: spektrogramda zaman eksenini ters cevirmek sesi geri
    sarmak, frekans eksenini ters cevirmek tiz ile pesi takas etmektir --
    ikisi de fiziksel olarak anlamsiz (PLAN 7.2 acikca yasakliyor).

    Kuresel RNG kullaniyor; train tarafinda set_deterministic() tohumu
    sabitledigi icin ayni tohum + ayni veri sirasi ayni sonucu verir.
    """
    B, _, H, W = x.shape
    for eksen, boy in ((2, H), (3, W)):
        uygula = torch.rand(B, device=x.device) < p
        if not bool(uygula.any()):
            continue
        en_fazla = max(2, int(boy * frac))
        for serit_no in range(2):                # PLAN 7.2: RASTGELE 1-2 serit
            if serit_no == 1:                    # ikincisi %50 olasilikla
                uygula = uygula & (torch.rand(B, device=x.device) < 0.5)
                if not bool(uygula.any()):
                    break
            genislik = torch.randint(1, en_fazla, (B,), device=x.device)
            bas = (torch.rand(B, device=x.device)
                   * (boy - genislik).clamp(min=1).float()).long()
            aralik = torch.arange(boy, device=x.device)[None, :]
            serit = (aralik >= bas[:, None]) & (aralik < (bas + genislik)[:, None])
            serit = serit & uygula[:, None]
            m = serit[:, None, :, None] if eksen == 2 else serit[:, None, None, :]
            x = x.masked_fill(m, 0.0)
    return x


def yukleyici(kume, batch=64, karistir=None, isci=4, pin=None):
    """
    DataLoader kurar.

    persistent_workers: her epoch'ta isci sureclerini yeniden kurmak, her
    birinin h5py dosyasini yeniden acmasi demek. 220 bin ornekte bu
    gereksiz maliyet.
    """
    if karistir is None:
        karistir = kume.egitim
    if pin is None:                      # GPU yoksa uyari basip bos yere ugrasir
        pin = torch.cuda.is_available()
    return DataLoader(kume, batch_size=batch, shuffle=karistir,
                      num_workers=isci, pin_memory=pin,
                      persistent_workers=isci > 0, drop_last=False)


# ---------------------------------------------------------------
# KENDI KENDINE TEST
# ---------------------------------------------------------------
def self_test(onbellek):
    cizgi = "-" * 70
    print("=" * 70)
    print("GERCEK VERI KUMESI -- BIRIM TESTI")
    print("=" * 70)

    print(f"\n[1] viridis tablosu")
    print(cizgi)
    lut = viridis_lut()
    print(f"  sekil {lut.shape}  ilk {lut[0].tolist()}  son {lut[255].tolist()}")
    assert lut.shape == (256, 3)
    assert lut[0].tolist() == [68, 1, 84], "viridis basi yanlis"
    assert lut[255].tolist() == [253, 231, 37], "viridis sonu yanlis"
    print(f"  [x] matplotlib olmadan viridis uretiliyor")

    print(f"\n[2] Veri kumesi")
    print(cizgi)
    k = OnbellekKumesi(onbellek, egitim=False)
    print(f"  kaynak {k.kaynak_csv} | {len(k):,} ornek | onbellek sekli {k.sekil}")
    print(f"  siniflar {k.siniflar}")
    print(f"  sinif sayilari {k.sinif_sayilari().tolist()}")
    print(f"  sinif agirliklari {[round(float(v), 3) for v in k.sinif_agirliklari()]}")

    x, y = k[0]
    print(f"  ornek 0 -> {tuple(x.shape)} {x.dtype}, etiket {y}")
    assert x.dtype == torch.uint8, "Dataset uint8 dondurmeli (donusum GPU'da)"
    assert x.shape == (3,) + k.sekil, f"beklenmeyen sekil {tuple(x.shape)}"
    print(f"  [x] Dataset uint8 dondururuyor (PCIe trafigi 30 kat az)")

    print(f"\n[3] hazirla() -- olcekle + normalize et")
    print(cizgi)
    xb = torch.stack([k[i][0] for i in range(4)])
    h = hazirla(xb, egitim=False)
    print(f"  {tuple(xb.shape)} {xb.dtype}  ->  {tuple(h.shape)} {h.dtype}")
    assert h.shape == (4, 3, GIRDI_H, GIRDI_W)
    assert h.dtype == torch.float32 and torch.isfinite(h).all()
    print(f"  deger araligi [{h.min():.2f}, {h.max():.2f}]  ortalama {h.mean():+.3f}")
    bayt_uint8 = xb.numel()
    bayt_float = h.numel() * 4
    print(f"  batch basina: uint8 {bayt_uint8/1e6:.2f} MB  vs  "
          f"float {bayt_float/1e6:.2f} MB   ({bayt_float/bayt_uint8:.0f}x)")
    print(f"  [x] Donusum dogru")

    print(f"\n[4] Belirlenimcilik")
    print(cizgi)
    assert torch.equal(k[3][0], k[3][0]), "Dataset belirlenimci degil"
    a = hazirla(xb, egitim=False)
    b = hazirla(xb, egitim=False)
    print(f"  egitim=False, iki cagri arasi maks fark: {(a-b).abs().max():.2e}")
    assert torch.equal(a, b), "degerlendirme yolu belirlenimci degil"
    torch.manual_seed(0); c = hazirla(xb, egitim=True)
    torch.manual_seed(0); d = hazirla(xb, egitim=True)
    print(f"  egitim=True, ayni tohumla   : {(c-d).abs().max():.2e}")
    assert torch.equal(c, d), "ayni tohum farkli maskeleme uretti"
    print(f"  [x] Tekrar uretilebilir")

    print(f"\n[5] Maskeleme")
    print(cizgi)
    torch.manual_seed(1)
    m = hazirla(xb, egitim=True)
    sifir_once = float((a == 0).float().mean())
    sifir_sonra = float((m == 0).float().mean())
    print(f"  sifir piksel orani: maskelemesiz %{100*sifir_once:.2f}  ->  "
          f"maskeli %{100*sifir_sonra:.2f}")
    assert not torch.equal(m, a), "maskeleme hic uygulanmiyor"
    assert sifir_sonra > sifir_once, "maskeleme sifir bolge eklemedi"
    print(f"  [x] Maskeleme aktif, degerlendirmede kapali")

    print(f"\n[6] Gri secenegi")
    print(cizgi)
    kg = OnbellekKumesi(onbellek, renk="gri")
    xg, _ = kg[0]
    print(f"  gri {tuple(xg.shape)} {xg.dtype}  3 kanal birebir ayni mi: "
          f"{bool(torch.equal(xg[0], xg[1]) and torch.equal(xg[1], xg[2]))}")
    assert torch.equal(xg[0], xg[1]), "gri modda kanallar ayni olmali"
    assert not torch.equal(xg[0], x[0]) or k.renk == "gri"
    print(f"  [x] Calisiyor")

    print(f"\n[7] Cok isci ile okuma  (h5py tutamaci her iscide ayri mi)")
    print(cizgi)

    # SADECE ILK N BATCH. Tum kumeyi belege toplamak 220.834 ornekte
    # 19.7 GB eder ve surec OOM ile oldurulur -- bir kez oldu.
    N_BATCH = 10

    def ilk_batchler(isci):
        cikan = []
        for i, (xb_, _) in enumerate(yukleyici(k, batch=8, karistir=False,
                                               isci=isci)):
            if i >= N_BATCH:
                break
            cikan.append(xb_)
        return torch.cat(cikan)

    x2, x0 = ilk_batchler(2), ilk_batchler(0)
    print(f"  ilk {N_BATCH} batch ({x0.shape[0]} ornek, "
          f"{x0.numel()/1e6:.1f} MB) karsilastiriliyor")
    print(f"  isci=0 ile isci=2 arasindaki maks fark: "
          f"{(x2.int() - x0.int()).abs().max()}")
    assert torch.equal(x2, x0), ("isci sayisi sonucu degistiriyor -- h5py "
                                 "tutamaci sureclerde paylasiliyor olabilir")
    print(f"  [x] Cok iscili okuma tek iscili okumayla BIREBIR ayni")

    print(f"\n[8] Model ile uc uca")
    print(cizgi)
    from model import DASNet, count_parameters
    net = DASNet(attention="sk", n_classes=3).eval()
    xd, _ = next(iter(yukleyici(k, batch=8, karistir=False, isci=0)))
    with torch.no_grad():
        cikti = net(hazirla(xd, egitim=False))
    print(f"  onbellek {tuple(xd.shape)} -> hazirla -> model -> "
          f"{tuple(cikti.shape)}")
    print(f"  {count_parameters(net):,} parametre")
    assert cikti.shape == (xd.shape[0], 3)
    print(f"  [x] Onbellek -> hazirla -> model zinciri calisiyor")

    print(f"\n{'=' * 70}")
    print("TUM TESTLER GECTI.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("onbellek", help="onbellek_*.h5 yolu")
    sys.exit(self_test(ap.parse_args().onbellek))
