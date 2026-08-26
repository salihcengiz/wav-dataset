"""
GERCEK VERI -- ADIM 5: PYTORCH VERI KUMESI

onbellek_kur.py'nin urettigi uint8 spektrogram onbelleginden model girdisi
uretir. Egitim ve degerlendirme AYNI sinifi kullanir.

    onbellek (129 x 231 uint8)
      -> renklendir (viridis 3 kanal, veya gri)
      -> 224 x 320'ye olcekle
      -> [0,1] + ImageNet normalizasyonu
      -> (istege bagli) zaman/frekans maskeleme        [yalnizca egitimde]
      -> (3, 224, 320) float32 tensor

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
        True ise maskeleme uygulanir. Degerlendirmede MUTLAKA False.
    renk : "viridis" | "gri"
    maske_p / maske_frac :
        Zaman ve frekans maskelemesi (SpecAugment ruhunda). PLAN 7.2'den:
        serit genisligi eksenin en fazla %10'u. Cevirme YOK -- spektrogramda
        zaman veya frekans eksenini ters cevirmek fiziksel olarak anlamsiz.
    """

    def __init__(self, yol, indeksler=None, egitim=False, renk="viridis",
                 girdi=(GIRDI_H, GIRDI_W), maske_p=0.5, maske_frac=0.10,
                 tohum=42):
        if h5py is None:
            raise ImportError("h5py gerekli")
        if renk not in ("viridis", "gri"):
            raise ValueError(f"renk 'viridis' veya 'gri' olmali, {renk!r} degil")

        self.yol = str(yol)
        self.egitim = egitim
        self.renk = renk
        self.girdi = tuple(girdi)
        self.maske_p = maske_p
        self.maske_frac = maske_frac
        self.tohum = tohum
        self._f = None                       # her iscide ayri acilacak

        with h5py.File(self.yol, "r") as f:
            self.n_toplam = int(f["spektrogram"].shape[0])
            self.sekil = tuple(f["spektrogram"].shape[1:])
            self.etiketler_tum = f["etiket"][:]
            self.siniflar = [s.decode() if isinstance(s, bytes) else str(s)
                             for s in f.attrs.get("siniflar", [])]
            self.kaynak_csv = str(f.attrs.get("kaynak_csv", "?"))

        self.indeksler = (np.arange(self.n_toplam) if indeksler is None
                          else np.asarray(indeksler, dtype=np.int64))
        self.etiketler = self.etiketler_tum[self.indeksler]

        self._lut = viridis_lut() if renk == "viridis" else None
        self._mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
        self._std = torch.tensor(NORM_STD).view(3, 1, 1)

    # --- h5py tutamaci: her iscide ayri ---
    @property
    def dosya(self):
        if self._f is None:
            self._f = h5py.File(self.yol, "r")
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
        """
        durum = self.__dict__.copy()
        durum["_f"] = None
        return durum

    def __len__(self):
        return len(self.indeksler)

    def __getitem__(self, i):
        j = int(self.indeksler[i])
        u = self.dosya["spektrogram"][j]            # (129, 231) uint8
        y = int(self.etiketler_tum[j])

        if self._lut is not None:
            rgb = self._lut[u]                      # (129, 231, 3) uint8
            x = torch.from_numpy(np.ascontiguousarray(
                rgb.transpose(2, 0, 1)))            # (3, H, W)
        else:
            x = torch.from_numpy(u.astype(np.uint8))[None].repeat(3, 1, 1)

        # Olcekle -> [0,1] -> ImageNet normalizasyonu
        x = F.interpolate(x[None].float(), size=self.girdi,
                          mode="bilinear", align_corners=False,
                          antialias=True)[0] / 255.0
        x = (x - self._mean) / self._std

        if self.egitim:
            x = self._maskele(x, j)
        return x, y

    def _maskele(self, x, tohum_ek):
        """
        Zaman (dikey serit) ve frekans (yatay serit) maskelemesi.

        Maskelenen bolge 0 yapiliyor -- normalize edilmis uzayda 0, veri
        setinin ortalamasina karsilik gelir. "Bilgi yok" demenin dogru yolu
        budur; siyah (=-mean/std) demek degil.
        """
        g = torch.Generator().manual_seed(self.tohum * 1_000_003 + tohum_ek)
        _, H, W = x.shape
        for eksen, boy in ((1, H), (2, W)):
            if torch.rand(1, generator=g).item() > self.maske_p:
                continue
            for _ in range(int(torch.randint(1, 3, (1,), generator=g))):
                genislik = int(torch.randint(
                    1, max(2, int(boy * self.maske_frac)), (1,), generator=g))
                bas = int(torch.randint(0, max(1, boy - genislik), (1,),
                                        generator=g))
                if eksen == 1:
                    x[:, bas:bas + genislik, :] = 0.0
                else:
                    x[:, :, bas:bas + genislik] = 0.0
        return x

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
    print(f"  ornek 0 -> girdi {tuple(x.shape)} {x.dtype}, etiket {y}")
    assert x.shape == (3, GIRDI_H, GIRDI_W), f"girdi sekli {tuple(x.shape)}"
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    print(f"  deger araligi [{x.min():.2f}, {x.max():.2f}]  ortalama {x.mean():+.3f}")
    print(f"  [x] Sekil ve tip dogru")

    print(f"\n[3] Belirlenimcilik -- ayni indeks ayni tensoru vermeli")
    print(cizgi)
    a, _ = k[3]
    b, _ = k[3]
    print(f"  degerlendirme modunda maks fark: {(a-b).abs().max():.2e}")
    assert torch.equal(a, b), "degerlendirme modu belirlenimci degil"
    ke = OnbellekKumesi(onbellek, egitim=True)
    a, _ = ke[3]
    b, _ = ke[3]
    print(f"  egitim modunda maks fark      : {(a-b).abs().max():.2e}  "
          f"(maskeleme tohuma bagli -> 0 olmali)")
    assert torch.equal(a, b), "egitim modu ayni indekste farkli sonuc verdi"
    print(f"  [x] Tekrar uretilebilir")

    print(f"\n[4] Maskeleme gercekten uyguluyor mu")
    print(cizgi)
    farkli = sum(1 for i in range(min(20, len(k)))
                 if not torch.equal(k[i][0], ke[i][0]))
    print(f"  20 ornegin {farkli}'i egitim modunda farkli (maskelenmis)")
    assert farkli > 0, "maskeleme hic uygulanmiyor"
    print(f"  [x] Maskeleme aktif")

    print(f"\n[5] Gri secenegi")
    print(cizgi)
    kg = OnbellekKumesi(onbellek, renk="gri")
    xg, _ = kg[0]
    print(f"  gri girdi {tuple(xg.shape)}  3 kanal ayni mi: "
          f"{torch.allclose(xg[0]*0.229+0.485, xg[1]*0.224+0.456, atol=1e-3)}")
    assert xg.shape == x.shape
    print(f"  [x] Calisiyor")

    print(f"\n[6] DataLoader (isci=0)")
    print(cizgi)
    dl = yukleyici(k, batch=8, isci=0)
    xb, yb = next(iter(dl))
    print(f"  batch girdi {tuple(xb.shape)}  etiket {tuple(yb.shape)} {yb.tolist()}")
    assert xb.shape == (min(8, len(k)), 3, GIRDI_H, GIRDI_W)
    print(f"  [x] Batch uretiliyor")

    print(f"\n[7] Cok isci ile okuma  (h5py tutamaci her iscide ayri mi)")
    print(cizgi)
    dl2 = yukleyici(k, batch=8, karistir=False, isci=2)
    x2 = torch.cat([b[0] for b in dl2])
    x0 = torch.cat([b[0] for b in yukleyici(k, batch=8, karistir=False, isci=0)])
    print(f"  isci=0 ile isci=2 ciktilari arasindaki maks fark: "
          f"{(x2 - x0).abs().max():.2e}")
    assert torch.equal(x2, x0), ("isci sayisi sonucu degistiriyor -- h5py "
                                 "tutamaci sureclerde paylasiliyor olabilir")
    print(f"  [x] Cok iscili okuma tek iscili okumayla BIREBIR ayni")

    print(f"\n[8] Model ile uc uca")
    print(cizgi)
    from model import DASNet, count_parameters
    m = DASNet(attention="sk", n_classes=3).eval()
    with torch.no_grad():
        cikti = m(xb)
    print(f"  {tuple(xb.shape)} -> {tuple(cikti.shape)}  "
          f"({count_parameters(m):,} parametre)")
    assert cikti.shape == (xb.shape[0], 3)
    print(f"  [x] Model onbellekten gelen girdiyi kabul ediyor")

    print(f"\n{'=' * 70}")
    print("TUM TESTLER GECTI.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("onbellek", help="onbellek_*.h5 yolu")
    sys.exit(self_test(ap.parse_args().onbellek))
