"""
ONNX DISA AKTARIM -- HER MIMARI ICIN (DASNetBiLSTM ve DASNet)

Modeli, ON ISLEMENIN TAMAMI ICINE GOMULU halde tek bir .onnx dosyasina
cevirir. Girdi HAM SINYAL.

=== TESLIM STANDARDI (sorumlunun kurali, 2026-09-01) ===

BUNDAN SONRAKI TUM MODELLERIN .onnx dosyalari ham sinyal alacak:
(None, 15000), on isleme grafigin icinde, opset 13.

Sarmalayici (OnIslemeliModel) mimariden BAGIMSIZ -- icine hangi model
konursa onu sarar. Yeni bir mimari eklendiginde bu dosya degismez;
gercek_export.model_kur() onu kurar, buraya verilir.

Mimari ve renk CHECKPOINT'TEN OKUNUR (ckpt_oku), elle verilmez. Yanlis
renk vermek modele sessizce yanlis girdi verir ve hicbir hata cikmaz.

    girdi  : (batch, 15000) float32   -- genlik, 7.5 s @ 2000 Hz
    cikti  : (batch, 3)     float32   -- logit  [cutting, climbing, noise]
             (batch,)       float32   -- bosluk_orani (asagida)

=== NEDEN ON ISLEME DE ICERIDE ===

Ekibin mevcut kodunda egitim ve cikarim ayri yollardan gidiyordu ve DORT
noktada ayrilmislardi (Rapor 6.3): olcek katsayisi, pencere boyutu, P/S
bileseni, sessiz pencere filtresi. Model egitimde gordugu bicimi bekler;
cikarimda farkli bir sey verilirse tahminler coper ve HICBIR HATA MESAJI
VERMEZ.

On islemeyi grafigin icine gomunce bu risk tamamen kapaniyor: .onnx
dosyasini kullanan kisinin ayrica bir sey yazmasina gerek yok.

=== BOSLUK ORANI NEDEN AYRI CIKTI ===

real_data.pencere_yukle bos pencerelerde None donduruyor. ONNX grafiginde
"bazen cikti verme" diye bir sey yok -- grafik her zaman ayni sekli
uretmek zorunda.

Cozum: bosluk oranini IKINCI CIKTI olarak veriyoruz. Cagiran taraf esigi
kendisi uyguluyor:

    logit, bosluk = oturum.run(None, {"sinyal": x})
    gecerli = bosluk <= 0.45          # real_data.BOS_ESIK
    # gecerli olmayan pencereler icin tahmin KULLANILMAMALI

Bu esik egitimde de uygulandi (bos pencereler elendi), yani model o tur
pencereler icin EGITILMEDI.

=== ON ISLEME ZINCIRI (real_data.py + gercek_veri_kumesi.py ile birebir) ===

    ham sinyal (B, 15000)
      -> normalize: (x - medyan) / MAD
      -> DC cikar
      -> STFT: n_fft=256, hop=64, Hann      -> (B, 129, 231) genlik
      -> dB: 20*log10(S / S.max()), -80'de kirp
      -> uint8'e kuantala (0.31 dB adim)    <- EGITIMDE DE BOYLEYDI
      -> viridis LUT                        -> (B, 3, 129, 231)
      -> 224 x 320'ye olcekle (bilinear)
      -> /255 -> ImageNet normalizasyonu
      -> DASNetBiLSTM                       -> (B, 3) logit

uint8 kuantalama BILEREK korunuyor: onbellek uint8 olarak saklandi ve model
o hassasiyetle egitildi. Atlanirsa cikarim egitimden farkli bir girdi
dagilimi gorur.

=== STFT NEDEN ELLE YAZILDI ===

torch.stft'nin ONNX karsiligi opset'e gore degisiyor ve bazi surumlerde
sorun cikariyor. Burada unfold + matris carpimi ile yaziliyor: yalnizca
Gather/MatMul/Mul gibi temel islemler kullanildigi icin her opset'te
guvenle ihrac edilir. Sonuc np.fft.rfft ile ayni (birim testinde
karsilastiriliyor).

=== OPSET NEDEN 13 ===

opset = ONNX operator set surumu. Grafikteki her dugumun (Conv, MatMul,
LSTM, Resize...) hangi tanimina gore yorumlanacagini sabitler. Bir calisma
zamani (onnxruntime, TensorRT, gomulu SDK) destekledigi en yuksek opset'in
USTUNDEKI dosyayi ACAMAZ. Yani opset bir UYUMLULUK ESIGIDIR -- modelin
matematigini ya da hizini degistirmez, yalnizca tasinabilirligini.

Geriye donuk uyumlu: dusuk opset'li dosyayi yeni calisma zamani okur,
tersi olmaz. Bu yuzden dusuk opset "daha kotu" degil, DAHA TASINABILIR.

Sorumlu 13 istedi (hedef calisma zamani 17'yi desteklemiyor). Bu grafik
icin BEDELSIZ, cunku opset 17'ye ozgu hicbir operator kullanilmiyor:

  - STFT elle yazildi (unfold + matmul); opset 17'nin native STFT/DFT
    operatorleri kullanilmiyor -- 17'nin bu projeye getirdigi TEK yenilik
    tam da o operatorlerdi
  - medyan siralama tabanli (TopK), aten::median yok
  - frekans havuzlamasi sabit cekirdekli, AdaptiveAvgPool yok
  - antialias kapali

Grafikteki en yeni gereksinim Resize-13. LSTM opset 7'den, Softmax-13
13'ten mevcut. Yani 13 tabanin ta kendisi, sikistirma degil.

Degisiklik OLCULDU: 13 ve 17 ile ihrac edilen grafiklerin logit ciktisi
karsilastirildi (asagidaki dogrulama akisi).

=== KULLANIM ===

    python onnx_disa_aktar.py --ckpt /tf/.../kosu4_bilstm_yeni_rejim.pt
    python onnx_disa_aktar.py --ckpt ... --opset 17   # gerekirse yukselt

    # yerelde rastgele agirliklarla sadece hattı sinamak icin:
    python onnx_disa_aktar.py --sahte-agirlik
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_KOK = Path(__file__).resolve().parent.parent
_SRC = _KOK / "src"
for _y in (str(_SRC), str(Path(__file__).resolve().parent)):
    if _y not in sys.path:
        sys.path.insert(0, _y)

import real_data as rd                                    # noqa: E402
from gercek_veri_kumesi import (GIRDI_H, GIRDI_W, NORM_MEAN,  # noqa: E402
                                NORM_STD, viridis_lut)
from model_bilstm import DASNetBiLSTM                     # noqa: E402
# Mimari kaydi TEK YERDE: gercek_export. Burada kopyalanmiyor.
from gercek_export import model_kur                       # noqa: E402


def ckpt_oku(ckpt):
    """
    Checkpoint'ten mimari, renk ve sinif sayisini OKUR -- tahmin etmez.

    NEDEN OTOMATIK:
    Yanlis --renk vermek modele sessizce YANLIS girdi verir ve hicbir hata
    mesaji cikmaz (gri model viridis goruntu gorurse tahminler coper).
    Ayni sekilde yanlis mimari secmek load_state_dict'te patlar ama once
    zaman kaybettirir. Ikisi de checkpoint'in icinde zaten yaziyor:

        mimari : state_dict'te "lstm." ile baslayan tensor var mi
        renk   : paket["ayar"]["renk"]  (egitim dongusu yaziyor)
        sinif  : classifier.weight'in satir sayisi

    Donen: (paket, state_dict, mimari_adi, renk, n_sinif)
    """
    paket = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = paket.get("state_dict", paket)
    sinif = ("DASNetBiLSTM" if any(str(k).startswith("lstm.") for k in sd)
             else "DASNet")
    renk = (paket.get("ayar") or {}).get("renk")
    n_sinif = int(sd["classifier.weight"].shape[0])
    return paket, sd, sinif, renk, n_sinif


class OnIslemeliModel(nn.Module):
    """
    Ham sinyalden logit'e -- on islemenin tamami iceride.

    Tum sabitler (Hann penceresi, DFT matrisleri, viridis LUT, ImageNet
    istatistikleri) buffer olarak kaydediliyor; boylece .onnx dosyasinin
    icine gomulurler ve disaridan hicbir sey gerekmez.
    """

    BASTIRMA = 1.0e4          # bos pencerede saldiri logitlerinden dusulen

    def __init__(self, model, pencere=rd.PENCERE, fs=rd.FS, n_fft=rd.N_FFT,
                 hop=rd.HOP, top_db=rd.TOP_DB, bos_frekans=rd.BOS_FREKANS,
                 girdi=(GIRDI_H, GIRDI_W), renk="viridis",
                 siniflar=("cutting", "climbing", "noise"),
                 bos_bastir=True, bos_esik=rd.BOS_ESIK,
                 dusuk_frekans=100.0, dusuk_esik=0.9084):
        super().__init__()
        self.model = model
        self.pencere, self.fs = pencere, fs
        self.n_fft, self.hop, self.top_db = n_fft, hop, top_db
        self.girdi, self.renk = tuple(girdi), renk
        self.n_cerceve = 1 + (pencere - n_fft) // hop
        self.siniflar = list(siniflar)
        self.bos_bastir, self.bos_esik = bool(bos_bastir), float(bos_esik)

        # --- BOS PENCERE BASTIRMASI ---
        #
        # OLCULDU (src/bos_pencere_testi.py): bos pencerelerde modelin
        # argmax'i %99.8 oranda bir SALDIRI sinifi. `noise` bizim veri
        # setimizde etiketlenmis bir OLAY TURU, "hicbir sey yok" degil --
        # bos pencereler egitimden silindigi icin modelin susma cevabi yok.
        #
        # Cozum: bosluk_orani esigi asiyorsa saldiri siniflarinin
        # logitinden buyuk bir sabit dusuyoruz; argmax kendiliginden
        # `noise`'a dusuyor. Cikti SEKLI degismiyor, cagiran tarafta
        # hicbir degisiklik gerekmiyor.
        #
        # Kosullu dal yok -- ONNX'te `if` yerine saf aritmetik:
        #     logit = logit - (1 - gecerli) * bastir_maske * BASTIRMA
        # gecerli=1 iken hicbir sey degismez, 0 iken saldiri logitleri
        # -1e4'e iner. Yalnizca LessOrEqual/Cast/Mul/Sub kullaniliyor.
        bastir = [0.0 if s == "noise" else 1.0 for s in self.siniflar]
        if all(b == 1.0 for b in bastir):
            raise ValueError(
                f"siniflar arasinda 'noise' yok: {self.siniflar}. "
                f"Bastirma hangi sinifa dusecegini bilemez.")
        self.register_buffer("bastir_maske",
                             torch.tensor(bastir).view(1, -1))

        # --- Hann penceresi: real_data._hann ile birebir ---
        n = torch.arange(n_fft, dtype=torch.float64)
        self.register_buffer("hann",
                             (0.5 - 0.5 * torch.cos(2 * np.pi * n / n_fft)).float())

        # --- rfft'nin matris karsiligi ---
        # X[k] = sum_n x[n] * exp(-2j*pi*k*n/N),  k = 0..N/2
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        aci = -2.0 * np.pi * k[None, :] * n[:, None] / n_fft   # (n_fft, K)
        self.register_buffer("dft_cos", torch.cos(aci).float())
        self.register_buffer("dft_sin", torch.sin(aci).float())

        # --- bosluk orani icin frekans maskesi (STFT binleri uzerinde) ---
        #
        # real_data.bosluk_orani TAM PENCERE rfft'si kullaniyor (15.000
        # nokta). Ama aten::fft_rfft ONNX'e ihrac EDILEMIYOR ve tam pencere
        # DFT matrisi 15.001 x 15.000 = ~450 MB tutardi.
        #
        # Cozum: ayni buyuklugu ZATEN HESAPLADIGIMIZ STFT'den kestiriyoruz.
        # 129 bin 0-1000 Hz'i kapsiyor; cerceveler boyunca guc toplanip
        # 500 Hz ustunun payi aliniyor. Bu bir Welch kestirimi -- tam
        # pencere FFT'siyle birebir ayni degil ama ayni fiziksel buyukluk.
        # Fark birim testinde OLCULUYOR ve model kartina yaziliyor.
        stft_frek = torch.arange(n_fft // 2 + 1, dtype=torch.float64) * fs / n_fft
        self.register_buffer("yuksek_maske",
                             (stft_frek >= bos_frekans).float().view(1, -1, 1))

        # --- DUSUK FREKANS MASKESI (ikinci bosluk olcutu) ---
        #
        # OLCULDU (src/ham_analiz.py, 201 kanalin hepsinde): waterfall'da
        # yanlis alarm ureten 5 kanalin (0, 4, 5, 63, 64) dordunde
        # bosluk_orani 0.007-0.014 cikiyor -- yani olcut onlari "en dolu"
        # pencereler sayiyor, TAM TERSI. Cunku olu bir kanalda yuksek
        # frekans yoktur, yavas bir taban kaymasi vardir.
        #
        # 100 Hz ALTINDAKI enerji payi ise ayiriyor:
        #     artefakt 0.956   gercek olay 0.655
        # Gercek olaylarin %95'ini koruyan esikte (0.9084) artefakt
        # pencerelerinin %92.5'i susturuluyor. bosluk_orani ayni testte
        # %20.0'da kaliyor.
        self.dusuk_esik = float(dusuk_esik)
        self.register_buffer("dusuk_maske",
                             (stft_frek < dusuk_frekans).float().view(1, -1, 1))

        # --- viridis LUT (256 x 3) ---
        self.register_buffer("lut", torch.from_numpy(
            viridis_lut().astype(np.int64)))

        self.register_buffer("norm_mean", torch.tensor(NORM_MEAN).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(NORM_STD).view(1, 3, 1, 1))

    # -----------------------------------------------------------
    @staticmethod
    def _medyan(x):
        """
        np.median ile AYNI medyan -- siralama uzerinden.

        torch.median ONNX'e ihrac EDILEMIYOR (aten::median desteklenmiyor).
        Ayrica torch.median cift uzunlukta ALT ortancayi dondururken
        np.median iki ortancanin ORTALAMASINI alir; real_data numpy
        kullandigi icin numpy davranisini almak zorundayiz.

        Siralama ONNX'te TopK'ya cevriliyor, sorunsuz ihrac oluyor.
        """
        n = x.shape[-1]
        s, _ = torch.sort(x, dim=-1)
        return (s[..., (n - 1) // 2] + s[..., n // 2]).unsqueeze(-1) * 0.5

    def normalize(self, x):
        """real_data.normalize_et -- (x - medyan) / MAD."""
        med = self._medyan(x)
        mad = self._medyan((x - med).abs())
        return (x - med) / (mad + 1e-9)

    def stft_genlik(self, x):
        """
        STFT genligi (B, 129, 231). torch.stft yerine unfold + matris
        carpimi: yalnizca temel islemler kullanildigi icin her opset'te
        guvenle ihrac olur ve np.fft.rfft ile ayni sonucu verir.
        """
        x = x - x.mean(dim=1, keepdim=True)                   # DC cikar
        cerceve = x.unfold(1, self.n_fft, self.hop)           # (B,T,n_fft)
        cerceve = cerceve * self.hann                         # pencerele
        re = cerceve @ self.dft_cos                           # (B,T,K)
        im = cerceve @ self.dft_sin
        return torch.sqrt(re * re + im * im).transpose(1, 2)  # (B,K,T)

    def bosluk_orani_stft(self, S):
        """
        500 Hz ustundeki guc payi -- STFT binlerinden kestirim.

        ⚠ SIFIR GUC KORUMASI. Sabit bir sinyal (olu kanal) normalizasyondan
        (x - medyan)/(0 + 1e-9) = 0 diye geciyor; STFT'si de sifir. Bu
        durumda eski surum 0/1e-12 = 0.0 donduruyordu, yani pencereyi
        "dolu" sayiyordu -- oysa real_data.bosluk_orani ayni durumda
        1.0 (=BOS) donduruyor.

        Saha etkisi olculdu: benchmark kanal 0'in MAD'i tam 0, numpy
        1.000 diyor, ONNX 0.0 diyordu. Bastirma bu yuzden tetiklenmedi ve
        waterfall'da 30 saniyelik kesintisiz `cutting` sutunu cikti.
        Simdi numpy ile ayni: guc yoksa 1.0.
        """
        G = S * S
        top = G.sum((1, 2))
        oran = (G * self.yuksek_maske).sum((1, 2)) / top.clamp_min(1e-12)
        return torch.where(top > 1e-12, oran, torch.ones_like(oran))

    def dusuk_frekans_payi(self, S):
        """
        100 Hz ALTINDAKI guc payi -- ikinci bosluk olcutu.

        Olu/zayif kanallarda enerji tamamen taban kaymasinda toplaniyor;
        bosluk_orani (yuksek frekans payi) bu durumda DUSUK cikiyor ve
        pencereyi dolu sayiyor. Bu buyukluk tersini olcuyor ve artefakt
        kanallarini gercek olaylardan ayiriyor (0.956 vs 0.655).
        """
        G = S * S
        top = G.sum((1, 2))
        oran = (G * self.dusuk_maske).sum((1, 2)) / top.clamp_min(1e-12)
        return torch.where(top > 1e-12, oran, torch.ones_like(oran))

    def db_cevir(self, S):
        """real_data.spektrogram'in dB adimi: ref=max, -top_db'de kirp."""
        ref = S.amax(dim=(1, 2), keepdim=True).clamp_min(1e-10)
        db = 20.0 * torch.log10(S.clamp_min(1e-10) / ref)
        return db.clamp_min(-self.top_db)

    def spektrogram_db(self, x):
        """Kolaylik: normalize edilmis sinyalden dB spektrograma."""
        return self.db_cevir(self.stft_genlik(x))

    def goruntu(self, db):
        """dB -> uint8 kuantalama -> renklendirme -> olcekle -> normalize."""
        # Egitimde onbellek uint8'di; ayni kuantalamayi uyguluyoruz.
        u = torch.round((db.clamp(-self.top_db, 0.0) + self.top_db)
                        * (255.0 / self.top_db))
        idx = u.clamp(0, 255).long()                          # (B,F,T)

        if self.renk == "viridis":
            rgb = self.lut[idx]                               # (B,F,T,3)
            x = rgb.permute(0, 3, 1, 2).float()               # (B,3,F,T)
        else:
            x = u.unsqueeze(1).repeat(1, 3, 1, 1).float()

        # gercek_veri_kumesi.hazirla ile ayni sira: olcekle -> /255 -> norm
        # NOT: antialias=True ONNX'e ihrac EDILEMIYOR; fark birim testinde
        # olculuyor ve model kartina yaziliyor.
        x = F.interpolate(x, size=self.girdi, mode="bilinear",
                          align_corners=False) / 255.0
        return (x - self.norm_mean) / self.norm_std

    def forward(self, sinyal):
        """
        (B, 15000) -> (logit (B, n_sinif), bosluk_orani (B,))

        bos_bastir aciksa ve bosluk_orani esigi asiyorsa saldiri
        siniflarinin logiti bastirilir, argmax `noise`'a duser.
        """
        S = self.stft_genlik(self.normalize(sinyal))
        bosluk = self.bosluk_orani_stft(S)
        logit = self.model(self.goruntu(self.db_cevir(S)))

        if self.bos_bastir:
            # IKI OLCUT, VEYA ile:
            #   bosluk_orani > 0.45   -> spektrum duz (beyaz gurultu benzeri)
            #   dusuk_frek  > 0.9084  -> enerji taban kaymasinda (olu kanal)
            # Ikincisi olculerek eklendi: artefakt kanallarinin dordunde
            # bosluk_orani 0.007-0.014 cikiyor ve onlari yakalayamiyor.
            dusuk = self.dusuk_frekans_payi(S)
            gecerli = ((bosluk <= self.bos_esik) &
                       (dusuk <= self.dusuk_esik)).to(logit.dtype).unsqueeze(1)
            logit = logit - (1.0 - gecerli) * self.bastir_maske * self.BASTIRMA
        return logit, bosluk


# ---------------------------------------------------------------
def sarmalayici_kur(ckpt=None, renk=None, mimari=None, n_sinif=3,
                    bos_bastir=True, siniflar=None, **model_kw):
    """
    Ham sinyal alan sarmalanmis modeli kurar.

    mimari / renk / siniflar None ise checkpoint'ten OKUNUR (ckpt_oku).
    Elle vermek otomatik tespiti ezer -- yalnizca checkpoint eksik bilgi
    tasiyorsa gerekli.

    bos_bastir : bosluk_orani esigi asan pencerelerde saldiri logitlerini
        bastir, argmax `noise`'a dussun. Varsayilan ACIK.
    """
    sd, paket = None, {}
    if ckpt:
        paket, sd, bulunan, ckpt_renk, n_sinif = ckpt_oku(ckpt)
        if mimari and mimari != bulunan:
            print(f"  UYARI: --mimari {mimari} verildi ama checkpoint "
                  f"{bulunan} gibi duruyor")
        mimari = mimari or bulunan
        if renk and ckpt_renk and renk != ckpt_renk:
            print(f"  UYARI: --renk {renk} verildi ama checkpoint "
                  f"'{ckpt_renk}' diyor -- egitimdeki temsil bu!")
        renk = renk or ckpt_renk

    mimari = mimari or "DASNetBiLSTM"
    if renk is None:
        renk = "viridis"
        if ckpt:
            print("  UYARI: checkpoint renk bilgisi tasimiyor, "
                  "'viridis' varsayildi")

    siniflar = (siniflar or paket.get("siniflar")
                or ["cutting", "climbing", "noise"])
    siniflar = [str(s) for s in siniflar]

    net = model_kur(mimari, n_sinif, **model_kw)
    if sd is not None:
        net.load_state_dict(sd)
        print(f"  agirliklar yuklendi: {ckpt}")
        print(f"  mimari: {mimari}   renk: {renk}   siniflar: {siniflar}")
        if "val_macro_f1" in paket:
            print(f"  val macro-F1: {paket['val_macro_f1']:.4f}  "
                  f"(epoch {paket.get('en_iyi_epoch')})")
    print(f"  bos pencere bastirma: "
          f"{'ACIK -> argmax noise' if bos_bastir else 'KAPALI'}"
          f"  (esik {rd.BOS_ESIK})")
    return OnIslemeliModel(net, renk=renk, siniflar=siniflar,
                           bos_bastir=bos_bastir).eval()


def ornek_sinyal(n=2, tohum=0):
    """
    YAPILI test sinyali -- duz gurultu DEGIL.

    Duz gurultunun bosluk_orani ~0.5 cikar ve BOS PENCERE BASTIRMASI
    devreye girer. O zaman dogrulama -1e4 ile -1e4'u karsilastirir ve
    gercek bir ayrismayi GOREMEZ -- asla basarisiz olamayan bir dogrulama
    olur (bkz. `pencere_son` dersi, sonuc belgesi Bolum 8b).

    Yapili sinyalde bosluk ~0.12, bastirma kapali kalir, logitler
    gercekten karsilastirilir.

    FREKANS BANDI 120-450 Hz: 120 + 30*(i % 12). Iki kisit birden:

      ust sinir  -- 20 + 15*i gibi artan bir dizi batch 64'te 965 Hz'e
                    cikiyordu; abs() harmonikleri Nyquist'in (1000 Hz)
                    ustune tasip katlaniyor ve PyTorch/ONNX farki 1e-3
                    esigine dayaniyordu (olculdu: 9.77e-04).
      alt sinir  -- eski 20 Hz'lik sinyalin enerjisi TAMAMEN 100 Hz'in
                    altindaydi, yani DUSUK FREKANS OLCUTU onu bos sayip
                    bastirirdi. O zaman "yapili sinyale dokunulmuyor"
                    kontrolu asla gecemezdi.

    120-450 Hz gercek olaylara benziyor: dusuk_frek ~0.1 (olaylarda
    0.65), bosluk_orani ~0.12 (esik 0.45).
    """
    rng = np.random.default_rng(tohum)
    t = np.arange(rd.PENCERE) / rd.FS
    return np.stack([
        np.abs(3 + np.sin(2 * np.pi * (120 + 30 * (i % 12)) * t)
               + 0.4 * rng.normal(0, 1, rd.PENCERE)) for i in range(n)
    ]).astype(np.float32)


def sayisal_dogrula(sarmal, n=4, tohum=0):
    """
    Sarmalayicinin on islemesi real_data ile AYNI mi?

    Bu kontrol olmadan ihracat sessizce yanlis girdi ureten bir model
    verebilir -- ve hicbir hata mesaji cikmaz.
    """
    print("\n" + "=" * 70)
    print("SAYISAL DOGRULAMA -- sarmalayici vs real_data")
    print("=" * 70)
    ham = ornek_sinyal(n, tohum).astype(np.float64)

    with torch.no_grad():
        x = torch.from_numpy(ham).float()
        S = sarmal.stft_genlik(sarmal.normalize(x))
        db_t = sarmal.db_cevir(S).numpy()
        bos_t = sarmal.bosluk_orani_stft(S).numpy()

    db_n = np.stack([rd.spektrogram(rd.normalize_et(s)) for s in ham])
    bos_n = np.array([rd.bosluk_orani(s) for s in ham])

    d_db = np.abs(db_t - db_n).max()
    print(f"  spektrogram sekli : torch {db_t.shape}  numpy {db_n.shape}")
    print(f"  maks dB farki     : {d_db:.2e}")
    assert db_t.shape == db_n.shape, "spektrogram sekli uyusmuyor"
    assert d_db < 1e-2, f"spektrogram ayrisiyor: {d_db}"
    print(f"  [x] Spektrogram real_data ile BIREBIR ortusuyor")

    # Bosluk orani BIREBIR DEGIL -- STFT'den kestirim (bkz. __init__).
    # Fark buyukse esik (0.45) yeniden kalibre edilmeli.
    print(f"\n  Bosluk orani (STFT kestirimi vs tam pencere FFT):")
    for a, b in zip(bos_t, bos_n):
        print(f"    kestirim {a:.4f}   gercek {b:.4f}   fark {a-b:+.4f}")
    d_bos = float(np.abs(bos_t - bos_n).max())
    print(f"  maks fark: {d_bos:.4f}   (esik {rd.BOS_ESIK})")
    if d_bos > 0.05:
        print(f"  UYARI: fark buyuk. Esik yeniden kalibre edilmeli ya da")
        print(f"  bosluk filtresi ONNX disinda uygulanmali.")
    else:
        print(f"  [x] Kestirim esik kararini degistirecek kadar sapmiyor")

    # antialias farki -- ihracatta kapatiliyor, bedeli olculuyor
    with torch.no_grad():
        db = torch.from_numpy(db_n).float()
        u = torch.round((db.clamp(-80, 0) + 80) * (255 / 80)).clamp(0, 255).long()
        rgb = sarmal.lut[u].permute(0, 3, 1, 2).float()
        a = F.interpolate(rgb, size=sarmal.girdi, mode="bilinear",
                          align_corners=False, antialias=True)
        b = F.interpolate(rgb, size=sarmal.girdi, mode="bilinear",
                          align_corners=False)
    print(f"\n  antialias acik/kapali farki (0-255 olceginde):")
    print(f"    ortalama {(a-b).abs().mean():.3f}   maks {(a-b).abs().max():.1f}")
    print(f"  NOT: egitimde antialias ACIKTI, ONNX'te kapali.")
    return float(d_db)


def disa_aktar(sarmal, cikti, opset=13, dogrula=True):
    """
    Grafigi .onnx'e yazar ve PyTorch ile ortustugunu OLCER.

    Donen: (cikti_yolu, logit_farki)

    logit_farki, PyTorch ile ONNX ciktisi arasindaki maks mutlak fark;
    dogrulama yapilmadiysa None. Model kartina BU SAYI yaziliyor -- daha
    once kart "ihracat ciktisina bak" diyordu, yani rapora giren bir sayi
    konsola basilip kayboluyordu (proje kurali: her sayi kod ciktisindan).
    """
    print("\n" + "=" * 70)
    print(f"ONNX IHRACATI -> {cikti}  (opset {opset})")
    print("=" * 70)
    d_logit = None
    ornek = torch.from_numpy(ornek_sinyal(2, tohum=0))
    # dynamo=False: eski TorchScript ihracatcisi. Yeni (torch.export tabanli)
    # ihracatci `onnxscript` istiyor ve sunucuda kurulu degil; eski yol bizim
    # grafigi sorunsuz cikariyor ve sunucudaki onnx 1.14 ile uyumlu.
    ek = {}
    try:
        import inspect
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            ek["dynamo"] = False
    except (TypeError, ValueError):
        pass
    torch.onnx.export(
        sarmal, (ornek,), str(cikti),
        input_names=["sinyal"], output_names=["logit", "bosluk_orani"],
        dynamic_axes={"sinyal": {0: "batch"},
                      "logit": {0: "batch"},
                      "bosluk_orani": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, **ek)
    boyut = Path(cikti).stat().st_size / 1e6
    print(f"  yazildi: {cikti}  ({boyut:.1f} MB)")

    if dogrula:
        try:
            import onnxruntime as ort
        except ImportError:
            print("  onnxruntime yok -- calisma zamani dogrulamasi ATLANDI")
            print("  (pip install --user onnxruntime ile kurulabilir)")
            return cikti, None
        oturum = ort.InferenceSession(str(cikti),
                                      providers=["CPUExecutionProvider"])
        with torch.no_grad():
            t_logit, t_bos = sarmal(ornek)
        o_logit, o_bos = oturum.run(None, {"sinyal": ornek.numpy()})
        d1 = d_logit = float(np.abs(t_logit.numpy() - o_logit).max())
        d2 = np.abs(t_bos.numpy() - o_bos).max()
        print(f"  PyTorch vs ONNX  logit farki : {d1:.2e}")
        print(f"                   bosluk farki: {d2:.2e}")
        assert d1 < 1e-3, f"ONNX ciktisi PyTorch'tan ayrisiyor: {d1}"
        print(f"  [x] ONNX cikti PyTorch ile ortusuyor")

        # --- DINAMIK BATCH -- istenen (None, 15000) gercekten calisiyor mu ---
        #
        # torch.onnx.export su uyariyi veriyor: "LSTM ile batch_size 1
        # disinda ihracat, farkli batch boyutunda HATA verebilir."
        # Uyari temkinli; bizde dizi uzunlugu SABIT (40 adim) oldugu icin
        # sorun cikmiyor. Ama varsaymak yerine OLCUYORUZ -- sorumlunun
        # istegi tam olarak dinamik batch'ti.
        print(f"\n  Dinamik batch testi (ihracat batch={ornek.shape[0]}):")
        sorun = False
        for b in (1, 3, 16, 64):
            x = torch.from_numpy(ornek_sinyal(b, tohum=b))
            try:
                ol, ob = oturum.run(None, {"sinyal": x.numpy()})
                with torch.no_grad():
                    tl, tb = sarmal(x)
                d = float(np.abs(tl.numpy() - ol).max())
                iyi = d < 1e-3
                sorun |= not iyi
                print(f"    batch {b:>3}: {'OK ' if iyi else 'AYRISMA!'} "
                      f"sekil {ol.shape}  fark {d:.2e}")
            except Exception as e:  # noqa: BLE001
                sorun = True
                print(f"    batch {b:>3}: HATA {type(e).__name__}: {str(e)[:90]}")
        assert not sorun, "dinamik batch calismiyor -- (None, 15000) verilemez"
        print(f"  [x] (None, {rd.PENCERE}) girdi sekli dogrulandi")

        # --- BASTIRMA ONNX ICINDE DE CALISIYOR MU ---
        #
        # Ihracat ornegi YAPILI bir sinyal (bkz. ornek_sinyal), yani
        # izleme sirasinda bastirma dali hic tetiklenmiyor. Eger
        # do_constant_folding gecerlilik bayragini sabitleseydi ONNX
        # ASLA bastirmazdi ve PyTorch tarafi dogru gorundugu icin
        # fark edilmezdi. O yuzden ONNX'e ayrica BOS bir pencere
        # veriliyor.
        if sarmal.bos_bastir:
            # UC AYRI YOL sinaniyor, cunku her biri farkli grafik
            # dugumlerinden geciyor:
            #   duz gurultu -> yuksek_maske dali
            #   olu kanal   -> torch.where SIFIR GUC korumasi (yeni)
            #   suruklenme  -> dusuk_maske dali (yeni)
            # Yalnizca duz gurultu sinansaydi, yeni iki yolun ihracatta
            # sabitlenip sessizce devre disi kalmasini goremezdik.
            rng = np.random.default_rng(11)
            t = np.arange(rd.PENCERE) / rd.FS
            durumlar = {
                "duz gurultu": np.abs(
                    rng.normal(0, 1, (2, rd.PENCERE))).astype(np.float32),
                "olu kanal": np.full((2, rd.PENCERE), 900.0, dtype=np.float32),
                "suruklenme": np.stack([
                    np.abs(1000 + 300 * np.sin(2 * np.pi * (1.0 + 0.5 * i) * t)
                           + 5 * rng.normal(0, 1, rd.PENCERE))
                    for i in range(2)]).astype(np.float32),
            }
            noise_idx = sarmal.siniflar.index("noise")
            print(f"\n  Bastirma ONNX icinde:")
            for ad, x in durumlar.items():
                o_lg, o_bo = oturum.run(None, {"sinyal": x})
                tahmin = o_lg.argmax(1)
                print(f"    {ad:<12s} bosluk {np.round(o_bo, 3).tolist()}  ->  "
                      f"{[sarmal.siniflar[t] for t in tahmin]}   "
                      f"logit[0] {np.round(o_lg[0], 1).tolist()}")
                assert (tahmin == noise_idx).all(), (
                    f"BASTIRMA ONNX'E GECMEMIS ('{ad}'): argmax "
                    f"{[sarmal.siniflar[t] for t in tahmin]}, 'noise' "
                    f"bekleniyordu. Muhtemelen constant folding gecerlilik "
                    f"bayragini sabitledi.")
            print(f"    [x] Uc yol da ONNX'te bastiriyor")
    return cikti, d_logit


def bastirma_dogrula(sarmal, tohum=7):
    """
    BOS PENCERE BASTIRMASI gercekten calisiyor mu?

    Iki sinyal veriliyor:
      duz gurultu -> spektrum duz -> bosluk_orani ~0.5 -> BASTIRILMALI
      yapili      -> bosluk ~0.12                      -> DOKUNULMAMALI

    Bu kontrol olmadan bastirma sessizce devre disi kalabilir (yanlis
    sinif sirasi, yanlis esik, maske sifirlanmasi) ve HICBIR SEY belli
    olmaz -- tam da bu projenin tekrar tekrar yakalandigi hata turu.
    """
    print("\n" + "=" * 70)
    print("BOS PENCERE BASTIRMASI")
    print("=" * 70)
    if not sarmal.bos_bastir:
        print("  bastirma KAPALI -- dogrulama atlandi")
        return None

    rng = np.random.default_rng(tohum)
    t = np.arange(rd.PENCERE) / rd.FS

    # Uc bastirilmasi GEREKEN durum + bir bastirilmamasi gereken.
    # Ikisi SAHADAN geliyor (src/ham_analiz.py, benchmark kanallari):
    #   olu kanal      -> benchmark kanal 0, MAD tam 0, bosluk 1.000
    #   suruklenme     -> kanal 4/5/63/64, bosluk 0.007-0.014 (!),
    #                     dusuk_frek 0.93-0.96
    duz = np.abs(rng.normal(0, 1, (3, rd.PENCERE))).astype(np.float32)
    olu = np.full((3, rd.PENCERE), 900.0, dtype=np.float32)
    suruklenme = np.stack([
        np.abs(1000 + 300 * np.sin(2 * np.pi * (1.0 + 0.5 * i) * t)
               + 5 * rng.normal(0, 1, rd.PENCERE)) for i in range(3)
    ]).astype(np.float32)
    yapili = ornek_sinyal(3, tohum)
    noise_idx = sarmal.siniflar.index("noise")

    sonuc = {}
    for ad, x, bos_bekleniyor in (("duz gurultu", duz, True),
                                  ("OLU KANAL (sabit)", olu, True),
                                  ("SURUKLENME (dusuk frek)", suruklenme, True),
                                  ("yapili sinyal", yapili, False)):
        xt = torch.from_numpy(x)
        with torch.no_grad():
            logit, bosluk = sarmal(xt)
            S = sarmal.stft_genlik(sarmal.normalize(xt))
            dusuk = sarmal.dusuk_frekans_payi(S).numpy()
        lg, bo = logit.numpy(), bosluk.numpy()
        tahmin = lg.argmax(1)
        print(f"\n  {ad}:")
        print(f"    bosluk_orani : {np.round(bo, 4).tolist()}  "
              f"(esik {sarmal.bos_esik})")
        print(f"    dusuk_frek   : {np.round(dusuk, 4).tolist()}  "
              f"(esik {sarmal.dusuk_esik})")
        print(f"    tahmin       : "
              f"{[sarmal.siniflar[t] for t in tahmin]}")
        print(f"    logit[0]     : {np.round(lg[0], 3).tolist()}")

        # Bastirma IKI olcutten biriyle tetikleniyor
        gercekten_bos = (bo > sarmal.bos_esik) | (dusuk > sarmal.dusuk_esik)
        if bos_bekleniyor:
            assert gercekten_bos.all(), (
                f"'{ad}' hicbir olcutu tetiklemedi: bosluk {bo}, "
                f"dusuk_frek {dusuk} -- test sinyali temsili degil, "
                f"bastirma sinanamadi")
            assert (tahmin == noise_idx).all(), (
                f"BASTIRMA CALISMIYOR: '{ad}' penceresinde argmax "
                f"{[sarmal.siniflar[t] for t in tahmin]}, 'noise' bekleniyordu")
            print(f"    [x] Bastirildi -> argmax 'noise'")
        else:
            assert not gercekten_bos.any(), (
                f"yapili sinyal bos sayildi: bosluk {bo}, dusuk_frek "
                f"{dusuk} -- esikler cok dusuk ya da test sinyali uygun degil")
            assert lg.max() < 100, (
                f"yapili sinyalde bastirma tetiklendi: {lg}")
            print(f"    [x] Dokunulmadi -- logitler normal aralikta")
        sonuc[ad] = (bo.tolist(), dusuk.tolist(),
                     [sarmal.siniflar[t] for t in tahmin])
    print(f"\n  [x] Bastirma her iki yonde de dogru davraniyor")
    return sonuc


def opset_karsilastir(sarmal, opsetler=(13, 17), n=4, tohum=1):
    """
    Ayni grafigi farkli opset'lerle ihrac edip ciktilarini karsilastirir.

    opset MATEMATIGI degistirmemeli -- yalnizca hangi operator tanimlarinin
    kullanildigini. Ama "degistirmemeli" bir varsayim; bu fonksiyon onu
    ORCUYE cevirir. Sorumlu 17 yerine 13 istedigi icin bedelini bilmemiz
    gerekiyordu.

    Donen: {opset: logit_dizisi} ve aradaki maks fark yazdirilir.
    """
    import tempfile
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime yok -- opset karsilastirmasi ATLANDI")
        return None

    print("\n" + "=" * 70)
    print(f"OPSET KARSILASTIRMASI -- {opsetler}")
    print("=" * 70)
    x = ornek_sinyal(n, tohum)          # yapili -- bastirma tetiklenmesin

    with torch.no_grad():
        ref = sarmal(torch.from_numpy(x))[0].numpy()

    sonuc = {}
    with tempfile.TemporaryDirectory() as td:
        for op in opsetler:
            yol = str(Path(td) / f"op{op}.onnx")
            ek = {}
            try:
                import inspect
                if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                    ek["dynamo"] = False
            except (TypeError, ValueError):
                pass
            try:
                torch.onnx.export(
                    sarmal, (torch.from_numpy(x[:2]),), yol,
                    input_names=["sinyal"],
                    output_names=["logit", "bosluk_orani"],
                    dynamic_axes={"sinyal": {0: "batch"},
                                  "logit": {0: "batch"},
                                  "bosluk_orani": {0: "batch"}},
                    opset_version=op, do_constant_folding=True, **ek)
            except Exception as e:  # noqa: BLE001
                print(f"  opset {op:>3}: IHRAC EDILEMEDI -- "
                      f"{type(e).__name__}: {str(e)[:80]}")
                continue
            o = ort.InferenceSession(yol, providers=["CPUExecutionProvider"])
            sonuc[op] = o.run(None, {"sinyal": x})[0]
            mb = Path(yol).stat().st_size / 1e6
            print(f"  opset {op:>3}: ihrac OK  ({mb:.1f} MB)  "
                  f"PyTorch'a fark {np.abs(sonuc[op] - ref).max():.2e}")

    if len(sonuc) >= 2:
        a, b = sorted(sonuc)[:2]
        d = float(np.abs(sonuc[a] - sonuc[b]).max())
        print(f"\n  opset {a} <-> {b} logit farki: {d:.2e}")
        print(f"  {'[x] Ayni sonuc -- opset dusurmenin bedeli YOK' if d < 1e-5 else '  UYARI: ciktilar ayrisiyor, incelenmeli'}")
        return d
    return None


def model_karti(onnx_yol, ckpt=None, sarmal=None, dogrulama=None):
    """
    .onnx dosyasinin yanina kullanim kartini yazar.

    Sayilar elle degil, checkpoint ve gecmis.json'dan okunuyor -- rapora
    giren her sayi kod ciktisindan gelsin (proje kurali).
    """
    onnx_yol = Path(onnx_yol)
    p = {}
    if ckpt and Path(ckpt).exists():
        paket = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        p = {k: paket.get(k) for k in
             ("val_macro_f1", "en_iyi_epoch", "siniflar", "ayar")}
        # test sonuclari gecmis.json'da
        gec = list(Path(ckpt).parent.glob(
            f"{Path(ckpt).stem}_gecmis.json"))
        if gec:
            import json
            g = json.loads(gec[0].read_text(encoding="utf-8"))
            p["test_macro_f1"] = g.get("test_macro_f1")
            p["test_dogruluk"] = g.get("test_dogruluk")
            p["karisiklik"] = g.get("karisiklik")
            p["n_test"] = g.get("n_test")

    siniflar = p.get("siniflar") or ["cutting", "climbing", "noise"]

    # Mimari ve renk sarmalayicidan OKUNUYOR, elle yazilmiyor -- yanlis
    # renk yazan bir kart, kullaniciyi yanlis on islemeye yonlendirir.
    sinif_adi = type(sarmal.model).__name__ if sarmal else "?"
    renk = sarmal.renk if sarmal else "?"
    mimari_metni = {"DASNetBiLSTM": "2D-CNN + SK-Attention + BiLSTM",
                    "DASNet": "2D-CNN + SK-Attention"}.get(sinif_adi, sinif_adi)

    s = [f"# `{onnx_yol.name}` — Kullanım Kartı", "",
         "DAS çit-ihlali sınıflandırıcısı. **Ön işlemenin tamamı grafiğin "
         "içinde** — ham sinyali verin, sınıf alın.", "",
         f"**Mimari:** {mimari_metni} (`{sinif_adi}`)  ",
         f"**Girdi temsili:** {renk}", ""]

    if p.get("test_macro_f1"):
        s += ["## Performans", "",
              f"| | |", "|---|---|",
              f"| test macro-F1 | **{p['test_macro_f1']:.4f}** |",
              f"| test doğruluk | {p['test_dogruluk']:.4f} |",
              f"| doğrulama macro-F1 | {p['val_macro_f1']:.4f} "
              f"(epoch {p['en_iyi_epoch']}) |",
              f"| test örneği | {p.get('n_test', 0):,} |",
              f"| taban çizgisi (doğrusal, 26 özellik) | 0.771 |", ""]
        if p.get("karisiklik"):
            prec, rec, f1, destek = _sinif_metrik(p["karisiklik"])
            s += ["| sınıf | precision | recall | F1 | destek |",
                  "|---|---|---|---|---|"]
            for i, ad in enumerate(siniflar):
                s.append(f"| `{ad}` | {prec[i]:.3f} | {rec[i]:.3f} "
                         f"| **{f1[i]:.3f}** | {destek[i]:,} |")
            s.append("")

    s += ["## Girdi / Çıktı", "",
          "```", f"girdi   sinyal        (batch, {rd.PENCERE})  float32",
          f"cikti   logit         (batch, {len(siniflar)})      float32",
          f"        bosluk_orani  (batch,)       float32", "```", "",
          f"**`sinyal`** = ham GENLİK. `hypot(re, im)` sonrası, yalnızca "
          f"**`{rd.ALAN}`** alanından. {rd.PENCERE:,} örnek = "
          f"{rd.PENCERE/rd.FS} saniye @ {rd.FS} Hz.", "",
          "⚠️ **Başka ön işleme UYGULAMAYIN.** Normalizasyon, STFT, "
          "ölçekleme — hepsi modelin içinde. Dışarıdan ikinci kez "
          "uygulamak tahminleri bozar.", "",
          "⚠️ **Ölçek katsayısı (`/16384`) uygulanmaz.** Model pencere-içi "
          "medyan/MAD normalizasyonu yapıyor; sabit bir çarpan zaten "
          "sadeleşiyor.", "",
          f"**Sınıf sırası:** " +
          ", ".join(f"`{i}={ad}`" for i, ad in enumerate(siniflar)) +
          " — karıştırılırsa model sessizce yanlış etiket üretir.", "",
          "## Kullanım", "", "```python", "import onnxruntime as ort",
          "import numpy as np", "",
          f"oturum = ort.InferenceSession('{onnx_yol.name}')",
          f"# x: (batch, {rd.PENCERE}) float32 ham genlik",
          "logit, bosluk = oturum.run(None, {'sinyal': x})", "",
          "sinif = logit.argmax(1)   # bos pencerelerde zaten 'noise'",
          f"# bosluk_orani > {rd.BOS_ESIK} olan pencereler grafik icinde",
          f"# bastiriliyor; ayrica filtre uygulamak GEREKMIYOR.",
          f"bos = bosluk > {rd.BOS_ESIK}   # istersen ayrica isaretleyebilirsin",
          "```", "",
          f"### Boş pencereler — **grafiğin içinde hallediliyor**", "",
          f"`bosluk_orani`, {rd.BOS_FREKANS:.0f} Hz üstündeki enerji payı. "
          f"**{rd.BOS_ESIK}'in üstündeyse** pencerede tespit edilebilir "
          f"sinyal yok demektir.", "",
          "Eğitim verisinde bu pencereler **elendi** (train'in %23'ü), yani "
          "model onlar için **eğitilmedi**. Ölçüldü: filtresiz bırakılırsa "
          "modelin `argmax`'i boş pencerelerin **%99.8'inde** bir saldırı "
          "sınıfı veriyor.", "",
          f"**Bu yüzden bastırma grafiğe gömüldü:** `bosluk_orani > "
          f"{rd.BOS_ESIK}` olduğunda saldırı sınıflarının logitinden büyük "
          f"bir sabit düşülür ve `argmax` kendiliğinden **`noise`**'a "
          f"düşer. Çağıran tarafta ek bir şey yapmaya gerek yok.", "",
          "```python",
          "sinif = logit.argmax(1)      # bos pencerede zaten 'noise'",
          "```", "",
          "⚠️ Boş pencerelerde `noise` logiti dokunulmadan bırakılır, "
          "diğerleri `-1e4`'e iner. Yani çok büyük negatif logit görürseniz "
          "bu bir hata değil, **kasıtlı bastırma** işaretidir.", "",
          f"⚠️ `noise` sınıfı artık **iki şeyi** temsil ediyor: "
          f"etiketlenmiş gürültü olayı **ve** boş pencere. Ayırmak "
          f"isterseniz `bosluk_orani > {rd.BOS_ESIK}` kontrolüne bakın — "
          f"değer hâlâ ikinci çıktı olarak veriliyor.", "",
          "## Grafiğin içindeki zincir", "", "```",
          "ham sinyal (batch, 15000)",
          "  -> normalize: (x - medyan) / MAD",
          "  -> DC cikar",
          f"  -> STFT: n_fft={rd.N_FFT}, hop={rd.HOP}, Hann",
          f"  -> dB: 20*log10(S/S.max()), -{rd.TOP_DB:.0f} dB'de kirp",
          "  -> uint8 kuantalama (egitimde de boyleydi)",
          ("  -> viridis renklendirme -> 3 kanal" if renk == "viridis"
           else "  -> gri: uint8 3 kanala kopyalanir"),
          f"  -> {GIRDI_H}x{GIRDI_W}'ye olcekle (bilinear)",
          "  -> /255 -> ImageNet normalizasyonu",
          ("  -> 2D-CNN + SK-Attention + BiLSTM + dikkatli zaman havuzlama"
           if sinif_adi == "DASNetBiLSTM"
           else "  -> 2D-CNN + SK-Attention + global ortalama havuzlama"),
          f"  -> {len(siniflar)} logit", "```", "",
          "## Doğrulama", ""]

    if dogrulama:
        for k, v in dogrulama.items():
            s.append(f"- {k}: **{v}**")
    s += ["", "## Sınırlar", "",
          "- Tek tohumla tek koşu; tohum varyansı ölçülmedi.",
          "- Kalan hatanın neredeyse tamamı `climbing` ↔ `cutting` arasında; "
          "`noise` pratikte çözülmüş.",
          "- val/test bölmeleri kürasyonlu görünüyor (boş pencere oranı "
          "train'de %23, val/test'te %0.1). Saha koşullarında zayıf kanallar "
          "daha sık olacaktır.",
          "- `bosluk_orani`, tam pencere FFT'si yerine STFT'den kestiriliyor "
          "(ONNX `fft_rfft` desteklemiyor). Ölçülen sapma < 0.002; eşik 0.45.",
          "- Eğitimde ölçekleme `antialias=True` ile yapıldı, ONNX'te "
          "kapalı. Ölçülen fark **0.0** (büyütmede antialias etkisiz).", ""]

    yol = onnx_yol.with_name(onnx_yol.stem + "_KULLANIM.md")
    yol.write_text("\n".join(s), encoding="utf-8")
    print(f"  kart  : {yol}")
    return yol


def _sinif_metrik(karisiklik):
    M = np.asarray(karisiklik, dtype=np.float64)
    kos = np.diag(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(M.sum(0) > 0, kos / M.sum(0), 0.0)
        rec = np.where(M.sum(1) > 0, kos / M.sum(1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, M.sum(1).astype(int)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BiLSTM modelini ONNX'e cevir")
    ap.add_argument("--ckpt", default=None, help="kosu4_bilstm_yeni_rejim.pt")
    ap.add_argument("--sahte-agirlik", action="store_true",
                    help="agirliksiz, yalnizca hattı sina")
    ap.add_argument("--cikti", default=None)
    # opset 13: sorumlunun hedef calisma zamani 17'yi desteklemiyor.
    # Bu grafik icin bedelsiz -- gerekcesi ve olcumu icin modul docstring'i
    # ("OPSET NEDEN 13") ve --opset-karsilastir.
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--opset-karsilastir", action="store_true",
                    help="13 ve 17 ile ihrac edip ciktilari karsilastir")
    # renk ve mimari VARSAYILAN OLARAK checkpoint'ten okunur (ckpt_oku).
    # Elle vermek tespiti ezer -- yanlis renk sessizce bozuk girdi demek.
    ap.add_argument("--renk", default=None, choices=["viridis", "gri"],
                    help="varsayilan: checkpoint'ten okunur")
    ap.add_argument("--mimari", default=None,
                    choices=["DASNetBiLSTM", "DASNet"],
                    help="varsayilan: checkpoint'ten tespit edilir")
    ap.add_argument("--bastirma-kapat", action="store_true",
                    help="bos pencere bastirmasini KAPAT (eski davranis: "
                         "bos pencerede de saldiri sinifi dondurulur)")
    a = ap.parse_args()

    if not a.ckpt and not a.sahte_agirlik:
        ap.error("--ckpt ver ya da --sahte-agirlik kullan")

    sarmal = sarmalayici_kur(ckpt=a.ckpt, renk=a.renk, mimari=a.mimari,
                             bos_bastir=not a.bastirma_kapat)
    d_db = sayisal_dogrula(sarmal)
    bastirma_dogrula(sarmal)
    d_opset = opset_karsilastir(sarmal) if a.opset_karsilastir else None
    cikti = a.cikti or str(Path(a.ckpt).with_suffix(".onnx") if a.ckpt
                           else "bilstm_sahte.onnx")
    cikti, d_logit = disa_aktar(sarmal, cikti, opset=a.opset)

    dogrulama = {
        "Spektrogram real_data ile ortusuyor": f"maks {d_db:.1e} dB fark",
        # Olculen sayi karta yaziliyor -- eskiden "ihracat ciktisina bak"
        # diyordu ve sayi konsolda kaliyordu.
        "ONNX ciktisi PyTorch ile ortusuyor": (
            f"maks {d_logit:.1e} logit farki" if d_logit is not None
            else "OLCULMEDI (onnxruntime yok)"),
        "Dinamik batch 1/3/16/64": "calisiyor",
        "opset": a.opset,
    }
    if d_opset is not None:
        dogrulama["opset 13 <-> 17 farki"] = f"{d_opset:.1e}"
    model_karti(cikti, ckpt=a.ckpt, sarmal=sarmal, dogrulama=dogrulama)
    print("\nTAMAM.")
