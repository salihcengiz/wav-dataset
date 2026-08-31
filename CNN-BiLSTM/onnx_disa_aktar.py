"""
CNN + BiLSTM -- ONNX DISA AKTARIM

Modeli, ON ISLEMENIN TAMAMI ICINE GOMULU halde tek bir .onnx dosyasina
cevirir. Girdi HAM SINYAL:

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

=== KULLANIM ===

    python onnx_disa_aktar.py --ckpt /tf/.../kosu4_bilstm_yeni_rejim.pt

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


class OnIslemeliModel(nn.Module):
    """
    Ham sinyalden logit'e -- on islemenin tamami iceride.

    Tum sabitler (Hann penceresi, DFT matrisleri, viridis LUT, ImageNet
    istatistikleri) buffer olarak kaydediliyor; boylece .onnx dosyasinin
    icine gomulurler ve disaridan hicbir sey gerekmez.
    """

    def __init__(self, model, pencere=rd.PENCERE, fs=rd.FS, n_fft=rd.N_FFT,
                 hop=rd.HOP, top_db=rd.TOP_DB, bos_frekans=rd.BOS_FREKANS,
                 girdi=(GIRDI_H, GIRDI_W), renk="viridis"):
        super().__init__()
        self.model = model
        self.pencere, self.fs = pencere, fs
        self.n_fft, self.hop, self.top_db = n_fft, hop, top_db
        self.girdi, self.renk = tuple(girdi), renk
        self.n_cerceve = 1 + (pencere - n_fft) // hop

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
        """500 Hz ustundeki guc payi -- STFT binlerinden kestirim."""
        G = S * S
        return (G * self.yuksek_maske).sum((1, 2)) / G.sum((1, 2)).clamp_min(1e-12)

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
        """(B, 15000) -> (logit (B,3), bosluk_orani (B,))"""
        S = self.stft_genlik(self.normalize(sinyal))
        bosluk = self.bosluk_orani_stft(S)
        return self.model(self.goruntu(self.db_cevir(S))), bosluk


# ---------------------------------------------------------------
def sarmalayici_kur(ckpt=None, renk="viridis", **model_kw):
    net = DASNetBiLSTM(**model_kw)
    if ckpt:
        paket = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        sd = paket.get("state_dict", paket)
        net.load_state_dict(sd)
        print(f"  agirliklar yuklendi: {ckpt}")
        if "val_macro_f1" in paket:
            print(f"  val macro-F1: {paket['val_macro_f1']:.4f}  "
                  f"(epoch {paket.get('en_iyi_epoch')})")
    return OnIslemeliModel(net, renk=renk).eval()


def sayisal_dogrula(sarmal, n=4, tohum=0):
    """
    Sarmalayicinin on islemesi real_data ile AYNI mi?

    Bu kontrol olmadan ihracat sessizce yanlis girdi ureten bir model
    verebilir -- ve hicbir hata mesaji cikmaz.
    """
    print("\n" + "=" * 70)
    print("SAYISAL DOGRULAMA -- sarmalayici vs real_data")
    print("=" * 70)
    rng = np.random.default_rng(tohum)
    t = np.arange(rd.PENCERE) / rd.FS
    ham = np.stack([
        np.abs(3 + np.sin(2 * np.pi * (20 + 15 * i) * t)
               + 0.4 * rng.normal(0, 1, rd.PENCERE)) for i in range(n)])

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


def disa_aktar(sarmal, cikti, opset=17, dogrula=True):
    print("\n" + "=" * 70)
    print(f"ONNX IHRACATI -> {cikti}  (opset {opset})")
    print("=" * 70)
    ornek = torch.randn(2, rd.PENCERE).abs() * 3 + 3
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
            return cikti
        oturum = ort.InferenceSession(str(cikti),
                                      providers=["CPUExecutionProvider"])
        with torch.no_grad():
            t_logit, t_bos = sarmal(ornek)
        o_logit, o_bos = oturum.run(None, {"sinyal": ornek.numpy()})
        d1 = np.abs(t_logit.numpy() - o_logit).max()
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
            x = torch.randn(b, rd.PENCERE).abs() * 3 + 3
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
    return cikti


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BiLSTM modelini ONNX'e cevir")
    ap.add_argument("--ckpt", default=None, help="kosu4_bilstm_yeni_rejim.pt")
    ap.add_argument("--sahte-agirlik", action="store_true",
                    help="agirliksiz, yalnizca hattı sina")
    ap.add_argument("--cikti", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--renk", default="viridis", choices=["viridis", "gri"])
    a = ap.parse_args()

    if not a.ckpt and not a.sahte_agirlik:
        ap.error("--ckpt ver ya da --sahte-agirlik kullan")

    sarmal = sarmalayici_kur(ckpt=a.ckpt, renk=a.renk)
    sayisal_dogrula(sarmal)
    cikti = a.cikti or str(Path(a.ckpt).with_suffix(".onnx") if a.ckpt
                           else "bilstm_sahte.onnx")
    disa_aktar(sarmal, cikti, opset=a.opset)
    print("\nTAMAM.")
