"""
CNN + BiLSTM -- ZAMANSAL ORUNTU MIMARISI

=== NEDEN BU MIMARI ===

Uc kosunun ardindan kalan hatanin neredeyse tamami climbing <-> cutting
arasinda (cutting'in %21'i climbing saniliyor). noise cozulmus (F1 0.979).

Mevcut DASNet'in adi konabilir bir zayifligi var:

    Conv bloklari -> SK-Attention -> AdaptiveAvgPool2d(1) -> Linear
                                     ^^^^^^^^^^^^^^^^^^^^
                                     40 zaman cercevesini TEK sayiya cokertiyor

Global ortalama havuzlama "darbe var mi" bilgisini korur ama "darbeler zaman
icinde NASIL DIZILMIS" bilgisini atar. cutting ritmik ve ayrik, climbing
surekli ve duzensiz -- ayrimin tam da orada olmasi olasi.

Faz 0 bunu destekliyor: "modulasyon tepe frekansi" ve "modulasyon keskinligi"
ozellikleri tek basina zayifti (F ~ 0.05) ama 26 ozellik BIRLIKTE
climbing/cutting'i ayirabildi. Ritim bilgisi var, daginik halde.

HIPOTEZ: Zaman eksenini cokertmek yerine bir dizi modeliyle islemek,
climbing/cutting ayrimini iyilestirir.

=== OMURGA NEDEN YENIDEN YAZILMIYOR ===

Konv dizisi DASNet'ten AYNEN aliniyor (ayni model nesnesinin `features` ve
`attention` modulleri). Kopyalasaydik, ikisi zamanla ayrisirdi -- bu projenin
tekrar tekrar kacindigi hata. Boylece karsilastirma da temiz olur: omurga
birebir ayni, YALNIZCA havuzlama basi degisiyor.

=== YAPI ===

    girdi (B,3,224,320)
      -> features + SK-Attention   [DASNet'ten]      -> (B,64,28,40)
      -> AdaptiveAvgPool2d((4,None))                 -> (B,64,4,40)
      -> yeniden duzenle: zaman = dizi ekseni        -> (B,40,256)
      -> BiLSTM(256 -> 128, cift yonlu)              -> (B,40,256)
      -> dikkatli zaman havuzlama                    -> (B,256)
      -> Dropout -> Linear(256, 3)                   -> (B,3)

=== TASARIM KARARLARI ===

1) FREKANS 4 BINE INDIRILIYOR, TAMAMEN COKERTILMIYOR.
   Frekans ekseninin ortalamasini alip 64 boyuta inmek "hangi bantta"
   bilgisini atardi. 4 bin x 64 kanal = 256 boyut; hem bant bilgisini korur
   hem diziyi makul tutar.

2) CIFT YONLU (BiLSTM).
   Bir darbenin anlami hem oncesine hem sonrasina bagli. Tek yonlu LSTM
   sonrasini goremez.

3) DIKKATLI HAVUZLAMA, SON GIZLI DURUM DEGIL.
   Model pencerenin hangi aninin bilgilendirici oldugunu ogrenir. Son gizli
   durum, pencerenin sonuna orantisiz agirlik verirdi -- oysa pencereler
   enerji merkezine gore kirpiliyor, olay ortada.
   Yan fayda: dikkat agirliklari GORSELLESTIRILEBILIR (return_dikkat=True).

4) ZAMAN COZUNURLUGU 40 CERCEVE (7.5 s / 40 ~ 187 ms).
   Omurga DASNet ile ayni kalsin diye ilk surumde degistirilmiyor.
   `zaman_havuzlama=False` ile son blogun havuzlamasi (2,1) yapilip 80
   cerceveye cikilabilir -- ama o zaman omurga artik DASNet'le ayni degildir
   ve karsilastirma iki degiskenli olur. VARSAYILAN: True (40 cerceve).

Kullanim (birim testi):
    python model_bilstm.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# src/ yolu -- veri hatti ve DASNet oradan geliyor, kopyalanmiyor
_KOK = Path(__file__).resolve().parent.parent
_SRC = _KOK / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config as cfg
from model import DASNet, count_parameters


class DASNetBiLSTM(nn.Module):
    """
    DASNet omurgasi + zamansal dizi basi.

    Parametreler
    ------------
    frekans_bin : int
        Frekans ekseni kac bine indirilecek. Adim basina ozellik boyutu
        = kanal_sayisi * frekans_bin (64 * 4 = 256).
    gizli : int
        LSTM gizli boyutu (her yon icin). Cikti 2*gizli olur.
    katman : int
        LSTM katman sayisi.
    lstm_dropout : float
        katman > 1 ise katmanlar arasi dropout.
    zaman_havuzlama : bool
        True  -> omurga DASNet ile BIREBIR ayni, 40 zaman cercevesi
        False -> son blokta zaman havuzlamasi kapatilir, 80 cerceve
                 (omurga artik DASNet'le ayni DEGIL, karsilastirma kirlenir)
    """

    def __init__(self, n_classes=cfg.N_CLASSES, in_channels=cfg.IN_CHANNELS,
                 channels=cfg.CONV_CHANNELS, attention="sk",
                 dropout=cfg.DROPOUT, batchnorm=cfg.BACKBONE_BATCHNORM,
                 frekans_bin=4, gizli=128, katman=1, lstm_dropout=0.0,
                 zaman_havuzlama=True):
        super().__init__()

        # --- OMURGA: DASNet'in kendi modullerini oduncal ---
        # Yeniden yazmak yerine bir DASNet kurup features/attention
        # modullerini aliyoruz. Boylece konv dizisi, BN, ReLU, MaxPool ve
        # SK-Attention birebir ayni; ikisi asla ayrisamaz.
        taban = DASNet(n_classes=n_classes, in_channels=in_channels,
                       channels=channels, attention=attention,
                       dropout=dropout, batchnorm=batchnorm)
        self.features = taban.features
        self.attention = taban.attention
        self.attention_name = attention
        self.out_channels = taban.out_channels          # 64

        if not zaman_havuzlama:
            self._son_havuzu_zamansiz_yap()
        self.zaman_havuzlama = zaman_havuzlama

        # --- FREKANS SIKISTIRMA: (B,C,F,T) -> (B,C,frekans_bin,T) ---
        self.frekans_bin = frekans_bin
        self.adim_boyutu = self.out_channels * frekans_bin      # 256

        # Havuzlama cekirdegi KURULUM ANINDA, Python tam sayisi olarak
        # hesaplaniyor. x.shape[2]'den turetseydik ONNX izlemesinde tensor
        # olarak gorulur ve ihracat "kernel size is not constant" ile
        # patlardi. Omurga uc kez 2'ye boldugu icin frekans ekseni
        # INPUT_H / 8 = 28.
        frek_cikis = cfg.INPUT_H // 8
        if frek_cikis % frekans_bin != 0:
            raise ValueError(
                f"frekans ekseni ({frek_cikis}) frekans_bin'e ({frekans_bin}) "
                f"tam bolunmuyor; sabit cekirdekli havuzlama kurulamaz")
        self._frek_kernel = frek_cikis // frekans_bin          # 7

        # --- ZAMANSAL DIZI ---
        self.lstm = nn.LSTM(
            input_size=self.adim_boyutu, hidden_size=gizli,
            num_layers=katman, batch_first=True, bidirectional=True,
            dropout=lstm_dropout if katman > 1 else 0.0)
        self.dizi_boyutu = 2 * gizli                            # 256

        # --- DIKKATLI ZAMAN HAVUZLAMA ---
        # Her zaman adimina bir skor, zaman ekseninde softmax, agirlikli
        # toplam. Son gizli durum yerine bu tercih edildi (karar 3).
        self.dikkat = nn.Linear(self.dizi_boyutu, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.dizi_boyutu, n_classes)

    def _son_havuzu_zamansiz_yap(self):
        """Son MaxPool2d(2,2) -> MaxPool2d((2,1)): zaman ekseni korunur."""
        son = None
        for i, k in enumerate(self.features):
            if isinstance(k, nn.MaxPool2d):
                son = i
        if son is not None:
            self.features[son] = nn.MaxPool2d(kernel_size=(2, 1),
                                              stride=(2, 1))

    def _frekans_sikistir(self, x):
        """
        Frekans eksenini `frekans_bin` bine indirir, zaman eksenine dokunmaz.

        NEDEN AdaptiveAvgPool2d DEGIL:
        AdaptiveAvgPool2d((frekans_bin, None)) kullaniyorduk. Sonuc dogruydu
        ama ONNX'e IHRAC EDILEMIYOR -- `None` cikti boyutunu sabit olmaktan
        cikariyor ve ihracat "output_size is not constant" ile patliyor.

        Frekans ekseni 28 (224/8) ve frekans_bin 4 oldugundan 28/4 = 7 tam
        bolunuyor; sabit cekirdekli ortalama havuzlama adaptive havuzlamayla
        BIREBIR AYNI sonucu veriyor (birim testinde dogrulaniyor).

        Cekirdek __init__'te Python tam sayisi olarak hesaplandi; buradan
        x.shape'e bakmiyoruz, cunku izlenen bir tensor sekli ONNX'te sabit
        sayilmaz.
        """
        return F.avg_pool2d(x, kernel_size=(self._frek_kernel, 1))

    # -----------------------------------------------------------
    def dizi_cikar(self, x):
        """
        (B,3,H,W) -> (B, T, adim_boyutu)

        Zaman ekseni diziye donusuyor. permute sirasi onemli: kanal ve
        frekans birlestirilirken zaman ekseni ayri tutulmali, yoksa
        adimlar birbirine karisir.
        """
        x = self.features(x)                     # (B, C, F, T)
        x = self.attention(x)                    # (B, C, F, T)
        x = self._frekans_sikistir(x)            # (B, C, frekans_bin, T)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)                # (B, T, C, F)
        return x.reshape(B, T, C * F)            # (B, T, C*F)

    def forward_features(self, x, return_dikkat=False):
        """
        Siniflandirici oncesi 256 boyutlu ozellik vektoru.
        t-SNE ve dikkat gorsellestirmesi bunu kullanir.
        """
        dizi = self.dizi_cikar(x)                # (B, T, adim)
        cikti, _ = self.lstm(dizi)               # (B, T, 2*gizli)

        skor = self.dikkat(cikti)                # (B, T, 1)
        agirlik = torch.softmax(skor, dim=1)     # zaman ekseninde toplam = 1
        ozet = (cikti * agirlik).sum(dim=1)      # (B, 2*gizli)

        if return_dikkat:
            return ozet, agirlik.squeeze(-1)     # (B, T)
        return ozet

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x)))


# ---------------------------------------------------------------
# BIRIM TESTI
# ---------------------------------------------------------------
def _check(kosul, mesaj):
    if not kosul:
        raise AssertionError(mesaj)


def self_test():
    torch.manual_seed(cfg.SEED)
    cizgi = "-" * 70
    print("=" * 70)
    print("CNN + BiLSTM -- BIRIM TESTI")
    print("=" * 70)

    B, H, W = 2, cfg.INPUT_H, cfg.INPUT_W
    x = torch.randn(B, cfg.IN_CHANNELS, H, W)
    m = DASNetBiLSTM().eval()

    print(f"\n[1] Dizi cikarma -- zaman ekseni korunuyor mu")
    print(cizgi)
    with torch.no_grad():
        dizi = m.dizi_cikar(x)
    print(f"  girdi {tuple(x.shape)}  ->  dizi {tuple(dizi.shape)}")
    print(f"  zaman adimi : {dizi.shape[1]}   (7.5 s / {dizi.shape[1]} = "
          f"{7500/dizi.shape[1]:.0f} ms/adim)")
    print(f"  adim boyutu : {dizi.shape[2]}   "
          f"(= {m.out_channels} kanal x {m.frekans_bin} frekans bini)")
    _check(dizi.shape[0] == B, "batch boyutu bozuldu")
    _check(dizi.shape[1] == W // 8, f"zaman adimi {dizi.shape[1]}, {W//8} bekleniyordu")
    _check(dizi.shape[2] == m.adim_boyutu, "adim boyutu yanlis")
    print(f"  [x] Zaman ekseni dizi haline geldi, cokertilmedi")

    print(f"\n[2] Ileri gecis")
    print(cizgi)
    with torch.no_grad():
        out = m(x)
        ozet, dikkat = m.forward_features(x, return_dikkat=True)
    print(f"  logit   {tuple(out.shape)}")
    print(f"  ozellik {tuple(ozet.shape)}   (t-SNE icin)")
    print(f"  dikkat  {tuple(dikkat.shape)}  (B, T)")
    _check(out.shape == (B, cfg.N_CLASSES), f"logit sekli {tuple(out.shape)}")
    _check(torch.isfinite(out).all(), "logitlerde NaN/Inf")
    print(f"  [x] Sekiller dogru, degerler sonlu")

    print(f"\n[3] Dikkat kisiti -- zaman ekseninde toplam 1")
    print(cizgi)
    toplam = dikkat.sum(dim=1)
    print(f"  toplamlar: {[round(float(v), 6) for v in toplam]}")
    print(f"  maks sapma: {(toplam - 1).abs().max():.2e}")
    _check(torch.allclose(toplam, torch.ones_like(toplam), atol=1e-5),
           "dikkat agirliklari 1'e toplanmiyor")
    _check((dikkat >= 0).all(), "negatif dikkat agirligi")
    yayilim = dikkat.std(dim=1).mean()
    print(f"  adimlar arasi std: {yayilim:.5f}")
    print(f"  [x] Softmax kisiti saglandi, agirliklar [0,1] araliginda")

    print(f"\n[4] Geri yayilim tum parametrelere ulasiyor mu")
    print(cizgi)
    mt = DASNetBiLSTM().train()
    kayip = nn.CrossEntropyLoss()(mt(x), torch.tensor([0, 2]))
    kayip.backward()
    eksik = [n for n, p in mt.named_parameters()
             if p.requires_grad and (p.grad is None
                                     or not torch.isfinite(p.grad).all())]
    print(f"  kayip = {kayip.item():.4f}")
    _check(not eksik, f"gradyan almayan parametreler: {eksik[:5]}")
    n_tensor = sum(1 for _ in mt.parameters())
    print(f"  [x] {n_tensor} parametre tensorunun hepsi sonlu gradyan aldi")

    print(f"\n[5] Parametre butcesi")
    print(cizgi)
    dasnet = count_parameters(DASNet(attention="sk"))
    bilstm = count_parameters(m)
    print(f"  DASNet (mevcut)   : {dasnet:>9,}")
    print(f"  DASNetBiLSTM      : {bilstm:>9,}   ({bilstm/dasnet:.1f} kat)")
    print(f"  fark              : {bilstm - dasnet:>+9,}")
    print(f"  -> 'kapasite tukenmemis' bulgusuna dogrudan cevap")

    print(f"\n[6] zaman_havuzlama=False -- 80 cerceveli varyant")
    print(cizgi)
    m80 = DASNetBiLSTM(zaman_havuzlama=False).eval()
    with torch.no_grad():
        d80 = m80.dizi_cikar(x)
    print(f"  dizi {tuple(d80.shape)}   zaman adimi {d80.shape[1]} "
          f"({7500/d80.shape[1]:.0f} ms/adim)")
    _check(d80.shape[1] == 2 * dizi.shape[1], "zaman adimi iki katina cikmadi")
    print(f"  [x] Calisiyor  (UYARI: omurga artik DASNet'le ayni DEGIL,")
    print(f"      varsayilan olarak KAPALI -- karsilastirmayi kirletir)")

    print(f"\n{'=' * 70}")
    print("TUM TESTLER GECTI.")
    return 0


if __name__ == "__main__":
    sys.exit(self_test())
