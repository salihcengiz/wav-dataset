"""
CNN + BiLSTM -- EGITIM KOSTURUCUSU

Bu dosyanin KENDI EGITIM DONGUSU YOKTUR. src/gercek_egitim.py'nin kos()
fonksiyonunu cagirir, yalnizca modeli ve rejimi degistirir.

=== NEDEN DONGU KOPYALANMIYOR ===

kos() icine dort ders gomulu (DURUM.md Bolum 6):

    A1  model secimi ve erken durdurma val macro-F1 izler, val_loss DEGIL
    A2  esitlikte EN ERKEN epoch secilir
    A3  set_deterministic() -- ayni tohum ayni sonuc
    +   her iyilesmede diske checkpoint, teste YALNIZCA BIR KEZ bakma

Kopyalasaydik bunlar zamanla ayrisirdi. Ayrica egitim/degerlendirme
donusumu (gercek_veri_kumesi.hazirla) tek fonksiyondan geciyor -- bu
projenin en temel ilkesi.

=== YENI EGITIM REJIMI ===

Uc kosunun ardindan olculen bulguya cevap veriyor
(GERCEK_VERI_EGITIM_SONUCLARI.md Bolum 5):

    ayar          eski   yeni   gerekce
    ------------  -----  -----  --------------------------------------
    maskeleme     0.5    0.0    Uc kosuda da dogrulama dogrulugu egitimin
                                USTUNDEYDI -- egitim metrikleri maskelenmis
                                girdilerde olculuyor. 21.101 bagimsiz
                                dosyayla maskelemeye gerek yok.
    maks epoch    40     80     Kosu 1 tavana carparken hala iyilesiyordu
    erken dur.    6      10     LR dususu 3 kotu epoch'ta tetikleniyor;
                                6 sabir dususe etkisini gosterecek yer
                                birakmiyor

=== ⚠ ATIF UYARISI ===

Bu kosu IKI degiskeni birden degistiriyor: mimari (BiLSTM) ve rejim
(maskeleme kapali + uzun butce). Sonuc 0.8843'u gecerse hangisinden
geldigi BILINEMEZ.

Ucuz telafi: sonuc iyiyse mevcut SK modelini de ayni rejimle kos --

    python ../src/gercek_egitim.py --kosu 1 --maske-p 0 --epoch 80 --sabir 10

O zaman atif netlesir. Bu, rapora yazilmasi gereken bir cekincedir.

=== KULLANIM ===

    python egitim_bilstm.py --hizli      # ~2 dk duman testi, kaydetmez
    python egitim_bilstm.py              # tam kosu

    # yerelde sahte onbellekle:
    python egitim_bilstm.py --hizli --veri <klasor> --cikti <klasor> \
        --batch 8 --isci 0
"""
import argparse
import sys
from functools import partial
from pathlib import Path

_KOK = Path(__file__).resolve().parent.parent
_SRC = _KOK / "src"
for _y in (str(_SRC), str(Path(__file__).resolve().parent)):
    if _y not in sys.path:
        sys.path.insert(0, _y)

from gercek_egitim import CIKTI, VERI, kos          # noqa: E402
from model_bilstm import DASNetBiLSTM               # noqa: E402

# Yeni rejim -- gerekceler modul docstring'inde
MASKE_P = 0.0
MAKS_EPOCH = 80
SABIR = 10
KOSU_NO = 4


def main():
    ap = argparse.ArgumentParser(description="CNN + BiLSTM egitimi")
    ap.add_argument("--veri", default=str(VERI))
    ap.add_argument("--cikti", default=str(CIKTI))
    ap.add_argument("--epoch", type=int, default=MAKS_EPOCH)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--isci", type=int, default=6)
    ap.add_argument("--maske-p", type=float, default=MASKE_P)
    ap.add_argument("--sabir", type=int, default=SABIR)
    ap.add_argument("--sinif-agirligi", action="store_true")
    ap.add_argument("--hizli", action="store_true",
                    help="kisa duman testi, sonuc raporlanmaz")

    # --- model dugmeleri ---
    ap.add_argument("--gizli", type=int, default=128,
                    help="LSTM gizli boyutu (her yon icin)")
    ap.add_argument("--frekans-bin", type=int, default=4,
                    help="frekans ekseni kac bine indirilecek")
    ap.add_argument("--katman", type=int, default=1,
                    help="LSTM katman sayisi")
    ap.add_argument("--zaman-cozunurlugu-artir", action="store_true",
                    help="son blokta zaman havuzlamasini kapat -> 80 cerceve. "
                         "UYARI: omurga artik DASNet ile ayni DEGIL, "
                         "karsilastirmayi kirletir")
    a = ap.parse_args()

    model_fn = partial(DASNetBiLSTM,
                       gizli=a.gizli,
                       frekans_bin=a.frekans_bin,
                       katman=a.katman,
                       zaman_havuzlama=not a.zaman_cozunurlugu_artir)

    print("=" * 78)
    print("CNN + BiLSTM  --  zamansal oruntu mimarisi")
    print("=" * 78)
    print(f"  hipotez : zaman eksenini cokertmek yerine dizi olarak islemek")
    print(f"            climbing/cutting ayrimini iyilestirir")
    print(f"  model   : gizli={a.gizli} frekans_bin={a.frekans_bin} "
          f"katman={a.katman} "
          f"zaman_cerceve={'80' if a.zaman_cozunurlugu_artir else '40'}")
    print(f"  rejim   : maske_p={a.maske_p} maks_epoch={a.epoch} "
          f"sabir={a.sabir}")
    if a.zaman_cozunurlugu_artir:
        print(f"  !!! omurga DASNet ile ayni DEGIL -- karsilastirma kirlenir")
    # Konsol ciktisi ASCII: Windows'ta cp1254 kodlamasi Unicode simgeleri
    # basamiyor ve surec UnicodeEncodeError ile oluyor.
    print(f"\n  ASILMASI GEREKEN: kosu 1, test macro-F1 0.8843")
    print(f"  UYARI: Bu kosu mimari VE rejimi birlikte degistiriyor; sonuc")
    print(f"    iyi cikarsa atif icin SK modelini de yeni rejimle kosmak")
    print(f"    gerekir (--kosu 1 --maske-p 0 --epoch 80 --sabir 10).")

    return kos(KOSU_NO, veri=a.veri, cikti=a.cikti, epoch=a.epoch,
               batch=a.batch, isci=a.isci,
               sinif_agirligi=a.sinif_agirligi, hizli=a.hizli,
               model_fn=model_fn, maske_p=a.maske_p, sabir=a.sabir)


if __name__ == "__main__":
    main()
