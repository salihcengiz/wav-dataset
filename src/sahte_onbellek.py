"""
YERELDE TEST ICIN SAHTE VERI URETICI

Gercek veri sunucudan cikamaz. Ama kodu sunucuya gondermeden once yerelde
denemek zorundayiz -- sunucuda her deneme bir tur kaybi ve GPU paylasimli.

Bu script, gercek verinin YAPISINI taklit eden kucuk bir kume uretir:

    sahte_veri/
        segment/record_*.bin.hdf5     ham veri (P/S alanli, int16 I/Q)
        train_final.csv               CSV indeksi (file/channel/event/window_*)
        val_final.csv
        test_final.csv
        onbellek_train_final_k0.h5    spektrogram onbellegi
        onbellek_val_final_k0.h5
        onbellek_test_final_k0.h5

Sayilar gercege benzemez (168 pencere vs 220.834) -- amac SONUC uretmek
degil, HATTIN CALISTIGINI dogrulamak. Sekiller, dtype'lar, alan adlari ve
dosya duzeni gercegiyle ayni.

=== NEDEN GEREKLI ===

Bu kumeyle yerelde su testler kosulabiliyor:

    python src/gercek_veri_kumesi.py <onbellek>     # 8 birim testi
    python src/gercek_egitim.py --kosu 1 --veri <klasor> --cikti <k> \
        --epoch 1 --batch 8 --isci 0                # egitim dongusu
    python CNN-BiLSTM/egitim_bilstm.py --hizli --veri <klasor> ...
    python src/gercek_rapor.py --cikti <k>          # rapor
    python src/onbellek_kur.py --kok <klasor> --k 0 # onbellek kurulumu

Bu oturumda yakalanan hatalarin cogu tam da bu testlerde cikti:
birim testinin tum kumeyi bellege toplamasi (OOM), lr gecmisinin
skalerle ezilmesi, erken durdurma mesajinin sabit SABIR basmasi,
ONNX'in dort ayri operatorde takilmasi.

=== KULLANIM ===

    python src/sahte_onbellek.py                    # ./sahte_veri altina
    python src/sahte_onbellek.py --hedef /tmp/x --dosya 20
"""
import argparse
import sys
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

# Gercek .bin.hdf5 dosyalarinin dtype'i: cift polarizasyon, int16 I/Q
DT = np.dtype([("P", [("re", "<i2"), ("im", "<i2")]),
               ("S", [("re", "<i2"), ("im", "<i2")])])

SINIFLAR = ["climbing", "cutting", "noise"]


def uret(hedef="sahte_veri", dosya=12, kanal=14, ornek=40_000, tohum=0):
    """Ham .bin.hdf5 + CSV indeksleri uretir."""
    if h5py is None:
        raise ImportError("h5py gerekli")
    hedef = Path(hedef)
    (hedef / "segment").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(tohum)

    satir = []
    for i in range(dosya):
        sinif = SINIFLAR[i % len(SINIFLAR)]
        yol = hedef / "segment" / f"record_SA4_{sinif.upper()}_{i}_raw.bin.hdf5"
        with h5py.File(yol, "w") as f:
            for j in range(kanal):
                a = np.zeros(ornek, dtype=DT)
                if sinif == "noise":
                    # noise: dusuk frekans agirlikli -> bos_mu False
                    t = np.arange(ornek) / rd.FS
                    x = 20 * np.sin(2 * np.pi * 8 * t) + rng.normal(0, 2, ornek)
                else:
                    # olay: ortada darbe kumesi
                    x = rng.normal(0, 3, ornek)
                    x[18_000:22_000] += rng.normal(0, 12, 4000)
                # Gercek veride ham deger araligi ±11 (Rapor 4.1)
                a["P"]["re"] = np.clip(x, -11, 11).astype("<i2")
                a["P"]["im"] = np.clip(rng.normal(0, 2, ornek), -11, 11).astype("<i2")
                f.create_dataset(str(160 + j), data=a)
                # Gercek CSV'de pencere uzunlugu degisken (Rapor 4.4)
                uzunluk = [7_500, 10_000, 15_000, 20_000][j % 4]
                satir.append({"file": str(yol), "channel": 160 + j,
                              "event": sinif, "window_start": 15_000,
                              "window_end": 15_000 + uzunluk})

    df = pd.DataFrame(satir)
    df.to_csv(hedef / "train_final.csv", index=False)
    # val/test daha kucuk -- gercekte de oyle
    df.iloc[::3].to_csv(hedef / "val_final.csv", index=False)
    df.iloc[1::3].to_csv(hedef / "test_final.csv", index=False)
    print(f"  ham veri : {dosya} dosya x {kanal} kanal = {len(df)} satir")
    return hedef


def onbellekle(hedef):
    """Uretilen CSV'lerden spektrogram onbelleklerini kurar."""
    import onbellek_kur
    for ad in ("train_final.csv", "val_final.csv", "test_final.csv"):
        onbellek_kur.kur(kok=hedef, csv=ad, k=0, ustune_yaz=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Yerel test icin sahte veri")
    ap.add_argument("--hedef", default="sahte_veri")
    ap.add_argument("--dosya", type=int, default=12)
    ap.add_argument("--kanal", type=int, default=14)
    ap.add_argument("--sadece-ham", action="store_true",
                    help="onbellek kurma, yalnizca ham veri + CSV")
    a = ap.parse_args()

    print("=" * 70)
    print(f"SAHTE VERI URETIMI -> {a.hedef}")
    print("=" * 70)
    h = uret(a.hedef, dosya=a.dosya, kanal=a.kanal)
    if not a.sadece_ham:
        onbellekle(h)
    print(f"\nHazir. Ornek kullanim:")
    print(f"  python src/gercek_veri_kumesi.py {h}/onbellek_train_final_k0.h5")
    print(f"  python src/gercek_egitim.py --kosu 1 --veri {h} "
          f"--cikti {h}/cikti --epoch 1 --batch 8 --isci 0")
