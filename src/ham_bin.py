"""
HAM .bin OKUYUCU -- benchmark kayitlarinin KIRPILMAMIS hali

=== NEDEN GEREKLI ===

`/tf/segment/Fence Benchmark Data/` altindaki `.hdf5` kopyalari olay
cevresine KIRPILMIS: record_26'da 201 kanaldan yalnizca 11'i var
(71-97). Saha waterfall'inda yanlis alarm ureten kanallar tam da eksik
olanlar -- o yuzden sorunu kopyalarla olcemiyorduk.

Ham dosyalar `/tf/rawData/2026_newdata/IGA_RECORDS/` altinda ve 201
kanalin hepsini tasiyor (9-17 kat buyuk).

=== FORMAT (2026-09-02'de cozuldu ve DOGRULANDI) ===

    baslik   16.384 bayt (16 KB), atlanir
    govde    uint16 little-endian, ZAMAN-oncelikli
             sekil = (ornek_sayisi, kanal_sayisi)
    deger    genlik -- ham `sampletype: RealUInt16`

Boyut kontrolu yedi dosyada da tam tutuyor:

    dosya_boyutu = 16384 + kanal * ornek * 2

DOGRULAMA: ayni kaydin .hdf5 kopyasinda kanal 71-97'nin dogru degerleri
var. Bu okuyucunun cikardigi degerlerle KOPYA arasindaki maks fark
**0** olcuLdu (kanal 71, 72, 73, 74). Yani format tahmin degil, kanit.

Alternatif dizilis (kanal-oncelikli) denendi ve tutmadi (fark ~9.6e3).

=== DEGER TURU FARKI -- DIKKAT ===

Egitim verimiz `.bin.hdf5` KARMASIK saklıyor: [('P', [('re','<i2'),
('im','<i2')])], ve genlik icin hypot(re, im) aliniyor.

Benchmark ham dosyalari ise GERCEK uint16 -- zaten genlik. hypot
UYGULANMAZ. `.hdf5` kopyasi da tek float ([('P','<f4')]) tutuyor ve bu
okuyucunun degerleriyle birebir ortusuyor.

=== KULLANIM ===

    from ham_bin import ham_ac, kanal_oku

    A, bilgi = ham_ac(ham_yol, kopya_yol)   # A: (ornek, kanal) memmap
    s = kanal_oku(A, kanal=0, bas=30000, son=45000)
"""
import os

import numpy as np

BASLIK = 16384          # bayt -- olculdu, yedi dosyada da ayni


def ham_bilgi(ham_yol, kopya_yol=None, n_kanal=None, n_ornek=None):
    """
    Ham dosyanin sekil bilgisini cikarir.

    kopya_yol verilirse `.hdf5` kopyasinin attrs'undan okunur (guvenli
    yol). Verilmezse n_kanal zorunlu; n_ornek dosya boyutundan turetilir.
    """
    boyut = os.path.getsize(ham_yol)
    attrs = {}
    if kopya_yol and os.path.exists(kopya_yol):
        import h5py
        with h5py.File(kopya_yol, "r") as f:
            attrs = dict(f.attrs)
            kn = sorted(int(k) for k in f.keys() if str(k).isdigit())
            if kn and n_ornek is None:
                n_ornek = int(f[str(kn[0])].shape[0])
        if n_kanal is None:
            n_kanal = int(attrs.get("channels", 0))

    if not n_kanal:
        raise ValueError("n_kanal bilinmiyor -- kopya_yol ver ya da elle gec")
    if n_ornek is None:
        n_ornek = (boyut - BASLIK) // (2 * n_kanal)

    beklenen = BASLIK + n_kanal * n_ornek * 2
    return {
        "boyut": boyut, "n_kanal": n_kanal, "n_ornek": n_ornek,
        "baslik": boyut - n_kanal * n_ornek * 2,
        "tutarli": abs(boyut - beklenen) < 2 * n_kanal,   # bir satir tolerans
        "attrs": attrs,
    }


def ham_ac(ham_yol, kopya_yol=None, n_kanal=None, n_ornek=None):
    """
    (ornek, kanal) seklinde uint16 memmap dondurur.

    memmap: 22 MB'lik dosyalar icin sart degil ama 31 GB'lik ham
    kayitlar da ayni formatta; belleğe almadan dilim okumak gerekiyor.
    """
    b = ham_bilgi(ham_yol, kopya_yol, n_kanal, n_ornek)
    if not b["tutarli"]:
        raise ValueError(
            f"boyut tutmuyor: {b['boyut']} bayt, "
            f"{b['n_kanal']} x {b['n_ornek']} x 2 + baslik bekleniyordu. "
            f"Baslik {b['baslik']} cikti (16384 olmali).")
    A = np.memmap(ham_yol, dtype="<u2", mode="r",
                  offset=b["baslik"], shape=(b["n_ornek"], b["n_kanal"]))
    return A, b


def kanal_oku(A, kanal, bas, son):
    """Tek kanaldan pencere. float64 -- real_data fonksiyonlari oyle bekliyor."""
    return np.asarray(A[bas:son, kanal], dtype=np.float64)


def dogrula(ham_yol, kopya_yol, n_kanal_test=4):
    """
    Okuyucuyu .hdf5 kopyasina karsi sinar.

    Bu kontrol olmadan format varsayimi sessizce yanlis olabilir ve
    butun analiz coper. Kopyada bulunan kanallar birebir karsilastiriliyor.
    """
    import h5py
    A, b = ham_ac(ham_yol, kopya_yol)
    with h5py.File(kopya_yol, "r") as f:
        kn = sorted(int(k) for k in f.keys() if str(k).isdigit())
        farklar = []
        for k in kn[:n_kanal_test]:
            ref = np.asarray(f[str(k)]["P"], dtype=np.float64)
            bizim = kanal_oku(A, k, 0, len(ref))
            farklar.append(float(np.abs(bizim - ref).max()))
    return farklar, b


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ham .bin okuyucu dogrulamasi")
    ap.add_argument("--ham", default="/tf/rawData/2026_newdata/IGA_RECORDS/"
                                     "record_26_202601051117_raw_port_1.bin")
    ap.add_argument("--kopya", default="/tf/segment/Fence Benchmark Data/"
                                       "record_26_202601051117_raw_port_1.bin.hdf5")
    a = ap.parse_args()
    farklar, b = dogrula(a.ham, a.kopya)
    print(f"boyut {b['boyut']:,}  kanal {b['n_kanal']}  ornek {b['n_ornek']}  "
          f"baslik {b['baslik']}")
    print(f"kopyayla maks fark: {farklar}")
    print("ESLESTI" if max(farklar) == 0 else "AYRISIYOR -- format yanlis")
