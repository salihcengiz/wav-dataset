# DAS 2D-CNN + SK — Gerçek Saha Verisi Modeli

**Dosya:** `das_2dcnn_sk_gercek_kosu3.pt`  
**Üretim:** 2026-08-27T08:46:44Z  
**Kod sürümü:** `f0b0d13c1770`

## Performans

| | |
|---|---|
| **test macro-F1** | **0.8737** |
| test doğruluk | 0.8392 |
| doğrulama macro-F1 | 0.8657 |
| taban çizgisi (doğrusal, 26 özellik) | 0.771 |
| **taban çizgisine göre** | **+0.1027** |

### Sınıf bazında (test)

| sınıf | precision | recall | F1 | destek |
|---|---|---|---|---|
| `cutting` | 0.834 | 0.757 | **0.794** | 17,386 |
| `climbing` | 0.822 | 0.882 | **0.851** | 21,910 |
| `noise` | 0.977 | 0.976 | **0.977** | 3,554 |

Karışıklık matrisi (satır = gerçek):

| | `cutting` | `climbing` | `noise` |
|---|---|---|---|
| **`cutting`** | 13,167 | 4,168 | 51 |
| **`climbing`** | 2,557 | 19,322 | 31 |
| **`noise`** | 68 | 16 | 3,470 |

## Eğitim

- Konfigürasyon: **gri_sifirdan** (koşu 3)
- Veri: train 220,834 / val 37,517 / test 42,850 pencere
- Bağımsız kayıt dosyası (train): 21,101
- 29 epoch koşuldu, en iyi epoch 23 (val macro-F1 izlendi)
- Adam, lr 0.001, batch 64, label smoothing 0.1, tohum 42
- Artırma: zaman/frekans maskeleme (1-2 serit, <=%10), cevirme YOK
- Bölmeler: ekibin train_final / val_final / test_final dosyalari; dosya/oturum/tarih duzeyinde cakisma yok

## Girdi — bu tarif birebir uygulanmalı

```
1. .bin.hdf5 ac, kanal veri kumesini oku
2. P alanindan: hypot(re, im)  -- genlik, FAZ DEGIL
3. CSV penceresini kes [window_start:window_end]
4. 15,000 ornege standartlastir (7.5 s @ 2000 Hz)
   uzunsa enerji merkezine kirp, kisaysa yansitmali doldur
5. Bos mu? (500 Hz ustu enerji payi > 0.45) -> bossa ELE
6. Normalize: pencere-ici (s - medyan) / MAD
7. STFT n_fft=256, hop=64 -> 129 x 231 dB
8. Renk: gri,  olcek 224x320 (bilinear, antialias=True)
9. [0,1] -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

⚠️ **Ölçek katsayısı (UYGULANMAZ (normalizasyon sadelestirir)).**

⚠️ **Sınıf sırası:** `{'cutting': 0, 'climbing': 1, 'noise': 2}` — karıştırılırsa model sessizce yanlış etiket üretir.

## Nasıl yüklenir

```python
import torch
from model import DASNet

paket = torch.load('das_2dcnn_sk_gercek_kosu3.pt', map_location='cpu', weights_only=False)
model = DASNet(attention='sk', n_classes=3)
model.load_state_dict(paket['state_dict'])
model.eval()

# on isleme: real_data.py + gercek_veri_kumesi.hazirla()
```

## Sınırlar

- Test skoru AYRI bir test setinde olculdu (42.850 pencere) -- egitimde de dogrulama secilirken de kullanilmadi.
- Kalan hatanin neredeyse tamami climbing <-> cutting arasinda; noise pratikte cozulmus durumda (F1 0.98).
- val/test bolmeleri kurasyonlu gorunuyor: bos pencere orani train'de %23, val/test'te %0.1. Saha kosullarinda zayif kanallar daha sik olacaktir.
- Bos pencereler cikarimda da elenmelidir (pencere_yukle None dondurur) -- model onlar icin egitilmedi.
- Girdi ON ISLEMESI birebir ayni olmalidir; farkli bir normalizasyon veya pencere boyutu sessizce bozuk tahmin uretir.
- Tek tohumla tek kosu. Tohum varyansi olculmedi.

## Paketin içindekiler

`mimari`, `girdi`, `on_isleme`, `siniflar`, `sinif_indeksi`, `performans`, `egitim`, `sinirlar`, `state_dict`. Yani model başka bir projede **bu dosyaya bakarak** yeniden kurulabilir.
