# `sk_gri_kosu3.onnx` — Kullanım Kartı

DAS çit-ihlali sınıflandırıcısı. **Ön işlemenin tamamı grafiğin içinde** — ham sinyali verin, sınıf alın.

**Mimari:** 2D-CNN + SK-Attention (`DASNet`)  
**Girdi temsili:** gri

## Performans

| | |
|---|---|
| test macro-F1 | **0.8737** |
| test doğruluk | 0.8392 |
| doğrulama macro-F1 | 0.8657 (epoch 23) |
| test örneği | 42,850 |
| taban çizgisi (doğrusal, 26 özellik) | 0.771 |

| sınıf | precision | recall | F1 | destek |
|---|---|---|---|---|
| `cutting` | 0.834 | 0.757 | **0.794** | 17,386 |
| `climbing` | 0.822 | 0.882 | **0.851** | 21,910 |
| `noise` | 0.977 | 0.976 | **0.977** | 3,554 |

## Girdi / Çıktı

```
girdi   sinyal        (batch, 15000)  float32
cikti   logit         (batch, 3)      float32
        bosluk_orani  (batch,)       float32
```

**`sinyal`** = ham GENLİK. `hypot(re, im)` sonrası, yalnızca **`P`** alanından. 15,000 örnek = 7.5 saniye @ 2000 Hz.

⚠️ **Başka ön işleme UYGULAMAYIN.** Normalizasyon, STFT, ölçekleme — hepsi modelin içinde. Dışarıdan ikinci kez uygulamak tahminleri bozar.

⚠️ **Ölçek katsayısı (`/16384`) uygulanmaz.** Model pencere-içi medyan/MAD normalizasyonu yapıyor; sabit bir çarpan zaten sadeleşiyor.

**Sınıf sırası:** `0=cutting`, `1=climbing`, `2=noise` — karıştırılırsa model sessizce yanlış etiket üretir.

## Kullanım

```python
import onnxruntime as ort
import numpy as np

oturum = ort.InferenceSession('sk_gri_kosu3.onnx')
# x: (batch, 15000) float32 ham genlik
logit, bosluk = oturum.run(None, {'sinyal': x})

sinif = logit.argmax(1)
gecerli = bosluk <= 0.45   # BU FILTRE ZORUNLU
```

### `bosluk_orani` neden var

500 Hz üstündeki enerji payı. Bu değer **0.45'in üstündeyse** pencerede tespit edilebilir sinyal yok demektir.

Eğitim verisinde bu pencereler **elendi** (train'in %23'ü). Yani model onlar için **eğitilmedi** — filtre uygulanmazsa o pencerelerde anlamsız ama kendinden emin tahminler üretir.

ONNX grafiği koşullu çıktı veremediği için filtre dışarıda uygulanmak zorunda.

## Grafiğin içindeki zincir

```
ham sinyal (batch, 15000)
  -> normalize: (x - medyan) / MAD
  -> DC cikar
  -> STFT: n_fft=256, hop=64, Hann
  -> dB: 20*log10(S/S.max()), -80 dB'de kirp
  -> uint8 kuantalama (egitimde de boyleydi)
  -> gri: uint8 3 kanala kopyalanir
  -> 224x320'ye olcekle (bilinear)
  -> /255 -> ImageNet normalizasyonu
  -> 2D-CNN + SK-Attention + global ortalama havuzlama
  -> 3 logit
```

## Doğrulama

- Spektrogram real_data ile ortusuyor: **maks 1.3e-03 dB fark**
- ONNX ciktisi PyTorch ile ortusuyor: **maks 2.8e-06 logit farki**
- Dinamik batch 1/3/16/64: **calisiyor**
- opset: **13**

## Sınırlar

- Tek tohumla tek koşu; tohum varyansı ölçülmedi.
- Kalan hatanın neredeyse tamamı `climbing` ↔ `cutting` arasında; `noise` pratikte çözülmüş.
- val/test bölmeleri kürasyonlu görünüyor (boş pencere oranı train'de %23, val/test'te %0.1). Saha koşullarında zayıf kanallar daha sık olacaktır.
- `bosluk_orani`, tam pencere FFT'si yerine STFT'den kestiriliyor (ONNX `fft_rfft` desteklemiyor). Ölçülen sapma < 0.002; eşik 0.45.
- Eğitimde ölçekleme `antialias=True` ile yapıldı, ONNX'te kapalı. Ölçülen fark **0.0** (büyütmede antialias etkisiz).
