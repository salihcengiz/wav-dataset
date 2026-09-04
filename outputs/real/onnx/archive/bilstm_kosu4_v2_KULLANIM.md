# `bilstm_kosu4_v2.onnx` — Kullanım Kartı

DAS çit-ihlali sınıflandırıcısı. **Ön işlemenin tamamı grafiğin içinde** — ham sinyali verin, sınıf alın.

**Mimari:** 2D-CNN + SK-Attention + BiLSTM (`DASNetBiLSTM`)  
**Girdi temsili:** viridis

## Performans

| | |
|---|---|
| test macro-F1 | **0.9390** |
| test doğruluk | 0.9225 |
| doğrulama macro-F1 | 0.9367 (epoch 19) |
| test örneği | 42,850 |
| taban çizgisi (doğrusal, 26 özellik) | 0.771 |

| sınıf | precision | recall | F1 | destek |
|---|---|---|---|---|
| `cutting` | 0.927 | 0.880 | **0.903** | 17,386 |
| `climbing` | 0.908 | 0.947 | **0.927** | 21,910 |
| `noise` | 0.993 | 0.982 | **0.987** | 3,554 |

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

oturum = ort.InferenceSession('bilstm_kosu4_v2.onnx')
# x: (batch, 15000) float32 ham genlik
logit, bosluk = oturum.run(None, {'sinyal': x})

sinif = logit.argmax(1)   # bos pencerelerde zaten 'noise'
# bosluk_orani > 0.45 olan pencereler grafik icinde
# bastiriliyor; ayrica filtre uygulamak GEREKMIYOR.
bos = bosluk > 0.45   # istersen ayrica isaretleyebilirsin
```

### Boş pencereler — **grafiğin içinde hallediliyor**

`bosluk_orani`, 500 Hz üstündeki enerji payı. **0.45'in üstündeyse** pencerede tespit edilebilir sinyal yok demektir.

Eğitim verisinde bu pencereler **elendi** (train'in %23'ü), yani model onlar için **eğitilmedi**. Ölçüldü: filtresiz bırakılırsa modelin `argmax`'i boş pencerelerin **%99.8'inde** bir saldırı sınıfı veriyor.

**Bu yüzden bastırma grafiğe gömüldü:** `bosluk_orani > 0.45` olduğunda saldırı sınıflarının logitinden büyük bir sabit düşülür ve `argmax` kendiliğinden **`noise`**'a düşer. Çağıran tarafta ek bir şey yapmaya gerek yok.

```python
sinif = logit.argmax(1)      # bos pencerede zaten 'noise'
```

⚠️ Boş pencerelerde `noise` logiti dokunulmadan bırakılır, diğerleri `-1e4`'e iner. Yani çok büyük negatif logit görürseniz bu bir hata değil, **kasıtlı bastırma** işaretidir.

⚠️ `noise` sınıfı artık **iki şeyi** temsil ediyor: etiketlenmiş gürültü olayı **ve** boş pencere. Ayırmak isterseniz `bosluk_orani > 0.45` kontrolüne bakın — değer hâlâ ikinci çıktı olarak veriliyor.

## Grafiğin içindeki zincir

```
ham sinyal (batch, 15000)
  -> normalize: (x - medyan) / MAD
  -> DC cikar
  -> STFT: n_fft=256, hop=64, Hann
  -> dB: 20*log10(S/S.max()), -80 dB'de kirp
  -> uint8 kuantalama (egitimde de boyleydi)
  -> viridis renklendirme -> 3 kanal
  -> 224x320'ye olcekle (bilinear)
  -> /255 -> ImageNet normalizasyonu
  -> 2D-CNN + SK-Attention + BiLSTM + dikkatli zaman havuzlama
  -> 3 logit
```

## Doğrulama

- Spektrogram real_data ile ortusuyor: **maks 7.9e-04 dB fark**
- ONNX ciktisi PyTorch ile ortusuyor: **maks 2.2e-06 logit farki**
- Dinamik batch 1/3/16/64: **calisiyor**
- opset: **13**

## Sınırlar

- Tek tohumla tek koşu; tohum varyansı ölçülmedi.
- Kalan hatanın neredeyse tamamı `climbing` ↔ `cutting` arasında; `noise` pratikte çözülmüş.
- val/test bölmeleri kürasyonlu görünüyor (boş pencere oranı train'de %23, val/test'te %0.1). Saha koşullarında zayıf kanallar daha sık olacaktır.
- `bosluk_orani`, tam pencere FFT'si yerine STFT'den kestiriliyor (ONNX `fft_rfft` desteklemiyor). Ölçülen sapma < 0.002; eşik 0.45.
- Eğitimde ölçekleme `antialias=True` ile yapıldı, ONNX'te kapalı. Ölçülen fark **0.0** (büyütmede antialias etkisiz).
