# DAS 2D-CNN + SK — Önceden Eğitilmiş Model

**Dosya:** `das_2dcnn_sk_v1.pt`
**Üretim tarihi:** 2026-08-24 09:42:55 UTC
**Kod sürümü:** `0b4b48dcbe39`

---

## Bu nedir?

Sentetik DAS spektrogramlarıyla eğitilmiş, üç çit-ihlali olayını sınıflandıran
bir 2D-CNN. Mimari You ve ark. (2025, IEEE Sensors Journal 25(22), 41320–41328)
makalesinden uyarlanmıştır.

**Amacı:** gerçek saha verisiyle çalışmaya başlarken **sıfırdan başlamamak.**
Omurga ağırlıkları, spektrogramlarda kenar/darbe/bant örüntülerini tanımayı
zaten öğrenmiş durumda.

| | |
|---|---|
| Parametre sayısı | 34,835 |
| Girdi | 224×320 (dikey=frekans, yatay=zaman), 3 kanal |
| Özellik boyutu | 64 (sınıflandırıcı öncesi) |
| Sınıflar | `chain_link_climbing`, `fence_cutting`, `metal_bending` |
| Eğitim örneği | 959 |
| **Etkin bağımsız kayıt** | **19** |

## Beklenen performans

**macro-F1 0.6222 ± 0.1662** (4 katlı, kaynak-gruplu çapraz doğrulama)

Katman katman: 0.5186, 0.8446, 0.4158, 0.7096

> ⚠️ **Bu modelin kendi başına dürüst bir test skoru YOKTUR.** Tüm veriyle
> eğitildi. Yukarıdaki sayı, aynı konfigürasyonun çapraz doğrulamadaki
> tahminidir. Bu modeli eğitildiği veri üzerinde ölçerseniz anlamsız derecede
> yüksek bir sonuç alırsınız.

## Nasıl yüklenir

```python
import torch
from model import DASNet, load_pretrained

bundle = torch.load('das_2dcnn_sk_v1.pt', map_location='cpu', weights_only=False)

# 1) Aynı 3 sınıfla kullanmak
model = DASNet(attention='sk')
model.load_state_dict(bundle['state_dict'])
model.eval()

# 2) Gerçek veride FARKLI sayıda sınıfla (transfer öğrenme)
model = DASNet(attention='sk', n_classes=YENI_SINIF_SAYISI)
load_pretrained(model, bundle)
# -> omurga + SK-Attention yüklenir, classifier sıfırdan başlar

# 3) Çok az saha verisi varsa: omurgayı dondur, sadece sınıf katmanını eğit
load_pretrained(model, bundle, freeze_backbone=True)
```

> ⚠️ **`load_state_dict(..., strict=False)` tek başına YETMEZ.** `strict=False`
> yalnızca eksik/fazla anahtarları tolere eder; sınıf sayısı değiştiğinde
> `classifier` katmanında **boyut uyuşmazlığı** hatası verir. `load_pretrained()`
> bu tensörleri atlayıp omurgayı yükler.

**Ön işleme aynı olmalı:**
```python
from torchvision.transforms import v2
tf = v2.Compose([
    v2.Resize((224, 320), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## Bilinmesi gerekenler

- Bu model TUM veriyle egitildi; kendi basina durust bir test skoru YOKTUR.
- Beklenen performans capraz dogrulama tahminidir (cv_performance).
- Egitim verisi ~959 spektrogram ama yalnizca 19 BAGIMSIZ kayittan turetilmistir.
- Sentetik veri gercek DAS verisinin yerini tutmaz; saha verisiyle fine-tuning sarttir.
- 'chain_link_climbing' sinifi akustik olarak tutarli bir kume degildir; orneklerinin ~%38'i 'metal_bending' ile karistirilmaktadir.
- 'Olay yok / normal' sinifi YOKTUR -- model 'hangi tehdit' sorusunu cevaplar, 'tehdit var mi' sorusunu cevaplayamaz.

## Paketin içindekiler

`bundle` sözlüğü şunları taşır: `architecture`, `input`, `classes`,
`training` (hiperparametreler + epoch kayıp geçmişi), `cv_performance`,
`caveats`, `state_dict`. Yani model başka bir projede **bu dosyaya bakarak**
yeniden kurulabilir.

## Katman modelleri

Çapraz doğrulamadaki 4 model ayrıca `outputs/checkpoints/` altında
duruyor (`fold_i_sk_best.pt`). Topluluk (ensemble) denemek veya
katman varyansını incelemek için kullanılabilir.
