# Model Başarısını Artırma — Seçenekler ve Plan

> Veri seti yetersizliği ve `chain_link_climbing` ↔ `metal_bending` benzerliği
> bilinçli olarak kapsam dışı bırakılmıştır. Bu belge **yalnızca model ve eğitim
> tarafında** yapılabilecekleri, ölçülen kanıtlara dayanarak sıralar.
>
> Başlangıç noktası: **macro-F1 0.628 ± 0.122** (SK-Attention, 4 katman).

---

## Önce iki kısıt

**1. Test setine bakarak ayar seçemeyiz.** 10 şey deneyip test F1'i en yüksek
olanı seçersek, raporladığımız sayı şişer. Değişiklikleri **gerekçeye göre**
seçmeliyiz, deneme-yanılmayla değil.

**2. Ölçüm hassasiyetimiz zayıf.** Standart sapma **±0.122**. Bu, 0.628 ile 0.66
arasındaki farkı güvenilir şekilde ayırt edemeyeceğimiz anlamına geliyor — o fark
gürültünün içinde kalır.

> Pratik sonucu: **az sayıda, büyük değişiklik** yapmalıyız. On tane küçük ayar
> denemek hem istatistiksel olarak anlamsız hem de test setini kirletir.

---

## A — Bozuk olanı düzelt (iyileştirme değil, tamir)

| # | Değişiklik | Kanıt | Etki |
|---|---|---|---|
| **A1** | Model seçimi ve erken durdurma **macro-F1**'i izlesin | Baseline'da 2 katman 1. epoch'ta donduruldu, test 0.162 | **Büyük** — ölçümü geçerli kılar |
| **A2** | Eşitlikte **en erken** epoch seçilsin | Katman 0'da doğrulama 10. epoch'ta tavana vurdu, 51 seçildi — 41 epoch boşuna ezber | Orta |
| **A3** | `torch.use_deterministic_algorithms` | Aynı tohum iki farklı sonuç verdi (0.558 / 0.572) | Performans değil, **tekrarlanabilirlik** |

Bunlar tartışmaya açık değil — yapılması gerekiyor.

---

## B — Yüksek beklentili, gerekçesi sağlam

### B1. Omurgaya BatchNorm ekle ⭐ en yüksek beklenti

**Kanıt:** Baseline katman 2'de `val_loss` 1.30'dan **3.48'e** fırladı. Bu,
normalizasyonsuz derin ağların klasik kararsızlık belirtisi — katmanlar arası
aktivasyon dağılımı kayıyor.

**Ne yapar:** Her evrişimden sonra aktivasyonları normalize eder. İki fayda birden:
- **Kararlılık** — eğitim zıplamaz, daha yüksek öğrenme hızına dayanır
- **Düzenlileştirme** — batch istatistikleri hafif gürültü katar, ezberi zorlaştırır

**Neden şu an yok:** Plan 6.1 bloğu harfiyen `Conv + ReLU + MaxPool` diyordu, ona
sadık kalındı (karar **K2**). Plan 6.4'teki sorun giderme tablosunda zaten
"ilk denenecek" diye işaretli.

**Maliyet:** 3 satır kod. Parametre artışı ~200.

### B2. Label smoothing (0.1)

**Kanıt:** Test kayıpları **1.61 / 0.69 / 1.91 / 1.76**, doğruluk ~0.68.
Yüksek kayıp + orta doğruluk = **emin bir şekilde yanlış**.

**Ne yapar:** Modele "%100 emin ol" demek yerine "%90 emin ol" der. Hedef etiketi
`[1, 0, 0]` yerine `[0.9, 0.05, 0.05]` yapar.

**Neden işe yarar:** Model olasılıkları 1.0'a itemediği için aşırı güven cezalanır.
Az veride bilinen ve güvenilir bir düzenleyici.

**Maliyet:** 1 satır (`CrossEntropyLoss(label_smoothing=0.1)`).

---

## C — Girdi temsili

### C1. Zaman ekseni çözünürlüğü ⭐ projeye özgü gerçek bir kayıp

Spektrogramın **gerçek** bilgi içeriği:

```
10 saniye × 2000 Hz = 20.000 örnek
STFT: n_fft=256, hop=64
  →  129 frekans bini  ×  313 zaman çerçevesi
```

Girdimizle karşılaştırma:

| Eksen | Gerçek bilgi | 224'te kalan | Durum |
|---|---|---|---|
| Frekans (dikey) | 129 | 224 | ✅ Bol bol yer var |
| **Zaman (yatay)** | **313** | **224** | ⚠️ **%28 sıkışma** |

**Zaman detayını gerçek çözünürlüğün altına indiriyoruz.** Bu veri setinde önemli
olabilir: `metal_bending`'in üç ayrı parlak darbesi, darbeler arası boşluklar.
Olayları ayıran şey büyük ölçüde **zamanlama örüntüsü**.

**Öneri:** Girdiyi `224×224` yerine **`224 (frekans) × 320 (zaman)`** yap. Model
`AdaptiveAvgPool` kullandığı için mimari değişikliği gerektirmiyor — sadece
dönüşüm ve konfigürasyon.

**Maliyet:** Hesap ~%43 artar (hâlâ çok hızlı). Kod değişikliği küçük.

### C2. Gri tonlama (1 kanal)

Spektrogramlar **viridis** renk haritasıyla kaydedildi: tek bir skaler değer 3 RGB
kanalına yayıldı. Yani girdinin 3 kanalı **tamamen fazlalık** — aynı bilgiyi üç kez
taşıyor.

Gri tonlamaya çevirirsek ilk katman `Conv(1→16)` olur, parametre 448'den 160'a iner.
Doğrulukta büyük fark beklenmiyor ama gereksiz karmaşayı kaldırır.

Plan 1.3 bunu "ablasyon fikri" olarak zaten not etmiş.

### C3. (Büyük iş) Ham STFT matrisini kullan

Şu anda model, verinin **çizilmiş bir resmini** görüyor:

```
STFT (129×313 sayı)
   → matplotlib viridis ile renklendir
   → 400×400 PNG'ye BÜYÜTEREK çiz
   → biz 224×224'e KÜÇÜLTÜYORUZ
   → model bunu görüyor
```

Her adımda bilgi bozuluyor. Ham STFT matrisini doğrudan `.npy` olarak kaydetseydik
model **gerçek veriyi** görürdü.

**Ama:** Bu, veri setini yeniden üretmeyi gerektirir (Faz 0'a döner) ve plan açıkça
PNG kullanmayı söylüyor. **Kapsam kararı verilmeli.** Potansiyeli yüksek ama iş yükü
büyük.

---

## D — Bedava kazançlar (yeniden eğitim gerekmez)

### D1. Test-zamanı artırma (TTA)

Her test görüntüsünü modele **birkaç farklı kırpmayla** ver, tahminlerin ortalamasını al.

- Elimizdeki checkpoint'lerle çalışır, **yeniden eğitim yok**
- Sızıntı riski yok (test etiketlerine dokunulmuyor)
- Tipik kazanç: **+%1–3**
- Maliyet: birkaç dakika

### D2. Veri setinin kendi normalizasyon istatistikleri

Şu an ImageNet ortalaması/std'si kullanılıyor. Kendi 959 görüntümüzün istatistiklerini
hesaplamak daha doğru. Etki küçük ama bedava ve metodolojik olarak temiz.
Plan 7.1 bunu da "ablasyon fikri" diye işaretlemiş.

---

## E — Daha büyük sapmalar (ayrı tartışma)

| Seçenek | Potansiyel | Bedeli |
|---|---|---|
| **Transfer öğrenme** (ImageNet'te önceden eğitilmiş omurga) | **Yüksek** — az veride en güçlü kaldıraç | Referans makalenin mimarisinden sapar; projenin amacı onu uyarlamaktı |
| Eğitim-zamanı MixUp (batch içinde canlı karıştırma) | Orta | Kod ekler; veri setindeki hazır MixUp'tan farklı, sızıntı riski yok |
| `SK_GROUPS` 32 → 16 | **Muhtemelen zararlı** | Kapasite artırır, zaten ezberleniyor |
| Modeli küçültmek | Düşük | 34k zaten çok küçük |

---

## Uygulama paketleri

### Paket 1 — Zorunlu (~40 dk)
**A1 + A2 + A3.** SK ve baseline'ı yeniden koş. İlk kez geçerli bir karşılaştırma.

### Paket 2 — Ana iyileştirme (~40 dk)
**B1 (BatchNorm) + B2 (label smoothing) + C1 (zaman çözünürlüğü 320).**

Üçü birlikte, tek seferde. Ayrı ayrı denemek 6 koşu eder, ±0.122 gürültüsünde
farkları ayırt edemeyiz ve test setini kirletiriz.

### Paket 3 — Bedava (~10 dk)
**D1 (TTA) + D2 (kendi normalizasyon istatistikleri).** Faz 4'te `evaluate.py`
yazarken eklenir.

---

## Uygulama öncesi tahmin (kayıt için)

```
Şu an:        0.628 ± 0.122
Paket 1:      belirsiz yön — ama SAYI GEÇERLİ olur
Paket 1+2:    0.68 – 0.73 bandı        <- BU TAHMIN TUTMADI (bkz. asagi)
+ Paket 3:    +0.01 – 0.03
```

---
---

# ÖLÇÜLEN SONUÇLAR

> Aşağısı uygulama sonrası eklenmiştir. Yukarıdaki bölümler tahmin/planlama
> aşamasında yazıldığı gibi bırakılmıştır.

## Ana tablo — üç konfigürasyon

| Konfigürasyon | SK-Attention | Baseline (SK'siz) | Fark | Eşleştirilmiş t |
|---|---|---|---|---|
| **Paket 0** — `val_loss` izleniyordu | 0.628 ± 0.122 | 0.442 ± 0.216 | +0.186 | *geçersiz* |
| **Paket 1** — A1+A2+A3 | 0.614 ± 0.204 | 0.548 ± 0.112 | +0.066 | 0.63 |
| **Paket 2** — +B1+B2+C1 | **0.622 ± 0.166** | **0.580 ± 0.135** | **+0.042** | **1.26** |

Anlamlılık için t ≈ 3.18 gerekirdi (df=3, p<0.05).

## Paket 1 — sonuç: amacına ulaştı ✅

**A1 (macro-F1 izleme)** — baseline'ın sabote edilmesi durdu:

| Baseline katman | Paket 0 | Paket 1 |
|---|---|---|
| 1 | en iyi epoch **1** → F1 **0.162** | en iyi epoch 3 → F1 **0.473** |
| 2 | en iyi epoch **1** → F1 **0.303** | en iyi epoch 16 → F1 **0.434** |

Baseline ortalaması 0.442 → 0.548, standart sapması 0.216 → 0.112 (yarıya indi).

**A2 (en erken epoch)** — teşhisimizi doğrudan kanıtladı. SK katman 0:

| | Seçilen epoch | Test F1 | Süre |
|---|---|---|---|
| Paket 0 | 51 | 0.572 | 170s |
| Paket 1 | **8** | 0.561 | **52s** |

43 epoch daha az eğitim, neredeyse aynı sonuç → o epoch'lar gerçekten boşunaydı.

**A3 (determinizm)** — doğrulandı. Aynı koşu iki kez çalıştırıldı, 18 epoch'un
tüm değerleri ve test skoru (0.561) **birebir aynı** çıktı.

## Paket 2 — sonuç: ana metrikte kazanç YOK ❌, ikincil göstergelerde VAR ✅

**Ana metrik:** SK 0.614 → 0.622 (+0.008). ±0.166 gürültüsünde anlamsız.
**Tahmin edilen 0.68–0.73 bandına ulaşılamadı.**

### Ama şunlar düzeldi:

**1. Test kayıpları düştü** (label smoothing kaybın tabanını *yükselttiği* halde):

| Katman | Paket 1 | Paket 2 |
|---|---|---|
| 0 | 0.855 | 1.037 |
| 1 | 0.647 | 0.603 |
| 2 | **2.029** | **1.476** |
| 3 | **1.270** | **0.851** |
| Toplam | 4.80 | **3.97** |

**2. SK'de aşırı öğrenme bayrağı kalktı** — Paket 1'de katman 1 bayrak alıyordu
(+0.298), Paket 2'de hiçbir katman almadı.

**3. Baseline'ın kayıp patlaması yumuşadı** — katman 2'de `val_loss` tepesi
3.98 → 2.38. BatchNorm işini yaptı ama tam çözmedi.

**4. SK'de doğrulama-test korelasyonu ilk kez pozitif** — +0.088 → **+0.597**.
(Uyarı: 4 noktayla hesaplandı, anlamlılık için ~0.95 gerekirdi. Sinyal, kanıt değil.)

## SK-Attention katkı sağlıyor mu?

Katman katman fark (SK − baseline):

| Katman | Paket 1 | Paket 2 |
|---|---|---|
| 0 | +0.003 | +0.033 |
| 1 | **+0.365** | +0.141 |
| 2 | **−0.128** | +0.006 |
| 3 | +0.024 | −0.009 |

Paket 2'de fark **küçüldü ama çok daha tutarlı** hale geldi: dört katman da
−0.01 ile +0.14 arasında. Paket 1'deki savrulma (+0.365 / −0.128) kayboldu.

**Raporlanacak ifade:**
> *SK-Attention üç konfigürasyonda da baseline'ın önünde çıktı (+0.19 / +0.07 /
> +0.04) ve son konfigürasyonda 4 katmanın 3'ünde pozitif fark verdi. Ancak
> 4 katmanlı çapraz doğrulamanın istatistiksel gücü, bu büyüklükteki bir farkı
> gürültüden ayırmaya yetmiyor.*

## Değişmeyen sorun: katman 2

| | SK | Baseline |
|---|---|---|
| Paket 0 | 0.463 | 0.303 |
| Paket 1 | 0.306 | 0.434 |
| Paket 2 | 0.416 | 0.410 |

Katman 2, test setinde **6 grup** taşıyan (diğerlerinde 4–5) ve eğitim setinde
`chain_link_climbing`'den yalnızca **3 kayıt** kalan katmandır. Hiçbir
hiperparametre bunu düzeltmedi — sorun hiperparametrede değil.

## Değişmeyen sorun: doğrulama seti güvenilmez

Baseline'da korelasyon Paket 2'de bile **−0.164**. Doğrulama setinde sınıf
başına yalnızca 1 kayıt olması yapısal bir kısıt; metrik değiştirmek çözmedi,
düzenlileştirme de çözmedi. Rapora **sınırlama** olarak yazılmalı.

---

# SONUÇ VE KARAR

Üç konfigürasyon denendi, ana metrik **0.614 / 0.622 / 0.628** aralığında
oturdu — aradaki farklar ±0.17'lik standart sapmanın çok altında, yani
**hiçbiri gerçek bir değişim değil**.

Bu, Faz 0'daki akustik analizin öngördüğünü doğruluyor: `chain_link_climbing`
tutarlı bir sınıf olmadığı sürece tavan bu civarda. Dördüncü, beşinci
konfigürasyonu denemek kazanç getirmez ve test setini kirletmeye başlar.

**Karar: hiperparametre ayarı burada bırakılmıştır.**

**Rapora yazılacak konfigürasyon sayısı: 3.**

**Sonuç olarak sunulacak konfigürasyon: Paket 2** — en yüksek skoru verdiği için
değil (Paket 0 daha yüksekti ama geçersizdi), **metodolojik olarak en sağlamı
olduğu için**: geçerli model seçimi, tekrar üretilebilir sonuçlar, standart
düzenlileştirme uygulanmış.

**Sıradaki:** Paket 3 (TTA + kendi normalizasyon istatistikleri, yeniden eğitim
gerektirmez) → Faz 4 (`evaluate.py`) → Faz 5 (rapor).
