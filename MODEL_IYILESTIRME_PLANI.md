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

## Gerçekçi beklenti

```
Şu an:        0.628 ± 0.122
Paket 1:      belirsiz yön — ama SAYI GEÇERLİ olur
Paket 1+2:    0.68 – 0.73 bandı
+ Paket 3:    +0.01 – 0.03
```

**0.85+ beklenmiyor** ve öyle bir sayı çıkarsa bir yerde sızıntı aranmalı. Sınıf
karışıklığı (chain ↔ metal) veri kaynaklı ve model tarafından çözülemez.

**Rapora yazılacak:** kaç konfigürasyon denendiği. Plan **2 konfigürasyon**
(Paket 1, sonra Paket 1+2). Bu dürüst ve savunulabilir bir sayı.

---

## Karar

Önce **Paket 1 tek başına** koşulacak, temiz bir "önce/sonra" karşılaştırması
alınacak, durulacak ve rapor verilecek. Ardından **Paket 1+2**.
