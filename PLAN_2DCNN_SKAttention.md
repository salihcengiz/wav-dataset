# PLAN: DAS Çit-İhlali Sınıflandırması — 2D-CNN + SK-Attention

> **Bu dosya, Claude Code için hazırlanmış eksiksiz bir uygulama planıdır.**
> Projeyi hiç bilmeyen birinin sıfırdan uygulayabilmesi için gereken tüm bağlam,
> veri seti detayı, mimari spesifikasyonu, tuzaklar ve kabul kriterleri içindedir.

---

## 0. PROJE BAĞLAMI

### 0.1 Nedir bu proje?

**Inosens** şirketinde yürütülen bir staj projesi. Amaç: **DAS (Distributed Acoustic
Sensing) / Φ-OTDR** tabanlı bir **çevre güvenliği / izinsiz giriş tespit sistemi
(PIDS — Perimeter Intrusion Detection System)** geliştirmek. Odak: **tel örgü çitler**.

DAS teknolojisi, standart tek modlu fiber optik kabloyu kilometrelerce uzunlukta
sürekli bir titreşim sensörü dizisine dönüştürür. Fiber üzerindeki her konum bağımsız
bir sanal sensör gibi davranır.

**Bu görevin kapsamı:** Elimizdeki sentetik spektrogram veri setiyle, üç çit-ihlali
olayını sınıflandıran bir derin öğrenme modeli eğitmek.

### 0.2 Kurulum yöntemi henüz kararlaştırılmadı

Fiberin çite monte mi edileceği yoksa yer altına mı gömüleceği **henüz karara
bağlanmamıştır**. Bu, veri toplama ve model performansını etkileyecek bir açık konudur
ama bu görevin kapsamı dışındadır. Herhangi bir varsayım yapma.

### 0.3 Referans literatür

Model mimarisi şu makaleden uyarlanmıştır:

> **You, J., Men, Y., Liu, Y., Zhang, K., Yu, M., Li, Y., Fang, Q., Zhou, R., Li, X.,
> Chen, H. (2025). "DAS-Based Perimeter Intrusion Detection Using 2D-CNN With
> SKAttention Mechanism." IEEE Sensors Journal, 25(22), 41320-41328.**
> DOI: 10.1109/JSEN.2025.3614201

Bu makale, fiberin **doğrudan çite S-şeklinde kablo bağlarıyla monte edildiği** gerçek
bir çit-PIDS deneyidir. 5 sınıf (Quiet, Hit, Shake, Blow, Drone), 6000 örnek, %99.3
doğruluk, %0.7 NAR (yanlış alarm oranı).

Makalenin kilit bulguları (bizim için doğrudan geçerli):
- 2D-CNN tek başına: **%95.5** doğruluk
- + SE-Attention: %98.2 | + Self-Attention: %97.8 | + CBAM: %98.9 | **+ SK-Attention: %99.3**
- SK-Attention'ın üstünlüğü: çoklu-kernel dinamik seçim mekanizması, akustik olarak
  benzer olayları (Hit vs Shake) diğer sabit-kernel dikkat yöntemlerinden daha iyi ayırıyor
- Ablasyon sonucu: **3×3 + 5×5 kernel ikilisi ve r=16 sıkıştırma oranı en iyi**
- Çıkarım süresi: 3.09 ms/örnek
- SNR -2 dB'ye kadar >%98 doğruluk korunuyor

---

## 1. VERİ SETİ

### 1.1 Kaynak

GitHub: `https://github.com/salihcengiz/wav-dataset`
İlgili klasör: **`synthetic_dataset/`**

### 1.2 Klasör yapısı

```
synthetic_dataset/
    fence_cutting/          # ~334 örnek
        *.wav
        *_spectrogram.png
    chain_link_climbing/    # ~334 örnek
        *.wav
        *_spectrogram.png
    metal_bending/          # ~332 örnek
        *.wav
        *_spectrogram.png
```

**Toplam: ~1000 örnek, 3 sınıf, dengeli dağılım.**

Bu görevde **sadece `*_spectrogram.png` dosyaları** kullanılacak. `.wav` dosyaları
ileride 1D model karşılaştırması için saklanmalı, silinmemeli.

### 1.3 Spektrogram özellikleri

| Özellik | Değer |
|---|---|
| Boyut | **400 × 400 piksel** (tümü aynı, doğrulandı) |
| Format | PNG, RGB |
| Renk haritası | `viridis` (matplotlib) |
| Eksen | Yok (eksensiz, kırpılmış — doğrudan CNN girdisi) |
| Kaynak sinyal | 10.0 saniye, 2000 Hz, 20000 örnek |
| STFT parametreleri | `n_fft=256`, `hop_length=64` → ~129 frekans bini × ~313 çerçeve |
| dB dönüşümü | `librosa.amplitude_to_db(ref=np.max)` |

> **Not — renk haritası hakkında:** Spektrogramlar viridis renk haritasıyla
> kaydedildiği için tek bir skaler değer 3 RGB kanalına yayılmıştır (bilgi fazlalığı).
> Bu, pratikte sorun değildir; olduğu gibi 3 kanallı girdi olarak kullanılabilir.
> Alternatif olarak gri tonlamaya çevirip tek kanal kullanmak da denenebilir
> (ablasyon fikri).

### 1.4 ⚠️ EN KRİTİK GERÇEK: Etkin örnek sayısı 1000 değil, **22**

Bu veri setindeki ~1000 spektrogram, **yalnızca 22 gerçek ses kaydından** türetilmiştir:

| Sınıf | Gerçek kaynak dosya sayısı |
|---|---|
| `fence_cutting` | **9** |
| `chain_link_climbing` | **9** |
| `metal_bending` | **4** |
| **TOPLAM** | **22** |

Her kaynak dosyadan ~26-59 varyant üretilmiştir (augmentasyon: hız/perde varyasyonu
±%15, genlik jitter ±3 dB, farklı SNR, farklı gürültü tipi, farklı pencere yerleşimi)
artı MixUp kombinasyonları.

**Aynı kaynaktan gelen varyantlar istatistiksel olarak bağımsız DEĞİLDİR.** Bu, tüm
metodolojiyi belirleyen tek en önemli kısıttır.

### 1.5 Dosya adlandırma şeması — meta veri buradan çıkarılacak

**Tek-kaynaklı augmentasyon varyantı:**
```
{kaynak}_snr{SNR}dB_{gürültü}_v{i}_spectrogram.png

Örnek: f0_snr0dB_pink_v18_spectrogram.png
       └┬┘ └─┬──┘ └┬─┘ └┬┘
        │    │     │    └─ varyant indeksi
        │    │     └────── gürültü tipi: pink | white
        │    └──────────── SNR (dB), 0-15 arası
        └───────────────── KAYNAK DOSYA ADI (= GRUP KİMLİĞİ)
```

**MixUp varyantı:**
```
mixup_{kaynak1}_x_{kaynak2}_snr{SNR}dB_{gürültü}_{k}_spectrogram.png

Örnek: mixup_f0_x_f7_snr14dB_white_92_spectrogram.png
             └┬┘   └┬┘
              │     └─ İKİNCİ ebeveyn kaynak
              └─────── BİRİNCİ ebeveyn kaynak
```

**Ayrıştırma (parsing) uyarıları:**
- Kaynak dosya adları alt çizgi içerebilir (örn. `bolt_cutter_02`). Bu yüzden
  `_snr` işaretinden bölmek güvenlidir, ilk `_` işaretinden değil.
- MixUp'ta `_x_` ayracı kullanılıyor. Eğer bir kaynak dosya adında `_x_` geçiyorsa
  ayrıştırma bozulur. **Tabloyu oluşturduktan sonra birkaç satırı gözle doğrula.**
- Önerilen yaklaşım: `mixup_` ön ekiyle başlayanları ayrı işle, kalanlarda
  `rsplit("_snr", 1)` kullan.

---

## 2. ⚠️ İKİ KRİTİK METODOLOJİK TUZAK

Bu bölüm planın kalbidir. Yanlış yapılırsa **tüm sonuçlar anlamsız olur.**

### 2.1 Tuzak #1: Rastgele train/test bölmesi

**Sorun:** `f0.wav`'ın bir varyantı eğitim setinde, başka bir varyantı test setinde
olursa, model genelleme değil **ezber** yapar. Sana %99 doğruluk gösterir ama gerçek
bir çit sesinde çuvallar.

**Çözüm:** Bölme **kaynak dosyaya göre gruplu** yapılmalı.
`f0`'ın **tüm** varyantları ya eğitimde ya testte olmalı — ikisinde birden asla.

Kullanılacak: `sklearn.model_selection.StratifiedGroupKFold`
- `groups` = kaynak dosya adı
- `y` = sınıf etiketi

### 2.2 Tuzak #2: MixUp sızıntısı (daha ince, kolayca gözden kaçar)

Gruplu bölme doğru yapılsa bile MixUp örnekleri sızıntı yaratır. `mixup_f0_x_f7_...`
dosyası **iki kaynaktan birden** bilgi taşır. `f0` test setindeyse ve bu MixUp örneği
eğitimdeyse, test kaynağının bilgisi eğitime sızmış olur.

**Zorunlu kural tablosu:**

| Örnek tipi | Karar |
|---|---|
| Tek-kaynaklı, kaynak ∈ **test** grubu | → **Test setine** |
| Tek-kaynaklı, kaynak ∉ test grubu | → Eğitime |
| MixUp, **her iki** ebeveyn de ∉ test grubu | → Eğitime |
| MixUp, **herhangi bir** ebeveyn ∈ test grubu | → **TAMAMEN ATILIR** |

**Gerekçe:** MixUp zaten bir *eğitim-zamanı* veri artırma tekniğidir. Test setinde
hiç bulunmamalıdır. Bu kural hem sızıntıyı kapatır hem metodolojik olarak doğrudur.

**Uygulama notu:** Bu, her katmanda (fold) eğitim seti boyutunun beklenenden küçük
olmasına yol açacaktır. Bu normaldir, panik yapma. Kaç örneğin atıldığını logla.

---

## 3. PROJE YAPISI (önerilen)

```
das-model-training/
├── data/
│   └── synthetic_dataset/          # wav-dataset reposundan kopyalanır
│       ├── fence_cutting/
│       ├── chain_link_climbing/
│       └── metal_bending/
├── src/
│   ├── metadata.py                 # Faz 0: dosya adı ayrıştırma → CSV
│   ├── splits.py                   # Faz 1: gruplu CV bölmeleri
│   ├── dataset.py                  # PyTorch Dataset + transform'lar
│   ├── model.py                    # Faz 2: 2D-CNN + SK-Attention
│   ├── train.py                    # Faz 3: eğitim döngüsü
│   ├── evaluate.py                 # Faz 4: metrikler, confusion matrix, t-SNE
│   └── config.py                   # Tüm hiperparametreler tek yerde
├── outputs/
│   ├── metadata.csv
│   ├── folds/
│   ├── checkpoints/
│   ├── figures/
│   └── results/
├── requirements.txt
└── README.md
```

---

## 4. FAZ 0 — Veri Hazırlığı ve Denetim

### 4.1 Görev

`synthetic_dataset/` altındaki tüm `*_spectrogram.png` dosyalarını tara ve şu sütunlara
sahip bir `outputs/metadata.csv` üret:

| Sütun | Tip | Açıklama |
|---|---|---|
| `filepath` | str | PNG'ye tam yol |
| `label` | str | `fence_cutting` / `chain_link_climbing` / `metal_bending` |
| `label_idx` | int | 0 / 1 / 2 |
| `is_mixup` | bool | MixUp örneği mi |
| `source_1` | str | Birincil kaynak dosya adı |
| `source_2` | str \| None | MixUp'ta ikinci ebeveyn, aksi halde `None` |
| `group_id` | str | Gruplu bölme için: tek-kaynaklıda `source_1`, MixUp'ta `source_1+"|"+source_2` |
| `snr_db` | int | Dosya adından çıkarılan SNR |
| `noise_kind` | str | `pink` / `white` |
| `variant_idx` | int | Varyant indeksi |

### 4.2 Zorunlu denetim çıktıları (konsola yazdır)

1. Toplam örnek sayısı, sınıf başına dağılım
2. **Benzersiz kaynak dosya sayısı — sınıf başına** (beklenen: 9 / 9 / 4)
3. Her kaynak dosyadan kaç varyant üretilmiş (min/ortalama/maks)
4. MixUp örneği sayısı ve oranı
5. SNR dağılımı (histogram özeti)
6. **Tüm PNG'lerin boyutu aynı mı?** (beklenen: hepsi 400×400)
7. Bozuk/okunamayan dosya var mı

### 4.3 Kabul kriteri

- [ ] `metadata.csv` üretildi
- [ ] Kaynak dosya sayıları 9/9/4 ile eşleşiyor (eşleşmiyorsa ayrıştırma hatalıdır — DUR ve düzelt)
- [ ] Rastgele 5 satır elle doğrulandı (dosya adı ↔ ayrıştırılmış alanlar)
- [ ] Tüm PNG'ler 400×400

---

## 5. FAZ 1 — Gruplu Çapraz Doğrulama Tasarımı

### 5.1 k seçimi — DİKKAT

`metal_bending` sınıfında **sadece 4 kaynak** var.

- `k=5` yaparsan → bazı katmanlarda test setinde **hiç metal_bending olmaz** → o katmanın
  metrikleri tanımsız olur. **KULLANMA.**
- **`k=4` (önerilen varsayılan)** → her katmanda her sınıftan en az 1 kaynak test setine düşer
- `Leave-One-Group-Out` (22 katman) → en dürüst tahmin, hesaplama maliyeti kabul edilebilir
  (model küçük). Zaman varsa **ikinci bir doğrulama olarak** çalıştır.

### 5.2 Uygulama

```
StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
  X = metadata index
  y = label_idx
  groups = source_1   ← NOT: MixUp'lar için group ataması aşağıdaki kurala göre
```

**MixUp örneklerinin ele alınması (Bölüm 2.2'deki tablo):**

Her katman için:
1. `StratifiedGroupKFold`'u **sadece tek-kaynaklı örnekler** üzerinde çalıştır
   (`is_mixup == False`), `groups = source_1`
2. Bu, o katmanın **test kaynak kümesini** (`test_sources`) belirler
3. Test seti = tek-kaynaklı örnekler, `source_1 ∈ test_sources`
4. Eğitim seti =
   - tek-kaynaklı örnekler, `source_1 ∉ test_sources`
   - **+** MixUp örnekleri, `source_1 ∉ test_sources` **VE** `source_2 ∉ test_sources`
5. Atılanlar = MixUp örnekleri, en az bir ebeveyn ∈ `test_sources`

### 5.3 Zorunlu çıktılar

Her katman için `outputs/folds/fold_{i}.json` kaydet:
- `test_sources` listesi
- eğitim/test örnek sayıları
- atılan MixUp örnek sayısı
- sınıf başına dağılım (eğitim ve test)

### 5.4 Kabul kriteri — SIZINTI TESTİ (zorunlu, otomatik)

Her katman için programatik olarak doğrula ve **başarısızsa hata fırlat**:

- [ ] `set(train_sources) ∩ set(test_sources) == ∅`
- [ ] Test setinde **hiç MixUp örneği yok**
- [ ] Eğitim setindeki hiçbir MixUp örneğinin hiçbir ebeveyni `test_sources` içinde değil
- [ ] Her katmanın test setinde her 3 sınıftan da en az 1 örnek var

> Bu testler kodda kalıcı olarak bulunmalı (assert), bir kereye mahsus manuel kontrol
> değil. Sonraki tüm sonuçların güvenilirliği buna dayanıyor.

---

## 6. FAZ 2 — Model Mimarisi

### 6.1 Genel yapı (Makale 4'ten uyarlanmış)

Makale 4'ün girdisi `(B, 1, 4000, 10)` zaman-uzay matrisiydi; bizimki spektrogram
görüntüsü. **Omurga aynı kalır, sadece girdi kanalı ve boyutu değişir.**

```
Girdi: (B, 3, 224, 224)          ← 400×400 PNG'den yeniden boyutlandırma
  │
  ├─ Conv2d(3  → 16, kernel=3×3, stride=1, padding=1) + ReLU + MaxPool2d(2×2, stride=2)
  ├─ Conv2d(16 → 32, kernel=3×3, stride=1, padding=1) + ReLU + MaxPool2d(2×2, stride=2)
  ├─ Conv2d(32 → 64, kernel=3×3, stride=1, padding=1) + ReLU + MaxPool2d(2×2, stride=2)
  │                                                    → (B, 64, 28, 28)
  ├─ SK-Attention Modülü (aşağıda detaylı)             → (B, 64, 28, 28)
  ├─ AdaptiveAvgPool2d(1)                              → (B, 64, 1, 1)
  ├─ Flatten                                           → (B, 64)
  ├─ Dropout(p=0.5)
  └─ Linear(64 → 3)                                    → (B, 3)  [logits]
```

Kayıp fonksiyonu: `CrossEntropyLoss` (softmax içinde)

**Parametre sayısı: ~50-100k.** Bu küçüklük 22 etkin örnek için **avantajdır** —
büyük bir model kesinlikle ezberlerdi.

### 6.2 SK-Attention modülü spesifikasyonu

Selective Kernel Networks (Li et al., CVPR 2019) temelli. Makale 4'ün ablasyonuyla
doğrulanmış hiperparametreler:

| Parametre | Değer | Anlamı |
|---|---|---|
| `M` | **2** | Dal sayısı |
| kernel boyutları | **3×3 ve 5×5** | Makale 4 ablasyonunda en iyi (7×7 dahil denenmiş) |
| `r` (reduction ratio) | **16** | Kanal sıkıştırma oranı (4/8/16/32 denenmiş, 16 en iyi) |
| `L` | **32** | Minimum kanal sayısı |
| `C` | 64 | Giriş kanal sayısı (omurganın son katmanı) |

**Üç aşama — Split / Fuse / Select:**

**1) SPLIT**
İki paralel dal, farklı kernel boyutlarıyla:
```
U_tilde = Conv3x3(X)   → grup konvolüsyonu + BatchNorm + ReLU
U_hat   = Conv5x5(X)   → grup konvolüsyonu + BatchNorm + ReLU
```
Her ikisi de `(B, C, H, W)` şeklinde.

> Uygulama notu: 5×5 yerine `dilation=2` ile 3×3 kullanmak yaygın ve hesaplama
> açısından daha ucuz bir eşdeğerdir. Orijinal SKNet makalesi bunu önerir.
> İlk uygulamada gerçek 5×5 kullan, ablasyon için dilated versiyonu dene.

**2) FUSE**
```
U = U_tilde + U_hat                          # eleman bazında toplama
s = GlobalAvgPool(U)                         → (B, C)      # uzamsal bilgiyi özetle
z = ReLU(BatchNorm(FC(s)))                   → (B, d)
    d = max(C/r, L) = max(64/16, 32) = 32
```

**3) SELECT**
```
a_logits = FC_A(z)   → (B, C)
b_logits = FC_B(z)   → (B, C)
[a, b] = softmax([a_logits, b_logits], dim=branch)   # kanal başına yumuşak dikkat
V = a·U_tilde + b·U_hat                              → (B, C, H, W)
```
Kısıt: her kanal için `a_c + b_c = 1` (softmax garantisi).

### 6.3 Ablasyon için gerekli varyantlar

Faz 4'te karşılaştırılacak (aynı gruplu bölmelerle):

1. **Baseline:** SK-Attention'sız düz 2D-CNN (3 konv bloğu → pool → FC)
2. **+ SE-Attention** (isteğe bağlı, Makale 4 karşılaştırması için)
3. **+ CBAM** (isteğe bağlı)
4. **+ SK-Attention** ← ana model

Makale 4'te bu sıralama %95.5 → %98.2 → %98.9 → %99.3 şeklindeydi. Bizim veri
setimizde aynı sıralamanın çıkması **beklenmez** (çok daha az veri), ama SK'nin
baseline'a göre iyileşme sağlaması beklenir.

### 6.4 ⚙️ UYGULAMA KARARLARI — planda belirtilmeyen, `src/model.py`'de verilen

> Bu bölüm Faz 2 uygulaması sırasında eklendi. Plan bu üç noktada bir değer
> vermiyordu; verilen kararlar, gerekçeleri ve **sorun çıkarsa ilk bakılacak
> yerler** aşağıda. Faz 3/4'te beklenmedik bir davranış görülürse önce bu tabloya
> dön.

| # | Karar | Değer | Gerekçe |
|---|---|---|---|
| **K1** | SK dallarında grup konvolüsyonu sayısı `G` | **32** | Plan "grup konvolüsyonu" diyor ama `G` vermiyor. Orijinal SKNet (Li ve ark., CVPR 2019) değeri; C=64 için grup başına 2 kanal. Ayrıca **zorunlu**: gruplamasız 5×5 dalı tek başına 64·64·25 = 102.400 parametre eder ve Bölüm 6.1'in ~50–100k hedefini tek başına aşardı. |
| **K2** | Omurgada BatchNorm | **YOK** | Bölüm 6.1 bloğu açıkça `Conv2d + ReLU + MaxPool2d` diye tarif ediyor; harfiyen uygulandı. BN yalnızca SK modülünün içinde (Bölüm 6.2 öyle tarif ediyor). |
| **K3** | SE ve CBAM varyantları | **Uygulandı** | Bölüm 6.3 bunları "isteğe bağlı" işaretlemiş ama ablasyon tablosunda (8.5) satırları var. Maliyeti düşük olduğu için Faz 4'te tablo eksiksiz doldurulabilsin diye eklendi. |

**Sonuç — ölçülen parametre sayıları:**

| Model | Parametre |
|---|---|
| Düz 2D-CNN (baseline) | 23.779 |
| **+ SK-Attention** | **34.723** (+10.944) |
| + SE | 24.291 |
| + CBAM | 24.389 |

⚠️ 34.723, Bölüm 6.1'in verdiği **~50–100k aralığının altındadır** — doğrudan K1'in
sonucu. Bölüm 6.1 küçüklüğü 19 etkin örnek için *avantaj* saydığından bu bir hata
değildir, ama **rapora not düşülmelidir**.

#### Sorun giderme — belirti → şüpheli karar → çözüm

| Belirti | Şüpheli | Denenecek çözüm |
|---|---|---|
| Eğitim kaybı zıplıyor, yakınsamıyor veya NaN veriyor | **K2** | Omurgaya `BatchNorm2d` ekle (her `Conv2d` sonrası, `ReLU` öncesi). BN'siz bir CNN'i Adam + `lr=1e-3` ile sıfırdan eğitmek kararsız olabilir — **ilk denenecek ablasyon budur**. |
| Model yeterince öğrenmiyor (eğitim doğruluğu da düşük — underfitting) | **K1** | `config.SK_GROUPS` değerini 16 veya 8'e düşür. Kapasite ve parametre sayısı artar, plandaki ~50–100k aralığına yaklaşılır. |
| SK, baseline'a göre iyileşme sağlamıyor | **K1** | Aynı şekilde `G`'yi düşür; grup konvolüsyonu kanallar arası karışımı sınırlıyor olabilir. Ayrıca Bölüm 6.2'deki `dilation=2` alternatifini dene. |
| Eğitim çok yavaş | **K3** | SE/CBAM ablasyon koşularını atla; ana karşılaştırma baseline vs SK'dir. |

---

## 7. FAZ 3 — Eğitim

### 7.1 Girdi ön işleme

```
1. PNG yükle (RGB, 400×400)
2. Resize → 224×224   (bilinear)
3. ToTensor → [0, 1]
4. Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   ← ImageNet istatistikleri; sıfırdan eğitimde kendi veri setinin
     ortalamasını hesaplamak da meşru bir alternatif (ablasyon fikri)
```

### 7.2 Eğitim-zamanı veri artırma (aşırı öğrenmeye karşı — ZORUNLU)

22 etkin örnekle bu olmazsa olmaz. SpecAugment ruhunda, spektrograma anlamlı olanlar:

| Teknik | Parametre | Gerekçe |
|---|---|---|
| **Zaman maskeleme** | Rastgele 1-2 dikey şerit, genişlik ≤ %10 | Olayın bir kısmı görünmese de sınıf değişmez |
| **Frekans maskeleme** | Rastgele 1-2 yatay şerit, yükseklik ≤ %10 | Bant kaybına dayanıklılık |
| **Rastgele kırpma** | `RandomResizedCrop(224, scale=(0.85, 1.0))` | Konum kaymasına dayanıklılık |
| **Parlaklık/kontrast jitter** | ±%10 | SNR varyasyonuna dayanıklılık |

**YAPMA:** Yatay çevirme (horizontal flip) — spektrogramda zaman eksenini ters çevirir,
fiziksel olarak anlamsızdır. Dikey çevirme — frekans eksenini ters çevirir, aynı şekilde
anlamsızdır.

### 7.3 Hiperparametreler

| Parametre | Değer | Not |
|---|---|---|
| Optimizer | Adam | Makale 4 de Adam kullanıyor |
| Öğrenme oranı | `1e-3` | |
| LR zamanlayıcı | `ReduceLROnPlateau(patience=5, factor=0.5)` | val_loss izler |
| Batch boyutu | 32 | Küçük veri; 16 da denenebilir |
| Maks. epoch | 100 | |
| **Erken durdurma** | `patience=10` | Makale 4 ile aynı: 10 epoch boyunca iyileşme yoksa dur |
| Dropout | 0.5 | |
| Weight decay | `1e-4` | Ek düzenlileştirme |
| Kayıp | CrossEntropyLoss | Sınıflar dengeli, ağırlıklandırma gerekmez |

### 7.4 Doğrulama seti

Her katmanın **eğitim** setinden, yine **gruplu** olarak %15 ayır (erken durdurma ve
LR zamanlaması için). Bu iç bölme de kaynak-gruplu olmalı — aynı sızıntı riski burada
da geçerli.

### 7.5 Eğitim döngüsü kuralları

- **Her katman için sıfırdan yeni model.** Ağırlıkları katmanlar arası taşıma.
- Her katmanın en iyi checkpoint'ini kaydet: `outputs/checkpoints/fold_{i}_best.pt`
- Rastgelelik tohumu sabitle (`torch.manual_seed`, `np.random.seed`), ama **her katman
  için farklı tohum kullanma** — katman varyansı gerçek olsun.
- Eğitim/doğrulama kayıp ve doğruluk eğrilerini kaydet (`outputs/figures/fold_{i}_curves.png`)

### 7.6 Kabul kriteri

- [ ] Tüm katmanlar hatasız tamamlandı
- [ ] Her katmanın kayıp eğrisi kaydedildi
- [ ] **Eğitim ve doğrulama eğrileri ciddi şekilde ayrışmıyor** — ayrışıyorsa aşırı
      öğrenme var, veri artırmayı güçlendir veya modeli küçült

---

## 8. FAZ 4 — Değerlendirme

### 8.1 Birincil metrik: macro-F1

**Doğruluk (accuracy) tek başına raporlanmayacak.** Gerekçe — projedeki literatür
analizinden: Makale 2'de (Rahman 2024) KNN modeli %98 doğruluk elde etmiş ama
F1 = 0.31 çıkmış. Dengesiz/az veride doğruluk ciddi şekilde yanıltıcıdır.

Raporlanacak metrikler:
1. **macro-F1** (birincil)
2. Sınıf bazında precision / recall / F1
3. Genel doğruluk (ikincil)
4. Karışıklık matrisi (tüm katmanlar toplanmış)

### 8.2 ⚠️ Katmanlar arası varyans — mutlaka raporla

`metal_bending`'de test setine **tek bir kaynak** düşeceği için sonuçlar katmanlar
arasında **yüksek varyans** gösterecektir.

**Ortalamayı tek başına raporlama.** Her metriği `ortalama ± standart sapma` şeklinde
yaz. Makale 4 de 5-katlı CV'de bunu yapıyor.

### 8.3 SNR'ye göre kırılım analizi

Dosya adlarında SNR bilgisi zaten mevcut. Test sonuçlarını SNR kovalarına ayır:

| Kova | Aralık |
|---|---|
| Düşük | 0-5 dB |
| Orta | 5-10 dB |
| Yüksek | 10-15 dB |

Her kova için macro-F1 hesapla. Bu, Makale 4'ün gürültü dayanıklılık analizinin
doğrudan karşılığıdır ve rapor için güçlü bir bölüm olur.

### 8.4 t-SNE görselleştirmesi

Makale 4'teki gibi (Fig. 7). Son katmandaki (AdaptiveAvgPool sonrası, 64 boyutlu)
özellik vektörlerini çıkar, t-SNE ile 2 boyuta indir, sınıflara göre renklendir.

**İki versiyon üret ve yan yana koy:**
- SK-Attention'sız baseline modelin özellikleri
- SK-Attention'lı modelin özellikleri

Beklenen: SK'li versiyonda sınıflar daha ayrık kümeler oluşturur.

### 8.5 Ablasyon tablosu

| Model | macro-F1 (ort ± std) | Doğruluk | Parametre | Çıkarım süresi |
|---|---|---|---|---|
| Düz 2D-CNN (SK'siz) | | | | |
| 2D-CNN + SK-Attention | | | | |

İsteğe bağlı olarak SE ve CBAM satırları da eklenebilir.

### 8.6 Kabul kriteri

- [ ] Tüm metrikler `ortalama ± std` formatında
- [ ] Karışıklık matrisi üretildi
- [ ] SNR kırılımı üretildi
- [ ] t-SNE görselleri üretildi (baseline vs SK)
- [ ] Ablasyon tablosu dolduruldu

---

## 9. FAZ 5 — Raporlama

Sorumluya sunarken **mutlaka** belirtilecek üç nokta:

### 9.1 Metodolojik şeffaflık (en önemlisi)

> "Train/test bölmesi **kaynak dosyaya göre gruplu** yapıldı, rastgele değil.
> Çünkü veri setindeki ~1000 örnek yalnızca 22 gerçek kayıttan türetilmiştir;
> rastgele bölme, aynı kaydın varyantlarının hem eğitimde hem testte bulunmasına
> yol açar ve yapay olarak yüksek (ama anlamsız) doğruluk üretir.
> Ayrıca MixUp örnekleri, ebeveynlerinden herhangi biri test setindeyse tamamen
> dışlanmıştır."

### 9.2 Etkin örnek sayısı uyarısı

> "Mutlak doğruluk rakamları temkinli yorumlanmalıdır: etkin bağımsız örnek sayısı
> 1000 değil 22'dir. Bu sonuçlar mimarinin çalıştığını gösterir, saha performansını
> garanti etmez."

### 9.3 Katmanlar arası varyans

> "Özellikle `metal_bending` sınıfında yalnızca 4 kaynak kayıt bulunduğundan,
> katmanlar arası standart sapma yüksektir ve raporlanmıştır."

---

## 10. BİLİNEN EKSİKLER VE SONRAKİ ADIMLAR

Bu görevin kapsamı dışında ama **rapora not düşülmeli**:

### 10.1 "Normal / olay yok" sınıfı eksik

Veri setindeki üç sınıfın **üçü de tehdittir**. Bu model *"hangi tehdit?"* sorusunu
cevaplar ama *"tehdit var mı?"* sorusunu cevaplayamaz — gerçek bir PIDS'in en temel
sorusu budur.

Karşılaştırma: Makale 4'te `Quiet` sınıfı, Makale 3'te (Tomasov) `regular` sınıfı vardı.

**Sonraki adım:** Sessizlik / rüzgâr / arka plan gürültüsü sınıfı eklenmeli.

### 10.2 NAR (yanlış alarm oranı) hesaplanamıyor

"Normal" sınıfı olmadığı için "yanlış alarm" tanımlanamaz. Makale 4 ile tam
hizalanmak (NAR %0.7 karşılaştırması) için o sınıf gereklidir.

### 10.3 Rüzgâr sınıfı hiçbir açık veri setinde yok

Çit PIDS'de en kritik yanlış alarm kaynağıdır. Kendi saha verimizle toplanması gerekir.

### 10.4 Diğer açık konular

- Fiber kurulum yöntemi (gömülü vs çite monteli) henüz kararlaştırılmadı
- Sentetik veri gerçek DAS verisinin yerini tutmaz; saha verisiyle fine-tuning şart
- Açık küme tanıma (open-set recognition): beklenmedik olay tiplerini reddetme
  mekanizması ileride eklenmeli (Makale 4'ün sonuç bölümünde önerilmiş)

---

## 11. BAĞIMLILIKLAR

```
torch>=2.0
torchvision
scikit-learn>=1.3        # StratifiedGroupKFold için
numpy
pandas
pillow
matplotlib
seaborn                  # confusion matrix görselleştirme
tqdm
```

> `StratifiedGroupKFold`, scikit-learn 0.24+ ile geldi. 1.3+ önerilir.

---

## 12. UYGULAMA SIRASI (Claude Code için adım adım)

1. **Faz 0** → `src/metadata.py` yaz, çalıştır, `outputs/metadata.csv` üret,
   denetim çıktılarını incele. **Kaynak sayıları 9/9/4 değilse DUR ve ayrıştırmayı düzelt.**
2. **Faz 1** → `src/splits.py` yaz, gruplu bölmeleri üret, **sızıntı testlerini
   assert olarak yaz ve çalıştır.** Hepsi geçmeden ilerleme.
3. **Faz 2** → `src/model.py` yaz. SK-Attention modülünü izole olarak test et:
   rastgele bir `(2, 64, 28, 28)` tensörü ver, çıktının aynı şekilde olduğunu
   ve softmax kısıtının (`a+b=1`) sağlandığını doğrula.
4. **Faz 3** → `src/train.py` yaz. **Önce tek bir katmanı** çalıştır, kayıp eğrisini
   incele, aşırı öğrenme var mı bak. Sonra tüm katmanları çalıştır.
5. **Faz 4** → `src/evaluate.py` yaz, tüm metrikleri ve görselleri üret.
6. **Ablasyon** → SK'siz baseline'ı aynı bölmelerle eğit, karşılaştır.
7. **Faz 5** → Sonuçları README'ye yaz, şeffaflık notlarını ekle.

---

## 13. HIZLI REFERANS — Kritik Sayılar

| | |
|---|---|
| Sınıf sayısı | 3 |
| Toplam örnek | ~1000 |
| **Etkin bağımsız örnek** | **22** (9 / 9 / 4) |
| Spektrogram boyutu | 400×400 → 224×224'e resize |
| Model girdisi | (B, 3, 224, 224) |
| CV katman sayısı | **4** (StratifiedGroupKFold) |
| SK parametreleri | M=2, kernel 3×3 & 5×5, r=16, L=32 |
| Konv kanalları | 3 → 16 → 32 → 64 |
| Birincil metrik | **macro-F1** (ortalama ± std) |
| Erken durdurma sabrı | 10 epoch |
