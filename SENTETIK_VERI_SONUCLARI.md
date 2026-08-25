# Sentetik Veri Aşaması — Tam Kayıt

> Aşama 1'in (sentetik spektrogramlarla model geliştirme) eksiksiz kaydı.
> Tüm sayılar doğrudan ölçümdür. Kod yorumlarında dağınık duran bulgular
> burada toplanmıştır.
>
> **Durum: TAMAMLANDI.** Çıktı: `outputs/pretrained/das_2dcnn_sk_v1.pt`
> **Sonuç: macro-F1 0.622 ± 0.166**

---

## 1. VERİ SETİ

Kendi ürettiğimiz veri (`synth_das_pipeline.py`). Ham ses kayıtlarından
sentetik DAS spektrogramları üretiliyor.

### Boru hattı

```
Ham .wav (Freesound)
  -> augmentasyon: speed_change ±%15, genlik jitter ±3 dB
  -> 2000 Hz'e indirgeme
  -> 10 saniyelik pencereye yerlestirme
  -> rastgele SNR (0-15 dB) pink/white gurultu
  -> STFT -> 400x400 viridis PNG
  -> ayrica MixUp: ayni siniftan iki kaydin karisimi
```

### Denetim sonrası son durum

| | Başlangıç | Temizlik sonrası |
|---|---|---|
| Spektrogram | 1.000 | **959** |
| chain_link_climbing | 334 | 334 |
| fence_cutting | 334 | **293** |
| metal_bending | 332 | 332 |
| Tek-kaynaklı | 700 | **674** |
| MixUp | 300 (%30) | **285 (%29.7)** |
| **Bağımsız kaynak** | **22** (9/9/4) | **19** (7/8/4) |

Diğer denetim sonuçları:

- Kaynak başına varyant: `chain_link_climbing` ve `fence_cutting` **26**,
  `metal_bending` **58**
- SNR 0–15 dB düzgün dağılmış — kovalar: düşük 319, orta 307, yüksek 333
- 73 benzersiz MixUp çifti
- Tüm PNG'ler **400×400**, bozuk dosya **yok**
- 959/959 satırda dosya adı ↔ ayrıştırılmış alanlar tersine doğrulaması geçti

---

## 2. FAZ 0 ADLİ BULGULARI

Plan 22 bağımsız kayıt varsayıyordu. Ölçüm **19** olduğunu gösterdi.

### 2.1 Birebir kopya dosya

```
345070__metrostock99__chain-lock-on-fence-sound.wav
345070__metrostock99__chain-lock-on-fence-sound (1).wav
```

| Kanıt | Değer |
|---|---|
| MD5 | **aynı** (`23b801e52200`) |
| Korelasyon | **1.000** |

Tarayıcının kopya-indirme soneki. İki ayrı kaynak sayılsaydı
`StratifiedGroupKFold` birini eğitime birini teste koyabilir ve **birebir aynı
kayıt** hem eğitimde hem testte bulunurdu.

**Silindi:** 1 ham + 82 türev (41 PNG + 41 WAV) = **83 dosya**.
Türevler: 26 tek-kaynaklı + 15 MixUp varyantı.

### 2.2 Eşzamanlı mikrofon çiftleri

```
189219__starvolt__fence-rattle-1-rear   <-> 189220__...-1-front
189223__starvolt__fence-rattle-2-rear   <-> 189224__...-2-front
```

Aynı çit-sallanma olayının çitin iki tarafından **eşzamanlı** kaydı.

| Ölçüt | rattle-1 | rattle-2 | Kontrol ort. | Kontrol maks. |
|---|---|---|---|---|
| **Zarf korelasyonu** | **0.969** | **0.903** | 0.079 | 0.744 |
| **Log-spektrogram kor.** | **0.942** | **0.888** | 0.075 | 0.858 |
| En-iyi-gecikmeli dalga kor. | 0.635 | 0.636 | 0.041 | 0.226 |
| Onset örtüşmesi | %74 | %86 | %29 | %100 |
| **Optimum gecikme** | **0.0 ms** | **0.0 ms** | ±250 ms rastgele | — |

Kontrol grubu = aynı sınıftaki 70 ilgisiz çift.

**Gecikmenin tam 0.0 ms olması belirleyici:** iki bağımsız çekim asla örnek
düzeyinde hizalı olmaz. Bu, tek bir kayıt cihazının iki kanalı.

**Silinmedi, birleştirildi** — ortak `group_id` verildi. Tüm varyantlar
eğitimde kalır ama asla eğitim/test arasında bölünmez.

### 2.3 ⚠️ Sınıf tutarsızlığı — performans tavanının asıl sebebi

Sınıfların kendi içinde ne kadar tutarlı olduğu (log-spektrogram korelasyonu):

| Sınıf | Sınıf içi ort. | **Medyan** | Yorum |
|---|---|---|---|
| `metal_bending` | 0.647 | **0.743** | Sıkı, tutarlı küme |
| `fence_cutting` | 0.065 | 0.104 | Dağınık |
| `chain_link_climbing` | 0.034 | **−0.250** | **Küme değil** |

Sınıflar arası: chain↔fence 0.005 · chain↔metal 0.025 · fence↔metal 0.069

**`chain_link_climbing` üç alakasız kayıt kümesinden oluşuyor:**
`hupguy` (1) · `starvolt` (4) · `department64` (4). Ortak akustik imzaları yok.

**Ve `department64` kayıtlarının dördü de kendi sınıf arkadaşlarından çok
`metal_bending`'e benziyor:**

| Kayıt | Kendi sınıfına | `metal_bending`'e | Karar |
|---|---|---|---|
| `chainlinkfence_f04` | 0.023 | **0.351** | → metal |
| `chainlinkfence_f03` | 0.029 | **0.338** | → metal |
| `chainlinkfence_f01` | 0.031 | **0.128** | → metal |
| `chainlinkfence_f02` | −0.113 | −0.052 | → metal |

Sebebi kaynak adlarında görünüyor: `metal_bending`'in 4 kaynağından **3'ü**
adı düpedüz `rbh_chain-link-fence-01/02/03` olan kayıtlar.

Ayrıca `large-bolt-cutter` (fence_cutting), `metal_bending`'in dört kaynağına
da **0.585–0.668** benzerlikte — sınıf içi benzerliklerden bile yüksek.

> **Bu, hiperparametre ayarıyla çözülemeyecek bir veri sorunudur.** Üç
> konfigürasyon denendi, ana metrik 0.614–0.628 aralığında kaldı.

---

## 3. FAZ 1 — GRUPLU ÇAPRAZ DOĞRULAMA

`StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)`,
yalnızca tek-kaynaklı örnekler üzerinde, `groups = group_1`.

**k=5 kullanılamaz:** `metal_bending`'de yalnızca 4 grup var, bazı katmanların
test setinde hiç `metal_bending` olmazdı.

### Katman kompozisyonu (nihai)

| Katman | Test grubu | Eğitim | Doğrulama | Test | Atılan MixUp |
|---|---|---|---|---|---|
| 0 | 4 | 499 | 110 | 162 | 188 |
| 1 | 4 | 496 | 110 | 162 | 191 |
| 2 | **6** | 465 | 110 | **188** | 196 |
| 3 | 5 | 503 | 110 | 162 | 184 |

Katmanlar arası grup sayısı eşit değil — `StratifiedGroupKFold` **örnek**
sayısına göre dengeliyor, grup sayısına göre değil (starvolt grupları 52,
`metal_bending` grupları 58 örnek taşıyor).

### Eğitimdeki grup sayısı (sınıf başına)

```
katman 0: chain=5  fence=5  metal=2
katman 1: chain=5  fence=5  metal=2
katman 2: chain=3  fence=5  metal=2    <- en dar
katman 3: chain=4  fence=5  metal=2
```

**`metal_bending` her katmanda yalnızca 2 kaynaktan öğreniyor** — 4 grubun
1'i teste, 1'i doğrulamaya gidiyor. Kaçınılmaz.

**Katman 2 her konfigürasyonda en kötü sonucu verdi** (0.463 / 0.306 / 0.416)
— en dar eğitim setine sahip olan o.

### MixUp eleme kuralı

MixUp örneği iki kaynaktan bilgi taşır. Kural: **her iki ebeveyni de eğitim
grubundaysa** kullanılır, değilse tamamen atılır.

Bu kural doğrulama setine de uygulandı (planda yoktu, karar verildi):
sızıntılı bir doğrulama sinyali erken durdurmayı iyimser bir epoch'ta
durdurur ve model seçimini bozar.

### Doğrulama grubu optimizasyonu

Doğrulama grupları rastgele değil, **hayatta kalan MixUp'ı maksimize edecek**
şekilde seçiliyor (sınıf başına tüm kombinasyonlar deneniyor, 72–108 adet):

| Katman | Rastgele | Optimize | En kötü seçim |
|---|---|---|---|
| 0 | 85 | **97** | 64 |
| 1 | 87 | **94** | 66 |
| 2 | 67 | **89** | 63 |
| 3 | 71 | **101** | 61 |
| **Toplam** | 310 (%27.2) | **381 (%33.4)** | |

Yan fayda: doğrulama oranı katmanlar arasında eşitlendi (%17.9–19.1, önce
%18.4–24.6) — büyük starvolt grupları doğrulamaya düşmediği için.

### Kalıcı sızıntı testleri

Kodda `check()` → `LeakageError` olarak 7 test var. **Bilerek `assert`
kullanılmadı** — `python -O` altında assert'ler tamamen kaldırılır.

1. `set(train_groups) ∩ set(test_groups) == ∅`
2. Test setinde hiç MixUp yok
3. Eğitimdeki MixUp'ların hiçbir ebeveyni test gruplarında değil
4. Her katmanın test setinde 3 sınıftan da ≥1 örnek
5. Aynı kurallar doğrulama seti için
6. Bölmeler ayrık ve tüm veriyi kapsıyor (959 = 959)
7. Her grup tam olarak bir katmanda test edildi (674 = 674)

---

## 4. FAZ 2 — MODEL DOĞRULAMA

```
(B,3,224,320) -> (B,16,112,160) -> (B,32,56,80) -> (B,64,28,40)
              -> SK-Attention -> (B,64) -> Dropout(0.5) -> (B,3)
```

### Parametre dağılımı

| Katman | Hesap | Parametre |
|---|---|---|
| `Conv(3→16)` | 16 × (3·3·3 + 1) | 448 |
| `Conv(16→32)` | 32 × (3·3·16 + 1) | 4.640 |
| `Conv(32→64)` | 64 × (3·3·32 + 1) | 18.496 |
| `Linear(64→3)` | 64·3 + 3 | 195 |
| **SK modülü** | | **+10.944** |

| Varyant | BN öncesi | BN sonrası (Paket 2) |
|---|---|---|
| Baseline | 23.779 | 23.891 |
| **+ SK** | **34.723** | **34.835** |
| + SE | 24.291 | — |
| + CBAM | 24.389 | — |

BN artışı +112: BatchNorm 224 parametre ekliyor, kaldırılan conv bias'ları
112 çıkarıyor.

### Birim testi sonuçları

| Test | Sonuç |
|---|---|
| SK çıktı şekli girdiyle aynı | ✅ `(2,64,28,28)` → `(2,64,28,28)` |
| Softmax kısıtı `a_c + b_c = 1` | ✅ maks sapma **5.96e-08** |
| Ağırlıklar `[0,1]` aralığında | ✅ |
| Ağırlıklar kanal başına farklılaşıyor | ✅ std **0.052** — sabit 0.5/0.5'e çökmemiş |
| Geri yayılım tüm parametrelere ulaşıyor | ✅ 21/21 (BN sonrası 24/24) |
| `state_dict` toplam | 35.385 (34.835 parametre + **550 BN tamponu**) |

BN tamponları (`running_mean`, `running_var`, `num_batches_tracked`)
öğrenilen ağırlık değil, **hatırlanan istatistik** — ama kaydedilmeleri gerek.

---

## 5. FAZ 3 — EĞİTİM SONUÇLARI

Detaylı sonuçlar: **`MODEL_IYILESTIRME_PLANI.md`** → "ÖLÇÜLEN SONUÇLAR"

Özet:

| Konfigürasyon | SK | Baseline | Fark | t |
|---|---|---|---|---|
| Paket 0 (`val_loss`) | 0.628 ± 0.122 | 0.442 ± 0.216 | +0.186 | *geçersiz* |
| Paket 1 (`val_macro_f1`) | 0.614 ± 0.204 | 0.548 ± 0.112 | +0.066 | 0.63 |
| **Paket 2** (+BN, LS, 224×320) | **0.622 ± 0.166** | 0.580 ± 0.135 | +0.042 | 1.26 |

**SK'nin üstünlüğü istatistiksel olarak gösterilemedi** (anlamlılık için
t ≈ 3.18 gerekirdi). Üç konfigürasyonda da önde ama fark gürültünün içinde.

### ⚠️ Doğrulama seti güvenilmez — önemli metodolojik bulgu

Seçilen epoch'taki doğrulama F1 ile test F1 karşılaştırması (Paket 1, SK):

| Katman | Doğrulama F1 | Test F1 | Fark |
|---|---|---|---|
| 0 | **1.000** | 0.572 | −0.43 |
| 1 | **0.537** | **0.787** | **+0.25** |
| 2 | 0.577 | 0.463 | −0.11 |
| 3 | 0.963 | 0.688 | −0.28 |

Sıralamalar neredeyse **ters**. Korelasyon **+0.088** (Paket 2'de +0.597'ye
çıktı ama 4 noktayla hesaplandığı için kanıt değil).

**Sebebi yapısal:** doğrulama setinde sınıf başına **1 kayıt** var. Ölçtüğü
şey "model genelliyor mu" değil, "bu 3 kayıt eğitimdekilere benziyor mu".
Metrik değiştirmek çözmedi, düzenlileştirme de çözmedi.

---

## 6. HATA ANALİZİ

4 katmanın test tahminleri havuzlanmış (674 örnek, Paket 0):

```
                     precision  recall  f1-score  support
chain_link_climbing      0.758   0.521     0.618      234
fence_cutting            0.665   0.803     0.728      208
metal_bending            0.634   0.716     0.672      232
                                 macro avg 0.672
```

Karışıklık matrisi (satır = gerçek):

```
                     chain   fence   metal
chain_link_climbing    122      24   →  88     %38'i metal saniliyor
fence_cutting           33     167       8
metal_bending            6   →  60     166     %26'si fence saniliyor
```

**Baskın hata: `chain_link_climbing` → `metal_bending` (88/234 = %38).**
Bölüm 2.3'teki akustik analiz bunu birebir açıklıyor.

İkincil hata `metal_bending` → `fence_cutting` (60/232) de açıklanıyor:
`large-bolt-cutter` kaydı `metal_bending` kaynaklarına 0.585–0.668 benzerlikte.

> **İki farklı macro-F1 var, karıştırma:**
> **0.628 ± 0.122** = her katmanda ayrı hesapla, sonra ortala (raporlanan)
> **0.672** = 674 tahmini tek havuzda topla (oynaklığı gizler)

---

## 7. NEDEN 0.62'DE TIKANDI

| Sebep | Kanıt |
|---|---|
| **19 bağımsız kayıt** | Sentetik boru hattı 1000 örnek üretti ama 19 kaynaktan |
| **Sınıf tutarsızlığı** | `chain_link_climbing` küme değil (medyan −0.250) |
| **`metal_bending` 4 kaynak** | Her katmanda eğitimde yalnızca 2 grup kalıyor |
| **Doğrulama güvenilmez** | Sınıf başına 1 kayıt, testle korelasyon ~0 |

Üç konfigürasyon denendi (0.614 / 0.622 / 0.628), farklar ±0.17 standart
sapmanın altında — **hiçbiri gerçek bir değişim değil.**

**Karar: hiperparametre ayarı bırakıldı.** Dördüncü konfigürasyon kazanç
getirmezdi ve test setini kirletmeye başlardı.

---

## 8. YAPILMAYANLAR

Plan Bölüm 8 ve 9'daki bazı işler tamamlanmadı — gerçek veriye geçildiği için.

- [ ] `src/evaluate.py` (Faz 4) — SNR kırılımı, t-SNE (SK'li vs SK'siz özellik
      uzayı), ablasyon tablosu, çıkarım süresi
- [ ] Paket 3 — TTA + veri setinin kendi normalizasyon istatistikleri
- [ ] SE / CBAM ablasyon koşuları (kod hazır, `--attention se|cbam`)
- [ ] Leave-One-Group-Out (19 katman) — istatistiksel gücü ~2 katına çıkarırdı;
      Paket 2'deki t=1.26, 19 katmanla ~2.7'ye gelirdi
- [ ] 2 sınıflı teşhis koşusu — `chain_link_climbing` + `metal_bending`
      birleştirilip veri tavanının ne kadara mal olduğunu ölçmek

**Not:** Bunların hiçbiri önceden eğitilmiş modeli etkilemiyor. Model paketi
üretildi, doğrulandı ve teslim edildi.

---

## 9. KULLANILAN KOD

| Dosya | Faz | Ne yapar |
|---|---|---|
| `src/config.py` | — | Tüm yollar ve hiperparametreler |
| `src/metadata.py` | 0 | Dosya adı ayrıştırma → `outputs/metadata.csv` + 7 denetim |
| `src/splits.py` | 1 | Gruplu CV bölmeleri + 7 kalıcı sızıntı testi |
| `src/model.py` | 2 | `DASNet` + SK/SE/CBAM + `load_pretrained()` |
| `src/dataset.py` | 3 | uint8 önbellek, artırma, `SpecMasking` |
| `src/train.py` | 3 | Eğitim döngüsü, determinizm, checkpoint |
| `src/export_model.py` | — | Nihai model paketi + `--verify` modu |
| `colab_train.ipynb` | 3 | Colab eğitim akışı |

Çalıştırma sırası:

```bash
python src/metadata.py                        # Faz 0
python src/splits.py                          # Faz 1
python src/model.py                           # Faz 2 birim testi
python src/dataset.py                         # veri yukleyici testi
python src/train.py --fold all --attention sk # Faz 3
python src/train.py --fold all --attention none
python src/export_model.py --attention sk     # nihai paket
python src/export_model.py --verify outputs/pretrained/das_2dcnn_sk_v1.pt
```
