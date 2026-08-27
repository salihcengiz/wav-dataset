# Gerçek Saha Verisi — Faz 0 Denetim Raporu

> `/tf/start_training/RELATIONNET/FENCE_DATA_NEW/` veri seti üzerinde yapılan
> ölçümler. Sentetik veri setinde uyguladığımız Faz 0 denetiminin gerçek veri
> karşılığı. Tüm sayılar doğrudan ölçümdür, tahmin değildir.

---

## 1. ÖZET

| | |
|---|---|
| Toplam pencere | 373.908 (train 293.469 / val 37.559 / test 42.880) |
| Benzersiz kayıt dosyası | 28.154 |
| **Benzersiz oturum** | **14.490** (train 11.175 / val 1.650 / test 1.665) |
| Sınıflar | `climbing`, `cutting`, `noise` |
| Örnekleme frekansı | **2000 Hz** (doğrulandı) |
| Bölme kalitesi | ✅ **temiz** — dosya/oturum/tarih çakışması sıfır |

**Sentetik veriye göre ölçek 600 kat arttı:** orada 19 bağımsız kayıt vardı,
burada 14.490 oturum.

### Üç ana bulgu

1. ✅ **Bölmeler grup-temiz.** Sentetik veride tüm Faz 1'i bunu sağlamak için
   yazmıştık; burada hazır ve doğru geliyor.
2. ✅ **"noise" sınıfı var.** Sentetik verinin en büyük eksiğiydi. Artık
   *"tehdit var mı?"* sorusu cevaplanabilir ve yanlış alarm oranı ölçülebilir.
3. ❌ **`climbing` ile `cutting` test edilen hiçbir özellikle ayrılmıyor.**

---

## 1.5 VERİLEN KARARLAR

Denetim sonrası, ölçümlere dayanarak alınan kararlar. Ön işleme hattı bunlara
göre kurulacak.

| Konu | Karar | Gerekçe |
|---|---|---|
| **Dosya biçimi** | **`.bin.hdf5`** | val/test'in %100'ü, train'in %97.8'i, ana deponun ezici çoğunluğu. `.sdf` yalnızca bir kampanyada |
| **Pencere** | **15.000 örnek = 7.5 s @ 2000 Hz** | `support_set_creator.py` ile aynı; CSV'deki en yaygın uzunluk; frekans düşürülmediği için bant kaybı yok |
| **Sinyal** | `hypot(re, im)` — **yalnızca `P`** | Ekibin yöntemi. Faz kullanılamaz (ölçüldü: std 1.814 = rastgele) |
| **Standartlaştırma** | uzunsa enerji merkezine kırp, kısaysa yansıtmalı doldur | Ekibin kuralı; kendi kuralımızı uydurmuyoruz |
| **Boş pencereler** | **Yükleme sırasında ayıklanacak** | Olay pencerelerinin ~%25'i spektral olarak boş |
| **Normalizasyon** | **Pencere-içi (zorunlu)** | Kanaldan kanala genlik 36 kat değişiyor; ölçek katsayısı sorununu da çözer |
| **Ölçek katsayısı** | Uygulanmayacak — **gereksiz** | Kendi hattımızda eğitim ve çıkarım aynı fonksiyonu kullanacak; pencere-içi normalizasyon sabit katsayıyı zaten sadeleştirir |

### Pencere boyutu neden 7.5 s (10 s değil)

Üçü birbirine bağlı: `örnek = süre × frekans`

| Seçenek | Örnek | Süre | Frekans | Bedeli |
|---|---|---|---|---|
| A ✅ | 15.000 | **7.5 s** | 2000 Hz | — |
| B | 20.000 | 10 s | 2000 Hz | Örnek 15.000 değil |
| C | 15.000 | 10 s | **1500 Hz** | **750–1000 Hz bandı kaybedilir** (enerjinin ~%22'si orada) |

Seçenek A alındı: frekans düşürülmediği için hiçbir bant kaybedilmiyor.

> **Aktarıma etkisi yok.** Spektrogram 224×320'ye ölçekleniyor; 7.5 s ile 10 s
> arasındaki tek fark piksel başına düşen süre (23 ms vs 31 ms).

### `P` / `S` terminolojisi — teyit edilmeli

Dosyanın kök öznitelikleri `polarization: 2`, `port: 1` diyor ve `P`/`S`
optikte p- ve s-polarize ışığı ifade eder. Sorumlu bunları **port** olarak
tanımladı. Rapor metninde hangisinin kullanılacağı teyit edilmeli.

**Eylem değişmiyor: `P` kullanılacak.**

---

## 2. VERİ YAPISI

### 2.1 CSV = indeks, HDF5 = veri

CSV sütunları: `file, channel, event, window_start, window_end`

Her satır bir **(kayıt, fiber kanalı, zaman penceresi)** üçlüsü. `window_*`
değerleri örnek indeksidir (saniye değil).

Etiketler HDF5 dosyalarının içinde `labels` veri kümesinde de gömülü:
`[window_start, window_end, channel_start, channel_end, ?, ?]`

- `.sdf` dosyalarında CSV satırları `labels` ile **birebir aynı**
- `.bin` dosyalarında CSV, etiketi saran **sabit uzunlukta** bir pencere üretmiş

### 2.2 İki farklı dosya biçimi

| | `.bin.hdf5` | `.sdf.hdf5` |
|---|---|---|
| train payı | %97.8 (286.941) | %2.2 (6.528) |
| val / test payı | **%100** | **%0** |
| Veri tipi | `[('P', [('re','<i2'),('im','<i2')]), ('S', ...)]` | `[('P', '<f4')]` |
| İçerik | **ham karmaşık I/Q**, çift polarizasyon | işlenmiş tek gerçek değer |
| polarization | 2 | 1 |
| resolution | 4.09 m | 10.0 m |
| Zamanlama | ✅ tutarlı | ⚠️ 2 kat uyuşmazlık (bkz. 6.2) |

**Karar: `.bin.hdf5` ile çalışılmalı.** Val/test'in tamamı bu biçimde;
`.sdf` için ölçebileceğimiz bir karşılık yok.

### 2.3 Dosya adı meta verisi

```
/tf/segment/2020.06.22/record_SA4_CIT_TIRMANMA_31_202006220826_30min_raw.bin.hdf5
             └─ tarih ┘        └SA4┘└─ TIRMANMA ─┘└31┘└─ 202006220826 ─┘
                                saha    olay tipi  no      zaman damgası
```

`30min` **orijinal** kaydın süresidir; dosyanın kendisi ondan kesilmiş bir
segmenttir (ölçülen: 20.3 saniye).

---

## 3. BÖLME KALİTESİ ✅

```
                dosya    oturum    tarih
train ∩ val  :      0         0        0
test  ∩ train:      0         0        0
test  ∩ val  :      0         0        0
```

Üç düzeyde de çakışma yok. **Bölmeleri kuran kişi doğru yapmış.**

### Sınıf ve saha dağılımı

| | climbing | cutting | noise | toplam |
|---|---|---|---|---|
| train | 146.855 | 122.080 | 24.534 | 293.469 |
| val | 18.191 | 15.786 | 3.582 | 37.559 |
| test | 21.910 | 17.386 | 3.584 | 42.880 |

Saha (train): SA12 187.133 · SA4 98.970 · `?` 6.847 (`.sdf`) · SA1 519

⚠️ `noise` sınıfı train'in yalnızca **%8.4'ü** — dengesiz.

### Kanal örtüşmesi (not)

val'in 84 kanalının 84'ü, test'in 130 kanalının 130'u train'de de var.
Kanal = fiber üzerindeki fiziksel konum; farklı günlerde farklı olaylar
olduğu için olay sızıntısı değil. Ama modelin kanala özgü imza öğrenme
riski not edilmeli.

---

## 4. SİNYAL ÇIKARMA

### 4.1 Faz DEĞİL, genlik

Ekibin kodu (`support_set_creator.py`, `sequence_few_shot_*.py`):

```python
s = np.hypot(s_re, s_im)        # |z| = √(re² + im²)
```

**Faz kullanılamaz.** Ölçüldü: `unwrap(angle(z))` sonrası `diff` her kanalda
std = **1.814** veriyor — `[-π,π]` düzgün dağılımın std'si tam olarak
`π/√3 = 1.8138`. Yani faz saf rastgele.

Sebebi: ham `re`/`im` değerleri `int16` alanında ama gerçek aralık **±11**
(22 benzersiz değer). `|z| ≈ 3` iken faz kuantalanması ~19° — gerçek faz
değişimini yutuyor.

### 4.2 ⚠️ Ölçek katsayısı tutarsızlığı

`test.py` (çıkarım):
```python
fb = FotasBinFile(bin_path, datatype=MAGNITUDE)
signals = fb.data[0, 0, :, :].astype(np.float32) / 16384.0
```

`support_set_creator.py` (eğitim):
```python
s = np.hypot(s_re, s_im)        # olcek katsayisi YOK
```

**Çıkarım `/16384` uyguluyor, eğitim uygulamıyor.** Ayrıca çıkarım orijinal
`.bin` dosyasını okuyor, eğitim dönüştürülmüş `.hdf5`'i. İki yol arasında
tutarlılık doğrulanmalı.

### 4.3 Pencere standartlaştırma (ekibin kuralı)

```python
if n > W:                                  # UZUNSA
    c = Σ(i · s²) / Σ(s²)                  # enerji agirlik merkezi
    s = s[c - W//2 : c + W//2]             # oraya ortalayarak kirp
elif n < W:                                # KISAYSA
    s = np.pad(s, (sol, sag), mode="reflect")
```

Bu kural benimsenmeli — kendi kuralımızı uydurmamalıyız.

### 4.4 ⚠️ Pencere boyutu tutarsızlığı

| Dosya | Değer |
|---|---|
| `support_set_creator.py` | `WINDOW_SIZE = 15000` (7.5 s) |
| `test.py` | `VALTEST_WIN = 20000` (10 s) |
| CSV'deki en yaygın | 15000 |

**Karar verildi: 15.000 örnek = 7.5 saniye @ 2000 Hz** (bkz. Bölüm 1.5).

Sonucu: val/test pencerelerinin **%57'si** zaten 15.000 veya daha uzun;
kısa olanlar (7.500 ve 10.000) yansıtmalı doldurma ile uzatılacak.

CSV'deki uzunluk dağılımı:

| örnek | süre | val | test |
|---|---|---|---|
| 7.500 | 3.75 s | 6.662 | 3.871 |
| 10.000 | 5 s | 8.395 | 14.734 |
| 15.000 | 7.5 s | 21.238 | 22.556 |
| 20.000 | 10 s | 1.264 | 1.719 |

Train'de ise `.bin` için **315 farklı** uzunluk (7.500–35.634), `.sdf` için
**1.433 farklı** (6.361–80.000). Eğitim ve test dağılımları uyuşmuyor.

---

## 5. SINIF AYRILABİLİRLİĞİ — ANA BULGU

Her pencere kendi medyanına/MAD'ine normalize edildikten sonra (genlik
seviyesi kısayolunu ortadan kaldırmak için) ölçüldü.

### 5.1 ✅ Genlik seviyesi kısayolu YOK

```
ham_ort   F = 0.164   zayif
```

Ham genlik seviyesi sınıfları ayırt etmiyor. Model bu bilgiyi kısayol olarak
kullanamaz.

### 5.2 ✅ `noise` ayrılabiliyor

| Özellik | climbing | cutting | noise | F(3 sınıf) |
|---|---|---|---|---|
| 0–25 Hz enerji payı | 0.316 | 0.371 | **0.665** | 0.597 |
| zarf std | 0.277 | 0.280 | **0.519** | **0.431** |
| eşik üstü zaman | 0.022 | 0.017 | **0.075** | 0.373 |
| spektral merkez | 265 Hz | 247 Hz | **122 Hz** | — |

`noise` enerjisinin üçte ikisi 25 Hz altında; olaylar enerjiyi bantlara yayıyor.

### 5.3 ❌ `climbing` / `cutting` ayrılmıyor

**Spektral özellikler** (8 bant, 0–1000 Hz):

| bant | climbing | cutting | F(cli/cut) |
|---|---|---|---|
| 0–25 | 0.3163 | 0.3712 | 0.013 |
| 50–100 | 0.1055 | 0.0840 | 0.075 |
| **100–150** | 0.0654 | 0.0520 | **0.112** ← en iyisi |
| 250–400 | 0.0874 | 0.0810 | 0.005 |
| 700–1000 | 0.1187 | 0.1253 | 0.001 |

**Zaman yapısı özellikleri** (zarf, modülasyon spektrumu, ritim):

| özellik | F(cli/cut) |
|---|---|
| tepe faktörü | 0.001 |
| zarf std | **0.000** |
| modülasyon tepe frekansı | 0.056 |
| modülasyon keskinliği | 0.047 |
| eşik üstü zaman oranı | 0.005 |

**Hiçbiri F > 0.12.** Spektrogramlar da gözle ayırt edilemiyor.

> ⚠️ **BU BÖLÜMÜN SONUCU 5.4'TE DÜZELTİLDİ.** Yukarıdaki F değerleri her
> özelliği **tek tek** ölçüyor. Ayırt edici bilgi tek bir özellikte toplanmamış,
> birçoğuna dağılmış olabilir — nitekim öyle çıktı. Tek değişkenli testlerin
> klasik körlüğü.

---

## 5.4 DÜZELTME — `climbing`/`cutting` AYRILIYOR

Bölüm 5.3'ün "ayrılmıyor" sonucu **yanlıştı.** Ön işleme hattı
(`src/real_data.py`) kurulduktan sonra tekrar ölçüldü.

### Yöntem

Ekibin hazır bölmeleri kullanıldı (grup-temiz olduğu Bölüm 3'te doğrulandı):
`train_final.csv` ile eğitim, `val_final.csv` ile ölçüm.

Her pencere tam hattan geçirildi: `hypot` → 15.000 örneğe standartlaştır →
boş pencereleri ele → pencere-içi normalize → STFT → spektrogram.

Spektrogramdan **26 özellik** çıkarıldı (8 bandın zaman içindeki ortalama/
std/maks değerleri + zarf istatistikleri), ardından **doğrusal lojistik
regresyon** eğitildi.

### Sonuç — `P` bileşeni

| Sınıf | precision | recall | F1 |
|---|---|---|---|
| **noise** | 0.987 | 0.975 | **0.981** |
| cutting | 0.637 | 0.725 | 0.678 |
| climbing | 0.700 | 0.613 | 0.653 |
| | | **macro-F1** | **0.771** |

Karışıklık matrisi (satır = gerçek):

```
              climbing  cutting  noise
climbing            49       31      0
cutting             21       58      1
noise                0        2     77
```

**Karışıklığın tamamı `climbing` ↔ `cutting` arasında.** `noise` hiçbir olayla
karışmıyor.

### İkili karşılaştırmalar (şans = 0.500)

| | doğruluk |
|---|---|
| climbing vs noise | **0.981** |
| cutting vs noise | **0.981** |
| **climbing vs cutting** | **0.662** |

160 doğrulama örneğiyle standart hata ≈ 0.037 → güven aralığı kabaca
**0.59–0.74**. Şansın belirgin şekilde üstünde.

### `P` / `S` karşılaştırması ✅ (Bölüm 8, madde 2 kapandı)

| | `P` | `S` |
|---|---|---|
| 3 sınıf macro-F1 | **0.771** | 0.729 |
| noise F1 | **0.981** | 0.932 |
| climbing vs cutting | 0.662 | 0.656 |

**`P` her ölçütte önde.** "Yalnızca `P` kullanılacak" kararı ölçümle doğrulandı.

### Bu sayılar TABAN, tavan değil

Yukarıdaki sonuçlar **doğrusal** bir sınıflandırıcının, spektrogramdan elle
çıkarılmış **26 sayı** üzerindeki performansı.

2D-CNN + SK-Attention modeli:
- Spektrogramın tamamını görecek (129×231), 26 sayıyı değil
- Doğrusal değil
- Zaman-frekans örüntülerini kendisi keşfedecek

**Aşılması gereken taban çizgisi: macro-F1 0.771.**

*(Karşılaştırma: sentetik veride 0.622 ± 0.166 almıştık.)*

### Boş pencere oranı — ikinci ölçüm (Bölüm 8, madde 3 kapandı)

Hat üzerinden, farklı örneklemle tekrarlandı:

| | 1. ölçüm | 2. ölçüm |
|---|---|---|
| climbing | %20 | %32 |
| cutting | %28 | %22 |
| noise | **%0** | **%0** |

Ortalama **~%27** olay penceresi eleniyor, `noise`'da iki ölçümde de **%0**.
Eşik (500 Hz üstü enerji payı > 0.45) doğru çalışıyor.

---

## 6. ETİKET KALİTESİ VE TUTARSIZLIKLAR

### 6.1 ⚠️ Olay pencerelerinin ~%25'i boş

```
climbing   %20 duz (beyaz gurultu) spektrum
cutting    %28
noise       %0
```

Düz spektrum = o pencerede tespit edilebilir sinyal yok. Muhtemel sebep:
`labels` kanal **aralığı** veriyor (örn. 163–172); aralığın kenarındaki
kanallarda sinyal zayıflamış olabilir.

İlgili: `test.py` çıkarımda sessiz pencereleri filtreliyor
(`np.mean(window**2) < 1e-4`), ama eğitim CSV'sinde bu filtre yok.

### 6.2 ⚠️ `.sdf` zamanlama uyuşmazlığı

```
153.750 ornek / 2000 Hz = 76.875 s      ama duration ozniteligi = 38.4375
```

Tam 2 kat. `.bin`'de böyle bir sorun yok (`40.629 / 2000 = 20.3145` =
`duration`). `file_format.py`'de `duration = numFrames / prf` olarak
tanımlı, yani `.sdf` veri kümesi kare sayısının iki katı örnek içeriyor.

### 6.3 🐛 Koddaki yorum/kod çelişkileri

| Dosya | Yorum | Kod |
|---|---|---|
| `sequence_few_shot_*.py` | `# Sadece S bileşeni alınır` | `f[ch][:]['P']` |
| `test.py` | `VALTEST_WIN = 20000  # 3 sn pencere` | 20000/2000 = **10 sn** |
| `test.py` | `VALTEST_STRIDE = WIN // 10  # %50 overlap` | %90 örtüşme |

Üçü de yanıltıcı. `P` mi `S` mi kullanılacağı **fark eder** — ölçümde
`S`'nin genliği `P`'ninkinin iki katı (7.0 vs 3.5).

**Tam konumlar** (yorum çelişkisi, üç dosyada da 61. satır):

```
/tf/start_training/RELATIONNET/sequence_few_shot_train.py       satir 61
/tf/start_training/RELATIONNET/sequence_few_shot_test.py        satir 61
/tf/start_training/RELATIONNET/sequence_few_shot_validation.py  satir 61

    s = f[ch][:]['P']  # Sadece S bileşeni alınır
```

`support_set_creator.py` (23. satır) aynı işi yapıyor ama orada yorum yok.

**Ölçüm `P`'yi destekliyor** (Bölüm 5.4) — yorum yanlış, kod doğru.

### 6.4 `labels` sınıf sütunu belirsiz

`.sdf` dosyasında 5. sütun hep `1`, CSV "climbing" diyor. `.bin` dosyasında
aynı sütun `0` ve `1` karışık, CSV yine "climbing" diyor.

`support_set_creator.py`'deki resmî harita: `{"cutting": 0, "climbing": 1, "noise": 2}`

**CSV'deki metin etiketlere güvenilmeli**, `labels` sütunu deşifre edilmemeli.

---

## 7. SORUMLUYA SORULACAKLAR

1. **Ölçek katsayısı:** Çıkarımda `/16384.0` var, eğitimde yok. Hangisi doğru?
   Eğitim de `FotasBinFile(datatype=MAGNITUDE)` yolundan mı geçmeli?
2. **`P` mi `S` mi?** Kod `P` alıyor, yorum `S` diyor. Genlikleri iki kat farklı.
3. **Pencere boyutu:** `support_set_creator.py` 15.000, `test.py` 20.000. Hangisi?
4. **`.sdf` zamanlaması:** `duration` ile örnek sayısı 2 kat uyuşmuyor. Gerçek
   frekans nedir?
5. **`.sdf` neden yalnızca eğitimde?** Val/test'te karşılığı yok, performansı
   ölçülemiyor.
6. **Boş pencereler:** Olay etiketli pencerelerin %25'inde sinyal yok. Eğitimde
   de sessiz pencere filtresi uygulanmalı mı?
7. **`climbing`/`cutting` ayrımı:** Mevcut RELATIONNET modelinin sınıf bazında
   performansı nedir? Bu iki sınıfı ayırabiliyor mu?
8. ⚠️ **PyTorch kurulumu:** Sunucuda `torch` yok, imaj TensorFlow 2.14.
   Modelimiz PyTorch (önceden eğitilmiş paket bir `state_dict`).
   `pip install torch` serbest mi, sunucunun internet erişimi var mı?
   Yoksa mimariyi Keras'a taşımak gerekir — 34.835 parametrelik küçük bir
   model ama SK-Attention'ın yeniden yazılması demek.
9. **GPU var mı?** TensorFlow cuDNN/cuBLAS eklentilerini kaydetmeye
   çalışıyor (GPU'lu imaj işareti) ama görünür GPU olduğu doğrulanmadı.

---

## 8. SONRAKİ ADIMLAR

### Tamamlananlar ✅
- [x] `S` bileşeniyle ölçümleri tekrarla → `P` daha iyi (Bölüm 5.4)
- [x] Boş pencereleri eleyip ayrışmayı tekrar ölç → macro-F1 0.771 (Bölüm 5.4)
- [x] Ön işleme hattını yaz → `src/real_data.py`, birim testleri geçti
- [~] Mevcut RELATIONNET sonuçları — **kapsam dışı bırakıldı**, kendi
      ölçümümüzü yaptık

### Sırada — gerçek eğitim hattı
- [x] Ortam kontrolü → **torch YOK**, GPU teyit edilmeli (Bölüm 8.5)
- [x] Yükleme hızı ölçümü → 3.87 ms/satır gruplu (Bölüm 8.5)
- [x] Alt örneklem stratejisi → **k=2** seçildi (Bölüm 8.5)
- [ ] `--kesif`: kenar/iç kanal boşluk oranı — kanal seçim kuralını doğrula
- [ ] Spektrogram önbelleği kur (`src/onbellek_kur.py`) + doğrula
- [ ] **PyTorch kurulumu** — sorumluya sorulacak (Bölüm 7, madde 8)

- [ ] PyTorch `Dataset`: `real_data.pencere_yukle` → `spektrogram` → 224×320
- [ ] `load_pretrained(model, bundle)` ile aktarım, `classifier` sıfırdan

### Spektrogram hattı — kesinleşmiş tarif

```
1. .bin.hdf5 ac, kanal veri kumesini oku
2. P alanindan: hypot(re, im)                 -> genlik
3. CSV penceresini kes [window_start:window_end]
4. 15.000 ornege standartlastir:
      uzunsa  -> enerji agirlik merkezine ortalayarak kirp
      kisaysa -> yansitmali doldur (reflect)
5. BOS MU kontrol et -> bossa ele
6. Pencere-ici normalizasyon: (s - medyan) / MAD
7. DC bilesenini cikar
8. STFT: n_fft=256, hop=64 @ 2000 Hz  -> 129 frekans x ~234 cerceve
9. 224 x 320 (frekans x zaman) olcekle
```

- [ ] Adımların hepsi **tek bir fonksiyonda** toplanacak — eğitim ve çıkarım
      aynı fonksiyonu çağıracak, ekibin yaşadığı tutarsızlık tekrarlanmayacak
- [ ] Boş pencere eşiği ölçümle belirlenecek (500+ Hz enerji payı > 0.45 → düz)

### Aktarım
- [ ] `das_2dcnn_sk_v1.pt` omurgası + yeni sınıflandırıcı
      (`load_pretrained(model, bundle)`)
- [ ] Sınıf sayısı 3 (climbing/cutting/noise) — sentetikteki 3 ile aynı sayı
      ama farklı sınıflar, `classifier` sıfırdan başlayacak
- [ ] Pencere 7.5 s @ 2000 Hz. Sentetik model 10 s ile eğitilmişti; spektrogram
      224×320'ye ölçeklendiği için aktarımı bozmuyor, sadece piksel başına
      düşen süre değişiyor (31 ms → 23 ms)

---

## 8.5 SUNUCU ORTAMI VE YÜKLEME ÖLÇÜMÜ (2026-08-26)

`src/sunucu_kontrol.py` ile sunucuda ölçüldü.

### Ortam

| | |
|---|---|
| İşletim sistemi | Linux 5.15, Python 3.11 |
| CPU | 8 (8'i kullanılabilir) |
| RAM | 16.5 GB |
| Disk boş | **646 GB** |
| **GPU** | ✅ **NVIDIA RTX 3090, 24 GB** (sürücü 535.183, CUDA 12.2) |
| **PyTorch** | ❌ **KURULU DEĞİL** |
| TensorFlow | 2.14.0 — GPU'yu görüyor |
| numpy / pandas / h5py | 1.26.4 / 3.0.3 / 3.9.0 |

⚠️ **En kritik bulgu: `torch` yok.** GPU var ve TensorFlow onu kullanıyor
(`list_physical_devices('GPU')` → `GPU:0`), ama bizim modelimiz PyTorch.

⚠️ **GPU paylaşımlı.** Ölçüm anında %85 kullanımda ve 5.3 GB dolu, süreç
tablosu boş — başka bir konteyner eğitim yapıyor. Bellek bizim için sorun
değil (34.835 parametre), ama hesap kuyruğu paylaşılıyor.

**Ağ erişimi ölçüldü — github, pypi ve download.pytorch.org üçü de
erişilebilir.** Yani `pip install` teknik olarak mümkün; Keras'a taşıma
zorunluluğu **ortadan kalktı**.

**Kurulacaksa CUDA 12.x tekerleği gerekir:**
`pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121`

⚠️ `--user` bilinçli: bu imaj ekip tarafından paylaşılıyor ve TensorFlow
2.14 `numpy<2` istiyor. Sistem `site-packages`'ına yazmak ekibin ortamını
bozabilir; `~/.local` altına kurmak geri alınabilir ve izole.
Kurulumdan sonra `numpy.__version__` hâlâ 1.26.4 mü, TensorFlow hâlâ
import ediliyor mu **doğrulanmalı**.

⚠️ numpy sunucuda **1.26.4**, yerelde 2.4.4. `real_data.py` sadece kararlı
API kullanıyor, sorun çıkmadı — ama yeni kod yazarken numpy 2'ye özgü
API'lerden kaçınılmalı.

### Yükleme hızı

| Ölçüm | Satır başına |
|---|---|
| Dağınık okuma (her satırda aç-kapa) | **6.50 ms** |
| Gruplu okuma (dosya nesnesi paylaşımlı) | **3.87 ms** |

Aşama kırılımı (gruplu): dilim okuma + `hypot` **%49** · STFT %22 ·
pencere/boş/normalize %18 · dosya açma %4.

**Darboğaz STFT değil, HDF5'ten okuma.** Yani `num_workers` artırmak tek
başına yetmez; asıl kazanç ön-hesaplamada.

Tek süreçli epoch süresi: k=2 için 3.6 dk, tam set için 18.5 dk.

### Boş pencere oranı — üçüncü ölçüm

Bu ölçümde **%19.0 ve %16.7** çıktı (Bölüm 5.4'te ~%27 idi). Fark beklenen:
bu örneklem `noise` sınıfını da içeriyor ve `noise`'da boş oran %0.

### Yedeklilik ve alt örneklem

train: 293.469 satır / **21.318 dosya** / dosya başına ortalama 10.9 kanal
(medyan 8, min 1, **maks 901**).

⚠️ Bir dosya **901 kanal** taşıyor — muhtemelen tüm fiberi etiketleyen bir
`noise` kaydı. Tek başına aşırı ağırlık yapabilir; alt örneklem bunu da
zaten sınırlıyor (k kanal alınıyor).

⚠️ val/test'te kanallar **bitişik değil** (aralıklar 1, 3, 4, 18, 53) —
train'de bitişik. val/test dosyaları elden geçirilmiş alt kümeler gibi
görünüyor.

**Alt örneklem sınıf dengesini düzeltiyor** (beklenmeyen yan fayda):

| k | satır | oran | noise payı |
|---|---|---|---|
| hepsi | 286.941 | %97.8 | %6.8 |
| 8 | 178.352 | %60.8 | %8.7 |
| 4 | 102.620 | %35.0 | %11.5 |
| **2** | **55.513** | **%18.9** | **%17.8** |
| 1 | 28.909 | %9.9 | %16.9 |

`noise` kayıtları kanal başına daha çok pencere taşıyor (2.4 vs climbing
1.2), o yüzden kanal budandıkça oranı yükseliyor. Bölüm 3'teki "`noise`
train'in %8.4'ü — dengesiz" uyarısı k=2'de büyük ölçüde çözülüyor.

**Asıl gerekçe:** `alt_orneklem` her dosyadan **en az bir kanal** aldığı
için k ne olursa olsun 21.318 dosyanın hepsi kalıyor. **k çeşitliliği
değil yedekliliği kesiyor.** Bağımsız birim dosya olduğuna göre k=2 bilgi
kaybı olmadan maliyeti %19'a indiriyor.

### 🐛 Düzeltilen hata — kanal seçimi uçları seçiyordu

`alt_orneklem`'in ilk sürümü `linspace(0, n-1, k)` kullanıyordu, yani k=2
için **ilk ve son kanalı**. Gerekçesi "hem güçlü hem zayıf kanaldan örnek
al" idi — ama bu gerekçe Bölüm 6.1 ile çelişiyor: kenar kanallar sinyalin
zayıfladığı yer, ve zayıf pencereler zaten `bos_mu` ile eleniyor. Yani
k=2'de sistematik olarak **en kötü iki kanal** seçiliyordu.

Yeni kural: `linspace(0, n-1, k+2)[1:-1]` — uçlar dışarıda, seçilenler
aralık içine eşit dağılıyor. (n=14 için k=1 → [7], k=2 → [4, 9].)

Bu kuralın gerekçesi hâlâ bir **tahmin**; `onbellek_kur.py --kesif`
kenar/iç boşluk oranını ölçüp doğrulayacak.

### Verilen kararlar

| Konu | Karar | Gerekçe |
|---|---|---|
| Ön-hesaplama | **Yapılacak** | Her epoch'ta okumak 4–19 dk ekler; darboğaz I/O |
| Önbellek biçimi | **uint8, 129×231** | dB [-80,0] → 0.31 dB adım. Sentetik model zaten 8-bit PNG görüyordu |
| Ölçekleme | Eğitim anında, önbellekte değil | 224×320 saklamak boyutu 2.4 kat artırır, bilgi eklemez |
| **Alt örneklem** | **k=0 — tüm satırlar** | Aşağıda |
| Çerçeve | Önbellek **çerçeve-bağımsız** | torch kurulumu beklenmeden üretilebilir |

Önbellek boyutu: **6.2 GB** (disk 646 GB boş), kurulum ~18.5 dk tek seferlik.

### Alt örneklem kararı — k=0, ve nedeni

İlk öneri k=2 idi. Gerekçesi şuydu: satırlar yedekli (aynı olay ~14 komşu
kanaldan kaydedilmiş) ve tüm satırlar kullanılırsa her dosya **kanal sayısı
kadar oy** kullanır — 901 kanallı bir dosya, 1 kanallı bir dosyadan 901 kat
fazla etkiler. k=2 her kayda eşit ağırlık verir ve `noise` payını %6.8'den
%17.8'e çıkarır.

**Karar k=0 (tüm satırlar) yönünde verildi** (kullanıcı kararı, 2026-08-26).
Karşı argüman geçerli: komşu kanallar birebir kopya değil — farklı mesafe,
farklı SNR — ve doğal bir veri artırma işlevi görüyorlar. "Yer yok" gerekçesi
de zayıftı: 6.2 GB, 646 GB boş diskte hiçbir şey.

**Mimari sonucu — bu karar hattı daha esnek yaptı.** Alt örneklem artık
kurulum kararı değil, **eğitim anı parametresi**:

```
onbellek_alt_kume(onbellek, kanal_basina=2)   -> k=2 alt kumesinin indeksleri
onbellek_alt_kume(onbellek, kanal_basina=None) -> tum veri
```

Aynı önbellekten her k sıfır maliyetle çıkıyor; yeniden kurulum gerekmiyor.
Doğrulandı: eğitim anındaki seçim, kurulum anındaki seçimle birebir aynı
satırları veriyor.

Böylece "eşit ağırlık" endişesi kaybolmuş değil — sadece çözümü veri atmak
yerine **eğitim anında örnekleme/ağırlıklandırma** oldu. Bu, bilgi kaybı
olmadan aynı sorunu çözer.

⚠️ Alt örneklem kullanılırsa kaç konfigürasyon denendiği rapora yazılmalı
(proje kuralı: test setine bakarak hiperparametre seçilmez).

---

## 8.6 ÖNBELLEK KURULDU (2026-08-26)

`src/onbellek_kur.py` ile üç önbellek kuruldu, üçü de doğrulandı.

| | seçilen satır | yazılan pencere | elenen (boş) | süre | boyut |
|---|---|---|---|---|---|
| train | 286.941 | **220.834** | 66.107 (**%23.0**) | 19.2 dk | 6.59 GB |
| val | 37.559 | **37.517** | 42 (**%0.1**) | 2.7 dk | 1.12 GB |
| test | 42.880 | **42.850** | 30 (**%0.1**) | 3.1 dk | 1.28 GB |

Hata: **0**. Doğrulama: her önbellekte 20 rastgele pencere kaynak dosyadan
yeniden hesaplandı, **20/20 birebir aynı** (uint8 sapması 0).

### 🔍 ANA BULGU — val/test elle ayıklanmış

train'de boş pencere oranı **%23.0**, val/test'te **%0.1**. 230 kat fark.

Daha önce toplanan üç ipucuyla birleşiyor:

| Gösterge | train | val/test |
|---|---|---|
| Kanallar bitişik mi | ✅ evet | ❌ hayır (aralıklar 3, 4, 18, 53) |
| Pencere uzunluğu çeşidi | 315 (`.bin`) | 4 |
| Dosya biçimi | %97.8 `.bin` | %100 `.bin` |
| Boş pencere | %23.0 | %0.1 |

**Sonuç: val/test kürasyonlu alt kümeler.** Bölmeleri kuran kişi olayı
gerçekten gösteren kanalları seçmiş; train ham bırakılmış.

**Metodolojik sonucu — boş pencere filtresi train'i val/test'e yaklaştırıyor.**
Filtre olmasaydı model, val/test'te hiç karşılaşmayacağı 66.107 sinyalsiz
pencereye `climbing`/`cutting` demeyi öğrenecekti. Rapor 6.1'in önerdiği
filtre, ölçülmemiş bir sezgiden ibaret değilmiş.

⚠️ **Çıkarım (deployment) için not:** gerçek fiberde zayıf kanallar da
olacak. Hattımız onlara `None` döndürüyor, yani "tespit yok" diyor —
yanlış sınıflandırmıyor. Doğru davranış, ama rapora yazılmalı.

### Sınıf bazında elenme — önceki ölçümlerle tutarlı

| sınıf | `.bin` satır | yazılan | elenen |
|---|---|---|---|
| climbing | 145.764 | 108.527 | %25.5 |
| cutting | 121.628 | 92.922 | %23.6 |
| noise | 19.549 | 19.385 | **%0.8** |

Bölüm 6.1'deki ilk ölçüm (climbing %20, cutting %28, noise %0) doğrulandı.

### Eğitim sonrası sınıf dağılımı

| | climbing | cutting | noise |
|---|---|---|---|
| train | %49.1 | %42.1 | **%8.8** |
| val | %48.5 | %42.1 | %9.4 |
| test | %51.1 | %40.6 | %8.3 |

Üç bölme de tutarlı. `noise` hâlâ ~%9 — dengesizlik sürüyor, ama taban
çizgisinde `noise` en kolay sınıftı (F1 0.981), asıl zorluk
`climbing`↔`cutting`. Yine de sınıf ağırlığı seçeneği açık tutulmalı.

---

## 8.7 EĞİTİM HIZI ÖLÇÜMÜ (2026-08-26)

`gercek_egitim.py --olcum`, RTX 3090 (paylaşımlı), batch 64, 6 işçi:

| Aşama | örnek/s | epoch tahmini |
|---|---|---|
| yalnız veri (DataLoader) | 1.624 | 2.3 dk |
| + GPU'ya + `hazirla()` | 1.632 | 2.3 dk |
| + tam adım (ileri/geri/optim) | **939** | **3.9 dk** |

### İki çıkarım

**1. GPU'daki dönüşüm bedava.** 1.624 → 1.632, fark yok. `hazirla()`'yı
GPU'ya taşıma kararı doğruydu — CPU'da yapılsaydı ölçülebilir bir maliyet
olurdu (batch başına 55 MB yerine 1.8 MB PCIe trafiği).

**2. Tek darboğaz veri okuma değil.** `1/939 − 1/1632` hesabından GPU
adımının tek başına ~**2.200 örnek/s** kapasiteli olduğu çıkıyor. Yani veri
okuma sonsuz hızlansa bile tavan 2.200 — epoch ancak 3.9 dk'dan 1.7 dk'ya
inerdi.

GPU adımının 34.835 parametrelik bir model için bu kadar yavaş olmasının
iki sebebi var: GPU **paylaşımlı** (ölçüm anında başka bir konteyner %85
kullanıyordu) ve `use_deterministic_algorithms(True)` hızlı cuDNN
algoritmalarını devre dışı bırakıyor. Determinizm bilinçli bir tercih
(DURUM.md Bölüm 6) ve bedeli kabul edilebilir.

### Reddedilen iyileştirme: önbelleği RAM'e almak

Önbellek `chunks=(64, 129, 231)` + LZF ile yazıldı. `shuffle=True` iken
rastgele erişimde HDF5 tek örnek için 64 örneklik bloğu açıyor; üstelik
h5py'nin varsayılan blok önbelleği 1 MB, blok 1.9 MB — hiç tutmuyor.

Çözüm olarak tüm train önbelleğini (6.58 GB) RAM'e almayı denedik,
**OOM ile öldürüldü** (16.5 GB makine, 6 işçi süreci ve CUDA bağlamı
ayaktayken). Yukarıdaki hesap zaten kazancın 3.9 → 1.7 dk ile sınırlı
olduğunu gösterdiği için **vazgeçildi**.

`bellege_al` seçeneği kodda duruyor ama varsayılan **kapalı** ve artık
`MemAvailable`'a bakıp yetmiyorsa OOM yerine açık bir `MemoryError`
fırlatıyor.

> Not: LZF neredeyse hiç sıkıştırmamış (6.58 GB ham → 6.59 GB dosya).
> Önbellek yeniden kurulursa `compression=None, chunks=(1, ...)` daha
> doğru olur — ama mevcut önbelleği yeniden kurmayı hak edecek bir kazanç
> değil.

---

## 9. SENTETİK VERİYLE KARŞILAŞTIRMA

| | Sentetik | Gerçek |
|---|---|---|
| Bağımsız örnek | **19 kayıt** | **14.490 oturum** |
| Pencere | 10 s @ 2000 Hz | 10 s @ 2000 Hz (aynı) |
| "Olay yok" sınıfı | ❌ yok | ✅ `noise` |
| Bölme | kendimiz kurduk (Faz 1) | ✅ hazır ve temiz |
| Etiket gürültüsü | yok (sentetik) | ⚠️ ~%25 boş pencere |
| Sınıf karışıklığı | `chain_link_climbing` ↔ `metal_bending` | `climbing` ↔ `cutting` |

**Not:** Sentetik veride de iki sınıf ayrılamıyordu, burada da. Farklı veri
setleri, aynı türden sorun — bu, sorumlunun "benzer durumların orada da
yaşanacağını düşünüyorum" öngörüsünü doğruluyor.
