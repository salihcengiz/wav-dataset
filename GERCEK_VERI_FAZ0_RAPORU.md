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

#### Bu ne demek, ne demek DEĞİL

**Demek:** Bu iki sınıf, standart akustik ve zaman-yapısı özellikleriyle
ayrılmıyor. İleride bir model yüksek `climbing`/`cutting` doğruluğu
raporlarsa, **kısayol/sızıntı açısından incelenmelidir.**

**Demek DEĞİL:** "CNN de ayıramaz." Elle tasarlanmış 15 özellik denendi;
derin öğrenmenin varlık sebebi insanın tasarlamadığı örüntüleri bulmaktır.

**Sınırlar:** sınıf başına 40–60 örnek · yalnızca `P` polarizasyonu ·
tek pencere boyutu (7.5 s) · tek STFT ayarı.

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

---

## 8. SONRAKİ ADIMLAR

### Hemen
- [ ] Mevcut RELATIONNET modelinin **sınıf bazında** sonuçlarını al —
      `climbing`/`cutting` bulgusunu bağımsız doğrular
- [ ] `S` polarizasyonuyla 5.3'teki ölçümleri tekrarla
- [ ] Boş pencereleri (düz spektrum) eleyip ayrışmayı tekrar ölç

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
