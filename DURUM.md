# DURUM — Buradan Başla

> Bu dosya, projeye yeni katılan biri (veya yeni bir sohbet) için giriş
> noktasıdır. Önce bunu oku, sonra aşağıdaki sıraya göre diğer dosyaları.

**Son güncelleme:** 2026-09-01

---

## 1. PROJE NEDİR

**Inosens** şirketinde bir staj projesi. DAS (Distributed Acoustic Sensing /
Φ-OTDR) tabanlı **çevre güvenliği sistemi** — fiber optik kablo kilometrelerce
uzunlukta sürekli bir titreşim sensörüne dönüşüyor. Odak: **tel örgü çitler**.

**Görev:** Çit-ihlali olaylarını sınıflandıran bir derin öğrenme modeli eğitmek.

**Mimari** şu makaleden uyarlandı:
> You, J. ve ark. (2025). *"DAS-Based Perimeter Intrusion Detection Using 2D-CNN
> With SKAttention Mechanism."* IEEE Sensors Journal, 25(22), 41320–41328.

---

## 2. ŞU AN NEREDEYİZ

Proje **iki aşamalıydı** ve birincisi bitti:

### Aşama 1 — Sentetik veri ✅ TAMAMLANDI

Kendi ürettiğimiz sentetik spektrogramlarla model geliştirildi ve
önceden-eğitilmiş bir model paketi çıkarıldı.

**Sonuç: macro-F1 0.622 ± 0.166** (4 katlı, kaynak-gruplu çapraz doğrulama)

Düşük görünüyor ama **dürüst**: ~960 spektrogram yalnızca **19 bağımsız ses
kaydından** türetilmişti. Rastgele bölme yapılsaydı %99 görünürdü ve yalan olurdu.

Üç konfigürasyon denendi (0.614 / 0.622 / 0.628), farklar ±0.17 standart
sapmanın altında kaldı. Tavanın sebebi **model değil veri**:
`chain_link_climbing` akustik olarak tutarlı bir sınıf değil.

**Çıktı:** `outputs/pretrained/das_2dcnn_sk_v1.pt` (34.835 parametre, 156 KB)

Tam kayıt: `SENTETIK_VERI_SONUCLARI.md`

### Aşama 2 — Gerçek saha verisi 🔄 DEVAM EDİYOR

Faz 0 denetimi, ön işleme hattı, spektrogram önbelleği ve **beş eğitim
koşusu** tamamlandı.

**Taban çizgisi: macro-F1 0.771** (doğrusal sınıflandırıcı, 26 özellik)
**Ulaşılan: macro-F1 0.9390** (CNN-BiLSTM, koşu 4) — **+0.168**

Kalan hata `climbing` ↔ `cutting` arasında; `noise` çözülmüş (F1 0.987).

Sıradaki işler Bölüm 5'in sonunda ve
`GERCEK_VERI_EGITIM_SONUCLARI.md` Bölüm 10'da.

---

## 3. OKUMA SIRASI

| # | Dosya | Ne için |
|---|---|---|
| **1** | **`GERCEK_VERI_EGITIM_SONUCLARI.md`** | **EN GÜNCEL.** Beş koşunun sonuçları, atıf hesabı, çürütülen teşhis, ONNX, kod değişiklikleri, açık işler |
| 2 | `GERCEK_VERI_FAZ0_RAPORU.md` | Gerçek veri denetimi, kararlar (1.5), ayrılabilirlik (5.4), ortam/hız ölçümleri (8.5), önbellek (8.6) |
| 3 | `src/real_data.py` | Ön işleme hattı. Docstring'ler gerekçeleri açıklıyor |
| 4 | `CNN-BiLSTM/model_bilstm.py` | Kazanan mimari. Docstring'de tasarım gerekçeleri |
| 5 | `FAZ_BILGISI.md` | Sinyalin faz bilgisi: neden kullanılamıyor, ileride nasıl kullanılır |
| 6 | `PLAN_2DCNN_SKAttention.md` | Orijinal plan. Metodoloji (2, 5), uygulama kararları (6.4) |
| 7 | `SENTETIK_VERI_SONUCLARI.md` | Aşama 1'in tam kaydı: adli bulgular, hata analizi, neden 0.62'de tıkandı |
| 8 | `MODEL_IYILESTIRME_PLANI.md` | Sentetik aşamadaki iyileştirme paketleri ve ölçülen sonuçları |
| — | `outputs/pretrained/MODEL_CARD.md` | Sentetik model. **Artık kullanılmıyor** (aktarım işe yaramadı, koşu 2) |

Sadece gerçek veriyle çalışacaksan **1–4** yeter. 7 ve 8, "neden bu kararlar
alındı" sorusunun cevabı.

**Kod:** `src/` altındaki dosyaların hepsinde uzun docstring'ler var ve **neden**
öyle yapıldığını anlatıyorlar. Bir karar tuhaf görünüyorsa docstring'e bak.

### Kod haritası

| dosya | ne yapar | durum |
|---|---|---|
| `src/real_data.py` | **Ön işleme hattı.** Ham `.bin.hdf5` → 1B sinyal → spektrogram. Eğitim de çıkarım da bunu çağırır | aktif |
| `src/onbellek_kur.py` | Spektrogram önbelleği kurar + doğrular. `alt_orneklem()` burada | aktif |
| `src/gercek_veri_kumesi.py` | PyTorch `Dataset` (uint8 döner) + `hazirla()` (GPU'da ölçekle/normalize) | aktif |
| `src/gercek_egitim.py` | **Eğitim döngüsü.** `kos()` tek giriş noktası; `--olcum` darboğaz ölçümü | aktif |
| `src/gercek_rapor.py` | Koşuları karşılaştırır, grafik + markdown üretir | aktif |
| `src/gercek_export.py` | Teslim paketi + model kartı. `MIMARILER` ile iki mimariyi de paketliyor (koşu 4 → BiLSTM). Kart, `.pt` (görüntü) ile `.onnx` (ham sinyal) ayrımını en üstte gösteriyor | aktif |
| `src/model.py` | `DASNet` + SK/SE/CBAM + `load_pretrained()` | aktif |
| `src/config.py` | Tüm hiperparametreler | aktif |
| `src/sahte_onbellek.py` | **Yerel test için sahte veri üretir** (Bölüm 8A) | aktif |
| `CNN-BiLSTM/model_bilstm.py` | **Kazanan mimari** | aktif |
| `CNN-BiLSTM/egitim_bilstm.py` | İnce koşturucu (`kos()` çağırır) | aktif |
| `CNN-BiLSTM/onnx_disa_aktar.py` | ONNX ihracatı, ön işleme gömülü | aktif |
| `src/sunucu_kontrol.py` | Sunucu ortam + hız ölçümü | bir kerelik, bitti |
| `src/inspect_csv_index.py` | Faz 0: CSV indeksini çöz | bir kerelik, bitti |
| `src/inspect_hdf5.py` | Faz 0: HDF5 yapısını incele | bir kerelik, bitti |
| `src/inspect_dataset.py` | Faz 0: veri seti genel denetim | bir kerelik, bitti |
| `src/metadata.py` `splits.py` `dataset.py` `train.py` `export_model.py` `synth_das_pipeline.py` `augment_only.py` | **Aşama 1 (sentetik)** — gerçek veride kullanılmıyor | arşiv |

---

## 4. GERÇEK VERİ — BİLİNMESİ GEREKENLER

### Veri nerede

Uzak sunucuda, VPN + JupyterLab ile erişiliyor. **Veri sunucudan çıkamaz.**
Çalışma yöntemi: kod yazılır → kullanıcı JupyterLab'de çalıştırır → çıktı
paylaşılır.

```
/tf/start_training/RELATIONNET/FENCE_DATA_NEW/   <- CSV indeksleri
/tf/segment/YYYY.MM.DD/                          <- asil veri (.bin.hdf5)
```

CSV'ler bir **indeks**: `file, channel, event, window_start, window_end`.
`file` sütunu ana veri setine mutlak yol veriyor.

### Ölçek

| | |
|---|---|
| Toplam pencere | 373.908 (train 293.469 / val 37.559 / test 42.880) |
| Benzersiz oturum | **14.490** |
| Sınıflar | `climbing`, `cutting`, **`noise`** |
| Örnekleme frekansı | 2000 Hz |

Sentetikte 19 bağımsız kayıt vardı — burada 14.490 oturum. **Ölçek artık sorun değil.**

### Bölmeler temiz ✅

train/val/test arasında **dosya, oturum ve tarih düzeyinde sıfır çakışma**.
Sentetikte Faz 1'i bunu sağlamak için yazmıştık; burada hazır geliyor.

### Verilmiş kararlar

| Konu | Karar |
|---|---|
| Dosya biçimi | `.bin.hdf5` (val/test'in %100'ü) |
| Sinyal | `hypot(re, im)`, yalnızca **`P`** alanı |
| Pencere | **15.000 örnek = 7.5 s @ 2000 Hz** |
| Standartlaştırma | uzunsa enerji merkezine kırp, kısaysa yansıtmalı doldur |
| Boş pencereler | elenir (~%27) |
| Normalizasyon | pencere-içi (medyan/MAD) — **zorunlu** |
| Ölçek katsayısı | uygulanmaz (normalizasyon sadeleştiriyor) |

Gerekçeler: `GERCEK_VERI_FAZ0_RAPORU.md` Bölüm 1.5

### Ölçülen taban çizgisi

Doğrusal sınıflandırıcı + 26 elle yapılmış özellik:

```
macro-F1 0.771
  noise     F1 0.981     <- neredeyse cozuldu
  cutting   F1 0.678
  climbing  F1 0.653     <- karisiklik burada
```

**2D-CNN bunu aşmalı.**

---

## 5. SIRADAKİ İŞ

**Gerçek eğitim hattı.** Önce iki pratik soru:

1. **Sunucuda PyTorch ve GPU var mı?** Yol `/tf/` bir TensorFlow imajına
   işaret ediyor; ekibin Keras modelleri var. Bizim modelimiz PyTorch.
2. **293.469 satır nasıl işlenecek?** Her satır bir HDF5 açma işlemi.
   Satırlar fazlasıyla yedekli (21.318 dosya × ~14 bitişik kanal, hepsi aynı
   olayı görüyor) — alt örneklemle başlamak mantıklı.

### ✅ Ölçüldü (2026-08-26, `src/sunucu_kontrol.py`)

**Cevap 1: GPU VAR (RTX 3090, 24 GB, CUDA 12.2) ama `torch` KURULU DEĞİL.**
İmaj TensorFlow 2.14 ve GPU'yu görüyor. 8 CPU, 16.5 GB RAM, 646 GB boş disk.
GPU **paylaşımlı** — ölçüm anında %85 kullanımda, 5.3 GB başkasında.

**Cevap 2: 3.87 ms/satır** (gruplu okuma). Darboğaz STFT değil, **HDF5'ten
okuma** (%49). Tam set tek süreçte 18.5 dk/epoch.

**Alt örneklem: k=0 — tüm satırlar kullanılıyor.** Önbellek "ham malzeme"
olarak kuruluyor, hiçbir satır atılmıyor (boş pencereler hariç). Alt
örneklem böylece kurulum kararı değil **eğitim anı parametresi** oldu:
`onbellek_alt_kume(onbellek, kanal_basina=k)` aynı önbellekten her k'yi
sıfır maliyetle üretiyor, yeniden kurulum gerekmiyor.

Ayrıntı ve tüm sayılar: `GERCEK_VERI_FAZ0_RAPORU.md` Bölüm 8.5

### ✅ Önbellekler kuruldu ve doğrulandı (2026-08-26)

| | pencere | boş elendi | boyut |
|---|---|---|---|
| train | **220.834** | %23.0 | 6.59 GB |
| val | **37.517** | %0.1 | 1.12 GB |
| test | **42.850** | %0.1 | 1.28 GB |

Üçünde de doğrulama 20/20 birebir geçti, hata 0.

⚠️ **Bulgu: val/test elle ayıklanmış.** train'de boş pencere %23, val/test'te
%0.1. Boş pencere filtremiz train'i val/test'e *yaklaştırıyor*. Ayrıntı:
`GERCEK_VERI_FAZ0_RAPORU.md` Bölüm 8.6

### Önbellek nasıl kuruldu (`src/onbellek_kur.py`)

Ön-hesaplama kararı ölçüme dayanıyor: darboğaz I/O olduğu için her epoch'ta
HDF5 okumak her epoch'a 4–19 dk ekler. Bir kez hesaplanıp **uint8** olarak
diske yazılıyor — tüm satırlarla **6.2 GB**, tek seferlik ~18.5 dk.

**torch gerektirmiyor** — numpy + h5py. Kurulum sorusu beklenmeden
ilerleyebilir, üretilen önbelleği PyTorch da Keras da okur.

Üç mod: `--kesif` (kanal seçim kuralını ölçümle doğrula) · varsayılan
(önbelleği kur) · `--dogrula` (önbelleği kaynaktan yeniden hesaplayıp
birebir karşılaştır).

### ✅ PyTorch veri kümesi yazıldı (`src/gercek_veri_kumesi.py`)

Önbellekten model girdisi üretir: uint8 → viridis → 224×320 → ImageNet
normalizasyonu. 8 birim testi geçiyor (çok işçili okumanın tek işçiliyle
birebir aynı olduğu dahil). Yerelde CPU torch ile test edildi.

İki ayrıntı önemli:
- **viridis gömülü.** Sentetik model viridis PNG görmüştü; aktarımın
  anlamlı olması için aynı temsil gerekiyor. `matplotlib` bağımlılığı yok.
- **`__getstate__` h5py tutamacını düşürüyor.** Olmasaydı `num_workers>0`
  Windows'ta çöker, Linux'ta *sessizce yanlış veri* okuyabilirdi.

### ✅ Eğitim hattı (`src/gercek_egitim.py`)

`kos()` fonksiyonu tek giriş noktası. İçine gömülü dersler: A1 (val
macro-F1 izlenir), A2 (eşitlikte en erken epoch), A3 (determinizm), her
iyileşmede diske checkpoint, teste **yalnızca bir kez** bakma.

Farklı mimariler bu döngüyü **kopyalamadan** kullanabilsin diye üç
parametre eklendi (geriye dönük uyumlu, A/B test edildi — 42 tensör
birebir aynı çıktı):

```python
kos(kosu, model_fn=None, maske_p=0.5, sabir=SABIR)
```

CLI: `--kosu 1..5`, `--maske-p`, `--sabir`, `--isci`, `--hizli`,
`--sinif-agirligi`, `--bellege-al`

### ✅ BEŞ KOŞU TAMAMLANDI (2026-08-28)

| # | mimari | girdi | başlangıç | rejim | **test macro-F1** |
|---|---|---|---|---|---|
| **4** | **CNN-BiLSTM** | viridis | sıfırdan | yeni | **0.9390** ← EN İYİ |
| 1 | 2D-CNN+SK | viridis | sıfırdan | eski | 0.8843 |
| 3 | 2D-CNN+SK | gri | sıfırdan | eski | 0.8737 |
| 5 | 2D-CNN+SK | gri | sıfırdan | yeni | 0.8704 |
| 2 | 2D-CNN+SK | viridis | aktarım | eski | 0.8658 |
| — | *doğrusal, 26 özellik* | — | — | — | *0.771* |

**Taban çizgisi +0.168 farkla aşıldı.**

Üç soru da cevaplandı:

- **Sentetik ön-eğitim işe yaramadı** (−0.019). Aşama 1'in *ağırlıkları*
  artık kullanılmıyor; mimarisi ve metodolojisi kullanılıyor.
- **Viridis zarar vermedi** (+0.011 lehine, tahminin tersi).
- **Zamansal dizi (BiLSTM) kazandırdı: +0.058.** Fark-farkı hesabıyla
  atfedildi — rejim etkisi ayrıca ölçüldü ve **sıfır** çıktı (−0.003).

Kazanç tam da hedeflenen yerde: `cutting` +0.090, `climbing` +0.066,
`noise` +0.008. Karışıklık yarıya indi.

⚠️ **Önceki teşhis ÇÜRÜTÜLDÜ.** Koşu 1–3'ten sonra "model yetersiz
öğreniyor, rejimi düzeltmek yeter" denmişti. Koşu 5 bunu ölçüp yalanladı:
maskelemeyi kapatmak ve bütçeyi artırmak **hiçbir şey kazandırmadı**.
Maskeleme yakınsamayı yavaşlatıyordu, tavanı belirlemiyordu. Tavanı
mimari belirliyordu.

⚠️ **Kalan belirsizlik:** BiLSTM aynı zamanda 12 kat büyük (430.932 vs
34.835 parametre). Kazancın zamansal modellemeden mi kapasiteden mi
geldiği ayrılmadı. Ayıracak koşu: geniş SK (`CONV_CHANNELS=(32,64,128)`)
+ yeni rejim.

**Tam kayıt — beş koşu, atıf hesabı, ONNX, kod değişiklikleri, açık işler:
`GERCEK_VERI_EGITIM_SONUCLARI.md`**

### ✅ CNN-BiLSTM mimarisi (`CNN-BiLSTM/`)

Omurga DASNet ile **birebir aynı** (modülleri ödünç alınıyor, kopyalanmıyor);
yalnızca havuzlama başı değişiyor:

```
... -> SK-Attention -> (B,64,28,40)
    -> frekansi 4 bine indir -> (B,40,256) dizi
    -> BiLSTM(256->128, cift yonlu) -> dikkatli zaman havuzlama -> (B,3)
```

`AdaptiveAvgPool2d(1)` zaman eksenini tek sayıya çökertiyordu; `cutting`
ritmik/ayrık, `climbing` sürekli/düzensiz olduğu için ayrım oradaydı.

### ✅ ONNX dışa aktarım (`CNN-BiLSTM/onnx_disa_aktar.py`)

Sorumlunun isteği: `(None, 15000)` girdi, yani **ham sinyal**. Bu yüzden
ön işlemenin tamamı grafiğin içine gömüldü.

```
girdi   sinyal       (batch, 15000)  ham genlik, P alani
cikti   logit        (batch, 3)      [cutting, climbing, noise]
        bosluk_orani (batch,)        > 0.45 ise tahmin KULLANILMAZ
```

Dört ONNX engeli çözüldü (`fft_rfft`, `median`, adaptive pooling,
antialias) — ayrıntı ve doğrulama sayıları sonuç belgesinde.

**opset 13** (2026-09-01, sorumlunun isteği — hedef çalışma zamanı 17'yi
desteklemiyor). Bedelsiz çıktı: 13 ile 17'nin logit farkı **0.00e+00**.
Sebebi, grafiğin zaten opset 17'ye özgü hiçbir operatör kullanmaması —
STFT elle yazılmış, medyan sıralama tabanlı, havuzlama sabit çekirdekli.
Ölçüm: `onnx_disa_aktar.py --opset-karsilastir`.

### ⏳ SIRADAKİ İŞLER

| # | iş | not |
|---|---|---|
| ~~1~~ | ✅ ~~Teslim paketini koşu 4'e güncelle~~ | **Kod hazır** (`MIMARILER`, iki mimari de yerelde test edildi). Sunucuda `python gercek_export.py --kosu 4` çalıştırılmalı |
| ~~2~~ | ~~`gercek_rapor.py`'yi beş koşuyla çalıştır~~ | **Ertelendi** (2026-09-01) — böyle bir rapor şu an sorumludan beklenmiyor |
| ~~3~~ | ✅ ~~ONNX kullanım kartını üret~~ | **Yapıldı.** `paket/bilstm_kosu4_KULLANIM.md`, performans tablosu dahil tam. İki küçük eksiği için sonuç belgesi Bölüm 10 |
| 4 | **Koşu 6: geniş SK + yeni rejim** | `CONV_CHANNELS=(32,64,128)`. Kazanç kapasiteden mi zamandan mı — tek kalan belirsizlik |
| 5 | Sorumluya özet | Beş koşu, en iyi model, ONNX teslimi |

---

## 6. ⚠️ ÖNEMLİ: TEKRARLANMAMASI GEREKEN HATALAR

Bu projede yapılıp düzeltilen hatalar. Yeni bir sohbet aynılarına düşmesin.

### Tek değişkenli F testine güvenmek

Gerçek veride `climbing`/`cutting` ayrımı için her özelliği **tek tek**
ölçtüm, F ≤ 0.112 çıktı, "ayrılmıyor" dedim. **Yanlıştı.** 26 özellik
birlikte kullanılınca doğrusal bir sınıflandırıcı %66.2 doğruluk aldı
(şans %50). Ayırt edici bilgi tek bir özellikte değil, birçoğuna dağılmıştı.

→ `GERCEK_VERI_FAZ0_RAPORU.md` Bölüm 5.4

### Faz çıkarmayı denemek

`.bin` dosyalarında `unwrap(angle(re + i·im))` + `diff` denedim. Sonuç saf
gürültü — std 1.814, yani `[-π,π]` düzgün dağılımın std'si (`π/√3`). Sebebi
ham değerlerin ±11 aralığında olması (22 benzersiz değer), faz kuantalanması
~19°. **Ekip de faz kullanmıyor, genlik kullanıyor.**

⚠️ Ama bu bir **fizik sınırı değil, veri hazırlama sınırı.** Faz, DAS'ta
genlikten teorik olarak üstün büyüklüktür (doğrusal, nicel, fading yok).
İleride kullanılabilir — ne gerektiği ve nasıl kullanılacağı:
**`FAZ_BILGISI.md`**

### Erken durdurmayı `val_loss` ile yapmak

Sentetik aşamada model seçimi `val_loss` izliyordu. Doğruluk yükselirken
kayıp arttığı durumlarda "ilerleme yok" sanıp modeli **1. epoch'ta**
dondurdu — baseline'ın 4 katmanından 2'si böyle bozuldu. **macro-F1'e
geçildi.**

→ `MODEL_IYILESTIRME_PLANI.md`, Paket 1 / A1

### Eğitim ile çıkarımı ayrı kodlamak

Ekibin mevcut kodunda eğitim ve çıkarım dört noktada ayrışıyor (ölçek
katsayısı, pencere boyutu, P/S, sessiz pencere filtresi). **Bizim hattımızda
tek fonksiyon var** — `real_data.pencere_yukle`. İki kod yolu olmadığı için
tutarsızlık imkânsız.

### Sarmalanmış modelin dış arayüzünü iç modülünkiyle karıştırmak

Sorumluya "model şu an `(None, 15000)` almıyor, `(None, 3, 224, 320)`
alıyor — ama zinciri ONNX'e gömüyorum" diye yazıldı. İki şekil aynı
cümlede geçti: biri **ihracattan önceki `nn.Module`'ün** girdisi, diğeri
**ihraç edilen grafiğin** arayüzü. Sorumlu manşete cevap verdi ("ham
sinyal vermen lazım, yeni model eğitmen gerekiyor") — oysa istediği şey
zaten yapılmıştı.

İki gün ve bir "yeni model eğitmek gerekir mi" yanlış çıkarımı buna gitti.

→ `torch.onnx.export` `nn.Module` iç içeliğini **düzleştirir**; çıkan
dosyada "sarmalayıcı" ile "asıl model" diye iki katman yoktur, tek bir
düz grafik vardır. Dolayısıyla "ön işleme modelin içinde mi" sorusunun
cevabı dosyaya bakılarak ölçülebilir bir olgudur.

→ **Ders:** teslim edilen artefaktın arayüzünden bahsederken iç modülün
şeklini hiç anma. `.pt` ile `.onnx` farklı girdiler alıyorsa bunu yan
yana yaz. `gercek_export.py`'nin ürettiği model kartı artık tam da bunu
yapıyor (en üstteki "Hangi dosyayı kullanmalısınız" tablosu).

### Şekil uyuyor diye anlam da uyuyor sanmak

Aktarımda `load_pretrained` yalnızca **şekil** uyuşmazlığını yakalıyordu.
Sentetik sınıflar `[chain_link_climbing, fence_cutting, metal_bending]`,
gerçek sınıflar `[cutting, climbing, noise]` — ikisi de **3 tane**. Şekiller
uyduğu için `classifier` katmanı sessizce yüklendi (`42/42 tensör`).

Ama bunlar farklı kavramlar, sıraları farklı ve `noise`'un sentetikte
karşılığı bile yok. MODEL_CARD zaten "classifier sıfırdan başlayacak"
diyordu; kod bunu sınıf sayısı tesadüfen eşleştiği için yapmıyordu.

→ `load_pretrained(..., atla=("classifier",))` eklendi. Sessiz hataydı:
hiçbir metrik göstermezdi, sadece 2. koşu yanıltıcı çıkardı.

### Handikapı kaldırmakla performansı artırmayı karıştırmak

Koşu 1–3'te doğrulama doğruluğu eğitimin üstündeydi. Sebebi doğru teşhis
edildi (maskeleme yalnızca eğitimde açık, metrikler bozuk girdide
ölçülüyor). Ama oradan **"maskelemeyi kapatmak skoru artırır"** diye bir
sonuç çıkarıldı ve koşu 5 bunu **yalanladı** — rejim düzeltmesi −0.003
getirdi, yani hiçbir şey.

Maskeleme yakınsamayı yavaşlatıyordu (koşu 1 epoch 39'da hâlâ tırmanıyor,
koşu 5 epoch 21'de yakınsıyor) ama **tavanı belirlemiyordu**.

→ "X bir handikap yaratıyor" ile "X'i kaldırmak sonucu iyileştirir" **ayrı
iddialar**. İkincisi ayrıca ölçülmeli.

### Paylaşılan ortamın koşu ortasında bozulabileceğini unutmak

2026-08-27'de koşu 1 başlarken öldü: `RuntimeError: Numpy is not available`.
Aynı kod saatler önce sorunsuz çalışmıştı.

**Belirti çok yanıltıcı:** kernel içinde `torch.from_numpy` çalışıyor, taze
bir süreçte çalışmıyor. Sebebi kernel'in numpy'ı belleğine saatler önce
yüklemiş olması; disk sonradan bozulunca kernel etkilenmedi.

İki tahmin tutmadı (import sırası, `python` vs `sys.executable`). Çözüm:

```
pip install --user --force-reinstall --no-cache-dir "numpy==1.26.4"
```

Aynı sırada `protobuf 6.33.6` bulundu (TF 2.14 `<5.0` istiyor) — biz
kurmadık. **Paylaşılan `/usr/local`'a başkaları da yazıyor.**

→ Ortam bozuk şüphesinde ilk test: `sys.executable` ile **taze bir
süreçte** `torch.from_numpy(np.zeros(3))`. Tanı bir dakika sürer.

### GPU paylaşımlı — dört kez engelledi

- %85 doluluk → epoch süresi 1.5 katına çıktı
- 0 boş bellek → koşu hiç başlayamadı (`CUDA out of memory`, 5 MiB boştu)
- 17 GB'lık bir iş girdi → epoch 2.5 → 5.5 dk
- 66 MiB boş → duman testi başlayamadı

→ Uzun koşuları **bekleyen başlatıcıyla** çalıştırın (Bölüm 8C'de örnek).
Sorumluya iletildi, "yapacak bir şey yok" dendi.

### `/dev/shm` 64 MB — DataLoader işçileri çöküyor

`Bus error ... out of shared memory`. Konteynerin paylaşılan bellek alanı
varsayılan **64 MB** (`df -h /dev/shm` ile bakılır); DataLoader işçileri
veriyi oradan geçiriyor.

Hesap: batch = 64 × 3 × 129 × 231 uint8 = **5.7 MB**. Kuyruk = 2 batch ×
işçi sayısı, ve `persistent_workers=True` ile üç yükleyici (train/val/test)
ayrı işçi grupları tutuyor. Çökme, train işçileri ayaktayken val
işçilerinin doğduğu **epoch sonu geçişinde** — talebin zirve yaptığı an.

→ `--isci 3` yeterli (veri okuma zaten darboğaz değil).
Kalıcı çözüm ekipte: konteyner `--shm-size=2g` ile başlatılmalı.

### Determinizmi açmayı unutmak

Aynı tohumla iki koşu farklı sonuç veriyordu (GPU'da konvolüsyon geri
yayılımı varsayılan olarak deterministik değil). `train.py` içinde
`set_deterministic()` var, kullan.

---

## 7. AÇIK SORULAR (sorumluya)

0. ✅ **PyTorch kurulumu — YAPILDI** (2026-08-26, `--user`). GPU koordinasyonu
   soruldu, **"yapacak bir şey yok"** yanıtı geldi; bekleyen başlatıcıyla
   çalışıyoruz.
0b. ⚠️ **Ortam bakımsız — bildirilmeli.** `protobuf 6.33.6` kurulu ama
   TensorFlow 2.14 `<5.0` istiyor (biz kurmadık). numpy bir kez bozuldu.
   `/dev/shm` 64 MB, DataLoader'ı çökertiyor — `--shm-size=2g` gerekiyor.
   Kalıcı çözüm: herkesin kendi `venv`'inde çalışması.
0c. ⚠️ **`bosluk_orani` filtresi.** ONNX modeli bu değeri çıktı veriyor ve
   `> 0.45` olan pencerelerin tahmini kullanılmamalı. Sorumluya iletilmeli.
1. `.sdf.hdf5` dosyalarında `duration` ile örnek sayısı 2 kat uyuşmuyor
   (`.bin`'de sorun yok). Gerçek frekans nedir?
2. `P`/`S` terminolojisi: dosya öznitelikleri `polarization: 2, port: 1`
   diyor, sorumlu "port" dedi. Rapor metninde hangisi kullanılmalı?
3. Ekibin kodunda üç yanlış yorum satırı var (`# Sadece S bileşeni alınır`
   yazıp `['P']` alması, `20000 # 3 sn` yazması vb.) — bilinçli mi, bakımsızlık mı?

---

## 8. ÇALIŞMA ORTAMLARI

Üç ayrı ortam kullanılıyor. Hangisinde ne yapıldığını bilmek önemli.

### A) Yerel makine (Windows) — kod yazma ve doğrulama

```
c:\Users\Cengiz\Desktop\inosens-internship\dataset
```

Kurulu: `torch 2.13.0+cpu` (**bilerek CPU-only**), `torchvision 0.28.0+cpu`,
`sklearn 1.9.0`, `pandas 3.0.5`, `numpy 2.4.4`, `h5py 3.16.0`,
`matplotlib`, `seaborn`, `librosa`, `soundfile`, `scipy`

**Burada eğitim yapılmaz** — kod yazılır, birim testleri koşturulur, kısa
smoke test'ler yapılır. GPU'lu torch bilerek kurulmadı.

Kabuk **PowerShell**. İki tuzak var:
- `Select-Object -First N` boru hattını erken kapatıp süreci öldürüyor →
  çıkış kodu 255. Hata sanma; tam çıktı için dosyaya yönlendir.
- Here-string (`@'...'@`) ile `python -c`'ye kod geçirmek tırnakları bozuyor.
  Uzun kod için geçici bir `.py` dosyası yaz.

Ek: `onnx 1.22`, `onnxruntime 1.29` kurulu (ONNX ihracatını yerelde test
etmek için). WSL Ubuntu'da **TeX Live 2022** var — `report/` altındaki
LaTeX raporu oradan derleniyor:
```
wsl -e bash -lc "cd /mnt/c/.../dataset/report && pdflatex rapor.tex && pdflatex rapor.tex"
```

#### ⭐ Yerelde nasıl test edilir — `src/sahte_onbellek.py`

Gerçek veri sunucudan çıkamıyor, ama kodu sunucuya göndermeden önce
denemek zorundayız (her deneme bir tur kaybı, GPU da paylaşımlı).

```bash
python src/sahte_onbellek.py --hedef sahte_veri
```

Gerçek verinin **yapısını** taklit eden küçük bir küme üretir: `.bin.hdf5`
dosyaları (aynı dtype, aynı ±11 değer aralığı), CSV indeksleri (değişken
pencere uzunlukları) ve spektrogram önbellekleri. Sayılar gerçeğe benzemez
(168 pencere vs 220.834) — amaç sonuç üretmek değil, **hattın çalıştığını
doğrulamak**.

```bash
python src/gercek_veri_kumesi.py sahte_veri/onbellek_train_final_k0.h5
python src/gercek_egitim.py --kosu 1 --veri sahte_veri --cikti sahte_veri/c \
    --epoch 1 --batch 8 --isci 0
python CNN-BiLSTM/egitim_bilstm.py --hizli --veri sahte_veri \
    --cikti sahte_veri/c --batch 8 --isci 0 --epoch 1
python CNN-BiLSTM/model_bilstm.py
python CNN-BiLSTM/onnx_disa_aktar.py --sahte-agirlik --cikti /tmp/t.onnx
```

**Bu oturumda yakalanan hataların çoğu tam da bu testlerde çıktı:** birim
testinin tüm kümeyi belleğe toplaması (OOM), `lr` geçmişinin skalerle
ezilmesi, erken durdurma mesajının sabit `SABIR` basması, ONNX'in dört
ayrı operatörde takılması.

⚠️ Kod değişikliği sunucuya gönderilmeden önce **eski yolun bozulmadığı**
da sınanmalı. `kos()` değiştirildiğinde `git stash` ile A/B testi yapıldı:
42 ağırlık tensörü ve val macro-F1 tam hassasiyetle aynı çıktı.

### B) Google Colab — sentetik aşamanın eğitimi (Aşama 1)

Notebook: `colab_train.ipynb` (repoda). Akış:

1. **Hücre 1** repoyu klonlar. `rm -rf` + `git clone` içeriyor, yani
   `outputs/` dahil her şeyi siler — kod güncellendiğinde bunu çalıştır
2. Drive bağla → **eğitimden ÖNCE**, sonra değil
3. Eğit → biter bitmez `shutil.copytree('outputs', drive_yolu)` ile yedekle

**Öğrenilen dersler:**
- Oturum boşta ~90 dk'da kopuyor, `/content` tamamen siliniyor.
  Bir kez 4 dakikalık koşuyu kaybettik çünkü Drive yedeği alınmamıştı
- `drive.mount` "credential propagation was unsuccessful" verebiliyor —
  üçüncü taraf çerezleri engelliyse. Tekrar denemek genelde çözüyor;
  çözmezse `files.download` ile zip indirmek alternatif
- Hücre 1'i tekrar çalıştırınca `outputs/` gittiği için, Paket 2 sonuçlarını
  Drive'dan geri kopyalamak gerekti (`export_model.py` CV geçmişine ihtiyaç
  duyuyor)
- T4 GPU'da SK eğitimi (4 katman) ~5 dk, Paket 2'de ~11 dk

Drive'daki yedekler: `MyDrive/das_outputs/paket1_*`, `paket2_*`, `pretrained_*`

### C) Uzak sunucu (JupyterLab) — gerçek veri

VPN + tarayıcı. **Veri sunucudan çıkamaz, eğitim orada olmalı.**

```
/tf/start_training/RELATIONNET/FENCE_DATA_NEW/   <- CSV'ler, calisma alani
/tf/segment/YYYY.MM.DD/                          <- asil veri
```

**Çalışma yöntemi:** kod yazılır → kullanıcı JupyterLab'de çalıştırır →
çıktı sohbete yapıştırılır. **Kimlik bilgisi asla istenmez.**

**Kod taşıma — `%%writefile` (2026-08-26'dan itibaren tercih edilen):**

Dosya yüklemek yerine notebook hücresi dosyayı diske yazar:

```python
%%writefile onbellek_kur.py
<dosyanin tam icerigi>
```

sonra

```python
import importlib, real_data, onbellek_kur
importlib.reload(onbellek_kur)   # icerigi degistikten sonra ZORUNLU
```

⚠️ **Hücreye fonksiyon TANIMI yapıştırılmaz.** Dosya diske yazılıp
`import` edildiği sürece kalite düşmez — yükleme ile birebir aynı sonuç.
Ama fonksiyonlar hücrede tanımlanırsa repodaki sürüm ile sunucuda gerçekten
çalışan sürüm sessizce ayrışır; bu projenin temel ilkesi "tek kod yolu"
(bkz. `real_data.py` docstring'i).

⚠️ **Kabuk komutları `!` ile:** `!nvidia-smi`, `!python3 x.py`. Öneksiz
yazılırsa `SyntaxError` verir.

Alternatif: sol paneldeki **yukarı ok (↑)** ile yükleme (hâlâ çalışıyor).

**Ölçülen ortam:** Linux, Python **3.11**, 8 CPU, 16.5 GB RAM, 646 GB boş
disk, **RTX 3090 (24 GB, CUDA 12.2)**, TensorFlow 2.14, numpy **1.26.4**
(yerelde 2.4.4 — numpy 2'ye özgü API kullanma), `/dev/shm` **64 MB**.

**PyTorch kuruldu** (2026-08-26): `torch 2.5.1+cu121`, `torchvision
0.20.1+cu121`, `--user` ile `/root/.local` altına. `onnx 1.14.1` ve
`onnxruntime 1.16.3` zaten kuruluydu.

Yerelde kod yazılıp test ediliyor; sunucuda **yalnızca `git pull`** var.
Sparse-checkout ayarlı:

```
git sparse-checkout set src outputs/pretrained CNN-BiLSTM
```

### Uzun koşuları arka planda başlatma

⚠️ IPython'un `!` operatörü **arka plan süreçlerini desteklemiyor**
(`&` ile bitince `OSError: Background processes not supported`).
`subprocess.Popen` kullanın; `start_new_session=True` süreci kernel'den
ayırır, notebook kapansa bile yaşar.

GPU paylaşımlı olduğu için **bekleyen başlatıcı** kalıbı:

```python
import subprocess, os, sys
KOD = os.path.abspath("kod/CNN-BiLSTM")     # veya kod/src
LOG = "/tf/start_training/RELATIONNET/FENCE_DATA_NEW/loglar/kosuN.log"
komut = (
    'while true; do '
    '  bos=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); '
    '  if [ "$bos" -ge 4000 ]; then echo "GPU hazir: ${bos} MiB"; break; fi; '
    '  echo "  $(date +%%H:%%M) bos=${bos} MiB, bekleniyor..."; sleep 120; '
    'done; '
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; '
    'exec "%s" egitim_bilstm.py --isci 3'
) % sys.executable
p = subprocess.Popen(["bash","-c",komut], cwd=KOD, stdout=open(LOG,"w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
print("PID", p.pid)
```

⚠️ **Bitişi `ps` ile kontrol etmeyin.** `Popen` nesnesi toplanmadığı için
biten süreç **zombie** olarak PID tablosunda kalır ve `ps` "çalışıyor" der.
Doğrusu `p.poll() is not None` ya da log'un sonundaki `toplam sure` satırı.

⚠️ `--isci 3` kullanın — `/dev/shm` 64 MB ve daha fazla işçi `Bus error`
veriyor.

İzleme:
```python
!grep -E "epoch |dogrul\.|sure " <log> | tail -15
!head -n 14 <log>      # baslik: hangi kosu, hangi rejim
```

---

## 9. ÇALIŞMA YÖNTEMİ

### ⚠️ Git: commit ve push HER ZAMAN kullanıcı tarafından yapılır

**Asistan `git commit` / `git push` çalıştırmaz.** İş bitince değişiklikleri
özetler ve çalıştırılacak komutu **metin olarak verir**; kullanıcı terminale
kendisi girer.

Aynı şey geri alma işlemleri için de geçerli (`reset`, `revert`, `stash pop`
gibi kalıcı sonucu olanlar).

Daha genel kural: **kendi başına karar verilip yapılan bu tür işlerde önce
sorulur.**

### Diğer kurallar

- **Kararlar dosyalara yazılır, sohbete değil.** Bu belge ve raporlar bu
  yüzden var. Bağlam dolduğunda yeni bir sohbet bu dosyaları okuyarak
  kaldığı yerden devam edebilmeli.
- **Uzak veri için:** kod yerelde yazılır ve sahte veriyle test edilir
  (Bölüm 8A) → push edilir → sunucuda `git pull` → çalıştırılır → çıktı
  paylaşılır. Kimlik bilgisi asla istenmez.
- **Ölçmeden karar verilmez.** Bu projede en değerli bulguların hepsi
  ölçümden çıktı (birebir kopya dosya, eşzamanlı mikrofon çiftleri, sınıf
  tutarsızlığı, boş pencereler, aktarımın işe yaramaması, rejim
  değişikliğinin hiçbir şey kazandırmaması).
- **Tahmin ölçümle çeliştiğinde ölçüm kazanır ve bu yazılır.** Bu oturumda
  üç tahmin çürütüldü: viridis zarar verecek (vermedi), aktarım yardım
  edecek (etmedi), rejim düzeltmesi skoru artıracak (artırmadı).
- **Test setine bakarak hiperparametre seçilmez.** Kaç konfigürasyon
  denendiği rapora yazılır. **Şu ana kadar: 5.**
- **Tek kod yolu.** Eğitim ve çıkarım aynı fonksiyonu çağırır; kod
  kopyalanmaz, import edilir. Ekibin dört tutarsızlığı iki kod yolu
  olduğu için doğmuştu.
