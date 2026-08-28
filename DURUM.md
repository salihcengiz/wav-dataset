# DURUM — Buradan Başla

> Bu dosya, projeye yeni katılan biri (veya yeni bir sohbet) için giriş
> noktasıdır. Önce bunu oku, sonra aşağıdaki sıraya göre diğer dosyaları.

**Son güncelleme:** 2026-08-25

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

Faz 0 denetimi **bitti**, ön işleme hattı **yazıldı ve test edildi**.
Sunucu ortamı ve yükleme hızı **ölçüldü** (Bölüm 5). Sırada spektrogram
önbelleğinin kurulması ve gerçek eğitim var.

**Taban çizgisi ölçüldü: macro-F1 0.771** (doğrusal sınıflandırıcı, 26 özellik)

⚠️ **Engel:** Sunucuda `torch` kurulu değil — sorumluya soruldu (Bölüm 7).

---

## 3. OKUMA SIRASI

| # | Dosya | Ne için |
|---|---|---|
| 1 | **`GERCEK_VERI_FAZ0_RAPORU.md`** | **Şu an aktif olan iş.** Gerçek veri denetimi, kararlar (Bölüm 1.5), ölçümler (Bölüm 5.4) |
| 2 | `src/real_data.py` | Ön işleme hattı. Docstring'ler gerekçeleri açıklıyor |
| 3 | `outputs/pretrained/MODEL_CARD.md` | Önceden eğitilmiş model: nasıl yüklenir, sınırları |
| 4 | `PLAN_2DCNN_SKAttention.md` | Orijinal plan. Metodoloji (Bölüm 2, 5) ve uygulama kararları (Bölüm 6.4) |
| 1b | **`GERCEK_VERI_EGITIM_SONUCLARI.md`** | **Üç koşunun sonuçları**, çekinceler, iyileştirme sırası |
| 5 | `SENTETIK_VERI_SONUCLARI.md` | Aşama 1'in tam kaydı: adli bulgular, katman kompozisyonu, hata analizi, neden 0.62'de tıkandı |
| 6 | `MODEL_IYILESTIRME_PLANI.md` | Sentetik aşamadaki iyileştirme paketleri ve ölçülen sonuçları |

Sadece gerçek veriyle çalışacaksan **1–3** yeter. 5 ve 6, "neden bu kararlar
alındı" sorusunun cevabı.

**Kod:** `src/` altındaki dosyaların hepsinde uzun docstring'ler var ve **neden**
öyle yapıldığını anlatıyorlar. Bir karar tuhaf görünüyorsa docstring'e bak.

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

### ✅ Eğitim hattı yazıldı (`src/gercek_egitim.py`)

- [x] **`torch` kuruldu** — 2.5.1+cu121, GPU görülüyor, numpy 1.26.4 korundu
- [x] Eğitim döngüsü yazıldı, üç koşu da yerelde sınandı
- [ ] **Sunucuda koş** — önce `--hizli` duman testi, sonra üç koşu

### ✅ ÜÇ KOŞU TAMAMLANDI (2026-08-27)

| # | girdi | başlangıç | **test macro-F1** |
|---|---|---|---|
| **1** | viridis | sıfırdan | **0.8843** ← en iyi |
| 3 | gri | sıfırdan | 0.8737 |
| 2 | viridis | aktarım | 0.8658 |
| — | *taban çizgisi* | *doğrusal* | *0.771* |

**Taban çizgisi +0.113 farkla aşıldı.**

İki soru da cevaplandı:
- **Sentetik ön-eğitim işe yaramadı** (−0.019, aktarım daha kötü). Aşama
  1'in ağırlıkları artık kullanılmıyor; mimari ve metodoloji kullanılıyor.
- **Viridis zarar vermedi** (+0.011 lehine, tahminin tersi).

⚠️ **Ana bulgu: model yetersiz öğreniyor.** Üç koşuda da doğrulama
doğruluğu eğitimin üstünde; koşu 1 epoch tavanına çarptığında hâlâ
iyileşiyordu. Darboğaz mimari değil, eğitim rejimi (maskeleme fazla,
kapasite küçük).

Tam kayıt, çekinceler ve iyileştirme sırası:
**`GERCEK_VERI_EGITIM_SONUCLARI.md`**

⚠️ **Aktarım varsayımı artık geçerli değil.** Önceden eğitilmiş paket "çok
az saha verisi varken sıfırdan başlamamak" için üretilmişti (MODEL_CARD).
Ama artık **220.834 pencere / 21.101 bağımsız dosya** var; sentetikte 959
pencere / **19** kayıt vardı. 1.100 kat daha fazla bağımsız kaynak.

Bu ölçekte sıfırdan eğitim muhtemelen en az aktarım kadar iyi. Karar
tahminle değil ölçümle verilmeli: iki koşu, aynı bölmeler, aynı tohum,
ikisi de raporlanır. Rapora "sentetik ön-eğitim işe yarıyor mu" diye somut
bir bölüm açar.

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

### Determinizmi açmayı unutmak

Aynı tohumla iki koşu farklı sonuç veriyordu (GPU'da konvolüsyon geri
yayılımı varsayılan olarak deterministik değil). `train.py` içinde
`set_deterministic()` var, kullan.

---

## 7. AÇIK SORULAR (sorumluya)

0. ⚠️ **PyTorch kurulumu — teknik engel yok, İZİN sorusu.** GPU var (RTX
   3090, CUDA 12.2), ağ açık (pypi + download.pytorch.org erişilebilir),
   yani `pip install --user torch` çalışır. Keras'a taşıma zorunluluğu
   kalktı. Sorulacak: paylaşılan imaja `--user` kurulum yapmamız uygun mu?
   Ayrıca GPU paylaşımlı (ölçümde %85 doluydu) — eğitim saati için
   koordinasyon gerekir mi?
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

**Ölçüldü (2026-08-26):** Linux, Python 3.11, 8 CPU, 16.5 GB RAM, 646 GB boş
disk, TensorFlow 2.14, numpy **1.26.4** (yerelde 2.4.4 — numpy 2'ye özgü
API kullanma). **`torch` KURULU DEĞİL**, GPU teyit edilmedi.

---

## 9. ÇALIŞMA YÖNTEMİ

- **Kararlar dosyalara yazılır, sohbete değil.** Bu belge ve raporlar bu
  yüzden var.
- **Uzak veri için:** kod yazılır → kullanıcı JupyterLab'de çalıştırır →
  çıktı paylaşılır. Kimlik bilgisi asla istenmez.
- **Ölçmeden karar verilmez.** Bu projede en değerli bulguların hepsi
  ölçümden çıktı (birebir kopya dosya, eşzamanlı mikrofon çiftleri, sınıf
  tutarsızlığı, boş pencereler).
- **Test setine bakarak hiperparametre seçilmez.** Kaç konfigürasyon
  denendiği rapora yazılır.
