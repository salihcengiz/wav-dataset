> ⚠️ **Bu dosya Aşama 1'i (sentetik veri) anlatır ve o kapsamda geçerlidir.**
> Proje Aşama 2'de (gerçek saha verisi) ve **giriş noktası
> [`DURUM.md`](DURUM.md)** — oradan başlayın.
>
> Aşağıda "sentetik veri ön eğitim için kullanılacak" beklentisi geçiyor.
> **Bu ölçüldü ve çürütüldü:** aktarım gerçek veride −0.019 getirdi
> (`GERCEK_VERI_EGITIM_SONUCLARI.md` Bölüm 4). Sentetik aşamanın kalıcı
> katkısı mimari ve metodoloji oldu, ağırlıklar değil.

# DAS Sentetik Veri Üretim Pipeline'ı

Ses kayıtlarından (fence cutting, metal bending, chain-link fence climbing)
DAS-benzeri sentetik faz/gerinim sinyali ve spektrogram üretir.

**Bu klasördeki iki script'i birlikte kullanıyorsun:**

| Dosya | Ne yapar |
|---|---|
| `synth_das_pipeline.py` | Tam pipeline: augmentasyon + gürültü karıştırma + downsample + pencereleme + spektrogram |
| `augment_only.py` | Sadece augmentasyon + pencereleme (gürültü/downsample YOK) — kalite kontrolü için |

`augment_only.py`, `synth_das_pipeline.py`'deki fonksiyonları import ettiği için **ikisi
de aynı klasörde olmalı**.

## Kurulum

```bash
pip install -r requirements.txt --break-system-packages
```

## Klasör Yapısı (girdi)

```
raw_sounds/
    fence_cutting/
        wire_cutter_01.wav
        bolt_cutter_02.wav
    metal_bending/
        sheet_metal_01.wav
        metal_bend_02.wav
    chain_link_climbing/
        chain_rattle_01.wav
        fence_climb_02.wav
```

Klasör adları serbest — script, `input_dir` altındaki her alt klasörü ayrı bir
sınıf olarak işler. Önerilen: sınıf başına en az 5-10 farklı gerçek kayıt (az
sayıda kayıttan augmentasyon + MixUp ile büyük bir set üretiliyor, aşağıda anlatılıyor).

---

## 1. Tam Pipeline — `synth_das_pipeline.py`

### Çalıştırma

**Sabit varyant sayısıyla:**
```bash
python synth_das_pipeline.py --input_dir raw_sounds --output_dir synthetic_dataset --n_variants 5
```

**Toplam hedef örnek sayısıyla (örn. 1000 örnek istendiğinde):**
```bash
python synth_das_pipeline.py --input_dir raw_sounds --output_dir synthetic_dataset --total_target 1000
```
Bu mod, sınıflar arasında hedefi eşit dağıtır ve her sınıftaki dosya sayısına göre
dosya başına kaç varyant üretilmesi gerektiğini otomatik hesaplar.

### Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--target_sr` | **2000** | Hedef DAS-benzeri örnekleme frekansı (Hz). 1000–5000 arası öneriliyor |
| `--window_sec` | 10.0 | Pencere uzunluğu (saniye) |
| `--noise_kinds` | pink white | Aralarından rastgele seçilecek arka plan gürültü tipleri |
| `--snr_min` / `--snr_max` | 0 / 15 | Olay/gürültü oranı aralığı (dB), her varyant için rastgele seçilir |
| `--n_variants` | 3 | (Sadece `--total_target` verilmediğinde) her ham dosyadan üretilecek sabit varyant sayısı |
| `--total_target` | — | Tüm veri setinde ulaşılmak istenen TOPLAM örnek sayısı; verilirse `--n_variants` yok sayılır |
| `--no_mixup` | kapalı (mixup açık) | Aynı sınıftaki dosyaları MixUp ile karıştırmayı devre dışı bırakır |
| `--seed` | 42 | Rastgelelik tohumu — aynı komutu tekrar çalıştırınca aynı sonucu üretir |

> ⚠️ **`--target_sr` hakkında not:** Varsayılan 2000 Hz. Yaptığımız testlerde 2000 Hz'e
> indirgeme sinyal enerjisinin ~%95'ini siliyor ve olay darbeleri gürültü eklenmeden
> ÖNCE bile ayırt edilmesi zorlaşabiliyor; 5000 Hz'de kayıp ~%89'a düşüyor ve darbeler
> daha net kalıyor (bkz. "Downsample Notu" bölümü). Sorumlunun verdiği "1-5 kHz" aralığı
> içinde kalmak kaydıyla, spektrogramları inceleyip event'lerin gürültüden yeterince
> ayırt edilebildiğini görene kadar `--target_sr` değerini yükseltmeyi (`--target_sr 5000`
> gibi) deneyebilirsin.

### Çıktı

Her varyant için:
- `.wav` — sentetik DAS-benzeri sinyal (indirgenmiş örnekleme frekansında, gürültüyle karıştırılmış)
- `_spectrogram.png` — STFT spektrogramı (eksensiz, doğrudan CNN girdisi olarak kullanılabilir)

Dosya adları kaynağı ve parametreleri gösterir, örn:
`test_wire_cutter_snr5dB_pink_v84.wav` (augmentasyon varyantı) veya
`mixup_test_bolt_cutter_x_test_wire_cutter_snr14dB_white_92.wav` (MixUp varyantı).

---

## 2. Sadece Augmentasyon Önizlemesi — `augment_only.py`

Tam pipeline'ın **sadece** augmentasyon + pencereleme kısmını çalıştırır; gürültü
karıştırma ve downsample YOKTUR. Çıktı dosyaları orijinal örnekleme hızında
(varsayılan 44100 Hz), gürültüsüz — yani **doğrudan kulakla değerlendirilebilir**.

Amaç: "sentetik ses garip geliyor" gibi bir şüphe oluştuğunda, sorunun augmentasyondan
mı yoksa gürültü+downsample adımından mı geldiğini ayrı ayrı dinleyerek teşhis edebilmek.

### Çalıştırma

```bash
python augment_only.py --input_dir raw_sounds --output_dir augmented_preview
```

### Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--preview_sr` | 44100 | DAS frekansı DEĞİL — sadece dosyalar arası tutarlılık için ortak, yüksek kaliteli bir onizleme hızı |
| `--window_sec` | 10.0 | Pencere uzunluğu (saniye) |
| `--n_variants` | 4 | Her ham dosyadan üretilecek varyant sayısı (1. varyant her zaman augmentasyonsuz — "temiz referans") |
| `--no_mixup` | kapalı | MixUp'ı devre dışı bırakır |
| `--seed` | 42 | Rastgelelik tohumu |

> **Neden `--preview_sr` var?** Farklı kaynak dosyaların orijinal örnekleme hızları
> birbirinden farklı olabilir (örn. biri 44100 Hz, biri 96000 Hz gibi profesyonel bir
> kayıt). MixUp iki sinyali doğrudan topladığı için aynı uzunlukta (dolayısıyla aynı
> örnekleme hızında) olmalarını gerektirir — bu yüzden her dosya, DAS frekansına değil
> ama ortak bir onizleme frekansına önce getiriliyor. Bu "DAS'a benzetme" değildir,
> sadece dosyalar arası tutarlılığı sağlar.

### Çıktıdaki dosya adlarına dikkat

- `..._orijinal_pencereli.wav` — hiç augmentasyon yok, sadece 10 saniyeye yerleştirilmiş (temiz referans)
- `..._augmented.wav` — speed_change + gain jitter uygulanmış
- `mixup_..._.wav` — iki farklı kaydın karışımı

---

## Pipeline Mantığı — Az Sayıda Kayıttan Büyük Veri Seti Üretme

**1000 gibi bir hedefe gerçek, birbirinden farklı ses kaydıyla ulaşmak pratik değildir.**
Bunun yerine bu pipeline, az sayıda gerçek kayıttan (önerilen: sınıf başına 5-10 farklı
kayıt) sistematik veri artırma ile büyük ve çeşitlendirilmiş bir sentetik set üretir —
Zhang & Chen (2024, MSF-DenseNet) makalesindeki 50-örnek + MixUp stratejisiyle aynı
yöntemsel mantık.

Adımlar (`synth_das_pipeline.py` — tam pipeline):
1. **Yükleme**: Ham `.wav`/`.mp3` mono'ya çevrilir, ORİJİNAL örnekleme hızında tutulur
2. **Augmentasyon** (orijinal sr'de, downsample'dan ÖNCE): `speed_change` ile hız/perde
   varyasyonu (±%15) + genlik jitter'i (±3 dB)
3. **Downsample**: `scipy.signal.resample_poly` ile hedef DAS frekansına indirgenir
4. **Pencereye yerleştirme**: Olay tipine göre iki farklı strateji (aşağıda detaylı anlatılıyor)
5. **MixUp** (opsiyonel, varsayılan açık): Aynı sınıftaki iki farklı kayıt karıştırılır
6. **Gürültü karıştırma**: Hedef SNR'ye göre ölçeklenmiş pink/white noise eklenir
7. **Spektrogram**: `librosa.stft` ile hesaplanır, kısa sinyaller için n_fft otomatik küçültülür

`augment_only.py`, yukarıdaki 1-2-4-5 adımlarını çalıştırır, 3 (downsample) ve 6
(gürültü) adımlarını atlar.

## İki Geçmiş Artefakt Düzeltmesi (arka plan bilgisi)

Geliştirme sürecinde iki ses kalitesi sorunu tespit edilip kaynağından düzeltildi —
mevcut kodda bu düzeltmeler zaten var, burada sadece nedenini not ediyoruz:

### Sorun 1: Kısa klipler "uğultu" gibi geliyordu

**Neden:** 150ms'lik tek bir "çıt" sesini 10 saniyeye sığdırmak için `np.tile` ile
**boşluksuz** art arda diziliyordu. Bu, tek bir darbeyi periyodik bir darbe dizisine
çeviriyordu.

**Çözüm:** `place_event_in_window` fonksiyonu kısa/darbeli olayları (kaynak, pencerenin
%40'ından kısaysa) 2-6 arasında rastgele sayıda kopya olarak, **aralarında rastgele
sessizlik boşluklarıyla** (0.3-2.0 saniye) pencereye serpiştiriyor.

### Sorun 2: Uzun, darbeli kayıtların varyantları "su altında" gibi geliyordu

**Neden:** `librosa.effects.pitch_shift`/`time_stretch` faz vokoderi kullanıyor —
darbeli/geçici seslerde (metal kesme, testereleme gibi) "fazlılık" (phasiness)
artefaktı üretiyor.

**Çözüm:** Bunların yerine `speed_change` fonksiyonu kullanılıyor — saf yeniden
örneklemeye dayanıyor, faz manipülasyonu içermediği için darbeli seslerde net kalıyor.

## Downsample Notu (gürültüden daha etkili bir faktör)

Test sürecinde şu bulgu ortaya çıktı: **downsample adımı, gürültü karıştırmadan çok
daha yıkıcı.** 2000 Hz'e indirgeme sinyal enerjisinin ~%95'ini siliyor; bu kayıp
gürültü eklenmeden ÖNCE bile gerçekleşiyor. 5000 Hz'de enerji kaybı ~%89'a düşüyor.

**Kavramsal not:** Ses kaydı ile gerçek DAS sinyali (fiber gerinim oranı) zaten temelde
farklı fiziksel büyüklüklerdir. Downsample sonrası üretilen sesin kulakla dinlendiğinde
orijinaline birebir benzememesi kısmen kaçınılmazdır — gerçek DAS sistemlerinin kendisi
de bu kadar dar bant genişliğinde çalışır (literatürde: Meng ve ark. 500 Hz, Zhang &
Chen 2000 Hz örnekleme). Başarı ölçütü "kulakla dinleyince orijinale benziyor mu"
değil, **"spektrogram sınıflar arası ayırt edici, tutarlı bir örüntü içeriyor mu"**
olmalı — nihai çıktı bir CNN'e görüntü olarak besleniyor, kulağa değil.

## Uzun/Sürekli Olay vs Kısa/Darbeli Olay Ayrımı

`place_event_in_window`, kaynak dosyanın uzunluğuna göre iki farklı strateji uyguluyor:

- **Uzun olay** (kaynak ≥ pencerenin %40'ı, örn. 4 saniyelik sürekli testereleme):
  Pencere içine **tek kopya**, rastgele bir konumda yerleştiriliyor, geri kalan kısım
  sessiz bırakılıyor (sonra gürültüyle dolduruluyor).
- **Kısa olay** (örn. 150ms'lik tek "çıt"): Birden fazla kopya + rastgele boşluklarla
  seyrek olarak yerleştiriliyor.

Bu eşiği değiştirmek istersen `place_event_in_window` fonksiyonundaki `long_event_ratio`
parametresini (varsayılan 0.4) ayarlayabilirsin.

## Şeffaflık Notu

Üretilen örneklerin büyük çoğunluğu, az sayıda gerçek kayıttan türetilmiş
**augmentasyon varyantlarıdır**, birbirinden bağımsız yüzlerce farklı olay kaydı değildir.
Bu, literatürde kabul görmüş standart bir veri artırma pratiğidir (bkz. MixUp,
Zhang ve ark. 2017; DAS bağlamında uygulaması: Zhang & Chen 2024), ancak sorumluna
rapor ederken bu ayrımı açıkça belirtmen önemlidir — tıpkı Zhang & Chen'in makalesinde
"50 orijinal + 30 MixUp örneği" şeklinde yaptığı gibi.

Bu pipeline'ın ürettiği veri, GERÇEK DAS verisinin yerini tutmaz — amaç, mimari
geliştirme ve pipeline doğrulama için "üzerinde çalışılabilir" bir başlangıç veri
seti üretmektir. Gerçek saha verisi toplandığında bu sentetik veri, ön eğitim
(pre-training) veya ek veri artırma amacıyla kullanılabilir.
