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

**Çıktı:** `outputs/pretrained/das_2dcnn_sk_v1.pt` (34.835 parametre, 156 KB)

### Aşama 2 — Gerçek saha verisi 🔄 DEVAM EDİYOR

Faz 0 denetimi **bitti**, ön işleme hattı **yazıldı ve test edildi**.
Sırada gerçek eğitim var.

**Taban çizgisi ölçüldü: macro-F1 0.771** (doğrusal sınıflandırıcı, 26 özellik)

---

## 3. OKUMA SIRASI

| # | Dosya | Ne için |
|---|---|---|
| 1 | **`GERCEK_VERI_FAZ0_RAPORU.md`** | **Şu an aktif olan iş.** Gerçek veri denetimi, kararlar (Bölüm 1.5), ölçümler (Bölüm 5.4) |
| 2 | `src/real_data.py` | Ön işleme hattı. Docstring'ler gerekçeleri açıklıyor |
| 3 | `PLAN_2DCNN_SKAttention.md` | Orijinal plan. Metodoloji (Bölüm 2, 5) ve uygulama kararları (Bölüm 6.4) |
| 4 | `outputs/pretrained/MODEL_CARD.md` | Önceden eğitilmiş model: nasıl yüklenir, sınırları |
| 5 | `MODEL_IYILESTIRME_PLANI.md` | Sentetik aşamadaki iyileştirme denemeleri ve ölçülen sonuçları |

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

Sonra:
- PyTorch `Dataset`: `real_data.pencere_yukle` → `spektrogram` → 224×320
- `load_pretrained(model, bundle)` ile aktarım, `classifier` sıfırdan
- train_final / val_final / test_final ile eğit

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

### Determinizmi açmayı unutmak

Aynı tohumla iki koşu farklı sonuç veriyordu (GPU'da konvolüsyon geri
yayılımı varsayılan olarak deterministik değil). `train.py` içinde
`set_deterministic()` var, kullan.

---

## 7. AÇIK SORULAR (sorumluya)

1. `.sdf.hdf5` dosyalarında `duration` ile örnek sayısı 2 kat uyuşmuyor
   (`.bin`'de sorun yok). Gerçek frekans nedir?
2. `P`/`S` terminolojisi: dosya öznitelikleri `polarization: 2, port: 1`
   diyor, sorumlu "port" dedi. Rapor metninde hangisi kullanılmalı?
3. Ekibin kodunda üç yanlış yorum satırı var (`# Sadece S bileşeni alınır`
   yazıp `['P']` alması, `20000 # 3 sn` yazması vb.) — bilinçli mi, bakımsızlık mı?

---

## 8. ÇALIŞMA YÖNTEMİ

- **Kararlar dosyalara yazılır, sohbete değil.** Bu belge ve raporlar bu
  yüzden var.
- **Uzak veri için:** kod yazılır → kullanıcı JupyterLab'de çalıştırır →
  çıktı paylaşılır. Kimlik bilgisi asla istenmez.
- **Ölçmeden karar verilmez.** Bu projede en değerli bulguların hepsi
  ölçümden çıktı (birebir kopya dosya, eşzamanlı mikrofon çiftleri, sınıf
  tutarsızlığı, boş pencereler).
- **Test setine bakarak hiperparametre seçilmez.** Kaç konfigürasyon
  denendiği rapora yazılır.
