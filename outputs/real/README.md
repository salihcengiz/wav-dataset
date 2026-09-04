# `outputs/real/` — Aşama 2 (gerçek saha verisi) çıktıları

Sunucudan indirilen eğitim ve teslim çıktıları.
İndirme: 2026-09-04, kaynak `/tf/start_training/RELATIONNET/FENCE_DATA_NEW/egitim_ciktilari/`

> ⚠️ **`outputs/` kökündeki diğer klasörler Aşama 1'e (sentetik veri) aittir**
> — `checkpoints/`, `figures/`, `folds/`, `pretrained/`, `results/`,
> `metadata.csv`. Karıştırma. Sentetik aşamanın ağırlıkları gerçek veride
> **kullanılmıyor** (aktarım ölçüldü, −0.019 zararlı çıktı).

---

## ⭐ Teslim edilen model

```
onnx/delivered/sk_gri_kosu3_v2.onnx
onnx/delivered/sk_gri_kosu3_v2_KULLANIM.md
```

Koşu 3 (2D-CNN + SK, **gri** girdi temsili), iki ölçütlü boşluk bastırması,
opset 13, IR 7. Girdi `(batch, 15000)` ham genlik; ön işlemenin tamamı
grafiğin içinde.

MLflow: `2026-09-02-MODEL-SK_GRI_V2_PERİMETER`
Saha başarımı (eşik 0.75): kaçırma %35.7, yanlış alarm %0.8

---

## Dizin yapısı

### `checkpoints/` — eğitim artıkları

Beş koşunun `.pt` dosyaları. Bunlar **teslim edilebilir paket değil**,
eğitim döngüsünün ürettiği ham checkpoint'ler (`state_dict` + koşu bilgisi).

| dosya | koşu | mimari · girdi | test macro-F1 |
|---|---|---|---|
| `kosu1_viridis_sifirdan.pt` | 1 | 2D-CNN+SK · viridis | 0.8843 |
| `kosu2_viridis_aktarim.pt` | 2 | 2D-CNN+SK · viridis · aktarım | 0.8658 |
| `kosu3_gri_sifirdan.pt` | 3 | 2D-CNN+SK · **gri** | 0.8737 |
| `kosu4_bilstm_yeni_rejim.pt` | 4 | **CNN-BiLSTM** · viridis | **0.9390** |
| `kosu5_gri_sifirdan_yeni_rejim.pt` | 5 | 2D-CNN+SK · gri · yeni rejim | 0.8704 |

Teslim edilen ONNX, **koşu 3**'ün ağırlıklarını taşır. Koşu 4 test setinde
daha yüksek skorlu ama sahada yanlış alarmı üç katı — gerekçe
`GERCEK_VERI_EGITIM_SONUCLARI.md` Bölüm 8'de.

### `checkpoints/interface/` — ⚠️ kökeni doğrulanmadı

`*_arayuz.pt` son ekli **dört** dosya: `kosu1`, `kosu2`, `kosu4`, `kosu5`.
Boyutları asıl checkpoint'leriyle aynı.

⚠️ Burada eksik olan tek şey `kosu3_gri_sifirdan_arayuz.pt`'dir — yani
**bu varyantın** koşu 3 için olanı. Koşu 3'ün **asıl checkpoint'i**
(`kosu3_gri_sifirdan.pt`, teslim edilen modelin ağırlıkları) bir üst
dizinde, `checkpoints/` altında **mevcuttur**.

**Bu dosyaların ne olduğu bu oturumda doğrulanmadı** — daha önceki bir
oturumda, muhtemelen test arayüzü için üretilmiş olabilirler. Kullanmadan
önce içerikleri incelenmeli. Ayrı klasöre alınmalarının sebebi bu.

### `package/` — teslim paketi (`.pt`)

`gercek_export.py`'nin ürettiği, kendi kendine yeten paket: ağırlıklar +
mimari + ön işleme tarifi + sınıf sırası + ölçülen performans + sınırlar.

⚠️ **Bu paket hâlâ koşu 3'ün ve `.pt` biçiminde — yani `(3, 224, 320)`
spektrogram GÖRÜNTÜSÜ alır, ham sinyal değil.** Ham sinyal için `onnx/`
altındaki dosyalar kullanılır. `gercek_export.py` koşu 4'ü de paketleyecek
şekilde güncellendi ama sunucuda çalıştırılmadı.

### `onnx/delivered/` — kullanılacak dosya

Yukarıda. Başka bir şey koyma.

### `onnx/archive/` — önceki sürümler

| dosya | ağırlık | bastırma | not |
|---|---|---|---|
| `sk_gri_kosu3.onnx` | koşu 3 | **yok** | ilk gri teslim |
| `sk_gri_kosu3_bastirmali.onnx` | koşu 3 | **v1** | ⚠️ **KULLANMA** — bastırması var ama çalışmıyor |
| `bilstm_kosu4.onnx` | koşu 4 | yok | sorumluya ilk teslim |
| `bilstm_kosu4_v2.onnx` | koşu 4 | v2 | üretildi, teslim edilmedi |
| `bilstm_kosu4_opset17_yedek.onnx` | koşu 4 | yok | opset 13'e geçmeden önceki yedek |

⚠️ **`sk_gri_kosu3_bastirmali.onnx` en yanıltıcı dosya:** bastırma kodu
içinde ama iki kusur yüzünden sahada hiç tetiklenmiyor (sıfır güç koruması
yok, ölçüt ölü kanalda ters çalışıyor). Sahada denendi ve kanal 0 hâlâ
`cutting` verdi; kusurlar ondan sonra bulundu. `_v2` bunların düzeltilmiş
hâlidir.

---

## ⚠️ Bu indirmede OLMAYAN dosyalar

`*_gecmis.json` — her koşunun eğitim geçmişi (epoch bazında kayıp/doğruluk,
karışıklık matrisi, test skorları). Sunucuda `egitim_ciktilari/` altında
duruyor.

Bunlar olmadan çalışmayan iki şey var:

- `gercek_export.py` — paketi kurarken performans sayılarını oradan okuyor
- `gercek_rapor.py` — öğrenme eğrilerini ve karşılaştırma grafiklerini
  oradan üretiyor

İleride bu grafikler yeniden üretilecekse `.json` dosyaları da indirilmeli.

---

## İlgili belgeler

| dosya | ne için |
|---|---|
| `STAJ_SONU_RAPORU.md` | Devir özeti — ne yapıldı, ne eksik, yol haritası |
| `GERCEK_VERI_EGITIM_SONUCLARI.md` | Beş koşu, atıf hesabı, ONNX, saha testi |
| `DURUM.md` | Giriş noktası, kod haritası, tekrarlanmaması gereken hatalar |
