# Staj Sonu Raporu — DAS Tabanlı Çevre Güvenliği Sınıflandırıcısı

> **Tarih:** 2026-09-04
> **Kapsam:** Projenin tamamı — sentetik aşamadan saha teslimine
>
> Bu belge bir **özet**tir. Ayrıntılar ve ölçüm kayıtları için:
> `DURUM.md` (giriş noktası) · `GERCEK_VERI_EGITIM_SONUCLARI.md` (eğitim ve
> saha) · `GERCEK_VERI_FAZ0_RAPORU.md` (veri denetimi)

---

## 1. PROBLEM

Fiber optik kablo, DAS (Φ-OTDR) ile kilometrelerce uzunlukta sürekli bir
titreşim sensörüne dönüşüyor. Görev: **tel örgü çitlerdeki ihlal
olaylarını sınıflandıran bir derin öğrenme modeli** geliştirmek.

Üç sınıf: `cutting` (kesme), `climbing` (tırmanma), `noise` (gürültü).

Mimari şu makaleden uyarlandı:
> You, J. ve ark. (2025). *DAS-Based Perimeter Intrusion Detection Using
> 2D-CNN With SKAttention Mechanism.* IEEE Sensors Journal, 25(22).

---

## 2. NE YAPILDI

### Aşama 1 — Sentetik veri (tamamlandı)

Kendi ürettiğimiz sentetik spektrogramlarla model geliştirildi.

**Sonuç: macro-F1 0.622 ± 0.166** (4 katlı, kaynak-gruplu çapraz doğrulama)

Düşük görünüyor ama **dürüst**: ~960 spektrogram yalnızca **19 bağımsız
ses kaydından** türetilmişti. Rastgele bölme yapılsaydı %99 görünürdü ve
yalan olurdu. Tavanın sebebi model değil veriydi.

**Bu aşamadan kalıcı olarak taşınanlar:** mimari, ön işleme disiplini,
sızıntı testleri, metodoloji. **Taşınmayan:** ağırlıklar (aktarım işe
yaramadı, aşağıda).

### Aşama 2 — Gerçek saha verisi

| | |
|---|---|
| Toplam pencere | 373.908 (train 293.469 / val 37.559 / test 42.880) |
| Benzersiz oturum | 14.490 |
| Örnekleme frekansı | 2000 Hz |
| Pencere | 15.000 örnek = 7.5 s |

Bölmeler dosya, oturum ve tarih düzeyinde **sıfır çakışma** ile geliyor.

**Taban çizgisi:** doğrusal sınıflandırıcı + 26 elle yapılmış özellik →
**macro-F1 0.771**

**Beş eğitim koşusu yapıldı, hiçbiri test setine bakılarak seçilmedi:**

| # | mimari | girdi | rejim | test macro-F1 |
|---|---|---|---|---|
| **4** | **CNN-BiLSTM** | viridis | yeni | **0.9390** |
| 1 | 2D-CNN+SK | viridis | eski | 0.8843 |
| 3 | 2D-CNN+SK | gri | eski | 0.8737 |
| 5 | 2D-CNN+SK | gri | yeni | 0.8704 |
| 2 | 2D-CNN+SK | viridis | aktarım | 0.8658 |
| — | *taban çizgisi* | — | — | *0.771* |

**Taban çizgisi +0.168 farkla aşıldı.**

**Kazancın nereden geldiği fark-farkı hesabıyla atfedildi:**

```
mimari (BiLSTM)      +0.058    <- kazancin tamami
rejim degisikligi    -0.003    <- sifir
sentetik aktarim     -0.019    <- ZARARLI
viridis vs gri       +0.011
```

Kazanç tam da hedeflenen yerde toplandı: `cutting` +0.090, `climbing`
+0.066, `noise` +0.008. `climbing`↔`cutting` karışıklığı yarıya indi.

### Teslim: ONNX

Sorumlunun isteği: model ham sinyal alsın. Bu yüzden **ön işlemenin
tamamı grafiğin içine gömüldü**.

```
girdi   sinyal        (batch, 15000)  float32   ham genlik, P alani
cikti   logit         (batch, 3)      float32   [cutting, climbing, noise]
        bosluk_orani  (batch,)        float32
```

opset **13**, IR **7**. Dört ONNX engeli çözüldü (`fft_rfft`, `median`,
adaptive pooling, antialias). Bu, ekibin yaşadığı dört ön işleme
tutarsızlığı riskini de tamamen kapatıyor — çağıran tarafın hiçbir şey
yazmasına gerek yok.

---

## 3. SAHA TESTİ VE ÇÖZÜLEN SORUN

Model MLflow üzerinden ekibin test arayüzüne yüklendi ve waterfall
görselleriyle incelendi. **Ortaya çıkan tablo, test setindeki 0.9390
ile çelişiyordu:** olay olmayan kanallarda yaygın ve kesintisiz yanlış
alarm.

### Teşhis zinciri

**1. `noise` sınıfı "boşluk" demek değil.** Bizim veri setimizde `noise`
**etiketlenmiş bir olay türü** (ortam gürültüsü, araç). Boş pencereler
eğitimden **silinmişti** (%23), `noise` diye etiketlenmemişti.

Sonuç: **modelin "hiçbir şey yok" cevabı yok.** Her pencerede üç sınıftan
birini seçmek zorunda. Ölçüldü: boş pencerelerin **%99.8'inde** bir
saldırı sınıfı seçiyor.

**2. Sebep mimari değil, GİRDİ TEMSİLİ.** İlk teşhis BiLSTM'in zaman
havuzlamasını suçladı; koşu 1 bunu çürüttü:

| koşu | mimari | renk | boş pencerede yanlış alarm |
|---|---|---|---|
| 1 | SK | **viridis** | %99.7 |
| 4 | BiLSTM | **viridis** | %99.7 |
| 3 | SK | **gri** | **%0.0** |
| 5 | SK | **gri** | %0.1 |

İki viridis modeli de başarısız, iki gri modeli de başarılı. **Viridis
bırakıldı** — zaten tek gerekçesi sentetik modelle temsil paritesiydi ve
o aktarım çürütülmüştü.

**3. Boşluk ölçütünün iki kusuru.** Modele "bu pencerede sinyal var mı"
sorusunu soran ikinci bir çıktı vardı (`bosluk_orani`) ama sahada
tetiklenmiyordu:

- **Sıfır güç:** sabit bir sinyalde numpy sürümü 1.0 (=boş), ONNX sürümü
  0.0 (=dolu) döndürüyordu. Tek satırlık tutarsızlık.
- **Ölçüt tersti:** `bosluk_orani` 500 Hz **üstündeki** payı ölçüyor.
  Ölü kanalda yüksek frekans yok, yavaş taban kayması var → oran düşük
  çıkıyor ve pencere "dolu" sayılıyor.

**4. Ham veri formatı çözüldü.** Sorunu ölçmek için 201 kanalın hepsi
gerekiyordu; elimizdeki `.hdf5` kopyaları olay çevresine kırpılmıştı
(6–14 kanal). Ham `.bin` formatı tersine mühendislikle çözüldü ve
kopyayla **birebir doğrulandı** (maks fark 0):

```
baslik  16.384 bayt · uint16 little-endian · ZAMAN-oncelikli (ornek, kanal)
```

Bu, yanlış alarmların **5 kanalda** toplandığını gösterdi (201'in 153'ü
hiç alarm üretmiyor).

### Çözüm

100 Hz **altındaki** enerji payı ikinci ölçüt olarak eklendi ve bastırma
grafiğe gömüldü:

```
bosluk_orani > 0.45  VEYA  dusuk_frek > 0.9084  ->  saldiri logitleri bastirilir
                                                     argmax kendiliginden `noise`
```

Gerçek olayların %95'ini koruyan eşikte artefakt pencerelerinin
**%92.5'i** susturuluyor (eski ölçüt: %20.0).

**Yeniden eğitim gerekmedi.** Çıktı şekli değişmedi, çağıran tarafta
hiçbir değişiklik gerekmiyor.

### Ölçülen sonuç

7 dosya, 201 kanal, 8.040 pencere:

| model | eşik | kaçırma | yanlış alarm |
|---|---|---|---|
| SK gri, bastırmasız | 0.90 | %39.3 | %3.2 |
| **SK gri, bastırmalı** | 0.90 | %40.8 | **%0.5** |
| BiLSTM, bastırmasız | 0.90 | %29.8 | **%55.5** |
| **BiLSTM, bastırmalı** | 0.90 | **%31.2** | %1.4 |

Bastırmasız BiLSTM sahada felaket (%55.5); bastırma onu **40 kat**
toparlıyor. Waterfall'da sürekli yanlış alarm veren kanalların hepsi
temizlendi, gerçek olay tespit edilmeye devam ediyor.

---

## 4. NEREYE KADAR GELİNDİ

✅ **Tamamlanan**

- Taban çizgisi +0.168 aşıldı (0.771 → 0.9390)
- Kazancın kaynağı ölçülerek atfedildi
- Ham sinyal alan, ön işlemesi gömülü ONNX teslimi (opset 13, IR 7)
- Saha yanlış alarm sorunu teşhis edildi ve çözüldü (%55.5 → %1.4)
- Ham veri formatı çözüldü, 201 kanalın hepsi okunabilir
- Üç kalıcı kural belgelendi: alfabetik sınıf sırası, gri temsil,
  ONNX her zaman ham sinyal

⚠️ **Açık kalan**

| # | konu | durum |
|---|---|---|
| 1 | **Kaçırma %25–31** | Asıl kalan sorun. İki hipotez denendi, ikisi de çürütüldü |
| 2 | **Koşu 7: BiLSTM + gri** | Kod hazır, GPU 6 saat boyunca başkası tarafından dolu tutulduğu için hiç başlayamadı |
| 3 | **`.sdf` örnekleme hızı** | 10 dosyada `duration × prf` ile örnek sayısı tam 2 kat uyuşmuyor (gerçek fs 4000 gibi). Sorumluya teyit ettirilmeli |
| 4 | **record_26 anomalisi** | %58.9 kaçırma, hiçbir açıklamaya uymuyor |
| 5 | **Eğitim etiket kalitesi** | Etiketler 224 kanala kadar yayılabiliyor; bazı gruplarda pencerelerin %92–100'ü boş |
| 6 | Teslim paketi (`.pt`) koşu 4'e güncelleme | Kod hazır, sunucuda çalıştırılmadı |

---

## 5. KAÇIRMA HAKKINDA NE BİLİYORUZ

Bu, projenin devredilen ana problemi. Ölçülenler:

**Doğrulanmış gözlemler**

- Kaçırma olayın **kanal kenarlarında merkezin 3 katı** (%37.5 / %31.7
  kenar, %11.7 merkez)
- Kaçırılan pencereler **düşük frekans baskın**: 0.802, yakalananlar
  0.587 — yüksek frekanslar önce sönümlendiği için zayıf/uzak olayların
  imzası
- Geniş olaylar daha çok kaçırılıyor (%15 → %35 → %54)

**Çürütülen hipotezler**

| hipotez | ölçüm |
|---|---|
| "GT bir zaman aralığı, sessiz anları da kapsıyor" | Kaçırmanın yalnızca %17'si. Kaçırmalar gerçek |
| "Eğitimde zayıf pencereleri elemişiz" | Elenenler saf gürültü (`mad` 0.45 vs 27.26), kaçırılanlarla ilgisiz |

**Bu ikisinin çürütülmesi problemi yeniden konumlandırıyor:** model o
pencereleri **gördü**, yine de tanımıyor. Yani sorun veri eksikliği
değil, **modelin temsil/ayırt etme kapasitesi**.

**Ve bir gerilim var:** bastırma ölçütümüz kaçırılan gerçek olaylarla
aynı eksende çalışıyor.

```
yakalanan gercek olay   dusuk_frek 0.587
KACIRILAN gercek olay              0.802
artefakt kanallar                  0.956
```

Eşiği düşürüp daha çok artefakt yakalamak, zayıf gerçek olayları da
susturur. **Yanlış alarm ile kaçırma aynı fiziksel büyüklüğün iki ucu.**

---

## 6. BUNDAN SONRA — YOL HARİTASI

### A. Kısa vade (kod hazır, yalnızca koşturulacak)

1. **Koşu 7: BiLSTM + gri.** Hiç denenmedi. BiLSTM'in kaçırma
   üstünlüğünü (%31.2 vs %40.8) gri temsilin boşluk sağlamlığıyla
   birleştirmesi bekleniyor — bastırma viridis'in zararını tam
   almıyor (BiLSTM y.alarm %1.4, SK %0.5). ~100 dk, GPU gerekir.
2. **Çalışma noktası seçimi.** Eşik eğrisi çıkarıldı; 0.90 üstü **saf
   kayıp** (yanlış alarm zaten dipte, sadece kaçırma artıyor). Sorumluyla
   birlikte 0.75 civarına inmek ölçüme göre iki eksende de daha iyi.
3. **Teslim paketini koşu 4'e güncelle** ve `.sdf` frekans sorusunu
   sorumluya teyit ettir.

### B. Orta vade — sorunun yeni konumuna göre

Kaçırma artık bir **veri** problemi değil, **model temsili** problemi
olarak görünüyor. Bu, daha önce değerlendirip ertelediğimiz yaklaşımları
yeniden anlamlı kılıyor:

**Kosinüs kafa (marjinsiz).** Şu anki son katman düz bir `Linear`; logit
sınırsız bir iç çarpım. Özellik ve ağırlığı normalize edip logiti
`[-1, 1]` aralığında bir **kosinüs benzerliğine** çevirmek, skoru "bu
örnek sınıf prototipine ne kadar yakın" diye okunabilir hâle getirir.
Sınırdaki zayıf örnekler için karar eşiği anlamlı olur. Ucuz, riski
düşük, ilk denenecek adım.

**Açısal marjin (ArcFace / CosFace).** Kosinüs kafanın üstüne bir marjin
ekleyerek sınıfları hiperküre üzerinde sıkı kümeler hâline getirir;
aradaki boşluk büyür. **Sınırdaki zayıf örnekleri ayırt etmek tam da bu
yöntemlerin iddia ettiği şey** — ve bizim kalan sorunumuz tam olarak bu.

⚠️ Çekince: normalizasyon **özellik normunu atıyor** ve OOD
literatüründe norm başlı başına güçlü bir sinyal. Doğru kullanım kosinüsü
sınıflandırma için, reddetme skorunu ayrı tutmak. 3 sınıfla `s`/`m`
ayarı hassas, denemeden tutmaz.

**Segmentasyon çerçevesi.** Pencere pencere sınıflandırmak yerine
(kanal × zaman) ızgarasında **olay maskesi** tahmin etmek. Model
komşuluğu görür ve olayların bitişik bloklar oluşturduğunu kendi
öğrenir — tek kanallı sahte sütunlar kendiliğinden elenir. Etiket
verisi buna uygun (`labels` zaten dikdörtgen veriyor). Büyük iş: mimari
ve veri hattı değişikliği.

**Denenmemesi önerilenler:** Dice loss (segmentasyon kaybı, bizim
problem tipimize uygun değil) ve genel maliyet duyarlı ağırlıklandırma
(sınırı var olan sınıflar arasında kaydırır, "hiçbiri" seçeneği
yaratmaz; etkisi büyük ölçüde eşik değiştirmekle aynı).

### C. Veri tarafı

- **Etiket kalitesi** incelenmeli: etiketlerin 224 kanala yayılması ve
  bazı gruplarda pencerelerin %92–100'ünün boş olması, eğitim sinyalini
  seyreltiyor olabilir.
- **`.sdf` örnekleme hızı** netleşmeli; doğruysa o dosyalarla yapılan
  tüm testler geçersiz.
- **Değerlendirme seti** kürasyonlu: val/test'te boş pencere oranı %0.1,
  train'de %23. Saha koşullarını temsil etmiyor. Gerçekçi bir
  değerlendirme seti kurulmalı.

---

## 7. METODOLOJİK NOTLAR

Bu projede en değerli bulguların hepsi **ölçümden** çıktı ve birçoğu
sezgiyi çürüttü. Kayda geçirilenler:

| tahmin | ölçüm |
|---|---|
| Sentetik ön-eğitim yardım eder | Etmedi (−0.019) |
| Viridis zarar verir | Vermedi (+0.011) — ama sahada felaket |
| Rejim düzeltmesi skoru artırır | Artırmadı (−0.003) |
| BiLSTM'in zaman havuzlaması suçlu | Değil; sebep renk temsili |
| GT sessiz anları kapsıyor | Kaçırmanın %17'si |
| Zayıf pencereleri elemişiz | Elenenler saf gürültü |

**İzlenen ilkeler:** tek kod yolu (eğitim ve çıkarım aynı fonksiyonu
çağırır) · ölçmeden karar verilmez · tahmin ölçümle çeliştiğinde ölçüm
kazanır ve bu yazılır · test setine bakarak hiperparametre seçilmez ·
kararlar dosyalara yazılır, sohbete değil.

**Tekrarlanmaması gereken hatalar** `DURUM.md` Bölüm 6'da toplandı —
aralarında "asla başarısız olamayan doğrulama yazmak", "yanlış
popülasyonda ölçüp *işe yaramıyor* demek", "açılamayan dosyayı yok
saymak" gibi bu projede fiilen yaşanmış on bir madde var.
