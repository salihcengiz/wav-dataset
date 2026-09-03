# Gerçek Veri — Eğitim Sonuçları

> Aşama 2'nin eğitim kaydı. **Beş koşu**, her biri önceden tasarlandı ve
> test setine bakmadan çalıştırıldı.
>
> **En iyi: CNN-BiLSTM, test macro-F1 0.9390** (taban çizgisi 0.771, **+0.168**)
>
> **Son güncelleme:** 2026-09-01

---

## 1. BEŞ KOŞU — ANA TABLO

| # | mimari | girdi | başlangıç | rejim | val | **test macro-F1** | en iyi/koşulan epoch |
|---|---|---|---|---|---|---|---|
| **4** | **CNN-BiLSTM** | viridis | sıfırdan | **yeni** | 0.9367 | **0.9390** | 19 / 29 |
| 1 | 2D-CNN+SK | viridis | sıfırdan | eski | 0.8719 | 0.8843 | 39 / 40 (tavan) |
| 3 | 2D-CNN+SK | gri | sıfırdan | eski | 0.8657 | 0.8737 | 23 / 29 |
| 5 | 2D-CNN+SK | gri | sıfırdan | **yeni** | 0.8617 | 0.8704 | 21 / 31 |
| 2 | 2D-CNN+SK | viridis | **aktarım** | eski | 0.8596 | 0.8658 | 18 / 24 |
| — | *doğrusal, 26 özellik* | — | — | — | *0.771* | — | — |

**Ortak ayarlar:** tohum 42, batch 64, Adam lr 1e-3, weight decay 1e-4,
label smoothing 0.1, izlenen metrik **val macro-F1**, determinizm açık,
train 220.834 / val 37.517 / test 42.850 pencere.

**Eski rejim:** maskeleme p=0.5, maks 40 epoch, erken durdurma sabrı 6.
**Yeni rejim:** maskeleme **kapalı**, maks 80 epoch, sabır 10.

### Sınıf bazında (test)

| sınıf | taban | koşu 2 | koşu 5 | koşu 3 | koşu 1 | **koşu 4** |
|---|---|---|---|---|---|---|
| cutting | 0.678 | 0.782 | 0.794 | 0.794 | 0.813 | **0.903** |
| climbing | 0.653 | 0.841 | 0.845 | 0.851 | 0.861 | **0.927** |
| noise | 0.981 | 0.975 | 0.972 | 0.977 | 0.979 | **0.987** |
| **macro** | **0.771** | 0.866 | 0.870 | 0.874 | 0.884 | **0.939** |

---

## 2. ATIF — KAZANÇ NEREDEN GELİYOR

Koşu 4 iki şeyi birden değiştiriyordu (mimari + rejim). Koşu 5 rejimi tek
başına izole etmek için koşuldu: koşu 3 ile **aynı mimari, aynı girdi**,
tek fark rejim.

```
rejim etkisi     (kosu 5 - kosu 3)  =  0.8704 - 0.8737  =  -0.0033
mimari + rejim   (kosu 4 - kosu 1)  =  0.9390 - 0.8843  =  +0.0547
                                                            --------
BiLSTM'e kalan   (fark-farki)                            =  +0.0580
```

**Kazancın tamamı mimariden.** Rejim değişikliği sıfır katkı verdi.

> Varsayım: rejim etkisinin gri ile viridis'te benzer olduğu. Temsil farkı
> ölçülmüştü ve küçüktü (viridis − gri = +0.0106), varsayım makul — ama
> varsayım.

### Hipotez desteklendi

Hipotez şuydu: *`cutting` ritmik ve ayrık, `climbing` sürekli ve düzensiz;
`AdaptiveAvgPool2d(1)` bu zamansal yapıyı çöpe atıyor.*

Kazanç tam da hedeflenen çiftte toplandı, `noise`'da değil:

| | koşu 1 | koşu 4 | fark |
|---|---|---|---|
| cutting | 0.813 | 0.903 | **+0.090** |
| climbing | 0.861 | 0.927 | **+0.066** |
| noise | 0.979 | 0.987 | +0.008 |

Karışıklık yarıya indi:

| | koşu 1 | koşu 4 |
|---|---|---|
| `cutting` → `climbing` | 3.675 (%21.1) | **2.084 (%12.0)** |
| `climbing` → `cutting` | 2.517 (%11.5) | **1.147 (%5.2)** |

---

## 3. ⚠️ ÇÜRÜTÜLEN TEŞHİS — KAYDA GEÇMELİ

Koşu 1–3'ten sonra şu teşhis konmuştu: *"Model yetersiz öğreniyor;
maskeleme fazla geliyor, epoch bütçesi yetmiyor. Rejimi düzeltmek skoru
artırır."*

**Koşu 5 bunu ölçüp yalanladı.** Rejim düzeltmesi tek başına **−0.0033**
getirdi, yani hiçbir şey.

Teşhisin **gözlem kısmı doğruydu**: üç koşuda da doğrulama doğruluğu
eğitimin üstündeydi ve sebebi maskelemeydi. Maskeleme kapatılınca ikisi
birbirine yapıştı (koşu 4 ve 5'te doğrulandı).

**Çıkarım kısmı yanlıştı.** Maskeleme yakınsamayı yavaşlatıyordu, tavanı
belirlemiyordu:

| | maskeleme | en iyi epoch | test |
|---|---|---|---|
| koşu 1 | açık | 39 (tavana dayandı) | 0.8843 |
| koşu 5 | kapalı | 21 (kendiliğinden durdu) | 0.8704 |

Maskelemesiz model **daha hızlı** yakınsadı ama **aynı yere**. Tavanı
belirleyen mimariydi.

**Ders:** "eğitim < doğrulama" gözlemi bir handikapın varlığını gösterir,
ama handikapı kaldırmanın performansı artıracağını göstermez. İkisi ayrı
iddia; ikincisi ayrıca ölçülmeli.

---

## 4. DİĞER İKİ SORU (koşu 1–3'ten)

### Sentetik ön-eğitim işe yarıyor mu? ❌ HAYIR

```
kosu 1 (sifirdan)  0.8843
kosu 2 (aktarim)   0.8658      fark -0.0185
```

Aktarımlı model hem düşük bitti hem erken durdu (en iyi epoch 18). Erken
epoch'larda da geride başladı (epoch 2: 0.7521 vs 0.7956) — oysa aktarımın
faydası en çok orada görünmeliydi.

19 bağımsız sentetik kayıttan öğrenilen filtreler, 21.101 gerçek dosyanın
yanında avans sağlamıyor; modelin çıkmak zorunda kaldığı bir kısıt gibi
davranıyor. `MODEL_CARD.md`'nin kendi uyarısı doğrulandı.

**Aşama 1 boşa gitmedi:** mimari, ön işleme disiplini, sızıntı testleri ve
metodoloji oradan geldi. İşe yaramayan tek şey **ağırlıkların aktarımı**.

**Karar: sentetik önceden eğitilmiş model artık kullanılmıyor.**

### Viridis zarar veriyor mu? ❌ HAYIR (tahminin tersi)

```
kosu 1 (viridis)  0.8843
kosu 3 (gri)      0.8737      fark +0.0106 viridis lehine
```

Beklenti viridis'in zarar vermesiydi. Ölçülen özellikleri:

- **Bilgi kaybı yok denecek kadar az:** 256 dB kademesinin **254'ü**
  benzersiz RGB'ye gidiyor (2 kademe çakışıyor).
- **Ama monoton değil:** R kanalında 109 artan / **42 azalan** adım,
  B kanalında 60 artan / **92 azalan**. Yalnızca G monoton (212/0).
  Yani "enerji arttıkça değer artar" ilişkisi iki kanalda yok — sıfırdan
  eğitilen bir ağ için gereksiz zorluk olmalıydı.

Ölçüm tersini söyledi. Fark küçük ama üç sınıfta da aynı yönde.

Tek net gözlem: viridis ile eğitim daha oynak ilerledi ve daha yavaş
yakınsadı (koşu 1, 40 epoch tavanına dayandı).

---

## 5. KAZANAN MİMARİ — CNN + BiLSTM

`CNN-BiLSTM/model_bilstm.py`. Omurga **DASNet ile birebir aynı** (aynı
model nesnesinin `features` ve `attention` modülleri ödünç alınıyor,
kopyalanmıyor). Yalnızca havuzlama başı değişiyor:

```
girdi (B,3,224,320)
  -> features + SK-Attention   [DASNet'ten]  -> (B,64,28,40)
  -> frekansi 4 bine indir                   -> (B,64,4,40)
  -> yeniden duzenle: zaman = dizi           -> (B,40,256)
  -> BiLSTM(256 -> 128, cift yonlu)          -> (B,40,256)
  -> dikkatli zaman havuzlama                -> (B,256)
  -> Dropout(0.5) -> Linear(256,3)           -> (B,3)
```

| | |
|---|---|
| parametre | **430.932** (DASNet'in 12.4 katı) |
| zaman adımı | 40 (188 ms/adım) |
| adım boyutu | 256 (64 kanal × 4 frekans bini) |

**Tasarım gerekçeleri:**

- **Frekans 4 bine indiriliyor, çökertilmiyor.** Ortalama alıp 64 boyuta
  inmek "hangi bantta" bilgisini atardı.
- **Çift yönlü.** Bir darbenin anlamı hem öncesine hem sonrasına bağlı.
- **Dikkatli havuzlama, son gizli durum değil.** Pencereler enerji
  merkezine göre kırpılıyor, olay ortada; son durum sona ağırlık verirdi.
  Yan fayda: `forward_features(x, return_dikkat=True)` ile dikkat
  ağırlıkları alınabiliyor — "model neye bakıyor" grafiği çıkarılabilir.
- **40 zaman adımı.** Omurga DASNet ile aynı kalsın diye. 80 adıma çıkaran
  `zaman_havuzlama=False` seçeneği var ama **varsayılan kapalı** — açılırsa
  omurga DASNet'ten ayrılır ve karşılaştırma kirlenir.

---

## 6. KALAN BELİRSİZLİK — KAPASİTE Mİ, ZAMAN MI?

BiLSTM aynı zamanda **12 kat daha büyük**. Kazancın zamansal modellemeden
mi yoksa sadece kapasiteden mi geldiğini **ayıramıyoruz**.

Ayıracak koşu (**koşu 6, henüz yapılmadı**): SK modelini
`config.CONV_CHANNELS = (32, 64, 128)` ile büyütüp yeni rejimle koşmak
(~124.000 parametre).

| sonuç | anlamı |
|---|---|
| Geniş SK ≈ BiLSTM | Kazanç **kapasiteden**, zamansal modelleme gereksiz |
| Geniş SK < BiLSTM | Kazanç **zamansal yapıdan** — hipotez tam doğrulanır |

Komut:
```
python gercek_egitim.py --kosu 5 --maske-p 0 --epoch 80 --sabir 10
# + config.CONV_CHANNELS degistirilmis olmali; ayri bir kosu numarasi verin
```

---

## 7. METODOLOJİK ÇEKİNCELER

Rapora yazılmalı:

1. **Her konfigürasyondan tek koşu, tek tohum.** 0.0033 ve 0.0106 gibi
   küçük farklar tohum varyansından ayrılamaz. **0.058'lik mimari farkı
   ise bu aralığın çok üstünde** — ama yine de tek koşu.
2. **Koşu 1 kesildi.** 40 epoch tavanına çarptığında hâlâ iyileşiyordu,
   yani 0.8843 onun tavanı değil. Bu, mimari farkını **olduğundan büyük**
   gösteriyor olabilir. Karşı kanıt: koşu 5 aynı mimariyi 80 epoch
   bütçesiyle koştu ve 21'de yakınsadı — yani SK'nin tavanı ~0.87.
3. **Koşu 2 kıl payı durdu.** Epoch 24'te 0.8590, en iyisi 0.8596 —
   0.0006 farkla iyileşme sayılmadı.
4. **Koşu süreleri karşılaştırılamaz.** GPU paylaşımlı ve yük değişkendi:
   koşu 3 = 103 dk, koşu 1 = 178 dk, koşu 2 = 132 dk, koşu 4 = 104 dk,
   koşu 5 = 79 dk. Doğruluk karşılaştırması etkilenmiyor.
5. **Taban çizgisi gürültülü.** 0.771, ~239 doğrulama örneğiyle ölçülmüştü;
   bizim skorlar 42.850 test örneğiyle.
6. **Denenen konfigürasyon sayısı: 5.** Hiçbiri test setine bakılarak
   seçilmedi.

---

## 8. ONNX DIŞA AKTARIM

`CNN-BiLSTM/onnx_disa_aktar.py`. Sorumlunun isteği: model `(None, 15000)`
girdiyle ONNX'e çevrilsin — yani **ham sinyal** girsin.

Modelimiz spektrogram görüntüsü alıyordu, dolayısıyla **ön işlemenin
tamamı grafiğin içine gömüldü**. Bu aynı zamanda ekibin yaşadığı dört
tutarsızlık riskini (ölçek katsayısı, pencere boyutu, P/S, sessiz pencere)
tamamen kapatıyor.

```
girdi   sinyal        (batch, 15000)  float32   ham genlik, P alani
cikti   logit         (batch, 3)      float32   [cutting, climbing, noise]
        bosluk_orani  (batch,)        float32
```

### Aşılan dört engel

| engel | çözüm | doğrulama |
|---|---|---|
| `aten::fft_rfft` (boşluk oranı) | Tam pencere FFT yerine STFT'den kestirim | sapma **0.0016**, eşik 0.45 |
| `aten::median` | Sıralama tabanlı medyan (`np.median` davranışı) | spektrogram farkı 1.3e-03 dB |
| `AdaptiveAvgPool2d((4,None))` | Sabit çekirdekli `avg_pool2d(7,1)` | **birebir aynı** (0.0) |
| `antialias=True` desteklenmiyor | Kapatıldı | **fark 0.0** — büyütmede etkisiz |

`model_bilstm.py`'deki havuzlama değişikliği ağırlıkları etkilemiyor
(havuzlamanın parametresi yok) ve çıktısı birebir aynı.

### Doğrulama sonuçları (gerçek ağırlıklarla, sunucuda)

```
PyTorch vs ONNX  logit farki : 3.70e-06        (opset 13, 2026-09-01)
bosluk farki                 : 2.98e-07
dinamik batch 1 / 3 / 16 / 64: 1.3e-05 / 1.8e-05 / 7.4e-05 / 1.3e-04
```

⚠️ Dinamik batch farkları koşudan koşuya değişir: o döngü her çağrıda
**yeni rastgele girdi** üretiyor (sabit tohum yok) ve batch büyüdükçe
float32 birikim hatası büyüyor. Eşik `1e-3`, logitler ~1-10 mertebesinde,
yani bağıl hata ~1e-5 — `argmax` ancak birebir eşitlikte değişir.
Önceki kayıt (opset 17, ≤3.7e-05) farklı bir çekilişti, regresyon değil.

LSTM için "dinamik batch hata verebilir" uyarısı geldi ama **ölçüldü,
sorun yok** — dizi uzunluğu sabit (40 adım).

### ✅ BOŞ PENCERE BASTIRMASI GRAFİĞE GÖMÜLDÜ (2026-09-01)

Ölçüm (Bölüm 8c) şunu gösterdi: **gri model bile** boş pencerelerde
`argmax`'ı %99.8 oranında bir saldırı sınıfına veriyor — sadece düşük
güvenle. Eşik uygulanırsa kurtuluyor, uygulanmazsa kurtulmuyor. Ve bir
sınıflandırıcıyı kullanmanın varsayılan yolu `argmax`.

Kök sebep: `noise` **etiketlenmiş bir olay türü**, "boşluk" değil. Boş
pencereler eğitimden silindi, `noise` diye etiketlenmedi — modelin susma
cevabı yok.

**Çözüm, yeniden eğitim olmadan:** grafik `bosluk_orani`'nı zaten
hesaplıyor. Eşiği aşıyorsa saldırı sınıflarının logitinden büyük bir
sabit düşülüyor, `argmax` kendiliğinden `noise`'a düşüyor.

```
gecerli = (bosluk_orani <= 0.45)             -> 1.0 / 0.0
logit   = logit - (1 - gecerli) * maske * 1e4
                              maske = [1, 1, 0]   (noise dokunulmaz)
```

Koşullu dal **yok** — ONNX'te `if` olmadığı için saf aritmetik:
`LessOrEqual → Cast → Mul → Sub`. Hepsi opset 13'te mevcut.

**Çıktı şekli değişmedi** (`batch, 3`), yani çağıran tarafta hiçbir
değişiklik gerekmiyor. Dosyayı değiştirip göndermek yeterli.

**Ölçülen davranış:**

| girdi | `bosluk_orani` | tahmin | logit |
|---|---|---|---|
| düz gürültü | 0.50 | **`noise`** | `[-1e4, -1e4, 0.05]` |
| yapılı sinyal | 0.12 | normal | dokunulmadı |

⚠️ **İki doğrulama eklendi** — ikisi de başarısız olabilir:
- `bastirma_dogrula()` — PyTorch tarafında her iki yönü de sınıyor
  (boşta bastırılıyor mu, doluda dokunulmuyor mu)
- `disa_aktar()` içinde **ONNX'e ayrıca boş pencere veriliyor.** İhracat
  örneği yapılı bir sinyal olduğu için bastırma dalı izleme sırasında
  tetiklenmiyor; `do_constant_folding` geçerlilik bayrağını sabitleseydi
  ONNX asla bastırmaz ve PyTorch tarafı doğru göründüğü için fark
  edilmezdi. Ölçüldü: **sabitlenmemiş**, ONNX de bastırıyor.

⚠️ Test sinyalleri de değişti (`ornek_sinyal`): eskiden düz gürültüydü,
`bosluk_orani ≈ 0.5` çıkıyor ve bastırma tetikleniyordu — doğrulama
`-1e4` ile `-1e4`'ü karşılaştırıp **asla başarısız olamayacak** hâle
gelirdi. Artık yapılı sinyal kullanılıyor (`bosluk ≈ 0.12`).

### 🔑 BASTIRMA SAHADA TETİKLENMEDİ — iki kusur bulundu (2026-09-02)

Bastırma laboratuvarda çalışıyordu ama waterfall'da kanal 0 hâlâ 30
saniye kesintisiz `cutting` veriyordu. Detailed Test: logitler
`+1.000 / −0.852 / −0.684`, yani `-1e4` yok — **tetiklenmemiş**.

Ham `.bin` formatı çözülüp 201 kanalın hepsi okununca sebep ölçüldü
(`src/ham_analiz.py`, 8.040 pencere). **Yanlış alarm 5 kanalda toplanmış**
(201'in 153'ü hiç alarm üretmiyor):

| kanal | alarm% | tutarlı% | baskın | `bosluk` | `dusuk_frek` | MAD |
|---|---|---|---|---|---|---|
| 5 | 100.0 | 100.0 | climbing | **0.007** | 0.957 | 829.5 |
| 0 | 100.0 | 100.0 | cutting | **1.000** | 1.000 | **0.0** |
| 4 | 92.5 | 100.0 | climbing | **0.014** | 0.931 | 404.1 |
| 63 | 62.5 | 87.5 | cutting | **0.011** | 0.952 | 689.6 |
| 64 | 32.5 | 82.5 | cutting | **0.012** | 0.941 | 677.6 |

**Kusur 1 — sıfır güç, iki uygulamada zıt sonuç.**
Kanal 0'ın MAD'i tam **0**, yani sinyal sabit. Normalizasyon
`(x−medyan)/(0+1e-9) = 0` veriyor, STFT'si sıfır.

```python
real_data.bosluk_orani :  if top <= 0: return 1.0    # BOS  ✓
bosluk_orani_stft      :  0 / 1e-12 = 0.0            # DOLU ✗
```

Tek satır. `torch.where(top > 1e-12, oran, 1.0)` ile numpy'la
hizalandı — `Where` opset 9'dan beri var, 13'te sorunsuz.

**Kusur 2 — ölçüt diğer dört kanalda TERS çalışıyor.**
`bosluk_orani` = 500 Hz **üstündeki** pay. Ölü/zayıf kanalda yüksek
frekans yok, yavaş taban kayması var → oran **0.007–0.014**, yani
tablodaki en düşük değerler. Ölçüt onları "en dolu" pencereler sayıyor.

Çözüm: **100 Hz altındaki pay** ikinci ölçüt olarak eklendi.

| ölçüt | artefakt | gerçek olay | eşik | **elenen** |
|---|---|---|---|---|
| **`dusuk_frek`** | 0.956 | 0.655 | 0.9084 | **%92.5** |
| `bosluk_orani` | 0.209 | 0.105 | 0.2506 | %20.0 |
| `mad` | 520 | 400 | 266 | %20.5 |
| `std` | 778 | 594 | 398 | %20.0 |

Gerçek olayların %95'i korunurken artefakt pencerelerinin **%92.5'i**
susturuluyor. Bastırma koşulu artık:

```
bosluk_orani > 0.45   VEYA   dusuk_frek > 0.9084
```

⚠️ Bu ölçüt daha önce "işe yaramaz" görünmüştü — çünkü kırpılmış
kanallarda ölçülmüştü ve orada model **zaten doğruydu**. Gerçek
popülasyonda tablo tersine döndü. **Yanlış popülasyonda ölçüm yapmak,
ölçüm yapmamaktan daha yanıltıcı.**

**Doğrulama üç durumda birden** (`bastirma_dogrula` + ONNX içinde):

| durum | `bosluk` | `dusuk_frek` | sonuç |
|---|---|---|---|
| düz gürültü | 0.507 | 0.103 | bastırıldı |
| **ölü kanal** (sabit) | **1.000** | 1.000 | bastırıldı |
| **sürüklenme** | **0.0002** | **0.9997** | bastırıldı |
| yapılı sinyal | 0.124 | 0.026 | dokunulmadı |

Sürüklenme satırı kritik: `bosluk_orani` 0.0002, yani **eski ölçüt
tamamen kördü**.

⚠️ Test sinyali `ornek_sinyal` de değişti: eskiden 20–185 Hz idi, yani
enerjisi tamamen 100 Hz altındaydı ve **yeni ölçüt onu bastırırdı** —
"yapılı sinyale dokunulmuyor" kontrolü asla geçemezdi. Artık 120–450 Hz.

### Gerçek takas eğrisi (201 kanal, 7.768 GT-dışı pencere)

```
esik   KACIRMA   Y.ALARM
0.00    15.4%     62.0%
0.50    24.6%      8.5%
0.75    34.2%      4.2%
0.90    39.7%      3.2%   <- arayuzun calisma noktasi
1.50    54.8%      1.2%
```

Önceki ölçüm (53 pencere, kırpılmış kanallar) yanlış alarmı **%0.0**
gösteriyordu. Gerçek sayı **%3.2**.

### ✅ SONUÇ ÖLÇÜLDÜ — eğri aşağı-sola kaydı

Aynı 8.040 pencerede iki model yan yana (`ham_analiz.py`, tek geçiş):

| eşik | bastırmasız kaçırma | bastırmasız y.alarm | **bastırmalı kaçırma** | **bastırmalı y.alarm** |
|---|---|---|---|---|
| 0.50 | 24.6% | 8.5% | **27.2%** | **2.0%** |
| 0.75 | 34.2% | 4.2% | **35.7%** | **0.8%** |
| 0.90 | 39.7% | 3.2% | **41.2%** | **0.5%** |
| 1.10 | 45.6% | 1.9% | **45.6%** | **0.3%** |

**Aynı eşikte (0.90):** kaçırma +1.5 puan, yanlış alarm −2.6 puan.
Kaldırılan her 1 puan yanlış alarm başına **0.56 puan** kaçırma bedeli.

**Ama asıl kazanç eğrinin kendisinde:**

```
bugunku (bastirmasiz @ 0.90)  ->  kacirma 39.7%   y.alarm 3.2%
bastirmali @ 0.50             ->  kacirma 27.2%   y.alarm 2.0%
                                  HER IKI EKSENDE DE DAHA IYI
```

Bastırma sadece takas yapmadı, **eğriyi aşağı-sola kaydırdı** — sınıf
ağırlığı / yeniden eğitim tartışırken hedeflenen kayma, tek bir ölçüt
düzeltmesiyle geldi. Eşik 1.10'da bastırma tamamen **bedava** (kaçırma
iki modelde de %45.6, yanlış alarm 1.9% → 0.3%).

**🔒 Önerilen çalışma noktası: bastırmalı model + eşik 0.75.**
Bugünküne göre kaçırma −4.0, yanlış alarm −2.4 puan.

⚠️ Çekince: GT-içi örneklem **272 pencere**, 7 dosyadan. Yanlış alarm
tarafı sağlam (7.768 pencere) ama kaçırma tahmini gürültülü olabilir.

### Saha doğrulaması (waterfall, record_26)

Bastırmalı model yüklenip aynı test tekrarlandı:
**kanal 0, 4, 5, 63, 64 sütunlarının hepsi kayboldu.** Kalan yanlış alarm
birkaç izole hücre (ch ~57, ~104, ~195). Gerçek olay kutusu (ch 90–97)
tespit edilmeye devam ediyor.

⚠️ **Bedeli:** `noise` artık iki şeyi temsil ediyor — etiketlenmiş
gürültü olayı **ve** boş pencere. Ayırmak isteyen `bosluk_orani`'na
bakabilir, değer hâlâ ikinci çıktı olarak veriliyor.

`--bastirma-kapat` ile eski davranışa dönülebilir.

### ⚠️ Kullanım şartları

- **Başka ön işleme uygulanmamalı** — hepsi içeride
- **`/16384` ölçek katsayısı uygulanmamalı**
- ~~`bosluk_orani > 0.45` olan pencerelerin tahmini kullanılmamalı~~ —
  **artık grafiğin içinde hallediliyor** (yukarıya bak). `argmax`
  doğrudan kullanılabilir; boş pencerelerde zaten `noise` döner.
  `bosluk_orani` çıktısı bilgi amaçlı duruyor

Çıktı: `egitim_ciktilari/paket/bilstm_kosu4.onnx` (2.1 MB) +
`bilstm_kosu4_KULLANIM.md` — ✅ **ikisi de üretildi** (2026-09-01 doğrulandı).
Kart `--ckpt` ile koşulduğu için performans tablosu dolu; sayılar
checkpoint ve `gecmis.json`'dan geliyor.

---

## 8c. ⚠️ SAHA TESTİ — macro-F1 YETMEDİ (2026-09-01)

Sorumlu BiLSTM'i kendi hattına taktı ve **MLflow waterfall görselleriyle**
inceledi. Hat doğru çalışıyor (ONNX tarafında sorun yok, ham sinyal
uyumlu). Ama sonuç beklenenden kötü:

> **Saldırı sınıfları ile `noise` olması gerekenden çok daha fazla
> birbirine karışıyor. Özellikle KENAR KANALLARDA.**

Bu, test setindeki `noise` F1 0.987 ile **çelişiyor gibi görünüyor**.
Çelişki değil — iki farklı şeyi ölçüyorlar.

### Neden bu, önceden yazdığımız sınırın tam olarak gerçekleşmesi

Kendi model kartımız bunu **öngörmüştü**:

> *"val/test bölmeleri kürasyonlu görünüyor (boş pencere oranı train'de
> %23, val/test'te %0.1). Saha koşullarında zayıf kanallar daha sık
> olacaktır."*

Ve `real_data.bos_mu` docstring'i şunu diyor:

> *"olay etiketli pencerelerin ~%25'i spektral olarak boş... Bunlar
> muhtemelen olaydan uzak kanallar — labels tablosu kanal ARALIĞI veriyor,
> aralık kenarındaki kanallarda sinyal zayıflamış olabilir."*

**Kenar kanal = zayıf sinyal = val/test'ten ayıklanmış pencere tipi.**
0.9390, bu pencerelerin bulunmadığı bir dağılımda ölçüldü. Saha o
pencereleri de içeriyor.

### Sınanacak hipotezler (hiçbiri henüz ölçülmedi)

| # | hipotez | nasıl sınanır |
|---|---|---|
| **1** | **`bosluk_orani > 0.45` filtresi saha testinde uygulanmıyor.** Model bu pencereler için eğitilmedi; filtresiz çalıştırılırsa kenar kanallarda anlamsız ama kendinden emin tahmin üretir | MLflow hattında filtrenin uygulanıp uygulanmadığını sor. **İlk bakılacak yer bu** |
| 2 | BiLSTM zayıf pencerelere SK'den **daha kırılgan**. Dikkatli zaman havuzlama tek bir sahte darbeye kilitlenebilir; `AdaptiveAvgPool2d(1)` onu ortalamaya karıştırıp söndürür | SK'yi de ONNX'e çevirip aynı waterfall'da karşılaştır — **sorumlunun istediği bu** |
| 3 | Kenar kanallarda etiketin kendisi gürültülü (kanal aralığının kenarı olayı zar zor görüyor ama "olay" etiketli) | `bosluk_orani`'na göre kanal-içi konumla çapraz tablo |

### ✅ ÖLÇÜLDÜ (2026-09-01, Detailed Test) — model "hiçbir şey yok" diyemiyor

İki pencere elle incelendi (çıktılar **ham logit**; arayüz 100 ile çarpıp
`%` gösteriyor — `161.7%` ve `−159.9%` bunu kanıtlıyor):

| pencere | gerçek | cutting | climbing | noise | modelin dediği |
|---|---|---|---|---|---|
| A | `climbing` | **+1.617** | −0.800 | −1.599 | `cutting` — emin ve **yanlış** |
| B | **boş** | −0.354 | **+0.808** | −1.110 | `climbing` — **hiç olay yok** |

**B satırı teşhisi veriyor:** pencere boş, ama modelin `noise` skoru
−1.110. Yani model boş bir pencerede `noise` seçeneğini *aktif olarak
reddediyor*.

**Sebep, eğitim tasarımının doğrudan sonucu:**

- `noise`, CSV'de **etiketlenmiş bir olay türü** (ortam gürültüsü, araç).
  "Pencerede bir şey yok" demek **değil**.
- Boş pencereler eğitimden **silindi** (`bos_mu`, train'in %23'ü),
  `noise` diye etiketlenmedi.

→ **Modelin "hiçbir şey yok" cevabı yok.** Üç olay sınıfı var ve her
pencerede birini seçmek zorunda. Boş pencere verilirse en çok neye
benziyorsa onu der, ve emin bir tonla der.

**Waterfall'daki her şey bundan çıkıyor:**

| gözlem | açıklama |
|---|---|
| Kanal 0 baştan sona `cutting` (30 s kesintisiz) | Ölü/zayıf kanal. Model susamıyor. Kanalın gürültü karakteri sabit olduğu için hep aynı sınıf |
| `climbing` ızgaranın her yerine dağılmış | Boş pencereler; bir şey seçmek zorunda |
| `noise` neredeyse hiç seçilmiyor | Çünkü `noise` sessizlik değil, bir olay türü |

**Mekanizma:** normalizasyon pencere-içi medyan/MAD. Ölü bir kanalda
gerçek sinyal yok, sadece gürültü var — normalizasyon o gürültüyü birim
ölçeğe kadar **büyütüyor** ve gerçek bir olay kadar "belirgin" görünen
anlamsız bir spektrogram çıkıyor.

### İki ayrı problem

| | ne | çözüm |
|---|---|---|
| **A** | Model susamıyor (boş pencerede emin tahmin) | **Yeniden eğitim DEĞİL.** `bosluk_orani > 0.45` olan pencereler modele hiç sorulmamalı. Model bu değeri zaten ikinci çıktı olarak üretiyor; **saha arayüzü okumuyor** |
| **B** | `climbing` ↔ `cutting` karışması | Bilinen, çözülmemiş sorun. Satır A'daki örnek: gerçek `climbing`, model +1.617 ile `cutting` diyor |

A'yı düzeltmek B'yi çözmez ama waterfall'ı okunabilir kılar — şu an B,
A'nın gürültüsü altında görünmüyor.

⚠️ **Sınıf eşiği (`CLASS THRESHOLDS`) A'nın çözümü değildir.** O, modelin
*çıktısına* uygulanan bir karar eşiği; `bosluk_orani` ise *girdinin*
geçerli olup olmadığını söylüyor. Eşiği düşürmek durumu kötüleştirir.
Gözlem: eşik 0.90 iken B penceresinin kazanan logiti 0.808, yani eşiğin
**altında** — eşik ham logite uygulanıyorsa o hücre waterfall'da
renklenmez. Doğrulanmalı.

### ✅✅ ÖLÇÜLDÜ (2026-09-01, `src/bos_pencere_testi.py`) — HİPOTEZ 2 DOĞRULANDI

Eğitim CSV'sinden **5.000 pencere** örneklendi, `bos_ele=False` ile
yüklendi ve `bosluk_orani > 0.45` olanlar ayrıldı. Boş oranı **%22.0** —
önbellek kurulumundaki %23 ile örtüşüyor (bağımsız doğrulama).

**Dört koşu, boş pencerelerde (model bunlar için EĞİTİLMEDİ):**

| koşu | mimari | **renk** | boş maks logit | dolu/boş ayrım | **logit>0.9 saldırı** |
|---|---|---|---|---|---|
| 1 | SK | **viridis** | **+1.943** | **0.80×** | **%99.7** |
| 4 | BiLSTM | **viridis** | +1.771 | 1.14× | **%99.7** |
| 3 | SK | **gri** | **+0.858** | **5.18×** | **%0.0** |
| 5 | SK | **gri** | +1.168 | 4.47× | %0.1 |

### 🔑 SEBEP MİMARİ DEĞİL, RENK TEMSİLİ

İki **viridis** modeli de başarısız, iki **gri** modeli de başarılı.
Mimari (SK ↔ BiLSTM) bu davranışla **ilgisiz**. Rejim de ilgisiz
(koşu 5, farklı rejim, gri ile aynı davranıyor).

Koşu 1 hepsinden kötü: ayrımı **0.80×**, yani boş pencerede dolu
pencereden **daha emin**. `>1.5` eşiğinde bile %99.6.

⚠️ **İlk teşhis yanlıştı.** Koşu 3 ↔ koşu 4 karşılaştırmasına bakıp
"BiLSTM'in dikkatli zaman havuzlaması boş pencerede bir adıma ağırlık
vermek zorunda" denmişti. Koşu 1 bunu çürüttü: aynı havuzlamayı
kullanmayan SK, viridis ile aynı — hatta daha kötü — davranıyor.
Karıştırıcı değişkeni ayırmadan mekanizma uydurmanın bedeli.

**Desen:** dört model de boş pencerede `climbing` diyor (%74–99.7).
Fark **güvende**: gri'de logit sıfıra çöküyor (+0.26), viridis'te gerçek
olay seviyesinde kalıyor (+1.6 ~ +1.8).

**Mekanizma hipotezi (ölçülmedi):** boş pencere, normalizasyon sonrası
neredeyse düz bir dB alanı veriyor. Gri'de düz alan → üç kanalda aynı
sabit → konvolüsyon yığını sönümlü tepki veriyor. Viridis'te düz bir dB
değeri **belirgin ve renkli bir sabite** eşleniyor; ilk katman
(Conv 3→16) bunu güçlü ve tutarlı bir örüntü olarak görüyor.

### Bu neyi açıklıyor

1. **Sorumlunun saha gözlemini.** Kenar kanallar = zayıf/boş pencereler.
   Sahada kullanılan model koşu 4 = **viridis** = ayırt edemeyen taraf.
2. **macro-F1'in neden yanılttığını.** 0.9390, boş pencerelerin
   **elendiği** bir test setinde ölçüldü. Saha onları da içeriyor.
3. **Teslim ettiğimiz iki modelin neden farklı davranacağını.**
   `sk_gri_kosu3.onnx` **gri** → sağlam. `bilstm_kosu4.onnx` **viridis**
   → sorunlu.

### 🔒 KARAR — VİRİDİS BIRAKILIYOR

Viridis'in tek gerekçesi, sentetik önceden-eğitilmiş modelle **temsil
paritesiydi** (`gercek_veri_kumesi.py`, "NEDEN VIRIDIS"). Koşu 2
aktarımın işe yaramadığını gösterdi ve *"sentetik model artık
kullanılmıyor"* kararı alındı — **yani gerekçe zaten düşmüştü.**
Kazancı +0.011 macro-F1'di; bedeli şimdi ölçüldü ve çok ağır.

**Bundan sonraki tüm eğitimler `renk="gri"`.**

### Kalan karar

- **`bosluk_orani > 0.45` filtresi saha hattında yine de ZORUNLU.**
  Gri modeller sağlam ama filtre ilkeli çözüm ve her modelde çalışır.
- **Model seçimi yalnızca macro-F1 ile yapılamaz.** Karşılaştırmalara
  "boş pencerede ayrım" ölçütü eklenmeli (`src/bos_pencere_testi.py`).
- **Koşu 7 değerli hâle geldi: BiLSTM + gri.** Hiç denenmedi. BiLSTM'in
  +0.065 kazancı mimariden geliyorsa, gri BiLSTM hem yüksek doğruluk hem
  boşluk sağlamlığı verebilir.

### Ders

**Tek bir toplam skor (macro-F1) dağılım kaymasını göstermez.** 0.9390 ile
0.8737 arasındaki 0.065'lik fark gerçek ama, *kürasyonlu bir test setinde*
gerçek. Sorumlunun görsel incelemesi bizim sayımızın ölçmediği bir şeyi
ölçtü.

→ Bundan sonra model karşılaştırmalarına **kenar kanal / düşük sinyal
dilimi** ayrı bir kesit olarak eklenmeli. Elimizde ölçüt de var:
`bosluk_orani`. Test setini `bosluk_orani` yüzdelik dilimlerine bölüp
her dilimde ayrı macro-F1 raporlamak, bu tartışmayı sayıya çevirir.

---

## 8d. 🔒 KURAL — SINIF SIRASI ALFABETİK (2026-09-01'den itibaren)

**Bundan sonraki TÜM eğitimlerde sınıflar alfabetik sıraya konur.**

```
0 = climbing
1 = cutting
2 = noise
```

**Eski modeller hariç** — koşu 1-5 ve sentetik aşama, aşağıdaki eski
sırayla eğitildi ve o hâlleriyle geçerli:

```
ESKI (kosu 1-5):  0 = cutting,  1 = climbing,  2 = noise
```

Eski sıra `onbellek_kur.SINIFLAR`'dan geliyordu ve o da ekibin
`support_set_creator.py`'sindeki haritayı izliyordu (Rapor 6.4). Yani
uydurma değildi — ama alfabetik olmadığı için her arayüzde ayrıca
belirtilmesi gerekiyor ve bir kez sorulmasına yol açtı.

**Neden alfabetik:** `sorted()`, `sklearn.LabelEncoder`,
`torchvision.ImageFolder` ve pandas `astype("category")` hepsi alfabetik
üretir. Alfabetik olmayan bir sıra, bu araçlardan herhangi biriyle
karşılaşan her yerde sessiz bir takas riski taşır — ve takas edilecek
ikili tam da birbirine en çok karışan `climbing`/`cutting` olur.

⚠️ **Değiştirilecek yer:** `onbellek_kur.SINIFLAR`. Değiştirildiğinde
**önbellek yeniden kurulmalı** (etiketler orada tam sayı olarak yazılı)
ve tüm modeller yeniden eğitilmelidir. Bu yüzden kural *bundan sonraki*
eğitimler için — mevcut önbellek ve modeller eski sırayla kalıyor.

⚠️ Eski bir modelin ONNX'i alfabetik sıra isteyen bir hatta verilecekse,
yeniden eğitim gerekmez: `classifier` katmanının satırları yeniden
sıralanır (`weight[[1,0,2]]`, `bias[[1,0,2]]`). Ama bu **açıkça
belgelenmeli**, yoksa iki farklı sıralı model dolaşıma girer.

---

## 8e. 🔒 KURAL — ONNX HER ZAMAN HAM SİNYAL ALIR

Sorumlu BiLSTM teslimini onayladı ve şunu kural hâline getirdi:

> **Bundan sonraki tüm modellerin ONNX dosyaları da ham sinyale uyumlu
> olacak** — `(None, 15000)`, ön işlemenin tamamı grafiğin içinde.

Yani `CNN-BiLSTM/onnx_disa_aktar.py`'deki `OnIslemeliModel` sarmalayıcısı
artık isteğe bağlı bir kolaylık değil, **teslim standardı**. Yeni bir
mimari eklendiğinde sarmalayıcı yeniden yazılmaz — `model_kur()` ile
kurulup aynı sarmalayıcıya verilir.

opset **13**, IR **7**, çıktılar `logit` + `bosluk_orani`.

---

## 8b. TEKNİK REFERANS

Koda bakmadan hatırlanması gereken sabitler.

### Önbellek dosya biçimi (`onbellek_*_k0.h5`)

| veri kümesi | tip | şekil | ne |
|---|---|---|---|
| `spektrogram` | uint8 | (N, 129, 231) | dB [−80, 0] kuantalanmış, adım 0.31 dB |
| `etiket` | uint8 | (N,) | 0=cutting, 1=climbing, 2=noise |
| `dosya_idx` | int32 | (N,) | `dosyalar` dizisine indeks |
| `kanal` | int32 | (N,) | fiber kanalı |
| `pencere_bas` | int64 | (N,) | CSV `window_start` |
| `pencere_son` | int64 | (N,) | CSV `window_end` |
| `dosyalar` | string | (M,) | izlenebilirlik |

`attrs`: `fs`, `pencere`, `alan`, `n_fft`, `hop`, `top_db`, `bos_esik`,
`bos_frekans`, `siniflar`, `k`, `ham_satir`, `secilen_satir`, `yazilan`,
`elenen_bos`, `hata`, `uretim`, `kaynak_csv`.

`chunks=(64, 129, 231)`, `compression="lzf"`.

⚠️ **`pencere_son` doğrulama için zorunlu.** Olmadan bir kaydı kaynaktan
yeniden hesaplayamayız (pencere uzunluğu CSV'de 1.727 farklı değer alıyor,
varsayılamaz). İlk sürümde yoktu ve `--dogrula` her kayıtta 55 dB fark
raporlayıp bunu "beklenen" diye açıklıyordu — **asla başarısız olamayan bir
doğrulama, doğrulama değildir.** Eklendikten sonra 20/20 birebir çıktı ve
kasten bozulan bir önbelleği (kanal +1) yakaladığı da sınandı.

### Eğitim hiperparametreleri (tümü)

```
tohum            42          batch            64
optimizer        Adam        lr               1e-3
weight_decay     1e-4        label_smoothing  0.1
izlenen metrik   val macro-F1 (kati esitsizlik -> en erken epoch)
LR zamanlayici   ReduceLROnPlateau(mode="max", factor=0.5, patience=3)
determinizm      cudnn.deterministic=True, benchmark=False,
                 use_deterministic_algorithms(True, warn_only=True),
                 CUBLAS_WORKSPACE_CONFIG=:4096:8  (torch import'undan ONCE)
eski rejim       maske_p=0.5, maks_epoch=40, sabir=6
yeni rejim       maske_p=0.0, maks_epoch=80, sabir=10
```

**Sınıf ağırlığı** (`--sinif-agirligi`) hesaplı ama **hiçbir koşuda
kullanılmadı**: `cutting 0.451, climbing 0.386, noise 2.163`
(N / (K·n_k), ortalaması 1'e ölçekli).

**Maskeleme** (yalnızca eski rejim): her eksende %50 olasılık, **rastgele
1–2 şerit**, şerit genişliği eksenin en fazla %10'u. Çevirme YOK. Maskelenen
bölge 0 yapılıyor (normalize uzayında veri seti ortalaması).

### ONNX yeniden üretim

```
opset_version=13,  do_constant_folding=True,  dynamo=False
```

⚠️ **opset 13** (2026-09-01'de 17'den düşürüldü, sorumlunun isteği).

opset = ONNX operatör kütüphanesinin sürümü; grafikteki her düğümün hangi
tanıma göre yorumlanacağını sabitler. Bir çalışma zamanı, desteklediği en
yüksek opset'in üstündeki dosyayı **açamaz** — yani opset bir uyumluluk
eşiğidir, matematiği ya da hızı değiştirmez. Geriye dönük uyumlu: düşük
opset'li dosyayı yeni çalışma zamanı okur, tersi olmaz.

13'e düşürmek bu grafik için **bedelsiz** çıktı çünkü opset 17'ye özgü
hiçbir operatör kullanılmıyor — 17'nin bu projeye getirdiği tek yenilik
native `STFT`/`DFT` operatörleriydi ve STFT zaten elle yazılmıştı
(unfold + matmul). Medyan `TopK` tabanlı, havuzlama sabit çekirdekli,
antialias kapalı. Grafikteki en yeni gereksinim `Resize-13`; `LSTM`
opset 7'den, `Softmax-13` 13'ten mevcut.

**Ölçüldü** (`--opset-karsilastir`) — iki ortamda da aynı sonuç:

```
                          sahte agirlik (yerel)   GERCEK agirlik (sunucu)
opset 13  PyTorch'a fark        4.10e-08                3.58e-06
opset 17  PyTorch'a fark        4.10e-08                3.58e-06
opset 13 <-> 17 logit farki     0.00e+00                0.00e+00
```

Dosya boyutu iki opset'te de aynı (2.147.296 bayt), dinamik batch 13'te de
çalışıyor. **Opset düşürmenin bedeli sıfır.**

⚠️ **`dynamo=False` zorunlu.** torch 2.9+ varsayılan olarak yeni
(torch.export tabanlı) ihracatçıyı kullanıyor ve o `onnxscript` istiyor —
sunucuda kurulu değil. Eski TorchScript ihracatçısı grafiği sorunsuz
çıkarıyor ve sunucudaki `onnx 1.14.1` ile uyumlu. Script bunu
`inspect.signature` ile yoklayıp geçiyor.

Girdi/çıktı adları: `sinyal` → `logit`, `bosluk_orani`.
Dinamik eksen: üçünde de `{0: "batch"}`.

**IR version 7.** opset'ten ayrı ikinci bir uyumluluk eşiği; 7 muhafazakâr
(ONNX 1.6 dönemi), opset 13'ü destekleyen her çalışma zamanı okur.
⚠️ Bu, dosyanın **sunucuda** üretilmesinin yan faydası: sunucudaki
`onnx 1.14.1` IR 7 yazıyor, yereldeki `onnx 1.22` daha yenisini yazardı.
**ONNX teslimleri sunucuda üretilmeli.**

Dosyadan doğrulama (teslimden önce koşulmalı):

```python
import onnx
m = onnx.load("bilstm_kosu4.onnx")
print([(o.domain or "ai.onnx", o.version) for o in m.opset_import])  # [('ai.onnx', 13)]
print(m.ir_version)                                                  # 7
print([i.name for i in m.graph.input])                               # ['sinyal']
```

### Rapor script'i ne üretiyor

`gercek_rapor.py` → `egitim_ciktilari/` altına:

- `ogrenme_egrileri.png` — 4 panel (kayıp, doğruluk, val macro-F1 + taban
  çizgisi, öğrenme oranı); düz=eğitim, kesikli=doğrulama
- `karisiklik_matrisi.png` — **satır bazında normalize** (recall). Ham
  sayıyla renklendirmek `noise`'u (3.554 örnek) `climbing`'in (21.910)
  yanında yanıltıcı biçimde soluk gösterirdi
- `sinif_bazinda_f1.png` — taban çizgisiyle yan yana
- `gercek_veri_sonuclari.md` — tablolar

Hesapladığı karşılaştırmalar: AKTARIM ETKİSİ (2−1), GİRDİ TEMSİLİ (3−1),
ZAMANSAL DİZİ (4−1, sınıf bazında kırılımla), REJİM ETKİSİ (5−3), ve
MİMARİ ETKİSİ (fark-farkı). Fark 0.01'in altındaysa "tohum gürültüsü
olabilir" uyarısı basıyor.

---

## 9. KOD DEĞİŞİKLİKLERİ

| dosya | değişiklik |
|---|---|
| `src/gercek_egitim.py` | `kos()` artık `model_fn`, `maske_p`, `sabir` alıyor (geriye dönük uyumlu, A/B test edildi: 42 tensör birebir aynı). KOSULAR 4 ve 5 eklendi. `--maske-p`, `--sabir` bayrakları. |
| `src/gercek_rapor.py` | Koşu 4 ve 5 tanınıyor; ZAMANSAL DIZI, REJIM ETKISI ve fark-farkı hesabı eklendi |
| `src/model.py` | `load_pretrained(..., atla=("classifier",))` |
| `CNN-BiLSTM/model_bilstm.py` | Yeni mimari; ONNX için sabit çekirdekli havuzlama |
| `CNN-BiLSTM/egitim_bilstm.py` | İnce koşturucu, kendi eğitim döngüsü yok |
| `CNN-BiLSTM/onnx_disa_aktar.py` | **Artık mimariden bağımsız** — `ckpt_oku()` mimariyi (`lstm.` tensörü var mı) ve rengi (`ayar.renk`) checkpoint'ten okuyor; `sarmalayici_kur()` `model_kur()` ile kuruyor. Kart mimariyi ve rengi yazıyor. Ölçüldü: aynı dB'den gri vs viridis girdi farkı **3.852** — yanlış renk sessizce bozuk girdi demek, o yüzden sorulmuyor, okunuyor |
| `CNN-BiLSTM/onnx_disa_aktar.py` | Sarmalayıcı + ihracat + doğrulama + kullanım kartı. **opset varsayılanı 13**; `opset_karsilastir()` ve `--opset-karsilastir` eklendi; `disa_aktar()` artık ölçülen logit farkını **döndürüyor** ve kart o sayıyı yazıyor (eskiden "ihracat çıktısına bak" diyordu) |
| `src/gercek_export.py` | **İki mimariyi de paketliyor.** `MIMARILER` (koşu → sınıf), `model_kur()` / `model_kur_mimariden()` / `mimari_cikar()`. `mimari` bloğu elle yazılmıyor, kurulan modelden okunuyor. `--kosu` varsayılanı 4. Kart, `.pt` (görüntü) ile `.onnx` (ham sinyal) ayrımını **en üstte** gösteriyor |

### Düzeltilen iki sessiz hata

- **`lr` geçmişi eziliyordu.** `gecmis.update({... "lr": LR ...})` epoch
  başına listeyi skalerle eziyordu. Koşu 3'ün lr geçmişi kayboldu
  (log'da var). Artık `lr_baslangic` ayrı anahtarda.
- **Erken durdurma mesajı sabit `SABIR` basıyordu.** `--sabir 10` ile
  koşulsa bile "6 epoch" yazıyordu. Kozmetik ama log okurken yanıltıcı.

---

## 10. AÇIK İŞLER

- [x] ~~**Teslim paketini koşu 4'e güncelle**~~ — **kod hazır**
      (2026-09-01). `gercek_export.py` artık `MIMARILER` ile iki mimariyi
      de kuruyor; `mimari` bloğu canlı modelden okunuyor. Yerelde sahte
      checkpoint'le iki yol da test edildi (BiLSTM 430.932 / DASNet 34.835
      parametre, `strict=True` yükleme, kart içeriği). **Sunucuda
      çalıştırılmalı:** `python gercek_export.py --kosu 4`
- [x] ~~Kart düzeltmesi (a)~~ — `disa_aktar()` ölçülen logit farkını
      döndürüyor, kart artık sayıyı yazıyor
- [x] ~~ONNX kullanım kartını sunucuda üret~~ — **yapıldı**,
      `paket/bilstm_kosu4_KULLANIM.md`. İçeriği doğrulandı: performans
      tablosu, sınıf metrikleri, kullanım şartları, sınırlar hepsi var.
      İki küçük eksik kaldı (aşağıda)
- [ ] ⚠️ **KAÇIRMA — henüz incelenmedi.** Saha waterfall'ında (benchmark
      `.bin` dosyası, record_26) **"weak climbing" GT kutusu neredeyse
      tamamen boş** — model o olayı kaçırmış. Yanındaki normal `climbing`
      kutusunu yakalamış. Ölçüm de bunu destekliyor: `.bin` taramasında
      **GT-içi pencerelerin %16.0'ına model `noise` diyor**, yani gerçek
      olayların altıda birini kaçırıyoruz.
      **Çevre güvenliğinde kaçırılan ihlal, fazladan alarmdan daha
      pahalıdır** — yanlış alarm sorunu kapandıktan sonra buna bakılacak.
- [ ] **Kart düzeltmesi (b)** — pencereyi **15.000 örneğe oturtma** tarifi
      kartta yok (uzunsa enerji merkezine kırp, kısaysa yansıtmalı doldur).
      `hypot(re,im)`/`P` seçimi gibi bu da grafiğin **dışında**, çağıranın
      yapması gerekiyor. Sorumluya iletilmeli
- [ ] **ONNX'i opset 13 ile sunucuda yeniden üret** ve kartıyla birlikte
      sorumluya gönder — gerçek ağırlıklarla doğrulama sayıları da
      tazelenir
- [ ] ~~`gercek_rapor.py`'yi beş koşuyla çalıştır~~ — **ertelendi**
      (2026-09-01): böyle bir rapor şu an sorumludan beklenmiyor
- [ ] **Koşu 6**: geniş SK + yeni rejim — kapasite mi zaman mı
- [ ] Sorumluya özet

**Kalan hata:** hâlâ `climbing` ↔ `cutting`. `cutting`'in %12'si `climbing`
sanılıyor. `noise` çözülmüş (F1 0.987).

**Kod:** `src/gercek_egitim.py` · `src/gercek_veri_kumesi.py` ·
`src/gercek_rapor.py` · `src/gercek_export.py` · `CNN-BiLSTM/*`
**Çıktılar:** `/tf/start_training/RELATIONNET/FENCE_DATA_NEW/egitim_ciktilari/`
