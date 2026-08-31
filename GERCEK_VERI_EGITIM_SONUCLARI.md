# Gerçek Veri — Eğitim Sonuçları

> Aşama 2'nin eğitim kaydı. **Beş koşu**, her biri önceden tasarlandı ve
> test setine bakmadan çalıştırıldı.
>
> **En iyi: CNN-BiLSTM, test macro-F1 0.9390** (taban çizgisi 0.771, **+0.168**)
>
> **Son güncelleme:** 2026-08-28

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
PyTorch vs ONNX  logit farki : 3.10e-06
dinamik batch 1/3/16/64      : hepsi OK, fark <= 3.7e-05
```

LSTM için "dinamik batch hata verebilir" uyarısı geldi ama **ölçüldü,
sorun yok** — dizi uzunluğu sabit (40 adım).

### ⚠️ Kullanım şartları

- **Başka ön işleme uygulanmamalı** — hepsi içeride
- **`/16384` ölçek katsayısı uygulanmamalı**
- **`bosluk_orani > 0.45` olan pencerelerin tahmini kullanılmamalı** —
  eğitimde bu pencereler elendi, model onlar için eğitilmedi

Çıktı: `egitim_ciktilari/paket/bilstm_kosu4.onnx` (2.1 MB) +
`bilstm_kosu4_KULLANIM.md` (script üretiyor).

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
opset_version=17,  do_constant_folding=True,  dynamo=False
```

⚠️ **`dynamo=False` zorunlu.** torch 2.9+ varsayılan olarak yeni
(torch.export tabanlı) ihracatçıyı kullanıyor ve o `onnxscript` istiyor —
sunucuda kurulu değil. Eski TorchScript ihracatçısı grafiği sorunsuz
çıkarıyor ve sunucudaki `onnx 1.14.1` ile uyumlu. Script bunu
`inspect.signature` ile yoklayıp geçiyor.

Girdi/çıktı adları: `sinyal` → `logit`, `bosluk_orani`.
Dinamik eksen: üçünde de `{0: "batch"}`.

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
| `CNN-BiLSTM/onnx_disa_aktar.py` | Sarmalayıcı + ihracat + doğrulama + kullanım kartı |

### Düzeltilen iki sessiz hata

- **`lr` geçmişi eziliyordu.** `gecmis.update({... "lr": LR ...})` epoch
  başına listeyi skalerle eziyordu. Koşu 3'ün lr geçmişi kayboldu
  (log'da var). Artık `lr_baslangic` ayrı anahtarda.
- **Erken durdurma mesajı sabit `SABIR` basıyordu.** `--sabir 10` ile
  koşulsa bile "6 epoch" yazıyordu. Kozmetik ama log okurken yanıltıcı.

---

## 10. AÇIK İŞLER

- [ ] **Teslim paketini koşu 4'e güncelle** — paket hâlâ koşu 3'ünkü.
      `gercek_export.py` `DASNet`'i sabit kuruyor, BiLSTM için küçük bir
      ekleme gerekiyor
- [ ] `gercek_rapor.py`'yi sunucuda beş koşuyla çalıştır (grafikler +
      markdown özeti)
- [ ] ONNX kullanım kartını sunucuda üret (script hazır, bir kez daha
      çalıştırılmalı)
- [ ] **Koşu 6**: geniş SK + yeni rejim — kapasite mi zaman mı
- [ ] Sorumluya özet

**Kalan hata:** hâlâ `climbing` ↔ `cutting`. `cutting`'in %12'si `climbing`
sanılıyor. `noise` çözülmüş (F1 0.987).

**Kod:** `src/gercek_egitim.py` · `src/gercek_veri_kumesi.py` ·
`src/gercek_rapor.py` · `src/gercek_export.py` · `CNN-BiLSTM/*`
**Çıktılar:** `/tf/start_training/RELATIONNET/FENCE_DATA_NEW/egitim_ciktilari/`
