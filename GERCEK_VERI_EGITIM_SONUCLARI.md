# Gerçek Veri — Eğitim Sonuçları ve İyileştirme Sırası

> Aşama 2'nin eğitim kaydı. Üç koşu önceden tasarlandı, test setine
> bakmadan; her koşuda **tek değişken** değişti.
>
> **Sonuç: test macro-F1 0.8843** (taban çizgisi 0.771, **+0.113**)
>
> **Tarih:** 2026-08-27

---

## 1. ÜÇ KOŞU

| # | girdi | başlangıç | val macro-F1 | **test macro-F1** | en iyi / koşulan epoch | nasıl durdu |
|---|---|---|---|---|---|---|
| **1** | viridis | sıfırdan | 0.8719 | **0.8843** | 39 / 40 | **tavan** (hâlâ iyileşiyordu) |
| 3 | gri | sıfırdan | 0.8657 | 0.8737 | 23 / 29 | erken durdurma |
| 2 | viridis | **aktarım** | 0.8596 | 0.8658 | 18 / 24 | erken durdurma |
| — | *taban çizgisi* | *doğrusal, 26 özellik* | *0.771* | — | — | — |

Ortak ayarlar: tohum 42, batch 64, Adam lr 1e-3, weight decay 1e-4,
label smoothing 0.1, maks 40 epoch, erken durdurma sabrı 6, izlenen metrik
**val macro-F1**, determinizm açık.

### Sınıf bazında (test)

| sınıf | koşu 1 | koşu 3 | koşu 2 | taban |
|---|---|---|---|---|
| cutting | **0.813** | 0.794 | 0.782 | 0.678 |
| climbing | **0.861** | 0.851 | 0.841 | 0.653 |
| noise | **0.979** | 0.977 | 0.975 | 0.981 |
| **macro** | **0.884** | 0.874 | 0.866 | 0.771 |

Sıralama üç sınıfın üçünde de aynı: **1 > 3 > 2**.

---

## 2. SORU 1 — Sentetik ön-eğitim işe yarıyor mu? ❌ HAYIR

```
koşu 1 (sıfırdan)   0.8843
koşu 2 (aktarım)    0.8658
                    --------
fark                -0.0185
```

Aktarımlı model hem daha düşük bitti hem **daha erken durdu** (en iyi epoch
18, epoch 24'te durdu). Sıfırdan başlayan model epoch 39'a kadar iyileşti.

Erken epoch'larda da geride başladı — oysa aktarımın faydası en çok orada
görünmeliydi:

| epoch | koşu 1 (sıfırdan) | koşu 2 (aktarım) |
|---|---|---|
| 2 | 0.7956 | 0.7521 |
| 6 | 0.8173 | 0.8068 |

**Yorum:** 19 bağımsız sentetik kayıttan öğrenilen filtreler, 21.101 gerçek
dosyanın yanında avans sağlamıyor; modelin çıkmak zorunda kaldığı bir kısıt
gibi davranıyor. `MODEL_CARD.md`'nin kendi uyarısı doğrulandı: *"sentetik
veri gerçek DAS verisinin yerini tutmaz."*

**Aşama 1 boşa gitmedi:** mimari, ön işleme disiplini, sızıntı testleri ve
metodoloji oradan geldi. İşe yaramayan tek şey **ağırlıkların aktarımı**.

**Karar: sentetik önceden eğitilmiş model artık kullanılmıyor.**

---

## 3. SORU 2 — Viridis zarar veriyor mu? ❌ HAYIR (tahmin yanlış çıktı)

```
koşu 1 (viridis)    0.8843
koşu 3 (gri)        0.8737
                    --------
fark                +0.0106   (viridis lehine)
```

Beklenti şuydu: viridis, tek bir dB değerini üç kanala **monoton olmayan**
biçimde dağıtan keyfi bir dönüşüm (ölçüldü: R kanalında 42, B kanalında 92
azalan adım). Sıfırdan eğitilen bir ağ için gereksiz zorluk olmalıydı.

Ölçüm tersini söyledi. Fark küçük ama üç sınıfın üçünde de aynı yönde.

**Tek net gözlem:** viridis ile eğitim **daha oynak** ilerledi (±0.10
sıçramalar) ve **daha yavaş yakınsadı** — 40 epoch bütçesi yetmedi.

---

## 4. METODOLOJİK ÇEKİNCELER

Bunlar rapora yazılmalı:

1. **Her konfigürasyondan tek koşu, tek tohum.** Farklar (0.011 ve 0.019)
   tohum varyansından ayrılamayacak kadar küçük olabilir. Kesin konuşmak
   için farklı tohumlarla tekrar gerekir.
2. **Koşu 1 kesildi, yakınsamadı.** 40 epoch tavanına çarptığında hâlâ
   iyileşiyordu. Yani **0.8843 onun tavanı değil** — daha uzun bütçeyle
   artabilir. Bu, aktarım karşılaştırmasını aktarım aleyhine değil,
   *lehine* çeviren bir çekince değil: koşu 2 aynı bütçeyi aldı ve
   kendiliğinden durdu.
3. **Koşu 2 kıl payı durdu.** Epoch 24'te 0.8590 aldı, en iyisi 0.8596'ydı —
   0.0006 farkla iyileşme sayılmadı. Birkaç epoch daha koşsa biraz
   yükselebilirdi, ama 0.8843'e yetişmesi için çok yol vardı.
4. **Koşu süreleri karşılaştırılamaz.** GPU paylaşımlı ve yük değişkendi:
   koşu 3 = 103 dk, koşu 1 = 178 dk, koşu 2 = 132 dk. Doğruluk
   karşılaştırması etkilenmiyor, yalnızca duvar saati.
5. **Taban çizgisi gürültülü.** 0.771, ~239 doğrulama örneğiyle ölçülmüştü;
   bizim skorlar 42.850 test örneğiyle. Yön kesin, tabanın hata payı geniş.

---

## 5. ANA BULGU — MODEL YETERSİZ ÖĞRENİYOR

Üç koşunun üçünde de **doğrulama doğruluğu eğitim doğruluğunun üstünde**:

| koşu | eğitim doğruluk | doğrulama doğruluk |
|---|---|---|
| 1 | 0.690 (epoch 2) | 0.757 |
| 3 | 0.821 (epoch 19) | 0.835 |
| 2 | ~0.79 | 0.826 |

Ezberleyen bir modelde bu asla görülmez. İki sebebi var ve ikisi de
düzeltilebilir:

### a) Maskeleme fazla geliyor

Eğitim sırasında spektrogramın rastgele şeritlerini siliyoruz — bazı zaman
aralıklarını ve frekans bantlarını karartıyoruz. Model **bozulmuş**
görüntülerle çalışıyor. Doğrulama ve testte ise hiçbir şey silinmiyor,
**temiz** görüntü veriliyor.

Yukarıdaki tabloda görünen fark tam olarak bu. Aynı model, iki farklı
zorluktaki sınavda ölçülüyor.

> Benzetme: kitabın bazı sayfaları yırtılmış hâlde ders çalışıp, sınava tam
> kitapla girmek. Sınavda daha iyi yapmanız normaldir.

**Neden yapıyorduk:** Sentetik aşamada elimizde **19 gerçek kayıt** vardı.
Model o 19 kaydı ezberlerdi. Sayfaları yırtmak, ezberlemek yerine anlamaya
zorluyordu — PLAN 7.2 bunu "19 etkin örnekle ZORUNLU" diye işaretlemişti.

**Neden artık gereksiz:** Şimdi **21.101 farklı kayıt** var. Ezberlenecek
bir şey yok, her örnek zaten farklı. Sayfaları yırtmaya devam etmek,
sebepsiz yere zorlaştırmak oluyor.

### b) Kapasite tükenmemiş

Model **34.835 parametreye** sahip. Parametre sayısı kabaca modelin
öğrenebileceği bilgi miktarı — ne kadar şey "aklında tutabileceği".

Bu sayı bilerek küçük seçilmişti ve gerekçesi PLAN 6.1'de yazılı: 19
bağımsız örnekle büyük bir model kesinlikle ezberlerdi, dolayısıyla
küçüklük **avantajdı**.

O gerekçe ortadan kalktı. 21.101 bağımsız dosya var ve model kapasitesinin
sınırına dayanmış görünüyor.

**Kanıt:** koşu 1, 40 epoch bütçesi bittiğinde **hâlâ iyileşiyordu**. Yani
"öğrenecek bir şeyim kalmadı" demedi, "vaktim doldu" dedi.

> Benzetme: küçük bir deftere çok şey yazmaya çalışmak. Defter yetmiyor.

### İkisi birlikte ne anlatıyor

Model **yetersiz öğreniyor** (underfitting), **ezberlemiyor**
(overfitting değil). Bu ayrım kritik, çünkü ikisinin çözümü **zıt**:

| durum | çözüm |
|---|---|
| Ezberliyor | veriyi zorlaştır, modeli küçült |
| **Yetersiz öğreniyor** ← biz buradayız | **zorlaştırmayı azalt, modeli büyüt** |

Şu an ikinci durumdayız ama birincinin ilacını veriyoruz. Sentetik aşamadan
devraldığımız ayarlar, o aşamanın kısıtlarına göre seçilmişti; veri 230 kat
büyüdüğünde birlikte gözden geçirilmediler.

**Sonuç: 0.884 bu modelin tavanı değil. Darboğaz mimari değil, eğitim
rejimi.**

---

## 6. İYİLEŞTİRME SIRASI (ölçülmedi, öneri)

| # | deney | maliyet | gerekçe |
|---|---|---|---|
| 1 | Maskelemeyi kapat + epoch tavanı 80 | 1 koşu | Bölüm 5'teki iki bulguya doğrudan cevap |
| 2 | Kanalları büyüt (16/32/64 → 32/64/128) | 1 koşu | Kapasite kısıtı kalktı; `config.CONV_CHANNELS` tek satır |
| 3 | Dikkat ablasyonu: yok / SE / CBAM vs SK | 3 koşu | `model.py`'de hazır, gerçek veride hiç denenmedi. "SK doğru seçim miydi" sorusunun cevabı |
| 4 | Sınıf ağırlığı (`--sinif-agirligi`) | 1 koşu | `noise` %8.8; ağırlıklar hesaplı (2.16×) ama denenmedi |
| 5 | Farklı mimari | yeni kod + koşu | En pahalı, en belirsiz |

1 ve 2 **kanıta dayalı** — üç koşuda da gözlenen bir örüntüye cevap veriyor.
5 ise "belki daha iyisi vardır" umuduna dayanıyor.

⚠️ **Proje kuralı:** kaç konfigürasyon denendiği rapora yazılır ve
hiperparametre **test setine bakarak** seçilmez. Şu ana kadar: **3
konfigürasyon**.

---

## 7. AÇIK İŞLER

- [ ] Teslim paketini koşu 1'e güncelle (`gercek_export.py --kosu 1`) —
      paket hâlâ koşu 3'ünkü
- [ ] `gercek_rapor.py` ile üç koşunun grafikleri ve markdown özeti
- [ ] Sorumluya özet
- [ ] Kalan hata hâlâ `climbing` ↔ `cutting`: `cutting`'in %21'i `climbing`
      sanılıyor. `noise` çözülmüş (F1 0.979).

**Kod:** `src/gercek_egitim.py` · `src/gercek_veri_kumesi.py` ·
`src/gercek_rapor.py` · `src/gercek_export.py`
**Çıktılar:** `/tf/.../egitim_ciktilari/` (sunucuda)
