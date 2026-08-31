# CNN-BiLSTM — Zamansal Örüntü Mimarisi

Projenin **en iyi modeli**: test macro-F1 **0.9390** (taban çizgisi 0.771).

Tam sonuçlar ve atıf hesabı: `../GERCEK_VERI_EGITIM_SONUCLARI.md`

---

## Neden bu mimari

Koşu 1–3'ten sonra kalan hatanın neredeyse tamamı `climbing` ↔ `cutting`
arasındaydı. Mevcut `DASNet`'in adı konabilir bir zayıflığı vardı:

```
Conv bloklari -> SK-Attention -> AdaptiveAvgPool2d(1) -> Linear
                                 ^^^^^^^^^^^^^^^^^^^^
                                 40 zaman cercevesini TEK sayiya cokertiyor
```

`cutting` ritmik ve ayrık, `climbing` sürekli ve düzensiz. Global ortalama
havuzlama "darbe var mı" bilgisini korur ama "darbeler zaman içinde nasıl
dizilmiş" bilgisini atar.

Faz 0 bunu destekliyordu: "modülasyon tepe frekansı" ve "modülasyon
keskinliği" özellikleri tek başına zayıftı (F ≈ 0.05) ama 26 özellik
**birlikte** iki sınıfı ayırabiliyordu — ritim bilgisi vardı, dağınık hâlde.

**Ölçüm hipotezi destekledi:** `cutting` +0.090, `climbing` +0.066,
`noise` +0.008. Karışıklık yarıya indi.

---

## Dosyalar

| dosya | ne |
|---|---|
| `model_bilstm.py` | `DASNetBiLSTM` + birim testi (`python model_bilstm.py`) |
| `egitim_bilstm.py` | İnce koşturucu — **kendi eğitim döngüsü yok**, `src/gercek_egitim.kos()` çağırır |
| `onnx_disa_aktar.py` | ONNX ihracatı: ön işleme grafiğe gömülü, `(None, 15000)` girdi |

Hepsi `src/`'den import ediyor, kopyalamıyor. Omurga bile `DASNet`'in
kendi modüllerinden ödünç alınıyor — ikisi asla ayrışamaz.

---

## Mimari

```
girdi (B,3,224,320)
  -> features + SK-Attention   [DASNet'ten aynen]  -> (B,64,28,40)
  -> frekansi 4 bine indir     avg_pool2d(7,1)     -> (B,64,4,40)
  -> yeniden duzenle           zaman = dizi        -> (B,40,256)
  -> BiLSTM(256 -> 128, cift yonlu)                -> (B,40,256)
  -> dikkatli zaman havuzlama  Linear(256->1)+softmax -> (B,256)
  -> Dropout(0.5) -> Linear(256,3)                 -> (B,3)
```

| | |
|---|---|
| parametre | **430.932** (DASNet'in 12.4 katı) |
| zaman adımı | 40 (188 ms/adım) |
| adım boyutu | 256 (64 kanal × 4 frekans bini) |

**Tasarım kararları ve gerekçeleri `model_bilstm.py` docstring'inde.** Özet:
frekans çökertilmiyor (bant bilgisi korunuyor), çift yönlü (darbenin anlamı
iki yöne de bağlı), dikkatli havuzlama (son gizli durum pencerenin sonuna
orantısız ağırlık verirdi).

⚠️ `zaman_havuzlama=False` 80 çerçeveye çıkarır ama omurga artık DASNet ile
aynı olmaz ve karşılaştırma kirlenir. **Varsayılan kapalı.**

⚠️ Frekans havuzlama çekirdeği `__init__`'te Python tam sayısı olarak
hesaplanıyor. `AdaptiveAvgPool2d((4,None))` doğru sonuç veriyordu ama ONNX'e
ihraç edilemiyordu ("output_size is not constant").

---

## Kullanım

```bash
python model_bilstm.py                    # birim testi
python egitim_bilstm.py --hizli           # duman testi
python egitim_bilstm.py                   # tam kosu (kosu 4)

python onnx_disa_aktar.py \
    --ckpt <...>/kosu4_bilstm_yeni_rejim.pt \
    --cikti <...>/paket/bilstm_kosu4.onnx
```

Sunucuda uzun koşular için **bekleyen başlatıcı** kalıbını kullanın
(`../DURUM.md` Bölüm 8C) — GPU paylaşımlı.

---

## Eğitim rejimi

Koşu 4 **yeni rejimle** koşuldu: maskeleme kapalı, maks 80 epoch, sabır 10.

⚠️ **Ama rejimin katkısı ölçüldü ve sıfır çıktı** (koşu 5: −0.003).
Kazancın tamamı mimariden geliyor. Ayrıntı: sonuç belgesi Bölüm 2–3.

---

## Kalan belirsizlik

BiLSTM aynı zamanda 12 kat daha büyük. Kazancın **zamansal modellemeden mi
kapasiteden mi** geldiği ayrılmadı.

Ayıracak koşu: SK modelini `config.CONV_CHANNELS=(32,64,128)` ile büyütüp
yeni rejimle koşmak (~124.000 parametre). Geniş SK BiLSTM'e yaklaşırsa
kapasite, yaklaşmazsa zamansal yapı.
