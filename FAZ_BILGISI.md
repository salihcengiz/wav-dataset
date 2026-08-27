# Sinyalin Faz Bilgisi — Not

> Sorumlunun isteği üzerine yazıldı: *"Sinyalin faz bilgisini bir yere yaz,
> ileride kullanabiliriz."*
>
> **Kısa cevap:** Faz, DAS'ta genlikten **teorik olarak üstün** olan
> büyüklüktür. Ama elimizdeki bu veri setinde **ölçtük ve kullanılamaz
> durumda** — sebebi fizik değil, veriye uygulanmış kuantalama.
> İleride kullanmak için verinin farklı bir aşamasına erişmek gerekiyor.

**Tarih:** 2026-08-27

---

## 1. Faz nedir

DAS / Φ-OTDR'de interrogator, fibere **eşevreli (koherent)** bir lazer
darbesi gönderir. Fiberdeki mikroskobik düzensizliklerden geri saçılan
(Rayleigh) ışık, geri döndüğünde kendi içinde girişim yapar. Dönen sinyal
karmaşık bir sayı olarak ölçülür:

```
z = re + i·im
```

Bunun iki bileşeni var:

| | ifade | fiziksel karşılığı |
|---|---|---|
| **Genlik** | `|z| = hypot(re, im)` | geri dönen ışığın şiddeti |
| **Faz** | `∠z = atan2(im, re)` | ışığın kat ettiği **optik yol uzunluğu** |

Fiber gerildiğinde (titreşim, ses, çite vurma) o bölgedeki optik yol uzunluğu
değişir ve bu **doğrudan faza** yansır.

## 2. Faz neden genlikten üstün

**Doğrusal ve nicel.** Standart bağıntı:

```
Δφ = (4πn/λ) · ΔL
```

`n` fiberin kırılma indisi, `λ` lazer dalga boyu, `ΔL` uzama. Katsayı sabit,
yani faz değişimi gerinime (strain) **doğru orantılı**. Ölçtüğünüz şey
"titreşim var mı" değil, "ne kadar" oluyor. `4π` çarpanı ışığın gidip
dönmesinden geliyor (yol iki kez kat ediliyor).

**Sönümleme (fading) yok.** Genlik tabanlı Φ-OTDR'ın bilinen sorunu: Rayleigh
girişimi bazı konumlarda yıkıcı olur ve orada genlik neredeyse sıfıra iner —
"fading spot". O kanal olayı hiç görmez, üstelik olay olduğu için değil,
optik tesadüf yüzünden. Faz bu sorundan etkilenmez.

**Tekdüze olmayan tepki.** Genlik ile gerinim arasındaki ilişki monoton bile
değildir; aynı genlik farklı gerinim değerlerine karşılık gelebilir.

**Pratikte kullanılan büyüklük genellikle *diferansiyel* fazdır:**
komşu iki konum arasındaki faz farkı, aradaki *gauge length* boyunca oluşan
gerinimi verir. Tek bir noktanın mutlak fazı sürüklenmeye (drift) açıktır.

---

## 3. Bu veri setinde ölçtük — faz kullanılamıyor ❌

`GERCEK_VERI_FAZ0_RAPORU.md` Bölüm 4.1'deki ölçüm:

```python
faz = np.unwrap(np.angle(re + 1j*im))
d = np.diff(faz)
d.std()   ->  1.814   (her kanalda)
```

**1.814 sayısı belirleyici.** `[-π, π]` aralığında **düzgün dağılımın**
standart sapması tam olarak:

```
π / √3 = 1.8138
```

Yani ardışık faz farkları **saf rastgele**. İçinde sinyal yok, sadece
gürültü var.

### Sebebi: kuantalama

Ham `re` / `im` değerleri `int16` alanında saklanmış ama **gerçek aralık
±11** — toplam 22 farklı tam sayı değeri.

Karmaşık düzlemde yarıçapı `r` olan bir noktada, tam sayı ızgarasının
sağladığı açısal çözünürlük kabaca `1/r` radyandır. Tipik `|z| ≈ 3` için:

```
1/3 radyan ≈ 19 derece
```

Gerçek faz değişimi bundan çok daha küçük olduğu için, kuantalama onu
tamamen yutuyor. Geriye kalan şey ızgara gürültüsü.

**Bu bir fizik sınırı değil, veri hazırlama sınırı.** Faz bilgisi fiberde
vardı; bize ulaşan dosyada yok.

### Ekip de faz kullanmıyor

`support_set_creator.py` ve `sequence_few_shot_*.py` dosyalarının hepsi
genlik alıyor:

```python
s = np.hypot(s_re, s_im)
```

Yani bu, bizim tercihimiz değil, verinin dayattığı durum.

---

## 4. İleride kullanmak için ne gerekir

Sırasıyla, en kolaydan en zora:

### 4.1 Interrogator'ın faz çıktısını istemek ⭐ en pratik

Modern DAS interrogator'larının çoğu fazı **kendisi demodüle edip** çıktı
verebiliyor (genellikle "phase", "strain" veya "strain-rate" adıyla).
Eğer cihaz bunu destekliyorsa, ham I/Q ile uğraşmaya gerek yok.

**Sorulacak:** Kullanılan interrogator faz/gerinim çıktısı verebiliyor mu?
Veren bir kayıt modu var mı?

### 4.2 Ham I/Q'yu kuantalama ÖNCESİ almak

`±11` aralığı, kayıt hattının bir yerinde ölçeklendirme + `int16`'ya
yuvarlama yapıldığını gösteriyor. O adımdan önceki veriye (veya daha geniş
dinamik aralıkla kaydedilmiş bir sürüme) erişilirse faz kurtarılabilir.

**Sorulacak:** `.bin` dosyaları üretilirken hangi ölçekleme uygulanıyor?
Kuantalama öncesi veri saklanıyor mu?

### 4.3 Yeni kayıtları uygun ayarla toplamak

Yeni saha kaydı yapılacaksa dinamik aralık faz demodülasyonunu kaldıracak
şekilde ayarlanmalı. `|z| ≈ 3` değil, en az birkaç yüz seviyesinde olmalı ki
açısal çözünürlük anlamlı olsun.

### 4.4 Elde edilirse nasıl kullanılır

```
1. Karmaşık I/Q'dan faz:      φ(t) = angle(re + i·im)
2. Zaman ekseninde aç:        φ_unwrap = unwrap(φ)
3. Diferansiyel (uzamsal):    Δφ = φ(kanal_2) - φ(kanal_1)
4. Gerinim hızı:              dΔφ/dt
5. Buradan spektrogram        -> mevcut 2D-CNN hattına GIRDI
```

Mevcut hattımız buna hazır: `real_data.pencere_yukle` genlik üretiyor, aynı
yapı faz için de kullanılabilir. Model tarafında iki seçenek var:

- **Ayrı kanal olarak:** girdi 3 kanal yerine genlik + faz spektrogramı
- **Faz yerine geçsin:** genliği tamamen bırakıp faz spektrogramıyla eğitmek

İkisi de ölçülerek karşılaştırılmalı — tahminle seçilmemeli.

### 4.5 Dikkat edilecek tuzaklar

- **Faz sarması (wrapping).** Ölçülen faz `2π` moduludur. Hızlı/şiddetli
  olaylarda ardışık örnekler arası değişim `π`'yi aşarsa `unwrap` yanlış
  açar ve sinyale sahte sıçramalar ekler. Örnekleme frekansı buna yetmeli.
- **Sürüklenme (drift).** Mutlak faz sıcaklıkla ve lazer kararsızlığıyla
  yavaşça kayar. Bu yüzden diferansiyel faz tercih edilir.
- **Düşük genlikte faz gürültülü olur.** `|z|` küçükken faz kestirimi
  güvenilmezdir — zaten bizim yaşadığımız sorunun özü bu.

---

## 5. Özet

| | |
|---|---|
| Faz teorik olarak genlikten üstün mü | **Evet** — doğrusal, nicel, fading yok |
| Bu veri setinde kullanılabilir mi | **Hayır** — ölçüldü, std 1.814 = saf rastgele |
| Sebep | `int16` alanında ±11 aralık, ~19° faz kuantalaması |
| Fizik sınırı mı | **Hayır**, veri hazırlama sınırı |
| İleride ne gerekir | Interrogator'ın faz çıktısı, veya kuantalama öncesi I/Q |
| Hattımız hazır mı | Evet — aynı pencereleme/STFT yapısı faz için de çalışır |

**Sorumluya sorulacak iki soru:**

1. Interrogator faz / gerinim çıktısı verebiliyor mu?
2. `.bin` dosyaları üretilirken uygulanan ölçekleme nedir; kuantalama
   öncesi ham I/Q saklanıyor mu?

**İlgili ölçümler:** `GERCEK_VERI_FAZ0_RAPORU.md` Bölüm 4.1
**İlgili kod:** `src/real_data.py` → `genlik_oku()` docstring'i
