"""
BOSLUK OLCUTU TARAMASI -- hangi buyukluk "burada bir sey yok"u dogru soyler?

=== NEDEN ===

Saha testinde kanal 0, 30 saniye boyunca kesintisiz `cutting` veriyor.
ONNX'e gomdugumuz bastirma TETIKLENMIYOR: Detailed Test'te logitler
+1.000 / -0.852 / -0.684, yani -1e4 yok.

Sebep, olcutumuzun kor noktasi. `bosluk_orani` = 500 Hz USTUNDEKI enerji
payi, yani spektrumun SEKLINI olcuyor, gucunu degil:

    beyaz gurultu        ~0.50   -> bos sayilir   DOGRU
    gercek olay          ~0.15   -> dolu sayilir  DOGRU
    olu kanal + suruklen  dusuk   -> dolu sayilir  YANLIS

Olu bir kanalda yuksek frekans yoktur, yavas bir taban kaymasi vardir.
Enerji dusuk frekansta toplanir, oran DUSUK cikar, pencere "dolu"
sayilir. Sonra medyan/MAD normalizasyonu o cilizligi birim olcege
buyutur ve model olay gibi gorunen bir spektrogram bulur.

Kok sebep tasarim kararimizda yazili (`real_data.normalize_et`):
"Kanaldan kanala genlik seviyesi 36 kata kadar degisiyor. Seviye SINIFLA
ilgili DEGIL." Dogru -- seviye hangi sinif oldugunu soylemez. Ama ORADA
BIR SEY OLUP OLMADIGINI soyleyen tek sey odur. Normalizasyonla tam da
simdi ihtiyacimiz olan bilgiyi attik.

=== NEDEN BENCHMARK .bin DOSYALARI ===

Egitim verisinde bos pencereler zaten elenmis, yani olcutun KACIRDIGI
pencereleri orada bulamayiz -- dairesel olur. Benchmark dosyalarinda ise
`labels` veri kumesi var: hangi (zaman, kanal) araliginda gercekten olay
oldugu yazili. Yani ETIKETLI bir ayirt etme problemi kurabiliyoruz.

Yalnizca .bin.hdf5 kullaniliyor: onlarda duration x prf = ornek sayisi
tam tutuyor (fs = 2000). .sdf.hdf5'te 2 kat uyusmazlik var (fs 4000
gibi), o yuzden disarida birakiliyor.

=== NE OLCULUYOR ===

Her pencere icin alti aday buyukluk + modelin cevabi. Sonra GT-ici ve
GT-disi pencereler ayrilip her adayin ayirma gucu raporlaniyor:

    "GT olaylarinin %95'ini korurken GT-disi pencerelerin yuzde kacini
     eleyebiliyoruz?"

En yuksek oran hangisindeyse dogru olcut odur.

=== KULLANIM ===

    python3 bosluk_olcut_tarama.py
    python3 bosluk_olcut_tarama.py --adim 4000 --ckpt .../kosu3_gri_sifirdan.pt
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

_KOK = Path(__file__).resolve().parent.parent
for _y in (str(_KOK / "src"), str(_KOK / "CNN-BiLSTM")):
    if _y not in sys.path:
        sys.path.insert(0, _y)

import real_data as rd                                    # noqa: E402

DIZIN = "/tf/segment/Fence Benchmark Data"
CIKTI = "/tf/start_training/RELATIONNET/FENCE_DATA_NEW/egitim_ciktilari"
SINIFLAR = ["cutting", "climbing", "noise"]

# Aday olcutler. (ad, yon) -- yon="dusuk" ise DUSUK deger BOS demek.
ADAYLAR = [
    ("bosluk_orani", "yuksek"),   # mevcut olcut: 500 Hz ustu enerji payi
    ("mad_ham", "dusuk"),         # ham sinyalin MAD'i -- mutlak zayiflik
    ("std_ham", "dusuk"),
    ("benzersiz", "dusuk"),       # kac farkli deger -- kuantalanmis olu kanal
    ("db_aralik", "dusuk"),       # dB spektrogram kontrasti (olcekten bagimsiz)
    ("db_p99_p50", "dusuk"),      # kontrastin dayanikli hali
    ("dusuk_frek", "yuksek"),     # 100 Hz alti enerji payi -- suruklenme
]


def pencere_olcutleri(s):
    """Ham pencereden aday buyuklukleri hesaplar."""
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    z = rd.normalize_et(s)
    db = rd.spektrogram(z)                     # (129, T), [-80, 0]

    # dusuk frekans payi -- tam pencere FFT'sinden
    x = s - s.mean()
    G = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / rd.FS)
    top = G.sum()

    return {
        "bosluk_orani": rd.bosluk_orani(s),
        "mad_ham": float(mad),
        "std_ham": float(s.std()),
        "benzersiz": float(len(np.unique(s))),
        "db_aralik": float(db.max() - db.min()),
        "db_p99_p50": float(np.percentile(db, 99) - np.percentile(db, 50)),
        "dusuk_frek": float(G[f < 100].sum() / top) if top > 0 else 1.0,
    }, z


def gt_araliklari(f):
    """labels -> [(ornek_bas, ornek_son, kanal_bas, kanal_son), ...]"""
    if "labels" not in f:
        return []
    return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in f["labels"][:]]


def gt_icinde(bas, son, kanal, kutular):
    """Pencere herhangi bir GT kutusuyla hem zaman hem kanal olarak kesisiyor mu?"""
    for ob, os_, kb, ks in kutular:
        if bas < os_ and son > ob and kb <= kanal <= ks:
            return True
    return False


def topla(dizin, adim, sarmal=None, cihaz="cpu"):
    import h5py
    kayitlar = []
    dosyalar = sorted(glob.glob(os.path.join(dizin, "*.bin.hdf5")))
    print(f".bin.hdf5 dosyasi: {len(dosyalar)}")

    for yol in dosyalar:
        with h5py.File(yol, "r") as f:
            kutular = gt_araliklari(f)
            kanallar = sorted(int(k) for k in f.keys() if k.isdigit())
            n_ornek = f[str(kanallar[0])].shape[0]
            print(f"\n  {os.path.basename(yol)[:46]:48s} "
                  f"kanal {kanallar[0]}..{kanallar[-1]} ({len(kanallar)})  "
                  f"ornek {n_ornek}  GT kutusu {len(kutular)}")
            for kb, ks in [(k[2], k[3]) for k in kutular][:3]:
                pass

            for kanal in kanallar:
                for bas in range(0, n_ornek - rd.PENCERE + 1, adim):
                    son = bas + rd.PENCERE
                    s = rd.genlik_oku(f, kanal, bas, son, alan=rd.ALAN)
                    if s is None or len(s) != rd.PENCERE:
                        continue
                    o, z = pencere_olcutleri(s)
                    o["gt"] = gt_icinde(bas, son, kanal, kutular)
                    o["kanal"] = kanal
                    o["bas"] = bas
                    o["_ham"] = s.astype(np.float32)
                    kayitlar.append(o)
    return kayitlar


def model_ekle(kayitlar, sarmal, cihaz, batch=64):
    import torch
    X = np.stack([k.pop("_ham") for k in kayitlar])
    sarmal = sarmal.to(cihaz).eval()
    L = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            lg, _ = sarmal(torch.from_numpy(X[i:i + batch]).to(cihaz))
            L.append(lg.cpu().numpy())
    L = np.concatenate(L)
    for k, l in zip(kayitlar, L):
        k["tahmin"] = SINIFLAR[int(l.argmax())]
        k["logit"] = float(l.max())
    return kayitlar


def rapor(kayitlar, koruma=0.95):
    gt = np.array([k["gt"] for k in kayitlar])
    print("\n" + "=" * 78)
    print(f"PENCERE: {len(kayitlar):,}   GT-ici: {gt.sum():,}   "
          f"GT-disi: {(~gt).sum():,}")
    print("=" * 78)

    if "tahmin" in kayitlar[0]:
        for ad, m in (("GT-ici ", gt), ("GT-disi", ~gt)):
            t = [k["tahmin"] for k, v in zip(kayitlar, m) if v]
            lg = np.array([k["logit"] for k, v in zip(kayitlar, m) if v])
            if not t:
                continue
            from collections import Counter
            c = Counter(t)
            print(f"  {ad}: " + "  ".join(
                f"{s} %{100*c.get(s,0)/len(t):.1f}" for s in SINIFLAR)
                + f"   |  logit>0.9: %{100*(lg>0.9).mean():.1f}")

    print(f"\n  Her olcut icin: GT olaylarinin %{100*koruma:.0f}'ini KORUYAN")
    print(f"  esikte, GT-disi pencerelerin yuzde kaci ELENIR?\n")
    print(f"  {'olcut':<16s} {'GT-ici ort':>11s} {'GT-disi ort':>12s} "
          f"{'esik':>10s} {'ELENEN':>8s}")
    print("  " + "-" * 62)

    sonuc = []
    for ad, yon in ADAYLAR:
        v = np.array([k[ad] for k in kayitlar], dtype=np.float64)
        ici, disi = v[gt], v[~gt]
        if len(ici) == 0 or len(disi) == 0:
            continue
        if yon == "dusuk":
            # dusuk = bos. GT'nin %koruma'sini korumak icin alt kuyruktan esik
            esik = np.quantile(ici, 1 - koruma)
            elenen = (disi < esik).mean()
        else:
            esik = np.quantile(ici, koruma)
            elenen = (disi > esik).mean()
        sonuc.append((elenen, ad, ici.mean(), disi.mean(), esik))
        print(f"  {ad:<16s} {ici.mean():>11.4g} {disi.mean():>12.4g} "
              f"{esik:>10.4g} {100*elenen:>7.1f}%")

    sonuc.sort(reverse=True)
    print("\n  " + "=" * 62)
    if sonuc:
        e, ad, _, _, esik = sonuc[0]
        print(f"  EN IYI: {ad}  ->  esik {esik:.4g}, "
              f"GT-disi pencerelerin %{100*e:.1f}'ini eler")
        mevcut = [s for s in sonuc if s[1] == "bosluk_orani"]
        if mevcut:
            print(f"  MEVCUT: bosluk_orani -> %{100*mevcut[0][0]:.1f}")
    return sonuc


def kanal_raporu(kayitlar, esik=0.9):
    """
    KANAL BAZINDA yanlis alarm dokumu.

    Saha waterfall'inda yanlis alarmin imzasi belirgin: TEK KANAL, SABIT
    SINIF, butun zaman boyunca. Kanal 0'da 30 saniye kesintisiz `cutting`
    gibi. Gercek bir cit olayi birden fazla BITISIK kanala baglanir (GT
    kutulari da oyle); tek kanalda sureklilik fiziksel olarak olay degil,
    KANAL ARTEFAKTIDIR.

    Bu yuzden iki sey birlikte raporlaniyor:
      saldiri%   -- GT disindaki pencerelerin kaci saldiri sinifi aldi
      tutarlilik -- o kanalin GT disi tahminlerinin kaci AYNI sinif
    Ikisi de yuksekse (mesela %80 saldiri + %95 tutarlilik) o kanal
    artefakt uretiyor demektir. Gercek gurultu cesitlenir.
    """
    from collections import Counter
    if "tahmin" not in kayitlar[0]:
        return []
    print("\n" + "=" * 78)
    print("KANAL BAZINDA -- GT DISI PENCERELER")
    print("=" * 78)
    print(f"  {'kanal':>5} {'n':>5} {'saldiri%':>9} {'tutarli%':>9} "
          f"{'baskin':>9} {'bosluk':>8} {'dusuk_fr':>9} {'mad':>8}")
    print("  " + "-" * 68)

    kanallar = sorted({k["kanal"] for k in kayitlar})
    supheli = []
    for kn in kanallar:
        d = [k for k in kayitlar if k["kanal"] == kn and not k["gt"]]
        if len(d) < 3:
            continue
        t = [k["tahmin"] for k in d]
        lg = np.array([k["logit"] for k in d])
        saldiri = np.array([x in ("cutting", "climbing") for x in t])
        emin_saldiri = (saldiri & (lg > esik)).mean()
        say = Counter(t)
        baskin, n_baskin = say.most_common(1)[0]
        tutarlilik = n_baskin / len(t)

        isaret = ""
        if emin_saldiri > 0.3 and tutarlilik > 0.8:
            isaret = "  <-- ARTEFAKT SUPHESI"
            supheli.append((kn, emin_saldiri, tutarlilik, baskin))
        print(f"  {kn:>5} {len(d):>5} {100*emin_saldiri:>8.1f}% "
              f"{100*tutarlilik:>8.1f}% {baskin:>9} "
              f"{np.mean([k['bosluk_orani'] for k in d]):>8.3f} "
              f"{np.mean([k['dusuk_frek'] for k in d]):>9.3f} "
              f"{np.mean([k['mad_ham'] for k in d]):>8.1f}{isaret}")

    if supheli:
        print(f"\n  {len(supheli)} kanal artefakt suphesi tasiyor.")
        print("  Bu kanallarin olcutlerini digerleriyle karsilastir --")
        print("  aralarindaki fark, dogru bosluk olcutunu verir.")
    else:
        print("\n  Artefakt suphesi tasiyan kanal YOK.")
        print("  Saha waterfall'indaki surekli sutunlar bu dosyada")
        print("  okuyabildigimiz kanallarda (71-97) gorunmuyor olabilir --")
        print("  arayuz tum kanallari (201) okuyor, biz 11'ini.")
    return supheli


def kacirma_raporu(kayitlar, esik=0.9):
    """GT-ici pencerelerin kaci KACIRILDI (noise dendi ya da esigin altinda)."""
    if "tahmin" not in kayitlar[0]:
        return
    ici = [k for k in kayitlar if k["gt"]]
    if not ici:
        return
    t = [k["tahmin"] for k in ici]
    lg = np.array([k["logit"] for k in ici])
    noise_dedi = np.array([x == "noise" for x in t])
    esik_alti = lg <= esik
    kacirildi = noise_dedi | esik_alti
    print("\n" + "=" * 78)
    print("KACIRMA -- GT ICI PENCERELER")
    print("=" * 78)
    print(f"  n={len(ici):,}   'noise' dedi: %{100*noise_dedi.mean():.1f}   "
          f"logit<={esik}: %{100*esik_alti.mean():.1f}")
    print(f"  TOPLAM KACIRILAN: %{100*kacirildi.mean():.1f}")
    print("  (cevre guvenliginde kacirilan ihlal, fazladan alarmdan pahalidir)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bosluk olcutu taramasi")
    ap.add_argument("--dizin", default=DIZIN)
    ap.add_argument("--adim", type=int, default=4000,
                    help="pencere adimi (arayuz 4000 kullaniyor)")
    ap.add_argument("--ckpt", default=os.path.join(CIKTI,
                                                   "kosu3_gri_sifirdan.pt"))
    ap.add_argument("--koruma", type=float, default=0.95)
    ap.add_argument("--cihaz", default=None)
    a = ap.parse_args()

    kayitlar = topla(a.dizin, a.adim)
    if not kayitlar:
        sys.exit("hic pencere toplanamadi -- --dizin dogru mu?")

    if a.ckpt and os.path.exists(a.ckpt):
        import torch
        from onnx_disa_aktar import sarmalayici_kur
        cihaz = a.cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
        # bastirma KAPALI: modelin ham davranisini gormek istiyoruz
        sarmal = sarmalayici_kur(ckpt=a.ckpt, bos_bastir=False)
        kayitlar = model_ekle(kayitlar, sarmal, cihaz)
    else:
        for k in kayitlar:
            k.pop("_ham", None)
        print(f"  checkpoint yok ({a.ckpt}) -- yalnizca olcutler")

    rapor(kayitlar, a.koruma)
    kanal_raporu(kayitlar)
    kacirma_raporu(kayitlar)
