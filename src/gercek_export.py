"""
GERCEK VERI -- ADIM 8: MODELI TESLIM EDILEBILIR PAKETE CEVIR

gercek_egitim.py zaten bir .pt yaziyor ama o bir EGITIM ARTIGI, teslim
edilebilir bir model degil. Ikisinin farki:

    kosu3_gri_sifirdan.pt          bu script'in urettigi paket
    ------------------------       --------------------------
    state_dict                     state_dict
    kosu / ayar / epoch            + mimari (kanallar, dikkat modulu, sinif sayisi)
                                   + ON ISLEME TARIFI (fs, pencere, n_fft, hop,
                                     alan, normalizasyon, bos esik, renk)
                                   + sinif adlari VE SIRASI
                                   + olculen performans (test + sinif bazinda)
                                   + egitim kosullari (veri boyutu, tohum, lr...)
                                   + SINIRLAR / uyarilar
                                   + kod surumu (git commit)

=== NEDEN BU FARK ONEMLI ===

Bir model dosyasi tek basina anlamsizdir. Alti ay sonra biri bu .pt'yi
acinca sunlari bilmek zorunda:

  - Girdi neye benziyor? 224x320 mi, 129x231 mi? Gri mi viridis mi?
    Hangi normalizasyon? Yanlis on isleme = coplenmis tahminler, ve
    HICBIR HATA MESAJI VERMEZ.
  - Siniflarin SIRASI ne? 0 cutting mi climbing mi? Yanlis sira =
    sessizce yanlis etiketler.
  - Bu skor nerede olculdu? Egitim setinde mi, ayri bir test setinde mi?

Sentetik asamada bu paketi uretmistik (export_model.py + MODEL_CARD.md) ve
gercek veriye gecerken tam da o dosya sayesinde nasil yukleyecegimizi
bilebildik. Ayni disiplini burada da uyguluyoruz.

=== KULLANIM ===

    python gercek_export.py --kosu 3
    python gercek_export.py --kosu 3 --dogrula     # uretilen paketi sina

Cikti:
    egitim_ciktilari/paket/das_2dcnn_gercek_kosu3.pt
    egitim_ciktilari/paket/MODEL_CARD_gercek.md
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_burada = str(Path(__file__).resolve().parent)
if _burada not in sys.path:
    sys.path.insert(0, _burada)

import real_data as rd
from gercek_veri_kumesi import GIRDI_H, GIRDI_W, NORM_MEAN, NORM_STD
from model import DASNet, count_parameters

VERI = Path("/tf/start_training/RELATIONNET/FENCE_DATA_NEW")
CIKTI = VERI / "egitim_ciktilari"
TABAN = 0.771


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_burada,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def sinif_metrikleri(karisiklik):
    M = np.asarray(karisiklik, dtype=np.float64)
    kosegen = np.diag(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(M.sum(0) > 0, kosegen / M.sum(0), 0.0)
        rec = np.where(M.sum(1) > 0, kosegen / M.sum(1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, M.sum(1).astype(int)


def paketle(kosu, cikti=CIKTI, hedef=None):
    cikti = Path(cikti)
    gecmisler = list(cikti.glob(f"kosu{kosu}_*_gecmis.json"))
    if not gecmisler:
        raise FileNotFoundError(
            f"kosu{kosu}_*_gecmis.json bulunamadi ({cikti}). "
            f"Kosu tamamlanmis mi?")
    g = json.loads(gecmisler[0].read_text(encoding="utf-8"))
    ad = g["ayar"]["ad"]
    ckpt_yol = cikti / f"kosu{kosu}_{ad}.pt"
    if not ckpt_yol.exists():
        raise FileNotFoundError(f"checkpoint yok: {ckpt_yol}")
    ckpt = torch.load(ckpt_yol, map_location="cpu", weights_only=False)

    siniflar = g["siniflar"]
    prec, rec, f1, destek = sinif_metrikleri(g["karisiklik"])

    paket = {
        "ad": f"das_2dcnn_sk_gercek_kosu{kosu}",
        "olusturma_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),

        # --- MIMARI: modeli sifirdan kurmak icin yeterli ---
        "mimari": {
            "sinif": "DASNet",
            "dikkat": "sk",
            "konv_kanallar": [16, 32, 64],
            "omurga_batchnorm": True,
            "dropout": 0.5,
            "sk": {"M": 2, "kernels": [3, 5], "r": 16, "L": 32, "gruplar": 32},
            "n_sinif": len(siniflar),
            "parametre": g["parametre"],
        },

        # --- GIRDI: bunlar olmadan model kullanilamaz ---
        "girdi": {
            "sekil": [3, GIRDI_H, GIRDI_W],
            "duzen": "dikey=frekans, yatay=zaman",
            "renk": g["ayar"]["renk"],
            "norm_mean": list(NORM_MEAN),
            "norm_std": list(NORM_STD),
            "olcekleme": "bilinear, antialias=True",
        },

        # --- ON ISLEME: ham .bin.hdf5'ten girdiye giden TAM tarif ---
        "on_isleme": {
            "kaynak": "real_data.py + onbellek_kur.py",
            "fs": rd.FS,
            "pencere_ornek": rd.PENCERE,
            "pencere_saniye": rd.PENCERE / rd.FS,
            "alan": rd.ALAN,
            "sinyal": "hypot(re, im)  -- genlik, FAZ DEGIL",
            "standartlastirma": "uzunsa enerji merkezine kirp, "
                                "kisaysa yansitmali doldur",
            "bos_pencere_esigi": rd.BOS_ESIK,
            "bos_pencere_frekans": rd.BOS_FREKANS,
            "normalizasyon": "pencere-ici (s - medyan) / MAD",
            "olcek_katsayisi": "UYGULANMAZ (normalizasyon sadelestirir)",
            "stft": {"n_fft": rd.N_FFT, "hop": rd.HOP, "top_db": rd.TOP_DB},
            "spektrogram_sekli": [rd.N_FFT // 2 + 1,
                                  1 + (rd.PENCERE - rd.N_FFT) // rd.HOP],
        },

        # --- SINIFLAR: SIRA KRITIK ---
        "siniflar": siniflar,
        "sinif_indeksi": {s: i for i, s in enumerate(siniflar)},

        # --- OLCULEN PERFORMANS ---
        "performans": {
            "test_macro_f1": g["test_macro_f1"],
            "test_dogruluk": g["test_dogruluk"],
            "test_kayip": g["test_kayip"],
            "val_macro_f1": g["en_iyi_val_macro_f1"],
            "taban_cizgisi_macro_f1": TABAN,
            "taban_cizgisine_gore": g["test_macro_f1"] - TABAN,
            "sinif_bazinda": {
                s: {"precision": round(float(prec[i]), 4),
                    "recall": round(float(rec[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "destek": int(destek[i])}
                for i, s in enumerate(siniflar)},
            "karisiklik_matrisi": g["karisiklik"],
            "karisiklik_aciklama": "satir = gercek, sutun = tahmin",
        },

        # --- EGITIM KOSULLARI ---
        "egitim": {
            "konfigurasyon": ad,
            "kosu": kosu,
            "n_train": g["n_train"], "n_val": g["n_val"], "n_test": g["n_test"],
            "bagimsiz_dosya_train": 21101,
            "epoch_kosulan": len(g["train_kayip"]),
            "en_iyi_epoch": g["en_iyi_epoch"],
            "izlenen_metrik": "val macro-F1",
            "tohum": g["tohum"], "batch": g["batch"], "lr": g["lr"],
            "label_smoothing": g["label_smoothing"],
            "sinif_agirligi": g["sinif_agirligi"],
            "optimizer": "Adam", "weight_decay": 1e-4,
            "artirma": "zaman/frekans maskeleme (1-2 serit, <=%10), "
                       "cevirme YOK",
            "toplam_dakika": g["toplam_dakika"],
            "bolmeler": "ekibin train_final / val_final / test_final "
                        "dosyalari; dosya/oturum/tarih duzeyinde cakisma yok",
        },

        "sinirlar": [
            "Test skoru AYRI bir test setinde olculdu (42.850 pencere) -- "
            "egitimde de dogrulama secilirken de kullanilmadi.",
            "Kalan hatanin neredeyse tamami climbing <-> cutting arasinda; "
            "noise pratikte cozulmus durumda (F1 0.98).",
            "val/test bolmeleri kurasyonlu gorunuyor: bos pencere orani "
            "train'de %23, val/test'te %0.1. Saha kosullarinda zayif "
            "kanallar daha sik olacaktir.",
            "Bos pencereler cikarimda da elenmelidir (pencere_yukle None "
            "dondurur) -- model onlar icin egitilmedi.",
            "Girdi ON ISLEMESI birebir ayni olmalidir; farkli bir "
            "normalizasyon veya pencere boyutu sessizce bozuk tahmin uretir.",
            "Tek tohumla tek kosu. Tohum varyansi olculmedi.",
        ],

        "state_dict": ckpt["state_dict"],
    }

    hedef = Path(hedef) if hedef else (cikti / "paket")
    hedef.mkdir(parents=True, exist_ok=True)
    pt = hedef / f"{paket['ad']}.pt"
    torch.save(paket, pt)
    kart = model_karti(paket, hedef)
    print(f"  paket : {pt}  ({pt.stat().st_size/1024:.0f} KB)")
    print(f"  kart  : {kart}")
    return pt


def model_karti(p, hedef):
    perf, eg = p["performans"], p["egitim"]
    s = [f"# DAS 2D-CNN + SK — Gerçek Saha Verisi Modeli", "",
         f"**Dosya:** `{p['ad']}.pt`  ",
         f"**Üretim:** {p['olusturma_utc']}  ",
         f"**Kod sürümü:** `{(p['git_commit'] or 'bilinmiyor')[:12]}`", "",
         "## Performans", "",
         "| | |", "|---|---|",
         f"| **test macro-F1** | **{perf['test_macro_f1']:.4f}** |",
         f"| test doğruluk | {perf['test_dogruluk']:.4f} |",
         f"| doğrulama macro-F1 | {perf['val_macro_f1']:.4f} |",
         f"| taban çizgisi (doğrusal, 26 özellik) | {perf['taban_cizgisi_macro_f1']:.3f} |",
         f"| **taban çizgisine göre** | **{perf['taban_cizgisine_gore']:+.4f}** |",
         "", "### Sınıf bazında (test)", "",
         "| sınıf | precision | recall | F1 | destek |", "|---|---|---|---|---|"]
    for ad, m in perf["sinif_bazinda"].items():
        s.append(f"| `{ad}` | {m['precision']:.3f} | {m['recall']:.3f} "
                 f"| **{m['f1']:.3f}** | {m['destek']:,} |")
    s += ["", "Karışıklık matrisi (satır = gerçek):", "",
          "| | " + " | ".join(f"`{c}`" for c in p["siniflar"]) + " |",
          "|---" * (len(p["siniflar"]) + 1) + "|"]
    for i, ad in enumerate(p["siniflar"]):
        s.append(f"| **`{ad}`** | "
                 + " | ".join(f"{v:,}" for v in perf["karisiklik_matrisi"][i])
                 + " |")

    oi = p["on_isleme"]
    s += ["", "## Eğitim", "",
          f"- Konfigürasyon: **{eg['konfigurasyon']}** (koşu {eg['kosu']})",
          f"- Veri: train {eg['n_train']:,} / val {eg['n_val']:,} / "
          f"test {eg['n_test']:,} pencere",
          f"- Bağımsız kayıt dosyası (train): {eg['bagimsiz_dosya_train']:,}",
          f"- {eg['epoch_kosulan']} epoch koşuldu, en iyi epoch "
          f"{eg['en_iyi_epoch']} ({eg['izlenen_metrik']} izlendi)",
          f"- Adam, lr {eg['lr']}, batch {eg['batch']}, "
          f"label smoothing {eg['label_smoothing']}, tohum {eg['tohum']}",
          f"- Artırma: {eg['artirma']}",
          f"- Bölmeler: {eg['bolmeler']}", "",
          "## Girdi — bu tarif birebir uygulanmalı", "",
          "```", "1. .bin.hdf5 ac, kanal veri kumesini oku",
          f"2. {oi['alan']} alanindan: {oi['sinyal']}",
          "3. CSV penceresini kes [window_start:window_end]",
          f"4. {oi['pencere_ornek']:,} ornege standartlastir "
          f"({oi['pencere_saniye']} s @ {oi['fs']} Hz)",
          f"   {oi['standartlastirma']}",
          f"5. Bos mu? ({oi['bos_pencere_frekans']:.0f} Hz ustu enerji payi > "
          f"{oi['bos_pencere_esigi']}) -> bossa ELE",
          f"6. Normalize: {oi['normalizasyon']}",
          f"7. STFT n_fft={oi['stft']['n_fft']}, hop={oi['stft']['hop']} "
          f"-> {oi['spektrogram_sekli'][0]} x {oi['spektrogram_sekli'][1]} dB",
          f"8. Renk: {p['girdi']['renk']},  olcek {p['girdi']['sekil'][1]}x"
          f"{p['girdi']['sekil'][2]} ({p['girdi']['olcekleme']})",
          f"9. [0,1] -> Normalize(mean={p['girdi']['norm_mean']}, "
          f"std={p['girdi']['norm_std']})", "```", "",
          f"⚠️ **Ölçek katsayısı ({oi['olcek_katsayisi']}).**", "",
          f"⚠️ **Sınıf sırası:** `{p['sinif_indeksi']}` — "
          f"karıştırılırsa model sessizce yanlış etiket üretir.", "",
          "## Nasıl yüklenir", "",
          "```python", "import torch", "from model import DASNet", "",
          f"paket = torch.load('{p['ad']}.pt', map_location='cpu', "
          f"weights_only=False)",
          f"model = DASNet(attention='sk', n_classes={len(p['siniflar'])})",
          "model.load_state_dict(paket['state_dict'])", "model.eval()", "",
          "# on isleme: real_data.py + gercek_veri_kumesi.hazirla()",
          "```", "", "## Sınırlar", ""]
    s += [f"- {c}" for c in p["sinirlar"]]
    s += ["", "## Paketin içindekiler", "",
          "`mimari`, `girdi`, `on_isleme`, `siniflar`, `sinif_indeksi`, "
          "`performans`, `egitim`, `sinirlar`, `state_dict`. Yani model "
          "başka bir projede **bu dosyaya bakarak** yeniden kurulabilir.", ""]

    yol = hedef / "MODEL_CARD_gercek.md"
    yol.write_text("\n".join(s), encoding="utf-8")
    return yol


def dogrula(pt):
    """Paket kendi tarifiyle gercekten yuklenebiliyor mu?"""
    print("=" * 78)
    print(f"PAKET DOGRULAMA -- {pt}")
    print("=" * 78)
    p = torch.load(str(pt), map_location="cpu", weights_only=False)

    print(f"  ad          : {p['ad']}")
    print(f"  siniflar    : {p['siniflar']}")
    print(f"  test macro-F1: {p['performans']['test_macro_f1']:.4f}")

    # 1) Paketin TARIFIYLE modeli kur -- disaridan bilgi kullanmadan
    m = DASNet(attention=p["mimari"]["dikkat"],
               n_classes=p["mimari"]["n_sinif"])
    eksik, fazla = m.load_state_dict(p["state_dict"], strict=True), None
    print(f"  [x] state_dict tam olarak yuklendi (strict=True)")
    n = count_parameters(m)
    assert n == p["mimari"]["parametre"], \
        f"parametre uyusmazligi: {n} != {p['mimari']['parametre']}"
    print(f"  [x] parametre sayisi eslesti: {n:,}")

    # 2) Paketin verdigi girdi sekliyle ileri gecis
    m.eval()
    x = torch.randn(2, *p["girdi"]["sekil"])
    with torch.no_grad():
        cikti = m(x)
    assert cikti.shape == (2, len(p["siniflar"]))
    assert torch.isfinite(cikti).all()
    print(f"  [x] girdi {tuple(x.shape)} -> logit {tuple(cikti.shape)}")

    # 3) On isleme tarifi hattaki gercek degerlerle tutarli mi
    oi = p["on_isleme"]
    assert oi["fs"] == rd.FS and oi["pencere_ornek"] == rd.PENCERE
    assert oi["stft"]["n_fft"] == rd.N_FFT and oi["stft"]["hop"] == rd.HOP
    print(f"  [x] on isleme tarifi real_data.py ile tutarli")

    # 4) Zorunlu alanlar
    for alan in ("mimari", "girdi", "on_isleme", "siniflar", "sinif_indeksi",
                 "performans", "egitim", "sinirlar", "state_dict"):
        assert alan in p, f"pakette '{alan}' yok"
    print(f"  [x] tum zorunlu alanlar mevcut ({len(p['sinirlar'])} sinir notu)")

    print(f"\n  DOGRULAMA GECTI -- paket kendi kendine yeter.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Modeli teslim paketine cevir")
    ap.add_argument("--kosu", type=int, default=3)
    ap.add_argument("--cikti", default=str(CIKTI))
    ap.add_argument("--hedef", default=None)
    ap.add_argument("--dogrula", default=None,
                    help="var olan bir paketi dogrula")
    a = ap.parse_args()
    if a.dogrula:
        sys.exit(dogrula(a.dogrula))
    yol = paketle(a.kosu, cikti=a.cikti, hedef=a.hedef)
    print()
    sys.exit(dogrula(yol))
