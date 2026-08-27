"""
GERCEK VERI -- ADIM 7: UC KOSUYU KARSILASTIR VE RAPORLA

gercek_egitim.py'nin yazdigi *_gecmis.json dosyalarini okur, karsilastirma
tablosunu ve sekilleri uretir, markdown ozet yazar.

=== NEDEN AYRI BIR SCRIPT ===

Sonuclari elle tabloya dokmek iki sorun uretir: (a) kopyalama hatasi,
(b) rapordaki sayi ile uretilen sayi arasinda sessiz ayrisma. Sentetik
asamada rapora giren her sayi kod ciktisindan geldi, burada da oyle olsun.

Ayrica model YENIDEN CALISTIRILMIYOR: sinif bazinda precision/recall/F1,
kaydedilmis KARISIKLIK MATRISINDEN tam olarak hesaplanabiliyor. Test
setine ikinci kez dokunmuyoruz.

=== NE URETIR ===

    1. Ana karsilastirma tablosu (val/test macro-F1, en iyi epoch, sure)
    2. Sinif bazinda precision / recall / F1  -- her kosu icin
    3. Karisiklik matrisleri
    4. Ogrenme egrileri  (matplotlib varsa)
    5. gercek_veri_sonuclari.md  -- rapora dogrudan girecek markdown

=== KULLANIM ===

    python gercek_rapor.py
    python gercek_rapor.py --cikti /tf/.../egitim_ciktilari

Eksik kosular sessizce atlanir -- ilk kosu biter bitmez calistirilabilir.

=== BAGIMLILIK ===

numpy. matplotlib VARSA sekil de uretir, yoksa yalnizca metin.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Asilmasi gereken taban cizgisi: dogrusal siniflandirici + 26 elle
# cikarilmis ozellik, ayni val setinde (Rapor Bolum 5.4).
TABAN = 0.771
TABAN_SINIF = {"noise": 0.981, "cutting": 0.678, "climbing": 0.653}

KOSU_ACIKLAMA = {
    1: "viridis girdi, sifirdan egitim  (referans)",
    2: "viridis girdi, sentetik aktarim (1'e karsi: aktarim ise yariyor mu)",
    3: "gri girdi, sifirdan egitim      (1'e karsi: viridis zarar veriyor mu)",
}

CIZGI = "-" * 78
CIFT = "=" * 78


def yukle(cikti):
    """Bulunan tum *_gecmis.json dosyalarini kosu numarasina gore dondurur."""
    bulunan = {}
    for yol in sorted(Path(cikti).glob("kosu*_gecmis.json")):
        d = json.loads(yol.read_text(encoding="utf-8"))
        bulunan[int(d["kosu"])] = d
    return bulunan


def sinif_metrikleri(karisiklik):
    """
    Karisiklik matrisinden sinif bazinda precision / recall / F1.

    Satir = gercek, sutun = tahmin.
        precision_k = M[k,k] / sutun_toplami[k]
        recall_k    = M[k,k] / satir_toplami[k]
    Model yeniden calistirilmiyor; test setine ikinci kez dokunulmuyor.
    """
    M = np.asarray(karisiklik, dtype=np.float64)
    kosegen = np.diag(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(M.sum(0) > 0, kosegen / M.sum(0), 0.0)
        rec = np.where(M.sum(1) > 0, kosegen / M.sum(1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, M.sum(1).astype(int)


def ana_tablo(kosular, yaz=print):
    yaz(f"\n{CIFT}")
    yaz("ANA KARSILASTIRMA")
    yaz(CIFT)
    yaz(f"  {'#':<3}{'konfigurasyon':<20}{'val F1':>9}{'test F1':>9}"
        f"{'test dog.':>11}{'epoch':>7}{'sure':>8}")
    yaz("  " + CIZGI[:60])
    for no in sorted(kosular):
        d = kosular[no]
        yaz(f"  {no:<3}{d['ayar']['ad']:<20}"
            f"{d['en_iyi_val_macro_f1']:>9.4f}{d['test_macro_f1']:>9.4f}"
            f"{d['test_dogruluk']:>11.4f}{d['en_iyi_epoch']:>7}"
            f"{d['toplam_dakika']:>7.0f}dk")
    yaz("  " + CIZGI[:60])
    yaz(f"  {'--':<3}{'TABAN (dogrusal)':<20}{TABAN:>9.4f}{'':>9}"
        f"{'':>11}{'':>7}{'':>8}")

    yaz(f"\n  Taban cizgisine gore fark (test macro-F1 - {TABAN}):")
    for no in sorted(kosular):
        d = kosular[no]
        fark = d["test_macro_f1"] - TABAN
        isaret = "ASILDI" if fark > 0 else "asilamadi"
        yaz(f"    kosu {no} ({d['ayar']['ad']:<18}) {fark:+.4f}   {isaret}")


def kosu_detay(no, d, yaz=print):
    yaz(f"\n{CIFT}")
    yaz(f"KOSU {no} -- {d['ayar']['ad']}")
    yaz(CIFT)
    yaz(f"  {KOSU_ACIKLAMA.get(no, '')}")
    yaz(f"  train {d['n_train']:,} | val {d['n_val']:,} | test {d['n_test']:,}")
    yaz(f"  parametre {d['parametre']:,} | tohum {d['tohum']} | "
        f"batch {d['batch']} | lr {d['lr']}")
    yaz(f"  en iyi epoch {d['en_iyi_epoch']} / {len(d['train_kayip'])} kosulan")

    siniflar = d["siniflar"]
    prec, rec, f1, destek = sinif_metrikleri(d["karisiklik"])
    yaz(f"\n  Sinif bazinda (test):")
    yaz(f"  {'sinif':<12}{'precision':>11}{'recall':>9}{'F1':>8}"
        f"{'destek':>10}{'taban F1':>10}{'fark':>8}")
    for i, s in enumerate(siniflar):
        tb = TABAN_SINIF.get(s)
        yaz(f"  {s:<12}{prec[i]:>11.3f}{rec[i]:>9.3f}{f1[i]:>8.3f}"
            f"{destek[i]:>10,}"
            + (f"{tb:>10.3f}{f1[i]-tb:>+8.3f}" if tb else f"{'-':>10}{'-':>8}"))
    yaz(f"  {'macro':<12}{prec.mean():>11.3f}{rec.mean():>9.3f}"
        f"{f1.mean():>8.3f}{destek.sum():>10,}{TABAN:>10.3f}"
        f"{f1.mean()-TABAN:>+8.3f}")

    M = np.asarray(d["karisiklik"])
    yaz(f"\n  Karisiklik matrisi (satir = gercek, sutun = tahmin):")
    yaz(f"  {'':<12}" + "".join(f"{s[:9]:>11}" for s in siniflar))
    for i, s in enumerate(siniflar):
        yaz(f"  {s:<12}" + "".join(
            f"{v:>11,}" if i != j else f"{('['+format(v, ',')+']'):>11}"
            for j, v in enumerate(M[i])))

    # En buyuk karisiklik cifti -- hikayenin ozeti genelde burada
    hata = M.copy()
    np.fill_diagonal(hata, 0)
    if hata.sum() > 0:
        i, j = np.unravel_index(hata.argmax(), hata.shape)
        yaz(f"\n  En buyuk karisiklik: {siniflar[i]} -> {siniflar[j]}  "
            f"({M[i, j]:,} ornek, {siniflar[i]} sinifinin "
            f"%{100*M[i,j]/max(M[i].sum(),1):.1f}'i)")

    # Asiri ogrenme gostergesi
    ep = d["en_iyi_epoch"] - 1
    if 0 <= ep < len(d["train_dogruluk"]):
        fark = d["train_dogruluk"][ep] - d["val_dogruluk"][ep]
        yaz(f"  En iyi epoch'ta egitim-dogrulama dogruluk farki: {fark:+.3f}"
            + ("   <- asiri ogrenme suphesi" if fark > 0.15 else ""))


def karsilastirmalar(kosular, yaz=print):
    """Tasarlanan iki karsilastirma: aktarim etkisi, viridis etkisi."""
    yaz(f"\n{CIFT}")
    yaz("TASARLANAN KARSILASTIRMALAR")
    yaz(CIFT)

    if 1 in kosular and 2 in kosular:
        a, b = kosular[1]["test_macro_f1"], kosular[2]["test_macro_f1"]
        yaz(f"\n  AKTARIM ETKISI  (kosu 2 - kosu 1, ikisi de viridis)")
        yaz(f"    sifirdan {a:.4f}  ->  aktarimli {b:.4f}   fark {b-a:+.4f}")
        yaz(f"    {_yorum(b - a, 'Sentetik on-egitim')}")
    else:
        yaz(f"\n  AKTARIM ETKISI  : kosu 1 ve 2 gerekli, henuz yok")

    if 1 in kosular and 3 in kosular:
        a, c = kosular[1]["test_macro_f1"], kosular[3]["test_macro_f1"]
        yaz(f"\n  GIRDI TEMSILI   (kosu 3 - kosu 1, ikisi de sifirdan)")
        yaz(f"    viridis {a:.4f}  ->  gri {c:.4f}   fark {c-a:+.4f}")
        yaz(f"    {_yorum(c - a, 'Gri temsil')}")
    else:
        yaz(f"\n  GIRDI TEMSILI   : kosu 1 ve 3 gerekli, henuz yok")

    yaz(f"\n  UYARI: her karsilastirma TEK kosu ciftine dayaniyor. Fark"
        f"\n  kucukse (|fark| < 0.01) tohum gurultusu olabilir; kesin konusmak"
        f"\n  icin farkli tohumlarla tekrar gerekir.")


def _yorum(fark, ad):
    if abs(fark) < 0.01:
        return f"-> {ad} olculebilir bir fark yaratmadi."
    return (f"-> {ad} {'FAYDA SAGLADI' if fark > 0 else 'ZARAR VERDI'} "
            f"({abs(fark):.4f}).")


def egriler(kosular, cikti):
    """Ogrenme egrileri. matplotlib yoksa sessizce atlanir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib yok -- sekiller atlandi)")
        return None

    fig, eks = plt.subplots(1, 3, figsize=(16, 4.2))
    for no in sorted(kosular):
        d = kosular[no]
        ep = np.arange(1, len(d["train_kayip"]) + 1)
        etiket = f"{no}: {d['ayar']['ad']}"
        eks[0].plot(ep, d["train_kayip"], label=etiket)
        eks[1].plot(ep, d["val_kayip"], label=etiket)
        eks[2].plot(ep, d["val_macro_f1"], label=etiket)
        eks[2].scatter([d["en_iyi_epoch"]], [d["en_iyi_val_macro_f1"]],
                       marker="o", zorder=5)
    for e, b in zip(eks, ("Egitim kaybi", "Dogrulama kaybi",
                          "Dogrulama macro-F1")):
        e.set_xlabel("epoch"); e.set_title(b); e.grid(alpha=.3); e.legend()
    eks[2].axhline(TABAN, ls="--", c="gray", lw=1)
    eks[2].annotate(f"taban {TABAN}", (0.02, TABAN + 0.005),
                    xycoords=("axes fraction", "data"), fontsize=8, color="gray")
    fig.suptitle("Gercek veri -- uc kosu")
    fig.tight_layout()
    yol = Path(cikti) / "kosu_egrileri.png"
    fig.savefig(yol, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  sekil: {yol}")
    return yol


def markdown(kosular, cikti):
    """Rapora dogrudan girecek markdown ozet."""
    s = ["# Gerçek Veri — Eğitim Sonuçları", "",
         f"> Üretildi: `gercek_rapor.py`. Aşılması gereken taban çizgisi "
         f"**macro-F1 {TABAN}** (doğrusal sınıflandırıcı, 26 özellik).", "",
         "## Ana tablo", "",
         "| # | konfigürasyon | val macro-F1 | test macro-F1 | test doğruluk "
         "| en iyi epoch | süre |", "|---|---|---|---|---|---|---|"]
    for no in sorted(kosular):
        d = kosular[no]
        s.append(f"| {no} | {d['ayar']['ad']} | {d['en_iyi_val_macro_f1']:.4f} "
                 f"| **{d['test_macro_f1']:.4f}** | {d['test_dogruluk']:.4f} "
                 f"| {d['en_iyi_epoch']} | {d['toplam_dakika']:.0f} dk |")
    s += ["| — | *taban (doğrusal)* | — | *0.771* | — | — | — |", ""]

    for no in sorted(kosular):
        d = kosular[no]
        siniflar = d["siniflar"]
        prec, rec, f1, destek = sinif_metrikleri(d["karisiklik"])
        s += [f"## Koşu {no} — {d['ayar']['ad']}", "",
              f"{KOSU_ACIKLAMA.get(no, '')}", "",
              "| sınıf | precision | recall | F1 | destek |",
              "|---|---|---|---|---|"]
        for i, ad in enumerate(siniflar):
            s.append(f"| {ad} | {prec[i]:.3f} | {rec[i]:.3f} | {f1[i]:.3f} "
                     f"| {destek[i]:,} |")
        s.append(f"| **macro** | {prec.mean():.3f} | {rec.mean():.3f} "
                 f"| **{f1.mean():.3f}** | {destek.sum():,} |")
        s += ["", "Karışıklık matrisi (satır = gerçek):", "",
              "| | " + " | ".join(siniflar) + " |",
              "|---" * (len(siniflar) + 1) + "|"]
        for i, ad in enumerate(siniflar):
            s.append(f"| **{ad}** | "
                     + " | ".join(f"{v:,}" for v in d["karisiklik"][i]) + " |")
        s.append("")

    yol = Path(cikti) / "gercek_veri_sonuclari.md"
    yol.write_text("\n".join(s), encoding="utf-8")
    print(f"  markdown: {yol}")
    return yol


def rapor(cikti):
    cikti = Path(cikti)
    kosular = yukle(cikti)
    print(CIFT)
    print(f"GERCEK VERI -- KOSU RAPORU   ({cikti})")
    print(CIFT)
    if not kosular:
        print(f"  *_gecmis.json bulunamadi. Kosu bitmemis olabilir.")
        return 1
    eksik = [n for n in (1, 2, 3) if n not in kosular]
    print(f"  bulunan kosular: {sorted(kosular)}"
          + (f"   eksik: {eksik}" if eksik else "   (hepsi tamam)"))

    ana_tablo(kosular)
    for no in sorted(kosular):
        kosu_detay(no, kosular[no])
    karsilastirmalar(kosular)
    egriler(kosular, cikti)
    markdown(kosular, cikti)
    print(f"\n{CIFT}\nRAPOR TAMAM.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Uc kosuyu karsilastir")
    ap.add_argument("--cikti",
                    default="/tf/start_training/RELATIONNET/FENCE_DATA_NEW/egitim_ciktilari")
    sys.exit(rapor(ap.parse_args().cikti))
