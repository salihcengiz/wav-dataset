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
    4: "CNN-BiLSTM, yeni rejim          (zamansal dizi + maskeleme kapali)",
    5: "SK + gri, YENI REJIM            (3'e karsi: rejim etkisi tek basina)",
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
    # "lr" epoch basina liste; baslangic degeri ayri anahtarda. Eski
    # dosyalarda ise "lr" skaler (bkz. gercek_egitim.py'deki ezme hatasi).
    lr0 = d.get("lr_baslangic")
    if lr0 is None:
        lr0 = d["lr"][0] if isinstance(d.get("lr"), list) else d.get("lr")
    yaz(f"  parametre {d['parametre']:,} | tohum {d['tohum']} | "
        f"batch {d['batch']} | lr {lr0}")
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

    if 1 in kosular and 4 in kosular:
        a, d4 = kosular[1]["test_macro_f1"], kosular[4]["test_macro_f1"]
        yaz(f"\n  ZAMANSAL DIZI   (kosu 4 - kosu 1)")
        yaz(f"    SK+havuzlama {a:.4f}  ->  BiLSTM {d4:.4f}   fark {d4-a:+.4f}")
        yaz(f"    {_yorum(d4 - a, 'BiLSTM basi')}")
        # Hipotez climbing/cutting hakkindaydi; noise zaten cozulmustu.
        # Kazanc yalnizca noise'da ise hipotez DESTEKLENMEMIS demektir.
        s1 = dict(zip(kosular[1]["siniflar"],
                      sinif_metrikleri(kosular[1]["karisiklik"])[2]))
        s4 = dict(zip(kosular[4]["siniflar"],
                      sinif_metrikleri(kosular[4]["karisiklik"])[2]))
        yaz(f"    sinif bazinda fark:")
        for ad in kosular[1]["siniflar"]:
            if ad in s4:
                yaz(f"      {ad:<10} {s1[ad]:.3f} -> {s4[ad]:.3f}  "
                    f"{s4[ad]-s1[ad]:+.3f}")
        zor = [c for c in ("climbing", "cutting") if c in s1 and c in s4]
        if zor:
            kazanc = sum(s4[c] - s1[c] for c in zor) / len(zor)
            yaz(f"    -> climbing/cutting ortalama fark: {kazanc:+.4f}")
            if kazanc <= 0:
                yaz(f"       HIPOTEZ DESTEKLENMEDI: zamansal dizi, hedeflenen")
                yaz(f"       climbing/cutting ayrimini iyilestirmedi.")
        yaz(f"\n    ATIF UYARISI: kosu 4 mimariyi VE rejimi (maskeleme kapali,")
        yaz(f"    uzun butce) birlikte degistiriyor. Fark hangisinden geldigi")
        yaz(f"    bilinmiyor. Atif icin SK modelini de ayni rejimle kosmak")
        yaz(f"    gerekir: --kosu 1 --maske-p 0 --epoch 80 --sabir 10")

    if 3 in kosular and 5 in kosular:
        e, y = kosular[3]["test_macro_f1"], kosular[5]["test_macro_f1"]
        rejim = y - e
        yaz(f"\n  REJIM ETKISI    (kosu 5 - kosu 3, ikisi de SK + gri)")
        yaz(f"    eski rejim {e:.4f}  ->  yeni rejim {y:.4f}   fark {rejim:+.4f}")
        yaz(f"    {_yorum(rejim, 'Yeni rejim (maskeleme kapali, uzun butce)')}")

        # MIMARI ETKISI -- fark-farki (difference in differences).
        # kosu4-kosu1 mimari VE rejimi birlikte tasiyor; rejim payini
        # kosu5-kosu3'ten cikarip kalani mimariye atfediyoruz.
        if 1 in kosular and 4 in kosular:
            birlikte = kosular[4]["test_macro_f1"] - kosular[1]["test_macro_f1"]
            mimari = birlikte - rejim
            yaz(f"\n  MIMARI ETKISI (tahmin, fark-farki)")
            yaz(f"    mimari + rejim (kosu 4 - kosu 1) : {birlikte:+.4f}")
            yaz(f"    yalniz rejim   (kosu 5 - kosu 3) : {rejim:+.4f}")
            yaz(f"    {'-' * 44}")
            yaz(f"    BiLSTM'e kalan                   : {mimari:+.4f}")
            if mimari > 0.02:
                yaz(f"    -> Kazancin BUYUK KISMI mimariden geliyor.")
            elif mimari > 0.005:
                yaz(f"    -> Mimari katki var ama rejim de onemli pay aliyor.")
            elif mimari > -0.005:
                yaz(f"    -> Kazanc neredeyse TAMAMEN rejimden; BiLSTM'in ek")
                yaz(f"       karmasikligi kendini savunmuyor.")
            else:
                yaz(f"    -> Rejim tek basina daha iyi; BiLSTM zarar veriyor.")
            temsil = kosular[1]["test_macro_f1"] - kosular[3]["test_macro_f1"]
            yaz(f"    VARSAYIM: rejim etkisi gri ile viridis'te benzer.")
            yaz(f"    (viridis - gri = {temsil:+.4f}, yani temsil farki kucuk;")
            yaz(f"     varsayim makul ama VARSAYIM -- rapora oyle yazilmali.)")

    yaz(f"\n  UYARI: her karsilastirma TEK kosu ciftine dayaniyor. Fark"
        f"\n  kucukse (|fark| < 0.01) tohum gurultusu olabilir; kesin konusmak"
        f"\n  icin farkli tohumlarla tekrar gerekir.")


def _yorum(fark, ad):
    if abs(fark) < 0.01:
        return f"-> {ad} olculebilir bir fark yaratmadi."
    return (f"-> {ad} {'FAYDA SAGLADI' if fark > 0 else 'ZARAR VERDI'} "
            f"({abs(fark):.4f}).")


def _plt():
    """matplotlib varsa dondurur, yoksa None. Sunucuda kurulu olmayabilir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("\n  (matplotlib yok -- sekiller atlandi)")
        return None


RENKLER = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red",
           5: "tab:purple"}


def egriler(kosular, cikti):
    """
    Ogrenme egrileri: kayip, dogruluk, macro-F1, ogrenme orani.

    Duz cizgi = egitim, kesikli = dogrulama. Egitim metrikleri MASKELENMIS
    girdiler uzerinde olculuyor (artirma yalnizca egitimde acik), bu yuzden
    dogrulamanin egitimin USTUNDE olmasi normaldir ve asiri ogrenme
    olmadiginin isaretidir.
    """
    plt = _plt()
    if plt is None:
        return None

    fig, eks = plt.subplots(2, 2, figsize=(13, 8))
    (a_kayip, a_dog), (a_f1, a_lr) = eks
    lr_var = [False]          # en az bir kosuda kullanilabilir lr gecmisi var mi

    for no in sorted(kosular):
        d = kosular[no]
        c = RENKLER.get(no, None)
        ep = np.arange(1, len(d["train_kayip"]) + 1)
        et = f"{no}: {d['ayar']['ad']}"

        a_kayip.plot(ep, d["train_kayip"], c=c, label=f"{et} (egitim)")
        a_kayip.plot(ep, d["val_kayip"], c=c, ls="--", label=f"{et} (dogrulama)")
        a_dog.plot(ep, d["train_dogruluk"], c=c, label=f"{et} (egitim)")
        a_dog.plot(ep, d["val_dogruluk"], c=c, ls="--", label=f"{et} (dogrulama)")
        a_f1.plot(ep, d["val_macro_f1"], c=c, label=et)
        # lr gecmisi bozuk/eksik olabilir: kosu 3'te "lr" anahtari skalerle
        # ezilmisti (gercek_egitim.py'de duzeltildi). Eski dosyalar da
        # okunabilsin diye uzunluk kontrolu yapiliyor.
        lr_gecmis = d.get("lr")
        if isinstance(lr_gecmis, (list, tuple)) and len(lr_gecmis) == len(ep):
            a_lr.plot(ep, lr_gecmis, c=c, label=et, drawstyle="steps-post")
            lr_var[0] = True
        a_f1.scatter([d["en_iyi_epoch"]], [d["en_iyi_val_macro_f1"]],
                     c=c, marker="o", s=60, zorder=5, edgecolors="k",
                     linewidths=.6)
        a_f1.annotate(f"en iyi: {d['en_iyi_val_macro_f1']:.4f}\n"
                      f"(epoch {d['en_iyi_epoch']})",
                      (d["en_iyi_epoch"], d["en_iyi_val_macro_f1"]),
                      textcoords="offset points", xytext=(8, -18),
                      fontsize=8, color=c)

    a_kayip.set_title("Kayip"); a_kayip.set_ylabel("kayip")
    a_dog.set_title("Dogruluk"); a_dog.set_ylabel("dogruluk")
    a_f1.set_title("Dogrulama macro-F1  (model secimi buna gore)")
    a_f1.set_ylabel("macro-F1")
    a_lr.set_title("Ogrenme orani"); a_lr.set_ylabel("lr")
    if lr_var[0]:
        a_lr.set_yscale("log")
    else:
        a_lr.text(0.5, 0.5,
                  "lr gecmisi kayitli degil\n(eski kosu dosyasi)",
                  ha="center", va="center", transform=a_lr.transAxes,
                  fontsize=10, color="gray")
        a_lr.set_xticks([]); a_lr.set_yticks([])

    # Taban cizgisi -- asilmasi gereken esik
    a_f1.axhline(TABAN, ls=":", c="crimson", lw=1.5)
    a_f1.annotate(f"taban cizgisi {TABAN} (dogrusal, 26 ozellik)",
                  (0.02, TABAN), xycoords=("axes fraction", "data"),
                  xytext=(0, 5), textcoords="offset points",
                  fontsize=8, color="crimson")

    for e in (a_kayip, a_dog, a_f1):
        e.set_xlabel("epoch"); e.grid(alpha=.3); e.legend(fontsize=8)
    if lr_var[0]:
        a_lr.set_xlabel("epoch"); a_lr.grid(alpha=.3); a_lr.legend(fontsize=8)

    fig.suptitle("Gercek saha verisi -- ogrenme egrileri\n"
                 "(duz = egitim, kesikli = dogrulama; egitim metrikleri "
                 "maskelenmis girdiler uzerinde)", fontsize=11)
    fig.tight_layout()
    yol = Path(cikti) / "ogrenme_egrileri.png"
    fig.savefig(yol, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  sekil: {yol}")
    return yol


def karisiklik_sekli(kosular, cikti):
    """
    Karisiklik matrisi isi haritasi -- her kosu icin bir panel.

    Renk SATIR BAZINDA normalize (yani recall). Ham sayilarla renklendirmek
    yaniltirdi: noise sinifinda 3.554 ornek var, climbing'de 21.910; ham
    sayida noise satiri hep soluk gorunur ve "kotu tanindigi" izlenimi verir.
    Hucrelerde hem yuzde hem ham sayi yaziyor.
    """
    plt = _plt()
    if plt is None:
        return None

    n = len(kosular)
    fig, eks = plt.subplots(1, n, figsize=(5.4 * n, 4.8), squeeze=False)
    for sut, no in enumerate(sorted(kosular)):
        d = kosular[no]
        siniflar = d["siniflar"]
        M = np.asarray(d["karisiklik"], dtype=np.float64)
        oran = M / np.maximum(M.sum(1, keepdims=True), 1)

        e = eks[0][sut]
        im = e.imshow(oran, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(siniflar)):
            for j in range(len(siniflar)):
                e.text(j, i, f"{oran[i, j]:.1%}\n{int(M[i, j]):,}",
                       ha="center", va="center", fontsize=9,
                       color="white" if oran[i, j] > 0.5 else "black")
        e.set_xticks(range(len(siniflar)), siniflar, rotation=20)
        e.set_yticks(range(len(siniflar)), siniflar)
        e.set_xlabel("tahmin"); e.set_ylabel("gercek")
        e.set_title(f"Kosu {no} -- {d['ayar']['ad']}\n"
                    f"test macro-F1 {d['test_macro_f1']:.4f}  "
                    f"(taban {TABAN}, {d['test_macro_f1']-TABAN:+.3f})",
                    fontsize=10)
        fig.colorbar(im, ax=e, fraction=0.046, label="satir orani (recall)")

    fig.suptitle("Karisiklik matrisi -- TEST seti "
                 "(satir = gercek, sutun = tahmin)", fontsize=12)
    fig.tight_layout()
    yol = Path(cikti) / "karisiklik_matrisi.png"
    fig.savefig(yol, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  sekil: {yol}")
    return yol


def sinif_sekli(kosular, cikti):
    """
    Sinif bazinda F1 -- taban cizgisiyle yan yana cubuk grafik.

    Ana hikaye bu grafikte: kazanc climbing ve cutting'de, noise'da degil.
    """
    plt = _plt()
    if plt is None:
        return None

    fig, e = plt.subplots(figsize=(1.8 + 2.6 * len(kosular), 4.4))
    siniflar = kosular[sorted(kosular)[0]]["siniflar"]
    x = np.arange(len(siniflar))
    grup = len(kosular) + 1
    gen = 0.8 / grup

    taban = [TABAN_SINIF.get(s, np.nan) for s in siniflar]
    e.bar(x - 0.4 + gen / 2, taban, gen, label="taban (dogrusal)",
          color="lightgray", edgecolor="gray")
    for k, no in enumerate(sorted(kosular)):
        d = kosular[no]
        _, _, f1, _ = sinif_metrikleri(d["karisiklik"])
        e.bar(x - 0.4 + gen * (k + 1.5), f1, gen,
              label=f"kosu {no}: {d['ayar']['ad']}", color=RENKLER.get(no))
        for i, v in enumerate(f1):
            e.text(x[i] - 0.4 + gen * (k + 1.5), v + 0.012, f"{v:.3f}",
                   ha="center", fontsize=8)

    e.set_xticks(x, siniflar)
    e.set_ylabel("F1"); e.set_ylim(0, 1.08)
    e.set_title("Sinif bazinda F1 -- taban cizgisiyle karsilastirma (test)")
    e.grid(axis="y", alpha=.3); e.legend(fontsize=9)
    fig.tight_layout()
    yol = Path(cikti) / "sinif_bazinda_f1.png"
    fig.savefig(yol, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  sekil: {yol}")
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
    eksik = [n for n in (1, 2, 3, 4, 5) if n not in kosular]
    print(f"  bulunan kosular: {sorted(kosular)}"
          + (f"   eksik: {eksik}" if eksik else "   (hepsi tamam)"))

    ana_tablo(kosular)
    for no in sorted(kosular):
        kosu_detay(no, kosular[no])
    karsilastirmalar(kosular)
    egriler(kosular, cikti)
    karisiklik_sekli(kosular, cikti)
    sinif_sekli(kosular, cikti)
    markdown(kosular, cikti)
    print(f"\n{CIFT}\nRAPOR TAMAM.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Uc kosuyu karsilastir")
    ap.add_argument("--cikti",
                    default="/tf/start_training/RELATIONNET/FENCE_DATA_NEW/egitim_ciktilari")
    sys.exit(rapor(ap.parse_args().cikti))
