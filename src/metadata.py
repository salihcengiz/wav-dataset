"""
FAZ 0 -- Veri Hazirligi ve Denetim (PLAN Bolum 4).

synthetic_dataset/ altindaki tum *_spectrogram.png dosyalarini tarar, dosya
adindan meta veriyi ayristirir ve outputs/metadata.csv uretir. Ardindan PLAN
Bolum 4.2'deki 7 zorunlu denetim ciktisini konsola yazar.

AYRISTIRMA STRATEJISI -- neden regex + dogrulama, neden naif split degil:

  Gercek kaynak dosya adlari "temiz" degil. Ornekler:
      138250__hupguy__chain-fence-hit_sequence-01r
      345070__metrostock99__chain-lock-on-fence-sound (1)
      7148__rhumphries__rbh_chain-link-fence-01
  Yani: cift alt cizgi, tire, bosluk ve parantez iceriyorlar. Ilk '_' isaretinden
  bolmek kesinlikle bozulur.

  Daha ince bir tuzak: fence_cutting sinifinda iki kaynak adindan biri digerinin
  TAM ONEKI ('...fence-sound' ve '...fence-sound (1)'). Bu yuzden onek-eslestirmeye
  dayali bir ayristirma da yanlis kaynaga atar.

  Cozum: sondan capalanmis regex ile ayristir (greedy '.+' en SON '_snr..' ayracini
  yakalar), sonra ayristirilan her kaynak adini raw_sounds/ altindaki GERCEK kaynak
  dosya adlari kumesine karsi dogrula. Boylece ayristirma bir tahmin degil,
  yer gercegiyle (ground truth) dogrulanmis bir eslesme olur.

Kullanim:
    python src/metadata.py
"""
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import config as cfg


# Sondan capalanmis desenler. '.+' greedy oldugu icin en SON '_snr{N}dB_...'
# ayraci yakalanir -- kaynak adinin kendisinde '_snr' gecse bile dogru calisir.
SINGLE_RE = re.compile(
    r"^(?P<src>.+)_snr(?P<snr>\d+)dB_(?P<noise>pink|white)_v(?P<idx>\d+)$"
)
MIXUP_RE = re.compile(
    r"^mixup_(?P<pair>.+)_snr(?P<snr>\d+)dB_(?P<noise>pink|white)_(?P<idx>\d+)$"
)

SPEC_SUFFIX = "_spectrogram"


class ParseError(RuntimeError):
    """Dosya adi ayristirilamadi veya yer gercegiyle eslesmedi."""


def load_known_sources():
    """
    raw_sounds/{sinif}/ altindaki gercek kayit adlarini oku.

    Bu, grup kimliginin YER GERCEGI'dir -- ayristirilan her source_1/source_2
    bu kumeye karsi dogrulanir.

    Donen: {sinif: set(kaynak_adi)}
    """
    if not cfg.RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"raw_sounds/ bulunamadi: {cfg.RAW_DIR}\n"
            "Ayristirmayi dogrulamak icin gercek kaynak dosya adlari gerekli."
        )
    known = {}
    for label in cfg.CLASSES:
        d = cfg.RAW_DIR / label
        if not d.is_dir():
            raise FileNotFoundError(f"Sinif klasoru yok: {d}")
        names = {p.stem for p in d.iterdir()
                 if p.is_file() and p.suffix.lower() in (".wav", ".mp3")}
        if not names:
            raise FileNotFoundError(f"'{d}' altinda ham ses dosyasi yok.")
        known[label] = names
    return known


def split_mixup_pair(pair, known):
    """
    'mixup_' on eki ve '_snr..' son eki soyulmus orta kismi iki ebeveyne ayirir.

    Ayrac '_x_' ama kaynak adlarinin kendisinde de '_x_' gecebilir (PLAN 1.5
    uyarisi). Bu yuzden butun olasi bolme noktalarini dene ve YALNIZCA iki
    yarisi da bilinen kaynak kumesinde olan bolmeyi kabul et. Tam olarak bir
    gecerli bolme yoksa hata firlat -- sessizce yanlis gruplama yapmaktansa
    durmak dogrudur.
    """
    candidates = []
    start = 0
    while True:
        i = pair.find("_x_", start)
        if i == -1:
            break
        left, right = pair[:i], pair[i + 3:]
        if left in known and right in known:
            candidates.append((left, right))
        start = i + 1

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ParseError(
            f"MixUp ebeveynleri cozulemedi: '{pair}'. "
            "'_x_' ayracinin hicbir bolmesi bilinen kaynak ciftine denk gelmiyor."
        )
    raise ParseError(
        f"MixUp ebeveynleri BELIRSIZ: '{pair}' -> {candidates}. "
        "Birden fazla gecerli bolme var, gruplama guvenilir degil."
    )


def parse_filename(png_path, label, known):
    """Tek bir *_spectrogram.png yolunu meta veri sozlugune cevirir."""
    tag = png_path.stem
    if not tag.endswith(SPEC_SUFFIX):
        raise ParseError(f"'{SPEC_SUFFIX}' son eki yok: {png_path.name}")
    tag = tag[: -len(SPEC_SUFFIX)]

    if tag.startswith("mixup_"):
        m = MIXUP_RE.match(tag)
        if m is None:
            raise ParseError(f"MixUp deseni eslesmedi: {png_path.name}")
        src1, src2 = split_mixup_pair(m.group("pair"), known)
        is_mixup = True
    else:
        m = SINGLE_RE.match(tag)
        if m is None:
            raise ParseError(f"Tek-kaynak deseni eslesmedi: {png_path.name}")
        src1 = m.group("src")
        if src1 not in known:
            raise ParseError(
                f"Ayristirilan kaynak '{src1}' raw_sounds/{label}/ icinde yok "
                f"(dosya: {png_path.name})"
            )
        src2 = None
        is_mixup = False

    # Kanonik grup: ayni fiziksel olayi paylasan kaynaklar tek gruba duser
    # (bkz. config.SOURCE_GROUP_MERGES -- starvolt on/arka ciftleri).
    grp1 = cfg.canonical_group(src1)
    grp2 = cfg.canonical_group(src2) if src2 is not None else None

    # Ornegin bagli oldugu BENZERSIZ grup kumesi. MixUp'ta iki ebeveyn ayni
    # kanonik gruba dusebilir (orn. rattle-1-rear x rattle-1-front) -- o zaman
    # ornek tek bir gruba baglidir.
    groups = sorted({g for g in (grp1, grp2) if g is not None})

    return {
        "filepath": str(png_path),
        "filename": png_path.name,
        "label": label,
        "label_idx": cfg.LABEL_TO_IDX[label],
        "is_mixup": is_mixup,
        "source_1": src1,
        "source_2": src2,
        # Gruplu bolmede FIILEN kullanilacak alanlar (PLAN 4.1 + Faz 0 duzeltmesi)
        "group_1": grp1,
        "group_2": grp2,
        "group_id": "|".join(groups),
        "n_groups": len(groups),
        "snr_db": int(m.group("snr")),
        "noise_kind": m.group("noise"),
        "variant_idx": int(m.group("idx")),
    }


def build_metadata():
    """Tum PNG'leri tara ve DataFrame uret. Ayristirma hatalarini toplu raporlar."""
    known = load_known_sources()
    rows, errors = [], []

    for label in cfg.CLASSES:
        d = cfg.DATA_DIR / label
        if not d.is_dir():
            raise FileNotFoundError(f"Sinif klasoru yok: {d}")
        pngs = sorted(d.glob(f"*{SPEC_SUFFIX}.png"))
        if not pngs:
            raise FileNotFoundError(f"'{d}' altinda spektrogram PNG'si yok.")
        for p in pngs:
            try:
                rows.append(parse_filename(p, label, known[label]))
            except ParseError as e:
                errors.append(str(e))

    if errors:
        print(f"\n[HATA] {len(errors)} dosya ayristirilamadi. Ilk 10:")
        for e in errors[:10]:
            print("   -", e)
        raise ParseError(f"{len(errors)} dosya ayristirilamadi -- DUR ve ayristirmayi duzelt.")

    df = pd.DataFrame(rows)
    return df, known


def inspect_pngs(df):
    """
    Denetim 6 ve 7: tum PNG'lerin boyutu ayni mi, bozuk dosya var mi.

    Image.load() dosyayi tam cozer -- bozuk/yarim yazilmis PNG burada patlar.
    """
    sizes = Counter()
    broken = []
    for fp in df["filepath"]:
        try:
            with Image.open(fp) as im:
                sizes[im.size] += 1
                im.load()
        except Exception as e:  # noqa: BLE001 -- her turlu bozulmayi yakalamak istiyoruz
            broken.append((fp, repr(e)))
    return sizes, broken


def audit(df, known):
    """PLAN Bolum 4.2 -- 7 zorunlu denetim ciktisi. Sert hatalari toplayip doner."""
    failures = []
    line = "-" * 72

    print("\n" + "=" * 72)
    print("FAZ 0 -- VERI DENETIMI")
    print("=" * 72)

    # --- 1) Toplam ornek sayisi ve sinif dagilimi ---
    print(f"\n[1] Toplam ornek: {len(df)}")
    print(line)
    for label in cfg.CLASSES:
        sub = df[df.label == label]
        n_mix = int(sub.is_mixup.sum())
        print(f"  {label:<22} {len(sub):>5}  (tek-kaynak {len(sub) - n_mix:>4} | mixup {n_mix:>4})")

    # --- 2) Benzersiz kaynak / etkin grup sayisi -- sinif basina ---
    print(f"\n[2] Benzersiz kaynak dosya ve ETKIN GRUP sayisi (KRITIK)")
    print(line)
    for label in cfg.CLASSES:
        sub = df[df.label == label]
        # Hem tek-kaynakli hem MixUp ebeveynlerini birlestir
        srcs = set(sub.source_1) | set(sub.source_2.dropna())
        grps = set(sub.group_1) | set(sub.group_2.dropna())
        exp_s = cfg.EXPECTED_SOURCE_COUNTS[label]
        exp_g = cfg.EXPECTED_GROUP_COUNTS[label]
        ok_s, ok_g = len(srcs) == exp_s, len(grps) == exp_g
        merged = exp_s - exp_g
        note = f"  ({merged} birlestirme)" if merged else ""
        print(f"  {label:<22} kaynak={len(srcs):>2}/{exp_s:<2} "
              f"raw_sounds={len(known[label]):>2}   "
              f"ETKIN GRUP={len(grps):>2}/{exp_g:<2}{note}   "
              f"{'OK' if ok_s and ok_g else '*** UYUSMUYOR ***'}")
        if not ok_s:
            failures.append(f"[2] {label}: {len(srcs)} kaynak bulundu, {exp_s} bekleniyordu")
        if not ok_g:
            failures.append(f"[2] {label}: {len(grps)} etkin grup bulundu, {exp_g} bekleniyordu")

        # Yer gercegiyle iki yonlu karsilastirma
        eksik = known[label] - srcs      # raw_sounds'ta var, veri setinde yok
        fazla = srcs - known[label]      # veri setinde var, raw_sounds'ta yok
        if eksik:
            print(f"      raw_sounds'ta var ama hic varyanti yok: {sorted(eksik)}")
            failures.append(f"[2] {label}: kaynaklarin varyanti yok -> {sorted(eksik)}")
        if fazla:
            print(f"      veri setinde var ama raw_sounds'ta yok: {sorted(fazla)}")
            failures.append(f"[2] {label}: bilinmeyen kaynak -> {sorted(fazla)}")

    total_grps = len(set(df.group_1) | set(df.group_2.dropna()))
    exp_total = cfg.N_EFFECTIVE_GROUPS
    print(f"\n  TOPLAM ETKIN BAGIMSIZ ORNEK: {total_grps}   "
          f"{'OK' if total_grps == exp_total else f'*** {exp_total} BEKLENIYORDU ***'}")
    print(f"  (Plan 22 varsayiyordu; Faz 0 denetimi 1 birebir kopya sildi ve")
    print(f"   2 eszamanli mikrofon ciftini birlestirdi -> {exp_total})")
    if total_grps != exp_total:
        failures.append(f"[2] Toplam etkin grup {total_grps}, {exp_total} bekleniyordu")

    # Birlestirmelerin gercekten uygulandigini dogrula
    print(f"\n  Uygulanan grup birlestirmeleri:")
    for grp in sorted(set(cfg.SOURCE_GROUP_MERGES.values())):
        members = sorted(s for s, g in cfg.SOURCE_GROUP_MERGES.items() if g == grp)
        n = int(((df.group_1 == grp) | (df.group_2 == grp)).sum())
        print(f"    {grp}  <- {len(members)} kaynak, {n} ornek")
        for mem in members:
            print(f"       {mem}")
        seen = set(df.source_1) | set(df.source_2.dropna())
        missing = [m for m in members if m not in seen]
        if missing:
            failures.append(f"[2] birlestirme uyesi veri setinde yok: {missing}")

    # --- 3) Kaynak basina varyant sayisi ---
    print(f"\n[3] Kaynak basina uretilen varyant sayisi")
    print(line)
    single = df[~df.is_mixup]
    for label in cfg.CLASSES:
        counts = single[single.label == label].source_1.value_counts()
        print(f"  {label:<22} tek-kaynakli varyant: min={counts.min():>3} "
              f"ort={counts.mean():>6.1f} maks={counts.max():>3}  ({len(counts)} kaynak)")
    # MixUp'ta ebeveyn basina katilim
    mix = df[df.is_mixup]
    if len(mix):
        parent_counts = Counter()
        for s1, s2 in zip(mix.source_1, mix.source_2):
            parent_counts[s1] += 1
            parent_counts[s2] += 1
        vals = np.array(list(parent_counts.values()))
        print(f"  {'(mixup ebeveyn katilimi)':<22} min={vals.min():>3} "
              f"ort={vals.mean():>6.1f} maks={vals.max():>3}")

    # --- 4) MixUp orani ---
    n_mix = int(df.is_mixup.sum())
    print(f"\n[4] MixUp ornekleri: {n_mix} / {len(df)}  ({100 * n_mix / len(df):.1f}%)")
    print(line)
    print(f"  UYARI: Bunlarin tamami test setinden dislanacak; ebeveyni test")
    print(f"  kaynagi olanlar ise TAMAMEN atilacak (PLAN Bolum 2.2).")

    # --- 5) SNR dagilimi ---
    print(f"\n[5] SNR dagilimi (dB)")
    print(line)
    snr = df.snr_db
    print(f"  min={snr.min()}  maks={snr.max()}  ortalama={snr.mean():.2f}  medyan={snr.median():.1f}")
    hist = snr.value_counts().sort_index()
    peak = int(hist.max())
    for v, c in hist.items():
        bar = "#" * max(1, round(40 * c / peak))
        print(f"   {v:>2} dB | {c:>4} {bar}")
    print("\n  SNR kovalari (PLAN 8.3 -- degerlendirmede kullanilacak):")
    for name, lo, hi in cfg.SNR_BUCKETS:
        n = int(((snr >= lo) & (snr < hi)).sum())
        print(f"   {name:<7} [{lo:>2}, {hi:>2}) dB : {n:>4} ornek")

    # --- 6) PNG boyutlari + 7) bozuk dosyalar ---
    print(f"\n[6] PNG boyut tutarliligi  /  [7] bozuk dosya taramasi")
    print(line)
    print(f"  {len(df)} PNG aciliyor ve tam olarak coozuluyor...")
    sizes, broken = inspect_pngs(df)
    for size, count in sizes.most_common():
        mark = "OK" if size == cfg.EXPECTED_PNG_SIZE else "*** BEKLENMEYEN BOYUT ***"
        print(f"  {size[0]} x {size[1]} : {count:>4} dosya   {mark}")
    if len(sizes) != 1:
        failures.append(f"[6] PNG boyutlari tekduze degil: {dict(sizes)}")
    elif next(iter(sizes)) != cfg.EXPECTED_PNG_SIZE:
        failures.append(f"[6] PNG boyutu {next(iter(sizes))}, {cfg.EXPECTED_PNG_SIZE} bekleniyordu")

    if broken:
        print(f"  *** {len(broken)} BOZUK DOSYA ***")
        for fp, err in broken[:10]:
            print(f"     {fp}  ->  {err}")
        failures.append(f"[7] {len(broken)} bozuk/okunamayan PNG")
    else:
        print(f"  Bozuk dosya yok.")

    return failures


def spot_check(df, n=5, seed=cfg.SEED):
    """
    PLAN 4.3 kabul kriteri: rastgele satirlari dosya adi <-> ayristirilmis
    alanlar seklinde elle dogrulanabilir bicimde yazdir.

    Tohum sabit -- ayni satirlar her calistirmada gorunur, tekrar dogrulanabilir.
    """
    print("\n" + "=" * 72)
    print(f"ELLE DOGRULAMA -- rastgele {n} satir (seed={seed})")
    print("=" * 72)
    # MixUp'lardan da en az birini goster: yari yariya ornekle
    mix = df[df.is_mixup].sample(min(2, int(df.is_mixup.sum())), random_state=seed)
    sing = df[~df.is_mixup].sample(n - len(mix), random_state=seed)
    for _, r in pd.concat([sing, mix]).iterrows():
        print(f"\n  DOSYA : {r.filename}")
        print(f"    label={r.label}  idx={r.label_idx}  is_mixup={r.is_mixup}")
        print(f"    source_1   = {r.source_1}")
        print(f"    source_2   = {r.source_2}")
        print(f"    group_1    = {r.group_1}")
        print(f"    group_2    = {r.group_2}")
        print(f"    group_id   = {r.group_id}   (n_groups={r.n_groups})")
        print(f"    snr={r.snr_db} dB  noise={r.noise_kind}  variant={r.variant_idx}")


def main():
    cfg.ensure_dirs()
    print(f"Veri klasoru : {cfg.DATA_DIR}")
    print(f"Ham kayitlar : {cfg.RAW_DIR}")

    df, known = build_metadata()
    failures = audit(df, known)
    spot_check(df)

    df.to_csv(cfg.METADATA_CSV, index=False)
    print("\n" + "=" * 72)
    print(f"metadata.csv yazildi: {cfg.METADATA_CSV}  ({len(df)} satir)")

    print("\nKABUL KRITERI (PLAN 4.3)")
    print("-" * 72)
    checks = [
        ("metadata.csv uretildi", cfg.METADATA_CSV.exists()),
        (f"Kaynak/grup sayilari beklenenle eslesiyor "
         f"(kaynak {'/'.join(str(cfg.EXPECTED_SOURCE_COUNTS[c]) for c in cfg.CLASSES)}, "
         f"grup {'/'.join(str(cfg.EXPECTED_GROUP_COUNTS[c]) for c in cfg.CLASSES)})",
         not any(f.startswith("[2]") for f in failures)),
        ("Tum PNG'ler 400x400, bozuk dosya yok",
         not any(f.startswith(("[6]", "[7]")) for f in failures)),
        ("Rastgele satirlar elle dogrulama icin yazdirildi", True),
    ]
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")

    if failures:
        print("\n*** FAZ 0 BASARISIZ -- DUR VE DUZELT ***")
        for f in failures:
            print("   -", f)
        return 1

    print("\nFAZ 0 TAMAM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
