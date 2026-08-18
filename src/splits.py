"""
FAZ 1 -- Gruplu Capraz Dogrulama Tasarimi (PLAN Bolum 5).

outputs/metadata.csv'yi okur, kaynak-gruplu k=4 capraz dogrulama bolmelerini
uretir ve outputs/folds/fold_{i}.json olarak yazar. Her katman icin PLAN
Bolum 5.4'teki sizinti testlerini KALICI olarak calistirir.

TEMEL KURAL (PLAN Bolum 2)
--------------------------
~959 spektrogram yalnizca 19 gercek kayittan turemistir. Rastgele bolme,
ayni kaydin varyantlarini hem egitime hem teste koyar ve modelin ezberini
"genelleme" diye raporlar. Bu yuzden bolme KAYNAK GRUBUNA gore yapilir.

MixUp ornekleri iki kaynaktan birden bilgi tasir. PLAN Bolum 2.2 kurali:

    tek-kaynakli, grup in test    -> TEST
    tek-kaynakli, grup not in test-> EGITIM
    mixup, HER IKI ebeveyn de egitimde -> EGITIM
    mixup, HERHANGI bir ebeveyn testte -> TAMAMEN ATILIR

DOGRULAMA SETI -- planda belirtilmeyen, burada verilen karar
------------------------------------------------------------
PLAN 7.4 egitim setinden gruplu %15 dogrulama ayirmayi soyluyor ama MixUp'in
ne olacagini yazmiyor. Ayni kurali dogrulama setine de uyguluyoruz: dogrulama
seti sadece tek-kaynakli orneklerden olusur ve bir ebeveyni dogrulama
gruplarinda olan MixUp ornegi egitimden de atilir.

Gerekce: erken durdurma ve ReduceLROnPlateau dogrulama kaybina bakiyor.
Sizintili bir dogrulama sinyali iyimser bir epoch'ta durmamiza yol acar ve
model secimini bozar -- test seti temiz olsa bile.

DOGRULAMA GRUBU SAYISI
----------------------
metal_bending'de toplam 4 grup var; k=4 ile her katmanin egitim setinde 3 grup
kalir. Bunun %15'i 0.45 grup eder -- yuvarlanirsa 0. O yuzden kural:

    n_val_groups = max(1, round(0.15 * egitimdeki_grup_sayisi))     [sinif basina]

Yani her siniftan en az 1 grup dogrulamaya gider. Bu, metal_bending icin
%15 degil ~%33 demektir (3 gruptan 1'i) ve egitimde sadece 2 grup birakir.
Sikisik ama kacinilmaz: her sinifin dogrulamada temsil edilmesi sart, aksi
halde dogrulama kaybi o sinif icin anlamsiz olur. Gercek oran her katman
icin JSON'a loglanir.

HANGI GRUPLAR DOGRULAMAYA GIDER -- MixUp korumali secim
--------------------------------------------------------
Dogrulama gruplarini rastgele secmek yerine, HAYATTA KALAN MixUp sayisini
maksimize eden secimi kullaniyoruz.

Neden fark ediyor: bazi kayitlar cok sayida MixUp ornegine ebeveynlik ediyor,
bazilari az (pipeline ciftleri rastgele sectigi icin). Cok ebeveynlik eden bir
grubu dogrulamaya koymak, onun tum MixUp cocuklarini da yok eder. Az ebeveynlik
edeni secmek ayni metodolojik garantiyi daha ucuza saglar.

Aday sayisi kucuk oldugu icin (sinif basina 1 grup -> katman basina 72-108
kombinasyon) hepsi denenir, en iyisi secilir. Olculen kazanc: MixUp hayatta
kalma orani %27.2 -> %33.4 (katmanlar boyunca +71 ornek).

Bu secim TEST setine dokunmaz ve kriteri tamamen yapisaldir (etiketlerle veya
model performansiyla ilgisi yok), dolayisiyla metodolojik bir taviz degildir.
Secim deterministiktir -- esitlik durumunda grup adlarina gore sirali ilk secim
alinir, boylece rastgelelik tohumundan bagimsiz olarak tekrar uretilebilir.

Kullanim:
    python src/splits.py
"""
import itertools
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as cfg


class LeakageError(AssertionError):
    """Sizinti testi basarisiz -- sonraki hicbir sonuc guvenilir degil."""


def check(condition, message):
    """
    Kalici dogrulama. Bilerek 'assert' anahtar kelimesi KULLANILMIYOR:
    python -O ile calistirildiginda assert'ler tamamen kaldirilir ve bu
    testlerin sessizce devre disi kalmasi kabul edilemez.
    """
    if not condition:
        raise LeakageError(message)


def row_group_set(row):
    """
    Bir ornegin BAGLI OLDUGU kanonik grup kumesi.

    Tek-kaynaklida tek eleman. MixUp'ta iki eleman -- ama starvolt
    birlestirmesinden sonra iki ebeveyn ayni gruba dusebilir
    (orn. rattle-1-rear x rattle-1-front), o zaman yine tek eleman.
    """
    groups = {row.group_1}
    g2 = row.group_2
    if isinstance(g2, str) and g2:
        groups.add(g2)
    return groups


def group_sets(df):
    """DataFrame'in her satiri icin grup kumesi (indeksle hizali liste)."""
    return [row_group_set(r) for r in df.itertuples(index=False)]


def build_group_label_map(df):
    """
    Her kanonik grup -> sinif etiketi. Bir grubun tek bir sinifa ait olmasi
    gerekir; degilse veri seti tutarsizdir.
    """
    mapping = defaultdict(set)
    for r in df.itertuples(index=False):
        for g in row_group_set(r):
            mapping[g].add(r.label)
    for g, labels in mapping.items():
        check(len(labels) == 1,
              f"'{g}' grubu birden fazla sinifa ait: {sorted(labels)}")
    return {g: next(iter(labels)) for g, labels in mapping.items()}


# Kombinasyon sayisi bunu asarsa hepsini denemek yerine tohumlu bir altkume
# ornekleriz. Bu veri setinde (19 grup) sinir asla asilmaz; ileride veri seti
# buyurse sessizce yavaslamak yerine graceful sekilde bozulmasi icin var.
MAX_VAL_COMBOS = 50_000


def count_surviving_mixup(gsets, is_mix, train_groups):
    """Iki ebeveyni de egitim gruplarinda olan -- yani kullanilabilir -- MixUp sayisi."""
    return sum(1 for j, m in enumerate(is_mix) if m and gsets[j] <= train_groups)


def choose_val_groups(non_test_groups, group_label, gsets, is_mix):
    """
    Dogrulama gruplarini sec: sinif basina en az 1 grup, ve HAYATTA KALAN
    MixUp sayisini maksimize eden kombinasyon (bkz. modul ust yorumu).

    Donen: (val_groups, secim istatistikleri)
    """
    per_class = {}
    for label in cfg.CLASSES:
        cands = sorted(g for g in non_test_groups if group_label[g] == label)
        check(len(cands) >= 2,
              f"'{label}' sinifinda egitimde {len(cands)} grup var; dogrulama "
              f"ayrildiktan sonra egitimde grup kalmaz.")
        n_val = max(1, round(cfg.VAL_FRACTION * len(cands)))
        n_val = min(n_val, len(cands) - 1)          # egitimde en az 1 grup kalsin
        per_class[label] = list(itertools.combinations(cands, n_val))

    combos = list(itertools.product(*(per_class[c] for c in cfg.CLASSES)))
    n_total = len(combos)
    if n_total > MAX_VAL_COMBOS:
        rng = np.random.default_rng(cfg.SEED)
        pick = rng.choice(n_total, size=MAX_VAL_COMBOS, replace=False)
        combos = [combos[i] for i in sorted(pick.tolist())]

    best = None
    worst_score = None
    for choice in combos:
        val_groups = {g for sub in choice for g in sub}
        score = count_surviving_mixup(gsets, is_mix, non_test_groups - val_groups)
        # Esitlikte grup adlarina gore sirali ilk secim -> deterministik
        key = (-score, tuple(sorted(val_groups)))
        if best is None or key < best[0]:
            best = (key, val_groups, score)
        if worst_score is None or score < worst_score:
            worst_score = score

    val_groups, score = best[1], best[2]
    stats = {
        "n_combinations": n_total,
        "n_evaluated": len(combos),
        "mixup_retained": score,
        "mixup_retained_worst_choice": worst_score,
    }
    return val_groups, stats


def class_counts(df):
    """Sinif basina ornek sayisi (tum siniflar, sifir olanlar dahil)."""
    vc = df.label.value_counts().to_dict()
    return {c: int(vc.get(c, 0)) for c in cfg.CLASSES}


def make_fold(df, gsets, test_groups):
    """
    Bir katmanin egitim / dogrulama / test bolmesini PLAN 2.2 kuralina gore kurar.

    Donen: (fold_dict, indeks kumeleri)
    """
    is_mix = df.is_mixup.to_numpy()
    all_groups = set().union(*gsets)
    non_test_groups = all_groups - test_groups

    group_label = build_group_label_map(df)
    val_groups, val_stats = choose_val_groups(non_test_groups, group_label,
                                              gsets, is_mix)
    train_groups = non_test_groups - val_groups

    test_idx, val_idx, train_idx, dropped_idx = [], [], [], []
    for i, (gs, mix) in enumerate(zip(gsets, is_mix)):
        if mix:
            # MixUp: SADECE tum ebeveynleri egitim gruplarindaysa egitime girer.
            # Bir ebeveyni bile test ya da dogrulama grubundaysa tamamen atilir.
            if gs <= train_groups:
                train_idx.append(i)
            else:
                dropped_idx.append(i)
        else:
            (g,) = gs
            if g in test_groups:
                test_idx.append(i)
            elif g in val_groups:
                val_idx.append(i)
            else:
                train_idx.append(i)

    return {
        "test_groups": sorted(test_groups),
        "val_groups": sorted(val_groups),
        "train_groups": sorted(train_groups),
        "val_selection": {"rule": "max-mixup-retention", **val_stats},
    }, (np.array(train_idx), np.array(val_idx), np.array(test_idx),
        np.array(dropped_idx))


def assert_no_leakage(fold_i, df, gsets, groups_info, idx):
    """
    PLAN Bolum 5.4 -- zorunlu sizinti testleri, arti Faz 0/1'de ortaya cikan
    kenar durumlar icin ek testler. Basarisizsa LeakageError firlatir.
    """
    train_idx, val_idx, test_idx, dropped_idx = idx
    test_groups = set(groups_info["test_groups"])
    val_groups = set(groups_info["val_groups"])
    train_groups = set(groups_info["train_groups"])
    tag = f"[fold {fold_i}]"

    def groups_of(indices):
        out = set()
        for i in indices:
            out |= gsets[i]
        return out

    # --- PLAN 5.4 #1: egitim ve test gruplari kesismiyor ---
    check(not (train_groups & test_groups),
          f"{tag} egitim ve test gruplari kesisiyor: {train_groups & test_groups}")
    check(not (groups_of(train_idx) & test_groups),
          f"{tag} egitim ORNEKLERI test grubuna dokunuyor: "
          f"{groups_of(train_idx) & test_groups}")

    # --- PLAN 5.4 #2: test setinde hic MixUp yok ---
    check(not df.is_mixup.to_numpy()[test_idx].any(),
          f"{tag} test setinde MixUp ornegi var")

    # --- PLAN 5.4 #3: egitimdeki MixUp'larin hicbir ebeveyni test grubunda degil ---
    mix_train = [i for i in train_idx if df.is_mixup.iat[i]]
    for i in mix_train:
        check(not (gsets[i] & test_groups),
              f"{tag} egitimdeki MixUp ebeveyni test grubunda: "
              f"{df.filename.iat[i]} -> {gsets[i] & test_groups}")

    # --- PLAN 5.4 #4: test setinde 3 sinifin da en az 1 ornegi var ---
    tc = class_counts(df.iloc[test_idx])
    for label, n in tc.items():
        check(n > 0, f"{tag} test setinde '{label}' sinifindan hic ornek yok")

    # --- EK #5: dogrulama setine de ayni kural (bu modulun karari) ---
    check(not (train_groups & val_groups),
          f"{tag} egitim ve dogrulama gruplari kesisiyor")
    check(not (val_groups & test_groups),
          f"{tag} dogrulama ve test gruplari kesisiyor")
    check(not df.is_mixup.to_numpy()[val_idx].any(),
          f"{tag} dogrulama setinde MixUp ornegi var")
    for i in mix_train:
        check(not (gsets[i] & val_groups),
              f"{tag} egitimdeki MixUp ebeveyni dogrulama grubunda: "
              f"{df.filename.iat[i]}")

    # --- EK #6: dogrulama setinde de 3 sinif temsil edilmeli (erken durdurma icin) ---
    vc = class_counts(df.iloc[val_idx])
    for label, n in vc.items():
        check(n > 0, f"{tag} dogrulama setinde '{label}' sinifindan hic ornek yok")

    # --- EK #7: bolmeler ayrik ve butun veriyi kapsiyor ---
    sets = [set(train_idx.tolist()), set(val_idx.tolist()),
            set(test_idx.tolist()), set(dropped_idx.tolist())]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            check(not (sets[a] & sets[b]),
                  f"{tag} bolmeler ortusuyor ({a},{b}): {sorted(sets[a] & sets[b])[:5]}")
    total = sum(len(s) for s in sets)
    check(total == len(df),
          f"{tag} bolmelerin toplami {total}, veri seti {len(df)}")

    # --- EK #8: atilanlar gercekten atilmali (hicbiri temiz olmamali) ---
    forbidden = test_groups | val_groups
    for i in dropped_idx:
        check(bool(gsets[i] & forbidden),
              f"{tag} gereksiz atilan ornek: {df.filename.iat[i]} "
              f"(hicbir ebeveyni test/dogrulama grubunda degil)")


def summarize_fold(df, groups_info, idx):
    """PLAN 5.3 -- fold JSON'una yazilacak ozet."""
    train_idx, val_idx, test_idx, dropped_idx = idx
    tr, va, te = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    n_train_single = int((~tr.is_mixup).sum())
    n_train_mix = int(tr.is_mixup.sum())
    denom = len(tr) + len(va)

    return {
        **groups_info,
        # test_sources: birlestirilmis gruplarin altindaki HAM dosya adlari
        "test_sources": sorted(
            set(te.source_1) | set(te.source_2.dropna())),
        "n_train": len(tr),
        "n_train_single": n_train_single,
        "n_train_mixup": n_train_mix,
        "n_val": len(va),
        "n_test": len(te),
        "n_dropped_mixup": int(len(dropped_idx)),
        "val_fraction_actual": round(len(va) / denom, 4) if denom else 0.0,
        "class_counts": {
            "train": class_counts(tr),
            "val": class_counts(va),
            "test": class_counts(te),
        },
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "test_idx": [int(i) for i in test_idx],
        "dropped_idx": [int(i) for i in dropped_idx],
    }


def main():
    cfg.ensure_dirs()
    df = pd.read_csv(cfg.METADATA_CSV)
    df["group_2"] = df["group_2"].where(df["group_2"].notna(), None)
    gsets = group_sets(df)
    group_label = build_group_label_map(df)

    print("=" * 74)
    print(f"FAZ 1 -- GRUPLU CAPRAZ DOGRULAMA  (k={cfg.N_SPLITS}, seed={cfg.SEED})")
    print("=" * 74)
    print(f"  ornek       : {len(df)}  ({int((~df.is_mixup).sum())} tek-kaynakli "
          f"+ {int(df.is_mixup.sum())} mixup)")
    print(f"  etkin grup  : {len(group_label)}")
    for label in cfg.CLASSES:
        n = sum(1 for g, l in group_label.items() if l == label)
        print(f"     {label:<22} {n} grup")
    check(len(group_label) == cfg.N_EFFECTIVE_GROUPS,
          f"{len(group_label)} grup bulundu, {cfg.N_EFFECTIVE_GROUPS} bekleniyordu")

    # Bolme SADECE tek-kaynakli ornekler uzerinde belirlenir (PLAN 5.2 adim 1).
    single = df[~df.is_mixup]
    sgkf = StratifiedGroupKFold(n_splits=cfg.N_SPLITS, shuffle=True,
                                random_state=cfg.SEED)
    splits = list(sgkf.split(single.index.to_numpy(),
                             single.label_idx.to_numpy(),
                             groups=single.group_1.to_numpy()))

    folds, all_test_groups = [], []
    for i, (_, te_pos) in enumerate(splits):
        test_groups = set(single.group_1.iloc[te_pos].unique())
        groups_info, idx = make_fold(df, gsets, test_groups)
        assert_no_leakage(i, df, gsets, groups_info, idx)
        summary = summarize_fold(df, groups_info, idx)
        folds.append(summary)
        all_test_groups.append(test_groups)

        out = cfg.FOLDS_DIR / f"fold_{i}.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                       encoding="utf-8")

        print(f"\n{'-' * 74}")
        print(f"KATMAN {i}   -> {out.name}")
        print(f"{'-' * 74}")
        print(f"  test gruplari ({len(test_groups)}):")
        for g in sorted(test_groups):
            print(f"     [{group_label[g]:<20}] {g}")
        vs = groups_info["val_selection"]
        print(f"  dogrulama gruplari ({len(groups_info['val_groups'])}):")
        for g in sorted(groups_info["val_groups"]):
            print(f"     [{group_label[g]:<20}] {g}")
        print(f"     secim: {vs['n_combinations']} kombinasyon denendi -> "
              f"MixUp korunan {vs['mixup_retained']} "
              f"(en kotu secim {vs['mixup_retained_worst_choice']} verirdi)")
        print(f"\n  egitim : {summary['n_train']:>4}  "
              f"({summary['n_train_single']} tek-kaynakli + "
              f"{summary['n_train_mixup']} mixup)")
        print(f"  dogrul.: {summary['n_val']:>4}  "
              f"(gercek oran {summary['val_fraction_actual']:.1%})")
        print(f"  test   : {summary['n_test']:>4}")
        print(f"  ATILAN mixup: {summary['n_dropped_mixup']:>4}  "
              f"(ebeveyni test/dogrulama grubunda)")
        print(f"  toplam : {summary['n_train'] + summary['n_val'] + summary['n_test'] + summary['n_dropped_mixup']}"
              f"  (= {len(df)})")
        print(f"\n  sinif dagilimi:")
        print(f"     {'':<22}{'egitim':>8}{'dogrul.':>9}{'test':>7}")
        for label in cfg.CLASSES:
            print(f"     {label:<22}{summary['class_counts']['train'][label]:>8}"
                  f"{summary['class_counts']['val'][label]:>9}"
                  f"{summary['class_counts']['test'][label]:>7}")

    # --- KATMANLAR ARASI TUTARLILIK ---
    print(f"\n{'=' * 74}")
    print("KATMANLAR ARASI DOGRULAMA")
    print("=" * 74)

    covered = set().union(*all_test_groups)
    check(covered == set(group_label),
          f"bazi gruplar hic test edilmedi: {set(group_label) - covered}")
    print(f"  [x] 19 grubun hepsi en az bir katmanda test edildi")

    for a in range(len(all_test_groups)):
        for b in range(a + 1, len(all_test_groups)):
            overlap = all_test_groups[a] & all_test_groups[b]
            check(not overlap,
                  f"katman {a} ve {b} ayni grubu test ediyor: {overlap}")
    print(f"  [x] Hicbir grup birden fazla katmanda test edilmedi")

    total_test = sum(f["n_test"] for f in folds)
    n_single = int((~df.is_mixup).sum())
    check(total_test == n_single,
          f"test ornekleri toplami {total_test}, tek-kaynakli sayisi {n_single}")
    print(f"  [x] Her tek-kaynakli ornek tam olarak bir kez test edildi "
          f"({total_test} = {n_single})")

    # --- OZET TABLO ---
    print(f"\n{'=' * 74}")
    print("OZET")
    print("=" * 74)
    print(f"  {'katman':<8}{'egitim':>8}{'dogrul.':>9}{'test':>7}{'atilan':>9}"
          f"{'val%':>8}")
    print("  " + "-" * 47)
    for i, f in enumerate(folds):
        print(f"  {i:<8}{f['n_train']:>8}{f['n_val']:>9}{f['n_test']:>7}"
              f"{f['n_dropped_mixup']:>9}{f['val_fraction_actual']:>8.1%}")
    arr = np.array([[f["n_train"], f["n_val"], f["n_test"], f["n_dropped_mixup"]]
                    for f in folds], dtype=float)
    print("  " + "-" * 47)
    print(f"  {'ort':<8}{arr[:, 0].mean():>8.1f}{arr[:, 1].mean():>9.1f}"
          f"{arr[:, 2].mean():>7.1f}{arr[:, 3].mean():>9.1f}")
    print(f"\n  Toplam atilan MixUp (katmanlar boyunca): {int(arr[:, 3].sum())}")
    print(f"  PLAN 2.2 notu: bu normaldir -- MixUp ornegi iki kaynaktan bilgi")
    print(f"  tasidigi icin ebeveyni test/dogrulama grubundaysa kullanilamaz.")

    print(f"\nKABUL KRITERI (PLAN 5.4)")
    print("-" * 74)
    for name in [
        "set(train_groups) & set(test_groups) == bos",
        "Test setinde hic MixUp ornegi yok",
        "Egitimdeki MixUp'larin hicbir ebeveyni test gruplarinda degil",
        "Her katmanin test setinde 3 sinifin da en az 1 ornegi var",
        "(ek) Ayni kurallar dogrulama seti icin de saglandi",
        "(ek) Bolmeler ayrik ve tum veriyi kapsiyor",
        "(ek) Her grup tam olarak bir katmanda test edildi",
    ]:
        print(f"  [x] {name}")
    print(f"\n  Bu testler splits.py icinde KALICI (check() -> LeakageError).")
    print(f"  python -O ile bile devre disi kalmazlar.")
    print(f"\nFAZ 1 TAMAM. {len(folds)} katman yazildi: {cfg.FOLDS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
