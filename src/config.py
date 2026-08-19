"""
Tum yollar ve hiperparametreler tek yerde (PLAN Bolum 3).

Not: Plan 'das-model-training/data/synthetic_dataset/' yerlesimi oneriyordu, ancak
veri seti bu repoda zaten kokte duruyor. Veriyi kopyalamak yerine mevcut konumu
okuyoruz; src/ ve outputs/ kokte.
"""
from pathlib import Path

# ---------------------------------------------------------------
# YOLLAR
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "synthetic_dataset"   # spektrogram PNG'leri (+ wav'lar)
RAW_DIR = ROOT / "raw_sounds"           # 22 gercek kaynak kayit -- grup kimliginin dogrulama referansi
OUT_DIR = ROOT / "outputs"

METADATA_CSV = OUT_DIR / "metadata.csv"
FOLDS_DIR = OUT_DIR / "folds"
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
RESULTS_DIR = OUT_DIR / "results"


def ensure_dirs():
    """Cikti klasorlerini olustur (varsa dokunma)."""
    for d in (OUT_DIR, FOLDS_DIR, CKPT_DIR, FIG_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# SINIFLAR
# ---------------------------------------------------------------
# Alfabetik sira -- deterministik ve torchvision.ImageFolder / sklearn ile tutarli.
CLASSES = ["chain_link_climbing", "fence_cutting", "metal_bending"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}
N_CLASSES = len(CLASSES)

# ---------------------------------------------------------------
# KAYNAK SAYILARI -- PLANDAN SAPMA (Faz 0 denetiminde bulundu)
# ---------------------------------------------------------------
# Plan 22 bagimsiz kayit (9/9/4) varsayiyordu. Faz 0 denetimi iki sorun buldu:
#
#   1) fence_cutting: '345070__...chain-lock-on-fence-sound (1).wav', ayni
#      dosyanin BAYT BAYT kopyasiydi (ayni MD5, r=1.000) -- tarayicinin kopya
#      indirme soneki. Silindi (turevleriyle birlikte, 83 dosya).
#      => 9 -> 8 kaynak
#
#   2) chain_link_climbing: starvolt 'rear'/'front' ciftleri, ayni fiziksel
#      olayin iki mikrofonla ESZAMANLI kaydiydi. Kanit: zarf korelasyonu
#      0.97/0.90, log-spektrogram korelasyonu 0.94/0.89, optimum gecikme
#      TAM 0.0 ms (kontrol grubu ortalamasi sirasiyla 0.08 / 0.08 / +-250ms).
#      Silinmedi -- ciftler tek gruba birlestirildi, boylece tum varyantlar
#      egitimde kalir ama asla egitim/test arasinda bolunmezler.
#      => 9 -> 7 etkin grup
#
# ETKIN BAGIMSIZ ORNEK: 22 -> 19
EXPECTED_SOURCE_COUNTS = {          # diskteki ham dosya sayisi
    "fence_cutting": 8,
    "chain_link_climbing": 9,
    "metal_bending": 4,
}

EXPECTED_GROUP_COUNTS = {           # birlestirmeden SONRA etkin grup sayisi
    "fence_cutting": 8,
    "chain_link_climbing": 7,
    "metal_bending": 4,
}

N_EFFECTIVE_GROUPS = sum(EXPECTED_GROUP_COUNTS.values())   # 19

# Ayni fiziksel olayi paylasan kaynaklar -> ortak grup kimligi.
# Bunlar ayri grup sayilirsa StratifiedGroupKFold birini egitime, digerini
# teste atabilir; model test olayini zaten gormus olur (olay duzeyinde sizinti).
SOURCE_GROUP_MERGES = {
    "189219__starvolt__fence-rattle-1-rear": "starvolt__fence-rattle-1",
    "189220__starvolt__fence-rattle-1-front": "starvolt__fence-rattle-1",
    "189223__starvolt__fence-rattle-2-rear": "starvolt__fence-rattle-2",
    "189224__starvolt__fence-rattle-2-front": "starvolt__fence-rattle-2",
}


def canonical_group(source_name):
    """Ham kaynak adini, gruplu bolmede kullanilacak kanonik grup kimligine cevirir."""
    return SOURCE_GROUP_MERGES.get(source_name, source_name)

# PLAN Bolum 1.3: tum spektrogramlar ayni boyutta olmali
EXPECTED_PNG_SIZE = (400, 400)

# ---------------------------------------------------------------
# CAPRAZ DOGRULAMA (PLAN Bolum 5.1)
# ---------------------------------------------------------------
# k=5 KULLANILAMAZ: metal_bending'de sadece 4 kaynak var, bazi katmanlarin
# test setinde hic metal_bending olmaz.
N_SPLITS = 4
SEED = 42

# Her katmanin egitim setinden ayrilacak, yine GRUPLU dogrulama orani (PLAN 7.4)
VAL_FRACTION = 0.15

# ---------------------------------------------------------------
# MODEL (PLAN Bolum 6)
# ---------------------------------------------------------------
INPUT_SIZE = 224            # 400x400 PNG -> 224x224
IN_CHANNELS = 3             # viridis RGB
CONV_CHANNELS = (16, 32, 64)
SK_M = 2                    # dal sayisi
SK_KERNELS = (3, 5)         # ablasyonda en iyi ikili
SK_R = 16                   # sikistirma orani
SK_L = 32                   # minimum kanal sayisi

# SK dallarindaki grup konvolusyonunun grup sayisi.
# PLAN "grup konvolusyonu" diyor ama G'yi vermiyor -- orijinal SKNet (Li ve ark.,
# CVPR 2019) G=32 kullanir; C=64 icin grup basina 2 kanal. Ayrica parametre
# butcesi icin zorunlu: gruplamasiz 5x5 dali tek basina 64*64*25 = 102.400
# parametre eder ve PLAN'in ~50-100k toplam hedefini asardi.
SK_GROUPS = 32
DROPOUT = 0.5

# ---------------------------------------------------------------
# EGITIM (PLAN Bolum 7.3)
# ---------------------------------------------------------------
OPTIMIZER = "adam"
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5

# ImageNet istatistikleri (PLAN 7.1)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------
# VERI ARTIRMA (PLAN Bolum 7.2) -- 19 etkin ornekle ZORUNLU
# ---------------------------------------------------------------
# YATAY/DIKEY CEVIRME YOK: spektrogramda zaman veya frekans eksenini ters
# cevirmek fiziksel olarak anlamsizdir (bkz. PLAN 7.2).
CROP_SCALE = (0.85, 1.0)    # RandomResizedCrop alan orani
JITTER = 0.10               # parlaklik/kontrast +-%10
MASK_MAX_FRAC = 0.10        # serit genisligi/yuksekligi <= eksenin %10'u
MASK_N = (1, 2)             # serit sayisi araligi
MASK_P = 0.5                # her eksen icin maskeleme uygulama olasiligi

# ---------------------------------------------------------------
# DEGERLENDIRME (PLAN Bolum 8.3)
# ---------------------------------------------------------------
SNR_BUCKETS = [("dusuk", 0, 5), ("orta", 5, 10), ("yuksek", 10, 16)]
