"""
Sadece Augmentasyon Testi (DAS donusumu OLMADAN)
----------------------------------------------------
Bu script, tam pipeline'in SADECE augmentasyon kismini calistirir:
  - speed_change (hiz/perde varyasyonu, faz-vokodersiz)
  - genlik jitter'i
  - 10 saniyelik pencereye yerlestirme (kisa olaylar icin bosluklu
    serpistirme, uzun olaylar icin tek kopya)
  - MixUp (ayni siniftaki iki dosyayi karistirma)

BILEREK YAPILMAYANLAR (bir sonraki asamada eklenecek):
  - Gurultu karistirma (white/pink noise)
  - Downsample (1-5 kHz DAS-benzeri frekansa indirgeme)

Amac: augmentasyon adiminin TEK BASINA orijinal seslerin karakterini
koruyup korumadigini, gurultu/downsample adimlarindan bagimsiz olarak
dinleyerek dogrulayabilmek. Cikan .wav dosyalari ORIJINAL ornekleme
hizinda (44100 Hz vb.) kaydedilir -- yani doğrudan kulakla
degerlendirilebilir, "DAS'a benzetilmis" hali degildir.

Kullanim:
    python augment_only.py --input_dir raw_sounds --output_dir augmented_preview
    python augment_only.py --input_dir raw_sounds --output_dir augmented_preview --n_variants 5 --no_mixup

Bu script synth_das_pipeline.py ile AYNI KLASORDE olmalidir (fonksiyonlari
oradan import eder, kod tekrarini onlemek icin).
"""

import os
import glob
import argparse
import itertools
import numpy as np
import soundfile as sf

from synth_das_pipeline import (
    load_audio, resample_signal, augment_signal, place_event_in_window, mixup, save_spectrogram,
)

DEFAULT_WINDOW_SEC = 10.0
DEFAULT_PREVIEW_SR = 44100  # DAS'a indirgeme YOK -- sadece dosyalar arasi ORTAK ve
                              # YUKSEK KALITELI bir orneklem hizi (mixup icin sart)


def process_label_augment_only(label, files, out_dir, window_sec, n_variants_per_file,
                                 use_mixup, rng, preview_sr, mixup_target=0, spectrogram_max_n_fft=2048):
    """
    Bir sinifin tum dosyalari icin SADECE augmentasyon + pencereleme uygular.
    Gurultu karistirma ve DAS-frekansina downsample YOKTUR.

    Onemli: Farkli kaynak dosyalarin ORIJINAL orneklem hizlari birbirinden
    farkli olabilir (ornegin biri 44100 Hz, digeri 96000 Hz gibi profesyonel
    kayitlarda sik gorulur). MixUp iki sinyali dogrudan topladigi icin ayni
    uzunlukta (dolayisiyla ayni orneklem hizinda) olmalarini gerektirir.
    Bu yuzden her dosya, DAS frekansina degil ama ORTAK bir onizleme
    frekansina (preview_sr, varsayilan 44100 Hz -- hala tam kalite) once
    indirgenir/yukseltilir. Bu adim "DAS'a benzetme" degildir, sadece
    dosyalar arasi tutarliligi saglar.
    """
    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    total_written = 0

    prepared = []  # [(base_name, y, sr), ...] -- MixUp icin referans
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        y_raw, orig_sr = load_audio(path)
        y_raw, sr = resample_signal(y_raw, orig_sr, preview_sr)

        for i in range(n_variants_per_file):
            apply_aug = i > 0  # ilk varyant augmentasyonsuz -- "pencerelenmis orijinal" referansi
            if apply_aug:
                y = augment_signal(y_raw, rng)
            else:
                y = y_raw.copy()

            y_windowed = place_event_in_window(y, sr, window_sec, rng)
            peak = np.max(np.abs(y_windowed))
            if peak > 0:
                y_windowed = y_windowed / peak

            tag = f"{base}_v{i}" + ("_augmented" if apply_aug else "_orijinal_pencereli")
            sf.write(os.path.join(label_dir, f"{tag}.wav"), y_windowed, sr)
            save_spectrogram(y_windowed, sr, os.path.join(label_dir, f"{tag}_spectrogram.png"),
                              max_n_fft=spectrogram_max_n_fft)
            total_written += 1

            if i == 0:
                prepared.append((base, y_windowed, sr))

    if use_mixup and len(prepared) >= 2 and mixup_target > 0:
        pairs = list(itertools.combinations(range(len(prepared)), 2))
        for k in range(mixup_target):
            i1, i2 = pairs[rng.integers(0, len(pairs))]
            b1, y1, sr1 = prepared[i1]
            b2, y2, sr2 = prepared[i2]
            mixed_sig = mixup(y1, y2, rng)

            tag = f"mixup_{b1}_x_{b2}_{k}"
            sf.write(os.path.join(label_dir, f"{tag}.wav"), mixed_sig, sr1)
            save_spectrogram(mixed_sig, sr1, os.path.join(label_dir, f"{tag}_spectrogram.png"),
                              max_n_fft=spectrogram_max_n_fft)
            total_written += 1

    print(f"[OK] Sinif '{label}': {len(files)} ham dosyadan {total_written} augmentasyon-onizleme ornegi uretildi")
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="SADECE augmentasyon onizlemesi (gurultu/downsample YOK)")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--preview_sr", type=int, default=DEFAULT_PREVIEW_SR,
                         help="DAS frekansi DEGIL -- dosyalar arasi tutarlilik icin ortak, "
                              "yuksek kaliteli bir onizleme orneklem hizi (varsayilan 44100 Hz)")
    parser.add_argument("--n_variants", type=int, default=4,
                         help="Her ham dosyadan uretilecek varyant sayisi (1. varyant her zaman augmentasyonsuz)")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    labels = [d for d in sorted(os.listdir(args.input_dir))
              if os.path.isdir(os.path.join(args.input_dir, d))]
    if not labels:
        raise RuntimeError(f"'{args.input_dir}' altinda sinif klasoru bulunamadi.")

    total_written = 0
    for label in labels:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, label, "*.wav")) +
            glob.glob(os.path.join(args.input_dir, label, "*.mp3"))
        )
        if not files:
            print(f"[UYARI] '{label}' klasorunde ses dosyasi yok, atlaniyor.")
            continue
        can_mixup = (not args.no_mixup) and len(files) >= 2
        mixup_target = args.n_variants if can_mixup else 0
        total_written += process_label_augment_only(
            label, files, args.output_dir,
            window_sec=args.window_sec,
            n_variants_per_file=args.n_variants,
            use_mixup=not args.no_mixup,
            rng=rng,
            preview_sr=args.preview_sr,
            mixup_target=mixup_target,
        )

    print(f"\nTamamlandi. Toplam {total_written} augmentasyon-onizleme ornegi: {args.output_dir}")
    print("\nNOT: Bu dosyalar DAS-benzeri degil -- orijinal ornekleme hizinda,")
    print("gurultusuz. Amac sadece augmentasyonun (speed_change/MixUp/pencereleme)")
    print("ses karakterini koruyup korumadigini dinleyerek dogrulamak.")


if __name__ == "__main__":
    main()
