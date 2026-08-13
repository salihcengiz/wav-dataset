"""
DAS Sentetik Veri Uretim Pipeline'i (v3 - gercekci pencereleme + faz-vokodersiz augmentasyon)
------------------------------------------------------------------------------------------------
Ses tabanli olay kayitlarini (fence cutting, metal bending,
chain-link fence climbing) DAS-benzeri faz/gerinim (strain rate)
sinyaline donusturur ve az sayida gercek kayittan buyuk, cesitlendirilmis
bir sentetik veri seti uretir.

v3 degisiklikleri (onceki surumdeki iki artefaktin duzeltilmesi):
  - "Ugultu" sorunu: kisa (ornegin 150ms) darbeli olaylar artik BOSLUKSUZ
    np.tile ile degil, aralarinda RASTGELE SESSIZLIK BOSLUKLARIYLA
    serpistirilerek pencereye yerlestiriliyor (place_event_in_window).
    Boylece tek bir "cit" sesi periyodik bir tona donusmuyor.
  - "Su altinda" sorunu: pitch-shift/time-stretch icin kullanilan
    librosa.effects fonksiyonlari FAZ VOKODERI tabanliydi ve darbeli/
    gecici seslerde "fazlilik" (phasiness) artefakti uretiyordu. Bunlarin
    yerine SAF YENIDEN-ORNEKLEMEYE dayanan speed_change() kullaniliyor --
    faz manipulasyonu icermedigi icin darbeli seslerde net kaliyor.

Adimlar:
  1) Ham ses dosyalarini yukle (.wav / .mp3), ORIJINAL ornekleme hizinda
  2) Augmentasyon: speed_change (hiz/perde, faz-vokodersiz) + genlik jitter'i
     -- downsample'dan ONCE, orijinal sr'de uygulanir
  3) Hedef DAS ornekleme frekansina indir (1-5 kHz araligi, DAS literaturuyle uyumlu)
  4) 10 saniyelik pencereye YERLESTIR: uzun/surekli olaylar tek kopya olarak
     rastgele konuma, kisa/darbeli olaylar birden fazla kopya + rastgele
     bosluklarla yerlestirilir (place_event_in_window)
  5) Arka plan gurultusuyle (white/pink noise) rastgele SNR'de karistir
  6) Hem sentetik .wav hem de STFT spektrogram (.png) uretir (kisa sinyaller
     icin otomatik kuculen n_fft ile)
  7) Opsiyonel: ayni siniftaki iki farkli kayit MixUp ile karistirilir
     (Zhang & Chen 2024, MSF-DenseNet makalesindeki kucuk-orneklem
     stratejisiyle ayni mantik)

Beklenen girdi klasor yapisi:
    input_dir/
        fence_cutting/
            angle_grinder_weldmesh.wav
            bolt_cutter_snip.wav
            ... (siniflar basina en az 5-10 farkli gercek kayit onerilir)
        metal_bending/
            ...
        chain_link_climbing/
            ...

Kullanim (toplam hedef ornek sayisina gore otomatik hesaplama):
    python synth_das_pipeline.py --input_dir raw_sounds --output_dir synthetic_dataset --total_target 1000

Kullanim (dosya basina sabit varyant sayisi ile):
    python synth_das_pipeline.py --input_dir raw_sounds --output_dir synthetic_dataset --n_variants 40
"""

import os
import glob
import argparse
import itertools
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from fractions import Fraction


# ---------------------------------------------------------------
# VARSAYILAN PARAMETRELER (komut satirindan degistirilebilir)
# ---------------------------------------------------------------
DEFAULT_TARGET_SR = 2000       # Hedef DAS-benzeri ornekleme frekansi (Hz): 1000-5000 onerilir
DEFAULT_WINDOW_SEC = 10.0      # Pencere uzunlugu (saniye)
DEFAULT_SNR_RANGE = (0.0, 15.0)  # Olay/gurultu orani araligi (dB), her varyant icin rastgele secilir
N_FFT = 256
HOP_LENGTH = 32


def load_audio(path):
    """Ses dosyasini yukle, mono'ya cevir. Orijinal ornekleme frekansini dondurur."""
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def resample_signal(y, orig_sr, target_sr):
    """Sinyali hedef ornekleme frekansina indirger (downsample)."""
    if orig_sr == target_sr:
        return y, target_sr
    y_rs = resample_poly(y, target_sr, orig_sr)
    return y_rs, target_sr


def generate_background_noise(n_samples, kind="pink", seed=None):
    """
    DAS arka plan / cevresel gurultusunu simule eder.
    kind='white' -> beyaz gurultu (basit, hizli, tum frekanslarda esit enerji)
    kind='pink'  -> pembe gurultu (1/f), gercek DAS taban gurultusune
                    (ruzgar, uzak trafik vb.) daha yakin bir spektral profil
    """
    rng = np.random.default_rng(seed)
    if kind == "white":
        noise = rng.normal(0, 1, n_samples)
    elif kind == "pink":
        white = rng.normal(0, 1, n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # sifira bolme koruması
        fft = fft / np.sqrt(freqs)
        noise = np.fft.irfft(fft, n=n_samples)
    else:
        raise ValueError("kind 'white' veya 'pink' olmali")
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak
    return noise


def mix_at_snr(signal, noise, snr_db):
    """Sinyali, hedef SNR (dB) degerine gore olceklenmis gurultuyle karistirir."""
    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = signal + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak
    return mixed


def adaptive_n_fft(n_samples, max_fft=2048, min_fft=64):
    """Kisa sinyaller icin STFT pencere boyutunu kucultur (asiri bulanikligi onler)."""
    n = max_fft
    while n > min_fft and n > n_samples // 4:
        n //= 2
    return max(n, min_fft)


def speed_change(y, rate):
    """
    Faz-vokoderi ICERMEYEN hiz/perde degisimi -- saf yeniden-orneklemeye
    (resampling) dayanir, tipki bir teybi/plagi farkli hizda calmak gibi.
    rate > 1  -> daha hizli VE daha tiz (sure kisalir)
    rate < 1  -> daha yavas VE daha pes (sure uzar)

    librosa.effects.pitch_shift / time_stretch, ikisi de icten ice faz
    vokoderi (STFT fazi manipule etme) kullanir. Faz vokoderleri darbeli/
    gecici (transient) agirlikli seslerde -- ornegin metal kesme, testereleme,
    vurma gibi -- bilinen bir "fazlilik" (phasiness) artefakti uretir; bu da
    kulakta "su altinda" hissi yaratir. Saf yeniden-orneklemeye dayanan bu
    fonksiyon faz manipulasyonu icermedigi icin darbeli seslerde net ve
    gercekci kalir.
    """
    ratio = 1.0 / rate  # yeni_uzunluk / eski_uzunluk
    frac = Fraction(ratio).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up < 1 or down < 1 or len(y) < 8:
        return y
    return resample_poly(y, up, down)


def place_event_in_window(y, sr, window_sec, rng, long_event_ratio=0.4,
                           min_reps=2, max_reps=6, min_gap_sec=0.3, max_gap_sec=2.0):
    """
    Olay sinyalini window_sec uzunlugunda bir pencereye yerlestirir.

    Iki farkli davranis:
      - UZUN olay (kaynak >= window_sec * long_event_ratio, ornegin zaten
        birkaç saniyelik surekli bir testereleme sesi): pencere icine
        RASTGELE BIR NOKTADAN tek kopya olarak yerlestirilir, geri kalani
        sessiz birakilir (sonradan arka plan gurultusuyle doldurulacak).
        Boylece dogal olarak var olan tekrarlanan hareket (ileri-geri
        testereleme gibi) tekrar tekrar dongulenerek yapay/periyodik hale
        getirilmez.
      - KISA olay (ornegin 150ms'lik tek bir "cit" sesi): np.tile ile
        BOSLUKSUZ art arda dizmek yerine, 2 ile 6 arasinda rastgele sayida
        kopya, ARALARINDA RASTGELE SESSIZLIK BOSLUKLARIYLA pencereye
        serpistirilir. Bu, gercek bir olayin (ornegin bir cit telini
        birkac kez kesmeye calisma) birden fazla ayri darbeden olustugu
        varsayimina dayanir ve tekrarlayan darbenin periyodik bir "ton"a
        donusmesini (uğultu artefaktini) tamamen ortadan kaldirir.
    """
    target_len = int(round(window_sec * sr))
    out = np.zeros(target_len, dtype=np.float64)
    if len(y) == 0:
        return out

    if len(y) >= target_len:
        start = rng.integers(0, len(y) - target_len + 1)
        return y[start:start + target_len]

    if len(y) >= target_len * long_event_ratio:
        # Uzun/surekli olay: tek kopya, rastgele konumda
        max_start = target_len - len(y)
        start = rng.integers(0, max_start + 1)
        out[start:start + len(y)] = y
        return out

    # Kisa/darbeli olay: birden fazla kopya, aralarinda rastgele bosluk
    n_reps = int(rng.integers(min_reps, max_reps + 1))
    min_gap = int(min_gap_sec * sr)
    max_gap = int(max_gap_sec * sr)

    # Toplam alan yetersizse tekrar sayisini azalt
    while n_reps > 1 and n_reps * len(y) + (n_reps - 1) * min_gap > target_len:
        n_reps -= 1

    pos = rng.integers(0, max(1, target_len // (n_reps + 1)))  # baslangic once bosluk
    for _ in range(n_reps):
        if pos + len(y) > target_len:
            break
        out[pos:pos + len(y)] = y
        gap = int(rng.integers(min_gap, max_gap + 1)) if max_gap > min_gap else min_gap
        pos += len(y) + gap
    return out


def mixup(y1, y2, rng, alpha=0.4):
    """
    MixUp veri artirma (Zhang ve ark. 2017; Zhang & Chen 2024'te DAS
    kucuk-orneklem sinifllandirmasinda kullanilmisti).
    Ayni siniftaki iki farkli kaydi rastgele bir agirlikla dogrusal
    olarak karistirarak sentetik ama fiziksel olarak makul bir uc-uncu
    varyant uretir. iki sinyal de ayni uzunlukta olmalidir.
    """
    lam = rng.beta(alpha, alpha)
    mixed = lam * y1 + (1 - lam) * y2
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak
    return mixed


def save_spectrogram(y, sr, out_path, max_n_fft=N_FFT):
    """STFT spektrogramini eksensiz, kirpilmis bir gorsel olarak kaydeder.
    Kisa sinyaller icin n_fft/hop_length otomatik kucultulur (adaptive_n_fft)."""
    n_fft_eff = adaptive_n_fft(len(y), max_fft=max_n_fft, min_fft=16)
    if n_fft_eff < 8 or len(y) < 8:
        return  # cok kisa sinyal, spektrogram anlamli olmaz
    hop_eff = max(4, n_fft_eff // 4)
    S = librosa.stft(y, n_fft=n_fft_eff, hop_length=hop_eff)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = plt.axes((0, 0, 1, 1))
    ax.axis("off")
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_eff,
                              x_axis=None, y_axis=None, cmap="viridis", ax=ax)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def augment_signal(y, rng, speed_range=(0.85, 1.15), gain_db_range=(-3.0, 3.0)):
    """
    Sinyale fiziksel olarak makul kucuk varyasyonlar uygular -- SADECE
    faz-vokoderi ICERMEYEN yontemlerle (bkz. speed_change ust yorumu):
      - speed_change: farkli kuvvet/hiz ile uretilen olayin hem sure hem
        perde farkini birlikte simule eder (teyp hizi degisimi mantigi)
      - gain jitter: mikrofon-kaynak mesafesi / darbe siddeti farkini simule eder
    Orijinal (yuksek ornekleme hizli) sinyal uzerinde, downsample'dan ONCE
    cagrilmalidir -- boylece resample_poly hassas calisir ve ardindan
    hedef DAS ornekleme hizina bir kez indirgeme yapilir.
    """
    rate = rng.uniform(*speed_range)
    y_aug = speed_change(y, rate)

    gain_db = rng.uniform(*gain_db_range)
    y_aug = y_aug * (10 ** (gain_db / 20))

    peak = np.max(np.abs(y_aug))
    if peak > 0:
        y_aug = y_aug / peak
    return y_aug


def load_and_prepare(path, target_sr, window_sec, rng, apply_augment=True):
    """
    Ham dosyayi yukler -> (opsiyonel) augmentasyon orijinal sr'de uygulanir
    -> hedef DAS ornekleme hizina indirgenir -> pencereye yerlestirilir.
    Donen sinyal HENUZ gurultuyle karistirilmamistir.
    """
    y, orig_sr = load_audio(path)
    if apply_augment:
        y = augment_signal(y, rng)
    y_ds, sr = resample_signal(y, orig_sr, target_sr)
    y_windowed = place_event_in_window(y_ds, sr, window_sec, rng)
    peak = np.max(np.abs(y_windowed))
    if peak > 0:
        y_windowed = y_windowed / peak
    return y_windowed, sr


def process_label(label, files, out_dir, target_sr, window_sec, snr_range,
                   noise_kinds, n_variants_per_file, use_mixup, rng, mixup_target=0):
    """
    Bir sinifin tum ham dosyalarindan sentetik varyantlar uretir.
    Her dosya icin: augmentasyon (speed_change/gain) + rastgele SNR'de
    gurultu karistirma ile n_variants_per_file adet ornek uretilir.
    use_mixup=True ise, sinifta 2+ dosya varsa ek olarak mixup_target
    adet MixUp varyanti da uretilir (ayni cift birden fazla kez, farkli
    SNR/gurultu ile kullanilabilir).
    """
    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    total_written = 0

    prepared = []  # [(base_name, y, sr), ...] -- MixUp icin ham (gurultusuz) hazirlanmis sinyaller
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        for i in range(n_variants_per_file):
            apply_aug = i > 0  # ilk varyant augmentasyonsuz -- MixUp icin "temiz" referans
            y_aug, sr = load_and_prepare(path, target_sr, window_sec, rng, apply_augment=apply_aug)

            snr_db = float(rng.uniform(*snr_range))
            noise_kind = noise_kinds[rng.integers(0, len(noise_kinds))]
            noise = generate_background_noise(len(y_aug), kind=noise_kind, seed=None)
            mixed = mix_at_snr(y_aug, noise, snr_db)

            tag = f"{base}_snr{snr_db:.0f}dB_{noise_kind}_v{i}"
            sf.write(os.path.join(label_dir, f"{tag}.wav"), mixed, sr)
            save_spectrogram(mixed, sr, os.path.join(label_dir, f"{tag}_spectrogram.png"))
            total_written += 1

            if i == 0:
                prepared.append((base, y_aug, sr))  # mixup icin referans sakla

    if use_mixup and len(prepared) >= 2 and mixup_target > 0:
        pairs = list(itertools.combinations(range(len(prepared)), 2))
        # Ayni ciftler farkli SNR/gurultu ile birden fazla kez kullanilabilir
        # (hedefe ulasmak icin), boylece cok az sayida ham dosya olsa bile
        # mixup_target sayisina yaklasilabilir.
        for k in range(mixup_target):
            i1, i2 = pairs[rng.integers(0, len(pairs))]
            b1, y1, sr1 = prepared[i1]
            b2, y2, sr2 = prepared[i2]
            mixed_sig = mixup(y1, y2, rng)

            snr_db = float(rng.uniform(*snr_range))
            noise_kind = noise_kinds[rng.integers(0, len(noise_kinds))]
            noise = generate_background_noise(len(mixed_sig), kind=noise_kind, seed=None)
            mixed = mix_at_snr(mixed_sig, noise, snr_db)

            tag = f"mixup_{b1}_x_{b2}_snr{snr_db:.0f}dB_{noise_kind}_{k}"
            sf.write(os.path.join(label_dir, f"{tag}.wav"), mixed, sr1)
            save_spectrogram(mixed, sr1, os.path.join(label_dir, f"{tag}_spectrogram.png"))
            total_written += 1

    print(f"[OK] Sinif '{label}': {len(files)} ham dosyadan {total_written} sentetik ornek uretildi")
    return total_written


def main():
    parser = argparse.ArgumentParser(
        description="Ses kayitlarindan sentetik DAS veri seti ureten pipeline")
    parser.add_argument("--input_dir", required=True,
                         help="Sinif basina alt klasor iceren ham ses klasoru")
    parser.add_argument("--output_dir", required=True, help="Cikti klasoru")
    parser.add_argument("--target_sr", type=int, default=DEFAULT_TARGET_SR,
                         help="Hedef ornekleme frekansi (Hz), 1000-5000 onerilir")
    parser.add_argument("--window_sec", type=float, default=DEFAULT_WINDOW_SEC,
                         help="Pencere uzunlugu (saniye)")
    parser.add_argument("--noise_kinds", nargs="+", default=["pink", "white"],
                         help="Kullanilacak arka plan gurultu tipleri (aralarindan rastgele secilir)")
    parser.add_argument("--snr_min", type=float, default=DEFAULT_SNR_RANGE[0])
    parser.add_argument("--snr_max", type=float, default=DEFAULT_SNR_RANGE[1])
    parser.add_argument("--n_variants", type=int, default=None,
                         help="Her ham ses dosyasindan uretilecek sabit varyant sayisi. "
                              "--total_target verilirse bu yok sayilir ve otomatik hesaplanir.")
    parser.add_argument("--total_target", type=int, default=None,
                         help="Tum veri setinde ulasilmak istenen TOPLAM sentetik ornek sayisi "
                              "(orn. 1000). Siniflar arasinda esit dagitilir, sinif icindeki "
                              "dosya sayisina gore dosya basina varyant sayisi otomatik hesaplanir.")
    parser.add_argument("--no_mixup", action="store_true",
                         help="MixUp varyant uretimini kapat (varsayilan: acik)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu (tekrar uretilebilirlik icin)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    labels = [d for d in sorted(os.listdir(args.input_dir))
              if os.path.isdir(os.path.join(args.input_dir, d))]
    if not labels:
        raise RuntimeError(
            f"'{args.input_dir}' altinda sinif klasoru bulunamadi.\n"
            "Beklenen yapi: input_dir/fence_cutting/*.wav, "
            "input_dir/metal_bending/*.wav, input_dir/chain_link_climbing/*.wav"
        )

    label_files = {}
    for label in labels:
        files = sorted(
            glob.glob(os.path.join(args.input_dir, label, "*.wav")) +
            glob.glob(os.path.join(args.input_dir, label, "*.mp3"))
        )
        if not files:
            print(f"[UYARI] '{label}' klasorunde ses dosyasi bulunamadi, atlaniyor.")
            continue
        label_files[label] = files

    if not label_files:
        raise RuntimeError("Hicbir sinif klasorunde ses dosyasi bulunamadi.")

    # --- dosya basina uretilecek varyant sayisini belirle ---
    if args.total_target is not None:
        per_label_target = args.total_target / len(label_files)
        print(f"Hedef: toplam {args.total_target} ornek, sinif basina ~{per_label_target:.0f} ornek\n")
    else:
        per_label_target = None

    total_written = 0
    for label, files in label_files.items():
        can_mixup = (not args.no_mixup) and len(files) >= 2
        if per_label_target is not None:
            if can_mixup:
                # Hedefin ~%70'i dogrudan augmentasyondan, ~%30'u MixUp'tan gelsin
                aug_target = round(per_label_target * 0.7)
                mixup_target = max(0, round(per_label_target) - aug_target)
            else:
                aug_target = round(per_label_target)
                mixup_target = 0
            n_variants_per_file = max(1, round(aug_target / len(files)))
        else:
            n_variants_per_file = args.n_variants or 3
            mixup_target = n_variants_per_file if can_mixup else 0

        written = process_label(
            label, files, args.output_dir,
            target_sr=args.target_sr,
            window_sec=args.window_sec,
            snr_range=(args.snr_min, args.snr_max),
            noise_kinds=args.noise_kinds,
            n_variants_per_file=n_variants_per_file,
            use_mixup=not args.no_mixup,
            rng=rng,
            mixup_target=mixup_target,
        )
        total_written += written

    print(f"\nTamamlandi. {len(label_files)} sinif icin toplam {total_written} sentetik ornek uretildi: {args.output_dir}")
    for label, files in label_files.items():
        n_out = len(glob.glob(os.path.join(args.output_dir, label, "*.wav")))
        print(f"  - {label}: {len(files)} ham dosya -> {n_out} sentetik ornek")


if __name__ == "__main__":
    main()
