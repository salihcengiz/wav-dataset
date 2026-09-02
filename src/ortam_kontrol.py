"""
ORTAM KONTROLU -- paylasilan sunucuda ne bozuldu?

=== NEDEN VAR ===

Sunucu paylasilan ve ortam tekrar tekrar bozuluyor:

  2026-08-27  numpy bozuldu, kosu 1 baslarken oldu
              ("RuntimeError: Numpy is not available")
  2026-09-02  --user ile kurulan paketlerin HEPSI silinmisti:
              torch, torchvision, onnx, onnxruntime, mlflow
  2026-09-02  mlflow-skinny kurulumu protobuf'u 4.24.3 -> 6.33.6
              yukseltti ve TENSORFLOW 2.14'U BOZDU (o `<5.0` istiyor).
              Bu sunucuda baskalari TF kullaniyor.

Her seferinde ayni teshis komutlarini elden yazmak yerine tek script.

=== NE ZAMAN KOSTURULUR ===

  - Her `pip install`ten SONRA (zorunlu -- baskalarinin TF'sini bozmus
    olabilirsin)
  - Uzun bir kosuya baslamadan ONCE
  - "dun calisiyordu, bugun calismiyor" durumunda ILK IS

=== KULLANIM ===

    python3 ortam_kontrol.py

Cikti "BOZUK" iceriyorsa DURUM.md Bolum 6'daki duzeltmelere bak.
protobuf icin:  pip install --user "protobuf==4.24.3"
"""
import importlib
import os
import shutil
import sys

# (modul, beklenen surum ya da None)  -- beklenen verilirse UYUSMAZLIK basar
PAKETLER = [
    ("numpy", "1.26.4"),            # 2 kullanma: sunucu 1.26.4
    ("google.protobuf", "4.24.3"),  # TF 2.14 <5.0 istiyor -- mlflow bunu bozar
    ("tensorflow", None),           # ekibin modelleri; biz kullanmiyoruz
    ("torch", None),
    ("torchvision", None),
    ("onnx", "1.14.1"),             # IR 7 yaziyor; yenisi daha yuksek IR yazar
    ("onnxruntime", None),
    ("mlflow", None),
    ("h5py", None),
    ("pandas", None),
    ("sklearn", None),
]


def paketler():
    print("=" * 68)
    print("PAKETLER  (beklenen surumler SUNUCU icin -- yerelde farkli olmasi")
    print("           normal, bkz. DURUM.md Bolum 8A)")
    print("=" * 68)
    kotu = 0
    for ad, beklenen in PAKETLER:
        try:
            m = importlib.import_module(ad)
            s = getattr(m, "__version__", "?")
            not_ = ""
            if beklenen and s != beklenen:
                not_ = f"   <-- UYUSMAZLIK, beklenen {beklenen}"
                kotu += 1
            print(f"  {ad:18s} OK      {s}{not_}")
        except Exception as e:            # noqa: BLE001
            print(f"  {ad:18s} BOZUK   {type(e).__name__}: {str(e)[:60]}")
            kotu += 1
    return kotu


def ortam():
    print("\n" + "=" * 68)
    print("ORTAM")
    print("=" * 68)
    print(f"  python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  HOME        : {os.environ.get('HOME')}")

    # /dev/shm -- 64 MB ise DataLoader iscileri "Bus error" ile coker
    try:
        t, _, b = shutil.disk_usage("/dev/shm")
        mb = t / 1e6
        print(f"  /dev/shm    : {mb:.0f} MB"
              + ("   <-- 64 MB, --isci 3'u asma (Bus error)" if mb < 200 else ""))
    except OSError:
        print("  /dev/shm    : okunamadi")

    try:
        import torch
        print(f"  CUDA        : {torch.cuda.is_available()}"
              + (f"  {torch.cuda.get_device_name(0)}"
                 if torch.cuda.is_available() else ""))
        # 2026-08-27 hatasinin birebir testi
        import numpy as np
        torch.from_numpy(np.zeros(3))
        print(f"  torch<-numpy: OK")
    except Exception as e:                # noqa: BLE001
        print(f"  torch/numpy : BOZUK  {type(e).__name__}: {str(e)[:60]}")
        return 1
    return 0


if __name__ == "__main__":
    n = paketler() + ortam()
    print("\n" + "=" * 68)
    if n:
        print(f"{n} sapma var.")
        print("  SUNUCUDA calisiyorsan: DURUM.md Bolum 6'ya bak.")
        print("  protobuf bozulduysa  : pip install --user \"protobuf==4.24.3\"")
        print("  YERELDE calisiyorsan : sapmalarin cogu beklenen "
              "(TF/mlflow yok, numpy 2.x, CPU torch).")
    else:
        print("Ortam sunucu beklentisiyle birebir.")
    sys.exit(1 if n else 0)
