"""
download_models.py
===================
Downloads all required pre-trained model files.

Run this ONCE before using the app:
    python download_models.py

Models downloaded:
1. Age estimation (Levi & Hassner 2015)
   - deploy_age.prototxt    (~3 KB)
   - age_net.caffemodel     (~44 MB)

2. dlib 68-point face landmarks (optional but improves drowsiness accuracy)
   - shape_predictor_68_face_landmarks.dat (~95 MB, compressed)
   - Note: requires manual download from dlib.net (too large for auto-download)

All models are saved to the models/ directory.
"""

import os
import sys
import urllib.request

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def progress_bar(block_num, block_size, total_size):
    """Shows download progress in terminal."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded / total_size * 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        sys.stdout.write(f"\r  [{bar}] {pct:.1f}%")
        sys.stdout.flush()


MODELS = [
    {
        "name": "Age Model Config (prototxt)",
        "url":  "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_age.prototxt",
        "dest": os.path.join(MODEL_DIR, "deploy_age.prototxt"),
    },
    {
        "name": "Age Model Weights (caffemodel)",
        "url":  "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
        "dest": os.path.join(MODEL_DIR, "age_net.caffemodel"),
    },
]


def download_all():
    print("\n" + "="*55)
    print("  Drowsiness Detection — Model Downloader")
    print("="*55 + "\n")

    for m in MODELS:
        dest = m["dest"]
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  ✓ {m['name']} already exists ({size_mb:.1f} MB)")
            continue

        print(f"  ↓ Downloading: {m['name']}")
        try:
            urllib.request.urlretrieve(m["url"], dest, reporthook=progress_bar)
            print()  # newline after progress bar
            size_mb = os.path.getsize(dest) / 1e6
            print(f"    Saved: {dest} ({size_mb:.1f} MB) ✓\n")
        except Exception as e:
            print(f"\n    ❌ Failed: {e}")
            print(f"    Manual download URL: {m['url']}")
            print(f"    Save to: {dest}\n")

    # dlib landmark model — manual step
    dlib_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(dlib_path):
        print("  ─── dlib Landmark Model (OPTIONAL but recommended) ───────")
        print("  For better drowsiness accuracy, download manually:")
        print("  1. Go to: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("  2. Extract the .dat file")
        print(f"  3. Place it at: {dlib_path}")
        print("  Without it, the system uses OpenCV eye cascade (still works).\n")
    else:
        print(f"  ✓ dlib landmark model found ✓")

    print("="*55)
    print("  Done! You can now run: python app.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    download_all()
