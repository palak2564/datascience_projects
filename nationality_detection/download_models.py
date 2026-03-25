"""
download_models.py
===================
Downloads all required pre-trained model weights.

Run once before using the system:
    python download_models.py

Downloads:
- Age estimation Caffe model (Levi & Hassner 2015) — ~44 MB
- DeepFace models auto-download on first use (~200 MB total)
"""

import os
import sys
import urllib.request

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded / total_size * 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        sys.stdout.write(f"\r  [{bar}] {pct:.1f}%")
        sys.stdout.flush()


MODELS = [
    {
        "name": "Age Model Config",
        "url":  "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_age.prototxt",
        "dest": os.path.join(MODEL_DIR, "deploy_age.prototxt"),
    },
    {
        "name": "Age Model Weights (~44 MB)",
        "url":  "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
        "dest": os.path.join(MODEL_DIR, "age_net.caffemodel"),
    },
]


def main():
    print("\n" + "="*52)
    print("  Nationality Detection — Model Downloader")
    print("="*52 + "\n")

    for m in MODELS:
        if os.path.exists(m["dest"]):
            mb = os.path.getsize(m["dest"]) / 1e6
            print(f"  ✓ {m['name']} already exists ({mb:.1f} MB)")
            continue
        print(f"  ↓ {m['name']}")
        try:
            urllib.request.urlretrieve(m["url"], m["dest"], reporthook=progress)
            print()
            mb = os.path.getsize(m["dest"]) / 1e6
            print(f"    Saved ({mb:.1f} MB) ✓\n")
        except Exception as e:
            print(f"\n    ❌ Failed: {e}")
            print(f"    URL: {m['url']}\n")

    print("\n  DeepFace models (race/emotion/age) will auto-download")
    print("  on first run — needs internet, ~200 MB total.\n")
    print("="*52)
    print("  Done! Run:  python app.py")
    print("="*52 + "\n")


if __name__ == "__main__":
    main()
