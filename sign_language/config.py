"""
config.py
==========
All configuration for the Sign Language Detection System.
"""

import os
from datetime import time as dtime

# ─── App ──────────────────────────────────────────────────────────────────────

APP_TITLE   = "◈  Sign Language Detector"
WINDOW_SIZE = "1150x720"

# ─── Time window ──────────────────────────────────────────────────────────────
# System only accepts input during this window

ACTIVE_START = dtime(18, 0)   # 6:00 PM
ACTIVE_END   = dtime(22, 0)   # 10:00 PM

# ─── Theme — neon / HUD ───────────────────────────────────────────────────────

THEME = {
    "bg":      "#0a0a0f",
    "surface": "#111118",
    "cyan":    "#00e5ff",
    "green":   "#76ff03",
    "warn":    "#ff9800",
    "danger":  "#ff3d00",
    "text":    "#d0d0e0",
    "dim":     "#444466",
    "btn_bg":  "#1a1a2e",
}

# ─── Labels ───────────────────────────────────────────────────────────────────

# 10 known words (custom trained)
KNOWN_WORDS = [
    "Hello", "ThankYou", "Yes", "No", "Please",
    "Sorry", "Help", "Eat", "Water", "More",
]

# Full ASL alphabet
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Combined label list (36 total)
ALL_LABELS = KNOWN_WORDS + ALPHABET

# ─── Model ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# MediaPipe gives 21 landmarks × 3 coords = 63 features
LANDMARK_FEATURES = 63

# Image resize for processing
IMG_SIZE = (224, 224)

# ─── Training ─────────────────────────────────────────────────────────────────

TRAIN_EPOCHS     = 50
TRAIN_BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
MIN_CONFIDENCE   = 0.60
