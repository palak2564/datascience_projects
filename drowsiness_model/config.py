"""
config.py
==========
All settings for the drowsiness detection system.
"""

import os

# ─── GUI Theme ────────────────────────────────────────────────────────────────

APP_TITLE   = "⚠  Drowsiness Detection System — Driver Safety AI"
WINDOW_SIZE = "1200x750"

THEME = {
    "bg":      "#0d0d0d",
    "surface": "#151515",
    "danger":  "#e63946",
    "warn":    "#f4a261",
    "ok":      "#57cc99",
    "text":    "#d4d4d4",
    "dim":     "#444444",
}

# ─── Detection ────────────────────────────────────────────────────────────────

EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.6
FRAME_SKIP        = 2
FACE_SIZE         = (100, 100)

# ─── Visualization Colors (BGR) ───────────────────────────────────────────────

COLOR_SLEEPING = (0,   30,  230)
COLOR_AWAKE    = (80,  200,  60)
COLOR_UNKNOWN  = (150, 150, 150)

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

SHAPE_PREDICTOR_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
AGE_DEPLOY_PATH      = os.path.join(MODEL_DIR, "deploy_age.prototxt")
AGE_MODEL_PATH       = os.path.join(MODEL_DIR, "age_net.caffemodel")

# ─── Age Model ────────────────────────────────────────────────────────────────

AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]

MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
