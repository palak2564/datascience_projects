"""
config.py
==========
Central configuration for the Nationality Detection System.
"""

import os

# ─── App ──────────────────────────────────────────────────────────────────────

APP_TITLE   = "🌐  Nationality Detection System"
WINDOW_SIZE = "1150x720"

# ─── Theme — editorial / travel-document aesthetic ────────────────────────────

THEME = {
    "bg":          "#f0ece4",   # warm parchment
    "card":        "#faf7f2",   # slightly lighter card
    "header_bg":   "#1a1f36",   # deep navy
    "header_text": "#f0ece4",
    "text":        "#1a1f36",
    "muted":       "#888070",
    "accent":      "#8b6914",   # warm gold
    "ok":          "#2e7d4f",
    "warn":        "#c47a1e",
    "danger":      "#c0392b",
    "info_bg":     "#e8f0f8",
}

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

AGE_DEPLOY_PATH = os.path.join(MODEL_DIR, "deploy_age.prototxt")
AGE_MODEL_PATH  = os.path.join(MODEL_DIR, "age_net.caffemodel")

# ─── Age Model ────────────────────────────────────────────────────────────────

AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ─── Nationality Mapping ──────────────────────────────────────────────────────
# Maps DeepFace raw ethnicity labels → human-readable nationality

NATIONALITY_MAP = {
    "indian":           "Indian",
    "black":            "African",
    "white":            "American",
    "asian":            "East Asian",
    "middle eastern":   "Middle Eastern",
    "latino hispanic":  "Latino/Hispanic",
}

# ─── Task Rules ───────────────────────────────────────────────────────────────
# Defines which attributes to predict per nationality group

TASK_RULES = {
    "indian": {
        "emotion":      True,
        "age":          True,
        "dress_colour": True,
    },
    "american": {
        "emotion":      True,
        "age":          True,
        "dress_colour": False,
    },
    "african": {
        "emotion":      True,
        "age":          False,
        "dress_colour": True,
    },
    "other": {
        "emotion":      True,
        "age":          False,
        "dress_colour": False,
    },
}

# ─── Dress Colour Palette ─────────────────────────────────────────────────────
# Named colours with (R, G, B) values used for nearest-neighbour colour naming.
# Expanded palette for better coverage of typical clothing colours.

DRESS_COLOUR_PALETTE = {
    "White":       (255, 255, 255),
    "Black":       (0,   0,   0),
    "Gray":        (128, 128, 128),
    "Light Gray":  (200, 200, 200),
    "Red":         (220, 30,  30),
    "Dark Red":    (140, 10,  10),
    "Pink":        (255, 153, 170),
    "Hot Pink":    (255, 69,  140),
    "Orange":      (255, 140, 0),
    "Yellow":      (255, 220, 50),
    "Lime Green":  (100, 210, 50),
    "Green":       (34,  139, 34),
    "Dark Green":  (0,   80,  0),
    "Olive":       (107, 142, 35),
    "Teal":        (0,   128, 128),
    "Cyan":        (0,   200, 210),
    "Sky Blue":    (135, 206, 235),
    "Blue":        (30,  100, 220),
    "Navy Blue":   (0,   0,   128),
    "Royal Blue":  (65,  105, 225),
    "Purple":      (128, 0,   128),
    "Violet":      (148, 0,   211),
    "Lavender":    (200, 162, 200),
    "Magenta":     (255, 0,   255),
    "Brown":       (139, 69,  19),
    "Tan":         (210, 180, 140),
    "Beige":       (245, 245, 220),
    "Cream":       (255, 253, 208),
    "Maroon":      (128, 0,   0),
    "Coral":       (255, 127, 80),
    "Khaki":       (195, 176, 145),
    "Gold":        (255, 215, 0),
    "Silver":      (192, 192, 192),
}
