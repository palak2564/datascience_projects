"""
config.py
==========
Central configuration for the animal detection project.

Instead of scattering magic numbers across files, they all live here.
Change stuff here, it applies everywhere.

This is the kind of thing that makes a project actually maintainable.
"""

# ─── Model Settings ────────────────────────────────────────────────────────────

# Default model size — options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
# n = nano (fastest, less accurate), x = extra large (slowest, most accurate)
DEFAULT_MODEL = "yolov8n"

# Default confidence threshold — boxes below this are ignored
DEFAULT_CONFIDENCE = 0.45

# IOU threshold for non-max suppression
# Higher = fewer overlapping boxes kept, lower = more boxes kept
IOU_THRESHOLD = 0.45

# Max detections per frame
MAX_DETECTIONS = 100


# ─── Animal Taxonomy ───────────────────────────────────────────────────────────

# All COCO class names that count as "animals" for our purposes
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    # Extended (would need custom dataset to detect these)
    "lion", "tiger", "leopard", "cheetah",
    "wolf", "fox", "deer", "rabbit",
    "monkey", "panda", "crocodile", "alligator",
    "snake", "eagle", "hawk",
}

# Carnivores — these get RED boxes and trigger the alert popup
CARNIVORE_SPECIES = {
    "cat", "bear", "lion", "tiger", "leopard", "cheetah",
    "wolf", "fox", "crocodile", "alligator", "snake",
    "eagle", "hawk", "bird",  # many birds are predatory
    "dog",  # domestic, but ancestrally predator
}

# Omnivores — YELLOW boxes (technically eat both, still flag as alert-worthy)
OMNIVORE_SPECIES = {"bear", "dog"}


# ─── Visual Settings ───────────────────────────────────────────────────────────

# BGR format (OpenCV uses BGR not RGB — classic gotcha)
COLOR_CARNIVORE  = (0,   30,  220)   # red
COLOR_OMNIVORE   = (0,  200,  255)   # yellow
COLOR_HERBIVORE  = (80,  200,  60)   # green
COLOR_DEFAULT    = (200, 200, 200)   # grey fallback

# Box line thickness (pixels)
BOX_THICKNESS_CARNIVORE = 3
BOX_THICKNESS_DEFAULT   = 2

# Label font scale
LABEL_FONT_SCALE = 0.55


# ─── GUI Settings ──────────────────────────────────────────────────────────────

WINDOW_TITLE = "🐾 Animal Detector — Wildlife AI"
WINDOW_SIZE  = "1200x800"
WINDOW_MIN   = (900, 600)

# Preview image max dimensions
PREVIEW_MAX_W = 800
PREVIEW_MAX_H = 550

# Theme colors
THEME = {
    "bg":       "#1a1a2e",
    "surface":  "#16213e",
    "accent":   "#e94560",
    "text":     "#eaeaea",
    "success":  "#4ecca3",
    "warn":     "#f5a623",
}


# ─── Output Settings ───────────────────────────────────────────────────────────

# Where to save annotated outputs by default
OUTPUT_DIR = "outputs"

# Log rotation — keep last N detection sessions in memory
MAX_HISTORY_ENTRIES = 50

# Carnivore alert auto-dismiss (milliseconds)
CARNIVORE_ALERT_TIMEOUT_MS = 8000


# ─── Video Settings ────────────────────────────────────────────────────────────

# Process every Nth frame to reduce CPU load on video files
# 1 = every frame, 2 = every other frame, etc.
VIDEO_FRAME_SKIP = 2

# Webcam device ID
WEBCAM_DEVICE_ID = 0
