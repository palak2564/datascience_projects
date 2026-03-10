"""
config.py
==========
Central configuration for the attendance system.
Change these to match your setup.
"""

from datetime import time as dtime
import os

# ─── Time Window ──────────────────────────────────────────────────────────────
# The system ONLY runs between these times.
# Outside this window, attendance_system.py will refuse to start.
ATTENDANCE_START = dtime(9, 30)   # 9:30 AM
ATTENDANCE_END   = dtime(10, 0)   # 10:00 AM

# ─── Camera / Source ──────────────────────────────────────────────────────────
WEBCAM_ID = 0  # 0 = default webcam, 1 = second webcam, etc.

# ─── Recognition Settings ─────────────────────────────────────────────────────
# Minimum confidence % to accept a recognition (0-100)
# LBPH: lower distance = better. We convert: confidence = 100 - distance
# 50 = fairly lenient, 70 = stricter, 80+ = very strict
CONFIDENCE_THRESHOLD = 50.0
RECOGNITION_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD

# ─── Face Detection Settings ──────────────────────────────────────────────────
# Size faces are resized to for recognition
FACE_SIZE = (100, 100)  # (width, height) — must match training size

# ─── Processing ───────────────────────────────────────────────────────────────
# Process every Nth frame (higher = faster but may miss faces)
FRAME_SKIP = 3

# Show live camera feed during attendance (requires display)
DISPLAY_FEED = True

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "known_faces")
MODEL_DIR           = os.path.join(BASE_DIR, "models")
OUTPUT_DIR          = os.path.join(BASE_DIR, "outputs")

# ─── Session ──────────────────────────────────────────────────────────────────
# Minimum seconds a student must be visible to count as present
# (prevents random passersby from being marked)
MIN_PRESENCE_SECONDS = 2  # not yet implemented — future enhancement
