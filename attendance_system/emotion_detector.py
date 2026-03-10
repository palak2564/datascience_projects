"""
utils/emotion_detector.py
==========================
Detects emotions from face crops.

Primary: DeepFace (uses FER/AffectNet model under the hood — very accurate)
Fallback: Simple OpenCV-based approximation (if DeepFace not installed)

Emotions detected: happy, sad, angry, surprised, fear, disgust, neutral

Why emotion detection in attendance? 
It's an interesting extra feature — imagine a teacher seeing that most
students look confused or tired during a lecture. Valuable feedback.
"""

import cv2
import numpy as np
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Try importing DeepFace — it's the best option but heavy
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    log.info("DeepFace loaded ✓")
except ImportError:
    DEEPFACE_AVAILABLE = False
    log.warning("DeepFace not installed. Using fallback emotion detection.")
    log.warning("Install with: pip install deepface")


class EmotionDetector:
    """
    Detects dominant emotion from a face image.

    Falls back gracefully if DeepFace isn't installed.
    """

    def __init__(self):
        self._cache = {}  # simple cache to avoid re-processing same face

    def detect(self, face_img: np.ndarray) -> str:
        """
        Returns dominant emotion string for a face image.
        e.g. "happy", "neutral", "sad", "angry", "surprised"
        """
        if face_img is None or face_img.size == 0:
            return "neutral"

        if DEEPFACE_AVAILABLE:
            return self._detect_deepface(face_img)
        else:
            return self._detect_fallback(face_img)

    def _detect_deepface(self, face_img: np.ndarray) -> str:
        """Use DeepFace for accurate emotion detection."""
        try:
            result = DeepFace.analyze(
                face_img,
                actions=["emotion"],
                enforce_detection=False,  # don't crash if face not re-detected
                silent=True,
            )
            # DeepFace returns a list when enforce_detection=False
            if isinstance(result, list):
                result = result[0]
            emotion = result["dominant_emotion"]
            return emotion.lower()
        except Exception as e:
            log.debug(f"DeepFace emotion failed: {e}")
            return "neutral"

    def _detect_fallback(self, face_img: np.ndarray) -> str:
        """
        Rough fallback emotion estimation when DeepFace isn't available.
        
        Uses simple heuristics:
        - Brightness variance → arousal proxy
        - Edge density → expression intensity proxy
        
        This is NOT accurate — it's purely a placeholder so the system
        still runs without DeepFace installed.
        """
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

            # Variance of pixel intensities (higher = more "expressive")
            variance = np.var(gray)

            # Edge density (mouth/eye movement creates edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Very rough heuristic
            if edge_density > 0.15:
                return "surprised"
            elif variance > 1800:
                return "happy"
            elif variance < 600:
                return "sad"
            else:
                return "neutral"

        except Exception:
            return "neutral"

    def detect_batch(self, face_imgs: list) -> list:
        """Process multiple faces at once."""
        return [self.detect(f) for f in face_imgs]
