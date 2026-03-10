"""
utils/face_recognizer.py
==========================
Handles face detection + recognition using OpenCV LBPH.

Two stages:
1. Detection  — Haar Cascade (finds WHERE faces are in the frame)
2. Recognition — LBPH model (says WHO that face belongs to)

The LBPH model must be pre-trained via train.py before this works.
"""

import cv2
import os
import pickle
import numpy as np
import logging
from typing import Tuple, List, Optional

from config import MODEL_DIR, FACE_SIZE, RECOGNITION_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

# Haar cascade paths
FRONTAL_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PROFILE_CASCADE = cv2.data.haarcascades + "haarcascade_profileface.xml"


class FaceRecognizer:
    """
    Wraps face detection (Haar Cascade) + recognition (LBPH).

    Usage:
        recognizer = FaceRecognizer()
        faces = recognizer.detect_faces(frame)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            name, confidence = recognizer.recognize(face_roi)
    """

    def __init__(self):
        self.model_path = os.path.join(MODEL_DIR, "lbph_model.yml")
        self.label_map_path = os.path.join(MODEL_DIR, "label_map.pkl")

        # Detection
        self.face_cascade = cv2.CascadeClassifier(FRONTAL_CASCADE)

        # Recognition model + labels
        self.recognizer = None
        self.label_map = {}
        self._load_model()

    def _load_model(self):
        """Load the trained LBPH model if it exists."""
        if not os.path.exists(self.model_path):
            log.warning(f"Model not found: {self.model_path} — run train.py first")
            return

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.model_path)

            with open(self.label_map_path, "rb") as f:
                self.label_map = pickle.load(f)

            log.info(f"Model loaded ✓ — {len(self.label_map)} students enrolled")
        except Exception as e:
            log.error(f"Failed to load model: {e}")

    def is_trained(self) -> bool:
        """Returns True if a trained model is loaded."""
        return self.recognizer is not None

    def detect_faces(self, frame: np.ndarray) -> List[tuple]:
        """
        Detects faces in a BGR frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return list(faces) if len(faces) > 0 else []

    def recognize(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Recognizes a face ROI.

        Returns:
            (student_name, confidence_score)
            confidence: 0-100, lower = more confident in LBPH
                        We invert it to 0-100% for readability

        Note: LBPH confidence is a DISTANCE — lower = better match.
        We convert: confidence% = max(0, 100 - distance)
        Threshold set in config.py (RECOGNITION_CONFIDENCE_THRESHOLD)
        """
        if self.recognizer is None:
            return "Unknown", 0.0

        # Preprocess
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        resized = cv2.resize(gray, FACE_SIZE)
        equalized = cv2.equalizeHist(resized)

        try:
            label_id, distance = self.recognizer.predict(equalized)
        except Exception as e:
            log.debug(f"Predict failed: {e}")
            return "Unknown", 0.0

        # Convert distance to readable confidence
        # LBPH distance ~0-100: 0=perfect match, 100=very different
        # Anything > 80 is usually a wrong person
        confidence_pct = max(0.0, 100.0 - distance)
        name = self.label_map.get(label_id, "Unknown")

        return name, confidence_pct

    def reload(self):
        """Reload the model from disk (useful after retraining)."""
        self._load_model()
