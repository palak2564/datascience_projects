"""
utils/detector.py
==================
Sign language detection inference pipeline.

Loads the trained model and runs prediction on:
- Static images (predict_image)
- Single video frames (predict_frame)
- Provides hand landmark overlay (draw_landmarks)

Model loading priority:
1. TF/Keras .h5 model (most accurate)
2. SVM .pkl fallback
3. Demo mode (random predictions — GUI still works)
"""

import cv2
import numpy as np
import os
import pickle
import logging
from typing import Optional

log = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
except ImportError:
    MP_AVAILABLE = False
    log.warning("MediaPipe not installed: pip install mediapipe")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from config import MODEL_DIR, LANDMARK_FEATURES, IMG_SIZE, ALL_LABELS


class SignLanguageDetector:
    """
    Runs sign language recognition on images/frames.

    Attributes:
        model:    Keras NN or sklearn SVM
        scaler:   StandardScaler for landmark features
        label_enc: LabelEncoder to decode predictions
        hands:    MediaPipe Hands instance
    """

    def __init__(self):
        self.model      = None
        self.model_type = None   # "nn" | "svm" | "demo"
        self.scaler     = None
        self.label_enc  = None

        # MediaPipe — shared instance
        self.hands_static = None   # for images
        self.hands_video  = None   # for video (different config)

        self._load_models()
        self._init_mediapipe()

    def _load_models(self):
        """Load saved model files."""
        le_path     = os.path.join(MODEL_DIR, "label_encoder.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        nn_path     = os.path.join(MODEL_DIR, "sign_model.h5")
        svm_path    = os.path.join(MODEL_DIR, "sign_svm.pkl")

        # Load label encoder
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                self.label_enc = pickle.load(f)
            log.info(f"Label encoder loaded: {list(self.label_enc.classes_)}")
        else:
            log.warning("Label encoder not found — run train.py first")

        # Load scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # Try NN first, then SVM
        if TF_AVAILABLE and os.path.exists(nn_path):
            try:
                self.model = tf.keras.models.load_model(nn_path)
                self.model_type = "nn"
                log.info("NN model loaded ✓")
            except Exception as e:
                log.warning(f"NN load failed: {e}")

        if self.model is None and os.path.exists(svm_path):
            with open(svm_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_type = "svm"
            log.info("SVM model loaded ✓")

        if self.model is None:
            self.model_type = "demo"
            log.warning("No trained model found — running in DEMO mode")
            log.warning("Run: python train.py  to train the model")

    def _init_mediapipe(self):
        """Initialise MediaPipe Hands instances."""
        if not MP_AVAILABLE:
            return
        self.hands_static = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands_video = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def is_ready(self) -> bool:
        return self.model_type != "demo"

    # ── Landmark extraction ────────────────────────────────────────────────────

    def _extract_landmarks(self, image: np.ndarray,
                           static: bool = True) -> Optional[np.ndarray]:
        """
        Extracts 63 normalised landmark features from an image.
        Returns None if no hand detected.
        """
        if not MP_AVAILABLE:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_inst = self.hands_static if static else self.hands_video
        result = hands_inst.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0]
        coords = []
        for pt in lm.landmark:
            coords.extend([pt.x, pt.y, pt.z])

        # Normalise relative to wrist (landmark 0)
        bx, by, bz = coords[0], coords[1], coords[2]
        normalised = []
        for i in range(0, len(coords), 3):
            normalised.extend([coords[i]-bx, coords[i+1]-by, coords[i+2]-bz])

        return np.array(normalised, dtype=np.float32)

    def _get_hand_landmarks_raw(self, image: np.ndarray, static: bool = True):
        """Returns raw MediaPipe result (for drawing)."""
        if not MP_AVAILABLE:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_inst = self.hands_static if static else self.hands_video
        return hands_inst.process(rgb)

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _predict_features(self, features: np.ndarray) -> tuple:
        """
        Runs inference on extracted landmark features.
        Returns (label, confidence).
        """
        if self.model_type == "demo":
            import random
            lbl = random.choice(ALL_LABELS[:10])
            return lbl, round(random.uniform(0.55, 0.95), 3)

        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)

        if self.model_type == "nn":
            probs = self.model.predict(features, verbose=0)[0]
            idx   = probs.argmax()
            conf  = float(probs[idx])
        elif self.model_type == "svm":
            probs = self.model.predict_proba(features)[0]
            idx   = probs.argmax()
            conf  = float(probs[idx])
        else:
            return "Unknown", 0.0

        label = (self.label_enc.inverse_transform([idx])[0]
                 if self.label_enc else str(idx))
        return label, conf

    def predict_image(self, image_path: str,
                      min_confidence: float = 0.6) -> dict:
        """
        Full pipeline on a static image file.
        Returns result dict with annotation.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read: {image_path}")

        image = cv2.resize(image, (640, 480))
        features = self._extract_landmarks(image, static=True)
        annotated = self._annotate_image(image.copy(), features)

        if features is None:
            return {
                "prediction": None,
                "confidence": 0.0,
                "hands_detected": 0,
                "annotated": annotated,
                "message": "No hand detected in image",
            }

        label, conf = self._predict_features(features)
        final_label = label if conf >= min_confidence else "Uncertain"

        # Draw prediction on image
        self._draw_prediction_label(annotated, final_label, conf)

        return {
            "prediction": final_label,
            "confidence": conf,
            "hands_detected": 1,
            "annotated": annotated,
        }

    def predict_frame(self, frame: np.ndarray,
                      min_confidence: float = 0.6) -> dict:
        """
        Fast prediction on a single webcam frame.
        Does NOT annotate — use draw_landmarks for display.
        """
        features = self._extract_landmarks(frame, static=False)

        if features is None:
            return {"prediction": None, "confidence": 0.0, "hands_detected": 0}

        label, conf = self._predict_features(features)
        final_label = label if conf >= min_confidence else None

        return {
            "prediction": final_label,
            "confidence": conf,
            "hands_detected": 1,
        }

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws hand landmarks + connections on frame.
        Returns annotated frame.
        Used for live webcam preview.
        """
        if not MP_AVAILABLE:
            return frame

        mp_result = self._get_hand_landmarks_raw(frame, static=False)
        if mp_result and mp_result.multi_hand_landmarks:
            for hand_lm in mp_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
        return frame

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _annotate_image(self, image: np.ndarray, features) -> np.ndarray:
        """Draw landmarks on image (static mode)."""
        if not MP_AVAILABLE or features is None:
            return image

        mp_result = self._get_hand_landmarks_raw(image, static=True)
        if mp_result and mp_result.multi_hand_landmarks:
            for hand_lm in mp_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
        return image

    def _draw_prediction_label(self, image: np.ndarray,
                                label: str, confidence: float):
        """Draw large prediction text on image."""
        h, w = image.shape[:2]
        text = f"{label}  {confidence:.0%}"

        # Background strip at bottom
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h-55), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        cv2.putText(image, text, (12, h-16),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1,
                    (0, 229, 255), 2, cv2.LINE_AA)

    def __del__(self):
        """Clean up MediaPipe instances."""
        if self.hands_static:
            self.hands_static.close()
        if self.hands_video:
            self.hands_video.close()
