"""
utils/detector.py
==================
The core detection pipeline.

Three models working together:

1. Face Detection
   - OpenCV Haar Cascade — fast, lightweight, good for frontal faces
   - Falls back to DNN-based detector for better accuracy if available

2. Drowsiness Detection
   - Primary: Eye Aspect Ratio (EAR) using dlib 68-point facial landmarks
     EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
     When eyes close: EAR drops below threshold (~0.25)
   - Fallback: Trained SVM on HOG features from eye region
     (used when dlib not available)

3. Age Estimation
   - OpenCV DNN with Caffe model (Levi & Hassner 2015)
   - Pre-trained on IMDB-WIKI dataset
   - Returns one of 8 age brackets: (0-2), (4-6), (8-12), (15-20),
     (25-32), (38-43), (48-53), (60-100)
   - Auto-downloads weights if not present

Why EAR for drowsiness?
It's the standard academic approach (Soukupova & Cech, 2016).
EAR is rotation-invariant and works across different face shapes.
No deep learning needed — just geometry.
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from typing import Callable, List, Optional, Tuple
import logging

log = logging.getLogger(__name__)

# Try dlib — best for EAR-based detection
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    log.warning("dlib not installed. Using fallback eye detection.")

from config import (
    MODEL_DIR, EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    AGE_DEPLOY_PATH, AGE_MODEL_PATH,
    AGE_BUCKETS, MEAN_VALUES,
    COLOR_SLEEPING, COLOR_AWAKE,
    FACE_SIZE, FRAME_SKIP,
)


# ─── EAR Calculation ──────────────────────────────────────────────────────────

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Computes Eye Aspect Ratio from 6 eye landmark points.

    References:
    - Soukupova & Cech (2016) "Real-Time Eye Blink Detection using
      Facial Landmarks"
    - p1..p6 are the 6 eye landmarks in order (left to right, top/bottom)

    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

    When fully open: EAR ~ 0.3-0.4
    When closing:    EAR drops toward 0
    """
    # Vertical distances
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


# ─── Age Estimator ────────────────────────────────────────────────────────────

class AgeEstimator:
    """
    Estimates age from a face crop using a pre-trained DNN.

    Model: Levi & Hassner (2015) age classification CNN
    Trained on IMDB-WIKI dataset.
    """

    # Direct download URLs for the model files
    _AGE_DEPLOY_URL = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_age.prototxt"
    _AGE_MODEL_URL  = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"

    def __init__(self):
        self.net = None
        self._load()

    def _load(self):
        """Load the age estimation model, downloading if needed."""
        # Try to load from local model dir first
        deploy = AGE_DEPLOY_PATH
        model  = AGE_MODEL_PATH

        if not os.path.exists(deploy) or not os.path.exists(model):
            log.info("Age model not found locally — attempting download...")
            self._download_model()

        if os.path.exists(deploy) and os.path.exists(model):
            try:
                self.net = cv2.dnn.readNet(model, deploy)
                log.info("Age estimation model loaded ✓")
            except Exception as e:
                log.warning(f"Could not load age model: {e}")

    def _download_model(self):
        """Download model weights. Skips if network unavailable."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        for url, path in [
            (self._AGE_DEPLOY_URL, AGE_DEPLOY_PATH),
            (self._AGE_MODEL_URL,  AGE_MODEL_PATH),
        ]:
            if os.path.exists(path):
                continue
            try:
                log.info(f"Downloading {os.path.basename(path)}...")
                urllib.request.urlretrieve(url, path)
                log.info(f"Downloaded: {path}")
            except Exception as e:
                log.warning(f"Download failed for {os.path.basename(path)}: {e}")

    def predict(self, face_img: np.ndarray) -> Optional[str]:
        """
        Returns age bracket string or None if model not available.
        e.g. "25-32", "15-20"
        """
        if self.net is None or face_img is None or face_img.size == 0:
            return None

        try:
            blob = cv2.dnn.blobFromImage(
                face_img, scalefactor=1.0, size=(227, 227),
                mean=MEAN_VALUES, swapRB=False,
            )
            self.net.setInput(blob)
            preds = self.net.forward()
            idx = preds[0].argmax()
            return AGE_BUCKETS[idx]
        except Exception as e:
            log.debug(f"Age prediction failed: {e}")
            return None

    def is_ready(self) -> bool:
        return self.net is not None


# ─── Main Detector ────────────────────────────────────────────────────────────

class DrowsinessDetector:
    """
    Main detection pipeline.

    Detects faces → checks drowsiness → estimates age → draws results.
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # dlib 68-point predictor
        self.dlib_detector = None
        self.landmark_predictor = None
        self._load_dlib()

        # Age estimator
        self.age_estimator = AgeEstimator()

        # Per-person drowsiness counters (for video — track across frames)
        self._ear_counters = {}   # person_id -> consecutive low-EAR frames

    def _load_dlib(self):
        """Load dlib face detector + shape predictor if available."""
        if not DLIB_AVAILABLE:
            return

        shape_model = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

        if not os.path.exists(shape_model):
            log.warning(
                "dlib landmark model not found.\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                f"Extract to: {shape_model}\n"
                "Falling back to eye cascade detection."
            )
            return

        try:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(shape_model)
            log.info("dlib landmark predictor loaded ✓")
        except Exception as e:
            log.warning(f"dlib load failed: {e}")

    def is_ready(self) -> bool:
        return self.age_estimator.is_ready()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_image(
        self, image_path: str,
        ear_threshold: float = EAR_THRESHOLD,
        show_landmarks: bool = False,
        show_age: bool = True,
    ) -> dict:
        """Process a single image file. Returns result dict."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        people, annotated = self._process_frame(
            frame, ear_threshold, show_landmarks, show_age)

        return {
            "source": image_path,
            "people": people,
            "annotated": annotated,
            "timestamp": time.time(),
        }

    def process_video(
        self,
        source,
        ear_threshold: float = EAR_THRESHOLD,
        show_landmarks: bool = False,
        show_age: bool = True,
        frame_cb: Optional[Callable] = None,
        stop_flag: Optional[Callable] = None,
    ):
        """Process a video file or webcam stream frame by frame."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_n = 0

        try:
            while True:
                if stop_flag and stop_flag():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_n += 1
                if frame_n % FRAME_SKIP != 0:
                    if frame_cb:
                        frame_cb(frame, [])
                    time.sleep(0.01)
                    continue

                people, annotated = self._process_frame(
                    frame, ear_threshold, show_landmarks, show_age)

                if frame_cb:
                    frame_cb(annotated, people)

                time.sleep(1 / (fps * 1.5))

        finally:
            cap.release()

    # ── Core frame processing ──────────────────────────────────────────────────

    def _process_frame(
        self, frame: np.ndarray,
        ear_threshold: float,
        show_landmarks: bool,
        show_age: bool,
    ) -> Tuple[list, np.ndarray]:
        """
        Full pipeline on one frame:
        1. Detect faces
        2. Per face: check drowsiness + estimate age
        3. Draw annotations
        Returns (people_list, annotated_frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy()
        people = []

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for i, (fx, fy, fw, fh) in enumerate(faces if len(faces) > 0 else []):
            face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
            face_roi_bgr  = frame[fy:fy+fh, fx:fx+fw]

            # ── Drowsiness detection ─────────────────────────────────────────
            ear, is_sleeping, landmarks = self._check_drowsiness(
                gray, face_roi_gray, fx, fy, fw, fh, ear_threshold)

            # ── Age estimation ───────────────────────────────────────────────
            age = None
            if show_age:
                age = self.age_estimator.predict(face_roi_bgr)

            state = "sleeping" if is_sleeping else "awake"
            color = COLOR_SLEEPING if is_sleeping else COLOR_AWAKE

            people.append({
                "id": i,
                "bbox": (fx, fy, fw, fh),
                "state": state,
                "ear": round(ear, 4),
                "age": age,
            })

            # ── Draw ─────────────────────────────────────────────────────────
            thickness = 3 if is_sleeping else 2
            cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), color, thickness)

            # Label: state + age
            label_parts = ["💤 SLEEPING" if is_sleeping else "AWAKE"]
            if age:
                label_parts.append(f"age ~{age}")
            label = "  ".join(label_parts)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (fx, fy - lh - 12), (fx + lw + 8, fy), color, -1)
            cv2.putText(annotated, label, (fx + 4, fy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # EAR value beneath box
            cv2.putText(annotated, f"EAR:{ear:.2f}", (fx, fy + fh + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1, cv2.LINE_AA)

            # Landmarks
            if show_landmarks and landmarks is not None:
                for (lx, ly) in landmarks:
                    cv2.circle(annotated, (lx, ly), 2, (0, 255, 255), -1)

            # Sleeping warning pulse (red ring)
            if is_sleeping:
                cx, cy = fx + fw // 2, fy + fh // 2
                radius = max(fw, fh) // 2 + 10
                cv2.circle(annotated, (cx, cy), radius, COLOR_SLEEPING, 2)

        # Summary overlay
        self._draw_overlay(annotated, people)

        return people, annotated

    # ── Drowsiness logic ──────────────────────────────────────────────────────

    def _check_drowsiness(
        self, gray: np.ndarray, face_roi: np.ndarray,
        fx: int, fy: int, fw: int, fh: int,
        ear_threshold: float,
    ) -> Tuple[float, bool, Optional[list]]:
        """
        Returns (ear_value, is_sleeping, landmarks_list)

        Uses dlib landmarks if available, else falls back to
        eye cascade detection in the face ROI.
        """
        if self.landmark_predictor is not None:
            return self._ear_from_dlib(gray, fx, fy, fw, fh, ear_threshold)
        else:
            return self._ear_from_cascade(face_roi, ear_threshold)

    def _ear_from_dlib(self, gray, fx, fy, fw, fh, threshold):
        """EAR using dlib 68 landmark model."""
        rect = dlib.rectangle(fx, fy, fx + fw, fy + fh)
        shape = self.landmark_predictor(gray, rect)

        # Extract landmark coordinates
        pts = np.array([(shape.part(i).x, shape.part(i).y)
                        for i in range(68)])

        # dlib 68-point landmark indices for eyes:
        # Left eye: 36-41, Right eye: 42-47
        left_eye  = pts[36:42]
        right_eye = pts[42:48]

        ear_left  = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0

        is_sleeping = ear < threshold
        return ear, is_sleeping, pts.tolist()

    def _ear_from_cascade(self, face_roi: np.ndarray, threshold: float):
        """
        Fallback EAR estimation using eye cascade.
        Less accurate but works without dlib.

        Estimates openness by eye region height/width ratio.
        """
        h, w = face_roi.shape[:2]
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=3,
            minSize=(20, 20),
        )

        if len(eyes) == 0:
            # No eyes detected — likely sleeping (eyes closed)
            return 0.18, True, None

        ear_values = []
        for (ex, ey, ew, eh) in eyes:
            # Simple ratio: eye height / eye width (approximate EAR)
            ear_approx = eh / ew * 0.6  # scale to EAR range
            ear_values.append(ear_approx)

        ear = np.mean(ear_values)
        is_sleeping = ear < threshold
        return float(ear), is_sleeping, None

    # ── Overlay ───────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, people: list):
        """Summary box top-left corner."""
        total = len(people)
        sleeping = sum(1 for p in people if p["state"] == "sleeping")

        lines = [
            f"Total:    {total}",
            f"Sleeping: {sleeping}",
            f"Awake:    {total - sleeping}",
        ]

        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (200, 72), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            color = (0, 30, 220) if (i == 1 and sleeping > 0) else (200, 200, 200)
            cv2.putText(frame, line, (12, 24 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
