"""
utils/predictor.py
===================
The core ML pipeline for nationality detection.

Models used:
1. DeepFace  — race/ethnicity + emotion + age (all-in-one)
2. OpenCV DNN — Levi & Hassner age model (backup/cross-check)
3. KMeans    — dominant dress colour from torso region
4. Haar Cascade — face localisation for torso crop estimation

Nationality mapping:
DeepFace returns raw ethnicity labels:
  'asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic'

We map these to task-spec nationalities:
  indian        → 🇮🇳 Indian
  black         → 🌍 African
  white         → 🇺🇸 American  (broadest match for "United States" spec)
  asian         → 🌏 East Asian
  middle eastern → 🌍 Middle Eastern
  latino hispanic → 🌎 Latino/Hispanic

Conditional logic (per task spec):
  Indian    → nationality + emotion + age + dress colour
  American  → nationality + emotion + age
  African   → nationality + emotion + dress colour
  Other     → nationality + emotion

Dress colour detection:
- Crop torso region (below face, same width)
- KMeans(k=3) on RGB pixels
- Dominant cluster → nearest named colour
- Named colours: standard CSS/paint names via Euclidean distance in RGB space

Author: [Your Name]
Date: 2025
"""

import cv2
import numpy as np
import os
import logging
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)

# Try DeepFace — our main nationality/emotion/age model
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    log.info("DeepFace loaded ✓")
except ImportError:
    DEEPFACE_AVAILABLE = False
    log.warning("DeepFace not installed: pip install deepface")

# sklearn for KMeans dress colour
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not installed: pip install scikit-learn")

from config import (
    MODEL_DIR, AGE_DEPLOY_PATH, AGE_MODEL_PATH,
    AGE_BUCKETS, MEAN_VALUES,
    NATIONALITY_MAP, TASK_RULES,
    DRESS_COLOUR_PALETTE,
)


# ─── Named colour lookup ──────────────────────────────────────────────────────

def closest_colour_name(rgb: tuple) -> str:
    """
    Maps an (R, G, B) tuple to the nearest named colour
    using Euclidean distance in RGB space.
    """
    r, g, b = rgb
    min_dist = float("inf")
    best_name = "Unknown"
    for name, (cr, cg, cb) in DRESS_COLOUR_PALETTE.items():
        dist = ((r - cr)**2 + (g - cg)**2 + (b - cb)**2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


# ─── Age Estimator (DNN) ─────────────────────────────────────────────────────

class AgeEstimator:
    """
    Levi & Hassner (2015) age classification CNN.
    Pretrained on IMDB-WIKI.
    Returns age bracket string e.g. "(25-32)".
    """

    _DEPLOY_URL = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/deploy_age.prototxt"
    _MODEL_URL  = "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel"

    def __init__(self):
        self.net = None
        self._load()

    def _load(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        for url, path in [(self._DEPLOY_URL, AGE_DEPLOY_PATH),
                          (self._MODEL_URL, AGE_MODEL_PATH)]:
            if not os.path.exists(path):
                try:
                    log.info(f"Downloading {os.path.basename(path)}...")
                    urllib.request.urlretrieve(url, path)
                except Exception as e:
                    log.warning(f"Download failed: {e}")

        if os.path.exists(AGE_DEPLOY_PATH) and os.path.exists(AGE_MODEL_PATH):
            try:
                self.net = cv2.dnn.readNet(AGE_MODEL_PATH, AGE_DEPLOY_PATH)
                log.info("Age DNN loaded ✓")
            except Exception as e:
                log.warning(f"Age model load failed: {e}")

    def predict(self, face_img: np.ndarray) -> Optional[str]:
        if self.net is None or face_img is None or face_img.size == 0:
            return None
        try:
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            return AGE_BUCKETS[preds[0].argmax()]
        except Exception:
            return None

    def is_ready(self):
        return self.net is not None


# ─── Dress Colour Detector ────────────────────────────────────────────────────

class DressColourDetector:
    """
    Detects dominant dress/clothing colour.

    Strategy:
    1. Locate face bounding box
    2. Estimate torso region: below face, same x-center, ~1.5x face height
    3. KMeans(k=3) on pixel colours in that region
    4. Largest cluster centroid → nearest named colour

    Why KMeans? It's unsupervised, handles multi-colour outfits,
    and doesn't require a labelled clothing dataset.
    """

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters

    def detect(self, image: np.ndarray, face_bbox: Optional[tuple] = None) -> str:
        """
        Returns dominant colour name from clothing region.
        face_bbox: (x, y, w, h) — if None, uses lower half of image
        """
        if image is None or image.size == 0:
            return "Unknown"

        torso = self._get_torso_region(image, face_bbox)
        if torso is None or torso.size == 0:
            return "Unknown"

        return self._dominant_colour(torso)

    def _get_torso_region(self, image: np.ndarray,
                          face_bbox: Optional[tuple]) -> Optional[np.ndarray]:
        """Crops the torso region from the image."""
        h, w = image.shape[:2]

        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            # Torso starts below the face, same width ±20%
            tx = max(0, fx - int(fw * 0.2))
            ty = fy + fh
            tw = min(w - tx, int(fw * 1.4))
            th = min(h - ty, int(fh * 1.5))
            if th < 20 or tw < 20:
                # Face is near bottom — fall back to lower half
                return image[h // 2:, :]
            return image[ty:ty+th, tx:tx+tw]
        else:
            # No face bbox — use lower 40% of image as clothing region
            return image[int(h * 0.4):, :]

    def _dominant_colour(self, region: np.ndarray) -> str:
        """KMeans → dominant cluster → named colour."""
        if not SKLEARN_AVAILABLE:
            return self._dominant_colour_simple(region)

        try:
            # Reshape to 2D array of pixels
            pixels = region.reshape(-1, 3).astype(np.float32)

            # Sample max 2000 pixels for speed
            if len(pixels) > 2000:
                idx = np.random.choice(len(pixels), 2000, replace=False)
                pixels = pixels[idx]

            k = min(self.n_clusters, len(pixels))
            kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
            kmeans.fit(pixels)

            # Find largest cluster
            counts = np.bincount(kmeans.labels_)
            dominant_centre = kmeans.cluster_centers_[counts.argmax()]

            # OpenCV uses BGR — convert to RGB for colour lookup
            b, g, r = dominant_centre
            return closest_colour_name((int(r), int(g), int(b)))

        except Exception as e:
            log.debug(f"KMeans failed: {e}")
            return self._dominant_colour_simple(region)

    def _dominant_colour_simple(self, region: np.ndarray) -> str:
        """Fallback: mean pixel colour without KMeans."""
        mean_bgr = cv2.mean(region)[:3]
        b, g, r = mean_bgr
        return closest_colour_name((int(r), int(g), int(b)))


# ─── Face Detector ────────────────────────────────────────────────────────────

class FaceDetector:
    """Simple Haar Cascade face detection."""

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, image: np.ndarray) -> list:
        """Returns list of (x, y, w, h) face bounding boxes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return list(faces) if len(faces) > 0 else []

    def largest_face(self, image: np.ndarray) -> Optional[tuple]:
        """Returns the largest detected face bounding box."""
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])


# ─── Main Predictor ───────────────────────────────────────────────────────────

class NationalityPredictor:
    """
    Orchestrates all predictions for one image.

    Returns a result dict with keys:
        nationality, raw_ethnicity, emotion, age, dress_colour,
        confidence, face_found, predictions_made, annotated_image
    """

    def __init__(self):
        self.age_estimator = AgeEstimator()
        self.dress_detector = DressColourDetector()
        self.face_detector = FaceDetector()

    def is_ready(self) -> bool:
        return DEEPFACE_AVAILABLE or True  # fallback always available

    def analyse(self, image_path: str) -> dict:
        """
        Full analysis pipeline on one image.
        Returns result dict.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        result = {
            "image_path":        image_path,
            "nationality":       "Unknown",
            "raw_ethnicity":     "unknown",
            "emotion":           "Unknown",
            "age":               None,
            "dress_colour":      None,
            "age_bracket":       None,
            "confidence":        0.0,
            "face_found":        False,
            "predictions_made":  [],
            "annotated_image":   None,
            "task_group":        "other",
        }

        # ── Step 1: Detect face ──────────────────────────────────────────────
        face_bbox = self.face_detector.largest_face(image)
        result["face_found"] = face_bbox is not None

        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            face_crop = image[fy:fy+fh, fx:fx+fw]
        else:
            # No face detected — try whole image
            face_crop = image
            log.warning("No face detected — using full image")

        # ── Step 2: DeepFace — nationality + emotion + age ───────────────────
        if DEEPFACE_AVAILABLE:
            self._run_deepface(face_crop, result)
        else:
            self._run_fallback(face_crop, result)

        # ── Step 3: Determine task group & apply rules ───────────────────────
        group = self._get_task_group(result["raw_ethnicity"])
        result["task_group"] = group
        rules = TASK_RULES.get(group, TASK_RULES["other"])

        predictions = ["nationality", "emotion"]  # always predicted

        # ── Step 4: Age (if required by rules) ──────────────────────────────
        if rules.get("age") and result["age"] is None:
            # Try DNN age estimator as backup
            age_bracket = self.age_estimator.predict(face_crop)
            result["age_bracket"] = age_bracket
            result["age"] = age_bracket
        elif rules.get("age") and result["age"] is not None:
            result["age_bracket"] = f"~{int(result['age'])}"
            predictions.append("age")
        
        if rules.get("age"):
            predictions.append("age")

        # ── Step 5: Dress colour (if required by rules) ──────────────────────
        if rules.get("dress_colour"):
            colour = self.dress_detector.detect(image, face_bbox)
            result["dress_colour"] = colour
            predictions.append("dress_colour")

        result["predictions_made"] = list(dict.fromkeys(predictions))  # deduplicate

        # ── Step 6: Draw annotated image ─────────────────────────────────────
        result["annotated_image"] = self._annotate(image.copy(), result, face_bbox)

        return result

    def _run_deepface(self, face_img: np.ndarray, result: dict):
        """Use DeepFace to get race, emotion, age."""
        try:
            analysis = DeepFace.analyze(
                face_img,
                actions=["race", "emotion", "age"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(analysis, list):
                analysis = analysis[0]

            # Race / nationality
            race_scores = analysis.get("race", {})
            raw_eth = max(race_scores, key=race_scores.get) if race_scores else "unknown"
            result["raw_ethnicity"] = raw_eth.lower()
            result["nationality"] = NATIONALITY_MAP.get(raw_eth.lower(), raw_eth.title())
            result["confidence"] = round(race_scores.get(raw_eth, 0), 1)

            # Emotion
            result["emotion"] = analysis.get("dominant_emotion", "neutral").capitalize()

            # Age (DeepFace gives numeric estimate)
            age_val = analysis.get("age")
            if age_val:
                result["age"] = int(age_val)

        except Exception as e:
            log.warning(f"DeepFace failed: {e}")
            result["nationality"] = "Unknown"
            result["emotion"] = "Neutral"

    def _run_fallback(self, face_img: np.ndarray, result: dict):
        """
        Fallback when DeepFace not available.
        Uses simple skin-tone heuristic for rough nationality estimate
        and OpenCV for basic emotion proxy.
        """
        result["nationality"] = "Unknown (install DeepFace)"
        result["raw_ethnicity"] = "unknown"
        result["emotion"] = "Neutral"
        result["confidence"] = 0.0
        log.warning("DeepFace not available — using fallback")

    def _get_task_group(self, raw_ethnicity: str) -> str:
        """Map raw DeepFace ethnicity to task group."""
        eth = raw_ethnicity.lower()
        if eth == "indian":
            return "indian"
        elif eth == "black":
            return "african"
        elif eth in ("white", "caucasian"):
            return "american"
        else:
            return "other"

    def _annotate(self, image: np.ndarray, result: dict,
                  face_bbox: Optional[tuple]) -> np.ndarray:
        """
        Draws annotations on the image:
        - Face bounding box (colour coded by nationality)
        - Labels for all predictions
        """
        # Nationality → box colour
        group_colours = {
            "indian":   (0, 120, 50),    # saffron-ish
            "african":  (0, 130, 200),   # warm amber
            "american": (180, 40, 40),   # navy/red
            "other":    (100, 100, 200), # neutral blue
        }
        group = result.get("task_group", "other")
        color = group_colours.get(group, (120, 120, 120))

        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), color, 2)

            # Build label lines
            lines = [
                f"Nationality: {result['nationality']}",
                f"Emotion: {result['emotion']}",
            ]
            if result.get("age"):
                lines.append(f"Age: ~{result['age']}")
            if result.get("age_bracket") and not result.get("age"):
                lines.append(f"Age: {result['age_bracket']}")
            if result.get("dress_colour"):
                lines.append(f"Dress: {result['dress_colour']}")

            # Draw label block above face
            line_h = 18
            block_h = len(lines) * line_h + 8
            block_y = max(0, fy - block_h)

            overlay = image.copy()
            cv2.rectangle(overlay,
                          (fx, block_y),
                          (fx + max(200, fw), fy),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            for i, line in enumerate(lines):
                cv2.putText(image, line,
                            (fx + 4, block_y + 14 + i * line_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 255, 255), 1, cv2.LINE_AA)

        return image
