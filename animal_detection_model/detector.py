"""
utils/detector.py
==================
The brain of the operation.

Wraps YOLOv8 (via ultralytics) to:
- Run inference on images and video frames
- Map COCO labels → animal names
- Tag each detected animal as carnivore / herbivore / omnivore
- Draw colored bounding boxes:
    RED   → carnivore
    GREEN → herbivore/other animal
    YELLOW → omnivore

Why YOLOv8? It's fast, accurate, and the pretrained COCO weights already
know 80 classes including a solid chunk of animals. For a student project
this is the right call — no need to train from scratch.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional

# Try importing ultralytics — give a helpful error if missing
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[detector] ultralytics not installed. Run: pip install ultralytics")


# ─── Animal taxonomy ──────────────────────────────────────────────────────────
# These are the COCO class names that are animals.
# I've manually categorized them into dietary groups.
# Could be expanded with a proper taxonomy database, but this works for the task.

ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
    # COCO v2 / extended classes
    "lion", "tiger", "leopard", "cheetah", "wolf", "fox",
    "deer", "rabbit", "squirrel", "monkey", "panda",
    "crocodile", "alligator", "snake", "eagle", "hawk",
}

# Carnivores in the wild — these get RED boxes
CARNIVORES = {
    "cat",        # domestic but instinct is predator
    "bear",       # technically omnivore but dangerous — marking carnivore
    "lion", "tiger", "leopard", "cheetah",
    "wolf", "fox",
    "crocodile", "alligator", "snake",
    "eagle", "hawk",
    "bird",       # many birds are predatory — flagging as potential carnivore
    "dog",        # flagging domestic dog as predator-adjacent (debatable, I know)
}

# Omnivores — YELLOW
OMNIVORES = {"bear", "dog"}  # bear + dog are technically omnivores, shown yellow

# COCO class index → label name (just the animal-relevant ones)
COCO_ANIMAL_IDS = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# Colors — BGR format for OpenCV
COLOR_CARNIVORE  = (0,   30,  220)   # red
COLOR_OMNIVORE   = (0,  200,  255)   # yellow-ish
COLOR_HERBIVORE  = (80,  200,  60)   # green
COLOR_DEFAULT    = (200, 200, 200)   # grey fallback

LABEL_BG_ALPHA = 0.6


# ─── Detection result dataclass ───────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple          # (x1, y1, x2, y2) in pixels
    is_carnivore: bool
    is_omnivore: bool
    diet_tag: str        # "carnivore" | "herbivore" | "omnivore"

    def to_dict(self):
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "is_carnivore": self.is_carnivore,
            "diet_tag": self.diet_tag,
        }


# ─── Main detector class ──────────────────────────────────────────────────────

class AnimalDetector:
    """
    Loads YOLOv8 and runs animal detection.

    If ultralytics isn't installed, falls back to a simple OpenCV
    HOG-based detector that only does "person" — but we swap it for
    a fake demo mode so the GUI still works.
    """

    def __init__(self, model_size: str = "yolov8n"):
        """
        model_size: one of yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        We default to nano (n) — fastest, good enough for demo.
        Swap to 's' or 'm' for better accuracy.
        """
        self.model_size = model_size
        self.model = None
        self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("[detector] Running in DEMO mode (no real detections)")
            return
        try:
            print(f"[detector] Loading {self.model_size}.pt ...")
            self.model = YOLO(f"{self.model_size}.pt")
            print("[detector] Model loaded ✓")
        except Exception as e:
            print(f"[detector] Failed to load model: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_image(self, image_path: str, confidence: float = 0.45,
                     show_labels: bool = True, show_conf: bool = True) -> dict:
        """
        Runs detection on a single image file.
        Returns dict with 'detections' list and 'annotated_frame'.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        detections = self._run_inference(frame, confidence)
        annotated = self._draw_detections(
            frame.copy(), detections,
            show_labels=show_labels, show_conf=show_conf
        )

        return {
            "source": image_path,
            "detections": [d.to_dict() for d in detections],
            "annotated_frame": annotated,
            "timestamp": time.time(),
        }

    def detect_video(
        self,
        source,
        confidence: float = 0.45,
        frame_callback: Optional[Callable] = None,
        stop_flag: Optional[Callable] = None,
        show_labels: bool = True,
        show_conf: bool = True,
    ):
        """
        Processes a video file or webcam stream frame by frame.
        Calls frame_callback(annotated_frame, detections) for each frame.
        stop_flag() should return True when caller wants to stop.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0

        try:
            while True:
                if stop_flag and stop_flag():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Skip every other frame for speed on video files
                frame_count += 1
                if frame_count % 2 != 0 and source != 0:
                    continue

                detections = self._run_inference(frame, confidence)
                annotated = self._draw_detections(
                    frame.copy(), detections,
                    show_labels=show_labels, show_conf=show_conf
                )

                if frame_callback:
                    frame_callback(annotated, [d.to_dict() for d in detections])

                # Tiny sleep to not peg the CPU
                time.sleep(1 / fps * 0.5)

        finally:
            cap.release()

    # ── Internal inference ─────────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray, confidence: float) -> List[Detection]:
        """
        Actually runs the model and returns a list of Detection objects.
        Falls back to demo detections if no model loaded.
        """
        if self.model is None:
            return self._demo_detections(frame)

        results = self.model(frame, conf=confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id].lower()

                # Only keep animals
                if not self._is_animal(label, cls_id):
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_carn = label in CARNIVORES
                is_omni = label in OMNIVORES

                if is_omni:
                    diet = "omnivore"
                elif is_carn:
                    diet = "carnivore"
                else:
                    diet = "herbivore"

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    is_carnivore=is_carn,
                    is_omnivore=is_omni,
                    diet_tag=diet,
                ))

        return detections

    def _is_animal(self, label: str, cls_id: int) -> bool:
        """Check if a YOLO class is an animal we care about."""
        if label in ANIMAL_CLASSES:
            return True
        if cls_id in COCO_ANIMAL_IDS:
            return True
        return False

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw_detections(
        self, frame: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True,
        show_conf: bool = True,
    ) -> np.ndarray:
        """
        Draws bounding boxes and labels on the frame.
        Returns annotated frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Pick color
            if det.is_omnivore:
                color = COLOR_OMNIVORE
            elif det.is_carnivore:
                color = COLOR_CARNIVORE
            else:
                color = COLOR_HERBIVORE

            # Box — thickness 3 for carnivores (more dramatic), 2 for others
            thickness = 3 if det.is_carnivore else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Flash a warning icon for carnivores
            if det.is_carnivore:
                cv2.putText(frame, "⚠", (x2 - 25, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Label
            if show_labels:
                label_text = det.label
                if show_conf:
                    label_text += f" {det.confidence:.0%}"
                label_text += f" [{det.diet_tag}]"

                # Background rect for label
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(
                    frame, label_text,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

        # Summary overlay (top-left)
        self._draw_summary_overlay(frame, detections)

        return frame

    def _draw_summary_overlay(self, frame: np.ndarray, detections: List[Detection]):
        """Draws a small summary box in the corner."""
        carnivore_count = sum(1 for d in detections if d.is_carnivore)
        total = len(detections)

        lines = [
            f"Total animals: {total}",
            f"Carnivores: {carnivore_count}",
            f"Herbivores: {total - carnivore_count}",
        ]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (210, 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            color = (0, 30, 220) if i == 1 and carnivore_count > 0 else (200, 200, 200)
            cv2.putText(frame, line, (14, 25 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ── Demo mode ──────────────────────────────────────────────────────────────

    def _demo_detections(self, frame: np.ndarray) -> List[Detection]:
        """
        Returns fake detections for testing the GUI without a real model.
        Draws random boxes — only used when ultralytics isn't installed.
        """
        h, w = frame.shape[:2]
        return [
            Detection("lion", 0.91, (50, 50, 250, 200), True, False, "carnivore"),
            Detection("zebra", 0.85, (300, 100, 550, 350), False, False, "herbivore"),
            Detection("elephant", 0.78, (w//2, h//3, w-50, h-50), False, False, "herbivore"),
        ]
