"""
tests/test_detector.py
=======================
Basic unit tests for the detector logic.

Not extensive — but enough to verify the carnivore classification
and bounding box drawing work as expected.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.detector import AnimalDetector, Detection, CARNIVORES, ANIMAL_CLASSES


class TestCarnivoreClassification:
    """Tests for the diet classification logic."""

    def test_cat_is_carnivore(self):
        assert "cat" in CARNIVORES

    def test_cow_not_carnivore(self):
        assert "cow" not in CARNIVORES

    def test_zebra_not_carnivore(self):
        assert "zebra" not in CARNIVORES

    def test_lion_is_carnivore(self):
        assert "lion" in CARNIVORES

    def test_eagle_is_carnivore(self):
        assert "eagle" in CARNIVORES


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_detection_to_dict(self):
        d = Detection(
            label="lion",
            confidence=0.92,
            bbox=(10, 20, 100, 200),
            is_carnivore=True,
            is_omnivore=False,
            diet_tag="carnivore",
        )
        result = d.to_dict()
        assert result["label"] == "lion"
        assert result["is_carnivore"] is True
        assert result["diet_tag"] == "carnivore"
        assert result["confidence"] == 0.92

    def test_herbivore_detection(self):
        d = Detection(
            label="zebra",
            confidence=0.88,
            bbox=(50, 50, 300, 300),
            is_carnivore=False,
            is_omnivore=False,
            diet_tag="herbivore",
        )
        assert d.diet_tag == "herbivore"
        assert not d.is_carnivore


class TestDetectorDemoMode:
    """
    Tests the detector in demo mode (no real model needed).
    This works even without ultralytics installed.
    """

    def setup_method(self):
        # Force demo mode by patching model to None
        self.detector = AnimalDetector.__new__(AnimalDetector)
        self.detector.model = None
        self.detector.model_size = "yolov8n"

    def test_demo_detections_returns_list(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector._demo_detections(dummy_frame)
        assert isinstance(detections, list)
        assert len(detections) > 0

    def test_demo_has_carnivore(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector._demo_detections(dummy_frame)
        carnivores = [d for d in detections if d.is_carnivore]
        assert len(carnivores) > 0  # demo always has a lion

    def test_draw_does_not_crash(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector._demo_detections(dummy_frame)
        result = self.detector._draw_detections(dummy_frame.copy(), detections)
        assert result is not None
        assert result.shape == dummy_frame.shape


class TestLogger:
    """Tests for the detection logger."""

    def test_log_and_summary(self):
        from utils.logger import DetectionLogger
        logger = DetectionLogger()
        logger.log([
            {"label": "lion", "confidence": 0.9, "diet_tag": "carnivore",
             "bbox": (0, 0, 100, 100), "is_carnivore": True},
            {"label": "zebra", "confidence": 0.85, "diet_tag": "herbivore",
             "bbox": (200, 200, 400, 400), "is_carnivore": False},
        ], source="test.jpg")

        summary = logger.get_summary()
        assert summary["total"] == 2
        assert summary["carnivores"] == 1
        assert summary["herbivores"] == 1
        assert "lion" in summary["species"]

    def test_export_csv(self, tmp_path):
        from utils.logger import DetectionLogger
        logger = DetectionLogger()
        logger.log([
            {"label": "cat", "confidence": 0.7, "diet_tag": "carnivore",
             "bbox": (0, 0, 50, 50), "is_carnivore": True},
        ], source="test.jpg")

        csv_path = str(tmp_path / "test_export.csv")
        logger.export_csv(csv_path)
        assert os.path.exists(csv_path)


if __name__ == "__main__":
    # Quick manual test without pytest
    print("Running basic checks...")

    # Carnivore check
    assert "cat" in CARNIVORES, "Cat should be carnivore"
    assert "cow" not in CARNIVORES, "Cow should NOT be carnivore"
    print("✓ Carnivore classification OK")

    # Detection dict
    d = Detection("lion", 0.9, (0, 0, 100, 100), True, False, "carnivore")
    dd = d.to_dict()
    assert dd["label"] == "lion"
    print("✓ Detection dataclass OK")

    print("\nAll checks passed! Run 'pytest tests/' for full suite.")
