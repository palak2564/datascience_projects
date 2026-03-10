"""
tests/test_detector.py
=======================
Unit tests for the drowsiness detection system.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.detector import eye_aspect_ratio, DrowsinessDetector
from utils.logger import DrowsinessLogger


# ─── EAR Tests ───────────────────────────────────────────────────────────────

class TestEAR:

    def test_open_eye_high_ear(self):
        """Open eye should have EAR > 0.25."""
        # Simulate 6 landmark points for a wide open eye
        eye = np.array([
            [0, 50],    # P1 - left corner
            [33, 80],   # P2 - upper left
            [67, 80],   # P3 - upper right
            [100, 50],  # P4 - right corner
            [67, 20],   # P5 - lower right
            [33, 20],   # P6 - lower left
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        assert ear > 0.25, f"Open eye EAR should be > 0.25, got {ear:.3f}"

    def test_closed_eye_low_ear(self):
        """Closed eye should have EAR < 0.25."""
        eye = np.array([
            [0,   50],
            [33,  53],  # almost flat top
            [67,  53],
            [100, 50],
            [67,  47],
            [33,  47],
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        assert ear < 0.25, f"Closed eye EAR should be < 0.25, got {ear:.3f}"

    def test_ear_non_negative(self):
        """EAR should never be negative."""
        eye = np.random.rand(6, 2) * 100
        ear = eye_aspect_ratio(eye)
        assert ear >= 0

    def test_ear_zero_width_no_crash(self):
        """Should not crash if all points are at same x (zero horizontal dist)."""
        eye = np.array([[50, 50]] * 6, dtype=float)
        ear = eye_aspect_ratio(eye)
        assert ear == 0.0  # zero denominator returns 0


# ─── DrowsinessDetector Tests ────────────────────────────────────────────────

class TestDrowsinessDetector:

    def setup_method(self):
        self.detector = DrowsinessDetector()

    def test_initializes(self):
        assert self.detector is not None
        assert self.detector.face_cascade is not None

    def test_process_image_returns_dict(self, tmp_path):
        """Create a dummy all-black image and run detection."""
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, img)

        result = self.detector.process_image(img_path)
        assert "people" in result
        assert "annotated" in result
        assert isinstance(result["people"], list)

    def test_annotated_same_shape(self, tmp_path):
        """Annotated frame should have same dimensions as input."""
        import cv2
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test2.jpg")
        cv2.imwrite(img_path, img)

        result = self.detector.process_image(img_path)
        assert result["annotated"].shape == (480, 640, 3)

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            self.detector.process_image("this_file_does_not_exist.jpg")


# ─── Logger Tests ─────────────────────────────────────────────────────────────

class TestLogger:

    def test_log_and_summary(self):
        logger = DrowsinessLogger()
        logger.log([
            {"id": 0, "state": "sleeping", "ear": 0.18, "age": "(25-32)"},
            {"id": 1, "state": "awake",    "ear": 0.31, "age": "(38-43)"},
        ], source="test.jpg")

        s = logger.get_summary()
        assert s["total"] == 2
        assert s["sleeping"] == 1
        assert s["awake"] == 1

    def test_csv_export(self, tmp_path):
        logger = DrowsinessLogger()
        logger.log([
            {"id": 0, "state": "sleeping", "ear": 0.18, "age": "(25-32)"},
        ], source="test.jpg")

        path = str(tmp_path / "log.csv")
        logger.export_csv(path)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()
        assert "sleeping" in content

    def test_empty_export(self, tmp_path):
        """Empty logger should still create a valid CSV."""
        logger = DrowsinessLogger()
        path = str(tmp_path / "empty.csv")
        logger.export_csv(path)
        assert os.path.exists(path)


# ─── Alert Tests ─────────────────────────────────────────────────────────────

class TestAlert:

    def test_alert_import(self):
        """Alert module should import cleanly."""
        from utils.alert import show_drowsiness_alert
        assert callable(show_drowsiness_alert)


if __name__ == "__main__":
    # Quick manual run
    print("Testing EAR function...")

    # Open eye
    eye_open = np.array([[0,50],[33,80],[67,80],[100,50],[67,20],[33,20]], dtype=float)
    ear = eye_aspect_ratio(eye_open)
    assert ear > 0.25, f"Expected > 0.25, got {ear}"
    print(f"  Open eye EAR: {ear:.3f} ✓")

    # Closed eye
    eye_closed = np.array([[0,50],[33,53],[67,53],[100,50],[67,47],[33,47]], dtype=float)
    ear = eye_aspect_ratio(eye_closed)
    assert ear < 0.25, f"Expected < 0.25, got {ear}"
    print(f"  Closed eye EAR: {ear:.3f} ✓")

    print("\nAll manual checks passed!")
