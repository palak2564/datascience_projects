"""
tests/test_sign_language.py
============================
Unit tests for the sign language detection system.

Run: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import PredictionLogger
from config import ACTIVE_START, ACTIVE_END, ALL_LABELS, KNOWN_WORDS, ALPHABET, LANDMARK_FEATURES


# ─── Config tests ─────────────────────────────────────────────────────────────

class TestConfig:

    def test_time_window(self):
        from datetime import time as dtime
        assert ACTIVE_START == dtime(18, 0), "Start should be 6 PM"
        assert ACTIVE_END   == dtime(22, 0), "End should be 10 PM"
        assert ACTIVE_START < ACTIVE_END

    def test_labels_count(self):
        assert len(KNOWN_WORDS) == 10
        assert len(ALPHABET) == 26
        assert len(ALL_LABELS) == 36

    def test_known_words(self):
        expected = ["Hello", "ThankYou", "Yes", "No", "Please",
                    "Sorry", "Help", "Eat", "Water", "More"]
        assert KNOWN_WORDS == expected

    def test_landmark_features(self):
        assert LANDMARK_FEATURES == 63  # 21 × 3


# ─── Logger tests ────────────────────────────────────────────────────────────

class TestLogger:

    def test_log_entry(self):
        logger = PredictionLogger()
        logger.log("Hello", 0.92, source="test.jpg", mode="image")
        assert len(logger.entries) == 1
        assert logger.entries[0]["prediction"] == "Hello"
        assert logger.entries[0]["confidence"] == 0.92

    def test_multiple_entries(self):
        logger = PredictionLogger()
        for pred in ["A", "B", "Hello", "Yes"]:
            logger.log(pred, 0.80)
        assert len(logger.entries) == 4

    def test_summary(self):
        logger = PredictionLogger()
        logger.log("Hello", 0.9)
        logger.log("Hello", 0.85)
        logger.log("Yes", 0.75)
        s = logger.get_summary()
        assert s["total"] == 3
        assert s["top_signs"][0][0] == "Hello"
        assert s["top_signs"][0][1] == 2

    def test_csv_export(self, tmp_path):
        logger = PredictionLogger()
        logger.log("Water", 0.88, source="test.jpg", mode="image")
        path = str(tmp_path / "test.csv")
        logger.export_csv(path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "Water" in content

    def test_empty_csv(self, tmp_path):
        logger = PredictionLogger()
        path = str(tmp_path / "empty.csv")
        logger.export_csv(path)
        assert os.path.exists(path)


# ─── Detector tests ───────────────────────────────────────────────────────────

class TestDetector:

    def setup_method(self):
        from utils.detector import SignLanguageDetector
        self.detector = SignLanguageDetector()

    def test_initialises(self):
        assert self.detector is not None

    def test_demo_mode_prediction(self, tmp_path):
        """In demo mode, prediction should return a valid label."""
        import cv2
        # Create dummy image
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        path = str(tmp_path / "dummy.jpg")
        cv2.imwrite(path, dummy)

        result = self.detector.predict_image(path)
        assert "prediction" in result
        assert "confidence" in result
        assert "hands_detected" in result
        assert "annotated" in result

    def test_annotated_is_ndarray(self, tmp_path):
        import cv2
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        path = str(tmp_path / "dummy2.jpg")
        cv2.imwrite(path, dummy)

        result = self.detector.predict_image(path)
        if result["annotated"] is not None:
            assert isinstance(result["annotated"], np.ndarray)

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            self.detector.predict_image("nonexistent_file.jpg")

    def test_draw_landmarks_returns_frame(self):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.draw_landmarks(dummy)
        assert result.shape == dummy.shape


# ─── Time window tests ────────────────────────────────────────────────────────

class TestTimeWindow:

    def test_window_within_day(self):
        from datetime import time as dtime
        assert ACTIVE_START.hour == 18
        assert ACTIVE_END.hour == 22
        duration_hours = ACTIVE_END.hour - ACTIVE_START.hour
        assert duration_hours == 4, f"Window should be 4 hours, got {duration_hours}"

    def test_example_times(self):
        from datetime import time as dtime
        inside_times  = [dtime(18, 0), dtime(19, 30), dtime(21, 59), dtime(20, 0)]
        outside_times = [dtime(0, 0), dtime(9, 0), dtime(17, 59), dtime(22, 1)]

        for t in inside_times:
            assert ACTIVE_START <= t <= ACTIVE_END, f"{t} should be inside window"

        for t in outside_times:
            assert not (ACTIVE_START <= t <= ACTIVE_END), f"{t} should be outside window"


if __name__ == "__main__":
    print("Running manual checks...")

    # Config
    assert len(KNOWN_WORDS) == 10, "Should have 10 words"
    assert len(ALL_LABELS) == 36, "Should have 36 total labels"
    print("✓ Config OK")

    # Logger
    log = PredictionLogger()
    log.log("Hello", 0.9)
    assert log.get_summary()["total"] == 1
    print("✓ Logger OK")

    # Time window
    from datetime import time as dtime
    assert ACTIVE_START == dtime(18, 0)
    assert ACTIVE_END == dtime(22, 0)
    print("✓ Time window OK")

    print("\nAll manual checks passed!")
