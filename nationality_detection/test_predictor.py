"""
tests/test_predictor.py
========================
Unit tests for the nationality detection system.

Run: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.predictor import closest_colour_name, DressColourDetector, NationalityPredictor
from utils.logger import PredictionLogger
from config import TASK_RULES, NATIONALITY_MAP, DRESS_COLOUR_PALETTE


# ─── Colour naming tests ──────────────────────────────────────────────────────

class TestColourNaming:

    def test_pure_red(self):
        result = closest_colour_name((255, 0, 0))
        assert result in ("Red", "Dark Red"), f"Got: {result}"

    def test_pure_blue(self):
        result = closest_colour_name((0, 0, 255))
        assert "blue" in result.lower() or "Blue" in result, f"Got: {result}"

    def test_pure_white(self):
        assert closest_colour_name((255, 255, 255)) == "White"

    def test_pure_black(self):
        assert closest_colour_name((0, 0, 0)) == "Black"

    def test_returns_string(self):
        result = closest_colour_name((100, 200, 50))
        assert isinstance(result, str) and len(result) > 0

    def test_all_palette_colours_findable(self):
        """Every colour in the palette should map to itself (distance 0)."""
        for name, rgb in DRESS_COLOUR_PALETTE.items():
            result = closest_colour_name(rgb)
            assert result == name, f"Expected {name}, got {result}"


# ─── Dress colour detector tests ─────────────────────────────────────────────

class TestDressColourDetector:

    def setup_method(self):
        self.detector = DressColourDetector()

    def test_solid_red_image(self):
        # Create a solid red BGR image
        red_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255  # red channel in BGR
        result = self.detector.detect(red_bgr)
        assert isinstance(result, str)
        assert "red" in result.lower() or "Red" in result

    def test_solid_blue_image(self):
        blue_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255  # blue channel in BGR
        result = self.detector.detect(blue_bgr)
        assert isinstance(result, str)
        assert "blue" in result.lower()

    def test_none_input(self):
        result = self.detector.detect(None)
        assert result == "Unknown"

    def test_with_face_bbox(self):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        # Face at top, clothing below
        result = self.detector.detect(img, face_bbox=(100, 20, 100, 80))
        assert isinstance(result, str)


# ─── Task rules tests ─────────────────────────────────────────────────────────

class TestTaskRules:

    def test_indian_rules(self):
        rules = TASK_RULES["indian"]
        assert rules["emotion"] is True
        assert rules["age"] is True
        assert rules["dress_colour"] is True

    def test_american_rules(self):
        rules = TASK_RULES["american"]
        assert rules["emotion"] is True
        assert rules["age"] is True
        assert rules["dress_colour"] is False

    def test_african_rules(self):
        rules = TASK_RULES["african"]
        assert rules["emotion"] is True
        assert rules["age"] is False
        assert rules["dress_colour"] is True

    def test_other_rules(self):
        rules = TASK_RULES["other"]
        assert rules["emotion"] is True
        assert rules["age"] is False
        assert rules["dress_colour"] is False

    def test_nationality_map_has_key_entries(self):
        assert "indian" in NATIONALITY_MAP
        assert "black" in NATIONALITY_MAP
        assert "white" in NATIONALITY_MAP
        assert NATIONALITY_MAP["indian"] == "Indian"
        assert NATIONALITY_MAP["black"] == "African"


# ─── Predictor task group mapping ────────────────────────────────────────────

class TestPredictor:

    def setup_method(self):
        self.predictor = NationalityPredictor()

    def test_task_group_indian(self):
        assert self.predictor._get_task_group("indian") == "indian"

    def test_task_group_african(self):
        assert self.predictor._get_task_group("black") == "african"

    def test_task_group_american(self):
        assert self.predictor._get_task_group("white") == "american"

    def test_task_group_other(self):
        assert self.predictor._get_task_group("asian") == "other"
        assert self.predictor._get_task_group("middle eastern") == "other"
        assert self.predictor._get_task_group("latino hispanic") == "other"

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            self.predictor.analyse("does_not_exist.jpg")

    def test_analyse_returns_required_keys(self, tmp_path):
        import cv2
        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        path = str(tmp_path / "test.jpg")
        cv2.imwrite(path, dummy)
        result = self.predictor.analyse(path)
        for key in ["nationality", "emotion", "task_group", "predictions_made",
                    "face_found", "annotated_image"]:
            assert key in result, f"Missing key: {key}"


# ─── Logger tests ─────────────────────────────────────────────────────────────

class TestLogger:

    def test_log_and_export(self, tmp_path):
        logger = PredictionLogger()
        logger.log({
            "nationality": "Indian", "task_group": "indian",
            "emotion": "Happy", "age": 28,
            "dress_colour": "Blue", "confidence": 82.0,
            "face_found": True,
        }, source="test.jpg")

        path = str(tmp_path / "log.csv")
        logger.export_csv(path)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()
        assert "Indian" in content
        assert "Happy" in content

    def test_export_single_txt(self, tmp_path):
        logger = PredictionLogger()
        result = {
            "nationality": "African", "task_group": "african",
            "emotion": "Neutral", "age": None,
            "dress_colour": "Red", "confidence": 74.0,
            "face_found": True,
        }
        path = str(tmp_path / "result.txt")
        logger.export_single(result, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "African" in content
        assert "Red" in content


if __name__ == "__main__":
    print("Running manual checks...")

    # Colour test
    assert closest_colour_name((255, 0, 0)) in ("Red", "Dark Red")
    print("✓ Red colour naming OK")

    assert closest_colour_name((0, 0, 0)) == "Black"
    print("✓ Black colour naming OK")

    # Task rules
    assert TASK_RULES["indian"]["age"] is True
    assert TASK_RULES["african"]["age"] is False
    assert TASK_RULES["american"]["dress_colour"] is False
    print("✓ Task rules OK")

    print("\nAll manual checks passed!")
