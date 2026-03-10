"""
tests/test_attendance.py
=========================
Unit tests for the attendance system components.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.attendance_log import AttendanceLog
from utils.emotion_detector import EmotionDetector


# ─── AttendanceLog tests ──────────────────────────────────────────────────────

class TestAttendanceLog:

    def setup_method(self):
        self.log = AttendanceLog()

    def test_mark_present(self):
        self.log.mark_present("Alice_Smith", emotion="happy", confidence=85.0)
        assert self.log.is_present("Alice_Smith")
        assert self.log.records["Alice_Smith"]["status"] == "present"
        assert self.log.records["Alice_Smith"]["emotion"] == "happy"

    def test_no_double_entry(self):
        """Same student should only be logged once even if seen multiple frames."""
        self.log.mark_present("Bob_Jones", emotion="neutral")
        self.log.mark_present("Bob_Jones", emotion="happy")  # second sighting
        assert self.log.present_count() == 1
        # Should keep the FIRST entry
        assert self.log.records["Bob_Jones"]["emotion"] == "neutral"

    def test_mark_absent(self):
        self.log.mark_absent("Charlie_Brown")
        assert not self.log.is_present("Charlie_Brown")
        assert self.log.records["Charlie_Brown"]["status"] == "absent"
        assert self.log.records["Charlie_Brown"]["time_in"] is None

    def test_present_count(self):
        self.log.mark_present("Alice_Smith", emotion="happy")
        self.log.mark_present("Bob_Jones", emotion="sad")
        self.log.mark_absent("Charlie_Brown")
        assert self.log.present_count() == 2

    def test_summary(self):
        self.log.mark_present("Alice_Smith", emotion="happy")
        self.log.mark_present("Bob_Jones", emotion="neutral")
        self.log.mark_absent("Charlie_Brown")

        summary = self.log.get_summary()
        assert summary["total"] == 3
        assert summary["present"] == 2
        assert summary["absent"] == 1
        assert abs(summary["rate"] - 66.67) < 0.1

    def test_to_list(self):
        self.log.mark_present("Alice_Smith", emotion="happy", confidence=88.5)
        self.log.mark_absent("Bob_Jones")

        rows = self.log.to_list()
        assert len(rows) == 2
        present_row = next(r for r in rows if "Alice" in r["Student Name"])
        assert present_row["Status"] == "Present"
        assert present_row["Emotion"] == "Happy"

    def test_clear(self):
        self.log.mark_present("Alice_Smith", emotion="happy")
        self.log.clear()
        assert self.log.present_count() == 0
        assert len(self.log.records) == 0


# ─── EmotionDetector tests ────────────────────────────────────────────────────

class TestEmotionDetector:

    def setup_method(self):
        self.detector = EmotionDetector()

    def test_returns_string(self):
        """Should always return a string even on garbage input."""
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.detector.detect(dummy)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_none_input(self):
        result = self.detector.detect(None)
        assert result == "neutral"

    def test_empty_input(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = self.detector.detect(empty)
        assert result == "neutral"

    def test_known_emotions(self):
        """Result should be one of the expected emotion strings."""
        valid_emotions = {
            "happy", "sad", "angry", "surprised", "fear",
            "disgust", "neutral", "surprise"
        }
        dummy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.detector.detect(dummy)
        assert result.lower() in valid_emotions


# ─── Report generation tests ──────────────────────────────────────────────────

class TestReportGenerator:

    def test_csv_export(self, tmp_path):
        from utils.report_generator import _export_csv

        records = {
            "Alice_Smith": {
                "status": "present",
                "time_in": datetime.now(),
                "emotion": "happy",
                "confidence": 88.0,
            },
            "Bob_Jones": {
                "status": "absent",
                "time_in": None,
                "emotion": "N/A",
                "confidence": 0.0,
            },
        }

        csv_path = str(tmp_path / "test_attendance.csv")
        start = datetime.now()
        end = start + timedelta(minutes=30)
        _export_csv(records, start, end, csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            content = f.read()
        assert "Alice" in content
        assert "Present" in content
        assert "Bob" in content

    def test_excel_export(self, tmp_path):
        from utils.report_generator import generate_excel_report

        records = {
            "Alice_Smith": {
                "status": "present",
                "time_in": datetime.now(),
                "emotion": "happy",
                "confidence": 88.0,
            },
        }

        excel_path = str(tmp_path / "test_attendance.xlsx")
        csv_path   = str(tmp_path / "test_attendance.csv")
        start = datetime.now()
        end = start + timedelta(minutes=30)

        generate_excel_report(records, start, end, excel_path, csv_path)

        assert os.path.exists(excel_path)
        assert os.path.getsize(excel_path) > 0


# ─── Time window tests ────────────────────────────────────────────────────────

class TestTimeWindow:

    def test_is_within_window(self):
        from datetime import time as dtime
        from config import ATTENDANCE_START, ATTENDANCE_END

        # Just verify config is sane
        assert ATTENDANCE_START < ATTENDANCE_END
        assert ATTENDANCE_START == dtime(9, 30)
        assert ATTENDANCE_END == dtime(10, 0)


if __name__ == "__main__":
    print("Running manual checks...")

    log = AttendanceLog()
    log.mark_present("Alice", emotion="happy")
    log.mark_present("Alice", emotion="sad")  # duplicate — should be ignored
    log.mark_absent("Bob")

    assert log.present_count() == 1
    assert log.records["Alice"]["emotion"] == "happy"  # first entry kept
    print("✓ AttendanceLog OK")

    det = EmotionDetector()
    result = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert isinstance(result, str)
    print("✓ EmotionDetector OK")

    print("\nAll manual checks passed!")
