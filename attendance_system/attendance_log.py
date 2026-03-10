"""
utils/attendance_log.py
========================
Tracks who's been marked present/absent during a session.

Handles:
- Marking present (with timestamp + emotion)
- Preventing duplicate entries (same student seen multiple times)
- Tracking all registered students (absent if not seen)
- Summary stats
"""

from datetime import datetime
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)


class AttendanceLog:
    """
    In-memory attendance log for one session.

    Structure of self.records:
    {
        "Alice_Smith": {
            "status": "present",         # or "absent"
            "time_in": datetime object,   # None if absent
            "emotion": "happy",           # emotion at recognition moment
            "confidence": 87.3,           # recognition confidence %
        },
        ...
    }
    """

    def __init__(self):
        self.records: Dict[str, dict] = {}
        self._seen_set = set()  # fast lookup for "already marked present"

    def mark_present(
        self,
        student_name: str,
        emotion: str = "neutral",
        timestamp: Optional[datetime] = None,
        confidence: float = 0.0,
    ):
        """
        Marks a student as present.
        If already marked present, this is a no-op (no double entries).
        """
        if student_name in self._seen_set:
            return  # already logged, skip

        self._seen_set.add(student_name)
        self.records[student_name] = {
            "status": "present",
            "time_in": timestamp or datetime.now(),
            "emotion": emotion,
            "confidence": round(confidence, 1),
        }
        log.info(f"PRESENT: {student_name} at {(timestamp or datetime.now()).strftime('%H:%M:%S')} — {emotion}")

    def mark_absent(self, student_name: str):
        """
        Marks a student as absent.
        Called at end of session for everyone not seen.
        """
        if student_name not in self.records:
            self.records[student_name] = {
                "status": "absent",
                "time_in": None,
                "emotion": "N/A",
                "confidence": 0.0,
            }

    def is_present(self, student_name: str) -> bool:
        return student_name in self._seen_set

    def present_count(self) -> int:
        return len(self._seen_set)

    def get_summary(self) -> dict:
        total = len(self.records)
        present = sum(1 for r in self.records.values() if r["status"] == "present")
        absent = total - present
        rate = (present / total * 100) if total > 0 else 0.0
        return {
            "total": total,
            "present": present,
            "absent": absent,
            "rate": rate,
        }

    def clear(self):
        """Reset for a new session."""
        self.records.clear()
        self._seen_set.clear()

    def to_list(self) -> list:
        """Flattens records into a list of dicts for DataFrame/Excel export."""
        rows = []
        for name, data in sorted(self.records.items()):
            rows.append({
                "Student Name": name.replace("_", " "),
                "Status": data["status"].capitalize(),
                "Time In": data["time_in"].strftime("%H:%M:%S") if data["time_in"] else "—",
                "Emotion": data["emotion"].capitalize(),
                "Confidence (%)": data["confidence"] if data["status"] == "present" else "—",
            })
        return rows
