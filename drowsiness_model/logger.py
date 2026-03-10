"""
utils/logger.py
================
Logs all detection events for later export.

Each entry: timestamp, source, total_people, sleeping_count, per-person data.
"""

import csv
import time
from datetime import datetime
from typing import List


class DrowsinessLogger:
    def __init__(self):
        self.entries = []

    def log(self, people: List[dict], source: str = "unknown"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for p in people:
            self.entries.append({
                "timestamp": ts,
                "source": str(source),
                "person_id": p.get("id", 0),
                "state": p.get("state", "unknown"),
                "ear": p.get("ear", ""),
                "age": p.get("age", ""),
            })

    def export_csv(self, path: str):
        if not self.entries:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=[
                    "timestamp", "source", "person_id", "state", "ear", "age"
                ]).writeheader()
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.entries[0].keys()))
            writer.writeheader()
            writer.writerows(self.entries)

    def get_summary(self) -> dict:
        total = len(self.entries)
        sleeping = sum(1 for e in self.entries if e["state"] == "sleeping")
        return {"total": total, "sleeping": sleeping, "awake": total - sleeping}
