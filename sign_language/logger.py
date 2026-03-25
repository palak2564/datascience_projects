"""
utils/logger.py
================
Logs all predictions to memory + exports CSV.
"""

import csv
import os
from datetime import datetime
from typing import List


class PredictionLogger:
    def __init__(self):
        self.entries: List[dict] = []

    def log(self, prediction: str, confidence: float,
            source: str = "unknown", mode: str = "image"):
        self.entries.append({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": prediction or "—",
            "confidence": round(confidence, 4),
            "source":     os.path.basename(str(source)),
            "mode":       mode,
        })

    def export_csv(self, path: str):
        if not self.entries:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=[
                    "timestamp", "prediction", "confidence", "source", "mode"
                ]).writeheader()
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.entries[0].keys()))
            writer.writeheader()
            writer.writerows(self.entries)

    def get_summary(self) -> dict:
        from collections import Counter
        total = len(self.entries)
        counts = Counter(e["prediction"] for e in self.entries)
        return {"total": total, "top_signs": counts.most_common(5)}
