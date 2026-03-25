"""
utils/logger.py
================
Logs all predictions to memory and exports to CSV / TXT.
"""

import csv
import os
from datetime import datetime
from typing import List


class PredictionLogger:
    def __init__(self):
        self.entries: List[dict] = []

    def log(self, result: dict, source: str = "unknown"):
        self.entries.append({
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":       os.path.basename(str(source)),
            "nationality":  result.get("nationality", ""),
            "task_group":   result.get("task_group", ""),
            "emotion":      result.get("emotion", ""),
            "age":          result.get("age", ""),
            "dress_colour": result.get("dress_colour", ""),
            "confidence":   result.get("confidence", ""),
            "face_found":   result.get("face_found", False),
        })

    def export_csv(self, path: str):
        if not self.entries:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.entries[0].keys()))
            writer.writeheader()
            writer.writerows(self.entries)

    def export_single(self, result: dict, path: str):
        """Export a single result to a nicely formatted text file."""
        lines = [
            "=" * 45,
            "  NATIONALITY DETECTION RESULT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 45,
            f"  Nationality  : {result.get('nationality', 'N/A')}",
            f"  Emotion      : {result.get('emotion', 'N/A')}",
        ]
        if result.get("age"):
            lines.append(f"  Age          : ~{result['age']} years")
        if result.get("age_bracket") and not result.get("age"):
            lines.append(f"  Age Range    : {result['age_bracket']}")
        if result.get("dress_colour"):
            lines.append(f"  Dress Colour : {result['dress_colour']}")
        lines += [
            f"  Confidence   : {result.get('confidence', 'N/A')}%",
            f"  Face Found   : {'Yes' if result.get('face_found') else 'No'}",
            "=" * 45,
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
