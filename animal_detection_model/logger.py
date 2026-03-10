"""
utils/logger.py
================
Handles logging all detections to memory and CSV export.

Keeps track of:
- What was detected
- When
- From which file
- Diet classification breakdown

This is the kind of thing you'd want for a real system — keeps
an audit trail of all detections the model made.
"""

import csv
import time
from datetime import datetime
from typing import List, Optional


class DetectionLogger:
    """
    Simple in-memory logger that can export to CSV.

    Each 'entry' is:
        timestamp | source_file | label | confidence | diet_tag | bbox
    """

    def __init__(self):
        self.entries = []

    def log(self, detections: List[dict], source: str = "unknown"):
        """
        Logs a batch of detections from a single frame/image.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for d in detections:
            self.entries.append({
                "timestamp": ts,
                "source": str(source),
                "label": d.get("label", "?"),
                "confidence": d.get("confidence", 0.0),
                "diet_tag": d.get("diet_tag", "unknown"),
                "bbox": str(d.get("bbox", "")),
                "is_carnivore": d.get("is_carnivore", False),
            })

    def get_summary(self) -> dict:
        """Returns a summary of all logged detections."""
        if not self.entries:
            return {"total": 0, "carnivores": 0, "herbivores": 0, "species": []}

        carnivores = [e for e in self.entries if e["is_carnivore"]]
        species_seen = list({e["label"] for e in self.entries})

        return {
            "total": len(self.entries),
            "carnivores": len(carnivores),
            "herbivores": len(self.entries) - len(carnivores),
            "species": sorted(species_seen),
        }

    def export_csv(self, path: str):
        """
        Exports all logged detections to a CSV file.
        """
        if not self.entries:
            # Write empty CSV with headers
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp", "source", "label", "confidence",
                    "diet_tag", "bbox", "is_carnivore"
                ])
                writer.writeheader()
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.entries[0].keys()))
            writer.writeheader()
            writer.writerows(self.entries)

        print(f"[logger] Exported {len(self.entries)} rows to {path}")

    def clear(self):
        self.entries.clear()
