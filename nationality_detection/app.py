"""
app.py — Nationality Detection System
=======================================
Predicts nationality, emotion, age, and dress colour from a face image.

Internship Task Extension — Nationality Detection Model

Conditional prediction logic (as per task spec):
  🇮🇳 Indian     → nationality + emotion + age + dress colour
  🇺🇸 American   → nationality + emotion + age
  🌍 African     → nationality + emotion + dress colour
  🌐 Other       → nationality + emotion only

GUI Design: editorial / world-map aesthetic
- Warm off-white background (#faf7f2)
- Deep navy text (#1a1f36)
- Country flag accent colours per prediction
- Clean serif headings (feels like a passport / travel document)
- Results shown as a structured "passport card"

Tech stack:
- Nationality: DeepFace (race/ethnicity attribute) + custom mapping
- Emotion: DeepFace (dominant emotion)
- Age: Levi & Hassner OpenCV DNN
- Dress colour: KMeans clustering on torso region (OpenCV + sklearn)
- GUI: tkinter with custom styled widgets

Author: [Your Name]
Date: 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
from datetime import datetime

from utils.predictor import NationalityPredictor
from utils.gui_components import (
    ImagePreviewPanel, ResultsPanel, ActionButton,
    SectionLabel, StatusFooter
)
from utils.logger import PredictionLogger
from config import THEME, APP_TITLE, WINDOW_SIZE


class NationalityApp:
    """
    Main application window.

    Layout:
    ┌──────────────────────────────────────────────────────┐
    │  HEADER — title strip                                 │
    ├────────────────────────┬─────────────────────────────┤
    │                        │  CONTROLS                    │
    │   IMAGE PREVIEW        │  (load, analyse buttons)     │
    │   (left, larger)       ├─────────────────────────────┤
    │                        │  RESULTS / OUTPUT PANEL      │
    │                        │  (passport-card style)       │
    ├────────────────────────┴─────────────────────────────┤
    │  STATUS FOOTER                                        │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=THEME["bg"])
        self.root.minsize(1000, 680)

        self.current_image_path = None
        self.is_processing = False
        self.last_result = None

        self.predictor = NationalityPredictor()
        self.logger = PredictionLogger()

        self._build_ui()
        self._bind_keys()

        self.status_var.set("Ready — upload a photo to begin  🌐")

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        content = tk.Frame(self.root, bg=THEME["bg"])
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        content.columnconfigure(0, weight=5)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        # Left — image preview
        self._build_preview_panel(content)

        # Right — controls + results
        self._build_right_panel(content)

        # Bottom — status
        self.status_var = tk.StringVar(value="")
        self.footer = StatusFooter(self.root, self.status_var)
        self.footer.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=THEME["header_bg"], height=62)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        # Left: title
        left = tk.Frame(hdr, bg=THEME["header_bg"])
        left.pack(side=tk.LEFT, padx=20, pady=8)

        tk.Label(
            left, text="NATIONALITY DETECTION",
            font=("Georgia", 18, "bold"),
            bg=THEME["header_bg"], fg=THEME["header_text"],
        ).pack(anchor="w")

        tk.Label(
            left, text="Face Analysis · Emotion · Age · Dress Colour",
            font=("Georgia", 9, "italic"),
            bg=THEME["header_bg"], fg=THEME["muted"],
        ).pack(anchor="w")

        # Right: model status dot
        self._status_dot = tk.Label(
            hdr, text="● Initialising...",
            font=("Courier New", 9),
            bg=THEME["header_bg"], fg=THEME["warn"],
        )
        self._status_dot.pack(side=tk.RIGHT, padx=20)
        self.root.after(1200, self._update_model_dot)

    def _update_model_dot(self):
        ready = self.predictor.is_ready()
        self._status_dot.configure(
            text="● Models Ready" if ready else "● Partial (fallback mode)",
            fg=THEME["ok"] if ready else THEME["warn"],
        )

    def _build_preview_panel(self, parent):
        frame = tk.Frame(parent, bg=THEME["card"], relief=tk.FLAT, bd=0)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)

        SectionLabel(frame, "INPUT IMAGE").pack(anchor="w", padx=14, pady=(10, 4))

        self.preview = ImagePreviewPanel(frame)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.preview.bind_click(self.load_image)

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=THEME["bg"])
        right.grid(row=0, column=1, sticky="nsew")

        # Controls card
        ctrl_card = tk.Frame(right, bg=THEME["card"], relief=tk.FLAT)
        ctrl_card.pack(fill=tk.X, pady=(0, 8))

        SectionLabel(ctrl_card, "UPLOAD & ANALYSE").pack(
            anchor="w", padx=14, pady=(10, 6))

        btn_frame = tk.Frame(ctrl_card, bg=THEME["card"])
        btn_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        ActionButton(
            btn_frame, "📂  Upload Image",
            command=self.load_image,
            style="primary",
        ).pack(fill=tk.X, pady=3)

        self.analyse_btn = ActionButton(
            btn_frame, "🔍  Analyse",
            command=self.run_analysis,
            style="accent",
        )
        self.analyse_btn.pack(fill=tk.X, pady=3)

        ActionButton(
            btn_frame, "🗑  Clear",
            command=self.clear_all,
            style="ghost",
        ).pack(fill=tk.X, pady=3)

        ActionButton(
            btn_frame, "💾  Save Results",
            command=self.save_results,
            style="ghost",
        ).pack(fill=tk.X, pady=3)

        # Info label
        tk.Label(
            ctrl_card,
            text="Supports: JPG, PNG, BMP, WEBP",
            font=("Courier New", 7),
            bg=THEME["card"], fg=THEME["muted"],
        ).pack(anchor="w", padx=14, pady=(0, 8))

        # Nationality conditional logic info
        info_card = tk.Frame(right, bg=THEME["info_bg"], relief=tk.FLAT)
        info_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            info_card,
            text="PREDICTION RULES",
            font=("Courier New", 8, "bold"),
            bg=THEME["info_bg"], fg=THEME["accent"],
        ).pack(anchor="w", padx=12, pady=(8, 2))

        rules = [
            "🇮🇳  Indian   → emotion + age + dress colour",
            "🇺🇸  American → emotion + age",
            "🌍  African  → emotion + dress colour",
            "🌐  Other    → nationality + emotion",
        ]
        for r in rules:
            tk.Label(
                info_card, text=r,
                font=("Courier New", 8),
                bg=THEME["info_bg"], fg=THEME["text"],
                anchor="w",
            ).pack(fill=tk.X, padx=12, pady=1)
        tk.Label(info_card, text="", bg=THEME["info_bg"]).pack(pady=3)

        # Results card
        results_card = tk.Frame(right, bg=THEME["card"], relief=tk.FLAT)
        results_card.pack(fill=tk.BOTH, expand=True)

        SectionLabel(results_card, "ANALYSIS RESULTS").pack(
            anchor="w", padx=14, pady=(10, 6))

        self.results_panel = ResultsPanel(results_card)
        self.results_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # ── Event Handlers ─────────────────────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        self.current_image_path = path
        self.preview.load(path)
        self.results_panel.clear()
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def run_analysis(self):
        if self.is_processing:
            self.status_var.set("Already processing — please wait...")
            return
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        self.is_processing = True
        self.analyse_btn.set_loading(True)
        self.results_panel.show_loading()
        self.status_var.set("🔍 Analysing face attributes...")

        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        try:
            result = self.predictor.analyse(self.current_image_path)
            self.root.after(0, lambda: self._on_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))
        finally:
            self.root.after(0, lambda: self.analyse_btn.set_loading(False))
            self.is_processing = False

    def _on_result(self, result: dict):
        self.last_result = result
        self.results_panel.display(result)
        self.logger.log(result, source=self.current_image_path)

        nat = result.get("nationality", "Unknown")
        emo = result.get("emotion", "Unknown")
        self.status_var.set(f"✓ Done — {nat} · {emo}")

    def _on_error(self, msg: str):
        self.results_panel.show_error(msg)
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Analysis Failed", msg)

    def clear_all(self):
        self.current_image_path = None
        self.last_result = None
        self.preview.clear()
        self.results_panel.clear()
        self.status_var.set("Cleared — upload a new image to continue")

    def save_results(self):
        if not self.last_result:
            messagebox.showinfo("Nothing to save", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("CSV", "*.csv")],
            initialfile=f"nationality_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            self.logger.export_single(self.last_result, path)
            self.status_var.set(f"Results saved → {path}")

    def _bind_keys(self):
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Return>", lambda e: self.run_analysis())
        self.root.bind("<Escape>", lambda e: self.clear_all())


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = NationalityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
