"""
utils/gui_components.py
========================
Custom tkinter widgets used by the main app.

Nothing revolutionary here — just cleaner than inline widget creation.
Includes:
- AnimatedButton (hover effects)
- DetectionResultPanel (live detection log)
- StatusBar
"""

import tkinter as tk
from tkinter import ttk
from typing import List


# ─── Color palette ─────────────────────────────────────────────────────────────

BG = "#1a1a2e"
SURFACE = "#16213e"
ACCENT = "#e94560"
TEXT = "#eaeaea"
SUCCESS = "#4ecca3"
WARN = "#f5a623"
FONT_MONO = "Courier New"


# ─── AnimatedButton ────────────────────────────────────────────────────────────

class AnimatedButton(tk.Button):
    """
    A button that changes color on hover.
    Simple but makes the UI feel less static.
    """

    def __init__(self, parent, hover_bg: str = "#555", font_size: int = 9, **kwargs):
        self._default_bg = kwargs.get("bg", "#333")
        self._hover_bg = hover_bg

        # Ensure font is set
        if "font" not in kwargs:
            kwargs["font"] = (FONT_MONO, font_size)

        kwargs.setdefault("relief", tk.FLAT)
        kwargs.setdefault("cursor", "hand2")
        kwargs.setdefault("padx", 8)
        kwargs.setdefault("pady", 6)
        kwargs.setdefault("bd", 0)
        kwargs.setdefault("activeforeground", "white")
        kwargs.setdefault("activebackground", hover_bg)

        super().__init__(parent, **kwargs)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, event):
        self.configure(bg=self._hover_bg)

    def _on_leave(self, event):
        self.configure(bg=self._default_bg)


# ─── DetectionResultPanel ──────────────────────────────────────────────────────

class DetectionResultPanel(tk.Frame):
    """
    Scrollable list of detection results.
    Each item shows: animal name, diet type, confidence.
    Carnivores shown in red/orange.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(bg=SURFACE, relief=tk.FLAT)
        self._build()

    def _build(self):
        # Scrollable text area
        self.text = tk.Text(
            self,
            bg=SURFACE,
            fg=TEXT,
            font=(FONT_MONO, 8),
            relief=tk.FLAT,
            state=tk.DISABLED,
            wrap=tk.WORD,
            height=12,
            padx=6,
            pady=4,
        )
        scrollbar = tk.Scrollbar(self, command=self.text.yview, bg=SURFACE)
        self.text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Tag colors
        self.text.tag_configure("carnivore", foreground="#ff6b6b")
        self.text.tag_configure("herbivore", foreground=SUCCESS)
        self.text.tag_configure("omnivore", foreground=WARN)
        self.text.tag_configure("header", foreground=ACCENT, font=(FONT_MONO, 8, "bold"))
        self.text.tag_configure("dim", foreground="#666")

    def update(self, detections: List[dict]):
        """Refresh the panel with new detections."""
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)

        if not detections:
            self.text.insert(tk.END, "No animals detected.\n", "dim")
            self.text.configure(state=tk.DISABLED)
            return

        self.text.insert(tk.END, f"── {len(detections)} detection(s) ──\n\n", "header")

        for i, d in enumerate(detections, 1):
            label = d["label"].title()
            conf = d["confidence"]
            diet = d["diet_tag"]
            tag = diet  # maps to text tag

            icon = "🔴" if diet == "carnivore" else ("🟡" if diet == "omnivore" else "🟢")
            self.text.insert(tk.END, f"{i}. {icon} {label}\n", tag)
            self.text.insert(tk.END, f"   conf: {conf:.1%}  [{diet}]\n", "dim")

        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)

    def clear(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)


# ─── StatusBar ────────────────────────────────────────────────────────────────

class StatusBar(tk.Frame):
    """
    Bottom status bar. Shows messages and has a tiny progress indicator.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=28, **kwargs)
        self.pack_propagate(False)

        self._msg_var = tk.StringVar(value="Ready")
        self._dot = tk.StringVar(value="●")

        self.label = tk.Label(
            self,
            textvariable=self._msg_var,
            font=(FONT_MONO, 9),
            bg=kwargs.get("bg", SURFACE),
            fg=kwargs.get("fg", TEXT),
            anchor="w",
        )
        self.label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Version / credit
        tk.Label(
            self,
            text="Animal Detector v1.0 · YOLOv8",
            font=(FONT_MONO, 8),
            bg=kwargs.get("bg", SURFACE),
            fg="#444",
        ).pack(side=tk.RIGHT, padx=10)

    def set_message(self, msg: str):
        self._msg_var.set(msg)
