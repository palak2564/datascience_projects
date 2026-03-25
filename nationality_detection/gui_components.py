"""
utils/gui_components.py
========================
Custom tkinter widgets for the Nationality Detection GUI.

Aesthetic: editorial / travel-document
- Warm off-white (#faf7f2) background — like aged paper
- Deep navy (#1a1f36) text
- Subtle card shadows via layered frames
- Georgia serif for headings (refined, international)
- Courier New for data labels (typewriter / passport feel)
- Results displayed as structured "passport card" with dividers

Components:
- ImagePreviewPanel: shows uploaded image
- ResultsPanel: structured output card
- ActionButton: styled buttons (primary/accent/ghost)
- SectionLabel: section header labels
- StatusFooter: bottom strip
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional

from config import THEME


# ─── Section Label ────────────────────────────────────────────────────────────

class SectionLabel(tk.Label):
    def __init__(self, parent, text: str, **kwargs):
        super().__init__(
            parent,
            text=text,
            font=("Courier New", 8, "bold"),
            bg=kwargs.pop("bg", THEME["card"]),
            fg=THEME["accent"],
            **kwargs,
        )


# ─── Action Button ────────────────────────────────────────────────────────────

class ActionButton(tk.Button):
    """
    Three styles:
    - primary: navy background, white text
    - accent:  warm amber/orange
    - ghost:   transparent outline look
    """

    STYLES = {
        "primary": {"bg": "#1a1f36", "fg": "#ffffff", "hover": "#2d3460"},
        "accent":  {"bg": "#c47a1e", "fg": "#ffffff", "hover": "#e09030"},
        "ghost":   {"bg": "#e8e3da", "fg": "#1a1f36", "hover": "#d4cfc5"},
    }

    def __init__(self, parent, text: str, command=None,
                 style: str = "primary", **kwargs):
        s = self.STYLES.get(style, self.STYLES["primary"])
        self._default_bg = s["bg"]
        self._hover_bg   = s["hover"]
        self._loading    = False

        super().__init__(
            parent,
            text=text,
            command=command,
            font=("Georgia", 9),
            bg=s["bg"],
            fg=s["fg"],
            relief=tk.FLAT,
            cursor="hand2",
            padx=10, pady=7,
            bd=0,
            activeforeground=s["fg"],
            activebackground=s["hover"],
            **kwargs,
        )
        self.bind("<Enter>", lambda _: self.configure(bg=self._hover_bg))
        self.bind("<Leave>", lambda _: self.configure(bg=self._default_bg))
        self._original_text = text

    def set_loading(self, loading: bool):
        self._loading = loading
        if loading:
            self.configure(text="⏳  Analysing...", state=tk.DISABLED)
        else:
            self.configure(text=self._original_text, state=tk.NORMAL)


# ─── Image Preview Panel ──────────────────────────────────────────────────────

class ImagePreviewPanel(tk.Frame):
    """
    Left panel that shows the uploaded image.
    Placeholder shown when no image loaded.
    Clicking the placeholder triggers load callback.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["card"], **kwargs)
        self._photo = None
        self._click_cb = None

        # Placeholder
        self._placeholder = tk.Label(
            self,
            text=(
                "╔══════════════════╗\n"
                "║                  ║\n"
                "║   📷  No image   ║\n"
                "║   uploaded yet   ║\n"
                "║                  ║\n"
                "║  Click to browse ║\n"
                "║                  ║\n"
                "╚══════════════════╝"
            ),
            font=("Courier New", 11),
            bg=THEME["card"],
            fg="#c8c0b0",
            justify=tk.CENTER,
            cursor="hand2",
        )
        self._placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=30)
        self._placeholder.bind("<Button-1>", lambda _: self._on_click())

        # Image label (hidden until loaded)
        self._img_label = tk.Label(self, bg=THEME["card"])

    def bind_click(self, callback):
        self._click_cb = callback

    def _on_click(self):
        if self._click_cb:
            self._click_cb()

    def load(self, path: str):
        """Load and display image from file path."""
        try:
            img = Image.open(path)
            self._display(img)
        except Exception as e:
            self._placeholder.configure(text=f"Preview failed:\n{e}")

    def load_from_array(self, frame: np.ndarray):
        """Display a BGR numpy array (annotated result)."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._display(img)
        except Exception:
            pass

    def _display(self, img: Image.Image):
        self.update_idletasks()
        w = max(self.winfo_width() - 20, 350)
        h = max(self.winfo_height() - 20, 280)
        img.thumbnail((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._placeholder.pack_forget()
        self._img_label.configure(image=self._photo, bg=THEME["card"])
        self._img_label.image = self._photo
        self._img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._img_label.bind("<Button-1>", lambda _: self._on_click())

    def clear(self):
        self._img_label.pack_forget()
        self._placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=30)
        self._photo = None


# ─── Results Panel ────────────────────────────────────────────────────────────

# Maps task group to a flag/icon for the passport card
GROUP_ICONS = {
    "indian":   ("🇮🇳", "#ff9933"),
    "african":  ("🌍", "#2e8b57"),
    "american": ("🇺🇸", "#3c3b6e"),
    "other":    ("🌐", "#4a5568"),
}

# Maps prediction keys to display labels
FIELD_LABELS = {
    "nationality":  "Nationality",
    "emotion":      "Emotion",
    "age":          "Age",
    "age_bracket":  "Age Range",
    "dress_colour": "Dress Colour",
    "confidence":   "Confidence",
}

EMOTION_ICONS = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "surprised": "😲", "fear": "😨", "disgust": "😒",
    "neutral": "😐", "contempt": "😤",
}


class ResultsPanel(tk.Frame):
    """
    Displays prediction results as a structured card.

    Looks like a mini passport/ID card:
    ┌──────────────────────────────┐
    │  🇮🇳  INDIAN                 │
    │  ─────────────────────────  │
    │  Emotion    : Happy 😊       │
    │  Age        : ~28            │
    │  Dress Colour: Navy Blue     │
    └──────────────────────────────┘
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["card"], **kwargs)
        self._widgets = []
        self._show_placeholder()

    def _show_placeholder(self):
        self._clear_widgets()
        lbl = tk.Label(
            self,
            text=(
                "Results will appear\n"
                "here after analysis.\n\n"
                "Upload an image and\n"
                "click  🔍 Analyse"
            ),
            font=("Georgia", 10, "italic"),
            bg=THEME["card"],
            fg=THEME["muted"],
            justify=tk.CENTER,
        )
        lbl.pack(expand=True, pady=30)
        self._widgets.append(lbl)

    def show_loading(self):
        self._clear_widgets()
        dots = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"]
        self._dot_idx = 0
        self._spinner_lbl = tk.Label(
            self,
            text="⠋  Analysing...",
            font=("Courier New", 11),
            bg=THEME["card"],
            fg=THEME["accent"],
        )
        self._spinner_lbl.pack(expand=True, pady=40)
        self._widgets.append(self._spinner_lbl)
        self._animate_spinner(dots)

    def _animate_spinner(self, dots):
        if self._spinner_lbl.winfo_exists():
            self._dot_idx = (self._dot_idx + 1) % len(dots)
            self._spinner_lbl.configure(
                text=f"{dots[self._dot_idx]}  Analysing...")
            self._spinner_lbl.after(120, lambda: self._animate_spinner(dots))

    def display(self, result: dict):
        """Build the passport-card style result display."""
        self._clear_widgets()

        group = result.get("task_group", "other")
        icon, accent = GROUP_ICONS.get(group, ("🌐", "#4a5568"))
        nationality = result.get("nationality", "Unknown")

        # ── Nationality header ────────────────────────────────────────────────
        hdr_frame = tk.Frame(self, bg=accent)
        hdr_frame.pack(fill=tk.X, padx=0, pady=(0, 0))
        self._widgets.append(hdr_frame)

        tk.Label(
            hdr_frame,
            text=f"  {icon}  {nationality.upper()}",
            font=("Georgia", 13, "bold"),
            bg=accent, fg="white",
            pady=10,
        ).pack(anchor="w", padx=10)

        # ── Confidence ───────────────────────────────────────────────────────
        conf = result.get("confidence", 0)
        if conf:
            tk.Label(
                hdr_frame,
                text=f"  Confidence: {conf:.1f}%",
                font=("Courier New", 8),
                bg=accent, fg="#ffffff99",
                pady=0,
            ).pack(anchor="w", padx=10)

        # ── Divider ───────────────────────────────────────────────────────────
        tk.Frame(self, bg="#e0d8cc", height=1).pack(fill=tk.X, padx=10, pady=6)

        # ── Fields ───────────────────────────────────────────────────────────
        fields_frame = tk.Frame(self, bg=THEME["card"])
        fields_frame.pack(fill=tk.X, padx=12)
        self._widgets.append(fields_frame)

        predictions = result.get("predictions_made", [])
        displayed_fields = set()

        for key in ["nationality", "emotion", "age", "age_bracket", "dress_colour"]:
            # Only show fields that were actually predicted
            if key == "nationality":
                pass  # always show (already in header, show smaller below)
            elif key not in predictions:
                continue

            if key in displayed_fields:
                continue
            displayed_fields.add(key)

            value = result.get(key)
            if value is None:
                continue

            label = FIELD_LABELS.get(key, key.replace("_", " ").title())

            row = tk.Frame(fields_frame, bg=THEME["card"])
            row.pack(fill=tk.X, pady=3)

            tk.Label(
                row, text=f"{label}",
                font=("Courier New", 9, "bold"),
                bg=THEME["card"], fg=THEME["muted"],
                width=14, anchor="w",
            ).pack(side=tk.LEFT)

            tk.Label(
                row, text=":",
                font=("Courier New", 9),
                bg=THEME["card"], fg=THEME["muted"],
            ).pack(side=tk.LEFT)

            # Format value nicely
            display_val = self._format_value(key, value, result)

            tk.Label(
                row, text=f"  {display_val}",
                font=("Georgia", 10),
                bg=THEME["card"], fg=THEME["text"],
                anchor="w",
            ).pack(side=tk.LEFT)

        # ── Divider + note about predictions ─────────────────────────────────
        tk.Frame(self, bg="#e0d8cc", height=1).pack(fill=tk.X, padx=10, pady=8)

        note = self._get_rule_note(group)
        tk.Label(
            self, text=note,
            font=("Courier New", 7, "italic"),
            bg=THEME["card"], fg=THEME["muted"],
            wraplength=240,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=14, pady=(0, 8))

        # ── Face detected indicator ───────────────────────────────────────────
        face_found = result.get("face_found", False)
        face_text = "✓ Face detected" if face_found else "⚠ No face detected (used full image)"
        face_color = THEME["ok"] if face_found else THEME["warn"]
        tk.Label(
            self, text=face_text,
            font=("Courier New", 7),
            bg=THEME["card"], fg=face_color,
        ).pack(anchor="w", padx=14, pady=(0, 6))

        for w in [hdr_frame, fields_frame]:
            self._widgets.append(w)

    def _format_value(self, key: str, value, result: dict) -> str:
        if key == "emotion":
            emotion_lower = str(value).lower()
            icon = EMOTION_ICONS.get(emotion_lower, "")
            return f"{value} {icon}".strip()
        elif key == "age":
            return f"~{value} years"
        elif key == "age_bracket":
            return f"{value}"
        elif key == "dress_colour":
            # Add a colour swatch emoji proxy
            colour_icons = {
                "Red": "🔴", "Blue": "🔵", "Green": "🟢", "Yellow": "🟡",
                "Orange": "🟠", "Purple": "🟣", "Black": "⚫", "White": "⚪",
                "Brown": "🟤", "Pink": "🌸", "Navy Blue": "🔵",
                "Gray": "⬜", "Grey": "⬜",
            }
            icon = colour_icons.get(value, "🎨")
            return f"{value} {icon}"
        return str(value)

    def _get_rule_note(self, group: str) -> str:
        notes = {
            "indian":   "Indian profile: emotion + age + dress colour predicted",
            "african":  "African profile: emotion + dress colour predicted",
            "american": "American profile: emotion + age predicted",
            "other":    "Other nationality: emotion predicted",
        }
        return notes.get(group, "")

    def show_error(self, msg: str):
        self._clear_widgets()
        lbl = tk.Label(
            self,
            text=f"⚠ Analysis failed\n\n{msg[:120]}",
            font=("Courier New", 9),
            bg=THEME["card"],
            fg="#cc3333",
            wraplength=250,
            justify=tk.CENTER,
        )
        lbl.pack(expand=True, pady=30)
        self._widgets.append(lbl)

    def clear(self):
        self._clear_widgets()
        self._show_placeholder()

    def _clear_widgets(self):
        for w in self.winfo_children():
            w.destroy()
        self._widgets = []


# ─── Status Footer ────────────────────────────────────────────────────────────

class StatusFooter(tk.Frame):
    def __init__(self, parent, status_var: tk.StringVar, **kwargs):
        super().__init__(parent, bg=THEME["header_bg"], height=26, **kwargs)
        self.pack_propagate(False)

        tk.Label(
            self, textvariable=status_var,
            font=("Courier New", 8),
            bg=THEME["header_bg"], fg=THEME["muted"],
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        tk.Label(
            self,
            text="Nationality Detector v1.0 · DeepFace + DNN",
            font=("Courier New", 7),
            bg=THEME["header_bg"], fg="#3a3a4a",
        ).pack(side=tk.RIGHT, padx=10)
