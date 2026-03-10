"""
utils/gui_components.py
========================
Custom tkinter widgets for the drowsiness detection GUI.

Aesthetic direction: dark industrial / vehicle HUD
- Near-black backgrounds (#0d0d0d, #151515)
- Danger red accents (#e63946)
- Amber warning (#f4a261)
- Monospace fonts everywhere (fits the "sensor readout" vibe)

Components:
- PreviewCanvas: shows images/video frames
- SidebarSection: labeled section box
- AnimatedButton: hover-animated button
- StatusBar: bottom info strip
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

from config import THEME


class PreviewCanvas(tk.Frame):
    """
    Left-side preview panel.
    Can display a static image, a video thumbnail, or live frames.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._photo = None

        self.label = tk.Label(
            self,
            text="📂  Click here or use Load buttons\nto open an image or video",
            font=("Courier New", 12),
            bg=THEME["surface"],
            fg="#444",
            cursor="hand2",
        )
        self.label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def load_file(self, path: str):
        """Show a static image from file path."""
        try:
            img = Image.open(path)
            self._display_pil(img)
        except Exception as e:
            self.label.configure(text=f"Preview failed: {e}", image="")

    def load_video_thumb(self, path: str):
        """Grab first frame from video and display as thumbnail."""
        try:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.show_frame(frame)
            else:
                self.label.configure(text="Video loaded — press Run", image="")
        except Exception:
            self.label.configure(text="Video loaded — press Run", image="")

    def show_frame(self, frame: np.ndarray):
        """Display a BGR numpy frame (used for live video)."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._display_pil(img)
        except Exception:
            pass

    def _display_pil(self, img: Image.Image):
        """Resize to fit and show a PIL image."""
        # Get current canvas size
        self.update_idletasks()
        w = max(self.winfo_width() - 20, 400)
        h = max(self.winfo_height() - 20, 300)

        img.thumbnail((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo, text="", bg=THEME["surface"])
        self.label.image = self._photo


class SidebarSection(tk.Frame):
    """
    A labeled card-style section in the sidebar.
    Returns self.inner for adding child widgets.
    """

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, bg=THEME["bg"], **kwargs)
        self.pack(fill=tk.X, padx=6, pady=5)

        tk.Label(
            self, text=title,
            font=("Courier New", 8, "bold"),
            bg=THEME["bg"], fg=THEME["danger"],
        ).pack(anchor="w", pady=(0, 2))

        self.inner = tk.Frame(self, bg=THEME["bg"])
        self.inner.pack(fill=tk.X)


class AnimatedButton(tk.Button):
    """
    Button with color-shift on hover.
    Defaults to the dark industrial theme.
    """

    def __init__(
        self, parent, text: str, command=None,
        accent: str = "#2a2a3e",
        bold: bool = False,
        **kwargs
    ):
        self._default_bg = accent
        self._hover_bg = _lighten(accent, 30)

        font_size = 9
        font_weight = "bold" if bold else "normal"

        super().__init__(
            parent,
            text=text,
            command=command,
            font=("Courier New", font_size, font_weight),
            bg=accent,
            fg=THEME["text"],
            relief=tk.FLAT,
            cursor="hand2",
            padx=8,
            pady=5,
            bd=0,
            activeforeground="white",
            activebackground=self._hover_bg,
            **kwargs
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        self.configure(bg=self._hover_bg)

    def _on_leave(self, _):
        self.configure(bg=self._default_bg)


class StatusBar(tk.Frame):
    """Bottom strip showing current status message."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["surface"], height=26, **kwargs)
        self.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")

        tk.Label(
            self,
            textvariable=self.status_var,
            font=("Courier New", 9),
            bg=THEME["surface"],
            fg=THEME["text"],
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        tk.Label(
            self,
            text="Drowsiness Detector v1.0 · EAR + DNN Age",
            font=("Courier New", 8),
            bg=THEME["surface"],
            fg="#3a3a3a",
        ).pack(side=tk.RIGHT, padx=10)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: int) -> str:
    """Lighten a hex color by adding `amount` to each RGB channel."""
    hex_color = hex_color.lstrip("#")
    try:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = min(255, r + amount)
        g = min(255, g + amount)
        b = min(255, b + amount)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#555"
