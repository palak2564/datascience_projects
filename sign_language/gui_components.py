"""
utils/gui_components.py
========================
Custom tkinter widgets for the Sign Language Detection GUI.

Aesthetic: neon / tech HUD
- Near-black (#0a0a0f) — like a darkened display panel
- Electric cyan (#00e5ff) — primary accent (communication = light waves)
- Toxic green (#76ff03) — success/active state
- Warning amber (#ff9800) — confidence indicator
- Monospace Consolas/Courier New — typewriter / terminal precision
- Glowing borders via tk highlightcolor

Components:
- PreviewCanvas: live video / image display
- PredictionDisplay: large animated prediction output
- HistoryPanel: scrollable log of past predictions
- NeonButton: glowing hover button
- SectionHeader: styled section title
- StatusStrip: bottom strip
- TimeWindowBadge: shows active/inactive status
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional

from config import THEME, ACTIVE_START, ACTIVE_END


# ─── Section Header ───────────────────────────────────────────────────────────

class SectionHeader(tk.Label):
    def __init__(self, parent, text: str, **kwargs):
        super().__init__(
            parent, text=text,
            font=("Consolas", 8, "bold"),
            bg=kwargs.pop("bg", THEME["surface"]),
            fg=THEME["cyan"],
            **kwargs,
        )


# ─── Neon Button ──────────────────────────────────────────────────────────────

class NeonButton(tk.Button):
    """
    Button with neon glow effect on hover.
    active=True highlights it as the selected tab.
    """

    def __init__(self, parent, text: str, command=None,
                 color: str = None, active: bool = False, **kwargs):
        self._color      = color or THEME["btn_bg"]
        self._active     = active
        self._hover_color = self._lighten(self._color, 40)
        self._active_bg   = THEME["cyan"]

        bg = self._active_bg if active else self._color
        fg = THEME["bg"] if active else THEME["text"]

        super().__init__(
            parent, text=text, command=command,
            font=("Consolas", 9),
            bg=bg, fg=fg,
            relief=tk.FLAT,
            cursor="hand2",
            padx=8, pady=6,
            bd=0,
            activeforeground=THEME["bg"],
            activebackground=THEME["cyan"],
            **kwargs,
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self.configure(bg=self._active_bg, fg=THEME["bg"])
        else:
            self.configure(bg=self._color, fg=THEME["text"])

    def _on_enter(self, _):
        if not self._active:
            self.configure(bg=self._hover_color)

    def _on_leave(self, _):
        if not self._active:
            self.configure(bg=self._color)

    @staticmethod
    def _lighten(hex_color: str, amount: int) -> str:
        hex_color = hex_color.lstrip("#")
        try:
            r = min(255, int(hex_color[0:2], 16) + amount)
            g = min(255, int(hex_color[2:4], 16) + amount)
            b = min(255, int(hex_color[4:6], 16) + amount)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#555"


# ─── Preview Canvas ───────────────────────────────────────────────────────────

class PreviewCanvas(tk.Frame):
    """
    Displays camera feed or uploaded image.
    Dark placeholder with ASCII-art camera icon when empty.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["surface"], **kwargs)
        self._photo = None
        self._label = tk.Label(
            self,
            text=(
                "\n"
                "  ┌──────────────────────────┐  \n"
                "  │                          │  \n"
                "  │    🤟  No feed active    │  \n"
                "  │                          │  \n"
                "  │  Upload image or start   │  \n"
                "  │  webcam to begin         │  \n"
                "  │                          │  \n"
                "  └──────────────────────────┘  \n"
            ),
            font=("Consolas", 11),
            bg=THEME["surface"],
            fg=THEME["dim"],
            cursor="hand2",
        )
        self._label.pack(fill=tk.BOTH, expand=True)

    def load_file(self, path: str):
        try:
            img = Image.open(path)
            self._display_pil(img)
        except Exception as e:
            self._label.configure(text=f"Preview error:\n{e}", image="")

    def show_frame(self, frame: np.ndarray):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._display_pil(Image.fromarray(rgb))
        except Exception:
            pass

    def _display_pil(self, img: Image.Image):
        self.update_idletasks()
        w = max(self.winfo_width() - 10, 400)
        h = max(self.winfo_height() - 10, 300)
        img.thumbnail((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._label.configure(image=self._photo, text="", bg=THEME["surface"])
        self._label.image = self._photo

    def reset(self):
        self._label.configure(
            text=(
                "\n"
                "  ┌──────────────────────────┐  \n"
                "  │    🤟  Feed stopped       │  \n"
                "  │  Start webcam to resume  │  \n"
                "  └──────────────────────────┘  \n"
            ),
            image="",
            bg=THEME["surface"],
            fg=THEME["dim"],
        )
        self._photo = None


# ─── Prediction Display ───────────────────────────────────────────────────────

class PredictionDisplay(tk.Frame):
    """
    Big animated output showing current prediction.
    Typewriter animation when new prediction arrives.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["surface"], **kwargs)
        self._anim_job = None

        # Big prediction text
        self._pred_var = tk.StringVar(value="—")
        self._pred_lbl = tk.Label(
            self,
            textvariable=self._pred_var,
            font=("Consolas", 32, "bold"),
            bg=THEME["surface"],
            fg=THEME["cyan"],
        )
        self._pred_lbl.pack(pady=(8, 2))

        # Confidence bar
        self._conf_frame = tk.Frame(self, bg=THEME["surface"])
        self._conf_frame.pack(fill=tk.X, padx=16, pady=(0, 4))

        self._conf_var = tk.StringVar(value="")
        tk.Label(
            self._conf_frame, textvariable=self._conf_var,
            font=("Consolas", 9),
            bg=THEME["surface"], fg=THEME["dim"],
        ).pack(anchor="w")

        self._bar_frame = tk.Frame(self._conf_frame, bg="#111", height=6)
        self._bar_frame.pack(fill=tk.X, pady=2)

        self._bar_fill = tk.Frame(self._bar_frame, bg=THEME["green"], height=6)
        self._bar_fill.place(x=0, y=0, relheight=1, relwidth=0)

        # Category label
        self._cat_var = tk.StringVar(value="")
        tk.Label(
            self, textvariable=self._cat_var,
            font=("Consolas", 8, "italic"),
            bg=THEME["surface"], fg=THEME["dim"],
        ).pack(pady=(0, 6))

    def show(self, prediction: str, confidence: float):
        """Animate in new prediction."""
        if self._anim_job:
            self.after_cancel(self._anim_job)

        # Confidence bar color
        if confidence >= 0.80:
            bar_color = THEME["green"]
        elif confidence >= 0.60:
            bar_color = THEME["warn"]
        else:
            bar_color = THEME["danger"]

        self._bar_fill.configure(bg=bar_color)
        self._bar_fill.place(relwidth=confidence)

        self._conf_var.set(f"Confidence: {confidence:.1%}")

        # Determine category
        from config import KNOWN_WORDS
        cat = "Word" if prediction in KNOWN_WORDS else "Letter"
        self._cat_var.set(f"Type: {cat}")

        # Typewriter animation
        self._typewriter(prediction, 0)

    def _typewriter(self, text: str, idx: int):
        """Reveals characters one by one."""
        if idx <= len(text):
            self._pred_var.set(text[:idx] + ("_" if idx < len(text) else ""))
            self._pred_lbl.configure(fg=THEME["cyan"])
            self._anim_job = self.after(60, lambda: self._typewriter(text, idx + 1))
        else:
            self._pred_var.set(text)

    def show_loading(self):
        self._pred_var.set("···")
        self._pred_lbl.configure(fg=THEME["dim"])
        self._conf_var.set("Analysing...")
        self._cat_var.set("")
        self._bar_fill.place(relwidth=0)

    def show_error(self):
        self._pred_var.set("ERR")
        self._pred_lbl.configure(fg=THEME["danger"])
        self._conf_var.set("Detection failed")

    def clear(self):
        self._pred_var.set("—")
        self._pred_lbl.configure(fg=THEME["cyan"])
        self._conf_var.set("")
        self._cat_var.set("")
        self._bar_fill.place(relwidth=0)


# ─── History Panel ────────────────────────────────────────────────────────────

class HistoryPanel(tk.Frame):
    """
    Scrollable log of past predictions.
    Each row: time | prediction | confidence | mode
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["surface"], **kwargs)

        self._text = tk.Text(
            self,
            bg=THEME["surface"],
            fg=THEME["text"],
            font=("Consolas", 8),
            relief=tk.FLAT,
            state=tk.DISABLED,
            wrap=tk.NONE,
            height=8,
            padx=6, pady=4,
        )
        sb = tk.Scrollbar(self, command=self._text.yview, bg="#111")
        self._text.configure(yscrollcommand=sb.set)

        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(fill=tk.BOTH, expand=True)

        self._text.tag_configure("time",  foreground=THEME["dim"])
        self._text.tag_configure("pred",  foreground=THEME["cyan"])
        self._text.tag_configure("conf",  foreground=THEME["green"])
        self._text.tag_configure("mode",  foreground=THEME["warn"])
        self._text.tag_configure("low",   foreground=THEME["danger"])
        self._text.tag_configure("hdr",   foreground=THEME["dim"],
                                          font=("Consolas", 7, "bold"))

        # Header row
        self._write_header()

    def _write_header(self):
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, f"{'TIME':8s} {'SIGN':16s} {'CONF':6s} {'MODE'}\n", "hdr")
        self._text.insert(tk.END, "─" * 42 + "\n", "hdr")
        self._text.configure(state=tk.DISABLED)

    def add_entry(self, entry: dict):
        self._text.configure(state=tk.NORMAL)
        t    = entry.get("time", "")
        pred = entry.get("pred", "—")
        conf = entry.get("conf", 0.0)
        mode = entry.get("mode", "—")

        conf_tag = "conf" if conf >= 0.6 else "low"

        self._text.insert(tk.END, f"{t:8s} ", "time")
        self._text.insert(tk.END, f"{pred:16s} ", "pred")
        self._text.insert(tk.END, f"{conf:.0%}  ".rjust(6), conf_tag)
        self._text.insert(tk.END, f"{mode}\n", "mode")

        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)

    def clear(self):
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._write_header()
        self._text.configure(state=tk.DISABLED)


# ─── Status Strip ─────────────────────────────────────────────────────────────

class StatusStrip(tk.Frame):
    def __init__(self, parent, status_var: tk.StringVar, **kwargs):
        super().__init__(parent, bg=THEME["surface"], height=24, **kwargs)
        self.pack_propagate(False)
        tk.Label(
            self, textvariable=status_var,
            font=("Consolas", 8),
            bg=THEME["surface"], fg=THEME["dim"],
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        tk.Label(
            self, text="Sign Language Detector v1.0 · MediaPipe + Neural Network",
            font=("Consolas", 7),
            bg=THEME["surface"], fg="#2a2a3a",
        ).pack(side=tk.RIGHT, padx=10)


# ─── Time Window Badge ────────────────────────────────────────────────────────

class TimeWindowBadge(tk.Frame):
    """
    Pill-shaped badge showing:
    - 🟢 ACTIVE (6PM–10PM)
    - 🔴 INACTIVE (outside hours)
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=THEME["surface"], **kwargs)

        self._dot = tk.Label(
            self, text="●", font=("Consolas", 10),
            bg=THEME["surface"], fg=THEME["dim"],
        )
        self._dot.pack(side=tk.LEFT)

        self._lbl = tk.Label(
            self, text="INACTIVE",
            font=("Consolas", 8, "bold"),
            bg=THEME["surface"], fg=THEME["dim"],
        )
        self._lbl.pack(side=tk.LEFT, padx=(2, 4))

        self._window_lbl = tk.Label(
            self,
            text=f"{ACTIVE_START.strftime('%I:%M %p')}–{ACTIVE_END.strftime('%I:%M %p')}",
            font=("Consolas", 8),
            bg=THEME["surface"], fg=THEME["dim"],
        )
        self._window_lbl.pack(side=tk.LEFT)

    def update_state(self, active: bool):
        if active:
            self._dot.configure(fg=THEME["green"])
            self._lbl.configure(text="ACTIVE", fg=THEME["green"])
        else:
            self._dot.configure(fg=THEME["danger"])
            self._lbl.configure(text="INACTIVE", fg=THEME["danger"])
