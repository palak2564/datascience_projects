"""
utils/carnivore_alert.py
=========================
The carnivore popup alert window.

When carnivores are detected, this fires a bold red modal dialog
with the count + names. Hard to miss.

Requirement from task brief:
  "highlight carnivorous animals in red and display a pop-up message
   indicating the number of detected carnivorous animals"
"""

import tkinter as tk
from tkinter import ttk
import time


def show_carnivore_alert(parent: tk.Tk, count: int, carnivores: list):
    """
    Shows a modal popup warning about detected carnivores.

    Args:
        parent:     The root tkinter window
        count:      Number of carnivores detected
        carnivores: List of detection dicts (with 'label', 'confidence')
    """
    # Build the modal window
    popup = tk.Toplevel(parent)
    popup.title("⚠️ Carnivore Alert")
    popup.configure(bg="#1a0a0a")
    popup.resizable(False, False)
    popup.grab_set()  # Makes it modal (blocks parent)

    # Center on screen
    w, h = 420, 320
    x = parent.winfo_x() + (parent.winfo_width() - w) // 2
    y = parent.winfo_y() + (parent.winfo_height() - h) // 2
    popup.geometry(f"{w}x{h}+{x}+{y}")

    # ── Warning icon + header ─────────────────────────────────────────────────
    tk.Label(
        popup,
        text="⚠️",
        font=("Arial", 40),
        bg="#1a0a0a",
        fg="#ff4444",
    ).pack(pady=(20, 5))

    tk.Label(
        popup,
        text=f"CARNIVORE{'S' if count > 1 else ''} DETECTED",
        font=("Courier New", 14, "bold"),
        bg="#1a0a0a",
        fg="#ff4444",
    ).pack()

    tk.Label(
        popup,
        text=f"{count} carnivorous animal{'s' if count > 1 else ''} found in this frame",
        font=("Courier New", 9),
        bg="#1a0a0a",
        fg="#cc8888",
    ).pack(pady=(2, 10))

    # ── List of carnivores detected ───────────────────────────────────────────
    list_frame = tk.Frame(popup, bg="#2a1010", relief=tk.FLAT, bd=0)
    list_frame.pack(fill=tk.X, padx=20, pady=5)

    for d in carnivores:
        name = d["label"].title()
        conf = d.get("confidence", 0)
        tk.Label(
            list_frame,
            text=f"  🔴  {name}  —  {conf:.0%} confidence",
            font=("Courier New", 9),
            bg="#2a1010",
            fg="#ff8888",
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=2)

    # ── Safety note ───────────────────────────────────────────────────────────
    tk.Label(
        popup,
        text="⚡ Exercise caution in this area",
        font=("Courier New", 8, "italic"),
        bg="#1a0a0a",
        fg="#884444",
    ).pack(pady=5)

    # ── Dismiss button ────────────────────────────────────────────────────────
    def dismiss():
        popup.grab_release()
        popup.destroy()

    btn = tk.Button(
        popup,
        text="  Dismiss  ",
        command=dismiss,
        font=("Courier New", 10, "bold"),
        bg="#cc2222",
        fg="white",
        relief=tk.FLAT,
        cursor="hand2",
        padx=16,
        pady=6,
        activebackground="#ff4444",
    )
    btn.pack(pady=12)

    # Auto-dismiss after 8 seconds (so it doesn't just hang there forever)
    popup.after(8000, lambda: dismiss() if popup.winfo_exists() else None)

    # Pulse animation — flashes the title color
    _pulse_label = popup.children.get("!label2")  # not reliable, skip for now
    popup.focus_set()
