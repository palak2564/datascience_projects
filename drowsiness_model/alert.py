"""
utils/alert.py
===============
The drowsiness popup alert.

Fires when sleeping people are detected.
Shows count, ages, and a dramatic warning message.
Auto-dismisses after 10 seconds (configurable).

Design: aggressive red — this is a SAFETY alert.
"""

import tkinter as tk
from typing import List


def show_drowsiness_alert(parent: tk.Tk, sleeping_people: List[dict]):
    """
    Displays a modal popup alerting about sleeping people.

    Args:
        parent:          Root tkinter window
        sleeping_people: List of detection dicts (state=="sleeping")
    """
    count = len(sleeping_people)

    popup = tk.Toplevel(parent)
    popup.title("🚨 DROWSINESS ALERT")
    popup.configure(bg="#0d0000")
    popup.resizable(False, False)
    popup.grab_set()  # modal

    # Center on screen
    w, h = 440, 360
    x = parent.winfo_x() + (parent.winfo_width() - w) // 2
    y = parent.winfo_y() + (parent.winfo_height() - h) // 2
    popup.geometry(f"{w}x{h}+{x}+{y}")

    # ── Top: big warning ─────────────────────────────────────────────────────
    tk.Label(popup, text="🚨", font=("Arial", 42),
             bg="#0d0000", fg="#ff2222").pack(pady=(18, 4))

    tk.Label(
        popup,
        text="DROWSY DRIVER ALERT",
        font=("Courier New", 15, "bold"),
        bg="#0d0000", fg="#ff2222",
    ).pack()

    tk.Label(
        popup,
        text=f"{count} person{'s' if count > 1 else ''} detected sleeping",
        font=("Courier New", 10),
        bg="#0d0000", fg="#cc5555",
    ).pack(pady=(4, 12))

    # ── Separator ────────────────────────────────────────────────────────────
    tk.Frame(popup, bg="#330000", height=1).pack(fill=tk.X, padx=20)

    # ── People list ──────────────────────────────────────────────────────────
    for i, p in enumerate(sleeping_people, 1):
        person_frame = tk.Frame(popup, bg="#180000")
        person_frame.pack(fill=tk.X, padx=20, pady=4)

        age_str = f"  —  estimated age: {p['age']}" if p.get("age") else ""
        ear_str = f"  (EAR: {p['ear']:.3f})" if p.get("ear") else ""

        tk.Label(
            person_frame,
            text=f"  💤  Person {i}{age_str}{ear_str}",
            font=("Courier New", 9),
            bg="#180000", fg="#ff8888",
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=3)

    # ── Warning text ─────────────────────────────────────────────────────────
    tk.Frame(popup, bg="#330000", height=1).pack(fill=tk.X, padx=20, pady=8)

    tk.Label(
        popup,
        text="⚡ PULL OVER IMMEDIATELY\nDO NOT OPERATE VEHICLE WHILE DROWSY",
        font=("Courier New", 9, "bold"),
        bg="#0d0000", fg="#ff6666",
        justify=tk.CENTER,
    ).pack(pady=4)

    # ── Auto-dismiss countdown ────────────────────────────────────────────────
    countdown_var = tk.StringVar(value="Auto-dismiss in 10s")
    tk.Label(popup, textvariable=countdown_var, font=("Courier New", 8),
             bg="#0d0000", fg="#553333").pack()

    remaining = [10]

    def tick():
        if not popup.winfo_exists():
            return
        remaining[0] -= 1
        if remaining[0] <= 0:
            _dismiss()
            return
        countdown_var.set(f"Auto-dismiss in {remaining[0]}s")
        popup.after(1000, tick)

    def _dismiss():
        if popup.winfo_exists():
            popup.grab_release()
            popup.destroy()

    popup.after(1000, tick)

    # ── Dismiss button ────────────────────────────────────────────────────────
    tk.Button(
        popup,
        text="  ✓  ACKNOWLEDGE  ",
        command=_dismiss,
        font=("Courier New", 10, "bold"),
        bg="#aa1111", fg="white",
        relief=tk.FLAT,
        cursor="hand2",
        padx=16, pady=6,
        activebackground="#ff3333",
    ).pack(pady=10)

    popup.focus_set()
