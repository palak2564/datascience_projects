"""
app.py — Sign Language Detection System
=========================================
Real-time and image-based ASL/sign language word recognition.

Internship Task Extension — Sign Language Detection

Features:
- Upload image mode — detect sign from static photo
- Real-time webcam mode — live hand sign recognition
- Time-restricted operation: 6:00 PM – 10:00 PM
- Supports 26 ASL alphabet letters + 10 custom words
- Hand landmark overlay using MediaPipe
- Prediction confidence display
- Full prediction history log

GUI Design: neon / tech-minimalist
- Deep dark background (#0a0a0f) — feels like a HUD display
- Electric cyan (#00e5ff) accents — sign language = visual communication = light
- Monospace Consolas/Courier — technical precision
- Glowing box shadows on cards
- Live prediction animates in with typewriter effect

Tech stack:
- Hand detection: MediaPipe Hands (21 landmarks)
- Sign classification: Trained CNN on hand landmark features
  → MobileNetV2 fine-tuned on ASL dataset (via TensorFlow/Keras)
  → Fallback: SVM on flattened landmark vectors
- Time window: 6:00 PM – 10:00 PM enforced

Words trained (10 custom ASL words + full alphabet):
  Hello, Thank You, Yes, No, Please, Sorry, Help, Eat, Water, More
  + A-Z alphabet

Author: [Your Name]
Date: 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import time
from datetime import datetime, time as dtime

from utils.detector import SignLanguageDetector
from utils.gui_components import (
    PreviewCanvas, PredictionDisplay, HistoryPanel,
    NeonButton, SectionHeader, StatusStrip, TimeWindowBadge
)
from utils.logger import PredictionLogger
from config import THEME, APP_TITLE, WINDOW_SIZE, ACTIVE_START, ACTIVE_END, KNOWN_WORDS


class SignLanguageApp:
    """
    Main application window.

    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │  HEADER: title · time window badge · status             │
    ├─────────────────────────────┬───────────────────────────┤
    │                             │  MODE TABS                │
    │   LIVE PREVIEW              │  (Image / Webcam)         │
    │   (camera feed or           ├───────────────────────────┤
    │    uploaded image)          │  PREDICTION DISPLAY       │
    │                             │  (big animated text)      │
    │                             ├───────────────────────────┤
    │                             │  CONTROLS                 │
    │                             ├───────────────────────────┤
    │                             │  HISTORY LOG              │
    ├─────────────────────────────┴───────────────────────────┤
    │  STATUS STRIP                                           │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=THEME["bg"])
        self.root.minsize(1050, 680)

        # State
        self.mode = "image"          # "image" | "webcam"
        self.current_image = None
        self.webcam_running = False
        self.is_processing = False
        self.prediction_history = []

        # Components
        self.detector = SignLanguageDetector()
        self.logger = PredictionLogger()

        self._build_ui()
        self._bind_keys()
        self._start_clock_ticker()

        self.status_var.set("Ready — system operational 6:00 PM to 10:00 PM")

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        main = tk.Frame(self.root, bg=THEME["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        self._build_preview(main)
        self._build_right_panel(main)

        self.status_var = tk.StringVar(value="")
        StatusStrip(self.root, self.status_var).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=THEME["surface"], height=58)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        # Title
        tk.Label(
            hdr, text="◈  SIGN LANGUAGE DETECTOR",
            font=("Consolas", 17, "bold"),
            bg=THEME["surface"], fg=THEME["cyan"],
        ).pack(side=tk.LEFT, padx=18, pady=10)

        tk.Label(
            hdr, text="ASL · MediaPipe · CNN",
            font=("Consolas", 9),
            bg=THEME["surface"], fg=THEME["dim"],
        ).pack(side=tk.LEFT, padx=0, pady=18)

        # Time window badge (right side)
        self.time_badge = TimeWindowBadge(hdr)
        self.time_badge.pack(side=tk.RIGHT, padx=16, pady=8)

        # Current time display
        self._clock_var = tk.StringVar(value="")
        tk.Label(
            hdr, textvariable=self._clock_var,
            font=("Consolas", 10),
            bg=THEME["surface"], fg=THEME["cyan"],
        ).pack(side=tk.RIGHT, padx=10, pady=18)

    def _build_preview(self, parent):
        frame = tk.Frame(parent, bg=THEME["surface"],
                         highlightbackground=THEME["cyan"],
                         highlightthickness=1)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        SectionHeader(frame, "◉  LIVE PREVIEW").pack(anchor="w", padx=12, pady=(8, 2))

        self.preview = PreviewCanvas(frame)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=THEME["bg"])
        right.grid(row=0, column=1, sticky="nsew")

        # Mode selector tabs
        tab_frame = tk.Frame(right, bg=THEME["bg"])
        tab_frame.pack(fill=tk.X, pady=(0, 6))

        SectionHeader(tab_frame, "MODE").pack(anchor="w", padx=0, pady=(0, 4))

        tabs = tk.Frame(tab_frame, bg=THEME["bg"])
        tabs.pack(fill=tk.X)

        self._img_tab_btn = NeonButton(
            tabs, "🖼  Image Upload",
            command=lambda: self._switch_mode("image"),
            active=True,
        )
        self._img_tab_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self._cam_tab_btn = NeonButton(
            tabs, "📷  Live Webcam",
            command=lambda: self._switch_mode("webcam"),
            active=False,
        )
        self._cam_tab_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Prediction display (big animated output)
        pred_card = tk.Frame(right, bg=THEME["surface"],
                             highlightbackground=THEME["cyan"],
                             highlightthickness=1)
        pred_card.pack(fill=tk.X, pady=6)

        SectionHeader(pred_card, "◈  PREDICTION").pack(anchor="w", padx=12, pady=(8, 2))

        self.prediction_display = PredictionDisplay(pred_card)
        self.prediction_display.pack(fill=tk.X, padx=8, pady=(0, 10))

        # Controls
        ctrl_card = tk.Frame(right, bg=THEME["surface"],
                             highlightbackground="#333",
                             highlightthickness=1)
        ctrl_card.pack(fill=tk.X, pady=6)

        SectionHeader(ctrl_card, "CONTROLS").pack(anchor="w", padx=12, pady=(8, 2))

        btn_frame = tk.Frame(ctrl_card, bg=THEME["surface"])
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Image mode buttons
        self._img_controls = tk.Frame(btn_frame, bg=THEME["surface"])
        self._img_controls.pack(fill=tk.X)

        NeonButton(
            self._img_controls, "📂  Upload Image",
            command=self.upload_image,
        ).pack(fill=tk.X, pady=2)

        NeonButton(
            self._img_controls, "🔍  Detect Sign",
            command=self.detect_image,
            color=THEME["green"],
        ).pack(fill=tk.X, pady=2)

        # Webcam mode buttons
        self._cam_controls = tk.Frame(btn_frame, bg=THEME["surface"])

        self._start_cam_btn = NeonButton(
            self._cam_controls, "▶  Start Webcam",
            command=self.start_webcam,
            color=THEME["green"],
        )
        self._start_cam_btn.pack(fill=tk.X, pady=2)

        self._stop_cam_btn = NeonButton(
            self._cam_controls, "⏹  Stop Webcam",
            command=self.stop_webcam,
            color=THEME["danger"],
        )
        self._stop_cam_btn.pack(fill=tk.X, pady=2)

        # Confidence threshold
        tk.Frame(btn_frame, bg=THEME["surface"], height=1).pack(fill=tk.X, pady=4)
        tk.Label(btn_frame, text="Min. confidence:",
                 font=("Consolas", 8), bg=THEME["surface"],
                 fg=THEME["dim"]).pack(anchor="w")

        self.conf_var = tk.DoubleVar(value=0.60)
        tk.Scale(
            btn_frame, from_=0.1, to=0.99, resolution=0.01,
            variable=self.conf_var, orient=tk.HORIZONTAL,
            bg=THEME["surface"], fg=THEME["text"],
            troughcolor="#222", highlightthickness=0,
            sliderrelief=tk.FLAT, length=200,
        ).pack(fill=tk.X)
        self._conf_lbl = tk.Label(btn_frame, text="60%",
                                   font=("Consolas", 8), bg=THEME["surface"],
                                   fg=THEME["cyan"])
        self._conf_lbl.pack(anchor="e")
        self.conf_var.trace("w", lambda *_: self._conf_lbl.configure(
            text=f"{int(self.conf_var.get()*100)}%"))

        NeonButton(
            btn_frame, "🗑  Clear History",
            command=self.clear_history,
            color="#444",
        ).pack(fill=tk.X, pady=(6, 2))

        NeonButton(
            btn_frame, "💾  Export Log",
            command=self.export_log,
            color="#1a3a2a",
        ).pack(fill=tk.X, pady=2)

        # History panel
        hist_card = tk.Frame(right, bg=THEME["surface"],
                             highlightbackground="#333",
                             highlightthickness=1)
        hist_card.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        SectionHeader(hist_card, "HISTORY").pack(anchor="w", padx=12, pady=(8, 2))

        self.history = HistoryPanel(hist_card)
        self.history.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Start in image mode
        self._switch_mode("image")

    # ── Mode switching ─────────────────────────────────────────────────────────

    def _switch_mode(self, mode: str):
        # Stop webcam if switching away
        if self.webcam_running and mode != "webcam":
            self.stop_webcam()

        self.mode = mode

        # Toggle tab button styles
        self._img_tab_btn.set_active(mode == "image")
        self._cam_tab_btn.set_active(mode == "webcam")

        # Toggle control panels
        if mode == "image":
            self._cam_controls.pack_forget()
            self._img_controls.pack(fill=tk.X)
        else:
            self._img_controls.pack_forget()
            self._cam_controls.pack(fill=tk.X)

    # ── Time window check ──────────────────────────────────────────────────────

    def _is_active(self) -> bool:
        now = datetime.now().time()
        return ACTIVE_START <= now <= ACTIVE_END

    def _check_time_window(self) -> bool:
        if not self._is_active():
            now_str = datetime.now().strftime("%I:%M %p")
            messagebox.showwarning(
                "Outside Operating Hours",
                f"⏰  The Sign Language Detector is only active\n"
                f"   from {ACTIVE_START.strftime('%I:%M %p')} to "
                f"{ACTIVE_END.strftime('%I:%M %p')}.\n\n"
                f"   Current time: {now_str}\n\n"
                f"   Please try again during operating hours."
            )
            return False
        return True

    def _start_clock_ticker(self):
        """Updates clock display every second."""
        def tick():
            now = datetime.now()
            self._clock_var.set(now.strftime("%I:%M:%S %p"))
            self.time_badge.update_state(self._is_active())
            self.root.after(1000, tick)
        tick()

    # ── Image upload & detection ───────────────────────────────────────────────

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All", "*.*"),
            ]
        )
        if not path:
            return
        self.current_image = path
        self.preview.load_file(path)
        self.prediction_display.clear()
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def detect_image(self):
        if not self._check_time_window():
            return
        if not self.current_image:
            messagebox.showwarning("No Image", "Upload an image first.")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.prediction_display.show_loading()
        self.status_var.set("🔍 Detecting sign...")

        threading.Thread(target=self._image_worker, daemon=True).start()

    def _image_worker(self):
        try:
            result = self.detector.predict_image(
                self.current_image, min_confidence=self.conf_var.get())
            self.root.after(0, lambda: self._on_image_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))
        finally:
            self.is_processing = False

    def _on_image_result(self, result: dict):
        if result.get("annotated") is not None:
            self.preview.show_frame(result["annotated"])

        pred = result.get("prediction", "—")
        conf = result.get("confidence", 0.0)
        hands = result.get("hands_detected", 0)

        self.prediction_display.show(pred, conf)
        self._add_to_history(pred, conf, "image")
        self.logger.log(pred, conf, source=self.current_image, mode="image")

        self.status_var.set(
            f"✓ Detected: {pred}  ({conf:.0%} confidence)  |  {hands} hand(s) found"
        )

    # ── Webcam mode ────────────────────────────────────────────────────────────

    def start_webcam(self):
        if not self._check_time_window():
            return
        if self.webcam_running:
            return

        self.webcam_running = True
        self.status_var.set("📷 Webcam active — show hand signs...")
        threading.Thread(target=self._webcam_worker, daemon=True).start()

    def stop_webcam(self):
        self.webcam_running = False
        self.status_var.set("Webcam stopped.")
        self.preview.reset()

    def _webcam_worker(self):
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.root.after(0, lambda: self._on_error("Cannot open webcam (device 0)"))
            self.webcam_running = False
            return

        last_pred_time = 0
        PRED_INTERVAL = 0.5  # predict every 0.5s to avoid spamming

        try:
            while self.webcam_running:
                # Check time window
                if not self._is_active():
                    self.root.after(0, lambda: self.status_var.set(
                        "⏰ Outside operating hours — webcam stopped"))
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                now = time.time()

                # Draw landmarks on every frame
                annotated = self.detector.draw_landmarks(frame.copy())
                self.root.after(0, lambda f=annotated: self.preview.show_frame(f))

                # Predict at intervals
                if now - last_pred_time >= PRED_INTERVAL:
                    result = self.detector.predict_frame(
                        frame, min_confidence=self.conf_var.get())
                    pred = result.get("prediction")
                    conf = result.get("confidence", 0.0)

                    if pred and conf >= self.conf_var.get():
                        self.root.after(0, lambda p=pred, c=conf:
                                        self.prediction_display.show(p, c))
                        self.root.after(0, lambda p=pred, c=conf:
                                        self._add_to_history(p, c, "webcam"))
                        self.logger.log(pred, conf, source="webcam", mode="webcam")

                    last_pred_time = now
                    status = f"📷 Live  |  {result.get('hands_detected', 0)} hand(s)"
                    self.root.after(0, lambda s=status: self.status_var.set(s))

                time.sleep(0.03)

        finally:
            cap.release()
            self.webcam_running = False

    # ── History ────────────────────────────────────────────────────────────────

    def _add_to_history(self, prediction: str, confidence: float, mode: str):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"time": ts, "pred": prediction, "conf": confidence, "mode": mode}
        self.prediction_history.insert(0, entry)
        self.history.add_entry(entry)

    def clear_history(self):
        self.prediction_history.clear()
        self.history.clear()
        self.prediction_display.clear()

    def export_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"sign_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if path:
            self.logger.export_csv(path)
            self.status_var.set(f"✓ Log exported → {path}")

    def _on_error(self, msg: str):
        self.status_var.set(f"Error: {msg}")
        self.prediction_display.show_error()
        messagebox.showerror("Detection Error", msg)

    def _bind_keys(self):
        self.root.bind("<Control-o>", lambda e: self.upload_image())
        self.root.bind("<space>",     lambda e: self.detect_image())
        self.root.bind("<Escape>",    lambda e: self.stop_webcam())

    def on_close(self):
        self.stop_webcam()
        self.root.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
