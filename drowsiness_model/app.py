"""
app.py — Drowsiness Detection System
======================================
Driver/Passenger Drowsiness Detection with Age Prediction

Internship Project Extension — Drowsiness Detection Task

What this does:
- Detects faces in images and video (including webcam)
- Classifies each person as AWAKE or DROWSY/SLEEPING
- Predicts estimated age for each detected person
- Highlights SLEEPING people with RED bounding boxes
- Shows a popup alert with count of sleeping people + their ages
- Full tkinter GUI: image load, video load, webcam, live preview
- Saves annotated outputs, exports CSV log

Tech stack:
- Face detection: OpenCV Haar Cascade (fast, reliable)
- Drowsiness: Eye Aspect Ratio (EAR) via dlib landmarks + trained SVM
- Age estimation: OpenCV DNN with pre-trained Caffe model
- GUI: tkinter (dark industrial theme — fits the vehicle safety context)

Author: [Your Name]
Date: 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import time
from datetime import datetime

from utils.detector import DrowsinessDetector
from utils.gui_components import PreviewCanvas, SidebarSection, AnimatedButton, StatusBar
from utils.alert import show_drowsiness_alert
from utils.logger import DrowsinessLogger
from config import THEME, APP_TITLE, WINDOW_SIZE


# ─── Main Application ─────────────────────────────────────────────────────────

class DrowsinessApp:
    """
    Main GUI window.

    Layout:
    ┌─────────────────────────────────────┐
    │  HEADER: title + model status       │
    ├──────────────────────┬──────────────┤
    │                      │  Load Media  │
    │   PREVIEW CANVAS     │  Detection   │
    │   (live annotated    │  Controls    │
    │    video/image)      │  Results Log │
    │                      │  Export      │
    ├──────────────────────┴──────────────┤
    │  STATUS BAR                         │
    └─────────────────────────────────────┘
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=THEME["bg"])
        self.root.minsize(1000, 650)

        # State
        self.current_file = None
        self.current_mode = None   # "image" | "video" | "webcam"
        self.is_processing = False
        self.video_running = False
        self.last_result = None
        self.detection_history = []

        # Components
        self.detector = DrowsinessDetector()
        self.logger = DrowsinessLogger()

        self._build_ui()
        self._bind_shortcuts()

        self.status.set("Ready — load an image or video to begin 😴")

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        # Main 2-column layout
        main = tk.Frame(self.root, bg=THEME["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)

        self._build_preview(main)
        self._build_sidebar(main)

        self.statusbar = StatusBar(self.root)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status = self.statusbar.status_var

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=THEME["surface"], height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="⚠  DROWSINESS DETECTOR",
            font=("Courier New", 17, "bold"),
            bg=THEME["surface"], fg=THEME["danger"],
        ).pack(side=tk.LEFT, padx=18, pady=10)

        tk.Label(
            hdr, text="Vehicle Safety · Real-time · Multi-person",
            font=("Courier New", 9),
            bg=THEME["surface"], fg="#666",
        ).pack(side=tk.LEFT, padx=0, pady=16)

        # Model indicator
        self._model_dot = tk.Label(
            hdr, text="● Model Loading...",
            font=("Courier New", 9),
            bg=THEME["surface"], fg=THEME["warn"],
        )
        self._model_dot.pack(side=tk.RIGHT, padx=16)

        # Check model after short delay
        self.root.after(800, self._update_model_status)

    def _update_model_status(self):
        if self.detector.is_ready():
            self._model_dot.configure(text="● Models Ready", fg=THEME["ok"])
        else:
            self._model_dot.configure(text="● Partial Model (EAR mode)", fg=THEME["warn"])

    def _build_preview(self, parent):
        frame = tk.Frame(parent, bg=THEME["surface"])
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)

        self.preview = PreviewCanvas(frame, bg=THEME["surface"])
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview.bind("<Button-1>", lambda e: self.load_image())

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=THEME["bg"], width=290)
        sb.grid(row=0, column=1, sticky="nsew", pady=4)
        sb.pack_propagate(False)

        # Load media
        s1 = SidebarSection(sb, "📂  LOAD MEDIA")
        AnimatedButton(s1.inner, "🖼  Load Image", self.load_image).pack(fill=tk.X, pady=2)
        AnimatedButton(s1.inner, "🎬  Load Video", self.load_video).pack(fill=tk.X, pady=2)
        AnimatedButton(s1.inner, "📷  Webcam Live", self.start_webcam,
                       accent=THEME["warn"]).pack(fill=tk.X, pady=2)

        # Detection settings
        s2 = SidebarSection(sb, "⚙  SETTINGS")

        tk.Label(s2.inner, text="EAR threshold (eye closure):",
                 bg=THEME["bg"], fg=THEME["text"], font=("Courier New", 8)).pack(anchor="w")
        self.ear_var = tk.DoubleVar(value=0.25)
        tk.Scale(s2.inner, from_=0.1, to=0.5, resolution=0.01,
                 variable=self.ear_var, orient=tk.HORIZONTAL,
                 bg=THEME["bg"], fg=THEME["text"],
                 troughcolor=THEME["surface"], highlightthickness=0,
                 sliderrelief=tk.FLAT).pack(fill=tk.X)
        self._ear_lbl = tk.Label(s2.inner, text="0.25", bg=THEME["bg"],
                                  fg=THEME["danger"], font=("Courier New", 8))
        self._ear_lbl.pack(anchor="e")
        self.ear_var.trace("w", lambda *_: self._ear_lbl.configure(
            text=f"{self.ear_var.get():.2f}"))

        self.show_landmarks_var = tk.BooleanVar(value=False)
        self._chk(s2.inner, "Show facial landmarks", self.show_landmarks_var)

        self.age_on_var = tk.BooleanVar(value=True)
        self._chk(s2.inner, "Show age prediction", self.age_on_var)

        self.alert_on_var = tk.BooleanVar(value=True)
        self._chk(s2.inner, "Enable popup alerts", self.alert_on_var)

        # Run
        s3 = SidebarSection(sb, "▶  DETECTION")
        AnimatedButton(s3.inner, "▶  Run Detection", self.run_detection,
                       accent=THEME["danger"], bold=True).pack(fill=tk.X, pady=4)
        AnimatedButton(s3.inner, "⏹  Stop", self.stop_processing,
                       accent="#444").pack(fill=tk.X, pady=2)

        # Results log
        s4 = SidebarSection(sb, "📊  RESULTS")
        self.result_text = tk.Text(
            s4.inner, height=9, bg=THEME["surface"], fg=THEME["text"],
            font=("Courier New", 8), relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.WORD, padx=5, pady=4,
        )
        self.result_text.pack(fill=tk.BOTH)
        self.result_text.tag_configure("sleeping", foreground="#ff5555")
        self.result_text.tag_configure("awake", foreground=THEME["ok"])
        self.result_text.tag_configure("hdr", foreground=THEME["danger"],
                                        font=("Courier New", 8, "bold"))
        self.result_text.tag_configure("dim", foreground="#555")

        # Export
        s5 = SidebarSection(sb, "💾  EXPORT")
        AnimatedButton(s5.inner, "💾  Save Annotated Image",
                       self.save_result, accent="#1a4a2a").pack(fill=tk.X, pady=2)
        AnimatedButton(s5.inner, "📋  Export CSV Log",
                       self.export_log, accent="#1a4a2a").pack(fill=tk.X, pady=2)

    def _chk(self, parent, text, var):
        tk.Checkbutton(
            parent, text=text, variable=var,
            bg=THEME["bg"], fg=THEME["text"], selectcolor="#333",
            activebackground=THEME["bg"], font=("Courier New", 8),
        ).pack(anchor="w", pady=1)

    # ── Event Handlers ─────────────────────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        self.current_file = path
        self.current_mode = "image"
        self.preview.load_file(path)
        self.status.set(f"Loaded: {os.path.basename(path)}")
        self._clear_results()

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if not path:
            return
        self.current_file = path
        self.current_mode = "video"
        self.preview.load_video_thumb(path)
        self.status.set(f"Video: {os.path.basename(path)} — press Run")

    def start_webcam(self):
        self.current_file = 0
        self.current_mode = "webcam"
        self.status.set("Webcam mode selected — press Run Detection")

    def run_detection(self):
        if self.is_processing:
            self.status.set("Already running! Hit Stop first.")
            return
        if self.current_file is None:
            messagebox.showwarning("No Input", "Please load an image or video first.")
            return

        self.is_processing = True
        self.status.set("🔍 Detecting...")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            ear_thresh = self.ear_var.get()
            show_lm = self.show_landmarks_var.get()
            show_age = self.age_on_var.get()

            if self.current_mode == "image":
                result = self.detector.process_image(
                    self.current_file, ear_thresh, show_lm, show_age)
                self.root.after(0, lambda: self._on_image_result(result))

            elif self.current_mode in ("video", "webcam"):
                self.video_running = True
                self.detector.process_video(
                    self.current_file, ear_thresh, show_lm, show_age,
                    frame_cb=self._on_frame,
                    stop_flag=lambda: not self.video_running,
                )
                self.root.after(0, lambda: self.status.set("Video finished ✓"))

        except Exception as e:
            self.root.after(0, lambda: self.status.set(f"Error: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.is_processing = False

    def _on_image_result(self, result):
        if result is None:
            return
        self.last_result = result
        self.preview.show_frame(result["annotated"])
        self._update_results(result["people"])
        self.logger.log(result["people"], source=self.current_file)
        self.detection_history.append(result)

        sleeping = [p for p in result["people"] if p["state"] == "sleeping"]
        if sleeping and self.alert_on_var.get():
            self.root.after(150, lambda: show_drowsiness_alert(
                self.root, sleeping))

        total = len(result["people"])
        ns = len(sleeping)
        msg = f"Done! {total} person(s) — {ns} sleeping ⚠️" if ns else \
              f"Done! {total} person(s) — all awake ✓"
        self.status.set(msg)

    def _on_frame(self, frame, people):
        self.root.after(0, lambda: self.preview.show_frame(frame))
        sleeping = [p for p in people if p["state"] == "sleeping"]
        self.root.after(0, lambda: self.status.set(
            f"Live: {len(people)} person(s) — {len(sleeping)} sleeping"
        ))

    def stop_processing(self):
        self.video_running = False
        self.is_processing = False
        self.status.set("Stopped.")

    def save_result(self):
        if not self.detection_history:
            messagebox.showinfo("Nothing to save", "Run detection first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        if path:
            import cv2
            cv2.imwrite(path, self.detection_history[-1]["annotated"])
            self.status.set(f"Saved → {path}")

    def export_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"drowsiness_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if path:
            self.logger.export_csv(path)
            self.status.set(f"Log exported → {path}")

    def _update_results(self, people: list):
        t = self.result_text
        t.configure(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        if not people:
            t.insert(tk.END, "No faces detected.\n", "dim")
            t.configure(state=tk.DISABLED)
            return

        sleeping = [p for p in people if p["state"] == "sleeping"]
        t.insert(tk.END, f"── {len(people)} person(s) found ──\n\n", "hdr")

        for i, p in enumerate(people, 1):
            is_sl = p["state"] == "sleeping"
            icon = "🔴" if is_sl else "🟢"
            tag = "sleeping" if is_sl else "awake"
            state_str = "SLEEPING ⚠" if is_sl else "Awake"
            age_str = f"~{p['age']}y" if p.get("age") else ""

            t.insert(tk.END, f"{i}. {icon} {state_str}  {age_str}\n", tag)
            ear = p.get("ear", 0)
            if ear:
                t.insert(tk.END, f"   EAR: {ear:.3f}\n", "dim")

        if sleeping:
            t.insert(tk.END, f"\n⚠️  {len(sleeping)} SLEEPING!\n", "sleeping")

        t.configure(state=tk.DISABLED)

    def _clear_results(self):
        t = self.result_text
        t.configure(state=tk.NORMAL)
        t.delete("1.0", tk.END)
        t.configure(state=tk.DISABLED)

    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Control-r>", lambda e: self.run_detection())
        self.root.bind("<Escape>", lambda e: self.stop_processing())

    def on_close(self):
        self.stop_processing()
        self.root.destroy()


# ─── Entry ────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
