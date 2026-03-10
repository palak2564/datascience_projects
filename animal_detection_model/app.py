"""
Animal Detection & Classification System
=========================================
Main GUI application — built with tkinter + OpenCV + YOLOv8 (via ultralytics)

Author: [Your Name]
Date: 2025
Internship Project Extension — Animal Detection Task

What this does:
- Detects animals in images and videos using YOLOv8
- Highlights carnivores in RED bounding boxes, herbivores in GREEN
- Shows a popup if carnivores are detected with count
- Has a clean GUI with image/video previews
- Logs all detections with timestamps
- Saves annotated outputs automatically
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import time
from datetime import datetime

# Local imports
from utils.detector import AnimalDetector
from utils.gui_components import DetectionResultPanel, StatusBar, AnimatedButton
from utils.logger import DetectionLogger
from utils.carnivore_alert import show_carnivore_alert


# ─── Constants ────────────────────────────────────────────────────────────────

APP_TITLE = "🐾 Animal Detector — Wildlife AI"
WINDOW_SIZE = "1200x800"
THEME_BG = "#1a1a2e"
THEME_SURFACE = "#16213e"
THEME_ACCENT = "#e94560"
THEME_TEXT = "#eaeaea"
THEME_SUCCESS = "#4ecca3"


# ─── Main Application Window ──────────────────────────────────────────────────

class AnimalDetectionApp:
    """
    Main application class. Handles all GUI logic and ties together
    the detector, logger, and display components.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=THEME_BG)
        self.root.minsize(900, 600)

        # State
        self.current_file = None
        self.is_processing = False
        self.video_running = False
        self.detection_history = []

        # Core components
        self.detector = AnimalDetector()
        self.logger = DetectionLogger()

        # Build the UI
        self._setup_ui()
        self._bind_shortcuts()

        # Welcome message in status
        self.status_bar.set_message("Ready! Load an image or video to get started 🐾")

    # ── UI Construction ────────────────────────────────────────────────────────

    def _setup_ui(self):
        """Builds the full UI layout."""
        # Top bar
        self._build_header()

        # Main content area (2 columns)
        content = tk.Frame(self.root, bg=THEME_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        content.columnconfigure(0, weight=3)  # preview takes more space
        content.columnconfigure(1, weight=1)  # sidebar

        # Left — preview panel
        self._build_preview_panel(content)

        # Right — controls + results
        self._build_sidebar(content)

        # Bottom — status bar
        self.status_bar = StatusBar(self.root, bg=THEME_SURFACE, fg=THEME_TEXT)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_header(self):
        header = tk.Frame(self.root, bg=THEME_SURFACE, height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        # Logo / title
        title_lbl = tk.Label(
            header,
            text="🐾  Animal Detection AI",
            font=("Courier New", 18, "bold"),
            bg=THEME_SURFACE,
            fg=THEME_ACCENT,
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=10)

        # Model status indicator
        model_status = tk.Label(
            header,
            text="● Model Ready",
            font=("Courier New", 10),
            bg=THEME_SURFACE,
            fg=THEME_SUCCESS,
        )
        model_status.pack(side=tk.RIGHT, padx=20)

        # Subtitle
        subtitle = tk.Label(
            header,
            text="YOLOv8 · Multi-species · Real-time",
            font=("Courier New", 9),
            bg=THEME_SURFACE,
            fg="#888",
        )
        subtitle.pack(side=tk.LEFT, padx=0, pady=15)

    def _build_preview_panel(self, parent):
        """Left side — where we show the image/video preview."""
        self.preview_frame = tk.Frame(parent, bg=THEME_SURFACE, relief=tk.FLAT)
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        # Preview label with placeholder
        self.preview_label = tk.Label(
            self.preview_frame,
            text="📂  Drop an image or video here\nor use the buttons →",
            font=("Courier New", 13),
            bg=THEME_SURFACE,
            fg="#555",
            cursor="hand2",
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Drag & drop hint
        self.preview_label.bind("<Button-1>", lambda e: self.load_image())

    def _build_sidebar(self, parent):
        """Right sidebar — buttons, settings, detection log."""
        sidebar = tk.Frame(parent, bg=THEME_BG, width=300)
        sidebar.grid(row=0, column=1, sticky="nsew", pady=5)
        sidebar.pack_propagate(False)

        # ── Load buttons
        load_section = self._make_section(sidebar, "📂 Load Media")

        btn_image = AnimatedButton(
            load_section,
            text="🖼  Load Image",
            command=self.load_image,
            bg="#2d2d5e",
            fg=THEME_TEXT,
            hover_bg=THEME_ACCENT,
        )
        btn_image.pack(fill=tk.X, pady=3)

        btn_video = AnimatedButton(
            load_section,
            text="🎬  Load Video",
            command=self.load_video,
            bg="#2d2d5e",
            fg=THEME_TEXT,
            hover_bg=THEME_ACCENT,
        )
        btn_video.pack(fill=tk.X, pady=3)

        btn_webcam = AnimatedButton(
            load_section,
            text="📷  Live Webcam",
            command=self.start_webcam,
            bg="#2d2d5e",
            fg=THEME_TEXT,
            hover_bg="#f5a623",
        )
        btn_webcam.pack(fill=tk.X, pady=3)

        # ── Detection controls
        det_section = self._make_section(sidebar, "🔍 Detection")

        self.conf_var = tk.DoubleVar(value=0.45)
        tk.Label(det_section, text="Confidence threshold:", bg=THEME_BG, fg=THEME_TEXT,
                 font=("Courier New", 8)).pack(anchor="w")
        conf_slider = ttk.Scale(det_section, from_=0.1, to=0.9,
                                variable=self.conf_var, orient=tk.HORIZONTAL)
        conf_slider.pack(fill=tk.X, pady=2)

        self.conf_label = tk.Label(det_section, text="0.45", bg=THEME_BG, fg=THEME_ACCENT,
                                   font=("Courier New", 8))
        self.conf_label.pack(anchor="e")
        self.conf_var.trace("w", self._update_conf_label)

        self.show_labels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            det_section, text="Show labels", variable=self.show_labels_var,
            bg=THEME_BG, fg=THEME_TEXT, selectcolor="#333",
            activebackground=THEME_BG, font=("Courier New", 9)
        ).pack(anchor="w", pady=2)

        self.show_conf_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            det_section, text="Show confidence", variable=self.show_conf_var,
            bg=THEME_BG, fg=THEME_TEXT, selectcolor="#333",
            activebackground=THEME_BG, font=("Courier New", 9)
        ).pack(anchor="w", pady=2)

        btn_run = AnimatedButton(
            det_section,
            text="▶  Run Detection",
            command=self.run_detection,
            bg=THEME_ACCENT,
            fg="white",
            hover_bg="#c73652",
            font_size=11,
        )
        btn_run.pack(fill=tk.X, pady=6)

        self.stop_btn = AnimatedButton(
            det_section,
            text="⏹  Stop",
            command=self.stop_processing,
            bg="#333",
            fg="#aaa",
            hover_bg="#555",
        )
        self.stop_btn.pack(fill=tk.X, pady=2)

        # ── Results panel
        res_section = self._make_section(sidebar, "📊 Detection Log")
        self.result_panel = DetectionResultPanel(res_section, bg=THEME_SURFACE)
        self.result_panel.pack(fill=tk.BOTH, expand=True)

        # ── Save / export
        save_section = self._make_section(sidebar, "💾 Export")
        AnimatedButton(
            save_section, text="💾  Save Result",
            command=self.save_result,
            bg="#1f4037", fg=THEME_TEXT, hover_bg=THEME_SUCCESS
        ).pack(fill=tk.X, pady=2)

        AnimatedButton(
            save_section, text="📋  Export Log (CSV)",
            command=self.export_log,
            bg="#1f4037", fg=THEME_TEXT, hover_bg=THEME_SUCCESS
        ).pack(fill=tk.X, pady=2)

    def _make_section(self, parent, title: str) -> tk.Frame:
        """Helper to create a labeled section box in the sidebar."""
        wrapper = tk.Frame(parent, bg=THEME_BG)
        wrapper.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(
            wrapper, text=title,
            font=("Courier New", 9, "bold"),
            bg=THEME_BG, fg=THEME_ACCENT
        ).pack(anchor="w", pady=(0, 3))

        inner = tk.Frame(wrapper, bg=THEME_BG)
        inner.pack(fill=tk.X)

        return inner

    # ── Event Handlers ─────────────────────────────────────────────────────────

    def load_image(self):
        """Opens file dialog and loads an image."""
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        self.current_file = path
        self.current_mode = "image"
        self._show_image_preview(path)
        self.status_bar.set_message(f"Loaded: {os.path.basename(path)}")
        self.result_panel.clear()

    def load_video(self):
        """Opens file dialog and loads a video."""
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if not path:
            return
        self.current_file = path
        self.current_mode = "video"
        self.status_bar.set_message(f"Video loaded: {os.path.basename(path)} — press Run Detection")
        self._show_video_thumbnail(path)

    def start_webcam(self):
        """Starts webcam live detection."""
        self.current_file = 0  # OpenCV uses 0 for default cam
        self.current_mode = "webcam"
        self.status_bar.set_message("Webcam mode — press Run Detection to start")

    def run_detection(self):
        """Kicks off detection in a background thread."""
        if self.is_processing:
            self.status_bar.set_message("Already running... hit Stop first!")
            return
        if self.current_file is None:
            messagebox.showwarning("No file", "Please load an image or video first!")
            return

        self.is_processing = True
        self.status_bar.set_message("🔍 Running detection...")

        thread = threading.Thread(target=self._detection_worker, daemon=True)
        thread.start()

    def _detection_worker(self):
        """Background worker that calls the detector and updates UI."""
        try:
            conf = self.conf_var.get()
            mode = getattr(self, "current_mode", "image")

            if mode == "image":
                result = self.detector.detect_image(
                    self.current_file,
                    confidence=conf,
                    show_labels=self.show_labels_var.get(),
                    show_conf=self.show_conf_var.get(),
                )
                self.root.after(0, lambda: self._handle_image_result(result))

            elif mode in ("video", "webcam"):
                self.video_running = True
                self.detector.detect_video(
                    self.current_file,
                    confidence=conf,
                    frame_callback=self._on_video_frame,
                    stop_flag=lambda: not self.video_running,
                )
                self.root.after(0, lambda: self.status_bar.set_message("Video processing done ✓"))

        except Exception as e:
            self.root.after(0, lambda: self.status_bar.set_message(f"Error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Detection Error", str(e)))
        finally:
            self.is_processing = False

    def _handle_image_result(self, result):
        """Called on the main thread after image detection finishes."""
        if result is None:
            return

        # Show annotated image
        self._display_annotated_image(result["annotated_frame"])

        # Update result panel
        self.result_panel.update(result["detections"])

        # Log it
        self.logger.log(result["detections"], source=self.current_file)
        self.detection_history.append(result)

        # Carnivore alert?
        carnivores = [d for d in result["detections"] if d["is_carnivore"]]
        if carnivores:
            self.root.after(100, lambda: show_carnivore_alert(self.root, len(carnivores), carnivores))

        # Summary
        total = len(result["detections"])
        self.status_bar.set_message(
            f"Done! Found {total} animal(s) — {len(carnivores)} carnivore(s) ⚠️" if carnivores
            else f"Done! Found {total} animal(s) — no carnivores detected ✓"
        )

    def _on_video_frame(self, frame, detections):
        """Called for each video frame — updates preview."""
        self.root.after(0, lambda: self._display_annotated_image(frame))
        carnivores = [d for d in detections if d["is_carnivore"]]
        msg = f"Video: {len(detections)} animal(s)"
        if carnivores:
            msg += f" — ⚠️ {len(carnivores)} CARNIVORE(S)"
        self.root.after(0, lambda: self.status_bar.set_message(msg))
        # Log every N frames only (avoid spamming)

    def stop_processing(self):
        self.video_running = False
        self.is_processing = False
        self.status_bar.set_message("Stopped.")

    def save_result(self):
        """Save the annotated result image to disk."""
        if not self.detection_history:
            messagebox.showinfo("Nothing to save", "Run detection first!")
            return
        last = self.detection_history[-1]
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        if path:
            import cv2
            cv2.imwrite(path, last["annotated_frame"])
            self.status_bar.set_message(f"Saved to {path} ✓")

    def export_log(self):
        """Export detection log as CSV."""
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if path:
            self.logger.export_csv(path)
            self.status_bar.set_message(f"Log exported to {path} ✓")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _show_image_preview(self, path: str):
        """Loads and displays image thumbnail in preview area."""
        try:
            from PIL import Image, ImageTk
            img = Image.open(path)
            img.thumbnail((800, 550))
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo  # keep reference!
        except Exception as e:
            self.preview_label.configure(text=f"Preview failed: {e}", image="")

    def _show_video_thumbnail(self, path: str):
        """Grabs first frame from video as preview thumbnail."""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self._display_annotated_image(frame)
        except Exception:
            self.preview_label.configure(text="Video loaded — press Run Detection")

    def _display_annotated_image(self, frame):
        """Shows a numpy frame (BGR) in the preview label."""
        try:
            from PIL import Image, ImageTk
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((800, 550))
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo
        except Exception as e:
            print(f"[display] Could not render frame: {e}")

    def _update_conf_label(self, *args):
        self.conf_label.configure(text=f"{self.conf_var.get():.2f}")

    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Control-r>", lambda e: self.run_detection())
        self.root.bind("<Escape>", lambda e: self.stop_processing())

    def on_close(self):
        self.stop_processing()
        self.root.destroy()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = AnimalDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
