"""
attendance_system.py
=====================
Smart Classroom Attendance System
===================================

Internship Task Extension — Attendance System using Face Recognition + Emotion Detection

What this does:
-  Recognizes registered students via webcam (or video file)
-  Marks them Present / Absent at the end of session
-  Detects their emotion at the moment of recognition
-  Only runs during a configured time window (default: 9:30 AM – 10:00 AM)
-  Saves a full attendance report to Excel + CSV with timestamps

How it works:
- Face detection: OpenCV Haar Cascade (fast, no GPU needed)
- Face recognition: LBPH (Local Binary Pattern Histogram) — trained on your dataset
- Emotion detection: DeepFace (wraps FER model under the hood)
- Output: openpyxl-formatted Excel report

Author: [Your Name]
Date: 2025
"""

import cv2
import os
import sys
import time
import logging
from datetime import datetime, time as dtime

# Local imports
from utils.face_recognizer import FaceRecognizer
from utils.emotion_detector import EmotionDetector
from utils.attendance_log import AttendanceLog
from utils.report_generator import generate_excel_report
from config import (
    ATTENDANCE_START, ATTENDANCE_END,
    WEBCAM_ID, CONFIDENCE_THRESHOLD,
    OUTPUT_DIR, STUDENT_DATASET_DIR,
    FRAME_SKIP, DISPLAY_FEED,
)

# ─── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("attendance_run.log"),
    ]
)
log = logging.getLogger(__name__)


# ─── Time window check ────────────────────────────────────────────────────────

def is_within_window() -> bool:
    """
    Returns True only if current time is within the attendance window.
    Outside this window, the system simply won't run.
    """
    now = datetime.now().time()
    return ATTENDANCE_START <= now <= ATTENDANCE_END


def time_until_window() -> str:
    """Returns a human-readable string for how long until the window opens."""
    now = datetime.now()
    start_today = now.replace(
        hour=ATTENDANCE_START.hour,
        minute=ATTENDANCE_START.minute,
        second=0, microsecond=0
    )
    if now < start_today:
        diff = start_today - now
        mins = int(diff.total_seconds() / 60)
        return f"{mins} minutes"
    return "window has passed for today"


# ─── Main system ──────────────────────────────────────────────────────────────

class AttendanceSystem:
    """
    Core attendance system orchestrator.

    Flow:
    1. Load trained face recognition model
    2. Open webcam / video source
    3. For each frame: detect faces → recognize → detect emotion → log
    4. At end: generate Excel + CSV report
    """

    def __init__(self, source=None, demo_mode: bool = False):
        """
        source: None = webcam, or path to video file for testing
        demo_mode: skips time window check (for testing outside class hours)
        """
        self.source = source if source is not None else WEBCAM_ID
        self.demo_mode = demo_mode
        self.running = False

        log.info("Initializing Attendance System...")

        # Load components
        self.recognizer = FaceRecognizer()
        self.emotion_detector = EmotionDetector()
        self.attendance_log = AttendanceLog()

        # Load student roster from dataset folder
        self.student_names = self._load_student_roster()
        log.info(f"Roster loaded: {len(self.student_names)} students registered")

    def _load_student_roster(self) -> list:
        """
        Reads the known_faces directory to build the student roster.
        Each subfolder = one student. Folder name = student name.
        """
        if not os.path.exists(STUDENT_DATASET_DIR):
            log.warning(f"Dataset directory not found: {STUDENT_DATASET_DIR}")
            return []

        students = [
            name for name in os.listdir(STUDENT_DATASET_DIR)
            if os.path.isdir(os.path.join(STUDENT_DATASET_DIR, name))
            and not name.startswith(".")
        ]
        return sorted(students)

    def run(self):
        """Main loop. Runs the attendance session."""

        # ── Time window check ────────────────────────────────────────────────
        if not self.demo_mode and not is_within_window():
            log.warning(
                f"Outside attendance window ({ATTENDANCE_START.strftime('%I:%M %p')} – "
                f"{ATTENDANCE_END.strftime('%I:%M %p')}). "
                f"Window opens in {time_until_window()}."
            )
            print(f"\n⏰ Attendance window is {ATTENDANCE_START.strftime('%I:%M %p')} – "
                  f"{ATTENDANCE_END.strftime('%I:%M %p')}")
            print(f"   Current time: {datetime.now().strftime('%I:%M %p')}")
            print(f"   Opens in: {time_until_window()}\n")
            return

        log.info(f"Starting attendance session — source: {self.source}")
        print(f"\n{'='*55}")
        print(f"  🎓 ATTENDANCE SYSTEM — {datetime.now().strftime('%A, %d %B %Y')}")
        print(f"  Window: {ATTENDANCE_START.strftime('%I:%M %p')} – {ATTENDANCE_END.strftime('%I:%M %p')}")
        print(f"  Students registered: {len(self.student_names)}")
        print(f"{'='*55}\n")

        # ── Check model ──────────────────────────────────────────────────────
        if not self.recognizer.is_trained():
            print("⚠️  Face recognition model not trained yet!")
            print("   Run: python train.py\n")
            log.error("Model not trained. Aborting.")
            return

        # ── Open video source ────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error(f"Cannot open source: {self.source}")
            print(f"❌ Cannot open camera/video: {self.source}")
            return

        self.running = True
        frame_count = 0
        session_start = datetime.now()

        print("▶  Running... Press Q to stop early.\n")

        try:
            while self.running:

                # Stop if we've gone past the window (live mode)
                if not self.demo_mode and not is_within_window():
                    log.info("Attendance window closed. Stopping.")
                    print("\n⏰ Attendance window closed.")
                    break

                ret, frame = cap.read()
                if not ret:
                    log.info("End of video / camera disconnected.")
                    break

                frame_count += 1

                # Skip frames for performance
                if frame_count % FRAME_SKIP != 0:
                    if DISPLAY_FEED:
                        cv2.imshow("Attendance System — press Q to quit", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue

                # ── Core processing ──────────────────────────────────────────

                # 1. Detect faces in frame
                faces = self.recognizer.detect_faces(frame)

                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue

                    # 2. Recognize who this is
                    student_name, confidence = self.recognizer.recognize(face_roi)

                    if confidence < CONFIDENCE_THRESHOLD:
                        # Unknown face
                        label = "Unknown"
                        color = (100, 100, 100)
                    else:
                        label = student_name
                        color = (80, 200, 60)  # green

                        # 3. Detect emotion
                        emotion = self.emotion_detector.detect(face_roi)

                        # 4. Log attendance (won't double-log same student)
                        self.attendance_log.mark_present(
                            student_name=student_name,
                            emotion=emotion,
                            timestamp=datetime.now(),
                        )

                        label = f"{student_name} ({emotion})"
                        log.info(f"  ✓ {student_name} — {emotion}")

                    # 5. Draw on frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Overlay: time remaining + count
                self._draw_overlay(frame)

                # Show feed (optional)
                if DISPLAY_FEED:
                    cv2.imshow("Attendance System — press Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user.")
                        break

        except KeyboardInterrupt:
            print("\nInterrupted.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False

        # ── Finalize & export ────────────────────────────────────────────────
        session_end = datetime.now()
        duration = (session_end - session_start).seconds // 60

        log.info(f"Session ended. Duration: {duration} min")
        self._finalize(session_start, session_end)

    def _draw_overlay(self, frame):
        """
        Draws a small info overlay on the frame:
        - Current time
        - How many students marked present so far
        """
        now_str = datetime.now().strftime("%H:%M:%S")
        present_count = self.attendance_log.present_count()
        total = len(self.student_names)

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (260, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Time: {now_str}", (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Present: {present_count}/{total}", (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 60), 1)

    def _finalize(self, session_start: datetime, session_end: datetime):
        """
        Called at end of session.
        Marks absent students and generates the Excel + CSV report.
        """
        print(f"\n{'='*55}")
        print("  📊 SESSION COMPLETE — Generating Report...")
        print(f"{'='*55}")

        # Mark everyone not seen as absent
        for student in self.student_names:
            if not self.attendance_log.is_present(student):
                self.attendance_log.mark_absent(student)

        # Print summary to console
        summary = self.attendance_log.get_summary()
        print(f"\n  Students registered : {summary['total']}")
        print(f"  Present             : {summary['present']} ✓")
        print(f"  Absent              : {summary['absent']} ✗")
        print(f"  Attendance rate     : {summary['rate']:.1f}%\n")

        print("  Present:")
        for name, data in self.attendance_log.records.items():
            if data["status"] == "present":
                t = data["time_in"].strftime("%H:%M:%S")
                print(f"    ✓ {name:25s} {t}  [{data['emotion']}]")

        print("\n  Absent:")
        for name, data in self.attendance_log.records.items():
            if data["status"] == "absent":
                print(f"    ✗ {name}")

        # Generate reports
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        date_str = session_start.strftime("%Y-%m-%d")

        excel_path = os.path.join(OUTPUT_DIR, f"attendance_{date_str}.xlsx")
        csv_path   = os.path.join(OUTPUT_DIR, f"attendance_{date_str}.csv")

        generate_excel_report(
            records=self.attendance_log.records,
            session_start=session_start,
            session_end=session_end,
            excel_path=excel_path,
            csv_path=csv_path,
        )

        print(f"\n  📁 Saved:")
        print(f"     Excel : {excel_path}")
        print(f"     CSV   : {csv_path}")
        print(f"\n{'='*55}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="🎓 Attendance System — Face Recognition + Emotion Detection"
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Video file path (default: webcam)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode — skip time window check")
    args = parser.parse_args()

    source = args.source
    if source is None:
        source = WEBCAM_ID
    elif source.isdigit():
        source = int(source)

    system = AttendanceSystem(source=source, demo_mode=args.demo)
    system.run()


if __name__ == "__main__":
    main()
