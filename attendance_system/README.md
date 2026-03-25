#  Smart Attendance System

> **Internship Project Extension** — Face recognition + emotion detection for automated classroom attendance.
> Runs during a configured time window, marks students present/absent, exports Excel + CSV report.

---

## Problem Statement

Manual roll calls are slow, error-prone, and waste class time. This system automates attendance using a webcam:

- **Recognizes students** by face using a trained LBPH model
- **Marks present/absent** automatically — only one entry per student per session
- **Detects emotion** at the moment of recognition (happy, sad, neutral, surprised...)
- **Time-restricted** — only runs between 9:30 AM and 10:00 AM (configurable)
- **Exports** a formatted Excel report + CSV with timestamps

---

## Project Structure

```
attendance_system/
│
├── attendance_system.py        # Main system (run this for attendance)
├── train.py                    # Train the face recognition model
├── config.py                   # All settings in one place
├── attendance_notebook.ipynb   # Experiments, accuracy plots, visualizations
├── requirements.txt
│
├── utils/
│   ├── face_recognizer.py      # Haar Cascade + LBPH recognition wrapper
│   ├── emotion_detector.py     # DeepFace emotion detection (with fallback)
│   ├── attendance_log.py       # Session log (present/absent tracking)
│   └── report_generator.py     # Excel + CSV export with formatting
│
├── dataset/
│   └── known_faces/            # Training data — one folder per student
│       ├── Alice_Smith/
│       │   ├── img001.jpg
│       │   └── ...
│       └── Bob_Jones/
│           └── ...
│
├── models/                     # Saved model files (created by train.py)
├── outputs/                    # Attendance reports saved here
└── tests/
    └── test_attendance.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1: Collect Training Data

Create a folder per student inside `dataset/known_faces/`. Each folder should have **20-50 photos** of that student.

```
dataset/known_faces/
    Alice_Smith/
        img001.jpg
        img002.jpg
        ...
    Bob_Jones/
        img001.jpg
        ...
```

**Or use the built-in webcam capture tool:**

```bash
python train.py --capture
```

This launches an interactive session to capture photos from your webcam.

---

## Step 2: Train the Model

```bash
python train.py
```

This trains an **LBPH (Local Binary Pattern Histogram)** model on your dataset.
Training typically takes 5–30 seconds depending on dataset size.

Options:
```bash
python train.py --dataset dataset/known_faces   # custom dataset path
python train.py --no-augment                    # skip data augmentation
```

---

## Step 3: Run Attendance

```bash
python attendance_system.py
```

The system will:
1. Check if current time is within the 9:30–10:00 AM window
2. Open the webcam and start scanning for faces
3. Recognize students, log presence + emotion
4. At the end of the window (or when you press Q), generate the report

**For testing outside class hours:**
```bash
python attendance_system.py --demo
```

**Use a video file instead of webcam:**
```bash
python attendance_system.py --source classroom_recording.mp4 --demo
```

---

## Output Report

Reports are saved in `outputs/` as:
- `attendance_YYYY-MM-DD.xlsx` — formatted Excel with color coding
- `attendance_YYYY-MM-DD.csv` — flat CSV with all data

**Excel Report contains:**
- Sheet 1: Full attendance table (green=present, red=absent) with time, emotion, confidence
- Sheet 2: Summary stats + emotion breakdown

**Sample output:**

| Student Name | Status | Time In | Emotion | Confidence (%) |
|---|---|---|---|---|
| Alice Smith | ✓ Present | 09:33:21 | Happy | 88.5 |
| Bob Jones | ✓ Present | 09:35:44 | Neutral | 82.1 |
| Charlie Brown | ✗ Absent | — | — | — |

---

## How It Works::

### Face Detection
Haar Cascade Classifier (OpenCV built-in) — fast, reliable for frontal faces in decent lighting.

### Face Recognition — LBPH
**Local Binary Pattern Histogram** is a classic ML algorithm:
1. Divides the face image into a grid of cells
2. For each pixel, compares it to its circular neighborhood → generates an 8-bit binary pattern
3. Builds a histogram of patterns per cell, concatenated into one feature vector
4. At recognition time, compares to stored histograms using Chi-squared distance

Why LBPH?
- Runs on CPU, trains in seconds
- Handles some lighting variation (especially with `equalizeHist`)
- Full control — no black-box neural network
- Meets the "train your own model" requirement

### Emotion Detection
Uses **DeepFace** which wraps a pre-trained FER (Facial Expression Recognition) model. Returns one of: `happy`, `sad`, `angry`, `surprised`, `fear`, `disgust`, `neutral`.

Falls back to a simple OpenCV-based heuristic if DeepFace isn't installed.

### Time Window
Hardcoded in `config.py`:
```python
ATTENDANCE_START = dtime(9, 30)   # 9:30 AM
ATTENDANCE_END   = dtime(10, 0)   # 10:00 AM
```
Outside this window, the system prints the time until next window and exits.

---

## Configuration

Edit `config.py`:

| Setting | Default | Description |
|---|---|---|
| `ATTENDANCE_START` | 9:30 AM | Window start |
| `ATTENDANCE_END` | 10:00 AM | Window end |
| `CONFIDENCE_THRESHOLD` | 50.0 | Min % to accept recognition |
| `FRAME_SKIP` | 3 | Process every Nth frame |
| `DISPLAY_FEED` | True | Show live video window |
| `FACE_SIZE` | (100, 100) | Face resize dimensions |

---

## Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- AttendanceLog (present/absent, no duplicates, summary)
- EmotionDetector (graceful failures, valid outputs)
- Report generation (CSV + Excel export)
- Time window config validation

---

## Model Accuracy

| Dataset size | LBPH Accuracy (approx) |
|---|---|
| 10 images/student | ~70% |
| 30 images/student | ~85% |
| 50+ images/student | ~90%+ |

See `attendance_notebook.ipynb` for full accuracy analysis and per-student breakdown.

---

## Limitations

- LBPH works best with frontal faces in consistent lighting
- Large pose changes (side profile) may not be recognized
- Performance drops with very similar-looking faces
- Emotion detection adds ~0.5s latency per frame

---

## Extensions

- Replace LBPH with dlib 128-d embeddings or FaceNet for higher accuracy
- Add MTCNN for better detection in crowded classrooms
- Web dashboard for weekly attendance reports
- SMS/email alerts for consistent absentees

---

## Tech Stack

- **Python 3.10**
- **OpenCV + opencv-contrib** — face detection + LBPH recognition
- **DeepFace** — emotion detection
- **openpyxl** — formatted Excel export
- **NumPy / pandas** — data handling
