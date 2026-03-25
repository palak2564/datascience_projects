# Drowsiness Detection System

> Detects multiple people, marks sleeping ones in RED, predicts their age, fires popup alerts.

---

## 📋 Problem Statement

Driver/passenger drowsiness causes thousands of road accidents annually. This system uses computer vision to:

- Detect **multiple people** in a single image or video frame
- Classify each as **AWAKE** (green box) or **SLEEPING** (red box)
- Predict **estimated age** for each person
- Show a **popup alert** with count of sleeping people and their ages
- Support **image, video file, and live webcam** input
- Full **GUI with live preview**

---

## Project Structure

```
drowsiness_detection/
│
├── app.py                          # Main GUI application (tkinter)
├── cli.py                          # Command-line interface
├── download_models.py              # Downloads pre-trained model weights
├── config.py                       # All settings
├── requirements.txt
├── drowsiness_notebook.ipynb       # Analysis, EAR theory, visualizations
│
├── utils/
│   ├── detector.py                 # Core pipeline: EAR + age estimation
│   ├── gui_components.py           # Custom dark-theme tkinter widgets
│   ├── alert.py                    # Drowsiness popup alert window
│   └── logger.py                   # CSV detection log
│
├── models/                         # Model weights (created by download_models.py)
├── samples/                        # Test images
├── outputs/                        # Saved results
└── tests/
    └── test_detector.py
```

---

## Setup

```bash
pip install -r requirements.txt
python download_models.py   # Downloads age estimation model (~44 MB)
```

### Optional: dlib landmark model (better accuracy)
```
1. Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Extract the .dat file
3. Place at: models/shape_predictor_68_face_landmarks.dat
```
Without it, the system falls back to OpenCV eye cascade (still works, slightly less accurate).

---

## Running the App

### GUI Mode
```bash
python app.py
```

**Features:**
- Load Image / Load Video / Live Webcam buttons
- Adjustable EAR threshold slider (default: 0.25)
- Toggle: landmark overlay, age display, popup alerts
- Live annotated preview
- Detection log in sidebar
- Save annotated result / Export CSV

### CLI Mode
```bash
python cli.py --image dashcam.jpg
python cli.py --image dashcam.jpg --output result.jpg
python cli.py --video footage.mp4 --show
python cli.py --webcam --show
python cli.py --image photo.jpg --ear 0.22 --csv log.csv
```

---

## How It Works

### Stage 1: Face Detection
OpenCV Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) — fast and reliable for frontal faces in vehicle dashcam scenarios.

### Stage 2: Drowsiness Detection via EAR
**Eye Aspect Ratio** (Soukupova & Cech, 2016):

```
EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
```

Where P1–P6 are the 6 eye landmark points.

- When eyes **open**: EAR ≈ 0.30–0.40
- When eyes **closed**: EAR drops below threshold (~0.25)

If EAR < threshold → person marked as sleeping.

**With dlib**: Uses 68 facial landmarks for accurate eye geometry.  
**Without dlib**: Falls back to OpenCV eye cascade + height/width ratio estimation.

### Stage 3: Age Estimation
**Levi & Hassner (2015)** deep CNN trained on IMDB-WIKI dataset.
- Input: 227×227 face crop
- Output: one of 8 age brackets: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- Accuracy: ~75-88% correct bracket prediction

### Visualization
| Color | Meaning |
|---|---|
| 🔴 Red box | Sleeping / drowsy |
| 🟢 Green box | Awake |
| Text label | State + estimated age |
| EAR value | Eye aspect ratio beneath box |
| Red ring | Pulse ring for sleeping people |

---

## Popup Alert

When sleeping people are detected, a modal popup fires automatically:
- Bold red warning design
- Count of sleeping people
- Age estimate per person
- Auto-dismisses after 10 seconds
- "PULL OVER IMMEDIATELY" safety message

---

## Configuration

Edit `config.py`:

| Setting | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | 0.25 | Eye closure threshold |
| `EAR_CONSEC_FRAMES` | 3 | Frames to confirm sleeping (video) |
| `FRAME_SKIP` | 2 | Process every Nth frame |
| `AGE_BUCKETS` | 8 brackets | Age classification labels |

---

## Model Performance

| Metric | Value |
|---|---|
| Drowsiness detection (EAR) | ~90% F1 at threshold 0.25 |
| Age estimation accuracy | ~75-88% per bracket |
| Detection speed | ~30 FPS (webcam, no age) / ~8 FPS (with age) |

See `drowsiness_notebook.ipynb` for full EAR threshold analysis, ROC curves, and age estimation confusion matrix.

---

## Limitations

- Haar cascade may miss faces in low light or side profiles
- EAR approach requires frontal/near-frontal face view
- Sunglasses or heavy eye makeup can affect EAR reading
- Age prediction is one of 8 broad brackets, not exact age

---

## Extensions

- Add head pose estimation to catch nodding (secondary drowsiness signal)
- Replace Haar with MTCNN for better multi-face detection
- Add audio alarm (beep on detection)
- ONNX export for mobile/embedded deployment in vehicles

---

## Tech Stack

- **Python 3.10**
- **OpenCV** — Haar cascade detection, DNN age model
- **dlib** (optional) — 68-point facial landmarks
- **tkinter** — GUI
- **Pillow** — image rendering
- **Matplotlib / Seaborn** — notebook visualizations
