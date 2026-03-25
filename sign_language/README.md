# 🤟 Sign Language Detection System

> **Internship Project Extension** — Real-time ASL sign language recognition using MediaPipe hand landmarks + neural network. Supports image upload and live webcam. Active 6:00 PM – 10:00 PM only.

---

## 📋 Problem Statement

Build a system that recognises hand signs in real-time, identifies known ASL words/letters, and operates within a restricted time window.

**Labels recognised (36 total):**
- 🔤 A–Z (26 ASL alphabet letters)
- 💬 10 custom words: `Hello`, `ThankYou`, `Yes`, `No`, `Please`, `Sorry`, `Help`, `Eat`, `Water`, `More`

**Operating hours:** 6:00 PM – 10:00 PM (configurable in `config.py`)

---

## 🗂 Project Structure

```
sign_language/
│
├── app.py                          # Main GUI (neon/HUD dark theme)
├── train.py                        # Model training + webcam capture tool
├── cli.py                          # CLI for Colab/headless use
├── config.py                       # Labels, time window, theme, paths
├── requirements.txt
├── sign_language_notebook.ipynb    # Colab notebook with full pipeline
│
├── utils/
│   ├── detector.py                 # MediaPipe + model inference
│   ├── gui_components.py           # Neon-themed custom widgets
│   └── logger.py                   # CSV prediction log
│
├── dataset/
│   └── train/                      # Training images per class
│       ├── A/ *.jpg
│       ├── Hello/ *.jpg
│       └── ...
│
├── models/                         # Saved model files
│   ├── sign_model.h5               # Keras NN (created by train.py)
│   ├── sign_svm.pkl                # SVM fallback
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   ├── training_history.png
│   └── confusion_matrix.png
│
└── tests/
    └── test_sign_language.py
```

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

---

## 📸 Step 1: Get Training Data

**Option A — Kaggle ASL Alphabet Dataset (recommended):**
```bash
pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d dataset/
```

**Option B — Capture your own via webcam:**
```bash
python train.py --capture
# Interactive tool: type label name → press SPACE to capture → Q to next label
```

---

## 🏋️ Step 2: Train the Model

```bash
python train.py
python train.py --epochs 80 --batch 64   # more epochs for better accuracy
```

Saves to `models/`:
- `sign_model.h5` — Keras neural network
- `label_encoder.pkl` — class name decoder
- `scaler.pkl` — feature normaliser
- `training_history.png` — accuracy/loss plots
- `confusion_matrix.png` — per-class accuracy

---

## 🖥 Step 3: Run the App

```bash
python app.py
```

### GUI Features:
| Feature | Description |
|---|---|
| Image Upload mode | Load any hand sign photo → detect |
| Live Webcam mode | Real-time detection from camera |
| Confidence slider | Adjust minimum confidence threshold |
| Typewriter animation | Prediction "types in" character by character |
| Confidence bar | Colour-coded: green/amber/red |
| History log | Timestamped record of all predictions |
| Export CSV | Save full session log |
| Time window badge | Live ● ACTIVE / ● INACTIVE indicator |

### CLI Mode (Colab-friendly):
```bash
python cli.py --image hand.jpg --demo    # --demo bypasses time window
python cli.py --image hand.jpg --output result.jpg
python cli.py --video signs.mp4 --demo
python cli.py --webcam --demo
python cli.py --image hand.jpg --csv log.csv
```

---

## ⏰ Time Window

The system **only accepts input between 6:00 PM and 10:00 PM**.

Outside this window:
- GUI shows a warning popup
- Webcam start is blocked
- Image detection is blocked

To override for testing:
```bash
python cli.py --image test.jpg --demo     # CLI: use --demo
```

Or change in `config.py`:
```python
ACTIVE_START = dtime(0, 0)    # midnight
ACTIVE_END   = dtime(23, 59)  # always active
```

---

## 🧠 How It Works

### Stage 1: Hand Detection — MediaPipe
MediaPipe Hands detects the hand and returns 21 3D landmarks (x, y, z coordinates).

### Stage 2: Feature Extraction
The 21 landmarks × 3 coordinates = **63 features**.
Normalised relative to wrist (landmark 0) to be translation-invariant.

```
Feature vector = [
  (lm1.x - wrist.x, lm1.y - wrist.y, lm1.z - wrist.z),
  (lm2.x - wrist.x, ...),
  ...  ×21 landmarks
]
```

### Stage 3: Classification — Neural Network
```
Input (63)
  → Dense(256) + BatchNorm + ReLU + Dropout(0.4)
  → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  → Dense(64) + ReLU + Dropout(0.2)
  → Dense(36, softmax)
Output: probabilities for each of 36 signs
```

Fallback: **SVM with RBF kernel** if TensorFlow not available.

### Why landmarks, not raw images?
- Robust to lighting, background, skin tone
- 63 floats vs 224×224×3 = ~150,000 pixels
- Trains in minutes on CPU
- Generalises better to unseen people

---

## 📊 Datasets

| Component | Dataset |
|---|---|
| ASL Alphabet (A-Z) | Kaggle: `grassknoted/asl-alphabet` (87K images) |
| Custom words | Self-captured via `python train.py --capture` |
| Hand landmarks | MediaPipe (pretrained, no dataset needed) |

---

## ⚠ Limitations

- Only frontal/near-frontal hand poses work well
- Two-handed signs not supported (single hand only)
- Dark environments may cause MediaPipe detection failure
- Some similar signs (e.g. M/N, R/U) may be confused

---

## 🛠 Tech Stack

- **Python 3.10**
- **MediaPipe** — hand landmark detection
- **TensorFlow / Keras** — neural network classifier
- **scikit-learn** — SVM fallback, preprocessing, metrics
- **OpenCV** — image/video I/O
- **tkinter** — GUI (neon HUD theme)
- **Pillow** — image rendering
