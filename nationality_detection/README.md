# Nationality Detection System

> **Internship Project Extension** — Predicts nationality from face images with conditional attribute analysis (emotion, age, dress colour) based on nationality group.

---

## Problem Statement

Given a face image, the system:
1. **Predicts nationality** using DeepFace race/ethnicity analysis
2. **Applies conditional prediction rules** based on nationality:

| Nationality | Predictions Made |
|---|---|
| Indian | Nationality + Emotion + Age + **Dress Colour** |
| American | Nationality + Emotion + Age |
| African | Nationality + Emotion + **Dress Colour** |
| Other | Nationality + Emotion only |

3. Displays results in a structured GUI output panel
4. Supports saving results and exporting CSV logs

---

## Project Structure

```
nationality_detection/
│
├── app.py                          # Main GUI (tkinter — editorial theme)
├── cli.py                          # Command-line interface
├── download_models.py              # Downloads age model weights
├── config.py                       # Settings, theme, colour palette, rules
├── requirements.txt
├── nationality_notebook.ipynb      # Colab notebook with full analysis
│
├── utils/
│   ├── predictor.py                # Core pipeline (DeepFace + age + colour)
│   ├── gui_components.py           # Styled widgets (passport card UI)
│   └── logger.py                   # CSV/TXT result export
│
├── models/                         # Downloaded model weights
├── samples/                        # Test images
├── outputs/                        # Saved results
└── tests/
    └── test_predictor.py
```

---

## Setup

```bash
pip install -r requirements.txt
python download_models.py   # Age model ~44 MB; DeepFace auto-downloads on first run
```

---

## Running

### GUI Mode
```bash
python app.py
```

**Features:**
- 📂 Upload Image button — opens file browser
- Live image preview panel (left)
- 🔍 Analyse button — runs full pipeline
- Results shown as structured "passport card" (right)
- Conditional fields shown based on nationality
- 💾 Save Results — exports TXT summary
- Keyboard shortcuts: Ctrl+O (open), Enter (analyse), Esc (clear)

### CLI Mode (Colab-friendly)
```bash
python cli.py --image face.jpg
python cli.py --image face.jpg --output annotated.jpg
python cli.py --image face.jpg --json
python cli.py --batch samples/
python cli.py --image face.jpg --csv log.csv
```

---

## How It Works

### Stage 1: Face Detection
OpenCV Haar Cascade — locates largest face in frame. Used for:
- Passing face crop to DeepFace
- Estimating torso region for dress colour extraction

### Stage 2: Nationality + Emotion + Age — DeepFace
DeepFace wraps several pre-trained models:
- **Race/ethnicity**: VGG-Face fine-tuned on race classification
- **Emotion**: MobileNet trained on FER-2013
- **Age**: DEX (Deep EXpectation) trained on IMDB-WIKI

Returns confidence % per ethnicity group.

**Raw labels → nationality mapping:**
```
indian          → Indian  
black           → African 
white           → American
asian           → East Asian
middle eastern  → Middle Eastern 
latino hispanic → Latino/Hispanic 
```

### Stage 3: Dress Colour — KMeans Clustering
1. Estimates torso region: below face box, ~1.5× face height
2. Samples up to 2000 pixels from that region
3. KMeans(k=3) finds 3 dominant colour clusters
4. Largest cluster centroid → mapped to nearest named colour (34 colour palette) via Euclidean distance in RGB space

### Stage 4: Conditional Logic
```python
TASK_RULES = {
    "indian":   {emotion: True, age: True, dress_colour: True},
    "american": {emotion: True, age: True, dress_colour: False},
    "african":  {emotion: True, age: False, dress_colour: True},
    "other":    {emotion: True, age: False, dress_colour: False},
}
```

---

## GUI Design

Aesthetic: **editorial / travel-document**
- Warm parchment background (`#f0ece4`)
- Deep navy header (`#1a1f36`)
- Georgia serif font for headings (refined, international)
- Courier New for data labels (typewriter / passport feel)
- Results card changes accent colour by nationality group
- Animated loading spinner during analysis

---

## Datasets Used

| Model | Dataset |
|---|---|
| DeepFace (race/ethnicity) | UTKFace, FairFace |
| DeepFace (emotion) | FER-2013 |
| DeepFace (age) | IMDB-WIKI |
| Age DNN (backup) | IMDB-WIKI |
| Dress colour | No dataset — KMeans + RGB distance |

---

## Limitations

- Race/ethnicity prediction is probabilistic — not always accurate
- Dress colour requires torso to be visible in frame
- Age estimation has ±5-8 year margin of error
- DeepFace requires ~200 MB model download on first use

---

## Tech Stack

- **Python 3.10**
- **DeepFace** — race + emotion + age
- **OpenCV** — face detection, image I/O, age DNN
- **scikit-learn** — KMeans for dress colour
- **tkinter** — GUI
- **Pillow** — image rendering
