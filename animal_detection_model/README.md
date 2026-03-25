# Animal Detection & Classification System 

> **Internship Project Extension** — Built on the training project base.
> Detects animals in images and videos, classifies dietary type, and alerts on carnivores.



## Problem Statement

Wildlife monitoring, zoo management, and ecological research all need automated tools to identify animals in images. Manual annotation is slow. This project uses **YOLOv8** to:

* Detect multiple animals in a single frame
* Classify each as **carnivore**, **herbivore**, or **omnivore**
* Highlight carnivores with **RED bounding boxes**
* Show a **popup alert** with the count of carnivores detected
* Support both images and video input (including live webcam)
* Export detection logs as CSV for analysis



## Demo

| Image Detection                            | Carnivore Alert                   |
| ------------------------------------------ | --------------------------------- |
| Green boxes = herbivores, Red = carnivores | Popup fires when carnivores found |



## Project Structure

```
animal_detection/

│
├── app.py                          # Main GUI application (tkinter)
├── cli.py                          # Command-line interface (no GUI)
├── requirements.txt
├── animal_detection_notebook.ipynb # Experiments, charts, model comparison
│
├── utils/
│   ├── detector.py                 # YOLOv8 wrapper + animal classification
│   ├── gui_components.py           # Custom tkinter widgets
│   ├── carnivore_alert.py          # Popup alert window
│   └── logger.py                   # Detection logging + CSV export
│
├── samples/                        # Put test images here
└── outputs/                        # Saved results go here
```



## Setup

### Requirements

* Python 3.8+
* pip

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

* `ultralytics` — YOLOv8 framework
* `opencv-python` — image/video processing
* `Pillow` — tkinter image display
* `matplotlib`, `seaborn` — notebook visualizations



## Running the App

### GUI Mode (recommended)

```bash
python app.py
```

**Features:**

* Load Image button — runs detection on any JPG/PNG
* Load Video button — processes video frame by frame
* Live Webcam — real-time detection
* Adjustable confidence threshold slider
* Live detection log in sidebar
* Save annotated result / Export CSV log

### CLI Mode (batch/headless)

```bash
# Detect animals in an image
python cli.py --image zoo.jpg

# Save annotated output
python cli.py --image zoo.jpg --output result.jpg

# Process a video file
python cli.py --video wildlife.mp4 --show

# Use a more accurate (but slower) model
python cli.py --image zoo.jpg --model yolov8s

# Export detection log
python cli.py --image zoo.jpg --export-csv detections.csv
```



## Methodology

### Model: YOLOv8 (Ultralytics)

YOLOv8 is a state-of-the-art, real-time object detection model. We use the pretrained COCO weights which include 10 animal classes out of the box:

| Class    | Type      |
| -------- | --------- |
| cat      | Carnivore |
| dog      | Omnivore  |
| bird     | Carnivore |
| bear     | Carnivore |
| horse    | Herbivore |
| sheep    | Herbivore |
| cow      | Herbivore |
| elephant | Herbivore |
| zebra    | Herbivore |
| giraffe  | Herbivore |

### Carnivore Classification

Carnivore detection uses a manually curated lookup table (`CARNIVORES` set in `detector.py`). A production system could use:

* A taxonomy API (e.g. GBIF, iNaturalist)
* A fine-tuned classification head trained on dietary behavior

### Bounding Box Colors

| Color  | Meaning   |
| ------ | --------- |
| Red    | Carnivore |
| Yellow | Omnivore  |
| Green  | Herbivore |

### Model Size Tradeoffs

| Model            | Speed | Accuracy | Use case           |
| ---------------- | ----- | -------- | ------------------ |
| yolov8n (nano)   | ~5ms  | Good     | Webcam / real-time |
| yolov8s (small)  | ~10ms | Better   | Video files        |
| yolov8m (medium) | ~20ms | Best     | Accurate analysis  |



## Results & Visualizations

See `animal_detection_notebook.ipynb` for:

* Single image detection examples
* Model speed comparison (nano vs small)
* Confidence distribution by species
* Carnivore vs herbivore breakdown pie chart



## Limitations

* COCO pretrained weights only detect 10 animal classes. Many wildlife species (lions, tigers, wolves) are NOT in COCO and won't be detected unless you use a custom dataset.
* Carnivore classification is rule-based, not learned.
* No temporal tracking between video frames (DeepSORT could be added).



## Future Extensions

* Fine-tune YOLOv8 on iNaturalist or OpenImages wildlife dataset
* Add DeepSORT multi-object tracking for video
* GPS-tagged alerts for field wildlife monitoring
* Mobile app wrapper (ONNX export for deployment)



## Tech Stack

* **Python 3.10**
* **YOLOv8** (Ultralytics)
* **OpenCV** — image/video processing
* **tkinter** — GUI
* **Matplotlib / Seaborn** — visualizations
* **Pillow** — image rendering in GUI


## License

MIT — free to use and extend.

