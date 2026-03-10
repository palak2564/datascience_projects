# Car Color Detection (YOLOv3 + OpenCV)

This is a Python-based computer vision tool designed for traffic analysis. It uses a YOLOv3 backbone for object detection and a custom HSV-based logic to filter vehicles by color.

The system includes a Tkinter-based GUI to handle model loading, image selection, and result visualization without needing to touch the command line during operation.

### Core Capabilities

* **Vehicle Detection:** Identifies cars, trucks, and buses.
* **Color Filtering:** Specifically isolates blue cars from other vehicle colors.
* **Pedestrian Tracking:** Monitors and counts people at traffic signals.
* **Data Export:** Saves processed images with labeled bounding boxes and a summary count.

---

### Technical Stack

* **Language:** Python 3.7+
* **Vision:** OpenCV (cv2)
* **Model:** YOLOv3 (weights trained on COCO)
* **Interface:** Tkinter & Pillow
* **Math:** NumPy

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/car-color-detection.git
cd car-color-detection
pip install -r requirements.txt

```

**2. Model Setup**
The YOLO weights are too large for standard Git versioning. You must manually download them:

1. Create a directory named `model_files`.
2. Download [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg).
3. Place them in the `model_files` directory alongside the provided `coco.names`.

---

### Workflow

1. **Initialization:** Run `python main_app.py`.
2. **Model Loading:** Use the "Load YOLO Model" button to link your `.cfg`, `.weights`, and `.names` files.
3. **Processing:** Upload an image and select "Detect Cars."
4. **Visual Legend:**
* **Red Bounding Box:** Blue car detected.
* **Blue Bounding Box:** Non-blue vehicle.
* **Green Bounding Box:** Pedestrian.



---

### Logic Overview

**Color Classification**
To handle varying light conditions (shadows, glare), the system converts BGR frames to the HSV (Hue, Saturation, Value) color space. This provides a more robust way to define "blue" than standard RGB ratios.

```python
# Internal Hue/Saturation check
if 90 <= average_hue <= 130 and average_saturation > 50:
    return "blue"

```

**Architecture**
The project follows a modular structure, separating the detection engine (`CarColorDetector`) from the interface logic (`CarDetectionGUI`). This allows the detection backend to be repurposed for web or CLI-based applications.

---

### Troubleshooting and Performance

* **Confidence Threshold:** If the model is picking up noise, increase `self.confidence_threshold` in the detector class to 0.6 or 0.7.
* **IO Performance:** YOLOv3 is computationally intensive. On CPU-only systems, expect a processing delay of 2–5 seconds per high-resolution image.
* **Pathing Errors:** Ensure the `results/` directory exists or that the script has permission to create it for saving outputs.

---

### Roadmap

* Implement video file stream processing.
* Connect to live RTSP camera feeds.
* Replace hardcoded HSV ranges with a KNN classifier for broader color support.
