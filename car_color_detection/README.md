# Car Color Detection System

## Project Overview

This is a machine learning-based car color detection system that uses YOLOv3 for object detection and custom color classification. The system can:

- Detect cars at traffic signals
- Identify car colors (specifically blue vs other colors)
- Count the number of cars
- Detect and count people at traffic signals
- Display results with color-coded bounding boxes

## Features

✅ **GUI Interface** - User-friendly graphical interface
✅ **Car Detection** - Detects cars, trucks, and buses
✅ **Color Classification** - Distinguishes blue cars from other colors
✅ **People Detection** - Counts people at traffic signals
✅ **Visual Feedback** - Color-coded bounding boxes:
  - 🔴 Red boxes for blue cars
  - 🔵 Blue boxes for other colored cars
  - 🟢 Green boxes for people
✅ **Statistics Display** - Shows counts on the image
✅ **Save Results** - Export detected images

## Technology Stack

- **Python 3.7+**
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computations
- **Tkinter** - GUI framework
- **Pillow** - Image handling
- **YOLOv3** - Object detection model

## Installation

### 1. Clone or Download the Project

```bash
git clone https://github.com/yourusername/car-color-detection.git
cd car-color-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download YOLO Model Files

Download the required YOLO model files:

**Option 1: Using wget (Linux/Mac)**
```bash
# Create model_files directory
mkdir model_files
cd model_files

# Download weights
wget https://pjreddie.com/media/files/yolov3.weights

# Download config
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# Return to main directory
cd ..
```

**Option 2: Manual Download**
1. Download `yolov3.weights` (237 MB) from: https://pjreddie.com/media/files/yolov3.weights
2. Download `yolov3.cfg` from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
3. The `coco.names` file is already included in this repository

## Project Structure

```
car_color_detection/
│
├── main_app.py                 # Main application file
├── requirements.txt            # Python dependencies
├── coco.names                 # Class labels for YOLO
├── README.md                  # This file
├── setup_instructions.txt     # Detailed setup guide
│
├── model_files/               # YOLO model files (create this)
│   ├── yolov3.weights        # Download separately (237 MB)
│   ├── yolov3.cfg            # Download separately
│   └── coco.names            # Copy from root directory
│
├── sample_images/            # Test images (optional)
│   └── traffic_scene.jpg
│
└── results/                  # Saved output images (created automatically)
```

## Usage

### Running the Application

```bash
python main_app.py
```

### Step-by-Step Guide

1. **Launch Application**
   - Run the Python script
   - GUI window will open

2. **Load YOLO Model**
   - Click "Load YOLO Model" button
   - Select `yolov3.cfg` file
   - Select `yolov3.weights` file
   - Select `coco.names` file
   - Wait for "Model loaded successfully" message

3. **Select Image**
   - Click "Select Image" button
   - Choose an image file (JPG, PNG, etc.)
   - Original image will appear on the left

4. **Detect Cars**
   - Click "Detect Cars" button
   - Wait for processing (may take a few seconds)
   - Detected image appears on the right with:
     * Red boxes around blue cars
     * Blue boxes around other colored cars
     * Green boxes around people
     * Statistics displayed at the top

5. **Save Results**
   - Click "Save Result" button
   - Choose location and filename
   - Image saved successfully

## Code Structure

### Main Components

#### 1. CarColorDetector Class
Handles all detection logic:
- `load_yolo_model()` - Loads YOLO neural network
- `detect_objects_in_image()` - Performs object detection
- `identify_car_color()` - Classifies car colors
- `draw_boxes_on_image()` - Draws bounding boxes and labels

#### 2. CarDetectionGUI Class
Manages the user interface:
- `create_widgets()` - Creates GUI components
- `load_model()` - Model loading interface
- `select_image()` - Image selection dialog
- `detect_cars()` - Triggers detection process
- `save_result()` - Saves output image

### Color Detection Algorithm

The system uses HSV color space for better color detection:

```python
def identify_car_color(self, car_image):
    # Convert to HSV
    hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
    
    # Calculate average values
    average_hue = np.mean(hsv_image[:, :, 0])
    average_saturation = np.mean(hsv_image[:, :, 1])
    
    # Classify as blue or other
    if 90 <= average_hue <= 130 and average_saturation > 50:
        return "blue"
    else:
        return "other"
```

## Clean Code Practices Used

This project follows Python best practices and clean code principles:

### ✅ Meaningful Variable Names
- `image_height` instead of `h`
- `confidence_threshold` instead of `conf_thresh`
- `detected_boxes` instead of `boxes`

### ✅ Clear Function Names
- `detect_objects_in_image()` - clearly states what it does
- `identify_car_color()` - easy to understand
- `draw_boxes_on_image()` - descriptive

### ✅ Proper Comments
- Docstrings for all classes and functions
- Inline comments explaining complex logic
- Section headers for organization

### ✅ Consistent Naming Convention
- snake_case for variables and functions
- CamelCase for classes
- UPPER_CASE for constants

### ✅ Modular Design
- Separate classes for detection logic and GUI
- Single responsibility principle
- Easy to maintain and extend

## Customization

### Adjusting Detection Parameters

Edit these values in `CarColorDetector.__init__()`:

```python
# Detection confidence (0.0 to 1.0)
self.confidence_threshold = 0.5  # Increase for fewer detections

# Box overlap threshold
self.box_overlap_threshold = 0.3  # Adjust for overlapping boxes
```

### Changing Box Colors

Modify color definitions (BGR format):

```python
self.blue_car_color = (0, 0, 255)     # Red for blue cars
self.other_car_color = (255, 0, 0)    # Blue for others
self.person_box_color = (0, 255, 0)   # Green for people
```

### Adding More Colors

Extend the `identify_car_color()` method:

```python
def identify_car_color(self, car_image):
    hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
    average_hue = np.mean(hsv_image[:, :, 0])
    
    # Red
    if average_hue < 10 or average_hue > 170:
        return "red"
    # Blue
    elif 90 <= average_hue <= 130:
        return "blue"
    # Green
    elif 40 <= average_hue <= 80:
        return "green"
    # Add more colors as needed
    else:
        return "other"
```

## Performance Tips

- **Image Size**: Smaller images process faster (resize large images)
- **CPU vs GPU**: This version uses CPU (add GPU support for faster processing)
- **Model Version**: YOLOv4 or YOLOv5 may be faster and more accurate
- **Batch Processing**: Process multiple images sequentially

## Troubleshooting

### Common Issues

**1. Model fails to load**
- Check file paths are correct
- Ensure all three files (weights, config, names) are present
- Verify weights file is not corrupted (should be ~237 MB)

**2. Low detection accuracy**
- Adjust `confidence_threshold` (try 0.3 for more detections)
- Use higher quality images
- Ensure good lighting in images

**3. Wrong colors detected**
- Adjust HSV thresholds in `identify_car_color()`
- Consider using a trained color classifier instead
- Ensure images have good color quality

**4. Slow performance**
- Reduce image size before processing
- Use YOLOv4-tiny for faster processing
- Consider GPU acceleration

**5. GUI doesn't appear**
- Check Tkinter is installed: `python -m tkinter`
- Update Pillow: `pip install --upgrade Pillow`
- Try running as administrator

## Future Enhancements

- [ ] Add support for video file processing
- [ ] Real-time webcam detection
- [ ] Train custom color classifier with ML
- [ ] Add more color categories
- [ ] Implement vehicle tracking across frames
- [ ] Export statistics to CSV/Excel
- [ ] Add confidence sliders in GUI
- [ ] Support for multiple image formats
- [ ] GPU acceleration option

## Learning Resources

- **YOLO**: https://pjreddie.com/darknet/yolo/
- **OpenCV**: https://opencv.org/
- **Python Clean Code**: https://realpython.com/python-pep8/
- **Computer Vision**: https://pyimagesearch.com/

## Credits

- **YOLO** by Joseph Redmon
- **OpenCV** community
- **COCO Dataset** for class labels

## License

This project is created for educational purposes as part of an internship assignment.

## Contact

For questions or suggestions:
- Create an issue in the repository
- Email: your.email@example.com

---

**Happy Detecting! 🚗🔍**
