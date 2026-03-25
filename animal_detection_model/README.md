Animal Detection & Classification SystemInternship Project Extension — Built on the training project base.Detects animals in images and videos, classifies dietary type, and alerts on carnivores.Problem StatementWildlife monitoring, zoo management, and ecological research all need automated tools to identify animals in images. Manual annotation is slow. This project uses YOLOv8 to:Detect multiple animals in a single frameClassify each as carnivore, herbivore, or omnivoreHighlight carnivores with RED bounding boxesShow a popup alert with the count of carnivores detectedSupport both images and video input (including live webcam)Export detection logs as CSV for analysisDemoImage DetectionCarnivore AlertGreen boxes = herbivores, Red = carnivoresPopup fires when carnivores foundProject Structureanimal_detection/
│
├── app.py                         # Main GUI application (tkinter)
├── cli.py                         # Command-line interface (no GUI)
├── requirements.txt
├── animal_detection_notebook.ipynb # Experiments, charts, model comparison
│
├── utils/
│   ├── detector.py                # YOLOv8 wrapper + animal classification
│   ├── gui_components.py           # Custom tkinter widgets
│   ├── carnivore_alert.py          # Popup alert window
│   └── logger.py                   # Detection logging + CSV export
│
├── samples/                        # Put test images here
└── outputs/                        # Saved results go here
SetupRequirementsPython 3.8+pipInstall dependenciesBashpip install -r requirements.txt
This installs:ultralytics — YOLOv8 frameworkopencv-python — image/video processingPillow — tkinter image displaymatplotlib, seaborn — notebook visualizationsRunning the AppGUI Mode (recommended)Bashpython app.py
Features:Load Image button — runs detection on any JPG/PNGLoad Video button — processes video frame by frameLive Webcam — real-time detectionAdjustable confidence threshold sliderLive detection log in sidebarSave annotated result / Export CSV logCLI Mode (batch/headless)Bash# Detect animals in an image
python cli.py --image zoo.jpg

# Save annotated output
python cli.py --image zoo.jpg --output result.jpg

# Process a video file
python cli.py --video wildlife.mp4 --show

# Use a more accurate (but slower) model
python cli.py --image zoo.jpg --model yolov8s

# Export detection log
python cli.py --image zoo.jpg --export-csv detections.csv
MethodologyModel: YOLOv8 (Ultralytics)YOLOv8 is a state-of-the-art, real-time object detection model. We use the pretrained COCO weights which include 10 animal classes out of the box:ClassTypecatCarnivoredogOmnivorebirdCarnivorebearCarnivorehorseHerbivoresheepHerbivorecowHerbivoreelephantHerbivorezebraHerbivoregiraffeHerbivoreCarnivore ClassificationCarnivore detection uses a manually curated lookup table (CARNIVORES set in detector.py). A production system could use:A taxonomy API (e.g. GBIF, iNaturalist)A fine-tuned classification head trained on dietary behaviorBounding Box ColorsColorMeaningRedCarnivoreYellowOmnivoreGreenHerbivoreModel Size TradeoffsModelSpeedAccuracyUse caseyolov8n (nano)~5msGoodWebcam / real-timeyolov8s (small)~10msBetterVideo filesyolov8m (medium)~20msBestAccurate analysisResults & VisualizationsSee animal_detection_notebook.ipynb for:Single image detection examplesModel speed comparison (nano vs small)Confidence distribution by speciesCarnivore vs herbivore breakdown pie chartLimitationsCOCO pretrained weights only detect 10 animal classes. Many wildlife species (lions, tigers, wolves) are NOT in COCO and won't be detected unless you use a custom dataset.Carnivore classification is rule-based, not learned.No temporal tracking between video frames (DeepSORT could be added).Future ExtensionsFine-tune YOLOv8 on iNaturalist or OpenImages wildlife datasetAdd DeepSORT multi-object tracking for videoGPS-tagged alerts for field wildlife monitoringMobile app wrapper (ONNX export for deployment)Tech StackPython 3.10YOLOv8 (Ultralytics)OpenCV — image/video processingtkinter — GUIMatplotlib / Seaborn — visualizationsPillow — image rendering in GUILicenseMIT — free to use and extend.
