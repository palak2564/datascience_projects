"""
Car Color Detection System
A simple GUI application to detect cars, identify their colors, and count people
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class CarColorDetector:
    """Main class for detecting cars and their colors"""
    
    def __init__(self):
        # Color definitions for different car colors
        self.blue_car_color = (0, 0, 255)  # Red rectangle for blue cars (BGR format)
        self.other_car_color = (255, 0, 0)  # Blue rectangle for other cars
        self.person_box_color = (0, 255, 0)  # Green for people
        
        # Detection confidence threshold
        self.confidence_threshold = 0.5
        self.box_overlap_threshold = 0.3
        
        # Model files
        self.yolo_config_file = None
        self.yolo_weights_file = None
        self.class_names = []
        self.neural_network = None
        
    def load_yolo_model(self, config_path, weights_path, names_path):
        """Load the YOLO model for object detection"""
        try:
            # Read class names from file
            with open(names_path, 'r') as file:
                self.class_names = file.read().strip().split('\n')
            
            # Load neural network
            self.neural_network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            self.neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            return True
        except Exception as error:
            print(f"Error loading YOLO model: {error}")
            return False
    
    def detect_objects_in_image(self, input_image):
        """Detect cars and people in the image"""
        image_height, image_width = input_image.shape[:2]
        
        # Prepare image for neural network
        image_blob = cv2.dnn.blobFromImage(
            input_image, 
            1/255.0, 
            (416, 416), 
            swapRB=True, 
            crop=False
        )
        
        # Run detection
        self.neural_network.setInput(image_blob)
        layer_names = self.neural_network.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.neural_network.getUnconnectedOutLayers()]
        detection_results = self.neural_network.forward(output_layers)
        
        # Lists to store detection information
        detected_boxes = []
        confidence_scores = []
        detected_classes = []
        
        # Process each detection
        for output in detection_results:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    box_width = int(detection[2] * image_width)
                    box_height = int(detection[3] * image_height)
                    
                    # Calculate corner coordinates
                    top_left_x = int(center_x - box_width / 2)
                    top_left_y = int(center_y - box_height / 2)
                    
                    detected_boxes.append([top_left_x, top_left_y, box_width, box_height])
                    confidence_scores.append(float(confidence))
                    detected_classes.append(class_id)
        
        # Apply non-maximum suppression to remove overlapping boxes
        final_indices = cv2.dnn.NMSBoxes(
            detected_boxes, 
            confidence_scores, 
            self.confidence_threshold, 
            self.box_overlap_threshold
        )
        
        return detected_boxes, confidence_scores, detected_classes, final_indices
    
    def identify_car_color(self, car_image):
        """Identify the dominant color of the car"""
        # Convert to HSV color space for better color detection
        hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
        
        # Calculate average color values
        average_hue = np.mean(hsv_image[:, :, 0])
        average_saturation = np.mean(hsv_image[:, :, 1])
        average_value = np.mean(hsv_image[:, :, 2])
        
        # Simple color classification based on HSV
        # Blue color range in HSV
        if 90 <= average_hue <= 130 and average_saturation > 50:
            return "blue"
        else:
            return "other"
    
    def draw_boxes_on_image(self, image, boxes, scores, classes, indices):
        """Draw bounding boxes on detected objects"""
        car_count = 0
        blue_car_count = 0
        other_car_count = 0
        people_count = 0
        
        result_image = image.copy()
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = classes[i]
                class_name = self.class_names[class_id]
                confidence = scores[i]
                
                # Check if detected object is a car
                if class_name in ['car', 'truck', 'bus']:
                    car_count += 1
                    
                    # Extract car region for color detection
                    car_region = image[y:y+h, x:x+w]
                    
                    if car_region.size > 0:
                        car_color = self.identify_car_color(car_region)
                        
                        if car_color == "blue":
                            box_color = self.blue_car_color  # Red box for blue cars
                            blue_car_count += 1
                            label = f"Blue {class_name} {confidence:.2f}"
                        else:
                            box_color = self.other_car_color  # Blue box for other cars
                            other_car_count += 1
                            label = f"{class_name} {confidence:.2f}"
                    else:
                        box_color = self.other_car_color
                        label = f"{class_name} {confidence:.2f}"
                    
                    # Draw bounding box
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(result_image, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Check if detected object is a person
                elif class_name == 'person':
                    people_count += 1
                    
                    # Draw bounding box for person
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), 
                                self.person_box_color, 2)
                    cv2.putText(result_image, f"Person {confidence:.2f}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.person_box_color, 2)
        
        # Add summary text on image
        summary_text = f"Cars: {car_count} | Blue Cars: {blue_car_count} | Other Cars: {other_car_count} | People: {people_count}"
        cv2.putText(result_image, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image, car_count, blue_car_count, people_count


class CarDetectionGUI:
    """GUI for the car color detection application"""
    
    def __init__(self, window):
        self.window = window
        self.window.title("Car Color Detection System")
        self.window.geometry("1200x800")
        
        # Initialize detector
        self.detector = CarColorDetector()
        
        # Variables
        self.current_image = None
        self.processed_image = None
        self.image_path = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI components"""
        # Title
        title_label = tk.Label(
            self.window, 
            text="Car Color Detection System", 
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.window)
        control_frame.pack(pady=10)
        
        # Buttons
        load_model_button = tk.Button(
            control_frame, 
            text="Load YOLO Model", 
            command=self.load_model,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        load_model_button.grid(row=0, column=0, padx=5)
        
        select_image_button = tk.Button(
            control_frame, 
            text="Select Image", 
            command=self.select_image,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10
        )
        select_image_button.grid(row=0, column=1, padx=5)
        
        detect_button = tk.Button(
            control_frame, 
            text="Detect Cars", 
            command=self.detect_cars,
            font=("Arial", 12),
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=10
        )
        detect_button.grid(row=0, column=2, padx=5)
        
        save_button = tk.Button(
            control_frame, 
            text="Save Result", 
            command=self.save_result,
            font=("Arial", 12),
            bg="#9C27B0",
            fg="white",
            padx=20,
            pady=10
        )
        save_button.grid(row=0, column=3, padx=5)
        
        # Image display frame
        display_frame = tk.Frame(self.window)
        display_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Original image
        original_frame = tk.LabelFrame(
            display_frame, 
            text="Original Image", 
            font=("Arial", 12)
        )
        original_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        self.original_image_label = tk.Label(original_frame)
        self.original_image_label.pack(padx=10, pady=10)
        
        # Processed image
        processed_frame = tk.LabelFrame(
            display_frame, 
            text="Detected Image", 
            font=("Arial", 12)
        )
        processed_frame.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        
        self.processed_image_label = tk.Label(processed_frame)
        self.processed_image_label.pack(padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.window, 
            text="Ready. Please load YOLO model first.", 
            font=("Arial", 11),
            bg="#f0f0f0",
            pady=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):
        """Load YOLO model files"""
        messagebox.showinfo(
            "Load Model", 
            "Please select the YOLO configuration file (.cfg)"
        )
        config_path = filedialog.askopenfilename(
            title="Select YOLO Config File",
            filetypes=[("Config files", "*.cfg")]
        )
        
        if not config_path:
            return
        
        messagebox.showinfo(
            "Load Model", 
            "Please select the YOLO weights file (.weights)"
        )
        weights_path = filedialog.askopenfilename(
            title="Select YOLO Weights File",
            filetypes=[("Weights files", "*.weights")]
        )
        
        if not weights_path:
            return
        
        messagebox.showinfo(
            "Load Model", 
            "Please select the class names file (.names or .txt)"
        )
        names_path = filedialog.askopenfilename(
            title="Select Class Names File",
            filetypes=[("Text files", "*.txt *.names")]
        )
        
        if not names_path:
            return
        
        # Load model
        self.status_label.config(text="Loading YOLO model...")
        self.window.update()
        
        success = self.detector.load_yolo_model(config_path, weights_path, names_path)
        
        if success:
            self.status_label.config(text="YOLO model loaded successfully!")
            messagebox.showinfo("Success", "YOLO model loaded successfully!")
        else:
            self.status_label.config(text="Failed to load YOLO model.")
            messagebox.showerror("Error", "Failed to load YOLO model. Please check the files.")
    
    def select_image(self):
        """Select an image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.current_image = cv2.imread(file_path)
            
            # Display original image
            self.display_image(self.current_image, self.original_image_label)
            self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
    
    def detect_cars(self):
        """Perform car detection on the selected image"""
        if self.detector.neural_network is None:
            messagebox.showwarning("Warning", "Please load YOLO model first!")
            return
        
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        self.status_label.config(text="Detecting cars and people...")
        self.window.update()
        
        # Detect objects
        boxes, scores, classes, indices = self.detector.detect_objects_in_image(
            self.current_image
        )
        
        # Draw boxes and get counts
        result_image, car_count, blue_count, people_count = self.detector.draw_boxes_on_image(
            self.current_image, boxes, scores, classes, indices
        )
        
        self.processed_image = result_image
        
        # Display processed image
        self.display_image(result_image, self.processed_image_label)
        
        # Update status
        status_message = f"Detection complete! Cars: {car_count}, Blue Cars: {blue_count}, People: {people_count}"
        self.status_label.config(text=status_message)
    
    def display_image(self, cv_image, label_widget):
        """Display image in tkinter label"""
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit in GUI
        max_width = 550
        max_height = 550
        height, width = rgb_image.shape[:2]
        
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to PIL format
        pil_image = Image.fromarray(resized_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        label_widget.config(image=photo)
        label_widget.image = photo  # Keep reference
    
    def save_result(self):
        """Save the processed image"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save!")
            return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        
        if save_path:
            cv2.imwrite(save_path, self.processed_image)
            self.status_label.config(text=f"Image saved: {os.path.basename(save_path)}")
            messagebox.showinfo("Success", "Image saved successfully!")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = CarDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
