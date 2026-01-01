"""
Car Color Detection System - Enhanced Version
This version includes improved color detection using K-means clustering
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from sklearn.cluster import KMeans


class ImprovedCarColorDetector:
    """Enhanced car color detector with better color identification"""
    
    def __init__(self):
        # Box colors for different detections
        self.blue_car_box_color = (0, 0, 255)  # Red box for blue cars
        self.other_car_box_color = (255, 0, 0)  # Blue box for other cars
        self.person_box_color = (0, 255, 0)  # Green box for people
        
        # Detection settings
        self.minimum_confidence = 0.5
        self.overlap_threshold = 0.3
        
        # Model variables
        self.neural_network = None
        self.object_classes = []
        
        # Color definitions in BGR
        self.color_ranges = {
            'blue': [(90, 50, 50), (130, 255, 255)],    # HSV range for blue
            'red': [(0, 50, 50), (10, 255, 255)],       # HSV range for red (part 1)
            'red2': [(170, 50, 50), (180, 255, 255)],   # HSV range for red (part 2)
            'white': [(0, 0, 200), (180, 30, 255)],     # HSV range for white
            'black': [(0, 0, 0), (180, 255, 50)],       # HSV range for black
        }
    
    def setup_model(self, config_file, weights_file, names_file):
        """Initialize YOLO detection model"""
        try:
            # Load class names
            with open(names_file, 'r') as file:
                self.object_classes = file.read().strip().split('\n')
            
            # Load YOLO network
            self.neural_network = cv2.dnn.readNetFromDarknet(config_file, weights_file)
            self.neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            return True
        except Exception as error_message:
            print(f"Model loading error: {error_message}")
            return False
    
    def find_dominant_color(self, image_region):
        """Find dominant color in car region using K-means clustering"""
        # Reshape image to list of pixels
        pixels = image_region.reshape(-1, 3)
        
        # Remove very dark pixels (shadows)
        bright_pixels = pixels[pixels.mean(axis=1) > 30]
        
        if len(bright_pixels) < 10:
            return "unknown"
        
        # Use K-means to find dominant colors
        number_of_colors = 3
        kmeans = KMeans(n_clusters=number_of_colors, random_state=42, n_init=10)
        kmeans.fit(bright_pixels)
        
        # Get most common color
        unique_labels, label_counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_color_index = unique_labels[np.argmax(label_counts)]
        dominant_color_bgr = kmeans.cluster_centers_[dominant_color_index]
        
        # Convert to HSV for better color classification
        dominant_color_bgr_uint8 = np.uint8([[dominant_color_bgr]])
        dominant_color_hsv = cv2.cvtColor(dominant_color_bgr_uint8, cv2.COLOR_BGR2HSV)[0][0]
        
        return self.classify_color_by_hsv(dominant_color_hsv)
    
    def classify_color_by_hsv(self, hsv_color):
        """Classify color based on HSV values"""
        hue, saturation, value = hsv_color
        
        # White - low saturation, high value
        if saturation < 30 and value > 200:
            return "white"
        
        # Black - low value
        elif value < 50:
            return "black"
        
        # Gray - low saturation, medium value
        elif saturation < 30:
            return "gray"
        
        # Blue
        elif 90 <= hue <= 130:
            return "blue"
        
        # Red (wraps around at 180)
        elif hue < 10 or hue > 170:
            return "red"
        
        # Yellow
        elif 20 <= hue < 40:
            return "yellow"
        
        # Green
        elif 40 <= hue < 80:
            return "green"
        
        # Silver/Gray
        else:
            return "other"
    
    def detect_vehicles_and_people(self, input_image):
        """Detect cars, trucks, buses and people in image"""
        image_height, image_width = input_image.shape[:2]
        
        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(
            input_image,
            1/255.0,
            (416, 416),
            swapRB=True,
            crop=False
        )
        
        # Run detection
        self.neural_network.setInput(blob)
        layer_names = self.neural_network.getLayerNames()
        output_layer_names = [layer_names[i - 1] for i in self.neural_network.getUnconnectedOutLayers()]
        detections = self.neural_network.forward(output_layer_names)
        
        # Storage for detections
        bounding_boxes = []
        confidence_values = []
        class_identifiers = []
        
        # Process each detection
        for detection_output in detections:
            for single_detection in detection_output:
                class_probabilities = single_detection[5:]
                class_id = np.argmax(class_probabilities)
                confidence = class_probabilities[class_id]
                
                if confidence > self.minimum_confidence:
                    # Calculate bounding box
                    box_center_x = int(single_detection[0] * image_width)
                    box_center_y = int(single_detection[1] * image_height)
                    box_width = int(single_detection[2] * image_width)
                    box_height = int(single_detection[3] * image_height)
                    
                    box_x = int(box_center_x - box_width / 2)
                    box_y = int(box_center_y - box_height / 2)
                    
                    bounding_boxes.append([box_x, box_y, box_width, box_height])
                    confidence_values.append(float(confidence))
                    class_identifiers.append(class_id)
        
        # Remove overlapping boxes
        selected_indices = cv2.dnn.NMSBoxes(
            bounding_boxes,
            confidence_values,
            self.minimum_confidence,
            self.overlap_threshold
        )
        
        return bounding_boxes, confidence_values, class_identifiers, selected_indices
    
    def annotate_image_with_detections(self, original_image, boxes, confidences, classes, indices):
        """Draw boxes and labels on detected objects"""
        total_vehicles = 0
        blue_vehicles = 0
        other_vehicles = 0
        people_detected = 0
        
        output_image = original_image.copy()
        
        if len(indices) > 0:
            for index in indices.flatten():
                box_x, box_y, box_width, box_height = boxes[index]
                object_class_id = classes[index]
                object_class_name = self.object_classes[object_class_id]
                detection_confidence = confidences[index]
                
                # Handle vehicle detection
                if object_class_name in ['car', 'truck', 'bus']:
                    total_vehicles += 1
                    
                    # Extract vehicle region
                    vehicle_region = original_image[box_y:box_y+box_height, box_x:box_x+box_width]
                    
                    if vehicle_region.size > 0:
                        # Detect vehicle color
                        vehicle_color = self.find_dominant_color(vehicle_region)
                        
                        if vehicle_color == "blue":
                            box_color = self.blue_car_box_color
                            blue_vehicles += 1
                            label_text = f"Blue {object_class_name} {detection_confidence:.2f}"
                        else:
                            box_color = self.other_car_box_color
                            other_vehicles += 1
                            label_text = f"{vehicle_color.title()} {object_class_name} {detection_confidence:.2f}"
                    else:
                        box_color = self.other_car_box_color
                        label_text = f"{object_class_name} {detection_confidence:.2f}"
                    
                    # Draw box and label
                    cv2.rectangle(output_image, (box_x, box_y), 
                                (box_x + box_width, box_y + box_height), 
                                box_color, 2)
                    cv2.putText(output_image, label_text, (box_x, box_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Handle person detection
                elif object_class_name == 'person':
                    people_detected += 1
                    
                    # Draw box and label
                    cv2.rectangle(output_image, (box_x, box_y),
                                (box_x + box_width, box_y + box_height),
                                self.person_box_color, 2)
                    cv2.putText(output_image, f"Person {detection_confidence:.2f}",
                              (box_x, box_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.person_box_color, 2)
        
        # Add summary information
        summary = f"Vehicles: {total_vehicles} | Blue: {blue_vehicles} | Other: {other_vehicles} | People: {people_detected}"
        cv2.putText(output_image, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_image, total_vehicles, blue_vehicles, people_detected


# The GUI class would be similar to the main version
# Just replace the detector instantiation with ImprovedCarColorDetector
