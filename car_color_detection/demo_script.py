"""
Demo Script - Quick Test Without GUI
This script demonstrates the detection functionality without GUI
Useful for testing and debugging
"""

import cv2
import numpy as np


def simple_car_detection_demo():
    """
    Simple demonstration of car detection
    This version uses hardcoded paths for quick testing
    """
    
    print("=" * 60)
    print("Car Color Detection - Simple Demo")
    print("=" * 60)
    
    # File paths - UPDATE THESE TO MATCH YOUR SETUP
    config_path = "model_files/yolov3.cfg"
    weights_path = "model_files/yolov3.weights"
    names_path = "coco.names"
    test_image_path = "sample_images/test.jpg"
    
    # Check if files exist
    import os
    missing_files = []
    
    if not os.path.exists(config_path):
        missing_files.append(config_path)
    if not os.path.exists(weights_path):
        missing_files.append(weights_path)
    if not os.path.exists(names_path):
        missing_files.append(names_path)
    
    if missing_files:
        print("\n✗ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download YOLO model files first.")
        print("See setup_instructions.txt for details.")
        return
    
    print("\n✓ All model files found!")
    
    # Load class names
    print("\nLoading class names...")
    with open(names_path, 'r') as file:
        class_names = file.read().strip().split('\n')
    print(f"✓ Loaded {len(class_names)} classes")
    
    # Load YOLO network
    print("\nLoading YOLO network...")
    neural_network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✓ YOLO network loaded")
    
    # Check for test image
    if not os.path.exists(test_image_path):
        print(f"\n✗ Test image not found: {test_image_path}")
        print("\nPlease provide a test image or update the path in this script.")
        return
    
    # Load and process image
    print(f"\nLoading image: {test_image_path}")
    input_image = cv2.imread(test_image_path)
    
    if input_image is None:
        print("✗ Failed to load image")
        return
    
    image_height, image_width = input_image.shape[:2]
    print(f"✓ Image loaded: {image_width}x{image_height}")
    
    # Prepare image for detection
    print("\nRunning detection...")
    image_blob = cv2.dnn.blobFromImage(
        input_image,
        1/255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )
    
    # Run detection
    neural_network.setInput(image_blob)
    layer_names = neural_network.getLayerNames()
    output_layers = [layer_names[i - 1] for i in neural_network.getUnconnectedOutLayers()]
    detections = neural_network.forward(output_layers)
    
    # Process detections
    detected_boxes = []
    confidence_scores = []
    detected_classes = []
    
    confidence_threshold = 0.5
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                box_width = int(detection[2] * image_width)
                box_height = int(detection[3] * image_height)
                
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)
                
                detected_boxes.append([x, y, box_width, box_height])
                confidence_scores.append(float(confidence))
                detected_classes.append(class_id)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        detected_boxes,
        confidence_scores,
        confidence_threshold,
        0.3
    )
    
    # Count objects
    car_count = 0
    people_count = 0
    
    print("\nDetections:")
    print("-" * 60)
    
    if len(indices) > 0:
        for i in indices.flatten():
            class_id = detected_classes[i]
            class_name = class_names[class_id]
            confidence = confidence_scores[i]
            
            if class_name in ['car', 'truck', 'bus']:
                car_count += 1
                print(f"  Vehicle: {class_name} (confidence: {confidence:.2f})")
            elif class_name == 'person':
                people_count += 1
                print(f"  Person (confidence: {confidence:.2f})")
            else:
                print(f"  {class_name} (confidence: {confidence:.2f})")
    else:
        print("  No objects detected")
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total vehicles: {car_count}")
    print(f"  Total people: {people_count}")
    print(f"  Total detections: {len(indices)}")
    
    # Draw boxes on image
    output_image = input_image.copy()
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = detected_boxes[i]
            class_id = detected_classes[i]
            class_name = class_names[class_id]
            
            # Choose color
            if class_name in ['car', 'truck', 'bus']:
                color = (255, 0, 0)  # Blue for vehicles
            elif class_name == 'person':
                color = (0, 255, 0)  # Green for people
            else:
                color = (0, 0, 255)  # Red for others
            
            # Draw box
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, class_name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save result
    output_path = "demo_output.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"\n✓ Result saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    simple_car_detection_demo()
