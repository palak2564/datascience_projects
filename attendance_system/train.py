"""
train.py
=========
Trains the face recognition model on your student dataset.

You need to run this ONCE before running attendance_system.py.

Dataset structure expected:
    dataset/known_faces/
        Alice_Smith/
            img1.jpg
            img2.jpg
            img3.jpg
            ...
        Bob_Jones/
            img1.jpg
            img2.jpg
            ...

The more images per student, the better the recognition accuracy.
Aim for at least 20-30 images per person, with varied lighting/angles.

Algorithm used: LBPH (Local Binary Pattern Histogram)
- Why LBPH? It's lightweight, trains in seconds, no GPU needed, and works
  reasonably well for controlled indoor environments like classrooms.
- It's a classic ML algorithm — fits the "train your own model" requirement.
- Alternative: use dlib's 128-d face embeddings for much better accuracy
  (see utils/face_recognizer.py for the dlib option)

Usage:
    python train.py
    python train.py --dataset dataset/known_faces --augment
"""

import cv2
import os
import sys
import pickle
import argparse
import numpy as np
import logging
from pathlib import Path

from config import STUDENT_DATASET_DIR, MODEL_DIR, FACE_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Haar Cascade for face detection during training
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# ─── Data augmentation helpers ────────────────────────────────────────────────

def augment_image(img: np.ndarray) -> list:
    """
    Returns a list of augmented versions of the input image.
    Helps when you don't have many training images per student.

    Augmentations:
    - Horizontal flip
    - Brightness +/- 30
    - Slight rotation (±10°)
    """
    augmented = [img]

    # Flip
    augmented.append(cv2.flip(img, 1))

    # Brightness boost
    bright = cv2.convertScaleAbs(img, alpha=1, beta=30)
    augmented.append(bright)

    # Brightness reduce
    dark = cv2.convertScaleAbs(img, alpha=1, beta=-30)
    augmented.append(dark)

    # Rotate +10°
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
    augmented.append(cv2.warpAffine(img, M, (w, h)))

    # Rotate -10°
    M2 = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
    augmented.append(cv2.warpAffine(img, M2, (w, h)))

    return augmented


def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """
    Standardize a face crop before training:
    - Convert to grayscale
    - Resize to fixed FACE_SIZE
    - Histogram equalize (improves lighting robustness)
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    resized = cv2.resize(gray, FACE_SIZE)
    equalized = cv2.equalizeHist(resized)
    return equalized


# ─── Main training logic ──────────────────────────────────────────────────────

def load_dataset(dataset_dir: str, use_augmentation: bool = True):
    """
    Walks the dataset directory and loads face images + labels.

    Returns:
        faces: list of preprocessed face np arrays
        labels: list of integer label IDs
        label_map: dict mapping label int -> student name string
    """
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    student_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ])

    if not student_dirs:
        log.error(f"No student folders found in {dataset_dir}")
        log.info("Create folders like: dataset/known_faces/StudentName/img1.jpg")
        return None, None, None

    log.info(f"Found {len(student_dirs)} students: {student_dirs}")

    for student_name in student_dirs:
        student_path = os.path.join(dataset_dir, student_name)
        label_map[label_id] = student_name

        image_files = [
            f for f in os.listdir(student_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not image_files:
            log.warning(f"  No images found for {student_name} — skipping")
            label_id += 1
            continue

        student_faces = 0

        for img_file in image_files:
            img_path = os.path.join(student_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                log.warning(f"  Couldn't read {img_path}")
                continue

            # Try to detect and crop face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

            if len(detected) > 0:
                # Use the largest detected face
                x, y, w, h = max(detected, key=lambda r: r[2] * r[3])
                face_crop = img[y:y+h, x:x+w]
            else:
                # No face detected — use whole image (might be pre-cropped)
                face_crop = img

            processed = preprocess_face(face_crop)

            if use_augmentation:
                for aug_face in augment_image(processed):
                    proc = cv2.resize(aug_face, FACE_SIZE) if aug_face.shape[:2] != FACE_SIZE else aug_face
                    if len(proc.shape) == 3:
                        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    faces.append(proc)
                    labels.append(label_id)
                    student_faces += 1
            else:
                faces.append(processed)
                labels.append(label_id)
                student_faces += 1

        log.info(f"  {student_name}: {len(image_files)} images → {student_faces} samples (after augmentation)")
        label_id += 1

    log.info(f"\nTotal training samples: {len(faces)} across {len(label_map)} students")
    return faces, labels, label_map


def train(dataset_dir: str, model_dir: str, use_augmentation: bool = True):
    """
    Full training pipeline:
    1. Load dataset
    2. Train LBPH model
    3. Save model + label map
    """
    if not os.path.exists(dataset_dir):
        print(f"\n❌ Dataset directory not found: {dataset_dir}")
        print("   Create it and add student photo folders inside.\n")
        print("   Structure:")
        print("   dataset/known_faces/")
        print("       Alice_Smith/")
        print("           photo1.jpg")
        print("           photo2.jpg")
        print("           ...")
        sys.exit(1)

    log.info("=" * 50)
    log.info("TRAINING FACE RECOGNITION MODEL")
    log.info("=" * 50)

    # Load data
    faces, labels, label_map = load_dataset(dataset_dir, use_augmentation)
    if faces is None or len(faces) == 0:
        log.error("No training data loaded. Aborting.")
        sys.exit(1)

    # Train LBPH
    log.info("Training LBPH model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,        # pixel radius for circular LBP
        neighbors=8,     # number of neighbors sampled
        grid_x=8,        # number of cells in horizontal direction
        grid_y=8,        # number of cells in vertical direction
    )
    recognizer.train(faces, np.array(labels))

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lbph_model.yml")
    label_map_path = os.path.join(model_dir, "label_map.pkl")

    recognizer.save(model_path)
    with open(label_map_path, "wb") as f:
        pickle.dump(label_map, f)

    log.info(f"\n✓ Model saved: {model_path}")
    log.info(f"✓ Label map saved: {label_map_path}")
    log.info(f"✓ Students enrolled: {list(label_map.values())}")

    print(f"\n{'='*50}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Students enrolled: {len(label_map)}")
    print(f"  Training samples:  {len(faces)}")
    print(f"  Model saved to:    {model_path}")
    print(f"{'='*50}\n")
    print("  You can now run: python attendance_system.py\n")

    return recognizer, label_map


# ─── Demo dataset generator ───────────────────────────────────────────────────

def create_demo_dataset(base_dir: str):
    """
    Creates a demo dataset using your webcam.
    Captures N frames per student — useful for testing.

    Not super necessary but nice to have for quick demos.
    """
    print("\n📸 DEMO DATASET CREATOR")
    print("This will capture face images from your webcam.")
    print("Type student names one by one. Type 'done' to finish.\n")

    os.makedirs(base_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)

    while True:
        name = input("Enter student name (or 'done'): ").strip()
        if name.lower() == "done":
            break
        if not name:
            continue

        student_dir = os.path.join(base_dir, name.replace(" ", "_"))
        os.makedirs(student_dir, exist_ok=True)

        print(f"\nCapturing for {name}...")
        print("Look at the camera. Press SPACE to capture, Q to finish this student.\n")

        count = 0
        while count < 50:  # capture up to 50 images
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display, f"Capturing: {name} [{count}/50]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=capture  Q=done",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                img_path = os.path.join(student_dir, f"img_{count:03d}.jpg")
                cv2.imwrite(img_path, frame)
                count += 1
                print(f"  Saved: {img_path}")
            elif key == ord('q'):
                break

        print(f"Captured {count} images for {name}\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Dataset saved to: {base_dir}")
    print("  Run 'python train.py' to train the model.\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--dataset", type=str, default=STUDENT_DATASET_DIR)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--capture", action="store_true",
                        help="Capture new training images from webcam")
    args = parser.parse_args()

    if args.capture:
        create_demo_dataset(args.dataset)
    else:
        train(
            dataset_dir=args.dataset,
            model_dir=MODEL_DIR,
            use_augmentation=not args.no_augment,
        )
