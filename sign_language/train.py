"""
train.py — Sign Language Model Training
=========================================
Trains a CNN classifier on hand landmark features extracted via MediaPipe.

Architecture:
- Input: 63 features (21 hand landmarks × 3: x, y, z normalised)
- Model: Fully-connected neural network (MLP / dense layers)
  Layer 1: Dense(256, relu) + BatchNorm + Dropout(0.4)
  Layer 2: Dense(128, relu) + BatchNorm + Dropout(0.3)
  Layer 3: Dense(64, relu) + Dropout(0.2)
  Output:  Dense(N_classes, softmax)

Why landmark features not raw images?
- More robust to lighting, skin tone, background
- Much smaller input (63 floats vs 224×224×3 image)
- Trains in minutes on CPU
- Works identically at inference time with live webcam

Dataset:
- ASL Alphabet Dataset (Kaggle) — 87,000 images, 29 classes (A-Z + delete/space/nothing)
- Custom 10 words captured via webcam capture tool (train.py --capture)

Labels (36 total):
  A-Z (26 letters) + Hello, ThankYou, Yes, No, Please, Sorry, Help, Eat, Water, More

Usage:
    python train.py                        # train on dataset/
    python train.py --capture              # capture new training images
    python train.py --eval                 # evaluate saved model
    python train.py --epochs 50 --batch 32
"""

import os
import sys
import cv2
import numpy as np
import pickle
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Try TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    log.info(f"TensorFlow {tf.__version__} loaded ✓")
except ImportError:
    TF_AVAILABLE = False
    log.warning("TensorFlow not found — will train SVM fallback")

# sklearn for SVM fallback + metrics
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

# MediaPipe for landmark extraction
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_hands = mp.solutions.hands
except ImportError:
    MP_AVAILABLE = False
    log.error("MediaPipe not installed: pip install mediapipe")

from config import (
    MODEL_DIR, DATASET_DIR, KNOWN_WORDS,
    ALL_LABELS, IMG_SIZE, LANDMARK_FEATURES,
)


# ─── Landmark extractor ───────────────────────────────────────────────────────

def extract_landmarks(image: np.ndarray, hands_model) -> np.ndarray:
    """
    Extracts 21 hand landmarks from an image using MediaPipe.
    Returns flattened array of 63 values (x, y, z for each landmark),
    normalised relative to bounding box.
    Returns None if no hand detected.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_model.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]
    coords = []
    for pt in lm.landmark:
        coords.extend([pt.x, pt.y, pt.z])

    # Normalise: subtract wrist (landmark 0) to make translation-invariant
    base_x, base_y, base_z = coords[0], coords[1], coords[2]
    normalised = []
    for i in range(0, len(coords), 3):
        normalised.extend([
            coords[i]   - base_x,
            coords[i+1] - base_y,
            coords[i+2] - base_z,
        ])

    return np.array(normalised, dtype=np.float32)


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(dataset_dir: str) -> tuple:
    """
    Walks dataset directory structure:
        dataset/train/
            A/ *.jpg
            B/ *.jpg
            ...
            Hello/ *.jpg
            ...

    Extracts landmark features for each image.
    Returns (X, y_str) where X is (N, 63) and y_str is list of label strings.
    """
    if not MP_AVAILABLE:
        log.error("MediaPipe required for training")
        sys.exit(1)

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )

    X, y = [], []
    label_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not label_dirs:
        log.error(f"No class folders in {dataset_dir}")
        log.info("Structure: dataset/train/A/*.jpg, dataset/train/Hello/*.jpg ...")
        sys.exit(1)

    log.info(f"Found {len(label_dirs)} classes: {label_dirs}")
    skipped = 0

    for label in label_dirs:
        label_path = os.path.join(dataset_dir, label)
        img_files = [
            f for f in os.listdir(label_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        label_count = 0
        for fname in img_files:
            img = cv2.imread(os.path.join(label_path, fname))
            if img is None:
                skipped += 1
                continue

            # Resize for faster processing
            img = cv2.resize(img, IMG_SIZE)
            lm = extract_landmarks(img, hands)

            if lm is None:
                skipped += 1
                continue

            X.append(lm)
            y.append(label)
            label_count += 1

        log.info(f"  {label:15s}: {label_count:4d} samples")

    hands.close()
    log.info(f"\nTotal: {len(X)} samples  |  Skipped: {skipped} (no hand detected)")
    return np.array(X), np.array(y)


# ─── Neural Network model ─────────────────────────────────────────────────────

def build_nn_model(n_classes: int) -> "keras.Model":
    """
    Dense neural network for landmark classification.
    Input: 63 normalised landmark coordinates.
    """
    model = keras.Sequential([
        layers.Input(shape=(LANDMARK_FEATURES,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.4),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(n_classes, activation="softmax"),
    ], name="sign_language_nn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── SVM fallback ────────────────────────────────────────────────────────────

def train_svm(X_train, y_train_enc):
    """
    Trains an SVM classifier as fallback when TensorFlow isn't available.
    Works surprisingly well on 63-feature landmark vectors.
    """
    log.info("Training SVM (TF fallback)...")
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    clf.fit(X_train, y_train_enc)
    return clf


# ─── Training pipeline ────────────────────────────────────────────────────────

def train(dataset_dir: str, model_dir: str, epochs: int = 50, batch_size: int = 32):
    """Full training pipeline."""
    os.makedirs(model_dir, exist_ok=True)

    log.info("=" * 55)
    log.info("  SIGN LANGUAGE MODEL TRAINING")
    log.info("=" * 55)

    # ── Load data ────────────────────────────────────────────────────────────
    train_path = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_path):
        # Try using dataset_dir directly
        train_path = dataset_dir

    X, y_str = load_dataset(train_path)

    if len(X) == 0:
        log.error("No training data loaded. Check dataset structure.")
        sys.exit(1)

    # ── Encode labels ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    n_classes = len(le.classes_)
    log.info(f"\nClasses ({n_classes}): {list(le.classes_)}")

    # ── Scale features ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train/val split ──────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    log.info(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # ── Train model ──────────────────────────────────────────────────────────
    if TF_AVAILABLE:
        log.info(f"\nTraining Neural Network ({epochs} epochs)...")
        model = build_nn_model(n_classes)
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Save NN
        model_path = os.path.join(model_dir, "sign_model.h5")
        model.save(model_path)
        log.info(f"\n✓ NN model saved: {model_path}")

        # Plot training history
        _plot_history(history, model_dir)

        # Evaluate
        y_pred = model.predict(X_val).argmax(axis=1)

    else:
        log.info("\nTraining SVM fallback...")
        clf = train_svm(X_train, y_train)
        y_pred = clf.predict(X_val)

        svm_path = os.path.join(model_dir, "sign_svm.pkl")
        with open(svm_path, "wb") as f:
            pickle.dump(clf, f)
        log.info(f"✓ SVM model saved: {svm_path}")

    # ── Save label encoder + scaler ──────────────────────────────────────────
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_val, y_pred)
    log.info(f"\n{'='*55}")
    log.info(f"  Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    log.info(f"{'='*55}")

    print(classification_report(y_val, y_pred,
                                  target_names=le.classes_))

    # Confusion matrix
    _plot_confusion_matrix(y_val, y_pred, le.classes_, model_dir)

    log.info(f"\n✓ Training complete! All files saved to: {model_dir}")
    print(f"\nRun:  python app.py  to launch the GUI")


def _plot_history(history, model_dir: str):
    """Plot training/val accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train", color="#00e5ff")
    axes[0].plot(history.history["val_accuracy"], label="Val", color="#76ff03")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    axes[1].plot(history.history["loss"], label="Train", color="#ff5722")
    axes[1].plot(history.history["val_loss"], label="Val", color="#ff9800")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle("Training History — Sign Language NN", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_history.png"), dpi=150)
    plt.close()
    log.info("✓ Training history plot saved")


def _plot_confusion_matrix(y_true, y_pred, classes, model_dir: str):
    """Plot normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_size = max(10, len(classes) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(cm_norm, annot=len(classes) <= 20,
                xticklabels=classes, yticklabels=classes,
                cmap="Blues", fmt=".2f", ax=ax,
                linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix — Sign Language")

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    log.info("✓ Confusion matrix saved")


# ─── Webcam capture tool ──────────────────────────────────────────────────────

def capture_training_data(dataset_dir: str):
    """
    Interactive webcam tool to capture training images.
    Creates dataset/train/<label>/ folders with captured images.
    """
    if not MP_AVAILABLE:
        print("MediaPipe required for capture: pip install mediapipe")
        return

    import tkinter as tk
    from tkinter import simpledialog

    print("\n📸 TRAINING DATA CAPTURE")
    print("=" * 45)
    print(f"Available labels: {', '.join(ALL_LABELS)}")
    print("=" * 45)

    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)

    while True:
        label = input("\nEnter label to capture (or 'done'): ").strip()
        if label.lower() == "done":
            break
        if label not in ALL_LABELS:
            print(f"  ⚠ '{label}' not in known labels. Allowed: {ALL_LABELS}")
            cont = input("  Capture anyway? (y/n): ").strip().lower()
            if cont != "y":
                continue

        save_dir = os.path.join(dataset_dir, "train", label)
        os.makedirs(save_dir, exist_ok=True)

        existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
        print(f"\nCapturing for '{label}'  (existing: {existing} images)")
        print("Press SPACE to capture, Q to finish this label\n")

        cap = cv2.VideoCapture(0)
        count = 0

        while count < 200:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display = frame.copy()
            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        display, hlm, mp_hands.HAND_CONNECTIONS)
                cv2.putText(display, "HAND DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No hand", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, f"{label}  [{count}/200]  SPACE=capture  Q=done",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
            cv2.imshow("Capture Training Data", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                fname = os.path.join(save_dir, f"{label}_{existing+count:04d}.jpg")
                cv2.imwrite(fname, frame)
                count += 1
                print(f"  Saved {count}/200", end="\r")
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n  ✓ Captured {count} images for '{label}'")

    hands.close()
    print("\nCapture session complete.")
    print(f"Dataset saved to: {dataset_dir}/train/")
    print("Run: python train.py  to train the model")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Model Training")
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--capture", action="store_true",
                        help="Capture training images via webcam")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate saved model on val set")
    args = parser.parse_args()

    if args.capture:
        capture_training_data(args.dataset)
    else:
        train(
            dataset_dir=args.dataset,
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch,
        )
