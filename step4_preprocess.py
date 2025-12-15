import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# Use your custom dataset path
DATASET_DIR = "custom_dataset"
IMG_SIZE = 64

gestures = []
labels = []
class_map = {}  # to map numeric labels → gesture names

print("[INFO] Loading dataset...")

# Only consider gesture folders inside custom_dataset
gesture_folders = sorted(os.listdir(DATASET_DIR))
for label, folder in enumerate(gesture_folders):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    class_map[label] = folder  # store mapping: 0 → move_mouse, 1 → left_click

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gestures.append(img.flatten())
        labels.append(label)

gestures = np.array(gestures)
labels = np.array(labels)

if gestures.size == 0:
    print("[ERROR] No images loaded; check dataset path and contents.")
    exit()

print(f"[INFO] Dataset loaded: {gestures.shape[0]} samples, {gestures.shape[1]} features each.")
print(f"[INFO] Classes found: {class_map}")

X_train, X_test, y_train, y_test = train_test_split(
    gestures, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"[INFO] Training samples: {X_train.shape}, Testing samples: {X_test.shape}")

os.makedirs("processed", exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test, class_map), "processed/gesture_data.pkl")

print("[INFO] Preprocessing complete. Saved to processed/gesture_data.pkl")
