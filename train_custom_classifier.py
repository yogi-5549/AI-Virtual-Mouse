import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
DATASET_DIR = "custom_dataset"
MODEL_PATH = "processed/custom_gesture_model.pkl"
IMG_SIZE = 64

# Load dataset
print("[INFO] Loading dataset...")
X, y = [], []
class_map = {}

for idx, folder in enumerate(sorted(os.listdir(DATASET_DIR))):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    class_map[idx] = folder
    print(f"[INFO] Loading class {idx}: {folder}")

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.flatten())
        y.append(idx)

X = np.array(X)
y = np.array(y)

print(f"[INFO] Loaded {len(X)} samples with {len(class_map)} classes.")

# Train classifier
print("[INFO] Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

# Evaluate
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"[INFO] Accuracy: {acc:.4f}")
print("[INFO] Classification Report:")
print(classification_report(y, y_pred))

# Save model
os.makedirs("processed", exist_ok=True)
joblib.dump((clf, class_map), MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
