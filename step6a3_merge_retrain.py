import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

IMG_SIZE = 64

def load_dataset(base_dir):
    gestures, labels = [], []
    label_map = {}  # map gesture folder name to label id
    label_id = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

                # get gesture name from parent folder (e.g., "01_palm")
                gesture_name = os.path.basename(os.path.dirname(img_path))

                # assign numeric label
                if gesture_name not in label_map:
                    label_map[gesture_name] = label_id
                    label_id += 1

                gestures.append(thresh.flatten())
                labels.append(label_map[gesture_name])

    print(f"[INFO] Loaded {len(gestures)} images from {base_dir} with {len(label_map)} gesture classes.")
    return np.array(gestures), np.array(labels)


print("[INFO] Loading Kaggle dataset...")
X1, y1 = load_dataset("datasets/leapGestRecog")


print("[INFO] Loading My dataset...")
X2, y2 = load_dataset("my_dataset")

# Combine
X = np.vstack([X1, X2])
y = np.hstack([y1, y2])

print(f"[INFO] Combined dataset: {X.shape[0]} samples, {X.shape[1]} features each.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
print("[INFO] Training Random Forest on combined dataset...")
clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Accuracy on combined dataset: {acc:.4f}")
print("[INFO] Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("processed", exist_ok=True)
joblib.dump(clf, "processed/gesture_model_custom.pkl")
print("[INFO] New model saved to processed/gesture_model_custom.pkl")
