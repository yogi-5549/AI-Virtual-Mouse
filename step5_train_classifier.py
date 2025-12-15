import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("[INFO] Loading preprocessed data...")
X_train, X_test, y_train, y_test, class_map = joblib.load("processed/gesture_data.pkl")

print(f"[INFO] Training classifier on {len(class_map)} classes: {class_map}")

# Train Random Forest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Model accuracy: {acc * 100:.2f}%")

# Save model along with class_map
joblib.dump((clf, class_map), "processed/custom_gesture_model.pkl")
print("[INFO] Model saved to processed/custom_gesture_model.pkl")
