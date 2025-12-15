import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
clf = joblib.load("processed/gesture_model.pkl")

# Parameters
IMG_SIZE = 64

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

print("[INFO] Starting live gesture prediction... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_class = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of the hand
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))

            # Crop and preprocess
            margin = 20
            xmin, ymin = max(0, xmin - margin), max(0, ymin - margin)
            xmax, ymax = min(w, xmax + margin), min(h, ymax + margin)

            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size > 0:
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                # Apply threshold to make it binary (closer to LeapGestRecog dataset style)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

                feature = thresh.flatten().reshape(1, -1)
                # Predict gesture
                predicted_class = clf.predict(feature)[0]

    # Display prediction
    if predicted_class is not None:
        cv2.putText(frame, f"Predicted: {predicted_class}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Step 6A - Live Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
