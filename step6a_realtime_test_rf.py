import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import joblib
import time

# Load model & class map
clf, class_map = joblib.load("processed/custom_gesture_model.pkl")

IMG_SIZE = 64
last_action_time = 0
COOLDOWN = 1.0  # seconds

# Mediapipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

print("[INFO] Starting real-time gesture control (2 gestures)... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

            margin = 20
            xmin, ymin = max(0, xmin - margin), max(0, ymin - margin)
            xmax, ymax = min(w, xmax + margin), min(h, ymax + margin)

            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size > 0:
                gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                feature = gray.flatten().reshape(1, -1)

                # Prediction
                pred = clf.predict(feature)[0]
                gesture = class_map[pred]
                cv2.putText(frame, f"{gesture}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Execute action with cooldown
                current_time = time.time()
                if current_time - last_action_time > COOLDOWN:
                    if gesture == "move_mouse":
                        x, y = int(hand_landmarks.landmark[8].x * screen_w), int(hand_landmarks.landmark[8].y * screen_h)
                        pyautogui.moveTo(x, y, duration=0.1)
                    elif gesture == "left_click":
                        pyautogui.click()
                        last_action_time = current_time

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
