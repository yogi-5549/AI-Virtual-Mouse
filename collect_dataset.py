import cv2
import os
import time

# Define your 6 gestures
GESTURES = {
    1: "move_mouse",
    2: "left_click",
 
}

DATASET_DIR = "custom_dataset"
IMG_SIZE = 64  # keep consistent with training
CAPTURE_INTERVAL = 0.2  # seconds between saved frames

# Create directories if not exist
for g_id, g_name in GESTURES.items():
    os.makedirs(os.path.join(DATASET_DIR, f"{g_id}_{g_name}"), exist_ok=True)

cap = cv2.VideoCapture(0)

print("[INFO] Starting dataset collection...")
print("Instructions:")
print(" - Show the gesture when prompted on screen.")
print(" - Press SPACE to start auto-saving frames.")
print(" - Press 'n' to move to the next gesture.")
print(" - Press 'q' to quit.")

for g_id, g_name in GESTURES.items():
    print(f"\n[INFO] Collecting for gesture {g_id}: {g_name}")
    count = 0
    save = False
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror-like preview
        frame = cv2.flip(frame, 1)

        # Display instructions on screen
        cv2.putText(frame, f"Gesture: {g_id} - {g_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE=Start/Stop Save | n=Next | q=Quit",
                    (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # Toggle saving with SPACE
        if key == ord(" "):
            save = not save
            print("[INFO] Saving:" if save else "[INFO] Paused saving")

        # Save frame at interval
        if save and (time.time() - last_capture_time) >= CAPTURE_INTERVAL:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            save_path = os.path.join(DATASET_DIR, f"{g_id}_{g_name}", f"{count}.png")
            cv2.imwrite(save_path, gray)
            count += 1
            last_capture_time = time.time()

        # Move to next gesture
        if key == ord("n"):
            print(f"[INFO] Finished gesture {g_id} ({count} images)")
            break

        # Quit program
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("[INFO] Dataset collection finished.")
cap.release()
cv2.destroyAllWindows()
