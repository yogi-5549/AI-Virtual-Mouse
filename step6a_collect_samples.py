import cv2
import os

# Parameters
SAVE_DIR = "my_dataset"
IMG_SIZE = 64
SAMPLES_PER_CLASS = 200   # how many images per gesture

# Create folders 0–9
for i in range(10):
    os.makedirs(os.path.join(SAVE_DIR, str(i)), exist_ok=True)

cap = cv2.VideoCapture(0)
print("[INFO] Press a number key (0–9) to start recording that gesture.")
print("[INFO] Press 'q' to quit.")

current_label = None
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # If recording
    if current_label is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        save_path = os.path.join(SAVE_DIR, str(current_label), f"{count}.png")
        cv2.imwrite(save_path, gray)
        count += 1
        cv2.putText(display, f"Recording {current_label}: {count}/{SAMPLES_PER_CLASS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if count >= SAMPLES_PER_CLASS:
            print(f"[INFO] Finished class {current_label}")
            current_label = None
            count = 0

    cv2.imshow("Sample Collector", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(10)]:
        current_label = int(chr(key))
        count = 0
        print(f"[INFO] Recording gesture {current_label}")

cap.release()
cv2.destroyAllWindows()
