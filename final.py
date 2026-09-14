import cv2
import numpy as np
import os
import pickle
import sqlite3
from tensorflow.keras.models import load_model

# ==========================
#       LOAD MODEL
# ==========================
MODEL_PATH = "cnn_model_final.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file not found: cnn_model_final.h5")

model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# ==========================
#       LOAD LABELS
# ==========================
if not os.path.exists("labels.txt"):
    raise FileNotFoundError("❌ labels.txt not found!")

labels = [line.strip() for line in open("labels.txt", "r").readlines()]
print(f"✅ Loaded {len(labels)} labels.")

# ==========================
#   LOAD HAND HISTOGRAM
# ==========================
def get_hand_hist():
    if not os.path.exists("hist"):
        raise FileNotFoundError("❌ hist file not found! Run set_hand_histogram.py first.")
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

hist = get_hand_hist()
print("✅ Loaded histogram!")

# ==========================
#      IMAGE SIZE
# ==========================
# Use first gesture image to determine size
sample_img = None
for root, dirs, files in os.walk("gestures"):
    if files:
        sample_img = cv2.imread(os.path.join(root, files[0]), 0)
        break

if sample_img is None:
    raise FileNotFoundError("❌ No gesture sample images found in gestures/")

image_x, image_y = sample_img.shape
print(f"📏 Image size: {image_x} x {image_y}")

# ==========================
#   PREPROCESS FUNCTIONS
# ==========================
def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_x, image_y))
    gray = gray.reshape(1, image_x, image_y, 1) / 255.0
    return gray

# ==========================
#   PREDICT FUNCTION
# ==========================
def predict_gesture(img):
    pred = model.predict(img)
    class_id = np.argmax(pred)
    confidence = np.max(pred)
    if class_id < len(labels):
        return labels[class_id], confidence
    return f"Class {class_id}", confidence

# ==========================
#    CAMERA LOGIC
# ==========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("⚠ Camera 0 failed. Trying camera 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ ERROR: No camera available!")

print("🎥 Webcam started — press Q to quit")

x, y, w, h = 300, 100, 300, 300  # region of interest

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Frame read failed.")
        continue

    frame = cv2.flip(frame, 1)

    # Draw region of interest
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y + h, x:x + w]

    # Predict
    processed = preprocess_roi(roi)
    gesture_name, confidence = predict_gesture(processed)

    # Display
    cv2.putText(frame, f"{gesture_name} ({confidence:.2f})",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited successfully.")

