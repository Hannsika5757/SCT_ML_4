# asl_letter_translator_fixed.py
import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
from collections import deque
import time

# ===== CONFIG =====
MODEL_FILE = "asl_cnn_model.h5"
CLASS_FILE = "class_indices.pkl"
IMG_SIZE = (64, 64)       # Match the training size
SMOOTHING = 5             # Frames to average predictions
LETTER_THRESHOLD = 7      # Consecutive frames to confirm a letter
CONFIDENCE_THRESHOLD = 0.7

# ===== Load model & class labels =====
model = tf.keras.models.load_model(MODEL_FILE)
with open(CLASS_FILE, "rb") as f:
    class_indices = pickle.load(f)
indices_class = {v: k for k, v in class_indices.items()}

# ===== MediaPipe Hands =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== Queues for smoothing & building words =====
pred_queue = deque(maxlen=SMOOTHING)
letter_queue = deque(maxlen=LETTER_THRESHOLD)
current_word = ""

# ===== Start webcam =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

prev_time = 0
print("🎥 ASL Letter Translator (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    letter_detected = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Compute bounding box
        h, w, _ = frame.shape
        x_min = w
        y_min = h
        x_max = y_max = 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        pad = 15
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

        # Crop hand
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size != 0:
            hand_resized = cv2.resize(hand_img, IMG_SIZE)
            hand_normalized = hand_resized / 255.0
            pred_input = np.expand_dims(hand_normalized, axis=0)

            # Predict
            pred = model.predict(pred_input, verbose=0)[0]
            pred_queue.append(pred)
            avg_pred = np.mean(pred_queue, axis=0)
            idx = np.argmax(avg_pred)
            confidence = avg_pred[idx]

            if confidence > CONFIDENCE_THRESHOLD:
                letter_detected = indices_class[idx]
                letter_queue.append(letter_detected)

                # Confirm letter after consecutive frames
                if len(letter_queue) == LETTER_THRESHOLD and all(l == letter_detected for l in letter_queue):
                    current_word += letter_detected
                    letter_queue.clear()

            # Display label & bounding box
            cv2.putText(frame, f"{indices_class[idx]} ({confidence*100:.1f}%)", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    else:
        pred_queue.clear()
        letter_queue.clear()

    # Show current word
    cv2.putText(frame, f"Word: {current_word}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("ASL Letter Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"👋 Final Word: {current_word}")
