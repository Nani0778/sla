import cv2
import mediapipe as mp
import numpy as np
import os

DATA_PATH = "data/raw"
LABEL = "hello"  # Change this per gesture

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sample_count = 0

while cap.isOpened() and sample_count < 100:
    ret, frame = cap.read()
    if not ret:
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        keypoints = np.array(keypoints)

        os.makedirs(DATA_PATH, exist_ok=True)
        np.save(f"{DATA_PATH}/{LABEL}_{sample_count}", keypoints)
        sample_count += 1

    cv2.imshow("Capture", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
