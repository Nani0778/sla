import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/gesture_model.h5")
GESTURE_LABELS = ['hello', 'thanks', 'yes', 'no']  # Update with actual labels

def predict_gesture(keypoints):
    keypoints = np.array(keypoints).reshape(1, -1)
    prediction = model.predict(keypoints)
    return GESTURE_LABELS[np.argmax(prediction)]
