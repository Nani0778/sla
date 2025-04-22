import flet as ft
import threading
import cv2
import mediapipe as mp
import sys
import os
import time

# Import your app and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.gesture_predictor import predict_gesture
from app.speech_output import speak
from utils.grammar_corrector import correct_grammar
from utils.feedback_loop import update_model_with_feedback

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Global variable to update UI
recognized_text = ""

def gesture_recognition_loop(update_text_callback):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                keypoints = []
                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                gesture = predict_gesture(keypoints)
                corrected = correct_grammar(gesture)
                update_text_callback(corrected)
                speak(corrected)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Flet UI function
def main(page: ft.Page):
    page.title = "Sign Language to Speech"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    output_text = ft.Text("Waiting for gesture...", size=30)
    page.add(output_text)

    def update_ui(text):
        output_text.value = f"Detected: {text}"
        page.update()

    # Run gesture recognition in a separate thread
    threading.Thread(target=gesture_recognition_loop, args=(update_ui,), daemon=True).start()

ft.page(target=main)
