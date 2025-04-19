import cv2
import mediapipe as mp
from app.gesture_predictor import predict_gesture
from app.speech_output import speak
from utils.grammar_corrector import correct_grammar
from utils.feedback_loop import update_model_with_feedback

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
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
            print("Detected:", corrected)
            speak(corrected)

    cv2.imshow("Sign Language App", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
