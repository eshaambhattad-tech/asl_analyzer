"""
infer.py

Real-time ASL alphabet recognition from webcam.
Loads the trained classifier and runs inference on every frame.

Usage:
    python3 infer.py

Press 'q' to quit.
"""

import pickle

import cv2
import mediapipe as mp

MODEL_PATH = "model/asl_classifier.pkl"
CONFIDENCE_THRESHOLD = 0.5  # below this, show "?" instead of a letter

with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)


def normalize_landmarks(hand_landmarks):
    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    wrist_x, wrist_y, wrist_z = coords[0]
    translated = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in coords]

    max_dist = max((x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated)
    if max_dist == 0:
        max_dist = 1e-6

    normalized = [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated]

    flat = []
    for x, y, z in normalized:
        flat.extend([x, y, z])
    return flat


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        features = normalize_landmarks(hand_landmarks)
        proba = clf.predict_proba([features])[0]
        confidence = proba.max()
        predicted = clf.classes_[proba.argmax()]

        if confidence >= CONFIDENCE_THRESHOLD:
            label_text = predicted.upper()
            color = (0, 255, 0)
        else:
            label_text = "?"
            color = (0, 165, 255)

        cv2.putText(frame, label_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)
        cv2.putText(frame, f"{confidence:.0%}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(frame, "No hand", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
