"""
capture.py

Records normalized MediaPipe hand landmarks labeled with ASL letters,
and saves them to a CSV for training a classifier.

Usage:
    python3 capture.py

Controls:
    - Press a letter key (a-y, excluding j and z, which involve motion)
      to START capturing samples for that letter. Landmarks are saved
      every frame while capturing is active.
    - Press the SAME letter key again (or SPACE) to STOP capturing.
    - Press 'q' to quit and save the CSV.

Tips:
    - Aim for ~150-200 samples per letter.
    - Move your hand slightly (angle, distance, position) while capturing
      so the model doesn't overfit to one exact pose.
    - Watch the on-screen sample counter for the current letter.
"""

import csv
import os
import string
import time

import cv2
import mediapipe as mp

# --- Config ---------------------------------------------------------------

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "landmarks.csv")

# Static letters only -- J and Z involve motion and need a different
# approach (sequence data), so we skip them in Phase 1.
VALID_LETTERS = set(string.ascii_lowercase) - {"j", "z"}

NUM_LANDMARKS = 21  # MediaPipe hand model always returns 21 points

# --- Landmark normalization -------------------------------------------------


def normalize_landmarks(hand_landmarks):
    """
    Convert MediaPipe's raw (x, y, z) landmarks into a scale- and
    position-invariant feature vector.

    Steps:
      1. Translate so the wrist (landmark 0) is the origin.
      2. Scale so the farthest landmark from the wrist has distance 1.

    This means the model learns hand *shape*, not where your hand is
    in the frame or how close it is to the camera.
    """
    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    wrist_x, wrist_y, wrist_z = coords[0]
    translated = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in coords]

    max_dist = max(
        (x**2 + y**2 + z**2) ** 0.5 for x, y, z in translated
    )
    if max_dist == 0:
        max_dist = 1e-6  # avoid divide-by-zero on a degenerate frame

    normalized = [
        (x / max_dist, y / max_dist, z / max_dist) for x, y, z in translated
    ]

    # Flatten to a single 63-length feature vector: x0,y0,z0,x1,y1,z1,...
    flat = []
    for x, y, z in normalized:
        flat.extend([x, y, z])
    return flat


# --- Setup -------------------------------------------------------------


def ensure_output_file():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_FILE)
    if not file_exists:
        header = ["label"] + [
            f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
        ]
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    return OUTPUT_FILE


def main():
    output_path = ensure_output_file()

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
        raise RuntimeError(
            "Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1)."
        )

    csv_file = open(output_path, "a", newline="")
    writer = csv.writer(csv_file)

    # Running counts per letter, loaded from any existing data so counts
    # persist across sessions.
    counts = {letter: 0 for letter in VALID_LETTERS}
    if os.path.isfile(output_path):
        with open(output_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1

    current_label = None
    capturing = False
    last_capture_time = 0.0
    capture_interval = 0.05  # seconds between saved samples (~20/sec max)

    print("Press a letter key (a-y, excluding j/z) to start/stop capturing.")
    print("Press 'q' to quit and save.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_present = bool(results.multi_hand_landmarks)

            if hand_present:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                if capturing and current_label is not None:
                    now = time.time()
                    if now - last_capture_time >= capture_interval:
                        features = normalize_landmarks(hand_landmarks)
                        writer.writerow([current_label] + features)
                        counts[current_label] += 1
                        last_capture_time = now

            # --- On-screen overlay ---
            status_color = (0, 0, 255) if not capturing else (0, 255, 0)
            status_text = (
                f"Capturing: {current_label.upper()}"
                if capturing and current_label
                else "Paused -- press a letter key"
            )
            cv2.putText(
                frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2,
            )

            if current_label is not None:
                count_text = f"{current_label.upper()} samples: {counts[current_label]}"
                cv2.putText(
                    frame, count_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )

            if not hand_present:
                cv2.putText(
                    frame, "No hand detected", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            total = sum(counts.values())
            cv2.putText(
                frame, f"Total samples: {total}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )

            cv2.imshow("ASL Data Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key) if key != 255 else None

            if key_char == "q":
                break
            elif key_char == " ":
                capturing = False
            elif key_char in VALID_LETTERS:
                if capturing and current_label == key_char:
                    # Pressing the same letter again stops capture
                    capturing = False
                else:
                    current_label = key_char
                    capturing = True

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        csv_file.close()

        print("\nSession summary:")
        for letter in sorted(counts):
            print(f"  {letter.upper()}: {counts[letter]} samples")
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
