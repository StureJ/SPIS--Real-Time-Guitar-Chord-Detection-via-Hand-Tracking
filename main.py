import cv2
import mediapipe as mp
import joblib
from collections import deque

from filtering import extract_fingertips
from feature_distances import compute_distances
from windowing import WindowBuffer
from window_features import window_to_features

model = joblib.load("Model/chord_model.pkl")
scaler = joblib.load("Model/chord_scaler.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

window = WindowBuffer(size=20)
median_filter = None
buf = deque(maxlen=10)
cap = cv2.VideoCapture(1)

import numpy as np
from collections import deque

class MedianFilter:
    def __init__(self, num_features, kernel_size=3):
        self.kernel_size = kernel_size
        self.buffers = [deque(maxlen=kernel_size) for _ in range(num_features)]

    def apply(self, x):
        filtered = []

        for i, val in enumerate(x):
            self.buffers[i].append(val)
            filtered.append(np.median(self.buffers[i]))

        return np.array(filtered)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        _, frame = cap.read()

        #BOX 1: Webcam & hand tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        chord = None
        if not results.multi_hand_landmarks:
            cv2.imshow("Hand Signal", cv2.flip(bgr, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            continue

        hand = results.multi_hand_landmarks[0]

        #BOX 2: Fingertip extraction/selection.
        fingertips = extract_fingertips(hand)

        #BOX 3: Feature extraction (distances)
        distances = compute_distances(fingertips)

        #Median filter to reduce the jittering.
        if median_filter is None:
            median_filter = MedianFilter(num_features=len(distances), kernel_size=3) #kernel size of 3 to reduce latency

        # Apply the filtering
        distances = median_filter.apply(distances)

        #BOX 4: Windowing
        window.add(distances)

        #BOX 5: window > ML features > prediction
        if window.is_full():
            features = scaler.transform(window_to_features(window.get_window()).reshape(1, -1))
            raw = model.predict(features)[0]
            buf.append(raw)
            chord = max(set(buf), key=buf.count)

        mp_drawing.draw_landmarks(
            bgr, hand, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        out = cv2.flip(bgr, 1)

        if chord:
            cv2.putText(out, f"Chord: {chord}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Hand Signal", out)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()