def extract_fingertips(hand_landmarks):
    #Remove additional joints from Media Pipe, focus only on fingertips.
    TIP_IDS = [4, 8, 12, 16, 20] #From MediaPipe

    fingertips = [
        (
            hand_landmarks.landmark[i].x,
            hand_landmarks.landmark[i].y,
            hand_landmarks.landmark[i].z
        )
        for i in TIP_IDS
    ]

    return {
        "thumb": fingertips[0],
        "index": fingertips[1],
        "middle": fingertips[2],
        "ring": fingertips[3],
        "pinky": fingertips[4],
    }