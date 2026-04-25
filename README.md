# SPIS

![demo](example.gif)


🎸 Overview
A real-time system that detects guitar chords from hand movements using signal processing and machine learning.

⚙️ How it works
Tracks hand landmarks with MediaPipe
Extracts fingertip distances
Applies median filtering to stabilize noisy hand tracking signals 
Applies sliding window
Generates features over time
Predicts chords using a trained SVM model

🧠 Tech Stack
Python • OpenCV • MediaPipe • NumPy • Scikit-learn
