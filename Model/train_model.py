import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_FILE = "training_data.csv"
MODEL_FILE = "chord_model.pkl"
SCALER_FILE = "chord_scaler.pkl"
# ────────────────────────────────────────────────────────────────────────────

# Load data
df = pd.read_csv(INPUT_FILE, header=None)
X = df.iloc[:, 1:].values   # features
y = df.iloc[:, 0].values    # labels

print(f"Loaded {len(X)} samples across {len(set(y))} chords: {sorted(set(y))}")
print(f"Feature vector size: {X.shape[1]}")

#Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#Train SVM
model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
model.fit(X_train, y_train)

#Evaluate - Own interest, not relevant.
y_pred = model.predict(X_test)
print("\n── Classification Report ──────────────────────")
print(classification_report(y_test, y_pred))
print("── Confusion Matrix ───────────────────────────")
print(confusion_matrix(y_test, y_pred, labels=sorted(set(y))))

#Save model and scaler
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
print(f"\nModel saved → {MODEL_FILE}")
print(f"Scaler saved → {SCALER_FILE}")