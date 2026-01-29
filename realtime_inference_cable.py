"""
Script Name: Real-Time Inference (Wired/USB)
=============================================================================
This script performs real-time Human Activity Recognition (HAR) using data
streamed via a USB serial connection (e.g., Arduino/ESP32 via cable).

Operational Flow:
    1.  **Connection:** Establishes a Serial connection to the microcontroller.
    2.  **Buffering:** Collects incoming data into a buffer until 'WINDOW_SIZE' is reached.
    3.  **Pre-processing:** Applies the SAME `StandardScaler` used in training.
        *Critial:* Without scaling, predictions will be random/garbage.
    4.  **Inference:**
        - If Classic Model: Extracts statistical features -> Predicts.
        - If Deep Model: Reshapes to (1, 100, 6) -> Predicts.
    5.  **Smoothing:** Uses a majority vote mechanism (Mode) over the last 5
        predictions to stabilize output.

Requirements:
    - 'processed_data/scaler.pkl' must exist.
    - Trained models must exist in 'models/'.
"""

import serial
import time
import numpy as np
import joblib
import os
import warnings
from scipy.stats import skew, kurtosis
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/cu.usbserial-110"  # CHANGE THIS to your USB Port
BAUD_RATE = 2000000
WINDOW_SIZE = 100
STEP = 50
HISTORY_LIMIT = 5

LABEL_NAMES = [
    "Biceps_Curl",
    "Dumbbell_Shoulder_Shrug",
    "Front_Raise",
    "Lateral_Raise",
    "Sitting",
    "Walking",
]

decision_history = []


def extract_features_realtime(window_data):
    """
    Extracts statistical features from a single data window for real-time inference.

    This function mirrors the logic in `feature_engineering.py` but is optimized
    for a single sample input.

    Args:
        window_data (np.array): Scaled sensor data of shape (WINDOW_SIZE, 6).

    Returns:
        np.array: Feature vector of shape (1, 42).
    """
    features_list = []
    for j in range(6):
        axis_data = window_data[:, j]
        features_list.extend(
            [
                np.mean(axis_data),
                np.std(axis_data),
                np.min(axis_data),
                np.max(axis_data),
                np.median(axis_data),
                skew(axis_data),
                kurtosis(axis_data),
            ]
        )
    return np.array(features_list).reshape(1, -1)


def main():
    """
    Main loop for real-time data capture and classification via USB.
    """
    print("=" * 60)
    print("   WIRED (USB) REAL-TIME HAR SYSTEM")
    print("=" * 60)
    print("\nSelect Model:\n [1] Random Forest\n [2] XGBoost\n [3] CNN\n [4] LSTM")

    choice = input("\nEnter selection (1-4): ").strip()

    model_configs = {
        "1": ("models/random_forest.pkl", "classic", "Random Forest"),
        "2": ("models/xgboost.pkl", "classic", "XGBoost"),
        "3": ("models/cnn_model.keras", "dl", "CNN"),
        "4": ("models/lstm_model.keras", "dl", "LSTM"),
    }

    if choice not in model_configs:
        return
    m_path, m_type, d_name = model_configs[choice]

    print(f"\nLoading Model: {d_name}...")
    try:
        if m_type == "classic":
            model = joblib.load(m_path)
        else:
            import tensorflow as tf

            model = tf.keras.models.load_model(m_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Opening Port: {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        ser.flushInput()
        time.sleep(2)
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    buffer = []
    print("\n[SYSTEM READY]...\n")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or "," not in line:
                    continue

                parts = line.split(",")
                if len(parts) != 6:
                    continue

                try:
                    buffer.append([float(x) for x in parts])
                except ValueError:
                    continue

                if len(buffer) == WINDOW_SIZE:
                    data_np = np.array(buffer)

                    if m_type == "classic":
                        features = extract_features_realtime(data_np)
                        pred_id = int(model.predict(features)[0])
                    else:
                        input_dl = data_np.reshape(1, WINDOW_SIZE, 6)
                        pred_probs = model.predict(input_dl, verbose=0)
                        pred_id = np.argmax(pred_probs, axis=1)[0]

                    label = LABEL_NAMES[pred_id]
                    decision_history.append(label)
                    if len(decision_history) > HISTORY_LIMIT:
                        decision_history.pop(0)
                    final_decision = Counter(decision_history).most_common(1)[0][0]

                    print(
                        f"Instant: {label.ljust(20)} | >>> FINAL: \033[1;32m{final_decision.upper()}\033[0m"
                    )
                    buffer = buffer[STEP:]

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if "ser" in locals():
            ser.close()


if __name__ == "__main__":
    main()
