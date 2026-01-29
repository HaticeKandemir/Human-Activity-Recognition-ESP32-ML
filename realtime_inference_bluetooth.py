"""
Script Name: Real-Time Inference (Wired/Bluetooth-Serial)
=============================================================================
This script provides real-time inference capabilities for data streams
coming through a serial interface (which can be a USB cable or a
Bluetooth SPP connection mapping to a serial port).

Features:
    - **Scaler Integration:** Loads the specific `StandardScaler` used during
      training to ensure the live data distribution matches the training data.
    - **Model Flexibility:** Supports both Feature-based (Classic) and
      Raw-signal (Deep Learning) models.
    - **Smoothing Logic:** Implements a sliding window buffer for input and
      a history buffer for output stabilization.

Usage:
    Run the script and select the desired model (1-4). ensure the SERIAL_PORT
    constant matches your device's address.
"""

import serial
import time
import numpy as np
import joblib
import os
import warnings
from scipy.stats import skew, kurtosis
from collections import Counter

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/cu.usbserial-110"
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
    Extracts statistical features from a scaled window for real-time inference.

    This function must match the logic in `feature_engineering.py` EXACTLY
    to ensure model compatibility.

    Args:
        window_data (np.array): Scaled sensor data of shape (WINDOW_SIZE, 6).

    Returns:
        np.array: Reshaped feature vector (1, 42).
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
    Main driver loop for handling Serial data and performing inference.
    """
    print("=" * 60)
    print("   WIRED (USB) REAL-TIME HAR SYSTEM (SCALED)")
    print("=" * 60)
    print("\n[1] Random Forest  [2] XGBoost  [3] CNN  [4] LSTM")
    choice = input("Select Model: ").strip()

    model_configs = {
        "1": ("models/random_forest.pkl", "classic", "Random Forest"),
        "2": ("models/xgboost.pkl", "classic", "XGBoost"),
        "3": ("models/cnn_model.keras", "dl", "CNN"),
        "4": ("models/lstm_model.keras", "dl", "LSTM"),
    }

    if choice not in model_configs:
        print("Invalid selection.")
        return
    m_path, m_type, d_name = model_configs[choice]

    # 1. LOAD RESOURCES
    print(f"\nLoading Scaler and Model ({d_name})...")
    try:
        # Load Scaler (Essential!)
        scaler = joblib.load("processed_data/scaler.pkl")

        if m_type == "classic":
            model = joblib.load(m_path)
        else:
            import tensorflow as tf

            model = tf.keras.models.load_model(m_path)
    except Exception as e:
        print(f"Error loading resources: {e}")
        print("Did you run 'data_preprocessor.py' and training scripts?")
        return

    # 2. CONNECT USB
    print(f"Opening Serial Port: {SERIAL_PORT}...")
    try:
        # Lower timeout is usually fine for wired connections
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        ser.flushInput()
        time.sleep(2)  # Wait for connection to stabilize
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    buffer = []
    print("\n[SYSTEM READY] Listening for data...\n")

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
                    vals = [float(x) for x in parts]
                    buffer.append(vals)
                except ValueError:
                    continue

                # Buffer Full? Make Prediction
                if len(buffer) == WINDOW_SIZE:
                    # A. Convert to Numpy
                    raw_window = np.array(buffer)

                    # B. APPLY SCALING (Match Training)
                    scaled_window = scaler.transform(raw_window)

                    # C. Predict
                    if m_type == "classic":
                        features = extract_features_realtime(scaled_window)
                        pred_id = int(model.predict(features)[0])
                    else:
                        input_dl = scaled_window.reshape(1, WINDOW_SIZE, 6)
                        pred_probs = model.predict(input_dl, verbose=0)
                        pred_id = np.argmax(pred_probs, axis=1)[0]

                    label = LABEL_NAMES[pred_id]

                    # D. Smoothing
                    decision_history.append(label)
                    if len(decision_history) > HISTORY_LIMIT:
                        decision_history.pop(0)
                    final_dec = Counter(decision_history).most_common(1)[0][0]

                    # E. Print Result
                    print(
                        f"Instant: {label.ljust(20)} | FINAL: \033[1;32m{final_dec.upper()}\033[0m"
                    )

                    # Slide Window
                    buffer = buffer[STEP:]

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
