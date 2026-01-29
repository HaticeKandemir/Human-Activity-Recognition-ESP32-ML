"""
Script Name: Data Preprocessor
=============================================================================
This script serves as the foundation of the HAR (Human Activity Recognition)
pipeline. It handles the transformation of raw sensor data into a format
suitable for Machine Learning and Deep Learning models.

Pipeline Stages:
    1.  **Load:** Reads raw CSV sensor data from the 'human_activity' folder.
    2.  **Clean:** Removes identifiers from labels (e.g., 'Walking_12' -> 'Walking').
    3.  **Windowing:** Applies a Sliding Window technique (Size: 100, Step: 50).
    4.  **Split:** Performs a stratified train/test split (80/20).
    5.  **Scale:** Applies StandardScaler to normalize sensor data (Mean=0, Var=1).
        *Critical:* The scaler is fitted only on training data to avoid leakage
        and saved for real-time inference.
    6.  **Save:** Exports processed arrays (.npy) and objects (.pkl).

Input: Raw CSV files in 'human_activity/'.
Output: Processed .npy files and .pkl objects in 'processed_data/'.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# --- CONFIGURATION ---
RANDOM_SEED = 47
WINDOW_SIZE = 100
STEP_SIZE = 50
DATA_PATH = "human_activity"
OUTPUT_DIR = "processed_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def clean_label(label_str):
    """
    Standardizes label strings by removing numeric suffixes.

    Example: 'Walking_123' becomes 'Walking'.

    Args:
        label_str (str): The raw label string from the CSV.

    Returns:
        str: The cleaned label string. Returns 'Unknown' if input is not a string.
    """
    if not isinstance(label_str, str):
        return "Unknown"
    parts = label_str.split("_")
    if parts[-1].isdigit():
        return "_".join(parts[:-1]).strip()
    return label_str.strip()


def load_data(folder_path):
    """
    Scans the specified folder, reads CSV files, and generates sliding windows.

    This function iterates through all CSV files, cleans the labels, handles
    missing values, and segments the continuous sensor stream into fixed-size windows.

    Args:
        folder_path (str): Path to the directory containing raw CSV files.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - X_list (np.array): Sensor data windows. Shape: (N_samples, 100, 6).
            - y_list (np.array): Corresponding labels. Shape: (N_samples,).
            Returns (None, None) if no files are found.
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"ERROR: No files found in {folder_path}")
        return None, None

    X_list = []
    y_list = []
    print(f"Processing {len(all_files)} files...")

    for file in all_files:
        try:
            df = pd.read_csv(file)
            if df.empty or len(df) < WINDOW_SIZE:
                continue

            # Label Processing
            raw_label = df["Label"].iloc[0]
            label = clean_label(raw_label)

            # Feature Selection & Cleaning
            sensor_cols = [
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "accel_x",
                "accel_y",
                "accel_z",
            ]
            # Force numeric, handle errors, fill NaNs
            data = (
                df[sensor_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .values.astype("float32")
            )

            # Sliding Window
            for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
                window = data[i : i + WINDOW_SIZE]
                X_list.append(window)
                y_list.append(label)

        except Exception as e:
            print(f"Skipping {os.path.basename(file)}: {e}")

    return np.array(X_list), np.array(y_list)


def main():
    """
    Main execution driver for the data preprocessing pipeline.
    """
    print(">>> 1. DATA PREPROCESSING STARTED")

    # 1. Load Raw Windows
    X, y = load_data(DATA_PATH)
    if X is None:
        return

    print(f"Total Windows Created: {len(X)}")

    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    # 3. Stratified Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=RANDOM_SEED, stratify=y_encoded
    )

    # 4. SCALING (CRITICAL STEP)
    # We must reshape to (N*100, 6) to fit the scaler, then reshape back.
    print("Applying StandardScaler...")
    N_train, T, F = X_train.shape
    N_test, _, _ = X_test.shape

    scaler = StandardScaler()

    # Fit on TRAIN data only to prevent data leakage
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, T, F)
    X_test_scaled = scaler.transform(X_test_flat).reshape(N_test, T, F)

    # Save Scaler (Needed for Real-Time Inference)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    # 5. Save Processed & Scaled Data
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y, palette="viridis")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))

    print("\nPreprocessing & Scaling Completed Successfully.")
    print(f"Train Set Shape: {X_train_scaled.shape}")
    print(f"Test Set Shape:  {X_test_scaled.shape}")


if __name__ == "__main__":
    main()
