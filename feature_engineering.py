"""
Script Name: Feature Engineering
=============================================================================
This script handles the extraction of statistical features from the preprocessed
(and scaled) sensor windows. These features are designed for traditional
Machine Learning models (Random Forest, XGBoost, etc.) that cannot accept
raw 3D arrays as input.

Methodology:
    - Input: Scaled window data of shape (N_samples, 100, 6).
    - Process: Flattens the time dimension by calculating statistics per axis.
    - Features per Axis: Mean, Std Dev, Min, Max, Median, Skewness, Kurtosis.
    - Total Features: 6 axes * 7 stats = 42 features per sample.

Input: 'X_train.npy', 'X_test.npy' from the 'processed_data' directory.
Output: 'X_train_features.npy', 'X_test_features.npy' (Shape: N, 42).
"""

import numpy as np
import os
from scipy.stats import skew, kurtosis

INPUT_DIR = "processed_data"


def extract_statistical_features(X_windows):
    """
    Extracts statistical features from a batch of sensor windows.

    For each of the 6 sensor axes (Gyro X/Y/Z, Accel X/Y/Z), this function
    calculates 7 statistical metrics, resulting in a 1D feature vector of size 42.

    Args:
        X_windows (np.array): Input 3D array of shape (N_samples, 100, 6).

    Returns:
        np.array: A 2D Feature Matrix of shape (N_samples, 42).
    """
    features_list = []
    for i in range(X_windows.shape[0]):
        window = X_windows[i]
        window_features = []
        for j in range(6):  # For each axis
            axis_data = window[:, j]
            window_features.extend(
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
        features_list.append(window_features)
    return np.array(features_list)


def main():
    """
    Main driver for the feature engineering process.
    """
    print(">>> 2. FEATURE ENGINEERING STARTED")

    # 1. Load Scaled Windows
    try:
        X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    except FileNotFoundError:
        print("Error: Processed data not found. Run 'data_preprocessor.py' first.")
        return

    # 2. Extract Features
    print("Extracting features for Training Set...")
    X_train_features = extract_statistical_features(X_train)

    print("Extracting features for Test Set...")
    X_test_features = extract_statistical_features(X_test)

    # 3. Save Features
    np.save(os.path.join(INPUT_DIR, "X_train_features.npy"), X_train_features)
    np.save(os.path.join(INPUT_DIR, "X_test_features.npy"), X_test_features)

    print("\nFeature Engineering Completed.")
    print(f"Feature Matrix Shape: {X_train_features.shape}")


if __name__ == "__main__":
    main()
