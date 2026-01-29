"""
Script Name: Classic Model Trainer
=============================================================================
This script trains and evaluates traditional Machine Learning models:
Random Forest, XGBoost, and AdaBoost.

Key Features:
    - **Grid Search:** Uses GridSearchCV with 3-fold Cross Validation to find
      optimal hyperparameters.
    - **Optimization:** Explores deep trees, various learning rates, and
      estimators.
    - **Evaluation:** Calculates Accuracy, F1-Score, and Inference Latency.
    - **Visualization:** Generates Confusion Matrices for performance analysis.
    - **Output:** Saves trained models (.pkl) and a results summary (.csv).

Input: Feature matrices (X_train_features.npy, etc.) from feature engineering.
Output: Trained models in 'models/' and metrics in 'results/'.
"""

import numpy as np
import pandas as pd
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

INPUT_DIR = "processed_data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
RANDOM_SEED = 47

for folder in [MODEL_DIR, RESULTS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_cm(y_true, y_pred, classes, name):
    """
    Generates and saves a Confusion Matrix heatmap.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names.
        name (str): Name of the model (for filename and title).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"cm_{name.lower()}.png"))
    plt.close()


def perform_grid_search(model, param_grid, X_train, y_train, name):
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        model (estimator): The scikit-learn compatible model instance.
        param_grid (dict): Dictionary of parameters to test.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        name (str): Name of the model for logging.

    Returns:
        tuple: (best_estimator, training_duration, best_params_string)
    """
    print(f"\n>>> Grid Search for: {name}...")
    grid = GridSearchCV(
        model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1
    )

    start = time.time()
    grid.fit(X_train, y_train)
    duration = time.time() - start

    print(f"Best Params: {grid.best_params_}")
    return grid.best_estimator_, duration, str(grid.best_params_)


def evaluate(model, X_test, y_test):
    """
    Evaluates the model on the test set and measures latency.

    Args:
        model (estimator): The trained model.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.

    Returns:
        tuple: (predictions, accuracy, f1_score, latency_per_sample)
    """
    start = time.time()
    y_pred = model.predict(X_test)
    latency = (time.time() - start) / len(X_test)

    acc = accuracy_score(y_test, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    return y_pred, acc, f1, latency


def main():
    """
    Main driver for training classic ML models.
    """
    print(">>> 3. CLASSIC MODEL TRAINING STARTED")
    X_train = np.load(
        os.path.join(INPUT_DIR, "X_train_features.npy"), allow_pickle=True
    )
    X_test = np.load(os.path.join(INPUT_DIR, "X_test_features.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"), allow_pickle=True)

    le = joblib.load(os.path.join(INPUT_DIR, "label_encoder.pkl"))
    classes = le.classes_

    # EXPANDED PARAMETER GRIDS
    models_config = {
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "criterion": ["gini", "entropy"],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=RANDOM_SEED, eval_metric="mlogloss"),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.1, 0.2, 0.3],  # Increased LR
                "max_depth": [3, 6, 9],  # Deeper trees
                "subsample": [0.8, 1.0],
            },
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(algorithm="SAMME", random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 1.0, 1.5],
            },
        },
    }

    results = []
    for name, config in models_config.items():
        best_model, train_time, best_params = perform_grid_search(
            config["model"], config["params"], X_train, y_train, name
        )

        y_pred, acc, f1, latency = evaluate(best_model, X_test, y_test)
        save_cm(y_test, y_pred, classes, name)

        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name.lower()}.pkl"))

        print(f"--> {name}: Acc={acc:.4f}, F1={f1:.4f}")
        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "F1_Score": f1,
                "Train_Time_Sec": train_time,
                "Inference_Latency_Sec": latency,
                "Best_Params": best_params,
            }
        )

    pd.DataFrame(results).to_csv(
        os.path.join(RESULTS_DIR, "results_classic.csv"), index=False
    )
    print("\nClassic Training Completed.")


if __name__ == "__main__":
    main()
