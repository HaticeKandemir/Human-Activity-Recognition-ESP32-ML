"""
Script Name: Deep Learning Model Trainer
=============================================================================
This script optimizes and trains Deep Learning models (CNN and LSTM) using
Keras Tuner for hyperparameter optimization.

Key Features:
    - **Architecture:** 1D-CNN (Spatial features) and LSTM (Temporal features).
    - **Input:** Scaled 3D time-series data (Batch, 100, 6).
    - **Tuning:** Optimizes filters, kernel sizes, units, dropout, and learning rates.
    - **Callbacks:** Implements EarlyStopping and ModelCheckpoint.
    - **Visualization:** Plots training history (Accuracy/Loss) and Confusion Matrices.

Input: Scaled arrays (X_train.npy, etc.) from the preprocessing step.
Output: Best .keras models in 'models/', tuner logs in 'tuner_results/',
        and metrics in 'results/'.
"""

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras_tuner as kt
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

INPUT_DIR = "processed_data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
TUNER_DIR = "tuner_results"
RANDOM_SEED = 47
MAX_TRIALS = 5  # Increase to 10 for even better results if you have time

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

for folder in [MODEL_DIR, RESULTS_DIR, TUNER_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_cm(y_true, y_pred, classes, name):
    """
    Generates and saves the Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"cm_{name.lower()}.png"))
    plt.close()


def plot_history(history, name):
    """
    Plots and saves the training accuracy and loss curves.

    Args:
        history (History): The Keras History object returned by fit().
        name (str): Model name for the plot title.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"history_{name.lower()}.png"))
    plt.close()


def build_cnn(hp):
    """
    Constructs a 1D Convolutional Neural Network (CNN) for Keras Tuner.

    Hyperparameters tuned:
    - Kernel Size (3 vs 5)
    - Number of Filters (Layer 1 & 2)
    - Dropout Rate
    - Dense Units
    - Learning Rate

    Args:
        hp (kt.HyperParameters): Hyperparameter container.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential()

    # Tunable Kernel Size (3 or 5)
    hp_kernel = hp.Choice("kernel_size", values=[3, 5])

    hp_filters1 = hp.Int("filters_1", 32, 128, step=32)
    model.add(Conv1D(filters=hp_filters1, kernel_size=hp_kernel, activation="relu"))
    model.add(MaxPooling1D(2))

    model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))

    hp_filters2 = hp.Int("filters_2", 64, 256, step=64)
    model.add(Conv1D(filters=hp_filters2, kernel_size=hp_kernel, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Flatten())

    model.add(Dense(hp.Int("dense_units", 32, 128, step=32), activation="relu"))
    model.add(Dense(6, activation="softmax"))

    hp_lr = hp.Choice("learning_rate", [1e-2, 1e-3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(hp):
    """
    Constructs a Long Short-Term Memory (LSTM) network for Keras Tuner.

    Hyperparameters tuned:
    - LSTM Units (Layer 1 & 2)
    - Dropout Rate
    - Dense Units
    - Learning Rate

    Args:
        hp (kt.HyperParameters): Hyperparameter container.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential()

    hp_lstm_1 = hp.Int("lstm_units_1", 32, 128, step=32)
    model.add(LSTM(hp_lstm_1, return_sequences=True))
    model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))

    hp_lstm_2 = hp.Int("lstm_units_2", 32, 128, step=32)
    model.add(LSTM(hp_lstm_2))

    model.add(Dense(hp.Int("dense_units", 32, 64, step=16), activation="relu"))
    model.add(Dense(6, activation="softmax"))

    hp_lr = hp.Choice("learning_rate", [1e-2, 1e-3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    """
    Main driver for the Deep Learning training and tuning process.
    """
    print(">>> 4. DEEP LEARNING TRAINING STARTED")
    X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"))

    le = joblib.load(os.path.join(INPUT_DIR, "label_encoder.pkl"))
    classes = le.classes_

    y_train_cat = to_categorical(y_train)

    builders = {"CNN": build_cnn, "LSTM": build_lstm}
    results = []

    for name, builder in builders.items():
        print(f"\n--- Tuning {name} ---")
        tuner = kt.RandomSearch(
            builder,
            objective="val_accuracy",
            max_trials=MAX_TRIALS,
            executions_per_trial=1,
            directory=TUNER_DIR,
            project_name=f"{name}_tuning",
            overwrite=True,
        )

        tuner.search(X_train, y_train_cat, epochs=12, validation_split=0.2, verbose=1)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best Params: {best_hps.values}")

        model = tuner.hypermodel.build(best_hps)
        path = os.path.join(MODEL_DIR, f"{name.lower()}_model.keras")
        callbacks = [
            EarlyStopping("val_loss", patience=6, restore_best_weights=True),
            ModelCheckpoint(path, "val_loss", save_best_only=True),
        ]

        print(f"Training BEST {name} fully...")
        start = time.time()
        hist = model.fit(
            X_train,
            y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )
        train_time = time.time() - start

        plot_history(hist, name)

        start_inf = time.time()
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        latency = (time.time() - start_inf) / len(X_test)

        acc = accuracy_score(y_test, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        save_cm(y_test, y_pred, classes, name)

        print(f"FINAL {name}: Acc={acc:.4f}")
        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "F1_Score": f1,
                "Train_Time_Sec": train_time,
                "Inference_Latency_Sec": latency,
                "Best_Params": str(best_hps.values),
            }
        )

    pd.DataFrame(results).to_csv(
        os.path.join(RESULTS_DIR, "results_deep.csv"), index=False
    )
    print("\nDeep Learning Completed.")


if __name__ == "__main__":
    main()
