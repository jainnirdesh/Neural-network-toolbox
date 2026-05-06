"""
lstm_pipeline.py
----------------
Complete, modular LSTM example using TensorFlow / Keras.

Functions provided (as requested):
- preprocess_data()
- create_sequences()
- build_lstm_model()
- train_model()
- evaluate_model()

Usage:
 - Run without args to use a synthetic sine wave dataset.
 - Or pass `--csv PATH --col COLUMN` to use your own CSV time-series data.

This script is beginner-friendly, with comments explaining each step.
"""

from typing import Tuple, Optional
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def preprocess_data(
    series: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    scale: bool = True,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Preprocess raw 1D series data.

    Steps:
    - Handles edge cases (nan, insufficient length).
    - Optionally scales data using MinMaxScaler to [0,1].
    - Splits into train / validation / test sets.

    Returns: X_train_series, X_val_series, X_test_series, scaler
    """
    if series is None:
        raise ValueError("Input series is None")
    series = np.asarray(series).astype(float)
    if np.isnan(series).any():
        # Simple strategy: forward-fill then back-fill any remaining NaNs
        s = pd.Series(series)
        s = s.fillna(method="ffill").fillna(method="bfill")
        series = s.values

    if len(series) < 10:
        raise ValueError("Series too short for training (need at least 10 values)")

    if scale:
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        else:
            series = scaler.transform(series.reshape(-1, 1)).flatten()
    else:
        scaler = None

    # Train / temp (val+test) split
    train_series, temp_series = train_test_split(series, test_size=(test_size + val_size), shuffle=False)
    # Now split temp into val and test maintaining order
    val_portion = val_size / (test_size + val_size)
    val_series, test_series = train_test_split(temp_series, test_size=(1 - val_portion), shuffle=False)

    return train_series, val_series, test_series, scaler


def create_sequences(series: np.ndarray, seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1D series into LSTM input sequences.

    Returns arrays X (num_samples, seq_len, 1) and y (num_samples,)
    """
    if series is None:
        raise ValueError("Series is None")
    series = np.asarray(series).astype(float)
    if len(series) <= seq_len:
        raise ValueError("Series length must be greater than sequence length")

    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def build_lstm_model(input_shape: Tuple[int, int], units: int = 50, dropout: float = 0.2) -> Sequential:
    """Builds a Sequential LSTM model with dropout and a dense output.

    - input_shape: (timesteps, features)
    - units: number of LSTM units
    - dropout: dropout rate between LSTM layers
    """
    model = Sequential()
    # First LSTM layer (returns sequences so we can stack another LSTM)
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    # Second LSTM layer (no return_sequences) for final representation
    model.add(LSTM(units // 2))
    model.add(Dropout(dropout))
    # Final dense layer for regression (single output)
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    model_dir: str = "models",
) -> Tuple[Sequential, dict]:
    """Train the model with EarlyStopping and ModelCheckpoint.

    Returns trained model and the training history dict.
    """
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "lstm_best.h5")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_path = os.path.join(model_dir, "lstm_final.h5")
    model.save(final_path)

    return model, history.history


def evaluate_model(model: Sequential, history: dict, X_test: np.ndarray, y_test: np.ndarray, scaler=None):
    """Evaluate and plot results: training curves and prediction vs actual.

    scaler: if provided, used to inverse-transform predictions and targets.
    """
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Predictions
    preds = model.predict(X_test)
    preds = preds.flatten()
    y_true = y_test.flatten()

    # Inverse scale if needed
    if scaler is not None:
        try:
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        except Exception:
            # If inverse transform fails, just continue with scaled values
            pass

    # Plot prediction vs actual
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title("Prediction vs Actual")
    plt.xlabel("Samples")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print simple numeric evaluation
    mse = np.mean((preds - y_true) ** 2)
    mae = np.mean(np.abs(preds - y_true))
    print(f"Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")


def _generate_sine(length: int = 1000, freq: float = 0.02, noise: float = 0.0) -> np.ndarray:
    """Generate a sine wave time series for demo and testing.
    This is useful when a user doesn't provide a dataset.
    """
    x = np.arange(length)
    y = np.sin(2 * np.pi * freq * x)
    if noise > 0:
        y = y + np.random.normal(scale=noise, size=y.shape)
    return y


def main(args):
    # Load data: either from CSV or synthetic
    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        df = pd.read_csv(args.csv)
        if args.col not in df.columns:
            raise ValueError(f"Column {args.col} not found in CSV")
        series = df[args.col].values
        project_goal = args.goal or "User-supplied time-series prediction"
        dataset_details = f"CSV: {args.csv}, column: {args.col}"
    else:
        series = _generate_sine(length=1200, freq=0.02, noise=0.05)
        project_goal = args.goal or "Sine wave prediction demo"
        dataset_details = "Synthetic sine wave"

    print("Project goal:", project_goal)
    print("Dataset details:", dataset_details)

    # Preprocess
    seq_len = args.seq_len
    train_s, val_s, test_s, scaler = preprocess_data(series, test_size=args.test_size, val_size=args.val_size)

    # Create sequences
    X_train, y_train = create_sequences(train_s, seq_len)
    X_val, y_val = create_sequences(val_s, seq_len)
    X_test, y_test = create_sequences(test_s, seq_len)

    # Edge case: if any set is empty because series halves are too short
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError("One of the train/val/test splits is too small for the chosen sequence length. Reduce seq_len or provide more data.")

    # Build model
    model = build_lstm_model(input_shape=(seq_len, 1), units=args.units, dropout=args.dropout)
    print(model.summary())

    # Train
    model, history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
    )

    # Evaluate
    evaluate_model(model, history, X_test, y_test, scaler=scaler)

    # Example prediction: take last seq_len values from test set / series
    example_input = series[-seq_len:]
    if scaler is not None:
        example_input = scaler.transform(example_input.reshape(-1, 1)).flatten()
    X_example = example_input.reshape(1, seq_len, 1)
    pred = model.predict(X_example).flatten()
    if scaler is not None:
        try:
            pred_val = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()[0]
        except Exception:
            pred_val = pred[0]
    else:
        pred_val = pred[0]

    print(f"Example prediction (next value): {pred_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM example pipeline (TensorFlow/Keras)")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file with time-series column")
    parser.add_argument("--col", type=str, default="value", help="Column name to use from CSV")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for LSTM input")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test set proportion")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation set proportion")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--units", type=int, default=64, help="LSTM units")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--goal", type=str, default=None, help="Project goal description")

    args = parser.parse_args()
    main(args)
