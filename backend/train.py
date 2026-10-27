"""Train a CNN from a CSV containing FEN positions and normalized targets."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import chess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from app import encode_board

def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if not {"fen", "target"}.issubset(frame.columns):
        raise ValueError("CSV must contain columns named 'fen' and 'target'")
    frame = frame.dropna(subset=["fen", "target"])
    boards, targets = [], []
    for index, row in frame.iterrows():
        try:
            board = chess.Board(str(row["fen"]))
            if not board.is_valid():
                raise ValueError("invalid board")
            target = float(row["target"])
            if not -1.0 <= target <= 1.0:
                raise ValueError("target must be in [-1, 1]")
            boards.append(encode_board(board))
            targets.append(target)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid row at CSV index {index}: {exc}") from exc
    if len(boards) < 10:
        raise ValueError("Provide at least 10 valid rows; a much larger dataset is recommended.")
    return np.asarray(boards, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def build_model():
    from tensorflow import keras
    layers = keras.layers
    model = keras.Sequential([
        keras.Input(shape=(8, 8, 12)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(96, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="tanh"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="CSV with fen,target columns")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/position_cnn.keras"))
    args = parser.parse_args()
    np.random.seed(args.seed)
    x, y = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=args.seed
    )
    from tensorflow import keras
    model = build_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(args.output), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        x_train, y_train, validation_split=0.2, epochs=args.epochs,
        batch_size=args.batch_size, callbacks=callbacks, verbose=1,
    )
    predictions = model.predict(x_test, verbose=0).reshape(-1)
    mse = mean_squared_error(y_test, predictions)
    mae = float(np.mean(np.abs(y_test - predictions)))
    print(f"Rows: {len(x)} | train: {len(x_train)} | test: {len(x_test)}")
    print(f"Best validation MSE: {min(history.history['val_loss']):.6f}")
    print(f"Held-out test MSE: {mse:.6f}")
    print(f"Held-out test MAE: {mae:.6f}")
    print(f"Saved model: {args.output.resolve()}")

if __name__ == "__main__":
    main()
