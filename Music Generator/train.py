"""
Trains the LSTM on preprocessed note sequences.

Run preprocessing.py first (or let this script call it automatically if
processed data isn't found yet).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from tensorflow import keras

from model import build_model
from preprocessing import PROCESSED_DIR, prepare_training_data

MODEL_DIR = Path(__file__).parent / "models_saved"


def load_or_prepare_data(sequence_length: int = 50):
    x_path, y_path, vocab_path = (
        PROCESSED_DIR / "X.npy", PROCESSED_DIR / "y.npy", PROCESSED_DIR / "vocab.pkl"
    )
    if x_path.exists() and y_path.exists() and vocab_path.exists():
        X, y = np.load(x_path), np.load(y_path)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"Loaded cached processed data: {X.shape[0]} sequences, "
              f"vocab size {len(vocab['token_to_int'])}")
        return X, y, vocab["token_to_int"], vocab["int_to_token"]

    print("No cached processed data found, running preprocessing...")
    X, y, token_to_int, int_to_token = prepare_training_data(sequence_length=sequence_length)
    return X, y, token_to_int, int_to_token


def train(epochs: int = 50, batch_size: int = 64, sequence_length: int = 50, val_split: float = 0.1):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, token_to_int, int_to_token = load_or_prepare_data(sequence_length)
    vocab_size = len(token_to_int)

    model = build_model(vocab_size=vocab_size, sequence_length=sequence_length)
    model.summary()

    checkpoint_path = MODEL_DIR / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="loss", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
    )

    # Save the final model + vocabulary together so generate.py is self-contained
    model.save(MODEL_DIR / "final_model.keras")
    with open(MODEL_DIR / "vocab.pkl", "wb") as f:
        pickle.dump({
            "token_to_int": token_to_int,
            "int_to_token": int_to_token,
            "sequence_length": sequence_length,
        }, f)

    print(f"\nModel + vocabulary saved to {MODEL_DIR}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the music generation LSTM")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=50)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length)
