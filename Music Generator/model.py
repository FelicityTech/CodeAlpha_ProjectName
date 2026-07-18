"""
LSTM model for note-sequence generation.

Standard next-token-prediction architecture: an Embedding layer turns
integer-encoded note/chord tokens into dense vectors, two stacked LSTM
layers learn temporal patterns across the sequence, and a final softmax
Dense layer predicts a probability distribution over the vocabulary for
"what note/chord comes next."
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(vocab_size: int, sequence_length: int, embedding_dim: int = 100,
                 lstm_units: int = 256, dropout: float = 0.3) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(sequence_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(lstm_units),
        layers.Dense(lstm_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check on the architecture with dummy sizes
    m = build_model(vocab_size=50, sequence_length=50)
    m.summary()
