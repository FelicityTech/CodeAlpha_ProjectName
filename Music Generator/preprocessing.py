"""
Preprocessing: turns a folder of MIDI files into integer-encoded note
sequences suitable for training an LSTM, using music21 to parse each file.

Encoding scheme:
- A single note becomes its pitch name + octave, e.g. "C4"
- A chord becomes its pitches joined by dots, e.g. "C4.E4.G4" (music21's
  own convention for representing chords as strings)
- Rests become the literal token "REST"

Each of these string tokens is treated as one "word" in a vocabulary,
exactly the same way word-level text generation treats words — which is
what makes an LSTM a reasonable architecture for this at all.
"""
import pickle
from pathlib import Path

import numpy as np
from music21 import chord, converter, note

MIDI_DIR = Path(__file__).parent / "data" / "midi"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def extract_notes(midi_dir: Path = MIDI_DIR) -> list[str]:
    """
    Parse every .mid file in midi_dir and return one flat list of note/chord
    tokens, in order, concatenated across all files.
    """
    midi_files = sorted(Path(midi_dir).glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(
            f"No .mid files found in {midi_dir}. Run prepare_data.py first, "
            f"or drop your own .mid files into that folder."
        )

    all_tokens = []
    for midi_path in midi_files:
        try:
            score = converter.parse(midi_path)
        except Exception as e:
            print(f"  skipping unparseable file {midi_path.name}: {e}")
            continue

        # flatten() merges all parts/voices into one timeline of events —
        # simpler than modeling multiple simultaneous instrument parts,
        # at the cost of losing which instrument played what.
        elements = score.flatten().notesAndRests

        for element in elements:
            if isinstance(element, note.Note):
                all_tokens.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                all_tokens.append(".".join(str(p) for p in element.pitches))
            elif isinstance(element, note.Rest):
                all_tokens.append("REST")

    print(f"Parsed {len(midi_files)} MIDI files -> {len(all_tokens)} note/chord/rest tokens")
    return all_tokens


def build_vocabulary(tokens: list[str]) -> tuple[dict, dict]:
    """Returns (token_to_int, int_to_token) mappings over the unique tokens seen."""
    unique_tokens = sorted(set(tokens))
    token_to_int = {tok: i for i, tok in enumerate(unique_tokens)}
    int_to_token = {i: tok for tok, i in token_to_int.items()}
    return token_to_int, int_to_token


def create_sequences(tokens: list[str], token_to_int: dict, sequence_length: int = 50):
    """
    Slide a window of `sequence_length` tokens across the token stream to
    build (input_sequence -> next_token) training pairs — the standard
    next-token-prediction framing for sequence models.

    Returns (X, y) as integer arrays: X shape (n_samples, sequence_length),
    y shape (n_samples,).
    """
    encoded = [token_to_int[t] for t in tokens]

    X, y = [], []
    for i in range(len(encoded) - sequence_length):
        X.append(encoded[i:i + sequence_length])
        y.append(encoded[i + sequence_length])

    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def prepare_training_data(midi_dir: Path = MIDI_DIR, sequence_length: int = 50, save: bool = True):
    """Full pipeline: parse MIDI -> tokens -> vocabulary -> (X, y) sequences."""
    tokens = extract_notes(midi_dir)
    token_to_int, int_to_token = build_vocabulary(tokens)
    X, y = create_sequences(tokens, token_to_int, sequence_length)

    print(f"Vocabulary size: {len(token_to_int)} unique tokens")
    print(f"Training sequences: {X.shape[0]} (each {sequence_length} tokens long)")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PROCESSED_DIR / "X.npy", X)
        np.save(PROCESSED_DIR / "y.npy", y)
        with open(PROCESSED_DIR / "vocab.pkl", "wb") as f:
            pickle.dump({"token_to_int": token_to_int, "int_to_token": int_to_token,
                         "sequence_length": sequence_length}, f)
        print(f"Saved processed data to {PROCESSED_DIR}")

    return X, y, token_to_int, int_to_token


if __name__ == "__main__":
    prepare_training_data()
