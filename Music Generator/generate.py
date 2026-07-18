"""
Generates a new note sequence from a trained model and writes it out as a
real MIDI file, using music21 to convert tokens back into Note/Chord
objects and a Stream.

Sampling uses temperature-scaled random sampling (not greedy argmax) —
greedy decoding on a music LSTM tends to collapse into short repeating
loops almost immediately, since the model just keeps picking its single
most-confident next note.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from music21 import chord, instrument, note, stream
from tensorflow import keras

MODEL_DIR = Path(__file__).parent / "models_saved"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_trained_model(model_dir: Path = MODEL_DIR):
    model_path = model_dir / "final_model.keras"
    if not model_path.exists():
        model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. Run train.py first."
        )

    model = keras.models.load_model(model_path)

    with open(model_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    return model, vocab["token_to_int"], vocab["int_to_token"], vocab["sequence_length"]


def sample_with_temperature(probabilities: np.ndarray, temperature: float = 1.0) -> int:
    """
    Temperature < 1.0 sharpens the distribution (more conservative,
    repetitive output). Temperature > 1.0 flattens it (more surprising,
    occasionally incoherent output). 1.0 = sample from the model's raw
    predicted distribution unchanged.
    """
    if temperature <= 0:
        return int(np.argmax(probabilities))

    logits = np.log(np.clip(probabilities, 1e-10, 1.0)) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    scaled_probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(scaled_probs), p=scaled_probs))


def generate_tokens(model, token_to_int, int_to_token, sequence_length: int,
                     n_tokens: int = 200, temperature: float = 1.0, seed_tokens: list[str] | None = None):
    vocab_size = len(token_to_int)

    if seed_tokens:
        seed_encoded = [token_to_int.get(t, np.random.randint(vocab_size)) for t in seed_tokens]
    else:
        seed_encoded = list(np.random.randint(0, vocab_size, size=sequence_length))

    # Pad/truncate seed to exactly sequence_length
    if len(seed_encoded) < sequence_length:
        pad = list(np.random.randint(0, vocab_size, size=sequence_length - len(seed_encoded)))
        seed_encoded = pad + seed_encoded
    seed_encoded = seed_encoded[-sequence_length:]

    generated = list(seed_encoded)
    pattern = list(seed_encoded)

    for _ in range(n_tokens):
        input_seq = np.array(pattern[-sequence_length:]).reshape(1, sequence_length)
        probabilities = model.predict(input_seq, verbose=0)[0]
        next_idx = sample_with_temperature(probabilities, temperature)

        generated.append(next_idx)
        pattern.append(next_idx)

    generated_tokens = [int_to_token[i] for i in generated[sequence_length:]]  # drop the seed itself
    return generated_tokens


def tokens_to_midi(tokens: list[str], output_path: Path, note_duration: float = 0.5,
                    instrument_name: str = "Piano"):
    """Converts a list of note/chord/REST tokens back into a real MIDI file."""
    output_stream = stream.Stream()
    output_stream.append(instrument.fromString(instrument_name) if hasattr(instrument, "fromString")
                          else instrument.Piano())

    offset = 0.0
    for token in tokens:
        if token == "REST":
            r = note.Rest()
            r.duration.quarterLength = note_duration
            r.offset = offset
            output_stream.append(r)
        elif "." in token:
            pitches = token.split(".")
            new_chord = chord.Chord(pitches)
            new_chord.duration.quarterLength = note_duration
            new_chord.offset = offset
            output_stream.append(new_chord)
        else:
            new_note = note.Note(token)
            new_note.duration.quarterLength = note_duration
            new_note.offset = offset
            output_stream.append(new_note)
        offset += note_duration

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stream.write("midi", fp=str(output_path))
    print(f"Saved generated MIDI to {output_path}")


def generate(n_tokens: int = 200, temperature: float = 1.0, output_name: str = "generated.mid"):
    model, token_to_int, int_to_token, sequence_length = load_trained_model()
    tokens = generate_tokens(model, token_to_int, int_to_token, sequence_length,
                              n_tokens=n_tokens, temperature=temperature)
    output_path = OUTPUT_DIR / output_name
    tokens_to_midi(tokens, output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new music from a trained model")
    parser.add_argument("--n-tokens", type=int, default=200, help="Number of notes/chords to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="Sampling temperature: <1 conservative, >1 experimental")
    parser.add_argument("--output", default="generated.mid")
    args = parser.parse_args()

    generate(n_tokens=args.n_tokens, temperature=args.temperature, output_name=args.output)
