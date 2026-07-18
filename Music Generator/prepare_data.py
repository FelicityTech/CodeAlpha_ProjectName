"""
Collects MIDI training data.

Primary source: J.S. Bach's chorales, bundled directly with music21's
corpus (433 pieces, no internet download required). These get converted
to real .mid files in data/midi/, so the rest of the pipeline works on
actual MIDI data exactly as the assignment specifies — not on music21's
internal Score objects directly.

You can also just drop your own .mid files (classical, jazz, whatever)
into data/midi/ — preprocessing.py globs that folder regardless of where
the files came from.
"""
import argparse
from pathlib import Path

from music21 import corpus

MIDI_DIR = Path(__file__).parent / "data" / "midi"


def export_bach_chorales(n: int = 60):
    """Export the first n Bach chorales from music21's bundled corpus as .mid files."""
    MIDI_DIR.mkdir(parents=True, exist_ok=True)

    chorale_paths = corpus.getComposer("bach")[:n]
    exported = 0
    skipped = 0

    for path in chorale_paths:
        try:
            score = corpus.parse(path)
            out_path = MIDI_DIR / f"{path.stem}.mid"
            score.write("midi", fp=str(out_path))
            exported += 1
        except Exception as e:
            # A handful of corpus entries are fragments or have parsing quirks;
            # skip them rather than letting one bad file kill the whole export.
            skipped += 1
            print(f"  skipped {path.name}: {e}")

    print(f"Exported {exported} Bach chorales to {MIDI_DIR} ({skipped} skipped)")
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training MIDI data")
    parser.add_argument(
        "--n", type=int, default=60,
        help="Number of Bach chorales to export (max 433). Default: 60",
    )
    args = parser.parse_args()
    export_bach_chorales(args.n)
