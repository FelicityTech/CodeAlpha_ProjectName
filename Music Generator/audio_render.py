"""
Renders a MIDI file to a playable WAV file.

Primary path: FluidSynth (a real software synthesizer) + a General MIDI
soundfont, if both are available on the system — this gives genuine
instrument timbre (piano, strings, etc.), not just tones.

Fallback: a small dependency-free synthesizer built with just numpy,
for environments where installing FluidSynth + a soundfont isn't
practical (e.g. some Windows setups). It renders each note as a decaying
sine wave with a couple of harmonics layered in — recognizably
piano-ish, not concert-hall quality, but it always works and needs
nothing beyond what's already in requirements.txt.
"""
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

# Common install locations for a General MIDI soundfont across platforms
SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf3/FluidR3_GM.sf3",
    "C:/soundfonts/FluidR3_GM.sf2",
]


def find_soundfont() -> str | None:
    for path in SOUNDFONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def render_with_fluidsynth(midi_path: Path, wav_path: Path, soundfont: str) -> bool:
    if not shutil.which("fluidsynth"):
        return False

    result = subprocess.run(
        ["fluidsynth", "-ni", soundfont, str(midi_path), "-F", str(wav_path), "-r", "44100"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not wav_path.exists():
        print(f"FluidSynth failed: {result.stderr[:500]}")
        return False
    return True


def render_with_numpy_synth(midi_path: Path, wav_path: Path, sample_rate: int = 44100) -> bool:
    """
    Dependency-free fallback: parse the MIDI with music21, then render each
    note/chord as a sum of a fundamental sine wave plus two quieter
    harmonics, with a simple exponential decay envelope so notes don't end
    abruptly. This is a basic additive synthesizer, not a sample-based one
    — it won't sound like a real piano, but every note will be audible and
    at the correct pitch/timing.
    """
    from music21 import chord, converter, note

    score = converter.parse(midi_path)
    events = score.flatten().notesAndRests

    total_duration_s = float(score.duration.quarterLength) * 0.5 + 2.0  # matches note_duration used in generate.py
    n_samples = int(total_duration_s * sample_rate)
    buffer = np.zeros(n_samples, dtype=np.float64)

    def midi_note_to_freq(midi_number: int) -> float:
        return 440.0 * (2 ** ((midi_number - 69) / 12))

    def synthesize_tone(freq: float, duration_s: float) -> np.ndarray:
        t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
        envelope = np.exp(-3.0 * t / max(duration_s, 1e-6))  # exponential decay
        tone = (
            1.00 * np.sin(2 * np.pi * freq * t)
            + 0.35 * np.sin(2 * np.pi * freq * 2 * t)   # 1st harmonic
            + 0.15 * np.sin(2 * np.pi * freq * 3 * t)   # 2nd harmonic
        )
        return tone * envelope

    for element in events:
        offset_s = float(element.offset) * 0.5  # matches note_duration used in generate.py
        dur_s = max(float(element.duration.quarterLength) * 0.5, 0.1)
        start_sample = int(offset_s * sample_rate)

        if isinstance(element, note.Note):
            pitches = [element.pitch.midi]
        elif isinstance(element, chord.Chord):
            pitches = [p.midi for p in element.pitches]
        else:
            continue  # rests produce silence, nothing to add

        for midi_num in pitches:
            tone = synthesize_tone(midi_note_to_freq(midi_num), dur_s) * 0.2
            end_sample = start_sample + len(tone)
            if end_sample > len(buffer):
                buffer = np.pad(buffer, (0, end_sample - len(buffer)))
            buffer[start_sample:end_sample] += tone

    # Normalize to avoid clipping, then write as 16-bit PCM WAV
    peak = np.max(np.abs(buffer)) if np.max(np.abs(buffer)) > 0 else 1.0
    normalized = (buffer / peak * 0.9 * 32767).astype(np.int16)

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(normalized.tobytes())

    return True


def render_to_audio(midi_path: Path, wav_path: Path | None = None) -> Path:
    midi_path = Path(midi_path)
    if wav_path is None:
        wav_path = midi_path.with_suffix(".wav")
    wav_path = Path(wav_path)

    soundfont = find_soundfont()
    if soundfont and render_with_fluidsynth(midi_path, wav_path, soundfont):
        print(f"Rendered audio with FluidSynth ({Path(soundfont).name}) -> {wav_path}")
        return wav_path

    print("FluidSynth/soundfont not available — using built-in numpy synthesizer fallback")
    render_with_numpy_synth(midi_path, wav_path)
    print(f"Rendered audio with fallback synthesizer -> {wav_path}")
    return wav_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render a MIDI file to WAV audio")
    parser.add_argument("midi_path", help="Path to the input .mid file")
    parser.add_argument("--output", default=None, help="Output .wav path (default: same name as input)")
    args = parser.parse_args()

    render_to_audio(Path(args.midi_path), Path(args.output) if args.output else None)
