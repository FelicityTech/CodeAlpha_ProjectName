# Music Generation with AI (LSTM + music21)

Trains an LSTM on real MIDI data to learn note patterns, then generates
new music sequences and writes them out as playable MIDI — and, optionally,
rendered audio.

Bundled with 40 of J.S. Bach's chorales (real classical MIDI data, exported
from music21's built-in corpus — no external download needed) as a working
example. Drop your own `.mid` files (classical, jazz, film scores,
whatever) into `data/midi/` and the entire pipeline works unchanged.

## Project Structure

```
music_generation/
├── data/
│   ├── midi/                    # training MIDI files (40 Bach chorales included)
│   └── processed/                # cached preprocessed sequences (generated)
├── models_saved/
│   ├── best_model.keras           # checkpoint with lowest training loss
│   ├── final_model.keras           # final trained model
│   └── vocab.pkl                    # note<->integer vocabulary mapping
├── output/
│   ├── generated_demo.mid            # example output (included)
│   └── generated_demo.wav             # example output, rendered to audio
├── prepare_data.py                # exports Bach corpus to data/midi/
├── preprocessing.py                # MIDI -> token sequences (music21)
├── model.py                         # LSTM architecture (tf.keras)
├── train.py                          # training script
├── generate.py                        # generates new sequences, writes MIDI
├── audio_render.py                     # MIDI -> WAV (FluidSynth or fallback)
├── test_pipeline.py                     # automated verification tests
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

**Optional, for higher-quality audio rendering:** install FluidSynth and a
General MIDI soundfont. Without these, audio rendering automatically falls
back to a built-in synthesizer (see [Audio Rendering](#audio-rendering)
below) — MIDI generation and playback in any MIDI player always works
regardless.

- **Linux (Debian/Ubuntu):** `sudo apt-get install fluidsynth fluid-soundfont-gm`
- **macOS:** `brew install fluidsynth fluid-synth` and download a GM soundfont
  (e.g. FluidR3_GM.sf2) to `/usr/share/sounds/sf2/`
- **Windows:** download FluidSynth from the [FluidSynth releases
  page](https://github.com/FluidSynth/fluidsynth/releases), add it to your
  PATH, and download a GM soundfont (e.g. FluidR3_GM.sf2) to
  `C:\soundfonts\`

## Usage

**1. Get training data** (already included — re-run to change how many chorales):
```bash
python prepare_data.py --n 60
```

**2. Preprocess into training sequences:**
```bash
python preprocessing.py
```

**3. Train the model:**
```bash
python train.py --epochs 100 --batch-size 64
```

**4. Generate new music:**
```bash
python generate.py --n-tokens 200 --temperature 0.8 --output my_song.mid
```

**5. Render to audio:**
```bash
python audio_render.py output/my_song.mid
```

**Run the test suite:**
```bash
python test_pipeline.py
```

## How It Works

```
data/midi/*.mid  →  preprocessing.py  →  model.py + train.py  →  generate.py  →  audio_render.py
  (real MIDI)         (music21 parses      (LSTM learns            (samples a         (MIDI -> WAV,
                       into note/chord      note-to-note             new note           FluidSynth or
                       tokens)              transition patterns)     sequence)          fallback synth)
```

### 1. Data Collection (`prepare_data.py`)

Exports Bach chorales from music21's bundled corpus (433 available, 40
included by default) as real `.mid` files in `data/midi/`. This satisfies
"collect MIDI data" literally — the training pipeline reads actual MIDI
files from disk, not music21 Score objects directly — while avoiding any
external download or licensing questions, since the corpus ships with
music21 itself.

### 2. Preprocessing (`preprocessing.py`)

Each MIDI file is parsed with music21 and flattened into a single timeline
of notes, chords, and rests. Each event becomes a string token:
- A note → its pitch name, e.g. `"C4"`
- A chord → its pitches joined by dots, e.g. `"C4.E4.G4"`
- A rest → the literal token `"REST"`

These tokens form a vocabulary (51 unique tokens across the 40 bundled
chorales), and the token stream is turned into sliding-window training
pairs: 50 consecutive tokens as input, the 51st as the target — the same
next-token-prediction framing used for text generation.

### 3. Model (`model.py`)

```
Embedding → LSTM(256, return_sequences) → Dropout → LSTM(256) → Dense(256, relu) → Dropout → Dense(vocab_size, softmax)
```
~975K parameters for the included dataset. Trained with sparse categorical
crossentropy (each target is a single vocabulary index, not one-hot) and
Adam.

### 4. Generation (`generate.py`)

Starting from either a random seed or a real seed sequence, the model
predicts a probability distribution over the vocabulary for "what comes
next," one token at a time, feeding each prediction back in as input for
the next step (autoregressive generation). Sampling uses **temperature**
rather than always picking the single most likely token — pure greedy
decoding on a music LSTM collapses into short repeating loops almost
immediately, since the model just keeps re-picking its most confident
choice.

- `temperature < 1.0` → more conservative, more repetitive
- `temperature = 1.0` → samples the model's raw predicted distribution
- `temperature > 1.0` → more experimental, occasionally incoherent

Generated tokens are converted back into `music21` Note/Chord/Rest objects
and written to a real `.mid` file.

### 5. Audio Rendering (`audio_render.py`)

Two paths, automatically chosen:
- **FluidSynth + a General MIDI soundfont**, if installed — a real software
  synthesizer, genuine instrument timbre.
- **Built-in numpy fallback** if FluidSynth isn't available — synthesizes
  each note as a fundamental sine wave plus two quieter harmonics with an
  exponential decay envelope. Not a real piano sound, but zero extra
  dependencies, works everywhere `requirements.txt` already supports, and
  every note is audible at the correct pitch and timing.

## About the Included Demo Model

The trained checkpoint and `output/generated_demo.*` files included in this
project were trained for only a handful of epochs — enough to verify the
entire pipeline genuinely works end-to-end (loss decreasing, valid MIDI
output, valid audio output), **not** enough for musically convincing
results. LSTM music generation typically needs 100+ epochs to produce
output that sounds intentional rather than close-to-random; on this
dataset size, that's a multi-hour CPU training run (a GPU, or a cloud
notebook like Colab, will be dramatically faster). Run `train.py` yourself
with more epochs for real results — `test_pipeline.py` passing confirms
the code is correct; it doesn't confirm the *music* is good, and no amount
of testing can substitute for actually training it properly.

## Known Limitations

- **Single-timeline flattening** — `preprocessing.py` flattens all
  instrument parts into one sequence, losing which instrument played what.
  Fine for the bundled 4-voice chorales (SATB voices get merged into one
  melodic line), but loses information for multi-instrument arrangements.
- **Fixed note duration** — generated notes all get the same duration
  (`note_duration=0.5` quarter notes in `generate.py`); the model doesn't
  currently learn or predict rhythm/duration, only pitch content.
- **No rhythm/dynamics modeling** — velocity (loudness) and precise timing
  nuances from the original performances aren't modeled or reproduced.

## Possible Improvements

- **Predict duration alongside pitch** — extend the vocabulary/model to a
  two-headed output (pitch token + duration bucket) instead of assuming a
  fixed note length, for far more natural-sounding rhythm.
- **Per-instrument sequences** — instead of flattening all parts together,
  train separate sequences per instrument/voice and recombine them at
  generation time.
- **Attention/Transformer architecture** — modern symbolic music models
  (e.g. Music Transformer) use self-attention instead of LSTMs and handle
  long-range musical structure (repeated themes, phrase structure)
  noticeably better than a plain LSTM.
- **GAN-based generation** — the assignment mentions GANs as an
  alternative; they're harder to train stably than an LSTM here but can
  produce more globally coherent pieces rather than token-by-token
  next-note prediction.

## Tech Stack

| Component | Library |
|---|---|
| MIDI parsing, note/chord extraction, MIDI writing | music21 |
| LSTM model, training | TensorFlow / Keras |
| Sequence math | NumPy |
| Audio synthesis | FluidSynth (optional, external) or built-in numpy synth |
