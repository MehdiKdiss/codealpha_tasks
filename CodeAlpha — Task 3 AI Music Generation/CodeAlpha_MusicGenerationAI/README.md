# 🎹 AI Piano Music Generation — CodeAlpha Task 3

**Artificial Intelligence Internship — CodeAlpha**
An LSTM-based generative model trained on real human piano performances, capable of composing new, original piano music — plus an interactive web app for one-click generation and playback.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Tokenization Scheme](#tokenization-scheme)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Known Edge Cases & Limitations](#known-edge-cases--limitations)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Credits](#credits)

---

## Overview

This project generates new piano music using a deep learning model trained on the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) dataset — ~200 hours of real, expressively-performed classical piano recordings captured with sub-millisecond MIDI precision from the International Piano-e-Competition.

**Approach:** music is represented as a sequence of discrete symbolic events (not raw audio, not a fixed piano-roll grid). A 3-layer LSTM is trained to predict the next event in the sequence, then sampled autoregressively at generation time to compose new pieces — the same core idea as a text language model, applied to music.

The result is a full pipeline: raw MIDI → tokenized event sequences → trained LSTM → newly generated token sequences → reconstructed MIDI → optional rendered audio — plus a Streamlit app that wraps the whole thing behind a "Generate" button, no terminal required to use it.

## Demo

Run the interactive app (see [Setup & Usage](#setup--usage)) for one-click generation:
- Adjustable temperature (creativity) and generation length
- In-browser audio playback + MIDI/WAV download
- Browse and replay previously generated files, or upload your own `.mid`

Pre-generated samples are available in `outputs/generated_midi/` and `outputs/generated_audio/`.

---

## Architecture

| Component | Detail |
|---|---|
| Model type | 3-layer stacked LSTM language model |
| Embedding dimension | 256 |
| Hidden size | 512 |
| Layers | 3 |
| Dropout | 0.3 |
| Total parameters | ~8.5M |
| Vocabulary size | 3,557 tokens |
| Framework | PyTorch |

**Generation strategy:** autoregressive sampling using the LSTM's carried hidden state (not a fixed sliding context window) — the seed sequence is processed once to initialize the hidden state, then each subsequent token is generated one at a time by feeding it back in along with the carried state. This lets the model draw on compressed memory of the entire generated sequence so far, rather than being capped to a fixed lookback window.

---

## Tokenization Scheme

Music is encoded as a stream of two event types:

- **`NOTE_<pitch>_<duration>`** — a note at MIDI pitch `<pitch>`, held for `<duration>` quarter-lengths (e.g. `NOTE_60_0.5` = middle C, eighth-note duration)
- **`TIME_SHIFT_<amount>`** — advances a shared time cursor forward by `<amount>` quarter-lengths, without sounding a note

Durations and time shifts are quantized to the nearest **0.25 quarter-length** (sixteenth-note resolution). Polyphony and chords emerge naturally from this scheme: multiple `NOTE_` tokens appearing with no `TIME_SHIFT_` between them are understood to sound simultaneously, at the current cursor position.

Two special tokens complete the vocabulary: `<PAD>` (index 0) and `<UNK>` (index 1, for any event type encountered in validation/test files that never appeared in training). Both are explicitly excluded from generation sampling — the model is never allowed to "generate" padding or an unknown token.

**Vocabulary is built exclusively from the training split**, to avoid any validation/test leakage into the token space used for training.

---

## Dataset

[MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) (MIDI-only), full dataset — no subsampling:

| Split | Performances |
|---|---|
| Train | 962 |
| Validation | 137 |
| Test | 177 |
| **Total** | **1,276** (~7M note events) |

The official, competition-curated split is used as-is (no manual re-stratification by composer), matching the split used in the original MAESTRO paper and the majority of published work built on this dataset — keeping results comparable and avoiding a source of pipeline bugs for uncertain benefit.

---

## Project Structure

```
CodeAlpha_MusicGenerationAI/
├── data/
│   ├── raw/                  # MAESTRO MIDI files + CSV (download separately)
│   └── processed/            # Tokenized shards + vocab.json + manifest.json
├── src/
│   ├── preprocessing.py      # MIDI -> event tokens -> sharded, resumable dataset
│   ├── dataset.py            # Lazy-loading PyTorch Dataset over token shards
│   ├── model.py              # MusicLSTM architecture
│   ├── train.py              # Training loop (checkpointing, resume, early stopping)
│   ├── generate.py           # Autoregressive generation + MIDI/audio conversion
│   ├── plot_loss.py          # Train/val loss curve plotting
│   └── app.py                # Streamlit web app
├── outputs/
│   ├── checkpoints/          # best_model.pt + epoch checkpoints
│   ├── generated_midi/       # Sample outputs
│   ├── generated_audio/      # Rendered .wav samples (if FluidSynth set up)
│   ├── loss_history.json
│   └── loss_curve.png
├── run-app.py                 # Double-click launcher for the web app (no terminal needed)
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why LSTM over a Transformer.** For a dataset and timeline of this scale, an LSTM is simpler to train reliably, needs far less data/compute to reach a coherent result, and is the architecture explicitly requested by the task brief. A Transformer is a reasonable future upgrade (see [Future Improvements](#future-improvements)) but was not necessary to produce musically coherent output here.

**Why event-based tokens over a fixed piano-roll grid.** A quantized-time-step grid (e.g. one row per 16th note, one column per pitch) wastes vocabulary space on "nothing happened" steps and struggles with wide dynamic range in note duration. The event-based scheme used here (`NOTE`/`TIME_SHIFT`) keeps the vocabulary compact (3,557 tokens) and lets the model spend its capacity on musically meaningful transitions.

**Why the full dataset, not a subset.** Training on all 1,276 files (rather than a smaller curated subset) gives the model more stylistic variety to generalize from, at the cost of longer preprocessing and training time — a tradeoff made deliberately given available hardware (RTX 4060, 32GB RAM) and time budget.

**Why hidden-state generation over a sliding context window.** An LSTM's hidden state is designed to carry compressed memory of an entire sequence. Re-feeding a fixed-size window at every generation step (an earlier iteration of this pipeline) works, but artificially caps how much history the model can draw on and is computationally wasteful — carrying the hidden state forward one token at a time is both more efficient and truer to how the architecture is meant to be used.

**Why early stopping.** Training was capped with a patience of 5 epochs on validation loss. The model's best weights (epoch 3, val loss 2.8056) were automatically preserved and used for generation, even though training continued for 5 more epochs before stopping — later epochs showed train loss continuing to fall while validation loss rose, a clear overfitting signal that early stopping correctly caught.

---

## Known Edge Cases & Limitations

**Micro-timing quantization behavior.** Because MAESTRO consists of real human performances (not quantized MIDI), notes are frequently played with natural timing variation smaller than the chosen 0.25 quarter-length quantization grid — rolled chords, rubato, fast ornaments, and similar expressive timing. When two notes fall within this sub-grid interval, the tokenizer collapses them into the same time step rather than emitting a negative time shift, effectively treating them as a chord. This affected roughly **30% of all note events dataset-wide**, most heavily in expressive/virtuosic pieces (e.g. Schubert sonatas, full recitals — up to **~50% of notes** in the most extreme cases). Round-trip decoding (`preprocessing.py`'s built-in sanity check) confirms this produces musically coherent, correctly structured output rather than data corruption — this is an inherent and expected consequence of symbolic quantization at this resolution, not a processing error.

**No velocity/dynamics.** The tokenization scheme encodes pitch, duration, and timing, but not note velocity. Every generated note plays at a fixed volume — the model captures melodic, harmonic, and rhythmic structure, but not expressive dynamics (crescendo, accents, soft passages).

**Fixed-length training windows, variable-length generation.** The model is trained on sliding windows of 100 tokens (stride 10), but generation is not limited to that length — the hidden-state-carrying approach allows arbitrarily long generated sequences, though quality beyond the training window length is not explicitly validated.

---

## Results

Trained on the full 1,276-file dataset (866,834 training windows, sequence length 100, stride 10). Early stopping (patience 5) halted training at epoch 8, correctly restoring the epoch 3 weights as the best model:

| Epoch | Train loss | Val loss | |
|---|---|---|---|
| 1 | 3.137 | 2.893 | improved |
| 2 | 2.547 | 2.822 | improved |
| **3** | **2.387** | **2.806** | **best — used for generation** |
| 4 | 2.300 | 2.809 | overfitting begins |
| 5–8 | ↓ (train) | ↑ (val) | early stopping triggered |

**Best validation loss: 2.8056** — a perplexity of roughly 16–17 out of the 3,557-token vocabulary. As a sanity anchor: a freshly-initialized, correctly-wired model should start at a loss equal to `ln(3557) ≈ 8.18` (uniform random guessing); training was confirmed to start almost exactly there before descending, confirming correct model wiring from step one. See `outputs/loss_curve.png` for the full curve.

**Generation samples** were compared across three temperatures (0.7, 1.0, 1.3 — controlling sampling randomness):

| Temperature | Character |
|---|---|
| 0.7 | Denser, more rhythmically confident, higher note density |
| 1.0 | Balanced variety and coherence |
| 1.3 | Sparser, more varied pitch choices, longer phrases |

An earlier generation approach (fixed sliding-window re-feeding) was found to occasionally produce degenerate, highly repetitive output at low temperature — a known LSTM failure mode from limited effective context. Switching to hidden-state-carrying generation resolved this, producing more diverse, structurally coherent samples across all tested temperatures.

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download [MAESTRO v3.0.0 (MIDI-only, 56MB)](https://magenta.tensorflow.org/datasets/maestro) and its metadata CSV, then place them at:
```
data/raw/maestro-v3.0.0.csv
data/raw/maestro-v3.0.0-midi/maestro-v3.0.0/   (extracted MIDI files)
```

### 3. Preprocess

```bash
python -m src.preprocessing --sample 10   # quick sanity check on 10 files first
python -m src.preprocessing               # full run, all 1,276 files (resumable)
```

### 4. Train

```bash
python -m src.train --stride 10 --batch-size 256 --sanity-check   # quick check
python -m src.train --stride 10 --batch-size 256                  # full training
python -m src.train --stride 10 --batch-size 256 --resume auto    # resume if interrupted
```

### 5. Generate (command line)

```bash
python -m src.generate --temperatures "0.7,1.0,1.3" --num-tokens 300
```

### 6. Loss curve

```bash
python -m src.plot_loss
```

### 7. Interactive web app (recommended)

**Easiest way:** double-click `run_app.py` in the project root — no terminal typing required.

**Or manually:**
```bash
python -m streamlit run src/app.py
```

### Optional: enable audio rendering

Both `generate.py` and `app.py` produce `.mid` files regardless of audio setup. For rendered `.wav` audio and in-browser playback, install [FluidSynth](https://github.com/FluidSynth/fluidsynth) (e.g. `choco install fluidsynth` on Windows) and a free soundfont such as [FluidR3_GM](https://member.keymusician.com/Member/FluidR3_GM/index.html), saved as `soundfont.sf2` in the project root.

---

## Troubleshooting

**FluidSynth hangs or drops into an interactive shell on Windows.** Some Windows builds of FluidSynth don't correctly parse the combined `-ni` flag shorthand, causing it to enter an interactive prompt instead of rendering and exiting — which then looks like a frozen process. This is fixed in `generate.py`'s `render_audio()` by splitting the flags (`-n -i`) and placing output flags before the positional soundfont/MIDI arguments, plus a hard 90-second subprocess timeout and `stdin=subprocess.DEVNULL` as defensive measures against the same class of issue recurring.

**`streamlit` or `fluidsynth` "not recognized" in PowerShell.** Usually a PATH issue from a user-level pip/Chocolatey install. Prefer invoking via `python -m streamlit run ...` (uses the currently active Python install directly) over relying on `streamlit` being on PATH.

---

## Future Improvements

- Add velocity/dynamics to the tokenization scheme for expressive playback
- Transformer-based architecture as an alternative to the LSTM
- Multi-instrument generation (MAESTRO is solo piano only)
- Beam search or nucleus sampling as alternatives to plain temperature sampling
- Package the FluidSynth + soundfont setup into the app itself for a fully self-contained install

---

## Credits

Built as part of the [CodeAlpha](https://www.codealpha.tech) Artificial Intelligence Internship — Task 3: Music Generation with AI.

Dataset: [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) (Google Magenta / International Piano-e-Competition), CC BY-NC-SA 4.0.
