# Phase 2 — Preprocessing Implementation Plan (Revised)

## Goal

Implement `src/preprocessing.py`: a multiprocessed, resumable pipeline that tokenizes all 1,276 MAESTRO MIDI files into integer token streams for LSTM training.

## Key Design Decisions

### Event-Based Tokenization with Explicit Timing

Instead of `"pitch_duration"`, we use an **event representation** that preserves musical timing:

```
TIME_SHIFT_0.5     ← advance time by 0.5 quarter-lengths
NOTE_60_1.0        ← MIDI pitch 60, duration 1.0 quarter-lengths
NOTE_64_1.0        ← same time position = chord with the note above
TIME_SHIFT_0.25    ← advance time
NOTE_67_0.5        ← single note at new position
```

- Multiple `NOTE` tokens at the same time position represent a **chord** (no TIME_SHIFT between them).
- Both durations and time shifts are quantized to **0.25 quarter-lengths** (sixteenth-note resolution).
- Rests/timing gaps are captured naturally by `TIME_SHIFT` tokens.

### Token-Stream Shards (No Materialized Windows)

Each shard stores only the **full encoded token stream** for one MIDI file:

```json
{
  "tokens": "torch.tensor([...], dtype=torch.int32)",
  "split": "train",
  "source_file": "2018/file.midi",
  "num_tokens": 5432
}
```

`dataset.py` will later create sliding windows **lazily**:
```python
x = tokens[start:start+seq_length]   # input
y = tokens[start+seq_length]          # target
```

The manifest tracks possible window count per file, but windows are never materialized during preprocessing.

### Vocabulary with Reserved Tokens

| Index | Token | Purpose |
|-------|-------|---------|
| 0 | `<PAD>` | Padding for batching |
| 1 | `<UNK>` | Unknown tokens (val/test tokens not seen in training) |
| 2+ | Sorted event tokens | `NOTE_*` and `TIME_SHIFT_*` from training data |

OOV counts are reported per split (train should be 0, val/test may have some).

### Atomic Shard Writes & Resume Safety

1. Worker writes to `shard_XXXX.pt.tmp`
2. `torch.save` completes → atomic rename to `shard_XXXX.pt`
3. Main process verifies final shard exists → updates manifest
4. On resume: skip a file only if **manifest entry exists AND shard file exists AND `torch.load` succeeds**
5. Failed files tracked separately in `failed_files` — never silently marked complete

### Two-Pass Architecture

| Pass | Scope | Workers Return | Main Process Does |
|------|-------|---------------|-------------------|
| **A – Vocab** | Training files only (962) | Set of unique token strings | Union → build `vocab.json` |
| **B – Encode** | All files (1,276) | Write shard to disk atomically | Update manifest per-file |

### Worker Default & Windows Safety

```python
workers = min(8, max(1, os.cpu_count() - 2))
```

All multiprocessing guarded by `if __name__ == "__main__":` + `freeze_support()`.

### Sample Mode Isolation

`--sample N` writes to `data/processed_sample/` (separate from `data/processed/`).
Report includes: vocab size, per-split OOV counts, token/window counts, processed/failed counts, round-trip decode check, and extrapolated full-run time.

## Proposed Changes

#### [MODIFY] [preprocessing.py](file:///c:/Users/GIGABYTE/Desktop/Code%20Alpha/CodeAlpha%20—%20Task%203%20AI%20Music%20Generation/CodeAlpha_MusicGenerationAI/src/preprocessing.py)

~400 lines implementing the full pipeline.

## Verification Plan

1. `python src/preprocessing.py --sample 10` — sanity check
2. Review vocab size, OOV counts, round-trip decode, estimated runtime
3. User approves → full 1,276-file run
