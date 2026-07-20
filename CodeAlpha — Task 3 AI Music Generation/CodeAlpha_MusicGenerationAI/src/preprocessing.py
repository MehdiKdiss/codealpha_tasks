"""
Phase 2 - Preprocessing Pipeline for MAESTRO v3.0.0 MIDI Dataset

Event-based tokenization:
  TIME_SHIFT_X  - advance time by X quarter-lengths (quantized to 0.25)
  NOTE_P_D      - note at MIDI pitch P, duration D quarter-lengths (quantized to 0.25)

Multiple NOTE tokens at the same time position represent a chord.
Durations and time shifts quantized to 0.25 quarter-lengths (sixteenth-note resolution).

Output per shard:
  {"tokens": tensor(int32), "split": str, "source_file": str, "num_tokens": int}

Windows for training (x = tokens[i:i+seq_len], y = tokens[i+seq_len]) are created
lazily by dataset.py, NOT materialized here.

Resume-safe via manifest.json with atomic shard writes.
Windows-safe multiprocessing with ProcessPoolExecutor + freeze_support.
"""

import os
import sys
import csv
import json
import time
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import torch
from tqdm import tqdm
import music21

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DURATION_RESOLUTION = 0.25  # Quantize durations and time shifts to nearest 0.25 QL
SEQ_LENGTH = 100            # Default sequence length for training windows


def get_default_workers():
    """Safe worker count: min(8, cpu_count - 2), at least 1."""
    cpu = os.cpu_count() or 4
    return min(8, max(1, cpu - 2))


# ─────────────────────────────────────────────────────────────────────────────
# Path Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_paths(sample_mode=False):
    """Return all relevant paths. Uses separate dir for sample mode."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")

    if sample_mode:
        processed_dir = os.path.join(base_dir, "data", "processed_sample")
    else:
        processed_dir = os.path.join(base_dir, "data", "processed")

    return {
        "base_dir": base_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "csv_path": os.path.join(raw_dir, "maestro-v3.0.0.csv"),
        "midi_base": os.path.join(raw_dir, "maestro-v3.0.0-midi", "maestro-v3.0.0"),
        "vocab_path": os.path.join(processed_dir, "vocab.json"),
        "manifest_path": os.path.join(processed_dir, "manifest.json"),
        "shards_dir": os.path.join(processed_dir, "shards"),
        "warnings_log": os.path.join(base_dir, "outputs", "preprocessing_warnings.log"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_duration(dur, resolution=DURATION_RESOLUTION):
    """
    Round a duration to the nearest multiple of resolution.
    Any positive duration is clamped to at least one tick (resolution),
    so very short notes become sixteenth notes rather than being dropped.
    """
    if dur <= 0:
        return 0.0
    ticks = max(1, round(dur / resolution))
    return round(ticks * resolution, 4)


# ─────────────────────────────────────────────────────────────────────────────
# MIDI Parsing → Event Tokens
# ─────────────────────────────────────────────────────────────────────────────

def parse_midi_to_tokens(midi_path):
    """
    Parse a MIDI file into a list of event tokens.

    Token types:
      TIME_SHIFT_X   - time advance by X quarter-lengths
      NOTE_P_D       - note-on at MIDI pitch P with duration D

    Multiple NOTE tokens at the same time position form a chord.
    Events are sorted by (offset, pitch) for deterministic ordering.

    Returns:
        tokens: list of token strings
        warnings: list of warning/error messages
    """
    tokens = []
    parse_warnings = []

    try:
        score = music21.converter.parse(midi_path)
        flat = score.flatten()

        # Collect all note events as (offset, pitch, duration)
        events = []
        for element in flat.notesAndRests:
            if isinstance(element, music21.note.Rest):
                continue

            offset = float(element.offset)
            dur = quantize_duration(element.quarterLength)
            if dur <= 0:
                continue

            if isinstance(element, music21.note.Note):
                events.append((offset, element.pitch.midi, dur))
            elif isinstance(element, music21.chord.Chord):
                # Each pitch in a chord becomes a separate NOTE event
                for p in element.pitches:
                    events.append((offset, p.midi, dur))

        if not events:
            parse_warnings.append(f"No note events found in {midi_path}")
            return tokens, parse_warnings

        # Sort by (offset, pitch) for deterministic ordering
        events.sort(key=lambda e: (e[0], e[1]))

        # Convert to token sequence with TIME_SHIFT events
        current_time = 0.0
        for offset, pitch, dur in events:
            # Quantize the time shift to avoid floating-point drift
            time_shift = quantize_duration(offset - current_time)
            if time_shift > 0:
                tokens.append(f"TIME_SHIFT_{time_shift}")
                current_time += time_shift
            elif offset - current_time < 0:
                # Event before current time - log but don't emit negative shift
                parse_warnings.append(
                    f"Negative time shift at offset {offset:.4f} "
                    f"(current_time={current_time:.4f}) in {os.path.basename(midi_path)}"
                )

            tokens.append(f"NOTE_{pitch}_{dur}")

    except Exception as e:
        parse_warnings.append(f"PARSE ERROR in {midi_path}: {str(e)}")

    return tokens, parse_warnings


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(all_token_sets):
    """
    Build vocabulary from a list of sets of unique tokens.
    Reserves: <PAD>=0, <UNK>=1. All other tokens sorted alphabetically from index 2.
    """
    all_tokens = set()
    for token_set in all_token_sets:
        all_tokens.update(token_set)

    sorted_tokens = sorted(all_tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, token in enumerate(sorted_tokens, start=2):
        vocab[token] = i

    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Window Counting
# ─────────────────────────────────────────────────────────────────────────────

def count_possible_windows(num_tokens, seq_length):
    """Number of (input[seq_length], target[1]) windows from a token stream."""
    return max(0, num_tokens - seq_length)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest I/O (atomic)
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path):
    """Load the processing manifest. Returns None if missing or corrupt."""
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_manifest(manifest, manifest_path):
    """Save manifest atomically: write to .tmp, then rename."""
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    # Atomic rename (Windows: must remove existing first)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    os.rename(tmp_path, manifest_path)


def new_manifest(config):
    """Create a fresh manifest with the given config."""
    return {
        "config": config,
        "vocab_built": False,
        "processed_files": {},
        "failed_files": {},
        "total_tokens": 0,
        "total_possible_windows": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Worker Functions (run in subprocesses)
# ─────────────────────────────────────────────────────────────────────────────

def worker_extract_tokens(args):
    """
    Pass A worker: parse one MIDI file, return its unique token set.
    Returns: (filename, set_of_unique_tokens, total_token_count, warnings)
    """
    midi_filename, midi_base = args
    full_path = os.path.join(midi_base, midi_filename.replace("/", os.sep))
    tokens, warnings = parse_midi_to_tokens(full_path)
    unique = set(tokens)
    return midi_filename, unique, len(tokens), warnings


def worker_process_file(args):
    """
    Pass B worker: parse MIDI, encode tokens to integers, save shard atomically.
    Returns: (filename, num_tokens, num_oov, shard_path, warnings, success, error_msg)
    """
    midi_filename, midi_base, vocab, seq_length, shard_path, split = args
    full_path = os.path.join(midi_base, midi_filename.replace("/", os.sep))

    try:
        tokens, parse_warnings = parse_midi_to_tokens(full_path)

        if len(tokens) == 0:
            return (midi_filename, 0, 0, shard_path, parse_warnings,
                    False, "No tokens extracted")

        # Convert to integer indices; unknowns → <UNK>=1
        unk_id = vocab.get("<UNK>", 1)
        token_indices = []
        oov_count = 0
        for t in tokens:
            idx = vocab.get(t)
            if idx is not None:
                token_indices.append(idx)
            else:
                token_indices.append(unk_id)
                oov_count += 1

        # Build shard data
        shard_data = {
            "tokens": torch.tensor(token_indices, dtype=torch.int32),
            "split": split,
            "source_file": midi_filename,
            "num_tokens": len(token_indices),
        }

        # Atomic write: save to .tmp, then rename
        tmp_path = shard_path + ".tmp"
        torch.save(shard_data, tmp_path)
        if os.path.exists(shard_path):
            os.remove(shard_path)
        os.rename(tmp_path, shard_path)

        return (midi_filename, len(token_indices), oov_count, shard_path,
                parse_warnings, True, None)

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (midi_filename, 0, 0, shard_path, [], False, error_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Round-Trip Decode/Re-Encode (sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def decode_tokens_to_music21(token_strings):
    """
    Decode event token strings back to (offset, music21.note.Note) pairs.
    TIME_SHIFT tokens advance the current time; NOTE tokens create notes.
    """
    elements = []
    current_time = 0.0

    for token in token_strings:
        if token.startswith("TIME_SHIFT_"):
            # TIME_SHIFT_X → advance by X quarter-lengths
            shift = float(token.split("_", 2)[2])
            current_time += shift
        elif token.startswith("NOTE_"):
            # NOTE_P_D → note at current_time with pitch P, duration D
            parts = token.split("_")
            pitch = int(parts[1])
            dur = float(parts[2])
            note = music21.note.Note(pitch)
            note.quarterLength = dur
            elements.append((current_time, note))

    return elements


def reencode_from_music21(elements):
    """
    Re-encode (offset, note) pairs back to event token strings.
    Should produce identical output to the original encoding.
    """
    tokens = []
    # Sort by (offset, pitch) - same deterministic order as the encoder
    elements_sorted = sorted(elements, key=lambda e: (e[0], e[1].pitch.midi))

    current_time = 0.0
    for offset, note in elements_sorted:
        time_shift = quantize_duration(offset - current_time)
        if time_shift > 0:
            tokens.append(f"TIME_SHIFT_{time_shift}")
            current_time += time_shift
        tokens.append(
            f"NOTE_{note.pitch.midi}_{quantize_duration(note.quarterLength)}"
        )

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Preprocess MAESTRO v3.0.0 MIDI dataset into token shards"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only the first N files for a quick sanity check",
    )
    parser.add_argument(
        "--workers", type=int, default=get_default_workers(),
        help=f"Number of worker processes (default: {get_default_workers()})",
    )
    parser.add_argument(
        "--seq-length", type=int, default=SEQ_LENGTH,
        help=f"Sequence length for training windows (default: {SEQ_LENGTH})",
    )
    args = parser.parse_args()

    seq_length = args.seq_length
    num_workers = args.workers
    sample_n = args.sample
    sample_mode = sample_n is not None

    paths = get_paths(sample_mode=sample_mode)

    print("====================================================")
    print("|    Phase 2 - MAESTRO Preprocessing Pipeline     |")
    print("====================================================")
    print(f"  Workers:              {num_workers}")
    print(f"  Sequence Length:      {seq_length}")
    print(f"  Duration Resolution:  {DURATION_RESOLUTION} quarter-lengths")
    print(f"  Output Directory:     {paths['processed_dir']}")
    if sample_mode:
        print(f"  MODE:                 SAMPLE (first {sample_n} files)")
    print()

    # Ensure output directories exist
    os.makedirs(paths["processed_dir"], exist_ok=True)
    os.makedirs(paths["shards_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(paths["warnings_log"]), exist_ok=True)

    # ─── Load CSV ────────────────────────────────────────────────────────
    records = []
    with open(paths["csv_path"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if sample_mode:
        records = records[:sample_n]

    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "validation"]
    test_records = [r for r in records if r["split"] == "test"]

    print(f"  Files to process: {len(records)}  "
          f"(Train: {len(train_records)}, Val: {len(val_records)}, "
          f"Test: {len(test_records)})")

    # ─── Config for manifest ─────────────────────────────────────────────
    run_config = {
        "seq_length": seq_length,
        "duration_resolution": DURATION_RESOLUTION,
        "total_expected": len(records),
    }

    # ─── Load or Initialize Manifest ─────────────────────────────────────
    manifest = load_manifest(paths["manifest_path"])

    if manifest is not None:
        if manifest.get("config") != run_config:
            print("  ! Config mismatch - clearing old cache and starting fresh.")
            manifest = new_manifest(run_config)
            # Clean old shards
            if os.path.exists(paths["shards_dir"]):
                for f in os.listdir(paths["shards_dir"]):
                    fp = os.path.join(paths["shards_dir"], f)
                    if os.path.isfile(fp):
                        os.remove(fp)
            # Remove old vocab
            if os.path.exists(paths["vocab_path"]):
                os.remove(paths["vocab_path"])
        else:
            print("  Loaded existing manifest for resume.")
    else:
        manifest = new_manifest(run_config)

    # Check if already fully complete
    all_filenames = set(r["midi_filename"] for r in records)
    already_processed = set(manifest.get("processed_files", {}).keys())
    already_failed = set(manifest.get("failed_files", {}).keys())

    if (all_filenames.issubset(already_processed | already_failed)
            and manifest.get("vocab_built")):
        print("\n[OK] All files already processed (verified via manifest). Skipping.")
        print(f"  Total tokens:           {manifest.get('total_tokens', 'N/A'):,}")
        print(f"  Total possible windows: {manifest.get('total_possible_windows', 'N/A'):,}")
        print(f"  Failed:                 {len(already_failed)}")
        # Still load vocab for sample report
        with open(paths["vocab_path"], "r") as f:
            vocab = json.load(f)
        if sample_mode:
            _print_sanity_check(manifest, vocab, paths, seq_length, sample_n, 0.0)
        return

    total_start = time.time()

    # =====================================================================
    # PASS A: Vocabulary Building (training files only)
    # =====================================================================

    if not manifest.get("vocab_built") or not os.path.exists(paths["vocab_path"]):
        print("\n=== Pass A: Building Vocabulary (training files only) ===")

        train_filenames = [r["midi_filename"] for r in train_records]
        if not train_filenames:
            print("  ! No training files in this subset - building vocab from all files.")
            train_filenames = [r["midi_filename"] for r in records]

        worker_args = [(f, paths["midi_base"]) for f in train_filenames]

        all_token_sets = []
        total_train_tokens = 0
        all_warnings = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker_extract_tokens, arg): arg[0]
                for arg in worker_args
            }
            with tqdm(total=len(futures), desc="Building vocab", unit="file") as pbar:
                for future in as_completed(futures):
                    filename, unique_tokens, num_tokens, warns = future.result()
                    all_token_sets.append(unique_tokens)
                    total_train_tokens += num_tokens
                    all_warnings.extend(warns)
                    pbar.update(1)

        if all_warnings:
            with open(paths["warnings_log"], "a", encoding="utf-8") as f:
                f.write("=== Pass A Warnings ===\n")
                for w in all_warnings:
                    f.write(w + "\n")
            print(f"  Logged {len(all_warnings)} warnings to {paths['warnings_log']}")

        vocab = build_vocab(all_token_sets)
        with open(paths["vocab_path"], "w") as f:
            json.dump(vocab, f, indent=2)

        manifest["vocab_built"] = True
        save_manifest(manifest, paths["manifest_path"])

        print(f"  Vocabulary size:         {len(vocab)} tokens (incl. <PAD>, <UNK>)")
        print(f"  Total training tokens:   {total_train_tokens:,}")
        print(f"  Saved to:                {paths['vocab_path']}")
    else:
        print("\n[OK] Vocabulary already built. Loading from disk.")
        with open(paths["vocab_path"], "r") as f:
            vocab = json.load(f)
        print(f"  Vocabulary size: {len(vocab)} tokens")

    # =====================================================================
    # PASS B: Token Encoding & Shard Writing (all splits)
    # =====================================================================

    print("\n=== Pass B: Encoding & Writing Shards ===")

    # On resume: verify that each "processed" file has a valid shard on disk
    verified_processed = set()
    for fname, info in list(manifest.get("processed_files", {}).items()):
        shard_file = os.path.join(paths["shards_dir"], info["shard"])
        if os.path.exists(shard_file):
            try:
                torch.load(shard_file, weights_only=False)
                verified_processed.add(fname)
            except Exception:
                # Shard is corrupt - will reprocess
                pass

    # Prune manifest to only verified entries
    manifest["processed_files"] = {
        k: v for k, v in manifest["processed_files"].items()
        if k in verified_processed
    }
    # Recalculate totals from verified entries
    manifest["total_tokens"] = sum(
        v["num_tokens"] for v in manifest["processed_files"].values()
    )
    manifest["total_possible_windows"] = sum(
        v.get("num_possible_windows", 0)
        for v in manifest["processed_files"].values()
    )

    # Determine which files still need processing
    files_to_process = []
    for i, r in enumerate(records):
        fname = r["midi_filename"]
        if fname not in verified_processed and fname not in already_failed:
            shard_name = f"shard_{i:04d}.pt"
            shard_path = os.path.join(paths["shards_dir"], shard_name)
            files_to_process.append(
                (fname, r["split"], shard_path, shard_name)
            )

    if len(files_to_process) == 0:
        print("  All files already processed. Nothing to do.")
    else:
        skipped = len(records) - len(files_to_process)
        if skipped > 0:
            print(f"  Resuming: {skipped} already done, "
                  f"{len(files_to_process)} remaining.")
        else:
            print(f"  Processing {len(files_to_process)} files...")

        worker_args = [
            (f[0], paths["midi_base"], vocab, seq_length, f[2], f[1])
            for f in files_to_process
        ]

        all_warnings = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker_process_file, arg): arg[0]
                for arg in worker_args
            }
            with tqdm(total=len(futures), desc="Processing files",
                      unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    (filename, num_tokens, num_oov, shard_path,
                     warns, success, error_msg) = result

                    all_warnings.extend(warns)

                    # Find split and shard name for this file
                    rec = next(
                        r for r in records
                        if r["midi_filename"] == filename
                    )
                    split = rec["split"]
                    shard_name = os.path.basename(shard_path)

                    if success:
                        num_windows = count_possible_windows(
                            num_tokens, seq_length
                        )
                        manifest["processed_files"][filename] = {
                            "shard": shard_name,
                            "num_tokens": num_tokens,
                            "num_oov": num_oov,
                            "num_possible_windows": num_windows,
                            "split": split,
                        }
                        manifest["total_tokens"] += num_tokens
                        manifest["total_possible_windows"] += num_windows
                    else:
                        manifest["failed_files"][filename] = (
                            error_msg or "Unknown error"
                        )
                        print(f"\n  x FAILED: {filename} - {error_msg}")

                    save_manifest(manifest, paths["manifest_path"])
                    pbar.update(1)

        if all_warnings:
            with open(paths["warnings_log"], "a", encoding="utf-8") as f:
                f.write("=== Pass B Warnings ===\n")
                for w in all_warnings:
                    f.write(w + "\n")
            print(f"  Logged {len(all_warnings)} warnings to {paths['warnings_log']}")

    total_elapsed = time.time() - total_start

    # =====================================================================
    # Final Report
    # =====================================================================

    processed_count = len(manifest.get("processed_files", {}))
    failed_count = len(manifest.get("failed_files", {}))
    total_files = len(records)

    print("\n==================================================")
    print("           PREPROCESSING COMPLETE")
    print("==================================================")
    print(f"  Files processed:         {processed_count} / {total_files}")
    print(f"  Files failed:            {failed_count}")
    print(f"  Total tokens:            {manifest.get('total_tokens', 0):,}")
    print(f"  Total possible windows:  {manifest.get('total_possible_windows', 0):,}")
    print(f"  Vocabulary size:         {len(vocab)}")
    print(f"  Elapsed time:            {total_elapsed:.1f}s "
          f"({total_elapsed / 60:.1f} min)")

    if processed_count > 0:
        avg_time = total_elapsed / processed_count
        print(f"  Average time/file:       {avg_time:.2f}s")

    if failed_count > 0:
        print(f"\n  Failed files:")
        for fname, err in manifest.get("failed_files", {}).items():
            # Show first line of error only
            err_short = err.split("\n")[0] if err else "Unknown"
            print(f"    x {fname}: {err_short}")

    # =====================================================================
    # Sanity Check (Sample Mode Only)
    # =====================================================================

    if sample_mode:
        _print_sanity_check(
            manifest, vocab, paths, seq_length, sample_n, total_elapsed
        )


def _print_sanity_check(manifest, vocab, paths, seq_length, sample_n,
                         total_elapsed):
    """Print detailed sanity-check report for sample mode."""
    processed_count = len(manifest.get("processed_files", {}))
    failed_count = len(manifest.get("failed_files", {}))

    print("\n==================================================")
    print("         SANITY CHECK (Sample Mode)")
    print("==================================================")

    # Vocabulary size
    print(f"\n  Vocabulary size:       {len(vocab)} tokens")

    # OOV counts per split - recompute from saved shards for accuracy
    unk_id = vocab.get("<UNK>", 1)
    split_oov = {"train": 0, "validation": 0, "test": 0}
    split_tokens = {"train": 0, "validation": 0, "test": 0}

    for fname, info in manifest.get("processed_files", {}).items():
        shard_file = os.path.join(paths["shards_dir"], info["shard"])
        try:
            shard_data = torch.load(shard_file, weights_only=False)
            tokens_tensor = shard_data["tokens"]
            split = info["split"]
            oov = int((tokens_tensor == unk_id).sum().item())
            split_oov[split] += oov
            split_tokens[split] += len(tokens_tensor)
        except Exception:
            pass

    print(f"\n  --- OOV Token Counts ---")
    for split in ["train", "validation", "test"]:
        total = split_tokens[split]
        oov = split_oov[split]
        pct = (oov / total * 100) if total > 0 else 0.0
        print(f"    {split:>12}: {oov:>6} / {total:>8} tokens "
              f"({pct:.2f}% OOV)")

    # Counts
    print(f"\n  --- Counts ---")
    print(f"    Total tokens:           "
          f"{manifest.get('total_tokens', 0):,}")
    print(f"    Possible sequences:     "
          f"{manifest.get('total_possible_windows', 0):,}")
    print(f"    Processed files:        {processed_count}")
    print(f"    Failed files:           {failed_count}")

    # Round-trip decode example
    print(f"\n  --- Round-Trip Decode ---")
    first_shard = None
    for f in sorted(os.listdir(paths["shards_dir"])):
        if f.endswith(".pt"):
            first_shard = os.path.join(paths["shards_dir"], f)
            break

    if first_shard:
        data = torch.load(first_shard, weights_only=False)
        token_indices = data["tokens"].tolist()

        # Reverse vocab lookup
        idx_to_token = {v: k for k, v in vocab.items()}

        # Take first 20 tokens for the example
        sample_size = min(20, len(token_indices))
        sample_indices = token_indices[:sample_size]
        decoded_tokens = [
            idx_to_token.get(idx, "<UNK>") for idx in sample_indices
        ]

        print(f"    Source file: {data.get('source_file', 'unknown')}")
        print(f"    First {sample_size} token indices: {sample_indices}")
        print(f"    Decoded tokens:")
        for t in decoded_tokens:
            print(f"      {t}")

        # Decode to music21
        elements = decode_tokens_to_music21(decoded_tokens)
        print(f"\n    Reconstructed music21 objects ({len(elements)} notes):")
        for offset, note in elements[:10]:
            print(f"      offset={offset:<8.2f}  "
                  f"pitch={note.pitch.midi:<4}  "
                  f"dur={note.quarterLength}")

        # Re-encode back to tokens
        re_encoded = reencode_from_music21(elements)

        # Filter original to only event tokens (skip <PAD>, <UNK>)
        original_events = [
            t for t in decoded_tokens
            if t not in ("<PAD>", "<UNK>")
        ]

        # Remove trailing TIME_SHIFT from original_events since re-encoding won't recreate it
        # if there are no subsequent notes.
        while original_events and original_events[-1].startswith("TIME_SHIFT_"):
            original_events.pop()
            
        match = re_encoded == original_events
        print(f"\n    Original tokens:    {original_events[:8]}...")
        print(f"    Re-encoded tokens:  {re_encoded[:8]}...")
        print(f"    Round-trip match:   {'[OK] PASS' if match else 'x FAIL'}")

        if not match:
            # Show first mismatch for debugging
            for i, (a, b) in enumerate(
                zip(original_events, re_encoded)
            ):
                if a != b:
                    print(f"    First mismatch at index {i}: "
                          f"'{a}' vs '{b}'")
                    break

    # Time estimate
    if processed_count > 0 and total_elapsed > 0:
        avg_time = total_elapsed / processed_count
        full_estimate = avg_time * 1276
        print(f"\n  --- Time Estimate ---")
        print(f"    Sample ({sample_n} files):      {total_elapsed:.1f}s")
        print(f"    Avg time/file:              {avg_time:.2f}s")
        print(f"    Est. full run (1,276 files): {full_estimate:.0f}s "
              f"({full_estimate / 60:.1f} min / "
              f"{full_estimate / 3600:.1f} hr)")

    print("\n==================================================")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point (Windows multiprocessing guard)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    freeze_support()
    main()
