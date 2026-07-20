from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from music21 import note, pitch as m21pitch, stream

try:
    from .model import MusicLSTM
except ImportError:
    from model import MusicLSTM


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_vocab(base_dir: Path):
    vocab_path = base_dir / "data" / "processed" / "vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "token_to_idx" in raw:
            token_to_idx = raw["token_to_idx"]
        elif "vocab" in raw and isinstance(raw["vocab"], dict):
            token_to_idx = raw["vocab"]
        else:
            token_to_idx = raw
    elif isinstance(raw, list):
        token_to_idx = {tok: i for i, tok in enumerate(raw)}
    else:
        raise ValueError(f"Unsupported vocab format: {type(raw)}")

    token_to_idx = {tok: int(idx) for tok, idx in token_to_idx.items()}
    idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}

    return token_to_idx, idx_to_token


def get_special_ids(token_to_idx: dict) -> list[int]:
    special_ids = []
    for name in ("<PAD>", "<UNK>"):
        if name in token_to_idx:
            special_ids.append(token_to_idx[name])

    if not special_ids:
        special_ids = [0, 1]

    return special_ids


def load_seed_sequence(base_dir: Path, seed_length: int, seed_file: str | None):
    manifest_path = base_dir / "data" / "processed" / "manifest.json"
    shards_dir = base_dir / "data" / "processed" / "shards"

    if not manifest_path.exists():
        return None, None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    processed_files = manifest.get("processed_files", {})

    candidates = [
        (fname, info)
        for fname, info in processed_files.items()
        if info.get("split") == "validation"
        and (seed_file is None or fname == seed_file)
    ]

    if not candidates:
        return None, None

    fname, info = random.choice(candidates)
    shard_path = shards_dir / info["shard"]

    data = safe_torch_load(shard_path)
    tokens = data["tokens"]

    actual_length = min(seed_length, len(tokens))
    seed_tokens = tokens[:actual_length].tolist()

    return seed_tokens, fname


def load_model(checkpoint_path: Path, vocab_size: int, device: torch.device, args):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    embedding_dim = config.get("embedding_dim", args.embedding_dim)
    hidden_size = config.get("hidden_size", args.hidden_size)
    num_layers = config.get("num_layers", args.num_layers)
    dropout = config.get("dropout", args.dropout)

    model = MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    best_val_loss = checkpoint.get("best_val_loss")

    print(f"Loaded checkpoint from epoch {epoch}")
    if best_val_loss is not None:
        print(f"Recorded best validation loss: {best_val_loss:.6f}")

    return model


# ---------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------

@torch.no_grad()
def generate_tokens(
    model,
    seed_indices: list[int],
    num_new_tokens: int,
    temperature: float,
    device: torch.device,
    special_token_ids: list[int],
) -> list[int]:
    temperature = max(float(temperature), 1e-4)

    seed_tensor = torch.tensor(seed_indices, dtype=torch.long, device=device).unsqueeze(0)

    logits, hidden = model(seed_tensor)
    next_logits = logits[:, -1, :].clone()

    generated = list(seed_indices)

    for _ in range(num_new_tokens):
        scaled = next_logits / temperature

        for special_id in special_token_ids:
            scaled[:, special_id] = float("-inf")

        probs = F.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(int(next_token.item()))

        step_logits, hidden = model(next_token, hidden)
        next_logits = step_logits[:, -1, :]

    return generated


# ---------------------------------------------------------
# Token -> MIDI conversion
# ---------------------------------------------------------

def tokens_to_music21_stream(tokens: list[str]) -> stream.Stream:
    s = stream.Stream()
    current_time = 0.0

    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            try:
                shift = float(tok.replace("TIME_SHIFT_", ""))
            except ValueError:
                continue
            current_time += shift

        elif tok.startswith("NOTE_"):
            parts = tok.split("_")
            if len(parts) != 3:
                continue
            try:
                midi_pitch = int(parts[1])
                duration = float(parts[2])
            except ValueError:
                continue

            p = m21pitch.Pitch()
            p.midi = midi_pitch

            n = note.Note()
            n.pitch = p
            n.duration.quarterLength = max(duration, 0.0625)

            s.insert(current_time, n)

        # Special tokens (<PAD>, <UNK>) are silently skipped.

    return s


# ---------------------------------------------------------
# Optional audio rendering
# ---------------------------------------------------------

def find_soundfont(base_dir: Path, cli_soundfont: str | None) -> Path | None:
    if cli_soundfont:
        path = Path(cli_soundfont)
        if path.exists():
            return path
        print(f"Warning: specified soundfont not found: {path}")
        return None

    env_path = os.environ.get("SOUNDFONT_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        base_dir / "assets" / "soundfont.sf2",
        base_dir / "soundfont.sf2",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def render_audio(midi_path: Path, wav_path: Path, soundfont_path: Path | None) -> bool:
    fluidsynth_bin = shutil.which("fluidsynth")

    if fluidsynth_bin is None:
        print("FluidSynth not found on PATH. Skipping audio rendering.")
        return False

    if soundfont_path is None:
        print("No soundfont (.sf2) found. Skipping audio rendering.")
        return False

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                fluidsynth_bin, "-n", "-i",
                "-F", str(wav_path), "-r", "44100",
                str(soundfont_path), str(midi_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
            stdin=subprocess.DEVNULL,
        )
        print(f"Audio rendered: {wav_path}")
        return True
    except subprocess.TimeoutExpired:
        print(f"FluidSynth timed out after 90s rendering {midi_path.name} — treating as failed.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"FluidSynth failed: {exc.stderr}")
        return False


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate music from a trained MusicLSTM.")

    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best_model.pt"))
    parser.add_argument("--temperatures", type=str, default="0.7,1.0,1.3")
    parser.add_argument("--num-tokens", type=int, default=300)
    parser.add_argument("--seed-length", type=int, default=50)
    parser.add_argument("--seed-file", type=str, default=None)
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    token_to_idx, idx_to_token = load_vocab(base_dir)
    vocab_size = len(token_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    special_ids = get_special_ids(token_to_idx)
    print(f"Masked special token ids during sampling: {special_ids}")

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = (base_dir / checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, vocab_size, device, args)

    seed_indices, seed_source = load_seed_sequence(base_dir, args.seed_length, args.seed_file)

    if seed_indices is None:
        print("No validation shard found. Using fixed fallback seed.")
        fallback_token = "NOTE_60_1.0"
        if fallback_token in token_to_idx:
            seed_indices = [token_to_idx[fallback_token]]
        else:
            non_special = [i for t, i in token_to_idx.items() if i not in special_ids]
            seed_indices = [non_special[0]] if non_special else [0]
        seed_source = "fixed_fallback"
    else:
        print(f"Seed sampled from validation file: {seed_source} ({len(seed_indices)} tokens)")

    temperatures = [float(t.strip()) for t in args.temperatures.split(",")]

    midi_dir = base_dir / "outputs" / "generated_midi"
    audio_dir = base_dir / "outputs" / "generated_audio"
    tokens_dir = base_dir / "outputs" / "generated_tokens"
    midi_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)

    soundfont_path = None if args.no_audio else find_soundfont(base_dir, args.soundfont)

    if not args.no_audio and soundfont_path is None:
        print(
            "No soundfont detected. Audio rendering will be skipped for all "
            "samples. Set --soundfont <path> or SOUNDFONT_PATH to enable it."
        )

    for temperature in temperatures:
        print(f"\n=== Generating sample at temperature={temperature} ===")

        generated_ids = generate_tokens(
            model=model,
            seed_indices=seed_indices,
            num_new_tokens=args.num_tokens,
            temperature=temperature,
            device=device,
            special_token_ids=special_ids,
        )

        generated_tokens_str = [idx_to_token.get(i, "<UNK>") for i in generated_ids]

        temp_label = str(temperature).replace(".", "p")
        base_name = f"sample_temp{temp_label}"

        tokens_path = tokens_dir / f"{base_name}.json"
        with open(tokens_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "temperature": temperature,
                    "seed_source": seed_source,
                    "seed_length": len(seed_indices),
                    "total_length": len(generated_ids),
                    "token_ids": generated_ids,
                    "tokens": generated_tokens_str,
                },
                f,
                indent=2,
            )
        print(f"Raw tokens saved: {tokens_path}")

        midi_obj = tokens_to_music21_stream(generated_tokens_str)
        midi_path = midi_dir / f"{base_name}.mid"
        midi_obj.write("midi", fp=str(midi_path))
        print(f"MIDI saved: {midi_path}")

        if not args.no_audio:
            wav_path = audio_dir / f"{base_name}.wav"
            render_audio(midi_path, wav_path, soundfont_path)

    print("\nGeneration complete for all requested temperatures.")


if __name__ == "__main__":
    main()