from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import MaestroDataset
    from .model import MusicLSTM
except ImportError:
    from dataset import MaestroDataset
    from model import MusicLSTM


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def safe_torch_load(path, map_location="cpu"):
    """Load a PyTorch file with compatibility across PyTorch versions."""
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_vocab_size(base_dir: Path) -> int:
    vocab_path = base_dir / "data" / "processed" / "vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    if isinstance(vocab, dict):
        if "token_to_idx" in vocab:
            return len(vocab["token_to_idx"])

        if "vocab" in vocab and isinstance(vocab["vocab"], dict):
            return len(vocab["vocab"])

        return len(vocab)

    if isinstance(vocab, list):
        return len(vocab)

    raise ValueError(f"Unsupported vocabulary format: {type(vocab)}")


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def autocast_context(enabled: bool):
    if enabled:
        return torch.cuda.amp.autocast()

    return nullcontext()


def create_grad_scaler(enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=enabled)


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }

    if workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True

    return DataLoader(**kwargs)


# ---------------------------------------------------------
# Training and validation
# ---------------------------------------------------------

def calculate_loss(logits, targets, criterion):
    """
    Model output:
        logits:  (batch, sequence_length, vocab_size)

    Targets:
        targets: (batch, sequence_length)

    CrossEntropyLoss requires:
        logits:  (batch * sequence_length, vocab_size)
        targets: (batch * sequence_length)
    """
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    return criterion(logits, targets)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    use_amp,
    max_batches=None,
    gradient_clip=1.0,
):
    model.train()

    total_loss = 0.0
    batch_count = 0
    first_loss = None
    last_loss = None

    optimizer.zero_grad(set_to_none=True)

    iterable = loader
    total = len(loader)

    if max_batches is not None:
        iterable = islice(loader, max_batches)
        total = min(max_batches, len(loader))

    progress = tqdm(
        iterable,
        total=total,
        desc="Training",
        leave=False
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if x.ndim != 2 or y.ndim != 2:
            raise RuntimeError(
                f"Expected x and y to have shape (batch, seq_len), "
                f"got {x.shape} and {y.shape}"
            )

        with autocast_context(use_amp):
            logits, _ = model(x)
            loss = calculate_loss(logits, y, criterion)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Invalid training loss: {loss.item()}")

        scaler.scale(loss).backward()

        if gradient_clip is not None and gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach().item())

        if first_loss is None:
            first_loss = loss_value

        last_loss = loss_value
        total_loss += loss_value
        batch_count += 1

        progress.set_postfix(loss=f"{loss_value:.4f}")

    if batch_count == 0:
        raise RuntimeError("No training batches were processed.")

    return {
        "loss": total_loss / batch_count,
        "first_batch_loss": first_loss,
        "last_batch_loss": last_loss,
        "batches": batch_count,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    use_amp,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    batch_count = 0

    iterable = loader
    total = len(loader)

    if max_batches is not None:
        iterable = islice(loader, max_batches)
        total = min(max_batches, len(loader))

    progress = tqdm(
        iterable,
        total=total,
        desc="Validation",
        leave=False
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_context(use_amp):
            logits, _ = model(x)
            loss = calculate_loss(logits, y, criterion)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Invalid validation loss: {loss.item()}")

        loss_value = float(loss.detach().item())
        total_loss += loss_value
        batch_count += 1

        progress.set_postfix(loss=f"{loss_value:.4f}")

    if batch_count == 0:
        raise RuntimeError("No validation batches were processed.")

    return total_loss / batch_count


# ---------------------------------------------------------
# Checkpoint functions
# ---------------------------------------------------------

def atomic_torch_save(state, path: Path):
    """
    Save to a temporary file and then replace the final file.
    This prevents incomplete checkpoint files after interruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(state, temporary_path)
    os.replace(str(temporary_path), str(path))


def build_checkpoint_state(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    history,
    best_val_loss,
    epochs_without_improvement,
    args,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "config": vars(args),
    }


def get_latest_checkpoint(checkpoint_dir: Path):
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))

    if not checkpoints:
        return None

    def epoch_number(path):
        match = re.search(r"epoch_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    checkpoints.sort(key=epoch_number)
    return checkpoints[-1]


def remove_old_checkpoints(checkpoint_dir: Path, keep_last: int):
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))

    def epoch_number(path):
        match = re.search(r"epoch_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    checkpoints.sort(key=epoch_number)

    for old_checkpoint in checkpoints[:-keep_last]:
        try:
            old_checkpoint.unlink()
        except OSError as exc:
            print(f"Warning: could not delete {old_checkpoint}: {exc}")


def save_loss_history(output_dir: Path, history):
    path = output_dir / "loss_history.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_resume_checkpoint(
    resume_argument,
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    if resume_argument == "auto":
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    else:
        checkpoint_path = Path(resume_argument)

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint was found for --resume."
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    start_epoch = int(checkpoint["epoch"]) + 1
    history = checkpoint.get("history", [])
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0
    )

    print(f"Resuming from epoch {start_epoch}")

    return (
        start_epoch,
        history,
        best_val_loss,
        epochs_without_improvement,
    )


# ---------------------------------------------------------
# Sanity check
# ---------------------------------------------------------

def run_sanity_check(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    use_amp,
    args,
    vocab_size,
):
    print("\n=== Phase 4 Sanity Check ===")
    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Stride: {args.stride}")
    print(f"Training batches available: {len(train_loader)}")
    print(f"Validation batches available: {len(val_loader)}")
    print(f"Sanity batches: {args.sanity_batches}")
    print(f"Mixed precision enabled: {use_amp}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()

    train_stats = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        max_batches=args.sanity_batches,
        gradient_clip=args.gradient_clip,
    )

    val_loss = validate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        max_batches=args.sanity_batches,
    )

    scheduler.step(val_loss)

    elapsed = time.perf_counter() - start_time

    history = [
        {
            "epoch": 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    ]

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "sanity_checkpoint.pt"

    checkpoint = build_checkpoint_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        history=history,
        best_val_loss=val_loss,
        epochs_without_improvement=0,
        args=args,
    )

    atomic_torch_save(checkpoint, checkpoint_path)

    # Verify that the checkpoint can be loaded into a new model.
    reloaded_model = MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    reloaded_checkpoint = safe_torch_load(
        checkpoint_path,
        map_location=device
    )

    reloaded_model.load_state_dict(
        reloaded_checkpoint["model_state_dict"]
    )

    checkpoint_reload_ok = True

    if device.type == "cuda":
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
    else:
        peak_allocated = None
        peak_reserved = None

    print("\n--- Sanity Results ---")
    print(f"Initial train batch loss: {train_stats['first_batch_loss']:.6f}")
    print(f"Final train batch loss:   {train_stats['last_batch_loss']:.6f}")
    print(f"Average train loss:       {train_stats['loss']:.6f}")
    print(f"Validation loss:          {val_loss:.6f}")
    print(f"Loss change:              "
          f"{train_stats['last_batch_loss'] - train_stats['first_batch_loss']:.6f}")
    print(f"Elapsed time:             {elapsed:.2f}s")
    print(f"Checkpoint:               {checkpoint_path}")
    print(f"Checkpoint reload:        {'PASS' if checkpoint_reload_ok else 'FAIL'}")

    if peak_allocated is not None:
        print(f"Peak GPU allocated:       {peak_allocated:.2f} MB")
        print(f"Peak GPU reserved:        {peak_reserved:.2f} MB")

    if train_stats["last_batch_loss"] < train_stats["first_batch_loss"]:
        print("Loss trend:               DECREASED")
    else:
        print(
            "Loss trend:               DID NOT DECREASE "
            "during this short sample"
        )

    print("\nSanity check complete.")
    print("Do not start full training until the results are reviewed.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the MAESTRO MusicLSTM model."
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Project root directory."
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        help="Input sequence length."
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window stride. Use 10 initially."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size."
    )

    parser.add_argument(
        "--workers",
        "--num-workers",
        dest="workers",
        type=int,
        default=2,
        help="DataLoader worker processes."
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension."
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="LSTM hidden size."
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of stacked LSTM layers."
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of epochs."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience."
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement."
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm."
    )

    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Number of recent epoch checkpoints to keep."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory."
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from latest checkpoint or a specified file."
    )

    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a limited training/validation test only."
    )

    parser.add_argument(
        "--sanity-batches",
        type=int,
        default=200,
        help="Number of train and validation batches in sanity mode."
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")

    if args.stride <= 0:
        raise ValueError("--stride must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.workers < 0:
        raise ValueError("--workers cannot be negative.")

    args.base_dir = args.base_dir.resolve()
    args.output_dir = (args.base_dir / args.output_dir).resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.cpu)
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print("=== Phase 4 Training Setup ===")
    print(f"Project root: {args.base_dir}")
    print(f"Device:       {device}")

    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(device)}")

    vocab_size = get_vocab_size(args.base_dir)
    print(f"Vocabulary:   {vocab_size}")

    print("\nLoading training dataset...")
    train_dataset = MaestroDataset(
        base_dir=str(args.base_dir),
        split="train",
        seq_len=args.seq_len,
        stride=args.stride,
    )

    print("Loading validation dataset...")
    val_dataset = MaestroDataset(
        base_dir=str(args.base_dir),
        split="validation",
        seq_len=args.seq_len,
        stride=args.stride,
    )

    print(f"Train windows: {len(train_dataset):,}")
    print(f"Val windows:   {len(val_dataset):,}")

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.workers,
        device=device,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        device=device,
    )

    model = MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    scaler = create_grad_scaler(use_amp)

    print("\nModel:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.sanity_check:
        run_sanity_check(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            args=args,
            vocab_size=vocab_size,
        )
        return

    start_epoch = 1
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume is not None:
        (
            start_epoch,
            history,
            best_val_loss,
            epochs_without_improvement,
        ) = load_resume_checkpoint(
            resume_argument=args.resume,
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    print("\n=== Full Training ===")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            gradient_clip=args.gradient_clip,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        }

        history.append(epoch_record)
        save_loss_history(args.output_dir, history)

        improved = val_loss < (best_val_loss - args.min_delta)

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_state = build_checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            history=history,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            args=args,
        )

        epoch_checkpoint = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        atomic_torch_save(checkpoint_state, epoch_checkpoint)

        if improved:
            best_checkpoint = checkpoint_dir / "best_model.pt"
            atomic_torch_save(checkpoint_state, best_checkpoint)

        remove_old_checkpoints(
            checkpoint_dir,
            keep_last=args.keep_checkpoints,
        )

        print(
            f"Train loss: {train_stats['loss']:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"LR: {current_lr:.8f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if improved:
            print("Validation loss improved. Best model saved.")
        else:
            print(
                f"No significant validation improvement. "
                f"Patience: {epochs_without_improvement}/{args.patience}"
            )

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Loss history: {args.output_dir / 'loss_history.json'}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()