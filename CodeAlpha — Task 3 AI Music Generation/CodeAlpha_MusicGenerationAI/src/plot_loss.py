from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train vs validation loss from loss_history.json.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.base_dir = args.base_dir.resolve()

    if args.output_dir is None:
        args.output_dir = args.base_dir / "outputs"

    history_path = args.output_dir / "loss_history.json"

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    best_epoch = min(history, key=lambda e: e["val_loss"])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation loss", marker="o")
    plt.axvline(
        best_epoch["epoch"],
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"Best epoch ({best_epoch['epoch']})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MusicLSTM Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = args.output_dir / "loss_curve.png"
    plt.savefig(output_path, dpi=150)

    print(f"Loss curve saved: {output_path}")
    print(f"Best epoch: {best_epoch['epoch']} (val_loss={best_epoch['val_loss']:.6f})")


if __name__ == "__main__":
    main()