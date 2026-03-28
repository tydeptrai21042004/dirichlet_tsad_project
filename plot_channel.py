from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dirichlet_tsad.data import TelemanomDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one channel with labels and predicted score.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--output", default="channel_plot.png")
    args = parser.parse_args()

    ds = TelemanomDataset(args.data_dir, target_index=args.target_index)
    rec = ds.get_channel(args.channel_id)
    df = pd.read_csv(args.scores_csv)

    x = rec.test[:, args.target_index]
    label = rec.labels
    score = df["score"].to_numpy()
    pred = df["pred"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x)
    axes[0].set_title(f"{args.channel_id} - test signal")
    axes[1].plot(score)
    axes[1].set_title("Anomaly score")
    axes[2].plot(label, label="label")
    axes[2].plot(pred, alpha=0.7, label="pred")
    axes[2].legend()
    axes[2].set_title("Ground truth vs prediction")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
