from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_METHODS = [
    "proposed_dirichlet",
    "moving_average",
    "ewma",
    "periodic_fft",
    "spectral_residual",
    "pca",
    "isolation_forest",
    "autoencoder",
    "lstm_forecast",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convenience wrapper to run all baselines.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/full_benchmark")
    parser.add_argument("--spacecraft", default="both", choices=["SMAP", "MSL", "both"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "run_experiment.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        args.output_dir,
        "--spacecraft",
        args.spacecraft,
        "--window-size",
        str(args.window_size),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--methods",
        *DEFAULT_METHODS,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
