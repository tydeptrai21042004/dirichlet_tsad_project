from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_experiment import default_output_dir, resolve_data_dir


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
    parser.add_argument("--data-dir", default=None, help="Optional raw dataset root. If omitted, Kaggle default is auto-detected.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to /kaggle/working/results on Kaggle.")
    parser.add_argument("--spacecraft", default="both", choices=["SMAP", "MSL", "both"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--alert-budget", type=float, default=0.005)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--threshold-warmup", type=int, default=-1)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--refractory", type=int, default=0)
    parser.add_argument("--bridge-gap", type=int, default=0)
    parser.add_argument("--if-contamination", type=float, default=0.01)
    parser.add_argument("--force-target-only", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = args.output_dir or default_output_dir()

    here = Path(__file__).resolve().parent
    run_script = here / "run_experiment.py"

    cmd = [
        sys.executable,
        str(run_script),
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--spacecraft",
        args.spacecraft,
        "--window-size",
        str(args.window_size),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--alert-budget",
        str(args.alert_budget),
        "--train-fraction",
        str(args.train_fraction),
        "--threshold-warmup",
        str(args.threshold_warmup),
        "--persistence",
        str(args.persistence),
        "--refractory",
        str(args.refractory),
        "--bridge-gap",
        str(args.bridge_gap),
        "--if-contamination",
        str(args.if_contamination),
        "--methods",
        *DEFAULT_METHODS,
    ]
    if args.force_target_only:
        cmd.append("--force-target-only")

    print("[INFO] Resolved data dir :", data_dir)
    print("[INFO] Output dir        :", output_dir)
    print("[INFO] Spacecraft subset :", args.spacecraft)
    print("[INFO] Methods           :", ", ".join(DEFAULT_METHODS))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
