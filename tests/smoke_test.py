from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def make_fake_dataset(root: Path) -> None:
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for spacecraft, channel_id in [("SMAP", "A-1"), ("MSL", "M-1")]:
        t_train = np.linspace(0, 20, 400)
        t_test = np.linspace(0, 20, 300)
        train = np.stack([
            np.sin(t_train) + 0.05 * rng.normal(size=len(t_train)),
            np.cos(t_train) + 0.05 * rng.normal(size=len(t_train)),
        ], axis=1).astype(np.float32)
        test = np.stack([
            np.sin(t_test) + 0.05 * rng.normal(size=len(t_test)),
            np.cos(t_test) + 0.05 * rng.normal(size=len(t_test)),
        ], axis=1).astype(np.float32)
        test[120:145, 0] += 3.0
        np.save(root / "train" / f"{channel_id}.npy", train)
        np.save(root / "test" / f"{channel_id}.npy", test)

    with open(root / "labeled_anomalies.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["channel_id", "spacecraft", "anomaly_sequences", "class", "num_values"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "channel_id": "A-1",
                "spacecraft": "SMAP",
                "anomaly_sequences": "[[120, 144]]",
                "class": "contextual",
                "num_values": 300,
            }
        )
        writer.writerow(
            {
                "channel_id": "M-1",
                "spacecraft": "MSL",
                "anomaly_sequences": "[[120, 144]]",
                "class": "contextual",
                "num_values": 300,
            }
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "telemanom"
        make_fake_dataset(root)
        out_dir = Path(tmp) / "out"
        cmd = [
            sys.executable,
            "run_experiment.py",
            "--data-dir",
            str(root),
            "--output-dir",
            str(out_dir),
            "--methods",
            "proposed_dirichlet",
            "periodic_fft",
            "moving_average",
            "pca",
            "--window-size",
            "32",
            "--epochs",
            "2",
        ]
        subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])
        agg = pd.read_csv(out_dir / "aggregate_metrics.csv")
        assert len(agg) == 4
        assert set(agg["method"]) == {"proposed_dirichlet", "periodic_fft", "moving_average", "pca"}
        print("Smoke test passed")


if __name__ == "__main__":
    main()
