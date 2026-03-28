from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.dirichlet_tsad.data import TelemanomDataset
from src.dirichlet_tsad.evaluation import (
    AggregateMetrics,
    aggregate_metrics,
    aggregate_metrics_to_frame,
    channel_metrics_to_frame,
    evaluate_channel,
)
from src.dirichlet_tsad.models import AVAILABLE_METHODS, build_detector
from src.dirichlet_tsad.thresholding import ThresholdConfig, apply_postprocessing, choose_threshold
from src.dirichlet_tsad.utils import parse_int_list


DEFAULT_KAGGLE_DATASET = Path("/kaggle/input/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
DEFAULT_KAGGLE_OUTPUT = Path("/kaggle/working/results")
DEFAULT_KAGGLE_WRAPPER = Path("/kaggle/working/telemanom_root")


def default_output_dir() -> str:
    if DEFAULT_KAGGLE_OUTPUT.parent.exists():
        return str(DEFAULT_KAGGLE_OUTPUT)
    return "outputs"


def _canonical_root_ready(root: Path) -> bool:
    return (
        root.exists()
        and (root / "labeled_anomalies.csv").exists()
        and (root / "train").is_dir()
        and (root / "test").is_dir()
    )


def _discover_telemanom_parts(raw_root: Path) -> Tuple[Path, Path, Path]:
    raw_root = raw_root.resolve()

    # Already in canonical format
    if _canonical_root_ready(raw_root):
        return raw_root / "labeled_anomalies.csv", raw_root / "train", raw_root / "test"

    # Kaggle dataset layout known in this conversation
    csv_path = raw_root / "labeled_anomalies.csv"
    train_dir = raw_root / "data" / "data" / "train"
    test_dir = raw_root / "data" / "data" / "test"
    if csv_path.exists() and train_dir.is_dir() and test_dir.is_dir():
        return csv_path, train_dir, test_dir

    # Fallback search
    csv_candidates = sorted(raw_root.rglob("labeled_anomalies.csv"))
    for csv in csv_candidates:
        parent = csv.parent
        if (parent / "train").is_dir() and (parent / "test").is_dir():
            return csv, parent / "train", parent / "test"
        if (parent / "data" / "data" / "train").is_dir() and (parent / "data" / "data" / "test").is_dir():
            return csv, parent / "data" / "data" / "train", parent / "data" / "data" / "test"

    raise FileNotFoundError(
        f"Could not find a Telemanom dataset layout under: {raw_root}\n"
        f"Expected either:\n"
        f"  root/labeled_anomalies.csv + root/train + root/test\n"
        f"or Kaggle layout:\n"
        f"  root/labeled_anomalies.csv + root/data/data/train + root/data/data/test"
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except Exception:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def prepare_telemanom_root(raw_root: Path, prepared_root: Path | None = None) -> Path:
    raw_root = raw_root.resolve()

    # If already canonical, use directly
    if _canonical_root_ready(raw_root):
        return raw_root

    csv_src, train_src, test_src = _discover_telemanom_parts(raw_root)

    if prepared_root is None:
        if DEFAULT_KAGGLE_WRAPPER.parent.exists():
            prepared_root = DEFAULT_KAGGLE_WRAPPER
        else:
            prepared_root = Path.cwd() / ".telemanom_root"

    prepared_root = prepared_root.resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)

    csv_dst = prepared_root / "labeled_anomalies.csv"
    train_dst = prepared_root / "train"
    test_dst = prepared_root / "test"

    if not csv_dst.exists():
        shutil.copy2(csv_src, csv_dst)
    _link_or_copy(train_src, train_dst)
    _link_or_copy(test_src, test_dst)

    if not _canonical_root_ready(prepared_root):
        raise RuntimeError(f"Prepared dataset root is incomplete: {prepared_root}")

    return prepared_root


def resolve_data_dir(user_value: str | None) -> str:
    if user_value is not None:
        return str(prepare_telemanom_root(Path(user_value)))

    if DEFAULT_KAGGLE_DATASET.exists():
        return str(prepare_telemanom_root(DEFAULT_KAGGLE_DATASET))

    raise FileNotFoundError(
        "No --data-dir was provided and the default Kaggle dataset path was not found."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Dirichlet TSAD and baselines on SMAP/MSL.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw or Telemanom-format data folder")
    parser.add_argument("--spacecraft", type=str, default="both", choices=["SMAP", "MSL", "both"], help="Subset to evaluate")
    parser.add_argument("--methods", nargs="+", default=["proposed_dirichlet"], choices=sorted(AVAILABLE_METHODS.keys()))
    parser.add_argument("--output-dir", type=str, default=default_output_dir())
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--force-target-only", action=argparse.BooleanOptionalAction, default=True, help="Use only target_index for every method for fair target-only comparison")

    parser.add_argument("--threshold-mode", type=str, default="alert_budget_under", choices=["fixed_quantile", "alert_budget_under", "alert_budget_closest"])
    parser.add_argument("--threshold-q", type=float, default=0.995)
    parser.add_argument("--alert-budget", type=float, default=0.005)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--threshold-warmup", type=int, default=-1, help="Warmup length ignored during threshold calibration. -1 = method-aware automatic warmup")
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--refractory", type=int, default=0, help="True refractory suppression after an alarm run")
    parser.add_argument("--bridge-gap", type=int, default=0, help="Merge nearby alarm runs if the gap length is <= bridge-gap")

    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--lags", type=str, default="1")
    parser.add_argument("--norm-window", type=int, default=256)

    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--skip-errors", dest="skip_errors", action="store_true", help="Skip failed channels/methods")
    parser.add_argument("--strict-errors", dest="skip_errors", action="store_false", help="Stop on the first error")
    parser.set_defaults(skip_errors=True)

    return parser.parse_args()


def detector_kwargs(method: str, args: argparse.Namespace) -> Dict:
    common = {"normalize": True}

    if method == "proposed_dirichlet":
        return {
            **common,
            "alpha": args.alpha,
            "lags": parse_int_list(args.lags),
            "norm_window": args.norm_window,
            "kappa": args.kappa,
            "target_index": args.target_index,
        }
    if method == "moving_average":
        return {**common, "window": args.window_size, "target_index": args.target_index}
    if method == "ewma":
        return {**common, "alpha": 0.05, "target_index": args.target_index}
    if method == "periodic_fft":
        return {**common, "alpha": args.alpha, "target_index": args.target_index}
    if method == "spectral_residual":
        return {**common, "target_index": args.target_index}
    if method == "pca":
        return {**common, "window_size": args.window_size, "n_components": 0.95}
    if method == "isolation_forest":
        return {**common, "window_size": args.window_size, "contamination": max(args.alert_budget, 1e-3)}
    if method == "autoencoder":
        return {
            **common,
            "window_size": args.window_size,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
        }
    if method == "lstm_forecast":
        return {
            **common,
            "window_size": args.window_size,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_index": args.target_index,
            "device": args.device,
        }
    raise KeyError(method)


def maybe_target_only(x: np.ndarray, target_index: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    if x.ndim != 2:
        return x
    if x.shape[1] <= 1:
        return x
    return x[:, [target_index]]


def method_threshold_warmup(method: str, args: argparse.Namespace) -> int:
    if args.threshold_warmup >= 0:
        return int(args.threshold_warmup)

    if method in {"pca", "isolation_forest", "autoencoder", "lstm_forecast", "moving_average"}:
        return int(args.window_size)
    if method == "proposed_dirichlet":
        lags = parse_int_list(args.lags)
        return int(max(lags) if len(lags) > 0 else 0)
    return 0


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Using data root :", args.data_dir)
    print("[INFO] Using output dir:", out_root.resolve())
    print("[INFO] Spacecraft      :", args.spacecraft)
    print("[INFO] Force target-only:", args.force_target_only)

    dataset = TelemanomDataset(args.data_dir, target_index=args.target_index)
    aggregate_rows: List[AggregateMetrics] = []

    for method in args.methods:
        method_dir = out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)

        threshold_cfg = ThresholdConfig(
            mode=args.threshold_mode,
            q=args.threshold_q,
            beta=args.alert_budget,
            train_fraction=args.train_fraction,
            warmup=method_threshold_warmup(method, args),
        )

        channel_rows = []
        y_true_all = []
        y_pred_all = []
        runtime_rows = []

        print(f"[INFO] Running method: {method}")

        for record in dataset.iter_channels(spacecraft=args.spacecraft):
            try:
                detector = build_detector(method, **detector_kwargs(method, args))
                train_x = maybe_target_only(record.train, args.target_index, args.force_target_only)
                test_x = maybe_target_only(record.test, args.target_index, args.force_target_only)

                t0 = time.perf_counter()
                detector.fit(train_x)
                train_scores = detector.score(train_x)
                test_scores = detector.score(test_x)
                fit_time = time.perf_counter() - t0

                threshold = choose_threshold(train_scores, threshold_cfg)
                pred = (test_scores > threshold).astype(np.int32)
                pred = apply_postprocessing(
                    pred,
                    persistence=args.persistence,
                    refractory=args.refractory,
                    bridge_gap=args.bridge_gap,
                )

                row = evaluate_channel(
                    channel_id=record.channel_id,
                    spacecraft=record.spacecraft,
                    y_true=record.labels,
                    y_pred=pred,
                    threshold=threshold,
                )
                channel_rows.append(row)
                y_true_all.append(record.labels)
                y_pred_all.append(pred)

                runtime_rows.append(
                    {
                        "channel_id": record.channel_id,
                        "spacecraft": record.spacecraft,
                        "fit_and_score_seconds": fit_time,
                        "train_length": len(record.train),
                        "test_length": len(record.test),
                        "n_features_original": int(record.train.shape[1]),
                        "n_features_used": int(train_x.shape[1]) if train_x.ndim == 2 else 1,
                        "force_target_only": bool(args.force_target_only),
                        "threshold_warmup": int(threshold_cfg.warmup),
                    }
                )

                pd.DataFrame(
                    {
                        "score": test_scores,
                        "pred": pred,
                        "label": record.labels,
                    }
                ).to_csv(method_dir / f"{record.channel_id}_scores.csv", index=False)

            except Exception as exc:
                if args.skip_errors:
                    print(f"[WARN] {method} / {record.channel_id} failed: {exc}")
                    continue
                raise

        ch_df = channel_metrics_to_frame(channel_rows)
        ch_df.to_csv(method_dir / "channel_metrics.csv", index=False)
        pd.DataFrame(runtime_rows).to_csv(method_dir / "runtime.csv", index=False)

        agg = aggregate_metrics(method, args.spacecraft, channel_rows, y_true_all, y_pred_all)
        aggregate_rows.append(agg)

        with open(method_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    agg_df = aggregate_metrics_to_frame(aggregate_rows)
    agg_df.to_csv(out_root / "aggregate_metrics.csv", index=False)
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
