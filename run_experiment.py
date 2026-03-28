from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Dirichlet TSAD and baselines on SMAP/MSL.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to Telemanom-format data folder")
    parser.add_argument("--spacecraft", type=str, default="both", choices=["SMAP", "MSL", "both"], help="Subset to evaluate")
    parser.add_argument("--methods", nargs="+", default=["proposed_dirichlet"], choices=sorted(AVAILABLE_METHODS.keys()))
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--threshold-mode", type=str, default="alert_budget_under", choices=["fixed_quantile", "alert_budget_under", "alert_budget_closest"])
    parser.add_argument("--threshold-q", type=float, default=0.995)
    parser.add_argument("--alert-budget", type=float, default=0.005)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--persistence", type=int, default=1)
    parser.add_argument("--refractory", type=int, default=0)
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
    parser.add_argument("--skip-errors", action="store_true")
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


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = TelemanomDataset(args.data_dir, target_index=args.target_index)
    threshold_cfg = ThresholdConfig(
        mode=args.threshold_mode,
        q=args.threshold_q,
        beta=args.alert_budget,
        train_fraction=args.train_fraction,
    )

    aggregate_rows: List[AggregateMetrics] = []

    for method in args.methods:
        method_dir = out_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        channel_rows = []
        y_true_all = []
        y_pred_all = []
        runtime_rows = []

        for record in dataset.iter_channels(spacecraft=args.spacecraft):
            try:
                detector = build_detector(method, **detector_kwargs(method, args))
                t0 = time.perf_counter()
                detector.fit(record.train)
                train_scores = detector.score(record.train)
                test_scores = detector.score(record.test)
                fit_time = time.perf_counter() - t0

                threshold = choose_threshold(train_scores, threshold_cfg)
                pred = (test_scores > threshold).astype(np.int32)
                pred = apply_postprocessing(pred, persistence=args.persistence, refractory=args.refractory)

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
                        "n_features": int(record.train.shape[1]),
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
