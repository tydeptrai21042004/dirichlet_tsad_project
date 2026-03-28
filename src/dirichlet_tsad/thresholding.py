from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ThresholdMode = Literal[
    "fixed_quantile",
    "alert_budget_under",
    "alert_budget_closest",
]


@dataclass
class ThresholdConfig:
    mode: ThresholdMode = "alert_budget_under"
    q: float = 0.995
    beta: float = 0.10
    train_fraction: float = 0.30
    warmup: int = 0  # ignore unstable first scores during calibration


def choose_threshold(scores: np.ndarray, config: ThresholdConfig) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0

    end = max(1, int(len(scores) * config.train_fraction))
    start = min(max(int(config.warmup), 0), max(end - 1, 0))

    subset = scores[start:end]
    if len(subset) == 0:
        subset = scores[:end]
    if len(subset) == 0:
        return 0.0

    if config.mode == "fixed_quantile":
        return float(np.quantile(subset, config.q))

    unique_scores = np.unique(subset)
    if len(unique_scores) == 1:
        return float(unique_scores[0])

    # rates decrease as threshold increases
    rates = np.asarray([(subset > thr).mean() for thr in unique_scores], dtype=np.float32)

    if config.mode == "alert_budget_under":
        valid = np.where(rates <= config.beta)[0]
        if len(valid) == 0:
            return float(unique_scores[-1])
        return float(unique_scores[valid[0]])

    if config.mode == "alert_budget_closest":
        idx = int(np.argmin(np.abs(rates - config.beta)))
        return float(unique_scores[idx])

    raise ValueError(f"Unsupported threshold mode: {config.mode}")


def _find_runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.int32)
    if len(x) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    padded = np.pad(x, (1, 1), mode="constant")
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return starts, ends


def apply_postprocessing(
    pred: np.ndarray,
    persistence: int = 1,
    refractory: int = 0,
    bridge_gap: int = 0,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int32).copy()
    n = len(pred)
    if n == 0:
        return pred

    # Step 1: remove short positive runs
    if persistence > 1:
        starts, ends = _find_runs(pred)
        out = np.zeros(n, dtype=np.int32)
        for s, e in zip(starts, ends):
            if (e - s + 1) >= persistence:
                out[s : e + 1] = 1
        pred = out

    # Step 2: optionally bridge small gaps between runs
    if bridge_gap > 0:
        starts, ends = _find_runs(pred)
        if len(starts) > 1:
            out = pred.copy()
            cur_e = ends[0]
            for s, e in zip(starts[1:], ends[1:]):
                gap = s - cur_e - 1
                if gap <= bridge_gap:
                    out[cur_e + 1 : s] = 1
                cur_e = e
            pred = out

    # Step 3: true refractory suppression: keep the first run, suppress runs that
    # re-trigger too soon after the previous kept run has ended.
    if refractory > 0:
        starts, ends = _find_runs(pred)
        if len(starts) > 1:
            out = np.zeros(n, dtype=np.int32)
            keep_s, keep_e = starts[0], ends[0]
            out[keep_s : keep_e + 1] = 1
            for s, e in zip(starts[1:], ends[1:]):
                gap = s - keep_e - 1
                if gap > refractory:
                    out[s : e + 1] = 1
                    keep_s, keep_e = s, e
            pred = out

    return pred.astype(np.int32)
