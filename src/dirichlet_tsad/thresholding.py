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
    beta: float = 0.005
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

    rates = np.asarray([(subset > thr).mean() for thr in unique_scores], dtype=np.float32)

    if config.mode == "alert_budget_under":
        valid = np.where(rates <= config.beta)[0]
        if len(valid) == 0:
            return float(unique_scores[-1])
        return float(unique_scores[valid[0]])

    if config.mode == "alert_budget_closest":
        idx = int(np.argmin(np.abs(rates - config.beta)))
        return float(unique_scores[idx])

    raise ValueError(f"Unknown threshold mode: {config.mode}")


def hysteresis_binarize(scores: np.ndarray, high: float, low: float | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if low is None:
        low = high
    low = min(float(low), float(high))
    out = np.zeros(len(scores), dtype=np.int32)
    active = False
    for i, s in enumerate(scores):
        if not active and s > high:
            active = True
        elif active and s < low:
            active = False
        out[i] = 1 if active else 0
    return out


def apply_postprocessing(
    pred: np.ndarray,
    persistence: int = 1,
    refractory: int = 0,
    bridge_gap: int = 0,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int32).copy()
    n = len(pred)
    out = np.zeros_like(pred)

    # Step 1: keep only runs with enough persistence
    i = 0
    while i < n:
        if pred[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pred[j] == 1:
            j += 1
        if j - i >= max(1, persistence):
            out[i:j] = 1
        i = j

    # Step 2: bridge nearby runs if the gap is small
    if bridge_gap > 0:
        runs = []
        i = 0
        while i < n:
            if out[i] == 0:
                i += 1
                continue
            j = i
            while j < n and out[j] == 1:
                j += 1
            runs.append((i, j))
            i = j
        for (cur_s, cur_e), (next_s, next_e) in zip(runs, runs[1:]):
            gap = next_s - cur_e
            if gap <= bridge_gap:
                out[cur_e:next_s] = 1

    # Step 3: refractory suppression after each accepted run
    if refractory > 0:
        final = np.zeros_like(out)
        i = 0
        block_until = -1
        while i < n:
            if out[i] == 0 or i < block_until:
                i += 1
                continue
            j = i
            while j < n and out[j] == 1:
                j += 1
            final[i:j] = 1
            block_until = j + refractory
            i = j
        out = final

    return out.astype(np.int32)
