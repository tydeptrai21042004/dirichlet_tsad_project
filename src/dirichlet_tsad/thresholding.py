from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdConfig:
    mode: str = "fixed_quantile"
    q: float = 0.995
    beta: float = 0.01
    train_fraction: float = 0.30
    warmup: int = 0


def choose_threshold(train_scores: np.ndarray, config: ThresholdConfig) -> float:
    scores = np.asarray(train_scores, dtype=np.float32)
    if len(scores) == 0:
        return 0.0

    start = max(0, int(config.warmup))
    end = max(start + 1, int(np.ceil(len(scores) * float(config.train_fraction))))
    end = min(end, len(scores))
    subset = scores[start:end]
    if len(subset) == 0:
        subset = scores

    if config.mode == "fixed_quantile":
        return float(np.quantile(subset, config.q))

    unique_scores = np.unique(subset)
    unique_scores = np.sort(unique_scores)[::-1]
    if len(unique_scores) == 0:
        return float(np.max(subset))

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

    # Step 2: merge nearby runs if the gap is small
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

    # Step 3: true refractory suppression after each positive run
    if refractory > 0:
        starts, ends = _find_runs(pred)
        out = pred.copy()
        for e in ends:
            sup_end = min(n, e + 1 + int(refractory))
            out[e + 1 : sup_end] = 0
        pred = out

    return pred.astype(np.int32)
