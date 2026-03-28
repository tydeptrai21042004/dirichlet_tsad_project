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


def choose_threshold(scores: np.ndarray, config: ThresholdConfig) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    n = max(1, int(len(scores) * config.train_fraction))
    subset = scores[:n]
    if config.mode == "fixed_quantile":
        return float(np.quantile(subset, config.q))

    unique_scores = np.unique(subset)
    if len(unique_scores) == 1:
        return float(unique_scores[0])

    rates = np.asarray([(subset > thr).mean() for thr in unique_scores])
    if config.mode == "alert_budget_under":
        valid = np.where(rates <= config.beta)[0]
        if len(valid) == 0:
            return float(unique_scores[-1])
        return float(unique_scores[valid[0]])
    if config.mode == "alert_budget_closest":
        idx = int(np.argmin(np.abs(rates - config.beta)))
        return float(unique_scores[idx])
    raise ValueError(f"Unsupported threshold mode: {config.mode}")


def apply_postprocessing(
    pred: np.ndarray,
    persistence: int = 1,
    refractory: int = 0,
) -> np.ndarray:
    pred = np.asarray(pred).astype(np.int32)
    n = len(pred)
    if persistence > 1:
        out = np.zeros(n, dtype=np.int32)
        run = 0
        for i in range(n):
            if pred[i]:
                run += 1
            else:
                run = 0
            if run >= persistence:
                out[i - persistence + 1 : i + 1] = 1
        pred = out
    if refractory > 0:
        out = pred.copy()
        cooldown = 0
        active = False
        for i in range(n):
            if cooldown > 0:
                if out[i] == 1:
                    out[i] = 0
                cooldown -= 1
            if out[i] == 1 and not active:
                cooldown = refractory
                active = True
            elif out[i] == 0:
                active = False
        pred = out
    return pred
