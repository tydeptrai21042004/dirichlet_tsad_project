from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class StandardScalerLike:
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale_ + self.mean_


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape={x.shape}")
    return x


def fit_standard_scaler(x: np.ndarray, eps: float = 1e-8) -> StandardScalerLike:
    x2 = ensure_2d(x)
    mean = x2.mean(axis=0)
    scale = x2.std(axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    return StandardScalerLike(mean_=mean, scale_=scale)


def rolling_mad(x: np.ndarray, window: int, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = max(0, i - window + 1)
        segment = x[s : i + 1]
        med = np.median(segment)
        mad = np.median(np.abs(segment - med))
        out[i] = 1.4826 * mad + eps
    return out


def causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1:
        return x.copy()
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        s = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[s]) / (i - s + 1)
    return out


def make_windows(
    x: np.ndarray,
    window_size: int,
    horizon: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = ensure_2d(x)
    n, d = x.shape
    if n < window_size + horizon:
        raise ValueError(
            f"Sequence too short for window_size={window_size}, horizon={horizon}, n={n}"
        )
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    end_positions: List[int] = []
    for end in range(window_size, n - horizon + 1, stride):
        xs.append(x[end - window_size : end])
        ys.append(x[end : end + horizon])
        end_positions.append(end)
    return np.stack(xs), np.stack(ys), np.asarray(end_positions, dtype=np.int64)


def scatter_window_scores(
    window_scores: np.ndarray,
    end_positions: np.ndarray,
    series_length: int,
    horizon: int = 1,
) -> np.ndarray:
    scores = np.zeros(series_length, dtype=np.float32)
    counts = np.zeros(series_length, dtype=np.float32)
    for score, end in zip(window_scores, end_positions):
        start = end
        stop = min(series_length, end + horizon)
        scores[start:stop] += float(score)
        counts[start:stop] += 1.0
    counts = np.where(counts == 0, 1.0, counts)
    return scores / counts


def segments_from_binary(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask).astype(bool)
    segments: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def parse_int_list(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]
