from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.fft import dst, idst

from ..utils import causal_moving_average, rolling_mad
from .base import BaseDetector


class DirichletResidualDetector(BaseDetector):
    name = "proposed_dirichlet"

    def __init__(
        self,
        alpha: float = 50.0,
        lags: Sequence[int] = (1,),
        lag_weights: Sequence[float] | None = None,
        norm_window: int = 256,
        kappa: float = 0.5,
        target_index: int = 0,
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.alpha = float(alpha)
        self.lags = tuple(sorted(set(int(l) for l in lags if int(l) > 0)))
        self.lag_weights = lag_weights
        self.norm_window = int(norm_window)
        self.kappa = float(kappa)
        self.target_index = int(target_index)

    def _fit_impl(self, train: np.ndarray) -> None:
        if self.lag_weights is None:
            self.weights_ = np.asarray([1.0 / l for l in self.lags], dtype=np.float32)
            self.weights_ = self.weights_ / self.weights_.sum()
        else:
            weights = np.asarray(self.lag_weights, dtype=np.float32)
            if len(weights) != len(self.lags):
                raise ValueError("lag_weights must have same length as lags")
            self.weights_ = weights / weights.sum()

    def _dirichlet_background(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        coeffs = dst(x, type=1, norm="ortho")
        k = np.arange(1, n + 1, dtype=np.float32)
        lamb = 4.0 * np.sin(np.pi * k / (2.0 * (n + 1.0))) ** 2
        mu = 1.0 / (1.0 + self.alpha * lamb)
        bg = idst(coeffs * mu, type=1, norm="ortho")
        return bg.astype(np.float32)

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        x = series[:, self.target_index].astype(np.float32)
        bg = self._dirichlet_background(x)
        residual = x - bg

        score = np.zeros_like(residual)
        for lag, weight in zip(self.lags, self.weights_):
            diff = np.zeros_like(residual)
            diff[lag:] = residual[lag:] - residual[:-lag]
            anti = np.abs(diff)
            sym = np.zeros_like(residual)
            sym[lag:] = np.abs(residual[lag:] + residual[:-lag])
            score += float(weight) * (anti + self.kappa * sym)

        norm = rolling_mad(score, window=self.norm_window)
        z = score / norm
        return np.maximum(z, 0.0).astype(np.float32)
