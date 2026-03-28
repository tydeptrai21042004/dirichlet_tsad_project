from __future__ import annotations

import numpy as np
from scipy.fft import rfft, irfft

from ..utils import causal_moving_average, rolling_mad
from .base import BaseDetector


class MovingAverageResidualDetector(BaseDetector):
    name = "moving_average"

    def __init__(self, window: int = 32, target_index: int = 0, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.window = int(window)
        self.target_index = int(target_index)

    def _fit_impl(self, train: np.ndarray) -> None:
        return None

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        x = series[:, self.target_index]
        bg = causal_moving_average(x, self.window)
        residual = np.abs(x - bg)
        return residual / rolling_mad(residual, max(16, self.window))


class EWMADetector(BaseDetector):
    name = "ewma"

    def __init__(self, alpha: float = 0.05, target_index: int = 0, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.alpha = float(alpha)
        self.target_index = int(target_index)

    def _fit_impl(self, train: np.ndarray) -> None:
        return None

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        x = series[:, self.target_index].astype(np.float32)
        ema = np.zeros_like(x)
        ema[0] = x[0]
        for i in range(1, len(x)):
            ema[i] = self.alpha * x[i] + (1.0 - self.alpha) * ema[i - 1]
        residual = np.abs(x - ema)
        return residual / rolling_mad(residual, 64)


class PeriodicFFTResidualDetector(BaseDetector):
    name = "periodic_fft"

    def __init__(self, alpha: float = 50.0, target_index: int = 0, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.alpha = float(alpha)
        self.target_index = int(target_index)

    def _fit_impl(self, train: np.ndarray) -> None:
        return None

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        x = series[:, self.target_index].astype(np.float32)
        n = len(x)
        X = rfft(x)
        k = np.arange(len(X), dtype=np.float32)
        lamb = 4.0 * np.sin(np.pi * k / max(1, n)) ** 2
        mu = 1.0 / (1.0 + self.alpha * lamb)
        bg = irfft(X * mu, n=n).astype(np.float32)
        residual = np.abs(x - bg)
        return residual / rolling_mad(residual, 64)
