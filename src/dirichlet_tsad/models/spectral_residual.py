from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .base import BaseDetector


class SpectralResidualDetector(BaseDetector):
    name = "spectral_residual"

    def __init__(self, mag_window: int = 3, score_window: int = 21, target_index: int = 0, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.mag_window = int(mag_window)
        self.score_window = int(score_window)
        self.target_index = int(target_index)

    def _fit_impl(self, train: np.ndarray) -> None:
        return None

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        x = series[:, self.target_index].astype(np.float32)
        if len(x) < 8:
            return np.zeros_like(x)
        spec = np.fft.fft(x)
        mag = np.abs(spec) + 1e-8
        log_mag = np.log(mag)
        avg_log = uniform_filter1d(log_mag, size=self.mag_window, mode="nearest")
        spectral_residual = np.exp(log_mag - avg_log)
        saliency = np.abs(np.fft.ifft(spec * spectral_residual))
        score = saliency / (uniform_filter1d(saliency, size=self.score_window, mode="nearest") + 1e-8)
        return score.astype(np.float32)
