from __future__ import annotations

import numpy as np
from scipy.fft import dst, idst

from .base import BaseDetector


class DirichletResidualDetector(BaseDetector):
    """Boundary-aware detector with two-scale Dirichlet filtering.

    Pipeline:
        slow/mid Dirichlet backgrounds -> residuals -> signed onset score
        -> sustain/energy confirmation -> shape coherence -> robust normalization
    """

    name = "proposed_dirichlet"

    def __init__(
        self,
        alpha: float = 50.0,
        alpha_fast_ratio: float = 0.25,
        lags: tuple[int, ...] = (1,),
        lag_weights: tuple[float, ...] | None = None,
        norm_window: int = 256,
        kappa: float = 0.5,
        target_index: int = 0,
        normalize: bool = True,
        sustain_window: int = 12,
        coherence_window: int = 12,
        band_weight: float = 0.5,
        down_weight: float = 0.9,
        use_residual_gate: bool = True,
        residual_gate_quantile: float = 0.95,
    ):
        super().__init__(normalize=normalize)
        self.alpha = float(alpha)
        self.alpha_fast_ratio = float(alpha_fast_ratio)
        clean_lags = tuple(sorted(set(int(l) for l in lags if int(l) > 0)))
        self.lags = clean_lags if len(clean_lags) > 0 else (1,)
        self.lag_weights = lag_weights
        self.norm_window = int(norm_window)
        self.kappa = float(kappa)
        self.target_index = int(target_index)
        self.sustain_window = int(sustain_window)
        self.coherence_window = int(coherence_window)
        self.band_weight = float(band_weight)
        self.down_weight = float(down_weight)
        self.use_residual_gate = bool(use_residual_gate)
        self.residual_gate_quantile = float(residual_gate_quantile)

    def _fit_impl(self, train: np.ndarray) -> None:
        if self.lag_weights is None:
            weights = np.asarray([1.0 / l for l in self.lags], dtype=np.float32)
            self.weights_ = weights / max(weights.sum(), 1e-6)
        else:
            weights = np.asarray(self.lag_weights, dtype=np.float32)
            if len(weights) != len(self.lags):
                raise ValueError("lag_weights must have same length as lags")
            self.weights_ = weights / max(weights.sum(), 1e-6)

        self.residual_gate_threshold_ = 0.0
        if self.use_residual_gate:
            x = train[:, self.target_index].astype(np.float32)
            r_mid, r_band = self._residuals(x)
            sustain = self._causal_rms(r_mid, self.sustain_window) * (1.0 + self.band_weight * self._causal_rms(r_band, self.sustain_window))
            self.residual_gate_threshold_ = float(np.quantile(sustain, self.residual_gate_quantile))

    def _dirichlet_background(self, x: np.ndarray, alpha: float) -> np.ndarray:
        n = len(x)
        if n == 0:
            return x.astype(np.float32)
        coeffs = dst(x, type=1, norm="ortho")
        k = np.arange(1, n + 1, dtype=np.float32)
        lamb = 4.0 * np.sin(np.pi * k / (2.0 * (n + 1.0))) ** 2
        mu = 1.0 / (1.0 + float(alpha) * lamb)
        bg = idst(coeffs * mu, type=1, norm="ortho")
        return bg.astype(np.float32)

    def _residuals(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha_fast = max(self.alpha * self.alpha_fast_ratio, 1e-4)
        bg_slow = self._dirichlet_background(x, self.alpha)
        bg_mid = self._dirichlet_background(x, alpha_fast)
        r_mid = x - bg_mid
        r_band = bg_mid - bg_slow
        return r_mid.astype(np.float32), r_band.astype(np.float32)

    @staticmethod
    def _rolling_median_mad_z(x: np.ndarray, window: int, eps: float = 1e-6) -> np.ndarray:
        n = len(x)
        z = np.zeros(n, dtype=np.float32)
        window = max(int(window), 1)
        for i in range(n):
            s = max(0, i - window + 1)
            seg = x[s : i + 1]
            med = float(np.median(seg))
            mad = float(np.median(np.abs(seg - med)))
            scale = 1.4826 * max(mad, eps)
            z[i] = (x[i] - med) / scale
        return z

    @staticmethod
    def _causal_mean_abs(x: np.ndarray, window: int) -> np.ndarray:
        x = np.abs(np.asarray(x, dtype=np.float32))
        n = len(x)
        out = np.zeros(n, dtype=np.float32)
        w = max(int(window), 1)
        csum = np.cumsum(np.insert(x, 0, 0.0))
        for i in range(n):
            s = max(0, i - w + 1)
            out[i] = (csum[i + 1] - csum[s]) / float(i - s + 1)
        return out

    @staticmethod
    def _causal_mean(x: np.ndarray, window: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        out = np.zeros(n, dtype=np.float32)
        w = max(int(window), 1)
        csum = np.cumsum(np.insert(x, 0, 0.0))
        for i in range(n):
            s = max(0, i - w + 1)
            out[i] = (csum[i + 1] - csum[s]) / float(i - s + 1)
        return out

    def _causal_rms(self, x: np.ndarray, window: int) -> np.ndarray:
        return np.sqrt(np.maximum(self._causal_mean(np.square(x), window), 0.0) + 1e-6).astype(np.float32)

    def _signed_onset(self, residual: np.ndarray) -> np.ndarray:
        a = np.zeros_like(residual, dtype=np.float32)
        for lag, weight in zip(self.lags, self.weights_):
            if lag <= 0 or lag >= len(residual):
                continue
            forward = np.zeros_like(residual, dtype=np.float32)
            backward = np.zeros_like(residual, dtype=np.float32)
            forward[:-lag] = residual[lag:]
            backward[lag:] = residual[:-lag]
            a += float(weight) * (forward - backward)
        return (self.kappa * a).astype(np.float32)

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        x = series[:, self.target_index].astype(np.float32)
        r_mid, r_band = self._residuals(x)

        onset_raw = self._signed_onset(r_mid)
        onset_up = np.maximum(onset_raw, 0.0)
        onset_down = np.maximum(-onset_raw, 0.0)
        onset = np.maximum(onset_up, self.down_weight * onset_down)

        sustain = self._causal_rms(r_mid, self.sustain_window)
        band = self._causal_rms(r_band, self.sustain_window)
        sustain_mix = sustain * (1.0 + self.band_weight * band)

        mean_signed = self._causal_mean(r_mid, self.coherence_window)
        mean_abs = self._causal_mean_abs(r_mid, self.coherence_window)
        coherence = np.clip(np.abs(mean_signed) / (mean_abs + 1e-6), 0.0, 1.0)

        raw_score = onset * np.sqrt(np.maximum(sustain_mix, 1e-6)) * np.sqrt(np.maximum(coherence, 1e-6))
        z = np.maximum(self._rolling_median_mad_z(raw_score, window=self.norm_window), 0.0)

        edge = max(self.lags) if len(self.lags) > 0 else 1
        if edge > 0:
            z[:edge] = 0.0
            z[-edge:] = 0.0

        if self.use_residual_gate:
            thr = max(float(getattr(self, "residual_gate_threshold_", 0.0)), 1e-6)
            gate = np.clip(sustain_mix / thr, 0.0, 1.0)
            z = z * gate

        return z.astype(np.float32)
