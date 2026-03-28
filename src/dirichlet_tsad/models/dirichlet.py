from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.fft import dst, idst

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
        use_residual_gate: bool = True,
        residual_gate_window: int = 5,
        residual_gate_quantile: float = 0.75,
    ):
        super().__init__(normalize=normalize)
        self.alpha = float(alpha)
        clean_lags = tuple(sorted(set(int(l) for l in lags if int(l) > 0)))
        self.lags = clean_lags if len(clean_lags) > 0 else (1,)
        self.lag_weights = lag_weights
        self.norm_window = int(norm_window)
        self.kappa = float(kappa)
        self.target_index = int(target_index)

        self.use_residual_gate = bool(use_residual_gate)
        self.residual_gate_window = int(residual_gate_window)
        self.residual_gate_quantile = float(residual_gate_quantile)

    def _fit_impl(self, train: np.ndarray) -> None:
        if self.lag_weights is None:
            weights = np.asarray([1.0 / l for l in self.lags], dtype=np.float32)
            self.weights_ = weights / weights.sum()
        else:
            weights = np.asarray(self.lag_weights, dtype=np.float32)
            if len(weights) != len(self.lags):
                raise ValueError("lag_weights must have same length as lags")
            self.weights_ = weights / weights.sum()

        # Fit a residual gate on train only
        self.residual_gate_threshold_ = 0.0
        if self.use_residual_gate:
            x = train[:, self.target_index].astype(np.float32)
            bg = self._dirichlet_background(x)
            residual = np.abs(x - bg)
            env = self._residual_envelope(residual)
            self.residual_gate_threshold_ = float(
                np.quantile(env, self.residual_gate_quantile)
            )

    def _dirichlet_background(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        coeffs = dst(x, type=1, norm="ortho")
        k = np.arange(1, n + 1, dtype=np.float32)
        lamb = 4.0 * np.sin(np.pi * k / (2.0 * (n + 1.0))) ** 2
        mu = 1.0 / (1.0 + self.alpha * lamb)
        bg = idst(coeffs * mu, type=1, norm="ortho")
        return bg.astype(np.float32)

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

    def _residual_envelope(self, residual_abs: np.ndarray) -> np.ndarray:
        w = max(int(self.residual_gate_window), 1)
        if w == 1:
            return residual_abs.astype(np.float32)
        kernel = np.ones(w, dtype=np.float32) / float(w)
        env = np.convolve(residual_abs.astype(np.float32), kernel, mode="same")
        return env.astype(np.float32)

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        x = series[:, self.target_index].astype(np.float32)
        bg = self._dirichlet_background(x)
        residual = x - bg

        # Paper-like antisymmetric score:
        # a_t = kappa * sum_l w_l * (r_{t+l} - r_{t-l})
        a = np.zeros_like(residual, dtype=np.float32)

        for lag, weight in zip(self.lags, self.weights_):
            if lag <= 0 or lag >= len(residual):
                continue

            forward = np.zeros_like(residual, dtype=np.float32)
            backward = np.zeros_like(residual, dtype=np.float32)

            # forward[t] = r_{t+lag}
            forward[:-lag] = residual[lag:]

            # backward[t] = r_{t-lag}
            backward[lag:] = residual[:-lag]

            a += float(weight) * (forward - backward)

        a *= self.kappa

        # Robust rolling normalization
        z = np.abs(self._rolling_median_mad_z(a, window=self.norm_window))

        # Make boundaries stricter because zero-padding can create spurious edge spikes
        edge = max(self.lags) if len(self.lags) > 0 else 1
        if edge > 0:
            z[:edge] = 0.0
            z[-edge:] = 0.0

        # Residual confirmation gate to reduce false positives
        if self.use_residual_gate:
            env = self._residual_envelope(np.abs(residual))
            thr = max(float(getattr(self, "residual_gate_threshold_", 0.0)), 1e-6)
            gate = np.clip(env / thr, 0.0, 1.0)
            z = z * gate

        return z.astype(np.float32)
