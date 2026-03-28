from __future__ import annotations

from .dirichlet import DirichletResidualDetector
from .smoothing import EWMADetector, MovingAverageResidualDetector, PeriodicFFTResidualDetector
from .spectral_residual import SpectralResidualDetector
from .window_baselines import AutoencoderWindowDetector, IsolationForestDetector, LSTMForecastDetector, PCADetector


AVAILABLE_METHODS = {
    "proposed_dirichlet": DirichletResidualDetector,
    "moving_average": MovingAverageResidualDetector,
    "ewma": EWMADetector,
    "periodic_fft": PeriodicFFTResidualDetector,
    "spectral_residual": SpectralResidualDetector,
    "pca": PCADetector,
    "isolation_forest": IsolationForestDetector,
    "autoencoder": AutoencoderWindowDetector,
    "lstm_forecast": LSTMForecastDetector,
}


def build_detector(name: str, **kwargs):
    if name not in AVAILABLE_METHODS:
        raise KeyError(f"Unknown method: {name}. Available: {sorted(AVAILABLE_METHODS)}")
    return AVAILABLE_METHODS[name](**kwargs)
