from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils import fit_standard_scaler, ensure_2d


class BaseDetector(ABC):
    name: str = "base"

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.scaler = None

    def fit(self, train: np.ndarray) -> "BaseDetector":
        x = ensure_2d(train)
        if self.normalize:
            self.scaler = fit_standard_scaler(x)
            x = self.scaler.transform(x)
        self._fit_impl(x)
        return self

    def score(self, series: np.ndarray) -> np.ndarray:
        x = ensure_2d(series)
        if self.normalize and self.scaler is not None:
            x = self.scaler.transform(x)
        s = self._score_impl(x)
        s = np.asarray(s, dtype=np.float32)
        if s.ndim != 1 or len(s) != len(x):
            raise ValueError(
                f"Detector {self.name} returned score shape {s.shape}; expected ({len(x)},)"
            )
        return s

    @abstractmethod
    def _fit_impl(self, train: np.ndarray) -> None:
        ...

    @abstractmethod
    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        ...
