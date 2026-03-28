from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from ..utils import make_windows, scatter_window_scores
from .base import BaseDetector


class PCADetector(BaseDetector):
    name = "pca"

    def __init__(self, window_size: int = 64, n_components: float | int = 0.95, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.window_size = int(window_size)
        self.n_components = n_components

    def _fit_impl(self, train: np.ndarray) -> None:
        windows, _, end_pos = make_windows(train, self.window_size, horizon=1, stride=1)
        self.end_positions_train_ = end_pos
        X = windows.reshape(len(windows), -1).astype(np.float32)
        if X.shape[0] < 2 or np.allclose(np.var(X, axis=0), 0.0):
            self.model_ = None
            return
        self.model_ = PCA(n_components=self.n_components, svd_solver="full", random_state=42)
        self.model_.fit(X)

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        windows, _, end_pos = make_windows(series, self.window_size, horizon=1, stride=1)
        X = windows.reshape(len(windows), -1).astype(np.float32)
        if self.model_ is None:
            return np.zeros(len(series), dtype=np.float32)
        recon = self.model_.inverse_transform(self.model_.transform(X))
        errs = np.mean((X - recon) ** 2, axis=1)
        scores = scatter_window_scores(errs, end_pos, len(series), horizon=1)
        return scores


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"

    def __init__(self, window_size: int = 32, contamination: float = 0.05, normalize: bool = True):
        super().__init__(normalize=normalize)
        self.window_size = int(window_size)
        self.contamination = float(contamination)

    def _fit_impl(self, train: np.ndarray) -> None:
        windows, _, _ = make_windows(train, self.window_size, horizon=1, stride=1)
        X = windows.reshape(len(windows), -1)
        self.model_ = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model_.fit(X)

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        windows, _, end_pos = make_windows(series, self.window_size, horizon=1, stride=1)
        X = windows.reshape(len(windows), -1)
        raw = -self.model_.score_samples(X)
        scores = scatter_window_scores(raw, end_pos, len(series), horizon=1)
        return scores


class AutoencoderWindowDetector(BaseDetector):
    name = "autoencoder"

    def __init__(
        self,
        window_size: int = 64,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        normalize: bool = True,
        device: str = "cpu",
    ):
        super().__init__(normalize=normalize)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device

    def _fit_impl(self, train: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        windows, _, _ = make_windows(train, self.window_size, horizon=1, stride=1)
        X = windows.reshape(len(windows), -1).astype(np.float32)
        input_dim = X.shape[1]

        class AE(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        self.model_ = AE(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        ds = TensorDataset(torch.from_numpy(X))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in dl:
                batch = batch.to(self.device)
                opt.zero_grad()
                recon = self.model_(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        import torch

        windows, _, end_pos = make_windows(series, self.window_size, horizon=1, stride=1)
        X = windows.reshape(len(windows), -1).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            batch = torch.from_numpy(X).to(self.device)
            recon = self.model_(batch).cpu().numpy()
        errs = np.mean((X - recon) ** 2, axis=1)
        return scatter_window_scores(errs, end_pos, len(series), horizon=1)


class LSTMForecastDetector(BaseDetector):
    name = "lstm_forecast"

    def __init__(
        self,
        window_size: int = 64,
        hidden_dim: int = 64,
        layers: int = 1,
        epochs: int = 12,
        batch_size: int = 128,
        lr: float = 1e-3,
        target_index: int = 0,
        normalize: bool = True,
        device: str = "cpu",
    ):
        super().__init__(normalize=normalize)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.target_index = int(target_index)
        self.device = device

    def _fit_impl(self, train: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        windows, y, _ = make_windows(train, self.window_size, horizon=1, stride=1)
        X = windows.astype(np.float32)
        target = y[:, 0, self.target_index].astype(np.float32)[:, None]
        input_dim = X.shape[-1]

        class ForecastLSTM(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, layers: int):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last)

        self.model_ = ForecastLSTM(input_dim, self.hidden_dim, self.layers).to(self.device)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(target))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in dl:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                opt.zero_grad()
                pred = self.model_(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                opt.step()

    def _score_impl(self, series: np.ndarray) -> np.ndarray:
        import torch

        windows, y, end_pos = make_windows(series, self.window_size, horizon=1, stride=1)
        X = windows.astype(np.float32)
        target = y[:, 0, self.target_index].astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X).to(self.device)).cpu().numpy().squeeze(-1)
        errs = np.abs(target - pred)
        return scatter_window_scores(errs, end_pos, len(series), horizon=1)
