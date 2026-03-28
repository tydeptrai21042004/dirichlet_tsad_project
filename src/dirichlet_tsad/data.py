from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd

from .utils import ensure_2d


@dataclass
class ChannelRecord:
    channel_id: str
    spacecraft: str
    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    anomaly_sequences: List[List[int]]
    anomaly_class: str | None


class TelemanomDataset:
    """Loader for the NASA/JPL Telemanom-formatted SMAP/MSL dataset.

    Expected structure:
        root/
          train/<channel>.npy
          test/<channel>.npy
          labeled_anomalies.csv
    """

    def __init__(self, root: str | Path, target_index: int = 0):
        self.root = Path(root)
        self.target_index = target_index
        self.train_dir = self.root / "train"
        self.test_dir = self.root / "test"
        self.labels_path = self.root / "labeled_anomalies.csv"

        if not self.train_dir.exists() or not self.test_dir.exists():
            raise FileNotFoundError(
                f"Expected train/ and test/ directories under {self.root}"
            )
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Expected labeled_anomalies.csv under {self.root}"
            )

        self.meta = pd.read_csv(self.labels_path)
        self.meta.columns = [c.strip().replace(" ", "_") for c in self.meta.columns]

        if "channel_id" not in self.meta.columns and "chan_id" in self.meta.columns:
            self.meta = self.meta.rename(columns={"chan_id": "channel_id"})

        if "anomaly_sequences" not in self.meta.columns:
            raise ValueError("labeled_anomalies.csv is missing anomaly_sequences")
        if "spacecraft" not in self.meta.columns:
            raise ValueError("labeled_anomalies.csv is missing spacecraft")

    def list_channels(self, spacecraft: str = "both") -> List[str]:
        df = self.meta
        if spacecraft.lower() != "both":
            df = df[df["spacecraft"].astype(str).str.upper() == spacecraft.upper()]
        # important: keep first appearance order, but remove duplicates like P-2
        return df["channel_id"].astype(str).drop_duplicates().tolist()

    def _load_array(self, split: str, channel_id: str) -> np.ndarray:
        path = self.root / split / f"{channel_id}.npy"
        if not path.exists():
            alt = self.root / split / f"{channel_id}.txt"
            if alt.exists():
                return ensure_2d(np.loadtxt(alt, dtype=np.float32))
            raise FileNotFoundError(f"Missing file for channel {channel_id}: {path}")
        arr = np.load(path)
        return ensure_2d(arr)

    def _parse_sequences(self, text) -> List[List[int]]:
        if isinstance(text, list):
            value = text
        else:
            value = ast.literal_eval(str(text))
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError(f"Invalid anomaly_sequences: {text}")

        out: List[List[int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            a, b = int(item[0]), int(item[1])
            out.append([a, b])
        return out

    def _merge_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        if not sequences:
            return []
        sequences = sorted(sequences, key=lambda x: (x[0], x[1]))
        merged = [sequences[0]]
        for s, e in sequences[1:]:
            last_s, last_e = merged[-1]
            # merge overlapping or touching intervals
            if s <= last_e + 1:
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])
        return merged

    def _make_labels(self, n: int, sequences: List[List[int]]) -> np.ndarray:
        y = np.zeros(n, dtype=np.int32)
        for start, end in sequences:
            start = max(0, int(start))
            end = min(n - 1, int(end))
            if end < start:
                continue
            y[start:end + 1] = 1
        return y

    def get_channel(self, channel_id: str) -> ChannelRecord:
        rows = self.meta[self.meta["channel_id"].astype(str) == str(channel_id)]
        if len(rows) == 0:
            raise KeyError(f"Channel not found: {channel_id}")

        train = self._load_array("train", channel_id)
        test = self._load_array("test", channel_id)

        spacecraft_values = rows["spacecraft"].astype(str).unique().tolist()
        spacecraft = spacecraft_values[0]

        all_sequences: List[List[int]] = []
        for _, row in rows.iterrows():
            all_sequences.extend(self._parse_sequences(row["anomaly_sequences"]))
        sequences = self._merge_sequences(all_sequences)

        labels = self._make_labels(len(test), sequences)

        anomaly_class = None
        if "class" in rows.columns:
            classes = rows["class"].dropna().astype(str).unique().tolist()
            if len(classes) == 1:
                anomaly_class = classes[0]
            elif len(classes) > 1:
                anomaly_class = "|".join(classes)

        return ChannelRecord(
            channel_id=str(channel_id),
            spacecraft=spacecraft,
            train=train,
            test=test,
            labels=labels,
            anomaly_sequences=sequences,
            anomaly_class=anomaly_class,
        )

    def iter_channels(self, spacecraft: str = "both") -> Iterator[ChannelRecord]:
        for channel_id in self.list_channels(spacecraft=spacecraft):
            yield self.get_channel(channel_id)
