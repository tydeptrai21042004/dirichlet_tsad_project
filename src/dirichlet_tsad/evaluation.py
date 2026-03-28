from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .utils import overlap, segments_from_binary


@dataclass
class ChannelMetrics:
    channel_id: str
    spacecraft: str
    point_precision: float
    point_recall: float
    point_f1: float
    segment_precision: float
    segment_recall: float
    segment_f1: float
    earlyhit_at_10: float
    median_delay: float
    false_alarm_segments: int
    n_gt_segments: int
    n_pred_segments: int
    threshold: float


@dataclass
class AggregateMetrics:
    split: str
    method: str
    point_precision_micro: float
    point_recall_micro: float
    point_f1_micro: float
    point_precision_macro: float
    point_recall_macro: float
    point_f1_macro: float
    segment_precision_macro: float
    segment_recall_macro: float
    segment_f1_macro: float
    earlyhit_at_10_macro: float
    median_delay_macro: float
    false_alarm_segments_total: int
    channels: int


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true.astype(int), y_pred.astype(int), average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def compute_segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    early_ratio: float = 0.10,
) -> Tuple[float, float, float, float, float, int, int, int]:
    gt_segments = segments_from_binary(y_true)
    pred_segments = segments_from_binary(y_pred)

    if len(pred_segments) == 0:
        segment_precision = 0.0
    else:
        tp_pred = sum(any(overlap(p, g) for g in gt_segments) for p in pred_segments)
        segment_precision = tp_pred / len(pred_segments)

    if len(gt_segments) == 0:
        segment_recall = 0.0
        earlyhit = 0.0
        median_delay = 0.0
    else:
        tp_gt = sum(any(overlap(g, p) for p in pred_segments) for g in gt_segments)
        segment_recall = tp_gt / len(gt_segments)

        early_hits: List[int] = []
        delays: List[int] = []
        for start, end in gt_segments:
            pred_indices = np.where(y_pred[start:end] == 1)[0]
            if len(pred_indices) == 0:
                continue
            first = start + int(pred_indices[0])
            delays.append(first - start)
            allowed = start + max(1, int(np.ceil((end - start) * early_ratio)))
            early_hits.append(1 if first < allowed else 0)
        earlyhit = float(np.mean(early_hits)) if early_hits else 0.0
        median_delay = float(np.median(delays)) if delays else float("nan")

    if segment_precision + segment_recall == 0:
        segment_f1 = 0.0
    else:
        segment_f1 = 2.0 * segment_precision * segment_recall / (segment_precision + segment_recall)

    false_alarm_segments = sum(not any(overlap(p, g) for g in gt_segments) for p in pred_segments)
    return (
        float(segment_precision),
        float(segment_recall),
        float(segment_f1),
        float(earlyhit),
        float(median_delay),
        int(false_alarm_segments),
        len(gt_segments),
        len(pred_segments),
    )


def evaluate_channel(
    channel_id: str,
    spacecraft: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> ChannelMetrics:
    pp, pr, pf1 = compute_point_metrics(y_true, y_pred)
    sp, sr, sf1, eh, md, fa, n_gt, n_pred = compute_segment_metrics(y_true, y_pred)
    return ChannelMetrics(
        channel_id=channel_id,
        spacecraft=spacecraft,
        point_precision=pp,
        point_recall=pr,
        point_f1=pf1,
        segment_precision=sp,
        segment_recall=sr,
        segment_f1=sf1,
        earlyhit_at_10=eh,
        median_delay=md,
        false_alarm_segments=fa,
        n_gt_segments=n_gt,
        n_pred_segments=n_pred,
        threshold=threshold,
    )


def aggregate_metrics(method: str, split: str, rows: Sequence[ChannelMetrics], y_true_all: List[np.ndarray], y_pred_all: List[np.ndarray]) -> AggregateMetrics:
    y_true_cat = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int32)
    y_pred_cat = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int32)
    micro_p, micro_r, micro_f1 = compute_point_metrics(y_true_cat, y_pred_cat)

    def mean_attr(name: str) -> float:
        values = np.asarray([getattr(r, name) for r in rows], dtype=np.float64)
        if len(values) == 0:
            return 0.0
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else 0.0

    return AggregateMetrics(
        split=split,
        method=method,
        point_precision_micro=micro_p,
        point_recall_micro=micro_r,
        point_f1_micro=micro_f1,
        point_precision_macro=mean_attr("point_precision"),
        point_recall_macro=mean_attr("point_recall"),
        point_f1_macro=mean_attr("point_f1"),
        segment_precision_macro=mean_attr("segment_precision"),
        segment_recall_macro=mean_attr("segment_recall"),
        segment_f1_macro=mean_attr("segment_f1"),
        earlyhit_at_10_macro=mean_attr("earlyhit_at_10"),
        median_delay_macro=mean_attr("median_delay"),
        false_alarm_segments_total=int(sum(r.false_alarm_segments for r in rows)),
        channels=len(rows),
    )


def channel_metrics_to_frame(rows: Sequence[ChannelMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def aggregate_metrics_to_frame(rows: Sequence[AggregateMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])
