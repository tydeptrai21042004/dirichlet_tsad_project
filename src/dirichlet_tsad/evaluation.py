from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .utils import overlap, segments_from_binary


@dataclass
class ChannelMetrics:
    channel_id: str
    spacecraft: str
    point_precision: float
    point_recall: float
    point_f1: float
    point_auc_roc: float
    point_auc_pr: float
    segment_precision: float
    segment_recall: float
    segment_f1: float
    earlyhit_at_05: float
    earlyhit_at_10: float
    earlyhit_at_20: float
    median_delay: float
    mean_delay: float
    p90_delay: float
    false_alarm_segments: int
    false_alarm_segments_per_1k: float
    n_gt_segments: int
    n_pred_segments: int
    mean_pred_segment_length: float
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
    point_auc_roc_macro: float
    point_auc_pr_macro: float
    segment_precision_macro: float
    segment_recall_macro: float
    segment_f1_macro: float
    earlyhit_at_05_macro: float
    earlyhit_at_10_macro: float
    earlyhit_at_20_macro: float
    median_delay_macro: float
    mean_delay_macro: float
    p90_delay_macro: float
    false_alarm_segments_total: int
    false_alarm_segments_per_1k_macro: float
    mean_pred_segment_length_macro: float
    channels: int


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true.astype(int), y_pred.astype(int), average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def compute_auc_metrics(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return float('nan'), float('nan')
    try:
        auc_roc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc_roc = float('nan')
    try:
        auc_pr = float(average_precision_score(y_true, scores))
    except Exception:
        auc_pr = float('nan')
    return auc_roc, auc_pr


def _event_detection_stats(gt_segments, pred_segments, y_pred, early_ratio: float):
    if len(gt_segments) == 0:
        return 0.0, 0.0, []
    hits = []
    delays = []
    for start, end in gt_segments:
        pred_indices = np.where(y_pred[start:end] == 1)[0]
        if len(pred_indices) == 0:
            continue
        first = start + int(pred_indices[0])
        delays.append(first - start)
        allowed = start + max(1, int(np.ceil((end - start) * early_ratio)))
        hits.append(1 if first < allowed else 0)
    return (float(np.mean(hits)) if hits else 0.0,
            float(np.median(delays)) if delays else float('nan'),
            delays)


def compute_segment_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float, int, int, int, float]:
    gt_segments = segments_from_binary(y_true)
    pred_segments = segments_from_binary(y_pred)

    if len(pred_segments) == 0:
        segment_precision = 0.0
        mean_pred_segment_length = 0.0
    else:
        tp_pred = sum(any(overlap(p, g) for g in gt_segments) for p in pred_segments)
        segment_precision = tp_pred / len(pred_segments)
        mean_pred_segment_length = float(np.mean([e - s for s, e in pred_segments]))

    if len(gt_segments) == 0:
        segment_recall = 0.0
        eh05, eh10, eh20 = 0.0, 0.0, 0.0
        median_delay, mean_delay, p90_delay = 0.0, 0.0, 0.0
    else:
        tp_gt = sum(any(overlap(g, p) for p in pred_segments) for g in gt_segments)
        segment_recall = tp_gt / len(gt_segments)
        eh05, _, delays05 = _event_detection_stats(gt_segments, pred_segments, y_pred, 0.05)
        eh10, median_delay, delays10 = _event_detection_stats(gt_segments, pred_segments, y_pred, 0.10)
        eh20, _, delays20 = _event_detection_stats(gt_segments, pred_segments, y_pred, 0.20)
        delays = delays10 if delays10 else (delays20 if delays20 else delays05)
        mean_delay = float(np.mean(delays)) if delays else float('nan')
        p90_delay = float(np.percentile(delays, 90)) if delays else float('nan')

    if segment_precision + segment_recall == 0:
        segment_f1 = 0.0
    else:
        segment_f1 = 2.0 * segment_precision * segment_recall / (segment_precision + segment_recall)

    false_alarm_segments = sum(not any(overlap(p, g) for g in gt_segments) for p in pred_segments)
    false_alarm_segments_per_1k = 1000.0 * false_alarm_segments / max(1, len(y_true))
    return (
        float(segment_precision),
        float(segment_recall),
        float(segment_f1),
        float(eh05),
        float(eh10),
        float(eh20),
        float(median_delay),
        float(mean_delay),
        float(p90_delay),
        int(false_alarm_segments),
        len(gt_segments),
        len(pred_segments),
        float(false_alarm_segments_per_1k),
        float(mean_pred_segment_length),
    )


def evaluate_channel(channel_id: str, spacecraft: str, y_true: np.ndarray, y_pred: np.ndarray, threshold: float, scores: np.ndarray | None = None) -> ChannelMetrics:
    pp, pr, pf1 = compute_point_metrics(y_true, y_pred)
    auc_roc, auc_pr = compute_auc_metrics(y_true, scores if scores is not None else y_pred)
    sp, sr, sf1, eh05, eh10, eh20, md, mean_d, p90_d, fa, n_gt, n_pred, fa1k, mean_pred_len = compute_segment_metrics(y_true, y_pred)
    return ChannelMetrics(
        channel_id=channel_id,
        spacecraft=spacecraft,
        point_precision=pp,
        point_recall=pr,
        point_f1=pf1,
        point_auc_roc=auc_roc,
        point_auc_pr=auc_pr,
        segment_precision=sp,
        segment_recall=sr,
        segment_f1=sf1,
        earlyhit_at_05=eh05,
        earlyhit_at_10=eh10,
        earlyhit_at_20=eh20,
        median_delay=md,
        mean_delay=mean_d,
        p90_delay=p90_d,
        false_alarm_segments=fa,
        false_alarm_segments_per_1k=fa1k,
        n_gt_segments=n_gt,
        n_pred_segments=n_pred,
        mean_pred_segment_length=mean_pred_len,
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
        point_auc_roc_macro=mean_attr("point_auc_roc"),
        point_auc_pr_macro=mean_attr("point_auc_pr"),
        segment_precision_macro=mean_attr("segment_precision"),
        segment_recall_macro=mean_attr("segment_recall"),
        segment_f1_macro=mean_attr("segment_f1"),
        earlyhit_at_05_macro=mean_attr("earlyhit_at_05"),
        earlyhit_at_10_macro=mean_attr("earlyhit_at_10"),
        earlyhit_at_20_macro=mean_attr("earlyhit_at_20"),
        median_delay_macro=mean_attr("median_delay"),
        mean_delay_macro=mean_attr("mean_delay"),
        p90_delay_macro=mean_attr("p90_delay"),
        false_alarm_segments_total=int(sum(r.false_alarm_segments for r in rows)),
        false_alarm_segments_per_1k_macro=mean_attr("false_alarm_segments_per_1k"),
        mean_pred_segment_length_macro=mean_attr("mean_pred_segment_length"),
        channels=len(rows),
    )


def channel_metrics_to_frame(rows: Sequence[ChannelMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def aggregate_metrics_to_frame(rows: Sequence[AggregateMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])
