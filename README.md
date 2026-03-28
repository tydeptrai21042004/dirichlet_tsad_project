# Dirichlet TSAD Benchmark for SMAP / MSL

A clean Python benchmark package for anomaly detection on the NASA/JPL Telemanom-formatted **SMAP** and **MSL** telemetry datasets.

This package includes:

- the **proposed boundary-aware Dirichlet residual detector**
- a **periodic FFT residual baseline** to directly test the boundary-condition claim
- additional competitive baselines
- strict **train-only threshold calibration**
- pointwise and event-level evaluation
- early-detection metrics such as **EarlyHit@10%** and **median delay**
- CSV outputs for per-channel results and aggregate benchmark tables

---

## Included methods

### Proposed method
- `proposed_dirichlet`
  - Dirichlet / DST-I low-pass background estimation
  - residual extraction
  - short-horizon antisymmetric multi-lag score
  - robust causal normalization with rolling MAD

### Classical / signal baselines
- `moving_average`
- `ewma`
- `periodic_fft`
- `spectral_residual`

### Machine learning baselines
- `pca`
- `isolation_forest`
- `autoencoder`
- `lstm_forecast`

> Note: `lstm_forecast` is a practical **Telemanom-style forecasting baseline**, not the exact official NASA Telemanom code.

---

## Dataset format expected

The package expects the standard Telemanom-style layout:

```text
data/telemanom/
  train/
    A-1.npy
    A-2.npy
    ...
  test/
    A-1.npy
    A-2.npy
    ...
  labeled_anomalies.csv
```

Each channel file can be:
- shape `(T,)` for univariate data
- shape `(T, D)` for multivariate telemetry or target-plus-exogenous features

By default, the proposed univariate residual-based methods use `target_index=0`.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SMAP / MSL data

```bash
python download_telemanom.py --output-dir data/telemanom
```

### 3. Run the proposed method

```bash
python run_experiment.py \
  --data-dir data/telemanom \
  --spacecraft both \
  --methods proposed_dirichlet \
  --output-dir outputs/proposed \
  --alpha 50 \
  --lags 1,2,4,8 \
  --kappa 0.5 \
  --norm-window 256 \
  --threshold-mode alert_budget_under \
  --alert-budget 0.005 \
  --persistence 2
```

### 4. Run a larger benchmark

```bash
python benchmark_all.py \
  --data-dir data/telemanom \
  --output-dir outputs/full_benchmark \
  --spacecraft both \
  --device cpu \
  --window-size 64 \
  --epochs 10
```

---

## Important scripts

### `download_telemanom.py`
Downloads and prepares the public Telemanom-format SMAP/MSL dataset.

### `run_experiment.py`
Main benchmark script.

Outputs:
- `aggregate_metrics.csv`
- one subfolder per method
- per-channel scores and predictions
- per-channel metrics
- runtime summary

### `benchmark_all.py`
Runs a broader method suite in one command.

### `plot_channel.py`
Plots one channel with signal, score, labels, and predictions.

### `tests/smoke_test.py`
Small synthetic test to verify the project end-to-end.

---

## Evaluation metrics

### Pointwise metrics
- Precision
- Recall
- F1

### Event / segment metrics
- Segment precision
- Segment recall
- Segment F1
- EarlyHit@10%
- Median delay
- False alarm segments

The benchmark saves both **per-channel** results and **macro / micro aggregate** summaries.

---

## Thresholding options

Available threshold modes:

- `fixed_quantile`
- `alert_budget_under`
- `alert_budget_closest`

Thresholds are always estimated from the **training score prefix only**:

```bash
--train-fraction 0.30
```

This matches a strict train-only calibration setting.

---

## Post-processing for false-positive control

Two simple controls are included:

- `--persistence K`
- `--refractory R`

Example:

```bash
--persistence 2 --refractory 10
```

This is useful when early detection is good but pointwise precision is weak.

---

## Recommended baseline package for your paper

If you want a stronger but still manageable comparison set, start with:

```text
proposed_dirichlet
periodic_fft
moving_average
ewma
spectral_residual
pca
isolation_forest
autoencoder
lstm_forecast
```

This gives you:
- a direct boundary-condition comparison
- classical signal baselines
- unsupervised ML baselines
- neural forecasting / reconstruction baselines

---

## Example commands

### Proposed method vs periodic baseline only

```bash
python run_experiment.py \
  --data-dir data/telemanom \
  --spacecraft both \
  --methods proposed_dirichlet periodic_fft \
  --output-dir outputs/dirichlet_vs_periodic \
  --alpha 50 \
  --lags 1,2,4,8 \
  --threshold-mode alert_budget_under \
  --alert-budget 0.005 \
  --persistence 2
```

### Classical baseline pack

```bash
python run_experiment.py \
  --data-dir data/telemanom \
  --spacecraft both \
  --methods moving_average ewma periodic_fft spectral_residual pca isolation_forest \
  --output-dir outputs/classical_pack \
  --window-size 64
```

### Neural baseline pack

```bash
python run_experiment.py \
  --data-dir data/telemanom \
  --spacecraft both \
  --methods autoencoder lstm_forecast \
  --output-dir outputs/neural_pack \
  --window-size 64 \
  --epochs 12 \
  --batch-size 128 \
  --device cpu
```

---

## Suggested ablations to run for the paper

### 1. Boundary-model ablation
```text
proposed_dirichlet vs periodic_fft vs moving_average
```

### 2. Score ablation
Try:
- `--lags 1`
- `--lags 1,2,4,8`

### 3. Thresholding ablation
Try:
- `fixed_quantile`
- `alert_budget_under`
- `alert_budget_closest`

### 4. False-positive control ablation
Try:
- `--persistence 1`
- `--persistence 2`
- `--persistence 3`
- with and without `--refractory`

### 5. Sensitivity
Sweep:
- `--alpha`
- `--kappa`
- `--window-size`
- `--alert-budget`

---

## Output structure

```text
outputs/
  aggregate_metrics.csv
  proposed_dirichlet/
    channel_metrics.csv
    runtime.csv
    A-1_scores.csv
    ...
  periodic_fft/
    channel_metrics.csv
    runtime.csv
    A-1_scores.csv
    ...
```

---

## Project structure

```text
.
├── README.md
├── requirements.txt
├── download_telemanom.py
├── run_experiment.py
├── benchmark_all.py
├── plot_channel.py
├── src/
│   └── dirichlet_tsad/
│       ├── data.py
│       ├── evaluation.py
│       ├── thresholding.py
│       ├── utils.py
│       └── models/
│           ├── __init__.py
│           ├── base.py
│           ├── dirichlet.py
│           ├── smoothing.py
│           ├── spectral_residual.py
│           └── window_baselines.py
└── tests/
    └── smoke_test.py
```

---

## Notes

1. This package is designed to be **clear, editable, and paper-friendly**, not ultra-optimized.
2. For the public SMAP/MSL benchmark, some baselines in the literature use different preprocessing, thresholding, and smoothing. For a fair comparison in your paper, run all methods under the same evaluation protocol and clearly report any exceptions.
3. If you want to compare against official external repos such as Telemanom, TranAD, OmniAnomaly, or MTAD-GAT, use this package's evaluation outputs as the common final metric layer.

---

## Smoke test

```bash
python tests/smoke_test.py
```

If the smoke test passes, the basic project wiring is working.


## Upgraded proposed_dirichlet

The default `proposed_dirichlet` detector now uses:
- two-scale Dirichlet backgrounds (`alpha` and `alpha_fast_ratio * alpha`)
- signed onset scoring from antisymmetric residual differences
- sustain confirmation from causal residual energy
- shape coherence weighting
- hysteresis thresholding and cleaner postprocessing

The benchmark also reports extra metrics including ROC-AUC, PR-AUC, EarlyHit@5/10/20, mean delay, P90 delay, false-alarm segments per 1k points, and mean predicted segment length.
