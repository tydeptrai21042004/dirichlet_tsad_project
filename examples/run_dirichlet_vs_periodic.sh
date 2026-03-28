#!/usr/bin/env bash
set -euo pipefail

python run_experiment.py \
  --data-dir data/telemanom \
  --spacecraft both \
  --methods proposed_dirichlet periodic_fft \
  --output-dir outputs/dirichlet_vs_periodic \
  --alpha 50 \
  --lags 1,2,4,8 \
  --kappa 0.5 \
  --norm-window 256 \
  --threshold-mode alert_budget_under \
  --alert-budget 0.005 \
  --persistence 2
