#!/usr/bin/env bash
set -euo pipefail

python benchmark_all.py \
  --data-dir data/telemanom \
  --output-dir outputs/full_benchmark \
  --spacecraft both \
  --device cpu \
  --window-size 64 \
  --epochs 10
