#!/bin/bash

set -euo pipefail

python src/eval/prepare_mmvu.py \
  --dataset-dir "data/mmvu/raw" \
  --processed-dir "data/mmvu/processed" \
  --grpo-output-dir "data/mmvu/grpo" \
  "$@"
