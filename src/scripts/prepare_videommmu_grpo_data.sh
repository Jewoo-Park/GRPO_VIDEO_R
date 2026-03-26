#!/bin/bash

set -euo pipefail

python src/eval/prepare_videommmu.py \
  --dataset-dir "data/video_mmmu/raw" \
  --processed-dir "data/video_mmmu/processed" \
  --grpo-output-dir "data/video_mmmu/grpo" \
  "$@"
