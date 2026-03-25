#!/bin/bash

set -euo pipefail

python src/eval/prepare_video_r1_grpo.py \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  "$@"
