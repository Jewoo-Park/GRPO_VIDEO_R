#!/bin/bash

set -euo pipefail

python src/eval/prepare_uvb_pipeline.py \
  --video-dir "data/urban_video_bench/raw" \
  --output-dir "data/urban_video_bench/processed" \
  --grpo-output-dir "data/urban_video_bench/grpo"
