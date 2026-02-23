#!/bin/bash

set -euo pipefail

python src/eval/uvb_to_grpo.py \
  --processed-dir "data/urban_video_bench/processed" \
  --output-dir "data/urban_video_bench/grpo"
