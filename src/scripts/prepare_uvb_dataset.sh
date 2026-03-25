#!/bin/bash

set -euo pipefail

echo "================================================"
echo "Preparing UrbanVideo-Bench benchmark"
echo "================================================"

python src/eval/prepare_uvb_pipeline.py \
  --dataset-id "EmbodiedCity/UrbanVideo-Bench" \
  --video-dir "data/urban_video_bench" \
  --output-dir "data/urban_video_bench/processed" \
  --grpo-output-dir "data/urban_video_bench/grpo"

echo "================================================"
echo "Done"
echo "Output: data/urban_video_bench/processed and data/urban_video_bench/grpo"
echo "================================================"
