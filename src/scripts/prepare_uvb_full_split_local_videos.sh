#!/bin/bash

set -euo pipefail

echo "================================================"
echo "UVB Full Evaluation Set + Local Video Frame Extraction"
echo "================================================"

python src/eval/prepare_uvb_pipeline.py \
  --dataset-id "EmbodiedCity/UrbanVideo-Bench" \
  --video-dir "data/videos" \
  --output-dir "data/urban_video_bench/processed" \
  --grpo-output-dir "data/urban_video_bench/grpo" \
  --num-frames 32 \
  --max-frame-size 768 \
  --skip-download

echo "================================================"
echo "Done"
echo "Processed files: data/urban_video_bench/processed and data/urban_video_bench/grpo"
echo "================================================"
