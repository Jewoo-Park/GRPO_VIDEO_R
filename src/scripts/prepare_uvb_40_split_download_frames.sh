#!/bin/bash

set -euo pipefail

echo "================================================"
echo "UVB 40% Sampling + 8:2 Split + Video Download + Frame Extraction"
echo "================================================"

python src/eval/prepare_uvb_pipeline.py \
  --input-jsonl "data/urban_video_bench/urban_video_bench_train.jsonl" \
  --sample-ratio 0.4 \
  --test-ratio 0.2 \
  --seed 42 \
  --dataset-id "EmbodiedCity/UrbanVideo-Bench" \
  --video-dir "data/urban_video_bench" \
  --output-dir "data/urban_video_bench/processed" \
  --num-frames 32 \
  --max-frame-size 768

echo "================================================"
echo "Done"
echo "Processed files: data/urban_video_bench/processed"
echo "================================================"
