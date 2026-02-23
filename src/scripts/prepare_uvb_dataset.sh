#!/bin/bash

set -euo pipefail

echo "================================================"
echo "Preparing UrbanVideo-Bench metadata"
echo "================================================"

python src/eval/prepare_urban_video_bench.py \
  --dataset-id "EmbodiedCity/UrbanVideo-Bench" \
  --split "train" \
  --output "data/urban_video_bench/urban_video_bench_train.jsonl"

echo "================================================"
echo "Done"
echo "Output: data/urban_video_bench/urban_video_bench_train.jsonl"
echo "================================================"
