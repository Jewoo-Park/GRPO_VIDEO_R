#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SFT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/train_lora_qwen25vl3b.yaml}"
MASTER_PORT="${MASTER_PORT:-12355}"

if [[ -n "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="${NUM_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
else
  NUM_GPUS="1"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

echo "[SFT] config: ${CONFIG_PATH}"
echo "[SFT] NUM_GPUS=${NUM_GPUS}"
echo "[SFT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    scripts/train_sft.py --config "${CONFIG_PATH}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/train_sft.py --config "${CONFIG_PATH}"
fi
