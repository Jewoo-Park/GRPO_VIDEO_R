#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SFT_TRAIN_CONFIG="${SFT_TRAIN_CONFIG:-configs/train_lora_qwen25vl3b.yaml}"
SFT_MERGE_CONFIG="${SFT_MERGE_CONFIG:-configs/merge_lora_qwen25vl3b.yaml}"
DO_SFT="${DO_SFT:-true}"
DO_MERGE="${DO_MERGE:-true}"
DO_GRPO="${DO_GRPO:-true}"

SFT_NUM_GPUS="${SFT_NUM_GPUS:-2}"
SFT_CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-0,1}"
SFT_MASTER_PORT="${SFT_MASTER_PORT:-12355}"

GRPO_NUM_GPUS="${GRPO_NUM_GPUS:-2}"
GRPO_TRAIN_NUM_GPUS="${GRPO_TRAIN_NUM_GPUS:-1}"
GRPO_CUDA_VISIBLE_DEVICES="${GRPO_CUDA_VISIBLE_DEVICES:-0,1}"
GRPO_QWEN_PATH="${GRPO_QWEN_PATH:-}"

resolve_sft_merged_model() {
  local candidates=(
    "${REPO_ROOT}/sft/outputs/qwen25vl3b_lora_merged_from_sft40"
    "${REPO_ROOT}/sft/outputs/qwen25vl3b_lora_merged_from_ck3855_v2"
    "${REPO_ROOT}/sft/outputs/qwen25vl3b_lora_merged_40"
    "${REPO_ROOT}/sft/outputs/qwen25vl3b_lora_merged"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}/config.json" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

if [[ "${DO_SFT,,}" == "true" ]]; then
  echo "[PIPELINE] Stage 1/3: SFT training"
  (
    cd sft
    NUM_GPUS="${SFT_NUM_GPUS}" \
    CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES}" \
    MASTER_PORT="${SFT_MASTER_PORT}" \
    CONFIG_PATH="${SFT_TRAIN_CONFIG}" \
    bash scripts/run_train.sh
  )
fi

if [[ "${DO_MERGE,,}" == "true" ]]; then
  echo "[PIPELINE] Stage 2/3: Merge SFT LoRA"
  (
    cd sft
    CONFIG_PATH="${SFT_MERGE_CONFIG}" bash scripts/run_merge.sh
  )
fi

if [[ "${DO_GRPO,,}" == "true" ]]; then
  echo "[PIPELINE] Stage 3/3: GRPO training"

  if [[ -n "${GRPO_QWEN_PATH}" ]]; then
    QWEN_PATH_FOR_GRPO="${GRPO_QWEN_PATH}"
  elif GRPO_QWEN_PATH="$(resolve_sft_merged_model)"; then
    QWEN_PATH_FOR_GRPO="${GRPO_QWEN_PATH}"
  else
    echo "[PIPELINE] ERROR: merged SFT model not found. Run DO_MERGE=true first." >&2
    exit 1
  fi
  echo "[PIPELINE] GRPO model path: ${QWEN_PATH_FOR_GRPO}"

  NUM_GPUS="${GRPO_NUM_GPUS}" \
  TRAIN_NUM_GPUS="${GRPO_TRAIN_NUM_GPUS}" \
  CUDA_VISIBLE_DEVICES="${GRPO_CUDA_VISIBLE_DEVICES}" \
  QWEN_PATH="${QWEN_PATH_FOR_GRPO}" \
  bash src/scripts/run_grpo_uvb_answer_only.sh
fi

echo "[PIPELINE] done"
