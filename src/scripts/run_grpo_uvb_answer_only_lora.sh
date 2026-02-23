#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/src/r1-v"

export DEBUG_MODE="true"
export LOG_PATH="./logs/uvb_grpo_answer_only_lora.log"
mkdir -p ./logs

QWEN_PATH="${QWEN_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-../../data/urban_video_bench/grpo/uvb_grpo_train.jsonl}"
TEST_FILE="${TEST_FILE:-../../data/urban_video_bench/grpo/uvb_grpo_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/uvb_grpo_answer_only_lora}"
RUN_NAME="${RUN_NAME:-uvb_grpo_answer_only_lora_$(date +%Y%m%d_%H%M%S)}"
DS_CONFIG="${DS_CONFIG:-./configs/zero1_no_optimizer.json}"
MASTER_PORT="${MASTER_PORT:-12347}"
REPORT_TO="${REPORT_TO:-wandb}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="${NUM_GPUS}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  else
    NUM_GPUS="1"
  fi
fi

# Reserve 1 GPU for vLLM when NUM_GPUS > 1 (same as run_grpo_uvb_answer_only.sh).
if [[ -n "${TRAIN_NUM_GPUS:-}" ]]; then
  TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS}"
elif [[ "${NUM_GPUS}" -gt 1 ]]; then
  TRAIN_NUM_GPUS="$((NUM_GPUS - 1))"
else
  TRAIN_NUM_GPUS="${NUM_GPUS}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

echo "[UVB-GRPO-LORA] QWEN_PATH=${QWEN_PATH}"
echo "[UVB-GRPO-LORA] TRAIN_FILE=${TRAIN_FILE}"
echo "[UVB-GRPO-LORA] TEST_FILE=${TEST_FILE}"
echo "[UVB-GRPO-LORA] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[UVB-GRPO-LORA] NUM_GPUS=${NUM_GPUS} (train processes: ${TRAIN_NUM_GPUS})"
echo "[UVB-GRPO-LORA] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[UVB-GRPO-LORA] LORA_R=${LORA_R}"
echo "[UVB-GRPO-LORA] LORA_ALPHA=${LORA_ALPHA}"
echo "[UVB-GRPO-LORA] LORA_DROPOUT=${LORA_DROPOUT}"

PYTHONPATH="./src" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${TRAIN_NUM_GPUS}" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="127.0.0.1" \
  --master_port="${MASTER_PORT}" \
  -m open_r1.grpo_uvb \
  --use_vllm true \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${QWEN_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --test_file "${TEST_FILE}" \
  --max_prompt_length 8192 \
  --max_completion_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-6 \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --min_pixels 3136 \
  --max_pixels 501760 \
  --num_train_epochs 1 \
  --run_name "${RUN_NAME}" \
  --save_steps 50 \
  --save_total_limit 2 \
  --save_only_model true \
  --report_to "${REPORT_TO}" \
  --temperature 1.0 \
  --num_generations 4 \
  --deepspeed "${DS_CONFIG}" \
  --use_peft true \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
