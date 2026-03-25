#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/src/r1-v"

export DEBUG_MODE="true"
export LOG_PATH="./logs/video_r1_uvb_grpo_answer_only.log"
mkdir -p ./logs

QWEN_PATH="${QWEN_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-../../data/video_r1/grpo/video_r1_grpo_train.jsonl}"
TEST_FILE="${TEST_FILE:-../../data/urban_video_bench/grpo/uvb_grpo_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/video_r1_uvb_grpo_answer_only}"
RUN_NAME="${RUN_NAME:-video_r1_uvb_grpo_answer_only_$(date +%Y%m%d_%H%M%S)}"
DS_CONFIG="${DS_CONFIG:-./configs/zero1_no_optimizer.json}"
MASTER_PORT="${MASTER_PORT:-12346}"
REPORT_TO="${REPORT_TO:-none}"
DATASET_NAME="${DATASET_NAME:-video_r1_train_uvb_eval}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.4}"
VLLM_DEVICE="${VLLM_DEVICE:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NCCL_SAFE_MODE="${NCCL_SAFE_MODE:-false}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_PIXELS="${MAX_PIXELS:-501760}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
USE_PEFT="${USE_PEFT:-true}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-}"
ANSWER_ACCURACY_WEIGHT="${ANSWER_ACCURACY_WEIGHT:-0.9}"
ANSWER_FORMAT_WEIGHT="${ANSWER_FORMAT_WEIGHT:-0.1}"
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

# When using vLLM, the trainer expects vLLM on a dedicated GPU (cuda:${TRAIN_NUM_GPUS}).
# So we use TRAIN_NUM_GPUS for torchrun and reserve 1 GPU for vLLM when NUM_GPUS > 1.
# Example: 6 A40s → 5 for training (cuda:0..4), 1 for vLLM (cuda:5).
# Override by setting TRAIN_NUM_GPUS explicitly (e.g. TRAIN_NUM_GPUS=6 to use all 6 for training; then vLLM shares cuda:0).
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

NCCL_SAFE_MODE_LC="$(printf '%s' "${NCCL_SAFE_MODE}" | tr '[:upper:]' '[:lower:]')"
USE_PEFT_LC="$(printf '%s' "${USE_PEFT}" | tr '[:upper:]' '[:lower:]')"
LOAD_IN_8BIT_LC="$(printf '%s' "${LOAD_IN_8BIT}" | tr '[:upper:]' '[:lower:]')"

if [[ "${NCCL_SAFE_MODE_LC}" == "true" ]]; then
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
fi

echo "[VIDEO-GRPO] QWEN_PATH=${QWEN_PATH}"
echo "[VIDEO-GRPO] TRAIN_FILE=${TRAIN_FILE}"
echo "[VIDEO-GRPO] TEST_FILE=${TEST_FILE}"
echo "[VIDEO-GRPO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[VIDEO-GRPO] NUM_GPUS=${NUM_GPUS} (train processes: ${TRAIN_NUM_GPUS}, 1 GPU reserved for vLLM when NUM_GPUS>1)"
echo "[VIDEO-GRPO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[VIDEO-GRPO] VLLM_GPU_UTIL=${VLLM_GPU_UTIL}"
echo "[VIDEO-GRPO] TORCH_DTYPE=${TORCH_DTYPE}"
echo "[VIDEO-GRPO] NCCL_SAFE_MODE=${NCCL_SAFE_MODE}"
echo "[VIDEO-GRPO] NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-<unset>}"
echo "[VIDEO-GRPO] NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-<unset>}"
echo "[VIDEO-GRPO] TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-<unset>}"
echo "[VIDEO-GRPO] MAX_PIXELS=${MAX_PIXELS} (frames are kept as-is from dataset)"
echo "[VIDEO-GRPO] NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "[VIDEO-GRPO] USE_PEFT=${USE_PEFT}"
echo "[VIDEO-GRPO] LOAD_IN_8BIT=${LOAD_IN_8BIT}"
echo "[VIDEO-GRPO] REWARD_WEIGHTS=${REWARD_WEIGHTS:-<default>}"
echo "[VIDEO-GRPO] ANSWER_ACCURACY_WEIGHT=${ANSWER_ACCURACY_WEIGHT:-<unset>}"
echo "[VIDEO-GRPO] ANSWER_FORMAT_WEIGHT=${ANSWER_FORMAT_WEIGHT:-<unset>}"

LORA_ARGS=()
if [[ "${USE_PEFT_LC}" == "true" ]]; then
  LORA_ARGS=(
    --use_peft true
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --lora_target_modules q_proj k_proj v_proj o_proj
  )
  echo "[VIDEO-GRPO] LORA_R=${LORA_R}"
  echo "[VIDEO-GRPO] LORA_ALPHA=${LORA_ALPHA}"
  echo "[VIDEO-GRPO] LORA_DROPOUT=${LORA_DROPOUT}"
else
  echo "[VIDEO-GRPO] Running full-parameter fine-tuning (USE_PEFT=false)"
fi

REWARD_ARGS=()
if [[ -n "${REWARD_WEIGHTS}" ]]; then
  REWARD_ARGS+=(--reward_weights "${REWARD_WEIGHTS}")
fi
if [[ -n "${ANSWER_ACCURACY_WEIGHT}" ]]; then
  REWARD_ARGS+=(--answer_accuracy_weight "${ANSWER_ACCURACY_WEIGHT}")
fi
if [[ -n "${ANSWER_FORMAT_WEIGHT}" ]]; then
  REWARD_ARGS+=(--answer_format_weight "${ANSWER_FORMAT_WEIGHT}")
fi

MODEL_QUANT_ARGS=()
if [[ "${LOAD_IN_8BIT_LC}" == "true" ]]; then
  MODEL_QUANT_ARGS+=(--load_in_8bit true)
fi

PYTHONPATH="./src" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="${TRAIN_NUM_GPUS}" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="127.0.0.1" \
  --master_port="${MASTER_PORT}" \
  -m open_r1.grpo_video \
  --use_vllm ${USE_VLLM:-true} \
  --dataset_name "${DATASET_NAME}" \
  --vllm_device "${VLLM_DEVICE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${QWEN_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --test_file "${TEST_FILE}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 1e-6 \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --bf16 true \
  --torch_dtype "${TORCH_DTYPE}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --min_pixels "${MIN_PIXELS}" \
  --max_pixels "${MAX_PIXELS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --run_name "${RUN_NAME}" \
  --save_steps 50 \
  --save_total_limit 2 \
  --save_only_model true \
  --report_to "${REPORT_TO}" \
  --temperature 1.0 \
  --num_generations "${NUM_GENERATIONS}" \
  --deepspeed "${DS_CONFIG}" \
  "${MODEL_QUANT_ARGS[@]}" \
  "${REWARD_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
