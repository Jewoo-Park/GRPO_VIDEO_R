#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SFT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/train_lora_qwen25vl3b.yaml}"
MASTER_PORT="${MASTER_PORT:-12355}"
MERGE_AFTER_TRAIN="${MERGE_AFTER_TRAIN:-true}"
USE_VISION="${USE_VISION:-}"

if [[ -z "${USE_VISION}" ]]; then
  read -r -p "[SFT] 이미지/프레임 입력을 사용할까요? [y/N]: " USE_VISION_REPLY
  if [[ "${USE_VISION_REPLY,,}" =~ ^(y|yes)$ ]]; then
    USE_VISION="true"
  else
    USE_VISION="false"
  fi
fi

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

echo "[SFT-PIPELINE] config: ${CONFIG_PATH}"
echo "[SFT-PIPELINE] USE_VISION=${USE_VISION}"
echo "[SFT-PIPELINE] NUM_GPUS=${NUM_GPUS}"
echo "[SFT-PIPELINE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    scripts/train_sft.py --config "${CONFIG_PATH}" --use-vision "${USE_VISION}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/train_sft.py --config "${CONFIG_PATH}" --use-vision "${USE_VISION}"
fi

if [[ "${MERGE_AFTER_TRAIN,,}" != "true" ]]; then
  exit 0
fi

readarray -t MERGE_VALUES < <(
  python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_name = cfg["model_name_or_path"]
adapter_dir = cfg["output_dir"]
export_dir = cfg.get("merge_output_dir") or f"{adapter_dir}_merged"

print(model_name)
print(adapter_dir)
print(export_dir)
PY
)

MODEL_NAME_OR_PATH="${MERGE_VALUES[0]}"
ADAPTER_NAME_OR_PATH="${MERGE_VALUES[1]}"
EXPORT_DIR="${MERGE_VALUES[2]}"

echo "[SFT-PIPELINE] merging adapter into ${EXPORT_DIR}"
python scripts/merge_lora.py \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --adapter-name-or-path "${ADAPTER_NAME_OR_PATH}" \
  --export-dir "${EXPORT_DIR}" \
  --remap-adapter-keys true
