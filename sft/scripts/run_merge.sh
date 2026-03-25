#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SFT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/merge_lora_qwen25vl3b.yaml}"

echo "[SFT-MERGE] config: ${CONFIG_PATH}"
python scripts/merge_lora.py --config "${CONFIG_PATH}"
