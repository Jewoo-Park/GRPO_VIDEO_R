#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv_realign}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_TORCH_CU124="${INSTALL_TORCH_CU124:-false}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-true}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d "${VENV_PATH}" ]]; then
    echo "[setup] Creating venv: ${VENV_PATH}"
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

echo "[setup] Using python: $(which python)"
python -m pip install --upgrade pip setuptools wheel

INSTALL_TORCH_CU124_LC="$(printf '%s' "${INSTALL_TORCH_CU124}" | tr '[:upper:]' '[:lower:]')"
INSTALL_FLASH_ATTN_LC="$(printf '%s' "${INSTALL_FLASH_ATTN}" | tr '[:upper:]' '[:lower:]')"

if [[ "${INSTALL_TORCH_CU124_LC}" == "true" ]]; then
  echo "[setup] Installing torch/cu124 pinned stack"
  python -m pip install \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
else
  echo "[setup] Skipping explicit torch/cu124 install (set INSTALL_TORCH_CU124=true to enable)"
fi

cd "${REPO_ROOT}/src/r1-v"

# Core project package (runtime only; skip heavy dev extras)
python -m pip install -e "."

# Runtime dependencies aligned with UVB GRPO runs
python -m pip install \
  "trl==0.14.0" \
  "peft==0.14.0" \
  "accelerate" \
  "deepspeed==0.15.4" \
  "datasets" \
  "yt-dlp" \
  "vllm==0.7.2" \
  "wandb>=0.19.1" \
  "tensorboardx" \
  "qwen_vl_utils" \
  "torchvision"

# Keep transformers at a known-good revision used by this repo.
python -m pip install "git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef"

if [[ "${INSTALL_FLASH_ATTN_LC}" == "true" ]]; then
  # flash-attn build can fail on unsupported environments (e.g., non-CUDA/macOS).
  # Continue setup and let runtime fallback to xformers/SDPA if unavailable.
  python -m pip install "flash-attn==2.6.3" --no-build-isolation || true
fi

echo "[setup] Done. Active venv: ${VIRTUAL_ENV:-<none>}"
