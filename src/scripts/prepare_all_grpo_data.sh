#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PREPARE_VIDEO_R1="${PREPARE_VIDEO_R1:-true}"
PREPARE_UVB="${PREPARE_UVB:-true}"
PREPARE_VIDEOMMMU="${PREPARE_VIDEOMMMU:-true}"
PREPARE_MMVU="${PREPARE_MMVU:-true}"

VIDEO_R1_ARGS="${VIDEO_R1_ARGS:-}"
UVB_ARGS="${UVB_ARGS:-}"
VIDEOMMMU_ARGS="${VIDEOMMMU_ARGS:-}"
MMVU_ARGS="${MMVU_ARGS:-}"

to_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

run_stage() {
  local title="$1"
  shift
  echo "================================================"
  echo "${title}"
  echo "================================================"
  "$@"
}

PREPARE_VIDEO_R1_LC="$(to_lower "${PREPARE_VIDEO_R1}")"
PREPARE_UVB_LC="$(to_lower "${PREPARE_UVB}")"
PREPARE_VIDEOMMMU_LC="$(to_lower "${PREPARE_VIDEOMMMU}")"
PREPARE_MMVU_LC="$(to_lower "${PREPARE_MMVU}")"

if [[ "${PREPARE_VIDEOMMMU_LC}" == "true" ]]; then
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "[prepare_all_grpo_data] ERROR: VideoMMMU is enabled but HF_TOKEN is not set." >&2
    echo "[prepare_all_grpo_data] Set HF_TOKEN (approved account) or disable with PREPARE_VIDEOMMMU=false." >&2
    exit 1
  fi
fi

if [[ "${PREPARE_VIDEO_R1_LC}" == "true" ]]; then
  if [[ -n "${VIDEO_R1_ARGS}" ]]; then
    run_stage "Prepare GRPO Train Set: Video-R1" bash src/scripts/prepare_video_r1_grpo_data.sh ${VIDEO_R1_ARGS}
  else
    run_stage "Prepare GRPO Train Set: Video-R1" bash src/scripts/prepare_video_r1_grpo_data.sh
  fi
fi

if [[ "${PREPARE_UVB_LC}" == "true" ]]; then
  if [[ -n "${UVB_ARGS}" ]]; then
    run_stage "Prepare GRPO Test Set 1: Urban Video Bench" bash src/scripts/prepare_uvb_grpo_data.sh ${UVB_ARGS}
  else
    run_stage "Prepare GRPO Test Set 1: Urban Video Bench" bash src/scripts/prepare_uvb_grpo_data.sh
  fi
fi

if [[ "${PREPARE_VIDEOMMMU_LC}" == "true" ]]; then
  if [[ -n "${VIDEOMMMU_ARGS}" ]]; then
    run_stage "Prepare GRPO Test Set 2: VideoMMMU" bash src/scripts/prepare_videommmu_grpo_data.sh ${VIDEOMMMU_ARGS}
  else
    run_stage "Prepare GRPO Test Set 2: VideoMMMU" bash src/scripts/prepare_videommmu_grpo_data.sh
  fi
fi

if [[ "${PREPARE_MMVU_LC}" == "true" ]]; then
  if [[ -n "${MMVU_ARGS}" ]]; then
    run_stage "Prepare GRPO Test Set 3: MMVU (multiple-choice only)" bash src/scripts/prepare_mmvu_grpo_data.sh ${MMVU_ARGS}
  else
    run_stage "Prepare GRPO Test Set 3: MMVU (multiple-choice only)" bash src/scripts/prepare_mmvu_grpo_data.sh
  fi
fi

echo "================================================"
echo "All requested GRPO datasets are prepared."
echo "================================================"
echo "Train:"
echo "  data/video_r1/grpo/video_r1_grpo_train.jsonl"
echo "Test 1:"
echo "  data/urban_video_bench/grpo/uvb_grpo_test.jsonl"
echo "Test 2:"
echo "  data/video_mmmu/grpo/videommmu_grpo_test.jsonl"
echo "Test 3:"
echo "  data/mmvu/grpo/mmvu_grpo_test.jsonl"
