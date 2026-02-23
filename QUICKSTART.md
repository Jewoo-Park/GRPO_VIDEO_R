# Quick Start: Urban Video Bench + Qwen2.5-VL GRPO

이 문서는 이 레포에서 **Urban Video Bench(UVB)** 데이터셋으로 **Qwen2.5-VL** 모델을 **GRPO** 방식으로 학습/추론하는 최소 실행 절차입니다.

**주의**: 아래 모든 명령은 **레포 루트**(TON_Codex 디렉터리)에서 실행하세요. 문제 발생 시 `TROUBLESHOOTING.md`를 참고하세요.

## One-click Training

데이터가 이미 준비되어 있으면, 아래 한 줄로 바로 학습 시작:

```bash
bash src/scripts/run_grpo_uvb_answer_only.sh
```

복구(경로 재생성/리사이즈) 포함 원클릭:

```bash
bash src/scripts/uvb_recover_and_train_a100.sh
```

## 0. Environment Setup

```bash
conda create -n r1-v python=3.11 -y
conda activate r1-v
bash setup.sh
```

선택: 환경 점검

```bash
bash src/scripts/check_environment.sh
```

## 1. UVB 메타데이터 준비

```bash
bash src/scripts/prepare_uvb_dataset.sh
```

생성 파일:
- `data/urban_video_bench/urban_video_bench_train.jsonl`

## 2. 40% 샘플링 + 8:2 분할 + 비디오 다운로드 + 프레임 추출

```bash
bash src/scripts/prepare_uvb_40_split_download_frames.sh
```

생성 디렉토리/파일:
- `data/urban_video_bench/processed/train_80.jsonl`
- `data/urban_video_bench/processed/test_20.jsonl`
- `data/urban_video_bench/processed/frames/`
- `data/urban_video_bench/processed/pipeline_summary.json`

## 3. GRPO 학습용 JSONL 변환

```bash
bash src/scripts/prepare_uvb_grpo_data.sh
```

생성 파일:
- `data/urban_video_bench/grpo/uvb_grpo_train.jsonl`
- `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`
- `data/urban_video_bench/grpo/uvb_grpo_summary.json`

## 4. GRPO 학습 실행 (Answer-only reward)

```bash
bash src/scripts/run_grpo_uvb_answer_only.sh
```

기본 설정(스크립트 내부):
- 모델: `Qwen/Qwen2.5-VL-3B-Instruct`
- 학습 파일: `data/urban_video_bench/grpo/uvb_grpo_train.jsonl`
- 검증 파일: `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`
- 출력: `src/r1-v/outputs/uvb_grpo_answer_only`
- 학습 방식: **LoRA 기본값 활성화** (`USE_PEFT=true`)
- 기본 메모리 설정: `NUM_GENERATIONS=8`, `MAX_PIXELS=131072`, `VLLM_GPU_UTIL=0.35`, `NUM_TRAIN_EPOCHS=2`

자주 바꾸는 옵션(환경변수):

```bash
QWEN_PATH=/path/to/your/model \
OUTPUT_DIR=./src/r1-v/outputs/uvb_custom \
REPORT_TO=none \
bash src/scripts/run_grpo_uvb_answer_only.sh
```

전체 파라미터 학습(LoRA 비활성화)로 바꾸려면:

```bash
USE_PEFT=false bash src/scripts/run_grpo_uvb_answer_only.sh
```

OOM 시 권장 조정:
- `NUM_GENERATIONS` 감소 (예: 8 -> 4)
- `--max_completion_length` 감소
- `MAX_PIXELS` 감소 (예: 131072 -> 65536)
- `NUM_GPUS`/`CUDA_VISIBLE_DEVICES` 조정

### 4-1. LoRA 방식으로 GRPO 학습 실행

LoRA 전용 실행 스크립트:

```bash
bash src/scripts/run_grpo_uvb_answer_only_lora.sh
```

자주 바꾸는 LoRA 옵션(환경변수):

```bash
LORA_R=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
REPORT_TO=none \
bash src/scripts/run_grpo_uvb_answer_only_lora.sh
```

참고:
- 이 레포의 `open_r1.grpo_uvb` 엔트리는 `get_peft_config(model_args)`를 통해 `--use_peft true` 시 LoRA를 활성화합니다.
- 기본 target module은 스크립트에 `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`로 지정되어 있습니다.

## 5. 학습 모델 평가 (vLLM)

```bash
python src/eval/uvb_eval_only.py \
  --model /path/to/checkpoint_or_model \
  --test-file data/urban_video_bench/grpo/uvb_grpo_test.jsonl \
  --device cuda:0 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 3136 \
  --max-completion-length 64 \
  --frames-per-sample 8 \
  --save-preds outputs/uvb_eval_preds.jsonl
```

출력 메트릭:
- `answer_accuracy`
- `answer_format_rate`

## One-command (복구 + 학습)

A100 환경에서 데이터 복구 + 학습을 한 번에 돌릴 때:

```bash
bash src/scripts/uvb_recover_and_train_a100.sh
```

## 빠른 확인 체크리스트

```bash
wc -l data/urban_video_bench/grpo/uvb_grpo_train.jsonl
wc -l data/urban_video_bench/grpo/uvb_grpo_test.jsonl
ls src/r1-v/outputs/uvb_grpo_answer_only
```

## 주의

실제 UVB GRPO 엔트리 모듈은 아래 경로입니다:
- `src/r1-v/src/open_r1/grpo_uvb.py`
