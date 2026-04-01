# GRPO UVB 실행 가이드

---

## 데이터 & SFT merged 모델 이미 있을 때 (지금 상황)

SFT LoRA를 베이스 모델에 merge한 모델(`sft/outputs/qwen25vl3b_lora_merged`)과 데이터(`data/video_r1/grpo/video_r1_grpo_train.jsonl`, `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`)가 준비되어 있다면, 아래만 하면 됨.

### 실행

```bash
cd /path/to/GRPO_Video/src/scripts

# merged SFT 모델 경로 지정
export QWEN_PATH="/path/to/GRPO_Video/sft/outputs/qwen25vl3b_lora_merged"

# Optional stability knobs (recommended on cloud GPU instances)
export TORCH_DTYPE="bfloat16"
export NCCL_SAFE_MODE="true"

# (선택) flash-attn rotary dtype assertion이 뜨는 환경이면 1회 적용
bash apply_rotary_dtype_hotfix.sh

# train/test 기본값이 ../../data/urban_video_bench/grpo/... 이므로
# 데이터가 프로젝트 루트의 data/urban_video_bench/grpo/ 아래에 있으면 생략 가능
./run_grpo_uvb_answer_only.sh
```

### GPU 4장 (A40) 쓰는 경우

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export QWEN_PATH="/path/to/GRPO_Video/sft/outputs/qwen25vl3b_lora_merged"
./run_grpo_uvb_answer_only.sh
```

### Reward 가중치 조정해서 실행

정답/포맷 보상 비중을 조정하려면 아래 환경변수를 사용:

- `REWARD_WEIGHTS`: `reward_funcs` 순서에 맞는 콤마 구분 가중치 (`answer_accuracy,answer_format`)
- `ANSWER_ACCURACY_WEIGHT`, `ANSWER_FORMAT_WEIGHT`: 개별 override (설정 시 `REWARD_WEIGHTS`보다 우선)

```bash
# 예시 1) 기본 권장 가중치 (accuracy 0.9 / format 0.1)
REWARD_WEIGHTS="0.9,0.1" ./run_grpo_uvb_answer_only.sh

# 예시 2) 개별 override
ANSWER_ACCURACY_WEIGHT="0.9" \
ANSWER_FORMAT_WEIGHT="0.1" \
./run_grpo_uvb_answer_only.sh
```

- 스크립트가 `src/r1-v`로 cd한 뒤 실행되므로, **데이터 경로**가 `REPO_ROOT/data/urban_video_bench/grpo/` 가 아니면 `TRAIN_FILE` / `TEST_FILE` 을 직접 지정.

### 결과

- **모델/체크포인트**: `src/r1-v/outputs/uvb_grpo_answer_only/`
- **테스트 추론**: 같은 폴더의 `test_predictions.jsonl` (학습 끝나면 자동 생성)

---

## SFT 이후 GRPO 실행

현재 운영 기준에서는 SFT와 merge를 먼저 끝낸 뒤, GRPO는 별도 스크립트로 실행한다.

대표 진입점:

```bash
cd /path/to/GRPO_Video
bash src/scripts/run_grpo_uvb_answer_only_lora.sh
```

---

## 자주 쓰는 환경 변수

| 변수 | 의미 | 예시 |
|------|------|------|
| `QWEN_PATH` | merged SFT 모델 경로 | `sft/outputs/qwen25vl3b_lora_merged` |
| `TRAIN_FILE` | 학습용 JSONL | `../../data/video_r1/grpo/video_r1_grpo_train.jsonl` |
| `TEST_FILE` | 테스트용 JSONL (비우면 테스트 추론 안 함) | `../../data/urban_video_bench/grpo/uvb_grpo_test.jsonl` |
| `OUTPUT_DIR` | 체크포인트/모델/테스트 결과 디렉터리 | `./outputs/uvb_grpo_answer_only` |
| `NUM_TRAIN_EPOCHS` | 에폭 수 | `3` |
| `NUM_GENERATIONS` | 프롬프트당 생성 개수 (G) | `8` |
| `VLLM_MAX_FRAMES` | 프레임 수 상한 (vLLM) | `8` |
| `TORCH_DTYPE` | 모델 로드 dtype (`bfloat16`, `float16`, `float32`) | `bfloat16` |
| `NCCL_SAFE_MODE` | NCCL 안정화 env 자동 설정 여부 | `true` |
| `REWARD_WEIGHTS` | 리워드 가중치 (`answer_accuracy,answer_format`) | `0.9,0.1` |
| `ANSWER_ACCURACY_WEIGHT` | 정답 보상 개별 가중치 override | `0.9` |
| `ANSWER_FORMAT_WEIGHT` | 포맷 보상 개별 가중치 override | `0.1` |

---

## 참고: 데이터 파이프라인 처음부터 할 때

데이터가 없을 때만:
`prepare_video_r1_grpo.py` 또는 `prepare_video_r1_grpo_data.sh`로 `data/video_r1/grpo/video_r1_grpo_train.jsonl` 생성.
`prepare_uvb_pipeline.py` 또는 `prepare_uvb_grpo_data.sh`로 `data/urban_video_bench/grpo/uvb_grpo_test.jsonl` 생성.
