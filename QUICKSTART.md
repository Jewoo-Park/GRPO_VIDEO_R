# GRPO_VIDEO_R Quick Start

이 문서는 현재 레포의 기준 파이프라인을 가장 짧게 따라갈 수 있도록 정리한 실행 가이드다.

현재 기본 흐름은 아래와 같다.

1. 데이터 로드
2. SFT 훈련
3. SFT 어댑터 결합
4. Video-R1을 통한 GRPO 훈련
5. GRPO 어댑터 결합
6. UVB Benchmark를 통한 검증

기준 구조는 다음이다.

- Train set: `Video-R1`
- Test benchmark: `Urban Video Bench`
- Additional benchmark 1: `VideoMMMU`
- Additional benchmark 2: `MMVU (multiple-choice only)`
- SFT base model: `Qwen/Qwen2.5-VL-3B-Instruct`

주의:

- 실제 SFT/GRPO 훈련은 CUDA GPU가 있는 Linux 환경 기준이다.
- macOS에서는 데이터 전처리 정도는 가능하지만, 실제 GRPO 학습과 vLLM 기반 평가는 권장하지 않는다.

---

## 0. 환경 준비

레포 루트에서:

```bash
bash setup.sh
```

환경 점검:

```bash
bash src/scripts/check_environment.sh
```

중요 확인 항목:

- `torch`
- `transformers`
- `peft`
- `trl`
- `datasets`
- `vllm`
- `deepspeed`

---

## 1. 데이터 로드

### 1-1. Train set: Video-R1

기본 train set은 아래 5개 subset을 사용한다.

- `LLaVA-Video-178K`
- `NeXT-QA`
- `PerceptionTest`
- `CLEVRER`
- `STAR`

전체 train set 준비:

```bash
python src/eval/prepare_video_r1_grpo.py \
  --dataset-id "Video-R1/Video-R1-data" \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  --subsets "PerceptionTest" \
  --num-frames 16
  --sample-ratio 1 \
  --download-mode subset-directories

```
여기서, 일부 샘플만 받으려고 할 때는 subsets 항목에 추가하면 된다.
위 상태는 가장 용량이 큰 LLaVA 데이터셋을 제외한 상태이다. 
다 쓰려면 그냥 지우면 된다.
현재는 프레임 개수가 16개로 지정되어 있으나, 32개로 설정할 수 있다.
  --subsets "NeXT-QA,PerceptionTest,CLEVRER,STAR" \

최종 train 입력 파일:

```text
data/video_r1/grpo/video_r1_grpo_train.jsonl
```

### 1-2. Test set: UVB

UVB benchmark 준비:

```bash
bash src/scripts/prepare_uvb_grpo_data.sh
```

최종 test 입력 파일:

```text
data/urban_video_bench/grpo/uvb_grpo_test.jsonl
```

이미 UVB가 준비돼 있으면 이 단계는 생략 가능하다.

### 1-3. 추가 평가셋: VideoMMMU / MMVU(mc)

VideoMMMU:

```bash
bash src/scripts/prepare_videommmu_grpo_data.sh
```

MMVU(mc only):

```bash
bash src/scripts/prepare_mmvu_grpo_data.sh
```

산출물:

```text
data/video_mmmu/grpo/videommmu_grpo_test.jsonl
data/mmvu/grpo/mmvu_grpo_test.jsonl
```

주의:

- VideoMMMU는 URL 기반 비디오 다운로드를 사용하므로 `yt-dlp`가 필요하다.
- MMVU는 Hugging Face 내부 비디오 파일을 직접 사용한다.
- VideoMMMU는 Hugging Face 접근 권한이 필요한 gated dataset일 수 있다.

---

## 2. SFT 훈련

SFT는 `sft/` 폴더에서 수행한다.

```bash
cd sft
CONFIG_PATH=configs/train_lora_qwen25vl3b.yaml \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/run_train.sh
```

산출물:

```text
sft/outputs/qwen25vl3b_lora_sft_40/
```

이 디렉터리에는 SFT LoRA adapter가 저장된다.

---

## 3. SFT 어댑터 결합

SFT LoRA adapter를 base model에 merge해서, 이후 GRPO의 시작 모델로 사용한다.

```bash
cd sft
CONFIG_PATH=configs/merge_lora_qwen25vl3b.yaml \
bash scripts/run_merge.sh
```

기본 merge 결과:

```text
sft/outputs/qwen25vl3b_lora_merged_from_sft40/
```

이 디렉터리가 Step 4의 `QWEN_PATH`가 된다.

---

## 4. Video-R1을 통한 GRPO 훈련

레포 루트에서 실행:

```bash
QWEN_PATH="$(pwd)/sft/outputs/qwen25vl3b_lora_merged_from_sft40" \
TRAIN_FILE="$(pwd)/data/video_r1/grpo/video_r1_grpo_train.jsonl" \
TEST_FILE="$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl" \
OUTPUT_DIR="$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only" \
NUM_GPUS=2 \
TRAIN_NUM_GPUS=1 \
CUDA_VISIBLE_DEVICES=0,1 \
bash src/scripts/run_grpo_uvb_answer_only.sh
```

설명:

- `QWEN_PATH`: Step 3에서 만든 merged SFT model
- `TRAIN_FILE`: Video-R1 train GRPO JSONL
- `TEST_FILE`: UVB benchmark JSONL
- `OUTPUT_DIR`: GRPO 학습 결과 저장 경로

기본적으로 이 스크립트는 `use_peft=true`로 돌기 때문에, GRPO 결과는 adapter 형태로 저장된다.

주요 산출물:

```text
src/r1-v/outputs/video_r1_uvb_grpo_answer_only/
```

---

## 5. GRPO 어댑터 결합

GRPO도 기본 실행이 LoRA/PEFT 기반이므로, 최종 추론용 모델이 필요하면 merge를 한 번 더 하는 것이 안전하다.

현재 레포에는 예시 설정 파일이 하나 있다.

```text
sft/configs/merge_lora_grpo_run12.yaml
```

이 파일의 의미는 다음과 같다.

- `model_name_or_path`: Step 3의 merged SFT model
- `adapter_name_or_path`: Step 4의 GRPO output directory
- `export_dir`: 최종 merged GRPO model directory

예시 설정:

```yaml
model_name_or_path: /absolute/path/to/sft/outputs/qwen25vl3b_lora_merged_from_sft40
adapter_name_or_path: /absolute/path/to/src/r1-v/outputs/video_r1_uvb_grpo_answer_only
export_dir: /absolute/path/to/src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged
remap_adapter_keys: true
```

설정이 맞는지 확인한 뒤 merge:

```bash
cd sft
CONFIG_PATH=configs/merge_lora_grpo_run12.yaml \
bash scripts/run_merge.sh
```

merge 결과 예시:

```text
src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged/
```

이 디렉터리를 Step 6의 최종 평가 모델로 사용하면 된다.

---

## 6. UVB Benchmark를 통한 검증

### 6-1. GRPO 훈련 중 자동 검증

`run_grpo_uvb_answer_only.sh`에 `TEST_FILE`이 지정되어 있으면, 학습 과정에서 UVB test를 같이 사용한다.

즉 Step 4 자체가 이미 `Video-R1 train + UVB test` 구조다.

### 6-2. 별도 평가

최종 merged GRPO model로 UVB benchmark만 다시 평가하려면:

```bash
python src/eval/uvb_eval_only.py \
  --model "$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged" \
  --test-file "$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl" \
  --device cuda:0 \
  --gpu-memory-utilization 0.6 \
  --save-preds "$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged/test_predictions.jsonl" \
  --save-json "$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged/test_metrics.json"
```

대표 출력:

- `answer_accuracy`
- `answer_format_rate`
- `reasoning_present_rate`

---

## 가장 짧은 실행 순서

처음부터 끝까지 가장 짧게 쓰면 아래 순서다.

```bash
# 0. 환경
bash setup.sh
bash src/scripts/check_environment.sh

# 1. 데이터
bash src/scripts/prepare_video_r1_grpo_data.sh --download-mode subset-directories
bash src/scripts/prepare_uvb_grpo_data.sh

# 2. SFT
cd sft
CONFIG_PATH=configs/train_lora_qwen25vl3b.yaml CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh

# 3. SFT merge
CONFIG_PATH=configs/merge_lora_qwen25vl3b.yaml bash scripts/run_merge.sh
cd ..

# 4. GRPO
QWEN_PATH="$(pwd)/sft/outputs/qwen25vl3b_lora_merged_from_sft40" \
TRAIN_FILE="$(pwd)/data/video_r1/grpo/video_r1_grpo_train.jsonl" \
TEST_FILE="$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl" \
OUTPUT_DIR="$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only" \
NUM_GPUS=2 \
TRAIN_NUM_GPUS=1 \
CUDA_VISIBLE_DEVICES=0,1 \
bash src/scripts/run_grpo_uvb_answer_only.sh

# 5. GRPO merge
cd sft
CONFIG_PATH=configs/merge_lora_grpo_run12.yaml bash scripts/run_merge.sh
cd ..

# 6. UVB eval
python src/eval/uvb_eval_only.py \
  --model "$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged" \
  --test-file "$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl" \
  --device cuda:0
```

---

## 핵심 최종 파일

데이터:

- Train: `data/video_r1/grpo/video_r1_grpo_train.jsonl`
- Test: `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`

모델:

- SFT merged model: `sft/outputs/qwen25vl3b_lora_merged_from_sft40/`
- GRPO adapter output: `src/r1-v/outputs/video_r1_uvb_grpo_answer_only/`
- GRPO merged model: `src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged/`

한 줄 요약:

```text
Video-R1로 학습하고, UVB로 검증한다.
```
