# GRPO_Video_2

`Qwen/Qwen2.5-VL-3B-Instruct`를 기반으로, 다음 순서의 연구 파이프라인을 수행하기 위한 레포입니다.

1. SFT 데이터셋으로 LoRA SFT
2. SFT LoRA 어댑터를 백본에 merge
3. merge된 모델을 시작점으로 GRPO 수행
4. GRPO LoRA 어댑터를 다시 merge
5. 최종 모델을 비디오 벤치마크로 평가

현재 레포는 특히 `Video-R1` 기반 GRPO 학습과 `Urban Video Bench`, `VideoMMMU`, `MMVU` 평가를 중심으로 구성되어 있습니다.

이 README는 현재 코드 기준의 실제 구조와 실행 흐름을 국문으로 정리한 문서입니다.

---

## 1. 이 레포가 하는 일

이 레포의 핵심 목적은 다음과 같습니다.

- Qwen2.5-VL 계열 모델에 대해 SFT를 수행할 수 있게 함
- SFT 결과를 merge하여 GRPO 시작 모델을 만듦
- 비디오/멀티모달 GRPO 학습 데이터를 공통 JSONL 포맷으로 정리함
- GRPO 학습 후 LoRA 어댑터만 저장함
- 저장된 어댑터를 다시 merge하여 최종 추론용 모델을 만듦
- 최종 모델을 여러 비디오 벤치마크로 평가함

즉 전체 철학은 다음과 같습니다.

**여러 원천 데이터셋을 공통 포맷으로 맞춘 뒤, SFT와 GRPO를 단계적으로 수행하고, 마지막에는 merge된 단일 추론 모델로 검증한다.**

---

## 2. 현재 지원하는 큰 흐름

현재 코드 기준으로 지원되는 전체 흐름은 아래와 같습니다.

### 2-1. SFT

- 텍스트 SFT
- 이미지/프레임을 포함한 멀티모달 SFT
- 출력 태그 형식:
  - `<ANSWER>...</ANSWER>`
  - `<COT>...</COT><ANSWER>...</ANSWER>`
  - `<LONG_COT>...</LONG_COT><ANSWER>...</ANSWER>`

### 2-2. SFT merge

- SFT LoRA 어댑터를 백본 `Qwen/Qwen2.5-VL-3B-Instruct`에 merge
- 키 mismatch가 발생하는 경우를 위해 adapter remap 로직 내장

### 2-3. GRPO

- 시작 모델: SFT merge 결과물
- 학습 데이터: `Video-R1` 기반 GRPO JSONL
- 학습 결과: GRPO LoRA 어댑터
- 생성은 vLLM 기반, 업데이트는 torch/deepspeed 기반

### 2-4. 최종 merge

- GRPO 어댑터를 SFT merge 모델 위에 다시 merge
- 최종 추론용 모델 생성

### 2-5. 평가

- Urban Video Bench
- VideoMMMU
- MMVU

위 3개 평가셋은 모두 공통 GRPO 입력 포맷으로 정리되어 있으며, 같은 방식으로 모델 입력에 연결됩니다.

---

## 3. 현재 상태에서 중요한 전제

이 문서를 보는 시점 기준으로 중요한 전제는 다음과 같습니다.

- **SFT 데이터셋은 아직 별도로 준비되어 있지 않을 수 있음**
- **GPU 서버에서 end-to-end 학습을 실제 실행한 것은 아님**
- 따라서 현재는 “정적 코드 검토 + 문법 검증 + 데이터 파이프라인 검증”이 완료된 상태로 보는 것이 정확함

즉,

- 코드 구조는 SFT -> merge -> GRPO -> merge -> 평가로 이어지도록 정리되어 있음
- 실제 대규모 GPU 서버에서 최종 실행 검증은 별도로 필요함

---

## 4. 디렉터리 구조

레포 주요 디렉터리는 다음과 같습니다.

```text
GRPO_Video_2/
├─ data/
├─ sft/
├─ src/
├─ video_r1_sft_annotator/
├─ QUICKSTART.md
├─ merge_readme.md
├─ REPO_STRUCTURE_AND_REVIEW.md
└─ setup.sh
```

### 4-1. `data/`

학습/평가용 데이터가 저장됩니다.

대표 구조:

```text
data/
├─ video_r1/
│  ├─ raw/
│  ├─ processed/
│  └─ grpo/
├─ urban_video_bench/
│  ├─ raw/
│  ├─ processed/
│  └─ grpo/
├─ video_mmmu/
│  ├─ raw/
│  ├─ processed/
│  └─ grpo/
└─ mmvu/
   ├─ raw/
   ├─ processed/
   └─ grpo/
```

### 4-2. `sft/`

SFT 관련 코드가 들어 있습니다.

대표 파일:

- `sft/scripts/train_sft.py`
- `sft/scripts/merge_lora.py`
- `sft/scripts/run_train.sh`
- `sft/scripts/run_merge.sh`
- `sft/scripts/run_pipeline.sh`
- `sft/configs/train_lora_qwen25vl3b.yaml`
- `sft/configs/merge_lora_qwen25vl3b.yaml`
- `sft/configs/merge_lora_grpo_run12.yaml`

### 4-3. `src/eval/`

데이터 준비와 평가 관련 코드가 들어 있습니다.

대표 파일:

- `src/eval/prepare_video_r1_grpo.py`
- `src/eval/prepare_uvb_pipeline.py`
- `src/eval/prepare_videommmu.py`
- `src/eval/prepare_mmvu.py`
- `src/eval/data_to_grpo.py`
- `src/eval/uvb_eval_only.py`

### 4-4. `src/r1-v/src/open_r1/`

GRPO 학습 핵심 코드입니다.

대표 파일:

- `src/r1-v/src/open_r1/grpo_video.py`
- `src/r1-v/src/open_r1/grpo_uvb.py`
- `src/r1-v/src/open_r1/trainer/grpo_trainer.py`
- `src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py`

### 4-5. `src/scripts/`

실행 스크립트 모음입니다.

대표 파일:

- `src/scripts/run_grpo_uvb_answer_only.sh`
- `src/scripts/run_grpo_uvb_answer_only_lora.sh`
- `src/scripts/prepare_all_grpo_data.sh`
- `src/scripts/check_environment.sh`
- `src/scripts/apply_rotary_dtype_hotfix.sh`

---

## 5. 공통 데이터 포맷

이 레포는 여러 데이터셋을 최종적으로 **공통 GRPO JSONL 포맷**으로 변환합니다.

주요 키는 다음과 같습니다.

- `video_id`
- `question_id`
- `question_category`
- `problem`
- `frames`
- `solution`

예시:

```json
{
  "video_id": "sample_001",
  "question_id": 1,
  "question_category": "Perception",
  "problem": "What is the person holding?\nA. Book\nB. Cup\nC. Phone\nD. Bag",
  "frames": [
    "../processed/frames/test/sample_001/frame_000.jpg",
    "../processed/frames/test/sample_001/frame_001.jpg"
  ],
  "solution": "<ANSWER>B</ANSWER>"
}
```

이 포맷의 장점은 다음과 같습니다.

- UVB, VideoMMMU, MMVU, Video-R1 train을 같은 입출력 인터페이스로 처리 가능
- 학습/평가 로직이 데이터셋별로 갈라지지 않음
- 최종 모델 입력을 모두 “프레임 기반 멀티모달 QA”로 통일 가능

---

## 6. SFT 파이프라인

### 6-1. 지원하는 SFT 데이터 형식

현재 `sft/scripts/train_sft.py`는 두 종류를 처리할 수 있습니다.

#### A. 텍스트 SFT

형식:

```json
[
  {
    "instruction": "문제 설명",
    "input": "추가 입력",
    "output": "<ANSWER>A</ANSWER>"
  }
]
```

#### B. 멀티모달 SFT

형식:

```json
[
  {
    "problem": "프레임을 보고 정답을 고르시오 ...",
    "solution": "<COT>...</COT><ANSWER>B</ANSWER>",
    "frames": ["frame_000.jpg", "frame_001.jpg"]
  }
]
```

또는:

```json
[
  {
    "problem": "이미지를 보고 답하시오 ...",
    "solution": "<ANSWER>C</ANSWER>",
    "image": "image_001.jpg"
  }
]
```

또는:

```json
[
  {
    "problem": "이미지들을 보고 답하시오 ...",
    "solution": "<LONG_COT>...</LONG_COT><ANSWER>D</ANSWER>",
    "images": ["img1.jpg", "img2.jpg"]
  }
]
```

상대경로는 **해당 JSON/JSONL 파일이 위치한 디렉터리 기준**으로 해석됩니다.

### 6-2. 이미지 사용 여부

SFT는 두 모드를 모두 지원합니다.

- `USE_VISION=false`
  - 텍스트만 사용
- `USE_VISION=true`
  - 이미지/프레임을 실제로 로드해서 사용

실행 시 `run_train.sh` 또는 `run_pipeline.sh`가 직접 물어볼 수도 있고, 환경변수로 고정할 수도 있습니다.

### 6-3. SFT LoRA target

현재 기본 설정은:

```yaml
lora_target_modules: auto
```

이 의미는 다음과 같습니다.

- 텍스트 SFT:
  - 언어 모듈 중심으로 LoRA 적용
- 비전 사용 SFT:
  - 언어 + 비전 선형 모듈까지 포함하여 LoRA 적용

즉 “이미지 사용 여부”에 따라 자동으로 더 넓은 범위의 모듈에 LoRA를 적용하도록 설계되어 있습니다.

### 6-4. SFT 실행 명령

가장 일반적인 실행:

```bash
cd sft
bash scripts/run_pipeline.sh
```

이 명령은:

1. 이미지/프레임 사용 여부를 물어봄
2. SFT 수행
3. 끝나면 자동으로 merge 수행

텍스트만:

```bash
cd sft
USE_VISION=false bash scripts/run_pipeline.sh
```

이미지/프레임 포함:

```bash
cd sft
USE_VISION=true bash scripts/run_pipeline.sh
```

학습만 하고 merge는 나중에:

```bash
cd sft
USE_VISION=true bash scripts/run_train.sh
```

### 6-5. SFT 결과물

기본 출력:

- LoRA 어댑터:
  - `sft/outputs/qwen25vl3b_lora_sft/`
- merge 결과:
  - `sft/outputs/qwen25vl3b_lora_merged/`

---

## 7. SFT merge 파이프라인

SFT 이후 LoRA 어댑터를 백본 모델에 병합합니다.

관련 파일:

- `sft/scripts/merge_lora.py`
- `sft/configs/merge_lora_qwen25vl3b.yaml`

기본 설정:

```yaml
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
adapter_name_or_path: ./outputs/qwen25vl3b_lora_sft
export_dir: ./outputs/qwen25vl3b_lora_merged
remap_adapter_keys: true
```

### 왜 `remap_adapter_keys: true`가 중요한가

Qwen2.5-VL 계열에서는 저장된 adapter key와 실제 backbone key가 다음 형태로 어긋나는 경우가 있습니다.

- `language_model.layers`
- `model.layers`
- `visual.blocks`
- `model.visual.blocks`

이 레포는 merge 전에 adapter key를 remap해서 이런 mismatch를 줄이도록 구성되어 있습니다.

즉, 과거에 있었던 “merge는 됐는데 실제 LoRA 가중치가 적용되지 않는” 문제를 막기 위한 장치입니다.

---

## 8. GRPO 파이프라인

### 8-1. 시작 모델

GRPO는 기본적으로 **SFT를 merge한 모델**을 시작점으로 사용합니다.

즉 순서는:

1. 백본
2. SFT LoRA 학습
3. SFT LoRA merge
4. merge된 모델을 GRPO 시작 모델로 사용

### 8-2. 학습 데이터

현재 학습용 주력 데이터는:

- `data/video_r1/grpo/video_r1_grpo_train.jsonl`

입니다.

이 파일은 `Video-R1` 원본을 전처리하여 만든 공통 포맷입니다.

### 8-3. 평가 데이터

대표 평가셋:

- `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`
- `data/video_mmmu/grpo/videommmu_grpo_test.jsonl`
- `data/mmvu/grpo/mmvu_grpo_test.jsonl`

### 8-4. 실행 스크립트

대표 실행 스크립트:

- `src/scripts/run_grpo_uvb_answer_only.sh`
- `src/scripts/run_grpo_uvb_answer_only_lora.sh`

실제로는 `TEST_FILE`만 바꾸면 UVB/MMMU/MMVU 모두 같은 구조로 사용 가능합니다.

예시:

```bash
QWEN_PATH="$(pwd)/sft/outputs/qwen25vl3b_lora_merged" \
TRAIN_FILE="$(pwd)/data/video_r1/grpo/video_r1_grpo_train.jsonl" \
TEST_FILE="$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl" \
OUTPUT_DIR="$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only" \
NUM_GPUS=2 \
TRAIN_NUM_GPUS=1 \
CUDA_VISIBLE_DEVICES=0,1 \
bash src/scripts/run_grpo_uvb_answer_only.sh
```

### 8-5. vLLM과 학습 GPU의 역할 분리

이 레포의 GRPO 핵심은 다음과 같습니다.

- 생성은 vLLM이 담당
- loss 계산/업데이트는 torch/deepspeed가 담당

가능하면 다음 구조를 권장합니다.

- 예: GPU 4장
  - 학습: 3장
  - vLLM: 1장

코드상으로는 vLLM을 별도 GPU에 올리도록 되어 있으며, 학습 step 사이에 최신 weight를 vLLM 엔진 쪽으로 다시 load합니다.

즉 “추론은 빠르게”, “가중치 업데이트는 기존 training stack으로”라는 분리 구조입니다.

---

## 9. GRPO LoRA target 정책

GRPO에서는 기본적으로 다음 target만 사용하는 것이 안전합니다.

```text
q_proj k_proj v_proj o_proj
```

이유:

- `gate_proj`, `up_proj`, `down_proj`는 언어 모델뿐 아니라 visual encoder MLP에도 넓게 걸릴 수 있음
- GRPO는 forward가 반복되기 때문에 activation memory가 더 쉽게 누적됨
- 결과적으로 OOM 위험이 커짐

현재 스크립트는 이 정책을 반영하도록 맞춰져 있습니다.

---

## 10. 필수 운영 패치와 안정화 포인트

이 레포는 과거 디버깅 결과를 반영해 몇 가지 중요한 안정화 포인트를 포함합니다.

### 10-1. `use_cache=False`

최신 transformers 환경에서는 기본 `use_cache=True`로 인해 `DynamicCache()`가 생성되고, Flash Attention 경로에서 padding 관련 오류가 날 수 있습니다.

따라서 trainer의 per-token logprob 계산 경로에서는 `use_cache=False`를 강제로 사용합니다.

반영 위치:

- `src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py`
- `src/r1-v/src/open_r1/trainer/grpo_trainer.py`

### 10-2. rotary dtype hotfix

일부 환경에서는 `q.float()`와 `cos/sin`의 dtype mismatch 때문에 아래 오류가 날 수 있습니다.

- `AssertionError: same dtype, got float32 and bfloat16`

이를 위해 레포에는 site-packages를 직접 패치하는 스크립트가 포함되어 있습니다.

관련 스크립트:

- `src/scripts/apply_rotary_dtype_hotfix.sh`

즉 이 버그는 “코드 안에 자동 반영”이 아니라, **필요한 환경에서 한 번 실행해 해결하는 방식**입니다.

### 10-3. Attention implementation

GRPO 스크립트 기본값은:

```text
flash_attention_2
```

입니다.

이는 SDPA 대비 메모리 사용량을 줄여 긴 프레임 입력에서 OOM을 피하는 데 도움이 됩니다.

---

## 11. 최종 merge

GRPO 완료 후에는 다시 LoRA 어댑터를 merge할 수 있습니다.

관련 파일:

- `sft/configs/merge_lora_grpo_run12.yaml`
- `sft/scripts/merge_lora.py`

현재 기본 예시는:

```yaml
model_name_or_path: ./outputs/qwen25vl3b_lora_merged
adapter_name_or_path: ../src/r1-v/outputs/uvb_grpo_run12
export_dir: ../src/r1-v/outputs/uvb_grpo_run12_merged
remap_adapter_keys: true
```

실제 실험 디렉터리에 맞게 `adapter_name_or_path`와 `export_dir`를 바꿔 쓰면 됩니다.

즉 순서는:

1. SFT merge 모델 준비
2. 그 모델로 GRPO 수행
3. GRPO LoRA 어댑터 저장
4. GRPO 어댑터를 다시 merge
5. 최종 모델로 평가

---

## 12. 평가 구조

### 12-1. 학습 중 test inference

GRPO 스크립트에서 `TEST_FILE`이 지정되어 있으면, 학습 완료 후 `test_predictions.jsonl`이 자동 생성됩니다.

### 12-2. 별도 평가

현재 별도 평가 스크립트는 UVB 이름으로 제공됩니다.

- `src/eval/uvb_eval_only.py`

하지만 실제 입력 형식은 공통 JSONL이라서, 같은 스키마를 쓰는 데이터에는 재사용 가능합니다.

즉 현재 구조는:

- 코드 이름은 UVB 전용처럼 보이지만
- 실제 입력 인터페이스는 공통 프레임 기반 QA 포맷

입니다.

---

## 13. 데이터 준비 스크립트

대표 데이터 준비 명령:

### Video-R1 train

```bash
bash src/scripts/prepare_video_r1_grpo_data.sh
```

### UVB

```bash
bash src/scripts/prepare_uvb_grpo_data.sh
```

### VideoMMMU

```bash
bash src/scripts/prepare_videommmu_grpo_data.sh
```

### MMVU

```bash
bash src/scripts/prepare_mmvu_grpo_data.sh
```

### 한 번에 모두

```bash
bash src/scripts/prepare_all_grpo_data.sh
```

---

## 14. 추천 실행 순서

### 14-1. 환경 준비

```bash
bash setup.sh
bash src/scripts/check_environment.sh
```

필요하면:

```bash
bash src/scripts/apply_rotary_dtype_hotfix.sh
```

### 14-2. 데이터 준비

```bash
bash src/scripts/prepare_all_grpo_data.sh
```

### 14-3. SFT

```bash
cd sft
bash scripts/run_pipeline.sh
```

### 14-4. GRPO

```bash
cd ..
QWEN_PATH="$(pwd)/sft/outputs/qwen25vl3b_lora_merged" \
TRAIN_FILE="$(pwd)/data/video_r1/grpo/video_r1_grpo_train.jsonl" \
TEST_FILE="$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl" \
OUTPUT_DIR="$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only" \
NUM_GPUS=2 \
TRAIN_NUM_GPUS=1 \
CUDA_VISIBLE_DEVICES=0,1 \
bash src/scripts/run_grpo_uvb_answer_only.sh
```

### 14-5. 최종 merge

```bash
cd sft
python scripts/merge_lora.py \
  --model-name-or-path ./outputs/qwen25vl3b_lora_merged \
  --adapter-name-or-path ../src/r1-v/outputs/video_r1_uvb_grpo_answer_only \
  --export-dir ../src/r1-v/outputs/video_r1_uvb_grpo_answer_only_merged \
  --remap-adapter-keys true
```

### 14-6. 벤치마크 평가

평가셋별로 `TEST_FILE`를 바꿔가며 실행하거나, 별도 eval 스크립트를 사용합니다.

---

## 15. 현재 기준으로 “되는 것”과 “아직 남은 것”

### 되는 것

- SFT -> merge 흐름 정리
- 텍스트 / 멀티모달 SFT 둘 다 지원
- GRPO 데이터 포맷 통일
- Video-R1 train + UVB/MMMU/MMVU test 데이터 준비 구조 정리
- SFT merge 모델을 시작점으로 GRPO 수행 가능
- GRPO 후 다시 merge 가능

### 아직 남은 것

- 실제 GPU 서버에서 end-to-end 실행 검증
- 실제 멀티모달 SFT 데이터셋으로 adapter save -> merge까지 실런
- UVB/MMMU/MMVU를 완전히 공통화한 별도 eval wrapper 정리

즉 현재 상태를 한 줄로 요약하면:

**코드 구조와 실행 흐름은 정리되어 있으며, 실제 대규모 서버에서의 최종 실험 검증만 남아 있다.**

---

## 16. 관련 문서

- [`QUICKSTART.md`](/Users/jw246/Desktop/NTU%20COSMO%20LAB/cloned%20Repos/GRPO_Video_2/QUICKSTART.md)
- [`merge_readme.md`](/Users/jw246/Desktop/NTU%20COSMO%20LAB/cloned%20Repos/GRPO_Video_2/merge_readme.md)
- [`sft/README.md`](/Users/jw246/Desktop/NTU%20COSMO%20LAB/cloned%20Repos/GRPO_Video_2/sft/README.md)
- [`src/eval/README.md`](/Users/jw246/Desktop/NTU%20COSMO%20LAB/cloned%20Repos/GRPO_Video_2/src/eval/README.md)
- [`REPO_STRUCTURE_AND_REVIEW.md`](/Users/jw246/Desktop/NTU%20COSMO%20LAB/cloned%20Repos/GRPO_Video_2/REPO_STRUCTURE_AND_REVIEW.md)

