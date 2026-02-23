# GRPO_Video 레포 흐름 및 RunPod 실행 가이드

이 문서는 **Urban Video Bench(UVB)** 데이터로 **Qwen2.5-VL**을 **GRPO** 방식으로 학습하는 이 레포의 전체 흐름과, **RunPod** 환경에서 실행할 때의 포인트를 정리한 것입니다.

---

## 1. 전체 흐름 요약

```
[1] 환경 설정 (setup.sh)
        ↓
[2] UVB 메타데이터 준비 (HuggingFace → JSONL)
        ↓
[3] 40% 샘플링 + 8:2 분할 + 비디오 다운로드 + 프레임 추출
        ↓
[4] GRPO 학습용 JSONL 변환 (frames 경로 + problem/solution)
        ↓
[5] GRPO 학습 (vLLM 생성 + LoRA/전체 파라미터 + 보상 함수)
        ↓
[6] (선택) 평가/추론
```

---

## 2. 디렉터리 구조와 역할

| 경로 | 역할 |
|------|------|
| `setup.sh` | conda + pip 의존성 설치 (r1-v 패키지, vLLM, flash-attn 등) |
| `src/r1-v/` | 학습 패키지 루트 (`open_r1` 모듈, DeepSpeed config) |
| `src/r1-v/src/open_r1/grpo_uvb.py` | **GRPO UVB 엔트리포인트** (데이터 로드, 트레이너 생성, 학습/테스트 추론) |
| `src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py` | vLLM으로 생성 + 보상 계산 + GRPO loss 학습 |
| `src/scripts/*.sh` | 데이터 준비 및 학습 실행 스크립트 |
| `src/eval/*.py` | UVB 메타데이터/파이프라인/GRPO 변환/평가 스크립트 |
| `data/urban_video_bench/` | UVB 메타데이터, 비디오, processed/frames, grpo JSONL |

---

## 3. 데이터 파이프라인 상세

### 3.1 메타데이터 준비 (`prepare_uvb_dataset.sh`)

- **실행**: `bash src/scripts/prepare_uvb_dataset.sh`
- **동작**: HuggingFace `EmbodiedCity/UrbanVideo-Bench` 데이터셋을 로드해 `train` 스플릿을 JSONL로 저장.
- **출력**: `data/urban_video_bench/urban_video_bench_train.jsonl`

### 3.2 샘플링·분할·다운로드·프레임 추출 (`prepare_uvb_40_split_download_frames.sh`)

- **실행**: `bash src/scripts/prepare_uvb_40_split_download_frames.sh`
- **동작** (`src/eval/prepare_uvb_pipeline.py`):
  1. 위 JSONL 로드 후 **question_category 기준 40% 샘플링**
  2. **8:2 stratified train/test 분할** → `train_80.jsonl`, `test_20.jsonl`
  3. HF에서 **mp4 비디오 다운로드** (`dataset-id`, `video-dir`)
  4. 각 비디오에서 **32프레임 추출**, 긴 변 768 이하로 리사이즈 후 `frames/train/<video_stem>/`, `frames/test/<video_stem>/`에 저장
- **출력**:
  - `data/urban_video_bench/processed/train_80.jsonl`, `test_20.jsonl`
  - `data/urban_video_bench/processed/frames/`
  - `pipeline_summary.json` 등

### 3.3 GRPO용 JSONL 변환 (`prepare_uvb_grpo_data.sh`)

- **실행**: `bash src/scripts/prepare_uvb_grpo_data.sh`
- **동작** (`src/eval/uvb_to_grpo.py`):  
  `processed/train_80.jsonl`, `test_20.jsonl`와 `processed/frames/`를 사용해, **프레임 경로 리스트**와 **problem/solution** 형식으로 GRPO용 JSONL 생성.  
  프레임이 없는 샘플은 제외.
- **출력**:
  - `data/urban_video_bench/grpo/uvb_grpo_train.jsonl`
  - `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`
  - `uvb_grpo_summary.json`

GRPO JSONL 한 줄 예시 형식:

- `video_id`, `question_id`, `question_category`, `problem` (질문 텍스트), `frames` (로컬 이미지 경로 리스트), `solution` (`<answer>...</answer>` 형식)

---

## 4. 학습(GRPO) 흐름

### 4.1 실행 경로

- **기본 학습**: `bash src/scripts/run_grpo_uvb_answer_only.sh`
- 단일 학습 진입점: `bash src/scripts/run_grpo_uvb_answer_only.sh`

스크립트는 **레포 루트가 아니라** `src/r1-v`로 이동한 뒤, `PYTHONPATH=./src`로 `torchrun` 실행:

```bash
cd "${REPO_ROOT}/src/r1-v"
PYTHONPATH="./src" torchrun --nproc_per_node="${NUM_GPUS}" ... -m open_r1.grpo_uvb ...
```

따라서 스크립트 내부의 `TRAIN_FILE`/`TEST_FILE` 기본값은 **src/r1-v 기준 상대 경로** (`../../data/...`)입니다.

### 4.2 엔트리포인트: `open_r1.grpo_uvb`

- **파일**: `src/r1-v/src/open_r1/grpo_uvb.py`
- **역할**:
  1. `GRPOUVBScriptArguments`, `GRPOConfig`, `ModelConfig` 파싱
  2. `train_file` / `test_file` JSONL 로드 후 `make_conversation_video`로 대화형 구조 생성:
     - `image_vllm`: 프레임 경로(리스트)
     - `solution`: 정답 문자열
     - `prompt`: system + user(이미지 토큰 + 문제 텍스트)
  3. **보상 함수**: `answer_accuracy`(정답 일치), `answer_format`(`<answer>...</answer>` 형식 준수) — 스크립트 인자로 선택 가능
  4. **트레이너**: `--use_vllm true`이면 `Qwen2VLGRPOVLLMTrainerModified` 사용
  5. 학습 후 `trainer.save_model(output_dir)`  
     테스트셋이 있으면 `run_test_inference()`로 예측하고 `test_predictions.jsonl` 저장

### 4.3 트레이너: `Qwen2VLGRPOVLLMTrainerModified`

- **파일**: `src/r1-v/src/open_r1/trainer/vllm_grpo_trainer_modified.py`
- **역할 요약**:
  - **vLLM**: 메인 프로세스에서 vLLM `LLM` 인스턴스 생성 (별도 GPU 권장: `vllm_device`, 기본 `cuda:num_processes`).  
    프롬프트(이미지+텍스트)당 `num_generations`개 생성.
  - **학습 모델**: Qwen2.5-VL (또는 Qwen2-VL/Aria) + 선택적 **LoRA** (`peft_config`).  
    DeepSpeed ZeRO-1 사용 시 ref 모델도 로드.
  - **스텝 흐름**:
    1. `_prepare_inputs`: 배치의 이미지 로드·리사이즈(`max_pixels`/`min_pixels`), 채팅 템플릿 적용 → `prompt_ids`, 이미지 등
    2. 주기적으로 학습 모델 가중치를 vLLM에 동기화(`load_weights`)
    3. vLLM으로 각 프롬프트당 `num_generations`개 생성
    4. 생성 결과를 모든 프로세스에 broadcast
    5. **보상 계산**: `reward_funcs`(여기선 정답/형식) → 그룹 내 정규화로 advantage 계산
    6. **GRPO loss**: policy logp vs ref logp, advantage, KL 페널티(`beta`)로 loss 계산 후 역전파

### 4.4 주요 환경변수 / 인자 (run_grpo_uvb_answer_only.sh)

- `QWEN_PATH`: 기본 `Qwen/Qwen2.5-VL-3B-Instruct`
- `TRAIN_FILE` / `TEST_FILE`: GRPO JSONL (기본 `../../data/.../uvb_grpo_train.jsonl`, `uvb_grpo_test.jsonl`)
- `OUTPUT_DIR`: 체크포인트 저장 (기본 `./outputs/uvb_grpo_answer_only`)
- `NUM_GPUS` / `CUDA_VISIBLE_DEVICES`: GPU 개수와 매핑
- `USE_PEFT` / `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT`: LoRA 사용 여부 및 하이퍼파라미터
- `VLLM_GPU_UTIL`: vLLM GPU 메모리 사용률
- `VLLM_MAX_FRAMES`: 학습 시 프롬프트당 최대 프레임 수 (기본 8). 환경변수로 지정.
- `VLLM_MAX_FRAMES_EVAL`: **테스트 추론 시** 프롬프트당 최대 프레임 수. 미설정 시 `VLLM_MAX_FRAMES`와 동일. 예: 학습 8장·테스트 16장으로 하려면 `VLLM_MAX_FRAMES_EVAL=16` 설정.
- `MAX_PIXELS` / `MIN_PIXELS`: 이미지 픽셀 상/하한
- `NUM_GENERATIONS`: 프롬프트당 생성 개수 (G)
- `MAX_PROMPT_LENGTH` / `MAX_COMPLETION_LENGTH`: 시퀀스 길이
- `DS_CONFIG`: DeepSpeed 설정 (기본 `configs/zero1_no_optimizer.json`)

---

## 5. RunPod에서 실행할 때

### 5.1 환경 요구사항

- **Python**: 3.11+ (스크립트/check_environment.sh 권장)
- **GPU**: VRAM 20GB+ 권장 (24GB+ 더 안정적), 멀티 GPU 가능
- **디스크**: 100GB+ 여유 (모델 ~30GB, 데이터·체크포인트 등)
- **네트워크**: HuggingFace 접근 (메타데이터, 모델, 데이터셋 다운로드)

### 5.2 RunPod에서 할 일

1. **이 레포 클론**  
   - RunPod 템플릿/스크립트에서 `git clone` 또는 코드 마운트 후 레포 루트로 이동.

2. **경로**  
   - 모든 스크립트는 **레포 루트**에서 실행하는 것을 전제로 함.  
     예: `bash src/scripts/run_grpo_uvb_answer_only.sh`  
   - 학습 시 실제 작업 디렉터리는 `src/r1-v`로 바뀌므로, `TRAIN_FILE`/`TEST_FILE`은 `src/r1-v` 기준 상대경로(`../../data/...`) 또는 절대 경로로 맞추면 됨.

3. **환경 구성**  
   ```bash
   conda create -n r1-v python=3.11 -y
   conda activate r1-v
   bash setup.sh
   ```
   필요 시 `bash src/scripts/check_environment.sh`로 점검.

4. **데이터 준비** (이미 있으면 생략 가능)  
   - 순서대로:
     - `bash src/scripts/prepare_uvb_dataset.sh`
     - `bash src/scripts/prepare_uvb_40_split_download_frames.sh`  
       (HF 토큰 필요할 수 있음: `HF_TOKEN` 또는 `HUGGINGFACE_HUB_TOKEN`)
     - `bash src/scripts/prepare_uvb_grpo_data.sh`

5. **GPU 개수에 맞춰 실행**  
   - **A40 6대**: `NUM_GPUS`를 지정하지 않으면 자동으로 6으로 감지되고, **학습 프로세스 5개 + vLLM 전용 1 GPU**로 동작합니다 (vLLM은 `cuda:5`). 그냥 실행하면 됨:
     ```bash
     bash src/scripts/run_grpo_uvb_answer_only.sh
     ```
   - 단일 GPU:
     ```bash
     NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash src/scripts/run_grpo_uvb_answer_only.sh
     ```
     (vLLM은 학습과 같은 `cuda:0`을 공유. OOM 시 `NUM_GENERATIONS`·`VLLM_GPU_UTIL` 등 조정.)
   - **직접 지정**: `TRAIN_NUM_GPUS=6`으로 두면 6개 모두 학습에 쓰고 vLLM은 `cuda:0`과 공유 (OOM 위험). 기본은 `NUM_GPUS > 1`일 때 `TRAIN_NUM_GPUS = NUM_GPUS - 1`로 vLLM 전용 1개 확보.
   - RunPod 인스턴스 GPU 수에 맞게 `NUM_GPUS`(또는 `CUDA_VISIBLE_DEVICES`)만 설정하면 됨.

6. **OOM 대응**  
   - `NUM_GENERATIONS` 감소 (예: 8→4)
   - `MAX_PIXELS` / `MAX_PROMPT_LENGTH` / `MAX_COMPLETION_LENGTH` 감소
   - `VLLM_GPU_UTIL` 조정
   - **학습 8장 / 테스트 16장**: 테스트만 16장 쓰려면 `VLLM_MAX_FRAMES_EVAL=16` (학습은 기본 8장 유지).

### 5.3 RunPod 관련 참고

- 이 레포에는 **RunPod 전용 스크립트나 Dockerfile은 없음**.  
  위 순서대로 conda + setup.sh + 데이터 스크립트 + 학습 스크립트만 실행하면 됨.
- **Volume/영구 디스크**: 데이터와 체크포인트를 유지하려면 RunPod Volume을 마운트한 뒤 `data/`, `src/r1-v/outputs/` 등을 그 안에 두는 구성을 권장.
- **HuggingFace**: 비공개 데이터셋이나 gated 모델 사용 시 `HF_TOKEN`(또는 `HUGGINGFACE_HUB_TOKEN`)을 RunPod 환경변수/시크릿에 설정.

---

## 6. 요약

- **데이터**: HuggingFace UVB → 메타 JSONL → 40% 샘플·8:2 분할 → 비디오 다운로드·프레임 추출 → GRPO용 JSONL.
- **학습**: `open_r1.grpo_uvb`가 JSONL을 로드해 `Qwen2VLGRPOVLLMTrainerModified`로 vLLM 생성 + 정답/형식 보상 + GRPO 업데이트.
- **RunPod**: 레포 루트에서 환경 설정 → 데이터 3단계 준비 → `NUM_GPUS`/경로/메모리만 맞춰 `run_grpo_uvb_answer_only.sh` 실행하면 됨.

추가로 RunPod용 Dockerfile이나 스타트업 스크립트가 필요하면 그에 맞춰 설계해 줄 수 있습니다.
