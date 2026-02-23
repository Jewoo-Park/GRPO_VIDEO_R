# 처음부터 실행할 때 생길 수 있는 문제 (Troubleshooting)

이 레포를 **처음부터 쭉** 돌릴 때 예상되는 문제와 대처 방법을 단계별로 정리했습니다.

---

## 0. 전제: 실행 위치

**모든 데이터 준비 스크립트와 학습 스크립트는 반드시 레포 루트에서 실행해야 합니다.**

- ✅ `cd /path/to/TON_Codex && bash src/scripts/prepare_uvb_dataset.sh`
- ❌ `cd /path/to/TON_Codex/src/scripts && bash prepare_uvb_dataset.sh`  
  → 이렇게 하면 `python src/eval/...`가 `src/scripts/src/eval`를 찾아 실패합니다.

QUICKSTART/문서에서 “레포 루트에서 실행”을 명시해 두는 것이 좋습니다.

---

## 1. 환경 설정 (setup.sh)

### 1.1 setup.sh 실행 위치

- `setup.sh`는 **레포 루트**에서 실행해야 합니다.  
  첫 줄이 `cd src/r1-v`이므로 상대 경로 기준이 레포 루트입니다.
- 다른 디렉터리에서 실행하면 `src/r1-v`를 찾지 못해 실패합니다.

### 1.2 Python 버전

- `setup.py`: `python_requires=">=3.10.9"`
- `check_environment.sh`: Python 3.11+ 권장
- RunPod/새 환경에서는 `conda create -n r1-v python=3.11` 등으로 3.11 사용을 권장합니다.

### 1.3 vLLM / transformers 버전 충돌

- `setup.py`에는 `vllm==0.6.6.post1`, `setup.sh`에서는 `pip install vllm==0.7.2`로 덮어씁니다.
- `setup.sh`가 나중에 실행되므로 최종적으로는 vLLM 0.7.2가 설치됩니다.
- **문제**: `pip install -e ".[dev]"`만 하고 `setup.sh`를 안 돌리면 0.6.6이 남아 있고, 실제 학습 스크립트는 0.7.2를 전제로 할 수 있어 호환성 문제가 생길 수 있습니다.
- **대처**: 반드시 **전체 `setup.sh`**를 한 번 끝까지 실행한 뒤 학습을 돌리세요.

### 1.4 flash-attn 설치 실패

- `pip install flash-attn --no-build-isolation`은 CUDA·컴파일 환경이 필요합니다.
- CPU 전용 머신이나 CUDA 버전이 맞지 않으면 빌드가 실패할 수 있습니다.
- RunPod 등 GPU 인스턴스에서는 보통 문제없지만, 실패 시:
  - CUDA 버전 확인 (`nvcc --version`, `nvidia-smi`)
  - [flash-attn](https://github.com/Dao-AILab/flash-attention) 레포의 설치 가이드에 맞춰 환경 맞추기

### 1.5 네트워크 / Git

- `pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef`  
  → GitHub 접근이 필요합니다. 회사 방화벽/프록시에서는 실패할 수 있습니다.
- **대처**: 프록시 설정 또는 미리 해당 커밋의 transformers를 설치해 두기.

---

## 2. 데이터 준비 1단계: prepare_uvb_dataset.sh

### 2.1 HuggingFace 접근

- `load_dataset("EmbodiedCity/UrbanVideo-Bench")`로 데이터셋을 내려받습니다.
- 네트워크 차단/제한이 있으면 실패합니다.
- 비공개/ gated 데이터셋이면 로그인 필요:
  - `huggingface-cli login` 또는 환경변수 `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`

### 2.2 데이터셋 스키마 변경

- 데이터셋 컬럼명이 바뀌면 이후 단계에서 키 에러가 날 수 있습니다.
- `uvb_to_grpo.py`는 `Question_id` / `question_id` 둘 다 보도록 수정해 두었습니다.  
  다른 컬럼명이 있다면 `prepare_urban_video_bench.py` 출력 JSONL을 확인한 뒤, 파이프라인/GRPO 변환 스크립트의 키를 맞춰 주세요.

---

## 3. 데이터 준비 2단계: prepare_uvb_40_split_download_frames.sh

### 3.1 실행 위치와 경로

- 반드시 **레포 루트**에서 실행.  
  `--input-jsonl "data/urban_video_bench/urban_video_bench_train.jsonl"` 등이 모두 cwd 기준 상대 경로입니다.

### 3.2 비디오 다운로드

- `snapshot_download(..., repo_id=dataset_id, allow_patterns=["*.mp4"])`  
  → 대용량 다운로드이며, 디스크 여유가 필요합니다.
- **HF_TOKEN** / **HUGGINGFACE_HUB_TOKEN**이 필요할 수 있습니다 (gated/private).

### 3.3 PyAV / 비디오 디코딩

- `import av` (PyAV)로 비디오를 디코딩합니다.
- 시스템에 **libav/ffmpeg**가 없거나 버전이 맞지 않으면 import 또는 디코딩 단계에서 실패할 수 있습니다.
- **대처**: `apt-get install libavdevice-dev libavformat-dev libavcodec-dev` 등 환경에 맞게 설치.

### 3.4 video_id와 프레임 디렉터리 이름

- 파이프라인은 프레임을 `frames_root / split_name / safe_stem(video_path.stem)` 아래에 저장합니다.
- `uvb_to_grpo.py`에서도 동일하게 `safe_stem(Path(video_id).stem)`으로 프레임 디렉터리를 찾도록 수정해 두었습니다.  
  `video_id`에 특수문자/점이 있어도 파이프라인과 일치해야 합니다.

### 3.5 디스크 용량

- 비디오 + 추출 프레임으로 수십 GB 이상 쓸 수 있습니다.  
  RunPod 등에서는 Volume 크기를 넉넉히 두는 것이 좋습니다.

---

## 4. 데이터 준비 3단계: prepare_uvb_grpo_data.sh

### 4.1 입력 파일 존재

- `data/urban_video_bench/processed/train_80.jsonl`, `test_20.jsonl`  
- `data/urban_video_bench/processed/frames/train/...`, `.../test/...`  
  위가 없으면 실패합니다. 2단계를 먼저 끝까지 실행했는지 확인하세요.

### 4.2 프레임이 하나도 없는 비디오

- 프레임 추출에 실패한 비디오는 해당 샘플이 GRPO JSONL에서 제외됩니다.
- `uvb_grpo_summary.json`의 `train_skipped_no_frames`, `test_skipped_no_frames`로 건너뛴 개수를 확인할 수 있습니다.
- 스킵이 과도하면 2단계 로그에서 비디오 누락/디코딩 에러를 확인하세요.

---

## 5. 학습: run_grpo_uvb_answer_only.sh

### 5.1 실행 위치

- 스크립트가 내부에서 `REPO_ROOT`를 구한 뒤 `cd "${REPO_ROOT}/src/r1-v"` 하므로, **어디서 실행하든 레포 내 스크립트 경로만 맞으면 됩니다.**  
  예: 레포 루트에서 `bash src/scripts/run_grpo_uvb_answer_only.sh`

### 5.2 학습/테스트 데이터 경로

- `TRAIN_FILE`, `TEST_FILE` 기본값은 **`src/r1-v` 기준** 상대 경로  
  `../../data/urban_video_bench/grpo/uvb_grpo_train.jsonl` 등입니다.
- 따라서 스크립트를 “레포 루트에서” 실행하면 `src/r1-v`로 이동한 뒤 상대 경로가 맞습니다.  
  다른 위치에서 스크립트만 호출해도 `REPO_ROOT` 덕분에 동작합니다.

### 5.3 GPU / vLLM

- **단일 GPU**: vLLM은 학습과 같은 `cuda:0`을 공유합니다.  
  OOM이 나면 `NUM_GENERATIONS`, `VLLM_GPU_UTIL`, `MAX_PIXELS` 등을 줄이세요.
- **다중 GPU (예: A40 6대)**:  
  스크립트가 자동으로 `TRAIN_NUM_GPUS = NUM_GPUS - 1`로 두어, 학습 5 + vLLM 1로 동작합니다.  
  별도 설정 없이 실행하면 됩니다.

### 5.4 HuggingFace 모델 다운로드

- `Qwen/Qwen2.5-VL-3B-Instruct`를 처음 불러올 때 HF에서 내려받습니다.
- 네트워크/방화벽 문제나 gated 모델이면 로그인/토큰 필요합니다.

### 5.5 OOM (Out of Memory)

- **조정 권장**: `NUM_GENERATIONS` 감소(예: 8→4), `MAX_PIXELS`/`MAX_PROMPT_LENGTH`/`MAX_COMPLETION_LENGTH` 감소, `VLLM_GPU_UTIL` 조정.
- DeepSpeed 설정은 기본 `configs/zero1_no_optimizer.json`.  
  메모리가 부족하면 `uvb_recover_and_train_a100.sh`에서 쓰는 식으로 zero2 offload 등을 검토할 수 있습니다.

### 5.6 configs 경로

- `DS_CONFIG` 기본값은 `./configs/zero1_no_optimizer.json`이고,  
  현재 디렉터리가 `src/r1-v`이므로 `src/r1-v/configs/zero1_no_optimizer.json`을 찾습니다.  
  정상 클론/설치라면 문제 없습니다.

---

## 6. 기타

### 6.1 데이터 경로가 절대 경로인 경우

- GRPO JSONL의 `frames` 필드는 **절대 경로**로 씁니다 (`str(p.resolve())`).  
  학습 시 다른 머신/컨테이너로 옮기면 해당 머신에 같은 경로가 없어 이미지를 못 찾을 수 있습니다.
- **대처**: 같은 환경에서 데이터 준비와 학습을 하거나, 상대 경로로 바꾸는 후처리/스크립트를 두거나, 데이터를 그 환경에 다시 준비합니다.

### 6.2 MASTER_PORT 충돌

- 여러 실험을 동시에 돌리면 `MASTER_PORT`가 겹쳐서 분산 학습이 꼬일 수 있습니다.
- **대처**: 실험마다 `MASTER_PORT=12347 bash src/scripts/run_grpo_uvb_answer_only.sh` 처럼 다른 포트를 지정합니다.

### 6.3 로그/출력

- 학습 로그는 `OUTPUT_DIR/training_log.txt`와 터미널에 tee로 남습니다.
- `DEBUG_MODE`, `LOG_PATH` 등은 스크립트에서 설정되어 있으므로, 문제 발생 시 해당 로그를 먼저 확인하면 좋습니다.

---

## 요약 체크리스트

| 단계 | 확인 사항 |
|------|-----------|
| 환경 | 레포 루트에서 `setup.sh` 전체 실행, Python 3.10.9+ (권장 3.11), vLLM 0.7.2·flash-attn 설치 확인 |
| 1단계 데이터 | 레포 루트에서 실행, HF 네트워크/로그인, `urban_video_bench_train.jsonl` 생성 확인 |
| 2단계 데이터 | 레포 루트에서 실행, 디스크·HF 토큰·PyAV/libav, `processed/frames` 및 `train_80.jsonl`/`test_20.jsonl` 확인 |
| 3단계 데이터 | `uvb_grpo_train.jsonl`/`uvb_grpo_test.jsonl` 및 `uvb_grpo_summary.json` 확인, 스킵 개수 확인 |
| 학습 | 레포 루트에서 스크립트 실행, GPU 개수·메모리·OOM 시 파라미터 조정, 필요 시 HF 모델 다운로드/로그인 |

이 문서는 `RUNPOD_FLOW.md`, `QUICKSTART.md`와 함께 보면 좋습니다.
