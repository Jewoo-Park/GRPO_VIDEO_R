# GRPO_Video 레포지토리 구조 및 상세 리뷰

이 문서는 현재 레포의 실제 구현 기준으로 디렉터리 구조, 데이터 흐름, 학습/평가 파이프라인, 핵심 코드 역할을 정리한 문서다.

샘플 검증 과정에서 특정 subset만 부분적으로 내려받아 테스트한 이력은 여기서 다루지 않는다. 이 문서는 최종 운영 기준인 `Video-R1 전체 선택 subset으로 학습하고, UVB로 평가하는 구조`를 기준으로 설명한다.

현재 기준 핵심 방향은 다음과 같다.

- Train Set: Video-R1
- Test Set 1: Urban Video Bench
- Test Set 2: VideoMMMU
- Test Set 3: MMVU (multiple-choice only)
- 최종 데이터 형식: 공통 GRPO JSONL
- 평가 벤치마크는 UVB를 유지

즉 이 레포는 더 이상 UVB만으로 학습하는 구조가 아니라, `Video-R1로 학습하고 UVB로 평가하는 구조`로 바뀌어 있다.

---

## 1. 전체 디렉터리 개요

```text
GRPO_Video/
├── setup.sh                    # 레포 공통 환경 설치
├── merge_readme.md             # SFT LoRA merge 가이드
├── analyze_train_log.ipynb     # 학습 로그 분석 노트북
├── parsed_train_metrics.csv    # 파싱된 학습 메트릭
│
├── data/
│   ├── video_r1/               # Train 데이터셋 (Video-R1)
│   │   ├── raw/                # 원본 manifest 및 다운로드한 subset 파일
│   │   ├── processed/          # 중간 산출물: train.jsonl, frames/train/...
│   │   └── grpo/               # 최종 학습 입력: video_r1_grpo_train.jsonl
│   │
│   └── urban_video_bench/      # Test Set 1 (UVB)
│       ├── processed/          # 중간 산출물: test.jsonl, frames/test/...
│       └── grpo/               # 최종 평가 입력: uvb_grpo_test.jsonl
│   ├── video_mmmu/             # Test Set 2 (VideoMMMU)
│   │   ├── raw/                # 메타데이터 및 URL 다운로드 비디오
│   │   ├── processed/          # 중간 산출물: test.jsonl, frames/test/...
│   │   └── grpo/               # 최종 평가 입력: videommmu_grpo_test.jsonl
│   └── mmvu/                   # Test Set 3 (MMVU mc only)
│       ├── raw/                # 메타데이터 및 HF 내부 비디오 파일
│       ├── processed/          # 중간 산출물: test.jsonl, frames/test/...
│       └── grpo/               # 최종 평가 입력: mmvu_grpo_test.jsonl
│
├── sft/                        # SFT 파이프라인
│   ├── configs/
│   ├── data/
│   ├── scripts/
│   ├── outputs/
│   └── README.md
│
├── src/
│   ├── eval/                   # 데이터셋 준비 및 오프라인 평가
│   │   ├── README.md
│   │   ├── prepare_video_r1_grpo.py
│   │   ├── prepare_uvb_pipeline.py
│   │   ├── prepare_videommmu.py
│   │   ├── prepare_mmvu.py
│   │   ├── data_to_grpo.py
│   │   ├── grpo_data_utils.py
│   │   ├── video_dataset_prep_utils.py
│   │   └── uvb_eval_only.py
│   │
│   ├── r1-v/                   # GRPO 학습 코드 (open_r1 기반)
│   │   ├── configs/
│   │   ├── src/open_r1/
│   │   │   ├── grpo_uvb.py
│   │   │   ├── grpo_video.py
│   │   │   └── trainer/
│   │   └── outputs/
│   │
│   └── scripts/
│       ├── prepare_video_r1_grpo_data.sh
│       ├── prepare_uvb_grpo_data.sh
│       ├── prepare_videommmu_grpo_data.sh
│       ├── prepare_mmvu_grpo_data.sh
│       ├── prepare_uvb_full_split_local_videos.sh
│       ├── run_grpo_uvb_answer_only.sh
│       ├── run_grpo_uvb_answer_only_lora.sh
│       ├── run_sft_grpo_a100x2.sh
│       ├── check_environment.sh
│       └── RUN_GRPO_UVB.md
│
├── docs/
└── analysis_plots/
```

---

## 2. 파이프라인 요약

### 2.1 현재 기준 End-to-End 흐름

```text
Video-R1 원본
-> prepare_video_r1_grpo.py
-> data/video_r1/processed/train.jsonl
-> data/video_r1/processed/frames/train/...
-> data_to_grpo.py
-> data/video_r1/grpo/video_r1_grpo_train.jsonl
-> GRPO 학습

Urban Video Bench 원본
-> prepare_uvb_pipeline.py
-> data/urban_video_bench/processed/test.jsonl
-> data/urban_video_bench/processed/frames/test/...
-> data_to_grpo.py
-> data/urban_video_bench/grpo/uvb_grpo_test.jsonl
-> 평가

VideoMMMU 원본
-> prepare_videommmu.py
-> data/video_mmmu/processed/test.jsonl
-> data/video_mmmu/processed/frames/test/...
-> data_to_grpo.py
-> data/video_mmmu/grpo/videommmu_grpo_test.jsonl
-> 평가

MMVU 원본
-> prepare_mmvu.py
-> data/mmvu/processed/test.jsonl
-> data/mmvu/processed/frames/test/...
-> data_to_grpo.py
-> data/mmvu/grpo/mmvu_grpo_test.jsonl
-> 평가
```

### 2.2 단계별 요약 표

| 단계 | 목적 | 스크립트/모듈 | 입력 | 출력 |
|------|------|----------------|------|------|
| 데이터 준비 (Train) | Video-R1를 학습용으로 전처리 | `prepare_video_r1_grpo.py` → `data_to_grpo.py` | HF `Video-R1/Video-R1-data` | `data/video_r1/grpo/video_r1_grpo_train.jsonl` |
| 데이터 준비 (Test 1) | UVB를 평가용으로 전처리 | `prepare_uvb_pipeline.py` → `data_to_grpo.py` | HF `EmbodiedCity/UrbanVideo-Bench` | `data/urban_video_bench/grpo/uvb_grpo_test.jsonl` |
| SFT | Qwen2.5-VL-3B에 LoRA SFT | `sft/scripts/run_train.sh` | `sft/data/*.json` | `sft/outputs/...` |
| Merge | LoRA 병합 | `sft/scripts/run_merge.sh` | merged 설정 파일 | merged 모델 디렉터리 |
| GRPO | Video-R1 train / UVB test로 학습 및 테스트 추론 | `run_grpo_uvb_answer_only.sh` → `open_r1.grpo_video` | `video_r1_grpo_train.jsonl`, `uvb_grpo_test.jsonl`, merged 모델 | `src/r1-v/outputs/...` |
| Eval | 저장된 모델로 UVB 테스트만 별도 평가 | `uvb_eval_only.py` | `--model`, `--test-file` | 터미널 메트릭, 선택 시 예측/JSON 파일 |

---

## 3. 데이터 형식 상세

### 3.1 최종 학습/평가 입력 형식: GRPO JSONL

최종 JSONL은 Train/Test 구분 없이 같은 스키마를 사용한다.

| 필드 | 설명 |
|------|------|
| `video_id` | 비디오 식별자 또는 원본 path |
| `question_id` | 질문 ID |
| `question_category` | 질문 카테고리 또는 데이터셋/소스 구분 |
| `problem` | 질문과 선택지를 포함한 최종 프롬프트 문자열 |
| `frames` | 프레임 이미지 경로 리스트 |
| `solution` | 정답. 항상 `<ANSWER>...</ANSWER>` 형식으로 정규화 |

예시 개념:

```json
{
  "video_id": "CLEVRER/train_videos/video_00001.mp4",
  "question_id": 123,
  "question_category": "CLEVRER",
  "problem": "Question: ...\nOptions:\nA. ...\nB. ...",
  "frames": [
    "../processed/frames/train/CLEVRER/.../frame_000.jpg",
    "../processed/frames/train/CLEVRER/.../frame_001.jpg"
  ],
  "solution": "<ANSWER>B</ANSWER>"
}
```

중요한 점:

- `frames`는 절대 경로가 아니라 JSONL 파일 위치 기준 상대 경로로 저장된다.
- `data_to_grpo.py`가 이를 최종 규칙에 맞게 통일한다.


### 3.2 중간 산출물 형식: processed JSONL

각 `prepare_*.py`는 먼저 `processed/*.jsonl`을 만든다.

이 파일은 최종 GRPO JSONL 직전 단계이며, 대개 아래 같은 정보를 가진다.

| 필드 | 설명 |
|------|------|
| `path` 또는 `video_id` | 원본 비디오 위치 |
| `problem` 또는 `question` | 질문 텍스트 |
| `solution` 또는 `answer` | 정답 |
| `question_category` | 카테고리 |
| `frame_subdir` | 프레임이 저장된 상대 서브디렉터리 |

즉 `processed/`는 사람이 보기에도 전처리된 중간 문제집이고, `grpo/`는 학습 코드가 바로 읽는 최종 문제집이다.

---

## 4. Train Set: Video-R1

### 4.1 사용 목적

이 레포에서 GRPO 학습용 데이터로 사용한다.

선택된 subset:

- `LLaVA-Video-178K`
- `NeXT-QA`
- `PerceptionTest`
- `CLEVRER`
- `STAR`

### 4.2 입력

- Hugging Face dataset: `Video-R1/Video-R1-data`
- manifest: `Video-R1-260k.json`

### 4.3 출력

중간 산출물:

- `data/video_r1/processed/train.jsonl`
- `data/video_r1/processed/frames/train/...`

최종 산출물:

- `data/video_r1/grpo/video_r1_grpo_train.jsonl`

### 4.4 실제 처리 흐름

`prepare_video_r1_grpo.py`가 하는 일:

1. 먼저 `Video-R1-260k.json`만 다운로드
2. manifest를 읽어 선택한 subset의 row만 필터링
3. `sample-ratio` 적용
4. 비디오 다운로드
5. archive 압축 해제
6. 비디오에서 프레임 추출
7. `processed/train.jsonl` 생성
8. `data_to_grpo.py`를 호출해 `video_r1_grpo_train.jsonl` 생성

### 4.5 다운로드 모드

#### `sampled-files`

- 샘플링된 row에 등장하는 개별 path만 받으려고 시도한다.
- 하지만 현재 Video-R1 저장소는 archive part 중심 구조라, 이 모드는 안정적으로 동작하지 않을 수 있다.
- 실제 로컬 비디오가 생성되지 않으면 fail-fast 하도록 구현되어 있다.

#### `subset-directories`

- 선택한 subset 디렉터리를 통째로 다운로드한다.
- 현재 가장 안정적인 방식이다.
- `sample-ratio`가 0.3이어도 다운로드 자체는 전체 subset 기준이 될 수 있다.
- 다만 최종 학습 row 수는 샘플링 비율이 적용된다.

### 4.6 예시 명령어

전체 train 준비:

```bash
python src/eval/prepare_video_r1_grpo.py \
  --dataset-id "Video-R1/Video-R1-data" \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  --sample-ratio 1 \
  --download-mode subset-directories
```

30% 샘플 train 준비:

```bash
python src/eval/prepare_video_r1_grpo.py \
  --dataset-id "Video-R1/Video-R1-data" \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  --sample-ratio 0.3 \
  --seed 42 \
  --download-mode subset-directories
```

선택 subset 전체를 준비:

```bash
python src/eval/prepare_video_r1_grpo.py \
  --dataset-id "Video-R1/Video-R1-data" \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  --subsets "LLaVA-Video-178K,NeXT-QA,PerceptionTest,CLEVRER,STAR" \
  --sample-ratio 1 \
  --download-mode subset-directories
```

---

## 5. Test Set 1: Urban Video Bench

### 5.1 사용 목적

GRPO 학습 후 테스트/평가 벤치마크로 사용한다.

### 5.2 입력

- Hugging Face dataset: `EmbodiedCity/UrbanVideo-Bench`

### 5.3 출력

중간 산출물:

- `data/urban_video_bench/processed/test.jsonl`
- `data/urban_video_bench/processed/frames/test/...`

최종 산출물:

- `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`

### 5.4 실제 처리 흐름

`prepare_uvb_pipeline.py`가 하는 일:

1. HF dataset split을 JSONL로 export
2. mp4 다운로드
3. 프레임 추출
4. 중간 split JSONL과 프레임 디렉터리 생성
5. 필요 시 `data_to_grpo.py`로 최종 GRPO JSONL 생성

### 5.5 주의

- 현재 학습 실행 경로에서 실제로 필요한 것은 `data/urban_video_bench/grpo/uvb_grpo_test.jsonl` 하나다.
- 즉 운영 기준에서는 UVB를 평가용 benchmark로만 사용한다.
- UVB 준비 스크립트 쪽에는 과거 sampling/split 로직의 흔적이 남아 있을 수 있지만, 학습 실행 관점에서는 이미 준비된 `uvb_grpo_test.jsonl`을 테스트 입력으로 사용하면 된다.

예시 명령어:

```bash
bash src/scripts/prepare_uvb_grpo_data.sh
```

---

## 6. Test Set 2 / Test Set 3

### 6.1 Test Set 2: VideoMMMU

담당 파일:

- `prepare_videommmu.py`

흐름:

```text
HF VideoMMMU metadata
-> 선택 config 로드
-> link_selected URL 비디오 다운로드
-> processed/test.jsonl
-> processed/frames/test/...
-> grpo/videommmu_grpo_test.jsonl
```

주의:

- VideoMMMU는 gated dataset일 수 있다.
- Hugging Face 인증이 필요할 수 있다.
- 비디오 다운로드에는 `yt-dlp`가 필요하다.


### 6.2 Test Set 3: MMVU (multiple-choice only)

담당 파일:

- `prepare_mmvu.py`

흐름:

```text
HF MMVU validation metadata
-> multiple-choice row만 필터링
-> HF 내부 video 파일 다운로드
-> processed/test.jsonl
-> processed/frames/test/...
-> grpo/mmvu_grpo_test.jsonl
```

주의:

- MMVU는 `multiple-choice` 질문만 사용한다.
- 비디오는 Hugging Face dataset 내부 파일을 직접 받으므로 유튜브 다운로드가 필요 없다.


---

## 7. 핵심 코드 설명

### 7.1 `data_to_grpo.py`

역할:

- `processed/*.jsonl` + `processed/frames/...`를 받아 최종 GRPO JSONL을 생성한다.
- Train/Test 두 split을 동시에 처리할 수도 있고, 단일 split만 처리할 수도 있다.

주요 특징:

- `problem`을 공통 포맷으로 정규화
- `solution`을 `<ANSWER>...</ANSWER>`로 정규화
- `frames`를 JSONL 기준 상대 경로로 저장

중요 함수:

- `to_grpo_rows()`
- `convert_single_split()`
- `convert_named_splits()`


### 7.2 `grpo_data_utils.py`

역할:

- 공통 문자열/정답/프레임 경로 처리

주요 기능:

- `normalize_answer()`
- `normalize_problem()`
- `resolve_frame_paths()`
- `pick_video_id()`, `pick_question_id()`, `pick_question_category()`


### 7.3 `video_dataset_prep_utils.py`

역할:

- 데이터셋 다운로드 및 프레임 추출 공통 유틸

주요 기능:

- Hugging Face `snapshot_download`
- mp4 인덱싱
- 비디오 path resolve
- 프레임 추출 및 resize
- JSON/JSONL 기록


### 7.4 `grpo_uvb.py` / `grpo_video.py`

현재 실제 의미는 UVB 전용이라기보다, 공통 비디오 GRPO 진입점에 가깝다.

- `grpo_uvb.py`
  - 기존 이름을 유지한 메인 구현
  - Train/Test JSONL을 읽고 GRPO 트레이너를 호출

- `grpo_video.py`
  - 공용 이름의 alias entrypoint
  - 실제 실행 스크립트에서는 이 경로를 사용하는 것이 자연스럽다

주요 포인트:

- Train/Test 모두 동일한 GRPO JSONL 형식 사용
- `frames`는 JSONL 위치 기준으로 절대 경로로 resolve
- reward는 다지선 정답 일치 여부와 포맷 일치 여부를 사용


### 7.5 `uvb_eval_only.py`

역할:

- 이미 만들어진 테스트 JSONL을 읽어서 모델 추론 후 메트릭을 계산한다.

메트릭 예:

- `answer_accuracy`
- `answer_format_rate`
- `reasoning_present_rate`
- reasoning type counts

즉 이 파일은 데이터 준비 스크립트가 아니라, 순수 평가 실행기다.

---

## 8. 실행 스크립트 관점 정리

### 8.1 Train 데이터 준비

```bash
bash src/scripts/prepare_video_r1_grpo_data.sh
```

또는 옵션 전달:

```bash
bash src/scripts/prepare_video_r1_grpo_data.sh --sample-ratio 0.3 --download-mode subset-directories
```


### 8.2 UVB 평가 데이터 준비

```bash
bash src/scripts/prepare_uvb_grpo_data.sh
```

이미 `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`이 준비돼 있다면 이 단계는 다시 돌릴 필요가 없다.


### 8.3 GRPO 실행

기본 실행 스크립트:

```bash
bash src/scripts/run_grpo_uvb_answer_only.sh
```

현재 기본값:

- Train file: `data/video_r1/grpo/video_r1_grpo_train.jsonl`
- Test file: `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`

즉 기본 동작 자체가 `Video-R1 train + UVB test` 기준이다.

---

## 9. 현재 기준 생성되는 핵심 파일

### Train

```text
data/video_r1/raw/Video-R1-260k.json
data/video_r1/processed/train.jsonl
data/video_r1/processed/frames/train/...
data/video_r1/grpo/video_r1_grpo_train.jsonl
```

### Test Set 1

```text
data/urban_video_bench/processed/test.jsonl
data/urban_video_bench/processed/frames/test/...
data/urban_video_bench/grpo/uvb_grpo_test.jsonl
```

운영 기준에서 실제 평가 입력으로 직접 쓰는 파일은 마지막의 `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`이다.

---

## 10. 중요 요약

현재 레포의 데이터셋 구조를 한 줄로 정리하면 이렇다.

```text
Video-R1로 학습하고, Urban Video Bench로 평가한다.
```

조금 더 정확히 쓰면:

```text
각 데이터셋별 prepare 스크립트는 processed/*.jsonl + processed/frames/...를 만들고,
최종 학습/평가 입력 JSONL은 항상 data_to_grpo.py가 공통 형식으로 만든다.
```

가장 중요한 최종 입력 파일은 아래 두 개다.

- Train: `data/video_r1/grpo/video_r1_grpo_train.jsonl`
- Test: `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`

즉 지금 이 레포에서 “실제로 모델이 읽는 문제집”은 위 두 파일이라고 이해하면 된다.
