# `src/eval` 데이터셋 준비/평가 파이프라인

## 개요
이 폴더는 이 레포에서 사용하는 비디오 데이터셋을 공통 형식으로 전처리하고, 최종적으로 GRPO 학습/평가에 바로 넣을 수 있는 JSONL을 만드는 역할을 담당한다.

현재 기준 데이터 흐름은 다음과 같다.

- Train Set: Video-R1
- Test Set 1: Urban Video Bench
- Test Set 2: VideoMMMU
- Test Set 3: MMVU (multiple-choice only)

핵심 원칙은 하나다.

1. 데이터셋별 `prepare_*.py`가 먼저 `processed/*.jsonl`과 `processed/frames/...`를 만든다.
2. 마지막 변환은 항상 `data_to_grpo.py`가 담당한다.

즉 공통 구조는 아래와 같다.

```text
원본 데이터셋
-> prepare_*.py
-> processed/*.jsonl + processed/frames/...
-> data_to_grpo.py
-> grpo/*.jsonl
-> 학습 또는 평가
```


## 현재 파일 구성

### 1. 데이터셋별 준비 스크립트

- `prepare_video_r1_grpo.py`
  - Video-R1 훈련 데이터를 준비한다.
  - `Video-R1-260k.json` manifest를 읽고, 선택한 subset에서 샘플링 후 프레임을 추출한다.
  - 중간 산출물 `processed/train.jsonl`을 만들고, 마지막에 `data_to_grpo.py`를 호출해 `video_r1_grpo_train.jsonl`을 생성한다.

- `prepare_uvb_pipeline.py`
  - Urban Video Bench를 단일 평가 벤치마크로 준비한다.
  - HF에서 메타데이터를 export하고, mp4를 다운로드하고, 프레임을 추출한다.
  - 중간 산출물 `processed/test.jsonl`을 만든 뒤, 마지막에 `data_to_grpo.py`를 호출해 `uvb_grpo_test.jsonl`을 생성한다.

- `prepare_videommmu.py`
  - VideoMMMU를 평가용 벤치마크로 준비한다.
  - Hugging Face metadata를 읽고, 각 row의 `link_selected` URL에서 비디오를 내려받고, 프레임을 추출한 뒤 최종 `videommmu_grpo_test.jsonl`을 만든다.
  - VideoMMMU는 gated dataset일 수 있으므로 `HF_TOKEN` 또는 `huggingface-cli login`이 필요할 수 있다.

- `prepare_mmvu.py`
  - MMVU validation split 중 `multiple-choice` 질문만 평가용 벤치마크로 준비한다.
  - Hugging Face metadata를 읽고, row의 `video`/`video_path`에 해당하는 HF 내부 비디오 파일을 다운로드하고, 프레임을 추출한 뒤 최종 `mmvu_grpo_test.jsonl`을 만든다.

### 2. 공통 변환기

- `data_to_grpo.py`
  - 이 폴더의 핵심 변환기다.
  - `processed/*.jsonl`과 `processed/frames/...`를 받아, 최종 GRPO JSONL을 생성한다.
  - Train/Test split 두 개를 동시에 처리할 수도 있고, 단일 split만 처리할 수도 있다.


### 3. 내부 공용 유틸

- `grpo_data_utils.py`
  - 질문/선택지 문자열 정규화
  - 정답 태그 `<ANSWER>...</ANSWER>` 정규화
  - frame path 해석
  - 공통 컬럼(`video_id`, `question_id`, `question_category`) 추출

- `video_dataset_prep_utils.py`
  - Hugging Face dataset 파일 다운로드
  - mp4 탐색 및 매칭
  - 비디오 프레임 추출
  - JSON/JSONL 저장


### 4. 평가 스크립트

- `uvb_eval_only.py`
  - 이미 만들어진 test JSONL을 읽어서 모델 추론 후 정확도와 포맷 메트릭을 계산한다.
  - 데이터 준비 스크립트가 아니라 실제 평가 실행기다.


## 데이터 흐름 상세

### A. Video-R1 Train Set

입력:

- Hugging Face dataset: `Video-R1/Video-R1-data`
- manifest: `Video-R1-260k.json`
- 사용 subset:
  - `LLaVA-Video-178K`
  - `NeXT-QA`
  - `PerceptionTest`
  - `CLEVRER`
  - `STAR`

흐름:

```text
Video-R1 원본
-> manifest 다운로드
-> 선택 subset 필터링
-> sample-ratio 적용
-> 비디오 다운로드
-> 프레임 추출
-> data/video_r1/processed/train.jsonl
-> data/video_r1/processed/frames/train/...
-> data/video_r1/grpo/video_r1_grpo_train.jsonl
```

중간 산출물:

- `data/video_r1/processed/train.jsonl`
- `data/video_r1/processed/frames/train/...`

최종 산출물:

- `data/video_r1/grpo/video_r1_grpo_train.jsonl`


### B. Urban Video Bench Test Set 1

입력:

- Hugging Face dataset: `EmbodiedCity/UrbanVideo-Bench`

흐름:

```text
Urban Video Bench 원본
-> 메타데이터 export
-> mp4 다운로드
-> 프레임 추출
-> data/urban_video_bench/processed/test.jsonl
-> data/urban_video_bench/processed/frames/test/...
-> data/urban_video_bench/grpo/uvb_grpo_test.jsonl
```

중간 산출물:

- `data/urban_video_bench/processed/test.jsonl`
- `data/urban_video_bench/processed/frames/test/...`

최종 산출물:

- `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`

주의:

- 현재 UVB는 train/test split을 따로 만들지 않는다.
- 전체를 하나의 test benchmark로 취급한다.


### C. VideoMMMU Test Set 2

흐름:

```text
VideoMMMU metadata
-> selected config 로드
-> link_selected URL 비디오 다운로드
-> processed/test.jsonl
-> processed/frames/test/...
-> grpo/videommmu_grpo_test.jsonl
```

기본 출력 구조:

```text
data/video_mmmu/raw/test.jsonl
data/video_mmmu/raw/videos/...
data/video_mmmu/processed/test.jsonl
data/video_mmmu/grpo/videommmu_grpo_test.jsonl
```

주의:

- VideoMMMU는 URL 기반 비디오를 내려받기 위해 `yt-dlp`가 필요하다.
- 또한 Hugging Face 접근 권한이 필요한 gated dataset일 수 있다.


### D. MMVU Test Set 3

흐름:

```text
MMVU validation metadata
-> multiple-choice row만 필터링
-> HF dataset 내부 video 파일 다운로드
-> processed/test.jsonl
-> processed/frames/test/...
-> grpo/mmvu_grpo_test.jsonl
```

기본 출력 구조:

```text
data/mmvu/raw/test.jsonl
data/mmvu/raw/videos/...
data/mmvu/processed/test.jsonl
data/mmvu/grpo/mmvu_grpo_test.jsonl
```

주의:

- MMVU는 `multiple-choice` 질문만 사용한다.
- 비디오는 Hugging Face dataset 내부 파일을 직접 받으므로 유튜브 다운로드가 필요 없다.


## `data_to_grpo.py`가 만드는 최종 형식

최종 GRPO JSONL은 각 row가 대략 아래 컬럼을 가진다.

- `video_id`
- `question_id`
- `question_category`
- `problem`
- `frames`
- `solution`

설명:

- `problem`
  - 질문과 선택지를 하나의 문자열로 정리한 값

- `frames`
  - 이미지 경로 리스트
  - 최종 JSONL 파일이 있는 디렉터리를 기준으로 한 상대 경로로 저장됨

- `solution`
  - 정답
  - 항상 `<ANSWER>...</ANSWER>` 형식으로 정규화됨


## 자주 쓰는 명령어

### 1. Video-R1 전체 train 준비

```bash
python src/eval/prepare_video_r1_grpo.py \
  --dataset-id "Video-R1/Video-R1-data" \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  --subsets "LLaVA-Video-178K,NeXT-QA,PerceptionTest,CLEVRER,STAR" \
  --num-frames 16 \
  --sample-ratio 1 \
  --download-mode subset-directories
```

여기서 특정 subset만 받으려면 `--subsets`를 줄이면 된다.
예를 들어 LLaVA를 빼고 싶으면 `--subsets "NeXT-QA,PerceptionTest,CLEVRER,STAR"`처럼 지정하면 된다.
프레임 개수는 `--num-frames 32`처럼 바꿀 수 있다.


### 2. Video-R1 30% 샘플 train 준비

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


### 3. UVB test set 준비

```bash
python src/eval/prepare_uvb_pipeline.py \
  --dataset-id "EmbodiedCity/UrbanVideo-Bench" \
  --video-dir "data/urban_video_bench" \
  --output-dir "data/urban_video_bench/processed" \
  --grpo-output-dir "data/urban_video_bench/grpo"
```


### 4. VideoMMMU test set 준비

```bash
python src/eval/prepare_videommmu.py \
  --dataset-dir "data/video_mmmu/raw" \
  --processed-dir "data/video_mmmu/processed" \
  --grpo-output-dir "data/video_mmmu/grpo"
```


### 5. MMVU test set 준비

```bash
python src/eval/prepare_mmvu.py \
  --dataset-dir "data/mmvu/raw" \
  --processed-dir "data/mmvu/processed" \
  --grpo-output-dir "data/mmvu/grpo"
```


### 4. 이미 만들어진 processed split을 수동 변환

단일 split:

```bash
python src/eval/data_to_grpo.py \
  --input data/urban_video_bench/processed/test.jsonl \
  --split-name test \
  --frames-root data/urban_video_bench/processed/frames \
  --output-dir data/urban_video_bench/grpo \
  --output-name uvb_grpo_test.jsonl
```

pair split:

```bash
python src/eval/data_to_grpo.py \
  --train-input data/some_dataset/processed/train.jsonl \
  --test-input data/some_dataset/processed/test.jsonl \
  --frames-root data/some_dataset/processed/frames \
  --output-dir data/some_dataset/grpo
```


### 5. UVB 오프라인 평가

```bash
python src/eval/uvb_eval_only.py \
  --model /path/to/model \
  --test-file data/urban_video_bench/grpo/uvb_grpo_test.jsonl
```


## Video-R1 다운로드 모드 설명

`prepare_video_r1_grpo.py`에는 현재 두 가지 다운로드 모드가 있다.

### `--download-mode subset-directories`

의미:

- 선택된 subset 디렉터리를 통째로 다운로드한다.
- 현재 가장 안정적으로 동작하는 모드다.

특징:

- `sample-ratio`가 0.3이어도 다운로드 용량은 여전히 클 수 있다.
- 다만 실제 전처리/학습에는 샘플링된 row만 사용한다.


### `--download-mode sampled-files`

의미:

- 샘플링된 row에 등장하는 `path`만 다운로드하려고 시도한다.

현재 상태:

- Video-R1 저장소 구조가 archive part 중심이라 이 모드는 현재 안정적으로 동작하지 않는다.
- 실제 로컬 비디오가 materialize되지 않으면 스크립트가 fail-fast 하도록 되어 있다.
- 현재 Video-R1에는 `subset-directories` 사용을 권장한다.


## 생성되는 주요 파일

### Video-R1

```text
data/video_r1/raw/Video-R1-260k.json
data/video_r1/processed/train.jsonl
data/video_r1/processed/frames/train/...
data/video_r1/grpo/video_r1_grpo_train.jsonl
```


### Urban Video Bench

```text
data/urban_video_bench/urban_video_bench_all.jsonl
data/urban_video_bench/processed/test.jsonl
data/urban_video_bench/processed/frames/test/...
data/urban_video_bench/grpo/uvb_grpo_test.jsonl
```


### Generic Test Benchmark

```text
data/test_set2/raw/test.jsonl
data/test_set2/videos/...
data/test_set2/processed/test.jsonl
data/test_set2/processed/frames/test/...
data/test_set2/grpo/test_set2_grpo_test.jsonl
```


## 요약

이 폴더의 현재 철학은 다음과 같다.

- 데이터셋별 준비는 각각의 `prepare_*.py`가 담당한다.
- 최종 학습/평가 입력 형식은 `data_to_grpo.py`가 일괄 통일한다.
- Train은 Video-R1, Test 1은 UVB, Test 2/3는 확장 가능한 템플릿 구조다.
- `processed/`는 중간 산출물, `grpo/`는 모델이 직접 읽는 최종 산출물이다.

가장 중요한 최종 입력 파일은 아래 두 개다.

- Train: `data/video_r1/grpo/video_r1_grpo_train.jsonl`
- Eval: `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`
