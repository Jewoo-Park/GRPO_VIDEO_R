# Qwen2.5-VL-3B SFT-only Pipeline

이 폴더는 ARM 레포의 SFT 데이터셋(`aqua_rat_multiple_choice`, `aqua_rat_open_form`)을 그대로 사용해서, `Qwen/Qwen2.5-VL-3B-Instruct`를 LoRA SFT 하기 위한 최소 파이프라인입니다.

## Folder structure

- `data/`: SFT 데이터셋(JSON)
- `configs/train_lora_qwen25vl3b.yaml`: 학습 설정
- `configs/merge_lora_qwen25vl3b.yaml`: LoRA 병합 설정
- `scripts/train_sft.py`: 학습 스크립트
- `scripts/merge_lora.py`: LoRA 병합 스크립트

## 1) Environment

```bash
cd sft
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Train (LoRA SFT)

```bash
cd sft
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_sft.py --config configs/train_lora_qwen25vl3b.yaml
# or
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh
```

체크포인트/어댑터는 `outputs/qwen25vl3b_lora_sft`에 저장됩니다.

## 3) Merge LoRA adapter

```bash
cd sft
python scripts/merge_lora.py --config configs/merge_lora_qwen25vl3b.yaml
# or
bash scripts/run_merge.sh
```

병합 모델은 `outputs/qwen25vl3b_lora_merged`에 저장됩니다.

## Notes

- 데이터 포맷은 현재 ARM 레포 SFT 데이터셋과 동일한 `instruction/input/output` JSON list를 그대로 사용합니다.
- Qwen2.5-VL-3B 로딩을 위해 `transformers>=4.51` 기준으로 작성했습니다.
- 이 파이프라인은 텍스트 SFT 기준이며, 추후 이미지 샘플을 추가하려면 데이터 파서를 확장하면 됩니다.
- 출력 태그 기반 학습을 지원합니다: `answer`, `cot`, `long_cot`.
- `CODE` CoT는 기본값으로 학습에서 제외됩니다 (`drop_code_cot: true`).
- `configs/train_lora_qwen25vl3b.yaml`에서 아래를 조정해 형식을 선택할 수 있습니다.
- 기본 LoRA target은 언어 모듈(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)로 고정되어 있습니다.

```yaml
reasoning_formats: [answer, cot, long_cot]
format_mix_strategy: expand
append_format_instruction: true
drop_code_cot: true
```

- 형식별 동작
  - `answer`: `<ANSWER>...</ANSWER>`만 학습
  - `cot`: `<COT>...</COT> + <ANSWER>...</ANSWER>` 학습
  - `long_cot`: `<LONG_COT>...</LONG_COT> + <ANSWER>...</ANSWER>` 학습

## VL 모델에 텍스트 SFT를 해도 되는가?

가능합니다. `Qwen2.5-VL-3B`도 언어 디코더를 포함하므로, 텍스트 샘플만으로 SFT를 진행할 수 있습니다.

다만 이 경우 모델은 주로 텍스트 추론 스타일에 맞춰지고, 시각 태스크 성능은 별도로 좋아지지 않습니다.  
시각 성능까지 유지/개선하려면 이후에 이미지 포함 샘플을 섞어서 추가 SFT를 진행하는 것이 안전합니다.
