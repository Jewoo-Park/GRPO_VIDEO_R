#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-VL-3B")
    parser.add_argument("--config", type=str, required=True, help="Path to train config YAML")
    return parser.parse_args()


### 학습 설정 데이터 클래스
@dataclass
class TrainConfig:
    model_name_or_path: str
    train_files: List[str]
    output_dir: str
    max_length: int = 4096
    val_size: float = 0.05
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Any = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    reasoning_formats: List[str] = field(default_factory=lambda: ["answer", "cot", "long_cot"])
    format_mix_strategy: str = "expand"
    append_format_instruction: bool = True
    drop_code_cot: bool = True


### 지원되는 추론 포맷
SUPPORTED_REASONING_FORMATS = {"answer", "cot", "long_cot"}
FORMAT_INSTRUCTIONS = {
    "answer": "Respond using only <ANSWER>...</ANSWER>.",
    "cot": "Respond using <COT>...</COT> followed by <ANSWER>...</ANSWER>.",
    "long_cot": "Respond using <LONG_COT>...</LONG_COT> followed by <ANSWER>...</ANSWER>.",
}

## YAML을 읽어서 딕셔너리로 만든 뒤, TrainConfig 필드 이름과 맞춰 인자로 넘겨 TrainConfig 인스턴스를 생성
def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


## 토크나이저 로드
def get_tokenizer(model_name_or_path: str):
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(model_name_or_path: str, bf16: bool, fp16: bool):
    dtype = None
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16

    # Qwen2.5-VL class name is not always available depending on transformers version.
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    return model


def load_raw_samples(train_files: List[str]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for path in train_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {path}")
        merged.extend(data)
    return merged


def to_chat_text(
    tokenizer,
    instruction: str,
    user_input: str,
    answer: str,
    format_instruction: str = "",
):
    user_text = instruction.strip()
    if user_input.strip():
        user_text = f"{user_text}\n\n{user_input.strip()}"
    if format_instruction.strip():
        user_text = f"{user_text}\n\n{format_instruction.strip()}"

    prompt_messages = [{"role": "user", "content": user_text}]
    full_messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer.strip()},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return prompt_text, full_text


def normalize_reasoning_formats(formats: List[str]) -> List[str]:
    normalized: List[str] = []
    for fmt in formats:
        key = str(fmt).strip().lower()
        if not key:
            continue
        if key not in SUPPORTED_REASONING_FORMATS:
            raise ValueError(
                f"Unsupported reasoning format: {fmt}. "
                f"Supported: {sorted(SUPPORTED_REASONING_FORMATS)}"
            )
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("reasoning_formats must contain at least one valid format")
    return normalized


## XML 스타일 태그 추출
def extract_tag_block(text: str, tag: str) -> Optional[str]:
    pattern = rf"(<{tag}>\s*.*?\s*</{tag}>)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


## 샘플 처리를 위한 타겟 빌드
def build_targets_for_sample(
    output_text: str,
    enabled_formats: List[str],
    drop_code_cot: bool,
    format_mix_strategy: str,
) -> List[Tuple[str, str]]:
    if drop_code_cot and extract_tag_block(output_text, "CODE") is not None:
        return []

    answer_block = extract_tag_block(output_text, "ANSWER")
    cot_block = extract_tag_block(output_text, "COT")
    long_cot_block = extract_tag_block(output_text, "LONG_COT")

    candidates: List[Tuple[str, str]] = []
    for fmt in enabled_formats:
        if fmt == "answer" and answer_block:
            candidates.append(("answer", answer_block))
        elif fmt == "cot" and cot_block and answer_block:
            candidates.append(("cot", f"{cot_block}\n{answer_block}"))
        elif fmt == "long_cot" and long_cot_block and answer_block:
            candidates.append(("long_cot", f"{long_cot_block}\n{answer_block}"))

    if format_mix_strategy == "single" and candidates:
        return [candidates[0]]
    if format_mix_strategy != "expand":
        raise ValueError("format_mix_strategy must be either 'expand' or 'single'")
    return candidates


def preprocess_samples(
    raw_samples: List[Dict[str, str]],
    tokenizer,
    config: TrainConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    enabled_formats = normalize_reasoning_formats(config.reasoning_formats)
    stats = Counter()
    tokenized = []
    for sample in raw_samples:
        instruction = str(sample.get("instruction", ""))
        user_input = str(sample.get("input", ""))
        output_text = str(sample.get("output", ""))
        if not instruction or not output_text:
            stats["skip_missing_fields"] += 1
            continue

        targets = build_targets_for_sample(
            output_text=output_text,
            enabled_formats=enabled_formats,
            drop_code_cot=config.drop_code_cot,
            format_mix_strategy=config.format_mix_strategy,
        )
        if not targets:
            if config.drop_code_cot and extract_tag_block(output_text, "CODE") is not None:
                stats["skip_code_cot"] += 1
            else:
                stats["skip_no_matching_format"] += 1
            continue

        for reasoning_format, answer in targets:
            format_instruction = (
                FORMAT_INSTRUCTIONS[reasoning_format] if config.append_format_instruction else ""
            )
            prompt_text, full_text = to_chat_text(
                tokenizer,
                instruction,
                user_input,
                answer,
                format_instruction=format_instruction,
            )

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_length,
            )["input_ids"]

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_length,
            )["input_ids"]

            if len(full_ids) == 0:
                stats["skip_empty_tokenized"] += 1
                continue

            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

            tokenized.append(
                {
                    "input_ids": full_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(full_ids),
                }
            )
            stats[f"kept_{reasoning_format}"] += 1

    stats["kept_total"] = len(tokenized)
    return tokenized, dict(stats)


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.seed)

    tokenizer = get_tokenizer(config.model_name_or_path)
    model = get_model(config.model_name_or_path, config.bf16, config.fp16)

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    raw_samples = load_raw_samples(config.train_files)
    tokenized_samples, preprocess_stats = preprocess_samples(raw_samples, tokenizer, config)

    print("[SFT] preprocessing stats:")
    for key in sorted(preprocess_stats.keys()):
        print(f"  - {key}: {preprocess_stats[key]}")
    print(f"[SFT] raw samples: {len(raw_samples)}")

    if len(tokenized_samples) < 2:
        raise ValueError("Need at least 2 tokenized samples for train/eval split")

    random.shuffle(tokenized_samples)
    eval_size = max(1, int(math.ceil(len(tokenized_samples) * config.val_size)))
    eval_size = min(eval_size, len(tokenized_samples) - 1)

    eval_samples = tokenized_samples[:eval_size]
    train_samples = tokenized_samples[eval_size:]
    print(f"[SFT] train samples: {len(train_samples)}")
    print(f"[SFT] eval samples: {len(eval_samples)}")

    train_dataset = Dataset.from_list(train_samples)
    eval_dataset = Dataset.from_list(eval_samples)

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        fp16=config.fp16,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=4,
        report_to="none",
    )

    collator = SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    train(cfg)
