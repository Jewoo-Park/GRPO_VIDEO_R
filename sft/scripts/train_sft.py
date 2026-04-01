#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DEFAULT_TEXT_LORA_SUFFIXES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-VL-3B")
    parser.add_argument("--config", type=str, required=True, help="Path to train config YAML")
    parser.add_argument(
        "--use-vision",
        type=parse_bool,
        default=None,
        help="Override config and enable/disable image/frame inputs.",
    )
    return parser.parse_args()


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_files: List[str]
    output_dir: str
    merge_output_dir: Optional[str] = None
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
    use_vision: bool = False
    max_visual_items: int = 16
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Any = "auto"
    reasoning_formats: List[str] = field(default_factory=lambda: ["answer", "cot", "long_cot"])
    format_mix_strategy: str = "expand"
    append_format_instruction: bool = True
    drop_code_cot: bool = True


SUPPORTED_REASONING_FORMATS = {"answer", "cot", "long_cot"}
FORMAT_INSTRUCTIONS = {
    "answer": "Respond using only <ANSWER>...</ANSWER>.",
    "cot": "Respond using <COT>...</COT> followed by <ANSWER>...</ANSWER>.",
    "long_cot": "Respond using <LONG_COT>...</LONG_COT> followed by <ANSWER>...</ANSWER>.",
}


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = TrainConfig(**raw)
    if cfg.merge_output_dir is None:
        cfg.merge_output_dir = f"{cfg.output_dir}_merged"
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_processor_and_tokenizer(model_name_or_path: str):
    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        processor = None
        tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if processor is not None and getattr(processor, "tokenizer", None) is not None:
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = tokenizer.pad_token
        processor.tokenizer.padding_side = "right"

    return processor, tokenizer


def get_model(model_name_or_path: str, bf16: bool, fp16: bool):
    dtype = None
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16

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


def load_raw_samples(train_files: List[str]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for raw_path in train_files:
        path = Path(raw_path).expanduser().resolve()
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list-like data in {path}")

        for sample in data:
            if not isinstance(sample, dict):
                raise ValueError(f"Expected object samples in {path}")
            item = dict(sample)
            item["__source_path"] = str(path)
            item["__source_dir"] = str(path.parent)
            merged.append(item)
    return merged


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


def extract_tag_block(text: str, tag: str) -> Optional[str]:
    pattern = rf"(<{tag}>\s*.*?\s*</{tag}>)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


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


def build_user_text(instruction: str, user_input: str, format_instruction: str = "") -> str:
    user_text = instruction.strip()
    if user_input.strip():
        user_text = f"{user_text}\n\n{user_input.strip()}"
    if format_instruction.strip():
        user_text = f"{user_text}\n\n{format_instruction.strip()}"
    return user_text


def normalize_media_candidates(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(normalize_media_candidates(item))
        return out
    if isinstance(value, dict):
        for key in ("path", "image", "image_path", "frame"):
            if key in value:
                return normalize_media_candidates(value[key])
    return []


def subsample_paths(paths: List[str], max_items: int) -> List[str]:
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    step = max(1, len(paths) // max_items)
    sampled = paths[::step][:max_items]
    return sampled if sampled else paths[:max_items]


def resolve_visual_paths(sample: Dict[str, Any], max_visual_items: int) -> Tuple[List[str], int, bool]:
    source_dir = Path(str(sample.get("__source_dir", ".")))
    candidates: List[str] = []
    had_visual_field = False
    for key in ("frames", "images", "image", "image_path", "image_vllm"):
        if key in sample:
            had_visual_field = True
            candidates.extend(normalize_media_candidates(sample.get(key)))

    resolved: List[str] = []
    missing_count = 0
    seen = set()
    for text in candidates:
        normalized = text.strip()
        if not normalized:
            continue
        path = Path(normalized)
        if not path.is_absolute():
            path = (source_dir / path).resolve()
        resolved_str = str(path)
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        if path.exists():
            resolved.append(resolved_str)
        else:
            missing_count += 1

    return subsample_paths(resolved, max_visual_items), missing_count, had_visual_field


def parse_sample_fields(
    sample: Dict[str, Any],
    use_vision: bool,
    max_visual_items: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    stats = Counter()
    instruction = str(sample.get("instruction", "")).strip()
    user_input = str(sample.get("input", "")).strip()
    output_text = str(sample.get("output", "")).strip()

    if not instruction or not output_text:
        problem = str(sample.get("problem", "")).strip()
        solution = str(sample.get("solution", "")).strip() or str(sample.get("answer", "")).strip()
        if problem and solution:
            instruction = problem
            if not user_input:
                user_input = str(sample.get("context", "")).strip()
            output_text = solution

    if not instruction or not output_text:
        stats["skip_missing_fields"] += 1
        return None, dict(stats)

    image_paths: List[str] = []
    if use_vision:
        image_paths, missing_media, had_visual_field = resolve_visual_paths(sample, max_visual_items)
        stats["missing_visual_files"] += missing_media
        if had_visual_field and not image_paths:
            stats["skip_missing_visuals"] += 1
            return None, dict(stats)
        if image_paths:
            stats["visual_sample"] += 1
        else:
            stats["text_only_sample"] += 1
    else:
        stats["text_only_sample"] += 1

    return {
        "instruction": instruction,
        "user_input": user_input,
        "output_text": output_text,
        "image_paths": image_paths,
    }, dict(stats)


def build_chat_texts(
    template_source,
    user_text: str,
    answer: str,
    image_paths: List[str],
) -> Tuple[str, str]:
    if image_paths:
        user_content: Any = [{"type": "image"} for _ in image_paths]
        user_content.append({"type": "text", "text": user_text})
    else:
        user_content = user_text

    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer.strip()},
    ]

    prompt_text = template_source.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = template_source.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


def preprocess_samples(
    raw_samples: List[Dict[str, Any]],
    template_source,
    tokenizer,
    config: TrainConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    enabled_formats = normalize_reasoning_formats(config.reasoning_formats)
    stats = Counter()
    processed = []

    for sample in raw_samples:
        parsed, sample_stats = parse_sample_fields(sample, config.use_vision, config.max_visual_items)
        stats.update(sample_stats)
        if parsed is None:
            continue

        targets = build_targets_for_sample(
            output_text=parsed["output_text"],
            enabled_formats=enabled_formats,
            drop_code_cot=config.drop_code_cot,
            format_mix_strategy=config.format_mix_strategy,
        )
        if not targets:
            if config.drop_code_cot and extract_tag_block(parsed["output_text"], "CODE") is not None:
                stats["skip_code_cot"] += 1
            else:
                stats["skip_no_matching_format"] += 1
            continue

        for reasoning_format, answer in targets:
            format_instruction = (
                FORMAT_INSTRUCTIONS[reasoning_format] if config.append_format_instruction else ""
            )
            user_text = build_user_text(
                parsed["instruction"],
                parsed["user_input"],
                format_instruction=format_instruction,
            )
            prompt_text, full_text = build_chat_texts(
                template_source=template_source,
                user_text=user_text,
                answer=answer,
                image_paths=parsed["image_paths"],
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

            processed.append(
                {
                    "prompt_text": prompt_text,
                    "full_text": full_text,
                    "prompt_length": min(len(prompt_ids), len(full_ids)),
                    "image_paths": parsed["image_paths"],
                }
            )
            modality_key = "visual" if parsed["image_paths"] else "text"
            stats[f"kept_{modality_key}_{reasoning_format}"] += 1

    stats["kept_total"] = len(processed)
    return processed, dict(stats)


def select_lora_target_modules(model, configured_targets: Any, use_vision: bool) -> List[str]:
    if isinstance(configured_targets, str) and configured_targets.strip().lower() != "auto":
        if configured_targets.strip().lower() == "all-linear":
            return [
                name
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head")
            ]
        return [configured_targets]

    if isinstance(configured_targets, list):
        return configured_targets

    targets: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.endswith("lm_head"):
            continue
        if use_vision:
            targets.append(name)
        elif any(name.endswith(suffix) for suffix in DEFAULT_TEXT_LORA_SUFFIXES):
            targets.append(name)

    deduped = sorted(set(targets))
    if not deduped:
        raise ValueError("Could not resolve any LoRA target modules from the current model.")
    return deduped


class SupervisedDataCollator:
    def __init__(self, tokenizer, processor, max_length: int, use_vision: bool):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.use_vision = use_vision

    def _load_image_list(self, image_paths: List[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return images

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = [f["full_text"] for f in features]
        prompt_lengths = [int(f["prompt_length"]) for f in features]
        batch_images = [self._load_image_list(f.get("image_paths", [])) for f in features]
        any_images = any(images for images in batch_images)

        common_kwargs = {
            "text": full_texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "add_special_tokens": False,
        }

        if any_images:
            if self.processor is None:
                raise ValueError("Visual SFT requires a processor that supports images.")
            model_inputs = self.processor(images=batch_images, **common_kwargs)
        elif self.processor is not None:
            model_inputs = self.processor(**common_kwargs)
        else:
            model_inputs = self.tokenizer(**common_kwargs)

        labels = model_inputs["input_ids"].clone()
        labels[model_inputs["attention_mask"] == 0] = -100

        seq_len = labels.shape[1]
        for idx, prompt_length in enumerate(prompt_lengths):
            labels[idx, : min(prompt_length, seq_len)] = -100

        model_inputs["labels"] = labels
        return model_inputs


def save_processor_or_tokenizer(processor, tokenizer, output_dir: str) -> None:
    if processor is not None:
        processor.save_pretrained(output_dir)
    else:
        tokenizer.save_pretrained(output_dir)


def train(config: TrainConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.seed)

    processor, tokenizer = get_processor_and_tokenizer(config.model_name_or_path)
    template_source = processor if processor is not None and hasattr(processor, "apply_chat_template") else tokenizer
    model = get_model(config.model_name_or_path, config.bf16, config.fp16)

    target_modules = select_lora_target_modules(
        model=model,
        configured_targets=config.lora_target_modules,
        use_vision=config.use_vision,
    )

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    print(f"[SFT] use_vision: {config.use_vision}")
    print(f"[SFT] resolved LoRA target module count: {len(target_modules)}")

    raw_samples = load_raw_samples(config.train_files)
    processed_samples, preprocess_stats = preprocess_samples(raw_samples, template_source, tokenizer, config)

    print("[SFT] preprocessing stats:")
    for key in sorted(preprocess_stats.keys()):
        print(f"  - {key}: {preprocess_stats[key]}")
    print(f"[SFT] raw samples: {len(raw_samples)}")

    if len(processed_samples) < 2:
        raise ValueError("Need at least 2 processed samples for train/eval split")

    random.shuffle(processed_samples)
    eval_size = max(1, int(math.ceil(len(processed_samples) * config.val_size)))
    eval_size = min(eval_size, len(processed_samples) - 1)

    eval_samples = processed_samples[:eval_size]
    train_samples = processed_samples[eval_size:]
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
        remove_unused_columns=False,
        report_to="none",
    )

    collator = SupervisedDataCollator(
        tokenizer=tokenizer,
        processor=processor,
        max_length=config.max_length,
        use_vision=config.use_vision,
    )

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
    save_processor_or_tokenizer(processor, tokenizer, config.output_dir)


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    if cli_args.use_vision is not None:
        cfg.use_vision = cli_args.use_vision
    train(cfg)
