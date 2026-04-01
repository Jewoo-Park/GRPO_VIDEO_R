#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def remap_adapter_keys_and_prepare_dir(adapter_name_or_path: str) -> str:
    """
    Remap known adapter key mismatches (e.g. language_model.layers, visual.blocks)
    so PeftModel.from_pretrained can load without 'missing adapter keys' warnings.
    Writes remapped adapter to a temp dir and returns that path.
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        raise ImportError("safetensors is required for remap_adapter_keys. pip install safetensors")
    adapter_path = os.path.abspath(adapter_name_or_path)
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(safetensors_path):
        return adapter_name_or_path
    sd = load_file(safetensors_path)
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace(".model.model.language_model.layers.", ".model.model.layers.")
        nk = nk.replace(".model.model.visual.blocks.", ".model.visual.blocks.")
        new_sd[nk] = v
    tmpdir = tempfile.mkdtemp(prefix="merge_lora_remap_")
    try:
        save_file(new_sd, os.path.join(tmpdir, "adapter_model.safetensors"))
        for fn in ("adapter_config.json", "README.md"):
            src = os.path.join(adapter_path, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmpdir, fn))
        return tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", type=str, default=None, help="Path to merge config YAML")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="Base model or merged SFT model path")
    parser.add_argument("--adapter-name-or-path", type=str, default=None, help="LoRA adapter directory")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to save merged weights")
    parser.add_argument(
        "--remap-adapter-keys",
        type=str,
        default=None,
        help="Override config and remap adapter keys before merge (true/false).",
    )
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_processor_or_tokenizer(model_name_or_path: str, export_dir: str) -> None:
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor.save_pretrained(export_dir)
        return
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(export_dir)


def get_base_model(model_name_or_path: str):
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )


def main() -> None:
    args = parse_args()
    if args.config is not None:
        cfg = load_yaml(args.config)
    else:
        cfg = {}

    model_name_or_path = args.model_name_or_path or cfg.get("model_name_or_path")
    adapter_name_or_path = args.adapter_name_or_path or cfg.get("adapter_name_or_path")
    export_dir = args.export_dir or cfg.get("export_dir")
    remap_adapter_keys = cfg.get("remap_adapter_keys", False)
    if args.remap_adapter_keys is not None:
        remap_adapter_keys = str(args.remap_adapter_keys).strip().lower() in {"1", "true", "yes", "y", "on"}

    if not model_name_or_path or not adapter_name_or_path or not export_dir:
        raise ValueError(
            "model_name_or_path, adapter_name_or_path, and export_dir must be provided "
            "either via --config or direct CLI flags."
        )

    os.makedirs(export_dir, exist_ok=True)

    adapter_to_remove = None
    if remap_adapter_keys:
        adapter_name_or_path = remap_adapter_keys_and_prepare_dir(adapter_name_or_path)
        adapter_to_remove = adapter_name_or_path

    base_model = get_base_model(model_name_or_path)
    peft_model = PeftModel.from_pretrained(base_model, adapter_name_or_path)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(export_dir, safe_serialization=True)
    save_processor_or_tokenizer(model_name_or_path, export_dir)

    if adapter_to_remove and os.path.isdir(adapter_to_remove):
        shutil.rmtree(adapter_to_remove, ignore_errors=True)


if __name__ == "__main__":
    main()
