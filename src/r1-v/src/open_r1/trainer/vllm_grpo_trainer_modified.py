# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import importlib
import inspect
from collections import defaultdict
from contextlib import nullcontext, contextmanager
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.utils.data
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer

import copy
from PIL import Image

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
import torch.nn as nn
from torch.utils.data import Sampler

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def _coerce_qwen25_text_config(cfg: Any) -> Any:
    """Normalize Qwen2.5-VL text config for transformers/vLLM compatibility."""
    if cfg is None:
        return cfg

    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        return cfg

    if isinstance(text_cfg, dict):
        text_cfg = Qwen2Config(**text_cfg)
        cfg.text_config = text_cfg

    # Some merged checkpoints keep canonical 3B text architecture only under
    # text_config, while top-level fields can drift to incompatible defaults.
    # Promote key fields to top-level so loaders agree on tensor shapes.
    promote_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rms_norm_eps",
        "hidden_act",
        "initializer_range",
        "tie_word_embeddings",
        "use_cache",
        "attention_dropout",
        "rope_theta",
    )
    for field_name in promote_fields:
        value = getattr(text_cfg, field_name, None)
        if value is not None:
            setattr(cfg, field_name, value)

    return cfg


def _patch_vllm_rope_scaling_conflict() -> None:
    """Make vLLM tolerate legacy/modern rope_scaling key conflicts in HF configs."""
    try:
        import vllm.transformers_utils.config as vllm_config
    except Exception:
        return

    # Newer vLLM versions removed/renamed this helper; skip patch in that case.
    if not hasattr(vllm_config, "patch_rope_scaling_dict"):
        return

    if getattr(vllm_config, "_codex_rope_patch_applied", False):
        return

    original_patch = vllm_config.patch_rope_scaling_dict

    def _patched_patch_rope_scaling_dict(rope_scaling):
        if isinstance(rope_scaling, dict):
            legacy_type = rope_scaling.get("type")
            modern_type = rope_scaling.get("rope_type")
            if legacy_type is not None and modern_type is not None and legacy_type != modern_type:
                rope_scaling = dict(rope_scaling)
                rope_scaling["rope_type"] = legacy_type
            elif legacy_type is not None and modern_type is None:
                rope_scaling = dict(rope_scaling)
                rope_scaling["rope_type"] = legacy_type
        return original_patch(rope_scaling)

    vllm_config.patch_rope_scaling_dict = _patched_patch_rope_scaling_dict
    vllm_config._codex_rope_patch_applied = True


def _build_vllm_profiling_patch():
    """Return a best-effort patch for vLLM profiling check across versions."""
    candidates = [
        ("vllm.worker.worker", "Worker", "_assert_memory_footprint_increased_during_profiling"),
        ("vllm.v1.worker.gpu_worker", "GPUWorker", "_assert_memory_footprint_increased_during_profiling"),
    ]
    for module_name, class_name, method_name in candidates:
        try:
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            if hasattr(worker_cls, method_name):
                return patch.object(worker_cls, method_name, return_value=None)
        except Exception:
            continue
    return nullcontext()

@contextmanager
def _temporary_cuda_device(device: str):
    """
    Temporarily set current CUDA device (torch.cuda.current_device()).
    vLLM/xFormers may allocate some tensors on the *current* device even if the model
    lives on a different one, so we scope the change to vLLM init/generate calls.
    """
    if not isinstance(device, str) or not device.startswith("cuda:") or not torch.cuda.is_available():
        yield
        return
    try:
        idx = int(device.split(":", 1)[1])
    except Exception:
        yield
        return
    prev = torch.cuda.current_device()
    torch.cuda.set_device(idx)
    try:
        yield
    finally:
        # Best-effort restore for the training process (usually cuda:0).
        try:
            torch.cuda.set_device(prev)
        except Exception:
            pass


def _peft_state_dict_to_merged_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix_strip: str = "base_model.model.",
    lora_alpha_override: Optional[float] = None,
) -> list[tuple[str, torch.Tensor]]:
    """
    Convert a PEFT model state_dict (with base_layer / lora_A / lora_B keys) into a
    single merged state_dict that vLLM can load (plain weight keys, no .base_layer).
    Merged weight = base_layer + (lora_B @ lora_A) * (lora_alpha / r).
    """
    if not state_dict:
        return []

    def strip_prefix(k: str) -> str:
        if prefix_strip and k.startswith(prefix_strip):
            return k[len(prefix_strip) :]
        if k.startswith("base_model."):
            return k[len("base_model.") :]
        return k

    # Collect keys and decide what to emit
    out: dict[str, torch.Tensor] = {}
    seen_base = set()

    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        short = strip_prefix(key)
        if ".base_layer.weight" in key:
            base_short = short.replace(".base_layer.weight", ".weight")
            lora_a_key = key.replace(".base_layer.weight", ".lora_A.default.weight")
            lora_b_key = key.replace(".base_layer.weight", ".lora_B.default.weight")
            if lora_a_key in state_dict and lora_b_key in state_dict:
                base_w = value
                lora_a = state_dict[lora_a_key]
                lora_b = state_dict[lora_b_key]
                r = lora_a.shape[0]
                lora_alpha = lora_alpha_override if lora_alpha_override is not None else 32
                scale = lora_alpha / max(r, 1)
                merged = base_w + (lora_b @ lora_a) * scale
                out[base_short] = merged.to(dtype=base_w.dtype, device=base_w.device)
            else:
                out[base_short] = value
            seen_base.add(base_short)
            continue
        if ".base_layer.bias" in key:
            base_short = short.replace(".base_layer.bias", ".bias")
            out[base_short] = value
            continue
        if short not in out and not short.endswith(".lora_A.default.weight") and not short.endswith(".lora_B.default.weight"):
            out[short] = value

    return list(out.items())


def _get_lora_alpha_from_model(model: Any) -> Optional[float]:
    """Try to get lora_alpha from a PEFT model for use in merge scaling."""
    if not is_peft_available():
        return None
    if not isinstance(model, PeftModel):
        return None
    cfg = getattr(model, "peft_config", None) or {}
    default = cfg.get("default") if isinstance(cfg, dict) else None
    if default is not None:
        return getattr(default, "lora_alpha", None)
    return None


def _filter_vllm_incompatible_weight_keys(
    items: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """
    Drop bitsandbytes-only metadata keys that vLLM loader does not accept.
    This enables USE_VLLM=true with 8-bit training paths.
    """
    filtered: list[tuple[str, torch.Tensor]] = []
    for key, value in items:
        if key.endswith(".SCB") or key.endswith(".weight_format") or ".SCB." in key:
            continue
        filtered.append((key, value))
    return filtered


class Qwen2VLGRPOVLLMTrainerModified(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        reward_weights: Optional[list[float]] = None,
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            model_type = None
            try:
                model_type = AutoConfig.from_pretrained(model_id).model_type
            except Exception:
                # Fall back to path/name heuristics if config probing fails.
                model_type = None
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            if model_type == "qwen2_vl" or "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif model_type == "qwen2_5_vl" or "Qwen2.5-VL" in model_id:
                model_init_kwargs.pop("use_cache", None)
                cfg = _coerce_qwen25_text_config(AutoConfig.from_pretrained(model_id))
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, config=cfg, **model_init_kwargs
                )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            ref_model_type = None
            try:
                ref_model_type = AutoConfig.from_pretrained(model_id).model_type
            except Exception:
                ref_model_type = None
            if ref_model_type == "qwen2_vl" or "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif ref_model_type == "qwen2_5_vl" or "Qwen2.5-VL" in model_id:
                model_init_kwargs.pop("use_cache", None)
                ref_cfg = _coerce_qwen25_text_config(AutoConfig.from_pretrained(model_id))
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, config=ref_cfg, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            model_type = getattr(getattr(model, "config", None), "model_type", "")
            # Prefer model_type so local paths (e.g. merged SFT) always get AutoProcessor when VL.
            is_multimodal_model = (
                model_type in {"qwen2_vl", "qwen2_5_vl"}
                or "Aria" in model_id
                or "Qwen" in model_id
            )
            if is_multimodal_model:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if model_type in {"qwen2_vl", "qwen2_5_vl"} or "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
                processing_class.tokenizer.padding_side = "left"
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):
                name = reward_func.config._name_or_path.split("/")[-1]
            else:
                name = reward_func.__name__
            self.reward_func_names.append(name)

        # Reward weights (default: uniform). Prefer explicit argument from script.
        # Backward-compatible env var fallback is kept for legacy runs.
        resolved_reward_weights = [1.0] * len(self.reward_funcs)
        if reward_weights is not None:
            if len(reward_weights) != len(resolved_reward_weights):
                raise ValueError(
                    "reward_weights length must match number of reward functions "
                    f"({len(resolved_reward_weights)}), got {len(reward_weights)}."
                )
            resolved_reward_weights = [float(x) for x in reward_weights]
        else:
            weights_env = os.getenv("UVB_REWARD_WEIGHTS", "").strip()
            if weights_env:
                parsed = [x.strip() for x in weights_env.split(",") if x.strip() != ""]
                if len(parsed) != len(resolved_reward_weights):
                    raise ValueError(
                        "UVB_REWARD_WEIGHTS length must match number of reward functions "
                        f"({len(resolved_reward_weights)}), got {len(parsed)}."
                    )
                resolved_reward_weights = [float(x) for x in parsed]

            acc_w_env = os.getenv("UVB_ANSWER_ACCURACY_WEIGHT")
            fmt_w_env = os.getenv("UVB_ANSWER_FORMAT_WEIGHT")
            if acc_w_env is not None:
                for idx, name in enumerate(self.reward_func_names):
                    if name == "answer_accuracy_reward":
                        resolved_reward_weights[idx] = float(acc_w_env)
            if fmt_w_env is not None:
                for idx, name in enumerate(self.reward_func_names):
                    if name == "answer_format_reward":
                        resolved_reward_weights[idx] = float(fmt_w_env)

        self.reward_weights = resolved_reward_weights

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        if hasattr(model, "warnings_issued") and isinstance(model.warnings_issued, dict):
            model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.use_vllm = args.use_vllm
        self.vllm_max_pixels = max_pixels
        self.vllm_min_pixels = min_pixels
        # Newer/older vLLM builds may silently drop max_pixels/min_pixels in
        # mm_processor_kwargs. Keep it opt-in and enforce pixel bounds via
        # local image resizing before generate().
        self.vllm_use_mm_processor_kwargs = (
            os.getenv("VLLM_USE_MM_PROCESSOR_KWARGS", "false").strip().lower()
            == "true"
        )
        self._vllm_device: Optional[str] = None
        # Keep vLLM multimodal batches small to avoid xformers vision kernel crashes.
        self.vllm_prompt_batch_size = max(
            1, int(os.getenv("VLLM_PROMPT_BATCH_SIZE", "1"))
        )
        self.vllm_max_frames = max(
            1, int(os.getenv("VLLM_MAX_FRAMES", "8"))
        )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    # Dedicated vLLM GPU when available (e.g. 6 GPUs → 5 train + 1 vLLM → cuda:5).
                    # Otherwise share cuda:0 with training (e.g. single GPU or TRAIN_NUM_GPUS=6).
                    if self.accelerator.num_processes < torch.cuda.device_count():
                        vllm_device = f"cuda:{self.accelerator.num_processes}"
                    else:
                        vllm_device = "cuda:0"
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == "cuda"
                    and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"cuda:{idx}" for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = _build_vllm_profiling_patch()
                self._vllm_device = vllm_device
                with world_size_patch, profiling_patch, _temporary_cuda_device(vllm_device):
                    _patch_vllm_rope_scaling_conflict()
                    print("vllm is running on: ", vllm_device)
                    print('model_path: ', model.name_or_path)
                    # Match vLLM image-item limit to the actual per-prompt frame cap (default 8 frames).
                    vllm_max_frames = getattr(args, "vllm_max_frames", None) or self.vllm_max_frames
                    max_mm_images = max(1, int(vllm_max_frames))
                    llm_kwargs = {
                        "model": model.name_or_path,
                        "gpu_memory_utilization": self.args.vllm_gpu_memory_utilization,
                        "dtype": torch.bfloat16,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        "enable_prefix_caching": True,
                        "enforce_eager": True,
                        "max_model_len": args.max_prompt_length + args.max_completion_length,
                        "limit_mm_per_prompt": {"image": max_mm_images},
                    }
                    model_type = getattr(getattr(model, "config", None), "model_type", "")
                    is_qwen_vl = (
                        model_type in {"qwen2_vl", "qwen2_5_vl"}
                        or "Qwen2-VL" in model_id
                        or "Qwen2.5-VL" in model_id
                    )
                    if self.vllm_use_mm_processor_kwargs and is_qwen_vl:
                        llm_kwargs["mm_processor_kwargs"] = {
                            "max_pixels": max_pixels,
                            "min_pixels": min_pixels,
                        }

                    # Try old API (`device`) first, then fallback to newer (`device_config`).
                    llm_kwargs["device"] = vllm_device
                    try:
                        self.llm = LLM(**llm_kwargs)
                    except TypeError as exc:
                        if "device" not in str(exc):
                            raise
                        llm_kwargs.pop("device", None)
                        llm_kwargs["device_config"] = vllm_device
                        try:
                            self.llm = LLM(**llm_kwargs)
                        except TypeError as exc2:
                            if "device_config" not in str(exc2):
                                raise
                            llm_kwargs.pop("device_config", None)
                            self.llm = LLM(**llm_kwargs)
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(
                "GRPOVLLMTrainerModified only supports vllm generation, please set --use_vllm True"
            )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _resize_image_to_pixel_bounds(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            return image
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        pixels = width * height
        target_pixels = pixels
        if self.vllm_max_pixels is not None and pixels > self.vllm_max_pixels:
            target_pixels = self.vllm_max_pixels
        elif self.vllm_min_pixels is not None and pixels < self.vllm_min_pixels:
            target_pixels = self.vllm_min_pixels

        if target_pixels == pixels:
            return image

        scale = (target_pixels / float(pixels)) ** 0.5
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return image.resize((new_w, new_h), Image.BICUBIC)

    def _load_image_item(self, item, max_frames=None):
        """Load and optionally subsample images. If item is a list, keep at most max_frames (default: self.vllm_max_frames)."""
        if max_frames is None:
            max_frames = self.vllm_max_frames
        if isinstance(item, str):
            return self._resize_image_to_pixel_bounds(Image.open(item).convert("RGB"))
        if isinstance(item, list):
            frames = [self._load_image_item(x, max_frames=None) for x in item]
            if len(frames) > max_frames:
                step = max(1, len(frames) // max_frames)
                frames = frames[::step][:max_frames]
            return frames
        if isinstance(item, Image.Image):
            return self._resize_image_to_pixel_bounds(item)
        return item

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # We need to keep all columns used in _prepare_inputs and reward_kwargs (e.g. image_vllm, solution).
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "image_vllm",
                "solution",
                "problem",
                "video_id",
                "question_id",
                "question_category",
            ]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        logits_to_keep,
    ):
        B = input_ids.shape[0]
        # All B completions share the same visual input (pixel_values was built with
        # repeat_interleave(B): [p0,p0,..,p0, p1,p1,..,p1, ...]).
        # Recover the original single-copy patches by striding with step B.
        pv_single = pixel_values[::B].to(model.device)
        thw_single = image_grid_thw[::B].to(device=model.device)

        per_token_logps = []
        for i in range(B):
            logits_i = model(
                input_ids[i : i + 1],
                attention_mask=attention_mask[i : i + 1],
                pixel_values=pv_single,
                image_grid_thw=thw_single,
                use_cache=False,
            ).logits  # (1, L, V)
            logits_i = logits_i[:, :-1, :]
            ids_i = input_ids[i : i + 1, -logits_to_keep:]
            logits_i = logits_i[:, -logits_to_keep:, :]
            log_probs = logits_i[0].log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=ids_i[0].unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        # Build reduced-frame examples first so chat template placeholder count
        # matches the actual number of frames passed to the processor.
        normalized_examples = []
        images = []
        for example in inputs:
            reduced = self._load_image_item(example["image_vllm"])
            ex = copy.deepcopy(example)
            ex["image_vllm"] = reduced
            if "image" in ex:
                ex["image"] = reduced
            if "frames" in ex:
                ex["frames"] = reduced
            # Critical: prompt was built with len(example["image_vllm"]) image placeholders (e.g. 16 or 32),
            # but we now pass only len(reduced) images (e.g. 8). Rebuild user content so placeholder count matches.
            if "prompt" in ex and len(ex["prompt"]) >= 2:
                user_content = ex["prompt"][1].get("content", [])
                text_part = ""
                if isinstance(user_content, list):
                    for t in user_content:
                        if isinstance(t, dict) and t.get("type") == "text":
                            text_part = t.get("text", "")
                            break
                if not text_part:
                    text_part = ex.get("problem", "")
                ex["prompt"] = [
                    ex["prompt"][0],
                    {"role": "user", "content": [{"type": "image"} for _ in reduced] + [{"type": "text", "text": text_part}]},
                ]
            normalized_examples.append(ex)
            images.append(reduced)
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in normalized_examples
        ]
        # print(f"prompts_text: {prompts_text}")
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=False,  # TODO: fix this, self.args.ds3_gather_for_generation,
                ) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        state_dict = unwrapped_model._orig_mod.state_dict()
                    else:
                        state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    # PEFT state_dict has base_layer/lora_A/lora_B; vLLM needs merged plain keys.
                    if any(".base_layer." in k for k in state_dict):
                        lora_alpha = _get_lora_alpha_from_model(unwrapped_model)
                        remapped_items = _peft_state_dict_to_merged_state_dict(
                            state_dict, lora_alpha_override=lora_alpha
                        )
                    else:
                        remapped_items = []
                        for key, value in state_dict.items():
                            if key.startswith("base_model.model."):
                                key = key[len("base_model.model.") :]
                            elif key.startswith("base_model."):
                                key = key[len("base_model.") :]
                            remapped_items.append((key, value))
                    llm_model.load_weights(
                        _filter_vllm_incompatible_weight_keys(remapped_items)
                    )
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            all_images = gather_object(images)
            # group into pairs
            all_multimodal_inputs = []

            use_naive_loop_sampling = False
            if use_naive_loop_sampling:
                # in this implementation, one sample will repeat `self.num_generations` times
                # it's not a efficient implementation, but safe to keep sampling diversity
                for prompt, image in zip(all_prompts_text, all_images):
                    for _ in range(self.num_generations):
                        all_multimodal_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                all_completion_ids = [None] * len(all_multimodal_inputs)
                for i in range(self.num_generations):
                    # Get the inputs for the current batch
                    batch_inputs = [all_multimodal_inputs[j] for j in range(i, len(all_multimodal_inputs), self.num_generations)]
                    if self.accelerator.is_main_process:
                        outputs = self.llm.generate(
                            batch_inputs,
                            sampling_params=self.sampling_params,
                            use_tqdm=False,
                        )
                        batch_completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                    else:
                        batch_completion_ids = [None] * len(batch_inputs)
                    # Place the results back into their original positions
                    for idx, completion_id in enumerate(batch_completion_ids):
                        all_completion_ids[i + idx * self.num_generations] = completion_id
                # Final completion IDs
                completion_ids = all_completion_ids

            # 2. Refer to TobiasLee's implementation suggestions
            # this is a better implementation for vLLM sampling.
            for prompt, image in zip(all_prompts_text, all_images):
                all_multimodal_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            # Create sampling params with num_generations
            if self.accelerator.is_main_process:
                # Clone to avoid modifying original params
                sampling_params = copy.deepcopy(self.sampling_params)
                sampling_params.n = self.num_generations
                # Single generate call with all prompts
                outputs = []
                with _temporary_cuda_device(self._vllm_device or "auto"):
                    for start in range(
                        0, len(all_multimodal_inputs), self.vllm_prompt_batch_size
                    ):
                        batch_inputs = all_multimodal_inputs[
                            start : start + self.vllm_prompt_batch_size
                        ]
                        outputs.extend(
                            self.llm.generate(
                                batch_inputs,
                                sampling_params=sampling_params,
                                use_tqdm=False,
                            )
                        )
                # Flatten outputs: [prompt1_gen1, prompt1_gen2, ..., prompt2_gen1, prompt2_gen2, ...]
                completion_ids = [out.token_ids for completion in outputs for out in completion.outputs]
            else:
                completion_ids = [None] * len(all_multimodal_inputs) * self.num_generations
            
            # broadcast and slice
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        else:
            raise ValueError("Only vLLM generation is supported in this version ")

        # below are the same with yifan's code
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        # Repeat visual inputs for each sampled completion (B -> B * num_generations).
        pixel_values = prompt_inputs["pixel_values"].repeat_interleave(self.num_generations, dim=0)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {
                    key: []
                    for key in inputs[0].keys()
                    if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )
        rewards_per_func = gather(rewards_per_func)
        reward_weights = torch.tensor(
            self.reward_weights, dtype=rewards_per_func.dtype, device=rewards_per_func.device
        )
        # Weighted sum of rewards from all reward functions
        rewards = (rewards_per_func * reward_weights.unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )
            self._metrics[f"rewards_weight/{reward_func_name}"].append(
                float(self.reward_weights[i])
            )

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        # print('prompt_ids: ', prompt_ids.size())
        # print('completion_ids: ', completion_ids.size())
        # actual_completion_lengths = completion_mask.sum(dim=1).long()  # Shape: number
        # min_length_index = actual_completion_lengths.argmin().item()
        # completion_ids= torch.cat([completion_ids, completion_ids[min_length_index].unsqueeze(0) ],dim=0)
        # completion_mask = torch.cat([completion_mask, completion_mask[min_length_index].unsqueeze(0) ],dim=0)
        # prompt_ids = torch.cat([prompt_ids, prompt_ids[min_length_index].unsqueeze(0) ],dim=0)
        # prompt_mask = torch.cat([prompt_mask, prompt_mask[min_length_index].unsqueeze(0) ],dim=0)
        # print('afterprompt_ids: ', prompt_ids.size())
        # print('aftercompletion_ids: ', completion_ids.size())
        # print('prompt_mask: ', prompt_mask.size())
        # print('completion_mask: ', completion_mask.size())
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # print('input_ids: ', input_ids.size())
        # print('attention_mask: ', attention_mask.size())
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        # print('pixel_values: ', pixel_values.size())
        # print('image_grid_thw: ', image_grid_thw.size())
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        )

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps)- (ref_per_token_logps - per_token_logps)- 1)

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss

    def run_test_inference(self) -> list[tuple[dict, str]]:
        """Run inference on eval_dataset (main process only), return list of (example_dict, completion_text)."""
        if not self.accelerator.is_main_process or self.eval_dataset is None:
            return []

        # Force sync latest model weights to vLLM
        self._last_loaded_step = -1
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=False,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()
        lora_alpha = _get_lora_alpha_from_model(self.model)
        if any(".base_layer." in k for k in state_dict):
            remapped_items = _peft_state_dict_to_merged_state_dict(
                state_dict, lora_alpha_override=lora_alpha
            )
        else:
            remapped_items = []
            for key, value in state_dict.items():
                if key.startswith("base_model.model."):
                    key = key[len("base_model.model.") :]
                elif key.startswith("base_model."):
                    key = key[len("base_model.") :]
                remapped_items.append((key, value))
        llm_model = (
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        )
        llm_model.load_weights(_filter_vllm_incompatible_weight_keys(remapped_items))
        self._last_loaded_step = self.state.global_step

        sampling_params = copy.deepcopy(self.sampling_params)
        sampling_params.n = 1

        # Test can use more frames than train (e.g. train 8, test 16). Env: VLLM_MAX_FRAMES_EVAL.
        eval_max_frames = int(os.getenv("VLLM_MAX_FRAMES_EVAL", str(self.vllm_max_frames)))

        results = []
        batch_size = 8
        for start in range(0, len(self.eval_dataset), batch_size):
            batch = [self.eval_dataset[i] for i in range(start, min(start + batch_size, len(self.eval_dataset)))]
            # Normalize: reduce frames to eval_max_frames and rebuild prompt so placeholder count matches.
            normalized_batch = []
            for ex in batch:
                reduced = self._load_image_item(ex["image_vllm"], max_frames=eval_max_frames)
                ex_copy = copy.deepcopy(ex)
                ex_copy["image_vllm"] = reduced
                if "prompt" in ex_copy and len(ex_copy["prompt"]) >= 2:
                    user_content = ex_copy["prompt"][1].get("content", [])
                    text_part = ""
                    if isinstance(user_content, list):
                        for t in user_content:
                            if isinstance(t, dict) and t.get("type") == "text":
                                text_part = t.get("text", "")
                                break
                    if not text_part:
                        text_part = ex_copy.get("problem", "")
                    ex_copy["prompt"] = [
                        ex_copy["prompt"][0],
                        {"role": "user", "content": [{"type": "image"} for _ in reduced] + [{"type": "text", "text": text_part}]},
                    ]
                normalized_batch.append(ex_copy)
            images = [x["image_vllm"] for x in normalized_batch]
            prompts_text = [
                maybe_apply_chat_template(ex, self.processing_class)["prompt"]
                for ex in normalized_batch
            ]
            all_multimodal_inputs = [
                {"prompt": p, "multi_modal_data": {"image": img}}
                for p, img in zip(prompts_text, images)
            ]
            outputs = []
            with _temporary_cuda_device(self._vllm_device or "auto"):
                for chunk_start in range(
                    0, len(all_multimodal_inputs), self.vllm_prompt_batch_size
                ):
                    chunk_inputs = all_multimodal_inputs[
                        chunk_start : chunk_start + self.vllm_prompt_batch_size
                    ]
                    outputs.extend(
                        self.llm.generate(
                            chunk_inputs,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )
                    )
            completion_ids = [out.token_ids for completion in outputs for out in completion.outputs]
            completion_texts = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            for i, ex in enumerate(batch):
                example_dict = {
                    "video_id": ex.get("video_id", ""),
                    "question_id": ex.get("question_id", 0),
                    "question_category": ex.get("question_category", ""),
                    "problem": ex.get("problem", ""),
                    "solution": ex.get("solution", ""),
                }
                results.append((example_dict, completion_texts[i]))
        return results

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
