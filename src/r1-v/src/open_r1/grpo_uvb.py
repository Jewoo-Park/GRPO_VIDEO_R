import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset

from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.trainer.utils import get_kbit_device_map, get_quantization_config


@dataclass
## 이 스크립터 전용 인자만 추가됨.
class GRPOVideoScriptArguments(ScriptArguments):
    dataset_name: str = field(default="video_grpo_local", metadata={"help": "Dataset name required by ScriptArguments"})
    reward_funcs: list[str] = field(
        default_factory=lambda: ["answer_accuracy", "answer_format"],
        metadata={"help": "Reward functions for multiple-choice video GRPO. Possible values: answer_accuracy, answer_format"},
    )
    max_pixels: Optional[int] = field(default=12845056, metadata={"help": "Maximum pixels per frame"})
    min_pixels: Optional[int] = field(default=3136, metadata={"help": "Minimum pixels per frame"})
    train_file: str = field(default="", metadata={"help": "Path to GRPO train JSONL"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Optional path to GRPO eval/test JSONL"})
    reward_weights: str = field(
        default="",
        metadata={"help": "Comma-separated reward weights aligned with --reward_funcs (e.g. '2.0,1.0')"},
    )
    answer_accuracy_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Optional weight override for answer_accuracy reward"},
    )
    answer_format_weight: Optional[float] = field(
        default=None,
        metadata={"help": "Optional weight override for answer_format reward"},
    )


GRPOUVBScriptArguments = GRPOVideoScriptArguments


## 답변 파싱 함수
def _extract_answer(text: str) -> str:
    # Accept both <ANSWER>...</ANSWER> and occasional case/tag variants.
    match = re.search(r"<ANSWER[S]?>(.*?)</ANSWER[S]?>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def _extract_choice_letter(text: str) -> str:
    """
    UVB is mostly multiple-choice (A-E). Models sometimes output:
    - just the letter: "A"
    - the full option text: "A. ...", "Option A", etc.
    For both training reward and eval logging, we compare the *choice letter*.
    """
    s = _extract_answer(text)
    # Prefer a standalone A-G letter.
    m = re.search(r"\b([A-G])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: first A-G anywhere (covers "A." patterns).
    m = re.search(r"([A-G])", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return _normalize(s)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def answer_accuracy_reward(completions, solution, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(completion_contents, solution):
        pred = _extract_choice_letter(content)
        gt = _extract_choice_letter(sol)
        reward = 1.0 if pred == gt else 0.0
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "./logs/uvb_grpo_reward.log")
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


# Accepted formats: answer-only, cot+answer, long_cot+answer (all must end with <ANSWER>...</ANSWER>).
_FORMAT_PATTERNS = (
    r"\s*<ANSWER[S]?>.*?</ANSWER[S]?>\s*",
    r"\s*<COT>.*?</COT>\s*<ANSWER[S]?>.*?</ANSWER[S]?>\s*",
    r"\s*<LONG_COT>.*?</LONG_COT>\s*<ANSWER[S]?>.*?</ANSWER[S]?>\s*",
)
_FLAGS = re.DOTALL


def _format_ok(content: str) -> int:
    """Return 1 if content matches one of the allowed formats (answer-only, cot+answer, long_cot+answer)."""
    for pattern in _FORMAT_PATTERNS:
        if re.fullmatch(pattern, content, _FLAGS):
            return 1
    return 0


def answer_format_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if _format_ok(x) else 0.0 for x in completion_contents]


def write_test_predictions_jsonl(examples_completions: list[tuple[dict, str]], output_path: str) -> None:
    """Write per-sample test predictions to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example, pred_raw in examples_completions:
            pred_answer = _extract_choice_letter(pred_raw)
            gt_answer = _extract_choice_letter(example["solution"])
            correct = 1 if pred_answer == gt_answer else 0
            format_ok = _format_ok(pred_raw)
            row = {
                "video_id": example.get("video_id", ""),
                "question_id": example.get("question_id", 0),
                "question_category": example.get("question_category", ""),
                "problem": example.get("problem", ""),
                "pred_raw": pred_raw,
                "pred_answer": pred_answer,
                "gt_answer": gt_answer,
                "correct": correct,
                "format_ok": format_ok,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


reward_funcs_registry = {
    "answer_accuracy": answer_accuracy_reward,
    "answer_format": answer_format_reward,
}


SYSTEM_PROMPT = """You are a video question answering assistant for multiple-choice questions. Use the provided video frames to choose the correct option.

The question will include a list of answer choices labeled by option letters.

Rules:
- Your final answer must be the option letter only.
- Do not output the option text in the final answer.
- If the question is simple and can be answered by direct observation, respond using only:
  <ANSWER>X</ANSWER>
  where X is the correct option letter.
- If the question requires multi-step reasoning, comparison, or cause-effect understanding, first give brief reasoning using:
  <COT>...</COT>
  and then give the final answer using:
  <ANSWER>X</ANSWER>
- For especially difficult questions requiring more detailed reasoning, you may use:
  <LONG_COT>...</LONG_COT><ANSWER>X</ANSWER>
- Your response must always end with <ANSWER>X</ANSWER>.
- Never output anything outside these tags except the reasoning inside <COT> or <LONG_COT>.

Examples:

User: [Video frames] What color is the car in the video?
Options:
A. Red
B. Blue
C. White
D. Black
Assistant: <ANSWER>A</ANSWER>

User: [Video frames] Why did the person in the video turn left after stopping at the sign?
Options:
A. To park the car immediately
B. To avoid a pedestrian crossing in front
C. To follow traffic rules and safely turn at the intersection
D. Because the road ahead was blocked by construction
Assistant: <COT>The person stopped at the sign, checked the surroundings, and then made a left turn. This indicates a deliberate turn at an intersection rather than stopping to park or reacting to an obstacle.</COT><ANSWER>C</ANSWER>

Now answer the question based on the video frames. Select the correct option letter from the provided choices."""


def main(script_args, training_args, model_args):
    # Bridge TRL ModelConfig quantization flags (e.g. --load_in_8bit/--load_in_4bit)
    # into trainer args so model_init_kwargs reach from_pretrained().
    model_init_kwargs = dict(getattr(training_args, "model_init_kwargs", {}) or {})
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_init_kwargs["quantization_config"] = quantization_config
        if getattr(model_args, "load_in_4bit", False):
            model_init_kwargs["device_map"] = get_kbit_device_map()
        elif getattr(model_args, "load_in_8bit", False):
            # Keep it explicit for 8-bit too; mirrors TRL script behavior.
            model_init_kwargs["device_map"] = get_kbit_device_map()
        if getattr(training_args, "use_vllm", False):
            print(
                "[VIDEO-GRPO] Warning: k-bit quantization with vLLM weight sync is experimental in this repo."
            )
    training_args.model_init_kwargs = model_init_kwargs

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    reward_weights = [1.0] * len(script_args.reward_funcs)
    if script_args.reward_weights.strip():
        parsed = [x.strip() for x in script_args.reward_weights.split(",") if x.strip()]
        if len(parsed) != len(reward_weights):
            raise ValueError(
                "reward_weights length must match reward_funcs length "
                f"({len(reward_weights)}), got {len(parsed)}."
            )
        reward_weights = [float(x) for x in parsed]
    if script_args.answer_accuracy_weight is not None:
        for i, name in enumerate(script_args.reward_funcs):
            if name == "answer_accuracy":
                reward_weights[i] = float(script_args.answer_accuracy_weight)
    if script_args.answer_format_weight is not None:
        for i, name in enumerate(script_args.reward_funcs):
            if name == "answer_format":
                reward_weights[i] = float(script_args.answer_format_weight)

    data_files = {"train": script_args.train_file}
    if script_args.test_file:
        data_files["test"] = script_args.test_file
    dataset = load_dataset("json", data_files=data_files)

    def resolve_frames_for_split(split_name: str, base_jsonl_path: Optional[str]) -> None:
        if split_name not in dataset or not base_jsonl_path:
            return
        base_dir = os.path.dirname(os.path.abspath(base_jsonl_path))

        def _resolve(example):
            frames = example.get("frames", [])
            resolved = []
            for frame_path in frames:
                if os.path.isabs(frame_path):
                    resolved.append(frame_path)
                else:
                    resolved.append(os.path.normpath(os.path.join(base_dir, frame_path)))
            example["frames"] = resolved
            return example

        dataset[split_name] = dataset[split_name].map(_resolve)

    resolve_frames_for_split("train", script_args.train_file)
    resolve_frames_for_split("test", script_args.test_file)

    def make_conversation_video(example):
        frame_tokens = [{"type": "image"} for _ in example["frames"]]
        frame_tokens.append({"type": "text", "text": example["problem"]})
        out = {
            "image_vllm": example["frames"],
            "solution": example["solution"],
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": frame_tokens},
            ],
        }
        if "video_id" in example:
            out["video_id"] = example["video_id"]
        if "question_id" in example:
            out["question_id"] = example["question_id"]
        if "question_category" in example:
            out["question_category"] = example["question_category"]
        out["problem"] = example["problem"]
        return out

    dataset = dataset.map(make_conversation_video)

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset and training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        reward_weights=reward_weights,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if script_args.test_file and "test" in dataset:
        examples_completions = trainer.run_test_inference()
        if examples_completions:
            out_path = os.path.join(training_args.output_dir, "test_predictions.jsonl")
            write_test_predictions_jsonl(examples_completions, out_path)
            print(f"[VIDEO-GRPO] Test predictions saved to {out_path}")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOVideoScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
