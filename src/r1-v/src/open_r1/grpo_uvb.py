import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOUVBScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["answer_accuracy", "answer_format"],
        metadata={"help": "Reward functions for UVB. Possible values: answer_accuracy, answer_format"},
    )
    max_pixels: Optional[int] = field(default=12845056, metadata={"help": "Maximum pixels per frame"})
    min_pixels: Optional[int] = field(default=3136, metadata={"help": "Minimum pixels per frame"})
    train_file: str = field(default="", metadata={"help": "Path to UVB GRPO train JSONL"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Path to UVB GRPO test JSONL"})


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def answer_accuracy_reward(completions, solution, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(completion_contents, solution):
        pred = _normalize(_extract_answer(content))
        gt = _normalize(_extract_answer(sol))
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards


def answer_format_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    pattern = r"\s*<answer>.*?</answer>\s*"
    return [1.0 if re.fullmatch(pattern, x, re.DOTALL | re.IGNORECASE) else 0.0 for x in completion_contents]


def _format_ok(content: str) -> int:
    pattern = r"\s*<answer>.*?</answer>\s*"
    return 1 if re.fullmatch(pattern, content, re.DOTALL | re.IGNORECASE) else 0


def write_test_predictions_jsonl(examples_completions: list[tuple[dict, str]], output_path: str) -> None:
    """Write per-sample test predictions to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example, pred_raw in examples_completions:
            pred_answer = _normalize(_extract_answer(pred_raw))
            gt_answer = _normalize(_extract_answer(example["solution"]))
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


SYSTEM_PROMPT = (
    "You are a video question answering assistant. Use the provided frames to answer the question. "
    "Output only one final answer inside <answer>...</answer>."
)


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    data_files = {"train": script_args.train_file}
    if script_args.test_file:
        data_files["test"] = script_args.test_file
    dataset = load_dataset("json", data_files=data_files)

    def make_conversation_video(example):
        frame_tokens = [{"type": "image"} for _ in example["frames"]]
        frame_tokens.append({"type": "text", "text": example["problem"]})
        out = {
            "image_vllm": example["frames"],
            "solution": example["solution"],
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
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
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if script_args.test_file and "test" in dataset:
        examples_completions = trainer.run_test_inference()
        if examples_completions:
            out_path = os.path.join(training_args.output_dir, "test_predictions.jsonl")
            write_test_predictions_jsonl(examples_completions, out_path)
            print(f"[UVB-GRPO] Test predictions saved to {out_path}")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name="urban-video-bench-local")


if __name__ == "__main__":
    parser = TrlParser((GRPOUVBScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
