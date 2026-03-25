import argparse
import json
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def resize_image_to_pixel_bounds(
    image: Image.Image, max_pixels: int | None, min_pixels: int | None
) -> Image.Image:
    if not isinstance(image, Image.Image):
        return image
    width, height = image.size
    if width <= 0 or height <= 0:
        return image

    pixels = width * height
    target_pixels = pixels
    if max_pixels is not None and pixels > max_pixels:
        target_pixels = max_pixels
    elif min_pixels is not None and pixels < min_pixels:
        target_pixels = min_pixels

    if target_pixels == pixels:
        return image

    scale = (target_pixels / float(pixels)) ** 0.5
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    payload = {
        "sessionId": "0a9c50",
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    try:
        with open(
            "/Users/jw246/Desktop/NTU COSMO LAB/cloned Repos/GRPO_Video/.cursor/debug-0a9c50.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def extract_answer(text: str) -> str:
    # Accept both <ANSWER>...</ANSWER> and occasional case/tag variants.
    m = re.search(r"<ANSWER[S]?>(.*?)</ANSWER[S]?>", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()

def extract_choice_letter(text: str) -> str:
    s = extract_answer(text)
    m = re.search(r"\b([A-G])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"([A-G])", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return norm(s)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_reasoning_type(text: str) -> str:
    if re.search(r"<COT>.*?</COT>", text, re.DOTALL):
        return "cot_tag"
    if re.search(r"<LONG_COT>.*?</LONG_COT>", text, re.DOTALL):
        return "long_cot_tag"
    return "none"


def extract_reasoning_text(text: str) -> str:
    m = re.search(r"<COT>(.*?)</COT>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<LONG_COT>(.*?)</LONG_COT>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def text_stats(text: str, tokenizer) -> dict:
    words = re.findall(r"\S+", text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "chars": len(text),
        "words": len(words),
        "tokens": len(token_ids),
    }


_FORMAT_PATTERNS = (
    r"\s*<ANSWER[S]?>.*?</ANSWER[S]?>\s*",
    r"\s*<COT>.*?</COT>\s*<ANSWER[S]?>.*?</ANSWER[S]?>\s*",
    r"\s*<LONG_COT>.*?</LONG_COT>\s*<ANSWER[S]?>.*?</ANSWER[S]?>\s*",
)


def format_ok(text: str) -> int:
    return int(any(re.fullmatch(p, text, re.DOTALL) for p in _FORMAT_PATTERNS))


def load_rows(path: Path, max_samples: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UVB GRPO checkpoint with vLLM inference.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=3136)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--frames-per-sample", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=501760)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-preds", type=str, default="")
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()
    run_id = f"eval_{int(time.time())}"

    test_path = Path(args.test_file)
    rows = load_rows(test_path, args.max_samples)
    if len(rows) == 0:
        raise SystemExit(f"No rows found in {test_path}")
    base_dir = test_path.resolve().parent
    # region agent log
    _agent_debug_log(
        hypothesis_id="H4",
        location="src/eval/uvb_eval_only.py:main:args",
        message="Eval config and dataset loaded",
        data={
            "frames_per_sample": args.frames_per_sample,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "total_rows": len(rows),
        },
        run_id=run_id,
    )
    # endregion

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 16},
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_completion_length, n=1)

    correct = 0
    format_ok_count = 0
    total = 0
    preds = []
    should_collect_rows = bool(args.save_preds or args.save_json)
    completion_chars = 0
    completion_words = 0
    completion_tokens = 0
    reasoning_chars = 0
    reasoning_words = 0
    reasoning_tokens = 0
    reasoning_types = Counter()
    has_reasoning = 0
    pred_letter_counts = Counter()
    gt_letter_counts = Counter()
    format_class_counts = Counter()
    strict_tag_letter_re = re.compile(r"^\s*<ANSWER[S]?>\s*[A-G]\s*</ANSWER[S]?>\s*$", re.IGNORECASE | re.DOTALL)
    broken_tag_re = re.compile(r"<ANSWER[S]?>|</ANSWER[S]?>", re.IGNORECASE)

    for idx, row in enumerate(rows):
        problem = row["problem"]
        resolved_frame_paths = []
        for p in row["frames"][: args.frames_per_sample]:
            frame_path = Path(p)
            if frame_path.is_absolute():
                resolved_frame_paths.append(frame_path)
            else:
                resolved_frame_paths.append((base_dir / frame_path).resolve())
        frames = [
            resize_image_to_pixel_bounds(
                Image.open(p).convert("RGB"),
                max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
            )
            for p in resolved_frame_paths
        ]
        if idx < 5:
            # region agent log
            _agent_debug_log(
                hypothesis_id="H4",
                location="src/eval/uvb_eval_only.py:main:frame_sampling",
                message="Resolved frame count for sample",
                data={
                    "sample_idx": idx,
                    "question_id": row.get("question_id"),
                    "resolved_frames": len(resolved_frame_paths),
                },
                run_id=run_id,
            )
            # endregion
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are a video question answering assistant. Use the provided frames to answer the question.\n\n"
                    "**Decision rule for reasoning:**\n"
                    "- For SIMPLE questions (direct observation from the video, one-step fact), answer directly. Use only <ANSWER>...</ANSWER>.\n"
                    "- For COMPLEX questions (multi-step reasoning, cause-effect, or comparison), show your reasoning in <COT>...</COT> or <LONG_COT>...</LONG_COT>, then give the final answer in <ANSWER>...</ANSWER>.\n\n"
                    "**Answer format (choose one):**\n"
                    "- Direct: <ANSWER>your answer</ANSWER>\n"
                    "- Short reasoning: <COT>brief reasoning</COT><ANSWER>your answer</ANSWER>\n"
                    "- Long reasoning: <LONG_COT>detailed reasoning</LONG_COT><ANSWER>your answer</ANSWER>\n"
                    "Your response must end with <ANSWER>...</ANSWER>.\n\n"
                    "**Examples:**\n\n"
                    "User: [Video frames] What color is the car in the video?\n"
                    "Assistant: <ANSWER>red</ANSWER>\n"
                    "[Direct observation - no reasoning needed]\n\n"
                    "User: [Video frames] Why did the person in the video turn left after stopping at the sign?\n"
                    "Assistant: <COT>The person stopped at the stop sign, looked both ways, then turned left. This suggests they were following traffic rules and turning at an intersection.</COT><ANSWER>To follow traffic rules and safely turn at the intersection.</ANSWER>\n"
                    "[Requires reasoning about actions and intent]\n\n"
                    "Now answer the question based on the video. Use simple format for direct questions and reasoning format for complex ones."
                )}],
            },
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in frames] + [{"type": "text", "text": problem}]),
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": frames}}], sampling_params=sp, use_tqdm=False)
        text = out[0].outputs[0].text

        pred = extract_choice_letter(text)
        gt = extract_choice_letter(row["solution"])
        ok = int(pred == gt)
        fmt = format_ok(text)
        if strict_tag_letter_re.fullmatch(text):
            format_class = "strict_tag_letter"
        elif broken_tag_re.search(text):
            format_class = "broken_or_partial_tag"
        elif fmt:
            format_class = "tagged_non_strict"
        elif re.match(r"^\s*[A-G]\.", text, re.IGNORECASE):
            format_class = "plain_option_sentence"
        elif re.match(r"^\s*[A-G]\s*$", text, re.IGNORECASE):
            format_class = "plain_letter"
        else:
            format_class = "other"
        reasoning_type = extract_reasoning_type(text)
        reasoning_text = extract_reasoning_text(text)
        full_len = text_stats(text, processor.tokenizer)
        reason_len = text_stats(reasoning_text, processor.tokenizer)

        total += 1
        correct += ok
        format_ok_count += fmt
        completion_chars += full_len["chars"]
        completion_words += full_len["words"]
        completion_tokens += full_len["tokens"]
        reasoning_chars += reason_len["chars"]
        reasoning_words += reason_len["words"]
        reasoning_tokens += reason_len["tokens"]
        reasoning_types[reasoning_type] += 1
        if reasoning_text:
            has_reasoning += 1
        pred_letter_counts[pred] += 1
        gt_letter_counts[gt] += 1
        format_class_counts[format_class] += 1

        if idx < 80:
            # region agent log
            _agent_debug_log(
                hypothesis_id="H1_H2_H3",
                location="src/eval/uvb_eval_only.py:main:sample_eval",
                message="Per-sample eval snapshot",
                data={
                    "sample_idx": idx,
                    "question_id": row.get("question_id"),
                    "pred": pred,
                    "gt": gt,
                    "correct": ok,
                    "format_ok": fmt,
                    "format_class": format_class,
                    "completion_tokens": full_len["tokens"],
                },
                run_id=run_id,
            )
            # endregion

        if should_collect_rows:
            preds.append(
                {
                    "video_id": row.get("video_id"),
                    "question_id": row.get("question_id"),
                    "question_category": row.get("question_category"),
                    "problem": row.get("problem"),
                    "pred_raw": text,
                    "detected_reasoning_type": reasoning_type,
                    "reasoning_text": reasoning_text,
                    "pred_answer": pred,
                    "gt_answer": gt,
                    "correct": ok,
                    "format_ok": fmt,
                    "completion_chars": full_len["chars"],
                    "completion_words": full_len["words"],
                    "completion_tokens": full_len["tokens"],
                    "reasoning_chars": reason_len["chars"],
                    "reasoning_words": reason_len["words"],
                    "reasoning_tokens": reason_len["tokens"],
                }
            )

    metrics = {
        "n": total,
        "answer_accuracy": (correct / total) if total else 0.0,
        "answer_format_rate": (format_ok_count / total) if total else 0.0,
        "reasoning_present_rate": (has_reasoning / total) if total else 0.0,
        "avg_completion_chars": (completion_chars / total) if total else 0.0,
        "avg_completion_words": (completion_words / total) if total else 0.0,
        "avg_completion_tokens": (completion_tokens / total) if total else 0.0,
        "avg_reasoning_chars": (reasoning_chars / total) if total else 0.0,
        "avg_reasoning_words": (reasoning_words / total) if total else 0.0,
        "avg_reasoning_tokens": (reasoning_tokens / total) if total else 0.0,
        "reasoning_type_counts": dict(reasoning_types),
    }
    # region agent log
    _agent_debug_log(
        hypothesis_id="H1_H2_H3_H4",
        location="src/eval/uvb_eval_only.py:main:summary",
        message="Eval summary counters",
        data={
            "metrics": metrics,
            "pred_letter_counts": dict(pred_letter_counts),
            "gt_letter_counts": dict(gt_letter_counts),
            "format_class_counts": dict(format_class_counts),
        },
        run_id=run_id,
    )
    # endregion
    print(metrics)

    if args.save_preds:
        out_path = Path(args.save_preds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in preds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved predictions: {out_path}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": metrics, "results": preds}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"saved report json: {out_path}")


if __name__ == "__main__":
    main()
