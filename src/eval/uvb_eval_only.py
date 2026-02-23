import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_reasoning_type(text: str) -> str:
    m = re.search(r"\[\s*REASONING_TYPE\s*:\s*([^\]]+)\]", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if re.search(r"<think>.*?</think>", text, re.IGNORECASE | re.DOTALL):
        return "think_tag"
    return "none"


def extract_reasoning_text(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(
        r"\[\s*REASONING_TYPE\s*:[^\]]+\]\s*(.*?)\s*(?:Final\s*Answer\s*:|<answer>|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-preds", type=str, default="")
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    rows = load_rows(test_path, args.max_samples)
    if len(rows) == 0:
        raise SystemExit(f"No rows found in {test_path}")

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
    format_ok = 0
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

    for row in rows:
        problem = row["problem"]
        frames = [Image.open(p).convert("RGB") for p in row["frames"][: args.frames_per_sample]]
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Output only one final answer inside <answer>...</answer>."}],
            },
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in frames] + [{"type": "text", "text": problem}]),
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": frames}}], sampling_params=sp, use_tqdm=False)
        text = out[0].outputs[0].text

        pred = norm(extract_answer(text))
        gt = norm(extract_answer(row["solution"]))
        ok = int(pred == gt)
        fmt = int(re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL) is not None)
        reasoning_type = extract_reasoning_type(text)
        reasoning_text = extract_reasoning_text(text)
        full_len = text_stats(text, processor.tokenizer)
        reason_len = text_stats(reasoning_text, processor.tokenizer)

        total += 1
        correct += ok
        format_ok += fmt
        completion_chars += full_len["chars"]
        completion_words += full_len["words"]
        completion_tokens += full_len["tokens"]
        reasoning_chars += reason_len["chars"]
        reasoning_words += reason_len["words"]
        reasoning_tokens += reason_len["tokens"]
        reasoning_types[reasoning_type] += 1
        if reasoning_text:
            has_reasoning += 1

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
        "answer_format_rate": (format_ok / total) if total else 0.0,
        "reasoning_present_rate": (has_reasoning / total) if total else 0.0,
        "avg_completion_chars": (completion_chars / total) if total else 0.0,
        "avg_completion_words": (completion_words / total) if total else 0.0,
        "avg_completion_tokens": (completion_tokens / total) if total else 0.0,
        "avg_reasoning_chars": (reasoning_chars / total) if total else 0.0,
        "avg_reasoning_words": (reasoning_words / total) if total else 0.0,
        "avg_reasoning_tokens": (reasoning_tokens / total) if total else 0.0,
        "reasoning_type_counts": dict(reasoning_types),
    }
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
