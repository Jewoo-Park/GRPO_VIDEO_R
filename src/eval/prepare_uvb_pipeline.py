import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from data_to_grpo import convert_single_split
from grpo_data_utils import safe_stem
from video_dataset_prep_utils import (
    download_dataset_files,
    extract_frames_for_rows,
    hf_token_from_env,
    write_json,
    write_jsonl,
)


def sample_rows_by_category(rows: list[dict], sample_ratio: float, seed: int) -> tuple[list[dict], dict[str, dict[str, int]]]:
    if not 0 < sample_ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if sample_ratio >= 1.0:
        counts = Counter(str(row.get("question_category") or "unknown") for row in rows)
        return rows, {category: {"total": total, "sampled": total} for category, total in counts.items()}

    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        category = str(row.get("question_category") or "unknown")
        grouped.setdefault(category, []).append(row)

    sampled_rows: list[dict] = []
    stats: dict[str, dict[str, int]] = {}
    for category in sorted(grouped.keys()):
        category_rows = grouped[category]
        total = len(category_rows)
        sample_n = min(total, max(1, round(total * sample_ratio)))
        sampled = rng.sample(category_rows, sample_n)
        sampled_rows.extend(sampled)
        stats[category] = {"total": total, "sampled": sample_n}
    rng.shuffle(sampled_rows)
    return sampled_rows, stats


def normalize_rows(dataset) -> tuple[list[dict], dict[str, int]]:
    normalized: list[dict] = []
    per_category_counts = Counter()

    for row in dataset:
        video_id = str(row.get("video_id") or "").strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        question_category = str(row.get("question_category") or "UVB").strip() or "UVB"
        question_id = row.get("Question_id")
        if question_id is None or question_id == "":
            question_id = row.get("question_id")
        if not video_id or not question or not answer:
            continue

        normalized.append(
            {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "video_id": video_id,
                "video_path": f"videos/{video_id}",
                "question_category": question_category,
                "dataset_name": "urban_video_bench",
            }
        )
        per_category_counts[question_category] += 1

    return normalized, dict(per_category_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Urban Video Bench into the common GRPO evaluation format.")
    parser.add_argument("--dataset-id", type=str, default="EmbodiedCity/UrbanVideo-Bench")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--video-dir", type=str, default="data/urban_video_bench")
    parser.add_argument("--output-dir", type=str, default="data/urban_video_bench/processed")
    parser.add_argument("--processed-name", type=str, default="test.jsonl")
    parser.add_argument("--grpo-output-dir", type=str, default="data/urban_video_bench/grpo")
    parser.add_argument("--grpo-output-name", type=str, default="uvb_grpo_test.jsonl")
    parser.add_argument("--summary-name", type=str, default="uvb_summary.json")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-frame-size", type=int, default=768)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    token = hf_token_from_env()
    video_dir = Path(args.video_dir)
    processed_dir = Path(args.output_dir)

    dataset = load_dataset(args.dataset_id, split=args.split, token=token)
    filtered_rows, per_category_counts = normalize_rows(dataset)
    sampled_rows, sampling_stats = sample_rows_by_category(
        rows=filtered_rows,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )

    if not args.skip_download:
        video_patterns = sorted(
            {f"videos/{row['video_id']}" for row in sampled_rows if str(row.get("video_id") or "").strip()}
        )
        download_dataset_files(
            dataset_id=args.dataset_id,
            local_dir=video_dir,
            allow_patterns=video_patterns,
            token=token,
        )

    raw_dump_path = video_dir / "test.jsonl"
    write_jsonl(raw_dump_path, sampled_rows)

    processed_rows, frame_stats = extract_frames_for_rows(
        rows=sampled_rows,
        split_name="test",
        video_root=video_dir,
        frame_root=processed_dir / "frames",
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
        frame_subdir_builder=lambda row, resolved_video_path: str(
            Path(safe_stem(str(row.get("question_category") or "uvb"))) / safe_stem(resolved_video_path.stem)
        ),
        progress_label="uvb_test",
    )

    processed_path = processed_dir / args.processed_name
    write_jsonl(processed_path, processed_rows)

    grpo_summary = convert_single_split(
        input_path=processed_path,
        split_name="test",
        output_dir=Path(args.grpo_output_dir),
        output_name=args.grpo_output_name,
        frames_root=processed_dir / "frames",
    )

    summary = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "filtered_rows": len(filtered_rows),
        "sampled_rows": len(sampled_rows),
        "sample_ratio": args.sample_ratio,
        "sampling_seed": args.seed,
        "sampling_stats": sampling_stats,
        "per_category_counts": per_category_counts,
        "raw_dump_path": str(raw_dump_path.resolve()),
        "processed_path": str(processed_path.resolve()),
        "frame_stats": frame_stats,
        "grpo_summary": grpo_summary,
    }
    write_json(Path(args.grpo_output_dir) / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
