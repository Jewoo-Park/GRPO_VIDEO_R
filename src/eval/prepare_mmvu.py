import argparse
import json
import random
from pathlib import Path
from urllib.parse import urlparse

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


def is_multiple_choice(row: dict) -> bool:
    return str(row.get("question_type") or "").strip().lower() == "multiple-choice"


def sample_rows(rows: list[dict], sample_ratio: float, seed: int) -> tuple[list[dict], dict[str, int]]:
    if not 0 < sample_ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    total = len(rows)
    if sample_ratio >= 1.0 or total == 0:
        return rows, {"total": total, "sampled": total}
    rng = random.Random(seed)
    sample_n = min(total, max(1, round(total * sample_ratio)))
    sampled = rng.sample(rows, sample_n)
    return sampled, {"total": total, "sampled": sample_n}


def choices_to_list(choices: object) -> list[str]:
    if isinstance(choices, dict):
        ordered: list[str] = []
        for key in sorted(choices.keys()):
            value = str(choices[key] or "").strip()
            if value:
                ordered.append(f"{key}. {value}")
        return ordered
    if isinstance(choices, list):
        return [str(item).strip() for item in choices if str(item).strip()]
    return []


def derive_video_path(row: dict) -> str:
    video_path = str(row.get("video_path") or "").strip()
    if video_path:
        return video_path.lstrip("./")

    video_url = str(row.get("video") or "").strip()
    if not video_url:
        return ""

    marker = "/resolve/main/"
    if marker in video_url:
        return video_url.split(marker, 1)[1].lstrip("/")

    parsed = urlparse(video_url)
    return parsed.path.lstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MMVU multiple-choice validation rows into the common GRPO evaluation format.")
    parser.add_argument("--dataset-id", type=str, default="yale-nlp/MMVU")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--dataset-dir", type=str, default="data/mmvu/raw")
    parser.add_argument("--processed-dir", type=str, default="data/mmvu/processed")
    parser.add_argument("--processed-name", type=str, default="test.jsonl")
    parser.add_argument("--grpo-output-dir", type=str, default="data/mmvu/grpo")
    parser.add_argument("--grpo-output-name", type=str, default="mmvu_grpo_test.jsonl")
    parser.add_argument("--summary-name", type=str, default="mmvu_summary.json")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-frame-size", type=int, default=768)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    token = hf_token_from_env()
    dataset_dir = Path(args.dataset_dir)

    dataset = load_dataset(args.dataset_id, split=args.split, token=token)

    filtered_rows: list[dict] = []
    skipped_rows = 0

    for row in dataset:
        if not is_multiple_choice(row):
            skipped_rows += 1
            continue

        video_path = derive_video_path(row)
        options = choices_to_list(row.get("choices"))
        if not video_path or len(options) < 2:
            skipped_rows += 1
            continue

        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        question_category = str(
            metadata.get("subfield") or metadata.get("field") or metadata.get("domain") or "MMVU"
        ).strip()

        filtered_rows.append(
            {
                "question_id": row.get("id"),
                "question": str(row.get("question") or "").strip(),
                "options": options,
                "answer": row.get("answer"),
                "path": video_path,
                "video_path": video_path,
                "video_id": Path(video_path).name,
                "question_category": question_category or "MMVU",
                "dataset_name": "mmvu_mc",
            }
        )

    sampled_rows, sampling_stats = sample_rows(
        rows=filtered_rows,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )

    if not args.skip_download:
        allow_patterns = sorted({row["video_path"] for row in sampled_rows})
        download_dataset_files(
            dataset_id=args.dataset_id,
            local_dir=dataset_dir,
            allow_patterns=allow_patterns,
            token=token,
        )

    processed_dir = Path(args.processed_dir)
    raw_dump_path = dataset_dir / "test.jsonl"
    write_jsonl(raw_dump_path, sampled_rows)

    processed_rows, frame_stats = extract_frames_for_rows(
        rows=sampled_rows,
        split_name="test",
        video_root=dataset_dir,
        frame_root=processed_dir / "frames",
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
        frame_subdir_builder=lambda row, resolved_video_path: str(
            Path(safe_stem(str(row.get("question_category") or "mmvu"))) / safe_stem(resolved_video_path.stem)
        ),
        progress_label="mmvu_test",
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
        "skipped_rows": skipped_rows,
        "raw_dump_path": str(raw_dump_path.resolve()),
        "processed_path": str(processed_path.resolve()),
        "frame_stats": frame_stats,
        "grpo_summary": grpo_summary,
    }
    write_json(Path(args.grpo_output_dir) / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
