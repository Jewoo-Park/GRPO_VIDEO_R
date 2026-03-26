import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from data_to_grpo import convert_single_split
from grpo_data_utils import safe_stem
from video_dataset_prep_utils import (
    download_video_url,
    extract_frames_for_rows,
    find_existing_video_file,
    hf_token_from_env,
    stable_name_from_url,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIGS = ["Adaptation", "Comprehension", "Perception"]


def is_multiple_choice(row: dict) -> bool:
    options = row.get("options")
    return isinstance(options, list) and len(options) >= 2


def sample_rows_by_config(rows: list[dict], sample_ratio: float, seed: int) -> tuple[list[dict], dict[str, dict[str, int]]]:
    if not 0 < sample_ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if sample_ratio >= 1.0:
        counts = Counter(str(row.get("source_subset") or "unknown") for row in rows)
        return rows, {subset: {"total": total, "sampled": total} for subset, total in counts.items()}

    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        subset = str(row.get("source_subset") or "unknown")
        grouped.setdefault(subset, []).append(row)

    sampled_rows: list[dict] = []
    stats: dict[str, dict[str, int]] = {}
    for subset in sorted(grouped.keys()):
        subset_rows = grouped[subset]
        total = len(subset_rows)
        sample_n = min(total, max(1, round(total * sample_ratio)))
        sampled = rng.sample(subset_rows, sample_n)
        sampled_rows.extend(sampled)
        stats[subset] = {"total": total, "sampled": sample_n}
    rng.shuffle(sampled_rows)
    return sampled_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VideoMMMU into the common GRPO evaluation format.")
    parser.add_argument("--dataset-id", type=str, default="lmms-lab/VideoMMMU")
    parser.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-dir", type=str, default="data/video_mmmu/raw")
    parser.add_argument("--processed-dir", type=str, default="data/video_mmmu/processed")
    parser.add_argument("--processed-name", type=str, default="test.jsonl")
    parser.add_argument("--grpo-output-dir", type=str, default="data/video_mmmu/grpo")
    parser.add_argument("--grpo-output-name", type=str, default="videommmu_grpo_test.jsonl")
    parser.add_argument("--summary-name", type=str, default="videommmu_summary.json")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-frame-size", type=int, default=768)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    selected_configs = [config.strip() for config in args.configs.split(",") if config.strip()]
    if not selected_configs:
        raise ValueError("At least one VideoMMMU config must be selected")

    token = hf_token_from_env()
    dataset_dir = Path(args.dataset_dir)
    videos_dir = dataset_dir / "videos"

    filtered_rows: list[dict] = []
    per_config_counts = Counter()
    skipped_rows = 0

    for config_name in selected_configs:
        dataset = load_dataset(args.dataset_id, name=config_name, split=args.split, token=token)
        for row in dataset:
            if not is_multiple_choice(row):
                skipped_rows += 1
                continue
            url = str(row.get("link_selected") or "").strip()
            if not url:
                skipped_rows += 1
                continue

            row_copy = {
                "question_id": row.get("id"),
                "question": str(row.get("question") or "").strip(),
                "options": row.get("options") or [],
                "answer": row.get("answer"),
                "video_url": url,
                "question_category": str(
                    row.get("question_type") or row.get("qa_type") or config_name
                ).strip(),
                "dataset_name": "videommmu",
                "source_subset": config_name,
            }
            filtered_rows.append(row_copy)
            per_config_counts[config_name] += 1

    sampled_rows, sampling_stats = sample_rows_by_config(
        rows=filtered_rows,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )

    url_cache: dict[str, str] = {}
    download_attempted = 0
    download_succeeded = 0

    for row in sampled_rows:
        url = row["video_url"]
        cached_path = url_cache.get(url)
        if cached_path is None:
            output_stem = videos_dir / str(row["source_subset"]) / stable_name_from_url(
                url,
                prefix=safe_stem(str(row["source_subset"]).lower()),
            )
            if args.skip_download:
                downloaded = find_existing_video_file(output_stem)
            else:
                download_attempted += 1
                try:
                    downloaded = download_video_url(url=url, output_stem=output_stem, quiet=False)
                except Exception:
                    downloaded = None
            if downloaded is not None:
                cached_path = str(downloaded.resolve().relative_to(dataset_dir.resolve()))
                url_cache[url] = cached_path
                download_succeeded += 1

        if cached_path is not None:
            row["video_path"] = cached_path
            row["video_id"] = Path(cached_path).name

    prepared_rows = [row for row in sampled_rows if row.get("video_path")]
    processed_dir = Path(args.processed_dir)
    raw_dump_path = dataset_dir / "test.jsonl"
    write_jsonl(raw_dump_path, sampled_rows)

    processed_rows, frame_stats = extract_frames_for_rows(
        rows=prepared_rows,
        split_name="test",
        video_root=dataset_dir,
        frame_root=processed_dir / "frames",
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
        frame_subdir_builder=lambda row, resolved_video_path: str(
            Path(str(row.get("source_subset") or "unknown")) / safe_stem(resolved_video_path.stem)
        ),
        progress_label="videommmu_test",
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
        "selected_configs": selected_configs,
        "split": args.split,
        "filtered_rows": len(filtered_rows),
        "sampled_rows": len(sampled_rows),
        "prepared_rows": len(prepared_rows),
        "sample_ratio": args.sample_ratio,
        "sampling_seed": args.seed,
        "sampling_stats": sampling_stats,
        "per_config_counts": dict(per_config_counts),
        "skipped_rows": skipped_rows,
        "download_attempted": download_attempted,
        "download_succeeded": download_succeeded,
        "raw_dump_path": str(raw_dump_path.resolve()),
        "processed_path": str(processed_path.resolve()),
        "frame_stats": frame_stats,
        "grpo_summary": grpo_summary,
    }
    write_json(Path(args.grpo_output_dir) / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
