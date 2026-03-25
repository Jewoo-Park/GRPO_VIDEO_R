'''
명령어
python src/eval/prepare_video_r1_grpo.py \
  --dataset-id "Video-R1/Video-R1-data" \
  --dataset-dir "data/video_r1/raw" \
  --processed-dir "data/video_r1/processed" \
  --output-dir "data/video_r1/grpo" \
  --sample-ratio 0.3 \
  --seed 42 \
  --download-mode subset-directories


'''


import argparse
import json
import random
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from data_to_grpo import convert_single_split
from grpo_data_utils import frame_key_from_path
from video_dataset_prep_utils import (
    build_video_index,
    download_dataset_files,
    extract_frames_for_rows,
    hf_token_from_env,
    resolve_video_path,
    write_json,
    write_jsonl,
)


DEFAULT_SUBSETS = [
    "LLaVA-Video-178K",
    "NeXT-QA",
    "PerceptionTest",
    "CLEVRER",
    "STAR",
]


def normalize_repo_path(path_text: str) -> str:
    return path_text.strip().lstrip("./")


def download_manifest(dataset_id: str, local_dir: Path, manifest_name: str) -> None:
    download_dataset_files(
        dataset_id=dataset_id,
        local_dir=local_dir,
        allow_patterns=[manifest_name],
        token=hf_token_from_env(),
    )


def download_sampled_video_files(dataset_id: str, local_dir: Path, manifest_name: str, sampled_rows: list[dict]) -> dict[str, int]:
    requested_paths = sorted(
        {
            normalize_repo_path(str(row.get("path") or ""))
            for row in sampled_rows
            if str(row.get("path") or "").strip()
        }
    )
    allow_patterns = [manifest_name, *requested_paths]
    download_dataset_files(
        dataset_id=dataset_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        token=hf_token_from_env(),
    )
    return {"requested_video_files": len(requested_paths)}


def extract_archives(dataset_root: Path) -> dict[str, int]:
    stats = {"zip_archives": 0, "tar_archives": 0, "archives_extracted": 0}
    for archive in dataset_root.rglob("*"):
        if not archive.is_file():
            continue

        suffixes = archive.suffixes
        sentinel = archive.with_name(f".{archive.name}.extracted")
        if sentinel.exists():
            continue

        if archive.suffix.lower() == ".zip":
            stats["zip_archives"] += 1
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(archive.parent)
            sentinel.write_text("", encoding="utf-8")
            stats["archives_extracted"] += 1
            continue

        if suffixes[-2:] == [".tar", ".gz"] or archive.suffix.lower() in {".tar", ".tgz"}:
            stats["tar_archives"] += 1
            with tarfile.open(archive) as tf:
                tf.extractall(archive.parent)
            sentinel.write_text("", encoding="utf-8")
            stats["archives_extracted"] += 1
    return stats


def match_subset(row: dict, subsets: list[str]) -> str | None:
    candidates = [
        str(row.get("data_source") or "").strip(),
        str(row.get("path") or "").strip().lstrip("./"),
    ]
    for subset in subsets:
        for candidate in candidates:
            if candidate == subset or candidate.startswith(f"{subset}/"):
                return subset
    return None


def is_multiple_choice(row: dict) -> bool:
    options = row.get("options")
    if isinstance(options, list) and len(options) >= 2:
        return True
    problem_type = str(row.get("problem_type") or row.get("original_question_type") or "").lower()
    return "multiple" in problem_type and "choice" in problem_type


def sample_rows_by_subset(rows: list[dict], sample_ratio: float, seed: int) -> tuple[list[dict], dict[str, dict[str, int]]]:
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
    parser = argparse.ArgumentParser(
        description="Prepare Video-R1 train data into an intermediate processed split, then convert it with data_to_grpo."
    )
    parser.add_argument("--dataset-id", type=str, default="Video-R1/Video-R1-data")
    parser.add_argument("--dataset-dir", type=str, default="data/video_r1/raw")
    parser.add_argument("--manifest-name", type=str, default="Video-R1-260k.json")
    parser.add_argument("--processed-dir", type=str, default="data/video_r1/processed")
    parser.add_argument("--processed-name", type=str, default="train.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/video_r1/grpo")
    parser.add_argument("--output-name", type=str, default="video_r1_grpo_train.jsonl")
    parser.add_argument("--summary-name", type=str, default="video_r1_grpo_summary.json")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-frame-size", type=int, default=768)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsets", type=str, default=",".join(DEFAULT_SUBSETS))
    parser.add_argument(
        "--download-mode",
        type=str,
        default="sampled-files",
        choices=("sampled-files", "subset-directories"),
        help="sampled-files downloads only sampled video paths; subset-directories downloads whole selected subset folders.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-archive-extract", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_subsets = [subset.strip() for subset in args.subsets.split(",") if subset.strip()]
    if not selected_subsets:
        raise ValueError("At least one subset must be selected")

    download_manifest(
        dataset_id=args.dataset_id,
        local_dir=dataset_dir,
        manifest_name=args.manifest_name,
    )

    archive_stats = {"zip_archives": 0, "tar_archives": 0, "archives_extracted": 0}
    manifest_path = dataset_dir / args.manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dataset = load_dataset("json", data_files=str(manifest_path), split="train")

    per_subset_counts = Counter()
    skipped_non_video_or_non_mc = 0
    filtered_rows: list[dict] = []

    for row in dataset:
        subset = match_subset(row, selected_subsets)
        if subset is None:
            continue
        if str(row.get("data_type") or "").lower() != "video" or not is_multiple_choice(row):
            skipped_non_video_or_non_mc += 1
            continue

        row_copy = dict(row)
        row_copy["question_category"] = subset
        row_copy["source_subset"] = subset
        row_copy["dataset_name"] = "video_r1_train"
        filtered_rows.append(row_copy)
        per_subset_counts[subset] += 1

    sampled_rows, sampling_stats = sample_rows_by_subset(
        rows=filtered_rows,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )

    download_stats = {"requested_video_files": 0}
    if not args.skip_download:
        if args.download_mode == "sampled-files":
            download_stats = download_sampled_video_files(
                dataset_id=args.dataset_id,
                local_dir=dataset_dir,
                manifest_name=args.manifest_name,
                sampled_rows=sampled_rows,
            )
        else:
            download_dataset_files(
                dataset_id=args.dataset_id,
                local_dir=dataset_dir,
                allow_patterns=[args.manifest_name, *(f"{subset}/**" for subset in selected_subsets)],
                token=hf_token_from_env(),
            )

    if not args.skip_archive_extract:
        archive_stats = extract_archives(dataset_dir)

    if args.download_mode == "sampled-files":
        video_index = build_video_index(dataset_dir)
        resolved_count = 0
        for row in sampled_rows:
            if resolve_video_path(row, dataset_dir, video_index) is not None:
                resolved_count += 1
                if resolved_count >= 1:
                    break
        if resolved_count == 0:
            raise RuntimeError(
                "sampled-files mode did not materialize any local videos. "
                "This dataset repository appears to expose archive parts rather than direct mp4 paths. "
                "Rerun with --download-mode subset-directories."
            )

    def build_frame_subdir(row: dict, resolved_video_path: Path) -> str:
        dataset_root = dataset_dir.resolve()
        if resolved_video_path.is_relative_to(dataset_root):
            relative_video = resolved_video_path.relative_to(dataset_root)
        else:
            relative_video = Path(resolved_video_path.name)
        subset = str(row.get("source_subset") or row.get("question_category") or "unknown")
        return str(Path(subset) / frame_key_from_path(relative_video.as_posix()))

    processed_rows, frame_stats = extract_frames_for_rows(
        rows=sampled_rows,
        split_name="train",
        video_root=dataset_dir,
        frame_root=processed_dir / "frames",
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
        frame_subdir_builder=build_frame_subdir,
        progress_label="video_r1_train",
    )

    processed_path = processed_dir / args.processed_name
    write_jsonl(processed_path, processed_rows)

    grpo_summary = convert_single_split(
        input_path=processed_path,
        split_name="train",
        output_dir=output_dir,
        output_name=args.output_name,
        frames_root=processed_dir / "frames",
    )

    summary = {
        "dataset_id": args.dataset_id,
        "manifest": str(manifest_path.resolve()),
        "selected_subsets": selected_subsets,
        "filtered_rows": len(filtered_rows),
        "sampled_rows": len(sampled_rows),
        "sample_ratio": args.sample_ratio,
        "sampling_seed": args.seed,
        "sampling_stats": sampling_stats,
        "download_mode": args.download_mode,
        "download_stats": download_stats,
        "processed_rows": len(processed_rows),
        "per_subset_counts": dict(per_subset_counts),
        "skipped_non_video_or_non_multiple_choice": skipped_non_video_or_non_mc,
        "num_frames_per_video": args.num_frames,
        "max_frame_size": args.max_frame_size,
        "processed_path": str(processed_path.resolve()),
        "archive_stats": archive_stats,
        "frame_stats": frame_stats,
        "grpo_summary": grpo_summary,
    }
    write_json(output_dir / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
