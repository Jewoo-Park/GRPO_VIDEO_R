import argparse
import json
from pathlib import Path

from data_to_grpo import convert_single_split
from grpo_data_utils import safe_stem
from video_dataset_prep_utils import extract_frames_for_rows, load_jsonl, write_json, write_jsonl


def prepare_generic_test_benchmark(default_dataset_name: str, default_root: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Prepare {default_dataset_name} into the common GRPO evaluation format."
    )
    parser.add_argument("--dataset-name", type=str, default=default_dataset_name)
    parser.add_argument("--input-jsonl", type=str, default=f"data/{default_root}/raw/test.jsonl")
    parser.add_argument("--video-dir", type=str, default=f"data/{default_root}/videos")
    parser.add_argument("--processed-dir", type=str, default=f"data/{default_root}/processed")
    parser.add_argument("--processed-name", type=str, default="test.jsonl")
    parser.add_argument("--grpo-output-dir", type=str, default=f"data/{default_root}/grpo")
    parser.add_argument("--grpo-output-name", type=str, default=f"{default_root}_grpo_test.jsonl")
    parser.add_argument("--summary-name", type=str, default=f"{default_root}_summary.json")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-frame-size", type=int, default=768)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    rows = load_jsonl(input_path)
    prepared_rows = []
    for row in rows:
        row_copy = dict(row)
        row_copy["dataset_name"] = args.dataset_name
        prepared_rows.append(row_copy)

    processed_dir = Path(args.processed_dir)
    processed_rows, frame_stats = extract_frames_for_rows(
        rows=prepared_rows,
        split_name="test",
        video_root=Path(args.video_dir),
        frame_root=processed_dir / "frames",
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
        frame_subdir_builder=lambda row, resolved_video_path: safe_stem(resolved_video_path.stem),
        progress_label=args.dataset_name,
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
        "dataset_name": args.dataset_name,
        "input_rows": len(rows),
        "processed_rows": len(processed_rows),
        "processed_path": str(processed_path.resolve()),
        "frame_stats": frame_stats,
        "grpo_summary": grpo_summary,
    }
    write_json(Path(args.grpo_output_dir) / args.summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare_generic_test_benchmark(default_dataset_name="generic_test_benchmark", default_root="generic_test_benchmark")
