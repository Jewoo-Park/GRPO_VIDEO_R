import argparse
import json
from pathlib import Path

from grpo_data_utils import (
    dump_jsonl,
    load_records,
    normalize_answer,
    normalize_problem,
    pick_question_category,
    pick_question_id,
    pick_video_id,
    relativize_paths,
    resolve_frame_paths,
)


def to_grpo_rows(
    split_rows: list[dict],
    split_name: str,
    source_path: Path,
    frames_root: Path | None,
    output_dir: Path,
) -> tuple[list[dict], int]:
    out: list[dict] = []
    skipped = 0
    for item in split_rows:
        frames = resolve_frame_paths(
            row=item,
            split_name=split_name,
            frames_root=frames_root,
            source_base_dir=source_path.parent.resolve(),
        )
        if len(frames) == 0:
            skipped += 1
            continue

        out.append(
            {
                "video_id": pick_video_id(item),
                "question_id": pick_question_id(item),
                "question_category": pick_question_category(item),
                "problem": normalize_problem(item),
                "frames": relativize_paths(frames, output_dir),
                "solution": normalize_answer(str(item.get("solution") or item.get("answer") or "")),
            }
        )
    return out, skipped


def convert_single_split(
    input_path: Path,
    split_name: str,
    output_dir: Path,
    output_name: str,
    frames_root: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_records(input_path)
    split_out, split_skipped = to_grpo_rows(rows, split_name, input_path, frames_root, output_dir)
    output_path = output_dir / output_name
    dump_jsonl(output_path, split_out)
    return {
        "input_path": str(input_path.resolve()),
        "split_name": split_name,
        "frames_root": str(frames_root.resolve()) if frames_root is not None else "",
        "in": len(rows),
        "out": len(split_out),
        "skipped_no_frames": split_skipped,
        "output_path": str(output_path.resolve()),
    }


def convert_named_splits(
    split_configs: list[tuple[str, Path, str]],
    output_dir: Path,
    frames_root: Path | None = None,
) -> dict:
    summary: dict[str, object] = {
        "frames_root": str(frames_root.resolve()) if frames_root is not None else "",
        "splits": {},
    }
    for split_name, input_path, output_name in split_configs:
        summary["splits"][split_name] = convert_single_split(
            input_path=input_path,
            split_name=split_name,
            output_dir=output_dir,
            output_name=output_name,
            frames_root=frames_root,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert processed split files plus extracted frames into the GRPO JSONL schema."
    )
    parser.add_argument("--processed-dir", type=str, default="data/urban_video_bench/processed")
    parser.add_argument("--output-dir", type=str, default="data/urban_video_bench/grpo")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--split-name", type=str, default="")
    parser.add_argument("--output-name", type=str, default="")
    parser.add_argument("--train-input", type=str, default="")
    parser.add_argument("--test-input", type=str, default="")
    parser.add_argument("--frames-root", type=str, default="")
    parser.add_argument("--train-output-name", type=str, default="uvb_grpo_train.jsonl")
    parser.add_argument("--test-output-name", type=str, default="uvb_grpo_test.jsonl")
    parser.add_argument("--summary-name", type=str, default="uvb_grpo_summary.json")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_root) if args.frames_root else processed / "frames"

    single_input_mode = bool(args.input.strip())
    paired_mode = bool(args.train_input.strip() or args.test_input.strip())

    if single_input_mode:
        if not args.split_name.strip():
            raise ValueError("--split-name is required when using --input")
        if not args.output_name.strip():
            raise ValueError("--output-name is required when using --input")
        summary = convert_single_split(
            input_path=Path(args.input),
            split_name=args.split_name,
            output_dir=output,
            output_name=args.output_name,
            frames_root=frames_root,
        )
    else:
        if paired_mode:
            split_configs: list[tuple[str, Path, str]] = []
            if args.train_input.strip():
                split_configs.append(("train", Path(args.train_input), args.train_output_name))
            if args.test_input.strip():
                split_configs.append(("test", Path(args.test_input), args.test_output_name))
        else:
            split_configs = [
                ("train", processed / "train_80.jsonl", args.train_output_name),
                ("test", processed / "test_20.jsonl", args.test_output_name),
            ]
        summary = convert_named_splits(
            split_configs=split_configs,
            output_dir=output,
            frames_root=frames_root,
        )

    (output / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
