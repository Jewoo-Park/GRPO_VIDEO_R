import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_stem(text: str) -> str:
    """Match prepare_uvb_pipeline's frame directory naming (alnum, -, _ only)."""
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def to_grpo_rows(split_rows: list[dict], split_name: str, frames_root: Path) -> tuple[list[dict], int]:
    out: list[dict] = []
    skipped = 0
    for item in split_rows:
        video_id = item["video_id"]
        # Pipeline uses safe_stem(video_path.stem) for frame dirs; use same here.
        video_stem = safe_stem(Path(video_id).stem)
        frame_dir = frames_root / split_name / video_stem
        frames = sorted(str(p.resolve()) for p in frame_dir.glob("frame_*.jpg"))

        # Require extracted frames to ensure training-time path validity.
        if len(frames) == 0:
            skipped += 1
            continue

        answer = str(item.get("answer", "")).strip()
        # HF dataset may use "question_id" or "Question_id".
        question_id = item.get("Question_id") or item.get("question_id")
        out.append(
            {
                "video_id": video_id,
                "question_id": question_id,
                "question_category": item.get("question_category", "unknown"),
                "problem": f"Question: {item.get('question', '').strip()}",
                "frames": frames,
                "solution": f"<answer>{answer}</answer>",
            }
        )
    return out, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UVB split JSONL + extracted frames to GRPO JSONL format.")
    parser.add_argument("--processed-dir", type=str, default="data/urban_video_bench/processed")
    parser.add_argument("--output-dir", type=str, default="data/urban_video_bench/grpo")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    output = Path(args.output_dir)
    frames_root = processed / "frames"

    train_rows = load_jsonl(processed / "train_80.jsonl")
    test_rows = load_jsonl(processed / "test_20.jsonl")

    train_out, train_skipped = to_grpo_rows(train_rows, "train", frames_root)
    test_out, test_skipped = to_grpo_rows(test_rows, "test", frames_root)

    train_out_path = output / "uvb_grpo_train.jsonl"
    test_out_path = output / "uvb_grpo_test.jsonl"
    dump_jsonl(train_out_path, train_out)
    dump_jsonl(test_out_path, test_out)

    summary = {
        "train_in": len(train_rows),
        "test_in": len(test_rows),
        "train_out": len(train_out),
        "test_out": len(test_out),
        "train_skipped_no_frames": train_skipped,
        "test_skipped_no_frames": test_skipped,
        "train_path": str(train_out_path.resolve()),
        "test_path": str(test_out_path.resolve()),
    }
    (output / "uvb_grpo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
