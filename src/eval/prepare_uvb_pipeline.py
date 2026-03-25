### → 샘플링/분할 → 비디오 다운로드 → 프레임 추출

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image


def load_jsonl(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def grouped_sample(data: List[dict], sample_ratio: float, seed: int) -> Tuple[List[dict], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[dict]] = {}
    for item in data:
        category = item.get("question_category", "unknown")
        grouped.setdefault(category, []).append(item)

    sampled_all: List[dict] = []
    stats: Dict[str, Dict[str, int]] = {}
    for category in sorted(grouped.keys()):
        items = grouped[category]
        total = len(items)
        sample_n = max(1, round(total * sample_ratio)) if total > 0 else 0
        sample_n = min(sample_n, total)
        sampled = rng.sample(items, sample_n) if sample_n > 0 else []
        sampled_all.extend(sampled)
        stats[category] = {"total": total, "sampled": sample_n}
    return sampled_all, stats


def stratified_train_test_split(data: List[dict], test_ratio: float, seed: int) -> Tuple[List[dict], List[dict], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[dict]] = {}
    for item in data:
        category = item.get("question_category", "unknown")
        grouped.setdefault(category, []).append(item)

    train_rows: List[dict] = []
    test_rows: List[dict] = []
    stats: Dict[str, Dict[str, int]] = {}

    for category in sorted(grouped.keys()):
        items = grouped[category]
        rng.shuffle(items)
        n = len(items)
        if n <= 1:
            test_n = 0
            train_n = n
        else:
            test_n = max(1, round(n * test_ratio))
            if test_n >= n:
                test_n = n - 1
            train_n = n - test_n

        test_split = items[:test_n]
        train_split = items[test_n:]
        train_rows.extend(train_split)
        test_rows.extend(test_split)
        stats[category] = {"sampled": n, "train": len(train_split), "test": len(test_split)}

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows, stats


def download_videos(dataset_id: str, local_dir: Path, token: Optional[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["*.mp4"],
        token=token,
        resume_download=True,
    )


def build_video_index(video_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in video_root.rglob("*.mp4"):
        index[p.name] = p
    return index


def resolve_video_path(sample: dict, video_root: Path, index: Dict[str, Path]) -> Optional[Path]:
    for key in ("video_id", "video", "video_path", "video_name"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            as_path = Path(value)
            if as_path.exists():
                return as_path
            joined = video_root / value
            if joined.exists():
                return joined
            by_name = index.get(Path(value).name)
            if by_name and by_name.exists():
                return by_name
    return None


def load_video_frames(video_path: Path, num_frames: int = 32, max_size: int = 768) -> List[Image.Image]:
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        total_frames = stream.frames
        if not total_frames:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        idx_set = set(indices.tolist())

        frames: List[Image.Image] = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in idx_set:
                img = frame.to_image()
                if max(img.size) > max_size:
                    ratio = max_size / float(max(img.size))
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                frames.append(img)
            if len(frames) >= num_frames:
                break

        container.close()
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
        return frames[:num_frames]
    except Exception:
        return [Image.new("RGB", (224, 224)) for _ in range(num_frames)]


def safe_stem(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def extract_frames_for_samples(
    samples: List[dict],
    split_name: str,
    video_root: Path,
    frame_root: Path,
    num_frames: int,
    max_size: int,
) -> Dict[str, int]:
    frame_root.mkdir(parents=True, exist_ok=True)
    video_index = build_video_index(video_root)

    processed_by_video: Dict[str, str] = {}
    missing_videos = 0
    decoded = 0

    for i, sample in enumerate(samples):
        video_path = resolve_video_path(sample, video_root, video_index)
        if video_path is None:
            missing_videos += 1
            continue

        key = str(video_path.resolve())
        if key in processed_by_video:
            continue

        video_name = video_path.stem
        output_dir = frame_root / split_name / safe_stem(video_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = load_video_frames(video_path=video_path, num_frames=num_frames, max_size=max_size)
        for frame_idx, frame in enumerate(frames):
            frame.save(output_dir / f"frame_{frame_idx:03d}.jpg", format="JPEG", quality=95)

        processed_by_video[key] = str(output_dir)
        decoded += 1
        if decoded % 50 == 0:
            print(f"[{split_name}] decoded videos: {decoded} (sample index: {i})")

    return {"decoded_videos": decoded, "missing_videos": missing_videos}


def main() -> None:
    parser = argparse.ArgumentParser(description="UVB sampling/splitting/video download/frame extraction pipeline.")
    parser.add_argument("--input-jsonl", type=str, default="data/urban_video_bench/urban_video_bench_train.jsonl")
    # Default to full dataset (no sampling). If sampling is needed later, change this default
    # or pass --sample-ratio explicitly at runtime (e.g., 0.4 for 40%).
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-id", type=str, default="EmbodiedCity/UrbanVideo-Bench")
    parser.add_argument("--video-dir", type=str, default="data/urban_video_bench")
    parser.add_argument("--output-dir", type=str, default="data/urban_video_bench/processed")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-frame-size", type=int, default=768)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HF download and use already-downloaded local videos",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    video_dir = Path(args.video_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {input_path}. Run src/scripts/prepare_uvb_dataset.sh first."
        )

    print("=" * 70)
    print("Step 1/4: Load metadata and sample by question_category")
    print("=" * 70)
    rows = load_jsonl(input_path)
    print(f"Loaded rows: {len(rows)}")
    sampled_rows, sample_stats = grouped_sample(rows, sample_ratio=args.sample_ratio, seed=args.seed)
    print(f"Sampling ratio: {args.sample_ratio:.3f}")
    print(f"Sampled rows: {len(sampled_rows)}")

    print("=" * 70)
    print("Step 2/4: Stratified train/test split (8:2)")
    print("=" * 70)
    train_rows, test_rows, split_stats = stratified_train_test_split(
        sampled_rows, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows:  {len(test_rows)}")

    sample_percent = int(round(args.sample_ratio * 100))
    sampled_filename = f"sampled_{sample_percent}_percent.json"
    write_json(output_dir / sampled_filename, sampled_rows)
    write_jsonl(output_dir / "train_80.jsonl", train_rows)
    write_jsonl(output_dir / "test_20.jsonl", test_rows)
    write_json(output_dir / "sample_stats.json", sample_stats)
    write_json(output_dir / "split_stats.json", split_stats)

    print("=" * 70)
    print("Step 3/4: Download mp4 files from HF dataset repo")
    print("=" * 70)
    if args.skip_download:
        print(f"Skipping download. Using existing local videos under: {video_dir}")
    else:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        download_videos(dataset_id=args.dataset_id, local_dir=video_dir, token=hf_token)
        print(f"Video download completed under: {video_dir}")

    print("=" * 70)
    print("Step 4/4: Decode videos and save sampled frames")
    print("=" * 70)
    frame_root = output_dir / "frames"
    train_stats = extract_frames_for_samples(
        samples=train_rows,
        split_name="train",
        video_root=video_dir,
        frame_root=frame_root,
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
    )
    test_stats = extract_frames_for_samples(
        samples=test_rows,
        split_name="test",
        video_root=video_dir,
        frame_root=frame_root,
        num_frames=args.num_frames,
        max_size=args.max_frame_size,
    )

    summary = {
        "input_rows": len(rows),
        "sampling_ratio": args.sample_ratio,
        "sampled_filename": sampled_filename,
        "sampled_rows": len(sampled_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_frame_stats": train_stats,
        "test_frame_stats": test_stats,
        "num_frames_per_video": args.num_frames,
        "max_frame_size": args.max_frame_size,
    }
    write_json(output_dir / "pipeline_summary.json", summary)

    print("=" * 70)
    print("UVB pipeline complete")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
