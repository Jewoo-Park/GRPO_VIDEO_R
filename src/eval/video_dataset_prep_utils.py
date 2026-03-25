import json
import os
from pathlib import Path
from typing import Callable, Optional

import av
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_dataset_files(
    dataset_id: str,
    local_dir: Path,
    allow_patterns: list[str],
    token: Optional[str] = None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token,
        resume_download=True,
    )


def download_mp4_dataset(dataset_id: str, local_dir: Path, token: Optional[str] = None) -> None:
    download_dataset_files(
        dataset_id=dataset_id,
        local_dir=local_dir,
        allow_patterns=["*.mp4"],
        token=token,
    )


def build_video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in video_root.rglob("*.mp4"):
        index.setdefault(path.name, path)
    return index


def resolve_video_path(
    sample: dict,
    video_root: Path,
    index: dict[str, Path],
    keys: tuple[str, ...] = ("video_id", "video", "video_path", "video_name", "path"),
) -> Optional[Path]:
    for key in keys:
        value = sample.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if candidate.exists():
            return candidate.resolve()
        joined = video_root / value
        if joined.exists():
            return joined.resolve()
        stripped = video_root / value.lstrip("./")
        if stripped.exists():
            return stripped.resolve()
        by_name = index.get(Path(value).name)
        if by_name is not None and by_name.exists():
            return by_name.resolve()
    return None


def load_video_frames(video_path: Path, num_frames: int, max_size: int) -> list[Image.Image]:
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        total_frames = stream.frames
        if not total_frames:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        idx_set = set(indices.tolist())

        frames: list[Image.Image] = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in idx_set:
                image = frame.to_image()
                if max(image.size) > max_size:
                    ratio = max_size / float(max(image.size))
                    resized = (max(1, int(image.size[0] * ratio)), max(1, int(image.size[1] * ratio)))
                    image = image.resize(resized, Image.Resampling.LANCZOS)
                frames.append(image)
            if len(frames) >= num_frames:
                break

        container.close()
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))
        return frames[:num_frames]
    except Exception:
        return [Image.new("RGB", (224, 224)) for _ in range(num_frames)]


def extract_frames_for_rows(
    rows: list[dict],
    split_name: str,
    video_root: Path,
    frame_root: Path,
    num_frames: int,
    max_size: int,
    frame_subdir_builder: Callable[[dict, Path], str],
    progress_label: str,
) -> tuple[list[dict], dict[str, int]]:
    frame_root.mkdir(parents=True, exist_ok=True)
    video_index = build_video_index(video_root)
    cached_by_video: dict[Path, str] = {}

    processed_rows: list[dict] = []
    missing_videos = 0
    decoded_videos = 0

    for idx, row in enumerate(rows):
        video_path = resolve_video_path(row, video_root, video_index)
        if video_path is None:
            missing_videos += 1
            continue

        resolved_video = video_path.resolve()
        frame_subdir = cached_by_video.get(resolved_video)
        if frame_subdir is None:
            frame_subdir = frame_subdir_builder(row, resolved_video)
            output_dir = frame_root / split_name / frame_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            existing_frames = sorted(output_dir.glob("frame_*.jpg"))
            if len(existing_frames) < num_frames:
                frames = load_video_frames(video_path=resolved_video, num_frames=num_frames, max_size=max_size)
                for frame_idx, frame in enumerate(frames):
                    frame.save(output_dir / f"frame_{frame_idx:03d}.jpg", format="JPEG", quality=95)

            cached_by_video[resolved_video] = frame_subdir
            decoded_videos += 1
            if decoded_videos % 50 == 0:
                print(f"[{progress_label}] decoded videos: {decoded_videos} (row index: {idx})")

        row_copy = dict(row)
        row_copy["frame_subdir"] = frame_subdir
        processed_rows.append(row_copy)

    stats = {
        "input_rows": len(rows),
        "processed_rows": len(processed_rows),
        "decoded_videos": decoded_videos,
        "missing_videos": missing_videos,
    }
    return processed_rows, stats


def hf_token_from_env() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
