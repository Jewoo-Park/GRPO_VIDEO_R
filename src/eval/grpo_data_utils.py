import json
import os
import re
from pathlib import Path
from typing import Iterable


ANSWER_TAG_RE = re.compile(r"<\s*answer[s]?\s*>(.*?)</\s*answer[s]?\s*>", re.IGNORECASE | re.DOTALL)
FRAME_GLOB_PATTERNS = ("frame_*.jpg", "frame_*.jpeg", "frame_*.png", "*.jpg", "*.jpeg", "*.png")
FRAME_LIST_KEYS = ("frames", "frame_paths", "images")
QUESTION_ID_KEYS = ("question_id", "Question_id", "problem_id")
VIDEO_ID_KEYS = ("video_id", "video", "video_path", "video_name", "path")


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_stem(text: str) -> str:
    keep: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def frame_key_from_path(path_text: str) -> str:
    path = Path(path_text)
    parts = [safe_stem(part) for part in path.with_suffix("").parts if part not in ("", ".", "..")]
    return "__".join(part for part in parts if part) or safe_stem(path.stem)


def normalize_answer(answer_or_solution: str) -> str:
    raw = str(answer_or_solution or "").strip()
    match = ANSWER_TAG_RE.search(raw)
    answer = match.group(1).strip() if match else raw
    return f"<ANSWER>{answer}</ANSWER>"


def format_options(options: object) -> list[str]:
    if isinstance(options, list):
        return [str(opt).strip() for opt in options if str(opt).strip()]
    if isinstance(options, str) and options.strip():
        return [line.strip() for line in options.splitlines() if line.strip()]
    return []


def normalize_problem(row: dict) -> str:
    raw_problem = str(row.get("problem") or row.get("question") or "").strip()
    if raw_problem and not raw_problem.lower().startswith("question:"):
        raw_problem = f"Question: {raw_problem}"

    options = format_options(row.get("options") or row.get("choices") or row.get("choice"))
    if not options:
        return raw_problem

    option_block = "\n".join(options)
    has_all_options = raw_problem and all(option in raw_problem for option in options)
    if has_all_options:
        return raw_problem

    if raw_problem:
        return f"{raw_problem}\nOptions:\n{option_block}"
    return f"Question:\nOptions:\n{option_block}"


def pick_question_id(row: dict):
    for key in QUESTION_ID_KEYS:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def pick_video_id(row: dict) -> str:
    for key in VIDEO_ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def pick_question_category(row: dict) -> str:
    for key in ("question_category", "data_source", "dataset_name", "source_subset"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _resolve_existing_path(path_text: str, base_dir: Path, frames_root: Path | None) -> Path | None:
    candidate = Path(path_text)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidates = [base_dir / path_text]
    if frames_root is not None:
        candidates.append(frames_root / path_text)
    candidates.append(Path(path_text))

    for maybe in candidates:
        if maybe.exists():
            return maybe.resolve()
    return None


def resolve_frame_paths(row: dict, split_name: str, frames_root: Path | None, source_base_dir: Path) -> list[Path]:
    for key in FRAME_LIST_KEYS:
        value = row.get(key)
        if isinstance(value, list) and value:
            resolved = []
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    continue
                path = _resolve_existing_path(item.strip(), source_base_dir, frames_root)
                if path is not None:
                    resolved.append(path)
            if resolved:
                return sorted(resolved)

    frame_dir_value = row.get("frame_dir")
    if isinstance(frame_dir_value, str) and frame_dir_value.strip():
        resolved_dir = _resolve_existing_path(frame_dir_value.strip(), source_base_dir, frames_root)
        if resolved_dir is not None and resolved_dir.is_dir():
            return collect_frames_from_dir(resolved_dir)

    if frames_root is None:
        return []

    if isinstance(row.get("frame_subdir"), str) and row["frame_subdir"].strip():
        candidate_dir = frames_root / split_name / row["frame_subdir"].strip()
        if candidate_dir.exists():
            return collect_frames_from_dir(candidate_dir)

    video_id = pick_video_id(row)
    if not video_id:
        return []
    candidate_dir = frames_root / split_name / safe_stem(Path(video_id).stem)
    if candidate_dir.exists():
        return collect_frames_from_dir(candidate_dir)
    return []


def collect_frames_from_dir(frame_dir: Path) -> list[Path]:
    frames: list[Path] = []
    for pattern in FRAME_GLOB_PATTERNS:
        frames.extend(frame_dir.glob(pattern))
    unique_frames = sorted({frame.resolve() for frame in frames if frame.is_file()})
    return unique_frames


def relativize_paths(paths: Iterable[Path], start_dir: Path) -> list[str]:
    start_dir = start_dir.resolve()
    return [os.path.relpath(path.resolve(), start=start_dir) for path in paths]
