import argparse
import json
from pathlib import Path

from datasets import load_dataset


def export_split_to_jsonl(split_data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in split_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load UrbanVideo-Bench from Hugging Face and export one split to JSONL."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="EmbodiedCity/UrbanVideo-Bench",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to export (default: train)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/urban_video_bench/urban_video_bench_train.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset_id)
    print("Dataset loaded!")
    print(f"Available splits: {list(ds.keys())}")

    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available splits: {list(ds.keys())}")

    split_data = ds[args.split]
    print(f"\n{args.split} split:")
    print(f"  Number of samples: {len(split_data)}")
    print(f"  Features: {split_data.features}")

    output_path = Path(args.output)
    export_split_to_jsonl(split_data, output_path)
    print(f"\nSaved {len(split_data)} samples to: {output_path}")


if __name__ == "__main__":
    main()
