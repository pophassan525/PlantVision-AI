"""
PlantVision AI — Train/Validation/Test Split Script (Phase 1)

Purpose:
    Split the cleaned dataset in data/raw/<class>/ into data/train/<class>/,
    data/validation/<class>/, and data/test/<class>/, using a fixed random
    seed for reproducibility. Files are MOVED by default (copy is optional).

    Run this AFTER check_dataset.py and remove_duplicates.py — this script
    assumes data/raw is already clean (0 corrupted files, 0 duplicates).

Usage:
    python scripts/split_dataset.py --data_dir data/raw --output_dir data
    (default ratios: 70% train / 15% validation / 15% test)

    To copy instead of move (keeps data/raw intact):
    python scripts/split_dataset.py --copy
"""

import argparse
import json
import random
import shutil
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42  # fixed seed so the split is reproducible — same split every run


def split_dataset(data_dir: Path, output_dir: Path, train_ratio: float,
                   val_ratio: float, test_ratio: float, copy_files: bool):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(SEED)

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    split_summary = {}

    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    test_dir = output_dir / "test"
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    action = shutil.copy2 if copy_files else shutil.move

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # remainder goes to test, so rounding never drops an image
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dest_class_dir = output_dir / split_name / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                action(str(f), str(dest_class_dir / f.name))

        split_summary[class_name] = {
            "total": n, "train": n_train, "validation": n_val, "test": n_test
        }
        print(f"  {class_name}: {n} -> train={n_train}, val={n_val}, test={n_test}")

    return split_summary


def main():
    parser = argparse.ArgumentParser(description="Split PlantVision AI dataset into train/validation/test.")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                         help="Source directory with one subfolder per class (default: data/raw)")
    parser.add_argument("--output_dir", type=str, default="data",
                         help="Where to create train/ validation/ test/ subfolders (default: data)")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--copy", action="store_true",
                         help="Copy files instead of moving them (keeps data/raw unchanged, uses more disk space)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Source data directory not found: {data_dir}")

    print(f"Splitting dataset from {data_dir} (seed={SEED}, "
          f"ratios={args.train_ratio}/{args.val_ratio}/{args.test_ratio})")
    print(f"Mode: {'COPY (data/raw stays intact)' if args.copy else 'MOVE (data/raw will become empty)'}\n")

    summary = split_dataset(
        data_dir, output_dir,
        args.train_ratio, args.val_ratio, args.test_ratio,
        copy_files=args.copy
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "split_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "ratios": {"train": args.train_ratio, "validation": args.val_ratio, "test": args.test_ratio},
            "mode": "copy" if args.copy else "move",
            "per_class": summary,
            "totals": {
                "train": sum(v["train"] for v in summary.values()),
                "validation": sum(v["validation"] for v in summary.values()),
                "test": sum(v["test"] for v in summary.values()),
            }
        }, f, indent=2, ensure_ascii=False)

    totals = {
        "train": sum(v["train"] for v in summary.values()),
        "validation": sum(v["validation"] for v in summary.values()),
        "test": sum(v["test"] for v in summary.values()),
    }
    print("\n=== SUMMARY ===")
    print(f"Train images:      {totals['train']}")
    print(f"Validation images: {totals['validation']}")
    print(f"Test images:       {totals['test']}")
    print(f"\nFull per-class report written to: {report_path}")


if __name__ == "__main__":
    main()
