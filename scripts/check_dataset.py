"""
PlantVision AI — Dataset Inspection Script (Phase 1)

Purpose:
    Inspect a real image dataset directory (organized as one subfolder per class)
    and produce an honest report: per-class image counts, corrupted-image detection,
    exact-duplicate detection, and class-imbalance flags.

    This script does NOT train anything and does NOT invent numbers — every value
    in its output is measured directly from the files on disk.

Usage:
    python scripts/check_dataset.py --data_dir data/raw

Output:
    - Prints a summary report to the console
    - Writes a full report to reports/dataset_inspection_report.json
    - Writes a human-readable summary to reports/dataset_inspection_summary.md
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def compute_file_hash(file_path: Path) -> str:
    """Compute an MD5 hash of the file's raw bytes (for exact-duplicate detection)."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_dataset(data_dir: Path) -> dict:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(
            f"No class subfolders found inside {data_dir}. "
            "Expected one folder per class (e.g. Tomato___healthy)."
        )

    class_counts = {}
    corrupted_files = []
    hash_to_paths = defaultdict(list)
    total_images = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]
        valid_count = 0

        for img_path in image_files:
            # Corruption check: try to actually open and verify the image
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_count += 1
            except (UnidentifiedImageError, OSError):
                corrupted_files.append(str(img_path))
                continue

            # Duplicate check: hash the file content
            try:
                file_hash = compute_file_hash(img_path)
                hash_to_paths[file_hash].append(str(img_path))
            except OSError:
                pass

        class_counts[class_name] = valid_count
        total_images += valid_count
        print(f"  {class_name}: {valid_count} valid images")

    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    # Class imbalance check: flag classes far from the mean
    if class_counts:
        mean_count = sum(class_counts.values()) / len(class_counts)
        imbalanced_classes = {
            cls: count for cls, count in class_counts.items()
            if count < 0.5 * mean_count or count > 2 * mean_count
        }
    else:
        mean_count = 0
        imbalanced_classes = {}

    report = {
        "data_dir": str(data_dir),
        "num_classes": len(class_dirs),
        "total_valid_images": total_images,
        "class_counts": class_counts,
        "mean_images_per_class": round(mean_count, 1),
        "num_corrupted_files": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "num_duplicate_groups": len(duplicates),
        "duplicate_groups": duplicates,
        "imbalanced_classes": imbalanced_classes,
    }
    return report


def write_markdown_summary(report: dict, out_path: Path):
    lines = [
        "# Dataset Inspection Summary",
        "",
        f"- Data directory: `{report['data_dir']}`",
        f"- Number of classes: {report['num_classes']}",
        f"- Total valid images: {report['total_valid_images']}",
        f"- Mean images per class: {report['mean_images_per_class']}",
        f"- Corrupted files found: {report['num_corrupted_files']}",
        f"- Exact-duplicate groups found: {report['num_duplicate_groups']}",
        "",
        "## Per-class counts",
        "",
        "| Class | Images |",
        "|---|---|",
    ]
    for cls, count in sorted(report["class_counts"].items()):
        lines.append(f"| {cls} | {count} |")

    if report["imbalanced_classes"]:
        lines += ["", "## Classes flagged for imbalance (< 0.5x or > 2x the mean)", ""]
        for cls, count in report["imbalanced_classes"].items():
            lines.append(f"- {cls}: {count} images")
    else:
        lines += ["", "## Class imbalance", "", "No classes flagged — distribution looks reasonably balanced."]

    if report["num_corrupted_files"] > 0:
        lines += ["", "## Corrupted files", ""]
        for f in report["corrupted_files"][:50]:
            lines.append(f"- {f}")
        if report["num_corrupted_files"] > 50:
            lines.append(f"- ... and {report['num_corrupted_files'] - 50} more (see JSON report)")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Inspect PlantVision AI image dataset.")
    parser.add_argument(
        "--data_dir", type=str, default="data/raw",
        help="Path to the dataset directory (one subfolder per class). Default: data/raw"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    print(f"Inspecting dataset at: {data_dir}\n")
    report = inspect_dataset(data_dir)

    json_path = reports_dir / "dataset_inspection_report.json"
    md_path = reports_dir / "dataset_inspection_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_markdown_summary(report, md_path)

    print("\n=== SUMMARY ===")
    print(f"Classes found:       {report['num_classes']}")
    print(f"Total valid images:  {report['total_valid_images']}")
    print(f"Mean per class:      {report['mean_images_per_class']}")
    print(f"Corrupted files:     {report['num_corrupted_files']}")
    print(f"Duplicate groups:    {report['num_duplicate_groups']}")
    print(f"Imbalanced classes:  {len(report['imbalanced_classes'])}")
    print(f"\nFull report written to: {json_path}")
    print(f"Readable summary written to: {md_path}")


if __name__ == "__main__":
    main()
