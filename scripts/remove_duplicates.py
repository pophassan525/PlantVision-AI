"""
PlantVision AI — Duplicate Removal Script (Phase 1)

Purpose:
    Read the duplicate groups found by check_dataset.py
    (reports/dataset_inspection_report.json) and remove all but one file
    from each group of exact duplicates.

    Nothing is guessed here — this only acts on duplicate groups that were
    actually measured by check_dataset.py's MD5 hashing.

Usage:
    Run check_dataset.py FIRST (it writes the report this script reads).
    python scripts/remove_duplicates.py

    Add --dry_run to only print what WOULD be deleted, without deleting anything:
    python scripts/remove_duplicates.py --dry_run
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Remove exact-duplicate images found by check_dataset.py")
    parser.add_argument(
        "--report", type=str, default="reports/dataset_inspection_report.json",
        help="Path to the JSON report produced by check_dataset.py"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="If set, only print what would be deleted without actually deleting files."
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(
            f"Report not found at {report_path}. Run check_dataset.py first — "
            "this script depends on its output and does not re-scan the dataset itself."
        )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    duplicate_groups = report.get("duplicate_groups", {})

    if not duplicate_groups:
        print("No duplicate groups found in the report. Nothing to do.")
        return

    total_deleted = 0
    total_kept = 0

    for file_hash, paths in duplicate_groups.items():
        # Keep the first path (sorted, for determinism), delete the rest
        sorted_paths = sorted(paths)
        keep = sorted_paths[0]
        to_delete = sorted_paths[1:]

        print(f"\nGroup ({file_hash[:8]}...): {len(paths)} identical files")
        print(f"  KEEP:   {keep}")
        total_kept += 1

        for path_str in to_delete:
            path = Path(path_str)
            if args.dry_run:
                print(f"  WOULD DELETE: {path}")
            else:
                if path.exists():
                    path.unlink()
                    print(f"  DELETED: {path}")
                else:
                    print(f"  SKIP (already gone): {path}")
            total_deleted += 1

    print("\n=== SUMMARY ===")
    print(f"Duplicate groups processed: {len(duplicate_groups)}")
    print(f"Files kept (one per group): {total_kept}")
    if args.dry_run:
        print(f"Files that WOULD be deleted: {total_deleted} (dry run — nothing was actually deleted)")
        print("\nRun again without --dry_run to actually delete these files.")
    else:
        print(f"Files deleted: {total_deleted}")
        print("\nRecommended next step: re-run check_dataset.py to confirm duplicate_groups is now 0.")


if __name__ == "__main__":
    main()
