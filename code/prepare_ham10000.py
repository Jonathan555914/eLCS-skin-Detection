"""Organize HAM10000 images into binary benign/malignant folders."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


CLASS_MAP = {
    "mel": "malignant",
    "bcc": "malignant",
    "nv": "benign",
    "bkl": "benign",
    "df": "benign",
    "vasc": "benign",
}
EXCLUDED_LABELS = {"akiec"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_images(source_directories: list[Path]) -> dict[str, Path]:
    """Index image files by filename stem across all source directories."""
    image_index: dict[str, Path] = {}
    for source_directory in source_directories:
        for extension in IMAGE_EXTENSIONS:
            for image_path in source_directory.rglob(f"*{extension}"):
                image_index[image_path.stem] = image_path
    return image_index


def organize_ham10000(
    metadata_file: Path,
    source_directories: list[Path],
    output_directory: Path,
    copy_files: bool = False,
) -> Counter:
    """Link or copy HAM10000 images into binary class folders."""
    image_index = find_images(source_directories)
    if not image_index:
        raise ValueError("No HAM10000 image files were found in the source directories.")

    for class_name in ("benign", "malignant", "excluded"):
        (output_directory / class_name).mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    missing_ids: list[str] = []
    manifest_path = output_directory.parent / "HAM10000_binary_manifest.csv"

    with metadata_file.open(newline="") as metadata_obj, manifest_path.open("w", newline="") as manifest_obj:
        reader = csv.DictReader(metadata_obj)
        fieldnames = list(reader.fieldnames or []) + ["binary_class", "source_path", "output_path"]
        writer = csv.DictWriter(manifest_obj, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            image_id = row["image_id"]
            diagnosis = row["dx"]
            source_path = image_index.get(image_id)
            if source_path is None:
                missing_ids.append(image_id)
                continue

            if diagnosis in CLASS_MAP:
                binary_class = CLASS_MAP[diagnosis]
            elif diagnosis in EXCLUDED_LABELS:
                binary_class = "excluded"
            else:
                raise ValueError(f"Unsupported HAM10000 diagnosis: {diagnosis}")

            output_path = output_directory / binary_class / source_path.name
            if not output_path.exists():
                if copy_files:
                    shutil.copy2(source_path, output_path)
                else:
                    output_path.hardlink_to(source_path)

            counts[binary_class] += 1
            counts[diagnosis] += 1
            writer.writerow(
                {
                    **row,
                    "binary_class": binary_class,
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                }
            )

    if missing_ids:
        sample = ", ".join(missing_ids[:10])
        raise ValueError(f"Missing {len(missing_ids)} images. First missing IDs: {sample}")

    print(f"Manifest: {manifest_path}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 for binary eLCS training.")
    parser.add_argument("metadata_file", type=Path)
    parser.add_argument("source_directories", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "metadata" / "data" / "images",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files instead of making space-efficient hard links.")
    args = parser.parse_args()

    counts = organize_ham10000(args.metadata_file, args.source_directories, args.output, args.copy)
    print("HAM10000 binary dataset prepared:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
