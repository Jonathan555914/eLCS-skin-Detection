"""Extract skin-image handcrafted features into eLCS-ready CSV files.

Expected image layout:

    metadata/data/images/
        benign/
            image1.jpg
        malignant/
            image2.jpg

Output format:
    feature columns + final ``Class`` column
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "handcrafted_features"))

from dwt_feature import extract_dwt_features  # noqa: E402
from glcm_feature import extract_glcm_features  # noqa: E402
from hog_feature import extract_hog_features  # noqa: E402
from lbp_feature import extract_lbp_features  # noqa: E402
from rgb_feature import extract_rgb_features  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_LABELS = ("benign", "malignant")


def load_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Read a color image and resize it to keep feature length stable."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)


def summarize(values: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    """Summarize large vectors so eLCS receives compact numeric features."""
    values = np.asarray(values, dtype=np.float64).ravel()
    names = [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_p25",
        f"{prefix}_p50",
        f"{prefix}_p75",
    ]
    features = [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 75)),
    ]
    return names, features


def extract_image_features(
    image_path: str | Path,
    image_size: tuple[int, int] = (128, 128),
    feature_set: str = "all",
) -> tuple[list[str], list[float]]:
    """Extract selected handcrafted features from one image."""
    image = load_image(Path(image_path), image_size)
    names: list[str] = []
    features: list[float] = []
    selected = {part.strip().lower() for part in feature_set.split("+")}

    if "all" in selected or "lbp" in selected:
        _, lbp_hist = extract_lbp_features(image, radius=1, n_points=8, method="uniform")
        lbp_features = np.zeros(10, dtype=np.float64)
        lbp_features[: min(10, len(lbp_hist))] = lbp_hist[:10]
        names.extend([f"lbp_{index}" for index in range(10)])
        features.extend(float(value) for value in lbp_features)

    if "all" in selected or "hog" in selected:
        hog_features = extract_hog_features(image, visualize=False)
        hog_names, hog_summary = summarize(hog_features, "hog")
        names.extend(hog_names)
        features.extend(hog_summary)

    if "all" in selected or "dwt" in selected:
        dwt_features, _ = extract_dwt_features(image, wavelet="haar", level=2, feature_type="stats")
        names.extend([f"dwt_{index}" for index in range(len(dwt_features))])
        features.extend(float(value) for value in dwt_features)

    if "all" in selected or "rgb" in selected or "color" in selected:
        rgb_names, rgb_features = extract_rgb_features(image)
        names.extend(rgb_names)
        features.extend(float(value) for value in rgb_features)

    if "all" in selected or "glcm" in selected:
        glcm_names, glcm_features = extract_glcm_features(image)
        names.extend(glcm_names)
        features.extend(float(value) for value in glcm_features)

    if not names:
        raise ValueError("No valid features selected. Use lbp, hog, dwt, rgb, glcm, or all.")

    return names, features


def iter_labeled_images(image_root: Path, labels: tuple[str, ...]) -> list[tuple[Path, str]]:
    """Collect image paths from one subfolder per class."""
    rows = []
    for label in labels:
        label_dir = image_root / label
        if not label_dir.is_dir():
            continue
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((image_path, label))
    return rows


def build_feature_dataset(
    image_root: str | Path,
    output_file: str | Path,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    image_size: tuple[int, int] = (128, 128),
    feature_set: str = "all",
) -> Path:
    """Create a CSV dataset from benign/malignant image folders."""
    image_root = Path(image_root)
    output_file = Path(output_file)
    labeled_images = iter_labeled_images(image_root, labels)
    if not labeled_images:
        raise ValueError(f"No images found under {image_root} for labels: {', '.join(labels)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    header = None
    with output_file.open("w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        for image_path, label in labeled_images:
            feature_names, features = extract_image_features(image_path, image_size, feature_set)
            if header is None:
                header = feature_names + ["Class"]
                writer.writerow(header)
            writer.writerow([f"{value:.10g}" for value in features] + [label])

    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract handcrafted skin-image features for eLCS.")
    parser.add_argument("image_root", type=Path, help="Folder containing benign/ and malignant/ subfolders.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "metadata" / "features" / "ALL" / "skin_features.csv")
    parser.add_argument(
        "--features",
        default="all",
        help="Feature set: all, lbp, hog, dwt, rgb, glcm, or combinations like rgb+glcm.",
    )
    parser.add_argument("--labels", default="benign,malignant", help="Comma-separated class folder names.")
    parser.add_argument("--image-size", type=int, default=128, help="Square resize dimension in pixels.")
    args = parser.parse_args()

    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    output_file = build_feature_dataset(
        args.image_root,
        args.output,
        labels=labels,
        image_size=(args.image_size, args.image_size),
        feature_set=args.features,
    )
    print(f"Wrote feature dataset: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
