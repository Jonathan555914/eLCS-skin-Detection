"""Gray-Level Co-occurrence Matrix texture features for skin lesions."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


GLCM_PROPERTIES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
)


def extract_glcm_features(
    image: np.ndarray,
    levels: int = 32,
    distances: tuple[int, ...] = (1, 2, 4),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> tuple[list[str], np.ndarray]:
    """Extract rotation-aware GLCM texture statistics.

    The grayscale image is quantized before building co-occurrence matrices.
    Each GLCM property is summarized across several pixel distances and angles.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    quantized = np.floor(gray.astype(np.float64) * levels / 256.0)
    quantized = np.clip(quantized, 0, levels - 1).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    names: list[str] = []
    features: list[float] = []
    for property_name in GLCM_PROPERTIES:
        values = graycoprops(matrix, property_name).ravel()
        prefix = property_name.lower()
        names.extend([f"glcm_{prefix}_mean", f"glcm_{prefix}_std"])
        features.extend([float(np.mean(values)), float(np.std(values))])

    return names, np.asarray(features, dtype=np.float64)

