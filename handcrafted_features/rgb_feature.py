"""Color-distribution features for skin lesion images."""

from __future__ import annotations

import cv2
import numpy as np


def _channel_statistics(channel: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    """Calculate robust statistics for one image channel."""
    values = channel.astype(np.float64).ravel()
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    skewness = float(np.mean(centered ** 3) / (std ** 3 + 1e-12))

    names = [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_skew",
        f"{prefix}_p25",
        f"{prefix}_p50",
        f"{prefix}_p75",
    ]
    features = [
        mean,
        std,
        skewness,
        float(np.percentile(values, 25)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 75)),
    ]
    return names, features


def extract_rgb_features(image: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Extract RGB and HSV color statistics.

    RGB describes the amount and variation of red, green, and blue. HSV adds
    hue, saturation, and brightness, which can better separate lesion color
    from illumination.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    names: list[str] = []
    features: list[float] = []
    for color_space, channel_names in ((rgb, ("r", "g", "b")), (hsv, ("h", "s", "v"))):
        for channel, channel_name in zip(cv2.split(color_space), channel_names):
            channel_feature_names, channel_features = _channel_statistics(channel, channel_name)
            names.extend(channel_feature_names)
            features.extend(channel_features)

    return names, np.asarray(features, dtype=np.float64)

