"""Frozen per-plane percentile normalization used by the research models."""

from __future__ import annotations

import numpy as np


def normalize_image(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.8,
) -> np.ndarray:
    """Return float32 [0, 1] intensities without changing the pixel grid.

    Every stack plane is normalized independently. The float32 conversion,
    percentile calculation, subtraction, division and clipping order match the
    frozen research implementation. Constant images become valid zero images.
    """
    image = np.asarray(image)
    if image.ndim != 2 or not all(image.shape):
        raise ValueError("Normalization requires a nonempty 2D image")
    if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise ValueError("Normalization requires real numeric pixels")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    image_float = image.astype(np.float32, copy=False)
    if not np.isfinite(image_float).all():
        raise ValueError("Image values overflow float32 normalization")
    low, high = np.percentile(image_float, [lower_percentile, upper_percentile])
    if high <= low:
        return np.zeros(image_float.shape, dtype=np.float32)
    normalized = (image_float - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
