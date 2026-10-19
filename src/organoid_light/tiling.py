"""Lossless-size overlapping sliding-window inference for independent 2D images."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _window_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _blend_weights(tile_size: int) -> np.ndarray:
    """Create positive center-weighted blending weights without zero borders."""

    one_dimensional = np.hanning(tile_size).astype(np.float32)
    weights = np.outer(one_dimensional, one_dimensional)
    weights /= max(float(weights.max()), 1e-6)
    return np.maximum(weights, 1e-3).astype(np.float32)


def _pad_image(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    original_shape = image.shape
    pad_height = max(0, tile_size - image.shape[0])
    pad_width = max(0, tile_size - image.shape[1])
    if pad_height == 0 and pad_width == 0:
        return image, original_shape
    mode = "reflect" if min(image.shape) > 1 else "edge"
    padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode=mode)
    return padded, original_shape


def predict_sliding_window(
    model: object,
    normalized_image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128,
    prediction_batch_size: int = 4,
) -> np.ndarray:
    """Predict a probability map with exactly the input H×W pixel grid.

    Patch probabilities are blended before thresholding. This prevents hard
    seams that would occur if each tile were converted to a binary mask first.
    """

    normalized_image = np.asarray(normalized_image)
    if normalized_image.ndim != 2 or not all(normalized_image.shape):
        raise ValueError(f"Expected a 2D normalized image, got {normalized_image.shape}")
    if not np.issubdtype(normalized_image.dtype, np.number) or np.iscomplexobj(normalized_image):
        raise ValueError("Normalized image must have real numeric pixels")
    if not np.isfinite(normalized_image).all():
        raise ValueError("Normalized image contains NaN or infinite values")
    if normalized_image.min() < 0 or normalized_image.max() > 1:
        raise ValueError("Normalized image must be in [0, 1]")
    if tile_size <= 0 or tile_size % 16 != 0:
        raise ValueError("tile_size must be positive and divisible by 16")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if prediction_batch_size <= 0:
        raise ValueError("prediction_batch_size must be positive")

    image, original_shape = _pad_image(normalized_image, tile_size)
    stride = tile_size - overlap
    y_starts = _window_starts(image.shape[0], tile_size, stride)
    x_starts = _window_starts(image.shape[1], tile_size, stride)
    coordinates = [(y, x) for y in y_starts for x in x_starts]

    probability_sum = np.zeros(image.shape, dtype=np.float32)
    weight_sum = np.zeros(image.shape, dtype=np.float32)
    weights = _blend_weights(tile_size)

    for batch_start in range(0, len(coordinates), prediction_batch_size):
        batch_coordinates = coordinates[batch_start : batch_start + prediction_batch_size]
        batch = np.stack(
            [
                image[y : y + tile_size, x : x + tile_size]
                for y, x in batch_coordinates
            ],
            axis=0,
        )[..., np.newaxis].astype(np.float32, copy=False)
        predictions = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        if predictions.shape != batch.shape:
            raise ValueError(
                f"Model output shape {predictions.shape} does not match input {batch.shape}"
            )

        if not np.isfinite(predictions).all():
            raise ValueError("Model output contains NaN or infinite values")
        if predictions.min() < 0 or predictions.max() > 1:
            raise ValueError("Model probability output must be in [0, 1]")

        for index, (y, x) in enumerate(batch_coordinates):
            tile_probability = predictions[index, ..., 0]
            probability_sum[y : y + tile_size, x : x + tile_size] += (
                tile_probability * weights
            )
            weight_sum[y : y + tile_size, x : x + tile_size] += weights

    if np.any(weight_sum <= 0):
        raise RuntimeError("Sliding-window inference left pixels without prediction weights")
    probability = probability_sum / weight_sum
    height, width = original_shape
    result = probability[:height, :width]
    if result.shape != original_shape:
        raise RuntimeError(f"Output shape {result.shape} differs from input {original_shape}")
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def predict_many(
    model: object,
    images: Sequence[np.ndarray],
    tile_size: int,
    overlap: int,
    prediction_batch_size: int,
) -> list[np.ndarray]:
    """Predict several already-normalized images in input order."""

    return [
        predict_sliding_window(
            model,
            image,
            tile_size=tile_size,
            overlap=overlap,
            prediction_batch_size=prediction_batch_size,
        )
        for image in images
    ]
