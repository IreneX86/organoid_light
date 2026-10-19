"""Frozen per-organelle inference using one shared normalization/tile path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .nuc_postprocess import postprocess_nucleus_masks
from .preprocessing import normalize_image
from .tiling import predict_sliding_window


class Segmenter:
    """Reuse loaded inference-only models; execute ensemble members sequentially."""

    def __init__(self, spec: dict, model_paths: list[Path], tf: object, device: str,
                 batch_size: int = 4):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.spec, self.tf, self.device, self.batch_size = spec, tf, device, batch_size
        self.models = []
        with tf.device(device):
            for path in model_paths:
                model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
                if tuple(model.input_shape) != (None, None, None, 1):
                    raise ValueError(f"Unexpected model input shape: {model.input_shape}")
                if tuple(model.output_shape) != (None, None, None, 1):
                    raise ValueError(f"Unexpected model output shape: {model.output_shape}")
                self.models.append(model)
        if len(self.models) != len(spec["assets"]):
            raise ValueError("Model set is incomplete")

    def probabilities(self, image: np.ndarray):
        settings = self.spec["settings"]
        normalized = normalize_image(image, *settings["percentiles"])
        with self.tf.device(self.device):
            for model in self.models:
                yield predict_sliding_window(
                    model, normalized, tile_size=settings["tile_size"],
                    overlap=settings["overlap"], prediction_batch_size=self.batch_size,
                )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.spec["organelle"] == "ld":
            votes = np.zeros(image.shape, dtype=np.uint8)
            for probability, asset in zip(self.probabilities(image), self.spec["assets"]):
                votes += probability >= asset["threshold"]
            return votes >= self.spec["settings"]["minimum_votes"]
        probability = next(self.probabilities(image))
        if self.spec["organelle"] == "nuc":
            mask, _ = postprocess_nucleus_masks(
                probability >= self.spec["settings"]["low_threshold"],
                probability >= self.spec["settings"]["core_threshold"],
            )
            return mask
        return probability >= self.spec["assets"][0]["threshold"]
