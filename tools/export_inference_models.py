"""Export frozen Keras checkpoints without training state or private paths.

This release-maintenance tool does not train a model. Its private source map is
a JSON list of {id, path, sha256, asset} records; keep that map outside Git.
Every export must retain the architecture, every weight bit and deterministic
CPU predictions. Public verification records contain hashes, never source paths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architecture(config: dict[str, Any]) -> dict[str, Any]:
    """Training configuration is the only permitted model-config difference."""
    return {key: value for key, value in config.items() if key != "compile_config"}


def assert_portable(value: Any) -> None:
    """Reject absolute filesystem references and training-loss registrations."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert_portable(key)
            assert_portable(item)
    elif isinstance(value, list):
        for item in value:
            assert_portable(item)
    elif isinstance(value, str):
        if (value.startswith(("/", "\\\\", "~"))
                or re.match(r"^[A-Za-z]:[\\/]", value)
                or "organoid_segmentation>" in value):
            raise ValueError("Export contains a private path or training registration")


def without_shared_ids(value: Any) -> Any:
    """Ignore Keras object-sharing IDs when checking its reserialization.

    Keras can drop shared dtype-policy IDs on reload. These process-specific
    IDs are serialization metadata, not graph/layer settings. The final file
    nevertheless retains the original architecture JSON verbatim.
    """
    if isinstance(value, dict):
        return {key: without_shared_ids(item) for key, item in value.items()
                if key != "shared_object_id"}
    if isinstance(value, list):
        return [without_shared_ids(item) for item in value]
    return value


def remove_compile_configuration(
    path: Path, source_config: dict[str, Any]
) -> dict[str, Any]:
    """Keep the original graph config and omit all compile configuration."""
    with zipfile.ZipFile(path) as source:
        contents = [(info, source.read(info.filename)) for info in source.infolist()]
    saved_config = json.loads(dict((i.filename, b) for i, b in contents)["config.json"])
    if (without_shared_ids(architecture(saved_config))
            != without_shared_ids(architecture(source_config))):
        raise ValueError("Keras reserialization changed the model architecture")
    config = architecture(source_config)
    assert_portable(config)
    with zipfile.ZipFile(path, "w") as destination:
        for info, data in contents:
            if info.filename == "config.json":
                data = json.dumps(config, separators=(",", ":")).encode("utf-8")
            destination.writestr(info, data)
    return config


def export_one(record: dict[str, str], destination: Path) -> dict[str, Any]:
    import h5py
    import numpy as np
    import tensorflow as tf

    source = Path(record["path"]).expanduser().resolve()
    asset_name = record["asset"]
    if (Path(asset_name).name != asset_name or not asset_name.endswith(".keras")
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", asset_name)):
        raise ValueError("Asset must be a simple .keras filename")
    output = destination / asset_name
    if output.exists():
        raise FileExistsError(f"Refusing to replace release asset: {asset_name}")
    source_hash = sha256(source)
    if source_hash != record["sha256"]:
        raise ValueError(f"Source hash mismatch for {record['id']}")
    with zipfile.ZipFile(source) as archive:
        source_config = json.loads(archive.read("config.json"))

    with tempfile.TemporaryDirectory(prefix=".export_", dir=destination) as temporary:
        candidate = Path(temporary) / asset_name
        # compile=False deliberately excludes optimizer, loss and metrics.
        original = tf.keras.models.load_model(source, compile=False, safe_mode=True)
        original.save(candidate, overwrite=False)
        export_config = remove_compile_configuration(candidate, source_config)
        if architecture(source_config) != architecture(export_config):
            raise ValueError(f"Architecture changed for {record['id']}")
        with zipfile.ZipFile(candidate) as archive:
            if archive.testzip() is not None:
                raise ValueError("Exported Keras ZIP failed CRC verification")
            if set(archive.namelist()) != {"metadata.json", "config.json", "model.weights.h5"}:
                raise ValueError("Unexpected Keras archive members")
            weights_path = Path(temporary) / "inspection.weights.h5"
            weights_path.write_bytes(archive.read("model.weights.h5"))
        with h5py.File(weights_path, "r") as weights:
            names: list[str] = []
            weights.visit(names.append)
            if any("optimizer" in name.lower() for name in names):
                raise ValueError("Optimizer state remained in inference export")
        weights_path.unlink()

        reloaded = tf.keras.models.load_model(candidate, compile=False, safe_mode=True)
        before, after = original.get_weights(), reloaded.get_weights()
        if len(before) != len(after) or any(
            a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b)
            for a, b in zip(before, after)
        ):
            raise ValueError(f"Weight values changed for {record['id']}")
        # Deterministic synthetic pixels are not biological data. On the same
        # CPU/runtime the untouched and exported graph must match exactly.
        pixels = np.random.default_rng(20261019).random((1, 64, 64, 1), dtype=np.float32)
        with tf.device("/CPU:0"):
            first = original(pixels, training=False).numpy()
            second = reloaded(pixels, training=False).numpy()
        if not np.isfinite(first).all() or not np.array_equal(first, second):
            raise ValueError(f"Deterministic CPU output changed for {record['id']}")
        result = {
            "id": record["id"],
            "asset": asset_name,
            "source_sha256": source_hash,
            "source_bytes": source.stat().st_size,
            "sha256": sha256(candidate),
            "bytes": candidate.stat().st_size,
            "architecture_equal_excluding_compile_config": True,
            "weight_arrays": len(before),
            "weight_scalars": sum(int(a.size) for a in before),
            "weights_bitwise_equal": True,
            "optimizer_state_present": False,
            "compile_config_present": False,
            "private_paths_present": False,
            "synthetic_test_shape": [1, 64, 64, 1],
            "synthetic_prediction_exact_equal": True,
            "synthetic_max_absolute_error": float(np.max(np.abs(first - second))),
            "tensorflow_version": tf.__version__,
            "keras_version": tf.keras.__version__,
            "numpy_version": np.__version__,
        }
        # Hard-link creation cannot overwrite an existing release file.
        os.link(candidate, output)
        del original, reloaded, before, after
        tf.keras.backend.clear_session()
        gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.source_map.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Source map must be a nonempty list")
    if any(not isinstance(row, dict) or set(row) != {"id", "path", "sha256", "asset"}
           for row in records):
        raise ValueError("Each source record requires exactly id, path, sha256, asset")
    if (len({row["id"] for row in records}) != len(records)
            or len({row["asset"] for row in records}) != len(records)):
        raise ValueError("Source identifiers and asset filenames must be unique")
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    verification = destination / "EXPORT_VERIFICATION.json"
    if verification.exists():
        raise FileExistsError("Refusing to replace an existing export verification")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    results = []
    for record in records:
        result = export_one(record, destination)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    with verification.open("x", encoding="utf-8") as stream:
        json.dump({"status": "PASS", "exports": results}, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
