"""Privately compare release inference against an existing frozen research run.

The source map and cases JSON contain private local paths and must stay outside
Git. A case contains organelle, input and golden_mask, plus an optional public
id alias. The resulting report contains hashes/counts only, never source paths
or pixels. Historical CPU/GPU mask differences are reported separately from
the strict same-runtime implementation comparison.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Read-only reference audits must not create cache files in the research tree.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(code_dir: Path, name: str):
    """Load the old leaf modules under distinct names, without sys.path edits."""
    alias = "_organoid_reference_" + name
    spec = importlib.util.spec_from_file_location(alias, code_dir / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError("Reference module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def release_models(records: dict[str, dict], spec: dict, models_dir: Path):
    from organoid_light.models import verify_asset
    organelle = spec["organelle"]
    result = []
    for index, asset in enumerate(spec["assets"]):
        source_id = f"ld-fold{index}" if organelle == "ld" else organelle
        source_record = records[source_id]
        original = Path(source_record["path"]).expanduser().resolve()
        exported = models_dir / asset["filename"]
        if sha256(original) != source_record["sha256"]:
            raise ValueError("Original checkpoint SHA256 failed")
        if asset["source_sha256"] != source_record["sha256"]:
            raise ValueError("Release does not originate from the expected checkpoint")
        verify_asset(exported, asset)
        if organelle != "nuc":
            threshold = json.loads((original.parent / "threshold.json").read_text())["threshold"]
            if threshold != asset["threshold"]:
                raise ValueError("Released threshold differs from original validation lock")
        result.append((source_id, original, exported, asset))
    return result


def original_mask(probabilities: list, organelle: str, spec: dict, nuc_reference):
    import numpy as np
    if organelle == "ld":
        votes = np.zeros(probabilities[0].shape, dtype=np.uint8)
        for probability, asset in zip(probabilities, spec["assets"]):
            votes += probability >= asset["threshold"]
        return votes >= 1
    if organelle == "nuc":
        return nuc_reference.postprocess_nucleus_masks(
            probabilities[0] >= 0.03, probabilities[0] >= 0.45
        )[0]
    return probabilities[0] >= spec["assets"][0]["threshold"]


def run_cli(case_dir: Path, organelle: str, models_dir: Path, source: Path, raw, expected):
    """Run real CLI inference and verify a hash/mtime-preserving resume."""
    import numpy as np
    import tifffile
    from organoid_light.tiff_io import inspect_tiff, validate_mask_tiff

    inputs = case_dir / "input"
    inputs.mkdir()
    shutil.copyfile(source, inputs / "single.tif")
    tifffile.imwrite(inputs / "stack.tif", np.stack([raw, raw]),
                     photometric="minisblack", metadata={"axes": "ZYX"})
    output = case_dir / "output"
    command = [sys.executable, "-m", "organoid_light", "--organelle", organelle,
               "--input", str(inputs), "--output", str(output), "--device", "cpu",
               "--models-dir", str(models_dir), "--batch-size", "4"]
    environment = os.environ.copy()
    package_source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_source, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    started = time.monotonic()
    with (case_dir / "cli.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment, timeout=3600)
    if process.returncode:
        raise RuntimeError("CLI reference run failed; local private cli.log retained")
    single, stack = (tifffile.imread(output / name) for name in ("single.tif", "stack.tif"))
    single_mismatch = int(np.count_nonzero((single > 0) != expected))
    if single.shape != expected.shape or single_mismatch:
        raise ValueError("CLI single-image mask differs from same-runtime reference")
    expected_stack_shape = (2, *expected.shape)
    if stack.shape != expected_stack_shape:
        raise ValueError("CLI stack dimensions changed")
    stack_mismatches = [int(np.count_nonzero((plane > 0) != expected)) for plane in stack]
    if any(stack_mismatches):
        raise ValueError("CLI stack plane differs from the single-plane reference")
    for name in ("single.tif", "stack.tif"):
        validate_mask_tiff(output / name, inspect_tiff(inputs / name))
    before = {name: (sha256(output / name), (output / name).stat().st_mtime_ns)
              for name in ("single.tif", "stack.tif")}
    with (case_dir / "resume.log").open("w", encoding="utf-8") as log:
        resumed = subprocess.run(command + ["--resume"], stdout=log,
                                 stderr=subprocess.STDOUT, env=environment, timeout=180)
    after = {name: (sha256(output / name), (output / name).stat().st_mtime_ns)
             for name in before}
    if resumed.returncode or before != after:
        raise ValueError("CLI resume changed a completed mask or failed")
    state = json.loads((output / "organoid-light.json").read_text())
    if state.get("status") != "complete" or any(
        row.get("status") != "complete" for row in state["files"].values()
    ):
        raise ValueError("CLI completion journal failed")
    if not (output / "quality.csv").is_file():
        raise ValueError("CLI quality report missing")
    resume_loaded_model = "Using " in (case_dir / "resume.log").read_text()
    if resume_loaded_model:
        raise ValueError("Completed resume unexpectedly initialized model inference")
    return {
        "exit_code": process.returncode,
        "single_mask_mismatched_pixels": single_mismatch,
        "stack_shape": list(stack.shape),
        "stack_planes_mismatched_pixels": stack_mismatches,
        "stack_is_two_repeated_real_planes": True,
        "uint8_0_255_minisblack_verified": True,
        "quality_report_present": True,
        "resume_exit_code": resumed.returncode,
        "resume_preserved_mask_hashes_and_mtimes": before == after,
        "resume_reinitialized_model": resume_loaded_model,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def validate_case(case: dict, index: int, records: dict, models_dir: Path,
                  reference_io, reference_tiles, reference_nuc, private_root: Path, tf):
    import numpy as np
    import tifffile
    from organoid_light.engine import Segmenter
    from organoid_light.models import model_spec
    from organoid_light.preprocessing import normalize_image

    started = time.monotonic()
    alias = case.get("id", f"case_{index:03d}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
        raise ValueError("Case id must be a portable public alias")
    organelle = case["organelle"]
    spec = model_spec(organelle)
    source = Path(case["input"]).expanduser().resolve()
    golden = Path(case["golden_mask"]).expanduser().resolve()
    source_hash = sha256(source)
    raw = reference_io.read_tiff_2d(source)
    normalized = reference_io.normalize_image(raw, 1.0, 99.8)
    if not np.array_equal(normalized, normalize_image(raw, 1.0, 99.8)):
        raise ValueError("Package preprocessing differs from original")
    pairs = release_models(records, spec, models_dir)
    original_probabilities, probability_records = [], []
    for source_id, original_path, exported_path, asset in pairs:
        maps = []
        for path in (original_path, exported_path):
            model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
            with tf.device("/CPU:0"):
                probability = reference_tiles.predict_sliding_window(
                    model, normalized, tile_size=512, overlap=128, prediction_batch_size=4
                )
            maps.append(probability)
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        equal = bool(np.array_equal(maps[0], maps[1]))
        if not equal or not all(np.isfinite(p).all() for p in maps):
            raise ValueError("Original/exported same-runtime probabilities differ")
        original_probabilities.append(maps[0])
        probability_records.append({
            "id": source_id, "source_sha256": asset["source_sha256"],
            "exported_sha256": asset["sha256"], "probability_exact_equal": equal,
            "max_absolute_error": float(np.max(np.abs(maps[0] - maps[1]))),
        })
        print(f"REFERENCE_MODEL_PASS case={alias} model={source_id}", flush=True)

    expected = original_mask(original_probabilities, organelle, spec, reference_nuc)
    segmenter = Segmenter(spec, [row[2] for row in pairs], tf, "/CPU:0", 4)
    package_mask = segmenter(raw)
    package_mismatch = int(np.count_nonzero(package_mask != expected))
    if package_mask.shape != raw.shape or package_mask.dtype != np.bool_ or package_mismatch:
        raise ValueError("Package Segmenter differs from original reference mask")
    del segmenter, original_probabilities
    tf.keras.backend.clear_session()
    gc.collect()

    historical = tifffile.imread(golden)
    if (historical.shape != raw.shape or historical.dtype != np.uint8
            or not np.isin(historical, [0, 255]).all()):
        raise ValueError("Historical golden mask has invalid encoding or dimensions")
    historical_mismatch = int(np.count_nonzero((historical > 0) != expected))
    case_dir = private_root / alias
    case_dir.mkdir()
    cli = run_cli(case_dir, organelle, models_dir, source, raw, expected)
    if sha256(source) != source_hash:
        raise ValueError("Original input changed during validation")
    result = {
        "case_id": alias, "organelle": organelle, "status": "PASS",
        "shape": list(raw.shape), "input_dtype": str(raw.dtype),
        "input_sha256": source_hash, "historical_mask_sha256": sha256(golden),
        "normalization_exact_equal": True, "model_probability_checks": probability_records,
        "package_vs_same_runtime_reference_mismatched_pixels": package_mismatch,
        "historical_vs_same_runtime_reference_mismatched_pixels": historical_mismatch,
        "historical_difference_policy": "reported separately; not silently accepted as exact historical reproduction",
        "foreground_pixels": int(np.count_nonzero(expected)), "cli": cli,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    print(f"REFERENCE_CASE_PASS case={alias} historical_pixel_differences={historical_mismatch}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-code-dir", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--organelle", choices=["mito", "ld", "nuc"])
    args = parser.parse_args()
    records = {row["id"]: row for row in json.loads(args.source_map.read_text())}
    cases = json.loads(args.cases.read_text())
    cases = [row for row in cases if not args.organelle or row["organelle"] == args.organelle]
    if not cases:
        raise ValueError("No matching reference cases")
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing validation report")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    code_dir = args.research_code_dir.expanduser().resolve()
    names = ["tiff_io", "sliding_window", "nuc_postprocess"]
    references = [load_reference(code_dir, name) for name in names]
    private_root = Path(tempfile.mkdtemp(prefix="organoid-light-reference-"))
    report = {
        "status": "RUNNING", "device": "CPU", "scope": "implementation-equivalence, not accuracy validation",
        "versions": {name: importlib.metadata.version(name)
                     for name in ["tensorflow", "keras", "numpy", "scipy", "tifffile"]},
        "python": sys.version.split()[0],
        "reference_code_sha256": {name: sha256(code_dir / f"{name}.py") for name in names},
        "cases": [],
    }
    try:
        for index, case in enumerate(cases):
            report["cases"].append(validate_case(
                case, index, records, args.models_dir.resolve(), *references, private_root, tf
            ))
        report["status"] = "PASS"
        shutil.rmtree(private_root)
    except Exception as error:
        report["status"] = "FAIL"
        report["error_type"] = type(error).__name__
        # Logs and copied pixels remain only in the private temporary directory.
        print(f"Private diagnostic files retained at {private_root}", file=sys.stderr)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
