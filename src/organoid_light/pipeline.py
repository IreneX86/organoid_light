"""Auditable folder/stack inference with crash-safe, hash-verified resume."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from filelock import FileLock

from . import __version__
from .models import sha256_file
from .tiff_io import (
    TIFF_SUFFIXES, inspect_tiff, iter_tiff_planes, validate_mask_tiff, write_mask_tiff,
)


MANIFEST_NAME = "organoid-light.json"
STAGING_NAME = ".organoid-light-staging"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked manifest: {path.name}")
    descriptor, name = tempfile.mkstemp(prefix=".journal-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for name in ("preprocessing.py", "tiling.py", "nuc_postprocess.py", "engine.py", "tiff_io.py"):
        digest.update(name.encode())
        # Normalize source newlines so Git checkout conventions are portable.
        digest.update(files("organoid_light").joinpath(name).read_text(encoding="utf-8")
                      .replace("\r\n", "\n").encode("utf-8"))
    return digest.hexdigest()


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.as_posix())]


def discover_inputs(input_path: Path, recursive: bool = False) -> tuple[Path, list[Path]]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in TIFF_SUFFIXES or path.name.startswith("."):
            raise ValueError("Input must be a real .tif/.tiff image or a TIFF folder")
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    iterator = path.rglob("*") if recursive else path.iterdir()
    images = sorted((p for p in iterator if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES
                     and not any(part.startswith(".") for part in p.relative_to(path).parts)),
                    key=lambda p: _natural_key(p.relative_to(path)))
    if not images:
        raise ValueError("No real TIFF images found (hidden/AppleDouble files are excluded)")
    names = [p.relative_to(path).as_posix().casefold() for p in images]
    if len(names) != len(set(names)):
        raise ValueError("Input paths collide on case-insensitive Windows/macOS filesystems")
    if any(p.is_symlink() or not p.resolve().is_relative_to(path) for p in images):
        raise ValueError("Symlinked inputs outside the selected data directory are unsupported")
    return path, images


def _safe_output(output: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe output path")
    result = output / relative
    for candidate in [result, *result.parents]:
        if candidate == output.parent:
            break
        if candidate.is_symlink():
            raise ValueError("Symlinked output paths are not supported")
    if not result.resolve().is_relative_to(output):
        raise ValueError("Output path escapes destination")
    return result


def _write_quality(output: Path, state: dict) -> None:
    temporary = output / ".quality.csv.tmp"
    if temporary.is_symlink() or (output / "quality.csv").is_symlink():
        raise ValueError("Refusing symlinked quality report")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file", "plane", "height", "width", "foreground_pixels",
                             "foreground_fraction", "warning"])
            for name, record in state["files"].items():
                quality = record["quality"]
                for index, count in enumerate(quality["foreground_pixels"]):
                    warning = "empty_mask" if index in quality["empty_planes"] else (
                        "all_foreground" if index in quality["all_foreground_planes"] else "")
                    writer.writerow([name, index, *record["input"]["shape"][-2:], count,
                                     quality["foreground_fraction"][index], warning])
        os.replace(temporary, output / "quality.csv")
    finally:
        temporary.unlink(missing_ok=True)


def run_batch(input_path: Path, output_dir: Path, spec: dict, make_predictor,
              *, resume: bool = False, recursive: bool = False, batch_size: int = 4) -> dict:
    """Run a frozen method. Factory returns (plane_predictor, runtime_metadata).

    Each file has a durable prepared record *before* publication. After a
    crash, resume can therefore distinguish a verified output from unrelated
    pre-existing masks; only unfinished files are inferred again. A multipage
    stack is one transaction and resumes at file boundaries, not Z boundaries.
    """
    source_root, images = discover_inputs(input_path, recursive)
    output = output_dir.expanduser().absolute()
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = output.resolve()
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError("Output must be outside the input directory")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Inspect all headers and source hashes before downloading/loading any model.
    snapshots, layouts = {}, {}
    for path in images:
        name = path.relative_to(source_root).as_posix()
        info = inspect_tiff(path)
        snapshots[name] = {"sha256": sha256_file(path), "shape": list(info.shape),
                           "dtype": info.dtype, "axes": info.axes}
        layouts[name] = info
    identity = {"package_version": __version__, "implementation_sha256": _implementation_hash(),
                "model": spec, "batch_size": batch_size, "encoding": "uint8_0_255_minisblack"}
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".organoid-light.lock"
    if lock_path.is_symlink():
        raise ValueError("Refusing symlinked output lock")
    with FileLock(str(lock_path), timeout=0):
        manifest_path = output / MANIFEST_NAME
        if manifest_path.exists() or manifest_path.is_symlink():
            if not resume:
                raise FileExistsError("Output has a run manifest; use --resume to verify and continue")
            if manifest_path.is_symlink():
                raise ValueError("Refusing symlinked run manifest")
            state = json.loads(manifest_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != 1 or state.get("identity") != identity:
                raise ValueError("Resume model/method/version/batch size differs from the original run")
            if {k: v.get("input") for k, v in state.get("files", {}).items()} != snapshots:
                raise ValueError("Resume input files or hashes changed; use a fresh output directory")
        else:
            unexpected = [p for p in output.iterdir() if p.name not in {lock_path.name, ".DS_Store"}]
            if unexpected:
                raise FileExistsError("Output is not empty; existing masks cannot be adopted")
            state = {"schema_version": 1, "identity": identity, "started_utc": _now(),
                     "status": "running", "files": {
                         name: {"input": snapshot, "status": "pending"}
                         for name, snapshot in snapshots.items()}}
            _atomic_json(manifest_path, state)
        # Reject extra output TIFFs; they cannot be silently mistaken for this run.
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
                  if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES
                  and STAGING_NAME not in p.relative_to(output).parts}
        if actual - set(snapshots):
            raise ValueError("Output contains TIFFs not recorded in this run")
        staging = output / STAGING_NAME
        if staging.is_symlink():
            raise ValueError("Refusing symlinked staging directory")
        staging.mkdir(exist_ok=True)
        predictor = None
        try:
            for index, path in enumerate(images, start=1):
                name = path.relative_to(source_root).as_posix()
                record, info = state["files"][name], layouts[name]
                target = _safe_output(output, name)
                stage = staging / (hashlib.sha256(name.encode()).hexdigest() + ".tif")
                if stage.is_symlink():
                    raise ValueError("Refusing symlinked staged output")
                if record["status"] in {"prepared", "complete"}:
                    candidate = target if target.exists() else stage
                    if not candidate.is_file() or sha256_file(candidate) != record["output_sha256"]:
                        raise ValueError(f"Resume output is missing or changed: {name}")
                    quality = validate_mask_tiff(candidate, info)
                    if quality != record["quality"]:
                        raise ValueError(f"Resume quality record differs: {name}")
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(stage, target)
                    stage.unlink(missing_ok=True)
                    record["status"] = "complete"
                    _atomic_json(manifest_path, state)
                    print(f"Verified {index}/{len(images)}: {name}", flush=True)
                    continue
                if target.exists():
                    raise FileExistsError(f"Unverified output cannot be overwritten: {name}")
                if record["status"] not in {"running", "pending"}:
                    raise ValueError(f"Unknown resume record state: {name}")
                # A stage from an interrupted prediction is never treated as a
                # completed result; this path is package-owned and not public.
                stage.unlink(missing_ok=True)
                if predictor is None:
                    predictor, state["runtime"] = make_predictor()
                    state.setdefault("runtime_history", []).append({
                        "started_utc": _now(), **state["runtime"],
                    })
                # Completed outputs keep their original backend record even
                # when an interrupted batch resumes on a different machine.
                record["runtime_id"] = len(state["runtime_history"]) - 1
                record["status"] = "running"
                state["status"] = "running"
                _atomic_json(manifest_path, state)
                quality = write_mask_tiff(stage, (predictor(plane) for plane in iter_tiff_planes(path)), info)
                if sha256_file(path) != record["input"]["sha256"]:
                    raise ValueError(f"Input changed during prediction: {name}")
                record.update(status="prepared", output_sha256=sha256_file(stage), quality=quality)
                _atomic_json(manifest_path, state)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(stage, target)
                record["status"] = "complete"
                _atomic_json(manifest_path, state)
                stage.unlink()
                print(f"Saved {index}/{len(images)}: {name} ({info.plane_count} plane(s))", flush=True)
            _write_quality(output, state)
            state.update(status="complete", finished_utc=_now())
            state.pop("error_type", None)
            _atomic_json(manifest_path, state)
        except BaseException as error:
            state.update(status="interrupted", error_type=type(error).__name__)
            _atomic_json(manifest_path, state)
            raise
    return state
