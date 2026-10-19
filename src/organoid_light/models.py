"""Versioned model catalog and integrity-checked GitHub Release downloads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path

from filelock import FileLock
from platformdirs import user_cache_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog() -> dict:
    return json.loads(files("organoid_light").joinpath("model_manifest.json").read_text(encoding="utf-8"))


def model_spec(organelle: str) -> dict:
    try:
        return catalog()["models"][organelle]
    except KeyError as error:
        raise ValueError("organelle must be mito, ld, or nuc") from error


def default_models_dir() -> Path:
    return user_cache_path("organoid-light", appauthor=False) / catalog()["release_tag"]


def verify_asset(path: Path, asset: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked model: {path}")
    if path.stat().st_size != asset["bytes"] or sha256_file(path) != asset["sha256"]:
        raise ValueError(
            f"Model integrity check failed: {path.name}. Move the corrupt file aside "
            "and run 'organoid-light models download' again."
        )


def ensure_models(organelle: str, directory: Path | None = None) -> list[Path]:
    """Reuse verified files; download missing assets without accepting stale weights."""
    directory = (directory or default_models_dir()).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for asset in model_spec(organelle)["assets"]:
        destination = directory / asset["filename"]
        lock_path = destination.with_name(destination.name + ".lock")
        if lock_path.is_symlink():
            raise ValueError(f"Refusing symlinked model lock: {lock_path.name}")
        with FileLock(str(lock_path), timeout=120):
            if destination.exists() or destination.is_symlink():
                verify_asset(destination, asset)
            else:
                print(f"Downloading {asset['filename']} ({asset['bytes'] / 1e6:.1f} MB)", flush=True)
                request = urllib.request.Request(asset["url"], headers={
                    "User-Agent": "organoid-light/1.0", "Accept": "application/octet-stream",
                })
                fd, temporary_name = tempfile.mkstemp(prefix=".download-", dir=directory)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(fd, "wb") as target, urllib.request.urlopen(
                        request, timeout=60
                    ) as response:
                        written = 0
                        while chunk := response.read(1024 * 1024):
                            written += len(chunk)
                            if written > asset["bytes"]:
                                raise ValueError("Release asset exceeds its registered size")
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                    verify_asset(temporary, asset)
                    # The per-asset process lock serializes package downloads.
                    if destination.exists() or destination.is_symlink():
                        verify_asset(destination, asset)
                    else:
                        os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            paths.append(destination)
    return paths
