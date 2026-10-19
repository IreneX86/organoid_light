"""Strict, plane-wise TIFF reading and atomic same-grid binary mask writing."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


TIFF_SUFFIXES = frozenset({".tif", ".tiff"})


@dataclass(frozen=True)
class TiffInfo:
    """Image layout; a Z stack is processed as independent 2D planes."""

    shape: tuple[int, ...]
    axes: str
    dtype: str
    plane_shape: tuple[int, int]
    plane_count: int
    is_stack: bool


def _inspect_open(tiff: tifffile.TiffFile) -> TiffInfo:
    if len(tiff.series) != 1:
        raise ValueError("Expected exactly one TIFF image series")
    series = tiff.series[0]
    shape = tuple(int(value) for value in series.shape)
    axes = series.axes
    if not shape or not all(shape):
        raise ValueError("Empty TIFF images are not supported")
    # Q/I represent ordinary sequential grayscale pages without calibrated Z
    # metadata. No ordering is inferred across files; no 3D model is applied.
    if axes == "YX" and len(shape) == 2:
        plane_count, plane_shape = 1, shape
    elif axes in {"ZYX", "QYX", "IYX"} and len(shape) == 3:
        plane_count, plane_shape = shape[0], shape[1:]
    else:
        raise ValueError(
            f"Unsupported TIFF axes {axes!r}, shape {shape}; expected YX or "
            "grayscale ZYX/QYX/IYX. Select channels/time points explicitly first."
        )
    dtype = np.dtype(series.dtype)
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"Expected real numeric TIFF pixels, got {dtype}")
    if len(series.pages) != plane_count:
        raise ValueError("TIFF must store one 2D plane per page for bounded-memory reading")
    for page in series.pages:
        # TiffFrame inherits metadata from its keyframe; its shape is retained.
        keyframe = getattr(page, "keyframe", page)
        if tuple(page.shape) != plane_shape:
            raise ValueError("TIFF pages do not share one consistent 2D pixel grid")
        if keyframe.samplesperpixel != 1 or keyframe.photometric not in (0, 1):
            raise ValueError("RGB, palette and multi-sample TIFF images are not supported")
        if np.dtype(page.dtype) != dtype:
            raise ValueError("TIFF pages have inconsistent pixel types")
    return TiffInfo(shape, axes, str(dtype), plane_shape, plane_count, len(shape) == 3)


def inspect_tiff(path: str | Path) -> TiffInfo:
    """Read TIFF layout without loading the complete pixel array."""
    with tifffile.TiffFile(path) as tiff:
        return _inspect_open(tiff)


def iter_tiff_planes(path: str | Path) -> Iterator[np.ndarray]:
    """Yield finite, real, 2D grayscale arrays, loading one page at a time.

    Spatial singleton axes remain intact: a 1×W image is not squeezed to 1D.
    A singleton Z axis remains a stack with one plane. Channel/time axes are
    rejected even when singleton, to prevent silent semantic reinterpretation.
    """
    with tifffile.TiffFile(path) as tiff:
        info = _inspect_open(tiff)
        for page in tiff.series[0].pages:
            plane = np.asarray(page.asarray())
            if plane.shape != info.plane_shape or not np.isfinite(plane).all():
                raise ValueError("TIFF contains invalid dimensions or NaN/infinite pixels")
            yield plane


def _encode_mask(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(plane)
    if array.shape != shape:
        raise ValueError(f"Mask shape {array.shape} differs from input plane {shape}")
    if array.dtype != np.bool_ and not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Masks must be bool or integer binary arrays")
    values = set(np.unique(array).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("Masks must contain only 0/1 or 0/255")
    return (array > 0).astype(np.uint8) * 255


def validate_mask_tiff(path: str | Path, info: TiffInfo) -> dict[str, object]:
    """Verify shape, black background, encoding, and every saved mask plane."""
    foreground: list[int] = []
    with tifffile.TiffFile(path) as tiff:
        actual = _inspect_open(tiff)
        if actual.shape != info.shape or actual.plane_shape != info.plane_shape:
            raise ValueError(f"Saved mask shape {actual.shape} differs from input {info.shape}")
        if actual.dtype != "uint8" or actual.plane_count != info.plane_count:
            raise ValueError("Saved mask must be uint8 with the input plane count")
        for page in tiff.series[0].pages:
            if getattr(page, "keyframe", page).photometric != 1:
                raise ValueError("Saved mask must be MINISBLACK: background 0 is black")
            plane = np.asarray(page.asarray())
            if not np.all((plane == 0) | (plane == 255)):
                raise ValueError("Saved mask contains values other than 0 and 255")
            foreground.append(int(np.count_nonzero(plane)))
    pixels = int(np.prod(info.plane_shape))
    return {
        "shape": list(info.shape),
        "dtype": "uint8",
        "plane_count": info.plane_count,
        "foreground_pixels": foreground,
        "foreground_fraction": [value / pixels for value in foreground],
        "empty_planes": [index for index, value in enumerate(foreground) if value == 0],
        "all_foreground_planes": [index for index, value in enumerate(foreground) if value == pixels],
    }


def write_mask_tiff(
    path: str | Path, planes: Iterable[np.ndarray], info: TiffInfo
) -> dict[str, object]:
    """Atomically create one same-grid TIFF; never replace an existing target.

    Planes are consumed once and saved as uint8 0/255 MINISBLACK. The temporary
    file is verified before an exclusive hard-link publication in the same
    directory, avoiding both partial output and overwrite races. ZIP/network
    filesystems without hard-link support fail safely without publishing.
    """
    output_path = Path(path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".organoid-mask-", suffix=".tif", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    source = iter(planes)

    def encoded_planes() -> Iterator[np.ndarray]:
        for _ in range(info.plane_count):
            try:
                plane = next(source)
            except StopIteration as exc:
                raise ValueError("Too few mask planes for the input TIFF") from exc
            yield _encode_mask(plane, info.plane_shape)

    try:
        tifffile.imwrite(
            temporary_path,
            encoded_planes(),
            shape=info.shape,
            dtype=np.uint8,
            photometric="minisblack",
            metadata={"axes": info.axes},
            compression="deflate",
            bigtiff=int(np.prod(info.shape)) >= 2**32 - 2**25,
        )
        try:
            next(source)
        except StopIteration:
            pass
        else:
            raise ValueError("Too many mask planes for the input TIFF")
        quality = validate_mask_tiff(temporary_path, info)
        # Windows FlushFileBuffers requires a write-capable file handle.
        with temporary_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(temporary_path, output_path)
        return quality
    finally:
        temporary_path.unlink(missing_ok=True)
