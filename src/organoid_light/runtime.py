"""Configure one visible CPU/GPU runtime without importing training code."""

from __future__ import annotations

import importlib.metadata
import os
import sys
import sysconfig
from pathlib import Path


def bootstrap_cuda_loader(argv: list[str], device: str) -> None:
    """Expose Linux pip CUDA libraries before the process loader initializes.

    Re-exec happens before opening output locks or importing TensorFlow. This
    reproduces the verified research GPU environment without CUDA modules or
    modifying site-packages/system libraries. Existing GPU visibility is kept.
    """
    if (sys.platform != "linux" or device == "cpu"
            or os.environ.get("ORGANOID_LIGHT_CUDA_BOOTSTRAPPED") == "1"):
        return
    roots = {Path(sysconfig.get_path(key)) for key in ("purelib", "platlib")}
    libraries = sorted({str(path) for root in roots
                        for path in (root / "nvidia").glob("*/lib") if path.is_dir()})
    if not libraries:
        return
    environment = os.environ.copy()
    existing = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(
        libraries + [value for value in existing if value]
    ))
    environment["ORGANOID_LIGHT_CUDA_BOOTSTRAPPED"] = "1"
    os.execve(sys.executable, [sys.executable, "-m", "organoid_light", *argv], environment)


def configure_runtime(device: str = "auto") -> tuple[object, dict]:
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu, or gpu")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Keep TensorFlow's native errors visible, while omitting routine INFO chatter.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            'TensorFlow is missing. Install with: python -m pip install ".[cpu]" '
            '(or ".[gpu]" on supported Linux NVIDIA systems).'
        ) from error
    if tf.__version__ != "2.21.0":
        raise RuntimeError(f"This release requires TensorFlow 2.21.0; found {tf.__version__}")
    physical = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        chosen = "/CPU:0"
    elif physical:
        # Respect any scheduler/container CUDA_VISIBLE_DEVICES restriction, and
        # select only the first device within that already-visible set.
        tf.config.set_visible_devices([physical[0]], "GPU")
        tf.config.experimental.set_memory_growth(physical[0], True)
        chosen = "/GPU:0"
    elif device == "gpu":
        raise RuntimeError("GPU requested, but TensorFlow found none; no CPU fallback was used")
    else:
        chosen = "/CPU:0"
    with tf.device(chosen):
        value = tf.linalg.matmul(tf.constant([[1., 2.]]), tf.constant([[3.], [4.]]))
    if float(value.numpy()[0, 0]) != 11.0 or chosen.split(":")[0][1:] not in value.device:
        raise RuntimeError(f"Device computation failed on {chosen}: {value.device}")
    logical = tf.config.list_logical_devices("GPU")
    if chosen == "/GPU:0" and len(logical) != 1:
        raise RuntimeError("Expected exactly one logical GPU after device restriction")
    environment = {
        "python": sys.version.split()[0], "tensorflow": tf.__version__,
        "keras": importlib.metadata.version("keras"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "tifffile": importlib.metadata.version("tifffile"),
        "device": chosen, "matmul_device": value.device,
        "visible_gpu_count": len(logical), "build": tf.sysconfig.get_build_info(),
    }
    return tf, environment
