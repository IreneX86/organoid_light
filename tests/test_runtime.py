"""GPU loader and device isolation checks with mocked runtimes, not GPU work."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from organoid_light.cli import main
from organoid_light.runtime import bootstrap_cuda_loader, configure_runtime


class CudaBootstrapTests(unittest.TestCase):
    def test_linux_pip_libraries_prepend_preserving_visibility_and_reexec_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "nvidia/cublas/lib", root / "nvidia/cudnn/lib"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            environment = {
                "CUDA_VISIBLE_DEVICES": "GPU-assigned-by-scheduler",
                "NVIDIA_VISIBLE_DEVICES": "GPU-assigned-by-container",
                "LD_LIBRARY_PATH": os.pathsep.join(["/existing/libs", str(first)]),
            }
            args = ["--organelle", "mito", "--input", "data", "--output", "masks"]
            with patch.dict(os.environ, environment, clear=True):
                with patch("organoid_light.runtime.sys.platform", "linux"), \
                        patch("organoid_light.runtime.sysconfig.get_path", return_value=str(root)), \
                        patch("organoid_light.runtime.os.execve") as execute:
                    bootstrap_cuda_loader(args, "gpu")
                    execute.assert_called_once()
                    executable, command, passed = execute.call_args.args
                    self.assertEqual(executable, sys.executable)
                    self.assertEqual(command, [sys.executable, "-m", "organoid_light", *args])
                    self.assertEqual(passed["CUDA_VISIBLE_DEVICES"], environment["CUDA_VISIBLE_DEVICES"])
                    self.assertEqual(passed["NVIDIA_VISIBLE_DEVICES"], environment["NVIDIA_VISIBLE_DEVICES"])
                    self.assertEqual(passed["LD_LIBRARY_PATH"].split(os.pathsep),
                                     [str(first), str(second), "/existing/libs"])
                    self.assertEqual(passed["ORGANOID_LIGHT_CUDA_BOOTSTRAPPED"], "1")
                    with patch.dict(os.environ, passed, clear=True):
                        bootstrap_cuda_loader(args, "gpu")
                    execute.assert_called_once()

    def test_cpu_and_nonlinux_do_not_probe_or_reexec(self) -> None:
        for platform, device in (("linux", "cpu"), ("darwin", "gpu"), ("win32", "auto")):
            with self.subTest(platform=platform, device=device):
                with patch.dict(os.environ, {}, clear=True), \
                        patch("organoid_light.runtime.sys.platform", platform), \
                        patch("organoid_light.runtime.sysconfig.get_path") as get_path, \
                        patch("organoid_light.runtime.os.execve") as execute:
                    bootstrap_cuda_loader([], device)
                    get_path.assert_not_called()
                    execute.assert_not_called()

    def test_missing_pip_libraries_need_no_reexec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True), \
                    patch("organoid_light.runtime.sys.platform", "linux"), \
                    patch("organoid_light.runtime.sysconfig.get_path", return_value=directory), \
                    patch("organoid_light.runtime.os.execve") as execute:
                bootstrap_cuda_loader([], "auto")
                execute.assert_not_called()

    def test_cli_bootstrap_happens_before_inference_and_model_download(self) -> None:
        with patch("organoid_light.runtime.bootstrap_cuda_loader", side_effect=SystemExit(99)), \
                patch("organoid_light.pipeline.run_batch") as run, \
                patch("organoid_light.cli.ensure_models") as download:
            with self.assertRaises(SystemExit) as caught:
                main(["--organelle", "mito", "--input", "synthetic", "--output", "masks"])
            self.assertEqual(caught.exception.code, 99)
            run.assert_not_called()
            download.assert_not_called()


def fake_tensorflow(physical_count: int, actual_device: str, logical_count: int):
    physical = [f"physical_gpu_{index}" for index in range(physical_count)]
    logical = [f"logical_gpu_{index}" for index in range(logical_count)]
    config = SimpleNamespace(
        list_physical_devices=MagicMock(return_value=physical),
        set_visible_devices=MagicMock(),
        list_logical_devices=MagicMock(return_value=logical),
        experimental=SimpleNamespace(set_memory_growth=MagicMock()),
    )
    value = SimpleNamespace(numpy=lambda: np.array([[11.0]]), device=actual_device)
    return SimpleNamespace(
        __version__="2.21.0", config=config,
        device=MagicMock(side_effect=lambda _: contextlib.nullcontext()),
        constant=np.asarray,
        linalg=SimpleNamespace(matmul=MagicMock(return_value=value)),
        sysconfig=SimpleNamespace(get_build_info=lambda: {"is_cuda_build": True}),
    )


class RuntimeTests(unittest.TestCase):
    def configure(self, fake, device):
        with patch.dict(sys.modules, {"tensorflow": fake}), \
                patch("organoid_light.runtime.importlib.metadata.version", return_value="synthetic"):
            return configure_runtime(device)

    def test_requested_gpu_missing_fails_before_computation(self) -> None:
        fake = fake_tensorflow(0, "/device:CPU:0", 0)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "GPU requested"):
            self.configure(fake, "gpu")
        fake.linalg.matmul.assert_not_called()

    def test_gpu_fallback_to_cpu_is_a_failure(self) -> None:
        fake = fake_tensorflow(1, "/job:localhost/device:CPU:0", 1)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "Device computation failed"):
            self.configure(fake, "gpu")

    def test_only_first_already_visible_gpu_selected(self) -> None:
        fake = fake_tensorflow(3, "/job:localhost/device:GPU:0", 1)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,6,7"}, clear=True):
            _, metadata = self.configure(fake, "gpu")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4,6,7")
        fake.config.set_visible_devices.assert_called_once_with(["physical_gpu_0"], "GPU")
        fake.config.experimental.set_memory_growth.assert_called_once_with("physical_gpu_0", True)
        self.assertEqual(metadata["visible_gpu_count"], 1)
        self.assertEqual(metadata["device"], "/GPU:0")

    def test_cpu_explicitly_hides_all_gpus(self) -> None:
        fake = fake_tensorflow(3, "/job:localhost/device:CPU:0", 0)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,6,7"}, clear=True):
            _, metadata = self.configure(fake, "cpu")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
        fake.config.set_visible_devices.assert_called_once_with([], "GPU")
        fake.config.experimental.set_memory_growth.assert_not_called()
        self.assertEqual(metadata["device"], "/CPU:0")

    def test_multiple_logical_gpus_fail_closed(self) -> None:
        fake = fake_tensorflow(2, "/device:GPU:0", 2)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "exactly one logical GPU"):
            self.configure(fake, "gpu")


if __name__ == "__main__":
    unittest.main()
