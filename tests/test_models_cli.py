"""No-network asset integrity checks and dependency-light CLI tests."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from organoid_light.models import ensure_models, model_spec, verify_asset
from organoid_light.pipeline import run_batch


class ModelDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.payload = b"synthetic checkpoint bytes; not an actual model"
        self.asset = {
            "filename": "synthetic.keras",
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "url": "https://example.invalid/synthetic.keras",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def download(self, body: bytes):
        with patch("organoid_light.models.model_spec", return_value={"assets": [self.asset]}):
            with patch("organoid_light.models.urllib.request.urlopen", return_value=io.BytesIO(body)) as fetch:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = ensure_models("mito", self.root)
                return result, fetch.call_count

    def test_valid_download_and_cached_reuse_require_matching_size_and_hash(self) -> None:
        paths, calls = self.download(self.payload)
        self.assertEqual(calls, 1)
        self.assertEqual(paths, [self.root / self.asset["filename"]])
        self.assertEqual(paths[0].read_bytes(), self.payload)
        _, calls = self.download(b"a cached model must never request this")
        self.assertEqual(calls, 0)
        self.assertEqual(list(self.root.glob(".download-*")), [])

    def test_truncated_oversized_and_bad_hash_downloads_are_not_published(self) -> None:
        for bad in (self.payload[:-1], self.payload + b"x", b"x" * len(self.payload)):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                self.download(bad)
            self.assertFalse((self.root / self.asset["filename"]).exists())
            self.assertEqual(list(self.root.glob(".download-*")), [])

    def test_existing_corrupt_asset_is_never_replaced_implicitly(self) -> None:
        target = self.root / self.asset["filename"]
        target.write_bytes(b"corrupt existing checkpoint")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.download(self.payload)
        self.assertEqual(target.read_bytes(), b"corrupt existing checkpoint")

    def test_symlinked_model_refused(self) -> None:
        real = self.root / "actual.keras"
        real.write_bytes(self.payload)
        target = self.root / self.asset["filename"]
        try:
            target.symlink_to(real)
        except OSError:
            self.skipTest("Host does not permit symlink creation")
        with self.assertRaisesRegex(ValueError, "symlinked"):
            verify_asset(target, self.asset)

    def test_symlinked_download_lock_is_rejected_before_opening_or_network(self) -> None:
        protected = self.root / "protected.txt"
        protected.write_bytes(b"must remain unchanged")
        lock = self.root / (self.asset["filename"] + ".lock")
        try:
            lock.symlink_to(protected)
        except OSError:
            self.skipTest("Host does not permit symlink creation")
        with patch("organoid_light.models.model_spec", return_value={"assets": [self.asset]}), \
                patch("organoid_light.models.FileLock") as locking, \
                patch("organoid_light.models.urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "symlinked model lock"):
                ensure_models("mito", self.root)
            locking.assert_not_called()
            network.assert_not_called()
        self.assertEqual(protected.read_bytes(), b"must remain unchanged")
        self.assertTrue(lock.is_symlink())
        self.assertFalse((self.root / self.asset["filename"]).exists())


class CliTests(unittest.TestCase):
    def run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        # Fail immediately if a help/list/complete-resume path tries to import
        # TensorFlow. This works even in a Python environment without TF.
        code = """
import importlib.abc
import sys
class RejectTensorFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tensorflow' or fullname.startswith('tensorflow.'):
            raise AssertionError('TensorFlow must not be imported here')
sys.meta_path.insert(0, RejectTensorFlow())
from organoid_light.cli import main
result = main(sys.argv[1:])
assert 'tensorflow' not in sys.modules
raise SystemExit(result)
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        return subprocess.run([sys.executable, "-c", code, *args], text=True,
                              capture_output=True, timeout=30, env=env)

    def test_help_and_model_list_without_tensorflow(self) -> None:
        for arguments in (["--help"], ["models", "--help"], ["models", "list"], ["--version"]):
            with self.subTest(arguments=arguments):
                result = self.run_cli(arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip())
        models = self.run_cli(["models", "list"])
        for organelle in ("mito:", "ld:", "nuc:"):
            self.assertIn(organelle, models.stdout)

    def test_cli_usage_errors_have_nonzero_status(self) -> None:
        for arguments in ([], ["--organelle", "unknown"], ["models", "unknown"]):
            with self.subTest(arguments=arguments):
                result = self.run_cli(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error", result.stderr.lower())

    def test_missing_input_fails_without_tensorflow_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_cli(["--organelle", "mito", "--input", str(root / "missing"),
                                   "--output", str(root / "output")])
            self.assertEqual(result.returncode, 1)
            self.assertIn("Input does not exist", result.stderr)
            self.assertFalse((root / "output").exists())

    def test_completed_resume_is_available_without_tensorflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, outputs = root / "inputs", root / "outputs"
            inputs.mkdir()
            tifffile.imwrite(inputs / "slice.tif", np.zeros((9, 13), np.uint16),
                             photometric="minisblack", metadata={"axes": "YX"})
            with contextlib.redirect_stdout(io.StringIO()):
                run_batch(inputs, outputs, model_spec("mito"),
                          lambda: (lambda p: p > 0, {"device": "synthetic"}))
            result = self.run_cli(["--organelle", "mito", "--input", str(inputs),
                                   "--output", str(outputs), "--resume"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Verified 1/1", result.stdout)
            self.assertIn("COMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
