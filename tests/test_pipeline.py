"""Folder transactions and recovery, using only synthetic TIFFs and a fake model."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from organoid_light.models import model_spec
from organoid_light.pipeline import MANIFEST_NAME, STAGING_NAME, discover_inputs, run_batch


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.outputs = self.root / "outputs"
        self.spec = model_spec("mito")
        self.predicted: list[int] = []
        self.factories = 0

    def tearDown(self) -> None:
        self.temp.cleanup()

    def source(self, name: str, data: np.ndarray, axes: str = "YX") -> Path:
        path = self.inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": axes})
        return path

    def factory(self):
        self.factories += 1

        def predict(plane):
            self.predicted.append(int(plane[0, 0]))
            return plane > 0

        return predict, {"device": "synthetic"}

    def run_batch(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_batch(self.inputs, self.outputs, self.spec, self.factory, **kwargs)

    def test_folder_2d_and_stack_preserve_grid_and_names(self) -> None:
        self.source("slice_2.tif", np.ones((17, 23), np.uint16))
        stack = np.zeros((3, 19, 21), np.uint16)
        stack[1, 4:12, 3:18] = 2
        self.source("slice_10.tiff", stack, "ZYX")
        state = self.run_batch()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(self.factories, 1)
        self.assertEqual(len(self.predicted), 4)
        self.assertEqual(list(state["files"]), ["slice_2.tif", "slice_10.tiff"])
        np.testing.assert_array_equal(tifffile.imread(self.outputs / "slice_10.tiff"), (stack > 0) * 255)
        self.assertEqual(tifffile.imread(self.outputs / "slice_2.tif").shape, (17, 23))
        quality = (self.outputs / "quality.csv").read_text()
        self.assertIn("empty_mask", quality)
        self.assertIn("all_foreground", quality)
        self.assertNotIn(str(self.inputs), json.dumps(state))

    def test_partial_failure_resume_does_not_recompute_completed_file(self) -> None:
        self.source("first.tif", np.ones((9, 13), np.uint16))
        self.source("second.tif", np.full((9, 13), 2, np.uint16))
        calls = []

        def failing_factory():
            def predict(plane):
                calls.append(int(plane[0, 0]))
                if calls[-1] == 2:
                    raise RuntimeError("synthetic interruption")
                return plane > 0
            return predict, {"device": "synthetic"}

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "interruption"):
            run_batch(self.inputs, self.outputs, self.spec, failing_factory)
        first_bytes = (self.outputs / "first.tif").read_bytes()
        interrupted = json.loads((self.outputs / MANIFEST_NAME).read_text())
        self.assertEqual(interrupted["files"]["first.tif"]["status"], "complete")
        self.assertEqual(interrupted["status"], "interrupted")
        completed = self.run_batch(resume=True)
        self.assertEqual(self.predicted, [2])
        self.assertEqual(completed["status"], "complete")
        self.assertEqual((self.outputs / "first.tif").read_bytes(), first_bytes)

    def test_partial_resume_preserves_per_file_runtime_provenance(self) -> None:
        self.source("first.tif", np.ones((9, 13), np.uint16))
        self.source("second.tif", np.full((9, 13), 2, np.uint16))

        def cpu_factory():
            def predict(plane):
                if plane[0, 0] == 2:
                    raise RuntimeError("synthetic CPU interruption")
                return plane > 0
            return predict, {"device": "/CPU:0", "backend": "first-session"}

        def gpu_factory():
            return lambda plane: plane > 0, {"device": "/GPU:0", "backend": "resumed-session"}

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            run_batch(self.inputs, self.outputs, self.spec, cpu_factory)
        original_first = (self.outputs / "first.tif").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            completed = run_batch(self.inputs, self.outputs, self.spec, gpu_factory, resume=True)
        history = completed["runtime_history"]
        self.assertEqual(len(history), 2)
        first_runtime = history[completed["files"]["first.tif"]["runtime_id"]]
        second_runtime = history[completed["files"]["second.tif"]["runtime_id"]]
        self.assertEqual(first_runtime["device"], "/CPU:0")
        self.assertEqual(first_runtime["backend"], "first-session")
        self.assertEqual(second_runtime["device"], "/GPU:0")
        self.assertEqual(second_runtime["backend"], "resumed-session")
        self.assertTrue(first_runtime["started_utc"])
        self.assertTrue(second_runtime["started_utc"])
        self.assertEqual((self.outputs / "first.tif").read_bytes(), original_first)
        with patch.object(self, "factory", side_effect=AssertionError("No new runtime needed")):
            verified = self.run_batch(resume=True)
        self.assertEqual(verified["runtime_history"], history)

    def test_prepared_file_recovers_after_publication_crash_without_model(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        real_link = os.link

        def fail_at_publication(source, destination, *args, **kwargs):
            if Path(destination) == self.outputs / "slice.tif":
                raise KeyboardInterrupt()
            return real_link(source, destination, *args, **kwargs)

        with patch("organoid_light.pipeline.os.link", side_effect=fail_at_publication):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        interrupted = json.loads((self.outputs / MANIFEST_NAME).read_text())
        self.assertEqual(interrupted["files"]["slice.tif"]["status"], "prepared")
        self.assertFalse((self.outputs / "slice.tif").exists())
        self.assertEqual(len(list((self.outputs / STAGING_NAME).glob("*.tif"))), 1)
        before_calls = self.predicted[:]
        with patch.object(self, "factory", side_effect=AssertionError("Model must not load")):
            completed = self.run_batch(resume=True)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(self.predicted, before_calls)
        self.assertTrue((self.outputs / "slice.tif").is_file())
        self.assertEqual(list((self.outputs / STAGING_NAME).glob("*.tif")), [])

    def test_completed_resume_never_loads_predictor(self) -> None:
        self.source("empty.tif", np.zeros((9, 13), np.uint16))
        initial = self.run_batch()
        self.assertEqual(initial["files"]["empty.tif"]["quality"]["foreground_pixels"], [0])
        with patch.object(self, "factory", side_effect=AssertionError("Unexpected model load")):
            resumed = self.run_batch(resume=True)
        self.assertEqual(resumed["status"], "complete")

    def test_partial_stack_never_publishes_and_resumes_at_file_boundary(self) -> None:
        stack = np.stack([np.full((9, 13), value, np.uint16) for value in (1, 2, 3)])
        self.source("stack.tif", stack, "ZYX")

        def failing_factory():
            def predict(plane):
                if plane[0, 0] == 2:
                    raise RuntimeError("interrupted within stack")
                return plane > 0
            return predict, {"device": "synthetic"}

        with self.assertRaisesRegex(RuntimeError, "within stack"):
            run_batch(self.inputs, self.outputs, self.spec, failing_factory)
        self.assertFalse((self.outputs / "stack.tif").exists())
        self.assertEqual(list((self.outputs / STAGING_NAME).glob("*.tif")), [])
        completed = self.run_batch(resume=True)
        self.assertEqual(self.predicted, [1, 2, 3])
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(tifffile.imread(self.outputs / "stack.tif").shape, (3, 9, 13))

    def test_prepared_record_recovers_after_link_before_journal_commit(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        real_link = os.link

        def fail_after_publication(source, destination, *args, **kwargs):
            result = real_link(source, destination, *args, **kwargs)
            if Path(destination) == self.outputs / "slice.tif":
                raise KeyboardInterrupt()
            return result

        with patch("organoid_light.pipeline.os.link", side_effect=fail_after_publication):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        self.assertTrue((self.outputs / "slice.tif").exists())
        state = json.loads((self.outputs / MANIFEST_NAME).read_text())
        self.assertEqual(state["files"]["slice.tif"]["status"], "prepared")
        with patch.object(self, "factory", side_effect=AssertionError("Unexpected model load")):
            self.assertEqual(self.run_batch(resume=True)["status"], "complete")
        self.assertEqual(list((self.outputs / STAGING_NAME).glob("*.tif")), [])

    def test_changed_input_is_rejected_before_predictor_load(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        self.run_batch()
        self.source("slice.tif", np.full((9, 13), 2, np.uint16))
        with patch.object(self, "factory", side_effect=AssertionError("Unexpected model load")):
            with self.assertRaisesRegex(ValueError, "input files or hashes changed"):
                self.run_batch(resume=True)

    def test_changed_output_is_rejected_and_left_untouched(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        self.run_batch()
        mask = self.outputs / "slice.tif"
        mask.write_bytes(b"corrupt mask bytes")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.run_batch(resume=True)
        self.assertEqual(mask.read_bytes(), b"corrupt mask bytes")

    def test_method_and_batch_size_mismatch_rejected(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        self.run_batch()
        original = self.spec
        self.spec = copy.deepcopy(self.spec)
        self.spec["version"] = "different-method"
        with self.assertRaisesRegex(ValueError, "model/method/version/batch size"):
            self.run_batch(resume=True)
        self.spec = original
        with self.assertRaisesRegex(ValueError, "model/method/version/batch size"):
            self.run_batch(resume=True, batch_size=1)

    def test_unknown_outputs_cannot_be_adopted_or_overwritten(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        self.outputs.mkdir()
        unrelated = self.outputs / "slice.tif"
        unrelated.write_bytes(b"existing user result")
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            self.run_batch(resume=True)
        self.assertEqual(unrelated.read_bytes(), b"existing user result")
        self.assertEqual(self.factories, 0)

    def test_extra_output_after_completed_run_is_rejected(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        self.run_batch()
        extra = self.outputs / "unrelated.tif"
        extra.write_bytes(b"do not touch")
        with self.assertRaisesRegex(ValueError, "not recorded"):
            self.run_batch(resume=True)
        self.assertEqual(extra.read_bytes(), b"do not touch")

    def test_output_must_not_be_same_or_nested_in_input(self) -> None:
        self.source("slice.tif", np.ones((9, 13), np.uint16))
        for output in (self.inputs, self.inputs / "masks"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                run_batch(self.inputs, output, self.spec, self.factory)
        self.assertFalse((self.inputs / "masks").exists())
        self.assertEqual(self.factories, 0)

    def test_recursive_duplicate_basenames_mirror_folders(self) -> None:
        self.source("cell1/slice.tif", np.zeros((9, 13), np.uint16))
        self.source("cell2/slice.tif", np.ones((9, 13), np.uint16))
        state = self.run_batch(recursive=True)
        self.assertEqual(set(state["files"]), {"cell1/slice.tif", "cell2/slice.tif"})
        self.assertFalse(tifffile.imread(self.outputs / "cell1/slice.tif").any())
        self.assertTrue(tifffile.imread(self.outputs / "cell2/slice.tif").all())

    def test_case_collisions_rejected_when_host_can_represent_them(self) -> None:
        self.source("a.tif", np.zeros((9, 13), np.uint16))
        self.source("A.tif", np.ones((9, 13), np.uint16))
        if len(list(self.inputs.iterdir())) != 2:
            self.skipTest("Host filesystem is already case-insensitive")
        with self.assertRaisesRegex(ValueError, "collide"):
            discover_inputs(self.inputs)

    def test_hidden_appledouble_files_are_never_read(self) -> None:
        self.source("slice.tif", np.zeros((9, 13), np.uint16))
        (self.inputs / "._slice.tif").write_bytes(b"not a TIFF")
        self.source(".hidden/slice.tif", np.ones((9, 13), np.uint16))
        state = self.run_batch(recursive=True)
        self.assertEqual(list(state["files"]), ["slice.tif"])


if __name__ == "__main__":
    unittest.main()
