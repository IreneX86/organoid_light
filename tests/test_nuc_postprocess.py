"""TensorFlow-free checks for the reviewed nucleus-mask postprocessing policy."""

from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from organoid_light.nuc_postprocess import _guarded_close, postprocess_nucleus_masks


CONNECT8 = np.ones((3, 3), dtype=bool)


class NucPostprocessTests(unittest.TestCase):
    def test_empty_mask_is_a_valid_true_negative(self) -> None:
        empty = np.zeros((37, 51), dtype=np.uint8)
        result, report = postprocess_nucleus_masks(empty, empty)
        self.assertEqual(result.shape, empty.shape)
        self.assertEqual(result.dtype, np.bool_)
        self.assertFalse(result.any())
        self.assertEqual(report["raw_components"], 0)
        self.assertEqual(report["components_after_split"], 0)
        self.assertEqual(report["accepted_splits"], [])
        json.dumps(report)  # The CLI may save this report directly.

    def test_rejects_grayscale_or_unpaired_high_core(self) -> None:
        low = np.zeros((32, 49), dtype=np.uint8)
        high = low.copy()
        low[2, 3] = 128
        with self.assertRaisesRegex(ValueError, "binary values|only 0/1"):
            postprocess_nucleus_masks(low, high)
        low[2, 3] = 0
        high[2, 3] = 255
        with self.assertRaisesRegex(ValueError, "subset"):
            postprocess_nucleus_masks(low, high)
        with self.assertRaisesRegex(ValueError, "identical pixel grids"):
            postprocess_nucleus_masks(low, np.zeros((31, 49), dtype=bool))

    def test_small_component_removed_and_hole_filled(self) -> None:
        low = np.zeros((96, 128), dtype=bool)
        low[15:55, 12:52] = True
        low[29:38, 26:35] = False
        low[70:78, 70:78] = True  # 64 pixels is below the raw 500 px cutoff.
        result, report = postprocess_nucleus_masks(low, np.zeros_like(low))
        self.assertFalse(result[72, 72])
        self.assertTrue(result[33, 30])
        self.assertEqual(report["raw_components"], 2)
        self.assertEqual(report["retained_components"], 1)
        self.assertEqual(report["components_after_split"], 1)
        self.assertFalse(np.any(ndimage.binary_fill_holes(result) & ~result))

    def test_closing_does_not_join_two_independent_nuclei(self) -> None:
        low = np.zeros((100, 120), dtype=bool)
        low[20:65, 15:45] = True
        low[20:65, 49:79] = True
        result, report = postprocess_nucleus_masks(low, np.zeros_like(low))
        _, count = ndimage.label(result, CONNECT8)
        self.assertEqual(count, 2)
        self.assertEqual(report["components_before_split"], 2)
        self.assertEqual(report["components_after_split"], 2)
        self.assertTrue(result[35, 30])
        self.assertTrue(result[35, 64])

    def test_closing_drops_foreground_unconnected_to_any_source(self) -> None:
        # A radius-4 close can make a detached island between nearby objects.
        # Lower the area guard only for this tiny geometric regression fixture;
        # production still requires 500-pixel source and final components.
        base = np.zeros((21, 21), dtype=bool)
        base[12, 1] = base[12, 8] = base[17, 4] = True
        base_labels, source_count = ndimage.label(base, CONNECT8)
        self.assertEqual(source_count, 3)
        with patch("organoid_light.nuc_postprocess._MIN_AREA", 1):
            closed, reverted_groups = _guarded_close(base, base_labels)
        np.testing.assert_array_equal(closed, base)
        self.assertEqual(reverted_groups, 0)

    def test_global_median_hole_uses_remove_only_smoothing(self) -> None:
        # Median5 bridges a narrow background opening in this large object,
        # making a hole despite leaving its component count unchanged.
        closed = np.zeros((80, 80), dtype=bool)
        closed[5:75, 5:75] = True
        closed[5:40, 40:41] = False
        closed[30:35, 39:42] = False
        self.assertFalse(np.any(ndimage.binary_fill_holes(closed) & ~closed))
        global_median = ndimage.median_filter(closed, size=5, mode="nearest")
        self.assertTrue(np.any(ndimage.binary_fill_holes(global_median) & ~global_median))
        # Isolate the smoother: a real radius-4 close would fill this fixture's
        # narrow opening before the median runs.
        with patch("organoid_light.nuc_postprocess._guarded_close", return_value=(closed, 0)):
            result, report = postprocess_nucleus_masks(closed, np.zeros_like(closed))
        self.assertEqual(report["smoothing_mode"], "per_component_remove_only_median5")
        self.assertFalse(np.any(ndimage.binary_fill_holes(result) & ~result))
        self.assertFalse(np.any(result & ~closed))

    def test_two_high_cores_split_one_component_at_image_edge(self) -> None:
        low = np.zeros((120, 180), dtype=bool)
        low[15:100, 0:160] = True  # Touches the left image boundary.
        high = np.zeros_like(low)
        high[30:80, 0:25] = True
        high[30:80, 115:145] = True
        result, report = postprocess_nucleus_masks(low, high)
        labels, count = ndimage.label(result, CONNECT8)
        self.assertEqual(count, 2)
        self.assertEqual(len(report["accepted_splits"]), 1)
        self.assertFalse(np.any(result & ~low))
        self.assertFalse(np.any(high & ~result))
        self.assertNotEqual(labels[50, 10], labels[50, 130])
        self.assertTrue(np.all(np.bincount(labels.ravel())[1:] >= 500))
        self.assertFalse(np.any(ndimage.binary_fill_holes(result) & ~result))
        # Digest produced by the frozen research implementation on this wholly
        # synthetic fixture. It protects the exact reviewed separation shape.
        self.assertEqual(
            hashlib.sha256(result.astype(np.uint8).tobytes()).hexdigest(),
            "c6540a3adcc2297853501cae13d474ddc45a9562d7f78a989a82a9dc63f34dc4",
        )

    def test_three_strong_cores_are_ambiguous_and_unchanged(self) -> None:
        low = np.zeros((110, 240), dtype=bool)
        low[15:95, 5:225] = True
        high = np.zeros_like(low)
        for x0 in (20, 90, 160):
            high[30:70, x0 : x0 + 25] = True
        result, report = postprocess_nucleus_masks(low, high)
        self.assertEqual(report["accepted_splits"], [])
        self.assertEqual(len(report["ambiguous_multiple_cores"]), 1)
        self.assertEqual(report["components_after_split"], 1)
        # The pre-split median may round outer corners, but an ambiguous
        # multi-core component must receive no additional split-side edits.
        cleaned_without_cores, _ = postprocess_nucleus_masks(low, np.zeros_like(low))
        np.testing.assert_array_equal(result, cleaned_without_cores)

    def test_invalid_split_is_skipped_without_changing_cleaned_component(self) -> None:
        low = np.zeros((120, 180), dtype=bool)
        low[15:100, 0:160] = True
        high = np.zeros_like(low)
        high[30:80, 0:25] = True
        high[30:80, 115:145] = True
        cleaned_without_cores, _ = postprocess_nucleus_masks(low, np.zeros_like(low))
        with patch("organoid_light.nuc_postprocess._split_pair", side_effect=ValueError("unsafe boundary")):
            result, report = postprocess_nucleus_masks(low, high)
        np.testing.assert_array_equal(result, cleaned_without_cores)
        self.assertEqual(report["accepted_splits"], [])
        self.assertEqual(len(report["skipped_two_core_candidates"]), 1)
        self.assertIn("unsafe boundary", report["skipped_two_core_candidates"][0]["reason"])



if __name__ == "__main__":
    unittest.main()
