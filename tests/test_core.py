"""Synthetic numerical and file-contract regression tests (no model download)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from organoid_light.preprocessing import normalize_image
from organoid_light.tiff_io import (
    inspect_tiff,
    iter_tiff_planes,
    validate_mask_tiff,
    write_mask_tiff,
)
from organoid_light.tiling import predict_sliding_window


class IdentityModel:
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return batch


class FrozenNumericsTests(unittest.TestCase):
    def test_percentile_normalization_matches_frozen_expression(self) -> None:
        # Separate float32 percentile and arithmetic steps are intentional.
        image = np.random.default_rng(42).integers(0, 65536, (67, 83), dtype=np.uint16)
        converted = image.astype(np.float32)
        low, high = np.percentile(converted, [1, 99.8])
        expected = np.clip((converted - low) / (high - low), 0, 1).astype(np.float32)
        np.testing.assert_array_equal(normalize_image(image), expected)
        np.testing.assert_array_equal(normalize_image(np.full((7, 9), 31)), np.zeros((7, 9)))

    def test_independent_plane_normalization_preserves_intensity_scaling(self) -> None:
        first = np.arange(63, dtype=np.float32).reshape(7, 9)
        second = first * 100 + 5000
        np.testing.assert_allclose(normalize_image(first), normalize_image(second), atol=1e-7)
        self.assertEqual(normalize_image(first).dtype, np.float32)

    def test_normalization_rejects_invalid_values_and_axes(self) -> None:
        for image in (np.array([[np.nan]]), np.array([[np.inf]]),
                      np.empty((0, 7)), np.zeros((2, 4, 4)), np.ones((2, 3), complex)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                normalize_image(image)
        with self.assertRaises(ValueError):
            normalize_image(np.ones((2, 3)), 99, 1)

    def test_identity_tiles_keep_grid_and_edges(self) -> None:
        for shape in ((1, 33), (17, 1), (7, 11), (16, 16), (35, 43), (67, 83)):
            for overlap in (0, 4, 15):
                image = np.random.default_rng(12).random(shape, dtype=np.float32)
                with self.subTest(shape=shape, overlap=overlap):
                    result = predict_sliding_window(IdentityModel(), image, 16, overlap, 3)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_allclose(result, image, rtol=3e-6, atol=3e-6)

    def test_model_failures_are_not_hidden_by_clipping(self) -> None:
        class InvalidModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
                return np.full_like(batch, self.value)

        for value in (np.nan, np.inf, -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                predict_sliding_window(InvalidModel(value), np.zeros((9, 12)), 16, 4, 1)
        with self.assertRaises(ValueError):
            predict_sliding_window(IdentityModel(), np.full((9, 12), np.nan), 16, 4, 1)


class TiffContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def source(self, data: np.ndarray, axes: str = "YX", name: str = "source.tif") -> Path:
        path = self.root / name
        tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": axes})
        return path

    def test_2d_atomic_mask_is_same_grid_uint8_minisblack(self) -> None:
        path = self.source(np.zeros((17, 23), np.uint16))
        info = inspect_tiff(path)
        mask = np.zeros(info.plane_shape, bool)
        mask[2:7, 3:12] = True
        output = self.root / "output" / path.name
        qc = write_mask_tiff(output, [mask], info)
        self.assertEqual(qc["foreground_pixels"], [45])
        self.assertEqual(qc["shape"], [17, 23])
        self.assertEqual(qc, validate_mask_tiff(output, info))
        with tifffile.TiffFile(output) as file:
            self.assertEqual(file.pages[0].photometric, 1)
            self.assertEqual(file.asarray().dtype, np.uint8)
            np.testing.assert_array_equal(file.asarray(), mask * 255)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_mask_tiff(output, [~mask], info)
        self.assertEqual(before, output.read_bytes())

    def test_stack_reads_and_writes_each_plane_without_losing_axes(self) -> None:
        data = np.arange(3 * 7 * 9, dtype=np.uint16).reshape(3, 7, 9)
        path = self.source(data, "ZYX")
        info = inspect_tiff(path)
        self.assertEqual(info.shape, data.shape)
        self.assertEqual(info.plane_count, 3)
        self.assertTrue(info.is_stack)
        for plane, expected in zip(iter_tiff_planes(path), data, strict=True):
            np.testing.assert_array_equal(plane, expected)
        output = self.root / "mask.tif"
        write_mask_tiff(output, (normalize_image(p) >= .5 for p in iter_tiff_planes(path)), info)
        actual = tifffile.imread(output)
        self.assertEqual(actual.shape, data.shape)
        for plane in actual[1:]:
            np.testing.assert_array_equal(plane, actual[0])

    def test_singleton_spatial_and_z_axes_are_preserved(self) -> None:
        for index, (shape, axes) in enumerate((((1, 8), "YX"), ((8, 1), "YX"), ((1, 7, 9), "ZYX"))):
            with self.subTest(shape=shape):
                path = self.source(np.zeros(shape, np.uint16), axes, f"source{index}.tif")
                info = inspect_tiff(path)
                output = self.root / f"mask{index}.tif"
                qc = write_mask_tiff(output, (p > 0 for p in iter_tiff_planes(path)), info)
                self.assertEqual(tuple(qc["shape"]), shape)
                self.assertEqual(tifffile.imread(output).shape, shape)
                self.assertEqual(qc["empty_planes"], [0])

    def test_generic_multipage_stack_is_accepted(self) -> None:
        path = self.root / "multipage.tif"
        tifffile.imwrite(path, np.zeros((5, 7, 9), np.uint16), photometric="minisblack", metadata=None)
        info = inspect_tiff(path)
        self.assertIn(info.axes, {"QYX", "IYX"})
        self.assertEqual(len(list(iter_tiff_planes(path))), 5)

    def test_rgb_channel_time_multiseries_and_corrupt_files_rejected(self) -> None:
        rgb = self.root / "rgb.tif"
        tifffile.imwrite(rgb, np.zeros((7, 9, 3), np.uint8), photometric="rgb")
        paths = [rgb]
        for axes in ("CYX", "TYX"):
            paths.append(self.source(np.zeros((1, 7, 9), np.uint8), axes, f"{axes}.tif"))
        multiple = self.root / "multiple.tif"
        with tifffile.TiffWriter(multiple) as writer:
            writer.write(np.zeros((7, 9), np.uint16), photometric="minisblack")
            writer.write(np.zeros((8, 10), np.uint16), photometric="minisblack")
        paths.append(multiple)
        corrupt = self.root / "corrupt.tif"
        corrupt.write_bytes(b"not a TIFF")
        paths.append(corrupt)
        for path in paths:
            with self.subTest(name=path.name), self.assertRaises((ValueError, tifffile.TiffFileError)):
                inspect_tiff(path)

    def test_nan_rejected_when_plane_is_read(self) -> None:
        data = np.zeros((7, 9), np.float32)
        data[3, 2] = np.nan
        path = self.source(data)
        with self.assertRaises(ValueError):
            list(iter_tiff_planes(path))

    def test_bad_masks_leave_no_partial_or_overwritten_output(self) -> None:
        info = inspect_tiff(self.source(np.zeros((7, 9), np.uint16)))
        output = self.root / "mask.tif"
        for masks in ([], [np.ones((7, 8), bool)], [np.full((7, 9), 127, np.uint8)],
                      [np.full((7, 9), np.nan)], [np.zeros((7, 9), bool)] * 2):
            with self.assertRaises((ValueError, StopIteration)):
                write_mask_tiff(output, masks, info)
            self.assertFalse(output.exists())
            self.assertEqual(list(self.root.glob(".organoid-mask-*")), [])

    def test_dangling_symlink_is_not_overwritten(self) -> None:
        info = inspect_tiff(self.source(np.zeros((7, 9), np.uint16)))
        output = self.root / "mask.tif"
        try:
            output.symlink_to(self.root / "missing.tif")
        except OSError:
            self.skipTest("Host does not permit symlink creation")
        with self.assertRaises(FileExistsError):
            write_mask_tiff(output, [np.zeros((7, 9), bool)], info)
        self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
