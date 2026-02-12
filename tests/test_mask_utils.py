"""
Tests for bactimate.gui.mask_utils (binary -> instance label conversion for Omnipose).
Run with: pytest tests/test_mask_utils.py -v
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bactimate.gui.mask_utils import convert_single_mask_to_omnipose, convert_masks_in_folder


def test_convert_binary_mask_returns_converted():
    """Binary 0/255 mask should be converted to instance labels and return ('converted', '')."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mask.tif"
        # Two separate regions (2 cells) as 255, background 0
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[2:4, 2:4] = 255
        arr[6:8, 6:8] = 255
        tifffile.imwrite(path, arr)
        status, msg = convert_single_mask_to_omnipose(path, out_path=path)
        assert status == "converted", msg
        out = tifffile.imread(path)
        assert out.ndim == 2
        unique = np.unique(out)
        assert 0 in unique
        assert len(unique) == 3  # 0, 1, 2 (two cells)


def test_convert_already_instance_returns_skipped():
    """Already instance labels (0,1,2,...) should be skipped and return ('skipped', '')."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mask.tif"
        arr = np.zeros((5, 5), dtype=np.uint32)
        arr[1:2, 1:2] = 1
        arr[3:4, 3:4] = 2
        tifffile.imwrite(path, arr)
        out_path = Path(d) / "out.tif"
        status, msg = convert_single_mask_to_omnipose(path, out_path=out_path)
        assert status == "skipped", msg
        out = tifffile.imread(out_path)
        np.testing.assert_array_equal(out, arr)


def test_convert_nonexistent_returns_error():
    """Nonexistent file should return ('error', message)."""
    status, msg = convert_single_mask_to_omnipose(Path("/nonexistent/mask.tif"))
    assert status == "error"
    assert msg


def test_convert_masks_in_folder_empty_dir():
    """Empty folder or no matching masks should return 0 converted, 0 skipped."""
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        converted, skipped, errors, out_dir = convert_masks_in_folder(folder, "_mask")
        assert converted == 0
        assert skipped == 0
        assert out_dir == folder / "omnipose_masks"


def test_convert_masks_in_folder_creates_output_dir():
    """convert_masks_in_folder should create omnipose_masks and write converted masks."""
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        # One binary mask
        arr = np.zeros((5, 5), dtype=np.uint8)
        arr[1:3, 1:3] = 255
        tifffile.imwrite(str(folder / "img_mask.tif"), arr)
        converted, skipped, errors, out_dir = convert_masks_in_folder(folder, "_mask")
        assert out_dir.is_dir()
        assert (out_dir / "img_mask.tif").exists()
        assert converted == 1 or skipped == 1
        assert not errors


if __name__ == "__main__":
    test_convert_binary_mask_returns_converted()
    test_convert_already_instance_returns_skipped()
    test_convert_nonexistent_returns_error()
    test_convert_masks_in_folder_empty_dir()
    test_convert_masks_in_folder_creates_output_dir()
    print("All mask_utils tests passed.")
