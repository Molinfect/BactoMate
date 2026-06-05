"""
Tests for image stack/plane coercion helpers used by FI/foci analysis.
"""
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bactomate.modules.image_stack_utils import coerce_plane_to_2d, coerce_stack_to_tyx


def test_coerce_plane_to_2d_yxc_uses_first_channel():
    arr = np.zeros((12, 15, 3), dtype=np.uint16)
    arr[:, :, 1] = 7
    out = coerce_plane_to_2d(arr)
    assert out.shape == (12, 15)
    np.testing.assert_array_equal(out, arr[:, :, 0])


def test_coerce_plane_to_2d_tyx_uses_first_frame():
    arr = np.arange(4 * 20 * 30, dtype=np.uint16).reshape(4, 20, 30)
    out = coerce_plane_to_2d(arr)
    assert out.shape == (20, 30)
    np.testing.assert_array_equal(out, arr[0])


def test_coerce_stack_to_tyx_from_2d():
    arr = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    out = coerce_stack_to_tyx(arr)
    assert out is not None
    assert out.shape == (1, 6, 8)
    np.testing.assert_array_equal(out[0], arr)


def test_coerce_stack_to_tyx_from_yxc():
    arr = np.zeros((10, 14, 2), dtype=np.uint16)
    arr[:, :, 0] = 3
    arr[:, :, 1] = 9
    out = coerce_stack_to_tyx(arr)
    assert out is not None
    assert out.shape == (1, 10, 14)
    np.testing.assert_array_equal(out[0], arr[:, :, 0])


def test_coerce_stack_to_tyx_from_tcyx():
    arr = np.arange(5 * 2 * 20 * 30, dtype=np.uint16).reshape(5, 2, 20, 30)
    out = coerce_stack_to_tyx(arr)
    assert out is not None
    assert out.shape == (5, 20, 30)
    np.testing.assert_array_equal(out, arr[:, 0, :, :])


def test_coerce_stack_to_tyx_returns_none_for_unsupported_ndim():
    arr = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    out = coerce_stack_to_tyx(arr)
    assert out is None
