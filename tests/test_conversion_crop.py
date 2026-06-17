"""Tests for conversion crop ROI helpers."""
from __future__ import annotations

import numpy as np
import pytest

from bactomate.modules.conversion_crop import (
    apply_yx_crop,
    format_roi_status,
    normalize_roi,
    parse_crop_cli_args,
    roi_display_to_native,
    roi_native_to_display,
)


def test_normalize_roi_clamps_and_rejects_tiny() -> None:
    assert normalize_roi(10, 20, 100, 80, 200, 150) == (10, 20, 100, 80)
    assert normalize_roi(100, 80, 10, 20, 200, 150) == (10, 20, 100, 80)
    assert normalize_roi(0, 0, 8, 8, 200, 150) is None


def test_apply_yx_crop_tyx_tcyx_tczxyx() -> None:
    tyx = np.arange(2 * 64 * 64, dtype=np.uint16).reshape(2, 64, 64)
    roi = (10, 12, 40, 36)
    out = apply_yx_crop(tyx, roi)
    assert out.shape == (2, 24, 30)
    assert out[0, 0, 0] == tyx[0, 12, 10]

    tcyx = np.arange(2 * 3 * 64 * 64, dtype=np.uint16).reshape(2, 3, 64, 64)
    out2 = apply_yx_crop(tcyx, roi)
    assert out2.shape == (2, 3, 24, 30)

    tczyx = np.arange(2 * 2 * 2 * 64 * 64, dtype=np.uint16).reshape(2, 2, 2, 64, 64)
    out3 = apply_yx_crop(tczyx, roi)
    assert out3.shape == (2, 2, 2, 24, 30)


def test_roi_display_native_roundtrip() -> None:
    native = (100, 200)
    display = (50, 100)
    native_roi = (20, 10, 120, 60)
    disp = roi_native_to_display(native_roi, native, display)
    assert disp is not None
    back = roi_display_to_native(disp, native, display)
    assert back is not None
    assert back[0] == pytest.approx(20, abs=1)
    assert back[1] == pytest.approx(10, abs=1)


def test_format_roi_status() -> None:
    assert format_roi_status(None) == "Crop: full image"
    assert "w=50" in format_roi_status((10, 20, 60, 70))


def test_parse_crop_cli_args() -> None:
    assert parse_crop_cli_args([1, 2, 3, 4]) == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        parse_crop_cli_args([1, 2, 3])
