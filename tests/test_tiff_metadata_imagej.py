"""Tests for ImageJ-style TIFF dimension and calibration inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bactomate.modules.tiff_metadata import (
    _pixel_size_um_from_tiff_page,
    get_tiff_file_info,
    infer_tiff_stack_dims,
    normalize_tiff_array_to_tcyx,
    read_calibration_from_path,
    read_metadata_from_path,
    resolve_calibration_for_mask,
    resolve_original_for_mask,
)

HOOK_TIF = Path("/Users/philipppopp/Desktop/hans/Hook-mal_001.tif")


def test_infer_imagej_three_channel_single_frame():
    """images=3 and channels=3 with 3 pages → 1 timepoint, 3 channels (not T=3)."""
    t, c, z = infer_tiff_stack_dims(
        num_pages=3,
        imagej_metadata={"images": 3, "channels": 3, "mode": "color"},
        series_axes="CYX",
        series_shape=(3, 121, 121),
    )
    assert (t, c, z) == (1, 3, 1)


def test_infer_imagej_hyperstack_images_16_channels_2_slices_8():
    t, c, z = infer_tiff_stack_dims(
        num_pages=16,
        imagej_metadata={"images": 16, "channels": 2, "slices": 8},
    )
    assert (t, c, z) == (1, 2, 8)


def test_infer_explicit_frames_overrides_images():
    t, c, z = infer_tiff_stack_dims(
        num_pages=30,
        imagej_metadata={"images": 30, "channels": 3, "frames": 10},
    )
    assert (t, c, z) == (10, 3, 1)


def test_normalize_cyx_to_tcyx():
    arr = np.zeros((3, 64, 64), dtype=np.uint16)
    out = normalize_tiff_array_to_tcyx(arr, num_time=1, num_channels=3, series_axes="CYX")
    assert out.shape == (1, 3, 64, 64)


def test_normalize_multipage_stack():
    arr = np.zeros((6, 32, 32), dtype=np.uint16)
    out = normalize_tiff_array_to_tcyx(arr, num_time=2, num_channels=3, num_z=1)
    assert out.shape == (2, 3, 32, 32)


@pytest.mark.skipif(not HOOK_TIF.is_file(), reason="local Hook-mal_001.tif fixture not present")
def test_hook_mal_001_file_info_matches_fiji():
    info = get_tiff_file_info(HOOK_TIF)
    assert info is not None
    assert info["num_time"] == 1
    assert info["num_channels"] == 3
    assert info["pixel_size"] == pytest.approx(0.1083, rel=0.02)
    assert info.get("channel_names_source") == "luts"
    assert info["channel_names"] == ["Red", "Green", "Gray"]

    meta = read_metadata_from_path(HOOK_TIF)
    assert meta["frame_count"] == 1
    assert meta["channel_count"] == 3
    assert meta.get("channel_names") == ["Red", "Green", "Gray"]
    ps, _ = read_calibration_from_path(HOOK_TIF)
    assert ps == pytest.approx(0.1083, rel=0.02)


def test_placeholder_resolution_one_one_is_not_one_um():
    """Unset TIFF (1,1) + NONE must not become 1.0 µm/px."""
    class _Page:
        tags = {
            282: type("T", (), {"value": (1, 1)})(),
            296: type("T", (), {"value": 1})(),
        }

    ps = _pixel_size_um_from_tiff_page(_Page(), {})
    assert ps == 0.0


def test_write_minimal_imagej_tiff_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    out = tmp_path / "ij_composite.tif"
    stack = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    tifffile.imwrite(
        out,
        stack,
        imagej=True,
        metadata={
            "images": 3,
            "channels": 3,
            "mode": "color",
            "spacing": 0.5,
            "unit": "micron",
        },
        resolution=(9.2308, 9.2308),
        resolutionunit=1,
    )
    info = get_tiff_file_info(out)
    assert info is not None
    assert info["num_time"] == 1
    assert info["num_channels"] == 3


def test_resolve_calibration_for_filtered_mask_case_insensitive_source(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tif_dir = tmp_path / "Tif_files"
    mask_dir = tif_dir / "_segmentation_cellpose_sam" / "filtered_masks"
    mask_dir.mkdir(parents=True)
    source = tif_dir / "2_Phase60x_phase.tif"
    mask = mask_dir / "2_Phase60X_phase_mask_filtered.tif"
    stack = np.zeros((2, 8, 8), dtype=np.uint16)
    tifffile.imwrite(
        source,
        stack,
        imagej=True,
        metadata={"axes": "TYX", "frames": 2, "images": 2, "finterval": 120.0},
        resolution=(1e4 / 0.108333, 1e4 / 0.108333),
        resolutionunit=3,
    )
    tifffile.imwrite(mask, stack, imagej=True, metadata={"axes": "TYX", "frames": 2, "images": 2})

    assert resolve_original_for_mask(mask) == source
    ps, tp, src, kind = resolve_calibration_for_mask(mask)
    assert src == source
    assert kind == "original"
    assert ps == pytest.approx(0.108333, rel=1e-4)
    assert tp == pytest.approx(2.0)
