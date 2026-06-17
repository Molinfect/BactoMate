"""Tests for rewriting TIFF metadata with backup preservation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bactomate.modules.tiff_metadata import (
    TiffMetadataOverrides,
    get_tiff_file_info,
    read_calibration_from_path,
    read_metadata_from_path,
    rewrite_tiff_with_metadata,
    validate_tiff_metadata_overrides,
)


def _write_minimal_tiff(path: Path, frames: int = 5, height: int = 12, width: int = 12) -> None:
    tifffile = pytest.importorskip("tifffile")
    stack = np.arange(frames * height * width, dtype=np.uint16).reshape(frames, height, width)
    tifffile.imwrite(
        path,
        stack,
        imagej=True,
        metadata={
            "axes": "TYX",
            "images": frames,
            "frames": frames,
            "slices": 1,
            "channels": 1,
            "mode": "grayscale",
            "loop": False,
        },
        photometric="minisblack",
    )


def test_validate_rejects_wrong_channel_count(tmp_path: Path) -> None:
    tif = tmp_path / "single_ch.tif"
    _write_minimal_tiff(tif)
    ok, err = validate_tiff_metadata_overrides(
        tif,
        TiffMetadataOverrides(channel_names=["A", "B"]),
    )
    assert not ok
    assert "channel count" in err.lower()


def test_rewrite_updates_calibration_and_preserves_data(tmp_path: Path) -> None:
    tif = tmp_path / "series.tif"
    _write_minimal_tiff(tif, frames=4)
    before = np.asarray(pytest.importorskip("tifffile").imread(tif))

    backup = rewrite_tiff_with_metadata(
        tif,
        TiffMetadataOverrides(
            pixel_size_um=0.1083,
            time_per_frame_min=0.011492,
            acquisition_notes="Manual fallback calibration",
        ),
        backup=True,
    )
    assert backup == tif.with_name(tif.name + ".bak")
    assert backup.is_file()

    after = np.asarray(pytest.importorskip("tifffile").imread(tif))
    assert after.shape == before.shape
    assert np.array_equal(after, before)

    ps, tp = read_calibration_from_path(tif)
    assert ps == pytest.approx(0.1083, rel=1e-3)
    assert tp == pytest.approx(0.011492, rel=1e-3)

    meta = read_metadata_from_path(tif)
    assert meta.get("pixel_size_um") == pytest.approx(0.1083, rel=1e-3)
    assert meta.get("time_per_frame_min") == pytest.approx(0.011492, rel=1e-3)


def test_rewrite_updates_channel_names(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    out = tmp_path / "composite.tif"
    stack = np.zeros((3, 10, 10), dtype=np.uint16)
    tifffile.imwrite(
        out,
        stack,
        imagej=True,
        metadata={"images": 3, "channels": 3, "mode": "color"},
        photometric="minisblack",
    )

    rewrite_tiff_with_metadata(
        out,
        TiffMetadataOverrides(
            pixel_size_um=0.2,
            channel_names=["Phase", "GFP", "RFP"],
        ),
        backup=False,
    )

    info = get_tiff_file_info(out)
    assert info is not None
    assert info["channel_names"] == ["Phase", "GFP", "RFP"]
    meta = read_metadata_from_path(out)
    assert meta.get("channel_names") == ["Phase", "GFP", "RFP"]


def test_rewrite_acquisition_fields_in_info(tmp_path: Path) -> None:
    tif = tmp_path / "notes.tif"
    _write_minimal_tiff(tif, frames=2)

    rewrite_tiff_with_metadata(
        tif,
        TiffMetadataOverrides(
            pixel_size_um=0.15,
            objective_name="Plan Apo",
            objective_magnification=60.0,
            exposure_ms=120.5,
            optics="DIC",
            capturing="Andor",
            acquisition_notes="Corrected after partial extraction",
        ),
        backup=False,
    )

    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffFile(tif) as tf:
        ij = getattr(tf, "imagej_metadata", None) or {}
        info_text = str(ij.get("Info") or "")
    assert "Plan Apo" in info_text
    assert "60" in info_text
    assert "DIC" in info_text
    assert "Andor" in info_text
    assert "Corrected after partial extraction" in info_text


def test_derived_tiff_time_edit_preserves_source_pixel_size(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    source = tmp_path / "2_Phase60x.tif"
    derived_dir = tmp_path / "Tif_files"
    derived_dir.mkdir()
    derived = derived_dir / "2_Phase60x_phase.tif"

    data = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    tifffile.imwrite(
        source,
        data,
        imagej=True,
        metadata={
            "axes": "TYX",
            "images": 3,
            "frames": 3,
            "pixelSize": 0.1083,
            "unit": "um",
        },
        resolution=(1e4 / 0.1083, 1e4 / 0.1083),
        resolutionunit=3,
        photometric="minisblack",
    )
    tifffile.imwrite(
        derived,
        data[:, :4, :4],
        imagej=True,
        metadata={
            "axes": "TYX",
            "images": 3,
            "frames": 3,
            "finterval": 120.0,
            "unit": "um",
        },
        photometric="minisblack",
    )

    meta = read_metadata_from_path(derived)
    assert meta.get("pixel_size_um") == pytest.approx(0.1083, rel=1e-3)
    assert meta.get("time_per_frame_min") == pytest.approx(2.0)

    rewrite_tiff_with_metadata(
        derived,
        TiffMetadataOverrides(time_per_frame_min=2.0),
        backup=False,
    )

    ps, tp = read_calibration_from_path(derived)
    assert ps == pytest.approx(0.1083, rel=1e-3)
    assert tp == pytest.approx(2.0)
