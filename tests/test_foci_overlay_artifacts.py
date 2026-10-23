from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.mark.parametrize(('threshold', 'neighbor_intensity'), [(0.0, 0.01), (0.8999, 89.995)])
def test_overlay_preserves_zero_and_decimal_blob_thresholds(tmp_path, threshold, neighbor_intensity):
    from bactomate.gui.foci_overlay_artifacts import FociOverlayStyle, write_foci_overlay_tif

    foci = np.zeros((9, 9), dtype=np.float32)
    foci[4, 4] = 100.0
    foci[4, 5] = neighbor_intensity
    foci_path = tmp_path / 'sample_foci.tif'
    tifffile.imwrite(foci_path, foci)
    output_path = tmp_path / 'overlay.tif'
    style = FociOverlayStyle()
    assert write_foci_overlay_tif(
        foci_image_path=foci_path,
        output_path=output_path,
        rows=[{'y': 4, 'x': 4, 'intensity': 100.0}],
        style=style,
        blob_settings={'foci_blob_threshold_frac': threshold, 'foci_blob_max_radius': 1},
    )
    overlay = tifffile.imread(output_path)
    assert tuple(overlay[0, 4, 6]) == style.outline_rgb


def test_overlay_paths_from_csv_names() -> None:
    from bactomate.gui.foci_overlay_artifacts import (
        foci_overlay_path_for_csv,
        filtered_overlay_path_for_filtered_csv,
    )

    assert foci_overlay_path_for_csv(Path("/tmp/A_foci.csv")).name == "A_foci_overlay.tif"
    assert foci_overlay_path_for_csv(Path("/tmp/A_foci_GFP.csv")).name == "A_foci_overlay_GFP.tif"
    assert filtered_overlay_path_for_filtered_csv(Path("/tmp/A_foci_filtered.csv")).name == "A_filtered_overlay.tif"
    assert (
        filtered_overlay_path_for_filtered_csv(Path("/tmp/A_foci_filtered_GFP.csv")).name
        == "A_filtered_overlay_GFP.tif"
    )


def test_write_foci_overlay_snapshot(tmp_path: Path) -> None:
    from bactomate.gui.foci_overlay_artifacts import write_foci_overlay_tif

    foci = np.zeros((64, 64), dtype=np.float32)
    foci[10, 11] = 100.0
    foci_path = tmp_path / "A_foci_GFP.tif"
    tifffile.imwrite(foci_path, foci)

    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[5:20, 5:20] = 1
    mask_path = tmp_path / "A_masks.tif"
    tifffile.imwrite(mask_path, mask)

    out_path = tmp_path / "A_overlay.tif"
    rows = [{"y": 10, "x": 11, "intensity": 100.0, "probability": 0.9}]  # no frame => frame 0
    ok = write_foci_overlay_tif(
        foci_image_path=foci_path,
        output_path=out_path,
        rows=rows,
        mask_path=mask_path,
    )
    assert ok
    arr = tifffile.imread(out_path)
    assert arr.shape == (1, 64, 64, 3)
    # Spotiflow peak pixel is red (255,30,30) at (10,11).
    assert tuple(int(x) for x in arr[0, 10, 11, :]) == (255, 30, 30)


def test_write_foci_overlay_time_sequence_frames(tmp_path: Path) -> None:
    from bactomate.gui.foci_overlay_artifacts import write_foci_overlay_tif

    foci = np.zeros((3, 32, 32), dtype=np.float32)
    foci[0, 1, 1] = 10.0
    foci[1, 2, 2] = 20.0
    foci[2, 3, 3] = 30.0
    foci_path = tmp_path / "B_foci.tif"
    # Explicitly store as multi-page grayscale, not RGB.
    tifffile.imwrite(foci_path, foci, photometric="minisblack")

    mask = np.zeros((3, 32, 32), dtype=np.uint16)
    mask[:, 0:10, 0:10] = 1
    mask_path = tmp_path / "B_masks.tif"
    tifffile.imwrite(mask_path, mask, photometric="minisblack")

    out_path = tmp_path / "B_overlay.tif"
    rows = [
        {"frame": 0, "y": 1, "x": 1, "intensity": 10.0, "probability": 0.1},
        {"frame": 2, "y": 3, "x": 3, "intensity": 30.0, "probability": 0.2},
    ]
    ok = write_foci_overlay_tif(
        foci_image_path=foci_path,
        output_path=out_path,
        rows=rows,
        mask_path=mask_path,
    )
    assert ok
    arr = tifffile.imread(out_path)
    assert arr.shape == (3, 32, 32, 3)
    assert tuple(int(x) for x in arr[0, 1, 1, :]) == (255, 30, 30)
    assert tuple(int(x) for x in arr[2, 3, 3, :]) == (255, 30, 30)
    # Frame 1 should not have a marker at (2,2) since no row for frame 1.
    assert tuple(int(x) for x in arr[1, 2, 2, :]) != (255, 30, 30)

