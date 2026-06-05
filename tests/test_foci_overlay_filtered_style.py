"""Tests for distinct filtered vs segmentation foci overlay artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def test_filtered_overlay_uses_orange_blob_style(tmp_path: Path) -> None:
    from bactomate.gui.foci_overlay_artifacts import (
        FociOverlayStyle,
        write_filtered_foci_overlay_for_csv,
        write_foci_overlay_tif,
    )

    foci = np.zeros((32, 32), dtype=np.float32)
    foci[10, 12] = 200.0
    foci[10, 13] = 150.0
    foci_path = tmp_path / "sample_foci.tif"
    tifffile.imwrite(foci_path, foci)

    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[8:20, 8:20] = 1
    mask_path = tmp_path / "sample_mask_filtered.tif"
    tifffile.imwrite(mask_path, mask)

    csv_path = tmp_path / "sample_foci_filtered.csv"
    csv_path.write_text("y,x,intensity,probability,area_px\n10,12,200,0.9,4\n", encoding="utf-8")

    out_path, ok = write_filtered_foci_overlay_for_csv(
        foci_image_path=foci_path,
        filtered_csv_path=csv_path,
        rows=[{"frame": 0, "y": 10, "x": 12, "intensity": 200.0, "area_px": 4}],
        mask_path=mask_path,
        filter_settings={"foci_blob_threshold_frac": 0.8, "foci_blob_max_radius": 5},
    )
    assert ok
    assert out_path.name.endswith("_filtered_overlay.tif")
    arr = tifffile.imread(out_path)
    # Orange blob region (not pure grayscale background).
    assert int(arr[0, 10, 12, 0]) > 50
    assert int(arr[0, 10, 12, 1]) > 50
    assert int(arr[0, 10, 12, 2]) < 200

    seg_path = tmp_path / "sample_foci_overlay.tif"
    ok_seg = write_foci_overlay_tif(
        foci_image_path=foci_path,
        output_path=seg_path,
        rows=[{"frame": 0, "y": 10, "x": 12, "intensity": 200.0}],
        mask_path=mask_path,
        style=FociOverlayStyle.segmentation(),
    )
    assert ok_seg
    seg = tifffile.imread(seg_path)
    # Segmentation uses yellow dot at peak.
    assert tuple(int(x) for x in seg[0, 10, 12, :]) == (255, 255, 0)
