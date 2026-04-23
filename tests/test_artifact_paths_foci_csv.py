from __future__ import annotations

from pathlib import Path


def test_foci_csv_path_short_canonical_name():
    from bactimate.gui.artifact_paths import foci_csv_path, filtered_foci_csv_path

    pipeline_dir = Path("/tmp/pipeline")
    mask_path = Path("/tmp/pipeline/G183A-004_Phase60x_mask_filtered.tif")
    foci_source = Path("/tmp/G183A-004_foci_GFP.tif")

    assert foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source).name == "G183A-004_foci_GFP.csv"
    assert (
        filtered_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source).name
        == "G183A-004_foci_filtered_GFP.csv"
    )


def test_foci_csv_path_channel_extraction_from_suffix():
    from bactimate.gui.artifact_paths import foci_csv_path

    pipeline_dir = Path("/tmp/pipeline")
    mask_path = Path("/tmp/pipeline/G183A-004_Phase60x_masks.tif")

    assert (
        foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=Path("/tmp/G183A-004_foci_dsRed.tif")).name
        == "G183A-004_foci_dsRed.csv"
    )

