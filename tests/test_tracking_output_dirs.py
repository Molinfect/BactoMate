"""Tracking output folder naming (movement vs lineage)."""
from __future__ import annotations

from pathlib import Path

from bactomate.gui.workers.tracking_io import (
    stem_from_tracking_output_folder,
    tracking_output_dir,
    tracking_run_folder_name,
)


def test_tracking_run_folder_name_lineage_vs_movement() -> None:
    assert tracking_run_folder_name("6_18771_Phage_1-1_masks", lineage=False) == (
        "6_18771_Phage_1-1_masks_tracking"
    )
    assert tracking_run_folder_name("6_18771_Phage_1-1_masks", lineage=True) == (
        "6_18771_Phage_1-1_masks_lineage_tracking"
    )


def test_stem_from_tracking_output_folder() -> None:
    assert stem_from_tracking_output_folder("sample_masks_lineage_tracking") == "sample_masks"
    assert stem_from_tracking_output_folder("sample_masks_tracking") == "sample_masks"


def test_tracking_output_dir_under_tracking_root(tmp_path: Path) -> None:
    sample = tmp_path / "Tif_files"
    seg = sample / "_segmentation_omni"
    seg.mkdir(parents=True)
    mask = seg / "6_18771_Phage_1-1_masks.tif"
    mask.write_bytes(b"x")

    movement = tracking_output_dir(mask, track_action="movement")
    lineage = tracking_output_dir(mask, track_action="lineage")
    assert movement == sample / "_tracking" / "6_18771_Phage_1-1_masks_tracking"
    assert lineage == sample / "_tracking" / "6_18771_Phage_1-1_masks_lineage_tracking"
