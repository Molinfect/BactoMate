"""Filtering auto-path must match where Omnipose actually writes masks."""

from __future__ import annotations

from pathlib import Path

from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab
from bactomate.gui.tabs.segmentation import SegmentationTab


def test_resolve_output_dir_uses_tif_parent_folder_name(tmp_path: Path) -> None:
    tif_dir = tmp_path / "Tif_files"
    tif_dir.mkdir()
    tiff = tif_dir / "sample_phase.tif"
    tiff.write_bytes(b"")
    out = OmniposeSegTab.resolve_segmentation_output_dir_for_tiff(
        tiff, "_segmentation_bact_phase_omni"
    )
    assert out == tif_dir / "_segmentation_bact_phase_omni"


def test_resolve_output_dir_per_file_when_not_under_tif(tmp_path: Path) -> None:
    tiff = tmp_path / "sample_phase.tif"
    tiff.write_bytes(b"")
    out = OmniposeSegTab.resolve_segmentation_output_dir_for_tiff(
        tiff, "_segmentation_bact_phase_omni"
    )
    assert out == tmp_path / "sample_phase_omnipose_segmentation"


def test_find_existing_output_under_parent_experiment_folder(tmp_path: Path) -> None:
    """Picking MOI10/ must discover MOI10/Tif_files/_segmentation_* (not MOI10/_segmentation_*)."""
    tif_dir = tmp_path / "Tif_files"
    seg = tif_dir / "_segmentation_bact_phase_omni"
    seg.mkdir(parents=True)
    (seg / "sample_phase_masks.tif").write_bytes(b"")
    phantom = tmp_path / "_segmentation_bact_phase_omni"
    assert not phantom.exists()
    found = SegmentationTab._find_existing_segmentation_output(
        tmp_path, "_segmentation_bact_phase_omni"
    )
    assert found == seg
