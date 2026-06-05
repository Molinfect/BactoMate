from __future__ import annotations

from pathlib import Path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_masks_in_segmentation_subfolder(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import discover_mask_tiffs_for_workspace

    seg = tmp_path / "_segmentation_cellpose_sam"
    seg.mkdir()
    _touch(seg / "T7_001_masks.tif")
    _touch(seg / "T7_001-1_masks.tif")

    masks = discover_mask_tiffs_for_workspace(tmp_path)
    assert {p.name for p in masks} == {"T7_001_masks.tif", "T7_001-1_masks.tif"}


def test_discover_masks_from_phase_path_lists_all(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import discover_mask_tiffs_for_workspace

    seg = tmp_path / "_segmentation_cellpose_sam"
    seg.mkdir()
    _touch(seg / "T7_001_masks.tif")
    _touch(seg / "T7_001-1_masks.tif")
    phase = tmp_path / "T7_001-1.tif"
    _touch(phase)

    masks = discover_mask_tiffs_for_workspace(phase)
    assert {p.name for p in masks} == {"T7_001_masks.tif", "T7_001-1_masks.tif"}


def test_discover_masks_directly_in_segmentation_folder(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import discover_mask_tiffs_for_workspace

    seg = tmp_path / "_segmentation_cellpose_sam"
    seg.mkdir()
    _touch(seg / "T7_001_masks.tif")
    _touch(seg / "T7_001-1_masks.tif")

    masks = discover_mask_tiffs_for_workspace(seg)
    assert {p.name for p in masks} == {"T7_001_masks.tif", "T7_001-1_masks.tif"}
