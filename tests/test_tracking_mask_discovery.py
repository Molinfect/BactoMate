"""Tracking mask file discovery (no GUI)."""
from pathlib import Path

from bactomate.gui.artifact_paths import is_cell_mask_tiff
from bactomate.gui.tabs.tracking import TrackingTab


def test_is_cell_mask_tiff(tmp_path: Path):
    for name in ("T7_001_masks.tif", "T7_001_mask_filtered.tif"):
        p = tmp_path / name
        p.write_bytes(b"x")
        assert is_cell_mask_tiff(p)
    for name in ("T7_001.tif", "T7_001_trajectories.tif"):
        p = tmp_path / name
        p.write_bytes(b"x")
        assert not is_cell_mask_tiff(p)


def test_collect_mask_files_excludes_original(tmp_path: Path):
    d = tmp_path / "Tif_files"
    d.mkdir()
    (d / "T7_001.tif").write_bytes(b"x")
    (d / "T7_001_masks.tif").write_bytes(b"x")
    found = TrackingTab._collect_mask_files(d)
    assert len(found) == 1
    assert found[0].name == "T7_001_masks.tif"


def test_collect_mask_files_segmentation_subfolder(tmp_path: Path):
    parent = tmp_path / "sample"
    parent.mkdir()
    seg = parent / "_segmentation_bact_phase_omni"
    seg.mkdir()
    (seg / "T7_001_masks.tif").write_bytes(b"x")
    (parent / "T7_001.tif").write_bytes(b"x")
    found = TrackingTab._collect_mask_files(parent)
    assert any(p.name == "T7_001_masks.tif" for p in found)
    assert not any(p.name == "T7_001.tif" for p in found)


def test_mask_stack_frame_count_single_page_3d_tiff(tmp_path: Path):
    import numpy as np
    import tifffile

    p = tmp_path / "sample_mask_filtered.tif"
    tifffile.imwrite(p, np.zeros((10, 16, 16), dtype=np.uint16))
    tab = TrackingTab.__new__(TrackingTab)
    assert tab._mask_stack_frame_count(p) == 10
