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


def test_infer_tracking_workspace_prefers_parent_with_multiple_masks(tmp_path: Path):
    root = tmp_path / "swimming"
    root.mkdir()
    seg_a = root / "T7_001_omnipose_segmentation"
    seg_b = root / "T7_002_omnipose_segmentation"
    seg_a.mkdir()
    seg_b.mkdir()
    mask_a = seg_a / "T7_001_masks.tif"
    mask_b = seg_b / "T7_002_masks.tif"
    mask_a.write_bytes(b"x")
    mask_b.write_bytes(b"x")
    ws = TrackingTab._infer_tracking_workspace(mask_a)
    assert ws.resolve() == root.resolve()
    assert len(TrackingTab._collect_mask_files(ws)) == 2


def test_infer_tracking_workspace_keeps_single_mask_folder(tmp_path: Path):
    seg = tmp_path / "one_sample_segmentation"
    seg.mkdir()
    mask = seg / "sample_masks.tif"
    mask.write_bytes(b"x")
    ws = TrackingTab._infer_tracking_workspace(mask)
    assert ws.resolve() == seg.resolve()
    import numpy as np
    import tifffile

    p = tmp_path / "sample_mask_filtered.tif"
    tifffile.imwrite(p, np.zeros((10, 16, 16), dtype=np.uint16))
    tab = TrackingTab.__new__(TrackingTab)
    assert tab._mask_stack_frame_count(p) == 10
