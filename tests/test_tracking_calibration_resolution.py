"""Calibration resolution: prefer original phase TIFF over mask metadata."""

from pathlib import Path

from bactomate.modules.tiff_metadata import (
    _search_dirs_for_original,
    resolve_calibration_for_mask,
    resolve_original_for_mask,
)


def test_search_dirs_includes_sample_base_parent(tmp_path: Path):
    swimming = tmp_path / "swimming"
    tif_files = swimming / "Tif_files"
    seg = tif_files / "_segmentation_bact_phase_omni"
    seg.mkdir(parents=True)
    mask = seg / "T7_001_masks.tif"
    mask.touch()
    dirs = _search_dirs_for_original(mask)
    assert swimming in dirs
    assert tif_files in dirs


def test_resolve_original_finds_phase_tiff_at_sample_root(tmp_path: Path):
    swimming = tmp_path / "swimming"
    tif_files = swimming / "Tif_files"
    seg = tif_files / "_segmentation_bact_phase_omni"
    seg.mkdir(parents=True)
    original = swimming / "T7_001.tif"
    original.write_bytes(b"x")
    mask = seg / "T7_001_masks.tif"
    mask.write_bytes(b"x")
    found = resolve_original_for_mask(mask)
    assert found == original


def test_calibration_prefers_original_over_mask_metadata(tmp_path: Path, monkeypatch):
    swimming = tmp_path / "swimming"
    seg = swimming / "Tif_files" / "_segmentation_bact_phase_omni"
    seg.mkdir(parents=True)
    original = swimming / "T7_001.tif"
    original.write_bytes(b"x")
    mask = seg / "T7_001_masks.tif"
    mask.write_bytes(b"x")

    def fake_read(path: Path):
        if path == original:
            return (0.325, 0.05)
        if path == mask:
            return (0.108, 0.0)
        return (0.0, 0.0)

    import bactomate.modules.tiff_metadata as tm

    monkeypatch.setattr(tm, "read_calibration_from_path", fake_read)
    ps, tp, src, kind = resolve_calibration_for_mask(mask)
    assert ps == 0.325
    assert kind == "original"
    assert src == original
