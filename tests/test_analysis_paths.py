from __future__ import annotations

from pathlib import Path


def test_base_dir_for_mask_from_filtered_masks_subfolder() -> None:
    from bactomate.gui.analysis_paths import ANALYSIS_SUBDIR, analysis_dir_from_base, base_dir_for_mask

    tif_root = Path("/tmp/Tif_files")
    seg = tif_root / "_segmentation_cellpose_sam"
    mask = seg / "filtered_masks" / "sample_mask_filtered.tif"

    assert base_dir_for_mask(mask) == tif_root
    assert analysis_dir_from_base(base_dir_for_mask(mask)) == tif_root / ANALYSIS_SUBDIR


def test_resolve_analysis_workbook_prefers_filtered_stem(tmp_path: Path) -> None:
    from bactomate.gui.analysis_paths import resolve_analysis_workbook_for_mask

    tif_root = tmp_path / "Tif_files"
    seg = tif_root / "_segmentation_cellpose_sam"
    seg.mkdir(parents=True)
    analysis_dir = tif_root / "_analysis"
    analysis_dir.mkdir(parents=True)
    raw_mask = seg / "sample_Phase60X_masks.tif"
    raw_mask.write_bytes(b"")
    filtered_wb = analysis_dir / "sample_Phase60X_mask_filtered_analysis.xlsx"
    filtered_wb.write_bytes(b"PK\x03\x04")

    found = resolve_analysis_workbook_for_mask(raw_mask)
    assert found is not None
    assert found.name == "sample_Phase60X_mask_filtered_analysis.xlsx"
