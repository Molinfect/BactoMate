from __future__ import annotations

from pathlib import Path


def test_foci_csv_path_under_foci_detection_folder():
    from bactomate.gui.artifact_paths import (
        FOCI_DETECTION_DIRNAME,
        FOCI_FILTERED_DIRNAME,
        filtered_foci_csv_path,
        foci_csv_path,
    )

    pipeline_dir = Path("/tmp/pipeline")
    mask_path = Path("/tmp/pipeline/G183A-004_Phase60x_mask_filtered.tif")
    foci_source = Path("/tmp/G183A-004_foci_GFP.tif")

    det = foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source)
    assert det.name == "G183A-004_foci_GFP.csv"
    assert det.parent.name == FOCI_DETECTION_DIRNAME
    assert det.parent.parent == pipeline_dir

    filt = filtered_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source)
    assert filt.name == "G183A-004_foci_filtered_GFP.csv"
    assert filt.parent.name == FOCI_FILTERED_DIRNAME
    assert filt.parent.parent == pipeline_dir


def test_foci_csv_path_channel_extraction_from_suffix():
    from bactomate.gui.artifact_paths import FOCI_DETECTION_DIRNAME, foci_csv_path

    pipeline_dir = Path("/tmp/pipeline")
    mask_path = Path("/tmp/pipeline/G183A-004_Phase60x_masks.tif")

    p = foci_csv_path(
        pipeline_dir, mask_path=mask_path, foci_source_path=Path("/tmp/G183A-004_foci_dsRed.tif")
    )
    assert p.name == "G183A-004_foci_dsRed.csv"
    assert p.parent.name == FOCI_DETECTION_DIRNAME


def test_filtered_mask_path_under_filtered_masks_folder():
    from bactomate.gui.artifact_paths import (
        FILTERED_MASKS_DIRNAME,
        filtered_mask_path_for_mask,
        resolve_filtered_mask_path,
    )

    pipeline_dir = Path("/tmp/pipeline")
    raw = Path("/tmp/pipeline/sample_Phase_masks.tif")
    p = filtered_mask_path_for_mask(raw)
    assert p.name == "sample_Phase_mask_filtered.tif"
    assert p.parent.name == FILTERED_MASKS_DIRNAME
    assert p.parent.parent == pipeline_dir


def test_infer_pipeline_dir_from_filtered_masks_subfolder(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import FILTERED_MASKS_DIRNAME, infer_pipeline_dir_from_mask

    seg = tmp_path / "_segmentation_cellpose_sam"
    filt_dir = seg / FILTERED_MASKS_DIRNAME
    filt_dir.mkdir(parents=True)
    mask = filt_dir / "sample_mask_filtered.tif"
    mask.write_bytes(b"")
    assert infer_pipeline_dir_from_mask(mask) == seg


def test_discover_foci_source_tiffs_finds_file_conversion_layout(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import discover_foci_source_tiffs

    tif_dir = tmp_path / "Tif_files"
    tif_dir.mkdir()
    foci = tif_dir / "sample_foci_GFP.tif"
    foci.write_bytes(b"")
    found = discover_foci_source_tiffs("sample_Phase60x", tif_dir)
    assert any(p.name == "sample_foci_GFP.tif" for p in found)


def test_resolve_filtered_mask_path_prefers_subfolder(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import (
        filtered_mask_path_for_mask,
        resolve_filtered_mask_path,
    )

    pipeline_dir = tmp_path / "_segmentation_cellpose_sam"
    pipeline_dir.mkdir()
    raw = pipeline_dir / "sample_masks.tif"

    canonical = filtered_mask_path_for_mask(raw)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"")

    legacy = pipeline_dir / "sample_mask_filtered.tif"
    legacy.write_bytes(b"")

    assert resolve_filtered_mask_path(raw) == canonical


def test_resolve_segmentation_foci_csv_prefers_foci_detection(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import (
        foci_csv_path,
        resolve_segmentation_foci_csv,
    )

    pipeline_dir = tmp_path / "_segmentation_cellpose_sam"
    pipeline_dir.mkdir()
    mask_path = pipeline_dir / "sample_Phase_masks.tif"
    foci_source = tmp_path / "sample_foci_GFP.tif"

    canonical = foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("frame,y,x,intensity,probability\n0,1,2,3.0,0.9\n", encoding="utf-8")

    # Legacy flat file should not win over canonical subfolder file.
    legacy_flat = pipeline_dir / "sample_foci_GFP.csv"
    legacy_flat.write_text("frame,y,x,intensity,probability\n0,9,9,1.0,0.1\n", encoding="utf-8")

    resolved = resolve_segmentation_foci_csv(
        pipeline_dir, mask_path=mask_path, foci_source_path=foci_source
    )
    assert resolved == canonical
