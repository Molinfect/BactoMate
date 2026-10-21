from __future__ import annotations

from pathlib import Path


def test_foci_tiff_stem_matches_ssb_file_conversion_name():
    from bactomate.gui.artifact_paths import foci_tiff_stem_matches_prefix

    assert foci_tiff_stem_matches_prefix("ssb-mNG_cl1__fi_foci_GFP", "ssb-mNG_cl1")
    assert not foci_tiff_stem_matches_prefix("other__fi_foci_GFP", "ssb-mNG_cl1")


def test_discover_foci_source_tiffs_finds_double_underscore_fi_foci(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import discover_foci_source_tiffs

    tif_dir = tmp_path / "Tif_files"
    tif_dir.mkdir()
    foci = tif_dir / "ssb-mNG_cl1__fi_foci_GFP.tif"
    foci.write_bytes(b"")
    found = discover_foci_source_tiffs(
        "ssb-mNG_cl1__Phase60X_mask_filtered",
        tif_dir,
    )
    assert any(p.name == "ssb-mNG_cl1__fi_foci_GFP.tif" for p in found)


def test_extracts_channel_from_double_underscore_foci_name():
    from bactomate.gui.artifact_paths import extract_channel_tag_for_mask_stem

    tag = extract_channel_tag_for_mask_stem(
        "EM20142_NoPhage__foci_GFP",
        "EM20142_NoPhage__mCherry",
    )
    assert tag == "GFP"


def test_mcherry_mask_pairs_with_matching_numbered_foci_only(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import foci_csv_path, resolve_foci_paths_for_mask

    tif_dir = tmp_path / "Tif_files"
    seg = tif_dir / "_segmentation_cellpose_sam"
    tif_dir.mkdir()
    seg.mkdir()
    mask = seg / "EM20140_NoPhage_001_mCherry_masks.tif"
    right = tif_dir / "EM20140_NoPhage_001_foci_GFP.tif"
    wrong = tif_dir / "EM20140_NoPhage_002_foci_GFP.tif"
    broad = tif_dir / "EM20140_Bas08_MOI1_001_foci_GFP.tif"
    for p in (mask, right, wrong, broad):
        p.write_bytes(b"x")

    kept = resolve_foci_paths_for_mask(mask)
    assert kept == [right]
    assert foci_csv_path(seg, mask_path=mask, foci_source_path=right).name == (
        "EM20140_NoPhage_001_foci_GFP.csv"
    )


def test_double_underscore_mcherry_mask_pairs_with_double_underscore_foci(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import foci_csv_path, resolve_foci_paths_for_mask

    tif_dir = tmp_path / "Tif_files"
    seg = tif_dir / "_segmentation_cellpose_sam"
    tif_dir.mkdir()
    seg.mkdir()
    mask = seg / "EM20140_NoPhage__mCherry_masks.tif"
    right = tif_dir / "EM20140_NoPhage__foci_GFP.tif"
    wrong = tif_dir / "EM20140_NoPhage_001_foci_GFP.tif"
    for p in (mask, right, wrong):
        p.write_bytes(b"x")

    kept = resolve_foci_paths_for_mask(mask)
    assert kept == [right]
    assert foci_csv_path(seg, mask_path=mask, foci_source_path=right).name == (
        "EM20140_NoPhage_foci_GFP.csv"
    )
