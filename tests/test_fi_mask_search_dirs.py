"""Production companion discovery, including historical conversion layouts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from bactomate.gui.analysis_paths import artifact_search_dirs_for_mask, base_dir_for_mask
from bactomate.gui.artifact_paths import (
    discover_foci_companions,
    discover_foci_source_tiffs,
    extract_channel_tag_for_mask_stem,
    resolve_fi_inputs_for_mask,
    resolve_fi_paths_for_mask,
    resolve_foci_paths_for_mask,
)
from bactomate.gui.filter_preview_utils import collect_background_paths_for_mask


@pytest.mark.parametrize("mask_folder", ["", "_segmentation_cellpose_sam", "_segmentation_cellpose_sam/filtered_masks"])
@pytest.mark.parametrize(("prefix", "phase", "other"), [
    ("sample_001_", "Phase60X", "sample_002_"),
    ("ssb-mNG_cl1__", "Phase60X", "ssb-mNG_cl1_001_"),
    ("EM19893_t=120_+Ara_001_", "DIA", "EM19893_t=120_+Ara_002_"),
    ("sample_f1_", "phase", "sample_f2_"),
    ("sample[1]_001_", "phase", "sample[1]_002_"),
    ("drug_analysis_001_", "phase", "drug_analysis_002_"),
    ("drug_maskspecial_001_", "phase", "drug_maskspecial_002_"),
])
def test_production_discovery_preserves_acquisition_and_search_locations(
    tmp_path: Path, mask_folder: str, prefix: str, phase: str, other: str,
) -> None:
    sample = tmp_path / "Tif_files"
    folder = sample / mask_folder
    folder.mkdir(parents=True)
    mask_suffix = "_mask_filtered" if folder.name == "filtered_masks" else "_masks"
    mask = folder / f"{prefix}{phase}{mask_suffix}.tif"
    phase_path = sample / f"{prefix}{phase}.tif"
    explicit = sample / f"{prefix}fi_GFP.tif"
    same_channel = sample / f"{prefix}foci_GFP.tif"
    foci_only = sample / f"{prefix}foci_dsRed.tiff"
    wrong = sample / f"{other}fi_foci_GFP.tif"
    qc = sample / f"{prefix}fi_foci_overlay_GFP.tif"
    for p in (mask, phase_path, explicit, same_channel, foci_only, wrong, qc):
        p.write_bytes(b"historical fixture")
    before = {p: p.read_bytes() for p in sample.rglob("*") if p.is_file()}

    assert sample in artifact_search_dirs_for_mask(mask)
    if mask_folder:
        assert base_dir_for_mask(mask) == sample
    from bactomate.gui.tabs._analyze_masks_base import AnalyzeMasksBase
    from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab

    analyze = AnalyzeMasksBase.__new__(AnalyzeMasksBase)
    segment = OmniposeSegTab.__new__(OmniposeSegTab)
    assert resolve_foci_paths_for_mask(mask) == [foci_only, same_channel]
    assert analyze._resolve_foci_paths_for_mask(mask) == [foci_only, same_channel]
    assert segment._find_foci_sources_for_input(phase_path) == [foci_only, same_channel]
    assert analyze._resolve_fi_inputs_for_mask(mask) == [explicit, foci_only]
    assert resolve_fi_inputs_for_mask(mask) == [explicit, foci_only]
    assert set(collect_background_paths_for_mask(mask)) == {
        phase_path, explicit, same_channel, foci_only,
    }
    assert {p: p.read_bytes() for p in before} == before


def test_fi_foci_conversion_name_is_available_for_explicit_fi_discovery(tmp_path: Path) -> None:
    mask = tmp_path / "ssb-mNG_cl1__Phase60X_masks.tif"
    fi = tmp_path / "ssb-mNG_cl1__fi_foci_GFP.tif"
    wrong = tmp_path / "ssb-mNG_cl1_001_fi_foci_GFP.tif"
    for p in (mask, fi, wrong):
        p.touch()
    assert resolve_fi_paths_for_mask(mask) == [fi]
    assert resolve_fi_inputs_for_mask(mask) == [fi]


@pytest.mark.parametrize(("fi_name", "foci_name"), [
    ("sample_001_fi_GFP", "sample_001_foci_GFP"),
    ("sample_001_fi_GFP", "sample_001_fi_foci_GFP"),
    ("sample_001_fi_GFP", "sample_001_GFP_fi_foci"),
    ("sample_001__fi_GFP", "sample_001__fi_foci_GFP"),
    ("sample_001_FI_GFP", "sample_001_FOCI_GFP"),
])
def test_explicit_fi_covers_the_same_foci_channel(tmp_path: Path, fi_name: str, foci_name: str) -> None:
    mask = tmp_path / "sample_001_Phase60X_masks.tif"
    explicit = tmp_path / f"{fi_name}.tif"
    duplicate = tmp_path / f"{foci_name}.tif"
    foci_only = tmp_path / "sample_001_foci_dsRed.tif"
    for p in (mask, explicit, duplicate, foci_only):
        p.touch()
    assert resolve_fi_inputs_for_mask(mask) == [explicit, foci_only]


def test_historical_channel_labels_are_retained() -> None:
    assert extract_channel_tag_for_mask_stem("sample_001_GFP_fi_foci", "sample_001_DIA") == "GFP_fi"
    assert extract_channel_tag_for_mask_stem("sample__fi_foci_GFP", "sample__Phase60X") == "GFP"



def test_segmentation_companions_exclude_input_and_alias(tmp_path: Path) -> None:
    input_path = tmp_path / "sample_fi_foci_GFP.tif"
    input_path.touch()
    alias = tmp_path / "sample_foci_GFP.tif"
    alias.hardlink_to(input_path)
    companion = tmp_path / "sample_foci_dsRed.tif"
    companion.touch()
    assert discover_foci_companions(input_path) == [companion]


@pytest.mark.parametrize("input_stem", ["sample_001_DIA", "sample_f1_DIA"])
def test_legacy_fallback_cannot_drop_replicate_or_frame(tmp_path: Path, input_stem: str) -> None:
    own = tmp_path / f"{input_stem.rsplit('_', 1)[0]}_foci_GFP.tif"
    shorter = tmp_path / "sample_foci_GFP.tif"
    for p in (own, shorter):
        p.touch()
    assert discover_foci_source_tiffs(input_stem, tmp_path) == [own]


def test_discovery_does_not_rewrite_historical_tiffs_or_sidecars(tmp_path: Path) -> None:
    mask = tmp_path / "sample_001_DIA_mask_filtered.tif"
    foci = tmp_path / "sample_001_GFP_fi_foci.TIFF"
    image = np.arange(6, dtype=np.uint16).reshape(2, 3)
    for p in (mask, foci):
        tifffile.imwrite(p, image, metadata={"axes": "YX", "legacy": True})
    csv = tmp_path / "sample_001_foci_GFP.csv"
    csv.write_text("frame,cell_id,foci_id,y,x\n0,17,1,1,2\n")
    before = {p: p.read_bytes() for p in (mask, foci, csv)}
    assert resolve_foci_paths_for_mask(mask) == [foci]
    assert resolve_fi_inputs_for_mask(mask) == [foci]
    assert {p: p.read_bytes() for p in before} == before
    np.testing.assert_array_equal(tifffile.imread(foci), image)


@pytest.mark.parametrize("location", ["", "_segmentation"])
def test_preview_preserves_base_phase_and_legacy_child_search(tmp_path: Path, location: str) -> None:
    mask = tmp_path / "sample_Phase60X_masks.tif"
    mask.touch()
    folder = tmp_path / location
    folder.mkdir(exist_ok=True)
    base_phase = folder / "sample.tif"
    base_phase.touch()
    assert collect_background_paths_for_mask(mask) == [base_phase]


def test_pairing_does_not_confuse_samples_with_equal_replicate_tokens(tmp_path: Path) -> None:
    mask = tmp_path / "sampleA_cl1_001_Phase60X_masks.tif"
    right = tmp_path / "sampleA_cl1_001_fi_foci_GFP.tif"
    wrong = tmp_path / "sampleB_cl1_001_fi_foci_GFP.tif"
    for p in (mask, right, wrong):
        p.touch()
    assert resolve_fi_inputs_for_mask(mask) == [right]
    assert collect_background_paths_for_mask(mask) == [right]


def test_foci_role_alias_survives_scanning_but_excludes_segmentation_input(tmp_path: Path) -> None:
    phase = tmp_path / "sample_DIA.tif"
    phase.touch()
    foci = tmp_path / "sample_foci_GFP.tif"
    foci.hardlink_to(phase)
    assert discover_foci_source_tiffs(phase.stem, tmp_path, tmp_path) == [foci]
    assert discover_foci_companions(phase) == []


@pytest.mark.parametrize("acquisition", [
    "drug_analysis_001",
    "drug_maskspecial_001",
    "drug_mask_filtered_001",
    "drug_overlay_001",
    "drug_analysis_visualization_001",
    "drug_filter_visualization_001",
    "drug_trajectories_001",
])
def test_artifact_words_in_acquisitions_preserve_inputs_and_exclude_qc(
    tmp_path: Path, acquisition: str,
) -> None:
    from bactomate.gui.fi_contour_filter_preview import discover_background_paths_for_mask
    from bactomate.gui.foci_overlay_artifacts import foci_overlay_path_for_csv
    from bactomate.gui.tabs._analyze_masks_analyze import AnalyzeTab
    from bactomate.gui.tabs._analyze_masks_filtering import FilteringTab
    from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab

    sample = tmp_path / "Tif_files"
    mask_dir = sample / "_segmentation_cellpose_sam" / "filtered_masks"
    mask_dir.mkdir(parents=True)
    mask = mask_dir / f"{acquisition}_phase_mask_filtered.tif"
    phase = sample / f"{acquisition}_phase.tif"
    foci = sample / f"{acquisition}_foci_GFP.tif"
    foci_only = sample / f"{acquisition}_foci_dsRed.tif"
    qc_paths = [
        sample / f"{foci.stem}{suffix}.tif" for suffix in (
            "_masks", "_mask_filtered", "_masks_tracked", "_overlay", "_overlay_tracked",
            "_analysis", "_analysis_visualization", "_filter_visualization", "_trajectories",
        )
    ] + [
        foci_overlay_path_for_csv(foci.with_suffix(".csv")),
        sample / f"{acquisition}_foci_overlay.tif",
        sample / f"{acquisition}_fi_foci_overlay_GFP.tif",
    ]
    for p in [mask, phase, foci, foci_only] + qc_paths:
        p.write_bytes(b"historical fixture")
    before = {p: p.read_bytes() for p in sample.rglob("*") if p.is_file()}
    foci_expected = [foci_only, foci]

    segment = OmniposeSegTab.__new__(OmniposeSegTab)
    analyze = AnalyzeTab.__new__(AnalyzeTab)
    filtering = FilteringTab.__new__(FilteringTab)
    assert segment._find_foci_sources_for_input(phase) == foci_expected
    assert analyze._resolve_foci_paths_for_mask(mask) == foci_expected
    assert analyze._resolve_fi_inputs_for_mask(mask) == foci_expected
    assert filtering._resolve_foci_paths_for_mask(mask) == foci_expected
    assert set(collect_background_paths_for_mask(mask)) == {phase, foci, foci_only}
    assert set(discover_background_paths_for_mask(filtering, mask)) == {phase, foci, foci_only}
    assert {p: p.read_bytes() for p in before} == before
