"""
Contract tests for the Analyze Masks stage split.

These tests intentionally focus on **stage separation** (widget surface + mode behavior),
not on running the full image analysis pipeline (which is integration-tested elsewhere).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QWidget
from unittest.mock import patch

from bactomate.gui.tabs.analyze_masks import AnalyzeMasksTab, AnalyzeTab, FilteringTab


@pytest.fixture
def qt_parent(qapp):
    """Top-level parent widget (required on macOS offscreen to avoid abort)."""
    parent = QWidget()
    yield parent
    parent.deleteLater()


def test_filtering_tab_exposes_only_filtering_controls(qt_parent):
    tab = FilteringTab(parent=qt_parent)
    assert tab._stage_key() == "filtering"
    assert not hasattr(tab, "analyze_fi")
    assert tab.preview_action_btn.text() == "Preview filters"
    assert "Filter" in tab.primary_action_btn.text()
    # Peak-filter widgets live in the preview surface, not the main tab shell.
    assert not hasattr(tab, "filter_prob_enabled")


def test_analyze_tab_exposes_only_analyze_controls_and_hides_preview(qt_parent):
    tab = AnalyzeTab(parent=qt_parent)
    assert tab._stage_key() == "analyze"
    assert hasattr(tab, "analyze_fi")
    assert hasattr(tab, "use_filtered_masks_cb")
    assert not hasattr(tab, "filter_prob_enabled")
    assert tab.preview_action_btn.isVisible() is False
    assert "Analyze" in tab.primary_action_btn.text()


def test_factory_returns_expected_tabs_and_rejects_full_mode(qt_parent):
    t1 = AnalyzeMasksTab(parent=qt_parent, mode="filtering")
    assert isinstance(t1, FilteringTab)
    t2 = AnalyzeMasksTab(parent=qt_parent, mode="analyze")
    assert isinstance(t2, AnalyzeTab)
    with pytest.raises(ValueError):
        AnalyzeMasksTab(parent=qt_parent, mode="full")


def test_update_fi_feedback_marks_filtered_mask_and_counts_only_real_fi_tasks(qt_parent):
    tab = AnalyzeTab(parent=qt_parent)
    tab.analyze_fi.setChecked(True)
    tab.fi_mode_match_by_name.setChecked(True)
    raw_mask = Path("/tmp/sample_fi_mCherry_masks.tif")
    filtered_mask = Path("/tmp/filtered_masks/sample_fi_mCherry_mask_filtered.tif")
    fi_path = Path("/tmp/sample_fi_mCherry.tif")
    foci_path = Path("/tmp/sample_fi_foci_GFP.tif")
    tab._get_mask_files_for_feedback = lambda: [raw_mask]
    tab._resolve_effective_mask_for_analysis = lambda _mask: filtered_mask
    tab._resolve_fi_inputs_for_mask = lambda _mask: [fi_path, foci_path]

    tab._update_fi_feedback()

    text = tab.fi_feedback.toPlainText()
    assert "1 mask file(s) -> 2 FI analyses".replace("->", "→") in text
    assert "sample_fi_mCherry_mask_filtered.tif (filtered; from sample_fi_mCherry_masks.tif)" in text
    assert "sample_fi_mCherry_masks.tif, " not in text
    assert "raw" not in text.lower()


def test_update_fi_feedback_never_says_raw_when_unfiltered(qt_parent):
    tab = AnalyzeTab(parent=qt_parent)
    tab.analyze_fi.setChecked(True)
    tab.fi_mode_match_by_name.setChecked(True)
    raw_mask = Path("/tmp/sample_fi_mCherry_masks.tif")
    fi_path = Path("/tmp/sample_fi_mCherry.tif")
    tab._get_mask_files_for_feedback = lambda: [raw_mask]
    tab._resolve_effective_mask_for_analysis = lambda mask: mask
    tab._resolve_fi_inputs_for_mask = lambda _mask: [fi_path]

    tab._update_fi_feedback()

    text = tab.fi_feedback.toPlainText()
    assert "sample_fi_mCherry_masks.tif (mask)" in text
    assert "raw" not in text.lower()


def test_resolve_effective_mask_finds_filtered_under_tif_files_workspace(qt_parent, tmp_path: Path):
    """Workspace folder (Tif_files/) must still resolve masks living under _segmentation_*/filtered_masks/."""
    tab = AnalyzeTab(parent=qt_parent)
    tab.use_filtered_masks_cb.setChecked(True)
    tif_root = tmp_path / "Tif_files"
    seg = tif_root / "_segmentation_cellpose_sam"
    filt_dir = seg / "filtered_masks"
    filt_dir.mkdir(parents=True)
    raw_mask = seg / "EM20129_NoPhage_fi_mCherry_masks.tif"
    filtered_mask = filt_dir / "EM20129_NoPhage_fi_mCherry_mask_filtered.tif"
    raw_mask.write_bytes(b"raw")
    filtered_mask.write_bytes(b"filt")
    tab.file_path.setText(str(tif_root))

    effective = tab._resolve_effective_mask_for_analysis(raw_mask)
    assert effective.resolve() == filtered_mask.resolve()
    assert tab._fi_feedback_mask_label(raw_mask, effective).endswith("(filtered; from EM20129_NoPhage_fi_mCherry_masks.tif)")


def test_companions_for_analysis_mask_uses_filtered_mask_and_drops_mask_tiffs_from_fi_rows(qt_parent, tmp_path: Path):
    tab = AnalyzeTab(parent=qt_parent)
    tab.fi_mode_match_by_name.setChecked(True)
    tab.analyze_fi.setChecked(True)
    tab.include_foci_from_detection.setChecked(False)

    raw_mask = tmp_path / "sample_fi_mCherry_masks.tif"
    filtered_dir = tmp_path / "filtered_masks"
    filtered_dir.mkdir()
    filtered_mask = filtered_dir / "sample_fi_mCherry_mask_filtered.tif"
    input_tiff = tmp_path / "sample_fi_mCherry.tif"
    foci_tiff = tmp_path / "sample_fi_foci_GFP.tif"
    for path in (raw_mask, filtered_mask, input_tiff, foci_tiff):
        path.write_bytes(b"stub")

    tab._resolve_effective_mask_for_analysis = lambda _mask: filtered_mask
    tab._resolve_fi_paths_for_mask = lambda _mask: [input_tiff]
    tab._resolve_fi_inputs_for_mask = lambda _mask: [input_tiff, foci_tiff]

    with patch("bactomate.gui.tabs._analyze_masks_analyze.resolve_phase_image_for_mask", return_value=input_tiff):
        companions = tab._companions_for_analysis_mask(raw_mask)

    assert companions["mask"] == [(filtered_mask, "filtered mask used")]
    assert companions["phase"] == [(input_tiff, "input image")]
    assert companions["fi"] == [(input_tiff, "fi"), (foci_tiff, "fi foci channel")]
