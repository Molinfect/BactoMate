"""
Contract tests for the Analyze Masks stage split.

These tests intentionally focus on **stage separation** (widget surface + mode behavior),
not on running the full image analysis pipeline (which is integration-tested elsewhere).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None

try:
    from PyQt6.QtWidgets import QApplication
    from bactimate.gui.tabs.analyze_masks import AnalyzeMasksTab, AnalyzeTab, FilteringTab

    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def test_filtering_tab_exposes_only_filtering_controls():
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 not available")
        return
    _app()
    tab = FilteringTab(parent=None)
    assert hasattr(tab, "filter_prob_enabled")
    assert hasattr(tab, "filter_min_int_enabled")
    assert hasattr(tab, "filter_per_cell_ratio_enabled")
    assert not hasattr(tab, "analyze_fi")
    assert tab.preview_btn.text() == "Select Filter"
    assert "Filter" in tab.analyze_btn.text()


def test_analyze_tab_exposes_only_analyze_controls_and_hides_preview():
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 not available")
        return
    _app()
    tab = AnalyzeTab(parent=None)
    assert hasattr(tab, "analyze_fi")
    assert hasattr(tab, "use_filtered_masks_cb")
    assert not hasattr(tab, "filter_prob_enabled")
    assert tab.preview_btn.isVisible() is False
    assert "Analyze" in tab.analyze_btn.text()


def test_factory_returns_expected_tabs_and_rejects_full_mode():
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 not available")
        return
    _app()
    t1 = AnalyzeMasksTab(parent=None, mode="filtering")
    assert isinstance(t1, FilteringTab)
    t2 = AnalyzeMasksTab(parent=None, mode="analyze")
    assert isinstance(t2, AnalyzeTab)
    try:
        AnalyzeMasksTab(parent=None, mode="full")
        assert False, "Expected ValueError for mode='full'"
    except ValueError:
        pass

