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
