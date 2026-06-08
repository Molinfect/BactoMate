"""Per-channel Spotiflow settings on the segmentation tab."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bactomate.gui.tabs.omnipose_seg import SPOTIFLOW_DETECTION_DEFAULTS, OmniposeSegTab


@pytest.fixture
def seg_tab(qapp):
    tab = OmniposeSegTab(None)
    tab.parent_window = MagicMock()
    tab.parent_window.log_output = MagicMock()
    tab.parent_window._save_session_log = None
    tab._spotiflow_detection_params = dict(SPOTIFLOW_DETECTION_DEFAULTS)
    tab._spotiflow_channel_params = {}
    return tab


def test_ensure_spotiflow_channel_entries_seeds_per_stack(seg_tab, tmp_path):
    a = tmp_path / "sample_foci_ssb.tif"
    b = tmp_path / "sample_foci_phage.tif"
    a.touch()
    b.touch()
    seg_tab._spotiflow_detection_params["prob_thresh"] = 0.42
    seg_tab._ensure_spotiflow_channel_entries([a, b])
    from bactomate.gui.filter_preview_utils import foci_channel_key_for_path

    k_a = foci_channel_key_for_path(a)
    k_b = foci_channel_key_for_path(b)
    assert k_a in seg_tab._spotiflow_channel_params
    assert k_b in seg_tab._spotiflow_channel_params
    assert seg_tab._spotiflow_channel_params[k_a]["prob_thresh"] == 0.42


def test_per_channel_override_does_not_bleed(seg_tab):
    seg_tab._spotiflow_channel_params = {
        "ssb": {"prob_thresh": 0.2, "use_optimized_threshold": False},
        "phage": {"prob_thresh": 0.8, "use_optimized_threshold": False},
    }
    ssb = seg_tab._get_spotiflow_detection_params(channel_key="ssb")
    phage = seg_tab._get_spotiflow_detection_params(channel_key="phage")
    assert ssb["prob_thresh"] == 0.2
    assert phage["prob_thresh"] == 0.8
