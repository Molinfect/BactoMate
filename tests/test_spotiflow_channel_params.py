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


def test_ensure_spotiflow_channel_entries_seeds_empty_overrides(seg_tab, tmp_path):
    a = tmp_path / "sample_foci_ssb.tif"
    b = tmp_path / "sample_foci_phage.tif"
    a.touch()
    b.touch()
    seg_tab._spotiflow_detection_params["prob_thresh"] = 0.42
    seg_tab._ensure_spotiflow_channel_entries([a, b])
    from bactomate.gui.filter_preview_utils import foci_channel_key_for_path

    k_a = foci_channel_key_for_path(a)
    k_b = foci_channel_key_for_path(b)
    assert seg_tab._spotiflow_channel_params[k_a] == {}
    assert seg_tab._spotiflow_channel_params[k_b] == {}
    assert seg_tab._get_spotiflow_detection_params(channel_key=k_a)["prob_thresh"] == 0.42


def test_channel_save_does_not_mutate_global_or_seed_other_channel(seg_tab, tmp_path):
    seg_tab._foci_slice = {"normalize": False, "perc_low": 2.0, "perc_high": 99.0,
                           "use_optimized_threshold": False, "prob_thresh": 0.9, "min_distance": 3}
    seg_tab._spotiflow_slice_from_preview_widgets = lambda dlg: dict(seg_tab._foci_slice, subpix=True)
    before = dict(seg_tab._spotiflow_detection_params)
    seg_tab._save_spotiflow_widgets_to_channel(MagicMock(), "ssb")
    assert seg_tab._spotiflow_detection_params == before
    entry = seg_tab._spotiflow_channel_params["ssb"]
    expected = {k for k, v in seg_tab._foci_slice.items() if SPOTIFLOW_DETECTION_DEFAULTS.get(k) != v}
    assert set(entry) == expected and "prob_thresh" in entry
    c = tmp_path / "s_foci_phage.tif"
    c.touch()
    seg_tab._ensure_spotiflow_channel_entries([c])
    from bactomate.gui.filter_preview_utils import foci_channel_key_for_path

    other = seg_tab._get_spotiflow_detection_params(channel_key=foci_channel_key_for_path(c))
    assert other["prob_thresh"] is None and other["use_optimized_threshold"] is True
    assert seg_tab._get_spotiflow_detection_params(channel_key="ssb")["prob_thresh"] == 0.9


def test_legacy_full_snapshot_session_is_stripped_on_load(seg_tab):
    legacy = dict(SPOTIFLOW_DETECTION_DEFAULTS, prob_thresh=0.3, use_optimized_threshold=False, model_name="old")
    seg_tab.set_session_params({"spotiflow_channels": {"ssb": legacy}, "spotiflow_model_name": "new"})
    entry = seg_tab._spotiflow_channel_params["ssb"]
    assert "model_name" not in entry and "subpix" not in entry
    resolved = seg_tab._get_spotiflow_detection_params(channel_key="ssb")
    assert resolved["model_name"] == "new"
    assert resolved["prob_thresh"] == 0.3


def test_per_channel_override_does_not_bleed(seg_tab):
    seg_tab._spotiflow_channel_params = {
        "ssb": {"prob_thresh": 0.2, "use_optimized_threshold": False},
        "phage": {"prob_thresh": 0.8, "use_optimized_threshold": False},
    }
    ssb = seg_tab._get_spotiflow_detection_params(channel_key="ssb")
    phage = seg_tab._get_spotiflow_detection_params(channel_key="phage")
    assert ssb["prob_thresh"] == 0.2
    assert phage["prob_thresh"] == 0.8


def test_legacy_per_sample_key_migrates_to_canonical(seg_tab, tmp_path):
    seg_tab.set_session_params({"spotiflow_channels": {"EM20129-ssb_foci": {"prob_thresh": 0.66, "use_optimized_threshold": False}}})
    a = tmp_path / "EM20129-ssb_foci.tif"
    b = tmp_path / "EM20130-ssb_foci.tif"
    seg_tab._ensure_spotiflow_channel_entries([a])
    assert seg_tab._spotiflow_channel_params["ssb"]["prob_thresh"] == 0.66
    # Another sample with the same channel shares the migrated settings.
    key_b = seg_tab._spotiflow_channel_key_for_path(b)
    assert key_b == "ssb"
    assert seg_tab._get_spotiflow_detection_params(channel_key=key_b)["prob_thresh"] == 0.66


def test_session_keys_are_case_normalized(seg_tab, tmp_path):
    seg_tab.set_session_params({"spotiflow_channels": {"GFP": {"prob_thresh": 0.25, "use_optimized_threshold": False}}})
    assert "gfp" in seg_tab._spotiflow_channel_params
    assert seg_tab._get_spotiflow_detection_params(channel_key="GFP")["prob_thresh"] == 0.25
    key = seg_tab._spotiflow_channel_key_for_path(tmp_path / "s_fi_foci_GFP.tif")
    assert seg_tab._get_spotiflow_detection_params(channel_key=key)["prob_thresh"] == 0.25


def test_truncated_legacy_key_migrates(seg_tab, tmp_path):
    seg_tab._spotiflow_channel_params = {"experimentA_longname_posi…": {"prob_thresh": 0.4}}
    p = tmp_path / "experimentA_longname_position01-GFP_foci.tif"
    key = seg_tab._spotiflow_channel_key_for_path(p)
    assert key == "gfp"
    assert seg_tab._spotiflow_channel_params["gfp"] == {"prob_thresh": 0.4}


def test_folder_preview_seg_compute_uses_legacy_entry():
    from bactomate.gui.filter_preview_utils import foci_channel_entry_for_path

    key, entry = foci_channel_entry_for_path({"EM20129-ssb_foci": {"prob_thresh": 0.5}}, Path("EM20129-ssb_foci.tif"))
    assert key == "ssb" and entry == {"prob_thresh": 0.5}


def test_channel_save_stores_only_differences_and_drops_equal_keys(seg_tab):
    seg_tab._spotiflow_detection_params["prob_thresh"] = 0.42
    seg_tab._spotiflow_slice_from_preview_widgets = lambda dlg: {**{k: SPOTIFLOW_DETECTION_DEFAULTS[k] for k in ("normalize", "perc_low", "perc_high", "use_optimized_threshold", "min_distance")}, "prob_thresh": 0.9}
    seg_tab._save_spotiflow_widgets_to_channel(MagicMock(), "ssb")
    assert seg_tab._spotiflow_channel_params["ssb"] == {"prob_thresh": 0.9}
    # Widget now equals global: override is dropped (inherits again).
    seg_tab._spotiflow_slice_from_preview_widgets = lambda dlg: {"prob_thresh": 0.42}
    seg_tab._save_spotiflow_widgets_to_channel(MagicMock(), "ssb")
    assert seg_tab._spotiflow_channel_params["ssb"] == {}
    assert seg_tab._get_spotiflow_detection_params(channel_key="ssb")["prob_thresh"] == 0.42
