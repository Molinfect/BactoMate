"""Regression tests for the Filtering Preview cache-validity rule.

Background
----------
The Filtering Preview window keeps a single ``_last_preview`` cache entry that
holds the result of the most recent ``Apply preview`` / ``Re-apply`` click.
That cache is consulted on every paint (frame change, source change, LUT
slider movement, etc.). A previous version of the cache check also compared
the cached settings signature against the *current* UI / tab settings, which
caused the overlay to silently revert to the unfiltered baseline whenever the
user adjusted any spinbox/checkbox without clicking Apply again.

The contract is now:

* Apply results are valid for a (mask, frame_idx) pair regardless of background View.
* Unapplied UI tweaks MUST NOT invalidate the cache.
* The cache is invalidated only by ``apply_current()`` (which clears
  ``_last_preview`` before the worker starts and refills it on success).

These tests pin that contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

try:
    from bactomate.gui.tabs._analyze_masks_preview_filtering import (
        is_filter_apply_cache_hit,
        is_preview_cache_hit,
    )
except Exception as e:  # pragma: no cover - import guard for headless environments
    pytest.skip(f"PyQt6 import failed: {e}", allow_module_level=True)


def _cached_entry(
    *,
    frame_idx: int = 0,
    source_key: tuple = ("none", -1),
    settings_sig: str = "sig-A",
    with_mask: bool = True,
) -> dict:
    entry: dict = {
        "frame_idx": int(frame_idx),
        "source_key": tuple(source_key),
        "settings_sig": str(settings_sig),
        "accepted": {1, 2},
        "rejected": {3},
        "pass_peaks": [],
        "fail_peaks": [],
        "info_text": "",
    }
    if with_mask:
        entry["mask_2d"] = np.zeros((4, 4), dtype=np.int32)
    return entry


def test_empty_cache_is_a_miss():
    assert is_preview_cache_hit({}, 0, ("none", -1)) is False
    assert is_preview_cache_hit(None, 0, ("none", -1)) is False


def test_cache_with_only_info_text_is_a_miss():
    assert is_preview_cache_hit({"info_text": "Error: foo"}, 0, ("none", -1)) is False


def test_cache_hit_when_frame_and_source_match():
    last = _cached_entry(frame_idx=2, source_key=("img", 0))
    assert is_preview_cache_hit(last, 2, ("img", 0)) is True


def test_cache_miss_when_frame_differs():
    last = _cached_entry(frame_idx=2, source_key=("img", 0))
    assert is_preview_cache_hit(last, 3, ("img", 0)) is False


def test_lut_window_spans_frame_detects_crushed_display() -> None:
    from bactomate.gui.tabs._analyze_masks_preview_filtering import _lut_window_spans_frame

    bg = np.linspace(0.0, 100.0, 64, dtype=np.float64).reshape(8, 8)
    assert _lut_window_spans_frame(5.0, 95.0, bg) is True
    # Stale high-gain LUT from a previous View → solid black on phase.
    assert _lut_window_spans_frame(500.0, 800.0, bg) is False
    assert _lut_window_spans_frame(float("nan"), 10.0, bg) is False
    # Foci-scale window still intersects phase intensities but crushes them.
    phase = np.linspace(800.0, 4000.0, 100, dtype=np.float64).reshape(10, 10)
    assert _lut_window_spans_frame(0.0, 65535.0, phase) is False
    assert _lut_window_spans_frame(700.0, 4200.0, phase) is True


def test_stretch_plane_to_uint8_recovers_from_crushed_foci_lut() -> None:
    """Regression: Phase must stay visible even when state still holds a foci LUT."""
    from bactomate.gui.tabs._analyze_masks_preview_filtering import stretch_plane_to_uint8

    phase = np.linspace(800.0, 4000.0, 1024, dtype=np.float64).reshape(32, 32)
    crushed = stretch_plane_to_uint8(phase, 20000.0, 34000.0)
    assert int(crushed.max()) >= 40
    assert float(crushed.mean()) > 10.0
    # Naive mapping with the crushed window is nearly black — helper must not.
    naive = ((phase - 20000.0) / (34000.0 - 20000.0) * 255.0).clip(0, 255)
    assert float(naive.mean()) < 1.0


def test_source_key_from_state_accepts_list() -> None:
    from bactomate.gui.tabs._analyze_masks_preview_filtering import _source_key_from_state

    class _State:
        source_key = ["file", 3]

    assert _source_key_from_state(_State(), lambda: ("none", -1)) == ("file", 3)


def test_source_key_from_state_prefers_visible_view_on_desync() -> None:
    """When dialog state and View disagree, paint the visible View channel."""
    from bactomate.gui.tabs._analyze_masks_preview_filtering import _source_key_from_state

    class _State:
        source_key = ["file", 2]  # Phase stuck in dialog state

    assert _source_key_from_state(_State(), lambda: ("file", 0)) == ("file", 0)


def test_preview_source_key_list_tuple_equality() -> None:
    from bactomate.gui.preview.navigation_controls import (
        normalize_preview_source_key,
        preview_source_keys_equal,
    )

    assert normalize_preview_source_key(["file", 2]) == ("file", 2)
    assert preview_source_keys_equal(("file", 2), ["file", 2])
    assert not preview_source_keys_equal(("file", 0), ["file", 2])


def test_filter_apply_cache_hit_when_only_view_source_differs():
    """Changing background View must not drop applied cell/foci overlays."""
    last = _cached_entry(frame_idx=0, source_key=("img", 0))
    assert is_filter_apply_cache_hit(last, 0) is True
    assert is_preview_cache_hit(last, 0, ("img", 1)) is False


def test_cache_miss_when_source_differs_for_full_preview_hit():
    last = _cached_entry(frame_idx=0, source_key=("img", 0))
    assert is_preview_cache_hit(last, 0, ("img", 1)) is False


def test_cache_hit_ignores_settings_signature_changes():
    """The bug: the previous _render() compared current UI settings to the
    cached settings_sig and fell back to the unfiltered baseline whenever
    they differed. The contract is now that unapplied UI tweaks (which only
    change the *current* settings sig, never the cached entry) must not
    invalidate the cached overlay.
    """

    last = _cached_entry(frame_idx=0, source_key=("img", 0), settings_sig="sig-A")

    assert is_preview_cache_hit(last, 0, ("img", 0)) is True


def test_cache_hit_handles_list_source_key():
    """source_key is sometimes passed as list; the helper must coerce to tuple."""

    last = _cached_entry(frame_idx=0, source_key=("img", 0))
    assert is_preview_cache_hit(last, 0, ["img", 0]) is True


def test_cache_miss_when_mask_missing_even_if_frame_matches():
    """An error-path entry that only carries info_text must not be reused."""

    last = _cached_entry(frame_idx=0, source_key=("img", 0), with_mask=False)
    assert is_preview_cache_hit(last, 0, ("img", 0)) is False
