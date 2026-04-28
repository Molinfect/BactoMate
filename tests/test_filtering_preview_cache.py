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

* The cache is valid for a (frame_idx, source_key) pair.
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
    from bactimate.gui.tabs._analyze_masks_preview_filtering import is_preview_cache_hit
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


def test_cache_miss_when_source_differs():
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
