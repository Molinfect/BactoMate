"""Tests for session tab payload normalization."""

from __future__ import annotations

from bactomate.gui.session_schema import LEGACY_UNET_TAB_KEY, normalize_session_tabs


def test_normalize_merges_legacy_unet_when_performed() -> None:
    tabs = {
        LEGACY_UNET_TAB_KEY: {"unet_model": "custom", "performed": True},
        "segmentation": {"create_overlay": False},
    }
    out = normalize_session_tabs(tabs)
    assert LEGACY_UNET_TAB_KEY not in out
    seg = out["segmentation"]
    assert seg["segmentation_backend"] == "unet"
    assert seg["unet_model"] == "custom"
    assert seg["create_overlay"] is False
    assert seg["performed"] is True


def test_normalize_keeps_explicit_segmentation_backend() -> None:
    tabs = {
        LEGACY_UNET_TAB_KEY: {"unet_model": "legacy", "performed": True},
        "segmentation": {"segmentation_backend": "omnipose", "model": "bact_phase_omni"},
    }
    out = normalize_session_tabs(tabs)
    seg = out["segmentation"]
    assert seg["segmentation_backend"] == "omnipose"
    assert seg["model"] == "bact_phase_omni"


def test_normalize_noop_without_legacy_unet() -> None:
    tabs = {"segmentation": {"segmentation_backend": "cellpose", "performed": True}}
    out = normalize_session_tabs(tabs)
    assert out == tabs
