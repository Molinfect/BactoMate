"""Canonical, sample-independent foci channel key (settings) vs. artifact channel id (CSV names)."""
from __future__ import annotations

from pathlib import Path

import pytest

from bactomate.gui.artifact_paths import (
    FOCI_CHANNEL_FALLBACK_TAG,
    foci_channel_id_from_source,
    foci_channel_key_from_path,
    foci_channel_tag_from_path,
)
from bactomate.gui.filter_preview_utils import (
    default_foci_channel_settings,
    foci_channel_key_for_path,
    legacy_foci_channel_keys_for_path,
    resolve_foci_settings_for_source,
    seed_foci_channel_settings,
    short_foci_stack_label,
)


@pytest.mark.parametrize(
    ("name", "tag"),
    [
        ("EM20129_foci_ssb.tif", "ssb"),
        ("sample_fi_foci_GFP.tif", "GFP"),
        ("sample__fi_foci_GFP.tif", "GFP"),
        ("G183A-004_foci_dsRed.tif", "dsRed"),
        ("sample_foci_overlay_GFP.tif", "GFP"),
        ("sample_foci_2.tif", "2"),
        ("EM20129-ssb_foci.tif", "ssb"),
        ("EM20129 ssb_foci.tif", "ssb"),
        ("sample_001_GFP_fi_foci.tif", "GFP"),
        ("sample_fi_GFP_foci.tif", "GFP"),
        ("sample_foci.tif", FOCI_CHANNEL_FALLBACK_TAG),
        ("EM20285_003_foci.tif", FOCI_CHANNEL_FALLBACK_TAG),
        ("sample_001_fi_foci.tif", FOCI_CHANNEL_FALLBACK_TAG),
        ("sample_foci_fi.tif", FOCI_CHANNEL_FALLBACK_TAG),
        ("sample_fi_mCherry.tif", "mCherry"),
        ("some_unrelated_name_that_is_definitely_longer_than_28.tif", "some_unrelated_name_that_is_definitely_longer_than_28"),
    ],
)
def test_canonical_tag_across_conventions(name: str, tag: str) -> None:
    assert foci_channel_tag_from_path(Path(name)) == tag
    assert foci_channel_key_from_path(Path(name)) == tag.casefold()
    assert foci_channel_key_for_path(Path(name)) == tag.casefold()


def test_key_is_sample_independent_and_case_insensitive() -> None:
    assert foci_channel_key_for_path(Path("EM20129-ssb_foci.tif")) == foci_channel_key_for_path(Path("EM20130-ssb_foci.tif"))
    assert foci_channel_key_for_path(Path("a_foci_GFP.tif")) == foci_channel_key_for_path(Path("b_fi_foci_gfp.tif")) == "gfp"
    # Labels keep the original case.
    assert short_foci_stack_label(Path("a_foci_GFP.tif")) == "GFP"


def test_long_names_do_not_collide_and_keys_never_truncate() -> None:
    gfp = Path("experimentA_longname_position01-GFP_foci.tif")
    mch = Path("experimentA_longname_position01-mCh_foci.tif")
    assert foci_channel_key_for_path(gfp) == "gfp"
    assert foci_channel_key_for_path(mch) == "mch"
    long_stem = "x" * 40
    assert foci_channel_key_for_path(Path(f"{long_stem}.tif")) == long_stem
    assert short_foci_stack_label(Path(f"{long_stem}.tif")).endswith("…")


def test_legacy_keys_for_path() -> None:
    assert legacy_foci_channel_keys_for_path(Path("EM20129-ssb_foci.tif")) == ["em20129-ssb_foci"]
    assert legacy_foci_channel_keys_for_path(Path("experimentA_longname_position01-GFP_foci.tif")) == ["experimenta_longname_posi…"]
    # Same key as before (only case changes) -> nothing to migrate beyond normalization.
    assert legacy_foci_channel_keys_for_path(Path("s_foci_GFP.tif")) == []


def test_filter_settings_migrate_legacy_key_to_canonical() -> None:
    custom = dict(default_foci_channel_settings(), foci_prob_thresh=0.77)
    settings = {"foci_channels": {"EM20129-ssb_foci": dict(custom), "_default": default_foci_channel_settings()}}
    seed_foci_channel_settings(settings, [Path("EM20129-ssb_foci.tif")])
    assert settings["foci_channels"]["ssb"]["foci_prob_thresh"] == 0.77
    # A second sample of the same channel now resolves to the same (migrated) settings.
    merged = resolve_foci_settings_for_source(settings, Path("EM20130-ssb_foci.tif"))
    assert merged["foci_prob_thresh"] == 0.77


def test_filter_settings_resolve_migrates_truncated_and_upper_case_keys() -> None:
    trunc = dict(default_foci_channel_settings(), foci_prob_thresh=0.33)
    upper = dict(default_foci_channel_settings(), foci_prob_thresh=0.55)
    settings = {"foci_channels": {"experimentA_longname_posi…": trunc, "GFP": upper}}
    merged = resolve_foci_settings_for_source(settings, Path("experimentA_longname_position01-mCh_foci.tif"))
    assert merged["foci_prob_thresh"] == 0.33
    merged = resolve_foci_settings_for_source(settings, Path("s_fi_foci_GFP.tif"))
    assert merged["foci_prob_thresh"] == 0.55


def test_filtering_tab_channel_accessors_are_case_insensitive() -> None:
    from bactomate.gui.tabs._analyze_masks_filtering import FilteringTab

    # Settings accessors only touch ``_filter_settings``; avoid building the QWidget.
    class _Stub:
        _default_filter_settings = FilteringTab._default_filter_settings
        _ensure_filter_settings = FilteringTab._ensure_filter_settings
        set_filter_settings = FilteringTab.set_filter_settings
        get_foci_channel_settings = FilteringTab.get_foci_channel_settings
        set_foci_channel_settings = FilteringTab.set_foci_channel_settings

    tab = _Stub()
    tab._filter_settings = {}
    tab.set_filter_settings({"foci_channels": {"GFP": dict(default_foci_channel_settings(), foci_prob_thresh=0.6)}})
    assert tab.get_foci_channel_settings("gfp")["foci_prob_thresh"] == 0.6
    tab.set_foci_channel_settings("Gfp", {"foci_prob_thresh": 0.1})
    assert tab.get_foci_channel_settings("GFP")["foci_prob_thresh"] == 0.1


@pytest.mark.parametrize(
    "foci_name",
    ["G183A-004_foci_GFP.tif", "G183A-004_fi_foci_dsRed.tif", "G183A-004__foci_mCherry.tif"],
)
def test_artifact_channel_id_matches_canonical_tag_for_explicit_foci_channel(foci_name: str) -> None:
    mask = Path("G183A-004_Phase60x_masks.tif")
    assert foci_channel_id_from_source(mask, Path(foci_name)) == foci_channel_tag_from_path(Path(foci_name))


@pytest.mark.parametrize(
    ("mask_name", "foci_name", "artifact_id", "tag"),
    [
        ("EM20129_masks.tif", "EM20129-ssb_foci.tif", "ssb_foci", "ssb"),
        ("EM20129-Membrane_masks.tif", "EM20129-ssb_foci.tif", "EM20129-ssb_foci", "ssb"),
        ("sample_001_masks.tif", "sample_001_GFP_fi_foci.tif", "GFP_fi_foci", "GFP"),
        ("sample_masks.tif", "sample_foci.tif", "foci", "foci"),
    ],
)
def test_artifact_channel_id_is_frozen_for_suffix_conventions(
    mask_name: str, foci_name: str, artifact_id: str, tag: str
) -> None:
    """On-disk CSV names stay mask-relative (existing pipelines); settings use the canonical tag."""
    assert foci_channel_id_from_source(Path(mask_name), Path(foci_name)) == artifact_id
    assert foci_channel_tag_from_path(Path(foci_name)) == tag


def _stub_tab():
    from bactomate.gui.tabs._analyze_masks_filtering import FilteringTab

    class _Stub:
        _default_filter_settings = FilteringTab._default_filter_settings
        _ensure_filter_settings = FilteringTab._ensure_filter_settings
        set_filter_settings = FilteringTab.set_filter_settings
        get_foci_channel_settings = FilteringTab.get_foci_channel_settings
        set_foci_channel_settings = FilteringTab.set_foci_channel_settings

    tab = _Stub()
    tab._filter_settings = {}
    return tab


def test_get_foci_channel_settings_does_not_create_entry_on_read() -> None:
    tab = _stub_tab()
    tab.set_filter_settings({"foci_channels": {"gfp": dict(default_foci_channel_settings(), foci_prob_thresh=0.6)}})
    got = tab.get_foci_channel_settings("mcherry")
    assert got == default_foci_channel_settings()
    assert "mcherry" not in tab._filter_settings["foci_channels"]


def test_foci_channel_settings_use_legacy_entry_when_source_given(tmp_path) -> None:
    tab = _stub_tab()
    src = tmp_path / "EM20129-ssb_foci.tif"
    legacy = legacy_foci_channel_keys_for_path(src)
    assert legacy
    canonical = foci_channel_key_for_path(src)
    tab.set_filter_settings({"foci_channels": {legacy[0]: dict(default_foci_channel_settings(), foci_prob_thresh=0.7)}})
    assert tab.get_foci_channel_settings(canonical, foci_source=src)["foci_prob_thresh"] == 0.7
    assert canonical not in tab._filter_settings["foci_channels"]
    tab.set_foci_channel_settings(canonical, {"foci_min_int": 5.0}, foci_source=src)
    entry = tab._filter_settings["foci_channels"][canonical]
    assert entry["foci_prob_thresh"] == 0.7 and entry["foci_min_int"] == 5.0


def test_normalize_foci_channel_map_collision_is_deterministic(caplog) -> None:
    from bactomate.gui.filter_preview_utils import normalize_foci_channel_map

    a = {"GFP": {"v": 1}, "gfp": {"v": 2}}
    b = {"gfp": {"v": 2}, "GFP": {"v": 1}}
    assert normalize_foci_channel_map(a) == normalize_foci_channel_map(b) == {"gfp": {"v": 2}}
    c = {"Gfp": {"v": 3}, "GFP": {"v": 1}}
    d = {"GFP": {"v": 1}, "Gfp": {"v": 3}}
    assert normalize_foci_channel_map(c) == normalize_foci_channel_map(d) == {"gfp": {"v": 1}}
