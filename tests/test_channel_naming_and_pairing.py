"""Conversion naming + sample/foci pairing (role markers + legacy LCP)."""
from __future__ import annotations

from pathlib import Path

from bactomate.gui.artifact_paths import (
    discover_foci_source_tiffs,
    foci_pairs_with_input,
    longest_common_underscore_prefix,
    sample_identity_stem,
    stem_has_role_marker,
)
from bactomate.modules.conversion_channel_names import (
    channel_name_looks_like_phase,
    conversion_frame_stem_suffix,
    default_fi_indices_for_channels,
    guess_phase_channel_index,
    split_channel_output_name,
)


def test_split_never_emits_bare_free_name():
    phase, used = split_channel_output_name("EM20285_003", "DIA", in_fi=False, in_foci=False, phase_used=False)
    assert phase == "EM20285_003_phase.tif"
    assert used is True
    fi, used2 = split_channel_output_name("EM20285_003", "TRITC", in_fi=False, in_foci=False, phase_used=True)
    assert fi == "EM20285_003_fi_TRITC.tif"
    assert used2 is True


def test_conversion_frame_stem_suffix_only_for_explicit_single_frame():
    assert conversion_frame_stem_suffix(frame_start_0based=0, explicit_single_frame=False) == ""
    assert conversion_frame_stem_suffix(frame_start_0based=0, explicit_single_frame=True) == "_f1"
    assert conversion_frame_stem_suffix(frame_start_0based=49, explicit_single_frame=True) == "_f50"
    # Natural T=1 full export is not explicit single-frame → no suffix.
    assert conversion_frame_stem_suffix(frame_start_0based=0, explicit_single_frame=False) == ""


def test_split_respects_explicit_foci_and_fi():
    name, _ = split_channel_output_name("s", "TRITC", in_fi=False, in_foci=True, phase_used=True)
    assert name == "s_foci_TRITC.tif"
    name, _ = split_channel_output_name("s", "GFP", in_fi=True, in_foci=True, phase_used=True)
    assert name == "s_fi_foci_GFP.tif"


def test_guess_phase_prefers_dia_over_channel_order():
    assert guess_phase_channel_index(["TRITC", "DIA"]) == 1
    assert guess_phase_channel_index(["GFP", "mScarlet"]) == 0  # no hint → fallback 0
    assert channel_name_looks_like_phase("Phase60X")
    assert channel_name_looks_like_phase("DIA")
    assert not channel_name_looks_like_phase("TRITC")


def test_guess_phase_uses_wavelength_flag():
    names = ["Ch1", "Ch2"]
    wls = [{"name": "Ch1"}, {"name": "Ch2", "phase_contrast": True}]
    assert guess_phase_channel_index(names, channel_wavelengths=wls) == 1


def test_default_fi_skips_phase_index():
    assert default_fi_indices_for_channels(2, ["DIA", "TRITC"]) == {1}
    assert default_fi_indices_for_channels(3, ["TRITC", "DIA", "GFP"]) == {0, 2}


def test_sample_identity_strips_role_markers():
    assert sample_identity_stem("EM20285_003_phase") == "EM20285_003"
    assert sample_identity_stem("EM20285_003_foci_TRITC") == "EM20285_003"
    assert sample_identity_stem("EM20285_003_fi_TRITC") == "EM20285_003"
    assert sample_identity_stem("EM20285_003_DIA") == "EM20285_003_DIA"  # free-form kept


def test_stem_has_role_marker():
    assert stem_has_role_marker("EM20285_003_phase")
    assert stem_has_role_marker("sample_fi_GFP")
    assert not stem_has_role_marker("EM20285_003_DIA")


def test_lcp_and_legacy_dia_pairing():
    assert longest_common_underscore_prefix("EM20285_003_DIA", "EM20285_003_foci_TRITC") == "EM20285_003"
    assert foci_pairs_with_input("EM20285_003_DIA", "EM20285_003_foci_TRITC")
    assert foci_pairs_with_input("EM20285_003_phase", "EM20285_003_foci_TRITC")
    assert not foci_pairs_with_input("EM20285_002_DIA", "EM20285_003_foci_TRITC")


def test_role_tagged_replicate_does_not_lcp_cross_pair():
    # Input has _phase → primary identity only (no LCP bleed to shorter replicate).
    assert not foci_pairs_with_input("ssb_cl1_001_phase", "ssb_cl1__foci_GFP")
    assert foci_pairs_with_input("ssb_cl1_001_phase", "ssb_cl1_001_foci_GFP")


def test_discover_foci_finds_legacy_dia_sibling(tmp_path: Path) -> None:
    tif_dir = tmp_path / "Tif_files"
    tif_dir.mkdir()
    (tif_dir / "EM20285_003_DIA.tif").write_bytes(b"")
    (tif_dir / "EM20285_003_foci_TRITC.tif").write_bytes(b"")
    (tif_dir / "EM20285_002_foci_TRITC.tif").write_bytes(b"")
    found = discover_foci_source_tiffs("EM20285_003_DIA", tif_dir)
    names = {p.name for p in found}
    assert "EM20285_003_foci_TRITC.tif" in names
    assert "EM20285_002_foci_TRITC.tif" not in names


def test_discover_foci_canonical_phase(tmp_path: Path) -> None:
    tif_dir = tmp_path / "Tif_files"
    tif_dir.mkdir()
    (tif_dir / "EM20285_003_phase.tif").write_bytes(b"")
    (tif_dir / "EM20285_003_foci_TRITC.tif").write_bytes(b"")
    found = discover_foci_source_tiffs("EM20285_003_phase", tif_dir)
    assert any(p.name == "EM20285_003_foci_TRITC.tif" for p in found)
