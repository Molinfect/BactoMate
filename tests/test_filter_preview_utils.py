"""Tests for filter preview mask / file pairing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bactomate.gui.filter_preview_utils import (
    collect_background_paths_for_mask,
    default_background_index,
    enrich_background_paths_for_mask,
    file_identity_key,
    filter_paths_for_mask,
    format_incremental_apply_plan,
    mask_stem_from_path,
    path_matches_mask_sample,
    phase_stack_candidates_for_mask,
    preview_view_label,
    replicate_token,
    same_replicate,
    short_foci_stack_label,
)


class _TabStub:
    def _candidate_match_stems(self, stem: str) -> list[str]:
        out: list[str] = [stem]
        low = stem.lower()
        if "_phase" in low:
            out.append(stem[: low.index("_phase")])
        if "_" in stem:
            out.append(stem.rsplit("_", 1)[0])
        try:
            from bactomate.gui.artifact_paths import sample_identity_stem

            sid = sample_identity_stem(stem)
            if sid and sid not in out:
                out.append(sid)
        except Exception:
            pass
        return out

    def _matches_fi_name(self, name: str, stem: str) -> bool:
        nl, sl = name.lower(), stem.lower()
        if not nl.startswith(sl):
            return False
        rest = nl[len(sl) :].lstrip("_")
        return rest.startswith("fi_foci") or (rest.startswith("fi") and "_foci" in rest)

    def _matches_foci_name(self, name: str, stem: str) -> bool:
        nl, sl = name.lower(), stem.lower()
        if not nl.startswith(sl):
            return False
        rest = nl[len(sl) :].lstrip("_")
        return rest.startswith("foci") or (rest.startswith("fi") and "_foci" in rest)

    def _extract_channel_tag_for_mask_stem(self, file_stem: str, mask_stem: str) -> str | None:
        for s in self._candidate_match_stems(mask_stem):
            if file_stem.lower().startswith(s.lower() + "_fi_foci_"):
                return file_stem.split("_fi_foci_", 1)[-1]
        return None

    def _preview_fi_path_candidates(self, mask_path: Path) -> list[Path]:
        folder = mask_path.parent
        return [p for p in folder.glob("*.tif") if "_foci" in p.name.lower()]


def test_short_foci_stack_label_extracts_channel() -> None:
    assert short_foci_stack_label(Path("sample_fi_foci_GFP.tif")) == "GFP"
    assert short_foci_stack_label(Path("sample_fi_foci_dsRed.tif")) == "dsRed"


def test_cell_and_foci_settings_signatures_are_independent() -> None:
    from bactomate.gui.filter_preview_utils import (
        cell_filter_settings_signature,
        default_foci_channel_settings,
        foci_channel_settings_signature,
    )

    cell_fs = {"pixel_size": 0.11, "min_aspect_ratio": 1.5}
    foci_a = dict(default_foci_channel_settings())
    foci_b = dict(foci_a)
    foci_b["foci_prob_thresh"] = 0.7
    assert cell_filter_settings_signature(cell_fs) != foci_channel_settings_signature(foci_a)
    assert foci_channel_settings_signature(foci_a) != foci_channel_settings_signature(foci_b)


def test_preview_view_label_shortens_foci_and_phase() -> None:
    assert preview_view_label(Path("sample_fi_foci_GFP.tif")) == "Foci: GFP"
    assert preview_view_label(Path("sample_fi_mCherry.tif")) == "FI: mCherry"
    assert preview_view_label(Path("sample_Phase60x.tif")) == "Phase / input"
    assert preview_view_label(Path("sample_fi_mCherry_masks.tif")) == "Cells (mask)"


def test_is_preview_background_tiff_excludes_masks() -> None:
    from bactomate.gui.filter_preview_utils import _is_preview_background_tiff

    assert _is_preview_background_tiff(Path("sample_fi_mCherry.tif"))
    assert _is_preview_background_tiff(Path("sample_fi_foci_GFP.tif"))
    assert not _is_preview_background_tiff(Path("sample_fi_mCherry_masks.tif"))
    assert not _is_preview_background_tiff(Path("sample_phase_mask_filtered.tif"))


def test_resolve_foci_settings_per_channel() -> None:
    from bactomate.gui.filter_preview_utils import (
        resolve_foci_settings_for_source,
        seed_foci_channel_settings,
    )

    base = {
        "pixel_size": 0.108,
        "foci_prob_enabled": False,
        "foci_prob_thresh": 0.5,
        "foci_channels": {
            "GFP": {"foci_prob_enabled": True, "foci_prob_thresh": 0.9},
            "dsRed": {"foci_prob_enabled": True, "foci_prob_thresh": 0.2},
        },
    }
    gfp = Path("sample_fi_foci_GFP.tif")
    red = Path("sample_fi_foci_dsRed.tif")
    seed_foci_channel_settings(base, [gfp, red])
    gfp_fs = resolve_foci_settings_for_source(base, gfp)
    red_fs = resolve_foci_settings_for_source(base, red)
    assert gfp_fs["foci_prob_thresh"] == 0.9
    assert red_fs["foci_prob_thresh"] == 0.2
    assert gfp_fs["pixel_size"] == 0.108


def test_path_matches_foci_when_mask_stem_includes_phase_suffix(tmp_path: Path) -> None:
    """Foci stacks named on the nd2 base stem must match Phase60x masks."""
    tab = _TabStub()
    mask = tmp_path / "EM15631_sample_Phase60x_masks.tif"
    phase = tmp_path / "EM15631_sample_Phase60x.tif"
    gfp = tmp_path / "EM15631_sample_fi_foci_GFP.tif"
    dsred = tmp_path / "EM15631_sample_fi_foci_dsRed.tif"
    for p in (mask, phase, gfp, dsred):
        p.write_bytes(b"x")
    assert path_matches_mask_sample(tab, mask, phase)
    assert path_matches_mask_sample(tab, mask, gfp)
    assert path_matches_mask_sample(tab, mask, dsred)


def test_is_cell_mask_tiff_excludes_phase_and_foci(tmp_path: Path) -> None:
    from bactomate.gui.artifact_paths import is_cell_mask_tiff

    mask = tmp_path / "sample_Phase60x_masks.tif"
    phase = tmp_path / "sample_Phase60x.tif"
    foci = tmp_path / "sample_fi_foci_GFP.tif"
    for p in (mask, phase, foci):
        p.write_bytes(b"x")
    assert is_cell_mask_tiff(mask)
    assert not is_cell_mask_tiff(phase)
    assert not is_cell_mask_tiff(foci)


def test_enrich_background_paths_adds_resolved_foci(tmp_path: Path) -> None:
    tab = _TabStub()
    mask = tmp_path / "ssb-mNG_cl1__Phase60X_masks.tif"
    phase = tmp_path / "ssb-mNG_cl1__Phase60X.tif"
    gfp = tmp_path / "ssb-mNG_cl1__fi_foci_GFP.tif"
    for p in (mask, phase, gfp):
        p.write_bytes(b"x")
    tab._preview_fi_path_candidates = lambda _m: [gfp]  # type: ignore[method-assign]
    out = enrich_background_paths_for_mask(tab, mask, [phase])
    names = {p.name for p in out}
    assert "ssb-mNG_cl1__Phase60X.tif" in names
    assert "ssb-mNG_cl1__fi_foci_GFP.tif" in names


def test_replicate_token_distinguishes_clones() -> None:
    assert replicate_token("ssb-mNG_cl1__Phase60X") == "__"
    assert replicate_token("ssb-mNG_cl1_001_Phase60X") == "_001_"
    assert replicate_token("ssb-mNG_cl1__fi_foci_GFP") == "__"
    assert replicate_token("ssb-mNG_cl1_001_fi_foci_GFP") == "_001_"


def test_same_replicate_blocks_mixed_clones() -> None:
    mask_a = Path("ssb-mNG_cl1__Phase60X_masks.tif")
    foci_a = Path("ssb-mNG_cl1__fi_foci_GFP.tif")
    foci_b = Path("ssb-mNG_cl1_001_fi_foci_GFP.tif")
    assert same_replicate(mask_a, foci_a)
    assert not same_replicate(mask_a, foci_b)


def test_filter_paths_for_mask_keeps_matching_foci_only(tmp_path: Path) -> None:
    tab = _TabStub()
    paths = []
    for name in (
        "ssb-mNG_cl1__Phase60X_masks.tif",
        "ssb-mNG_cl1__Phase60X.tif",
        "ssb-mNG_cl1__fi_foci_GFP.tif",
        "ssb-mNG_cl1_001_fi_foci_GFP.tif",
        "ssb-mNG_cl1_001_Phase60X.tif",
    ):
        p = tmp_path / name
        p.write_bytes(b"x")
        paths.append(p)
    mask = tmp_path / "ssb-mNG_cl1__Phase60X_masks.tif"
    out = filter_paths_for_mask(tab, mask, paths)
    names = {p.name for p in out}
    assert "ssb-mNG_cl1__Phase60X.tif" in names
    assert "ssb-mNG_cl1__fi_foci_GFP.tif" in names
    assert "ssb-mNG_cl1_001_fi_foci_GFP.tif" not in names
    assert "ssb-mNG_cl1_001_Phase60X.tif" not in names


def test_default_background_index_prefers_phase_matching_mask(tmp_path: Path) -> None:
    tab = _TabStub()
    paths = []
    for name in (
        "ssb-mNG_cl1__Phase60X_masks.tif",
        "ssb-mNG_cl1_001_fi_foci_GFP.tif",
        "ssb-mNG_cl1__fi_foci_GFP.tif",
        "ssb-mNG_cl1__Phase60X.tif",
    ):
        p = tmp_path / name
        p.write_bytes(b"x")
        paths.append(p)
    mask = tmp_path / "ssb-mNG_cl1__Phase60X_masks.tif"
    idx = default_background_index(tab, mask, paths)
    assert paths[idx].name == "ssb-mNG_cl1__Phase60X.tif"


def test_mask_stem_from_path() -> None:
    assert mask_stem_from_path(Path("a_Phase60X_masks.tif")) == "a_Phase60X"


def test_phase_stack_candidates_finds_paired_phase(tmp_path: Path) -> None:
    mask = tmp_path / "ssb-mNG_cl1__Phase60X_masks.tif"
    phase = tmp_path / "ssb-mNG_cl1__Phase60X.tif"
    for p in (mask, phase):
        p.write_bytes(b"x")
    found = phase_stack_candidates_for_mask(mask)
    assert phase in found


def test_phase_stack_dedupes_tif_case_variants(tmp_path: Path) -> None:
    phase = tmp_path / "ssb-mNG_cl1_001_Phase60X.tif"
    phase.write_bytes(b"x")
    mask = tmp_path / "ssb-mNG_cl1_001_Phase60X_masks.tif"
    mask.write_bytes(b"x")
    found = phase_stack_candidates_for_mask(mask)
    assert len(found) == 1
    assert found[0].name == "ssb-mNG_cl1_001_Phase60X.tif"


def test_filter_paths_dedupes_same_path_twice(tmp_path: Path) -> None:
    tab = _TabStub()
    phase = tmp_path / "ssb-mNG_cl1_001_Phase60X.tif"
    phase.write_bytes(b"x")
    mask = tmp_path / "ssb-mNG_cl1_001_Phase60X_masks.tif"
    mask.write_bytes(b"x")
    out = filter_paths_for_mask(tab, mask, [phase, phase])
    assert len(out) == 1


def test_filter_paths_dedupes_tif_spelling_on_case_insensitive_fs(tmp_path: Path) -> None:
    tab = _TabStub()
    phase = tmp_path / "ssb-mNG_cl1_001_Phase60X.tif"
    phase.write_bytes(b"x")
    mask = tmp_path / "ssb-mNG_cl1_001_Phase60X_masks.tif"
    mask.write_bytes(b"x")
    alias = tmp_path / "ssb-mNG_cl1_001_Phase60X.TIF"
    if not alias.is_file():
        return
    assert file_identity_key(phase) == file_identity_key(alias)
    out = filter_paths_for_mask(tab, mask, [phase, alias])
    assert len(out) == 1


def test_filter_paths_blocks_numbered_foci_for_unnumbered_mcherry_mask(tmp_path: Path) -> None:
    """Unnumbered ``NoPhage_mCherry`` must not pull in ``NoPhage_001/002_foci_GFP`` siblings."""
    tab = _TabStub()
    names = (
        "EM20142_NoPhage_mCherry_masks.tif",
        "EM20142_NoPhage_mCherry.tif",
        "EM20142_NoPhage_foci_GFP.tif",
        "EM20142_NoPhage_001_foci_GFP.tif",
        "EM20142_NoPhage_002_foci_GFP.tif",
    )
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"x")
        paths.append(p)
    mask = tmp_path / "EM20142_NoPhage_mCherry_masks.tif"
    out = filter_paths_for_mask(tab, mask, paths)
    kept = {p.name for p in out}
    assert "EM20142_NoPhage_mCherry.tif" in kept
    assert "EM20142_NoPhage_foci_GFP.tif" in kept
    assert "EM20142_NoPhage_001_foci_GFP.tif" not in kept
    assert "EM20142_NoPhage_002_foci_GFP.tif" not in kept


def test_collect_background_paths_includes_phase_and_foci(tmp_path: Path) -> None:
    tab = _TabStub()
    mask = tmp_path / "ssb-mNG_cl1__Phase60X_masks.tif"
    phase = tmp_path / "ssb-mNG_cl1__Phase60X.tif"
    foci = tmp_path / "ssb-mNG_cl1__fi_foci_GFP.tif"
    other = tmp_path / "ssb-mNG_cl1_001_fi_foci_GFP.tif"
    for p in (mask, phase, foci, other):
        p.write_bytes(b"x")
    out = collect_background_paths_for_mask(tab, mask)
    names = {p.name for p in out}
    assert "ssb-mNG_cl1__Phase60X.tif" in names
    assert "ssb-mNG_cl1__fi_foci_GFP.tif" in names
    assert "ssb-mNG_cl1_001_fi_foci_GFP.tif" not in names


def test_sync_filter_settings_from_preview_dialog_calls_ui_sync() -> None:
    from bactomate.gui.filter_preview_utils import sync_filter_settings_from_preview_dialog

    calls: list[str] = []

    class _Dlg:
        def _pf_ui_to_settings(self) -> None:
            calls.append("sync")

    tab = type("Tab", (), {"_filter_preview_dialog": _Dlg()})()
    assert sync_filter_settings_from_preview_dialog(tab) is True
    assert calls == ["sync"]


def test_sync_filter_settings_from_preview_dialog_no_dialog() -> None:
    from bactomate.gui.filter_preview_utils import sync_filter_settings_from_preview_dialog

    tab = type("Tab", (), {})()
    assert sync_filter_settings_from_preview_dialog(tab) is False


def test_batch_filter_percent_two_channels() -> None:
    from bactomate.gui.filter_batch_progress import batch_filter_percent, batch_foci_step_title

    assert batch_filter_percent(file_index=0, file_count=1, channel_index=0, channel_count=2, channel_phase=0.0) == 75
    assert batch_filter_percent(file_index=0, file_count=1, channel_index=1, channel_count=2, channel_phase=0.0) == 88
    assert batch_filter_percent(file_index=0, file_count=1, channel_index=1, channel_count=2, channel_phase=1.0) == 91
    assert batch_filter_percent(file_index=0, file_count=1, finishing_phase=0.0) == 92
    assert batch_filter_percent(file_index=0, file_count=1, finishing_phase=1.0) == 100
    assert "dsRed" in batch_foci_step_title(Path("sample_fi_foci_dsRed.tif"), 0, 2, "Scoring filters")


def test_format_filter_batch_detail_multiline() -> None:
    from bactomate.gui.filter_apply_progress import format_filter_batch_detail

    txt = format_filter_batch_detail(
        file_index=1,
        file_count=2,
        mask_name="sample_Phase60x_masks.tif",
        channel_name="sample_fi_foci_dsRed.tif",
        channel_index=1,
        channel_count=2,
        extra="1,234 peaks",
    )
    assert "\n" in txt
    assert "File 1/2:" in txt
    assert "Foci stack:" in txt
    assert "(1/2)" in txt
    assert "1,234 peaks" in txt


def test_format_incremental_apply_plan_cell_change_rescores_foci() -> None:
    plan = {
        "reuse_cells": False,
        "cell_changed": True,
        "channels_to_score": ["dsRed", "GFP"],
        "prefilled": {},
    }
    txt = format_incremental_apply_plan(plan)
    assert "recompute accept/reject" in txt
    assert "rescore dsRed, GFP" in txt
    assert "rejected cells automatically fail" in txt


def test_format_incremental_apply_plan_foci_only() -> None:
    plan = {
        "reuse_cells": True,
        "cell_changed": False,
        "channels_to_score": ["GFP"],
        "prefilled": {"dsRed": {}},
    }
    txt = format_incremental_apply_plan(plan)
    assert "reuse previous accept/reject" in txt
    assert "rescore GFP" in txt
