"""FI TIFF discovery must search the sample folder, not only filtered_masks/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bactomate.gui.analysis_paths import artifact_search_dirs_for_mask, base_dir_for_mask


class _FiMatchMixin:
    def _candidate_match_stems(self, stem: str) -> list[str]:
        out: list[str] = []

        def _add(s: str) -> None:
            s = (s or "").strip("_")
            if s and s not in out:
                out.append(s)

        _add(stem)
        low = stem.lower()
        if low.endswith("_mask_filtered"):
            _add(stem[: -len("_mask_filtered")])
        if "_phase" in low:
            _add(stem[: low.index("_phase")])
        if "_" in stem:
            _add(stem.rsplit("_", 1)[0])
        if "__" in stem:
            s2 = stem.split("__", 1)[0]
            _add(s2)
        return out

    def _matches_fi_name(self, name: str, stem: str) -> bool:
        if name == f"{stem}_fi" or name.startswith(f"{stem}_fi_"):
            return True
        nl, sl = name.lower(), stem.lower()
        if nl.startswith(sl):
            if len(nl) == len(sl):
                return nl in (f"{sl}_fi", f"{sl}_fi_foci")
            if nl[len(sl)] != "_":
                return False
            rest = nl[len(sl) :].lstrip("_")
            return rest.startswith("fi_foci") or (rest.startswith("fi") and "_foci" in rest)
        return False

    def _resolve_fi_paths_for_mask(self, mask_path: Path) -> list[Path]:
        from bactomate.gui.analysis_paths import artifact_search_dirs_for_mask

        stem = mask_path.stem.replace("_masks", "").replace("_mask_filtered", "")
        stems = self._candidate_match_stems(stem)
        found: list[Path] = []
        seen: set[str] = set()
        for d in artifact_search_dirs_for_mask(mask_path):
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if any(self._matches_fi_name(f.stem, s) for s in stems):
                    key = str(f.resolve())
                    if key not in seen:
                        seen.add(key)
                        found.append(f)
        return sorted(found, key=lambda p: p.name.lower())


def test_artifact_search_dirs_includes_sample_folder(tmp_path: Path) -> None:
    sample = tmp_path / "Tif_files"
    seg = sample / "_segmentation_cellpose_sam"
    filt = seg / "filtered_masks"
    filt.mkdir(parents=True)
    sample.mkdir(exist_ok=True)
    mask = filt / "ssb-mNG_cl1_001_Phase60X_mask_filtered.tif"
    fi = sample / "ssb-mNG_cl1_001_fi_foci_GFP.tif"
    mask.write_bytes(b"x")
    fi.write_bytes(b"x")

    assert base_dir_for_mask(mask) == sample
    dirs = artifact_search_dirs_for_mask(mask)
    assert sample in dirs


def test_resolve_fi_from_sample_folder_when_mask_in_filtered_masks(tmp_path: Path) -> None:
    sample = tmp_path / "Tif_files"
    seg = sample / "_segmentation_cellpose_sam"
    filt = seg / "filtered_masks"
    filt.mkdir(parents=True)
    mask = filt / "ssb-mNG_cl1_001_Phase60X_mask_filtered.tif"
    fi = sample / "ssb-mNG_cl1_001_fi_foci_GFP.tif"
    mask.write_bytes(b"x")
    fi.write_bytes(b"x")

    mixin = _FiMatchMixin()
    found = mixin._resolve_fi_paths_for_mask(mask)
    assert found == [fi]


def test_merge_fi_inputs_adds_foci_channel_when_only_one_explicit_fi(tmp_path: Path) -> None:
    from bactomate.gui.analysis_paths import merge_fi_input_paths

    mask = tmp_path / "sample_Phase60x_mask_filtered.tif"
    dsred_fi = tmp_path / "sample_fi_foci_dsRed.tif"
    gfp_foci = tmp_path / "sample_fi_foci_GFP.tif"
    for p in (mask, dsred_fi, gfp_foci):
        p.write_bytes(b"x")

    def _tag(p: Path) -> str:
        name = p.stem.lower()
        if name.endswith("_dsred"):
            return "dsred"
        if name.endswith("_gfp"):
            return "gfp"
        return ""

    merged = merge_fi_input_paths(
        mask_path=mask,
        explicit_fi=[dsred_fi],
        foci_stacks=[dsred_fi, gfp_foci],
        channel_tag_for_path=_tag,
    )
    names = {p.name for p in merged}
    assert "sample_fi_foci_dsRed.tif" in names
    assert "sample_fi_foci_GFP.tif" in names


def test_resolve_fi_excludes_sibling_replicate(tmp_path: Path) -> None:
    from bactomate.gui.filter_preview_utils import filter_paths_for_mask, same_replicate

    sample = tmp_path / "Tif_files"
    seg = sample / "_segmentation_cellpose_sam"
    filt = seg / "filtered_masks"
    filt.mkdir(parents=True)
    mask = filt / "ssb-mNG_cl1__Phase60X_mask_filtered.tif"
    fi_match = sample / "ssb-mNG_cl1__fi_foci_GFP.tif"
    fi_other = sample / "ssb-mNG_cl1_001_fi_foci_GFP.tif"
    for p in (mask, fi_match, fi_other):
        p.write_bytes(b"x")

    assert same_replicate(mask, fi_match)
    assert not same_replicate(mask, fi_other)

    class _Tab:
        def _matches_fi_name(self, name: str, stem: str) -> bool:
            return "fi_foci" in name.lower() and stem.split("_Phase")[0] in name

        def _matches_foci_name(self, name: str, stem: str) -> bool:
            return False

        def _extract_channel_tag_for_mask_stem(self, file_stem: str, mask_stem: str) -> str | None:
            return None

    raw = [fi_match, fi_other]
    kept = filter_paths_for_mask(_Tab(), mask, raw)
    assert kept == [fi_match]
