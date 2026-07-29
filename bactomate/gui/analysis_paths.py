"""
Shared rules for analysis output paths: base dir, _analysis folder, stem from mask, find analysis Excel.
Used by Analyze Masks and Tracking tabs to avoid duplicate logic (Phase 4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

ANALYSIS_SUBDIR = "_analysis"
ANALYSIS_VIS_SUBDIR = "visualizations"


def stem_from_mask_path(mask_path: Path) -> str:
    """Stem for naming (e.g. stem_analysis.xlsx). mask_path.stem with _masks removed."""
    return mask_path.stem.replace("_masks", "")


def artifact_search_dirs_for_mask(mask_path: Path) -> list[Path]:
    """
    Directories to search for sibling TIFFs (FI, foci stacks, phase) for one mask.

    Includes the sample folder from :func:`base_dir_for_mask` (e.g. parent of
  ``_segmentation_*``) so artifacts in ``Tif_files/`` are found when masks live in
    ``_segmentation_*/filtered_masks/``.
    """
    mask_path = Path(mask_path)
    parent = mask_path.parent
    search_dirs: list[Path] = []
    try:
        sample_dir = base_dir_for_mask(mask_path)
        if sample_dir.is_dir():
            search_dirs.append(sample_dir)
    except Exception:
        pass
    if parent.name.startswith("_segmentation"):
        search_dirs.extend([parent.parent, parent])
    else:
        search_dirs.append(parent)
        if parent.parent != parent:
            search_dirs.append(parent.parent)
    seen: set[str] = set()
    unique: list[Path] = []
    for d in search_dirs:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def base_dir_for_mask(mask_path: Path) -> Path:
    """
    Sample-level directory for pipeline artifacts (sibling of ``_segmentation_*``).

    One level up when the mask lives under ``Tif_*``, ``_segmentation*``, ``_analysis``,
    or ``filtered_masks/`` inside a segmentation folder.
    """
    base = Path(mask_path).parent
    if base.name == "filtered_masks" and "_segmentation" in base.parent.name:
        base = base.parent
    if base.name.startswith("Tif_") or "_segmentation" in base.name or base.name == "_analysis":
        base = base.parent
    return base


def analysis_dir_from_base(base_dir: Path) -> Path:
    """Directory where analysis Excel files live (base_dir / _analysis)."""
    return base_dir / ANALYSIS_SUBDIR


def analysis_visualizations_dir(analysis_dir: Path | str) -> Path:
    """QC images under ``_analysis/visualizations/`` (filter + analysis panels)."""
    return Path(analysis_dir) / ANALYSIS_VIS_SUBDIR


def find_analysis_excel_for_mask(mask_path: Path, base_dir: Path) -> Path | None:
    """Find analysis Excel for this mask in base_dir / _analysis (e.g. {stem}_analysis.xlsx or {stem}_masks_analysis.xlsx). Returns None if not found."""
    stem = stem_from_mask_path(mask_path)
    analysis_dir = analysis_dir_from_base(base_dir)
    if not analysis_dir.is_dir():
        return None
    for name in (f"{stem}_analysis.xlsx", f"{stem}_masks_analysis.xlsx"):
        p = analysis_dir / name
        if p.is_file():
            return p
    return None


def resolve_analysis_workbook_for_mask(mask_path: Path) -> Path | None:
    """
    Locate an existing analysis workbook for a raw or filtered mask.

    Tries the filtered-mask stem first (e.g. ``*_mask_filtered_analysis.xlsx``), then the
    original mask stem (e.g. ``*_masks`` → ``*_analysis.xlsx``).
    """
    mask_path = Path(mask_path)
    candidates: list[Path] = []
    try:
        from bactomate.gui.artifact_paths import filtered_mask_path_for_mask

        filt = filtered_mask_path_for_mask(mask_path)
        candidates.append(filt)
    except Exception:
        pass
    if mask_path not in candidates:
        candidates.append(mask_path)
    seen: set[str] = set()
    for mp in candidates:
        try:
            base = base_dir_for_mask(mp)
        except Exception:
            continue
        wb = find_analysis_excel_for_mask(mp, base)
        if wb is None:
            continue
        try:
            key = str(wb.resolve())
        except Exception:
            key = str(wb)
        if key in seen:
            continue
        seen.add(key)
        return wb
    return None


def merge_fi_input_paths(
    *,
    mask_path: Path,
    explicit_fi: list[Path],
    foci_stacks: list[Path],
    channel_tag_for_path: Callable[[Path], str],
    filter_paths: Callable[[Path, list[Path]], list[Path]] | None = None,
) -> list[Path]:
    """
    Build Analyze FI inputs: named FI TIFFs plus foci stacks for channels not already covered.

    When at least one ``*_fi*`` file exists, older logic returned only those files and skipped
    other channels (e.g. dsRed FI present but GFP only as ``*_fi_foci_GFP.tif``).
    """
    mask_path = Path(mask_path)
    explicit = [Path(p) for p in (explicit_fi or []) if Path(p).is_file()]
    foci = [Path(p) for p in (foci_stacks or []) if Path(p).is_file()]
    if not explicit:
        out = list(foci)
    else:
        covered_tags: set[str] = set()
        out: list[Path] = []
        seen: set[str] = set()
        for p in explicit:
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            tag = str(channel_tag_for_path(p) or "").strip().lower()
            if tag:
                covered_tags.add(tag)
        for p in foci:
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key in seen:
                continue
            tag = str(channel_tag_for_path(p) or "").strip().lower()
            if tag and tag in covered_tags:
                continue
            seen.add(key)
            out.append(p)
    if filter_paths is not None:
        try:
            return list(filter_paths(mask_path, out))
        except Exception:
            pass
    return out
