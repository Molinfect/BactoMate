"""
Shared rules for analysis output paths: base dir, _analysis folder, stem from mask, find analysis Excel.
Used by Analyze Masks and Tracking tabs to avoid duplicate logic (Phase 4).
"""
from pathlib import Path

ANALYSIS_SUBDIR = "_analysis"


def stem_from_mask_path(mask_path: Path) -> str:
    """Stem for naming (e.g. stem_analysis.xlsx). mask_path.stem with _masks removed."""
    return mask_path.stem.replace("_masks", "")


def base_dir_for_mask(mask_path: Path) -> Path:
    """Base directory for this mask. One level up if mask is inside Tif_*, _segmentation*, or _analysis."""
    base = mask_path.parent
    if base.name.startswith("Tif_") or "_segmentation" in base.name or base.name == "_analysis":
        base = base.parent
    return base


def analysis_dir_from_base(base_dir: Path) -> Path:
    """Directory where analysis Excel files live (base_dir / _analysis)."""
    return base_dir / ANALYSIS_SUBDIR


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
