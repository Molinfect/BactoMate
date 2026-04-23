"""
Stage artifact naming + discovery helpers.

These helpers are intentionally GUI-agnostic: they encode filename conventions for
Segmentation → Filtering → Analyze so both UI code and CLI/batch utilities can
write/read the same artifacts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bactimate.gui.analysis_paths import base_dir_for_mask, stem_from_mask_path


@dataclass(frozen=True)
class PipelineArtifacts:
    """Computed artifact paths for one mask + (optional) foci source."""

    pipeline_dir: Path
    mask_path: Path
    filtered_mask_path: Path
    foci_csv_path: Optional[Path] = None
    raw_foci_csv_path: Optional[Path] = None  # legacy name (kept for migration)
    filtered_foci_csv_path: Optional[Path] = None


def segmentation_pipeline_dir_from_input(
    input_path: Path,
    *,
    pipeline_folder_name: Optional[str] = None,
) -> Path:
    """
    Return the pipeline folder for outputs.

    Convention:
    - If `input_path` is a file: pipeline dir is next to the file.
    - If `input_path` is a folder: pipeline dir is inside it.
    - If `pipeline_folder_name` is provided, it is appended accordingly.

    This keeps the helper generic; specific backends can pass their existing
    `_segmentation_*` folder name.
    """
    p = Path(input_path)
    root = p.parent if p.is_file() else p
    if pipeline_folder_name:
        return root / pipeline_folder_name
    return root


def filtered_mask_path_for_mask(mask_path: Path) -> Path:
    """
    Filtering-stage mask artifact path.

    Uses the base stem (mask stem with `_masks` stripped) and emits:
      `{stem}_mask_filtered.tif` next to the original mask.
    """
    m = Path(mask_path)
    stem = stem_from_mask_path(m)
    return m.parent / f"{stem}_mask_filtered.tif"


def _sanitize_channel_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _base_id_stem_from_mask(mask_path: Path) -> str:
    """
    Short stem used for foci artifacts.

    Example:
      G183A-004_Phase60x_mask_filtered.tif -> G183A-004
      G183A-004_Phase60x_masks.tif         -> G183A-004
    """
    stem = stem_from_mask_path(Path(mask_path))
    low = stem.lower()
    if low.endswith("_mask_filtered"):
        stem = stem[: -len("_mask_filtered")]
    low = stem.lower()
    if low.endswith("_masks"):
        stem = stem[: -len("_masks")]
    low = stem.lower()
    if "_phase" in low:
        try:
            i = low.index("_phase")
            stem = stem[:i]
        except Exception:
            pass
    return stem.strip("_- .") or stem_from_mask_path(Path(mask_path))


def foci_channel_id_from_source(mask_path: Path, foci_source_path: Path) -> str:
    """
    Compute a stable channel id for naming foci CSV artifacts.

    Strategy:
    - Start with the foci source filename stem.
    - If it starts with the mask stem (with `_masks` stripped), remove that prefix.
    - Strip leading separators and sanitize for filenames.
    """
    src = Path(foci_source_path).stem
    # Prefer extracting the channel for common naming: <base>_foci_<channel>.tif
    m = re.match(r"^(?P<prefix>.+?)_foci_(?P<channel>.+)$", src, flags=re.IGNORECASE)
    if m:
        return _sanitize_channel_id(m.group("channel"))

    # Otherwise, strip known prefixes using both the full mask stem and the shortened base-id stem.
    base_full = stem_from_mask_path(Path(mask_path))
    base_short = _base_id_stem_from_mask(Path(mask_path))
    for base in (base_full, base_short):
        if base and src.startswith(base):
            src = src[len(base) :]
            break
    src = src.lstrip("_- .")
    return _sanitize_channel_id(src)


def foci_csv_path(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Path:
    """
    Segmentation-stage canonical foci detection CSV path.

    Naming:
      `{stem}_foci.csv` when channel id is empty, else
      `{stem}_foci_{channel}.csv`
    """
    stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    suffix = f"_{ch}" if ch else ""
    return Path(pipeline_dir) / f"{stem}_foci{suffix}.csv"


def raw_foci_csv_path(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Path:
    """Legacy segmentation-stage foci CSV path (`*_foci_raw*.csv`)."""
    stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    suffix = f"_{ch}" if ch else ""
    return Path(pipeline_dir) / f"{stem}_foci_raw{suffix}.csv"


def filtered_foci_csv_path(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> Path:
    """
    Filtering-stage filtered foci CSV path.

    Naming mirrors `raw_foci_csv_path` with `_foci_filtered`.
    """
    stem = _base_id_stem_from_mask(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    suffix = f"_{ch}" if ch else ""
    return Path(pipeline_dir) / f"{stem}_foci_filtered{suffix}.csv"


def legacy_foci_csv_candidates(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
) -> list[Path]:
    """
    Return likely legacy foci CSV paths for migration/compatibility.

    These cover older naming schemes that used the full mask stem as the prefix.
    """
    full_stem = stem_from_mask_path(Path(mask_path))
    ch = foci_channel_id_from_source(mask_path, foci_source_path)
    suffix = f"_{ch}" if ch else ""
    p = Path(pipeline_dir)
    return [
        p / f"{full_stem}_foci{suffix}.csv",
        p / f"{full_stem}_foci_raw{suffix}.csv",
        p / f"{full_stem}_foci_filtered{suffix}.csv",
    ]


def pipeline_artifacts_for_mask(
    mask_path: Path,
    *,
    pipeline_dir: Optional[Path] = None,
    foci_source_path: Optional[Path] = None,
) -> PipelineArtifacts:
    """
    Compute all stage artifact paths for a single mask.

    - `pipeline_dir` defaults to the mask's parent directory.
    - `foci_source_path` optionally yields raw/filtered CSV paths.
    """
    mask_path = Path(mask_path)
    pipeline_dir = Path(pipeline_dir) if pipeline_dir is not None else mask_path.parent
    filt = pipeline_dir / filtered_mask_path_for_mask(mask_path).name
    foci_csv = None
    raw_csv = None
    filt_csv = None
    if foci_source_path is not None:
        foci_csv = foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=Path(foci_source_path))
        raw_csv = raw_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=Path(foci_source_path))
        filt_csv = filtered_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=Path(foci_source_path))
    return PipelineArtifacts(
        pipeline_dir=pipeline_dir,
        mask_path=mask_path,
        filtered_mask_path=filt,
        foci_csv_path=foci_csv,
        raw_foci_csv_path=raw_csv,
        filtered_foci_csv_path=filt_csv,
    )


def infer_pipeline_dir_from_mask(mask_path: Path) -> Path:
    """
    Infer the pipeline directory from a mask path.

    If the mask lives in a `_segmentation_*` folder, return that folder;
    otherwise return the mask's parent directory.
    """
    m = Path(mask_path)
    if "_segmentation" in m.parent.name:
        return m.parent
    return m.parent


def infer_base_dir_from_any_mask_artifact(path: Path) -> Path:
    """
    Infer the base dir for a path that could be a mask or filtered mask.

    This reuses `base_dir_for_mask` which already handles `_segmentation_*`.
    """
    return base_dir_for_mask(Path(path))

