"""Constants and candidate helpers for folder preview grids."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

FOLDER_PREVIEW_MAX_FILES = 48


def folder_preview_enabled(n_candidates: int) -> bool:
    """True when more than one previewable file exists in the source location."""
    return int(n_candidates) > 1


def cap_folder_preview_paths(
    paths: Sequence[Path],
    *,
    max_files: int = FOLDER_PREVIEW_MAX_FILES,
) -> tuple[list[Path], int]:
    """Return (capped_paths, total_before_cap).

    Preserves input order. ``total_before_cap`` is the uncapped length.
    """
    items = [Path(p) for p in paths]
    total = len(items)
    limit = max(1, int(max_files))
    return items[:limit], total
