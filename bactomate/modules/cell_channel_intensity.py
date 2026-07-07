"""Whole-cell mean intensity on a fluorescence channel stack."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bactomate.modules.image_stack_utils import coerce_plane_to_2d


def load_mask_frames_by_index(mask_path: Path) -> dict[int, np.ndarray]:
    import tifffile

    arr = np.asarray(tifffile.imread(str(mask_path)), dtype=np.int32)
    if arr.ndim == 2:
        return {0: arr}
    return {int(i): np.asarray(arr[i], dtype=np.int32) for i in range(int(arr.shape[0]))}


def load_channel_stack_tyx(channel_path: Path) -> np.ndarray:
    import tifffile

    arr = np.asarray(tifffile.imread(str(channel_path)))
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr


def whole_cell_mean_intensity(
    mask_plane: np.ndarray,
    channel_plane: np.ndarray,
    cell_label: int,
) -> float:
    mask_plane = np.asarray(mask_plane, dtype=np.int32)
    channel_plane = np.asarray(coerce_plane_to_2d(channel_plane), dtype=np.float64)
    if channel_plane.shape != mask_plane.shape:
        return float("nan")
    cell_mask = mask_plane == int(cell_label)
    if not np.any(cell_mask):
        return float("nan")
    vals = channel_plane[cell_mask]
    return float(np.mean(vals)) if vals.size else float("nan")


def build_whole_cell_intensity_lookup(
    *,
    mask_frames: dict[int, np.ndarray],
    channel_stack: np.ndarray,
    frame_shift: int = 0,
) -> dict[tuple[int, int], float]:
    """
    Map ``(frame_1based, cell_id)`` → mean intensity inside the cell mask.

    ``frame_shift`` aligns CSV/mask frame indices with Per-Cell ``Frame`` column (same as foci integration).
    """
    lookup: dict[tuple[int, int], float] = {}
    channel_stack = np.asarray(channel_stack)
    if channel_stack.ndim == 2:
        channel_stack = channel_stack[np.newaxis, ...]
    for mask_idx, mask_plane in mask_frames.items():
        fr_1based = int(mask_idx) + 1 + int(frame_shift)
        ch_idx = max(0, min(int(mask_idx), int(channel_stack.shape[0]) - 1))
        ch_plane = channel_stack[ch_idx]
        for lab in np.unique(mask_plane):
            lab_i = int(lab)
            if lab_i <= 0:
                continue
            lookup[(fr_1based, lab_i)] = whole_cell_mean_intensity(mask_plane, ch_plane, lab_i)
    return lookup


def apply_whole_cell_intensity_columns(
    per_cell: Any,
    *,
    column_name: str,
    lookup: dict[tuple[int, int], float],
) -> Any:
    """Fill ``column_name`` on a Per-Cell Dataframe from ``lookup``."""
    import pandas as pd

    if per_cell is None or per_cell.empty:
        return per_cell
    cell_col = "Cell_ID" if "Cell_ID" in per_cell.columns else ("Cell Label" if "Cell Label" in per_cell.columns else None)
    if cell_col is None or "Frame" not in per_cell.columns:
        return per_cell
    if column_name not in per_cell.columns:
        per_cell[column_name] = np.nan

    def _val(fr: Any, cid: Any) -> float:
        try:
            fr_i = int(fr) if fr is not None and pd.notna(fr) else 0
        except Exception:
            fr_i = 0
        try:
            cid_i = int(cid) if cid is not None and pd.notna(cid) else -1
        except Exception:
            cid_i = -1
        return lookup.get((fr_i, cid_i), np.nan)

    per_cell[column_name] = [
        _val(fr, cid) for fr, cid in zip(per_cell["Frame"], per_cell[cell_col])
    ]
    return per_cell
