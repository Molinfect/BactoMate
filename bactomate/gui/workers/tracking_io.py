"""
I/O and lightweight helpers for the Trackastra tracking worker.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from bactomate.modules.tiff_stack_io import read_tiff_stack

TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE = 384
LARGE_STACK_VIS_FRAME_THRESHOLD = 150

__all__ = [
    "LARGE_STACK_VIS_FRAME_THRESHOLD",
    "TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE",
    "build_track_points_by_id",
    "downsample_stacks_for_preview",
    "read_tiff_stack",
    "should_write_tracking_visualization",
]


def build_track_points_by_id(tracks_df) -> dict[int, list[tuple[int, float, float]]]:
    """Index track points as ``track_id -> [(frame, y, x), ...]`` sorted by frame."""
    required = ("frame", "track_id", "y", "x")
    if tracks_df is None or not all(col in tracks_df.columns for col in required):
        return {}
    out: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    frames = tracks_df["frame"].to_numpy(dtype=int, copy=False)
    track_ids = tracks_df["track_id"].to_numpy(dtype=int, copy=False)
    ys = tracks_df["y"].to_numpy(dtype=float, copy=False)
    xs = tracks_df["x"].to_numpy(dtype=float, copy=False)
    for frame, tid, y, x in zip(frames, track_ids, ys, xs):
        out[int(tid)].append((int(frame), float(y), float(x)))
    for tid in out:
        out[tid].sort(key=lambda p: p[0])
    return dict(out)


def downsample_stacks_for_preview(
    masks: np.ndarray,
    imgs: np.ndarray,
    *,
    max_size: int = TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample (T,Y,X) stacks for fast in-memory preview tracking."""
    if masks.ndim != 3 or imgs.ndim != 3:
        return masks, imgs
    _t, h, w = masks.shape
    if max(h, w) <= max_size:
        return masks, imgs
    scale = max_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    sy = max(1, h // new_h)
    sx = max(1, w // new_w)
    return masks[:, ::sy, ::sx], imgs[:, ::sy, ::sx]


def should_write_tracking_visualization(
    mask_file,
    *,
    user_enabled: bool = True,
    track_action: str,
) -> bool:
    """Disk tracking visualizations are always written (except preview runs)."""
    del mask_file, user_enabled
    if track_action in ("preview", "preview_lineage"):
        return False
    return True
