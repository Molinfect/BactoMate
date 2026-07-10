"""
I/O and lightweight helpers for the Trackastra tracking worker.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from bactomate.modules.tiff_stack_io import read_tiff_stack

TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE = 384
LARGE_STACK_VIS_FRAME_THRESHOLD = 150

__all__ = [
    "LARGE_STACK_VIS_FRAME_THRESHOLD",
    "TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE",
    "TrackingExportVizStyle",
    "assign_cell_ids_to_tracks_df",
    "blend_path_pixel_on_frame",
    "build_excel_frame_label_to_cell_id",
    "build_track_points_by_id",
    "downsample_stacks_for_preview",
    "draw_track_centroid_marker",
    "overlay_label_for_track",
    "read_tiff_stack",
    "resolve_track_at_centroid",
    "should_draw_track_id_labels",
    "should_write_tracking_visualization",
    "tracking_export_viz_style",
]


@dataclass(frozen=True)
class TrackingExportVizStyle:
    """Resolution-aware drawing parameters for tracking TIFF exports."""

    font_size: int
    label_scale: int
    stroke_width: int
    path_bg_weight: float
    path_fg_weight: float
    dot_radius: int
    dot_outline: int
    label_offset_x: int
    label_offset_y: int
    thick_path: bool


def tracking_export_viz_style(height: int, width: int) -> TrackingExportVizStyle:
    """Pick label/path sizes scaled to the output frame (small crops need thin lines + small text)."""
    min_dim = max(1, min(int(height), int(width)))
    if min_dim <= 128:
        return TrackingExportVizStyle(
            font_size=5,
            label_scale=1,
            stroke_width=1,
            path_bg_weight=0.62,
            path_fg_weight=0.38,
            dot_radius=1,
            dot_outline=0,
            label_offset_x=2,
            label_offset_y=-2,
            thick_path=False,
        )
    if min_dim <= 256:
        return TrackingExportVizStyle(
            font_size=7,
            label_scale=1,
            stroke_width=1,
            path_bg_weight=0.58,
            path_fg_weight=0.42,
            dot_radius=2,
            dot_outline=0,
            label_offset_x=2,
            label_offset_y=-3,
            thick_path=False,
        )
    if min_dim <= 512:
        return TrackingExportVizStyle(
            font_size=9,
            label_scale=1,
            stroke_width=1,
            path_bg_weight=0.55,
            path_fg_weight=0.45,
            dot_radius=3,
            dot_outline=1,
            label_offset_x=3,
            label_offset_y=-4,
            thick_path=False,
        )
    return TrackingExportVizStyle(
        font_size=11,
        label_scale=2,
        stroke_width=2,
        path_bg_weight=0.30,
        path_fg_weight=0.70,
        dot_radius=4,
        dot_outline=1,
        label_offset_x=4,
        label_offset_y=-6,
        thick_path=True,
    )


def should_draw_track_id_labels(n_tracks: int, height: int, width: int) -> bool:
    """Skip numeric labels when a tiny crop would become unreadable."""
    if n_tracks <= 0:
        return False
    h = max(1, int(height))
    w = max(1, int(width))
    min_dim = min(h, w)
    area = h * w
    if min_dim <= 160:
        return int(n_tracks) <= max(6, area // 300)
    return int(n_tracks) <= max(12, area // 200)


def blend_path_pixel_on_frame(
    frame_rgb: np.ndarray,
    y: int,
    x: int,
    color: tuple[int, int, int],
    *,
    bg_weight: float,
    fg_weight: float,
) -> None:
    h, w = frame_rgb.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return
    frame_rgb[y, x, 0] = np.clip(frame_rgb[y, x, 0] * bg_weight + color[0] * fg_weight, 0, 255).astype(np.uint8)
    frame_rgb[y, x, 1] = np.clip(frame_rgb[y, x, 1] * bg_weight + color[1] * fg_weight, 0, 255).astype(np.uint8)
    frame_rgb[y, x, 2] = np.clip(frame_rgb[y, x, 2] * bg_weight + color[2] * fg_weight, 0, 255).astype(np.uint8)


def build_excel_frame_label_to_cell_id(excel_file: str | Path) -> dict[tuple[int, int], int]:
    """Map (0-based frame, segmentation Cell Label) -> analysis Cell_ID from the workbook."""
    from pathlib import Path as _Path

    import pandas as pd

    path = _Path(excel_file)
    if not path.is_file():
        return {}
    try:
        excel_data = pd.read_excel(path, sheet_name="Per-Cell Data")
    except Exception:
        try:
            excel_data = pd.read_excel(path, sheet_name=0)
        except Exception:
            return {}
    cell_id_col = None
    for col in ("Cell_ID", "cell_id", "Cell ID", "CellID"):
        if col in excel_data.columns:
            cell_id_col = col
            break
    if cell_id_col is None or "Cell Label" not in excel_data.columns:
        return {}
    frame_col = None
    for col in ("Frame", "frame", "t", "time", "Time"):
        if col in excel_data.columns:
            frame_col = col
            break
    if frame_col is None:
        return {}
    mapping: dict[tuple[int, int], int] = {}
    valid = excel_data[[frame_col, "Cell Label", cell_id_col]].dropna()
    for frame_val, orig_label, cell_id in zip(
        valid[frame_col].astype(int) - 1,
        valid["Cell Label"].astype(int),
        valid[cell_id_col].astype(int),
    ):
        mapping[(int(frame_val), int(orig_label))] = int(cell_id)
    return mapping


def assign_cell_ids_to_tracks_df(
    tracks_df,
    masks_input: np.ndarray | None,
    *,
    excel_file: str | Path | None = None,
    trackid_to_cellid: dict[int, int] | None = None,
):
    """Attach a per-detection Cell_ID column stable against Excel and export overlays."""
    import pandas as pd

    if tracks_df is None or len(tracks_df) == 0:
        return tracks_df
    if not hasattr(tracks_df, "columns") or "track_id" not in tracks_df.columns:
        return tracks_df
    excel_map = build_excel_frame_label_to_cell_id(excel_file) if excel_file else {}
    track_map = {int(k): int(v) for k, v in (trackid_to_cellid or {}).items()}
    out = tracks_df.copy()
    has_xy = "y" in out.columns and "x" in out.columns
    has_mask = masks_input is not None and getattr(masks_input, "ndim", 0) == 3
    cell_ids: list[int] = []
    for row in out.itertuples(index=False):
        tid = int(row.track_id)
        cell_id: int | None = None
        if has_mask and has_xy and "frame" in out.columns:
            frame_idx = int(row.frame)
            y = int(round(float(row.y)))
            x = int(round(float(row.x)))
            if 0 <= frame_idx < masks_input.shape[0]:
                h, w = masks_input.shape[1], masks_input.shape[2]
                yi = int(np.clip(y, 0, h - 1))
                xi = int(np.clip(x, 0, w - 1))
                label = int(masks_input[frame_idx, yi, xi])
                if label > 0 and excel_map:
                    cell_id = excel_map.get((frame_idx, label))
        if cell_id is None and tid in track_map:
            cell_id = track_map[tid]
        if cell_id is None:
            cell_id = tid
        cell_ids.append(int(cell_id))
    out["Cell_ID"] = cell_ids
    return out


def resolve_track_at_centroid(
    tracks_df,
    frame_idx: int,
    centroid_y: float,
    centroid_x: float,
    *,
    max_distance_px: float = 12.0,
) -> tuple[int | None, int | None]:
    """Match a mask region centroid to the nearest track row; return (track_id, cell_id_or_none)."""
    if tracks_df is None or not hasattr(tracks_df, "columns"):
        return None, None
    need = {"track_id", "frame", "y", "x"}
    if not need.issubset(tracks_df.columns):
        return None, None
    sub = tracks_df[tracks_df["frame"].astype(int) == int(frame_idx)]
    if sub.empty:
        return None, None
    ys = sub["y"].to_numpy(dtype=float, copy=False)
    xs = sub["x"].to_numpy(dtype=float, copy=False)
    dist = np.hypot(ys - float(centroid_y), xs - float(centroid_x))
    if dist.size == 0:
        return None, None
    idx = int(np.argmin(dist))
    if float(dist[idx]) > float(max_distance_px):
        return None, None
    row = sub.iloc[idx]
    track_id = int(row["track_id"])
    cell_id = None
    if "Cell_ID" in sub.columns:
        val = row["Cell_ID"]
        try:
            import pandas as pd

            if val is not None and not pd.isna(val):
                cell_id = int(val)
        except Exception:
            if val is not None:
                cell_id = int(val)
    return track_id, cell_id


def overlay_label_for_track(track_id: int | None, cell_id: int | None, region_label: int) -> str:
    """Label text for export overlays: prefer analysis Cell ID, else Trackastra track ID."""
    if cell_id is not None and (track_id is None or cell_id != track_id):
        return str(int(cell_id))
    if track_id is not None:
        return str(int(track_id))
    return str(int(region_label))


def draw_track_centroid_marker(
    frame_rgb: np.ndarray,
    y: int,
    x: int,
    color: tuple[int, int, int],
    style: TrackingExportVizStyle,
) -> None:
    """Small centroid dot (+ optional white ring) for export overlays."""
    if style.dot_radius <= 0:
        return
    h, w = frame_rgb.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return
    dot_size = int(style.dot_radius)
    outline_size = int(style.dot_outline)
    y_outline_start = max(0, y - dot_size - outline_size)
    y_outline_end = min(h, y + dot_size + outline_size + 1)
    x_outline_start = max(0, x - dot_size - outline_size)
    x_outline_end = min(w, x + dot_size + outline_size + 1)
    yy, xx = np.ogrid[y_outline_start:y_outline_end, x_outline_start:x_outline_end]
    center_y, center_x = y - y_outline_start, x - x_outline_start
    dist_sq = (yy - center_y) ** 2 + (xx - center_x) ** 2
    patch = frame_rgb[y_outline_start:y_outline_end, x_outline_start:x_outline_end]
    if outline_size > 0:
        outline_mask = (dist_sq <= (dot_size + outline_size) ** 2) & (dist_sq > dot_size ** 2)
        patch[outline_mask] = [255, 255, 255]
    dot_mask = dist_sq <= dot_size ** 2
    patch[dot_mask, 0] = color[0]
    patch[dot_mask, 1] = color[1]
    patch[dot_mask, 2] = color[2]


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
    """Disk tracking visualizations when enabled by the user (except preview runs)."""
    del mask_file
    if track_action in ("preview", "preview_lineage"):
        return False
    return bool(user_enabled)
