"""Shared calibrated track metrics and filter logic for preview and export."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_track_metrics(
    tracks_df: pd.DataFrame | None,
    pixel_size_um: float,
    time_per_frame_min: float,
) -> pd.DataFrame:
    """Compute metrics from native-pixel, zero-based track points without modifying them.

    Duration is the inclusive frame span; speed uses elapsed frame intervals,
    including gaps. A single detection has zero motion. Without positive spatial
    and temporal calibration, physical speed columns retain the px/frame fallback.
    Missing coordinates retain duration/counts with zero motion; missing required
    track_id/frame columns return an empty table with the metric schema.
    """
    cols = [
        "track_id",
        "trajectory_length",
        "duration_frames",
        "gap_frames",
        "duration_min",
        "first_frame",
        "last_frame",
        "displacement_px",
        "displacement_um",
        "path_length_px",
        "path_length_um",
        "mean_speed_px_frame",
        "mean_speed_um_s",
        "straight_line_speed_px_frame",
        "straight_line_speed_um_s",
        "straightness",
    ]
    if tracks_df is None or len(tracks_df) == 0 or "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
        return pd.DataFrame(columns=cols)
    need_xy = "y" in tracks_df.columns and "x" in tracks_df.columns
    rows = []
    for tid, g in tracks_df.groupby("track_id"):
        g = g.sort_values("frame")
        n = len(g)
        first_frame = int(g["frame"].iloc[0])
        last_frame = int(g["frame"].iloc[-1])
        duration = max(1, int(last_frame - first_frame + 1))
        elapsed_intervals = max(1, last_frame - first_frame)
        displacement_px = 0.0
        path_length_px = 0.0
        if need_xy and n >= 2:
            ys = g["y"].values
            xs = g["x"].values
            displacement_px = float(np.hypot(ys[-1] - ys[0], xs[-1] - xs[0]))
            path_length_px = float(np.sum(np.hypot(np.diff(ys), np.diff(xs))))
        displacement_um = displacement_px * pixel_size_um if pixel_size_um > 0 else displacement_px
        path_length_um = path_length_px * pixel_size_um if pixel_size_um > 0 else path_length_px
        straightness = float(displacement_px / path_length_px) if path_length_px > 0 else 0.0
        mean_speed_px_frame = path_length_px / elapsed_intervals
        duration_sec = elapsed_intervals * time_per_frame_min * 60.0 if time_per_frame_min > 0 else 0.0
        if pixel_size_um > 0 and duration_sec > 0:
            mean_speed_um_s = path_length_um / duration_sec
        else:
            mean_speed_um_s = mean_speed_px_frame
        duration_min = duration * time_per_frame_min if time_per_frame_min > 0 else 0.0
        straight_line_speed_px_frame = displacement_px / elapsed_intervals
        if duration_sec > 0 and pixel_size_um > 0:
            straight_line_speed_um_s = displacement_um / duration_sec
        else:
            straight_line_speed_um_s = straight_line_speed_px_frame
        gap_frames = max(0, int(duration) - int(n))
        rows.append(
            {
                "track_id": int(tid),
                "trajectory_length": int(n),
                "duration_frames": int(duration),
                "gap_frames": int(gap_frames),
                "duration_min": float(duration_min),
                "first_frame": int(first_frame),
                "last_frame": int(last_frame),
                "displacement_px": float(displacement_px),
                "displacement_um": float(displacement_um),
                "path_length_px": float(path_length_px),
                "path_length_um": float(path_length_um),
                "mean_speed_px_frame": float(mean_speed_px_frame),
                "mean_speed_um_s": float(mean_speed_um_s),
                "straight_line_speed_px_frame": float(straight_line_speed_px_frame),
                "straight_line_speed_um_s": float(straight_line_speed_um_s),
                "straightness": float(straightness),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def apply_track_filters(metrics_df: pd.DataFrame, params: dict[str, Any]) -> set[int]:
    """Return track_ids that pass all active filters."""
    if metrics_df.empty:
        return set()
    use_um = params.get("use_physical_units", True)
    use_speed_physical = params.get("use_speed_physical", use_um)
    disp_col = "displacement_um" if use_um else "displacement_px"
    path_col = "path_length_um" if use_um else "path_length_px"
    speed_col = "mean_speed_um_s" if use_speed_physical else "mean_speed_px_frame"
    mask = np.ones(len(metrics_df), dtype=bool)
    m = metrics_df
    if params.get("min_length") is not None:
        mask &= m["trajectory_length"] >= params["min_length"]
    if params.get("max_length") is not None:
        mask &= m["trajectory_length"] <= params["max_length"]
    if params.get("min_duration") is not None:
        mask &= m["duration_frames"] >= params["min_duration"]
    if params.get("max_duration") is not None:
        mask &= m["duration_frames"] <= params["max_duration"]
    if params.get("min_gaps") is not None:
        mask &= m["gap_frames"] >= params["min_gaps"]
    if params.get("max_gaps") is not None:
        mask &= m["gap_frames"] <= params["max_gaps"]
    if params.get("min_displacement") is not None:
        mask &= m[disp_col] >= params["min_displacement"]
    if params.get("max_displacement") is not None:
        mask &= m[disp_col] <= params["max_displacement"]
    if params.get("min_path_length") is not None:
        mask &= m[path_col] >= params["min_path_length"]
    if params.get("max_path_length") is not None:
        mask &= m[path_col] <= params["max_path_length"]
    if params.get("min_speed") is not None:
        mask &= m[speed_col] >= params["min_speed"]
    if params.get("max_speed") is not None:
        mask &= m[speed_col] <= params["max_speed"]
    if params.get("min_straightness") is not None:
        mask &= m["straightness"] >= params["min_straightness"]
    return set(metrics_df.loc[mask, "track_id"].astype(int).tolist())


def filter_params_run_info_rows(params: dict[str, Any]) -> list[tuple[str, str]]:
    """Key/value rows describing calibration and filter thresholds for Run info sheet."""
    rows: list[tuple[str, str]] = [
        ("pixel_size_um", str(params.get("pixel_size", 0.0))),
        ("time_per_frame_min", str(params.get("time_per_frame", 0.0))),
        ("use_physical_units", "yes" if params.get("use_physical_units") else "no"),
        ("use_speed_physical", "yes" if params.get("use_speed_physical") else "no"),
        ("stitch_enabled", "yes" if params.get("stitch_enabled") else "no"),
    ]
    if params.get("stitch_enabled"):
        rows.append(("stitch_max_gap_frames", str(params.get("stitch_max_gap_frames", ""))))
        rows.append(("stitch_max_jump", str(params.get("stitch_max_jump", ""))))
    for key in (
        "min_length",
        "max_length",
        "min_duration",
        "max_duration",
        "min_gaps",
        "max_gaps",
        "min_displacement",
        "max_displacement",
        "min_path_length",
        "max_path_length",
        "min_speed",
        "max_speed",
        "min_straightness",
    ):
        val = params.get(key)
        if val is not None:
            rows.append((key, str(val)))
    return rows
