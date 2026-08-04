"""Shared track-metric filter logic for tracking preview and export."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
