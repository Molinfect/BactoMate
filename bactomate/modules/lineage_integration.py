"""
Lineage integration: join Trackastra outputs with Analyze workbooks, build tables, QC, reports.

See docs/LINEAGE_ANALYSIS_JOIN_CONTRACT.md for the stable join contract.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

ANALYSIS_FRAME_BASE = 1
TRACKING_FRAME_BASE = 0
LINEAGE_SCHEMA_VERSION = 2
MASTER_LINEAGE_SUMMARY_NAME = "@summary-lineage.xlsx"
MASTER_LINEAGE_SHEET = "@summary-lineage"

LINK_STATUS_CANDIDATE = "candidate"
REVIEW_STATUS_CANDIDATE = "candidate"
REVIEW_STATUS_ACCEPTED = "accepted"
REVIEW_STATUS_REJECTED = "rejected"
REVIEW_STATUS_FLAGGED = "flagged"
VALID_REVIEW_STATUSES = (
    REVIEW_STATUS_CANDIDATE,
    REVIEW_STATUS_ACCEPTED,
    REVIEW_STATUS_REJECTED,
    REVIEW_STATUS_FLAGGED,
)


@dataclass
class LineagePreflightResult:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    mask_path: Optional[Path] = None
    resolved_mask_path: Optional[Path] = None
    analysis_workbook: Optional[Path] = None
    original_image: Optional[Path] = None
    n_frames: Optional[int] = None
    mask_is_filtered: bool = False
    linker_mode: str = ""
    model_name: str = ""
    fi_channels: list[str] = field(default_factory=list)
    foci_channels: list[str] = field(default_factory=list)
    pixel_size_um: float = 0.0
    time_per_frame_min: float = 0.0


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def resolve_lineage_mask_path(mask_path: Path, *, prefer_filtered: bool = True) -> Path:
    """Prefer filtered mask sibling when user selected raw *_masks.tif."""
    mask_path = Path(mask_path)
    if not prefer_filtered:
        return mask_path
    name_lower = mask_path.name.lower()
    if "_mask_filtered" in name_lower:
        return mask_path
    try:
        from bactomate.gui.artifact_paths import filtered_mask_path_for_mask, resolve_filtered_mask_path

        if name_lower.endswith(("_masks.tif", "_masks.tiff")):
            resolved = resolve_filtered_mask_path(mask_path)
            if resolved is not None and Path(resolved).is_file():
                return Path(resolved)
        filt = filtered_mask_path_for_mask(mask_path)
        if filt.is_file():
            return Path(filt)
    except Exception:
        pass
    return mask_path


def mask_filtered_status(mask_path: Path) -> bool:
    return "_mask_filtered" in Path(mask_path).name.lower()


def discover_lineage_input_channels(mask_path: Path) -> tuple[list[str], list[str]]:
    """Discover FI and foci TIFF siblings near the resolved mask."""
    from bactomate.gui.analysis_paths import base_dir_for_mask

    mask_path = Path(mask_path)
    fi_channels: list[str] = []
    foci_channels: list[str] = []
    search_dirs: list[Path] = []
    try:
        search_dirs.append(base_dir_for_mask(mask_path))
    except Exception:
        pass
    for d in (mask_path.parent, mask_path.parent.parent):
        if d.is_dir() and d not in search_dirs:
            search_dirs.append(d)
    for directory in search_dirs:
        for p in sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff")):
            name = p.name.lower()
            if "_fi_foci_" in name or "_foci_" in name:
                if p.name not in foci_channels:
                    foci_channels.append(p.name)
            elif "_fi_" in name and "_foci_" not in name:
                if p.name not in fi_channels:
                    fi_channels.append(p.name)
    return fi_channels, foci_channels


def preflight_lineage_run(
    mask_path: Path,
    *,
    linker_mode: str,
    model_name: str,
    prefer_filtered_mask: bool = True,
    require_analysis_workbook: bool = False,
) -> LineagePreflightResult:
    """Validate mask, analysis workbook, linker, and model before a lineage job."""
    from bactomate.gui.analysis_paths import resolve_analysis_workbook_for_mask
    from bactomate.gui.trackastra_utils import validate_trackastra_model_choice
    from bactomate.modules.tiff_metadata import resolve_original_for_mask

    out = LineagePreflightResult(
        ok=True,
        mask_path=Path(mask_path),
        linker_mode=str(linker_mode or ""),
        model_name=str(model_name or ""),
    )
    raw = Path(mask_path)
    if not raw.is_file():
        out.ok = False
        out.errors.append(f"Mask file not found: {raw}")
        return out

    resolved = resolve_lineage_mask_path(raw, prefer_filtered=prefer_filtered_mask)
    out.resolved_mask_path = resolved
    out.mask_is_filtered = mask_filtered_status(resolved)

    if prefer_filtered_mask and not out.mask_is_filtered and "_masks" in raw.name.lower():
        filt = resolve_lineage_mask_path(raw, prefer_filtered=True)
        if filt != raw and filt.is_file():
            out.warnings.append(
                f"No filtered mask found; using raw mask {raw.name}. "
                "Run Filtering first and analyze the filtered mask for consistent joins."
            )

    wb = resolve_analysis_workbook_for_mask(resolved)
    out.analysis_workbook = wb
    if require_analysis_workbook and wb is None:
        out.ok = False
        out.errors.append(
            "No analysis workbook found under _analysis/. Run Analyze on this mask first."
        )
    elif wb is None:
        out.warnings.append(
            "No analysis workbook found; lineage will run without Per-Cell FI/foci joins."
        )
    elif raw.resolve() != resolved.resolve():
        out.warnings.append(
            f"Analyze workbook matches filtered mask ({resolved.name}); "
            f"selected path was {raw.name}."
        )

    mode = (linker_mode or "").strip().lower()
    if mode == "greedy_nodiv":
        out.warnings.append(
            "greedy_nodiv cannot represent divisions; lineage runs will use greedy linking."
        )
    elif mode not in ("greedy", "ilp", "greedy_nodiv"):
        out.warnings.append(f"Unknown linking mode '{linker_mode}'; Trackastra may reject it.")

    if mode == "ilp":
        try:
            import motile  # noqa: F401
        except ImportError:
            out.warnings.append(
                "ILP mode may require motile and a solver (Gurobi/SCIP). "
                "If the run fails, switch to greedy."
            )

    ok_model, model_msg = validate_trackastra_model_choice(model_name)
    if not ok_model:
        out.ok = False
        out.errors.append(model_msg or f"Trackastra model unavailable: {model_name}")

    try:
        orig = resolve_original_for_mask(resolved)
        if orig is not None and Path(orig).is_file():
            out.original_image = Path(orig)
    except Exception:
        pass

    try:
        import tifffile

        with tifffile.TiffFile(str(resolved)) as tf:
            out.n_frames = len(tf.pages) if len(tf.pages) > 1 else 1
            if len(tf.pages) == 1:
                arr = tf.asarray()
                if arr.ndim == 3:
                    out.n_frames = int(arr.shape[0])
    except Exception:
        out.warnings.append("Could not read mask frame count from TIFF header.")

    try:
        from bactomate.modules.tiff_metadata import resolve_calibration_for_mask

        ps, tp, _, _ = resolve_calibration_for_mask(resolved)
        out.pixel_size_um = float(ps or 0.0)
        out.time_per_frame_min = float(tp or 0.0)
    except Exception:
        pass

    fi, foci = discover_lineage_input_channels(resolved)
    out.fi_channels = fi
    out.foci_channels = foci
    if not fi and not foci and wb is not None:
        out.warnings.append(
            "No FI/foci TIFF siblings found beside the mask; lineage will still run but "
            "channel overlays depend on Analyze workbook columns."
        )

    return out


def preflight_summary_lines(pf: LineagePreflightResult) -> list[str]:
    """Human-readable preflight summary for dialogs."""
    lines: list[str] = []
    if pf.resolved_mask_path:
        lines.append(f"Mask: {pf.resolved_mask_path.name}")
        lines.append(f"Filtered mask: {'Yes' if pf.mask_is_filtered else 'No'}")
    if pf.n_frames is not None:
        lines.append(f"Frames: {pf.n_frames}")
    if pf.analysis_workbook:
        lines.append(f"Analysis workbook: {pf.analysis_workbook.name}")
    else:
        lines.append("Analysis workbook: not found")
    if pf.original_image:
        lines.append(f"Phase image: {pf.original_image.name}")
    if pf.fi_channels:
        lines.append(f"FI channels: {', '.join(pf.fi_channels)}")
    if pf.foci_channels:
        lines.append(f"Foci channels: {', '.join(pf.foci_channels)}")
    if pf.pixel_size_um > 0:
        lines.append(f"Pixel size: {pf.pixel_size_um:.4f} µm/px")
    if pf.time_per_frame_min > 0:
        lines.append(f"Time / frame: {pf.time_per_frame_min:.6g} min")
    lines.append(f"Trackastra model: {pf.model_name or '—'}")
    lines.append(f"Linking mode: {pf.linker_mode or '—'}")
    return lines


def lineage_edges_from_napari_graph(napari_graph: Any, tracks_df: Any) -> Any:
    """Napari convention: graph[child_track_id] = [parent_track_id, ...]."""
    import pandas as pd

    cols = ["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
    empty = pd.DataFrame(columns=cols)
    if napari_graph is None or tracks_df is None or len(tracks_df) == 0:
        return empty
    if "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
        return empty
    first_frame = tracks_df.groupby("track_id")["frame"].min()
    rows = []
    seen: set[tuple[int, int]] = set()
    try:
        items = list(napari_graph.items())
    except Exception:
        items = []
    for child_raw, parents in items:
        try:
            child = int(child_raw)
        except (TypeError, ValueError):
            continue
        if parents is None:
            continue
        if isinstance(parents, (int, np.integer)):
            plist = [int(parents)]
        else:
            try:
                plist = [int(p) for p in parents if p is not None]
            except (TypeError, ValueError):
                continue
        for p in plist:
            key = (p, child)
            if key in seen:
                continue
            seen.add(key)
            if child not in first_frame.index:
                continue
            div_f = int(first_frame.loc[child])
            pf = tracks_df.loc[tracks_df["track_id"] == p, "frame"]
            before = pf[pf < div_f]
            parent_last = int(before.max()) if len(before) > 0 else (int(pf.max()) if len(pf) > 0 else "")
            rows.append(
                {
                    "parent_track_id": p,
                    "daughter_track_id": child,
                    "division_frame": div_f,
                    "parent_last_frame": parent_last,
                    "link_status": "candidate",
                }
            )
    return pd.DataFrame(rows, columns=cols + ["link_status"]) if rows else empty


def lineage_founder_map(edges_df: Any, tracks_df: Any) -> dict[int, int]:
    parent_of: dict[int, int] = {}
    if edges_df is not None and len(edges_df) > 0:
        for _, row in edges_df.iterrows():
            parent_of[int(row["daughter_track_id"])] = int(row["parent_track_id"])
    cache: dict[int, int] = {}

    def founder(tid: int) -> int:
        if tid in cache:
            return cache[tid]
        p = parent_of.get(tid)
        if p is None:
            cache[tid] = tid
            return tid
        f = founder(p)
        cache[tid] = f
        return f

    out: dict[int, int] = {}
    if tracks_df is not None and "track_id" in tracks_df.columns:
        for t in tracks_df["track_id"].unique():
            out[int(t)] = founder(int(t))
    return out


def track_lineage_metadata(edges_df: Any, tracks_df: Any) -> Any:
    """Per-track generation, branch depth, parent, daughters, founder."""
    import pandas as pd

    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = defaultdict(list)
    if edges_df is not None and len(edges_df) > 0:
        for _, row in edges_df.iterrows():
            p, d = int(row["parent_track_id"]), int(row["daughter_track_id"])
            parent_of[d] = p
            children_of[p].append(d)

    def subtree_depth(tid: int) -> int:
        ch = children_of.get(tid, [])
        if not ch:
            return 0
        return 1 + max(subtree_depth(c) for c in ch)

    def generation(tid: int) -> int:
        g = 0
        t = tid
        while t in parent_of:
            t = parent_of[t]
            g += 1
        return g

    founder_map = lineage_founder_map(edges_df, tracks_df)
    rows = []
    if tracks_df is not None and "track_id" in tracks_df.columns:
        for tid in sorted(tracks_df["track_id"].unique()):
            tid = int(tid)
            cell_id = None
            if "Cell_ID" in tracks_df.columns:
                sub = tracks_df.loc[tracks_df["track_id"] == tid, "Cell_ID"]
                if len(sub) > 0:
                    v = sub.iloc[0]
                    cell_id = None if pd.isna(v) else v
            rows.append(
                {
                    "track_id": tid,
                    "lineage_id": founder_map.get(tid, tid),
                    "founder_track_id": founder_map.get(tid, tid),
                    "parent_track_id": parent_of.get(tid),
                    "daughter_track_ids": json.dumps(sorted(children_of.get(tid, []))),
                    "n_daughters": len(children_of.get(tid, [])),
                    "generation": generation(tid),
                    "branch_depth": subtree_depth(tid),
                    "Cell_ID": cell_id,
                }
            )
    return pd.DataFrame(rows)


def load_per_cell_table(excel_path: Path) -> Any:
    import pandas as pd

    path = Path(excel_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return pd.read_excel(path, sheet_name="Per-Cell Data")
    except Exception:
        return pd.read_excel(path, sheet_name=0)


def join_analysis_to_lineage_points(
    lineage_points_df: Any,
    per_cell_df: Any,
    *,
    analysis_frame_base: int = ANALYSIS_FRAME_BASE,
) -> Any:
    """Left-join Per-Cell measurements onto lineage_points by frame + Cell_ID."""
    import pandas as pd

    if lineage_points_df is None or len(lineage_points_df) == 0:
        return lineage_points_df
    if per_cell_df is None or len(per_cell_df) == 0:
        return lineage_points_df
    pts = lineage_points_df.copy()
    pc = per_cell_df.copy()
    frame_col = next((c for c in ("Frame", "frame") if c in pc.columns), None)
    cell_col = next((c for c in ("Cell_ID", "cell_id", "CellID") if c in pc.columns), None)
    if frame_col is None or cell_col is None or "Cell_ID" not in pts.columns or "frame" not in pts.columns:
        return pts
    measure_cols = [
        c
        for c in pc.columns
        if c not in (frame_col, cell_col, "Cell Label") and not str(c).startswith("Unnamed")
    ]
    pc_merge = pc[[frame_col, cell_col] + measure_cols].copy()
    pc_merge["frame"] = pc_merge[frame_col].astype(int) - int(analysis_frame_base) + int(TRACKING_FRAME_BASE)
    if cell_col != "Cell_ID":
        pc_merge["Cell_ID"] = pc_merge[cell_col].astype(int)
    else:
        pc_merge["Cell_ID"] = pc_merge["Cell_ID"].astype(int)
    drop_src = [c for c in (frame_col, cell_col) if c not in ("frame", "Cell_ID")]
    pc_merge = pc_merge.drop(columns=drop_src, errors="ignore")
    pc_merge = pc_merge.drop_duplicates(subset=["frame", "Cell_ID"], keep="first")
    pts["Cell_ID"] = pts["Cell_ID"].astype(int)
    return pts.merge(pc_merge, on=["frame", "Cell_ID"], how="left")


def _track_point_lookup(tracks_df: Any) -> dict[tuple[int, int], tuple[float, float]]:
    out: dict[tuple[int, int], tuple[float, float]] = {}
    if tracks_df is None or len(tracks_df) == 0:
        return out
    if "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
        return out
    for r in tracks_df.itertuples(index=False):
        tid = _safe_int(getattr(r, "track_id", None))
        fr = _safe_int(getattr(r, "frame", None))
        y = _safe_float(getattr(r, "y", None))
        x = _safe_float(getattr(r, "x", None))
        if tid is not None and fr is not None and y is not None and x is not None:
            out[(tid, fr)] = (y, x)
    return out


def build_lineage_events(
    edges_df: Any,
    tracks_df: Any,
    area_df: Any = None,
    *,
    time_per_frame_min: float = 0.0,
    lineage_tracks_meta: Any = None,
    pixel_size_um: float = 0.0,
) -> Any:
    """One row per candidate division with mother/daughter size snapshots."""
    import pandas as pd

    cols = [
        "parent_track_id",
        "daughter_track_id",
        "division_frame",
        "division_frame_1based",
        "division_time_min",
        "parent_last_frame",
        "lineage_id",
        "generation",
        "branch_depth",
        "mother_area_before",
        "daughter_area_after",
        "area_asymmetry",
        "inter_division_interval_frames",
        "division_distance_px",
        "division_distance_um",
        "mother_y",
        "mother_x",
        "daughter_y",
        "daughter_x",
        "link_status",
        "review_status",
    ]
    if edges_df is None or len(edges_df) == 0:
        return pd.DataFrame(columns=cols)
    founder = lineage_founder_map(edges_df, tracks_df)
    meta_by_tid: dict[int, dict[str, Any]] = {}
    if lineage_tracks_meta is not None and hasattr(lineage_tracks_meta, "columns"):
        for r in lineage_tracks_meta.itertuples(index=False):
            tid = _safe_int(getattr(r, "track_id", None))
            if tid is not None:
                meta_by_tid[tid] = {
                    "generation": _safe_int(getattr(r, "generation", None)),
                    "branch_depth": _safe_int(getattr(r, "branch_depth", None)),
                }
    area_by_track_frame: dict[tuple[int, int], float] = {}
    if area_df is not None and len(area_df) > 0 and "track_id" in area_df.columns and "frame" in area_df.columns:
        for r in area_df.itertuples(index=False):
            tid = _safe_int(getattr(r, "track_id", None))
            fr = _safe_int(getattr(r, "frame", None))
            ar = _safe_float(getattr(r, "area", None))
            if tid is not None and fr is not None and ar is not None and ar > 0:
                area_by_track_frame[(tid, fr)] = ar
    point_lookup = _track_point_lookup(tracks_df)
    prior_div_by_parent: dict[int, list[int]] = defaultdict(list)
    rows = []
    for er in edges_df.itertuples(index=False):
        p = _safe_int(getattr(er, "parent_track_id", None))
        d = _safe_int(getattr(er, "daughter_track_id", None))
        div_f = _safe_int(getattr(er, "division_frame", None))
        plf = _safe_int(getattr(er, "parent_last_frame", None))
        if p is None or d is None or div_f is None:
            continue
        lid = founder.get(p) or founder.get(d)
        plf_use = plf if plf is not None else max(0, div_f - 1)
        mother_a = area_by_track_frame.get((p, plf_use))
        daughter_a = area_by_track_frame.get((d, div_f))
        asym = np.nan
        if mother_a and daughter_a and mother_a > 0:
            asym = float(daughter_a) / float(mother_a)
        mother_pt = point_lookup.get((p, plf_use)) or point_lookup.get((p, div_f))
        daughter_pt = point_lookup.get((d, div_f))
        dist_px = np.nan
        if mother_pt and daughter_pt:
            dist_px = float(np.hypot(daughter_pt[0] - mother_pt[0], daughter_pt[1] - mother_pt[1]))
        dist_um = dist_px * pixel_size_um if pixel_size_um > 0 and not np.isnan(dist_px) else np.nan
        interval = None
        if p in prior_div_by_parent and prior_div_by_parent[p]:
            interval = int(div_f - max(prior_div_by_parent[p]))
        prior_div_by_parent[p].append(div_f)
        d_meta = meta_by_tid.get(d, {})
        rows.append(
            {
                "parent_track_id": p,
                "daughter_track_id": d,
                "division_frame": div_f,
                "division_frame_1based": int(div_f) + int(ANALYSIS_FRAME_BASE) - int(TRACKING_FRAME_BASE),
                "division_time_min": float(div_f) * time_per_frame_min if time_per_frame_min > 0 else np.nan,
                "parent_last_frame": plf,
                "lineage_id": lid,
                "generation": d_meta.get("generation"),
                "branch_depth": d_meta.get("branch_depth"),
                "mother_area_before": mother_a,
                "daughter_area_after": daughter_a,
                "area_asymmetry": asym,
                "inter_division_interval_frames": interval,
                "division_distance_px": dist_px,
                "division_distance_um": dist_um,
                "mother_y": mother_pt[0] if mother_pt else np.nan,
                "mother_x": mother_pt[1] if mother_pt else np.nan,
                "daughter_y": daughter_pt[0] if daughter_pt else np.nan,
                "daughter_x": daughter_pt[1] if daughter_pt else np.nan,
                "link_status": getattr(er, "link_status", LINK_STATUS_CANDIDATE)
                if hasattr(er, "link_status")
                else LINK_STATUS_CANDIDATE,
                "review_status": getattr(er, "review_status", REVIEW_STATUS_CANDIDATE)
                if hasattr(er, "review_status")
                else REVIEW_STATUS_CANDIDATE,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def build_lineage_qc(
    edges_df: Any,
    tracks_df: Any,
    lineage_tracks_df: Any,
    *,
    lineage_events_df: Any = None,
    lineage_points_df: Any = None,
    max_daughter_distance_px: float = 80.0,
    area_asymmetry_min: float = 0.15,
    area_asymmetry_max: float = 0.85,
) -> Any:
    """QC flags for suspicious lineage links."""
    import pandas as pd

    qc_cols = ["flag_type", "track_id", "related_track_id", "frame", "severity", "message"]
    rows: list[dict[str, Any]] = []
    if tracks_df is None or len(tracks_df) == 0:
        return pd.DataFrame(columns=qc_cols)
    daughters_per_parent: dict[int, list[int]] = defaultdict(list)
    if edges_df is not None and len(edges_df) > 0:
        for er in edges_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            if p is not None and d is not None:
                daughters_per_parent[p].append(d)
    for p, dlist in daughters_per_parent.items():
        if len(dlist) > 2:
            rows.append(
                {
                    "flag_type": "many_daughters",
                    "track_id": p,
                    "related_track_id": ",".join(str(x) for x in dlist),
                    "frame": None,
                    "severity": "warning",
                    "message": f"Parent track {p} has {len(dlist)} candidate daughters.",
                }
            )
    if "track_id" in tracks_df.columns and "frame" in tracks_df.columns:
        for tid, g in tracks_df.groupby("track_id"):
            frames = sorted(g["frame"].astype(int).unique().tolist())
            if len(frames) >= 2:
                gaps = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
                if any(gap > 1 for gap in gaps):
                    rows.append(
                        {
                            "flag_type": "track_gap",
                            "track_id": int(tid),
                            "related_track_id": None,
                            "frame": None,
                            "severity": "info",
                            "message": f"Track {tid} has frame gap(s) in detection.",
                        }
                    )
    if lineage_events_df is not None and hasattr(lineage_events_df, "columns") and len(lineage_events_df) > 0:
        for er in lineage_events_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            div_f = _safe_int(getattr(er, "division_frame", None))
            dist = _safe_float(getattr(er, "division_distance_px", None))
            asym = _safe_float(getattr(er, "area_asymmetry", None))
            if dist is not None and dist > max_daughter_distance_px:
                rows.append(
                    {
                        "flag_type": "daughter_far_from_mother",
                        "track_id": p,
                        "related_track_id": str(d) if d is not None else None,
                        "frame": div_f,
                        "severity": "warning",
                        "message": f"Division {p}->{d} at frame {div_f}: daughter centroid {dist:.1f} px from mother.",
                    }
                )
            if asym is not None and (asym < area_asymmetry_min or asym > area_asymmetry_max):
                rows.append(
                    {
                        "flag_type": "area_asymmetry_extreme",
                        "track_id": p,
                        "related_track_id": str(d) if d is not None else None,
                        "frame": div_f,
                        "severity": "warning",
                        "message": f"Division {p}->{d}: area ratio {asym:.2f} outside typical range.",
                    }
                )
    if lineage_points_df is not None and hasattr(lineage_points_df, "columns") and len(lineage_points_df) > 0:
        if "Cell_ID" in lineage_points_df.columns:
            measure_cols = [
                c
                for c in lineage_points_df.columns
                if c not in ("track_id", "frame", "y", "x", "lineage_id", "Cell_ID", "cell_id")
            ]
            if measure_cols:
                joined = lineage_points_df.dropna(subset=["Cell_ID"])
                missing = lineage_points_df[lineage_points_df["Cell_ID"].isna()]
                if len(missing) > 0 and "track_id" in missing.columns:
                    for tid in sorted(missing["track_id"].dropna().unique().tolist())[:20]:
                        rows.append(
                            {
                                "flag_type": "missing_analyze_join",
                                "track_id": int(tid),
                                "related_track_id": None,
                                "frame": None,
                                "severity": "info",
                                "message": f"Track {tid} has points without Per-Cell join.",
                            }
                        )
    return pd.DataFrame(rows, columns=qc_cols) if rows else pd.DataFrame(columns=qc_cols)


def build_lineage_growth_timeseries(
    area_df: Any,
    founder_by_track: dict[int, int],
    *,
    time_per_frame_min: float = 0.0,
) -> Any:
    """Per-lineage summed area over time for growth-curve plotting."""
    import pandas as pd

    cols = ["lineage_id", "frame", "frame_1based", "time_min", "total_area", "n_tracks", "metric_used"]
    if area_df is None or not hasattr(area_df, "columns") or len(area_df) == 0:
        return pd.DataFrame(columns=cols)
    if "track_id" not in area_df.columns or "frame" not in area_df.columns or "area" not in area_df.columns:
        return pd.DataFrame(columns=cols)
    df = area_df.copy()
    df["lineage_id"] = df["track_id"].map(lambda t: founder_by_track.get(int(t), int(t)) if _safe_int(t) is not None else None)
    df = df.dropna(subset=["lineage_id", "frame", "area"])
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    metric = "mask_px"
    if "area_source" in df.columns:
        sources = sorted(set(df["area_source"].astype(str).tolist()))
        analysis_sources = [s for s in sources if s.startswith("analysis:")]
        if analysis_sources:
            metric = analysis_sources[0]
    rows = []
    for (lid, fr), g in df.groupby(["lineage_id", "frame"]):
        rows.append(
            {
                "lineage_id": int(lid),
                "frame": int(fr),
                "frame_1based": int(fr) + int(ANALYSIS_FRAME_BASE) - int(TRACKING_FRAME_BASE),
                "time_min": float(fr) * time_per_frame_min if time_per_frame_min > 0 else np.nan,
                "total_area": float(g["area"].sum()),
                "n_tracks": int(g["track_id"].nunique()),
                "metric_used": metric,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["lineage_id", "frame"])


def apply_event_review_status(
    events_df: Any,
    *,
    parent_track_id: int,
    daughter_track_id: int,
    review_status: str,
) -> Any:
    """Update review_status for one division event (in-memory or before export)."""
    import pandas as pd

    if events_df is None or not hasattr(events_df, "columns"):
        return events_df
    status = str(review_status or REVIEW_STATUS_CANDIDATE).strip().lower()
    if status not in VALID_REVIEW_STATUSES:
        status = REVIEW_STATUS_FLAGGED
    out = events_df.copy()
    if "review_status" not in out.columns:
        out["review_status"] = REVIEW_STATUS_CANDIDATE
    mask = (out["parent_track_id"] == int(parent_track_id)) & (out["daughter_track_id"] == int(daughter_track_id))
    out.loc[mask, "review_status"] = status
    return out


def enrich_lineage_tables(
    base_tables: dict[str, Any],
    *,
    tracks_df: Any,
    edges_df: Any,
    area_df: Any = None,
    per_cell_df: Any = None,
    time_per_frame_min: float = 0.0,
    pixel_size_um: float = 0.0,
) -> dict[str, Any]:
    """Add events, QC, analysis joins, and track metadata to core lineage tables from TrackingWorker."""
    import pandas as pd

    from bactomate.modules.lineage_viz import build_lineage_tree_layout

    out = dict(base_tables)
    founder = lineage_founder_map(edges_df, tracks_df)
    lineage_tracks_meta = track_lineage_metadata(edges_df, tracks_df)
    lineage_points = out.get("lineage_points")
    if per_cell_df is not None and lineage_points is not None and len(lineage_points) > 0:
        lineage_points = join_analysis_to_lineage_points(lineage_points, per_cell_df)
    lineage_events = build_lineage_events(
        edges_df,
        tracks_df,
        area_df=area_df,
        time_per_frame_min=time_per_frame_min,
        lineage_tracks_meta=lineage_tracks_meta,
        pixel_size_um=pixel_size_um,
    )
    lineage_qc = build_lineage_qc(
        edges_df,
        tracks_df,
        lineage_tracks_meta,
        lineage_events_df=lineage_events,
        lineage_points_df=lineage_points,
    )
    lineage_growth_timeseries = build_lineage_growth_timeseries(
        area_df, founder, time_per_frame_min=time_per_frame_min
    )
    tree_nodes, tree_edges = build_lineage_tree_layout(edges_df, lineage_tracks_meta)
    measure_cols = []
    if lineage_points is not None and len(lineage_points) > 0:
        skip = {"track_id", "frame", "y", "x", "lineage_id", "Cell_ID", "cell_id"}
        measure_cols = [c for c in lineage_points.columns if c not in skip]
    lineage_measurements = pd.DataFrame()
    if measure_cols and lineage_points is not None:
        keep = ["lineage_id", "track_id", "frame", "Cell_ID"] + measure_cols
        keep = [c for c in keep if c in lineage_points.columns]
        lineage_measurements = lineage_points[keep].copy()
        if "frame" in lineage_measurements.columns:
            lineage_measurements["frame_1based"] = (
                lineage_measurements["frame"].astype(int) + ANALYSIS_FRAME_BASE - TRACKING_FRAME_BASE
            )
    out["lineage_tracks_meta"] = lineage_tracks_meta
    out["lineage_points"] = lineage_points
    out["lineage_events"] = lineage_events
    out["lineage_qc"] = lineage_qc
    out["lineage_measurements"] = lineage_measurements
    out["lineage_growth_timeseries"] = lineage_growth_timeseries
    out["lineage_tree_nodes"] = tree_nodes
    out["lineage_tree_edges"] = tree_edges
    out["founder_by_track"] = founder
    return out


LINEAGE_CANONICAL_TABLE_KEYS = (
    "lineage_summary",
    "lineage_growth",
    "lineage_growth_timeseries",
    "lineage_edges",
    "lineage_tracks",
    "lineage_cells",
    "lineage_points",
    "lineage_events",
    "lineage_qc",
    "lineage_measurements",
    "lineage_tracks_meta",
    "lineage_tree_nodes",
    "lineage_tree_edges",
)


def canonical_lineage_tables(tables: dict[str, Any]) -> dict[str, Any]:
    """Subset for tables/ directory writes (excludes founder_by_track dict)."""
    return {k: tables[k] for k in LINEAGE_CANONICAL_TABLE_KEYS if k in tables and tables[k] is not None}


def lineage_run_info_rows(
    *,
    mask_path: Path,
    analysis_workbook: Optional[Path],
    linker_mode: str,
    model_name: str,
    preflight: Optional[LineagePreflightResult] = None,
) -> list[tuple[str, str]]:
    rows = [
        ("Schema version", str(LINEAGE_SCHEMA_VERSION)),
        ("Mask file", str(Path(mask_path).resolve())),
        ("Analysis workbook", str(analysis_workbook.resolve()) if analysis_workbook else "—"),
        ("Analysis frame base", str(ANALYSIS_FRAME_BASE)),
        ("Tracking frame base", str(TRACKING_FRAME_BASE)),
        ("Trackastra model", model_name),
        ("Linking mode", linker_mode),
        ("Division links", "candidate (Trackastra); verify in lineage QC / inspector"),
        ("Review statuses", "candidate | accepted | rejected | flagged"),
    ]
    if preflight is not None:
        if preflight.resolved_mask_path and preflight.resolved_mask_path != Path(mask_path):
            rows.append(("Resolved mask", str(preflight.resolved_mask_path.resolve())))
        rows.append(("Mask is filtered", "Yes" if preflight.mask_is_filtered else "No"))
        if preflight.n_frames is not None:
            rows.append(("Frames", str(preflight.n_frames)))
        if preflight.fi_channels:
            rows.append(("FI channels", ", ".join(preflight.fi_channels)))
        if preflight.foci_channels:
            rows.append(("Foci channels", ", ".join(preflight.foci_channels)))
        if preflight.pixel_size_um > 0:
            rows.append(("Pixel size (µm/px)", f"{preflight.pixel_size_um:.6g}"))
        if preflight.time_per_frame_min > 0:
            rows.append(("Time / frame (min)", f"{preflight.time_per_frame_min:.6g}"))
        for w in preflight.warnings:
            rows.append(("Warning", w))
    return rows


def write_lineage_report_workbook(
    output_path: Path,
    tables: dict[str, Any],
    *,
    run_info_rows: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Write {stem}_lineage_report.xlsx with integrated sheets."""
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheets: list[tuple[str, Any]] = []
    if run_info_rows:
        sheets.append(("Run Info", pd.DataFrame(run_info_rows, columns=["Setting", "Value"])))
    order = [
        ("Lineage Summary", "lineage_summary"),
        ("Division Events", "lineage_events"),
        ("Track Lineage", "lineage_tracks_meta"),
        ("Growth Timeseries", "lineage_growth_timeseries"),
        ("Lineage Points", "lineage_points"),
        ("Growth", "lineage_growth"),
        ("QC Flags", "lineage_qc"),
        ("Edges", "lineage_edges"),
        ("Measurements", "lineage_measurements"),
        ("Tree Nodes", "lineage_tree_nodes"),
    ]
    for sheet_name, key in order:
        df = tables.get(key)
        if df is not None and hasattr(df, "columns") and len(df.columns) > 0:
            sheets.append((sheet_name, df))
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets:
            safe = name[:31]
            df.to_excel(writer, sheet_name=safe, index=False)


def write_batch_summary_lineage(
    output_path: Path,
    lineage_summary_frames: list[Any],
) -> None:
    """Stack lineage_summary tables into @summary-lineage.xlsx."""
    import pandas as pd

    if not lineage_summary_frames:
        return
    parts = []
    for item in lineage_summary_frames:
        if isinstance(item, tuple) and len(item) >= 2:
            df, sample = item[0], item[1]
        else:
            df, sample = item, ""
        if df is None or not hasattr(df, "columns") or len(df) == 0:
            continue
        chunk = df.copy()
        chunk.insert(0, "Sample", str(sample))
        parts.append(chunk)
    if not parts:
        return
    out = pd.concat(parts, ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name=MASTER_LINEAGE_SHEET[:31], index=False)


def edges_and_founders_from_graph(napari_graph: Any, tracks_df: Any) -> tuple[Any, dict[int, int]]:
    edges = lineage_edges_from_napari_graph(napari_graph, tracks_df)
    founders = lineage_founder_map(edges, tracks_df)
    return edges, founders


def export_lineage_artifacts(
    output_dir: Path,
    tables: dict[str, Any],
    *,
    stem: str,
) -> list[Path]:
    """Write static PNG figures for lineage overview (tree, growth, divisions)."""
    from bactomate.modules.lineage_viz import export_lineage_figures

    fig_dir = Path(output_dir) / "figures"
    return export_lineage_figures(fig_dir, tables, stem=stem)
