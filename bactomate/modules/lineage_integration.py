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
    workbook_frames: Optional[int] = None
    frame_count_mismatch: bool = False
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


def _workbook_summary_frame_count(workbook_path: Path) -> tuple[int | None, bool]:
    """Return (expected mask frame count, is_1_indexed) from Summary or Per-Cell Data."""
    import pandas as pd

    path = Path(workbook_path)
    df = None
    for sheet in ("Summary", "Per-Cell Data"):
        try:
            df = pd.read_excel(path, sheet_name=sheet)
            break
        except Exception:
            continue
    if df is None:
        try:
            xl = pd.ExcelFile(path)
            if not xl.sheet_names:
                return None, True
            df = pd.read_excel(path, sheet_name=xl.sheet_names[0])
        except Exception:
            return None, True
    frame_col = next((c for c in ("Frame", "frame", "t") if c in df.columns), None)
    if frame_col is None:
        return None, True
    vals = pd.to_numeric(df[frame_col], errors="coerce").dropna().astype(int)
    if len(vals) == 0:
        return None, True
    unique = sorted(vals.unique().tolist())
    is_1_indexed = (0 not in unique) and min(unique) >= 1
    max_frame = int(max(unique))
    if is_1_indexed:
        return max_frame, True
    return max_frame + 1, False


def preflight_lineage_run(
    mask_path: Path,
    *,
    linker_mode: str,
    model_name: str,
    prefer_filtered_mask: bool = True,
    require_analysis_workbook: bool = False,
    strict_lineage: bool = False,
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
        if filt == raw:
            msg = (
                f"No filtered mask found; using raw mask {raw.name}. "
                "Run Filtering first and analyze the filtered mask for consistent joins."
            )
            if strict_lineage:
                out.ok = False
                out.errors.append(
                    "Filtered mask required for lineage tracking. Run Filtering on this mask stack first."
                )
            else:
                out.warnings.append(msg)

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

    if wb is not None:
        try:
            from bactomate.gui.analysis_paths import stem_from_mask_path

            expected_stem = stem_from_mask_path(resolved)
            wb_stem = wb.stem
            for suffix in ("_masks_analysis", "_analysis"):
                if wb_stem.endswith(suffix):
                    wb_mask_stem = wb_stem[: -len(suffix)]
                    break
            else:
                wb_mask_stem = None
            if wb_mask_stem and wb_mask_stem != expected_stem:
                msg = (
                    f"Analysis workbook ({wb.name}) may not match mask stem ({expected_stem}). "
                    "Re-run Analyze on the mask you plan to track."
                )
                if strict_lineage:
                    out.ok = False
                    out.errors.append(msg)
                else:
                    out.warnings.append(msg)
        except Exception:
            pass

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

    if wb is not None and out.n_frames is not None:
        wb_frames, _ = _workbook_summary_frame_count(wb)
        if wb_frames is not None:
            out.workbook_frames = wb_frames
            if int(wb_frames) != int(out.n_frames):
                out.frame_count_mismatch = True
                msg = (
                    f"Mask has {out.n_frames} frame(s) but analysis Summary suggests {wb_frames}. "
                    "Re-analyze the mask or pick the matching mask file."
                )
                if strict_lineage:
                    out.ok = False
                    out.errors.append(msg)
                else:
                    out.warnings.append(msg)

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
    if pf.workbook_frames is not None:
        lines.append(f"Analysis frames: {pf.workbook_frames}")
    if pf.frame_count_mismatch:
        lines.append("Frame count mismatch: Yes")
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
        "has_post_frame",
        "qc_warning",
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
    max_frame = None
    if tracks_df is not None and hasattr(tracks_df, "columns") and "frame" in tracks_df.columns and len(tracks_df) > 0:
        try:
            max_frame = int(tracks_df["frame"].dropna().astype(int).max())
        except (TypeError, ValueError):
            max_frame = None
    rows = []
    for er in edges_df.itertuples(index=False):
        p = _safe_int(getattr(er, "parent_track_id", None))
        d = _safe_int(getattr(er, "daughter_track_id", None))
        div_f = _safe_int(getattr(er, "division_frame", None))
        plf = _safe_int(getattr(er, "parent_last_frame", None))
        if p is None or d is None or div_f is None:
            continue
        has_post_frame = max_frame is None or int(div_f) < int(max_frame)
        qc_warning = "" if has_post_frame else "terminal_frame_division_unconfirmed"
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
        link_status = getattr(er, "link_status", LINK_STATUS_CANDIDATE) if hasattr(er, "link_status") else LINK_STATUS_CANDIDATE
        review_status = (
            getattr(er, "review_status", REVIEW_STATUS_CANDIDATE)
            if hasattr(er, "review_status")
            else REVIEW_STATUS_CANDIDATE
        )
        if not has_post_frame and str(link_status).lower() != "manual":
            review_status = REVIEW_STATUS_FLAGGED
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
                "has_post_frame": bool(has_post_frame),
                "qc_warning": qc_warning,
                "link_status": link_status,
                "review_status": review_status,
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
            has_post_frame = getattr(er, "has_post_frame", True)
            dist = _safe_float(getattr(er, "division_distance_px", None))
            asym = _safe_float(getattr(er, "area_asymmetry", None))
            if not bool(has_post_frame):
                rows.append(
                    {
                        "flag_type": "terminal_frame_division",
                        "track_id": p,
                        "related_track_id": str(d) if d is not None else None,
                        "frame": div_f,
                        "severity": "warning",
                        "message": (
                            f"Division {p}->{d} is at the final frame; no next movie frame "
                            "exists to confirm the post-division state."
                        ),
                    }
                )
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
        um_sources = [s for s in sources if "mask_um2" in s.lower() or "µm" in s.lower() or "um2" in s.lower()]
        if analysis_sources:
            metric = analysis_sources[0]
        elif um_sources:
            metric = um_sources[0]
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


def filter_lineage_edges_for_curation(
    edges_df: Any,
    events_df: Any = None,
    *,
    strict: bool = False,
) -> Any:
    """
    Return edges for curated lineage views.

    strict=False: drop rejected divisions; keep candidate, accepted, flagged, manual.
    strict=True: keep only accepted and manual links.
    """
    import pandas as pd

    if edges_df is None or not hasattr(edges_df, "columns") or len(edges_df) == 0:
        return edges_df
    status_by_edge: dict[tuple[int, int], str] = {}
    if events_df is not None and hasattr(events_df, "columns") and len(events_df) > 0:
        for row in events_df.itertuples(index=False):
            try:
                key = (int(getattr(row, "parent_track_id")), int(getattr(row, "daughter_track_id")))
                status_by_edge[key] = str(getattr(row, "review_status", REVIEW_STATUS_CANDIDATE)).lower()
            except (TypeError, ValueError):
                continue
    rows: list[dict[str, Any]] = []
    for _, er in edges_df.iterrows():
        p = _safe_int(er.get("parent_track_id"))
        d = _safe_int(er.get("daughter_track_id"))
        if p is None or d is None:
            continue
        link_status = str(er.get("link_status", LINK_STATUS_CANDIDATE)).lower()
        if link_status == "manual":
            rows.append(er.to_dict())
            continue
        review = status_by_edge.get((p, d), REVIEW_STATUS_CANDIDATE)
        if review == REVIEW_STATUS_REJECTED:
            continue
        if strict and review != REVIEW_STATUS_ACCEPTED:
            continue
        rows.append(er.to_dict())
    if not rows:
        return pd.DataFrame(columns=list(edges_df.columns))
    return pd.DataFrame(rows)


def area_table_for_lineage_growth(tables: dict[str, Any]) -> Any:
    """Pick the best per-track area table available for growth curves."""
    import pandas as pd

    area_df = tables.get("track_area")
    if area_df is not None and hasattr(area_df, "columns") and len(area_df) > 0:
        out = area_df.copy()
        pts = tables.get("lineage_points")
        if pts is not None and hasattr(pts, "columns") and len(pts) > 0 and "area_source" in pts.columns:
            manual = pts[pts["area_source"].astype(str) == "manual_mask_px"].copy()
            if len(manual) > 0 and {"track_id", "frame", "area"}.issubset(manual.columns):
                manual = manual[["track_id", "frame", "area", "area_source"]]
                existing = set()
                if {"track_id", "frame"}.issubset(out.columns):
                    existing = {
                        (int(r.track_id), int(r.frame))
                        for r in out[["track_id", "frame"]].dropna().itertuples(index=False)
                    }
                manual = manual[
                    ~manual.apply(lambda r: (int(r["track_id"]), int(r["frame"])) in existing, axis=1)
                ]
                if len(manual) > 0:
                    for col in out.columns:
                        if col not in manual.columns:
                            manual[col] = pd.NA
                    out = pd.concat([out, manual[out.columns]], ignore_index=True)
        return out
    pts = tables.get("lineage_points")
    if pts is None or not hasattr(pts, "columns") or len(pts) == 0:
        return None
    col = None
    source = "mask_px"
    if "Area (µm²)" in pts.columns:
        col = "Area (µm²)"
        source = "analysis:Area (µm²)"
    elif "area" in pts.columns:
        col = "area"
    if col is None or not {"track_id", "frame"}.issubset(pts.columns):
        return None
    out = pts[["track_id", "frame", col]].copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col])
    if len(out) == 0:
        return None
    out = out.rename(columns={col: "area"})
    out["area_source"] = source
    return out


def calibrate_area_table(area_df: Any, pixel_size_um: float) -> Any:
    """Convert mask-count areas to µm² when calibration is available."""
    if area_df is None or not hasattr(area_df, "columns") or len(area_df) == 0:
        return area_df
    if "area" not in area_df.columns or "area_source" not in area_df.columns:
        return area_df
    if float(pixel_size_um or 0.0) <= 0:
        return area_df
    import pandas as pd

    out = area_df.copy()
    src = out["area_source"].astype(str)
    mask_rows = src.isin(["mask_px", "manual_mask_px"])
    if mask_rows.any():
        scale = float(pixel_size_um) ** 2
        out.loc[mask_rows, "area"] = pd.to_numeric(out.loc[mask_rows, "area"], errors="coerce") * scale
        out.loc[src == "mask_px", "area_source"] = "mask_um2_from_metadata"
        out.loc[src == "manual_mask_px", "area_source"] = "manual_mask_um2_from_metadata"
    return out


def rebuild_curated_lineage_views(
    tables: dict[str, Any],
    tracks_df: Any,
    *,
    strict: bool = False,
    time_per_frame_min: float = 0.0,
) -> dict[str, Any]:
    """Recompute tree, meta, and growth timeseries from curated division links."""
    from bactomate.modules.lineage_viz import build_lineage_tree_layout

    edges_src = tables.get("lineage_edges")
    events = tables.get("lineage_events")
    curated_edges = filter_lineage_edges_for_curation(edges_src, events, strict=strict)
    tracks_src = tables.get("lineage_points")
    if tracks_src is None or not hasattr(tracks_src, "columns") or "track_id" not in tracks_src.columns:
        tracks_src = tracks_df
    founder = lineage_founder_map(curated_edges, tracks_src)
    meta = tables.get("lineage_tracks_meta")
    if meta is None or not hasattr(meta, "columns") or len(meta) == 0:
        meta = track_lineage_metadata(curated_edges, tracks_src)
    cal = tables.get("lineage_calibration") or {}
    if not time_per_frame_min:
        time_per_frame_min = float(cal.get("time_per_frame_min", 0.0) or 0.0)
    area_df = area_table_for_lineage_growth(tables)
    area_df = calibrate_area_table(area_df, float(cal.get("pixel_size_um", 0.0) or 0.0))
    growth_ts = build_lineage_growth_timeseries(area_df, founder, time_per_frame_min=time_per_frame_min)
    nodes, tree_edges = build_lineage_tree_layout(curated_edges, meta)
    biology_labels = build_biology_track_labels(curated_edges, tracks_src, meta)
    return {
        "lineage_edges_curated": curated_edges,
        "lineage_tracks_meta_curated": meta,
        "lineage_growth_timeseries_curated": growth_ts,
        "lineage_tree_nodes_curated": nodes,
        "lineage_tree_edges_curated": tree_edges,
        "founder_by_track_curated": founder,
        "biology_track_labels": biology_labels,
    }


def build_biology_track_labels(
    edges_df: Any,
    tracks_df: Any,
    lineage_tracks_meta: Any = None,
    *,
    founder_lineage_order: Optional[list[int]] = None,
) -> Any:
    """
    Map each track_id to a biology-facing label (Cell 1, Cell 1.1, Cell 1.2, …).

    Labels follow the curated lineage tree, not raw Trackastra IDs.
    """
    import pandas as pd

    cols = ["track_id", "biology_label", "lineage_id", "lineage_display", "branch_role", "is_manual"]
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = defaultdict(list)
    manual_children: set[int] = set()
    if edges_df is not None and hasattr(edges_df, "columns") and len(edges_df) > 0:
        for _, row in edges_df.iterrows():
            p = _safe_int(row.get("parent_track_id"))
            d = _safe_int(row.get("daughter_track_id"))
            if p is None or d is None:
                continue
            parent_of[d] = p
            children_of[p].append(d)
            if str(row.get("link_status", "")).lower() == "manual":
                manual_children.add(d)

    track_ids: set[int] = set()
    if tracks_df is not None and hasattr(tracks_df, "columns") and "track_id" in tracks_df.columns:
        for v in tracks_df["track_id"].dropna().unique():
            tid = _safe_int(v)
            if tid is not None:
                track_ids.add(tid)
    if lineage_tracks_meta is not None and hasattr(lineage_tracks_meta, "columns") and "track_id" in lineage_tracks_meta.columns:
        for v in lineage_tracks_meta["track_id"].dropna().unique():
            tid = _safe_int(v)
            if tid is not None:
                track_ids.add(tid)

    founder_map = lineage_founder_map(edges_df, tracks_df)
    founders = [int(x) for x in (founder_lineage_order or sorted(set(founder_map.values())))]
    lineage_number = {lid: idx + 1 for idx, lid in enumerate(founders)}
    labels: dict[int, dict[str, Any]] = {}

    def _role_for(tid: int, biology_label: str) -> str:
        if tid < 0 or tid in manual_children:
            return "manual_daughter"
        if biology_label.count(".") == 0:
            return "founder"
        return "daughter"

    def _assign_subtree(root_tid: int, root_label: str, lineage_id: int) -> None:
        if root_tid in labels:
            return
        labels[root_tid] = {
            "track_id": root_tid,
            "biology_label": root_label,
            "lineage_id": lineage_id,
            "lineage_display": f"Lineage {lineage_number.get(lineage_id, '?')}",
            "branch_role": _role_for(root_tid, root_label),
            "is_manual": bool(root_tid < 0 or root_tid in manual_children),
        }
        children = sorted(set(children_of.get(root_tid, [])))
        for i, child in enumerate(children, 1):
            _assign_subtree(child, f"{root_label}.{i}", lineage_id)

    tracks_by_lineage: dict[int, list[int]] = defaultdict(list)
    for tid in track_ids:
        lid = int(founder_map.get(tid, tid))
        tracks_by_lineage[lid].append(tid)

    for lid in founders:
        lineage_tracks = tracks_by_lineage.get(lid, [])
        roots = [t for t in lineage_tracks if parent_of.get(t) is None or parent_of.get(t) not in track_ids]
        if not roots:
            roots = [lid] if lid in track_ids else (sorted(lineage_tracks)[:1] if lineage_tracks else [lid])
        if lid in roots:
            roots = [lid] + [r for r in sorted(roots) if r != lid]
        else:
            roots = sorted(roots)
        cell_num = lineage_number.get(lid, 1)
        for root_idx, root in enumerate(roots):
            root_label = f"Cell {cell_num}" if len(roots) == 1 else f"Cell {cell_num}.{root_idx + 1}"
            _assign_subtree(int(root), root_label, lid)

    for tid in sorted(track_ids):
        if tid in labels:
            continue
        lid = int(founder_map.get(tid, tid))
        cell_num = lineage_number.get(lid, 1)
        fallback = f"Cell {cell_num}?" if len(founders) == 1 else f"track {tid}"
        labels[tid] = {
            "track_id": tid,
            "biology_label": fallback,
            "lineage_id": lid,
            "lineage_display": f"Lineage {lineage_number.get(lid, '?')}",
            "branch_role": "orphan",
            "is_manual": tid < 0,
        }

    if not labels:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(list(labels.values()), columns=cols)


def biology_label_map(labels_df: Any) -> dict[int, str]:
    """track_id -> biology_label from a biology_track_labels table."""
    out: dict[int, str] = {}
    if labels_df is None or not hasattr(labels_df, "columns") or len(labels_df) == 0:
        return out
    if not {"track_id", "biology_label"}.issubset(labels_df.columns):
        return out
    for row in labels_df.itertuples(index=False):
        tid = _safe_int(getattr(row, "track_id", None))
        if tid is None:
            continue
        out[tid] = str(getattr(row, "biology_label", tid))
    return out


def division_daughters_at_frame(events_df: Any, parent_id: int, division_frame: int) -> list[int]:
    """Sorted daughter track IDs for one mother at one division frame."""
    import pandas as pd

    if events_df is None or not isinstance(events_df, pd.DataFrame) or len(events_df) == 0:
        return []
    if not {"parent_track_id", "daughter_track_id", "division_frame"}.issubset(events_df.columns):
        return []
    sub = events_df[
        (events_df["parent_track_id"].astype(int) == int(parent_id))
        & (events_df["division_frame"].astype(int) == int(division_frame))
    ]
    if len(sub) == 0:
        return []
    return sorted(int(x) for x in sub["daughter_track_id"].dropna().unique().tolist())


def format_biology_division_split(
    parent_id: int,
    daughter_id: int,
    division_frame: int,
    events_df: Any,
    label_map: dict[int, str],
) -> str:
    """Biology-facing division text, grouping co-split daughters on one row."""
    daughters = division_daughters_at_frame(events_df, parent_id, division_frame)
    if not daughters:
        daughters = [int(daughter_id)]
    parent_lbl = label_map.get(int(parent_id), f"track {parent_id}")
    daughter_lbls = [label_map.get(int(d), f"track {d}") for d in daughters]
    if len(daughter_lbls) > 1:
        return f"{parent_lbl} → {' + '.join(daughter_lbls)} @ frame {int(division_frame) + 1}"
    return f"{parent_lbl} → {daughter_lbls[0]} @ frame {int(division_frame) + 1}"


def refresh_lineage_events_table(
    tables: dict[str, Any],
    tracks_df: Any,
    *,
    preserve_review: bool = True,
) -> Any:
    """Rebuild lineage_events from edges while keeping user review decisions."""
    import pandas as pd

    edges = tables.get("lineage_edges")
    if edges is None or not hasattr(edges, "columns") or len(edges) == 0:
        return tables.get("lineage_events")
    review_map: dict[tuple[int, int], str] = {}
    old = tables.get("lineage_events")
    if preserve_review and old is not None and hasattr(old, "columns") and len(old) > 0:
        for row in old.itertuples(index=False):
            p = _safe_int(getattr(row, "parent_track_id", None))
            d = _safe_int(getattr(row, "daughter_track_id", None))
            if p is None or d is None:
                continue
            review_map[(p, d)] = str(getattr(row, "review_status", REVIEW_STATUS_CANDIDATE)).lower()
    meta = tables.get("lineage_tracks_meta")
    cal = tables.get("lineage_calibration") or {}
    events = build_lineage_events(
        edges,
        tracks_df,
        area_df=area_table_for_lineage_growth(tables),
        time_per_frame_min=float(cal.get("time_per_frame_min", 0.0) or 0.0),
        lineage_tracks_meta=meta,
        pixel_size_um=float(cal.get("pixel_size_um", 0.0) or 0.0),
    )
    if events is None or not hasattr(events, "columns") or len(events) == 0:
        return events
    for i, row in events.iterrows():
        key = (int(row["parent_track_id"]), int(row["daughter_track_id"]))
        if key in review_map:
            events.at[i, "review_status"] = review_map[key]
        elif str(row.get("link_status", "")).lower() == "manual":
            events.at[i, "review_status"] = REVIEW_STATUS_ACCEPTED
        if (
            "has_post_frame" in events.columns
            and not bool(events.at[i, "has_post_frame"])
            and str(row.get("link_status", "")).lower() != "manual"
        ):
            events.at[i, "review_status"] = REVIEW_STATUS_FLAGGED
    return events


def resolve_lineage_area_unit(
    tables: dict[str, Any],
    *,
    growth_ts: Any = None,
) -> str:
    """Consistent area unit for summary text and plots (µm² vs px)."""
    from bactomate.modules.lineage_viz import lineage_area_unit_label

    if growth_ts is None:
        growth_ts = tables.get("lineage_growth_timeseries_curated")
    if growth_ts is None:
        growth_ts = tables.get("lineage_growth_timeseries")
    if growth_ts is not None and hasattr(growth_ts, "columns") and len(growth_ts) > 0:
        unit = lineage_area_unit_label(growth_ts)
        if unit == "µm²":
            return unit

    pts = tables.get("lineage_points")
    if pts is not None and hasattr(pts, "columns"):
        if "Area (µm²)" in pts.columns:
            return "µm²"
        if "area_source" in pts.columns:
            src = " ".join(str(v) for v in pts["area_source"].dropna().unique().tolist()).lower()
            if "analysis:" in src or "mask_um2" in src or "µm" in src or "um2" in src:
                return "µm²"

    area_df = tables.get("track_area")
    if area_df is not None and hasattr(area_df, "columns") and "area_source" in area_df.columns:
        src = " ".join(str(v) for v in area_df["area_source"].dropna().unique().tolist()).lower()
        if "analysis:" in src or "mask_um2" in src:
            return "µm²"

    cal = tables.get("lineage_calibration") or {}
    if float(cal.get("pixel_size_um", 0.0) or 0.0) > 0 and growth_ts is not None:
        unit = lineage_area_unit_label(growth_ts)
        if unit == "µm²":
            return unit

    return lineage_area_unit_label(growth_ts) if growth_ts is not None else "px"


def compute_lineage_analysis_summary(
    tables: dict[str, Any],
    *,
    lineage_id: Optional[int] = None,
    growth_ts: Any = None,
    untracked_unresolved: int = 0,
) -> dict[str, Any]:
    """Build a compact summary dict for the Analysis panel."""
    import pandas as pd

    out: dict[str, Any] = {
        "lineage_id": lineage_id,
        "start_frame": None,
        "end_frame": None,
        "n_descendants_final": None,
        "n_divisions_accepted": 0,
        "n_divisions_rejected": 0,
        "n_divisions_candidate": 0,
        "n_divisions_flagged": 0,
        "untracked_unresolved": int(untracked_unresolved),
        "area_start": None,
        "area_end": None,
        "area_fold_change": None,
        "growth_rate": None,
        "doubling_time": None,
        "growth_unit": None,
        "area_unit": "px",
    }
    if lineage_id is None:
        return out
    lid = int(lineage_id)
    summary = tables.get("lineage_summary")
    if summary is not None and hasattr(summary, "columns") and len(summary) > 0 and "lineage_id" in summary.columns:
        row = summary[summary["lineage_id"].astype(int) == lid]
        if len(row):
            r0 = row.iloc[0]
            out["start_frame"] = _safe_int(r0.get("start_frame"))
            out["end_frame"] = _safe_int(r0.get("end_frame"))
            out["n_descendants_final"] = _safe_int(r0.get("n_tracks"))
    events = tables.get("lineage_events")
    if events is not None and hasattr(events, "columns") and len(events) > 0:
        if "lineage_id" in events.columns:
            ev = events[events["lineage_id"].astype(int) == lid]
        else:
            ev = events
        if "review_status" in ev.columns:
            for status_col, key in (
                (REVIEW_STATUS_ACCEPTED, "n_divisions_accepted"),
                (REVIEW_STATUS_REJECTED, "n_divisions_rejected"),
                (REVIEW_STATUS_CANDIDATE, "n_divisions_candidate"),
                (REVIEW_STATUS_FLAGGED, "n_divisions_flagged"),
            ):
                status_ev = ev[ev["review_status"].astype(str).str.lower() == status_col]
                if {"parent_track_id", "division_frame"}.issubset(status_ev.columns):
                    out[key] = int(status_ev[["parent_track_id", "division_frame"]].drop_duplicates().shape[0])
                else:
                    out[key] = int(len(status_ev))
    growth = tables.get("lineage_growth")
    if growth is not None and hasattr(growth, "columns") and len(growth) > 0 and "lineage_id" in growth.columns:
        gr = growth[growth["lineage_id"].astype(int) == lid]
        if len(gr):
            g0 = gr.iloc[0]
            out["growth_rate"] = _safe_float(g0.get("growth"))
            out["doubling_time"] = _safe_float(g0.get("doubling_time"))
            out["growth_unit"] = g0.get("growth_unit")
    if growth_ts is None:
        growth_ts = tables.get("lineage_growth_timeseries_curated")
        if growth_ts is None:
            growth_ts = tables.get("lineage_growth_timeseries")
    if growth_ts is not None and hasattr(growth_ts, "columns") and len(growth_ts) > 0:
        sub = growth_ts.copy()
        if "lineage_id" in sub.columns:
            sub = sub[sub["lineage_id"].astype(int) == lid]
        if len(sub) and "total_area" in sub.columns:
            sub = sub.sort_values("frame")
            out["area_start"] = _safe_float(sub.iloc[0]["total_area"])
            out["area_end"] = _safe_float(sub.iloc[-1]["total_area"])
            if out["area_start"] and out["area_end"] and out["area_start"] > 0:
                out["area_fold_change"] = float(out["area_end"]) / float(out["area_start"])
            out["area_unit"] = resolve_lineage_area_unit(tables, growth_ts=sub)
    pts = tables.get("lineage_points")
    if out["n_descendants_final"] is None and pts is not None and hasattr(pts, "columns") and "lineage_id" in pts.columns:
        sub = pts[pts["lineage_id"].astype(int) == lid]
        if len(sub) and "track_id" in sub.columns:
            out["n_descendants_final"] = int(sub["track_id"].nunique())
    return out


def aggregate_division_review_status(statuses: list[str]) -> str:
    """Combine per-edge review flags for one biological division (worst case wins)."""
    normalized = [str(s or REVIEW_STATUS_CANDIDATE).strip().lower() for s in statuses]
    if any(s == REVIEW_STATUS_REJECTED for s in normalized):
        return REVIEW_STATUS_REJECTED
    if any(s == REVIEW_STATUS_FLAGGED for s in normalized):
        return REVIEW_STATUS_FLAGGED
    if normalized and all(s == REVIEW_STATUS_ACCEPTED for s in normalized):
        return REVIEW_STATUS_ACCEPTED
    if normalized and all(s == REVIEW_STATUS_CANDIDATE for s in normalized):
        return REVIEW_STATUS_CANDIDATE
    return REVIEW_STATUS_CANDIDATE


def collapse_lineage_events_for_display(events_df: Any) -> Any:
    """One row per mother cell division (groups co-split daughter edges)."""
    import pandas as pd

    if events_df is None or not isinstance(events_df, pd.DataFrame) or len(events_df) == 0:
        return events_df
    required = {"parent_track_id", "daughter_track_id", "division_frame"}
    if not required.issubset(events_df.columns):
        return events_df
    rows: list[dict[str, Any]] = []
    grouped = events_df.groupby(
        [events_df["parent_track_id"].astype(int), events_df["division_frame"].astype(int)],
        sort=False,
    )
    for (parent_id, division_frame), sub in grouped:
        sub = sub.sort_values("daughter_track_id")
        first = sub.iloc[0]
        review_vals = (
            sub["review_status"].astype(str).tolist()
            if "review_status" in sub.columns
            else [REVIEW_STATUS_CANDIDATE] * len(sub)
        )
        link_vals = (
            sub["link_status"].astype(str).str.lower().tolist()
            if "link_status" in sub.columns
            else ["candidate"] * len(sub)
        )
        has_post = True
        if "has_post_frame" in sub.columns:
            has_post = bool(sub["has_post_frame"].astype(bool).all())
        row: dict[str, Any] = {
            "parent_track_id": int(parent_id),
            "daughter_track_id": int(sub.iloc[0]["daughter_track_id"]),
            "division_frame": int(division_frame),
            "review_status": aggregate_division_review_status(review_vals),
            "link_status": "manual" if any(v == "manual" for v in link_vals) else "candidate",
            "has_post_frame": has_post,
        }
        if "division_frame_1based" in sub.columns:
            row["division_frame_1based"] = int(sub.iloc[0]["division_frame_1based"])
        elif "division_frame" in sub.columns:
            row["division_frame_1based"] = int(division_frame) + 1
        if "generation" in sub.columns:
            row["generation"] = first.get("generation")
        rows.append(row)
    return pd.DataFrame(rows)


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


def apply_division_review_status(
    events_df: Any,
    *,
    parent_track_id: int,
    division_frame: int,
    review_status: str,
) -> Any:
    """Update review_status for every daughter edge at one mother division."""
    import pandas as pd

    if events_df is None or not hasattr(events_df, "columns"):
        return events_df
    status = str(review_status or REVIEW_STATUS_CANDIDATE).strip().lower()
    if status not in VALID_REVIEW_STATUSES:
        status = REVIEW_STATUS_FLAGGED
    out = events_df.copy()
    if "review_status" not in out.columns:
        out["review_status"] = REVIEW_STATUS_CANDIDATE
    mask = (out["parent_track_id"].astype(int) == int(parent_track_id)) & (
        out["division_frame"].astype(int) == int(division_frame)
    )
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
    area_df = calibrate_area_table(area_df, pixel_size_um)
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
    out["track_area"] = area_df
    out["lineage_calibration"] = {
        "pixel_size_um": float(pixel_size_um or 0.0),
        "time_per_frame_min": float(time_per_frame_min or 0.0),
    }
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
    out["biology_track_labels"] = build_biology_track_labels(edges_df, tracks_df, lineage_tracks_meta)
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
    """Write static PNG figures for lineage overview and figures/analysis/ detail plots."""
    from bactomate.modules.lineage_viz import export_lineage_figures

    fig_dir = Path(output_dir) / "figures"
    return export_lineage_figures(fig_dir, tables, stem=stem)


def _read_lineage_table_file(path: Path) -> Any:
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffixes[-2:] == [".csv", ".gz"] or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_lineage_tables_from_output_dir(output_dir: Path) -> dict[str, Any]:
    """Load lineage tables from a completed ``_tracking/{stem}_tracking/`` folder."""
    import json

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"No tables/ in {output_dir}")
    tables: dict[str, Any] = {}
    meta_json = tables_dir / "metadata.json"
    if meta_json.is_file():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            tables["lineage_calibration"] = {
                "pixel_size_um": float(meta.get("pixel_size_um", 0.0) or 0.0),
                "time_per_frame_min": float(meta.get("time_per_frame_min", 0.0) or 0.0),
            }
        except Exception:
            pass
    for key in (
        "lineage_points",
        "lineage_edges",
        "lineage_tracks_meta",
        "lineage_events",
        "lineage_qc",
        "lineage_summary",
        "lineage_growth",
        "lineage_growth_timeseries",
        "lineage_tree_nodes",
        "lineage_tree_edges",
        "lineage_measurements",
        "track_area",
        "biology_track_labels",
    ):
        for ext in (".parquet", ".csv.gz", ".csv"):
            p = tables_dir / f"{key}{ext}"
            if p.is_file():
                tables[key] = _read_lineage_table_file(p)
                break
    if tables.get("lineage_points") is None:
        raise FileNotFoundError("lineage_points table not found")
    return tables


def load_lineage_tables_from_report_xlsx(xlsx_path: Path) -> dict[str, Any]:
    """Load lineage tables from ``{stem}_lineage_report.xlsx``."""
    import pandas as pd
    from bactomate.modules.lineage_viz import LINEAGE_REPORT_SHEET_TO_KEY

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.is_file():
        raise FileNotFoundError(xlsx_path)
    xl = pd.ExcelFile(xlsx_path)
    tables: dict[str, Any] = {}
    for sheet_name, key in LINEAGE_REPORT_SHEET_TO_KEY.items():
        if sheet_name in xl.sheet_names:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
            if df is not None and hasattr(df, "columns") and len(df.columns) > 0:
                tables[key] = df
    if tables.get("lineage_points") is None:
        raise FileNotFoundError("Lineage Points sheet not found in workbook")
    return tables


def _is_lineage_tracking_output_dir(path: Path) -> bool:
    """True when ``path`` looks like ``_tracking/{stem}_tracking/`` output."""
    path = Path(path)
    if not path.is_dir():
        return False
    if (path / "tables").is_dir():
        return True
    return any(path.glob("*_lineage_report.xlsx"))


def discover_lineage_tracking_dirs(search_root: Path) -> list[Path]:
    """
    Find completed lineage tracking folders under a user-selected path.

    Supports:
    - ``…/{stem}_tracking/`` directly under the folder
    - ``…/_tracking/{stem}_tracking/`` (standard BactoMate layout)
    - nested ``…/_tracking/*`` when the child contains ``tables/`` or a report
    """
    search_root = Path(search_root).resolve()
    if not search_root.is_dir():
        return []
    found: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen or not _is_lineage_tracking_output_dir(resolved):
            return
        seen.add(key)
        found.append(resolved)

    if _is_lineage_tracking_output_dir(search_root):
        add(search_root)

    for pattern in ("*_tracking", "*/*_tracking"):
        for candidate in search_root.glob(pattern):
            if candidate.is_dir():
                add(candidate)

    tracking_root = search_root / "_tracking"
    if tracking_root.is_dir():
        for candidate in tracking_root.iterdir():
            if candidate.is_dir():
                add(candidate)

    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_lineage_source(source: Path) -> tuple[Path, str, Path]:
    """
    Resolve a user-selected path to (tables_parent_dir, stem, figures_dir).

    Accepts a tracking output folder, a parent folder containing ``_tracking/``,
    or a ``*_lineage_report.xlsx`` file.
    """
    source = Path(source).resolve()
    if source.is_file() and source.suffix.lower() == ".xlsx" and "lineage_report" in source.name.lower():
        stem = source.stem
        if stem.endswith("_lineage_report"):
            stem = stem[: -len("_lineage_report")]
        parent = source.parent
        fig_dir = parent / "figures"
        if (parent / "tables").is_dir():
            return parent, stem, fig_dir
        return parent, stem, fig_dir
    if source.is_dir():
        if (source / "tables").is_dir():
            stem = source.name
            if stem.endswith("_tracking"):
                stem = stem[: -len("_tracking")]
            return source, stem, source / "figures"
        reports = sorted(source.glob("*_lineage_report.xlsx"))
        if reports:
            rep = reports[0]
            stem = rep.stem
            if stem.endswith("_lineage_report"):
                stem = stem[: -len("_lineage_report")]
            parent = rep.parent
            return parent, stem, parent / "figures"
        nested = sorted(source.glob("*_tracking"))
        for d in nested:
            if (d / "tables").is_dir():
                stem = d.name
                if stem.endswith("_tracking"):
                    stem = stem[: -len("_tracking")]
                return d, stem, d / "figures"
        discovered = discover_lineage_tracking_dirs(source)
        if discovered:
            parent = discovered[0]
            stem = parent.name
            if stem.endswith("_tracking"):
                stem = stem[: -len("_tracking")]
            return parent, stem, parent / "figures"
    raise FileNotFoundError(f"Could not resolve lineage source: {source}")


def load_lineage_tables_from_source(source: Path) -> tuple[dict[str, Any], Path, str, Path]:
    """Load tables from tracking output folder or lineage report workbook."""
    parent, stem, fig_dir = resolve_lineage_source(source)
    source = Path(source).resolve()
    if source.is_file() and source.suffix.lower() == ".xlsx":
        tables = load_lineage_tables_from_report_xlsx(source)
    else:
        tables = load_lineage_tables_from_output_dir(parent)
    return tables, parent, stem, fig_dir


LINEAGE_CURATION_SCHEMA_VERSION = 2
LINEAGE_CURATION_FILENAME = "lineage_curation.json"
LINEAGE_CURATION_LEGACY_FILENAME = "lineage_manual_edits.json"


def resolve_lineage_curation_path(mask_path: Path, *, output_dir: Optional[Path] = None) -> Path:
    """Staging path for lineage curation (written during preview QC, read on full export)."""
    if output_dir is not None:
        return Path(output_dir) / "tables" / LINEAGE_CURATION_FILENAME
    mask_path = resolve_lineage_mask_path(Path(mask_path))
    try:
        from bactomate.gui.analysis_paths import base_dir_for_mask

        base = base_dir_for_mask(mask_path)
    except Exception:
        base = mask_path.parent
    stem = mask_path.stem
    return base / "_tracking" / f"{stem}_tracking" / "tables" / LINEAGE_CURATION_FILENAME


def _curation_legacy_path(path: Path) -> Path:
    return path.parent / LINEAGE_CURATION_LEGACY_FILENAME


def pack_lineage_curation_payload(
    tables: dict[str, Any],
    *,
    mask_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Serialize inspector tables into a JSON-friendly curation bundle."""
    import pandas as pd

    edits = tables.get("manual_lineage_edits")
    edit_rows: list[dict[str, Any]] = []
    if edits is not None and hasattr(edits, "columns") and len(edits) > 0:
        edit_rows = edits.where(edits.notna(), None).to_dict(orient="records")

    ignored_raw = tables.get("manual_ignored_untracked", set()) or set()
    ignored: list[list[int]] = []
    for item in ignored_raw:
        try:
            ignored.append([int(item[0]), int(item[1])])
        except (TypeError, ValueError, IndexError):
            continue
    ignored.sort()

    division_reviews: list[dict[str, Any]] = []
    events = tables.get("lineage_events")
    if events is not None and hasattr(events, "columns") and len(events) > 0:
        for row in events.itertuples(index=False):
            p = _safe_int(getattr(row, "parent_track_id", None))
            d = _safe_int(getattr(row, "daughter_track_id", None))
            div_f = _safe_int(getattr(row, "division_frame", None))
            if p is None or d is None or div_f is None:
                continue
            division_reviews.append(
                {
                    "parent_track_id": p,
                    "daughter_track_id": d,
                    "division_frame": div_f,
                    "review_status": str(getattr(row, "review_status", REVIEW_STATUS_CANDIDATE)).lower(),
                    "mother_y": _safe_float(getattr(row, "mother_y", None)),
                    "mother_x": _safe_float(getattr(row, "mother_x", None)),
                    "daughter_y": _safe_float(getattr(row, "daughter_y", None)),
                    "daughter_x": _safe_float(getattr(row, "daughter_x", None)),
                }
            )

    payload: dict[str, Any] = {
        "schema_version": LINEAGE_CURATION_SCHEMA_VERSION,
        "manual_lineage_edits": edit_rows,
        "manual_ignored_untracked": ignored,
        "division_reviews": division_reviews,
    }
    if mask_path is not None:
        payload["mask_path"] = str(Path(mask_path).resolve())
    return payload


def save_lineage_curation(path: Path, tables: dict[str, Any], *, mask_path: Optional[Path] = None) -> Path:
    """Write curation bundle next to the tracking output tables/."""
    path = Path(path)
    payload = pack_lineage_curation_payload(tables, mask_path=mask_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def load_lineage_curation(path: Path) -> Optional[dict[str, Any]]:
    """Load curation bundle (falls back to legacy manual-edits JSON)."""
    path = Path(path)
    if not path.is_file():
        legacy = _curation_legacy_path(path)
        if legacy.is_file():
            path = legacy
        else:
            return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "division_reviews" not in data and data.get("manual_lineage_edits") is not None:
        data.setdefault("division_reviews", [])
    return data


def _track_point_at_frame(tracks_df: Any, track_id: int, frame: int) -> Optional[tuple[float, float]]:
    if tracks_df is None or not hasattr(tracks_df, "columns"):
        return None
    if not {"track_id", "frame", "y", "x"}.issubset(tracks_df.columns):
        return None
    sub = tracks_df[
        (tracks_df["track_id"].astype(int) == int(track_id)) & (tracks_df["frame"].astype(int) == int(frame))
    ]
    if len(sub) == 0:
        sub = tracks_df[tracks_df["track_id"].astype(int) == int(track_id)].sort_values("frame")
        if len(sub) == 0:
            return None
        row = sub.iloc[-1]
    else:
        row = sub.iloc[0]
    y = _safe_float(row.get("y"))
    x = _safe_float(row.get("x"))
    if y is None or x is None:
        return None
    return float(y), float(x)


def nearest_track_id_at_frame(
    tracks_df: Any,
    frame: int,
    y: float,
    x: float,
    *,
    max_dist_px: float = 120.0,
) -> Optional[int]:
    """Pick the track whose centroid is closest to (y, x) on a given frame."""
    if tracks_df is None or not hasattr(tracks_df, "columns"):
        return None
    if not {"track_id", "frame", "y", "x"}.issubset(tracks_df.columns):
        return None
    sub = tracks_df[tracks_df["frame"].astype(int) == int(frame)]
    if len(sub) == 0:
        return None
    best_tid: Optional[int] = None
    best_d2 = float(max_dist_px) ** 2
    for row in sub.itertuples(index=False):
        py = _safe_float(getattr(row, "y", None))
        px = _safe_float(getattr(row, "x", None))
        tid = _safe_int(getattr(row, "track_id", None))
        if py is None or px is None or tid is None:
            continue
        d2 = (float(py) - float(y)) ** 2 + (float(px) - float(x)) ** 2
        if d2 <= best_d2:
            best_d2 = d2
            best_tid = int(tid)
    return best_tid


def _match_division_review_to_edge(
    review: dict[str, Any],
    edges_df: Any,
    tracks_df: Any,
    *,
    max_dist_px: float = 100.0,
) -> Optional[tuple[int, int]]:
    """Map a saved division review onto an edge in a fresh tracking run."""
    import pandas as pd

    if edges_df is None or not hasattr(edges_df, "columns") or len(edges_df) == 0:
        return None
    div_f = _safe_int(review.get("division_frame"))
    p_hint = _safe_int(review.get("parent_track_id"))
    d_hint = _safe_int(review.get("daughter_track_id"))
    if div_f is None:
        return None
    if p_hint is not None and d_hint is not None:
        exact = edges_df[
            (edges_df["division_frame"].astype(int) == int(div_f))
            & (edges_df["parent_track_id"].astype(int) == int(p_hint))
            & (edges_df["daughter_track_id"].astype(int) == int(d_hint))
        ]
        if len(exact) > 0:
            return int(p_hint), int(d_hint)
    my = _safe_float(review.get("mother_y"))
    mx = _safe_float(review.get("mother_x"))
    dy = _safe_float(review.get("daughter_y"))
    dx = _safe_float(review.get("daughter_x"))
    if my is None or mx is None or dy is None or dx is None:
        return None
    candidates = edges_df[edges_df["division_frame"].astype(int) == int(div_f)]
    if len(candidates) == 0:
        return None
    best_key: Optional[tuple[int, int]] = None
    best_score = float(max_dist_px) ** 2 * 2.0
    for er in candidates.itertuples(index=False):
        p = _safe_int(getattr(er, "parent_track_id", None))
        d = _safe_int(getattr(er, "daughter_track_id", None))
        if p is None or d is None:
            continue
        mpt = _track_point_at_frame(tracks_df, p, int(div_f)) or _track_point_at_frame(
            tracks_df, p, max(0, int(div_f) - 1)
        )
        dpt = _track_point_at_frame(tracks_df, d, int(div_f))
        if mpt is None or dpt is None:
            continue
        score = (mpt[0] - my) ** 2 + (mpt[1] - mx) ** 2 + (dpt[0] - dy) ** 2 + (dpt[1] - dx) ** 2
        if score < best_score:
            best_score = score
            best_key = (p, d)
    return best_key


def _append_table_row(df: Any, row: dict[str, Any]) -> Any:
    import pandas as pd

    row_df = pd.DataFrame([row])
    if df is None or not hasattr(df, "columns") or len(df) == 0:
        return row_df
    out = df.copy()
    for col in row_df.columns:
        if col not in out.columns:
            out[col] = np.nan
    for col in out.columns:
        if col not in row_df.columns:
            row_df[col] = np.nan
    return pd.concat([out, row_df[out.columns]], ignore_index=True)


def _next_free_track_id(tracks_df: Any, masks_tracked: Any) -> int:
    used: set[int] = set()
    if tracks_df is not None and hasattr(tracks_df, "columns") and "track_id" in tracks_df.columns:
        for v in tracks_df["track_id"].dropna().unique():
            tid = _safe_int(v)
            if tid is not None and tid > 0:
                used.add(int(tid))
    if masks_tracked is not None and getattr(masks_tracked, "ndim", 0) == 3:
        for fi in range(int(masks_tracked.shape[0])):
            for v in np.unique(masks_tracked[fi]):
                tid = _safe_int(v)
                if tid is not None and tid > 0:
                    used.add(int(tid))
    candidate = max(used) + 1 if used else 1
    while candidate in used:
        candidate += 1
    return int(candidate)


def _apply_manual_lineage_edits(
    edits: list[dict[str, Any]],
    *,
    tracks_df: Any,
    edges_df: Any,
    masks_tracked: Any,
    masks_input: Any = None,
) -> tuple[Any, Any, Any, int]:
    """Replay orange-cell manual assignments onto a fresh tracking run."""
    import pandas as pd

    if not edits:
        return tracks_df, edges_df, masks_tracked, 0
    manual_counter = 0
    for edit in edits:
        try:
            manual_counter = max(manual_counter, abs(int(edit.get("manual_track_id", 0) or 0)))
        except (TypeError, ValueError):
            continue
    applied = 0
    for edit in edits:
        frame = _safe_int(edit.get("frame"))
        label = _safe_int(edit.get("untracked_label"))
        if frame is None or label is None:
            continue
        y = _safe_float(edit.get("y"))
        x = _safe_float(edit.get("x"))
        area = _safe_float(edit.get("area_px")) or 0.0
        if y is None or x is None:
            continue
        parent_tid = _safe_int(edit.get("parent_track_id"))
        py = _safe_float(edit.get("parent_y"))
        px = _safe_float(edit.get("parent_x"))
        if py is not None and px is not None:
            remapped = nearest_track_id_at_frame(tracks_df, frame, py, px)
            if remapped is not None:
                parent_tid = remapped
        elif parent_tid is not None:
            if tracks_df is not None and hasattr(tracks_df, "columns"):
                exists = (
                    (tracks_df["track_id"].astype(int) == int(parent_tid))
                    & (tracks_df["frame"].astype(int) == int(frame))
                ).any()
                if not exists:
                    parent_tid = nearest_track_id_at_frame(tracks_df, frame, y, x)
        if parent_tid is None:
            continue
        lineage_id = _safe_int(edit.get("lineage_id")) or int(parent_tid)
        manual_tid = _safe_int(edit.get("manual_track_id"))
        if manual_tid is None:
            manual_counter += 1
            manual_tid = _next_free_track_id(tracks_df, masks_tracked)
        elif int(manual_tid) < 0:
            manual_tid = _next_free_track_id(tracks_df, masks_tracked)
        else:
            manual_tid = int(manual_tid)

        edge_row = {
            "parent_track_id": int(parent_tid),
            "daughter_track_id": int(manual_tid),
            "division_frame": int(frame),
            "parent_last_frame": max(0, int(frame) - 1),
            "link_status": "manual",
        }
        if edges_df is None or not hasattr(edges_df, "columns"):
            edges_df = pd.DataFrame(columns=list(edge_row.keys()))
        edge_exists = (
            len(edges_df) > 0
            and "daughter_track_id" in edges_df.columns
            and (edges_df["daughter_track_id"].astype(int) == int(manual_tid)).any()
        )
        if not edge_exists:
            edges_df = _append_table_row(edges_df, edge_row)

        point_row = {
            "track_id": int(manual_tid),
            "frame": int(frame),
            "y": float(y),
            "x": float(x),
            "lineage_id": int(lineage_id),
        }
        if tracks_df is None or not hasattr(tracks_df, "columns"):
            tracks_df = pd.DataFrame(columns=["track_id", "frame", "y", "x"])
        point_exists = (
            len(tracks_df) > 0
            and (tracks_df["track_id"].astype(int) == int(manual_tid))
            & (tracks_df["frame"].astype(int) == int(frame))
        ).any()
        if not point_exists:
            tracks_df = _append_table_row(tracks_df, point_row)

        if masks_tracked is not None and getattr(masks_tracked, "ndim", 0) == 3:
            if int(frame) < int(masks_tracked.shape[0]):
                region = None
                if masks_input is not None and getattr(masks_input, "ndim", 0) == 3:
                    if int(frame) < int(masks_input.shape[0]):
                        region = np.asarray(masks_input[int(frame)]) == int(label)
                if region is None or not np.any(region):
                    yy = int(round(float(y)))
                    xx = int(round(float(x)))
                    mf = np.asarray(masks_tracked[int(frame)])
                    if 0 <= yy < mf.shape[0] and 0 <= xx < mf.shape[1]:
                        lbl = int(mf[yy, xx])
                        if lbl > 0:
                            region = mf == lbl
                if region is not None and np.any(region):
                    masks_tracked = np.array(masks_tracked, copy=True)
                    masks_tracked[int(frame)][region] = int(manual_tid)
        applied += 1
    return tracks_df, edges_df, masks_tracked, applied


def apply_lineage_curation_to_run(
    curation: dict[str, Any],
    *,
    tracks_df: Any,
    edges_df: Any,
    masks_tracked: Any,
    masks_input: Any = None,
) -> tuple[Any, Any, Any, dict[tuple[int, int], str], dict[str, int]]:
    """
    Apply saved lineage curation onto a fresh Trackastra run.

    Returns updated (tracks_df, edges_df, masks_tracked), division review map, and stats.
    """
    import pandas as pd

    stats = {"manual_edits_applied": 0, "division_reviews_applied": 0, "divisions_dropped": 0}
    review_map: dict[tuple[int, int], str] = {}
    if not curation:
        return tracks_df, edges_df, masks_tracked, review_map, stats

    edits = curation.get("manual_lineage_edits") or []
    if isinstance(edits, list) and edits:
        tracks_df, edges_df, masks_tracked, n_manual = _apply_manual_lineage_edits(
            edits,
            tracks_df=tracks_df,
            edges_df=edges_df,
            masks_tracked=masks_tracked,
            masks_input=masks_input,
        )
        stats["manual_edits_applied"] = int(n_manual)

    division_reviews = curation.get("division_reviews") or []
    if isinstance(division_reviews, list):
        for rev in division_reviews:
            if not isinstance(rev, dict):
                continue
            status = str(rev.get("review_status", REVIEW_STATUS_CANDIDATE)).lower()
            if status not in VALID_REVIEW_STATUSES:
                status = REVIEW_STATUS_CANDIDATE
            key = _match_division_review_to_edge(rev, edges_df, tracks_df)
            if key is None:
                continue
            review_map[key] = status
            stats["division_reviews_applied"] += 1

    if review_map and edges_df is not None and hasattr(edges_df, "columns") and len(edges_df) > 0:
        kept: list[dict[str, Any]] = []
        for er in edges_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            if p is None or d is None:
                continue
            link_status = str(getattr(er, "link_status", LINK_STATUS_CANDIDATE)).lower()
            if link_status == "manual":
                kept.append(er._asdict())
                continue
            status = review_map.get((p, d))
            if status == REVIEW_STATUS_REJECTED:
                stats["divisions_dropped"] += 1
                continue
            kept.append(er._asdict())
        if kept:
            edges_df = pd.DataFrame(kept)
        else:
            edges_df = pd.DataFrame(columns=list(edges_df.columns))

    return tracks_df, edges_df, masks_tracked, review_map, stats


def apply_curation_review_map_to_events(events_df: Any, review_map: dict[tuple[int, int], str]) -> Any:
    """Overlay saved division review decisions onto a freshly built events table."""
    if events_df is None or not hasattr(events_df, "columns") or len(events_df) == 0:
        return events_df
    if not review_map:
        return events_df
    out = events_df.copy()
    if "review_status" not in out.columns:
        out["review_status"] = REVIEW_STATUS_CANDIDATE
    for i, row in out.iterrows():
        key = (int(row["parent_track_id"]), int(row["daughter_track_id"]))
        if key in review_map:
            out.at[i, "review_status"] = review_map[key]
    return out
