"""
Lineage tree layout and static figure export (GUI-agnostic).

Used by lineage_inspector and TrackingWorker exports.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

LINEAGE_TREE_NODE_COLS = ("track_id", "lineage_id", "generation", "x", "y", "parent_track_id")
LINEAGE_TREE_EDGE_COLS = ("parent_track_id", "daughter_track_id", "division_frame")
LINEAGE_TREE_FAMILY_GAP = 1.5
MAX_COMBINED_TREE_LINEAGES = 6
MIN_LINEAGE_EXPORT_DPI = 600
def _lineage_export_dpi(dpi: int | float | None) -> int:
    """Clamp lineage PNG export DPI to a publication-quality minimum."""
    try:
        requested = int(round(float(dpi))) if dpi is not None else MIN_LINEAGE_EXPORT_DPI
    except (TypeError, ValueError):
        requested = MIN_LINEAGE_EXPORT_DPI
    return max(MIN_LINEAGE_EXPORT_DPI, requested)
def _save_lineage_figure(fig: Any, path: Path, *, dpi: int) -> None:
    """Save a lineage figure and release its memory promptly."""
    try:
        fig.savefig(path, dpi=dpi)
    finally:
        try:
            fig.clear()
        except Exception:
            pass


LINEAGE_REPORT_SHEET_TO_KEY: dict[str, str] = {
    "Lineage Summary": "lineage_summary",
    "Division Events": "lineage_events",
    "Track Lineage": "lineage_tracks_meta",
    "Growth Timeseries": "lineage_growth_timeseries",
    "Lineage Points": "lineage_points",
    "Growth": "lineage_growth",
    "QC Flags": "lineage_qc",
    "Edges": "lineage_edges",
    "Measurements": "lineage_measurements",
    "Tree Nodes": "lineage_tree_nodes",
}


def lineage_area_unit_label(growth_ts_df: Any) -> str:
    """Return the display unit for lineage total_area values."""
    if growth_ts_df is not None and hasattr(growth_ts_df, "columns") and "metric_used" in growth_ts_df.columns:
        metric = " ".join(str(v) for v in growth_ts_df["metric_used"].dropna().unique().tolist()).lower()
        if "µm" in metric or "um" in metric or "mask_um2" in metric or "analysis:" in metric:
            return "µm²"
        if "mask_px" in metric or "manual_mask_px" in metric:
            return "px"
    return "px"


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _apply_integer_frame_xaxis(ax: Any, frames: Any = None) -> None:
    """Use whole-number frame ticks (frames are discrete, not fractional)."""
    from matplotlib.ticker import MaxNLocator

    if frames is not None:
        arr = np.asarray(frames, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            lo = int(np.floor(arr.min()))
            hi = int(np.ceil(arr.max()))
            if lo == hi:
                hi = lo + 1
            ax.set_xlim(lo - 0.4, hi + 0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def build_lineage_tree_layout(
    edges_df: Any,
    tracks_meta_df: Any,
    *,
    lineage_ids: Optional[list[int]] = None,
    combine_families: bool = False,
) -> tuple[Any, Any]:
    """
    Compute simple top-down tree positions for matplotlib rendering.

    By default, when ``lineage_ids`` is omitted, every family is laid out with local
    coordinates (filter by ``lineage_id`` before plotting one tree). Pass a single
    ``lineage_ids`` entry to build one family. Pass ``combine_families=True`` to place
    up to :data:`MAX_COMBINED_TREE_LINEAGES` families side by side for small overviews.

    Returns (nodes_df, edges_df) with columns suitable for plotting.
    """
    import pandas as pd

    empty_nodes = pd.DataFrame(columns=list(LINEAGE_TREE_NODE_COLS))
    empty_edges = pd.DataFrame(columns=list(LINEAGE_TREE_EDGE_COLS))
    if tracks_meta_df is None or not hasattr(tracks_meta_df, "columns") or len(tracks_meta_df) == 0:
        return empty_nodes, empty_edges
    if "track_id" not in tracks_meta_df.columns:
        return empty_nodes, empty_edges

    meta = tracks_meta_df.copy()
    if "generation" not in meta.columns:
        meta["generation"] = 0
    if "lineage_id" not in meta.columns:
        meta["lineage_id"] = meta["track_id"]
    if "parent_track_id" not in meta.columns:
        meta["parent_track_id"] = None

    if lineage_ids is not None:
        lid_set = {int(x) for x in lineage_ids}
        meta = meta[meta["lineage_id"].isin(lid_set)]

    by_lineage: dict[int, list[int]] = {}
    for row in meta.itertuples(index=False):
        tid = _safe_int(getattr(row, "track_id", None))
        lid = _safe_int(getattr(row, "lineage_id", None))
        if tid is None:
            continue
        by_lineage.setdefault(lid if lid is not None else tid, []).append(tid)

    sorted_families = sorted(by_lineage.items(), key=lambda x: int(x[0]))
    if combine_families:
        families = sorted_families[:MAX_COMBINED_TREE_LINEAGES]
        use_offset = len(families) > 1
    else:
        families = sorted_families
        use_offset = False

    lineage_offsets: dict[int, float] = {}
    cursor = 0.0
    if use_offset:
        for lid, tids in families:
            sub = meta[meta["track_id"].isin(tids)].copy()
            if len(sub) == 0:
                continue
            lineage_offsets[int(lid)] = cursor
            max_gen_width = 1
            for _gen, gdf in sub.groupby("generation"):
                max_gen_width = max(max_gen_width, len(gdf))
            cursor += float(max_gen_width) + LINEAGE_TREE_FAMILY_GAP

    node_rows: list[dict[str, Any]] = []
    for lid, tids in families:
        sub = meta[meta["track_id"].isin(tids)].copy()
        if len(sub) == 0:
            continue
        x_offset = lineage_offsets.get(int(lid), 0.0) if use_offset else 0.0
        for gen, gdf in sub.groupby("generation"):
            ids = sorted(int(x) for x in gdf["track_id"].tolist())
            n = max(1, len(ids))
            for i, tid in enumerate(ids):
                row = sub[sub["track_id"] == tid].iloc[0]
                node_rows.append(
                    {
                        "track_id": int(tid),
                        "lineage_id": int(lid),
                        "generation": int(gen),
                        "x": float(i) - (n - 1) / 2.0 + x_offset,
                        "y": float(gen),
                        "parent_track_id": row.get("parent_track_id"),
                    }
                )

    nodes_df = pd.DataFrame(node_rows, columns=list(LINEAGE_TREE_NODE_COLS)) if node_rows else empty_nodes
    visible_tids = set(int(t) for t in nodes_df["track_id"].tolist()) if len(nodes_df) > 0 else set()
    edge_rows: list[dict[str, Any]] = []
    if edges_df is not None and hasattr(edges_df, "columns") and len(edges_df) > 0:
        for er in edges_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            div_f = _safe_int(getattr(er, "division_frame", None))
            if p is not None and d is not None and p in visible_tids and d in visible_tids:
                edge_rows.append(
                    {
                        "parent_track_id": p,
                        "daughter_track_id": d,
                        "division_frame": div_f,
                    }
                )
    edges_out = pd.DataFrame(edge_rows, columns=list(LINEAGE_TREE_EDGE_COLS)) if edge_rows else empty_edges
    return nodes_df, edges_out


def filter_lineage_tree_edges(edges_df: Any, track_ids: set[int]) -> Any:
    """Keep only edges whose endpoints are in ``track_ids``."""
    import pandas as pd

    if edges_df is None or not isinstance(edges_df, pd.DataFrame) or len(edges_df) == 0:
        return edges_df
    if not track_ids:
        return edges_df.iloc[0:0]
    return edges_df[
        edges_df["parent_track_id"].isin(track_ids) & edges_df["daughter_track_id"].isin(track_ids)
    ]


def plot_lineage_tree(
    ax: Any,
    nodes_df: Any,
    edges_df: Any,
    *,
    founder_colors: Optional[dict[int, tuple[float, float, float]]] = None,
    highlight_track_id: Optional[int] = None,
    track_labels: Optional[dict[int, str]] = None,
    title: str = "Lineage tree",
) -> None:
    """Draw a simple founder-colored lineage tree on a matplotlib Axes."""
    import pandas as pd

    ax.clear()
    if nodes_df is None or not isinstance(nodes_df, pd.DataFrame) or len(nodes_df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No lineage tree", ha="center", va="center", transform=ax.transAxes)
        return

    pos = {int(r.track_id): (float(r.x), float(r.y)) for r in nodes_df.itertuples(index=False)}
    lid_by_tid = {int(r.track_id): int(r.lineage_id) for r in nodes_df.itertuples(index=False)}
    gen_by_tid = {int(r.track_id): int(r.generation) for r in nodes_df.itertuples(index=False)}

    from matplotlib import colormaps

    cmap = colormaps.get_cmap("tab20")
    founders = sorted(set(lid_by_tid.values()))
    founder_color = {}
    for i, fid in enumerate(founders):
        c = cmap((i % 20) / 20.0)
        founder_color[int(fid)] = (float(c[0]), float(c[1]), float(c[2]))
    if founder_colors:
        founder_color.update(founder_colors)

    if edges_df is not None and isinstance(edges_df, pd.DataFrame) and len(edges_df) > 0:
        for er in edges_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            if p is None or d is None or p not in pos or d not in pos:
                continue
            x1, y1 = pos[p]
            x2, y2 = pos[d]
            ax.plot([x1, x2], [y1, y2], color="#888888", linewidth=1.2, zorder=1)

    for tid, (x, y) in pos.items():
        lid = lid_by_tid.get(tid, tid)
        color = founder_color.get(lid, (0.5, 0.5, 0.5))
        size = 90 if highlight_track_id is not None and int(tid) == int(highlight_track_id) else 55
        edge = "yellow" if highlight_track_id is not None and int(tid) == int(highlight_track_id) else "white"
        ax.scatter([x], [y], s=size, c=[color], edgecolors=edge, linewidths=1.5, zorder=2)
        node_label = track_labels.get(int(tid), str(tid)) if track_labels else str(tid)
        ax.text(x, y + 0.08, node_label, ha="center", va="bottom", fontsize=7, color="#e8eef6")

    ax.set_title(title)
    ax.set_xlabel("Branch position")
    ax.set_ylabel("Generation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def _division_events_for_lineage(
    events_df: Any,
    *,
    lineage_id: Optional[int] = None,
    include_candidates: bool = True,
) -> Any:
    """Filter division events for overlay markers."""
    import pandas as pd

    if events_df is None or not isinstance(events_df, pd.DataFrame) or len(events_df) == 0:
        return pd.DataFrame()
    ev = events_df.copy()
    if lineage_id is not None and "lineage_id" in ev.columns:
        ev = ev[ev["lineage_id"].astype(int) == int(lineage_id)]
    if "review_status" in ev.columns:
        ev = ev[ev["review_status"].astype(str).str.lower() != "rejected"]
        if not include_candidates:
            ev = ev[ev["review_status"].astype(str).str.lower() == "accepted"]
    if "division_frame" not in ev.columns:
        return pd.DataFrame()
    return ev


def plot_lineage_growth(
    ax: Any,
    growth_ts_df: Any,
    *,
    lineage_id: Optional[int] = None,
    log_scale: bool = False,
    title: str = "Lineage growth",
    events_df: Any = None,
    show_division_markers: bool = True,
    include_candidate_divisions: bool = True,
) -> None:
    """Plot total lineage area over time with optional division markers."""
    import pandas as pd

    ax.clear()
    if growth_ts_df is None or not isinstance(growth_ts_df, pd.DataFrame) or len(growth_ts_df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No growth data", ha="center", va="center", transform=ax.transAxes)
        return
    sub = growth_ts_df.copy()
    if lineage_id is not None and "lineage_id" in sub.columns:
        sub = sub[sub["lineage_id"] == int(lineage_id)]
    if len(sub) == 0 or "frame" not in sub.columns or "total_area" not in sub.columns:
        ax.set_title(title)
        return
    sub = sub.sort_values("frame")
    unit = lineage_area_unit_label(sub)
    frames = sub["frame"].astype(float).to_numpy() + 1.0
    y_raw = sub["total_area"].astype(float).to_numpy()
    if log_scale:
        y = np.log(np.clip(y_raw, 1e-9, None))
        ylab = f"log(total area [{unit}])"
    else:
        y = y_raw
        ylab = f"Total area ({unit})"
    ax.plot(frames, y, marker="o", markersize=4, color="#4da3ff", label="Total area", zorder=2)
    if show_division_markers and events_df is not None:
        ev = _division_events_for_lineage(
            events_df,
            lineage_id=lineage_id,
            include_candidates=include_candidate_divisions,
        )
        if len(ev) > 0:
            frame_to_y = {float(f): float(v) for f, v in zip(frames, y)}
            legend_seen: set[str] = set()
            for er in ev.itertuples(index=False):
                div_f = _safe_int(getattr(er, "division_frame", None))
                if div_f is None:
                    continue
                status = str(getattr(er, "review_status", "candidate")).lower()
                color = "#6fcf97" if status == "accepted" else "#f2c94c" if status == "flagged" else "#ff9f43"
                label = f"{status} division"
                legend_label = label if label not in legend_seen else None
                legend_seen.add(label)
                div_x = float(div_f) + 1.0
                y_at = frame_to_y.get(div_x)
                if y_at is None and len(frames) > 0:
                    y_at = float(np.interp(div_x, frames, y))
                if y_at is not None:
                    ax.axvline(div_x, color=color, linestyle="--", alpha=0.55, linewidth=1.0, zorder=1, label=legend_label)
                    ax.scatter([div_x], [y_at], s=36, c=color, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylab)
    ax.set_title(title if title != "Lineage growth" or lineage_id is None else f"{title} — lineage {lineage_id}")
    _apply_integer_frame_xaxis(ax, frames)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, loc="best")


def plot_lineage_cell_cycle_area(
    ax: Any,
    area_df: Any,
    edges_df: Any = None,
    *,
    lineage_id: Optional[int] = None,
    label_map: Optional[dict[int, str]] = None,
    events_df: Any = None,
    include_candidate_divisions: bool = True,
    title: str = "Cell-cycle area trace",
) -> None:
    """Plot per-branch cell area through time with mother→daughter connectors."""
    import pandas as pd

    ax.clear()
    if area_df is None or not isinstance(area_df, pd.DataFrame) or len(area_df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No cell area data", ha="center", va="center", transform=ax.transAxes)
        return
    if not {"track_id", "frame", "area"}.issubset(area_df.columns):
        ax.set_title(title)
        return
    sub = area_df.copy()
    if lineage_id is not None and "lineage_id" in sub.columns:
        sub = sub[sub["lineage_id"].astype(int) == int(lineage_id)]
    if len(sub) == 0:
        ax.set_title(title)
        return
    unit = "px"
    if "area_source" in sub.columns:
        src = " ".join(str(v) for v in sub["area_source"].dropna().unique().tolist()).lower()
        if "um" in src or "µm" in src or "analysis:" in src:
            unit = "µm²"
    sub["frame"] = pd.to_numeric(sub["frame"], errors="coerce")
    sub["area"] = pd.to_numeric(sub["area"], errors="coerce")
    sub = sub.dropna(subset=["frame", "area", "track_id"]).sort_values(["track_id", "frame"])
    if len(sub) == 0:
        ax.set_title(title)
        return

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    tids = sorted(int(t) for t in sub["track_id"].dropna().unique().tolist())
    color_by_tid = {tid: cmap(i % 20) for i, tid in enumerate(tids)}
    point_lookup: dict[tuple[int, int], float] = {}
    for row in sub.itertuples(index=False):
        point_lookup[(int(row.track_id), int(row.frame))] = float(row.area)
    for tid, g in sub.groupby("track_id"):
        tid = int(tid)
        g = g.sort_values("frame")
        lbl = label_map.get(tid, str(tid)) if label_map else str(tid)
        display_frame = g["frame"].astype(float) + 1.0
        ax.plot(
            display_frame,
            g["area"],
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            color=color_by_tid.get(tid, "#4da3ff"),
            alpha=0.9,
            label=lbl,
        )

    if edges_df is not None and isinstance(edges_df, pd.DataFrame) and len(edges_df) > 0:
        for er in edges_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            div_f = _safe_int(getattr(er, "division_frame", None))
            plf = _safe_int(getattr(er, "parent_last_frame", None))
            if p is None or d is None or div_f is None:
                continue
            if lineage_id is not None and "lineage_id" in sub.columns:
                if p not in tids and d not in tids:
                    continue
            parent_frame = plf if plf is not None else max(0, div_f - 1)
            ya = point_lookup.get((p, parent_frame)) or point_lookup.get((p, div_f))
            yb = point_lookup.get((d, div_f))
            if ya is None or yb is None:
                continue
            ax.plot([parent_frame + 1, div_f + 1], [ya, yb], color="#666666", linestyle=":", linewidth=0.9, alpha=0.7)

    if events_df is not None:
        ev = _division_events_for_lineage(
            events_df,
            lineage_id=lineage_id,
            include_candidates=include_candidate_divisions,
        )
        for er in ev.itertuples(index=False):
            div_f = _safe_int(getattr(er, "division_frame", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            if div_f is None or d is None:
                continue
            y = point_lookup.get((d, div_f))
            if y is None:
                continue
            status = str(getattr(er, "review_status", "candidate")).lower()
            color = "#d62728" if status == "accepted" else "#f2c94c"
            marker = "o" if status == "accepted" else "x"
            ax.scatter([div_f + 1], [y], s=30, c=color, marker=marker, edgecolors="white", linewidths=0.6, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"Cell area ({unit})")
    _apply_integer_frame_xaxis(ax, sub["frame"].astype(float) + 1.0)
    ax.grid(True, alpha=0.25)
    if len(tids) <= 10:
        ax.legend(fontsize=7, loc="best")


def plot_lineage_cell_cycle_area_overlay(
    ax: Any,
    area_df: Any,
    edges_df: Any = None,
    *,
    lineage_ids: list[int],
    label_map: Optional[dict[int, str]] = None,
    events_df: Any = None,
    include_candidate_divisions: bool = True,
    title: str = "Cell-cycle area — all lineages",
) -> None:
    """All founder families on one axes; color = lineage family, legend explains colors."""
    import pandas as pd
    from matplotlib import colormaps

    ax.clear()
    if area_df is None or not isinstance(area_df, pd.DataFrame) or len(area_df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No cell area data", ha="center", va="center", transform=ax.transAxes)
        return
    if not {"track_id", "frame", "area"}.issubset(area_df.columns):
        ax.set_title(title)
        return
    if not lineage_ids:
        plot_lineage_cell_cycle_area(
            ax,
            area_df,
            edges_df,
            label_map=label_map,
            events_df=events_df,
            include_candidate_divisions=include_candidate_divisions,
            title=title,
        )
        return

    sub = area_df.copy()
    if "lineage_id" in sub.columns:
        lid_set = {int(x) for x in lineage_ids}
        sub = sub[sub["lineage_id"].astype(int).isin(lid_set)]
    if len(sub) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No cell area data for selected lineages", ha="center", va="center", transform=ax.transAxes)
        return

    unit = "px"
    if "area_source" in sub.columns:
        src = " ".join(str(v) for v in sub["area_source"].dropna().unique().tolist()).lower()
        if "um" in src or "µm" in src or "analysis:" in src:
            unit = "µm²"
    sub["frame"] = pd.to_numeric(sub["frame"], errors="coerce")
    sub["area"] = pd.to_numeric(sub["area"], errors="coerce")
    sub = sub.dropna(subset=["frame", "area", "track_id"]).sort_values(["track_id", "frame"])
    if len(sub) == 0:
        ax.set_title(title)
        return

    cmap = colormaps.get_cmap("tab10")
    point_lookup: dict[tuple[int, int], float] = {}
    for row in sub.itertuples(index=False):
        point_lookup[(int(row.track_id), int(row.frame))] = float(row.area)

    tids = sorted(int(t) for t in sub["track_id"].dropna().unique().tolist())
    tid_to_lid: dict[int, int] = {}
    if "lineage_id" in sub.columns:
        for tid in tids:
            rows = sub[sub["track_id"].astype(int) == tid]["lineage_id"].dropna()
            if len(rows):
                tid_to_lid[tid] = int(rows.iloc[0])

    legend_seen: set[int] = set()
    for i, lid in enumerate(sorted(int(x) for x in lineage_ids)):
        family_color = cmap(i % 10)
        family_tids = [tid for tid in tids if tid_to_lid.get(tid, lid) == int(lid)]
        if not family_tids and int(lid) in tids:
            family_tids = [int(lid)]
        for tid in family_tids:
            g = sub[sub["track_id"].astype(int) == int(tid)].sort_values("frame")
            if len(g) == 0:
                continue
            show_legend = int(lid) not in legend_seen
            if show_legend:
                legend_seen.add(int(lid))
            ax.plot(
                g["frame"].astype(float) + 1.0,
                g["area"],
                marker="o",
                markersize=3.0,
                linewidth=1.1,
                color=family_color,
                alpha=0.88,
                label=f"Lineage {lid}" if show_legend else None,
                zorder=2,
            )

    if edges_df is not None and isinstance(edges_df, pd.DataFrame) and len(edges_df) > 0:
        for er in edges_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            div_f = _safe_int(getattr(er, "division_frame", None))
            plf = _safe_int(getattr(er, "parent_last_frame", None))
            if p is None or d is None or div_f is None:
                continue
            if p not in tids and d not in tids:
                continue
            parent_frame = plf if plf is not None else max(0, div_f - 1)
            ya = point_lookup.get((p, parent_frame)) or point_lookup.get((p, div_f))
            yb = point_lookup.get((d, div_f))
            if ya is None or yb is None:
                continue
            ax.plot(
                [parent_frame + 1, div_f + 1],
                [ya, yb],
                color="#666666",
                linestyle=":",
                linewidth=0.9,
                alpha=0.65,
                zorder=1,
            )

    if events_df is not None:
        for lid in lineage_ids:
            ev = _division_events_for_lineage(
                events_df,
                lineage_id=int(lid),
                include_candidates=include_candidate_divisions,
            )
            for er in ev.itertuples(index=False):
                div_f = _safe_int(getattr(er, "division_frame", None))
                d = _safe_int(getattr(er, "daughter_track_id", None))
                if div_f is None or d is None:
                    continue
                y = point_lookup.get((d, div_f))
                if y is None:
                    continue
                status = str(getattr(er, "review_status", "candidate")).lower()
                color = "#d62728" if status == "accepted" else "#f2c94c"
                marker = "o" if status == "accepted" else "x"
                ax.scatter(
                    [div_f + 1],
                    [y],
                    s=28,
                    c=color,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=4,
                )

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"Cell area ({unit})")
    _apply_integer_frame_xaxis(ax, sub["frame"].astype(float) + 1.0)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, loc="best", title="Founder family")


def _parse_numeric_list_cell(value: Any) -> list[float]:
    """Parse a scalar or comma/space-separated list of numbers from analysis export."""
    if value is None:
        return []
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.floating, np.integer)):
        try:
            v = float(value)
            return [v] if np.isfinite(v) else []
        except (TypeError, ValueError):
            return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        v = float(text)
        if np.isfinite(v):
            return [v]
    except ValueError:
        pass
    out: list[float] = []
    for part in re.split(r"[,;]+", text):
        token = part.strip()
        if not token:
            continue
        try:
            v = float(token)
            if np.isfinite(v):
                out.append(v)
        except ValueError:
            continue
    return out


def list_metric_cell_reduce(values: list[float], metric_column: str) -> float:
    """Collapse per-foci list values to one scalar per tracked cell."""
    if not values:
        return float("nan")
    cl = str(metric_column or "").lower()
    if cl.endswith("_mean_intensities") or "_foci_mean_intensities" in cl:
        return float(np.mean(values))
    if cl.endswith("_sizes_px") or "_foci_sizes_px" in cl:
        return float(np.sum(values))
    if cl.endswith("_intensities") or "_foci_intensities" in cl:
        return float(np.sum(values))
    return float(np.mean(values))


def prepare_lineage_metric_numeric_column(df: Any, metric_column: str) -> Any:
    """Return a float Series suitable for lineage metric plots."""
    import pandas as pd

    raw = df[metric_column]
    if pd.api.types.is_numeric_dtype(raw):
        return pd.to_numeric(raw, errors="coerce")
    sample = raw.dropna().head(32)
    needs_parse = False
    for val in sample:
        if isinstance(val, str) and ("," in val or ";" in val):
            needs_parse = True
            break
    if not needs_parse:
        return pd.to_numeric(raw, errors="coerce")
    return raw.map(lambda v: list_metric_cell_reduce(_parse_numeric_list_cell(v), metric_column))


def _metric_plot_title(base: str, lineage_id: Optional[int]) -> str:
    if lineage_id is None:
        return str(base)
    suffix = f"lineage {lineage_id}"
    if suffix in str(base):
        return str(base)
    return f"{base} — {suffix}"


def plot_lineage_metric_timeseries(
    ax: Any,
    lineage_points_df: Any,
    *,
    metric_column: str,
    lineage_id: Optional[int] = None,
    aggregate: str = "sum",
    log_scale: bool = False,
    title: str = "Lineage metric",
    events_df: Any = None,
    show_division_markers: bool = True,
    include_candidate_divisions: bool = True,
) -> None:
    """Plot a per-lineage aggregate of a measurement column over time."""
    import pandas as pd

    ax.clear()
    if lineage_points_df is None or not isinstance(lineage_points_df, pd.DataFrame) or len(lineage_points_df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No measurement data", ha="center", va="center", transform=ax.transAxes)
        return
    if metric_column not in lineage_points_df.columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, f"Column missing: {metric_column}", ha="center", va="center", transform=ax.transAxes)
        return
    sub = lineage_points_df.copy()
    if lineage_id is not None and "lineage_id" in sub.columns:
        sub = sub[sub["lineage_id"].astype(int) == int(lineage_id)]
    if len(sub) == 0 or "frame" not in sub.columns:
        ax.set_title(_metric_plot_title(title, lineage_id))
        return
    sub = sub.copy()
    sub["_metric_value"] = prepare_lineage_metric_numeric_column(sub, metric_column)
    sub = sub.dropna(subset=["_metric_value"])
    if len(sub) == 0:
        ax.set_title(_metric_plot_title(title, lineage_id))
        ax.text(
            0.5,
            0.5,
            "No numeric values for this metric.\n"
            "List columns (comma-separated per-foci values) may be empty for this lineage.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        return
    agg = str(aggregate or "sum").lower()
    if agg == "mean":
        ts = sub.groupby("frame", as_index=False)["_metric_value"].mean()
        ylab = f"Mean {metric_column}"
    elif agg == "max":
        ts = sub.groupby("frame", as_index=False)["_metric_value"].max()
        ylab = f"Max {metric_column}"
    else:
        ts = sub.groupby("frame", as_index=False)["_metric_value"].sum()
        ylab = f"Sum {metric_column}"
    ts = ts.sort_values("frame")
    frames = ts["frame"].astype(float).to_numpy() + 1.0
    y_raw = ts["_metric_value"].astype(float).to_numpy()
    if log_scale:
        y = np.log(np.clip(y_raw, 1e-9, None))
        ylab = f"log({ylab})"
    else:
        y = y_raw
    ax.plot(frames, y, marker="o", markersize=4, color="#bb86fc", label=metric_column, zorder=2)
    if show_division_markers and events_df is not None:
        ev = _division_events_for_lineage(
            events_df,
            lineage_id=lineage_id,
            include_candidates=include_candidate_divisions,
        )
        if len(ev) > 0:
            frame_to_y = {float(f): float(v) for f, v in zip(frames, y)}
            for er in ev.itertuples(index=False):
                div_f = _safe_int(getattr(er, "division_frame", None))
                if div_f is None:
                    continue
                div_x = float(div_f) + 1.0
                y_at = frame_to_y.get(div_x)
                if y_at is None and len(frames) > 0:
                    y_at = float(np.interp(div_x, frames, y))
                if y_at is not None:
                    ax.axvline(div_x, color="#888888", linestyle=":", alpha=0.45, linewidth=0.9, zorder=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylab)
    ax.set_title(_metric_plot_title(title, lineage_id))
    _apply_integer_frame_xaxis(ax, frames)
    ax.grid(True, alpha=0.25)


def _extract_whole_cell_fi_channel(column: str) -> Optional[str]:
    """Return channel label from ``Mean FI Intensity (whole cell) (Red)`` style columns."""
    m = re.match(r"^Mean FI Intensity \(whole cell\) \((.+)\)\s*$", str(column).strip())
    return str(m.group(1)).strip() if m else None


def _is_foci_only_whole_cell_fi_column(column: str, lineage_points_df: Any) -> bool:
    """
  True when whole-cell FI is on a channel that also has foci metrics.

  Analyze FI can run on a *_foci_* TIFF (mean intensity over the cell mask on that
  channel). That is usually not what users want when the channel is only for foci.
    """
    import pandas as pd

    channel = _extract_whole_cell_fi_channel(column)
    if channel is None or lineage_points_df is None or not isinstance(lineage_points_df, pd.DataFrame):
        return False
    cols = {str(c) for c in lineage_points_df.columns}
    markers = (
        f"{channel}_foci_count",
        f"foci_ids_{channel}",
        f"{channel}_foci_mean_intensity",
        f"{channel}_foci_max_intensity",
    )
    return any(m in cols for m in markers)


def _metric_column_has_plottable_data(lineage_points_df: Any, column: str) -> bool:
    """Skip columns that are empty or coloc placeholders with no real pairs."""
    import pandas as pd

    if lineage_points_df is None or not isinstance(lineage_points_df, pd.DataFrame):
        return False
    if column not in lineage_points_df.columns:
        return False
    vals = prepare_lineage_metric_numeric_column(lineage_points_df, column)
    try:
        arr = np.asarray(vals, dtype=float)
    except Exception:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    if "coloc" in str(column).lower():
        return bool(np.any(finite != 0))
    return True


def discover_lineage_metric_columns(lineage_points_df: Any) -> dict[str, list[str]]:
    """Group measurement columns from lineage_points by category."""
    import pandas as pd

    out = {"area": [], "fi": [], "foci": [], "coloc": [], "other": []}
    if lineage_points_df is None or not isinstance(lineage_points_df, pd.DataFrame):
        return out
    skip = {"track_id", "frame", "y", "x", "lineage_id", "cell_id", "area", "area_source"}
    for c in lineage_points_df.columns:
        cl = str(c).lower()
        if cl in skip:
            continue
        if cl.startswith("foci_ids") or "_foci_ids" in cl:
            continue
        if _is_foci_only_whole_cell_fi_column(str(c), lineage_points_df):
            continue
        if not _metric_column_has_plottable_data(lineage_points_df, str(c)):
            continue
        if "area" in cl and "µm" in cl:
            out["area"].append(str(c))
        elif "fi" in cl and "foci" not in cl:
            out["fi"].append(str(c))
        elif "foci" in cl:
            out["foci"].append(str(c))
        elif "coloc" in cl:
            out["coloc"].append(str(c))
        else:
            out["other"].append(str(c))
    for key in out:
        out[key] = sorted(out[key], key=_metric_column_sort_key)
    return out


def _metric_column_sort_key(column: str) -> tuple[int, str]:
    """Prefer cell-level scalar summaries over per-foci list columns in the UI."""
    cl = str(column or "").lower()
    if cl.endswith("_max_intensity") or cl.endswith("_mean_intensity") or cl.endswith("_total_intensity"):
        return (0, str(column))
    if cl.endswith("_mean_intensities") or cl.endswith("_intensities") or cl.endswith("_sizes_px"):
        return (2, str(column))
    return (1, str(column))


def default_metric_aggregate(metric_column: str) -> str:
    """Pick sum vs mean for lineage metric aggregation."""
    cl = str(metric_column or "").lower()
    if cl.endswith("_mean_intensities") or "_foci_mean_intensities" in cl:
        return "mean"
    if cl.endswith("_intensities") or "_foci_intensities" in cl:
        return "sum"
    if cl.endswith("_sizes_px") or "_foci_sizes_px" in cl:
        return "sum"
    if "intensity" in cl or "fi" in cl:
        return "mean"
    if "count" in cl:
        return "sum"
    return "sum"


def lineage_ids_from_tables(tables: dict[str, Any]) -> list[int]:
    """Sorted lineage ids present in summary or growth tables."""
    import pandas as pd

    for key in ("lineage_summary", "lineage_growth_timeseries", "lineage_tracks_meta"):
        df = tables.get(key)
        if df is not None and isinstance(df, pd.DataFrame) and "lineage_id" in df.columns:
            ids = sorted(int(x) for x in df["lineage_id"].dropna().unique().tolist())
            if ids:
                return ids
    return []


def _biology_label_map_from_tables(tables: dict[str, Any]) -> dict[int, str]:
    label_map: dict[int, str] = {}
    labels_df = tables.get("biology_track_labels")
    if labels_df is not None and hasattr(labels_df, "columns") and {"track_id", "biology_label"}.issubset(labels_df.columns):
        for row in labels_df.itertuples(index=False):
            tid = _safe_int(getattr(row, "track_id", None))
            if tid is not None:
                label_map[tid] = str(getattr(row, "biology_label", tid))
        return label_map
    meta = tables.get("lineage_tracks_meta")
    if meta is not None and hasattr(meta, "columns"):
        tid_col = "track_id" if "track_id" in meta.columns else None
        lbl_col = None
        for c in ("Cell_ID", "cell_id", "biology_label"):
            if c in meta.columns:
                lbl_col = c
                break
        if tid_col and lbl_col:
            for row in meta.itertuples(index=False):
                tid = _safe_int(getattr(row, tid_col, None))
                if tid is not None:
                    label_map[tid] = str(getattr(row, lbl_col, tid))
    return label_map


def _founder_map_from_tables(tables: dict[str, Any]) -> dict[int, int]:
    """track_id -> founder lineage_id from lineage tables."""
    founder = tables.get("founder_by_track_curated") or tables.get("founder_by_track") or {}
    if isinstance(founder, dict) and founder:
        return {int(k): int(v) for k, v in founder.items()}
    import pandas as pd

    meta = tables.get("lineage_tracks_meta_curated") or tables.get("lineage_tracks_meta")
    if meta is not None and isinstance(meta, pd.DataFrame) and {"track_id", "lineage_id"}.issubset(meta.columns):
        out: dict[int, int] = {}
        for row in meta.itertuples(index=False):
            tid = _safe_int(getattr(row, "track_id", None))
            lid = _safe_int(getattr(row, "lineage_id", None))
            if tid is not None and lid is not None:
                out[tid] = lid
        return out
    return {}


def _attach_lineage_id_to_area_df(area_df: Any, tables: dict[str, Any]) -> Any:
    """Ensure track_area rows carry lineage_id for per-family filtering."""
    import pandas as pd

    if area_df is None or not isinstance(area_df, pd.DataFrame) or len(area_df) == 0:
        return area_df
    if "lineage_id" in area_df.columns or "track_id" not in area_df.columns:
        return area_df
    founder = _founder_map_from_tables(tables)
    if not founder:
        return area_df
    out = area_df.copy()
    out["lineage_id"] = out["track_id"].map(lambda t: founder.get(int(t), int(t)) if t is not None else None)
    return out


def _track_area_from_tables(tables: dict[str, Any]) -> Any:
    """Return track_area table, synthesizing from lineage_points when needed."""
    import pandas as pd

    area_df = tables.get("track_area")
    if area_df is not None and isinstance(area_df, pd.DataFrame) and len(area_df) > 0:
        return _attach_lineage_id_to_area_df(area_df, tables)
    pts = tables.get("lineage_points")
    if pts is None or not isinstance(pts, pd.DataFrame) or len(pts) == 0:
        return None
    area_col = None
    for c in pts.columns:
        cl = str(c).lower()
        if "area" in cl and ("µm" in cl or "um" in cl):
            area_col = c
            break
    if area_col is None:
        return None
    cols = ["track_id", "frame", area_col]
    if "lineage_id" in pts.columns:
        cols.append("lineage_id")
    out = pts[cols].copy()
    out = out.rename(columns={area_col: "area"})
    out["area_source"] = f"analysis:{area_col}"
    return out


def plot_lineage_growth_overlay(
    ax: Any,
    growth_ts_df: Any,
    lineage_ids: list[int],
    *,
    log_scale: bool = False,
    title: str = "Lineage growth comparison",
    events_df: Any = None,
    include_candidate_divisions: bool = True,
) -> None:
    """Plot total area for multiple lineages on one axes."""
    import pandas as pd

    ax.clear()
    if growth_ts_df is None or not isinstance(growth_ts_df, pd.DataFrame) or len(growth_ts_df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No growth data", ha="center", va="center", transform=ax.transAxes)
        return
    if not lineage_ids:
        plot_lineage_growth(ax, growth_ts_df, log_scale=log_scale, title=title, events_df=events_df)
        return
    from matplotlib import colormaps

    cmap = colormaps.get_cmap("tab10")
    for i, lid in enumerate(lineage_ids):
        sub = growth_ts_df[growth_ts_df["lineage_id"] == int(lid)].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("frame")
        unit = lineage_area_unit_label(sub)
        frames = sub["frame"].astype(float).to_numpy() + 1.0
        y_raw = sub["total_area"].astype(float).to_numpy()
        y = np.log(np.clip(y_raw, 1e-9, None)) if log_scale else y_raw
        color = cmap(i % 10)
        ax.plot(frames, y, marker="o", markersize=3.5, color=color, label=f"Lineage {lid}", zorder=2)
    ylab = f"log(total area)" if log_scale else "Total area"
    if lineage_ids:
        ylab += f" ({lineage_area_unit_label(growth_ts_df)})"
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, loc="best")


def plot_lineage_metric_overlay(
    ax: Any,
    lineage_points_df: Any,
    metric_column: str,
    lineage_ids: list[int],
    *,
    aggregate: str = "sum",
    log_scale: bool = False,
    title: str = "Lineage metric comparison",
) -> None:
    """Plot one metric for multiple lineages on the same axes."""
    import pandas as pd

    ax.clear()
    if lineage_points_df is None or not isinstance(lineage_points_df, pd.DataFrame) or len(lineage_points_df) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No measurement data", ha="center", va="center", transform=ax.transAxes)
        return
    if metric_column not in lineage_points_df.columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, f"Column missing: {metric_column}", ha="center", va="center", transform=ax.transAxes)
        return
    from matplotlib import colormaps

    cmap = colormaps.get_cmap("tab10")
    agg = str(aggregate or "sum").lower()
    for i, lid in enumerate(lineage_ids):
        sub = lineage_points_df[lineage_points_df["lineage_id"].astype(int) == int(lid)].copy()
        if len(sub) == 0:
            continue
        sub["_metric_value"] = prepare_lineage_metric_numeric_column(sub, metric_column)
        sub = sub.dropna(subset=["_metric_value"])
        if len(sub) == 0:
            continue
        if agg == "mean":
            ts = sub.groupby("frame", as_index=False)["_metric_value"].mean()
        elif agg == "max":
            ts = sub.groupby("frame", as_index=False)["_metric_value"].max()
        else:
            ts = sub.groupby("frame", as_index=False)["_metric_value"].sum()
        ts = ts.sort_values("frame")
        frames = ts["frame"].astype(float).to_numpy() + 1.0
        y_raw = ts["_metric_value"].astype(float).to_numpy()
        y = np.log(np.clip(y_raw, 1e-9, None)) if log_scale else y_raw
        ax.plot(frames, y, marker="o", markersize=3.5, color=cmap(i % 10), label=f"Lineage {lid}", zorder=2)
    ax.set_xlabel("Frame")
    ylab = metric_column
    if log_scale:
        ylab = f"log({ylab})"
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, loc="best")


def _safe_metric_stem(metric_column: str) -> str:
    s = re.sub(r"[^\w]+", "_", str(metric_column).strip().lower())
    return s.strip("_")[:48] or "metric"


def export_lineage_analysis_figures(
    output_dir: Path,
    tables: dict[str, Any],
    *,
    stem: str,
    lineage_ids: Optional[list[int]] = None,
    metric_columns: Optional[list[str]] = None,
    dpi: int = MIN_LINEAGE_EXPORT_DPI,
) -> list[Path]:
    """Write per-lineage and overlay comparison figures under figures/analysis/."""
    from matplotlib.figure import Figure
    import pandas as pd

    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    export_dpi = _lineage_export_dpi(dpi)

    lids = [int(x) for x in (lineage_ids or lineage_ids_from_tables(tables))]
    if not lids:
        lids = []

    meta = tables.get("lineage_tracks_meta")
    edges = tables.get("lineage_edges")
    growth_ts = tables.get("lineage_growth_timeseries")
    events = tables.get("lineage_events")
    area_df = _track_area_from_tables(tables)
    pts = tables.get("lineage_points")
    label_map = _biology_label_map_from_tables(tables)

    metrics = list(metric_columns or [])
    if not metrics and pts is not None:
        discovered = discover_lineage_metric_columns(pts)
        metrics = discovered["area"][:1] + discovered["fi"][:2] + discovered["foci"][:2] + discovered["coloc"][:1]

    for lid in lids:
        if meta is not None:
            nodes, tree_edges = build_lineage_tree_layout(edges, meta, lineage_ids=[int(lid)])
            if len(nodes) > 0:
                fig = Figure(figsize=(5.5, 3.6), tight_layout=True)
                ax = fig.add_subplot(111)
                plot_lineage_tree(ax, nodes, tree_edges, track_labels=label_map, title=f"Lineage {lid} — family tree")
                p = analysis_dir / f"lineage_{lid}_tree.png"
                _save_lineage_figure(fig, p, dpi=export_dpi)
                written.append(p)

        if growth_ts is not None and isinstance(growth_ts, pd.DataFrame) and len(growth_ts) > 0:
            fig = Figure(figsize=(8, 3), tight_layout=True)
            axes = fig.subplots(1, 2)
            plot_lineage_growth(axes[0], growth_ts, lineage_id=int(lid), title=f"Growth — lineage {lid}", events_df=events)
            plot_lineage_growth(axes[1], growth_ts, lineage_id=int(lid), log_scale=True, title=f"Log growth — lineage {lid}", events_df=events)
            p = analysis_dir / f"lineage_{lid}_growth.png"
            _save_lineage_figure(fig, p, dpi=export_dpi)
            written.append(p)

        for metric_col in metrics:
            if pts is None or metric_col not in getattr(pts, "columns", []):
                continue
            fig = Figure(figsize=(7, 3), tight_layout=True)
            ax = fig.add_subplot(111)
            plot_lineage_metric_timeseries(
                ax,
                pts,
                metric_column=metric_col,
                lineage_id=int(lid),
                aggregate=default_metric_aggregate(metric_col),
                log_scale="intensity" in metric_col.lower(),
                title=f"{metric_col} — lineage {lid}",
                events_df=events,
            )
            p = analysis_dir / f"lineage_{lid}_{_safe_metric_stem(metric_col)}.png"
            _save_lineage_figure(fig, p, dpi=export_dpi)
            written.append(p)

    if lids and growth_ts is not None and isinstance(growth_ts, pd.DataFrame) and len(growth_ts) > 0:
        fig = Figure(figsize=(8, 3), tight_layout=True)
        axes = fig.subplots(1, 2)
        plot_lineage_growth_overlay(axes[0], growth_ts, lids, title="All lineages — growth", events_df=events)
        plot_lineage_growth_overlay(axes[1], growth_ts, lids, log_scale=True, title="All lineages — log growth", events_df=events)
        p = analysis_dir / f"{stem}_growth_overlay.png"
        _save_lineage_figure(fig, p, dpi=export_dpi)
        written.append(p)

    if lids and area_df is not None and isinstance(area_df, pd.DataFrame) and len(area_df) > 0:
        fig = Figure(figsize=(9, 4.5), tight_layout=True)
        ax = fig.add_subplot(111)
        plot_lineage_cell_cycle_area_overlay(
            ax,
            area_df,
            edges,
            lineage_ids=lids,
            label_map=label_map,
            events_df=events,
            title="Cell-cycle area — all lineages",
        )
        p = analysis_dir / f"{stem}_cell_cycle_area.png"
        _save_lineage_figure(fig, p, dpi=export_dpi)
        written.append(p)

    if lids and pts is not None and metrics:
        for metric_col in metrics:
            if metric_col not in pts.columns:
                continue
            fig = Figure(figsize=(7, 3), tight_layout=True)
            ax = fig.add_subplot(111)
            plot_lineage_metric_overlay(
                ax,
                pts,
                metric_col,
                lids,
                aggregate=default_metric_aggregate(metric_col),
                log_scale="intensity" in metric_col.lower(),
                title=f"Overlay — {metric_col}",
            )
            p = analysis_dir / f"{stem}_overlay_{_safe_metric_stem(metric_col)}.png"
            _save_lineage_figure(fig, p, dpi=export_dpi)
            written.append(p)

    if len(lids) > 1 and len(lids) <= MAX_COMBINED_TREE_LINEAGES and meta is not None:
        nodes, tree_edges = build_lineage_tree_layout(edges, meta, lineage_ids=lids, combine_families=True)
        if len(nodes) > 0:
            fig = Figure(figsize=(max(6, 2.5 * len(lids)), 4), tight_layout=True)
            ax = fig.add_subplot(111)
            plot_lineage_tree(ax, nodes, tree_edges, track_labels=label_map, title=f"{stem} — families 1–{len(lids)}")
            p = analysis_dir / f"{stem}_lineage_tree_combined.png"
            _save_lineage_figure(fig, p, dpi=export_dpi)
            written.append(p)

    return written


def export_lineage_figures(
    output_dir: Path,
    tables: dict[str, Any],
    *,
    stem: str,
    dpi: int = MIN_LINEAGE_EXPORT_DPI,
) -> list[Path]:
    """Write PNG overview figures: tree, growth overlays, division events, and analysis/."""
    from matplotlib.figure import Figure
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    export_dpi = _lineage_export_dpi(dpi)

    meta = tables.get("lineage_tracks_meta")
    edges = tables.get("lineage_edges")
    growth_ts = tables.get("lineage_growth_timeseries")
    events = tables.get("lineage_events")
    area_df = _track_area_from_tables(tables)
    label_map = _biology_label_map_from_tables(tables)
    lids = lineage_ids_from_tables(tables)
    n_lids = len(lids)

    if n_lids <= MAX_COMBINED_TREE_LINEAGES:
        nodes, tree_edges = build_lineage_tree_layout(
            edges,
            meta,
            lineage_ids=lids if lids else None,
            combine_families=n_lids > 1,
        )
        tree_title = f"{stem} — lineage tree"
        if n_lids > 1:
            tree_title = f"{stem} — lineage tree ({n_lids} families)"
    elif lids:
        nodes, tree_edges = build_lineage_tree_layout(edges, meta, lineage_ids=[int(lids[0])])
        tree_title = f"{stem} — lineage {lids[0]} (1 of {n_lids}; see figures/analysis/ for all)"
    else:
        nodes, tree_edges = build_lineage_tree_layout(edges, meta)

    if len(nodes) > 0:
        fig = Figure(figsize=(6, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        plot_lineage_tree(ax, nodes, tree_edges, track_labels=label_map, title=tree_title)
        p = output_dir / f"{stem}_lineage_tree.png"
        _save_lineage_figure(fig, p, dpi=export_dpi)
        written.append(p)

    if growth_ts is not None and isinstance(growth_ts, pd.DataFrame) and len(growth_ts) > 0:
        fig = Figure(figsize=(8, 3), tight_layout=True)
        axes = fig.subplots(1, 2)
        if lids:
            plot_lineage_growth_overlay(axes[0], growth_ts, lids, title="Lineage growth (all families)", events_df=events)
            plot_lineage_growth_overlay(axes[1], growth_ts, lids, log_scale=True, title="Log growth (all families)", events_df=events)
        else:
            plot_lineage_growth(axes[0], growth_ts, title="Lineage growth")
            plot_lineage_growth(axes[1], growth_ts, log_scale=True, title="Log growth")
        p = output_dir / f"{stem}_lineage_growth.png"
        _save_lineage_figure(fig, p, dpi=export_dpi)
        written.append(p)

    if area_df is not None and isinstance(area_df, pd.DataFrame) and len(area_df) > 0:
        fig = Figure(figsize=(9, 4.5), tight_layout=True)
        ax = fig.add_subplot(111)
        if lids:
            plot_lineage_cell_cycle_area_overlay(
                ax,
                area_df,
                edges,
                lineage_ids=lids,
                label_map=label_map,
                events_df=events,
                title="Cell-cycle area — all lineages",
            )
        else:
            plot_lineage_cell_cycle_area(
                ax,
                area_df,
                edges,
                label_map=label_map,
                events_df=events,
                title="Cell-cycle area trace",
            )
        p = output_dir / f"{stem}_cell_cycle_area.png"
        _save_lineage_figure(fig, p, dpi=export_dpi)
        written.append(p)

    if events is not None and isinstance(events, pd.DataFrame) and len(events) > 0:
        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)
        if "area_asymmetry" in events.columns:
            valid = events["area_asymmetry"].dropna()
            if len(valid) > 0:
                ax.hist(valid.astype(float), bins=min(12, max(3, len(valid))), color="#6fcf97", edgecolor="#333")
                ax.set_xlabel("Daughter / mother area ratio")
                ax.set_ylabel("Count")
                ax.set_title(f"{stem} — division asymmetry")
        p = output_dir / f"{stem}_division_events.png"
        _save_lineage_figure(fig, p, dpi=export_dpi)
        written.append(p)

    try:
        written.extend(export_lineage_analysis_figures(output_dir, tables, stem=stem, lineage_ids=lids, dpi=export_dpi))
    except Exception:
        pass

    return written
