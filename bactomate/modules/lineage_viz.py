"""
Lineage tree layout and static figure export (GUI-agnostic).

Used by lineage_inspector and TrackingWorker exports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

LINEAGE_TREE_NODE_COLS = ("track_id", "lineage_id", "generation", "x", "y", "parent_track_id")
LINEAGE_TREE_EDGE_COLS = ("parent_track_id", "daughter_track_id", "division_frame")


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def build_lineage_tree_layout(edges_df: Any, tracks_meta_df: Any) -> tuple[Any, Any]:
    """
    Compute simple top-down tree positions for matplotlib rendering.

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

    by_lineage: dict[int, list[int]] = {}
    for row in meta.itertuples(index=False):
        tid = _safe_int(getattr(row, "track_id", None))
        lid = _safe_int(getattr(row, "lineage_id", None))
        if tid is None:
            continue
        by_lineage.setdefault(lid if lid is not None else tid, []).append(tid)

    node_rows: list[dict[str, Any]] = []
    for lid, tids in sorted(by_lineage.items(), key=lambda x: int(x[0])):
        sub = meta[meta["track_id"].isin(tids)].copy()
        if len(sub) == 0:
            continue
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
                        "x": float(i) - (n - 1) / 2.0,
                        "y": -float(gen),
                        "parent_track_id": row.get("parent_track_id"),
                    }
                )

    nodes_df = pd.DataFrame(node_rows, columns=list(LINEAGE_TREE_NODE_COLS)) if node_rows else empty_nodes
    edge_rows: list[dict[str, Any]] = []
    if edges_df is not None and hasattr(edges_df, "columns") and len(edges_df) > 0:
        for er in edges_df.itertuples(index=False):
            p = _safe_int(getattr(er, "parent_track_id", None))
            d = _safe_int(getattr(er, "daughter_track_id", None))
            div_f = _safe_int(getattr(er, "division_frame", None))
            if p is not None and d is not None:
                edge_rows.append(
                    {
                        "parent_track_id": p,
                        "daughter_track_id": d,
                        "division_frame": div_f,
                    }
                )
    edges_out = pd.DataFrame(edge_rows, columns=list(LINEAGE_TREE_EDGE_COLS)) if edge_rows else empty_edges
    return nodes_df, edges_out


def plot_lineage_tree(
    ax: Any,
    nodes_df: Any,
    edges_df: Any,
    *,
    founder_colors: Optional[dict[int, tuple[float, float, float]]] = None,
    highlight_track_id: Optional[int] = None,
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
        ax.text(x, y + 0.08, str(tid), ha="center", va="bottom", fontsize=7, color="#e8eef6")

    ax.set_title(title)
    ax.set_xlabel("Branch position")
    ax.set_ylabel("Generation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def plot_lineage_growth(
    ax: Any,
    growth_ts_df: Any,
    *,
    lineage_id: Optional[int] = None,
    log_scale: bool = False,
    title: str = "Lineage growth",
) -> None:
    """Plot total lineage area over time."""
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
    y = sub["total_area"].astype(float)
    if log_scale:
        y = np.log(np.clip(y, 1e-9, None))
        ylab = "log(total area)"
    else:
        ylab = "Total area"
    ax.plot(sub["frame"], y, marker="o", markersize=4, color="#4da3ff")
    ax.set_xlabel("Frame (0-based)")
    ax.set_ylabel(ylab)
    ax.set_title(title if lineage_id is None else f"{title} — lineage {lineage_id}")
    ax.grid(True, alpha=0.25)


def export_lineage_figures(
    output_dir: Path,
    tables: dict[str, Any],
    *,
    stem: str,
    dpi: int = 120,
) -> list[Path]:
    """Write PNG overview figures: tree, growth, division events."""
    from matplotlib.figure import Figure
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    meta = tables.get("lineage_tracks_meta")
    edges = tables.get("lineage_edges")
    growth_ts = tables.get("lineage_growth_timeseries")
    events = tables.get("lineage_events")

    nodes, tree_edges = build_lineage_tree_layout(edges, meta)
    if len(nodes) > 0:
        fig = Figure(figsize=(6, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        plot_lineage_tree(ax, nodes, tree_edges, title=f"{stem} — lineage tree")
        p = output_dir / f"{stem}_lineage_tree.png"
        fig.savefig(p, dpi=dpi)
        written.append(p)

    if growth_ts is not None and isinstance(growth_ts, pd.DataFrame) and len(growth_ts) > 0:
        fig = Figure(figsize=(8, 3), tight_layout=True)
        axes = fig.subplots(1, 2)
        lids = sorted(growth_ts["lineage_id"].dropna().unique().tolist())[:6] if "lineage_id" in growth_ts.columns else [None]
        if lids and lids[0] is not None:
            plot_lineage_growth(axes[0], growth_ts, lineage_id=int(lids[0]), title="Lineage growth")
            plot_lineage_growth(axes[1], growth_ts, lineage_id=int(lids[0]), log_scale=True, title="Log growth")
        else:
            plot_lineage_growth(axes[0], growth_ts, title="Lineage growth")
            axes[1].set_visible(False)
        p = output_dir / f"{stem}_lineage_growth.png"
        fig.savefig(p, dpi=dpi)
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
        fig.savefig(p, dpi=dpi)
        written.append(p)

    return written
