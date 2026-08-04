"""Post-Trackastra proximity stitch: merge track fragments across short detection gaps."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def max_jump_px_from_params(params: dict[str, Any]) -> float:
    """Convert stitch max-jump spin value to pixels using calibration when available."""
    jump = float(params.get("stitch_max_jump", 2.0) or 2.0)
    pixel_size = float(params.get("pixel_size", 0.0) or 0.0)
    if pixel_size > 0 and bool(params.get("use_physical_units", True)):
        return float(jump) / float(pixel_size)
    return float(jump)


def remap_tracked_mask_labels(
    masks_tracked: Optional[np.ndarray],
    id_map: dict[int, int],
) -> Optional[np.ndarray]:
    """Remap tracked-mask label values according to ``old_id -> keep_id``."""
    if masks_tracked is None or not id_map:
        return masks_tracked
    arr = np.asarray(masks_tracked)
    out = arr.copy()
    for old_id, new_id in id_map.items():
        if int(old_id) == int(new_id):
            continue
        out[arr == int(old_id)] = int(new_id)
    return out


def stitch_tracks_by_proximity(
    tracks_df: pd.DataFrame,
    *,
    max_gap_frames: int = 2,
    max_jump_px: float = 2.0,
) -> tuple[pd.DataFrame, int, list[tuple[int, int]], dict[int, int]]:
    """
    Merge track fragments when one ends and another starts nearby after a short hole.

    ``gap = first_frame(B) - last_frame(A) - 1`` must satisfy
    ``1 <= gap <= max_gap_frames``. Endpoint distance must be ``<= max_jump_px``.

    Returns ``(stitched_df, n_merges, merge_pairs, id_map)`` where each pair is
    ``(kept_id, discarded_id)`` and ``id_map`` maps discarded -> final kept id.
    """
    need = {"track_id", "frame", "y", "x"}
    if tracks_df is None or getattr(tracks_df, "empty", True):
        empty = tracks_df if tracks_df is not None else pd.DataFrame()
        return empty, 0, [], {}
    if not need.issubset(set(tracks_df.columns)):
        return tracks_df.copy(), 0, [], {}

    max_gap = int(max(0, max_gap_frames))
    max_jump = float(max(0.0, max_jump_px))
    if max_gap < 1 or max_jump <= 0:
        return tracks_df.copy(), 0, [], {}

    df = tracks_df.copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)

    endpoints: dict[int, dict[str, Any]] = {}
    frames_by_tid: dict[int, set[int]] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        tid_i = int(tid)
        first = g.iloc[0]
        last = g.iloc[-1]
        endpoints[tid_i] = {
            "first_frame": int(first["frame"]),
            "last_frame": int(last["frame"]),
            "first_y": float(first["y"]),
            "first_x": float(first["x"]),
            "last_y": float(last["y"]),
            "last_x": float(last["x"]),
        }
        frames_by_tid[tid_i] = set(int(f) for f in g["frame"].tolist())

    tids = list(endpoints.keys())
    candidates: list[tuple[float, int, int]] = []
    for a in tids:
        ea = endpoints[a]
        for b in tids:
            if a == b:
                continue
            eb = endpoints[b]
            gap = int(eb["first_frame"]) - int(ea["last_frame"]) - 1
            if gap < 1 or gap > max_gap:
                continue
            dist = float(
                np.hypot(eb["first_y"] - ea["last_y"], eb["first_x"] - ea["last_x"])
            )
            if dist > max_jump:
                continue
            candidates.append((dist, a, b))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    parent: dict[int, int] = {t: t for t in tids}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _root_first_frame(root: int) -> int:
        return min(endpoints[t]["first_frame"] for t in tids if _find(t) == root)

    def _frames_of_root(root: int) -> set[int]:
        out: set[int] = set()
        for tid, frames in frames_by_tid.items():
            if _find(tid) == root:
                out |= frames
        return out

    used_as_end: set[int] = set()
    used_as_start: set[int] = set()
    merge_pairs: list[tuple[int, int]] = []

    for _dist, a, b in candidates:
        if a in used_as_end or b in used_as_start:
            continue
        ra, rb = _find(a), _find(b)
        if ra == rb:
            continue
        fa, fb = _root_first_frame(ra), _root_first_frame(rb)
        if fa < fb:
            keep, discard = ra, rb
        elif fb < fa:
            keep, discard = rb, ra
        else:
            keep, discard = (ra, rb) if ra <= rb else (rb, ra)

        if _frames_of_root(keep) & _frames_of_root(discard):
            continue

        parent[discard] = keep
        used_as_end.add(a)
        used_as_start.add(b)
        merge_pairs.append((int(keep), int(discard)))

    if not merge_pairs:
        return df, 0, [], {}

    resolved: dict[int, int] = {t: _find(t) for t in tids}
    id_map = {old: new for old, new in resolved.items() if old != new}
    if not id_map:
        return df, 0, [], {}

    out = df.copy()
    out["track_id"] = out["track_id"].map(lambda t: int(resolved.get(int(t), int(t))))
    return out, len(merge_pairs), merge_pairs, id_map


def remap_napari_lineage_graph(
    napari_graph: Any,
    id_map: dict[int, int],
) -> Any:
    """Remap napari lineage graph keys and parent lists through ``id_map``.

    Napari convention: ``graph[child_track_id] = [parent_track_id, ...]``.
    Self-parent edges after merge are dropped.
    """
    if not id_map or napari_graph is None:
        return napari_graph

    def _map_tid(tid: Any) -> int | None:
        try:
            t = int(tid)
        except (TypeError, ValueError):
            return None
        return int(id_map.get(t, t))

    if isinstance(napari_graph, dict):
        items = list(napari_graph.items())
    else:
        try:
            items = list(napari_graph.items())
        except Exception:
            return napari_graph

    out: dict[int, list[int]] = {}
    for child_raw, parents in items:
        child = _map_tid(child_raw)
        if child is None:
            continue
        if parents is None:
            plist: list[int] = []
        elif isinstance(parents, (int, np.integer)):
            mapped = _map_tid(parents)
            plist = [mapped] if mapped is not None else []
        else:
            try:
                plist = []
                for p in parents:
                    mapped = _map_tid(p)
                    if mapped is not None:
                        plist.append(mapped)
            except TypeError:
                plist = []
        plist = [p for p in plist if p != child]
        # Prefer first parents if multiple children collapse onto one key.
        if child in out:
            seen = set(out[child])
            for p in plist:
                if p not in seen:
                    out[child].append(p)
                    seen.add(p)
        else:
            out[child] = list(plist)
    return out


def remap_lineage_edges_df(edges_df: Any, id_map: dict[int, int]) -> Any:
    """Remap parent/daughter track IDs in a lineage edges table; drop self-edges."""
    if edges_df is None or not id_map or not hasattr(edges_df, "copy"):
        return edges_df
    if not {"parent_track_id", "daughter_track_id"}.issubset(getattr(edges_df, "columns", [])):
        return edges_df
    out = edges_df.copy()
    out["parent_track_id"] = out["parent_track_id"].map(
        lambda t: int(id_map.get(int(t), int(t))) if pd.notna(t) else t
    )
    out["daughter_track_id"] = out["daughter_track_id"].map(
        lambda t: int(id_map.get(int(t), int(t))) if pd.notna(t) else t
    )
    out = out[out["parent_track_id"] != out["daughter_track_id"]].reset_index(drop=True)
    return out


def remap_founder_by_track(founder_by_track: Any, id_map: dict[int, int]) -> dict[int, int]:
    """Remap founder map keys and values through ``id_map``."""
    if not id_map or not founder_by_track:
        return dict(founder_by_track or {})
    out: dict[int, int] = {}
    for tid, founder in dict(founder_by_track).items():
        try:
            t = int(id_map.get(int(tid), int(tid)))
            f = int(id_map.get(int(founder), int(founder)))
        except (TypeError, ValueError):
            continue
        out[t] = f
    return out


def track_graph_from_tracks_df(tracks_df: pd.DataFrame) -> Any:
    """Build a simple Trackastra-compatible DiGraph from stitched track detections.

    Chains consecutive detections of each ``track_id`` so CTC export matches remapped
    masks after proximity stitch (the pre-stitch Trackastra graph is stale).
    """
    import networkx as nx

    g = nx.DiGraph()
    need = {"track_id", "frame", "y", "x"}
    if tracks_df is None or not need.issubset(tracks_df.columns) or tracks_df.empty:
        return g
    for tid, group in tracks_df.groupby("track_id", sort=True):
        sub = group.sort_values("frame")
        prev_node = None
        for row in sub.itertuples(index=False):
            frame = int(getattr(row, "frame"))
            y = float(getattr(row, "y"))
            x = float(getattr(row, "x"))
            node = f"{int(tid)}_{frame}"
            g.add_node(node, time=frame, label=int(tid), coords=(y, x))
            if prev_node is not None:
                g.add_edge(prev_node, node)
            prev_node = node
    return g


def apply_stitch_from_params(
    tracks_df: pd.DataFrame,
    masks_tracked: Optional[np.ndarray] = None,
    params: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, Optional[np.ndarray], int, list[tuple[int, int]], dict[int, int]]:
    """Apply proximity stitch when enabled in ``params``; otherwise return inputs unchanged.

    Returns ``(tracks_df, masks, n_merges, pairs, id_map)``.
    """
    params = dict(params or {})
    if not bool(params.get("stitch_enabled", False)):
        return tracks_df, masks_tracked, 0, [], {}
    max_gap = int(params.get("stitch_max_gap_frames", 2) or 2)
    max_jump_px = max_jump_px_from_params(params)
    stitched, n_merges, pairs, id_map = stitch_tracks_by_proximity(
        tracks_df,
        max_gap_frames=max_gap,
        max_jump_px=max_jump_px,
    )
    if n_merges <= 0 or not id_map:
        return stitched, masks_tracked, 0, [], {}
    masks_out = remap_tracked_mask_labels(masks_tracked, id_map)
    return stitched, masks_out, n_merges, pairs, id_map
