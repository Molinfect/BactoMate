"""
Colocalization analysis between pairs of foci channels.

Matching strategy
-----------------
1. **KD-tree ball query** (``cKDTree.query_ball_point``) generates all candidate
   pairs within ``coloc_max_distance`` pixels.
2. **Connected-component decomposition** splits the bipartite candidate graph
   into independent sub-problems so we never build a dense N×M cost matrix
   when foci are spatially clustered.
3. **Linear sum assignment** (Hungarian algorithm) finds the globally optimal
   one-to-one pairing within each component.
4. ``coloc_threshold`` **classifies** each assigned pair as colocalized
   (distance ≤ threshold) or not, without affecting the assignment itself.

All distances are in **pixels**.
"""

from __future__ import annotations

import ast
import concurrent.futures
import itertools
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

LOGGER = logging.getLogger(__name__)

SPARSE_COMPONENT_THRESHOLD = 200_000


@dataclass
class ColocSettings:
    """Colocalization parameters (distances in pixels)."""

    coloc_max_distance: Optional[float] = 8.0
    coloc_threshold: Optional[float] = 4.0
    coloc_filter_boundaries: bool = True
    coloc_edge_tolerance: float = 0.0
    coloc_save_unmatched: bool = False
    coloc_n_randomizations: int = 0
    coloc_randomization_seed: int = 0
    coloc_one_to_one: bool = True
    coloc_mutual_nearest: bool = True


@dataclass
class ColocalizationOutputs:
    coloc_records: List[Dict[str, object]]
    pair_records: List[Dict[str, object]]
    unmatched_records: List[Dict[str, object]]


# ---------------------------------------------------------------------------
# KD-tree candidate generation + optimal one-to-one assignment
# ---------------------------------------------------------------------------

def _kdtree_radius_candidates(
    coords_primary: np.ndarray,
    coords_secondary: np.ndarray,
    radius_px: float,
    *,
    cell_ids_primary: Optional[np.ndarray] = None,
    cell_ids_secondary: Optional[np.ndarray] = None,
    filter_boundaries: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KD-tree ball query returning candidate edges within *radius_px*.

    Coordinates are ``(y, x)`` order.  When *filter_boundaries* is True only
    same-cell candidates with positive cell IDs are kept.

    Returns ``(row_indices, col_indices, distances)`` — all 1-D arrays.
    """
    n_p = coords_primary.shape[0]
    n_s = coords_secondary.shape[0]
    if n_p == 0 or n_s == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    tree = cKDTree(coords_secondary)
    neighbor_lists = tree.query_ball_point(coords_primary, r=float(radius_px))

    rows: List[int] = []
    cols: List[int] = []
    dists: List[float] = []
    for ia, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        jb_arr = np.asarray(neighbors, dtype=int)
        deltas = coords_secondary[jb_arr] - coords_primary[ia]
        d_arr = np.hypot(deltas[:, 0], deltas[:, 1])
        if filter_boundaries:
            cid_a = int(cell_ids_primary[ia])
            if cid_a <= 0:
                continue
            keep = cell_ids_secondary[jb_arr] == cid_a
            jb_arr = jb_arr[keep]
            d_arr = d_arr[keep]
        if jb_arr.size == 0:
            continue
        rows.extend([ia] * jb_arr.size)
        cols.extend(jb_arr.tolist())
        dists.extend(d_arr.tolist())

    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(dists, dtype=float),
    )


def _hungarian_on_cost(cost: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Run ``linear_sum_assignment`` on a cost matrix where ``inf`` marks
    forbidden assignments."""
    finite_mask = np.isfinite(cost)
    if not np.any(finite_mask):
        return [], np.array([], dtype=float)
    max_finite = float(np.nanmax(cost[finite_mask]))
    penalty = max(max_finite, 1.0) * (float(min(cost.shape)) + 1.0) + 1.0
    work = np.where(finite_mask, cost, penalty)
    ri, ci = linear_sum_assignment(work)
    keep = finite_mask[ri, ci]
    ri, ci = ri[keep], ci[keep]
    if ri.size == 0:
        return [], np.array([], dtype=float)
    return list(zip(ri.tolist(), ci.tolist())), cost[ri, ci]


def _optimal_by_components(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    distances: np.ndarray,
    n_primary: int,
    n_secondary: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Decompose the bipartite candidate graph into connected components and
    run Hungarian assignment on each independently."""
    node_count = n_primary + n_secondary
    n_edges = row_idx.size

    bip = coo_matrix(
        (np.ones(n_edges, dtype=np.uint8), (row_idx, n_primary + col_idx)),
        shape=(node_count, node_count),
    )
    graph = (bip + bip.T).tocsr()
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    comp_rows: Dict[int, List[int]] = {}
    comp_cols: Dict[int, List[int]] = {}
    for i in range(n_primary):
        comp_rows.setdefault(int(labels[i]), []).append(i)
    for j in range(n_secondary):
        comp_cols.setdefault(int(labels[n_primary + j]), []).append(j)

    edge_comp = labels[row_idx]
    edge_comps = set(edge_comp.tolist())

    pair_mapping: List[Tuple[int, int]] = []
    min_dists: List[float] = []
    for comp in sorted(edge_comps):
        r_arr = np.asarray(comp_rows.get(comp, []), dtype=int)
        c_arr = np.asarray(comp_cols.get(comp, []), dtype=int)
        if r_arr.size == 0 or c_arr.size == 0:
            continue
        row_map = {int(v): i for i, v in enumerate(r_arr)}
        col_map = {int(v): i for i, v in enumerate(c_arr)}
        mask = edge_comp == comp
        comp_r = row_idx[mask]
        comp_c = col_idx[mask]
        comp_d = distances[mask]
        block = np.full((r_arr.size, c_arr.size), np.inf, dtype=float)
        for r, c, d in zip(comp_r.tolist(), comp_c.tolist(), comp_d.tolist()):
            block[row_map[r], col_map[c]] = d
        pairs, dists = _hungarian_on_cost(block)
        for (bi, bj), dist in zip(pairs, dists.tolist()):
            pair_mapping.append((int(r_arr[bi]), int(c_arr[bj])))
            min_dists.append(dist)

    if not pair_mapping:
        return [], np.array([], dtype=float)
    order = np.argsort([p[0] for p in pair_mapping], kind="stable")
    return (
        [pair_mapping[i] for i in order],
        np.asarray([min_dists[i] for i in order]),
    )


def _optimal_one_to_one(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    distances: np.ndarray,
    n_primary: int,
    n_secondary: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Globally optimal one-to-one matching on sparse candidate edges.

    Uses connected-component decomposition when the full matrix would exceed
    ``SPARSE_COMPONENT_THRESHOLD`` entries.
    """
    if row_idx.size == 0:
        return [], np.array([], dtype=float)
    matrix_size = n_primary * n_secondary
    if matrix_size >= SPARSE_COMPONENT_THRESHOLD:
        return _optimal_by_components(
            row_idx, col_idx, distances, n_primary, n_secondary,
        )
    cost = np.full((n_primary, n_secondary), np.inf, dtype=float)
    cost[row_idx, col_idx] = distances
    return _hungarian_on_cost(cost)


def _match_pairs(
    coords_primary: np.ndarray,
    coords_secondary: np.ndarray,
    cell_ids_primary: np.ndarray,
    cell_ids_secondary: np.ndarray,
    *,
    filter_boundaries: bool,
    max_distance_px: Optional[float],
    primary_channel: Optional[str] = None,
    secondary_channel: Optional[str] = None,
    warn_cross_cell: bool = False,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Match primary↔secondary foci: KD-tree radius candidates → optimal
    one-to-one assignment.

    When *max_distance_px* is ``None`` all pairwise distances are considered
    (full cost matrix via ``cdist``).
    """
    n_p = int(coords_primary.shape[0])
    n_s = int(coords_secondary.shape[0])
    if n_p == 0 or n_s == 0:
        return [], np.array([], dtype=float)

    if warn_cross_cell and not filter_boundaries:
        cross = _count_cross_cell_pairs(cell_ids_primary, cell_ids_secondary)
        if cross > 0:
            LOGGER.warning(
                "coloc_filter_boundaries is disabled: %d candidate pairs cross "
                "cell boundaries between channels %s and %s.",
                cross, primary_channel, secondary_channel,
            )

    if max_distance_px is not None:
        row_idx, col_idx, dists = _kdtree_radius_candidates(
            coords_primary, coords_secondary, max_distance_px,
            cell_ids_primary=cell_ids_primary,
            cell_ids_secondary=cell_ids_secondary,
            filter_boundaries=filter_boundaries,
        )
        return _optimal_one_to_one(row_idx, col_idx, dists, n_p, n_s)

    cost = cdist(coords_primary, coords_secondary)
    if filter_boundaries:
        compat = (
            (cell_ids_primary[:, None] == cell_ids_secondary[None, :])
            & (cell_ids_primary[:, None] > 0)
        )
        cost = np.where(compat, cost, np.inf)
    return _hungarian_on_cost(cost)


# ---------------------------------------------------------------------------
# Row helpers (Analyze Masks / Excel Foci sheet)
# ---------------------------------------------------------------------------


def _row_channel_name(record: Dict[str, object]) -> str:
    raw = record.get("channel_name")
    if raw is None:
        raw = record.get("Channel_name")
    return str(raw).strip() if raw is not None else ""


def _row_frame(record: Dict[str, object]) -> Optional[int]:
    for key in ("frame", "Frame"):
        if key in record and record[key] is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                return None
    return None


def _filter_foci_rows_by_frame(
    rows: List[Dict[str, object]], frame: Optional[int]
) -> List[Dict[str, object]]:
    if frame is None:
        return list(rows)
    out: List[Dict[str, object]] = []
    for r in rows:
        fr = _row_frame(r)
        if fr is not None and fr == frame:
            out.append(r)
    return out


def _unique_frames_in_rows(rows: List[Dict[str, object]]) -> List[Optional[int]]:
    frames: List[int] = []
    for r in rows:
        fr = _row_frame(r)
        if fr is not None and fr not in frames:
            frames.append(fr)
    if not frames:
        return [None]
    return sorted(frames)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_colocalization_foci_table(
    foci_rows: List[Dict[str, object]],
    settings: ColocSettings,
    *,
    foci_channels: Optional[Sequence[str]] = None,
    phase_labels: Optional[np.ndarray] = None,
    frame: Optional[int] = None,
) -> ColocalizationOutputs:
    """Colocalization from foci rows (e.g. Analyze Masks ``foci_table`` / Foci sheet).

    Coordinates are ``(y, x)`` in pixels; no inter-channel registration is applied.

    When *frame* is ``None`` and rows contain multiple *frame* values, analysis
    runs **per frame** and each output record includes a ``frame`` field.
    """
    rows_in = list(foci_rows)
    if foci_channels is None:
        chans = sorted({_row_channel_name(r) for r in rows_in if _row_channel_name(r)})
    else:
        chans = [str(c).strip() for c in foci_channels if str(c).strip()]
    if len(chans) < 2:
        return ColocalizationOutputs([], [], [])

    _warn_ignored_matching_flags(settings)

    tolerance_px = float(settings.coloc_edge_tolerance)
    max_distance_px = settings.coloc_max_distance
    coloc_threshold_px = settings.coloc_threshold

    need_labels_for_random = settings.coloc_n_randomizations > 0
    if need_labels_for_random and phase_labels is None:
        LOGGER.warning(
            "coloc_n_randomizations > 0 but phase_labels is missing; skipping randomization."
        )

    if tolerance_px > 0 and phase_labels is None:
        LOGGER.warning(
            "coloc_edge_tolerance > 0 but phase_labels is missing; using cell_id from rows only."
        )

    filtered = _filter_foci_rows_by_frame(rows_in, frame)
    frame_groups = [frame] if frame is not None else _unique_frames_in_rows(filtered)

    coloc_records: List[Dict[str, object]] = []
    pair_records: List[Dict[str, object]] = []
    unmatched_records: List[Dict[str, object]] = []

    for fr in frame_groups:
        if fr is None:
            sub_rows = filtered
        else:
            sub_rows = [r for r in filtered if _row_frame(r) == fr]
        if not sub_rows:
            continue

        grouped = _group_foci_from_table_rows(
            sub_rows,
            foci_channels=chans,
            tolerance_px=tolerance_px,
            phase_labels=phase_labels,
        )

        centroid_methods = {"weighted_centroid", "external"}
        peak_methods = {"peak_local_max", "spotiflow", "skimage_peak"}
        for primary, secondary in itertools.combinations(chans, 2):
            m1 = grouped.get(primary, {}).get("detection_methods", set())
            m2 = grouped.get(secondary, {}).get("detection_methods", set())
            all_m = m1 | m2
            if all_m & centroid_methods and all_m & peak_methods:
                LOGGER.warning(
                    "Channels %s and %s use different detection semantics "
                    "(centroid vs peak); colocalization distances may be biased. "
                    "Consider using a uniform detection method across channels.",
                    primary, secondary,
                )

        for primary, secondary in itertools.combinations(chans, 2):
            pts_p = grouped.get(primary)
            pts_s = grouped.get(secondary)
            if not pts_p or not pts_s:
                continue
            record, pairs, unmatched = _analyze_pair(
                primary_channel=primary,
                secondary_channel=secondary,
                points_primary=pts_p,
                points_secondary=pts_s,
                settings=settings,
                max_distance_px=max_distance_px,
                coloc_threshold_px=coloc_threshold_px,
            )
            if len(frame_groups) > 1 or frame is not None:
                if fr is not None:
                    record["frame"] = int(fr)

            if (
                settings.coloc_n_randomizations > 0
                and phase_labels is not None
                and record.get("valid_pairs", 0) is not None
            ):
                rand_result = _run_randomization_test(
                    observed_valid=int(record.get("valid_pairs", 0) or 0),
                    points_primary=pts_p,
                    points_secondary=pts_s,
                    phase_labels=phase_labels,
                    settings=settings,
                    max_distance_px=max_distance_px,
                    coloc_threshold_px=coloc_threshold_px,
                )
                record.update(rand_result)

            coloc_records.append(record)
            for pr in pairs:
                prc = dict(pr)
                if len(frame_groups) > 1 or frame is not None:
                    if fr is not None:
                        prc["frame"] = int(fr)
                pair_records.append(prc)
            unmatched_records.extend(unmatched)

    return ColocalizationOutputs(
        coloc_records=coloc_records,
        pair_records=pair_records,
        unmatched_records=unmatched_records if settings.coloc_save_unmatched else [],
    )


def _group_foci_from_table_rows(
    foci_records: List[Dict[str, object]],
    *,
    foci_channels: Sequence[str],
    tolerance_px: float,
    phase_labels: Optional[np.ndarray],
) -> Dict[str, Dict[str, object]]:
    """Group foci by channel in image coordinates; optional mask snap for cell_id."""
    has_phase_labels = phase_labels is not None and phase_labels.ndim == 2
    height = width = 0
    if has_phase_labels:
        height, width = int(phase_labels.shape[0]), int(phase_labels.shape[1])
    distance_map = None
    nearest_indices = None
    if tolerance_px > 0 and has_phase_labels:
        distance_map, nearest_indices = distance_transform_edt(
            phase_labels == 0, return_indices=True, sampling=(1.0, 1.0)
        )

    grouped: Dict[str, Dict[str, object]] = {}
    for channel in foci_channels:
        grouped[channel] = {
            "coords": [],
            "cell_ids": [],
            "indices": [],
            "orig_coords": [],
            "foci_ids": [],
            "detection_methods": set(),
        }

    local_idx = 0
    for idx, record in enumerate(foci_records):
        channel = _row_channel_name(record)
        if channel not in grouped:
            continue
        y_orig, x_orig = _extract_foci_coordinates(record)
        if np.isnan(y_orig) or np.isnan(x_orig):
            continue
        y = float(y_orig)
        x = float(x_orig)
        try:
            cell_id = int(record.get("cell_id") or record.get("Cell_ID") or 0)
        except (TypeError, ValueError):
            cell_id = 0

        y_int = int(round(y))
        x_int = int(round(x))
        if has_phase_labels and np.isfinite(y) and np.isfinite(x):
            if 0 <= y_int < height and 0 <= x_int < width:
                cell_id = int(phase_labels[y_int, x_int])
            else:
                y_int = -1
                x_int = -1
            if (
                cell_id == 0
                and tolerance_px > 0
                and distance_map is not None
                and nearest_indices is not None
            ):
                if 0 <= y_int < height and 0 <= x_int < width:
                    distance_value = distance_map[y_int, x_int]
                    if distance_value <= tolerance_px:
                        nearest_y = nearest_indices[0, y_int, x_int]
                        nearest_x = nearest_indices[1, y_int, x_int]
                        cell_id = int(phase_labels[nearest_y, nearest_x])

        grouped[channel]["coords"].append([y, x])
        grouped[channel]["cell_ids"].append(cell_id)
        grouped[channel]["indices"].append(local_idx)
        local_idx += 1
        grouped[channel]["orig_coords"].append([y_orig, x_orig])
        fid = record.get("foci_id")
        if fid is None:
            fid = record.get("Foci_id")
        grouped[channel]["foci_ids"].append(
            int(fid) if fid is not None else idx + 1
        )
        dm = record.get("detection_method")
        if dm:
            grouped[channel]["detection_methods"].add(str(dm))

    centroid_methods = {"weighted_centroid", "external"}
    peak_methods = {"peak_local_max", "spotiflow", "skimage_peak"}
    for ch, data in grouped.items():
        methods = data["detection_methods"]
        if methods & centroid_methods and methods & peak_methods:
            LOGGER.warning(
                "Channel %s mixes centroid and peak detection methods %s; "
                "colocalization distances may be inconsistent.",
                ch, methods,
            )

    return grouped


def _extract_foci_coordinates(record: Dict[str, object]) -> Tuple[float, float]:
    """Return ``(y, x)`` coordinates from a focus record."""
    if "foci_location" in record and record["foci_location"] is not None:
        parsed = _parse_coordinate_pair(record["foci_location"])
        if parsed is not None:
            return parsed
    for yk, xk in (("y", "x"), ("Y", "X")):
        if yk in record and xk in record:
            yy = _as_float(record.get(yk, np.nan))
            xx = _as_float(record.get(xk, np.nan))
            if np.isfinite(yy) and np.isfinite(xx):
                return yy, xx
    return (
        _as_float(record.get("foci_center_y", np.nan)),
        _as_float(record.get("foci_center_x", np.nan)),
    )


def _parse_coordinate_pair(value: object) -> Optional[Tuple[float, float]]:
    """Parse ``(y, x)`` from list/tuple/ndarray or stringified literals."""
    if isinstance(value, np.ndarray):
        if value.size >= 2:
            return _as_float(value.flat[0]), _as_float(value.flat[1])
        return None
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return _as_float(value[0]), _as_float(value[1])
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return _as_float(parsed[0]), _as_float(parsed[1])
        stripped = text.strip("[]()")
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) >= 2:
            try:
                return _as_float(parts[0]), _as_float(parts[1])
            except Exception:
                return None
    return None


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Pair analysis
# ---------------------------------------------------------------------------

def _analyze_pair(
    primary_channel: str,
    secondary_channel: str,
    points_primary: Dict[str, Sequence],
    points_secondary: Dict[str, Sequence],
    settings: ColocSettings,
    max_distance_px: Optional[float],
    coloc_threshold_px: Optional[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    """Analyze a single channel pair.  All distances are in pixels."""
    coords1 = np.asarray(points_primary["coords"], dtype=float)
    coords2 = np.asarray(points_secondary["coords"], dtype=float)
    if coords1.size == 0 or coords2.size == 0:
        return (
            _empty_record(primary_channel, secondary_channel,
                          max_distance_px, coloc_threshold_px),
            [],
            [],
        )

    cell_ids1 = np.asarray(points_primary["cell_ids"], dtype=int)
    cell_ids2 = np.asarray(points_secondary["cell_ids"], dtype=int)

    pair_mapping, min_distances = _match_pairs(
        coords_primary=coords1,
        coords_secondary=coords2,
        cell_ids_primary=cell_ids1,
        cell_ids_secondary=cell_ids2,
        filter_boundaries=settings.coloc_filter_boundaries,
        max_distance_px=max_distance_px,
        primary_channel=primary_channel,
        secondary_channel=secondary_channel,
        warn_cross_cell=True,
    )

    if min_distances.size == 0:
        primary_unmatched = list(range(len(coords1)))
        secondary_unmatched = list(range(len(coords2)))
        record = _empty_record(
            primary_channel, secondary_channel,
            max_distance_px, coloc_threshold_px,
        )
        record["primary_unmatched_indices"] = primary_unmatched
        record["secondary_unmatched_indices"] = secondary_unmatched
        unmatched_records = _collect_unmatched_records(
            save_unmatched=settings.coloc_save_unmatched,
            primary_channel=primary_channel,
            secondary_channel=secondary_channel,
            primary_unmatched=primary_unmatched,
            secondary_unmatched=secondary_unmatched,
            primary_coordinates=points_primary["coords"],
            secondary_coordinates=points_secondary["coords"],
        )
        return record, [], unmatched_records

    # Classify using coloc_threshold
    valid_mask = np.ones_like(min_distances, dtype=bool)
    if coloc_threshold_px is not None:
        valid_mask = min_distances <= coloc_threshold_px

    # All-pair statistics
    mean_distance = float(np.mean(min_distances))
    median_distance = float(np.median(min_distances))
    max_distance_observed = float(np.max(min_distances))

    # Valid-pair statistics
    valid_dists = min_distances[valid_mask]
    valid_mean = float(np.mean(valid_dists)) if valid_dists.size else float("nan")
    valid_median = float(np.median(valid_dists)) if valid_dists.size else float("nan")
    valid_max = float(np.max(valid_dists)) if valid_dists.size else float("nan")

    total_pairs = len(pair_mapping)
    valid_pairs = int(valid_mask.sum())
    invalid_pairs = total_pairs - valid_pairs

    # Directional distances (in pixels)
    c1_idx = np.array([p[0] for p in pair_mapping])
    c2_idx = np.array([p[1] for p in pair_mapping])
    x_distances = coords2[c2_idx, 1] - coords1[c1_idx, 1]
    y_distances = coords2[c2_idx, 0] - coords1[c1_idx, 0]

    primary_assigned = {i for i, _ in pair_mapping}
    secondary_assigned = {j for _, j in pair_mapping}
    primary_unmatched = sorted(set(range(len(coords1))) - primary_assigned)
    secondary_unmatched = sorted(set(range(len(coords2))) - secondary_assigned)

    orig_coords1 = (
        np.asarray(points_primary["orig_coords"], dtype=float)
        if points_primary["orig_coords"]
        else np.empty((0, 2))
    )
    orig_coords2 = (
        np.asarray(points_secondary["orig_coords"], dtype=float)
        if points_secondary["orig_coords"]
        else np.empty((0, 2))
    )

    pair_records = _build_pair_records(
        primary_channel=primary_channel,
        secondary_channel=secondary_channel,
        pair_mapping=pair_mapping,
        min_distances=min_distances,
        x_distances=x_distances,
        y_distances=y_distances,
        valid_mask=valid_mask,
        primary_indices=points_primary["indices"],
        secondary_indices=points_secondary["indices"],
        primary_foci_ids=points_primary.get("foci_ids"),
        secondary_foci_ids=points_secondary.get("foci_ids"),
        primary_cell_ids=cell_ids1,
        secondary_cell_ids=cell_ids2,
        primary_coords=coords1,
        secondary_coords=coords2,
        primary_native_coords=orig_coords1,
        secondary_native_coords=orig_coords2,
        coloc_threshold_px=coloc_threshold_px,
    )

    valid_primary_set = {pair[0] for pair, v in zip(pair_mapping, valid_mask) if v}
    valid_secondary_set = {pair[1] for pair, v in zip(pair_mapping, valid_mask) if v}

    record: Dict[str, object] = {
        "primary_channel": primary_channel,
        "secondary_channel": secondary_channel,
        "distance_unit": "px",
        "max_distance": max_distance_px,
        "colocalisation_threshold": coloc_threshold_px,
        "coloc_mode": "one_to_one",
        "pair_ratio_primary": (
            len(valid_primary_set) / len(coords1) if len(coords1) > 0 else 0.0
        ),
        "pair_ratio_secondary": (
            len(valid_secondary_set) / len(coords2) if len(coords2) > 0 else 0.0
        ),
        "total_pairs": total_pairs,
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "max_distance_observed": max_distance_observed,
        "valid_mean_distance": valid_mean,
        "valid_median_distance": valid_median,
        "valid_max_distance": valid_max,
        "valid_distances": valid_dists.tolist(),
        "invalid_distances": (min_distances[~valid_mask]).tolist(),
        "primary_unmatched_indices": primary_unmatched,
        "secondary_unmatched_indices": secondary_unmatched,
    }

    unmatched_records = _collect_unmatched_records(
        save_unmatched=settings.coloc_save_unmatched,
        primary_channel=primary_channel,
        secondary_channel=secondary_channel,
        primary_unmatched=primary_unmatched,
        secondary_unmatched=secondary_unmatched,
        primary_coordinates=points_primary["coords"],
        secondary_coordinates=points_secondary["coords"],
    )

    return record, pair_records, unmatched_records


def _empty_record(
    primary_channel: str,
    secondary_channel: str,
    max_distance_px: Optional[float],
    coloc_threshold_px: Optional[float],
) -> Dict[str, object]:
    return {
        "primary_channel": primary_channel,
        "secondary_channel": secondary_channel,
        "distance_unit": "px",
        "max_distance": max_distance_px,
        "colocalisation_threshold": coloc_threshold_px,
        "coloc_mode": "one_to_one",
        "pair_ratio_primary": 0.0,
        "pair_ratio_secondary": 0.0,
        "total_pairs": 0,
        "valid_pairs": 0,
        "invalid_pairs": 0,
        "mean_distance": float("nan"),
        "median_distance": float("nan"),
        "max_distance_observed": float("nan"),
        "valid_mean_distance": float("nan"),
        "valid_median_distance": float("nan"),
        "valid_max_distance": float("nan"),
        "valid_distances": [],
        "invalid_distances": [],
        "primary_unmatched_indices": [],
        "secondary_unmatched_indices": [],
    }


# ---------------------------------------------------------------------------
# Pair / unmatched record builders
# ---------------------------------------------------------------------------

def _build_pair_records(
    primary_channel: str,
    secondary_channel: str,
    pair_mapping: List[Tuple[int, int]],
    min_distances: np.ndarray,
    x_distances: np.ndarray,
    y_distances: np.ndarray,
    valid_mask: np.ndarray,
    primary_indices: Sequence[int],
    secondary_indices: Sequence[int],
    primary_foci_ids: Optional[Sequence[int]],
    secondary_foci_ids: Optional[Sequence[int]],
    primary_cell_ids: Optional[Sequence[int]],
    secondary_cell_ids: Optional[Sequence[int]],
    primary_coords: np.ndarray,
    secondary_coords: np.ndarray,
    primary_native_coords: np.ndarray,
    secondary_native_coords: np.ndarray,
    coloc_threshold_px: Optional[float],
) -> List[Dict[str, object]]:
    """Build per-pair records.

    Coordinates are reported in two frames:
    - ``foci_coordinates`` / ``colocalized_foci_coordinates``: reference-channel
      ``(y, x)`` in pixels.
    - ``foci_native_coordinates`` / ``colocalized_foci_native_coordinates``:
      original channel ``(y, x)`` in pixels.
    """
    if not pair_mapping:
        return []
    primary_foci_ids = list(primary_foci_ids) if primary_foci_ids is not None else []
    secondary_foci_ids = list(secondary_foci_ids) if secondary_foci_ids is not None else []
    primary_cell_ids = list(primary_cell_ids) if primary_cell_ids is not None else []
    secondary_cell_ids = list(secondary_cell_ids) if secondary_cell_ids is not None else []
    records: List[Dict[str, object]] = []
    for idx, (pi, si) in enumerate(pair_mapping):
        cell_id_primary = int(primary_cell_ids[pi]) if pi < len(primary_cell_ids) else 0
        cell_id_secondary = int(secondary_cell_ids[si]) if si < len(secondary_cell_ids) else 0
        record: Dict[str, object] = {
            "foci_channel": primary_channel,
            "foci_idx": _resolve_foci_identifier(pi, primary_indices, primary_foci_ids),
            "foci_coordinates": _resolve_foci_coordinates(pi, primary_coords),
            "foci_native_coordinates": _resolve_foci_coordinates(pi, primary_native_coords),
            "cell_id_primary": (cell_id_primary if cell_id_primary > 0 else None),
            "colocalized_channel": secondary_channel,
            "colocalized_foci_idx": _resolve_foci_identifier(si, secondary_indices, secondary_foci_ids),
            "colocalized_foci_coordinates": _resolve_foci_coordinates(si, secondary_coords),
            "colocalized_foci_native_coordinates": _resolve_foci_coordinates(si, secondary_native_coords),
            "cell_id_secondary": (cell_id_secondary if cell_id_secondary > 0 else None),
            "same_cell": bool(cell_id_primary > 0 and cell_id_primary == cell_id_secondary),
            "distance_unit": "px",
            "distance": float(min_distances[idx]),
            "x_distance": float(x_distances[idx]),
            "x_distance_unit": "px",
            "y_distance": float(y_distances[idx]),
            "y_distance_unit": "px",
            "is_valid": bool(valid_mask[idx]) if valid_mask.size > idx else True,
        }
        if coloc_threshold_px is not None:
            record["colocalisation_threshold"] = float(coloc_threshold_px)
        records.append(record)
    return records


def _resolve_foci_identifier(
    mapped_index: int,
    global_indices: Sequence[int],
    declared_foci_ids: Sequence[int],
) -> int:
    """Resolve focus identifier, preferring declared IDs over computed ones."""
    if mapped_index < len(declared_foci_ids):
        try:
            return int(declared_foci_ids[mapped_index])
        except (TypeError, ValueError):
            pass
    if mapped_index < len(global_indices):
        return int(global_indices[mapped_index] + 1)
    return int(mapped_index + 1)


def _resolve_foci_coordinates(
    mapped_index: int,
    coordinates: np.ndarray,
) -> Sequence[float]:
    """Return ``(y, x)`` pixel coordinates for *mapped_index*."""
    if mapped_index < len(coordinates):
        c = coordinates[mapped_index]
        return [float(c[0]), float(c[1])]
    return [float("nan"), float("nan")]


def _build_unmatched_records(
    primary_channel: str,
    secondary_channel: str,
    unmatched_indices: List[int],
    coordinates: Sequence[Sequence[float]],
    channel_role: str,
) -> List[Dict[str, object]]:
    if not unmatched_indices:
        return []
    unmatched_set = set(unmatched_indices)
    coords_list = [
        [float(c[0]), float(c[1])]
        for idx, c in enumerate(coordinates)
        if idx in unmatched_set
    ]
    return [
        {
            "primary_channel": primary_channel,
            "secondary_channel": secondary_channel,
            "channel_role": channel_role,
            "distance_unit": "px",
            "unmatched_indices": unmatched_indices,
            "unmatched_coordinates": coords_list,
        }
    ]


def _collect_unmatched_records(
    *,
    save_unmatched: bool,
    primary_channel: str,
    secondary_channel: str,
    primary_unmatched: List[int],
    secondary_unmatched: List[int],
    primary_coordinates: Sequence[Sequence[float]],
    secondary_coordinates: Sequence[Sequence[float]],
) -> List[Dict[str, object]]:
    if not save_unmatched:
        return []
    records: List[Dict[str, object]] = []
    records.extend(
        _build_unmatched_records(
            primary_channel, secondary_channel,
            primary_unmatched, primary_coordinates, "primary",
        )
    )
    records.extend(
        _build_unmatched_records(
            primary_channel, secondary_channel,
            secondary_unmatched, secondary_coordinates, "secondary",
        )
    )
    return records


# ---------------------------------------------------------------------------
# Cross-cell warning helper
# ---------------------------------------------------------------------------

def _count_cross_cell_pairs(
    cell_ids_primary: np.ndarray,
    cell_ids_secondary: np.ndarray,
) -> int:
    """Count candidate pairs with both cell IDs > 0 and different labels."""
    primary_pos = cell_ids_primary[cell_ids_primary > 0]
    secondary_pos = cell_ids_secondary[cell_ids_secondary > 0]
    if primary_pos.size == 0 or secondary_pos.size == 0:
        return 0
    p_ids, p_counts = np.unique(primary_pos, return_counts=True)
    s_ids, s_counts = np.unique(secondary_pos, return_counts=True)
    s_lookup = {int(cid): int(cnt) for cid, cnt in zip(s_ids.tolist(), s_counts.tolist())}
    same_cell_pairs = 0
    for cid, count in zip(p_ids.tolist(), p_counts.tolist()):
        same_cell_pairs += int(count) * s_lookup.get(int(cid), 0)
    total = int(primary_pos.size) * int(secondary_pos.size)
    return int(total - same_cell_pairs)


# ---------------------------------------------------------------------------
# Deprecated matching-mode warnings
# ---------------------------------------------------------------------------

def _warn_ignored_matching_flags(settings: ColocSettings) -> None:
    if not settings.coloc_one_to_one:
        LOGGER.warning(
            "coloc_one_to_one=False is deprecated and ignored; using one-to-one pairing."
        )
    if not settings.coloc_mutual_nearest:
        LOGGER.warning(
            "coloc_mutual_nearest is deprecated and ignored; "
            "colocalization always uses one-to-one pairing."
        )


# ---------------------------------------------------------------------------
# Randomization significance testing
# ---------------------------------------------------------------------------

def _resolve_randomization_workers(n_iter: int) -> int:
    """Determine worker count for randomization matching."""
    if n_iter <= 1:
        return 1
    env_value = os.getenv("FLAGELLA_COLOC_RANDOMIZATION_WORKERS")
    if env_value is not None:
        try:
            parsed = int(env_value)
            if parsed >= 1:
                return min(parsed, n_iter)
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid FLAGELLA_COLOC_RANDOMIZATION_WORKERS=%s; "
                "expected integer >= 1.",
                env_value,
            )
    if n_iter < 16:
        return 1
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, n_iter, 8))


def _run_randomization_test(
    observed_valid: int,
    points_primary: Dict[str, Sequence],
    points_secondary: Dict[str, Sequence],
    phase_labels: np.ndarray,
    settings: ColocSettings,
    max_distance_px: Optional[float],
    coloc_threshold_px: Optional[float],
) -> Dict[str, object]:
    """Assess colocalization significance by randomizing secondary foci
    positions within cell masks and re-running matching."""
    n_iter = settings.coloc_n_randomizations
    rng = np.random.default_rng(settings.coloc_randomization_seed)

    cell_pixels: Dict[int, np.ndarray] = {}
    cell_ids_secondary = np.asarray(points_secondary["cell_ids"], dtype=int)
    for cid in np.unique(cell_ids_secondary):
        if cid <= 0:
            continue
        ys, xs = np.nonzero(phase_labels == cid)
        if ys.size > 0:
            cell_pixels[cid] = np.column_stack((ys.astype(float), xs.astype(float)))

    coords_primary = np.asarray(points_primary["coords"], dtype=float)
    coords_secondary = np.asarray(points_secondary["coords"], dtype=float)
    cell_ids_primary = np.asarray(points_primary["cell_ids"], dtype=int)

    eligible_positions: List[int] = []
    eligible_cell_ids: List[int] = []
    for k, cid in enumerate(cell_ids_secondary.tolist()):
        if cid > 0 and cid in cell_pixels:
            eligible_positions.append(int(k))
            eligible_cell_ids.append(int(cid))

    draw_indices = np.empty((n_iter, len(eligible_positions)), dtype=np.int64)
    for iteration in range(n_iter):
        for local_idx, cid in enumerate(eligible_cell_ids):
            draw_indices[iteration, local_idx] = int(
                rng.integers(0, len(cell_pixels[cid]))
            )

    def _iteration_valid_count(iteration: int) -> int:
        shuffled = coords_secondary.copy()
        if eligible_positions:
            row_draws = draw_indices[iteration]
            for local_idx, focus_idx in enumerate(eligible_positions):
                cid = eligible_cell_ids[local_idx]
                pix = cell_pixels[cid]
                shuffled[focus_idx] = pix[int(row_draws[local_idx])]
        _, matched_dists = _match_pairs(
            coords_primary=coords_primary,
            coords_secondary=shuffled,
            cell_ids_primary=cell_ids_primary,
            cell_ids_secondary=cell_ids_secondary,
            filter_boundaries=settings.coloc_filter_boundaries,
            max_distance_px=max_distance_px,
            warn_cross_cell=False,
        )
        if coloc_threshold_px is not None:
            return int(np.sum(matched_dists <= coloc_threshold_px))
        return int(len(matched_dists))

    workers = _resolve_randomization_workers(n_iter)
    if workers == 1:
        random_valid_counts = [_iteration_valid_count(i) for i in range(n_iter)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            random_valid_counts = list(
                executor.map(_iteration_valid_count, range(n_iter))
            )

    random_arr = np.asarray(random_valid_counts, dtype=float)
    p_value = float(np.sum(random_arr >= observed_valid) + 1) / float(n_iter + 1)
    return {
        "randomization_n": n_iter,
        "randomization_p_value": p_value,
        "randomization_mean_valid_pairs": (
            float(np.mean(random_arr)) if random_arr.size else float("nan")
        ),
        "randomization_std_valid_pairs": (
            float(np.std(random_arr)) if random_arr.size else float("nan")
        ),
    }
