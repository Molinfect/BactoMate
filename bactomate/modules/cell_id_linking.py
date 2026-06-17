"""
Frame-to-frame cell ID linking via Hungarian (linear sum) assignment.

Used by analyze_cell_areas and tracking workers when matching centroids
between consecutive frames or between mask stacks.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def centroids_from_mask(
    mask,
    labels: Iterable[int] | None = None,
    *,
    pixel_size: float = 1.0,
) -> dict[int, np.ndarray]:
    """Return ``{label: centroid}`` for instance labels in *mask* (centroid × pixel_size)."""
    from skimage.measure import regionprops

    label_set = set(int(x) for x in labels) if labels is not None else None
    out: dict[int, np.ndarray] = {}
    for region in regionprops(mask):
        label = int(region.label)
        if label <= 0:
            continue
        if label_set is not None and label not in label_set:
            continue
        out[label] = np.asarray(region.centroid, dtype=np.float64) * float(pixel_size)
    return out


def hungarian_link_centroids(
    prev_points: np.ndarray,
    prev_ids: np.ndarray,
    curr_points: np.ndarray,
    curr_labels: np.ndarray,
    max_distance: float,
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Globally optimal one-to-one links between previous IDs and current labels.

    Parameters
    ----------
    prev_points, curr_points
        (N, 2) and (M, 2) centroid coordinates in the same units as *max_distance*.
    prev_ids, curr_labels
        Persistent IDs and current-frame instance labels.
    max_distance
        Pairs farther apart are not linked.

    Returns
    -------
    linked
        List of ``(prev_id, curr_label)`` pairs.
    unlinked_curr_labels
        Current labels with no assignment within *max_distance*.
    """
    prev_ids = np.asarray(prev_ids, dtype=int)
    curr_labels = np.asarray(curr_labels, dtype=int)
    if prev_points.size == 0 or curr_points.size == 0:
        return [], curr_labels.astype(int).tolist()

    prev_points = np.asarray(prev_points, dtype=np.float64)
    curr_points = np.asarray(curr_points, dtype=np.float64)
    distances = cdist(prev_points, curr_points)
    cost = distances.astype(np.float64, copy=True)
    cost[cost > float(max_distance)] = np.inf

    finite = np.isfinite(cost)
    if not np.any(finite):
        return [], curr_labels.astype(int).tolist()

    max_finite = float(np.nanmax(cost[finite]))
    penalty = max(max_finite, 1.0) * (float(min(cost.shape)) + 1.0) + 1.0
    work = np.where(finite, cost, penalty)
    row_idx, col_idx = linear_sum_assignment(work)

    linked: list[tuple[int, int]] = []
    used_curr: set[int] = set()
    for i, j in zip(row_idx, col_idx):
        if not finite[i, j]:
            continue
        prev_id = int(prev_ids[i])
        curr_label = int(curr_labels[j])
        linked.append((prev_id, curr_label))
        used_curr.add(curr_label)

    unlinked = [int(l) for l in curr_labels if int(l) not in used_curr]
    return linked, unlinked
