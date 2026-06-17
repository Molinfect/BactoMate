"""Tests for Hungarian frame-to-frame cell ID linking."""

from __future__ import annotations

import numpy as np
import pytest

from bactomate.modules.cell_id_linking import centroids_from_mask, hungarian_link_centroids
from bactomate.scripts.analyze_cell_areas import (
    assign_persistent_cell_ids,
    link_persistent_cell_ids_raw,
)


def test_hungarian_link_simple_translation():
    prev_points = np.array([[10.0, 10.0], [30.0, 30.0]])
    prev_ids = np.array([1, 2])
    curr_points = np.array([[11.0, 10.5], [29.0, 31.0]])
    curr_labels = np.array([10, 20])

    linked, unlinked = hungarian_link_centroids(
        prev_points, prev_ids, curr_points, curr_labels, max_distance=5.0
    )
    assert sorted(linked) == [(1, 10), (2, 20)]
    assert unlinked == []


def test_hungarian_link_respects_max_distance():
    prev_points = np.array([[0.0, 0.0]])
    prev_ids = np.array([1])
    curr_points = np.array([[100.0, 0.0]])
    curr_labels = np.array([5])

    linked, unlinked = hungarian_link_centroids(
        prev_points, prev_ids, curr_points, curr_labels, max_distance=10.0
    )
    assert linked == []
    assert unlinked == [5]


def test_hungarian_link_resolves_crossing_better_than_greedy():
    """Two cells swap positions slightly — Hungarian should match motion, not index order."""
    prev_points = np.array([[0.0, 0.0], [10.0, 0.0]])
    prev_ids = np.array([1, 2])
    curr_points = np.array([[9.5, 0.0], [0.5, 0.0]])
    curr_labels = np.array([100, 101])

    linked, unlinked = hungarian_link_centroids(
        prev_points, prev_ids, curr_points, curr_labels, max_distance=2.0
    )
    assert sorted(linked) == [(1, 101), (2, 100)]
    assert unlinked == []


def test_centroids_from_mask_filters_labels():
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[2:8, 2:8] = 1
    mask[12:18, 12:18] = 2

    all_centroids = centroids_from_mask(mask, pixel_size=0.1)
    assert set(all_centroids) == {1, 2}

    filtered = centroids_from_mask(mask, labels=[2], pixel_size=0.1)
    assert set(filtered) == {2}


def test_link_persistent_cell_ids_raw_two_frame_stack():
    masks = np.zeros((2, 30, 30), dtype=np.uint16)
    masks[0, 5:10, 5:10] = 1
    masks[1, 6:11, 6:11] = 1

    out = link_persistent_cell_ids_raw(masks, pixel_size=1.0, max_distance_um=5.0)
    assert int(out[0].max()) == 1
    assert int(out[1].max()) == 1
    assert np.count_nonzero(out[1]) == np.count_nonzero(masks[1])


def test_assign_persistent_cell_ids_accepts_only():
    masks = np.zeros((2, 30, 30), dtype=np.uint16)
    masks[0, 5:10, 5:10] = 1
    masks[0, 15:20, 15:20] = 2
    masks[1, 6:11, 6:11] = 1
    masks[1, 16:21, 16:21] = 2

    frame_data = [
        {"accepted_labels": [1]},
        {"accepted_labels": [1]},
    ]
    out, mapping = assign_persistent_cell_ids(
        masks, frame_data, pixel_size=1.0, max_distance_um=5.0
    )
    assert mapping[(0, 1)] == 1
    assert mapping[(1, 1)] == 1
    assert (0, 2) not in mapping
    assert int(out[1][6:11, 6:11].max()) == 1
