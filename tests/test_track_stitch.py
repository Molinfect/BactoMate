"""Unit tests for proximity track stitch (gap closing)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bactomate.gui.track_stitch import (
    apply_stitch_from_params,
    remap_lineage_edges_df,
    remap_napari_lineage_graph,
    remap_tracked_mask_labels,
    stitch_tracks_by_proximity,
    track_graph_from_tracks_df,
)


def _tracks(*rows: tuple[int, int, float, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["track_id", "frame", "y", "x"])


def test_one_frame_gap_close_endpoints_merge_to_one_id():
    # Track 1 ends at frame 2; track 2 starts at frame 4 → gap=1, close jump.
    df = _tracks(
        (1, 0, 10.0, 10.0),
        (1, 1, 10.0, 11.0),
        (1, 2, 10.0, 12.0),
        (2, 4, 10.5, 13.0),
        (2, 5, 10.5, 14.0),
    )
    out, n, pairs, id_map = stitch_tracks_by_proximity(df, max_gap_frames=2, max_jump_px=5.0)
    assert n == 1
    assert pairs == [(1, 2)]
    assert id_map == {2: 1}
    assert set(out["track_id"].unique()) == {1}
    assert len(out) == 5


def test_far_jump_does_not_merge():
    df = _tracks(
        (1, 0, 10.0, 10.0),
        (1, 1, 10.0, 10.0),
        (2, 3, 80.0, 80.0),
        (2, 4, 81.0, 81.0),
    )
    out, n, pairs, id_map = stitch_tracks_by_proximity(df, max_gap_frames=4, max_jump_px=5.0)
    assert n == 0
    assert pairs == []
    assert id_map == {}
    assert set(out["track_id"].unique()) == {1, 2}


def test_gap_too_large_does_not_merge():
    df = _tracks(
        (1, 0, 10.0, 10.0),
        (1, 1, 10.0, 11.0),
        (2, 6, 10.0, 12.0),  # gap = 6-1-1 = 4
        (2, 7, 10.0, 13.0),
    )
    out, n, _pairs, _id_map = stitch_tracks_by_proximity(df, max_gap_frames=2, max_jump_px=20.0)
    assert n == 0
    assert set(out["track_id"].unique()) == {1, 2}


def test_competing_two_starts_only_nearest_merges():
    # Track 1 ends; 2 is closer than 3.
    df = _tracks(
        (1, 0, 0.0, 0.0),
        (1, 1, 0.0, 1.0),
        (2, 3, 0.0, 2.0),  # dist ~1
        (2, 4, 0.0, 3.0),
        (3, 3, 0.0, 8.0),  # dist ~7
        (3, 4, 0.0, 9.0),
    )
    out, n, pairs, id_map = stitch_tracks_by_proximity(df, max_gap_frames=2, max_jump_px=10.0)
    assert n == 1
    assert pairs == [(1, 2)]
    assert id_map == {2: 1}
    assert set(out["track_id"].unique()) == {1, 3}


def test_remap_tracked_mask_labels():
    masks = np.zeros((2, 4, 4), dtype=np.int32)
    masks[0, 1, 1] = 2
    masks[1, 2, 2] = 2
    out = remap_tracked_mask_labels(masks, {2: 1})
    assert out is not None
    assert int(out[0, 1, 1]) == 1
    assert int(out[1, 2, 2]) == 1


def test_apply_stitch_from_params_respects_enabled_flag():
    df = _tracks(
        (1, 0, 10.0, 10.0),
        (1, 1, 10.0, 11.0),
        (2, 3, 10.0, 12.0),
        (2, 4, 10.0, 13.0),
    )
    masks = np.zeros((5, 3, 3), dtype=np.int32)
    masks[3, 1, 1] = 2
    off, masks_off, n_off, _, id_off = apply_stitch_from_params(
        df, masks, {"stitch_enabled": False, "stitch_max_gap_frames": 2, "stitch_max_jump": 15.0}
    )
    assert n_off == 0
    assert id_off == {}
    assert set(off["track_id"].unique()) == {1, 2}
    assert int(masks_off[3, 1, 1]) == 2

    on, masks_on, n_on, pairs, id_on = apply_stitch_from_params(
        df,
        masks,
        {
            "stitch_enabled": True,
            "stitch_max_gap_frames": 2,
            "stitch_max_jump": 15.0,
            "pixel_size": 0.0,
            "use_physical_units": False,
        },
    )
    assert n_on == 1
    assert pairs == [(1, 2)]
    assert id_on == {2: 1}
    assert set(on["track_id"].unique()) == {1}
    assert int(masks_on[3, 1, 1]) == 1


def test_physical_jump_conversion_uses_pixel_size():
    df = _tracks(
        (1, 0, 0.0, 0.0),
        (1, 1, 0.0, 0.0),
        (2, 3, 0.0, 20.0),  # 20 px = 2 µm at 0.1 µm/px
    )
    # 1 µm max jump → 10 px → should NOT merge (20 px apart)
    out, _masks, n, _pairs, _id = apply_stitch_from_params(
        df,
        None,
        {
            "stitch_enabled": True,
            "stitch_max_gap_frames": 2,
            "stitch_max_jump": 1.0,  # µm
            "pixel_size": 0.1,
            "use_physical_units": True,
        },
    )
    assert n == 0
    assert set(out["track_id"].unique()) == {1, 2}

    out2, _masks2, n2, _pairs2, id2 = apply_stitch_from_params(
        df,
        None,
        {
            "stitch_enabled": True,
            "stitch_max_gap_frames": 2,
            "stitch_max_jump": 3.0,  # µm → 30 px
            "pixel_size": 0.1,
            "use_physical_units": True,
        },
    )
    assert n2 == 1
    assert id2 == {2: 1}
    assert set(out2["track_id"].unique()) == {1}


def test_remap_napari_lineage_graph_merges_ids_and_drops_self_parent():
    graph = {2: [1], 3: [2], 4: [1]}
    remapped = remap_napari_lineage_graph(graph, {2: 1, 3: 1})
    assert remapped[1] == []  # 2→1 and 3→1 collapsed; self-parents dropped
    assert remapped[4] == [1]


def test_remap_lineage_edges_df():
    edges = pd.DataFrame(
        {
            "parent_track_id": [1, 2],
            "daughter_track_id": [2, 3],
            "division_frame": [5, 8],
            "parent_last_frame": [4, 7],
        }
    )
    out = remap_lineage_edges_df(edges, {2: 1, 3: 1})
    # (1,2)→(1,1) dropped; (2,3)→(1,1) dropped
    assert len(out) == 0


def test_track_graph_from_tracks_df_chains_detections():
    df = _tracks((7, 0, 1.0, 2.0), (7, 1, 1.5, 2.5), (7, 3, 2.0, 3.0))
    g = track_graph_from_tracks_df(df)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    labels = {g.nodes[n]["label"] for n in g.nodes}
    assert labels == {7}
