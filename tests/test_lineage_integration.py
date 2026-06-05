from __future__ import annotations


def test_lineage_edges_from_napari_graph_parent_child() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import lineage_edges_from_napari_graph

    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 3, 3],
            "frame": [0, 1, 2, 3, 4, 5],
            "y": [10, 11, 20, 21, 30, 31],
            "x": [10, 11, 20, 21, 30, 31],
        }
    )
    graph = {2: [1], 3: [2]}
    edges = lineage_edges_from_napari_graph(graph, tracks)
    assert len(edges) == 2
    assert set(edges["daughter_track_id"].tolist()) == {2, 3}


def test_lineage_founder_map() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import lineage_edges_from_napari_graph, lineage_founder_map

    tracks = pd.DataFrame(
        {"track_id": [1, 2, 3], "frame": [0, 1, 2], "y": [0, 0, 0], "x": [0, 0, 0]}
    )
    graph = {2: [1], 3: [2]}
    edges = lineage_edges_from_napari_graph(graph, tracks)
    founders = lineage_founder_map(edges, tracks)
    assert founders[1] == 1
    assert founders[2] == 1
    assert founders[3] == 1


def test_track_lineage_metadata_generation_depth() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import (
        lineage_edges_from_napari_graph,
        track_lineage_metadata,
    )

    tracks = pd.DataFrame(
        {"track_id": [1, 2, 3], "frame": [0, 1, 2], "y": [0, 0, 0], "x": [0, 0, 0]}
    )
    edges = lineage_edges_from_napari_graph({2: [1], 3: [2]}, tracks)
    meta = track_lineage_metadata(edges, tracks)
    assert int(meta.loc[meta["track_id"] == 3, "generation"].iloc[0]) == 2


def test_join_analysis_to_lineage_points_frame_shift() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import join_analysis_to_lineage_points

    pts = pd.DataFrame({"track_id": [1], "frame": [0], "Cell_ID": [42], "y": [1], "x": [2]})
    per_cell = pd.DataFrame(
        {"Frame": [1], "Cell_ID": [42], "Area (µm²)": [12.5], "Cell Label": [7]}
    )
    out = join_analysis_to_lineage_points(pts, per_cell)
    assert float(out["Area (µm²)"].iloc[0]) == 12.5


def test_build_lineage_events_area_asymmetry() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import build_lineage_events

    edges = pd.DataFrame(
        [
            {
                "parent_track_id": 1,
                "daughter_track_id": 2,
                "division_frame": 2,
                "parent_last_frame": 1,
            }
        ]
    )
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2],
            "frame": [0, 1, 2],
            "y": [0, 0, 0],
            "x": [0, 0, 0],
        }
    )
    area = pd.DataFrame(
        {
            "track_id": [1, 1, 2],
            "frame": [0, 1, 2],
            "area": [100.0, 80.0, 40.0],
        }
    )
    events = build_lineage_events(edges, tracks, area_df=area)
    assert len(events) == 1
    assert float(events["mother_area_before"].iloc[0]) == 80.0
    assert float(events["daughter_area_after"].iloc[0]) == 40.0


def test_resolve_lineage_mask_prefers_filtered(tmp_path) -> None:
    from bactomate.modules.lineage_integration import resolve_lineage_mask_path

    seg = tmp_path / "_segmentation_test"
    filt_dir = seg / "filtered_masks"
    filt_dir.mkdir(parents=True)
    raw = seg / "sample_masks.tif"
    filt = filt_dir / "sample_mask_filtered.tif"
    raw.write_bytes(b"\x00")
    filt.write_bytes(b"\x00")
    resolved = resolve_lineage_mask_path(raw)
    assert resolved.name == "sample_mask_filtered.tif"


def test_preflight_warns_raw_without_filtered(tmp_path) -> None:
    from bactomate.modules.lineage_integration import preflight_lineage_run

    raw = tmp_path / "only_masks.tif"
    raw.write_bytes(b"\x00")
    pf = preflight_lineage_run(
        raw,
        linker_mode="greedy",
        model_name="general_2d",
        prefer_filtered_mask=True,
    )
    assert any("filtered" in w.lower() for w in pf.warnings) or pf.mask_is_filtered is False
