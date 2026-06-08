from __future__ import annotations

from pathlib import Path


def test_build_lineage_tree_layout_positions() -> None:
    import pandas as pd

    from bactomate.modules.lineage_viz import build_lineage_tree_layout

    meta = pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "lineage_id": [1, 1, 1],
            "generation": [0, 1, 2],
            "parent_track_id": [None, 1, 2],
            "branch_depth": [2, 1, 0],
        }
    )
    edges = pd.DataFrame(
        {
            "parent_track_id": [1, 2],
            "daughter_track_id": [2, 3],
            "division_frame": [1, 2],
        }
    )
    nodes, tree_edges = build_lineage_tree_layout(edges, meta)
    assert len(nodes) == 3
    assert len(tree_edges) == 2
    assert set(nodes["track_id"].tolist()) == {1, 2, 3}


def test_build_lineage_growth_timeseries_sums_by_lineage() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import build_lineage_growth_timeseries

    area = pd.DataFrame(
        {
            "track_id": [1, 2, 1, 2],
            "frame": [0, 0, 1, 1],
            "area": [10.0, 5.0, 12.0, 6.0],
            "area_source": ["mask_px", "mask_px", "mask_px", "mask_px"],
        }
    )
    founder = {1: 1, 2: 1}
    ts = build_lineage_growth_timeseries(area, founder, time_per_frame_min=2.0)
    assert len(ts) == 2
    assert float(ts.loc[ts["frame"] == 0, "total_area"].iloc[0]) == 15.0
    assert float(ts.loc[ts["frame"] == 1, "total_area"].iloc[0]) == 18.0
    assert int(ts.loc[ts["frame"] == 0, "frame_1based"].iloc[0]) == 1


def test_apply_event_review_status_updates_row() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import (
        REVIEW_STATUS_ACCEPTED,
        apply_event_review_status,
    )

    events = pd.DataFrame(
        [{"parent_track_id": 1, "daughter_track_id": 2, "review_status": "candidate"}]
    )
    out = apply_event_review_status(events, parent_track_id=1, daughter_track_id=2, review_status=REVIEW_STATUS_ACCEPTED)
    assert out["review_status"].iloc[0] == REVIEW_STATUS_ACCEPTED


def test_build_lineage_qc_flags_far_daughter() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import build_lineage_qc

    tracks = pd.DataFrame({"track_id": [1, 2], "frame": [0, 1], "y": [0, 0], "x": [0, 0]})
    edges = pd.DataFrame(
        [{"parent_track_id": 1, "daughter_track_id": 2, "division_frame": 1, "parent_last_frame": 0}]
    )
    events = pd.DataFrame(
        [
            {
                "parent_track_id": 1,
                "daughter_track_id": 2,
                "division_frame": 1,
                "division_distance_px": 120.0,
                "area_asymmetry": 0.5,
            }
        ]
    )
    meta = pd.DataFrame({"track_id": [1, 2], "generation": [0, 1], "branch_depth": [1, 0]})
    qc = build_lineage_qc(edges, tracks, meta, lineage_events_df=events)
    assert any(qc["flag_type"] == "daughter_far_from_mother")


def test_export_lineage_figures_writes_png(tmp_path) -> None:
    import pandas as pd

    from bactomate.modules.lineage_viz import export_lineage_figures

    tables = {
        "lineage_tracks_meta": pd.DataFrame(
            {
                "track_id": [1, 2],
                "lineage_id": [1, 1],
                "generation": [0, 1],
                "parent_track_id": [None, 1],
            }
        ),
        "lineage_edges": pd.DataFrame(
            {"parent_track_id": [1], "daughter_track_id": [2], "division_frame": [1]}
        ),
        "lineage_growth_timeseries": pd.DataFrame(
            {"lineage_id": [1, 1], "frame": [0, 1], "total_area": [10.0, 20.0]}
        ),
        "lineage_events": pd.DataFrame(
            {"area_asymmetry": [0.5, 0.6]},
        ),
    }
    written = export_lineage_figures(Path(tmp_path), tables, stem="sample")
    assert any(p.suffix == ".png" for p in written)
