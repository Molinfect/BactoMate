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


def test_build_lineage_tree_layout_offsets_independent_families() -> None:
    import pandas as pd

    from bactomate.modules.lineage_viz import build_lineage_tree_layout

    meta = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 5, 6],
            "lineage_id": [1, 1, 1, 2, 2, 2],
            "generation": [0, 1, 1, 0, 1, 1],
            "parent_track_id": [None, 1, 1, None, 4, 4],
        }
    )
    edges = pd.DataFrame(
        {
            "parent_track_id": [1, 1, 4, 4],
            "daughter_track_id": [2, 3, 5, 6],
            "division_frame": [5, 5, 5, 5],
        }
    )
    nodes, _ = build_lineage_tree_layout(edges, meta, lineage_ids=[1, 2], combine_families=True)
    founders = nodes[nodes["generation"] == 0]
    assert len(founders) == 2
    xs = sorted(float(x) for x in founders["x"].tolist())
    assert xs[1] - xs[0] > 0.5


def test_build_lineage_tree_layout_single_family_default() -> None:
    import pandas as pd

    from bactomate.modules.lineage_viz import build_lineage_tree_layout

    meta = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4],
            "lineage_id": [1, 1, 2, 2],
            "generation": [0, 1, 0, 1],
            "parent_track_id": [None, 1, None, 3],
        }
    )
    edges = pd.DataFrame(
        {"parent_track_id": [1, 3], "daughter_track_id": [2, 4], "division_frame": [1, 1]}
    )
    nodes, _ = build_lineage_tree_layout(edges, meta)
    assert set(nodes["lineage_id"].tolist()) == {1, 2}
    nodes_one, _ = build_lineage_tree_layout(edges, meta, lineage_ids=[2])
    assert set(nodes_one["lineage_id"].tolist()) == {2}

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


def test_discover_lineage_metric_columns_hides_foci_fi_and_empty_coloc() -> None:
    import pandas as pd

    from bactomate.modules.lineage_viz import discover_lineage_metric_columns

    pts = pd.DataFrame(
        {
            "track_id": [1, 2],
            "frame": [0, 1],
            "lineage_id": [1, 1],
            "Area (µm²)": [3.0, 4.0],
            "Mean FI Intensity (whole cell) (Red)": [1700.0, 1800.0],
            "Mean FI Intensity (whole cell) (Green)": [4800.0, 5000.0],
            "Green_foci_count": [1, 2],
            "Green_foci_max_intensity": [100.0, 120.0],
            "coloc_pairs_count": [0, 0],
        }
    )
    discovered = discover_lineage_metric_columns(pts)
    flat = [c for group in discovered.values() for c in group]
    assert "Mean FI Intensity (whole cell) (Red)" in flat
    assert "Mean FI Intensity (whole cell) (Green)" not in flat
    assert "coloc_pairs_count" not in flat
    assert "Green_foci_max_intensity" in flat


def test_plot_lineage_metric_timeseries_parses_foci_list_column() -> None:
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure

    from bactomate.modules.lineage_viz import (
        list_metric_cell_reduce,
        plot_lineage_metric_timeseries,
        prepare_lineage_metric_numeric_column,
    )

    assert list_metric_cell_reduce([5100.0, 6100.0], "Green_foci_mean_intensities") == 5600.0
    assert list_metric_cell_reduce([5000.0, 6000.0], "Green_foci_intensities") == 11000.0

    pts = pd.DataFrame(
        {
            "track_id": [1, 1, 2],
            "frame": [0, 1, 1],
            "lineage_id": [1, 1, 1],
            "Green_foci_mean_intensities": ["5100, 6100", "5200", ""],
            "Green_foci_max_intensity": [8000.0, 7900.0, 7600.0],
        }
    )
    parsed = prepare_lineage_metric_numeric_column(pts, "Green_foci_mean_intensities")
    assert float(parsed.iloc[0]) == 5600.0
    assert float(parsed.iloc[1]) == 5200.0
    assert not np.isfinite(float(parsed.iloc[2]))

    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    plot_lineage_metric_timeseries(
        ax,
        pts,
        metric_column="Green_foci_mean_intensities",
        lineage_id=1,
        aggregate="mean",
        title="Green_foci_mean_intensities — lineage 1",
    )
    assert len(ax.lines) == 1
    y = ax.lines[0].get_ydata()
    assert len(y) >= 2
    assert np.isfinite(y).all()


def test_plot_lineage_cell_cycle_area_uses_integer_frame_ticks() -> None:
    import pandas as pd
    from matplotlib.figure import Figure

    from bactomate.modules.lineage_viz import plot_lineage_cell_cycle_area

    area_df = pd.DataFrame(
        {
            "track_id": [2, 2, 2, 21, 21, 22, 22, 221, 221, 222, 222],
            "frame": [0, 1, 2, 6, 7, 6, 7, 17, 18, 17, 18],
            "area": [4.0, 5.0, 6.0, 7.0, 8.0, 7.5, 8.5, 9.0, 9.5, 9.2, 9.8],
            "area_source": ["analysis:um2"] * 11,
            "lineage_id": [2] * 11,
        }
    )
    edges = pd.DataFrame(
        {
            "parent_track_id": [2, 22],
            "daughter_track_id": [21, 221],
            "division_frame": [6, 17],
            "parent_last_frame": [5, 16],
        }
    )
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plot_lineage_cell_cycle_area(
        ax,
        area_df,
        edges,
        lineage_id=2,
        label_map={2: "Cell 2", 21: "Cell 2.1", 22: "Cell 2.2", 221: "Cell 2.2.1", 222: "Cell 2.2.2"},
        title="Lineage 2 cell-cycle area",
    )
    tick_vals = ax.get_xticks()
    assert len(tick_vals) > 0
    for tick in tick_vals:
        assert abs(tick - round(tick)) < 1e-6, f"non-integer frame tick: {tick}"


def test_export_lineage_figures_writes_png(tmp_path) -> None:
    import pandas as pd

    from bactomate.modules.lineage_viz import export_lineage_analysis_figures, export_lineage_figures

    tables = {
        "lineage_tracks_meta": pd.DataFrame(
            {
                "track_id": [1, 2, 3, 4],
                "lineage_id": [1, 1, 2, 2],
                "generation": [0, 1, 0, 1],
                "parent_track_id": [None, 1, None, 3],
            }
        ),
        "lineage_edges": pd.DataFrame(
            {"parent_track_id": [1, 3], "daughter_track_id": [2, 4], "division_frame": [1, 1]}
        ),
        "lineage_growth_timeseries": pd.DataFrame(
            {
                "lineage_id": [1, 1, 2, 2],
                "frame": [0, 1, 0, 1],
                "total_area": [10.0, 20.0, 8.0, 16.0],
            }
        ),
        "lineage_points": pd.DataFrame(
            {
                "track_id": [1, 2, 3, 4],
                "frame": [0, 1, 0, 1],
                "lineage_id": [1, 1, 2, 2],
                "Area (µm²)": [10.0, 12.0, 8.0, 9.0],
                "Green_foci_count": [1, 2, 3, 4],
            }
        ),
        "lineage_events": pd.DataFrame(
            {"area_asymmetry": [0.5, 0.6]},
        ),
    }
    written = export_lineage_figures(Path(tmp_path), tables, stem="sample")
    assert any(p.suffix == ".png" for p in written)
    analysis = export_lineage_analysis_figures(Path(tmp_path), tables, stem="sample", lineage_ids=[1, 2])
    assert any("analysis" in str(p) for p in analysis)
