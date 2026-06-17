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


def test_build_lineage_events_includes_review_and_1based_frame() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import build_lineage_events, track_lineage_metadata

    edges = pd.DataFrame(
        [
            {
                "parent_track_id": 1,
                "daughter_track_id": 2,
                "division_frame": 2,
                "parent_last_frame": 1,
                "link_status": "candidate",
            }
        ]
    )
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "frame": [0, 1, 2, 3],
            "y": [0.0, 0.0, 50.0, 51.0],
            "x": [0.0, 0.0, 0.0, 0.0],
        }
    )
    meta = track_lineage_metadata(edges, tracks)
    events = build_lineage_events(edges, tracks, lineage_tracks_meta=meta, pixel_size_um=0.1)
    assert int(events["division_frame_1based"].iloc[0]) == 3
    assert events["review_status"].iloc[0] == "candidate"
    assert float(events["division_distance_px"].iloc[0]) == 50.0


def test_resolve_lineage_area_unit_prefers_analysis_over_mislabeled_metric() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import (
        compute_lineage_analysis_summary,
        resolve_lineage_area_unit,
    )

    growth_ts = pd.DataFrame(
        {
            "lineage_id": [2, 2],
            "frame": [0, 9],
            "total_area": [3.1, 11.2],
            "metric_used": ["mask_px", "mask_px"],
        }
    )
    tables = {
        "lineage_points": pd.DataFrame(
            {
                "track_id": [2],
                "frame": [0],
                "lineage_id": [2],
                "Area (µm²)": [3.1],
            }
        ),
        "lineage_growth_timeseries_curated": growth_ts,
    }
    assert resolve_lineage_area_unit(tables, growth_ts=growth_ts) == "µm²"
    summary = compute_lineage_analysis_summary(tables, lineage_id=2, growth_ts=growth_ts)
    assert summary["area_unit"] == "µm²"
    assert summary["area_start"] == 3.1
    assert summary["area_end"] == 11.2


def test_enrich_lineage_tables_adds_growth_timeseries_and_tree() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import enrich_lineage_tables

    base = {
        "lineage_summary": pd.DataFrame(),
        "lineage_growth": pd.DataFrame(),
        "lineage_edges": pd.DataFrame(
            [{"parent_track_id": 1, "daughter_track_id": 2, "division_frame": 1, "parent_last_frame": 0}]
        ),
        "lineage_tracks": pd.DataFrame({"lineage_id": [1], "track_id": [1]}),
        "lineage_cells": pd.DataFrame(),
        "lineage_points": pd.DataFrame(
            {"track_id": [1, 2], "frame": [0, 1], "y": [0, 0], "x": [0, 0], "lineage_id": [1, 1], "Cell_ID": [1, 2]}
        ),
    }
    tracks = pd.DataFrame(
        {"track_id": [1, 2], "frame": [0, 1], "y": [0, 0], "x": [0, 0], "Cell_ID": [1, 2]}
    )
    area = pd.DataFrame({"track_id": [1, 2], "frame": [0, 1], "area": [10.0, 5.0], "area_source": ["mask_px"] * 2})
    out = enrich_lineage_tables(base, tracks_df=tracks, edges_df=base["lineage_edges"], area_df=area)
    assert "lineage_growth_timeseries" in out
    assert len(out["lineage_growth_timeseries"]) >= 1
    assert "lineage_tree_nodes" in out
    assert len(out["lineage_tree_nodes"]) >= 1


def test_build_biology_track_labels_tree() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import (
        biology_label_map,
        build_biology_track_labels,
        format_biology_division_split,
    )

    tracks = pd.DataFrame(
        {"track_id": [5, 1, 11], "frame": [0, 2, 2], "y": [0, 0, 0], "x": [0, 0, 0]}
    )
    edges = pd.DataFrame(
        [
            {"parent_track_id": 5, "daughter_track_id": 1, "division_frame": 2, "link_status": "candidate"},
            {"parent_track_id": 5, "daughter_track_id": 11, "division_frame": 2, "link_status": "candidate"},
        ]
    )
    labels = build_biology_track_labels(edges, tracks)
    label_map = biology_label_map(labels)
    assert label_map[5] == "Cell 1"
    assert label_map[1] == "Cell 1.1"
    assert label_map[11] == "Cell 1.2"
    events = pd.DataFrame(
        [
            {"parent_track_id": 5, "daughter_track_id": 1, "division_frame": 2},
            {"parent_track_id": 5, "daughter_track_id": 11, "division_frame": 2},
        ]
    )
    split = format_biology_division_split(5, 1, 2, events, label_map)
    assert split == "Cell 1 → Cell 1.1 + Cell 1.2 @ frame 3"


def test_terminal_frame_division_is_flagged() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import (
        REVIEW_STATUS_FLAGGED,
        build_lineage_events,
        refresh_lineage_events_table,
    )

    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2],
            "frame": [0, 1, 2],
            "y": [0, 0, 1],
            "x": [0, 0, 1],
        }
    )
    edges = pd.DataFrame(
        [
            {
                "parent_track_id": 1,
                "daughter_track_id": 2,
                "division_frame": 2,
                "parent_last_frame": 1,
                "link_status": "candidate",
            }
        ]
    )
    events = build_lineage_events(edges, tracks)
    assert bool(events["has_post_frame"].iloc[0]) is False
    assert events["review_status"].iloc[0] == REVIEW_STATUS_FLAGGED

    events["review_status"] = "accepted"
    refreshed = refresh_lineage_events_table({"lineage_edges": edges, "lineage_events": events}, tracks)
    assert refreshed["review_status"].iloc[0] == REVIEW_STATUS_FLAGGED


def test_write_lineage_report_workbook_sheets(tmp_path) -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import write_lineage_report_workbook

    tables = {
        "lineage_summary": pd.DataFrame({"lineage_id": [1], "n_tracks": [2]}),
        "lineage_events": pd.DataFrame(
            {"parent_track_id": [1], "daughter_track_id": [2], "review_status": ["candidate"]}
        ),
        "lineage_tracks_meta": pd.DataFrame({"track_id": [1, 2], "generation": [0, 1]}),
        "lineage_growth_timeseries": pd.DataFrame({"lineage_id": [1], "frame": [0], "total_area": [10.0]}),
        "lineage_qc": pd.DataFrame({"flag_type": ["track_gap"], "track_id": [1], "message": ["gap"]}),
    }
    out = tmp_path / "lineage_report.xlsx"
    write_lineage_report_workbook(out, tables, run_info_rows=[("Schema version", "2")])
    sheets = pd.read_excel(out, sheet_name=None)
    assert "Lineage Summary" in sheets
    assert "Division Events" in sheets
    assert "Growth Timeseries" in sheets


def test_apply_lineage_curation_manual_edit_and_reject_division() -> None:
    import numpy as np
    import pandas as pd

    from bactomate.modules.lineage_integration import (
        REVIEW_STATUS_ACCEPTED,
        REVIEW_STATUS_REJECTED,
        apply_lineage_curation_to_run,
    )

    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 3, 3, 8],
            "frame": [0, 1, 2, 9, 10, 11],
            "y": [10.0, 10.0, 12.0, 20.0, 20.0, 22.0],
            "x": [10.0, 10.0, 12.0, 30.0, 30.0, 32.0],
        }
    )
    edges = pd.DataFrame(
        [
            {
                "parent_track_id": 3,
                "daughter_track_id": 8,
                "division_frame": 11,
                "parent_last_frame": 10,
                "link_status": "candidate",
            }
        ]
    )
    masks_tracked = np.zeros((12, 40, 40), dtype=np.uint16)
    masks_input = np.zeros((12, 40, 40), dtype=np.uint16)
    masks_input[11, 15:18, 15:18] = 99

    curation = {
        "manual_lineage_edits": [
            {
                "frame": 11,
                "untracked_label": 99,
                "manual_track_id": -1,
                "parent_track_id": 3,
                "parent_y": 20.0,
                "parent_x": 30.0,
                "lineage_id": 3,
                "area_px": 9.0,
                "y": 16.0,
                "x": 16.0,
            }
        ],
        "division_reviews": [
            {
                "parent_track_id": 3,
                "daughter_track_id": 8,
                "division_frame": 11,
                "review_status": REVIEW_STATUS_ACCEPTED,
                "mother_y": 20.0,
                "mother_x": 30.0,
                "daughter_y": 22.0,
                "daughter_x": 32.0,
            },
            {
                "parent_track_id": 1,
                "daughter_track_id": 2,
                "division_frame": 2,
                "review_status": REVIEW_STATUS_REJECTED,
                "mother_y": 10.0,
                "mother_x": 10.0,
                "daughter_y": 12.0,
                "daughter_x": 12.0,
            },
        ],
    }
    tracks_out, edges_out, masks_out, review_map, stats = apply_lineage_curation_to_run(
        curation,
        tracks_df=tracks,
        edges_df=edges,
        masks_tracked=masks_tracked,
        masks_input=masks_input,
    )
    assert stats["manual_edits_applied"] == 1
    manual_ids = [tid for tid in tracks_out["track_id"].astype(int).tolist() if tid not in {1, 2, 3, 8}]
    assert len(manual_ids) == 1
    manual_id = int(manual_ids[0])
    assert int(masks_out[11, 16, 16]) == manual_id
    assert (3, 8) in review_map
    assert review_map[(3, 8)] == REVIEW_STATUS_ACCEPTED


def test_save_and_load_lineage_curation_roundtrip(tmp_path) -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import load_lineage_curation, save_lineage_curation

    tables = {
        "manual_lineage_edits": pd.DataFrame(
            [{"frame": 5, "untracked_label": 7, "manual_track_id": -1, "parent_track_id": 2}]
        ),
        "manual_ignored_untracked": {(5, 9)},
        "lineage_events": pd.DataFrame(
            [
                {
                    "parent_track_id": 2,
                    "daughter_track_id": 4,
                    "division_frame": 5,
                    "review_status": "accepted",
                    "mother_y": 1.0,
                    "mother_x": 2.0,
                    "daughter_y": 3.0,
                    "daughter_x": 4.0,
                }
            ]
        ),
    }
    path = tmp_path / "lineage_curation.json"
    save_lineage_curation(path, tables)
    loaded = load_lineage_curation(path)
    assert loaded is not None
    assert len(loaded["manual_lineage_edits"]) == 1
    assert loaded["division_reviews"][0]["review_status"] == "accepted"


def test_collapse_lineage_events_groups_binary_split() -> None:
    import pandas as pd

    from bactomate.modules.lineage_integration import (
        REVIEW_STATUS_ACCEPTED,
        REVIEW_STATUS_CANDIDATE,
        apply_division_review_status,
        collapse_lineage_events_for_display,
    )

    events = pd.DataFrame(
        [
            {
                "parent_track_id": 1,
                "daughter_track_id": 2,
                "division_frame": 7,
                "division_frame_1based": 8,
                "generation": 1,
                "review_status": REVIEW_STATUS_CANDIDATE,
                "link_status": "candidate",
                "has_post_frame": True,
            },
            {
                "parent_track_id": 1,
                "daughter_track_id": 3,
                "division_frame": 7,
                "division_frame_1based": 8,
                "generation": 1,
                "review_status": REVIEW_STATUS_CANDIDATE,
                "link_status": "candidate",
                "has_post_frame": True,
            },
        ]
    )
    collapsed = collapse_lineage_events_for_display(events)
    assert len(collapsed) == 1
    assert int(collapsed.iloc[0]["parent_track_id"]) == 1
    assert int(collapsed.iloc[0]["division_frame"]) == 7

    updated = apply_division_review_status(
        events, parent_track_id=1, division_frame=7, review_status=REVIEW_STATUS_ACCEPTED
    )
    assert (updated["review_status"] == REVIEW_STATUS_ACCEPTED).all()
    collapsed_after = collapse_lineage_events_for_display(updated)
    assert collapsed_after.iloc[0]["review_status"] == REVIEW_STATUS_ACCEPTED


def test_resolve_lineage_source_finds_tracking_under_parent_folder(tmp_path) -> None:
    from bactomate.modules.lineage_integration import discover_lineage_tracking_dirs, resolve_lineage_source

    tif_files = tmp_path / "Tif_files"
    out = tif_files / "_tracking" / "Merged-1_phase_mask_filtered_tracking"
    tables = out / "tables"
    tables.mkdir(parents=True)
    (tables / "lineage_points.parquet").write_bytes(b"")
    (out / "figures").mkdir()

    discovered = discover_lineage_tracking_dirs(tif_files)
    assert len(discovered) == 1
    assert discovered[0].name == "Merged-1_phase_mask_filtered_tracking"

    parent, stem, fig_dir = resolve_lineage_source(tif_files)
    assert parent == out
    assert stem == "Merged-1_phase_mask_filtered"
    assert fig_dir == out / "figures"
