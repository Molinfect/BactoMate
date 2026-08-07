"""Resolution-aware styling for tracking TIFF exports."""

import numpy as np

from bactomate.gui.workers.tracking_io import (
    blend_path_pixel_on_frame,
    should_draw_track_id_labels,
    tracking_export_viz_style,
)


def test_tracking_export_viz_style_small_crop_uses_thin_lines() -> None:
    style = tracking_export_viz_style(91, 96)
    assert style.thick_path is False
    assert style.font_size <= 8

    crop = tracking_export_viz_style(204, 232)
    assert crop.thick_path is False
    assert crop.label_scale >= 2


def test_tracking_export_upscale_factor_small_crop() -> None:
    from bactomate.gui.workers.tracking_io import tracking_export_upscale_factor, upscale_rgb_stack

    assert tracking_export_upscale_factor(204, 232) == 2
    assert tracking_export_upscale_factor(96, 91) == 4
    assert tracking_export_upscale_factor(1024, 1024) == 1
    stack = np.zeros((2, 10, 12, 3), dtype=np.uint8)
    up = upscale_rgb_stack(stack, 2)
    assert up.shape == (2, 20, 24, 3)


def test_overlay_export_upscale_keeps_native_coords_for_trajectories() -> None:
    """Disk export upscales, but in-memory overlay must stay native for track y/x."""
    from bactomate.gui.workers.tracking_io import tracking_export_upscale_factor, upscale_rgb_stack

    native = np.zeros((2, 96, 91, 3), dtype=np.uint8)
    upscale = tracking_export_upscale_factor(native.shape[1], native.shape[2])
    saved = upscale_rgb_stack(native, upscale)
    assert saved.shape == (2, 384, 364, 3)
    assert native.shape == (2, 96, 91, 3)


def test_scale_track_points_by_id_scales_coordinates() -> None:
    from bactomate.gui.workers.tracking_io import scale_track_points_by_id

    points = {3: [(0, 10.5, 20.0), (1, 12.0, 22.5)]}
    scaled = scale_track_points_by_id(points, 4)
    assert scaled[3][0] == (0, 42.0, 80.0)
    assert scaled[3][1] == (1, 48.0, 90.0)


def test_draw_trajectory_paths_on_frame_uses_antialiased_lines() -> None:
    from bactomate.gui.workers.tracking_io import draw_trajectory_paths_on_frame, tracking_export_viz_style

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    style = tracking_export_viz_style(40, 40)
    out = draw_trajectory_paths_on_frame(
        frame,
        track_points_by_id={1: [(0, 5.0, 5.0), (1, 35.0, 35.0)]},
        track_colors={1: (255, 0, 0)},
        unique_track_set={1},
        frame_idx=1,
        height=40,
        width=40,
        export_style=style,
        use_antialiased_paths=True,
    )
    assert int(out.sum()) > 0


def test_tracking_export_viz_style_large_frame_keeps_legacy_weight() -> None:
    style = tracking_export_viz_style(1024, 1024)
    assert style.thick_path is True
    assert style.label_scale == 2


def test_should_draw_track_id_labels_skips_dense_small_crops() -> None:
    assert should_draw_track_id_labels(30, 91, 96) is False
    assert should_draw_track_id_labels(10, 91, 96) is True


def test_blend_path_pixel_is_single_pixel() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    blend_path_pixel_on_frame(frame, 5, 5, (255, 0, 0), bg_weight=0.5, fg_weight=0.5)
    assert frame[5, 5, 0] == 127
    assert frame[4, 5, 0] == 0


def test_overlay_label_prefers_cell_id_when_mapped() -> None:
    from bactomate.gui.workers.tracking_io import overlay_label_for_track

    assert overlay_label_for_track(6, 3, 3) == "3"
    assert overlay_label_for_track(6, None, 3) == "6"
    assert overlay_label_for_track(6, 6, 3) == "6"


def test_compact_tracking_label_makes_namespace_explicit() -> None:
    from bactomate.gui.workers.tracking_io import compact_tracking_label

    assert compact_tracking_label(6, 3, cell_id_source="analysis", region_label=3) == "C3"
    assert compact_tracking_label(6, 6, cell_id_source="track", region_label=3) == "T6"
    assert compact_tracking_label(6, None, cell_id_source=None, region_label=3) == "T6"


def test_assign_cell_ids_uses_track_map_fallback() -> None:
    import pandas as pd

    from bactomate.gui.workers.tracking_io import assign_cell_ids_to_tracks_df

    tracks = pd.DataFrame({"track_id": [6], "frame": [0], "y": [5.0], "x": [5.0]})
    out = assign_cell_ids_to_tracks_df(tracks, None, trackid_to_cellid={6: 10})
    assert int(out.iloc[0]["Cell_ID"]) == 10
    assert str(out.iloc[0]["Cell_ID_Source"]) == "analysis"


def test_assign_cell_ids_from_mask_label_excel_map() -> None:
    import pandas as pd

    from bactomate.gui.workers.tracking_io import assign_cell_ids_to_tracks_df

    masks = np.zeros((1, 20, 20), dtype=np.uint16)
    masks[0, 10, 10] = 5
    tracks = pd.DataFrame({"track_id": [6], "frame": [0], "y": [10.0], "x": [10.0]})

    import bactomate.gui.workers.tracking_io as tio

    orig = tio.build_excel_frame_label_to_cell_id
    tio.build_excel_frame_label_to_cell_id = lambda _p: {(0, 5): 10}
    try:
        out = assign_cell_ids_to_tracks_df(tracks, masks, excel_file="dummy.xlsx")
    finally:
        tio.build_excel_frame_label_to_cell_id = orig
    assert int(out.iloc[0]["Cell_ID"]) == 10
    assert str(out.iloc[0]["Cell_ID_Source"]) == "analysis"


def test_assign_cell_ids_marks_track_fallback_source() -> None:
    import pandas as pd

    from bactomate.gui.workers.tracking_io import assign_cell_ids_to_tracks_df

    tracks = pd.DataFrame({"track_id": [6], "frame": [0], "y": [5.0], "x": [5.0]})
    out = assign_cell_ids_to_tracks_df(tracks, None)
    assert int(out.iloc[0]["Cell_ID"]) == 6
    assert str(out.iloc[0]["Cell_ID_Source"]) == "track"


def test_resolve_track_at_centroid_matches_nearest_row() -> None:
    import pandas as pd
    from bactomate.gui.workers.tracking_io import resolve_track_at_centroid

    df = pd.DataFrame(
        {
            "track_id": [6, 3],
            "frame": [0, 0],
            "y": [34.9, 10.0],
            "x": [43.2, 10.0],
            "Cell_ID": [3, 9],
        }
    )
    tid, cid = resolve_track_at_centroid(df, 0, 34.9, 43.2)
    assert tid == 6
    assert cid == 3
