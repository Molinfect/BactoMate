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
    assert style.font_size <= 6
    assert style.label_scale == 1


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


def test_assign_cell_ids_uses_track_map_fallback() -> None:
    import pandas as pd

    from bactomate.gui.workers.tracking_io import assign_cell_ids_to_tracks_df

    tracks = pd.DataFrame({"track_id": [6], "frame": [0], "y": [5.0], "x": [5.0]})
    out = assign_cell_ids_to_tracks_df(tracks, None, trackid_to_cellid={6: 10})
    assert int(out.iloc[0]["Cell_ID"]) == 10


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
