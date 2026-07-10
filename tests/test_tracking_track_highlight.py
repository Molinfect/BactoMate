"""Track highlight resolves mask by centroid label, not track_id == mask value."""

import numpy as np
import pandas as pd

from bactomate.gui.tabs.tracking import TrackingTab


def test_mask_region_at_centroid_uses_label_not_track_id():
    # Per-frame label 3 at centroid; track_id 785 must not select all pixels == 785.
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[8:12, 8:12] = 3
    mask[0:3, 0:3] = 785
    region = TrackingTab._mask_region_at_centroid(mask, 10, 10)
    assert region is not None
    assert region.sum() == 16
    assert not region[0, 0]


def test_mask_region_for_track_uses_tracks_df_centroid():
    masks = np.zeros((2, 20, 20), dtype=np.int32)
    masks[0, 8:12, 8:12] = 3
    masks[1, 14:18, 14:18] = 7
    tracks = pd.DataFrame(
        {
            "track_id": [785, 785],
            "frame": [0, 1],
            "y": [10, 16],
            "x": [10, 16],
        }
    )
    r0 = TrackingTab._mask_region_for_track_at_frame(masks, tracks, 0, 785)
    r1 = TrackingTab._mask_region_for_track_at_frame(masks, tracks, 1, 785)
    assert r0 is not None and r0.sum() == 16
    assert r1 is not None and r1.sum() == 16
    assert not r0[14, 14]
    assert not r1[10, 10]


def test_apply_highlight_draws_trajectory_up_to_frame():
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 2],
            "y": [5, 10, 15],
            "x": [5, 10, 15],
        }
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    out = TrackingTab._apply_selected_track_halo(frame, 1, 1, None, tracks)
    assert out[10, 10].any() or out[5, 5].any()


def test_apply_highlight_skips_frames_without_detection():
    tracks = pd.DataFrame(
        {
            "track_id": [1] * 3,
            "frame": [0, 2, 4],
            "y": [5, 10, 15],
            "x": [5, 10, 15],
        }
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    out = TrackingTab._apply_selected_track_halo(frame, 3, 1, None, tracks)
    assert np.array_equal(out, frame)


def test_draw_polyline_thickness_one_is_single_pixel():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    pts = np.array([[5, 5], [5, 7]], dtype=int)
    TrackingTab._draw_polyline_on_frame(frame, pts, color=(255, 0, 0), thickness=1)
    assert frame[5, 5, 0] == 255
    assert frame[5, 6, 0] == 255
    assert frame[4, 5, 0] == 0
    assert frame[6, 5, 0] == 0
