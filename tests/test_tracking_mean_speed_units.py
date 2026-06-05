"""Mean speed is reported in µm/s when calibration is set."""

import pandas as pd

from bactomate.gui.tabs.tracking import TrackingTab


def test_mean_speed_um_s():
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 2],
            "y": [0.0, 0.0, 0.0],
            "x": [0.0, 10.0, 20.0],
        }
    )
    # path = 20 px, pixel 0.325 µm/px → 6.5 µm; 3 frames at 1 s/frame → 2 s total
    metrics = TrackingTab._compute_track_metrics(tracks, pixel_size_um=0.325, time_per_frame_min=1.0 / 60.0)
    assert len(metrics) == 1
    assert abs(metrics.iloc[0]["mean_speed_um_s"] - 3.25) < 0.01
