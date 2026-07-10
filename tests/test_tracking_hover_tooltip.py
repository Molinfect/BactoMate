"""Tracking preview hover tooltip formatting."""

from bactomate.gui.tabs.tracking import TrackingTab


def test_format_track_hover_tooltip_includes_preview_metrics() -> None:
    tip = TrackingTab._format_track_hover_tooltip(
        501,
        9,
        69.5,
        131.5,
        {
            "trajectory_length": 8,
            "duration_frames": 9,
            "gap_frames": 1,
            "displacement_px": 12.0,
            "path_length_px": 18.0,
            "mean_speed_px_frame": 2.0,
            "straightness": 0.67,
        },
        use_um=False,
        use_speed_physical=False,
        passes_filter=True,
        preview_n_frames=20,
    )
    assert "Track ID: 501" in tip
    assert "Preview metrics only (first 20 frames)" in tip
    assert "Preview (20 frames): 8 detections" in tip
    assert "Gaps: 1 missed frame" in tip
    assert "Displacement: 12.00 px · path: 18.00 px" in tip
    assert "Mean speed: 2.000 px/frame · straightness: 0.67" in tip
    assert "Passes filters: Yes" in tip


def test_format_track_hover_tooltip_uses_physical_units() -> None:
    tip = TrackingTab._format_track_hover_tooltip(
        3,
        0,
        10.0,
        20.0,
        {
            "trajectory_length": 2,
            "duration_frames": 2,
            "gap_frames": 0,
            "displacement_um": 1.25,
            "path_length_um": 1.8,
            "mean_speed_um_s": 0.45,
            "straightness": 0.9,
        },
        use_um=True,
        use_speed_physical=True,
        passes_filter=False,
        preview_n_frames=20,
        cell_id=42,
    )
    assert "Cell ID: 42" in tip
    assert "Track ID: 3" in tip
    assert "Preview metrics only (first 20 frames)" in tip
    assert "Displacement: 1.25 µm" in tip
    assert "Mean speed: 0.450 µm/s" in tip
    assert "Passes filters: No" in tip


def test_format_track_hover_tooltip_skips_speed_for_single_detection() -> None:
    tip = TrackingTab._format_track_hover_tooltip(
        1,
        0,
        5.0,
        5.0,
        {
            "trajectory_length": 1,
            "duration_frames": 1,
            "gap_frames": 0,
            "displacement_px": 0.0,
            "path_length_px": 0.0,
            "mean_speed_px_frame": 0.0,
            "straightness": 0.0,
        },
        use_um=False,
        use_speed_physical=False,
        passes_filter=None,
        preview_n_frames=20,
    )
    assert "Mean speed" not in tip
