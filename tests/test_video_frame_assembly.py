"""Growth-curve video frame assembly smoke tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bactomate.gui.workers import (
    VIDEO_FI_LAYOUT_MERGED,
    VIDEO_FI_LAYOUT_SEPARATE,
    apply_video_montage_overlays,
    compose_growth_curve_combined_frame,
    load_growth_curve_video_source,
)


def test_compose_merged_frame_video_only() -> None:
    frames = np.stack(
        [np.full((16, 24, 3), fill, dtype=np.uint8) for fill in (10, 20, 30)],
        axis=0,
    )
    out = compose_growth_curve_combined_frame(
        1,
        video_frames=frames,
        fi_layout=VIDEO_FI_LAYOUT_MERGED,
    )
    assert out.shape == (16, 24, 3)
    assert out.dtype == np.uint8
    assert int(out.mean()) == 20


def test_compose_frame_with_graph_panel() -> None:
    video = np.full((12, 20, 3), 128, dtype=np.uint8)
    frames = np.array([0, 1, 2], dtype=float)
    areas = np.array([1.0, 1.5, 2.0], dtype=float)
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    out = compose_growth_curve_combined_frame(
        0,
        video_frames=video[np.newaxis, ...],
        frames=frames,
        relative_areas=areas,
        time_minutes=times,
        fi_layout=VIDEO_FI_LAYOUT_MERGED,
        graph_width_ratio=0.4,
    )
    assert out.ndim == 3
    assert out.shape[0] == 12
    assert out.shape[2] == 3
    assert out.shape[1] > 20


def test_separate_tiles_layout(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    phase = tmp_path / "sample_phase.tif"
    fi = tmp_path / "sample_fi_GFP.tif"
    stack = np.arange(3 * 10 * 10, dtype=np.uint8).reshape(3, 10, 10)
    tifffile.imwrite(phase, stack)
    tifffile.imwrite(fi, stack + 1)
    source = load_growth_curve_video_source(
        [phase, fi],
        fi_layout=VIDEO_FI_LAYOUT_SEPARATE,
    )
    assert source.fi_layout == VIDEO_FI_LAYOUT_SEPARATE
    assert source.tiles is not None
    assert len(source.tiles) >= 2
    out = compose_growth_curve_combined_frame(
        0,
        tiles=source.tiles,
        fi_layout=VIDEO_FI_LAYOUT_SEPARATE,
        reference_width=10,
    )
    assert out.shape[0] == 10
    assert out.shape[1] >= 10


def test_apply_video_montage_overlays_noop_when_disabled() -> None:
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    out = apply_video_montage_overlays(frames, show_mask_contours=False, show_foci=False)
    np.testing.assert_array_equal(out, frames)
