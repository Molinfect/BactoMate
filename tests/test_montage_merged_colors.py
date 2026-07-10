from __future__ import annotations

import numpy as np

from bactomate.gui.tabs.post_processing import PostProcessingTab


def _tab() -> PostProcessingTab:
    tab = PostProcessingTab.__new__(PostProcessingTab)
    tab.montage_tile_color_overrides = {}
    tab.montage_tile_percentile_overrides = {}
    return tab


def test_montage_recompose_merged_uses_channel_color_overrides() -> None:
    tab = _tab()
    ch1 = np.linspace(0, 200, 16, dtype=np.float64).reshape(4, 4).astype(np.uint16)
    ch2 = np.linspace(0, 100, 16, dtype=np.float64).reshape(4, 4).astype(np.uint16)
    channel_tiles = [
        (ch1, "sample Phase", (1.0, 1.0, 1.0)),
        (ch2, "sample GFP", (0.0, 1.0, 0.0)),
    ]
    channel_meta = [
        {"kind": "channel", "channel_norm": "phase"},
        {"kind": "channel", "channel_norm": "gfp"},
    ]
    tab.montage_tile_color_overrides = {
        "sample Phase": (1.0, 86 / 255.0, 0.0),
    }
    merged = tab._montage_recompose_merged_tile(channel_tiles, channel_meta)
    assert merged is not None
    assert merged.shape == (4, 4, 3)
    bright = merged[3, 3]
    # Phase should render orange (strong red, weaker green) rather than neutral gray.
    assert bright[0] > bright[2]
    assert bright[0] > 0


def test_montage_recompose_merged_keeps_hidden_channel_colors() -> None:
    """Merged-only display should still use overrides from excluded channel tiles."""
    tab = _tab()
    phase = np.linspace(40, 220, 9, dtype=np.float64).reshape(3, 3).astype(np.uint16)
    gfp = np.linspace(10, 180, 9, dtype=np.float64).reshape(3, 3).astype(np.uint16)
    tiles = [
        (phase, "sample Phase", (1.0, 1.0, 1.0)),
        (gfp, "sample GFP", (0.0, 1.0, 0.0)),
    ]
    meta = [
        {"kind": "channel", "channel_norm": "phase"},
        {"kind": "channel", "channel_norm": "gfp"},
    ]
    tab.montage_tile_color_overrides = {}
    default_merged = tab._montage_recompose_merged_tile(tiles, meta)
    tab.montage_tile_color_overrides = {"sample Phase": (1.0, 86 / 255.0, 0.0)}
    orange_merged = tab._montage_recompose_merged_tile(tiles, meta)
    assert default_merged is not None and orange_merged is not None
    assert not np.array_equal(default_merged, orange_merged)
