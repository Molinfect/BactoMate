"""Filter preview foci hover tooltip helpers."""
from __future__ import annotations

from pathlib import Path

from bactomate.gui.tabs._analyze_masks_preview_filtering import (
    _cached_foci_hover_data,
    _format_foci_peak_tooltip,
    _foci_hover_cache_key,
    build_foci_hover_data,
    collect_visible_foci_peaks_for_hover,
)


class _DlgStub:
    pass


def test_foci_peak_tooltip_includes_metrics() -> None:
    tip = _format_foci_peak_tooltip(
        {
            "y": 10,
            "x": 20,
            "intensity": 1500.5,
            "probability": 0.87,
            "area_px": 12,
            "mean_intensity": 1400.0,
            "rel_intensity": 2.3,
            "per_cell_ratio": 3.1,
            "cell_id": 5,
        },
        status="pass",
    )
    assert "pass" in tip.lower()
    assert "y=10" in tip
    assert "Area (px): 12" in tip
    assert "Frame-rel. intens.: 2.30" in tip
    assert "Per-cell ratio: 3.10" in tip
    assert "Cell ID: 5" in tip


def test_tooltip_cell_id_outside_shows_em_dash() -> None:
    import numpy as np
    from bactomate.gui.tabs._analyze_masks_preview_filtering import (
        _format_foci_metric_value,
        _format_foci_peak_tooltip,
    )

    assert _format_foci_metric_value({"cell_id": 0}, "cell_id") == "—"
    assert _format_foci_metric_value({"cell_id": 3}, "cell_id") == "3"
    mask = np.zeros((20, 20), dtype=np.int32)
    tip = _format_foci_peak_tooltip(
        {"y": 0, "x": 0, "cell_id": 0, "intensity": 10.0},
        status="fail",
        mask_2d=mask,
    )
    assert "Cell ID: —" in tip
    assert "Cell ID: 0" not in tip


def test_collect_visible_foci_peaks_includes_all_channels() -> None:
    sources = [Path("a_foci_GFP.tif"), Path("b_foci_dsRed.tif")]
    ch_map = {
        "GFP": {
            "pass_peaks": [{"y": 1, "x": 2}],
            "fail_peaks": [{"y": 3, "x": 4}],
        },
        "dsRed": {
            "pass_peaks": [{"y": 5, "x": 6}],
            "fail_peaks": [{"y": 7, "x": 8}],
        },
    }

    def _all_visible(_si: int, _fp) -> bool:
        return True

    def _peaks(ch_key: str, _fp: Path) -> tuple[list, list]:
        payload = ch_map[ch_key]
        return list(payload["pass_peaks"]), list(payload["fail_peaks"])

    p, f = collect_visible_foci_peaks_for_hover(
        overlay_sources=sources,
        channel_visible=_all_visible,
        channel_peaks=_peaks,
    )
    assert len(p) == 2
    assert len(f) == 2
    assert {x["_foci_channel"] for x in p} == {"GFP", "dsRed"}
    keys = {(int(x["y"]), int(x["x"])) for x in f}
    assert (3, 4) in keys and (7, 8) in keys


def test_build_foci_hover_combines_colocalized_channels() -> None:
    hover = build_foci_hover_data(
        [
            {"y": 10, "x": 10, "area_px": 9, "_foci_channel": "GFP"},
            {"y": 10, "x": 10, "area_px": 9, "_foci_channel": "dsRed"},
        ],
        [],
        (100, 100),
        (100, 100),
        {"foci_blob_max_radius": 5},
    )
    assert len(hover) == 1
    assert "GFP" in hover[0][2] and "dsRed" in hover[0][2]


def test_build_foci_hover_includes_fail_peaks() -> None:
    hover = build_foci_hover_data(
        [{"y": 1, "x": 1, "area_px": 9}],
        [{"y": 2, "x": 2, "area_px": 9, "fail_reasons": ["low probability"]}],
        (100, 100),
        (100, 100),
        {"foci_blob_max_radius": 5},
    )
    assert len(hover) == 2
    tips = [h[2] for h in hover]
    assert any("FAIL" in t for t in tips)


def test_build_foci_hover_data_scales_coordinates() -> None:
    peaks = [{"y": 100, "x": 200, "area_px": 9, "intensity": 1.0}]
    hover = build_foci_hover_data(
        peaks,
        [],
        (1000, 1000),
        (500, 500),
        {"foci_blob_max_radius": 5},
    )
    assert len(hover) == 1
    x, y, tip, r = hover[0]
    assert abs(x - 100.0) < 1e-3
    assert abs(y - 50.0) < 1e-3
    assert r > 0
    assert "pass" in tip.lower()


def test_foci_hover_cache_key_includes_geometry() -> None:
    k1 = _foci_hover_cache_key(
        3, "sig", draw_blobs=True, mask_shape=(100, 200), display_shape=(50, 100)
    )
    k2 = _foci_hover_cache_key(
        3, "sig", draw_blobs=True, mask_shape=(100, 200), display_shape=(60, 100)
    )
    assert k1 != k2


def test_cached_foci_hover_data_reuses_entry() -> None:
    dlg = _DlgStub()
    peaks = [{"y": 10, "x": 20, "area_px": 9, "intensity": 1.0}]
    key = _foci_hover_cache_key(
        0, "s", draw_blobs=False, mask_shape=(100, 100), display_shape=(50, 50)
    )
    first = _cached_foci_hover_data(
        dlg,
        cache_key=key,
        pass_peaks=peaks,
        fail_peaks=[],
        mask_shape=(100, 100),
        display_shape=(50, 50),
        settings={"foci_blob_max_radius": 5},
    )
    second = _cached_foci_hover_data(
        dlg,
        cache_key=key,
        pass_peaks=peaks,
        fail_peaks=[],
        mask_shape=(100, 100),
        display_shape=(50, 50),
        settings={"foci_blob_max_radius": 5},
    )
    assert first is not None and len(first) == 1
    assert second is first
