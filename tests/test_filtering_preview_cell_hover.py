"""Filter preview cell hover tooltip helpers."""
from __future__ import annotations

import numpy as np

from bactomate.gui.filter_preview_utils import (
    build_cell_hover_data,
    evaluate_cell_filter_details,
    format_cell_filter_tooltip,
)


def test_format_cell_filter_tooltip_includes_fail_reasons() -> None:
    tip = format_cell_filter_tooltip(
        {
            "label": 3,
            "status": "fail",
            "y": 10.0,
            "x": 20.0,
            "metrics": {"area_um2": 2.5, "aspect_ratio": 1.2},
            "fail_reasons": ["Area (µm²) 2.50 < min 5.00"],
        }
    )
    assert "Cell 3" in tip
    assert "FAIL" in tip
    assert "Area (µm²) 2.50" in tip
    assert "min 5.00" in tip


def test_evaluate_cell_filter_details_marks_small_area_fail() -> None:
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[10:20, 10:30] = 1
    settings = {
        "pixel_size": 0.108,
        "exclude_border": False,
        "min_aspect_ratio": 1.0,
        "max_aspect_ratio": 20.0,
        "min_eccentricity": 0.0,
        "min_solidity": 0.0,
        "min_area_um2": 5.0,
    }
    details = evaluate_cell_filter_details(mask, settings, accepted=set(), rejected={1})
    assert len(details) == 1
    assert details[0]["status"] == "fail"
    assert details[0]["fail_reasons"]


def test_build_cell_hover_data_scales_coordinates() -> None:
    hover = build_cell_hover_data(
        [{"label": 1, "status": "pass", "y": 10.0, "x": 20.0, "area_px": 100, "metrics": {}, "fail_reasons": []}],
        (20, 40),
        (40, 80),
    )
    assert len(hover) == 1
    x, y, tip, radius = hover[0]
    assert abs(x - 40.0) < 1e-6
    assert abs(y - 20.0) < 1e-6
    assert "Cell 1" in tip
    assert radius > 0
