"""Tests for spatial_metrics module."""
from __future__ import annotations

import numpy as np
import pytest

from bactomate.modules.spatial_metrics import (
    aggregate_foci_spatial,
    apply_temporal_metrics,
    cell_contour_mask,
    focus_spatial_metrics,
    frame_summary_spatial_metrics,
)


def _rod_mask(h: int = 40, w: int = 12) -> np.ndarray:
    mask = np.zeros((60, 60), dtype=bool)
    y0, x0 = 10, 24
    mask[y0 : y0 + h, x0 : x0 + w] = True
    return mask


def test_focus_spatial_metrics_on_rod_center():
    mask = _rod_mask()
    cy, cx = 30.0, 30.0
    m = focus_spatial_metrics(cy, cx, mask, pixel_size=0.1, contour_thickness=1)
    assert np.isfinite(m["distance_to_contour_um"])
    assert m["distance_to_contour_um"] > 0
    assert 0.0 <= m["longitudinal_position"] <= 1.0
    assert np.isfinite(m["distance_to_nearest_pole_um"])


def test_aggregate_foci_spatial_empty():
    agg = aggregate_foci_spatial([])
    assert np.isnan(agg["foci_min_dist_contour_um"])


def test_apply_temporal_metrics_deltas():
    all_frame_data = [
        {
            "frame": 1,
            "cells": [
                {"cell_id": 1, "label": 1, "area_um2": 1.0, "foci_count": 0, "mean_fi": 10.0},
            ],
        },
        {
            "frame": 2,
            "cells": [
                {"cell_id": 1, "label": 1, "area_um2": 1.5, "foci_count": 1, "mean_fi": 12.0},
            ],
        },
    ]
    apply_temporal_metrics(all_frame_data)
    cell = all_frame_data[1]["cells"][0]
    assert cell["delta_area_um2"] == pytest.approx(0.5)
    assert cell["delta_foci_count"] == 1
    assert cell["delta_mean_fi"] == pytest.approx(2.0)
    assert cell["time_of_first_focus"] == 2


def test_frame_summary_spatial_metrics():
    cells = [
        {"foci_count": 2, "foci_min_dist_contour_um": 0.2, "foci_membrane_proximal_fraction": 0.5},
        {"foci_count": 0},
    ]
    summary = frame_summary_spatial_metrics(cells)
    assert summary["Median Foci per Cell"] == 1.0
    assert summary["Fraction Cells with Foci"] == 0.5
    assert summary["Mean Min Foci-Contour Distance (µm)"] == pytest.approx(0.2)


def test_cell_contour_mask_non_empty():
    mask = _rod_mask()
    contour = cell_contour_mask(mask, contour_thickness=1)
    assert contour.any()
    assert not np.any(contour & ~mask)
