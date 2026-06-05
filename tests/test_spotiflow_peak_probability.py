"""Spotiflow predict() maps per-spot scores via details.prob, not a third spots column."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bactomate.modules.foci_utils import probability_from_spotiflow_details


def test_probability_from_details_prob_array() -> None:
    details = SimpleNamespace(prob=np.array([0.12, 0.87, 0.45]))
    assert probability_from_spotiflow_details(details, 1) == 0.87
    assert probability_from_spotiflow_details(details, 99) is None


def test_probability_fallback_third_spots_column() -> None:
    row = np.array([10.0, 20.0, 0.33])
    assert probability_from_spotiflow_details(None, 0, spots_row=row) == 0.33


def test_details_prob_preferred_over_spots_column() -> None:
    details = SimpleNamespace(prob=np.array([0.9]))
    row = np.array([1.0, 2.0, 0.1])
    assert probability_from_spotiflow_details(details, 0, spots_row=row) == 0.9
