from __future__ import annotations

import sys
import types

import numpy as np


def test_detect_peaks_spotiflow_reuses_cached_model(monkeypatch):
    from bactomate.modules import foci_utils

    foci_utils.clear_spotiflow_model_cache()
    calls = {"from_pretrained": 0}

    class FakeSpotiflow:
        @classmethod
        def from_pretrained(cls, model_name, map_location="auto"):
            calls["from_pretrained"] += 1
            return cls()

        def predict(self, image, **kwargs):
            return np.array([[1, 1, 0.9]], dtype=float), None

    spotiflow_pkg = types.ModuleType("spotiflow")
    spotiflow_model = types.ModuleType("spotiflow.model")
    spotiflow_model.Spotiflow = FakeSpotiflow
    monkeypatch.setitem(sys.modules, "spotiflow", spotiflow_pkg)
    monkeypatch.setitem(sys.modules, "spotiflow.model", spotiflow_model)

    img = np.ones((3, 3), dtype=np.float32)
    p1 = foci_utils.detect_peaks_spotiflow(img, model_name="general", normalize=False, raise_on_error=True)
    p2 = foci_utils.detect_peaks_spotiflow(img, model_name="general", normalize=False, raise_on_error=True)

    assert calls["from_pretrained"] == 1
    assert p1 and p2
    assert p1[0]["y"] == 1
    assert p2[0]["x"] == 1
    foci_utils.clear_spotiflow_model_cache()
