"""
Phase 1.3: Minimal tests for session apply (set_session_params / get_session_params round-trip).
Tests that applying session params to AnalyzeMasksTab updates widgets and can be read back.
Run with: pytest tests/test_session_apply.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None

try:
    from PyQt6.QtWidgets import QApplication
    from bactimate.gui.tabs.analyze_masks import AnalyzeTab
    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False


def test_analyze_masks_set_then_get_session_params_round_trip():
    """set_session_params then get_session_params returns equivalent values for the keys we set."""
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    tab = AnalyzeTab(parent=None)
    tab.parent_window = None
    # Values that map to widget types we know exist (from get_session_params)
    data = {
        "pixel_size": 0.2,
        "min_aspect_ratio": 1.8,
        "max_aspect_ratio": 8.0,
        "min_eccentricity": 0.7,
        "min_solidity": 0.9,
        "min_area": 5.0,
        "max_area": 500.0,
        "min_width": 0.5,
        "max_width": 20.0,
        "min_length": 1.0,
        "detailed": True,
        "create_vis": False,
        "export_csv": True,
        "exclude_border": True,
    }
    tab.set_session_params(data)
    out = tab.get_session_params()
    for key, expected in data.items():
        assert key in out, f"Missing key {key} in get_session_params()"
        assert out[key] == expected, f"Mismatch for {key}: got {out[key]!r}, expected {expected!r}"


def test_analyze_masks_apply_session_params_ignores_unknown_keys():
    """set_session_params with extra keys does not raise; unknown keys are ignored."""
    if not QT_AVAILABLE:
        if pytest:
            pytest.skip("PyQt6 or AnalyzeMasksTab not available")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    tab = AnalyzeTab(parent=None)
    tab.parent_window = None
    data = {
        "pixel_size": 0.15,
        "unknown_key_xyz": 999,
        "another_unknown": "ignored",
    }
    tab.set_session_params(data)
    out = tab.get_session_params()
    assert out["pixel_size"] == 0.15
    assert "unknown_key_xyz" not in out
    assert "another_unknown" not in out


if __name__ == "__main__":
    test_analyze_masks_set_then_get_session_params_round_trip()
    test_analyze_masks_apply_session_params_ignores_unknown_keys()
    print("All session_apply tests passed.")
