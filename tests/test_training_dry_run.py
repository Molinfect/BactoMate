"""Training validation / dry-run error surfaces without GPU work."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from bactomate.gui.tabs.training import TrainingTab
from bactomate.scripts.check_omnipose_pipeline import _check_training_data


def test_training_validate_pairs_missing_dir() -> None:
    tab = TrainingTab.__new__(TrainingTab)
    tab.train_dir_edit = MagicMock()
    tab.train_dir_edit.text.return_value = ""
    tab.img_filter_edit = MagicMock()
    tab.mask_filter_edit = MagicMock()
    tab._pair_status = MagicMock()
    tab._validate_pairs()
    tab._pair_status.setText.assert_called_with("Set a training directory.")


def test_check_training_data_dry_run_empty_dir(tmp_path: Path) -> None:
    result = _check_training_data(tmp_path, dry_run=True)
    assert result["ok"] is False
    assert "pair" in result.get("error", "").lower()
