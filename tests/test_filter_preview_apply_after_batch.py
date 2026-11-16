from __future__ import annotations

from pathlib import Path

import pytest


def test_request_filter_preview_apply_calls_stored_callback(tmp_path: Path) -> None:
    from bactomate.gui.filter_preview_utils import request_filter_preview_apply

    calls: list[str] = []

    class _Dlg:
        def isVisible(self) -> bool:
            return True

        def _pf_ui_to_settings(self) -> None:
            calls.append("sync")

        def _pf_apply_current(self) -> None:
            calls.append("apply")

    class _Tab:
        _filter_preview_dialog = _Dlg()

    assert request_filter_preview_apply(_Tab(), defer_ms=0) is True
    assert calls == ["sync", "apply"]


def test_request_filter_preview_apply_no_dialog() -> None:
    from bactomate.gui.filter_preview_utils import request_filter_preview_apply

    class _Tab:
        _filter_preview_dialog = None

    assert request_filter_preview_apply(_Tab()) is False


def test_wait_for_filter_preview_apply_idle_when_not_busy() -> None:
    pytest.importorskip("PyQt6")
    from bactomate.gui.filter_preview_utils import wait_for_filter_preview_apply_idle

    class _Lifecycle:
        def wait_until_idle(self, *, timeout_ms: int) -> bool:
            assert timeout_ms == 42
            return True

    class _Dlg:
        _filter_apply = _Lifecycle()

    assert wait_for_filter_preview_apply_idle(_Dlg(), timeout_ms=42) is True
