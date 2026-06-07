"""Qt thread helper smoke tests."""
from __future__ import annotations

from unittest.mock import MagicMock

from bactomate.gui.qt_thread_utils import stop_qthread_worker


def test_stop_qthread_worker_not_running() -> None:
    worker = MagicMock()
    worker.isRunning.return_value = False
    assert stop_qthread_worker(worker) is True


def test_stop_qthread_worker_requests_cancel() -> None:
    worker = MagicMock()
    worker.isRunning.return_value = True
    worker.wait.return_value = True
    assert stop_qthread_worker(worker) is True
    worker.request_cancel.assert_called_once()
