"""A preview thread that outlives the release timeout must be retired, not left on the owner."""
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QObject, QThread
from PyQt6.QtWidgets import QApplication

from bactomate.gui.filter_preview_apply import _safe_release_preview_worker
from bactomate.gui.qt_thread_utils import _OWNED_QTHREADS


class _BusyThread(QThread):
    """Ignores quit() until interrupted, like a worker stuck in a long computation."""

    def run(self):
        while not self.isInterruptionRequested():
            time.sleep(0.005)


def test_release_retires_thread_that_does_not_stop_in_time():
    app = QApplication.instance() or QApplication([])
    thread = _BusyThread()
    worker = QObject()
    owner = SimpleNamespace(_preview_worker=worker, _preview_thread=thread)
    thread.start()
    deadline = time.monotonic() + 2
    while not thread.isRunning() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert thread.isRunning()

    _safe_release_preview_worker(owner, wait_ms=20)

    assert owner._preview_worker is None
    assert owner._preview_thread is None
    assert thread.isInterruptionRequested()
    assert thread.wait(2000)
    for _ in range(20):
        app.processEvents()
    assert thread not in _OWNED_QTHREADS
