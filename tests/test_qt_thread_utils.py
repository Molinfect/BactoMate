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


_RETIRE_SCRIPT = r"""
import gc, sys, threading, time
from PyQt6.QtCore import QCoreApplication, QEvent, QThread, pyqtSignal
from bactomate.gui import qt_thread_utils as qtu

app = QCoreApplication(sys.argv)
gate = threading.Event()
release_threads = []
orig_release = qtu._release_owned_qthread

def spy(w):
    release_threads.append(threading.current_thread() is threading.main_thread())
    orig_release(w)

qtu._release_owned_qthread = spy

class Blocking(QThread):
    result_ready = pyqtSignal(object)
    def run(self):
        # Ignores interruption, like a native model load.
        gate.wait(5)
        self.result_ready.emit("late")

def pump():
    app.processEvents()
    # processEvents() alone never runs deleteLater(); flush deferred deletes explicitly.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)

results = []
MODE = sys.argv[1]
w = Blocking()
w.setObjectName("FociPeaksPreviewWorker")
w.result_ready.connect(results.append)
if MODE == "legacy":
    w.start()
else:
    qtu.start_owned_qthread(w)
time.sleep(0.05)
if MODE == "legacy":
    # Pre-fix cleanup: request cancel, disconnect, then delete while still running.
    w.requestInterruption()
    w.result_ready.disconnect(results.append)
    w.deleteLater()
else:
    qtu.retire_qthread(w, result_signal=w.result_ready, slot=results.append)
del w
gc.collect()
deadline = time.time() + 3
while time.time() < deadline:
    pump()
    time.sleep(0.01)
gate.set()
deadline = time.time() + 3
while qtu._OWNED_QTHREADS and time.time() < deadline:
    pump()
    time.sleep(0.01)
for _ in range(20):
    pump()
assert not qtu._OWNED_QTHREADS, "thread not released"
assert release_threads == [True], release_threads
assert results == [], results
print("OK")
"""


def _run_retire_script(mode: str):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", _RETIRE_SCRIPT, mode],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)},
    )


def test_retire_qthread_keeps_running_worker_alive() -> None:
    proc = _run_retire_script("retire")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_deleting_running_qthread_aborts_control() -> None:
    """Control: the pre-fix pattern reproduces the SIGABRT seen in the crash report."""
    proc = _run_retire_script("legacy")
    assert proc.returncode != 0
    assert "still running" in proc.stderr


def test_retire_qthread_deletes_never_started_worker() -> None:
    worker = MagicMock()
    worker.isRunning.return_value = False
    signal = MagicMock()
    slot = MagicMock()
    from bactomate.gui.qt_thread_utils import retire_qthread

    retire_qthread(worker, result_signal=signal, slot=slot)
    signal.disconnect.assert_called_once_with(slot)
    worker.deleteLater.assert_called_once()


_SHADOWED_SCRIPT = r"""
import gc, sys, threading, time
from PyQt6.QtCore import QCoreApplication, QEvent, QObject, QThread, pyqtSignal
from bactomate.gui import qt_thread_utils as qtu

app = QCoreApplication(sys.argv)
gate = threading.Event()

class ShadowedFinished(QThread):
    # Same shape as WorkerThread: custom finished fires from inside run().
    finished = pyqtSignal(bool, str)
    def run(self):
        gate.wait(5)
        self.finished.emit(True, "done")

def pump():
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)

owner = QObject()
w = ShadowedFinished(owner)
w.start()
time.sleep(0.05)
qtu.retire_qthread(w)
# Parent dialog/tab goes away while the thread is still blocked.
owner.deleteLater()
del owner, w
gc.collect()
for _ in range(50):
    pump()
    time.sleep(0.01)
assert len(qtu._OWNED_QTHREADS) == 1
gate.set()
deadline = time.time() + 3
while qtu._OWNED_QTHREADS and time.time() < deadline:
    pump()
    time.sleep(0.01)
assert not qtu._OWNED_QTHREADS, "released only via native QThread.finished"
print("OK")
"""


def test_retire_qthread_survives_parent_deletion_with_shadowed_finished() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", _SHADOWED_SCRIPT],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
