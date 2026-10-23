"""Request and Qt teardown ordering for the filtering preview's Apply action."""
from __future__ import annotations

from copy import deepcopy
import time
from typing import Callable

from PyQt6.QtCore import QObject, QThread, Qt
from PyQt6.QtWidgets import QApplication

from bactomate.gui.filter_apply_progress import FilterApplyProgressDialog, _qt_alive
from bactomate.gui.log_config import write_gui_log_line
from bactomate.gui.preview.preview_window import WorkerGuard


def _safe_release_preview_worker(owner, attr: str='_preview_worker', wait_ms: int=10000) -> None:
    w = getattr(owner, attr, None)
    thread = getattr(owner, '_preview_thread', None)
    try:
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(wait_ms):
                return
    except Exception:
        return
    try:
        if w is not None:
            w.deleteLater()
    except Exception:
        pass
    try:
        if thread is not None:
            thread.deleteLater()
    except Exception:
        pass
    try:
        setattr(owner, attr, None)
    except Exception:
        pass
    try:
        owner._preview_thread = None
    except Exception:
        pass


class FilterPreviewApplyLifecycle(QObject):
    """Serialize Apply snapshots, reject obsolete results, and close after teardown.

    ``start`` handles filtering/display policy; this module owns the request until
    its worker thread has finished, even after a result or cancellation arrives.
    """

    def __init__(self, dialog, apply_button, apply_current: Callable, start: Callable) -> None:
        super().__init__(dialog)
        self._dialog = dialog
        self._button = apply_button
        self._apply_current = apply_current
        self._start = start
        self.generation = 0
        self._active = False
        self._active_generation = None
        self._closing = False
        self._pending = None
        self._commit = None
        self._progress = None
        self._plan_line = ''
        dialog.set_worker_guard(WorkerGuard(
            is_running=lambda: self._active,
            wait=self._wait,
            request_stop=self._request_close,
            disconnect=self._disconnect,
            on_close_while_running='defer_hide',
        ))

    @property
    def idle(self) -> bool:
        return not self._active and self._pending is None

    def request(self, frame_idx: int, settings: dict, settings_sig: str, source_key: tuple) -> None:
        """Keep the newest submitted snapshot, including nested channel settings."""
        if self._closing:
            return
        self.generation += 1
        self._pending = (self.generation, frame_idx, deepcopy(settings), settings_sig, tuple(source_key))
        if self._active:
            self._set_status('Queued…', busy=True)
        else:
            self._start_pending()

    def _start_pending(self) -> None:
        request, self._pending = self._pending, None
        if request is None:
            return
        self._active = True
        self._active_generation = request[0]
        self._set_status('Apply current filters', busy=True)
        try:
            self._start(*request)
        except Exception as exc:
            write_gui_log_line('Preview', f'Could not start filter preview Apply: {exc}')
        finally:
            # Cache-only Apply has no worker, but participates in the same queue.
            if getattr(self._dialog, '_preview_thread', None) is None:
                self._finish()

    def run_worker(self, worker, commit: Callable, plan_line: str) -> None:
        self._commit = commit
        self._plan_line = plan_line
        prog = FilterApplyProgressDialog(self._dialog)
        # The Qt event loop delivers async progress; nested event pumping races teardown.
        prog.set_pump_events(False)
        prog.set_progress(0, 'Applying filters', plan_line or 'Starting…')
        self._progress = self._dialog._progress_dialog = prog
        prog.canceled.connect(self.cancel)
        prog.show()
        thread = QThread(self._dialog)
        thread.setObjectName('AnalyzeMasksPreviewWorkerThread')
        worker.moveToThread(thread)
        self._dialog._preview_worker = worker
        self._dialog._preview_thread = thread
        worker.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        worker.result_ready.connect(self._on_result, Qt.ConnectionType.QueuedConnection)
        worker.result_ready.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(self._on_finished)
        thread.start()

    def _set_status(self, label: str, *, busy: bool) -> None:
        self._dialog.set_extra('filter_apply_busy', busy)
        self._dialog.set_primary_action(self._apply_current, label=label, enabled=not busy, visible=False)
        self._button.setEnabled(not busy)
        self._button.setText(label)

    def _on_progress(self, pct: int, step: str, detail: str = '') -> None:
        if self.sender() is not getattr(self._dialog, '_preview_worker', None):
            return
        prog = self._progress
        if _qt_alive(prog) and not prog._closing:
            combined = '\n'.join(p for p in (self._plan_line, str(detail or '').strip()) if p)
            prog.set_progress(pct, step, combined)

    def _dismiss_progress(self) -> None:
        prog, self._progress = self._progress, None
        if getattr(self._dialog, '_progress_dialog', None) is prog:
            self._dialog._progress_dialog = None
        if not _qt_alive(prog):
            return
        try:
            prog.canceled.disconnect(self.cancel)
        except (TypeError, RuntimeError):
            pass
        prog._closing = True
        prog.close()
        prog.deleteLater()

    def _on_result(self, *result) -> None:
        if int(result[0]) != self._active_generation:
            return
        self._dismiss_progress()
        if self._closing or int(result[0]) != self.generation:
            return
        self._dialog.set_extra('filter_apply_busy', False)
        self._dialog.set_extra('filter_cell_recomputing', False)
        if self._commit is not None:
            self._commit(*result)

    def cancel(self) -> None:
        """Cancel current and queued work; a subsequent Apply can queue again."""
        self.generation += 1
        self._pending = None
        thread = getattr(self._dialog, '_preview_thread', None)
        if thread is not None:
            thread.requestInterruption()
            thread.quit()
        if self._active:
            self._set_status('Cancelling…', busy=True)
            if _qt_alive(self._progress):
                self._progress.set_progress(0, 'Cancelling', 'Stopping current apply…')

    def _request_close(self) -> None:
        self._closing = True
        self.cancel()
        self._dismiss_progress()

    def retire(self, parent) -> None:
        """Release a replaced preview after its worker stops, outside the mount."""
        self._dialog.finished.connect(self._dialog.deleteLater)
        self._dialog.close()
        # Mount replacement deletes its old widgets, including a deferred close.
        self._dialog.setParent(parent)

    def _wait(self, ms: int) -> bool:
        thread = getattr(self._dialog, '_preview_thread', None)
        return thread is None or bool(thread.wait(ms))

    def _disconnect(self) -> None:
        self._closing = True
        self._pending = None
        self._release_worker()
        if getattr(self._dialog, '_preview_thread', None) is None:
            self._active = False

    def _release_worker(self) -> None:
        self._dismiss_progress()
        worker = getattr(self._dialog, '_preview_worker', None)
        thread = getattr(self._dialog, '_preview_thread', None)
        for signal, slot in (
            (getattr(worker, 'progress', None), self._on_progress),
            (getattr(worker, 'result_ready', None), self._on_result),
            (getattr(worker, 'result_ready', None), thread.quit if thread is not None else None),
            (getattr(thread, 'finished', None), self._on_finished),
        ):
            if signal is not None and slot is not None:
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        _safe_release_preview_worker(self._dialog)
        self._commit = None

    def _on_finished(self) -> None:
        if self.sender() is not getattr(self._dialog, '_preview_thread', None):
            return
        self._finish()

    def _finish(self) -> None:
        if not _qt_alive(self._dialog):
            return
        self._release_worker()
        self._active = False
        self._active_generation = None
        self._dialog.set_extra('filter_cell_recomputing', False)
        self._set_status('Apply current filters', busy=False)
        if self._closing:
            # WorkerGuard hid the dialog; reject still emits finished for its owner.
            self._dialog.reject()
        else:
            self._start_pending()

    def wait_until_idle(self, *, timeout_ms: int = 3_600_000) -> bool:
        """Batch finalize waits through queued Apply and the final thread cleanup."""
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        while not self.idle:
            if time.monotonic() >= deadline:
                return False
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        return True
