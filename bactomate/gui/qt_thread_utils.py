"""Shared helpers for cooperative QThread shutdown in the GUI."""
from __future__ import annotations

from typing import Any, Callable, Optional

# Workers started via ``start_owned_qthread`` stay referenced here until Qt's native
# ``QThread.finished`` fires. Deleting (or garbage-collecting) a QThread whose run()
# is still executing aborts the process ("QThread: Destroyed while thread ... is still
# running"), and cooperative cancellation cannot interrupt a blocking call such as
# Spotiflow/Omnipose model loading.
_OWNED_QTHREADS: set[Any] = set()
_QUIT_HOOK_INSTALLED = False
_QUIT_WAIT_MS = 10_000


def _native_finished(worker: Any) -> Any:
    """Qt's own ``QThread.finished`` (emitted on thread exit), even if a subclass shadows it.

    Several workers (e.g. ``WorkerThread``) declare a custom ``finished`` result signal that
    fires from inside run(), i.e. before the thread has exited.
    """
    from PyQt6.QtCore import QThread

    if isinstance(worker, QThread):
        return QThread.finished.__get__(worker, QThread)
    return worker.finished


def _adopt(worker: Any) -> None:
    _OWNED_QTHREADS.add(worker)
    _native_finished(worker).connect(lambda *_, w=worker: _release_owned_qthread(w))
    _install_quit_hook()


def _release_owned_qthread(worker: Any) -> None:
    if worker not in _OWNED_QTHREADS:
        return
    _OWNED_QTHREADS.discard(worker)
    try:
        # finished is emitted just before run() fully unwinds; wait() returns promptly.
        worker.wait()
        worker.deleteLater()
    except RuntimeError:
        pass


def _on_app_about_to_quit() -> None:
    for worker in list(_OWNED_QTHREADS):
        try:
            worker.requestInterruption()
            if worker.wait(_QUIT_WAIT_MS):
                _OWNED_QTHREADS.discard(worker)
                continue
            # Still blocked in native code: hand ownership to C++ so interpreter teardown
            # does not run ~QThread on a live thread. The process is exiting anyway.
            import PyQt6.sip as sip

            sip.transferto(worker, None)
        except Exception:
            pass


def _install_quit_hook() -> None:
    global _QUIT_HOOK_INSTALLED
    if _QUIT_HOOK_INSTALLED:
        return
    try:
        from PyQt6.QtCore import QCoreApplication

        app = QCoreApplication.instance()
        if app is None:
            return
        app.aboutToQuit.connect(_on_app_about_to_quit)
        _QUIT_HOOK_INSTALLED = True
    except Exception:
        pass


def start_owned_qthread(worker: Any) -> None:
    """Start ``worker`` and keep it alive until it actually exits, then ``deleteLater`` it.

    ``worker`` should be created without a QObject parent so closing/deleting a preview
    dialog cannot destroy it mid-run. Result delivery must use a custom signal (not a
    shadowed ``finished``) so handlers never see a half-exited thread.
    """
    _adopt(worker)
    worker.start()


def retire_qthread(
    worker: Any,
    *,
    result_signal: Any = None,
    slot: Optional[Callable[..., Any]] = None,
) -> None:
    """Detach a worker without blocking: drop its result, request cancellation.

    Never deletes a running thread. Owned workers are deleted when Qt's native
    ``finished`` fires; workers that were never started are deleted immediately.
    """
    if worker is None:
        return
    if result_signal is not None and slot is not None:
        try:
            result_signal.disconnect(slot)
        except (TypeError, RuntimeError):
            pass
    try:
        if worker.isRunning():
            worker.requestInterruption()
            if worker not in _OWNED_QTHREADS:
                # Legacy / unowned thread: adopt it so it outlives its Python references.
                try:
                    worker.setParent(None)
                except Exception:
                    pass
                _adopt(worker)
                if worker.isFinished():
                    _release_owned_qthread(worker)
            return
        if worker in _OWNED_QTHREADS:
            if worker.isFinished():
                # Native finished already ran or is queued; release is idempotent.
                _release_owned_qthread(worker)
            return
        worker.deleteLater()
    except RuntimeError:
        pass


def stop_qthread_worker(
    worker: Any,
    *,
    wait_ms: int = 120_000,
    terminate_ms: int = 5_000,
) -> bool:
    """Request stop and wait for a QThread-like worker. Returns True when not running."""
    if worker is None:
        return True
    try:
        if not hasattr(worker, "isRunning") or not worker.isRunning():
            return True
        cancel_fn = getattr(worker, "request_cancel", None)
        if callable(cancel_fn):
            try:
                cancel_fn()
            except Exception:
                pass
        if hasattr(worker, "requestInterruption"):
            try:
                worker.requestInterruption()
            except Exception:
                pass
        if hasattr(worker, "quit"):
            try:
                worker.quit()
            except Exception:
                pass
        if worker.wait(int(wait_ms)):
            return True
        if hasattr(worker, "terminate"):
            try:
                worker.terminate()
            except Exception:
                pass
            return bool(worker.wait(int(terminate_ms)))
    except Exception:
        return False
    return False


def close_preview_widget(widget: Any, *, wait_ms: int = 120_000) -> None:
    """Close a preview widget so WorkerGuard / closeEvent can stop background workers."""
    if widget is None:
        return
    try:
        widget.close()
    except RuntimeError:
        pass
    worker = getattr(widget, "_seg_worker", None) or getattr(widget, "_preview_worker", None)
    if worker is not None:
        stop_qthread_worker(worker, wait_ms=wait_ms)


def dismiss_embedded_segmentation_preview(owner: Any) -> None:
    """Stop segmentation preview workers before clearing owner tab references."""
    if owner is None:
        return
    close_fn = getattr(owner, "_on_embedded_segmentation_preview_closed", None)
    if callable(close_fn):
        try:
            close_fn()
            return
        except Exception:
            pass
    prev = getattr(owner, "_embedded_segmentation_preview", None)
    if prev is not None:
        close_preview_widget(prev)
        try:
            owner._embedded_segmentation_preview = None
        except Exception:
            pass
    prev_u = getattr(owner, "_embedded_unet_preview", None)
    if prev_u is not None:
        close_preview_widget(prev_u)
        try:
            owner._embedded_unet_preview = None
        except Exception:
            pass
