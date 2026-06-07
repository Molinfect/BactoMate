"""Shared helpers for cooperative QThread shutdown in the GUI."""
from __future__ import annotations

from typing import Any, Optional


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
