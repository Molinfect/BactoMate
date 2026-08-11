"""Background queue that computes folder-preview tiles one at a time."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

TileComputeFn = Callable[[Path, int], tuple[Optional[np.ndarray], str]]
# Returns (rgb_uint8 HxWx3 or None, error_message)


def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert contiguous RGB uint8 array to QPixmap (copy-safe)."""
    from bactomate.gui.image_utils import to_qpixmap

    return to_qpixmap(np.asarray(rgb))


class FolderPreviewQueueWorker(QThread):
    """Process paths serially; emit one result per tile.

    Signals:
      tile_ready(index, path_str, pixmap_or_None, error_str)
      progress(done, total, path_str)
      finished_ok()
      failed(message)
    """

    tile_ready = pyqtSignal(int, str, object, str)
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: list[Path],
        compute_fn: TileComputeFn,
        *,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._paths = [Path(p) for p in paths]
        self._compute_fn = compute_fn
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True
        self.requestInterruption()

    def run(self) -> None:
        total = len(self._paths)
        try:
            for idx, path in enumerate(self._paths):
                if self._cancel or self.isInterruptionRequested():
                    break
                self.progress.emit(idx, total, path.name)
                err = ""
                pix: Optional[QPixmap] = None
                try:
                    rgb, err = self._compute_fn(path, idx)
                    if rgb is not None and not err:
                        pix = rgb_to_qpixmap(rgb)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    pix = None
                if self._cancel or self.isInterruptionRequested():
                    break
                self.tile_ready.emit(idx, str(path), pix, err or "")
                self.progress.emit(idx + 1, total, path.name)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
