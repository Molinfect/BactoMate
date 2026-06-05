"""Background load of mask / foci stacks when opening the filtering preview."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from bactomate.gui.utils import safe_imread
from bactomate.modules.image_stack_utils import coerce_stack_to_tyx


class FilterPreviewOpenWorker(QObject):
    """Load a mask TIFF off the GUI thread."""

    finished = pyqtSignal(object, str)  # masks np.ndarray | None, error message

    def __init__(self, mask_path: Path) -> None:
        super().__init__()
        self.setObjectName(type(self).__name__)
        self._mask_path = Path(mask_path)

    def run(self) -> None:
        err = ""
        masks = None
        try:
            raw = np.asarray(safe_imread(str(self._mask_path)))
            if raw.ndim == 2:
                masks = raw[np.newaxis, :, :]
            else:
                masks = raw
        except Exception as exc:
            err = str(exc)
        self.finished.emit(masks, err)


class FilterPreviewFociStacksWorker(QObject):
    """Preload foci channel TIFF stacks off the GUI thread (for faster first overlay)."""

    finished = pyqtSignal(object, str)  # dict[path_key, TYX stack] | None, error message

    def __init__(self, foci_paths: Sequence[Path]) -> None:
        super().__init__()
        self.setObjectName(type(self).__name__)
        self._foci_paths = [Path(p) for p in foci_paths if p is not None]

    def run(self) -> None:
        err = ""
        out: dict[str, np.ndarray] = {}
        try:
            for fp in self._foci_paths:
                if not fp.is_file():
                    continue
                try:
                    key = str(fp.resolve())
                except Exception:
                    key = str(fp)
                raw = np.asarray(safe_imread(str(fp)))
                stack = coerce_stack_to_tyx(raw)
                if stack is None or getattr(stack, "size", 0) == 0:
                    continue
                out[key] = stack
        except Exception as exc:
            err = str(exc)
            out = {}
        self.finished.emit(out if out else None, err)
