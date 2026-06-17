"""
Efficient TIFF time-series loading for tracking and analysis workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

# Above this page count, try memmap before a full RAM load.
LARGE_STACK_PAGE_THRESHOLD = 64

LogFn = Callable[[str], None] | None


def count_tiff_pages(path: str | Path) -> int | None:
    """Return the number of time frames in a TIFF stack."""
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tif:
            n_pages = len(tif.pages)
            if n_pages == 0:
                return None
            if n_pages == 1 and tif.series:
                shape = getattr(tif.series[0], "shape", None)
                if shape is not None and len(shape) >= 3:
                    return int(shape[0])
            return n_pages
    except Exception:
        return None


def read_tiff_stack(
    path: str | Path,
    *,
    label: str = "stack",
    frame_limit: int = 0,
    preview_mode: bool = False,
    log: LogFn = None,
) -> tuple[np.ndarray, int | None, bool]:
    """
    Load a (T, Y, X) stack from *path*.

    Returns ``(array, total_frame_count_or_none, was_partial_read)``.
  Preview mode reads only the first *frame_limit* pages. Large stacks use memmap when possible.
    """
    import tifffile

    p = str(path)
    limit = int(frame_limit or 0)

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    if preview_mode and limit > 0:
        try:
            with tifffile.TiffFile(p) as tif:
                n_pages = len(tif.pages)
            if n_pages > 0:
                n_use = min(limit, n_pages)
                arr = tifffile.imread(p, key=range(n_use))
                if getattr(arr, "ndim", 0) == 2:
                    arr = arr[np.newaxis, ...]
                _log(f"✓ {label}: preview-read first {arr.shape[0]} of {n_pages} frame(s)")
                return arr, n_pages, True
        except Exception as exc:
            _log(f"  {label}: partial preview read failed ({exc}); falling back to full read")

    n_pages: int | None = None
    try:
        with tifffile.TiffFile(p) as tif:
            n_pages = len(tif.pages)
        if n_pages and n_pages > LARGE_STACK_PAGE_THRESHOLD:
            try:
                arr = tifffile.memmap(p)
                if getattr(arr, "ndim", 0) == 2:
                    arr = arr[np.newaxis, ...]
                _log(f"✓ {label}: memmap {int(arr.shape[0])} frame(s) (of {n_pages})")
                return arr, n_pages, False
            except Exception as exc:
                _log(f"  {label}: memmap failed ({exc}); using full read")
    except Exception:
        pass

    arr = tifffile.imread(p)
    if getattr(arr, "ndim", 0) == 2:
        arr = arr[np.newaxis, ...]
    n_total = int(n_pages if n_pages is not None else arr.shape[0])
    _log(f"✓ {label}: loaded {int(arr.shape[0])} frame(s)")
    return arr, n_total, False
