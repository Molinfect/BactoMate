"""Mount and drive a FolderPreviewGrid inside PreviewSurface."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget

from bactomate.gui.folder_preview.constants import FOLDER_PREVIEW_MAX_FILES, cap_folder_preview_paths
from bactomate.gui.folder_preview.fingerprint import settings_fingerprint
from bactomate.gui.folder_preview.grid_widget import FolderPreviewGrid
from bactomate.gui.folder_preview.workers import FolderPreviewQueueWorker, TileComputeFn, rgb_to_qpixmap
from bactomate.gui.module_shell import PreviewSurface, mount_embedded_preview
from bactomate.gui.preview.preview_window import WorkerGuard


class FolderPreviewController(QObject):
    """Owns grid widget + queue worker lifecycle for one module tab."""

    def __init__(
        self,
        owner: QWidget,
        *,
        preview_surface: Optional[PreviewSurface],
        mount: Optional[QWidget],
        on_tile_open: Optional[Callable[[Path], None]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._owner = owner
        self._preview_surface = preview_surface
        self._mount = mount
        # Optional escape hatch; default double-click opens zoomable detail of the cached tile.
        self._on_tile_open = on_tile_open
        self._grid: Optional[FolderPreviewGrid] = None
        self._worker: Optional[FolderPreviewQueueWorker] = None
        self._fingerprint = ""
        self._params_snapshot: dict[str, Any] = {}
        # path_str -> RGB uint8 (and derived pixmap) for the current fingerprint
        self._rgb_cache: dict[str, np.ndarray] = {}
        self._pixmap_cache: dict[str, QPixmap] = {}

    @property
    def grid(self) -> Optional[FolderPreviewGrid]:
        return self._grid

    @property
    def is_running(self) -> bool:
        w = self._worker
        return w is not None and w.isRunning()

    def current_fingerprint(self) -> str:
        return self._fingerprint

    def mark_stale_if_params_changed(self, params: dict[str, Any]) -> None:
        if self._grid is None:
            return
        fp = settings_fingerprint(params)
        if fp and fp != self._fingerprint:
            self._grid.set_stale(True)

    def close(self) -> None:
        self.cancel()
        grid = self._grid
        self._grid = None
        if grid is not None:
            try:
                grid.close()
            except Exception:
                pass
            try:
                grid.deleteLater()
            except Exception:
                pass
        if self._preview_surface is not None and self._mount is not None:
            try:
                self._preview_surface.clear_mount()
            except Exception:
                pass

    def cancel(self) -> None:
        w = self._worker
        if w is None:
            return
        try:
            w.request_cancel()
        except Exception:
            pass
        try:
            w.wait(3000)
        except Exception:
            pass
        self._worker = None
        if self._grid is not None:
            self._grid.set_busy(False)

    def clear_result_cache(self) -> None:
        self._rgb_cache.clear()
        self._pixmap_cache.clear()

    def start(
        self,
        paths: Sequence[Path],
        params: dict[str, Any],
        compute_fn: TileComputeFn,
        *,
        max_files: int = FOLDER_PREVIEW_MAX_FILES,
        force_recompute: bool = False,
    ) -> FolderPreviewGrid:
        self.cancel()
        old_grid = self._grid
        self._grid = None
        if old_grid is not None:
            try:
                old_grid.deleteLater()
            except Exception:
                pass
        capped, total = cap_folder_preview_paths(paths, max_files=max_files)
        if not capped:
            raise ValueError("No files to preview in folder")

        self._params_snapshot = dict(params)
        self._params_snapshot["_paths"] = [Path(p) for p in capped]
        fp = settings_fingerprint({k: v for k, v in params.items() if k != "_paths"})
        if force_recompute or (fp and fp != self._fingerprint):
            self.clear_result_cache()
        self._fingerprint = fp

        grid = FolderPreviewGrid(self._owner)
        grid.set_paths(capped, showing_of_total=(len(capped), total))
        grid.cancel_requested.connect(self.cancel)
        grid.refresh_requested.connect(lambda: self._restart(compute_fn, max_files=max_files, force_recompute=True))
        grid.tile_activated.connect(self._open_tile)

        mount = self._mount
        surf = self._preview_surface
        if mount is not None:
            mount_embedded_preview(mount, grid, preview_surface=surf)

        self._grid = grid

        # Serve fully-cached runs without re-running segmentation / Spotiflow.
        if not force_recompute and all(str(p) in self._pixmap_cache for p in capped):
            grid.apply_cached_results(self._pixmap_cache)
            grid.set_busy(False)
            n = len(capped)
            grid.set_progress(n, n, "cached")
            return grid

        def cached_compute(path: Path, idx: int) -> tuple[Optional[np.ndarray], str]:
            key = str(path)
            cached = self._rgb_cache.get(key)
            if cached is not None and not force_recompute:
                return np.asarray(cached), ""
            rgb, err = compute_fn(path, idx)
            if rgb is not None and not err:
                arr = np.ascontiguousarray(rgb)
                self._rgb_cache[key] = arr
                try:
                    self._pixmap_cache[key] = rgb_to_qpixmap(arr)
                except Exception:
                    pass
            return rgb, err

        worker = FolderPreviewQueueWorker(capped, cached_compute, parent=self)
        worker.progress.connect(self._on_progress)
        worker.tile_ready.connect(self._on_tile_ready)
        worker.finished_ok.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        grid.set_busy(True)
        if capped:
            grid.mark_tile_busy(capped[0])
        worker.start()

        guard = WorkerGuard(
            is_running=lambda: self.is_running,
            wait=lambda ms: (self._worker.wait(ms) if self._worker is not None else True),
            request_stop=self.cancel,
            disconnect=lambda: None,
            on_close_while_running="defer_hide",
            warning_title="Folder preview in progress",
            warning_text="Please wait for folder preview to finish or click Cancel.",
        )
        grid.setProperty("bactomate_worker_guard", guard)
        return grid

    def _restart(self, compute_fn: TileComputeFn, *, max_files: int, force_recompute: bool = True) -> None:
        paths = list(self._params_snapshot.get("_paths") or [])
        if self._grid is not None and getattr(self._grid, "_paths", None):
            paths = list(self._grid._paths)
        if not paths:
            return
        params = dict(self._params_snapshot)
        params.pop("_paths", None)
        self.start(paths, params, compute_fn, max_files=max_files, force_recompute=force_recompute)

    def _open_tile(self, path_str: str) -> None:
        grid = self._grid
        pix = self._pixmap_cache.get(path_str)
        if pix is None and grid is not None:
            pix = grid.cached_pixmap(path_str)
        if grid is not None and pix is not None and not pix.isNull():
            grid.show_tile_detail(path_str, pix)
            return
        if self._on_tile_open is not None:
            try:
                self._on_tile_open(Path(path_str))
            except Exception:
                pass

    def _on_progress(self, done: int, total: int, name: str) -> None:
        if self._grid is None:
            return
        self._grid.set_progress(done, total, name)
        if 0 <= done < total and done < len(getattr(self._grid, "_paths", [])):
            try:
                nxt = self._grid._paths[done]
                self._grid.mark_tile_busy(nxt)
            except Exception:
                pass

    def _on_tile_ready(self, index: int, path_str: str, pixmap: object, error: str) -> None:
        if self._grid is None:
            return
        pix = pixmap if isinstance(pixmap, QPixmap) else None
        if pix is not None and not pix.isNull() and not error:
            self._pixmap_cache[path_str] = pix
        self._grid.set_tile_result(path_str, pix, error=error or "")

    def _on_finished(self) -> None:
        self._worker = None
        if self._grid is not None:
            self._grid.set_busy(False)
            n = len(getattr(self._grid, "_paths", []) or [])
            self._grid.set_progress(n, n, "done")

    def _on_failed(self, message: str) -> None:
        self._worker = None
        if self._grid is not None:
            self._grid.set_busy(False)
            self._grid._status.setText(f"Failed: {message}")
