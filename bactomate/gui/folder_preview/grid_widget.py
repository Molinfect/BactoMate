"""Scrollable thumbnail grid for folder-wide preview QC."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.layout_helpers import style_module_primary_button


class _TileWidget(QFrame):
    activated = pyqtSignal(str)

    def __init__(self, path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.setObjectName("FolderPreviewTile")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "#FolderPreviewTile { background: #121212; border: 1px solid #333; border-radius: 4px; }"
            "#FolderPreviewTile:hover { border-color: #4a9eff; }"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self._title = QLabel(self.path.name)
        self._title.setWordWrap(True)
        self._title.setStyleSheet("color: #d0d0d0; font-size: 10pt;")
        self._title.setToolTip(str(self.path))
        lay.addWidget(self._title)
        self._image = QLabel("…")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(120, 120)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image.setStyleSheet("background: #0a0a0a; color: #777; border-radius: 2px;")
        self._image.setScaledContents(False)
        lay.addWidget(self._image, 1)
        self._status = QLabel("Queued")
        self._status.setStyleSheet("color: #888; font-size: 9pt;")
        lay.addWidget(self._status)
        self._pixmap: Optional[QPixmap] = None
        self._tile_scale = 1.0

    def set_pending(self) -> None:
        self._status.setText("Queued")
        self._image.setText("…")
        self._pixmap = None

    def set_busy(self) -> None:
        self._status.setText("Computing…")

    def set_result(self, pixmap: Optional[QPixmap], error: str = "") -> None:
        if error:
            self._status.setText(error[:120])
            self._status.setStyleSheet("color: #e07070; font-size: 9pt;")
            self._image.setText("Error")
            self._pixmap = None
            return
        self._status.setStyleSheet("color: #888; font-size: 9pt;")
        self._status.setText("Ready — double-click to enlarge")
        self._pixmap = pixmap
        self._apply_pixmap()

    def set_tile_scale(self, scale: float) -> None:
        self._tile_scale = max(0.4, min(3.0, float(scale)))
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        base_w = max(140, int(220 * self._tile_scale))
        scaled = self._pixmap.scaled(
            base_w,
            base_w,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image.setPixmap(scaled)
        self._image.setText("")
        self._image.setMinimumSize(scaled.width(), scaled.height())

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.activated.emit(str(self.path))
        super().mouseDoubleClickEvent(event)


class FolderPreviewGrid(QWidget):
    """Grid surface with progress chrome, stale banner, cancel/refresh, and zoomable detail."""

    cancel_requested = pyqtSignal()
    refresh_requested = pyqtSignal()
    tile_activated = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("FolderPreviewGrid")
        self._tiles: dict[str, _TileWidget] = {}
        self._paths: list[Path] = []
        self._tile_scale = 1.0
        self._worker_running = False
        self._result_pixmaps: dict[str, QPixmap] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        self._stack = QStackedWidget(self)
        root.addWidget(self._stack, 1)

        overview = QWidget()
        overview_lay = QVBoxLayout(overview)
        overview_lay.setContentsMargins(0, 0, 0, 0)
        overview_lay.setSpacing(6)

        chrome = QHBoxLayout()
        self._status = QLabel("Folder preview")
        self._status.setStyleSheet("color: #c0c0c0; font-size: 11pt;")
        chrome.addWidget(self._status, 1)
        self._stale = QLabel("Settings changed — results may be stale")
        self._stale.setStyleSheet("color: #e0a060; font-size: 10pt; font-weight: 600;")
        self._stale.setVisible(False)
        chrome.addWidget(self._stale)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)
        self._refresh_btn.setEnabled(False)
        chrome.addWidget(self._refresh_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._cancel_btn.setEnabled(False)
        chrome.addWidget(self._cancel_btn)
        overview_lay.addLayout(chrome)

        self._hint = QLabel(
            "Double-click a tile to enlarge (zoom/pan). "
            "Scroll to browse; Ctrl+wheel zooms tile size. "
            "Foci are shown as compact QC dots (not full preview rings). "
            "Results are cached until settings change."
        )
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #888; font-size: 9pt;")
        overview_lay.addWidget(self._hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")
        self._host = QWidget()
        self._grid = QGridLayout(self._host)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(8)
        self._scroll.setWidget(self._host)
        overview_lay.addWidget(self._scroll, 1)
        self._stack.addWidget(overview)

        detail = QWidget()
        detail_lay = QVBoxLayout(detail)
        detail_lay.setContentsMargins(0, 0, 0, 0)
        detail_lay.setSpacing(8)
        detail_chrome = QHBoxLayout()
        self._back_btn = QPushButton("← Back to overview")
        style_module_primary_button(self._back_btn)
        self._back_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._back_btn.setMinimumWidth(160)
        self._back_btn.clicked.connect(self.show_overview)
        detail_chrome.addWidget(self._back_btn, 0)
        self._detail_title = QLabel("")
        self._detail_title.setStyleSheet("color: #dce4ee; font-size: 11pt; font-weight: 600;")
        self._detail_title.setWordWrap(True)
        detail_chrome.addWidget(self._detail_title, 1)
        detail_lay.addLayout(detail_chrome)
        detail_hint = QLabel(
            "Wheel zooms · left-drag pans · cached QC tile "
            "(compact foci dots — open single-file Preview for full rings)."
        )
        detail_hint.setStyleSheet("color: #888; font-size: 9pt;")
        detail_lay.addWidget(detail_hint)
        self._detail_canvas = ImageCanvas(detail)
        self._detail_canvas.pan_button = Qt.MouseButton.LeftButton
        self._detail_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag.")
        self._detail_canvas.setMinimumSize(200, 200)
        self._detail_canvas.setStyleSheet("background-color: #1e1e1e;")
        detail_lay.addWidget(self._detail_canvas, 1)
        self._stack.addWidget(detail)

    def show_overview(self) -> None:
        self._stack.setCurrentIndex(0)

    def show_tile_detail(self, path: str | Path, pixmap: QPixmap) -> None:
        if pixmap is None or pixmap.isNull():
            return
        p = Path(path)
        self._detail_title.setText(p.name)
        self._detail_title.setToolTip(str(p))
        self._detail_canvas.set_pixmap(pixmap, (pixmap.width(), pixmap.height()), reset_view=True)
        self._stack.setCurrentIndex(1)

    def is_detail_visible(self) -> bool:
        return self._stack.currentIndex() == 1

    def cached_pixmap(self, path: str | Path) -> Optional[QPixmap]:
        return self._result_pixmaps.get(str(path))

    def set_paths(self, paths: list[Path], *, showing_of_total: tuple[int, int] | None = None) -> None:
        self.show_overview()
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._tiles.clear()
        self._result_pixmaps.clear()
        self._paths = [Path(p) for p in paths]
        cols = 3
        for i, path in enumerate(self._paths):
            tile = _TileWidget(path)
            tile.activated.connect(self.tile_activated.emit)
            tile.set_tile_scale(self._tile_scale)
            self._tiles[str(path)] = tile
            self._grid.addWidget(tile, i // cols, i % cols)
        if showing_of_total is not None:
            shown, total = showing_of_total
            if total > shown:
                self._status.setText(f"Showing {shown} of {total} files")
            else:
                self._status.setText(f"{shown} file(s)")
        else:
            self._status.setText(f"{len(self._paths)} file(s)")
        self.set_stale(False)

    def set_progress(self, done: int, total: int, current_name: str = "") -> None:
        name = f" — {current_name}" if current_name else ""
        self._status.setText(f"Progress {done}/{total}{name}")

    def set_busy(self, running: bool) -> None:
        self._worker_running = bool(running)
        self._cancel_btn.setEnabled(bool(running))
        self._refresh_btn.setEnabled(not running)

    def set_stale(self, stale: bool) -> None:
        self._stale.setVisible(bool(stale))
        if stale and not self._worker_running:
            self._refresh_btn.setEnabled(True)

    def mark_tile_busy(self, path: str | Path) -> None:
        tile = self._tiles.get(str(path))
        if tile is not None:
            tile.set_busy()

    def set_tile_result(self, path: str | Path, pixmap: Optional[QPixmap], error: str = "") -> None:
        key = str(path)
        tile = self._tiles.get(key)
        if pixmap is not None and not pixmap.isNull() and not error:
            self._result_pixmaps[key] = pixmap
        elif error:
            self._result_pixmaps.pop(key, None)
        if tile is not None:
            tile.set_result(pixmap, error=error)

    def apply_cached_results(self, results: dict[str, QPixmap]) -> None:
        """Populate tiles from an existing cache without recomputing."""
        for path in self._paths:
            key = str(path)
            pix = results.get(key)
            if pix is not None and not pix.isNull():
                self.set_tile_result(key, pix)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self.is_detail_visible():
            super().wheelEvent(event)
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = 0.1 if delta > 0 else -0.1
            self._tile_scale = max(0.4, min(3.0, self._tile_scale + step))
            for tile in self._tiles.values():
                tile.set_tile_scale(self._tile_scale)
            event.accept()
            return
        super().wheelEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.cancel_requested.emit()
        super().closeEvent(event)
