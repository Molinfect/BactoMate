"""
Module shell: toolbar + horizontal splitter with pipeline controls (left) and preview surface (right).

The left pane holds the existing module tab (scrollable, adaptive width, shared typography + light combos).
The right pane lists files at the current path and hosts embedded previews below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PyQt6.QtCore import QByteArray, Qt, QTimer

from bactomate.gui.styles import MODULE_PANEL_COMBO_STYLESHEET, get_module_shell_left_pane_stylesheet
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Match main header / logo strip (see ``MainWindow`` title_bar and ``styles.py`` title_bar).
BRAND_ACCENT = "#4a9eff"


def detach_mount_widget(widget: Optional[QWidget]) -> None:
    """Remove a widget from the shell mount area.

    Preview dialogs rely on ``PreviewDialogBase.closeEvent`` (WorkerGuard) to stop
    background ``QThread`` workers. Calling ``deleteLater()`` alone skips that path
    and can trigger Qt fatal: QThread destroyed while thread is still running.
    """
    if widget is None:
        return
    try:
        widget.close()
    except RuntimeError:
        pass
    try:
        widget.deleteLater()
    except RuntimeError:
        pass


_NAV_BTN_ACTIVE = (
    "QPushButton {"
    " font-weight: 700;"
    " color: #ECF0F1;"
    " background: transparent;"
    " border: none;"
    f" border-bottom: 2px solid {BRAND_ACCENT};"
    " padding: 6px 10px 4px 10px;"
    "}"
    "QPushButton:hover { color: #ffffff; }"
)
# Module splitter: left = collapsible controls (~40%), right = preview (~60%). Left never scrolls horizontally.
_SPLITTER_LEFT_RATIO = 0.40
_MAX_LEFT_RATIO = 0.40
_LEFT_PANE_MIN_W = 260
_RIGHT_PANE_MIN_W = 400

_NAV_BTN_INACTIVE = (
    "QPushButton {"
    " font-weight: 500;"
    " color: #95a5a6;"
    " background: transparent;"
    " border: none;"
    " border-bottom: 2px solid transparent;"
    " padding: 6px 10px 4px 10px;"
    "}"
    "QPushButton:hover { color: #d0e8ff; border-bottom-color: #3d5a80; }"
)


class _ConstrainedPathComboBox(QComboBox):
    """Path picker combo whose popup follows the available pane width."""

    def showPopup(self) -> None:
        # Qt otherwise sizes the popup to the longest item, which can extend off-screen
        # for long analysis / mask filenames on small displays.
        try:
            popup_w = max(160, int(self.width()))
            view = self.view()
            view.setMinimumWidth(popup_w)
            view.setMaximumWidth(popup_w)
            view.setTextElideMode(Qt.TextElideMode.ElideMiddle)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        except Exception:
            pass
        super().showPopup()


class PreviewSurface(QWidget):
    """Right-hand column: convertible file list + embedded preview mount."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PreviewSurface")
        self.setMinimumWidth(260)
        self._pick_callback: Optional[Callable[[str], None]] = None
        self._preview_context = "file_conversion"
        self._empty_folder_msg = (
            "No ND2 / LIF / CZI / TIFF files found under this folder (recursive scan, within limits). "
            "Browse another folder or pick a file on the left."
        )
        self._idle_placeholder_msg = (
            "Pick a file from the dropdown above, or set a single file on the left. "
            "The interactive preview (LUT, time/channel) appears here."
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(4)

        self._file_combo = _ConstrainedPathComboBox()
        self._file_combo.setMinimumHeight(30)
        self._file_combo.setMinimumWidth(0)
        self._file_combo.setMinimumContentsLength(0)
        self._file_combo.setMaxVisibleItems(18)
        self._file_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._file_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        try:
            self._file_combo.view().setTextElideMode(Qt.TextElideMode.ElideMiddle)
            self._file_combo.view().setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        except Exception:
            pass
        self._file_combo.setStyleSheet(MODULE_PANEL_COMBO_STYLESHEET + " QComboBox { font-size: 12pt; }")
        self._file_combo.currentIndexChanged.connect(self._on_file_combo_changed)
        layout.addWidget(self._file_combo)
        self._preview_chrome_visible = True
        self._candidate_list_nonempty = False
        self._file_picker_enabled = True

        self._toolbar_host = QWidget()
        self._toolbar_layout = QHBoxLayout(self._toolbar_host)
        self._toolbar_layout.setContentsMargins(0, 8, 0, 10)
        self._toolbar_layout.setSpacing(8)
        self._toolbar_host.setVisible(False)
        layout.addWidget(self._toolbar_host)

        self._mount_area = QWidget()
        self._mount_area.setObjectName("PreviewMountArea")
        self._mount_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._mount_area.setMinimumHeight(120)
        self._mount_area.setStyleSheet("#PreviewMountArea { background-color: #1a1a1a; border-radius: 6px; }")
        ma = QVBoxLayout(self._mount_area)
        ma.setContentsMargins(4, 4, 4, 4)
        self._placeholder = QLabel(self._idle_placeholder_msg)
        self._placeholder.setWordWrap(True)
        self._placeholder.setTextFormat(Qt.TextFormat.RichText)
        self._placeholder.setStyleSheet("color: #9a9a9a; font-size: 11pt;")
        self._placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        ma.addWidget(self._placeholder)
        layout.addWidget(self._mount_area, 1)

    def mount_widget(self) -> QWidget:
        """Widget passed as ``embed_parent`` to embedded preview builders."""
        return self._mount_area

    def set_pick_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._pick_callback = cb

    def current_candidate_path(self) -> Optional[str]:
        """Resolved path for the shell file combo, or None if the placeholder row is selected."""
        data = self._file_combo.currentData()
        if data is None:
            return None
        text = str(data).strip()
        return text or None

    def set_preview_context(self, context: str) -> None:
        """Switch placeholder copy for file conversion vs segmentation (TIFF-only)."""
        self._preview_context = str(context)
        if self._preview_context == "segmentation":
            self._empty_folder_msg = (
                "No .tif / .tiff files found under this folder (recursive scan, within limits). "
                "Browse another folder or pick a file on the left."
            )
            self._idle_placeholder_msg = (
                "Use the TIFF dropdown above to browse inputs in the folder (no preview required), "
                "or set a path on the left. Segmentation preview opens below when you choose Preview."
            )
        elif self._preview_context == "filtering":
            self._empty_folder_msg = (
                "No mask .tif / .tiff files found under this folder. "
                "Set a mask folder on the left or run segmentation first."
            )
            self._idle_placeholder_msg = (
                "Choose a mask from the dropdown above, then click Preview filters on the left."
            )
        elif self._preview_context == "tracking":
            self._empty_folder_msg = (
                "No mask .tif / .tiff files found under this folder. "
                "Set a mask or filtered-mask path on the left, or run segmentation / filtering first."
            )
            self._idle_placeholder_msg = (
                "Select a mask from the dropdown, then run Preview tracking "
                "(quick or full time lapse in the setup dialog). "
                "Trackastra trajectory overlays appear here frame by frame."
            )
        elif self._preview_context == "post_processing_scalebar":
            self._empty_folder_msg = (
                "No image file found for the current scale-bar path. "
                "Set a TIFF/PNG/JPEG image on the left."
            )
            self._idle_placeholder_msg = (
                "Set an image path on the left, then click Preview scale bar. "
                "The interactive scale-bar preview appears here."
            )
        elif self._preview_context == "post_processing_video":
            self._empty_folder_msg = (
                "No video input file found for the current Video montage path. "
                "Set a TIFF file, folder, or channel stack list on the left."
            )
            self._idle_placeholder_msg = (
                "Set video input and analysis workbook paths on the left, then click Preview video montage. "
                "Tune settings here and use Create video montage when ready."
            )
        elif self._preview_context == "post_processing":
            self._empty_folder_msg = (
                "No montage-compatible image files found for the current Image montage path. "
                "Set a file or folder on the left."
            )
            self._idle_placeholder_msg = (
                "Set an Image montage path on the left, then click Preview image montage. "
                "The interactive montage preview appears here."
            )
        elif self._preview_context == "training":
            self._empty_folder_msg = ""
            self._idle_placeholder_msg = (
                "Training is configured on the left. This preview panel is not used in the Training module."
            )
        else:
            self._empty_folder_msg = (
                "No ND2 / LIF / CZI / TIFF files found under this folder (recursive scan, within limits). "
                "Browse another folder or pick a file on the left."
            )
            self._idle_placeholder_msg = (
                "Pick a file from the dropdown above, or set a single file on the left. "
                "The interactive preview (LUT, time/channel) appears here."
            )
        self._placeholder.setText(self._idle_placeholder_msg)

    def set_toolbar_widget(self, widget: Optional[QWidget]) -> None:
        """Optional row between the file combo and the preview mount (e.g. Preview segmentation)."""
        while self._toolbar_layout.count():
            item = self._toolbar_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        if widget is None:
            self._toolbar_host.setVisible(False)
            return
        self._toolbar_layout.addWidget(widget)
        self._toolbar_host.setVisible(True)

    def set_preview_chrome_visible(self, visible: bool) -> None:
        """Hide the TIFF picker row while an embedded preview fills the mount (avoids overlap/clutter)."""
        self._preview_chrome_visible = bool(visible)
        self._sync_preview_chrome_widgets()

    def set_file_picker_enabled(self, enabled: bool) -> None:
        """Hide the candidate-file dropdown when a module has no preview picker (e.g. Training)."""
        self._file_picker_enabled = bool(enabled)
        self._sync_preview_chrome_widgets()

    def _sync_preview_chrome_widgets(self) -> None:
        show_picker = (
            bool(self._preview_chrome_visible)
            and bool(self._candidate_list_nonempty)
            and bool(self._file_picker_enabled)
        )
        try:
            self._file_combo.setVisible(show_picker)
        except Exception:
            pass
        try:
            has_toolbar = bool(self._toolbar_layout.count())
            self._toolbar_host.setVisible(bool(self._preview_chrome_visible) and has_toolbar)
        except Exception:
            pass

    def set_candidate_files(
        self,
        paths: list[str],
        *,
        selected: Optional[str] = None,
        advance_placeholder: bool = True,
    ) -> None:
        """Populate the TIFF dropdown. When ``advance_placeholder`` is True (default), select the
        first listed file if none matches ``selected``. When False, keep the placeholder row until
        the user picks (browse-without-preview mode).
        """
        self._file_combo.blockSignals(True)
        try:
            self._file_combo.clear()
            placeholder = (
                "— Select mask —"
                if self._preview_context == "filtering"
                else "— Select file to preview —"
            )
            self._file_combo.addItem(placeholder, None)
            for p in paths:
                pp = Path(p)
                rp = str(pp.resolve())
                self._file_combo.addItem(pp.name, rp)
                idx = self._file_combo.count() - 1
                self._file_combo.setItemData(idx, rp, Qt.ItemDataRole.ToolTipRole)
            self._candidate_list_nonempty = bool(paths)
            self._sync_preview_chrome_widgets()
            if not paths:
                self._placeholder.setText(self._empty_folder_msg)
            else:
                self._placeholder.setText(self._idle_placeholder_msg)
                if selected:
                    self.set_selected_candidate(selected)
                if advance_placeholder and self._file_combo.currentData() is None:
                    for i in range(1, self._file_combo.count()):
                        if self._file_combo.itemData(i) is not None:
                            self._file_combo.setCurrentIndex(i)
                            break
        finally:
            self._file_combo.blockSignals(False)

    def set_selected_candidate(self, full_path: Optional[str]) -> None:
        """Select the row matching this path without invoking the pick callback."""
        if not full_path:
            return
        try:
            target = str(Path(full_path).resolve())
        except OSError:
            target = str(Path(full_path))
        self._file_combo.blockSignals(True)
        try:
            for i in range(self._file_combo.count()):
                data = self._file_combo.itemData(i)
                if data is None:
                    continue
                try:
                    if str(Path(str(data)).resolve()) == target:
                        self._file_combo.setCurrentIndex(i)
                        return
                except OSError:
                    if str(data) == target:
                        self._file_combo.setCurrentIndex(i)
                        return
        finally:
            self._file_combo.blockSignals(False)

    def _on_file_combo_changed(self, _idx: int) -> None:
        if self._file_combo.signalsBlocked() or self._pick_callback is None:
            return
        path = self._file_combo.currentData()
        if not path:
            return
        try:
            self._pick_callback(str(path))
        except Exception:
            pass

    def clear_mount(self) -> None:
        lay = self._mount_area.layout()
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is None:
                continue
            # Never destroy the idle placeholder; mirror ``mount_embedded_preview`` (detach only).
            if w is self._placeholder:
                try:
                    w.hide()
                    w.setParent(None)
                except RuntimeError:
                    pass
                continue
            detach_mount_widget(w)
        lay.addWidget(self._placeholder)
        self._placeholder.show()
        self._placeholder.setText(self._idle_placeholder_msg)
        self.set_preview_chrome_visible(True)


def mount_embedded_preview(
    mount: QWidget,
    preview_widget: QWidget,
    *,
    preview_surface: Optional[PreviewSurface] = None,
) -> None:
    """Place an embedded preview widget in the module shell right-hand mount area."""
    lay = mount.layout()
    if lay is None:
        lay = QVBoxLayout(mount)
        lay.setContentsMargins(0, 0, 0, 0)
        mount.setLayout(lay)
    placeholder = getattr(preview_surface, "_placeholder", None) if preview_surface is not None else None
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is None:
            continue
        if placeholder is not None and w is placeholder:
            w.hide()
            w.setParent(None)
            continue
        detach_mount_widget(w)
    lay.addWidget(preview_widget, 1)
    if preview_surface is not None:
        preview_surface.set_preview_chrome_visible(False)


class ModuleShellWidget(QWidget):
    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        module_id: str,
        content: QWidget,
        on_back: Optional[Callable[[], None]] = None,
        pipeline_navigation: Optional[Sequence[tuple[str, str]]] = None,
        on_open_module: Optional[Callable[[str], None]] = None,
        show_preview_picker: bool = True,
        show_preview_panel: bool = True,
    ) -> None:
        super().__init__(parent)
        self._module_id = str(module_id)
        self._nav_by_module: dict[str, QPushButton] = {}
        self._show_preview_panel = bool(show_preview_panel)
        self._splitter: Optional[QSplitter] = None
        self._preview = PreviewSurface(self)
        if not show_preview_picker:
            self._preview.set_preview_context("training")
            self._preview.set_file_picker_enabled(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QWidget()
        bar.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #252525, stop:1 #1a1a1a);"
            f"border-bottom: 1px solid {BRAND_ACCENT};"
        )
        bar_l = QHBoxLayout(bar)
        bar_l.setContentsMargins(10, 8, 12, 8)
        bar_l.setSpacing(12)

        self._back_btn = QPushButton("  ◀  Dashboard  ")
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.setToolTip("Open the main dashboard")
        self._back_btn.setStyleSheet(
            "QPushButton {"
            " background-color: #2d3d52; color: #e8f4ff; font-size: 11pt; font-weight: 600;"
            f" border: 1px solid {BRAND_ACCENT}; border-radius: 18px; padding: 6px 16px;"
            "}"
            f"QPushButton:hover {{ background-color: {BRAND_ACCENT}; color: #0d1117; border-color: #6bb3ff; }}"
            "QPushButton:pressed { background-color: #3d8ae6; }"
        )
        if on_back:
            self._back_btn.clicked.connect(on_back)
        bar_l.addWidget(self._back_btn)

        if pipeline_navigation and on_open_module:
            bar_l.addSpacing(8)
            for mid, label in pipeline_navigation:
                nav_btn = QPushButton(label)
                nav_btn.setFlat(True)
                nav_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                nav_btn.setToolTip(f"Open {label}")
                nav_btn.setStyleSheet(_NAV_BTN_INACTIVE)
                fnt = nav_btn.font()
                fnt.setPointSize(12)
                nav_btn.setFont(fnt)

                def _go(_checked: bool, m: str = mid) -> None:
                    try:
                        on_open_module(m)
                    except Exception:
                        pass

                nav_btn.clicked.connect(_go)
                self._nav_by_module[str(mid)] = nav_btn
                bar_l.addWidget(nav_btn)
        else:
            # Fallback: single title if navigation not wired (e.g. tests)
            title = QLabel(module_id.replace("_", " ").title())
            title.setStyleSheet("font-weight: bold; color: #ECF0F1; font-size: 14pt; letter-spacing: 0.5px;")
            bar_l.addWidget(title)

        bar_l.addStretch(1)
        outer.addWidget(bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content.setMinimumWidth(0)
        content.setStyleSheet(get_module_shell_left_pane_stylesheet())
        scroll.setWidget(content)
        scroll.setMinimumWidth(0)
        self._left_scroll = scroll

        if not self._show_preview_panel:
            outer.addWidget(scroll, 1)
            return

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(scroll)
        self._splitter.addWidget(self._preview)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 6)
        # Fixed 40/60 split: no drag grip (ratio is enforced on resize, not user-draggable).
        self._splitter.setHandleWidth(0)
        self._splitter.setStyleSheet(
            "QSplitter::handle {"
            " background: transparent;"
            " width: 0px;"
            " margin: 0;"
            " padding: 0;"
            " border: none;"
            " image: none;"
            "}"
        )
        try:
            handle = self._splitter.handle(1)
            if handle is not None:
                handle.setEnabled(False)
                handle.setVisible(False)
        except Exception:
            pass
        outer.addWidget(self._splitter, 1)
        QTimer.singleShot(0, self._apply_splitter_ratio)

    def _splitter_total_width(self) -> int:
        if self._splitter is None:
            return 0
        w = int(self._splitter.width())
        if w > 0:
            return w
        sizes = self._splitter.sizes()
        total = sum(sizes)
        return total if total > 0 else _LEFT_PANE_MIN_W + _RIGHT_PANE_MIN_W

    def _apply_splitter_ratio(self) -> None:
        """Keep ~40% controls / ~60% preview; left pane never wider than 40%."""
        if self._splitter is None:
            return
        total = self._splitter_total_width()
        if total < _LEFT_PANE_MIN_W + _RIGHT_PANE_MIN_W:
            return
        cap = int(round(total * _MAX_LEFT_RATIO))
        left = max(_LEFT_PANE_MIN_W, min(int(round(total * _SPLITTER_LEFT_RATIO)), cap, total - _RIGHT_PANE_MIN_W))
        right = max(_RIGHT_PANE_MIN_W, total - left)
        if left + right != total:
            left = total - right
        sizes = [left, right]
        if sizes != self._splitter.sizes():
            self._splitter.setSizes(sizes)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_splitter_ratio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_splitter_ratio)

    def preview_surface(self) -> PreviewSurface:
        return self._preview

    def set_workspace_line(self, text: str) -> None:
        """Workspace path is shown only in the Back button tooltip (not in the bar)."""
        tip = (text or "").strip()
        if tip:
            self._back_btn.setToolTip(f"Open the main dashboard\n\nWorkspace:\n{tip}")
        else:
            self._back_btn.setToolTip("Open the main dashboard")

    def restore_splitter_state(self, data: Optional[QByteArray]) -> None:
        if self._splitter is None:
            return
        if data is not None and not data.isEmpty():
            self._splitter.restoreState(data)
        QTimer.singleShot(0, self._apply_splitter_ratio)

    def splitter_state(self) -> bytes:
        if self._splitter is None:
            return b""
        return bytes(self._splitter.saveState())

    def set_active_pipeline_module(self, active_module_id: str) -> None:
        """Underline the active module in the top bar (same accent as the main BactoMate header)."""
        if not self._nav_by_module:
            return
        for mid, btn in self._nav_by_module.items():
            btn.setStyleSheet(_NAV_BTN_ACTIVE if mid == active_module_id else _NAV_BTN_INACTIVE)
