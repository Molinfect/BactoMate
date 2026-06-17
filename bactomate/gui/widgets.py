"""
Shared GUI widgets and helpers: drag-drop group box, clickable tooltip, info icon, file selection dialog.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

_MASK_PATH_ROLE = Qt.ItemDataRole.UserRole
_COMPANION_PATH_ROLE = Qt.ItemDataRole.UserRole + 1


def tif_files_output_dir(input_path: Union[str, Path]) -> Path:
    """``Tif_files`` folder for a microscopy file or folder input (matches ``convert_nd2_to_tif.py``)."""
    path = Path(input_path).expanduser()
    base = path.parent if path.is_file() else path
    return (base / "Tif_files").resolve()


_BACTOMATE_DIALOG_STYLESHEET = (
    "QDialog { background-color: #1e1e1e; }"
    "QLabel { color: #e8eef6; font-size: 11pt; padding: 2px 0; }"
    "QLabel#dialogTitle { color: #e8eef6; font-size: 12pt; font-weight: 600; padding-bottom: 4px; }"
)

def _dialog_stylesheet() -> str:
    from bactomate.gui.styles import card_aware_control_stylesheet

    return _BACTOMATE_DIALOG_STYLESHEET + card_aware_control_stylesheet()

_TASK_COMPLETE_DIALOG_STYLESHEET = _BACTOMATE_DIALOG_STYLESHEET  # extended at runtime via _dialog_stylesheet()


def show_confirm_dialog(
    parent: Optional[QWidget],
    *,
    title: str,
    message: str,
    confirm_label: str = "Yes",
    cancel_label: str = "No",
    default_confirm: bool = True,
) -> bool:
    """
    Dark-themed Yes/No prompt (replaces native ``QMessageBox.question``).
    """
    from bactomate.gui.layout_helpers import style_compact_action_button

    dlg = QDialog(parent)
    dlg.setWindowTitle(str(title))
    dlg.setStyleSheet(_dialog_stylesheet())
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumWidth(440)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 18, 20, 16)
    layout.setSpacing(12)

    title_lbl = QLabel(str(title))
    title_lbl.setObjectName("dialogTitle")
    title_lbl.setWordWrap(True)
    layout.addWidget(title_lbl)

    body = QLabel(str(message))
    body.setWordWrap(True)
    layout.addWidget(body)

    btn_row = QHBoxLayout()
    btn_row.setSpacing(10)
    btn_row.addStretch(1)

    cancel_btn = QPushButton(str(cancel_label))
    style_compact_action_button(cancel_btn, horizontal_padding=28)
    cancel_btn.clicked.connect(dlg.reject)

    confirm_btn = QPushButton(str(confirm_label))
    style_compact_action_button(confirm_btn, horizontal_padding=28)
    confirm_btn.setDefault(bool(default_confirm))
    confirm_btn.clicked.connect(dlg.accept)

    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(confirm_btn)
    layout.addLayout(btn_row)

    if default_confirm:
        confirm_btn.setFocus()
    else:
        cancel_btn.setFocus()

    return dlg.exec() == int(QDialog.DialogCode.Accepted)


def ask_choice_dialog(
    parent: Optional[QWidget],
    *,
    title: str,
    message: str,
    accept_label: str = "Yes",
    reject_label: str = "No",
    default_accept: bool = True,
) -> bool:
    """Alias for :func:`show_confirm_dialog` with accept/reject naming."""
    return show_confirm_dialog(
        parent,
        title=title,
        message=message,
        confirm_label=accept_label,
        cancel_label=reject_label,
        default_confirm=default_accept,
    )


_MESSAGE_DIALOG_KINDS = frozenset({"info", "warning", "error"})


def show_message_dialog(
    parent: Optional[QWidget],
    title: str,
    message: str,
    kind: str = "info",
) -> None:
    """
    Dark-themed informational popup (replaces native ``QMessageBox`` info/warning/error).
    """
    from bactomate.gui.layout_helpers import style_compact_action_button

    normalized_kind = str(kind or "info").strip().lower()
    if normalized_kind not in _MESSAGE_DIALOG_KINDS:
        normalized_kind = "info"

    dlg = QDialog(parent)
    dlg.setWindowTitle(str(title))
    dlg.setStyleSheet(_dialog_stylesheet())
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumWidth(420)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 18, 20, 16)
    layout.setSpacing(12)

    title_lbl = QLabel(str(title))
    title_lbl.setObjectName("dialogTitle")
    title_lbl.setWordWrap(True)
    layout.addWidget(title_lbl)

    body = QLabel(str(message))
    body.setWordWrap(True)
    if normalized_kind == "warning":
        body.setStyleSheet("color: #f5c842;")
    elif normalized_kind == "error":
        body.setStyleSheet("color: #ff8a80;")
    layout.addWidget(body)

    btn_row = QHBoxLayout()
    btn_row.setSpacing(10)
    btn_row.addStretch(1)

    ok_btn = QPushButton("OK")
    style_compact_action_button(ok_btn, horizontal_padding=28)
    ok_btn.setDefault(True)
    ok_btn.clicked.connect(dlg.accept)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    ok_btn.setFocus()
    dlg.exec()


def style_dialog_button_box(
    button_box: QDialogButtonBox,
    *,
    horizontal_padding: int = 20,
    fixed_height: int = 36,
    min_height: int = 44,
) -> None:
    """Apply BactoMate compact action styling to every button in a ``QDialogButtonBox``."""
    from bactomate.gui.layout_helpers import style_compact_action_button

    for std_btn in button_box.buttons():
        style_compact_action_button(std_btn, horizontal_padding=horizontal_padding)
        std_btn.setFixedHeight(fixed_height)
    button_box.setMinimumHeight(min_height)


def show_task_complete_dialog(
    parent: Optional[QWidget],
    *,
    title: str,
    message: str,
    output_dir: Union[str, Path, None] = None,
    on_ok: Optional[Callable[[], None]] = None,
) -> None:
    """
    Completion popup with optional *Open output folder*.

    Buttons use the same compact File…/Folder… styling as module path rows (not native QMessageBox).
    """
    from bactomate.gui.layout_helpers import style_compact_action_button

    out: Optional[Path] = None
    if output_dir is not None:
        try:
            candidate = Path(output_dir).expanduser()
            if candidate.is_file():
                candidate = candidate.parent
            out = candidate.resolve()
        except Exception:
            try:
                out = Path(output_dir)
                if out.is_file():
                    out = out.parent
            except Exception:
                out = None

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setStyleSheet(_dialog_stylesheet())
    dlg.setMinimumWidth(420)
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 18, 20, 16)
    layout.setSpacing(14)

    label = QLabel(message)
    label.setWordWrap(True)
    layout.addWidget(label)

    open_clicked = False
    ok_clicked = False
    btn_row = QHBoxLayout()
    btn_row.setSpacing(10)
    btn_row.addStretch(1)

    def _open_and_close() -> None:
        nonlocal open_clicked
        open_clicked = True
        dlg.accept()

    def _accept_ok() -> None:
        nonlocal ok_clicked
        ok_clicked = True
        dlg.accept()

    if out is not None and out.is_dir():
        open_btn = QPushButton("Open output folder")
        style_compact_action_button(open_btn, horizontal_padding=32)
        open_btn.clicked.connect(_open_and_close)
        btn_row.addWidget(open_btn)

    ok_btn = QPushButton("OK")
    style_compact_action_button(ok_btn, horizontal_padding=28)
    ok_btn.setDefault(True)
    ok_btn.clicked.connect(_accept_ok)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    dlg.exec()
    if open_clicked and out is not None and out.is_dir():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))
    if ok_clicked and callable(on_ok):
        on_ok()


class ProgressCancelButton(QPushButton):
    """Primary action button that can show an in-button progress fill while running (Cancel mode)."""

    _BASE_BG = QColor("#2d3a47")
    _BASE_BORDER = QColor("#4a6278")
    _FILL_BG = QColor("#4a9eff")  # single accent fill (matches app highlight / hover border)
    _TEXT = QColor("#e8eef6")
    _TEXT_DISABLED = QColor("#6a7a88")
    _RADIUS = 6

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._progress_active = False
        self._progress = 0
        self._progress_subtitle = ""

    def set_progress_active(self, active: bool) -> None:
        active = bool(active)
        if active == self._progress_active:
            return
        self._progress_active = active
        if not active:
            self._progress = 0
            self._progress_subtitle = ""
        self.update()

    def set_progress(self, value: int) -> None:
        value = max(0, min(100, int(value)))
        if value == self._progress:
            return
        self._progress = value
        if self._progress_active:
            self.update()

    def set_progress_subtitle(self, text: str) -> None:
        """Optional suffix on the same line as the main label (e.g. frame progress)."""
        text = str(text or "").strip()
        if text == self._progress_subtitle:
            return
        self._progress_subtitle = text
        if self._progress_active:
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self._progress_active:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -2, -2)
        radius = self._RADIUS

        path = QPainterPath()
        path.addRoundedRect(float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()), radius, radius)

        painter.fillPath(path, self._BASE_BG)
        painter.setPen(QPen(self._BASE_BORDER, 1))
        painter.drawPath(path)

        if self._progress > 0:
            fill_w = max(0, int(rect.width() * self._progress / 100.0))
            if fill_w > 0:
                fill_rect = QRect(rect.x(), rect.y(), fill_w, rect.height())
                fill_path = QPainterPath()
                fill_path.addRoundedRect(
                    float(fill_rect.x()),
                    float(fill_rect.y()),
                    float(fill_rect.width()),
                    float(fill_rect.height()),
                    radius,
                    radius,
                )
                painter.save()
                painter.setClipPath(path)
                painter.fillPath(fill_path, self._FILL_BG)
                painter.restore()

        main = (self.text() or "").strip()
        sub = (self._progress_subtitle or "").strip()
        label = f"{main}  {sub}" if main and sub else (sub or main)
        painter.setPen(self._TEXT if self.isEnabled() else self._TEXT_DISABLED)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)
        painter.end()


class ClickableTooltip(QLabel):
    """A tooltip-like widget that appears on click and disappears on click elsewhere"""
    _current_tooltip = None
    _event_filter = None

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setWindowFlags(
            Qt.WindowType.ToolTip
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setStyleSheet(
            "QLabel {"
            "background-color: #2d2d2d;"
            "color: #ffffff;"
            "padding: 10px;"
            "border: 1px solid #4a9eff;"
            "border-radius: 8px;"
            "font-size: 11pt;"
            "}"
        )
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.hide()

    @classmethod
    def show_tooltip(cls, text, icon_widget, parent_widget=None):
        """Show a tooltip next to the icon widget. Toggle off if already showing."""
        if cls._current_tooltip and cls._current_tooltip.isVisible():
            cls.hide_current()
            return
        if cls._current_tooltip:
            cls.hide_current()

        tooltip = cls(text, parent_widget)
        tooltip.setText(text)
        icon_global_pos = icon_widget.mapToGlobal(QPoint(0, 0))
        tooltip_pos = QPoint(icon_global_pos.x() + icon_widget.width() + 20, icon_global_pos.y())
        tooltip.setMaximumWidth(400)
        tooltip.adjustSize()

        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.geometry()
            if tooltip_pos.x() + tooltip.width() > screen_rect.width():
                tooltip_pos.setX(screen_rect.width() - tooltip.width() - 10)
            if tooltip_pos.y() + tooltip.height() > screen_rect.height():
                tooltip_pos.setY(icon_global_pos.y() - tooltip.height())

        tooltip.move(tooltip_pos)
        tooltip.show()
        cls._current_tooltip = tooltip

        if cls._event_filter:
            QApplication.instance().removeEventFilter(cls._event_filter)
        cls._event_filter = TooltipClickFilter(tooltip)
        QApplication.instance().installEventFilter(cls._event_filter)

    @classmethod
    def hide_current(cls):
        """Hide the currently shown tooltip"""
        if cls._current_tooltip:
            cls._current_tooltip.hide()
            cls._current_tooltip = None
        if cls._event_filter:
            QApplication.instance().removeEventFilter(cls._event_filter)
            cls._event_filter = None


class TooltipClickFilter(QObject):
    """Event filter to hide tooltip when clicking elsewhere or when app loses focus"""

    def __init__(self, tooltip, parent=None):
        super().__init__(parent)
        self.tooltip = tooltip

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.ApplicationDeactivate:
            if self.tooltip and self.tooltip.isVisible():
                ClickableTooltip.hide_current()
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            global_pos = event.globalPosition().toPoint()
            widget_under_mouse = QApplication.widgetAt(global_pos)

            if self.tooltip and self.tooltip.isVisible():
                tooltip_rect = QRect(
                    self.tooltip.mapToGlobal(QPoint(0, 0)),
                    self.tooltip.size(),
                )
                click_in_tooltip = tooltip_rect.contains(global_pos)

                if not click_in_tooltip:
                    click_on_icon = False
                    for widget in QApplication.allWidgets():
                        if isinstance(widget, QLabel) and widget.text() == "(?)":
                            icon_rect = QRect(
                                widget.mapToGlobal(QPoint(0, 0)),
                                widget.size(),
                            )
                            if icon_rect.contains(global_pos):
                                click_on_icon = True
                                break

                    if not click_on_icon:
                        ClickableTooltip.hide_current()
        return False


def create_info_icon(info_text, parent_widget=None, *, compact: bool = False):
    """Create a small (?) info icon label that shows tooltip-style info on click.
    compact=True: smaller control for tight toolbars (e.g. tracking action row)."""
    info_label = QLabel("(?)")
    info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if compact:
        style = (
            "QLabel {"
            "color: #7F8C8D;"
            "font-size: 7pt;"
            "font-weight: 600;"
            "text-decoration: none;"
            "background-color: #2d2d2d;"
            "border: 1px solid #4a4a4a;"
            "border-radius: 4px;"
            "padding: 0px 2px;"
            "min-width: 11px;"
            "min-height: 11px;"
            "max-width: 11px;"
            "max-height: 11px;"
            "}"
            "QLabel:hover {"
            "color: #4a9eff;"
            "background-color: #3a3a3a;"
            "border: 1px solid #4a9eff;"
            "}"
        )
    else:
        style = (
            "QLabel {"
            "color: #7F8C8D;"
            "font-size: 8pt;"
            "font-weight: 600;"
            "text-decoration: none;"
            "background-color: #2d2d2d;"
            "border: 1px solid #4a4a4a;"
            "border-radius: 6px;"
            "padding: 0px 3px;"
            "min-width: 14px;"
            "min-height: 14px;"
            "max-width: 14px;"
            "max-height: 14px;"
            "}"
            "QLabel:hover {"
            "color: #4a9eff;"
            "background-color: #3a3a3a;"
            "border: 1px solid #4a9eff;"
            "}"
        )
    info_label.setStyleSheet(style)
    info_label.setCursor(Qt.CursorShape.PointingHandCursor)

    def show_info(event):
        if ClickableTooltip._current_tooltip and ClickableTooltip._current_tooltip.isVisible():
            ClickableTooltip.hide_current()
        else:
            ClickableTooltip.show_tooltip(info_text, info_label, parent_widget)
        event.accept()

    info_label.mousePressEvent = show_info
    return info_label


# Shared with segmentation / U-Net / filter preview folder pickers.
_LIST_PICK_DIALOG_STYLE = (
    "QListWidget {"
    "background-color: #2b2b2b;"
    "color: #ffffff;"
    "font-size: 11pt;"
    "border: 1px solid #555555;"
    "border-radius: 4px;"
    "padding: 5px;"
    "}"
    "QListWidget::item {"
    "padding: 8px;"
    "border-bottom: 1px solid #3a3a3a;"
    "}"
    "QListWidget::item:selected {"
    "background-color: #5D6D7E;"
    "color: #ffffff;"
    "}"
    "QListWidget::item:hover {"
    "background-color: #3a3a3a;"
    "}"
)


def show_list_pick_dialog(
    parent,
    files,
    *,
    title: str = "Select file",
    message: str = "Choose a file:",
    default_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Single-selection file list (same UX as Omnipose / U-Net preview folder pickers).

    Returns the chosen path, or None if cancelled. When only one file is given,
    returns it without opening a dialog.
    """
    paths = sorted({Path(f) for f in files}, key=lambda p: p.name.lower())
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    from bactomate.gui.layout_helpers import style_compact_action_button

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(500, 400)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)

    layout = QVBoxLayout()
    layout.setContentsMargins(14, 14, 14, 18)
    layout.setSpacing(10)

    label = QLabel(message)
    label.setWordWrap(True)
    label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
    layout.addWidget(label)

    file_list = QListWidget()
    file_list.setStyleSheet(_LIST_PICK_DIALOG_STYLE)

    by_name: dict[str, Path] = {}
    default_row = 0
    default_resolved: Optional[Path] = None
    if default_path is not None:
        try:
            default_resolved = Path(default_path).resolve()
        except Exception:
            default_resolved = Path(default_path)

    for row, path in enumerate(paths):
        file_list.addItem(path.name)
        by_name[path.name] = path
        if default_resolved is not None:
            try:
                if path.resolve() == default_resolved:
                    default_row = row
            except Exception:
                if path == default_resolved:
                    default_row = row

    if file_list.count() > 0:
        file_list.setCurrentRow(default_row)

    file_list.itemDoubleClicked.connect(lambda _item: dialog.accept())
    layout.addWidget(file_list, 1)

    button_box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    style_dialog_button_box(button_box)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)

    footer = QWidget()
    footer_layout = QVBoxLayout(footer)
    footer_layout.setContentsMargins(0, 6, 0, 2)
    footer_layout.setSpacing(0)
    footer_layout.addWidget(button_box)
    layout.addWidget(footer, 0)

    dialog.setLayout(layout)
    dialog.raise_()
    dialog.activateWindow()

    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    item = file_list.currentItem()
    if item is None:
        return None
    return by_name.get(item.text())


def show_file_selection_dialog(
    parent,
    files,
    title="Select Files",
    message="Select files to process (all selected by default):",
    initial_filename_filter="",
    return_filename_filter=False,
    default_checked: Optional[Callable[[Path], bool]] = None,
):
    """
    Show a dialog with checkboxes for file selection.

    initial_filename_filter: if non-empty, pre-fill "Select matching" and apply it so
        files whose name contains this string (case-insensitive) are selected at open.
    default_checked: optional ``path -> bool``; when provided, sets each row's initial
        checkbox from this predicate (after list build, before initial_filename_filter).
    return_filename_filter: if True, return (selected_files, filter_string) or (None, None);
        otherwise return selected_files or None (backward compatible).

    Returns:
        List of selected file paths (or (list, filter_str) if return_filename_filter), or None if cancelled.
    """
    if not files:
        return (None, None) if return_filename_filter else None

    from bactomate.gui.layout_helpers import style_compact_action_button

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(600, 520)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)

    layout = QVBoxLayout()
    layout.setContentsMargins(14, 14, 14, 18)
    layout.setSpacing(10)

    label = QLabel(message)
    label.setWordWrap(True)
    label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
    layout.addWidget(label)

    button_layout = QHBoxLayout()
    select_all_btn = QPushButton("Select All")
    deselect_all_btn = QPushButton("Deselect All")
    style_compact_action_button(select_all_btn, horizontal_padding=20)
    style_compact_action_button(deselect_all_btn, horizontal_padding=20)
    button_layout.addWidget(select_all_btn)
    button_layout.addWidget(deselect_all_btn)
    # "Select matching" row: line edit + button
    filter_label = QLabel("Select TIF files containing:")
    filter_label.setStyleSheet("color: #cccccc;")
    filter_edit = QLineEdit()
    filter_edit.setPlaceholderText("e.g. phase")
    filter_edit.setClearButtonEnabled(True)
    if initial_filename_filter:
        filter_edit.setText(initial_filename_filter.strip())
    select_matching_btn = QPushButton("Select matching")
    select_matching_btn.setToolTip("Select all files whose name contains the text above (case-insensitive).")
    style_compact_action_button(select_matching_btn, horizontal_padding=20)
    button_layout.addWidget(filter_label)
    button_layout.addWidget(filter_edit, 1)
    button_layout.addWidget(select_matching_btn)
    button_layout.addStretch()
    layout.addLayout(button_layout)

    file_list = QListWidget()
    file_items = []
    for f in files:
        path = Path(f)
        item = QListWidgetItem(path.name)
        if default_checked is None:
            checked = True
        else:
            try:
                checked = bool(default_checked(path))
            except Exception:
                checked = True
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setData(Qt.ItemDataRole.UserRole, path)
        file_list.addItem(item)
        file_items.append(item)

    def select_all():
        for item in file_items:
            item.setCheckState(Qt.CheckState.Checked)

    def deselect_all():
        for item in file_items:
            item.setCheckState(Qt.CheckState.Unchecked)

    def select_matching():
        text = (filter_edit.text() or "").strip().lower()
        if not text:
            select_all()
            return
        for item in file_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            name = (path.name if hasattr(path, "name") else str(path)).lower()
            item.setCheckState(Qt.CheckState.Checked if text in name else Qt.CheckState.Unchecked)

    select_all_btn.clicked.connect(select_all)
    deselect_all_btn.clicked.connect(deselect_all)
    select_matching_btn.clicked.connect(select_matching)

    # Apply initial filter when dialog opens so re-run starts with matching files selected
    if initial_filename_filter and (initial_filename_filter.strip().lower()):
        select_matching()

    file_list.setMinimumHeight(120)
    layout.addWidget(file_list, 1)

    button_box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    style_dialog_button_box(button_box)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)

    footer = QWidget()
    footer_layout = QVBoxLayout(footer)
    footer_layout.setContentsMargins(0, 6, 0, 2)
    footer_layout.setSpacing(0)
    footer_layout.addWidget(button_box)
    layout.addWidget(footer, 0)

    dialog.setLayout(layout)
    dialog.raise_()
    dialog.activateWindow()

    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected_files = []
        for i in range(file_list.count()):
            item = file_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_files.append(item.data(Qt.ItemDataRole.UserRole))
        filter_used = (filter_edit.text() or "").strip() if return_filename_filter else ""
        if return_filename_filter:
            return (selected_files if selected_files else None, filter_used)
        return selected_files if selected_files else None
    if return_filename_filter:
        return (None, None)
    return None


def show_linked_mask_selection_dialog(
    parent,
    mask_files: list[Path],
    *,
    companion_for_mask: Callable[[Path], dict[str, Any]],
    title: str = "Select Files",
    message: str = "Select mask files to process:",
    right_header: str = "Linked artifacts (follow mask selection)",
    show_fi: bool = True,
    initial_filename_filter: str = "",
    return_filename_filter: bool = False,
    default_checked: Optional[Callable[[Path], bool]] = None,
) -> Union[list[Path], tuple[Optional[list[Path]], str], tuple[None, None], None]:
    """
  Show a two-column file picker: masks (left) and linked foci CSV / FI paths (right).

  ``companion_for_mask`` returns ``{"foci": [(csv_path, kind_label), ...], "fi": [path, ...]}``.
  Unchecking a mask unchecks and disables its linked rows on the right.
  Falls back to :func:`show_file_selection_dialog` when no companions exist for any mask.
    """
    masks = [Path(m) for m in mask_files]
    if not masks:
        return (None, None) if return_filename_filter else None

    has_companions = False
    for mp in masks:
        try:
            comp = companion_for_mask(mp) or {}
        except Exception:
            comp = {}
        foci = comp.get("foci") or []
        fi = comp.get("fi") or [] if show_fi else []
        if foci or fi:
            has_companions = True
            break
    if not has_companions:
        return show_file_selection_dialog(
            parent,
            masks,
            title=title,
            message=message,
            initial_filename_filter=initial_filename_filter,
            return_filename_filter=return_filename_filter,
            default_checked=default_checked,
        )

    from bactomate.gui.layout_helpers import style_compact_action_button

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(920, 540)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)

    layout = QVBoxLayout()
    layout.setContentsMargins(14, 14, 14, 18)
    layout.setSpacing(10)

    label = QLabel(message)
    label.setWordWrap(True)
    label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
    layout.addWidget(label)

    button_layout = QHBoxLayout()
    select_all_btn = QPushButton("Select All")
    deselect_all_btn = QPushButton("Deselect All")
    style_compact_action_button(select_all_btn, horizontal_padding=20)
    style_compact_action_button(deselect_all_btn, horizontal_padding=20)
    button_layout.addWidget(select_all_btn)
    button_layout.addWidget(deselect_all_btn)
    filter_label = QLabel("Select masks containing:")
    filter_label.setStyleSheet("color: #cccccc;")
    filter_edit = QLineEdit()
    filter_edit.setPlaceholderText("e.g. phase")
    filter_edit.setClearButtonEnabled(True)
    if initial_filename_filter:
        filter_edit.setText(initial_filename_filter.strip())
    select_matching_btn = QPushButton("Select matching")
    select_matching_btn.setToolTip("Select masks whose name contains the text above (case-insensitive).")
    style_compact_action_button(select_matching_btn, horizontal_padding=20)
    button_layout.addWidget(filter_label)
    button_layout.addWidget(filter_edit, 1)
    button_layout.addWidget(select_matching_btn)
    button_layout.addStretch()
    layout.addLayout(button_layout)

    splitter = QSplitter(Qt.Orientation.Horizontal)

    mask_list = QListWidget()
    mask_items: list[QListWidgetItem] = []
    for mp in masks:
        item = QListWidgetItem(mp.name)
        if default_checked is None:
            checked = True
        else:
            try:
                checked = bool(default_checked(mp))
            except Exception:
                checked = True
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setData(_MASK_PATH_ROLE, mp)
        mask_list.addItem(item)
        mask_items.append(item)

    companion_list = QListWidget()
    companion_rows: list[QListWidgetItem] = []

    def _add_header(mask_path: Path) -> None:
        header = QListWidgetItem(f"— {mask_path.name} —")
        header.setFlags(Qt.ItemFlag.ItemIsEnabled)
        header.setForeground(QColor("#9e9e9e"))
        header.setData(_MASK_PATH_ROLE, mask_path)
        companion_list.addItem(header)
        companion_rows.append(header)

    def _add_companion(mask_path: Path, text: str, companion_path: Path) -> None:
        row = QListWidgetItem(text)
        row.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
        )
        row.setCheckState(Qt.CheckState.Checked)
        row.setForeground(QColor("#e8eef6"))
        row.setData(_MASK_PATH_ROLE, mask_path)
        row.setData(_COMPANION_PATH_ROLE, companion_path)
        companion_list.addItem(row)
        companion_rows.append(row)

    for mp in masks:
        try:
            comp = companion_for_mask(mp) or {}
        except Exception:
            comp = {}
        foci_rows = list(comp.get("foci") or [])
        fi_rows = list(comp.get("fi") or []) if show_fi else []
        if not foci_rows and not fi_rows:
            continue
        _add_header(mp)
        for csv_path, kind in foci_rows:
            kind_lbl = str(kind or "csv").replace("_", " ")
            _add_companion(mp, f"  CSV ({kind_lbl}): {Path(csv_path).name}", Path(csv_path))
        for fi_entry in fi_rows:
            if isinstance(fi_entry, tuple):
                fi_path, fi_kind = fi_entry[0], str(fi_entry[1] or "fi")
            else:
                fi_path, fi_kind = fi_entry, "fi"
            kind_lbl = fi_kind.replace("_", " ")
            _add_companion(mp, f"  FI ({kind_lbl}): {Path(fi_path).name}", Path(fi_path))

    def _set_companion_row_active(row: QListWidgetItem, active: bool) -> None:
        flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
        if active:
            flags |= Qt.ItemFlag.ItemIsEnabled
            row.setForeground(QColor("#e8eef6"))
        else:
            row.setForeground(QColor("#6a7380"))
        row.setFlags(flags)

    def _sync_companions_for_mask(mask_path: Path, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        companion_list.blockSignals(True)
        for row in companion_rows:
            if row.data(_COMPANION_PATH_ROLE) is None:
                continue
            if Path(row.data(_MASK_PATH_ROLE)) != Path(mask_path):
                continue
            row.setCheckState(state)
            _set_companion_row_active(row, checked)
        companion_list.blockSignals(False)

    def _on_mask_changed(item: QListWidgetItem) -> None:
        if item is None:
            return
        mask_path = item.data(_MASK_PATH_ROLE)
        if mask_path is None:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        _sync_companions_for_mask(Path(mask_path), checked)

    mask_list.itemChanged.connect(_on_mask_changed)

    for item in mask_items:
        _sync_companions_for_mask(
            Path(item.data(_MASK_PATH_ROLE)),
            item.checkState() == Qt.CheckState.Checked,
        )

    left_col = QWidget()
    left_lay = QVBoxLayout(left_col)
    left_lay.setContentsMargins(0, 0, 4, 0)
    left_lay.setSpacing(4)
    left_lbl = QLabel("Mask TIFFs")
    left_lbl.setStyleSheet("color: #c8d0dc; font-weight: 600;")
    left_lay.addWidget(left_lbl)
    left_lay.addWidget(mask_list, 1)

    right_col = QWidget()
    right_lay = QVBoxLayout(right_col)
    right_lay.setContentsMargins(4, 0, 0, 0)
    right_lay.setSpacing(4)
    right_lbl = QLabel(right_header)
    right_lbl.setStyleSheet("color: #c8d0dc; font-weight: 600;")
    right_lbl.setWordWrap(True)
    right_lay.addWidget(right_lbl)
    right_lay.addWidget(companion_list, 1)

    splitter.addWidget(left_col)
    splitter.addWidget(right_col)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([420, 460])
    layout.addWidget(splitter, 1)

    def select_all() -> None:
        mask_list.blockSignals(True)
        for item in mask_items:
            item.setCheckState(Qt.CheckState.Checked)
        mask_list.blockSignals(False)
        for item in mask_items:
            _sync_companions_for_mask(Path(item.data(_MASK_PATH_ROLE)), True)

    def deselect_all() -> None:
        mask_list.blockSignals(True)
        for item in mask_items:
            item.setCheckState(Qt.CheckState.Unchecked)
        mask_list.blockSignals(False)
        for item in mask_items:
            _sync_companions_for_mask(Path(item.data(_MASK_PATH_ROLE)), False)

    def select_matching() -> None:
        text = (filter_edit.text() or "").strip().lower()
        mask_list.blockSignals(True)
        if not text:
            for item in mask_items:
                item.setCheckState(Qt.CheckState.Checked)
        else:
            for item in mask_items:
                name = Path(item.data(_MASK_PATH_ROLE)).name.lower()
                item.setCheckState(
                    Qt.CheckState.Checked if text in name else Qt.CheckState.Unchecked
                )
        mask_list.blockSignals(False)
        for item in mask_items:
            _sync_companions_for_mask(
                Path(item.data(_MASK_PATH_ROLE)),
                item.checkState() == Qt.CheckState.Checked,
            )

    select_all_btn.clicked.connect(select_all)
    deselect_all_btn.clicked.connect(deselect_all)
    select_matching_btn.clicked.connect(select_matching)
    if initial_filename_filter and initial_filename_filter.strip().lower():
        select_matching()

    button_box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    for std_btn in (
        button_box.button(QDialogButtonBox.StandardButton.Ok),
        button_box.button(QDialogButtonBox.StandardButton.Cancel),
    ):
        if std_btn is not None:
            style_compact_action_button(std_btn, horizontal_padding=20)
            std_btn.setFixedHeight(36)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    button_box.setMinimumHeight(44)

    footer = QWidget()
    footer_layout = QVBoxLayout(footer)
    footer_layout.setContentsMargins(0, 6, 0, 2)
    footer_layout.setSpacing(0)
    footer_layout.addWidget(button_box)
    layout.addWidget(footer, 0)

    dialog.setLayout(layout)
    dialog.raise_()
    dialog.activateWindow()

    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected_files: list[Path] = []
        for item in mask_items:
            if item.checkState() == Qt.CheckState.Checked:
                selected_files.append(Path(item.data(_MASK_PATH_ROLE)))
        filter_used = (filter_edit.text() or "").strip() if return_filename_filter else ""
        if return_filename_filter:
            return (selected_files if selected_files else None, filter_used)
        return selected_files if selected_files else None
    if return_filename_filter:
        return (None, None)
    return None


class _FileDropMixin:
    """Shared drag-and-drop handling for path pickers."""

    on_drop_callback = None
    _drag_over = False
    _base_stylesheet = ""

    def _accept_file_drop(self, event) -> bool:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def _drag_highlight(self) -> None:
        self._drag_over = True
        self.setStyleSheet("QWidget { border: 2px dashed #2196F3; border-radius: 6px; }")

    def _drag_restore(self) -> None:
        self._drag_over = False
        self.setStyleSheet(self._base_stylesheet)

    def _handle_drop(self, event) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        urls = event.mimeData().urls()
        if urls and self.on_drop_callback:
            dropped_path = urls[0].toLocalFile()
            if dropped_path:
                self.on_drop_callback(dropped_path)
        event.acceptProposedAction()
        self._drag_restore()


class PathDropWidget(QWidget, _FileDropMixin):
    """Flat drag-and-drop path row (no nested QGroupBox border)."""

    def __init__(self, parent=None, on_drop_callback=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.on_drop_callback = on_drop_callback

    def dragEnterEvent(self, event):
        if self._accept_file_drop(event):
            self._drag_highlight()

    def dragMoveEvent(self, event):
        self._accept_file_drop(event)

    def dragLeaveEvent(self, event):
        self._drag_restore()

    def dropEvent(self, event):
        self._handle_drop(event)


class DragDropGroupBox(QGroupBox, _FileDropMixin):
    """QGroupBox with drag-and-drop support for files and folders.
    Optional base_stylesheet: restored after drag leave/drop so custom padding etc. is kept."""
    def __init__(self, title, parent=None, on_drop_callback=None, base_stylesheet=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.on_drop_callback = on_drop_callback
        self._drag_over = False
        self._base_stylesheet = base_stylesheet or ""

    def dragEnterEvent(self, event):
        if self._accept_file_drop(event):
            self._drag_over = True
            self.setStyleSheet("QGroupBox { border: 2px dashed #2196F3; }")

    def dragMoveEvent(self, event):
        self._accept_file_drop(event)

    def dragLeaveEvent(self, event):
        self._drag_restore()

    def dropEvent(self, event):
        self._handle_drop(event)
