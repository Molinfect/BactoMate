"""
Shared GUI widgets and helpers: drag-drop group box, clickable tooltip, info icon, file selection dialog.
"""
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
    QVBoxLayout,
)
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, Qt


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


def show_file_selection_dialog(
    parent,
    files,
    title="Select Files",
    message="Select files to process (all selected by default):",
    initial_filename_filter="",
    return_filename_filter=False,
):
    """
    Show a dialog with checkboxes for file selection.

    initial_filename_filter: if non-empty, pre-fill "Select matching" and apply it so
        files whose name contains this string (case-insensitive) are selected at open.
    return_filename_filter: if True, return (selected_files, filter_string) or (None, None);
        otherwise return selected_files or None (backward compatible).

    Returns:
        List of selected file paths (or (list, filter_str) if return_filename_filter), or None if cancelled.
    """
    if not files:
        return (None, None) if return_filename_filter else None

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(600, 500)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)

    layout = QVBoxLayout()

    label = QLabel(message)
    label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
    layout.addWidget(label)

    button_layout = QHBoxLayout()
    select_all_btn = QPushButton("Select All")
    deselect_all_btn = QPushButton("Deselect All")
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
    button_layout.addWidget(filter_label)
    button_layout.addWidget(filter_edit, 1)
    button_layout.addWidget(select_matching_btn)
    button_layout.addStretch()
    layout.addLayout(button_layout)

    file_list = QListWidget()
    file_items = []
    for f in files:
        item = QListWidgetItem(f.name)
        item.setCheckState(Qt.CheckState.Checked)
        item.setData(Qt.ItemDataRole.UserRole, f)
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

    layout.addWidget(file_list)

    button_box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    layout.addWidget(button_box)

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


class DragDropGroupBox(QGroupBox):
    """QGroupBox with drag-and-drop support for files and folders.
    Optional base_stylesheet: restored after drag leave/drop so custom padding etc. is kept."""
    def __init__(self, title, parent=None, on_drop_callback=None, base_stylesheet=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.on_drop_callback = on_drop_callback
        self._drag_over = False
        self._base_stylesheet = base_stylesheet or ""

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag_over = True
            self.setStyleSheet("QGroupBox { border: 2px dashed #2196F3; }")
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drag_over = False
        self.setStyleSheet(self._base_stylesheet)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.on_drop_callback:
                dropped_path = urls[0].toLocalFile()
                if dropped_path:
                    self.on_drop_callback(dropped_path)
            event.acceptProposedAction()
            self._drag_over = False
            self.setStyleSheet(self._base_stylesheet)
        else:
            event.ignore()
