"""Small layout helpers for narrow module left panes (no horizontal scroll)."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.styles import (
    COMPACT_BROWSE_BUTTON_STYLESHEET,
    MODULE_PATH_ACTION_PANEL_STYLESHEET,
    PATH_ROW_BROWSE_BUTTON_STYLESHEET,
    PATH_ROW_LINE_EDIT_STYLESHEET,
)


def hbox_row_widget(row: QHBoxLayout, parent: Optional[QWidget] = None) -> QWidget:
    """Wrap a horizontal layout in a QWidget for use inside responsive panels."""
    w = QWidget(parent)
    w.setLayout(row)
    return w


def _qt_widget_alive(widget: Optional[QWidget]) -> bool:
    """Return False if the underlying C++ QObject was deleted."""
    if widget is None:
        return False
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False


class ResponsivePanel(QWidget):
    """
    Place child widgets in one horizontal row when wide; stack vertically when narrow.

    Reflows on resize so controls use horizontal space first (responsive module panes).
    """

    def __init__(self, parent: Optional[QWidget] = None, *, breakpoint: int = 520) -> None:
        super().__init__(parent)
        self._breakpoint = int(breakpoint)
        self._items: list[QWidget] = []
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(6)
        self._defer_reflow = False

    def add_widget(self, widget: QWidget) -> None:
        self._items.append(widget)
        widget.setParent(self)
        if not self._defer_reflow:
            self._reflow()

    def set_defer_reflow(self, defer: bool) -> None:
        """When True, batch add_widget calls and finish with reflow()."""
        self._defer_reflow = bool(defer)

    def reflow(self) -> None:
        self._reflow()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not _qt_widget_alive(self):
            return
        self._reflow()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not _qt_widget_alive(self):
            return
        self._reflow()

    def _reflow(self) -> None:
        if not _qt_widget_alive(self):
            return
        self._items = [w for w in self._items if _qt_widget_alive(w)]
        if not self._items:
            return
        # Reparent managed widgets before tearing down old layout containers.
        for w in self._items:
            if not _qt_widget_alive(w):
                continue
            w.setParent(self)
        while self._outer.count():
            item = self._outer.takeAt(0)
            container = item.widget()
            if container is not None:
                container.deleteLater()
        width = self.width()
        if width <= 0:
            width = self._breakpoint + 1
        if width >= self._breakpoint:
            row = QWidget(self)
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)
            for w in self._items:
                if not _qt_widget_alive(w):
                    continue
                w.show()
                lay.addWidget(w, 0)
            lay.addStretch(1)
            self._outer.addWidget(row)
        else:
            for w in self._items:
                if not _qt_widget_alive(w):
                    continue
                w.show()
                self._outer.addWidget(w)
        self.setMinimumHeight(max(self.sizeHint().height(), 38))
        self.updateGeometry()


def style_compact_browse_button(btn: QPushButton, *, row_height: int | None = None) -> None:
    """Apply shared compact styling and short labels for file/folder pickers."""
    text = (btn.text() or "").strip().lower()
    if "browse file" in text or text in ("file", "file…", "file..."):
        btn.setText("File…")
    elif "browse folder" in text or text in ("folder", "folder…", "folder..."):
        btn.setText("Folder…")
    if row_height is not None:
        _style_path_row_browse_button(btn, int(row_height))
        return
    style_compact_action_button(btn)


def _style_path_row_line_edit(edit: QLineEdit, height: int) -> None:
    edit.setStyleSheet(PATH_ROW_LINE_EDIT_STYLESHEET)
    edit.setFixedHeight(int(height))
    edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def _style_path_row_browse_button(btn: QPushButton, height: int) -> None:
    btn.setStyleSheet(PATH_ROW_BROWSE_BUTTON_STYLESHEET)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    btn.setFixedHeight(int(height))
    fm = btn.fontMetrics()
    text_w = int(fm.horizontalAdvance(btn.text() or ""))
    btn.setFixedWidth(max(72, text_w + 24))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)


def style_path_row_browse_button(btn: QPushButton) -> None:
    """Compact File…/Folder…-style button for module path strips and FI file picker."""
    _style_path_row_browse_button(btn, _PATH_ROW_CONTENT_HEIGHT)


def style_compact_action_button(btn: QPushButton, *, horizontal_padding: int = 24) -> None:
    """Compact secondary button (path row File…/Folder… style) without renaming label text."""
    btn.setStyleSheet(COMPACT_BROWSE_BUTTON_STYLESHEET)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    # 32px is tight with 1px border + padding on some styles — bottom edge can clip in dense rows.
    btn.setFixedHeight(34)
    fm = btn.fontMetrics()
    text_w = int(fm.horizontalAdvance(btn.text() or ""))
    btn.setFixedWidth(max(72, text_w + int(horizontal_padding)))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)


# Tall enough for border + padding under module-shell QLineEdit rules; +4px bottom inset in the row.
_PATH_ROW_CONTENT_HEIGHT = 34
_PATH_ROW_BOTTOM_MARGIN = 4


def stacked_path_block(
    *,
    path_label: str = "Path:",
    line_edit: QLineEdit,
    buttons: Sequence[QPushButton] = (),
    info_icon: Optional[QWidget] = None,
    parent: Optional[QWidget] = None,
    show_path_label: bool = True,
) -> QWidget:
    """One row: Path + info + text field + File…/Folder… (same as File Conversion)."""
    wrap = QWidget(parent)
    wrap.setMinimumWidth(0)
    wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    wrap.setFixedHeight(_PATH_ROW_CONTENT_HEIGHT + _PATH_ROW_BOTTOM_MARGIN)
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, _PATH_ROW_BOTTOM_MARGIN)
    lay.setSpacing(4)
    lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
    if show_path_label:
        lbl = QLabel(path_label)
        lbl.setFixedHeight(_PATH_ROW_CONTENT_HEIGHT)
        lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lay.addWidget(lbl, 0)
    if info_icon is not None:
        info_icon.setFixedSize(18, 18)
        info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lay.addWidget(info_icon, 0, Qt.AlignmentFlag.AlignVCenter)
    line_edit.setMinimumWidth(0)
    _style_path_row_line_edit(line_edit, _PATH_ROW_CONTENT_HEIGHT)
    lay.addWidget(line_edit, 1)
    lay.setStretchFactor(line_edit, 1)
    for btn in buttons:
        style_compact_browse_button(btn, row_height=_PATH_ROW_CONTENT_HEIGHT)
        lay.addWidget(btn, 0)
    return wrap


def new_module_path_action_panel(parent: QWidget) -> QWidget:
    """Bordered path + action-button strip (drag-and-drop only on the path row)."""
    from PyQt6.QtWidgets import QGroupBox

    panel = QGroupBox("", parent)
    panel.setFlat(True)
    panel.setObjectName("ModulePathActionPanel")
    panel.setStyleSheet(MODULE_PATH_ACTION_PANEL_STYLESHEET)
    panel.setMinimumWidth(0)
    panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    return panel


def build_path_drop_group(
    parent: QWidget,
    on_drop_callback,
    *,
    line_edit: QLineEdit,
    buttons: Sequence[QPushButton] = (),
    info_icon: Optional[QWidget] = None,
    show_path_label: bool = True,
    path_label: str = "Path:",
    nested_in_action_panel: bool = False,
) -> QWidget:
    """File Conversion–style path picker: one bordered group, dashed outline only while dragging."""
    from bactomate.gui.styles import PATH_DROP_GROUPBOX_NESTED_STYLESHEET, PATH_DROP_GROUPBOX_STYLESHEET
    from bactomate.gui.widgets import DragDropGroupBox

    sheet = PATH_DROP_GROUPBOX_NESTED_STYLESHEET if nested_in_action_panel else PATH_DROP_GROUPBOX_STYLESHEET
    file_group = DragDropGroupBox(
        "",
        parent=parent,
        on_drop_callback=on_drop_callback,
        base_stylesheet=sheet,
    )
    file_group.setFlat(True)
    file_group.setStyleSheet(sheet)
    file_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    outer = QVBoxLayout(file_group)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    line_edit.setAcceptDrops(False)
    outer.addWidget(
        stacked_path_block(
            line_edit=line_edit,
            buttons=buttons,
            info_icon=info_icon,
            parent=file_group,
            show_path_label=show_path_label,
            path_label=path_label,
        )
    )
    return file_group


def module_path_drop_row(
    parent: QWidget,
    on_drop_callback,
    *,
    line_edit: QLineEdit,
    buttons: Sequence[QPushButton] = (),
    info_icon: Optional[QWidget] = None,
    show_path_label: bool = True,
    path_label: str = "Path:",
) -> QWidget:
    """Path drop strip inside a module path/action card (same look as File Conversion)."""
    return build_path_drop_group(
        parent,
        on_drop_callback,
        line_edit=line_edit,
        buttons=buttons,
        info_icon=info_icon,
        show_path_label=show_path_label,
        path_label=path_label,
        nested_in_action_panel=True,
    )


def module_path_action_outer_layout(panel: QWidget) -> QVBoxLayout:
    """VBox inside the path/action panel (full width; height from Maximum policy)."""
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    return lay


def module_path_action_buttons_layout() -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    return lay


def style_module_primary_button(btn: QPushButton) -> None:
    """Full-width slim primary action (Process / Filter / Preview) in module panes."""
    btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    btn.setMinimumHeight(32)
    btn.setMaximumHeight(34)
    try:
        from bactomate.gui.styles import MODULE_ACTION_BUTTON_STYLESHEET

        btn.setStyleSheet(MODULE_ACTION_BUTTON_STYLESHEET)
    except Exception:
        pass


def style_module_action_button(btn: QPushButton) -> None:
    """Alias for primary module action buttons (full-width slim bar)."""
    style_module_primary_button(btn)


def wire_spin_activates_checkbox(
    spin: QSpinBox | QDoubleSpinBox,
    checkbox: QCheckBox,
    *,
    on_changed: Optional[Callable[[], None]] = None,
) -> None:
    """When the user edits a spinbox, auto-check the paired enable checkbox."""
    def _on_spin_changed(*_args: object) -> None:
        if not checkbox.isChecked():
            checkbox.blockSignals(True)
            try:
                checkbox.setChecked(True)
            finally:
                checkbox.blockSignals(False)
        if on_changed is not None:
            on_changed()

    spin.valueChanged.connect(_on_spin_changed)


def vbox_from_rows(*rows: QHBoxLayout, spacing: int = 6, parent: Optional[QWidget] = None) -> QWidget:
    """Stack horizontal rows vertically (replaces one wide HBox)."""
    wrap = QWidget(parent)
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for row in rows:
        lay.addLayout(row)
    return wrap
