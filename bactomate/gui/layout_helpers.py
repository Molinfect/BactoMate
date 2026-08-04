"""Small layout helpers for narrow module left panes (no horizontal scroll)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.styles import (
    COMPACT_BROWSE_BUTTON_STYLESHEET,
    MODULE_PANEL_COMBO_STYLESHEET,
    MODULE_PATH_ACTION_PANEL_STYLESHEET,
    PATH_ROW_BROWSE_BUTTON_STYLESHEET,
    PATH_ROW_LINE_EDIT_STYLESHEET,
    PREVIEW_NAV_COMBO_STYLESHEET,
    RESPONSIVE_COMPACT_BREAKPOINT,
    RESPONSIVE_MEDIUM_BREAKPOINT,
    RESPONSIVE_WIDE_BREAKPOINT,
    preview_settings_card_stylesheet,
)

_COMBO_POPUP_LISTVIEW_STYLESHEET = (
    "QListView {"
    " background-color: #1f1f1f;"
    " color: #ffffff;"
    " border: 1px solid #4a9eff;"
    " outline: 0;"
    " padding: 2px;"
    "}"
    "QListView::item {"
    " padding: 6px 12px;"
    " min-height: 24px;"
    "}"
    "QListView::item:selected {"
    " background-color: #4a9eff;"
    " color: #ffffff;"
    "}"
    "QListView::item:hover {"
    " background-color: #2a3540;"
    "}"
)


def _clamp_combo_popup_height(combo: QComboBox) -> None:
    """Keep dropdown lists compact when combos live inside scrollable preview panels."""
    view = combo.view()
    if view is None:
        return
    container = view.parentWidget()
    if container is None:
        return
    count = max(0, int(combo.count()))
    if count <= 0:
        return
    visible = min(count, max(4, int(combo.maxVisibleItems())))
    row_h = 28
    if count > 0:
        try:
            row_h = max(28, int(view.sizeHintForRow(0)))
        except Exception:
            row_h = 28
    frame = max(0, int(view.frameWidth()) * 2)
    height = row_h * visible + frame + 4
    container.setFixedHeight(height)
    try:
        view.scrollTo(combo.model().index(combo.currentIndex(), 0), QAbstractItemView.ScrollHint.PositionAtCenter)
    except Exception:
        pass


def ensure_dark_combo_popup(combo: QComboBox) -> None:
    """Use a styled QListView popup instead of the native Windows dropdown list."""
    view = combo.view()
    if view is None or view.objectName() != "bactomateDarkComboPopup":
        view = QListView()
        view.setObjectName("bactomateDarkComboPopup")
        combo.setView(view)
    view.setFrameShape(QFrame.Shape.NoFrame)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    view.setUniformItemSizes(True)
    view.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)
    view.setStyleSheet(_COMBO_POPUP_LISTVIEW_STYLESHEET)
    combo.setMaxVisibleItems(12)
    existing = (combo.styleSheet() or "").strip()
    if "combobox-popup" not in existing:
        combo.setStyleSheet((existing + " QComboBox { combobox-popup: 0; }").strip())
    if not getattr(combo, "_bactomate_popup_hooked", False):
        original_show_popup = combo.showPopup

        def _show_popup() -> None:
            original_show_popup()
            _clamp_combo_popup_height(combo)

        combo.showPopup = _show_popup  # type: ignore[method-assign]
        combo._bactomate_popup_hooked = True


def style_module_combo(combo: QComboBox) -> None:
    """Module-pane combo: dark field + dark dropdown popup (Windows-safe)."""
    combo.setStyleSheet(MODULE_PANEL_COMBO_STYLESHEET + " QComboBox { combobox-popup: 0; }")
    ensure_dark_combo_popup(combo)


def patch_all_combo_popups(root: QWidget) -> None:
    """Apply dark popup styling to every combo under *root*."""
    for combo in root.findChildren(QComboBox):
        ensure_dark_combo_popup(combo)


# Re-export breakpoints for tests and module code.
COMPACT_BREAKPOINT = RESPONSIVE_COMPACT_BREAKPOINT
MEDIUM_BREAKPOINT = RESPONSIVE_MEDIUM_BREAKPOINT
WIDE_BREAKPOINT = RESPONSIVE_WIDE_BREAKPOINT

PREVIEW_SETTINGS_SCROLL_MAX_HEIGHT = 340
PREVIEW_SETTINGS_SCROLL_MIN_HEIGHT = 72
PREVIEW_SETTINGS_SCROLL_EMBEDDED_MAX_HEIGHT = 280


def columns_for_width(
    width: int,
    *,
    min_cell_width: int = 148,
    compact_bp: int = COMPACT_BREAKPOINT,
    medium_bp: int = MEDIUM_BREAKPOINT,
    wide_bp: int = WIDE_BREAKPOINT,
    max_columns: int = 4,
) -> int:
    """Return a bounded column count for adaptive form grids."""
    w = max(0, int(width))
    cap = max(1, int(max_columns))
    if w < compact_bp:
        return 1
    max_fit = max(1, w // max(80, int(min_cell_width)))
    if w >= wide_bp:
        return min(cap, max_fit)
    if w >= medium_bp:
        return min(min(3, cap), max_fit)
    return min(min(2, cap), max_fit)


def style_preview_nav_spinbox(widget: QWidget) -> None:
    """Shared preview spinbox styling (same skin as settings-card spinboxes)."""
    from bactomate.gui.styles import card_aware_control_stylesheet

    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setStyleSheet(card_aware_control_stylesheet(compact_margins=True))
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


def style_preview_nav_combo(widget: QWidget) -> None:
    """Nav-row combo styling (32px tall, shared across preview toolbars)."""
    if isinstance(widget, QComboBox):
        widget.setStyleSheet(PREVIEW_NAV_COMBO_STYLESHEET + " QComboBox { combobox-popup: 0; }")
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        ensure_dark_combo_popup(widget)


def flex_control(
    widget: QWidget,
    min_width: int = 64,
    *,
    combo_style: bool = True,
    max_width: int | None = None,
) -> None:
    """Width-aware control; expands when useful but can be capped inside adaptive cells."""
    if combo_style and isinstance(widget, QComboBox):
        # Only apply nav combo sheet when parent has no card stylesheet (standalone nav rows).
        parent = widget.parentWidget()
        if parent is not None and not (parent.styleSheet() or "").strip():
            widget.setStyleSheet(PREVIEW_NAV_COMBO_STYLESHEET + " QComboBox { combobox-popup: 0; }")
        ensure_dark_combo_popup(widget)
    widget.setMinimumWidth(int(min_width))
    if max_width is not None and int(max_width) > 0:
        widget.setMaximumWidth(max(int(min_width), int(max_width)))
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    else:
        widget.setMaximumWidth(16777215)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def compact_control(widget: QWidget, width: int, *, combo_style: bool = True, fixed: bool = False) -> None:
    """Preview/module control sizing; defaults to expanding width unless ``fixed=True``."""
    if fixed:
        if combo_style and isinstance(widget, QComboBox):
            widget.setStyleSheet(PREVIEW_NAV_COMBO_STYLESHEET + " QComboBox { combobox-popup: 0; }")
            ensure_dark_combo_popup(widget)
        widget.setMinimumWidth(int(width))
        widget.setMaximumWidth(int(width))
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return
    flex_control(widget, min_width=int(width), combo_style=combo_style, max_width=max(int(width), 180))


def preview_settings_group(
    title: str = "",
    *,
    object_name: str = "PreviewSettingsCard",
    parent: QWidget | None = None,
) -> QGroupBox:
    """Standard blue-gray preview settings card with unified child control styling."""
    from PyQt6.QtWidgets import QGroupBox

    group = QGroupBox(title, parent)
    group.setObjectName(object_name)
    group.setStyleSheet(preview_settings_card_stylesheet(object_name))
    group.setMinimumWidth(0)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    return group


@dataclass(frozen=True)
class SettingsCardItem:
    """Declarative item for a shared settings card."""

    widget: QWidget
    label: str | None = None
    min_control_width: int = 64
    max_control_width: int | None = None


def setting_field(
    label: str,
    control: QWidget,
    *,
    min_width: int = 64,
    max_width: int | None = None,
) -> SettingsCardItem:
    """Labeled control cell for ``build_settings_card``."""
    return SettingsCardItem(
        widget=control,
        label=str(label),
        min_control_width=int(min_width),
        max_control_width=max_width,
    )


def setting_cell(widget: QWidget) -> SettingsCardItem:
    """Unlabeled cell (checkbox, button, custom row) for ``build_settings_card``."""
    return SettingsCardItem(widget=widget)


def build_settings_card(
    title: str = "",
    *,
    object_name: str = "PreviewSettingsCard",
    parent: QWidget | None = None,
    fields: Sequence[SettingsCardItem | QWidget] = (),
    footer_widgets: Sequence[QWidget] = (),
    min_cell_width: int = 160,
    label_min_width: int = 78,
    control_max_width: int = 120,
    max_columns: int = 4,
    h_spacing: int = 14,
    v_spacing: int = 6,
    margins: tuple[int, int, int, int] = (8, 2, 8, 4),
    spacing: int = 6,
    stylesheet: str | None = None,
) -> QWidget:
    """Build a standard themed settings card from declarative fields.

    Modules should create their widgets and pass them here instead of hand-wiring
    ``QVBoxLayout + AdaptiveFillGrid``. This keeps card spacing, width caps, and
    transparent wrappers consistent across previews and module panels.
    """
    group = preview_settings_group(title, object_name=object_name, parent=parent)
    if stylesheet:
        group.setStyleSheet(str(stylesheet))
    outer = QVBoxLayout(group)
    outer.setContentsMargins(*margins)
    outer.setSpacing(int(spacing))

    grid = AdaptiveFillGrid(
        group,
        min_cell_width=int(min_cell_width),
        label_min_width=int(label_min_width),
        control_max_width=int(control_max_width),
        max_columns=int(max_columns),
        h_spacing=int(h_spacing),
        v_spacing=int(v_spacing),
    )
    for field in fields:
        item = setting_cell(field) if isinstance(field, QWidget) else field
        if item.label:
            grid.add_field(
                item.label,
                item.widget,
                min_control_width=item.min_control_width,
                max_control_width=item.max_control_width,
            )
        else:
            grid.add_cell(item.widget)
    outer.addWidget(grid)

    for footer in footer_widgets:
        footer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        outer.addWidget(footer)

    try:
        group._settings_grid = grid  # type: ignore[attr-defined]
    except Exception:
        pass
    return group


def preview_settings_viewport_height(*, embedded: bool = False, max_height: int | None = None) -> int:
    """Standard cap for scrollable preview settings chrome."""
    if max_height is not None:
        return int(max_height)
    return PREVIEW_SETTINGS_SCROLL_EMBEDDED_MAX_HEIGHT if embedded else PREVIEW_SETTINGS_SCROLL_MAX_HEIGHT


class AdaptiveFillGrid(QWidget):
    """
    Reflow labeled fields into 1–4 equal-width columns based on available width.

    Controls stretch within each cell so cards use horizontal space efficiently.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        min_cell_width: int = 148,
        h_spacing: int = 8,
        v_spacing: int = 6,
        compact_bp: int = COMPACT_BREAKPOINT,
        medium_bp: int = MEDIUM_BREAKPOINT,
        wide_bp: int = WIDE_BREAKPOINT,
        label_min_width: int = 52,
        control_max_width: int = 160,
        max_columns: int = 4,
    ) -> None:
        super().__init__(parent)
        self._min_cell_width = int(min_cell_width)
        self._h_spacing = int(h_spacing)
        self._v_spacing = int(v_spacing)
        self._compact_bp = int(compact_bp)
        self._medium_bp = int(medium_bp)
        self._wide_bp = int(wide_bp)
        self._label_min_width = int(label_min_width)
        self._control_max_width = int(control_max_width)
        self._max_columns = int(max_columns)
        self._labeled: list[tuple[str, QWidget]] = []
        self._cells: list[QWidget] = []
        self._current_columns: int | None = None
        self._layout_dirty = True
        self._grid_host = QWidget(self)
        self._grid_host.setObjectName("AdaptiveFillGridHost")
        self._grid_host.setStyleSheet("#AdaptiveFillGridHost { background: transparent; }")
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(self._h_spacing)
        self._grid.setVerticalSpacing(self._v_spacing)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._grid_host)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def add_field(
        self,
        label: str,
        control: QWidget,
        *,
        min_control_width: int = 64,
        max_control_width: int | None = None,
    ) -> None:
        flex_control(
            control,
            min_width=min_control_width,
            max_width=self._control_max_width if max_control_width is None else max_control_width,
        )
        cell = self._make_labeled_cell(label, control)
        self._labeled.append((label, control))
        self._cells.append(cell)
        self._layout_dirty = True
        self._reflow()

    def add_cell(self, widget: QWidget) -> None:
        from PyQt6.QtWidgets import QPushButton

        if isinstance(widget, QPushButton):
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        else:
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        cell = self._wrap_grid_cell(widget)
        self._cells.append(cell)
        self._layout_dirty = True
        self._reflow()

    def reflow(self) -> None:
        self._reflow()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if _qt_widget_alive(self):
            self._reflow()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if _qt_widget_alive(self):
            self._reflow()

    def _wrap_grid_cell(self, widget: QWidget) -> QWidget:
        wrap = QWidget(self._grid_host)
        wrap.setObjectName("AdaptiveFillCell")
        wrap.setStyleSheet("#AdaptiveFillCell { background: transparent; }")
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(
            widget,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        lay.addStretch(1)
        wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        return wrap

    def _make_labeled_cell(self, label: str, control: QWidget) -> QWidget:
        wrap = QWidget(self._grid_host)
        wrap.setObjectName("AdaptiveFillCell")
        wrap.setStyleSheet("#AdaptiveFillCell { background: transparent; }")
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        if label:
            lbl = QLabel(label, wrap)
            lbl.setMinimumWidth(self._label_min_width)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lay.addWidget(lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(
            control,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        lay.addStretch(1)
        wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        return wrap

    def _column_count(self) -> int:
        return columns_for_width(
            self.width(),
            min_cell_width=self._min_cell_width,
            compact_bp=self._compact_bp,
            medium_bp=self._medium_bp,
            wide_bp=self._wide_bp,
            max_columns=self._max_columns,
        )

    def _reflow(self) -> None:
        if not _qt_widget_alive(self):
            return
        self._cells = [c for c in self._cells if _qt_widget_alive(c)]
        if not self._cells:
            return
        cols = max(1, self._column_count())
        if not self._layout_dirty and self._current_columns == cols:
            return

        for cell in self._cells:
            cell.setParent(self._grid_host)

        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(self._grid_host)

        for idx, cell in enumerate(self._cells):
            row, col = divmod(idx, cols)
            self._grid.addWidget(cell, row, col)
            self._grid.setColumnStretch(col, 1)
        self._current_columns = cols
        self._layout_dirty = False
        self.updateGeometry()


def build_preview_settings_scroll(
    content: QWidget,
    *,
    embedded: bool = False,
    max_height: int | None = None,
    min_height: int = PREVIEW_SETTINGS_SCROLL_MIN_HEIGHT,
    stretch: bool = False,
) -> QScrollArea:
    """Scrollable preview settings chrome with standard height cap."""
    return scrollable_control_panel(
        content,
        max_viewport_height=preview_settings_viewport_height(embedded=embedded, max_height=max_height),
        min_viewport_height=min_height,
        stretch=stretch or embedded,
    )


def control_row(
    label: str | QLabel | None,
    control: QWidget,
    *,
    spacing: int = 6,
    parent: QWidget | None = None,
) -> QWidget:
    """Label + control as one horizontal strip for responsive flow layouts."""
    wrap = QWidget(parent)
    wrap.setObjectName("TransparentControlRow")
    wrap.setStyleSheet("#TransparentControlRow { background: transparent; }")
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(int(spacing))
    if label is not None:
        lbl = QLabel(label, parent) if isinstance(label, str) else label
        lay.addWidget(lbl, 0)
    lay.addWidget(control, 0)
    wrap.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return wrap


def inline_controls(*widgets: QWidget, spacing: int = 4, parent: QWidget | None = None) -> QWidget:
    """Group widgets side-by-side (checkbox + spin, label + combo + info icon, etc.)."""
    wrap = QWidget(parent)
    wrap.setObjectName("TransparentInlineControls")
    wrap.setStyleSheet("#TransparentInlineControls { background: transparent; }")
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(int(spacing))
    for widget in widgets:
        lay.addWidget(widget, 0)
    wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return wrap


def labeled_control(
    label: str | QLabel,
    control: QWidget,
    *,
    parent: QWidget | None = None,
    spacing: int = 6,
) -> QWidget:
    """Label + control on one row; stretches to panel width without runtime reflow."""
    wrap = QWidget(parent)
    wrap.setObjectName("TransparentLabeledControl")
    wrap.setStyleSheet("#TransparentLabeledControl { background: transparent; }")
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(int(spacing))
    lbl = QLabel(label, parent) if isinstance(label, str) else label
    lay.addWidget(lbl, 0)
    lay.addWidget(control, 0)
    lay.addStretch(1)
    wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return wrap


def checkbox_with_control(
    checkbox: QCheckBox,
    control: QWidget,
    *,
    parent: QWidget | None = None,
    spacing: int = 4,
) -> QWidget:
    """Checkbox + paired control (spin/combo) as one stable row item."""
    return inline_controls(checkbox, control, spacing=spacing, parent=parent)


def stacked_control_rows(
    *rows: QHBoxLayout | QWidget,
    parent: QWidget | None = None,
    spacing: int = 6,
) -> QWidget:
    """Stack explicit horizontal rows vertically — deterministic preview layout."""
    wrap = QWidget(parent)
    wrap.setObjectName("TransparentStackedRows")
    wrap.setStyleSheet("#TransparentStackedRows { background: transparent; }")
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(int(spacing))
    for row in rows:
        if isinstance(row, QHBoxLayout):
            lay.addLayout(row)
        else:
            lay.addWidget(row)
    wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    return wrap


def stacked_control_grid(
    *cells: QWidget,
    columns: int = 2,
    parent: QWidget | None = None,
    h_spacing: int = 8,
    v_spacing: int = 4,
) -> QWidget:
    """Place control strips in a fixed grid (wraps to new rows by column count)."""
    wrap = QWidget(parent)
    wrap.setObjectName("TransparentStackedGrid")
    wrap.setStyleSheet("#TransparentStackedGrid { background: transparent; }")
    grid = QGridLayout(wrap)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(int(h_spacing))
    grid.setVerticalSpacing(int(v_spacing))
    cols = max(1, int(columns))
    for idx, cell in enumerate(cells):
        grid.addWidget(cell, idx // cols, idx % cols)
    wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    return wrap


def scrollable_control_panel(
    content: QWidget,
    *,
    parent: QWidget | None = None,
    max_viewport_height: int = 340,
    min_viewport_height: int = 72,
    stretch: bool = False,
) -> QScrollArea:
    """Scroll area for dense preview chrome; child keeps natural height."""
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(content)
    scroll.setMinimumHeight(int(min_viewport_height))
    if max_viewport_height > 0:
        scroll.setMaximumHeight(int(max_viewport_height))
    policy_v = (
        QSizePolicy.Policy.Expanding
        if stretch
        else QSizePolicy.Policy.Fixed
    )
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, policy_v)
    return scroll


class ResponsiveRowFlow(QWidget):
    """
    Pack child widgets into horizontal rows; wrap when the next item would overflow.

    Used for dense preview option strips (montage layout, foci filters) so controls
    reflow on narrow panels without hand-splitting every row.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        h_spacing: int = 8,
        v_spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._h_spacing = int(h_spacing)
        self._v_spacing = int(v_spacing)
        self._items: list[QWidget] = []
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(self._v_spacing)
        self._defer_reflow = False

    def add_widget(self, widget: QWidget) -> None:
        self._items.append(widget)
        widget.setParent(self)
        if not self._defer_reflow:
            self._reflow()

    def set_defer_reflow(self, defer: bool) -> None:
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
        for widget in self._items:
            widget.setParent(self)
        while self._outer.count():
            item = self._outer.takeAt(0)
            container = item.widget()
            if container is not None:
                container.deleteLater()

        width = self.width()
        if width <= 0:
            width = 640

        row_widget: QWidget | None = None
        row_layout: QHBoxLayout | None = None
        row_used = 0

        def _finish_row() -> None:
            nonlocal row_widget, row_layout, row_used
            if row_layout is not None:
                row_layout.addStretch(1)
            row_widget = None
            row_layout = None
            row_used = 0

        def _start_row() -> QHBoxLayout:
            nonlocal row_widget, row_layout, row_used
            _finish_row()
            row_widget = QWidget(self)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(self._h_spacing)
            self._outer.addWidget(row_widget)
            row_used = 0
            return row_layout

        for widget in self._items:
            if not _qt_widget_alive(widget):
                continue
            widget.show()
            item_width = max(widget.sizeHint().width(), widget.minimumSizeHint().width(), 1)
            if row_layout is None:
                lay = _start_row()
            elif row_used + self._h_spacing + item_width > width:
                lay = _start_row()
            else:
                lay = row_layout
            lay.addWidget(widget, 0)
            row_used += item_width + (self._h_spacing if row_used else 0)

        _finish_row()
        hint_h = max(int(self.sizeHint().height()), int(self.minimumSizeHint().height()), 1)
        self.setMinimumHeight(hint_h)
        self.updateGeometry()


def responsive_rows(
    *items: QWidget,
    parent: QWidget | None = None,
    h_spacing: int = 8,
    v_spacing: int = 6,
) -> ResponsiveRowFlow:
    """Build a wrapping row flow from pre-built control strips."""
    flow = ResponsiveRowFlow(parent, h_spacing=h_spacing, v_spacing=v_spacing)
    flow.set_defer_reflow(True)
    for item in items:
        flow.add_widget(item)
    flow.reflow()
    return flow


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
    btn.setObjectName("pathBrowseButton")
    btn.setStyleSheet(PATH_ROW_BROWSE_BUTTON_STYLESHEET)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    btn.setFixedHeight(int(height))
    fm = btn.fontMetrics()
    text_w = int(fm.boundingRect(btn.text() or "").width())
    pad = 28 if sys.platform == "win32" else 24
    btn.setFixedWidth(max(84 if sys.platform == "win32" else 72, text_w + pad))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)


def style_path_row_browse_button(btn: QPushButton) -> None:
    """Compact File…/Folder…-style button for module path strips and FI file picker."""
    _style_path_row_browse_button(btn, _PATH_ROW_CONTENT_HEIGHT)


def style_compact_action_button(
    btn: QPushButton,
    *,
    horizontal_padding: int = 24,
    dense: bool = False,
) -> None:
    """Compact secondary button (path row File…/Folder… style) without renaming label text."""
    if dense:
        btn.setStyleSheet(
            COMPACT_BROWSE_BUTTON_STYLESHEET
            + (
                "QPushButton {"
                " padding: 2px 10px;"
                " min-height: 0px;"
                " max-height: 30px;"
                " min-width: 0px;"
                "}"
            )
        )
    else:
        btn.setStyleSheet(
            COMPACT_BROWSE_BUTTON_STYLESHEET + " QPushButton { min-width: 0px; }"
        )
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    text = btn.text() or ""
    fm = btn.fontMetrics()
    # boundingRect can underestimate vs painted text (ellipsis, Windows Segoe UI); prefer advance.
    text_w = max(int(fm.horizontalAdvance(text)), int(fm.boundingRect(text).width()))
    # COMPACT_BROWSE_BUTTON_STYLESHEET: 12px horizontal padding per side + 1px border each side.
    chrome = 26
    win_fudge = 6 if sys.platform == "win32" else 2
    width = text_w + chrome + max(int(horizontal_padding), win_fudge)
    btn.setMinimumWidth(0)
    btn.setMaximumWidth(16777215)
    btn.setMinimumHeight(0)
    btn.setMaximumHeight(16777215)
    btn.adjustSize()
    width = max(width, btn.sizeHint().width() + win_fudge)
    btn.setFixedWidth(max(72, width))
    height_floor = 30 if dense else 34
    height = max(height_floor, btn.sizeHint().height() + (2 if sys.platform == "win32" else 0))
    btn.setFixedHeight(height)
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
