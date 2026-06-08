"""Unit tests for deterministic preview layout helpers."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QHBoxLayout, QSpinBox

from bactomate.gui.layout_helpers import (
    AdaptiveFillGrid,
    build_settings_card,
    checkbox_with_control,
    columns_for_width,
    compact_control,
    flex_control,
    labeled_control,
    preview_settings_group,
    scrollable_control_panel,
    setting_cell,
    setting_field,
    stacked_control_grid,
    stacked_control_rows,
    vbox_from_rows,
)
from bactomate.gui.styles import PREVIEW_NAV_COMBO_STYLESHEET, UI_THEME


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_flex_control_expands_width(qapp) -> None:
    spin = QSpinBox()
    flex_control(spin, 80, combo_style=False)
    assert spin.minimumWidth() == 80
    assert spin.maximumWidth() >= 16777215


def test_compact_control_fixed_mode(qapp) -> None:
    combo = QComboBox()
    combo.addItems(["A", "B"])
    compact_control(combo, 120, fixed=True)
    assert combo.styleSheet() == PREVIEW_NAV_COMBO_STYLESHEET
    assert combo.minimumWidth() == 120
    assert combo.maximumWidth() == 120


def test_compact_control_defaults_to_flex(qapp) -> None:
    spin = QSpinBox()
    compact_control(spin, 90, combo_style=False)
    assert spin.maximumWidth() == 180


def test_labeled_control_wraps_label_and_widget(qapp) -> None:
    from PyQt6.QtWidgets import QLabel

    spin = QSpinBox()
    row = labeled_control("Rows", spin)
    labels = row.findChildren(QLabel)
    spins = row.findChildren(QSpinBox)
    assert len(labels) == 1
    assert labels[0].text() == "Rows"
    assert spins == [spin]


def test_checkbox_with_control_groups_pair(qapp) -> None:
    cb = QCheckBox("Prob")
    spin = QSpinBox()
    strip = checkbox_with_control(cb, spin)
    assert strip.findChildren(QCheckBox) == [cb]
    assert strip.findChildren(QSpinBox) == [spin]


def test_stacked_control_grid_places_cells(qapp) -> None:
    a = checkbox_with_control(QCheckBox("A"), QSpinBox())
    b = checkbox_with_control(QCheckBox("B"), QSpinBox())
    grid = stacked_control_grid(a, b, columns=2)
    assert grid.layout() is not None
    assert grid.layout().count() == 2


def test_stacked_control_rows_and_scroll_panel(qapp) -> None:
    row = QHBoxLayout()
    row.addWidget(labeled_control("One", QSpinBox()))
    panel_body = stacked_control_rows(vbox_from_rows(row), spacing=4)
    scroll = scrollable_control_panel(panel_body, max_viewport_height=240)
    assert scroll.widget() is panel_body
    assert scroll.widgetResizable() is True


def test_adaptive_fill_grid_column_math(qapp) -> None:
    assert columns_for_width(320) == 1
    assert columns_for_width(520) == 2
    grid = AdaptiveFillGrid()
    grid.resize(900, 100)
    grid.add_field("A", QSpinBox())
    grid.add_field("B", QSpinBox())
    assert grid._column_count() >= 2


def test_preview_settings_group_theme(qapp) -> None:
    group = preview_settings_group("", object_name="TestMontage")
    assert UI_THEME["preview_card_bg"] in group.styleSheet()


def test_build_settings_card_accepts_fields_and_cells(qapp) -> None:
    cb = QCheckBox("Enable")
    spin = QSpinBox()
    card = build_settings_card(
        "",
        object_name="BuiltCard",
        fields=[
            setting_cell(cb),
            setting_field("Value", spin, min_width=72, max_width=96),
        ],
    )
    assert card.objectName() == "BuiltCard"
    assert UI_THEME["preview_card_bg"] in card.styleSheet()
    assert card.findChildren(QCheckBox) == [cb]
    assert card.findChildren(QSpinBox) == [spin]
    assert spin.minimumWidth() == 72
    assert spin.maximumWidth() == 96


def test_build_settings_card_supports_dense_filter_layout(qapp) -> None:
    spin = QSpinBox()
    card = build_settings_card(
        "Filter",
        object_name="DenseFilterCard",
        min_cell_width=160,
        max_columns=5,
        stylesheet="QGroupBox#DenseFilterCard { background-color: #1a1a1a; }",
        fields=[setting_field("Radius", spin, min_width=52, max_width=76)],
    )
    grid = getattr(card, "_settings_grid")
    grid.resize(1000, 80)
    grid.reflow()
    assert grid._column_count() == 5
    assert "#1a1a1a" in card.styleSheet()
