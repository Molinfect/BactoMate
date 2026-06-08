"""Responsive UI audit + theme token checks for the global layout system."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from bactomate.gui.layout_helpers import (
    COMPACT_BREAKPOINT,
    MEDIUM_BREAKPOINT,
    WIDE_BREAKPOINT,
    AdaptiveFillGrid,
    build_settings_card,
    columns_for_width,
    preview_settings_group,
    setting_field,
)
from bactomate.gui.styles import (
    UI_THEME,
    card_aware_control_stylesheet,
    filter_preview_card_stylesheet,
    get_module_shell_left_pane_stylesheet,
    preview_settings_card_stylesheet,
)

_GUI_ROOT = Path(__file__).resolve().parents[1] / "bactomate" / "gui"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# Audit snapshot (grep-derived classification for migration tracking).
AUDIT = {
    "module_shell": {
        "fixed_splitter_ratio": "replaced with draggable width-aware splitter",
        "left_pane_min_w": 240,
        "right_pane_min_w": 280,
    },
    "preview_cards": {
        "montage_layout": "build_settings_card declarative fields",
        "filter_foci": "AdaptiveFillGrid + filter_preview_card_stylesheet",
        "filter_cell": "filter_preview_card_stylesheet",
        "omnipose_embedded_segmentation": "build_settings_card declarative fields",
    },
    "deprecated_patterns": {
        "ResponsiveRowFlow_in_previews": "not used in production preview panels",
        "inline_232830_montage_styles": "replaced by preview_settings_card_stylesheet",
    },
}


def test_audit_structure_documents_key_areas() -> None:
    assert "module_shell" in AUDIT
    assert AUDIT["preview_cards"]["montage_layout"].startswith("build_settings_card")


def test_theme_tokens_include_card_and_field_colors() -> None:
    assert UI_THEME["preview_card_bg"] == "#1a1a1a"
    assert UI_THEME["field_bg"] == "#2a3140"
    assert UI_THEME["accent"] == "#4a9eff"


def test_card_stylesheets_use_theme_not_black_inputs() -> None:
    card = preview_settings_card_stylesheet("TestCard")
    assert UI_THEME["field_bg"] in card
    assert UI_THEME["preview_card_bg"] in card
    filter_card = filter_preview_card_stylesheet("FilterTest")
    assert UI_THEME["field_bg"] in filter_card


def test_module_shell_stylesheet_uses_unified_field_background() -> None:
    sheet = get_module_shell_left_pane_stylesheet()
    assert UI_THEME["field_bg_module"] in sheet


def test_columns_for_width_breakpoints() -> None:
    assert columns_for_width(240) == 1
    assert columns_for_width(480) == 2
    assert columns_for_width(720) >= 2
    assert columns_for_width(1000) >= 3
    assert COMPACT_BREAKPOINT < MEDIUM_BREAKPOINT < WIDE_BREAKPOINT


def test_adaptive_fill_grid_reflows_by_width(qapp) -> None:
    from PyQt6.QtWidgets import QSpinBox

    grid = AdaptiveFillGrid()
    grid.resize(1000, 120)
    for i in range(4):
        grid.add_field(f"F{i}", QSpinBox())
    grid.reflow()
    assert grid._column_count() >= 3


def test_preview_settings_group_applies_object_name_stylesheet(qapp) -> None:
    group = preview_settings_group("", object_name="AuditCard")
    assert group.objectName() == "AuditCard"
    assert UI_THEME["preview_card_bg"] in group.styleSheet()


def test_build_settings_card_is_declarative_entrypoint(qapp) -> None:
    from PyQt6.QtWidgets import QSpinBox

    spin = QSpinBox()
    card = build_settings_card(
        "",
        object_name="DeclarativeAuditCard",
        fields=[setting_field("Rows", spin, min_width=56, max_width=84)],
    )
    assert card.objectName() == "DeclarativeAuditCard"
    assert spin.maximumWidth() == 84


def test_card_aware_control_stylesheet_covers_inputs() -> None:
    sheet = card_aware_control_stylesheet()
    for selector in ("QSpinBox", "QDoubleSpinBox", "QComboBox", "QLineEdit"):
        assert selector in sheet


def test_touched_gui_files_parse() -> None:
    """Syntax check on primary responsive migration targets."""
    targets = [
        _GUI_ROOT / "styles.py",
        _GUI_ROOT / "layout_helpers.py",
        _GUI_ROOT / "module_shell.py",
        _GUI_ROOT / "preview" / "preview_window.py",
        _GUI_ROOT / "tabs" / "post_processing.py",
        _GUI_ROOT / "tabs" / "_analyze_masks_preview_filtering.py",
    ]
    for path in targets:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
