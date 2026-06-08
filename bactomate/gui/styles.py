"""
BactoMate global stylesheet and theme helpers.
"""


def get_modern_stylesheet() -> str:
    """
    Returns a comprehensive modern stylesheet for BactoMate with:
    - Modern color palette with gradients
    - Card-based design with shadows
    - Pill-shaped buttons with smooth transitions
    - Modern tabs with rounded corners
    - Enhanced input fields
    - Modern progress bars
    """
    return """
    /* ===== GLOBAL STYLES ===== */
    QMainWindow {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    QWidget {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: "Helvetica Neue", Helvetica, Arial;
        font-size: 11pt;
    }
    
    /* ===== MODERN TABS ===== */
    QTabWidget::pane {
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        background-color: #1f1f1f;
        top: -1px;
    }
    
    QTabBar::tab {
        background-color: #2d2d2d;
        color: #b0b0b0;
        border: none;
        padding: 8px 16px;
        margin-right: 3px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 500;
        font-size: 11pt;
        min-width: 80px;
    }
    
    QTabBar::tab:hover {
        background-color: #3a3a3a;
        color: #ffffff;
    }
    
    QTabBar::tab:selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        color: #ffffff;
        font-weight: 600;
    }
    
    QTabBar::tab:!selected {
        margin-top: 2px;
    }
    
    /* ===== MODERN GROUP BOXES (CARDS) ===== */
    QGroupBox {
        border: 1px solid #303844;
        border-radius: 8px;
        margin-top: 6px;
        padding-top: 10px;
        background-color: #1a1a1a;
        font-weight: 600;
        font-size: 11pt;
        color: #e0e0e0;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 2px 8px;
        background-color: #1a1a1a;
        color: #4a9eff;
        border-radius: 4px;
    }
    
    /* ===== MODERN BUTTONS ===== */
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #5a9eff, stop:1 #4a7fdd);
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 600;
        font-size: 11pt;
        min-height: 28px;
    }
    
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #6aaeff, stop:1 #5a8fed);
    }
    
    QPushButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a8eef, stop:1 #3a6fcd);
    }
    
    QPushButton:disabled {
        background-color: #3a3a3a;
        color: #666666;
    }
    
    /* Secondary button style */
    QPushButton[class="secondary"] {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a4a4a, stop:1 #3a3a3a);
    }
    
    QPushButton[class="secondary"]:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #5a5a5a, stop:1 #4a4a4a);
    }
    
    /* Analyze Masks: FI bar — label color matches QGroupBox::title (#4a9eff); font normal weight (not bold like default QPushButton) */
    QPushButton#analyzeFiToggleButton {
        font-weight: 400;
        font-size: 11pt;
    }
    QPushButton#analyzeFiToggleButton[class="secondary"] {
        color: #4a9eff;
    }
    QPushButton#analyzeFiToggleButton[class="secondary"]:hover {
        color: #6aaeff;
    }
    /* When on, same primary button look as elsewhere; white text for contrast on blue */
    QPushButton#analyzeFiToggleButton:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #5a9eff, stop:1 #4a7fdd);
        color: #ffffff;
        font-weight: 600;
    }
    QPushButton#analyzeFiToggleButton:checked:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #6aaeff, stop:1 #5a8fed);
        color: #ffffff;
    }
    QPushButton#analyzeFiToggleButton:checked:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a8eef, stop:1 #3a6fcd);
        color: #ffffff;
    }
    
    /* ===== MODERN INPUT FIELDS ===== */
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
        background-color: #1f1f1f;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        padding: 6px 10px;
        color: #ffffff;
        font-size: 11pt;
        selection-background-color: #4a9eff;
    }
    
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #4a9eff;
        background-color: #252525;
    }
    
    QLineEdit:hover, QTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
        border: 1px solid #4a4a4a;
    }

    /* ===== SPINBOX ARROWS (BETTER CONTRAST) ===== */
    /* Subtle inset seam between up/down — dark line reads as depth, not a bright divider. */
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #3a3a3a;
        border-bottom: 1px solid rgba(12, 28, 48, 0.55);
        border-top-right-radius: 6px;
        background-color: #344052;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border-left: 1px solid #3a3a3a;
        border-bottom-right-radius: 6px;
        background-color: #344052;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #46586f;
    }
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
        background-color: #2f3948;
    }
    QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled {
        background-color: #3d4d5c;
        border-left: 1px solid #3a3a3a;
        border-bottom: 1px solid rgba(0, 0, 0, 0.4);
    }
    QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {
        background-color: #3a4756;
        border-left: 1px solid #3a3a3a;
    }
    
    /* ===== MODERN COMBO BOXES ===== */
    QComboBox {
        background-color: #1f1f1f;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        padding: 6px 10px;
        color: #ffffff;
        font-size: 11pt;
        min-height: 28px;
    }
    
    QComboBox:hover {
        border: 1px solid #4a4a4a;
    }
    
    QComboBox:focus {
        border: 1px solid #4a9eff;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 24px;
        border-top-right-radius: 6px;
        border-bottom-right-radius: 6px;
        background-color: #3a3a3a;
    }
    
    QComboBox::drop-down:hover {
        background-color: #4a4a4a;
    }
    
    QComboBox QAbstractItemView {
        background-color: #1f1f1f;
        border: 1px solid #4a9eff;
        border-radius: 6px;
        selection-background-color: #4a9eff;
        selection-color: #ffffff;
        padding: 2px;
    }
    
    /* ===== MODERN CHECKBOXES ===== */
    QCheckBox {
        color: #e0e0e0;
        font-size: 11pt;
        spacing: 6px;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        background-color: #2d2d2d;
    }
    
    QCheckBox::indicator:hover {
        border: 1px solid #4a9eff;
    }
    
    QCheckBox::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        border: 1px solid #4a9eff;
    }
    
    
    /* ===== MODERN RADIO BUTTONS ===== */
    QRadioButton {
        color: #e0e0e0;
        font-size: 11pt;
        spacing: 6px;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #4a4a4a;
        border-radius: 9px;
        background-color: #2d2d2d;
    }
    
    QRadioButton::indicator:hover {
        border: 1px solid #4a9eff;
    }
    
    QRadioButton::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        border: 1px solid #4a9eff;
    }
    
    /* ===== MODERN PROGRESS BAR ===== */
    QProgressBar {
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        text-align: center;
        color: #ffffff;
        font-weight: 600;
        font-size: 10pt;
        background-color: #1f1f1f;
        height: 24px;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #4a9eff, stop:0.5 #5aaeff, stop:1 #4a9eff);
        border-radius: 5px;
    }
    
    /* ===== MODERN SCROLLBAR ===== */
    QScrollBar:vertical {
        border: none;
        background-color: #1f1f1f;
        width: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #4a4a4a;
        border-radius: 6px;
        min-height: 30px;
        margin: 2px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #5a5a5a;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        border: none;
        background-color: #1f1f1f;
        height: 12px;
        border-radius: 6px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #4a4a4a;
        border-radius: 6px;
        min-width: 30px;
        margin: 2px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #5a5a5a;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    
    /* ===== MODERN LIST WIDGET ===== */
    QListWidget {
        background-color: #1f1f1f;
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        padding: 3px;
        color: #ffffff;
    }
    
    QListWidget::item {
        padding: 6px;
        border-radius: 4px;
        margin: 1px;
    }
    
    QListWidget::item:hover {
        background-color: #2d2d2d;
    }
    
    QListWidget::item:selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4a9eff, stop:1 #357abd);
        color: #ffffff;
    }
    
    /* ===== MODERN DIALOG BUTTONS ===== */
    QDialogButtonBox QPushButton {
        min-width: 100px;
        min-height: 32px;
        padding: 6px 16px;
    }

    QMessageBox QPushButton {
        min-height: 32px;
        padding: 6px 16px;
    }
    
    /* ===== MODERN TOOLTIP ===== */
    QToolTip {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #4a9eff;
        border-radius: 6px;
        padding: 8px;
        font-size: 11pt;
    }
    
    /* ===== MODERN LABELS ===== */
    QLabel {
        color: #e0e0e0;
    }
    
    /* ===== MODERN TITLE BAR ===== */
    QWidget[class="title_bar"] {
        background: transparent;
        border-bottom: 1px solid #4a9eff;
    }
    """


# Flat bordered panel (no title chrome) for module sub-panels after outer CollapsibleSection headers.
FLAT_GROUPBOX_FRAME_STYLESHEET = (
    "QGroupBox { border: 1px solid #303844; border-radius: 8px; "
    "margin-top: 4px; padding: 8px 10px; background-color: #1a1a1a; min-width: 0; }"
    "QGroupBox QWidget { background: transparent; }"
)

# Drag-drop path strip: tight vertical padding so the line edit is not squashed inside nested group boxes.
PATH_DROP_GROUPBOX_STYLESHEET = (
    "QGroupBox { border: 1px solid #303844; border-radius: 8px; margin-top: 4px; "
    "padding: 8px 10px; background-color: #1a1a1a; min-width: 0; }"
    "QGroupBox QWidget { background: transparent; }"
)

# Same path-drop chrome as File Conversion, but without extra top margin inside a module path/action card.
PATH_DROP_GROUPBOX_NESTED_STYLESHEET = (
    "QGroupBox { border: 1px solid #303844; border-radius: 8px; margin-top: 0; "
    "padding: 8px 10px; background-color: #1a1a1a; min-width: 0; }"
    "QGroupBox QWidget { background: transparent; }"
)

# Path + preview/segment (or filter) actions: border with no extra space under the bottom button.
MODULE_PATH_ACTION_PANEL_STYLESHEET = (
    "QGroupBox#ModulePathActionPanel {"
    " border: 1px solid #303844;"
    " border-radius: 8px;"
    " margin-top: 4px;"
    " padding: 8px 10px 6px 10px;"
    " background-color: #1a1a1a;"
    " min-width: 0;"
    "}"
    "QGroupBox#ModulePathActionPanel QWidget { background: transparent; }"
)

# Secondary path pickers in module left panes (less prominent than primary action buttons).
MODULE_ACTION_BUTTON_STYLESHEET = (
    "QPushButton {"
    " background-color: #2d3a47;"
    " color: #e8eef6;"
    " border: 1px solid #4a6278;"
    " border-radius: 6px;"
    " padding: 5px 14px;"
    " font-size: 11pt;"
    " font-weight: 500;"
    " min-height: 28px;"
    " max-height: 34px;"
    "}"
    "QPushButton:hover { background-color: #364556; border-color: #4a9eff; }"
    "QPushButton:pressed { background-color: #24303a; }"
    "QPushButton:disabled { color: #6a7a88; background-color: #252a30; border-color: #3a3a3a; }"
)

COMPACT_BROWSE_BUTTON_STYLESHEET = (
    "QPushButton {"
    " background-color: #252d36;"
    " color: #b8d4f0;"
    " border: 1px solid #4a6278;"
    " border-radius: 4px;"
    " padding: 4px 12px;"
    " font-size: 10pt;"
    " font-weight: 500;"
    " min-height: 26px;"
    " min-width: 72px;"
    "}"
    "QPushButton:hover {"
    " background-color: #2a3540;"
    " color: #c8e0ff;"
    " border-color: #4a9eff;"
    "}"
    "QPushButton:pressed { background-color: #1e242c; }"
)

# Path row in module panes: overrides left-pane QLineEdit { min-height; padding } so borders are not clipped.
PATH_ROW_LINE_EDIT_STYLESHEET = (
    "QLineEdit {"
    " background-color: #1f1f1f;"
    " border: 1px solid #3a3a3a;"
    " border-radius: 6px;"
    " color: #ffffff;"
    " font-size: 12pt;"
    " padding: 3px 8px;"
    " min-height: 0px;"
    "}"
    "QLineEdit:focus { border: 1px solid #4a9eff; background-color: #252525; }"
    "QLineEdit:hover { border: 1px solid #4a4a4a; }"
)

PATH_ROW_BROWSE_BUTTON_STYLESHEET = (
    "QPushButton {"
    " background-color: #252d36;"
    " color: #b8d4f0;"
    " border: 1px solid #4a6278;"
    " border-radius: 4px;"
    " padding: 0px 12px;"
    " font-size: 10pt;"
    " font-weight: 500;"
    " min-height: 0px;"
    " min-width: 72px;"
    "}"
    "QPushButton:hover {"
    " background-color: #2a3540;"
    " color: #c8e0ff;"
    " border-color: #4a9eff;"
    "}"
    "QPushButton:pressed { background-color: #1e242c; }"
)


# ----- Responsive UI theme tokens (shared across module panes and preview cards) -----
RESPONSIVE_COMPACT_BREAKPOINT = 360
RESPONSIVE_MEDIUM_BREAKPOINT = 560
RESPONSIVE_WIDE_BREAKPOINT = 840

UI_THEME = {
    "app_bg": "#1a1a1a",
    "card_bg": "#1a1a1a",
    "preview_card_bg": "#1a1a1a",
    "preview_card_border": "#303844",
    "field_bg": "#2a3140",
    "field_bg_module": "#2a3140",
    "field_border": "#3a4555",
    "field_border_module": "#3a3a3a",
    "text": "#e8eef6",
    "muted_text": "#c8d0dc",
    "accent": "#4a9eff",
    "disabled_text": "#6a7a88",
}


def card_aware_control_stylesheet(*, compact_margins: bool = False) -> str:
    """Spin/combo/line styles that match preview cards (not near-black islands)."""
    margin = " margin: 0;" if compact_margins else " margin-top: 2px; margin-bottom: 2px;"
    fg = UI_THEME["field_bg"]
    border = UI_THEME["field_border"]
    text = UI_THEME["text"]
    return (
        f"QSpinBox, QDoubleSpinBox, QLineEdit {{"
        f" background-color: {fg};"
        f" color: {text};"
        f" border: 1px solid {border};"
        f" border-radius: 6px;"
        f" padding: 4px 8px;"
        f" min-height: 28px;{margin}"
        f"}}"
        f"QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {UI_THEME['accent']}; }}"
        f"QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {UI_THEME['accent']}; }}"
        f"QSpinBox::up-button, QDoubleSpinBox::up-button {{"
        f" subcontrol-origin: border;"
        f" subcontrol-position: top right;"
        f" width: 20px;"
        f" border-left: 1px solid {border};"
        f" border-bottom: 1px solid rgba(12, 28, 48, 0.55);"
        f" border-top-right-radius: 5px;"
        f" background-color: #344052;"
        f"}}"
        f"QSpinBox::down-button, QDoubleSpinBox::down-button {{"
        f" subcontrol-origin: border;"
        f" subcontrol-position: bottom right;"
        f" width: 20px;"
        f" border-left: 1px solid {border};"
        f" border-bottom-right-radius: 5px;"
        f" background-color: #344052;"
        f"}}"
        f"QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,"
        f"QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{"
        f" background-color: #46586f;"
        f"}}"
        f"QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,"
        f"QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{"
        f" background-color: #2f3948;"
        f"}}"
        f"QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,"
        f"QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {{"
        f" background-color: #2f3a48;"
        f" border-left: 1px solid {border};"
        f"}}"
        f"QComboBox {{"
        f" background-color: {fg};"
        f" color: {text};"
        f" border: 1px solid {border};"
        f" border-radius: 6px;"
        f" padding: 4px 8px;"
        f" min-height: 28px;{margin}"
        f"}}"
        f"QComboBox:hover {{ border-color: {UI_THEME['accent']}; }}"
        f"QComboBox::drop-down {{ border: none; width: 22px; }}"
        f"QComboBox QAbstractItemView {{"
        f" background-color: {fg};"
        f" color: {text};"
        f" selection-background-color: {UI_THEME['accent']};"
        f" selection-color: #ffffff;"
        f" border: 1px solid {border};"
        f" outline: none;"
        f"}}"
        f"QLabel {{ color: {UI_THEME['muted_text']}; background: transparent; }}"
        f"QCheckBox {{ color: {UI_THEME['text']}; spacing: 6px; }}"
    )


def preview_settings_card_stylesheet(object_name: str = "PreviewSettingsCard") -> str:
    """Compact dark settings card shared by preview and module controls."""
    return (
        f"QGroupBox#{object_name} {{"
        f" border: 1px solid {UI_THEME['preview_card_border']};"
        f" border-radius: 8px;"
        f" margin-top: 4px;"
        f" padding: 2px 10px 4px 10px;"
        f" background-color: {UI_THEME['preview_card_bg']};"
        f" min-width: 0;"
        f"}}"
        f"QGroupBox#{object_name}::title {{"
        f" color: {UI_THEME['accent']};"
        f" padding: 0 6px;"
        f" subcontrol-origin: margin;"
        f" left: 10px;"
        f" font-weight: 600;"
        f"}}"
        f"QGroupBox#{object_name} QWidget {{ background: transparent; }}"
        + card_aware_control_stylesheet(compact_margins=True)
    )


def filter_preview_card_stylesheet(object_name: str = "FilterPreviewCard") -> str:
    """Compatibility alias for dense filter cards; uses the shared card skin."""
    return preview_settings_card_stylesheet(object_name)


# Preview nav rows: no extra vertical margin so "View:" aligns with the combo on one baseline.
PREVIEW_NAV_COMBO_STYLESHEET = (
    "QComboBox {"
    f" background-color: {UI_THEME['field_bg']};"
    " color: #eeeeee;"
    f" border: 1px solid {UI_THEME['field_border']};"
    " border-radius: 6px;"
    " padding: 4px 8px;"
    " min-height: 28px;"
    " max-height: 32px;"
    " margin: 0;"
    "}"
    "QComboBox:hover { border-color: #4a9eff; }"
    "QComboBox::drop-down { border: none; width: 22px; }"
    "QComboBox QAbstractItemView {"
    f" background-color: {UI_THEME['field_bg']};"
    " color: #eeeeee;"
    " selection-background-color: #4a9eff;"
    " selection-color: #ffffff;"
    f" border: 1px solid {UI_THEME['field_border']};"
    " outline: none;"
    "}"
)

PREVIEW_NAV_SPINBOX_STYLESHEET = card_aware_control_stylesheet(compact_margins=True)


def preview_table_stylesheet() -> str:
    """Compact data table styling aligned with preview card theme tokens."""
    t = UI_THEME
    return (
        "QTableWidget {"
        f" background-color: {t['preview_card_bg']};"
        f" color: {t['text']};"
        f" gridline-color: {t['field_border']};"
        f" border: 1px solid {t['preview_card_border']};"
        " border-radius: 6px;"
        "}"
        "QTableWidget::item { padding: 1px 3px; }"
        "QTableWidget::item:selected {"
        f" background-color: {t['accent']};"
        " color: #ffffff;"
        "}"
        "QHeaderView::section {"
        f" background-color: {t['field_bg']};"
        f" color: {t['muted_text']};"
        " padding: 2px 2px;"
        " border: none;"
        f" border-right: 1px solid {t['field_border']};"
        " font-weight: 600;"
        "}"
        "QHeaderView::section:last { border-right: none; }"
    )

# Module dropdowns use the same palette as preview dropdowns, with module-pane vertical margins.
MODULE_PANEL_COMBO_STYLESHEET = (
    PREVIEW_NAV_COMBO_STYLESHEET
    + " QComboBox { min-height: 24px; margin-top: 6px; margin-bottom: 6px; }"
)

# Segmentation preview: cached-parameter picker — quieter than main nav combos.
PREVIEW_CACHE_RUN_COMBO_STYLESHEET = (
    "QComboBox {"
    f" background-color: {UI_THEME['field_bg']};"
    " color: #dce4e8;"
    f" border: 1px solid {UI_THEME['field_border']};"
    " border-radius: 5px;"
    " padding: 3px 8px;"
    " min-height: 28px;"
    " max-height: 32px;"
    " margin: 0;"
    " font-size: 11pt;"
    "}"
    "QComboBox:hover { border-color: #4a9eff; color: #ffffff; }"
    "QComboBox::drop-down { border: none; width: 20px; }"
    "QComboBox QAbstractItemView {"
    f" background-color: {UI_THEME['field_bg']};"
    " color: #dddddd;"
    " selection-background-color: #4a9eff;"
    " selection-color: #ffffff;"
    f" border: 1px solid {UI_THEME['field_border']};"
    " outline: none;"
    " font-size: 11pt;"
    "}"
)


def get_module_shell_left_pane_stylesheet() -> str:
    """Font bump + unified field/combo styling for module shell left scroll."""
    fg = UI_THEME["field_bg_module"]
    border = UI_THEME["field_border_module"]
    return (
        "QWidget { font-size: 12pt; }"
        "QLabel { font-size: 12pt; color: #dce4ee; }"
        f"QLineEdit {{ font-size: 13pt; min-height: 30px; padding: 4px 8px;"
        f" background-color: {fg}; border: 1px solid {border}; border-radius: 6px; color: #ffffff; }}"
        f"QLineEdit:focus {{ border-color: {UI_THEME['accent']}; }}"
        "QGroupBox { font-size: 12pt; font-weight: 600; border-color: #303844; background-color: #1a1a1a; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #aeb6c2; font-weight: 600; }"
        "QPushButton { font-size: 12pt; }"
        "QCheckBox { font-size: 12pt; color: #dce4ee; spacing: 6px; }"
        "QRadioButton { font-size: 12pt; }"
        f"QSpinBox, QDoubleSpinBox {{ font-size: 12pt; background-color: {fg};"
        f" border: 1px solid {border}; border-radius: 6px; padding: 4px 8px; color: #eeeeee; }}"
        f"QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {UI_THEME['accent']}; }}"
        "QTextEdit { font-size: 12pt; }"
        + MODULE_PANEL_COMBO_STYLESHEET
    )
