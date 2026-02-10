"""
BactiMate global stylesheet and theme helpers.
"""


def get_modern_stylesheet() -> str:
    """
    Returns a comprehensive modern stylesheet for BactiMate with:
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
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 14px;
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
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #3a3a3a;
        border-top-right-radius: 6px;
        background-color: #5a9eff;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border-left: 1px solid #3a3a3a;
        border-bottom-right-radius: 6px;
        background-color: #5a9eff;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #6aaeff;
    }
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
        background-color: #4a8eef;
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
