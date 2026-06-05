"""
Project dashboard: workspace registration and navigation into pipeline modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.layout_helpers import style_compact_action_button
from bactomate.gui.project_model import (
    MODULE_FILE_CONVERSION,
    MODULE_POST_PROCESSING,
    MODULE_SEGMENTATION,
    MODULE_TRACKING,
    MODULE_TRAINING,
)


class _ModuleCard(QFrame):
    """Hover highlight; click anywhere on the card to open the module."""

    def __init__(
        self,
        inner: QWidget,
        *,
        module_id: str,
        on_open: Callable[[str], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._module_id = module_id
        self._on_open = on_open
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("ModuleCardOuter")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to open this module")
        self.setStyleSheet(
            "#ModuleCardOuter {"
            " background-color: #252525;"
            " border: 2px solid #3d3d3d;"
            " border-radius: 12px;"
            " padding: 4px;"
            "}"
            "#ModuleCardOuter:hover {"
            " border-color: #4a9eff;"
            " background-color: #2a2a2e;"
            "}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(0)
        inner.setStyleSheet(
            "QWidget { background-color: transparent; border: none; }"
            "QLabel { background-color: transparent; }"
        )
        lay.addWidget(inner)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_open(self._module_id)
        super().mousePressEvent(event)


class DashboardWidget(QWidget):
    """Landing page: open/create workspace, jump to modules."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        on_open_workspace: Callable[[], None],
        on_clear_workspace: Callable[[], None],
        on_go_module: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self._on_open_workspace = on_open_workspace
        self._on_clear_workspace = on_clear_workspace
        self._on_go_module = on_go_module

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        inner = QWidget()
        inner.setMinimumWidth(0)
        # Enough height that module cards are not clipped when the log panel is open on short screens.
        inner.setMinimumHeight(560)

        root = QVBoxLayout(inner)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        ws_row = QHBoxLayout()
        self.workspace_label = QLabel("No workspace open")
        self.workspace_label.setObjectName("WorkspaceStatusLabel")
        self.workspace_label.setWordWrap(True)
        self.workspace_label.setStyleSheet(
            "#WorkspaceStatusLabel {"
            " color: #ffffff;"
            " font-size: 13pt;"
            " padding: 12px 14px;"
            " background: #252525;"
            " border-radius: 8px;"
            " border: 1px solid #3d3d3d;"
            "}"
        )
        self.workspace_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        ws_row.addWidget(self.workspace_label, 1)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        open_btn = QPushButton("Open workspace…")
        style_compact_action_button(open_btn)
        open_btn.setToolTip("Choose a folder; create or load bactomate_project.json")
        open_btn.clicked.connect(on_open_workspace)
        btn_row.addWidget(open_btn)
        clear_btn = QPushButton("Clear workspace")
        style_compact_action_button(clear_btn)
        clear_btn.setToolTip(
            "Reset BactoMate to a fresh state: clear the workspace, all module paths, "
            "and embedded previews. Does not delete files on disk."
        )
        clear_btn.clicked.connect(on_clear_workspace)
        btn_row.addWidget(clear_btn)
        ws_row.addLayout(btn_row)
        root.addLayout(ws_row)

        modules = QGroupBox("Modules")
        modules.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #4a9eff; font-size: 13pt; border: 1px solid #3d3d3d; "
            "border-radius: 8px; margin-top: 12px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }"
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        modules.setLayout(grid)

        cards = [
            ("File conversion", MODULE_FILE_CONVERSION, "ND2, LIF, CZI, or TIFF → TIFF stacks and metadata."),
            ("Segmentation", MODULE_SEGMENTATION, "Cellpose / Omnipose / U-Net, then mask filtering and quantify."),
            ("Tracking", MODULE_TRACKING, "Trackastra trajectories with optional phase overlay."),
            ("Post-processing", MODULE_POST_PROCESSING, "Scale bars, montages, and growth-curve videos."),
        ]
        for i, (title, mid, blurb) in enumerate(cards):
            grid.addWidget(self._make_card(title, blurb, mid), i // 2, i % 2)

        root.addWidget(modules)

        train_box = QGroupBox("Training")
        train_box.setStyleSheet(modules.styleSheet())
        train_outer = QVBoxLayout()
        train_outer.setContentsMargins(8, 8, 8, 8)
        train_outer.addWidget(
            self._make_card(
                "Training",
                "Train Omnipose or U-Net models on your labeled data in this workspace.",
                MODULE_TRAINING,
                full_width=True,
            )
        )
        train_box.setLayout(train_outer)
        root.addWidget(train_box)

        root.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _make_card(self, title: str, blurb: str, module_id: str, *, full_width: bool = False) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        t = QLabel(title)
        t.setStyleSheet("font-size: 15pt; font-weight: bold; color: #f0f0f0;")
        lay.addWidget(t)
        b = QLabel(blurb)
        b.setWordWrap(True)
        b.setStyleSheet("color: #c5c5c5; font-size: 11pt; line-height: 1.35;")
        lay.addWidget(b)
        if full_width:
            inner.setMinimumHeight(120)
        else:
            inner.setMinimumHeight(150)
        inner.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        for w in inner.findChildren(QWidget):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        card = _ModuleCard(inner, module_id=module_id, on_open=self._on_go_module, parent=self)
        if full_width:
            card.setMinimumHeight(148)
        else:
            card.setMinimumHeight(178)
        return card

    def set_workspace_display(self, path: Optional[Path]) -> None:
        if path is None:
            self.workspace_label.setText(
                "No workspace open — choose a folder. BactoMate will create bactomate_project.json there "
                "and suggest paths for each module from your files (Tif_files, masks, _fi / _foci, etc.)."
            )
        else:
            self.workspace_label.setText(f"Workspace: {path.resolve()}")
