"""
Session Logs tab: list recent session log files, show overview, apply params to current session.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.constants import LOGS_DIR, SESSION_TAB_KEYS


class SessionLogsTab(QWidget):
    """Tab to list session log files, show overview of saved settings, and apply them."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._selected_path = None
        self._loaded_data = None
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        list_group = QGroupBox("Recent session logs (last 10)")
        list_layout = QVBoxLayout()
        self.log_list = QListWidget()
        self.log_list.setMinimumHeight(120)
        self.log_list.itemSelectionChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.log_list)
        load_other_btn = QPushButton("Load other...")
        load_other_btn.setToolTip("Open a session log file that is not in the list above")
        load_other_btn.clicked.connect(self._load_other)
        list_layout.addWidget(load_other_btn)
        list_group.setLayout(list_layout)
        scroll_layout.addWidget(list_group)

        overview_group = QGroupBox("Overview of selected session")
        overview_layout = QVBoxLayout()
        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setMinimumHeight(200)
        self.overview_text.setPlaceholderText(
            "Select a log file above or click 'Load other...' to see parameters."
        )
        overview_layout.addWidget(self.overview_text)
        overview_group.setLayout(overview_layout)
        scroll_layout.addWidget(overview_group)

        self.apply_btn = QPushButton("Apply now to current session")
        self.apply_btn.setToolTip(
            "Apply the parameters from the selected log to all tabs (parameters only; file paths are not changed)."
        )
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_now)
        scroll_layout.addWidget(self.apply_btn)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

        self._refresh_list()

    def _refresh_list(self):
        """Refresh the list of last 10 log files."""
        self.log_list.clear()
        self._selected_path = None
        self._loaded_data = None
        self.overview_text.clear()
        self.apply_btn.setEnabled(False)
        if not LOGS_DIR.exists():
            return
        files = sorted(
            LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for p in files[:10]:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                label = f"{mtime.strftime('%Y-%m-%d %H:%M:%S')} — {p.name}"
            except (OSError, ValueError) as e:
                from gui.log_config import get_logger
                get_logger().debug("Session log list: could not format mtime for %s: %s", p.name, e)
                label = p.name
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.log_list.addItem(item)

    def _on_selection_changed(self):
        """When user selects an item, load that file and show overview."""
        items = self.log_list.selectedItems()
        if not items:
            self._selected_path = None
            self._loaded_data = None
            self.overview_text.clear()
            self.apply_btn.setEnabled(False)
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        self._load_and_show(path)

    def _load_other(self):
        """Open file dialog to load a session log file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open session log",
            str(LOGS_DIR) if LOGS_DIR.exists() else "",
            "JSON (*.json);;All files (*)",
        )
        if path:
            self._load_and_show(path)

    def _load_and_show(self, path):
        """Load JSON at path, store in _loaded_data, and show overview."""
        self._selected_path = path
        self._loaded_data = None
        self.overview_text.clear()
        self.apply_btn.setEnabled(False)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            from gui.log_config import get_logger
            get_logger().warning("Could not load session file: %s — %s", path, e)
            self.overview_text.setPlainText(f"Could not load file:\n{e}")
            return
        self._loaded_data = data
        lines = []
        schema = data.get("schema_version", "?")
        ts = data.get("timestamp", "?")
        lines.append(f"Schema version: {schema}  |  Saved: {ts}\n")
        tabs_data = data.get("tabs", data)
        if not isinstance(tabs_data, dict):
            tabs_data = {}
        for key, tab_label in SESSION_TAB_KEYS:
            if key not in tabs_data or not isinstance(tabs_data[key], dict):
                continue
            lines.append(f"\n——— {tab_label} ———")
            for k, v in sorted(tabs_data[key].items()):
                lines.append(f"  {k}: {v}")
        self.overview_text.setPlainText("\n".join(lines) if lines else "(No tab data)")
        self.apply_btn.setEnabled(True)

    def _apply_now(self):
        """Apply the currently loaded session params to the main window."""
        if not self._loaded_data or not self.parent_window:
            return
        tabs_data = self._loaded_data.get("tabs", self._loaded_data)
        if isinstance(tabs_data, dict):
            self.parent_window.apply_session_params(tabs_data)
        QMessageBox.information(
            self,
            "Applied",
            "Parameters from the selected session have been applied to the current session (parameters only).",
        )

    def showEvent(self, event):
        """Refresh list when tab is shown."""
        super().showEvent(event)
        self._refresh_list()
