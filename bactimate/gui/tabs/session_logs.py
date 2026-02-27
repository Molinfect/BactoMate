"""
Session Logs: list recent session log files, show overview, apply params to current session.
SessionLogsPanel = compact widget for Status & Session logs area.
SessionLogsTab = full tab (kept for backward compatibility if needed).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.constants import LOGS_DIR, MAX_SESSION_LOGS, SESSION_TAB_KEYS
from bactimate.gui.preferences import get_default_browse_dir


_RECENT_LOGS_SHOWN = 5

# Default values for session params (matches initial widget values). Used to show " (default)" in overview.
# Only params listed here are annotated; others are shown without a default hint.
SESSION_PARAM_DEFAULTS: dict[str, dict[str, Any]] = {
    "nd2": {
        "bleach_correct": False,
        "bleach_method": 0,
        "drift_correct": False,
        "max_frames": 100,
        "max_z_projection": False,
        "separate_channels": True,
    },
    "segmentation": {
        "cluster": False,
        "create_overlay": True,
        "diameter": 30.0,
        "flow_threshold": 0.4,
        "mask_threshold": 0.0,
        "model": "bact_phase_omni",
        "omnipose_batch": 16,
        "omnipose_lr": 0.01,
        "omnipose_n_epochs": 50,
        "omnipose_pretrained": "bact_phase_omni",
        "omnipose_save_each": 0,
        "omnipose_tyx_x": 256,
        "omnipose_tyx_y": 256,
        "omnipose_use_gpu": True,
        "omnipose_use_tiled": True,
        "use_gpu": True,
    },
    "unet_segment": {
        "unet_batch_size": 16,
        "unet_epochs": 50,
        "unet_img_filter": "",
        "unet_mask_filter": "_mask",
        "unet_model": "",
        "unet_model_name": "",
        "unet_use_gpu": True,
    },
    "analyze_masks": {
        "analyze_fi": False,
        "analyze_fi_contour": False,
        "fi_match_by_name": True,
        "create_combined_summary": False,
        "create_vis": True,
        "detailed": True,
        "exclude_border": True,
        "fi_contour_thickness": 1,
        "max_area": 0.0,
        "max_aspect_ratio": 10.0,
        "max_width": 0.0,
        "min_area": 0.0,
        "min_aspect_ratio": 1.5,
        "min_eccentricity": 0.5,
        "min_length": 0.0,
        "min_solidity": 0.7,
        "min_width": 0.0,
        "pixel_size": 0.108,
    },
    "tracking": {
        "device": "CPU",
        "mode": "greedy",
        "model": "general_2d",
        "filter_pixel_size": 0.108,
        "filter_time_per_frame": 1.0,
        "filter_time_per_frame_unit": "min",
    },
    "growth_curve_video": {
        "format_avi": True,
        "format_gif": False,
        "format_mp4": False,
        "format_tif": True,
        "fps": 5.0,
        "time_interval": 10.0,
        "yaxis_max": 0.0,
        "yaxis_min": 0.0,
    },
}


def _values_equal(saved: Any, default: Any) -> bool:
    """Compare saved value to default (handles int/float, list)."""
    if type(saved) == type(default):
        if isinstance(saved, list):
            return saved == default
        return saved == default
    if isinstance(saved, (int, float)) and isinstance(default, (int, float)):
        return float(saved) == float(default)
    return False


def format_session_overview(data: dict) -> str:
    """Build overview text for a session log (with ' (default)' on values that match app defaults)."""
    lines = []
    schema = data.get("schema_version", "?")
    ts = data.get("timestamp", "?")
    lines.append(f"Schema: {schema}  |  Saved: {ts}")
    lines.append("All values below are what was saved/used (reproducible). (default) = same as app default.\n")
    tabs_data = data.get("tabs", data)
    if not isinstance(tabs_data, dict):
        tabs_data = {}
    for key, tab_label in SESSION_TAB_KEYS:
        if key not in tabs_data or not isinstance(tabs_data[key], dict):
            continue
        lines.append(f"\n——— {tab_label} ———")
        defaults = SESSION_PARAM_DEFAULTS.get(key, {})
        for k, v in sorted(tabs_data[key].items()):
            default_val = defaults.get(k)
            is_default = default_val is not None and _values_equal(v, default_val)
            suffix = "  (default)" if is_default else ""
            lines.append(f"  {k}: {v}{suffix}")
    return "\n".join(lines) if lines else "(No tab data)"


class SessionLogsPanel(QWidget):
    """Compact session logs widget for the Status & Session logs area: list + log content preview + Load other + Apply."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._selected_path: Optional[str] = None
        self._loaded_data: Optional[dict] = None
        self._build_ui()
        self._refresh_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Session logs")
        title.setStyleSheet("font-weight: bold; color: #4a9eff;")
        layout.addWidget(title)
        self.log_list = QListWidget()
        # Fixed height for exactly 5 items so no scroll (avoids scroll gap below last entry)
        self.log_list.setFixedHeight(22 * _RECENT_LOGS_SHOWN + 4)
        self.log_list.setUniformItemSizes(True)
        self.log_list.setStyleSheet("QListWidget { padding: 0; margin: 0; outline: none; }")
        self.log_list.setToolTip(
            f"Last {_RECENT_LOGS_SHOWN} session logs. Up to {MAX_SESSION_LOGS} are kept on disk; this list shows the {_RECENT_LOGS_SHOWN} most recent."
        )
        self.log_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.log_list)
        content_label = QLabel("Log content")
        content_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(content_label)
        content_row = QHBoxLayout()
        self.content_text = QTextEdit()
        self.content_text.setReadOnly(True)
        self.content_text.setPlaceholderText("Select a log above or click Load other...")
        self.content_text.setMinimumHeight(70)
        self.content_text.setMaximumHeight(105)
        self.content_text.setStyleSheet("font-size: 10px;")
        content_row.addWidget(self.content_text, 1)
        btn_col = QVBoxLayout()
        load_btn = QPushButton("Load other...")
        load_btn.setToolTip("Open a session log file not in the list")
        load_btn.setMaximumHeight(22)
        load_btn.setStyleSheet("font-size: 11px;")
        load_btn.clicked.connect(self._load_other)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Apply parameters from the selected log to current session (parameters only)")
        self.apply_btn.setMaximumHeight(22)
        self.apply_btn.setStyleSheet("font-size: 11px;")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_now)
        btn_col.addWidget(load_btn)
        btn_col.addWidget(self.apply_btn)
        btn_col.addStretch()
        content_row.addLayout(btn_col, 0)
        layout.addLayout(content_row)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMinimumWidth(200)

    def _refresh_list(self) -> None:
        self.log_list.clear()
        self._selected_path = None
        self._loaded_data = None
        self.content_text.clear()
        self.apply_btn.setEnabled(False)
        if not LOGS_DIR.exists():
            return
        files = sorted(
            LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for p in files[:_RECENT_LOGS_SHOWN]:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
                label = f"{mtime.strftime('%Y-%m-%d %H:%M')} — {p.name}"
            except (OSError, ValueError):
                label = p.name
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.log_list.addItem(item)

    def _on_selection_changed(self) -> None:
        items = self.log_list.selectedItems()
        if not items:
            self._selected_path = None
            self._loaded_data = None
            self.content_text.clear()
            self.apply_btn.setEnabled(False)
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        self._load_and_show(path)

    def _load_other(self) -> None:
        try:
            if not self or not self.parent_window:
                return
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open session log",
                str(LOGS_DIR) if LOGS_DIR.exists() else get_default_browse_dir(),
                "JSON (*.json);;All files (*)",
            )
            if path:
                self._load_and_show(path)
        except Exception:
            from bactimate.gui.log_config import get_logger
            get_logger().exception("Session logs: error in Load other")

    def _format_overview(self, data: dict) -> str:
        """Build overview text for the log content box (same format as full tab; shows ' (default)' where applicable)."""
        return format_session_overview(data)

    def _load_and_show(self, path: str) -> None:
        self._selected_path = path
        self._loaded_data = None
        self.content_text.clear()
        self.apply_btn.setEnabled(False)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            from bactimate.gui.log_config import get_logger
            get_logger().warning("Could not load session file: %s — %s", path, e)
            self.content_text.setPlainText(f"Error loading file:\n{e}")
            return
        self._loaded_data = data
        self.content_text.setPlainText(self._format_overview(data))
        self.apply_btn.setEnabled(True)

    def _apply_now(self) -> None:
        try:
            if not self or not self._loaded_data or not self.parent_window:
                return
            apply_fn = getattr(self.parent_window, "apply_session_params", None)
            if not callable(apply_fn):
                return
            tabs_data = self._loaded_data.get("tabs", self._loaded_data)
            if isinstance(tabs_data, dict):
                apply_fn(tabs_data)
            QMessageBox.information(
                self,
                "Applied",
                "Parameters from the selected session have been applied (parameters only; paths unchanged).",
            )
        except Exception:
            from bactimate.gui.log_config import get_logger
            get_logger().exception("Session logs: error in Apply")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_list()


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

        list_group = QGroupBox(f"Recent session logs (last 10 shown; up to {MAX_SESSION_LOGS} kept)")
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
                from bactimate.gui.log_config import get_logger
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
            str(LOGS_DIR) if LOGS_DIR.exists() else get_default_browse_dir(),
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
            from bactimate.gui.log_config import get_logger
            get_logger().warning("Could not load session file: %s — %s", path, e)
            self.overview_text.setPlainText(f"Could not load file:\n{e}")
            return
        self._loaded_data = data
        self.overview_text.setPlainText(format_session_overview(data))
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
