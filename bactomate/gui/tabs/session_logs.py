"""
Session Logs: list recent session log files, show overview, apply params to current session.
SessionLogsPanel = compact widget for Status & Session logs area.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

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

from bactomate.gui.constants import LOGS_DIR, MAX_SESSION_LOGS, SESSION_TAB_KEYS
from bactomate.gui.preferences import get_default_browse_dir
from bactomate.gui.styles import COMPACT_BROWSE_BUTTON_STYLESHEET, MODULE_ACTION_BUTTON_STYLESHEET
from bactomate.gui.widgets import create_info_icon


_RECENT_LOGS_SHOWN = 5

_SESSION_LOG_LIST_STYLESHEET = (
    "QListWidget {"
    " background-color: #1f1f1f;"
    " color: #dce6f2;"
    " border: 1px solid #2d3a47;"
    " border-radius: 6px;"
    " padding: 2px;"
    " margin: 0;"
    " outline: none;"
    "}"
    "QListWidget::item { padding: 2px 6px; border-radius: 4px; }"
    "QListWidget::item:selected { background-color: #2d3a47; color: #ffffff; }"
    "QListWidget::item:hover { background-color: #263241; }"
)

_SESSION_LOG_TEXT_STYLESHEET = (
    "QTextEdit {"
    " background-color: #1f1f1f;"
    " color: #dce6f2;"
    " border: 1px solid #2d3a47;"
    " border-radius: 6px;"
    " padding: 6px 8px;"
    " font-size: 10px;"
    "}"
    "QTextEdit:focus { border-color: #4a9eff; background-color: #252525; }"
)

_SESSION_LOG_COMPACT_BUTTON_STYLESHEET = (
    COMPACT_BROWSE_BUTTON_STYLESHEET
    + "QPushButton { min-height: 22px; max-height: 24px; min-width: 92px; padding: 1px 10px; }"
)

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
        "detect_foci": False,
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
        "spotiflow_auto_tiling": True,
        "spotiflow_bg_subtract_radius": 0,
        "spotiflow_blur_sigma": 0.0,
        "spotiflow_min_distance": 1,
        "spotiflow_model_name": "general",
        "spotiflow_model_path": "",
        "spotiflow_n_tiles_x": 1,
        "spotiflow_n_tiles_y": 1,
        "spotiflow_normalize": True,
        "spotiflow_perc_high": 99.8,
        "spotiflow_perc_low": 1.0,
        "spotiflow_prob_thresh": None,
        "spotiflow_raise_on_error": False,
        "spotiflow_subpix": True,
        "spotiflow_use_optimized_threshold": True,
        "spotiflow_verbose": False,
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
    created = data.get("created_at") or data.get("timestamp", "?")
    updated = data.get("updated_at") or data.get("timestamp", "?")
    lines.append(f"Schema: {schema}  |  Created: {created}  |  Updated: {updated}")
    lines.append("All values below are the reusable analysis recipe saved for that app session. (default) = same as app default.\n")
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
    """Compact session logs widget for the Status & Session logs area: optional session replay, list, preview, Apply."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        on_rerun_session: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._on_rerun_session = on_rerun_session
        self._selected_path: Optional[str] = None
        self._loaded_data: Optional[dict] = None
        self._build_ui()
        self._refresh_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Session logs")
        title.setStyleSheet("font-weight: bold; color: #aeb6c2;")
        layout.addWidget(title)
        if self._on_rerun_session is not None:
            replay_row = QHBoxLayout()
            replay_row.setSpacing(10)
            rerun_btn = QPushButton("Re-run analysis from session…")
            rerun_btn.setMinimumHeight(42)
            rerun_btn.setMinimumWidth(160)
            rerun_btn.setStyleSheet(MODULE_ACTION_BUTTON_STYLESHEET)
            rerun_btn.setToolTip(
                "Choose a session log (JSON). Uses the current File conversion path as the new input folder or file."
            )
            rerun_btn.clicked.connect(self._on_rerun_session)
            replay_row.addWidget(rerun_btn, 0)
            replay_row.addWidget(
                create_info_icon(
                    "Pick a session JSON. The app loads parameters from that session (models, FI/foci, filters, etc.) "
                    "and runs the full pipeline on the path set in File conversion. Outputs go next to that input.",
                    parent_widget=self,
                ),
                0,
            )
            replay_row.addStretch(1)
            layout.addLayout(replay_row)
        self.log_list = QListWidget()
        # Fixed height for exactly 5 items so no scroll (avoids scroll gap below last entry)
        self.log_list.setFixedHeight(22 * _RECENT_LOGS_SHOWN + 4)
        self.log_list.setUniformItemSizes(True)
        self.log_list.setStyleSheet(_SESSION_LOG_LIST_STYLESHEET)
        self.log_list.setToolTip(
            f"Last {_RECENT_LOGS_SHOWN} reusable session recipe files. One JSON is created per app launch and updated while you work. Up to {MAX_SESSION_LOGS} are kept on disk."
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
        self.content_text.setStyleSheet(_SESSION_LOG_TEXT_STYLESHEET)
        content_row.addWidget(self.content_text, 1)
        btn_col = QVBoxLayout()
        load_btn = QPushButton("Load other...")
        load_btn.setToolTip("Open a session log file not in the list")
        load_btn.setMaximumHeight(22)
        load_btn.setStyleSheet(_SESSION_LOG_COMPACT_BUTTON_STYLESHEET)
        load_btn.clicked.connect(self._load_other)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Apply parameters from the selected log to current session (parameters only)")
        self.apply_btn.setMaximumHeight(22)
        self.apply_btn.setStyleSheet(_SESSION_LOG_COMPACT_BUTTON_STYLESHEET)
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
            from bactomate.gui.log_config import get_logger
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
            from bactomate.gui.log_config import get_logger
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
            from bactomate.gui.log_config import get_logger
            get_logger().exception("Session logs: error in Apply")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_list()
