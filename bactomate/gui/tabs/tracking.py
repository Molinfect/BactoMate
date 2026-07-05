"""
Tracking tab: run trackastra on mask TIFFs.
"""
from __future__ import annotations
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np

def _pd():
    """Lazy pandas import (avoids heavy import cost during GUI startup)."""
    import pandas as pd
    return pd

def _is_pandas_dataframe(obj: Any) -> bool:
    try:
        return isinstance(obj, _pd().DataFrame)
    except Exception:
        return False
TRACKING_PREVIEW_FRAME_LIMIT = 10
TRACKING_PREVIEW_FULL_STACK_WARN_FRAMES = 80
FILTER_CAL_SPIN_WIDTH = 80
FILTER_CAL_TIME_SPIN_WIDTH = 118
FILTER_METRIC_SPIN_WIDTH = 68
FILTER_METRIC_DEC_SPIN_WIDTH = 80
FILTER_GRID_COLUMNS = 2
FILTER_METRIC_CELL_MIN_HEIGHT = 34
FILTER_GRID_H_SPACING = 10
FILTER_GRID_V_SPACING = 6
METRICS_TABLE_COL_WEIGHTS = (36, 56, 72, 88, 44, 72, 72, 88, 72, 56)
METRICS_TABLE_COL_MIN_WIDTH = 28
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QButtonGroup, QDialog, QDialogButtonBox, QFrame, QProgressBar, QProgressDialog, QPushButton, QRadioButton, QScrollArea, QListWidget, QListWidgetItem, QSlider, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from bactomate.gui.analysis_paths import base_dir_for_mask, find_analysis_excel_for_mask, resolve_analysis_workbook_for_mask, stem_from_mask_path
from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.image_utils import scale_for_preview, to_qpixmap
from bactomate.gui.preferences import get_default_browse_dir
from bactomate.gui.utils import safe_imread
from bactomate.gui.session_params import apply_to_widgets, get_from_widgets
from bactomate.gui.widgets import ProgressCancelButton, create_info_icon, show_confirm_dialog, show_file_selection_dialog, show_message_dialog, style_dialog_button_box
from bactomate.modules.tiff_metadata import read_metadata_from_path, resolve_calibration_for_mask, resolve_calibration_from_trajectories_path, resolve_original_for_mask
from bactomate.gui.artifact_paths import collect_filtered_mask_files_in_folder, is_cell_mask_tiff
from bactomate.gui.layout_helpers import flex_control, module_path_action_buttons_layout, module_path_action_outer_layout, module_path_drop_row, new_module_path_action_panel, style_module_primary_button, vbox_from_rows
from bactomate.gui.module_shell import PreviewSurface, mount_embedded_preview
from bactomate.gui.preview.preview_window import CollapsibleSection, PreviewDialogBase, RenderResult
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET, MODULE_PANEL_COMBO_STYLESHEET
from bactomate.gui.trackastra_utils import TRACKASTRA_DEFAULT_MODEL, TRACKASTRA_SAM2_MODEL, format_trackastra_model_error, trackastra_pretrained_feats_installed, validate_trackastra_model_choice
from bactomate.gui.workers import TrackingPreviewResult, TrackingWorker
_TRACKING_PREVIEW_TIFF_CAP = 2500
TIME_UNIT_TO_MINUTES = {'min': 1.0, 'h': 60.0, 's': 1.0 / 60.0, 'ms': 1.0 / 60000.0}
_FILTER_TIP_DETECTIONS = 'Detections\nHow many frames have a linked detection (frame, y, x) for this track ID.\n\nExample: seen in 8 frames → detections = 8.\n\nPair with Duration and Gaps: gaps = duration − detections (missing frames inside the span).'
_FILTER_TIP_DURATION = 'Duration (frames)\nFrame span: last_frame − first_frame + 1 (counts empty frames in between).\n\nExample: first at frame 10, last at frame 49 → duration = 40 even with missed frames inside.\n\nUse with min detections and max gaps to drop flickering or sparse tracks.'
_FILTER_TIP_GAPS = 'Gaps (frames)\nMissing frames inside the track span: duration − detections.\n\nTrackastra kept one track ID across these gaps (linking), but the mask had no detection on those frames.\n\nExample: detections = 9, duration = 14 → gaps = 5.\n\nSet a low max gaps (e.g. 0–2) to require near-continuous presence. A cell that vanishes and reappears with a new ID is two tracks, not one track with gaps.'
_FILTER_TIP_DISPLACEMENT = 'Displacement\nStraight-line distance from the first to the last position along the track.\n\nUnits: µm if Pixel (µm/px) is set; otherwise pixels.\n\nExample: a cell drifts 15 µm from birth to exit → displacement ≈ 15 µm.\n\nA cell jittering in place has low displacement even if it moved a lot along its path (see Path length).'
_FILTER_TIP_PATH_LENGTH = 'Path length\nTotal distance travelled: sum of step lengths between consecutive positions.\n\nAlways ≥ displacement. Equal only for a perfectly straight path.\n\nExample A — wiggling in place: displacement ≈ 1 µm, path length ≈ 25 µm.\nExample B — straight swim: displacement ≈ 20 µm, path length ≈ 21 µm.\n\nFilter out “stuck” cells with low displacement but high path length, or fast swimmers with both high.'
_FILTER_TIP_MEAN_SPEED = 'Mean speed\nAverage speed along the track: path_length ÷ elapsed time.\n\nUnits: µm/s when both pixel size and Time / frame are set; otherwise px/frame.\n\nExample: path length 30 µm over 20 frames at 3 s/frame (60 s total) → mean speed ≈ 30 / 60 = 0.5 µm/s.\n\nRequires calibration for physical speed; set Time / frame to 0 to use px/frame only.'
_FILTER_TIP_STRAIGHTNESS = 'Min straightness\nHow straight the path is: displacement ÷ path_length (0–1).\n\n1.0 = perfectly straight; lower = more wandering or circling.\n\nExample: displacement 20 µm, path length 20 µm → straightness = 1.0.\nExample: displacement 2 µm, path length 40 µm → straightness = 0.05 (mostly jitter).\n\nUse a minimum (e.g. 0.3) to drop tracks that barely progress, or a high minimum to keep only directed motion.'


class SortableTrackingTableItem(QTableWidgetItem):
    """Numeric-aware table item for tracking/lineage QC tables."""

    def __init__(self, value: Any) -> None:
        super().__init__("" if value is None else str(value))
        self._sort_value = self._coerce_sort_value(value)

    @staticmethod
    def _coerce_sort_value(value: Any) -> tuple[int, Any]:
        try:
            if value is None:
                return (2, "")
            if isinstance(value, float) and np.isnan(value):
                return (2, "")
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, ("" if value is None else str(value)).lower())

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, SortableTrackingTableItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


def _fit_preview_metrics_table_columns(table: QTableWidget) -> None:
    """Size metrics columns proportionally to the table viewport (preview sidebar)."""
    hdr = table.horizontalHeader()
    n_cols = table.columnCount()
    if hdr is None or n_cols <= 0:
        return
    weights = list(METRICS_TABLE_COL_WEIGHTS[:n_cols])
    if len(weights) < n_cols:
        pad = weights[-1] if weights else 56
        weights.extend([pad] * (n_cols - len(weights)))
    total_weight = float(sum(weights)) or 1.0
    avail = int(table.viewport().width())
    if avail <= 0:
        vhdr = table.verticalHeader()
        vhdr_w = int(vhdr.width()) if vhdr is not None else 0
        avail = max(0, int(table.width()) - vhdr_w - 2 * int(table.frameWidth()))
    vbar = table.verticalScrollBar()
    if vbar is not None and vbar.isVisible():
        avail = max(0, avail - int(vbar.width()))
    avail = max(METRICS_TABLE_COL_MIN_WIDTH * n_cols, avail)
    hdr.setStretchLastSection(False)
    hdr.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    widths = [max(METRICS_TABLE_COL_MIN_WIDTH, int(avail * w / total_weight)) for w in weights]
    slack = avail - sum(widths)
    if slack and widths:
        widths[-1] = max(METRICS_TABLE_COL_MIN_WIDTH, widths[-1] + slack)
    elif slack < 0 and widths:
        widths[-1] = max(METRICS_TABLE_COL_MIN_WIDTH, widths[-1] + slack)
    for (col_i, w) in enumerate(widths):
        table.setColumnWidth(col_i, int(w))

class _MetricsTableColumnFitFilter(QObject):
    """Keep preview metrics columns fitted when the table or panel is resized."""

    def __init__(self, table: QTableWidget, *, host: QWidget | None=None) -> None:
        super().__init__(table)
        self._table = table
        self._host = host

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Resize and obj in (self._table, self._host):
            _fit_preview_metrics_table_columns(self._table)
        return super().eventFilter(obj, event)

class TrackingTab(QWidget):
    """Tab for cell tracking using trackastra"""
    workflow_step_finished = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget]=None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._performed_this_session = False
        self._module_preview_mount: Optional[QWidget] = None
        self._preview_surface: Optional[PreviewSurface] = None
        self._embedded_tracking_preview: Optional[QDialog] = None
        self._last_preview_in_memory = False
        self._last_tracking_preview_result: Optional[TrackingPreviewResult] = None
        self._tracking_candidate_timer = QTimer(self)
        self._tracking_candidate_timer.setSingleShot(True)
        self._tracking_candidate_timer.timeout.connect(self.refresh_tracking_candidate_list)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        scroll_widget.setMinimumWidth(0)
        scroll_widget.setStyleSheet('QWidget { font-size: 12pt; }QLabel { font-size: 12pt; }QLineEdit { font-size: 13pt; }QGroupBox { font-size: 12pt; font-weight: 600; }QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #aeb6c2; font-weight: 600; }QPushButton { font-size: 12pt; }QCheckBox { font-size: 12pt; }QSpinBox, QDoubleSpinBox { font-size: 12pt; }' + MODULE_PANEL_COMBO_STYLESHEET)
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(12)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        run_content = QWidget()
        run_inner = QVBoxLayout(run_content)
        run_inner.setContentsMargins(0, 0, 0, 0)
        run_inner.setSpacing(6)
        self._build_run_tracking_section(run_inner)
        scroll_layout.addWidget(CollapsibleSection('Run tracking', run_content, parent=self, expanded=True))
        lineage_content = QWidget()
        lineage_inner = QVBoxLayout(lineage_content)
        lineage_inner.setContentsMargins(0, 0, 0, 0)
        lineage_inner.setSpacing(6)
        self._build_run_lineage_tracking_section(lineage_inner)
        scroll_layout.addWidget(CollapsibleSection('Lineage tracking', lineage_content, parent=self, expanded=False))
        analysis_content = QWidget()
        analysis_inner = QVBoxLayout(analysis_content)
        analysis_inner.setContentsMargins(0, 0, 0, 0)
        analysis_inner.setSpacing(6)
        from bactomate.gui.lineage_analysis_panel import LineageAnalysisPanel
        self._lineage_analysis = LineageAnalysisPanel(self)
        analysis_inner.addWidget(self._lineage_analysis.path_bar())
        self._lineage_analysis_section = CollapsibleSection('Lineage analysis', analysis_content, parent=self, expanded=False)
        scroll_layout.addWidget(self._lineage_analysis_section)
        self._tracking_params_host = scroll_widget
        self._tracking_params_panel = self._build_tracking_params()
        self._filter_tracks_panel = self._build_filter_tracks_panel()
        self._tracking_params_panel.setParent(self._tracking_params_host)
        self._filter_tracks_panel.setParent(self._tracking_params_host)
        self._tracking_params_panel.hide()
        self._filter_tracks_panel.hide()
        scroll_layout.addStretch(1)
        self.file_path.textChanged.connect(self._update_tracking_button_state)
        self.file_path.textChanged.connect(self._update_filter_ranges_from_trajectory)
        self.file_path.textChanged.connect(self._schedule_tracking_candidate_refresh)
        self.lineage_file_path.textChanged.connect(self._update_lineage_tracking_button_state)
        self._update_tracking_button_state()
        self._update_lineage_tracking_button_state()
        QTimer.singleShot(0, self._update_filter_ranges_from_trajectory)
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        self._tracking_left_scroll = scroll
        layout.addWidget(scroll)
        self.setLayout(layout)

    @staticmethod
    def _qt_widget_alive(widget: QWidget | None) -> bool:
        if widget is None:
            return False
        try:
            from shiboken6 import isValid
            return bool(isValid(widget))
        except Exception:
            try:
                widget.objectName()
                return True
            except RuntimeError:
                return False

    def _restore_tracking_preview_panels(self) -> None:
        """Reparent shared parameter panels before the preview dialog is destroyed."""
        host = getattr(self, '_tracking_params_host', None) or self
        for panel in (getattr(self, '_tracking_params_panel', None), getattr(self, '_filter_tracks_panel', None)):
            if not self._qt_widget_alive(panel):
                continue
            try:
                panel.setParent(host)
                panel.hide()
            except RuntimeError:
                pass

    def _linking_mode_text(self, default: str='greedy') -> str:
        combo = getattr(self, 'mode_combo', None)
        if not self._qt_widget_alive(combo):
            return default
        try:
            return (combo.currentText() or default).strip()
        except RuntimeError:
            return default

    def _set_linking_mode_text(self, mode: str) -> None:
        combo = getattr(self, 'mode_combo', None)
        if not self._qt_widget_alive(combo):
            return
        try:
            combo.setCurrentText(mode)
        except RuntimeError:
            pass

    def _main_window(self) -> QWidget:
        """Return the main window so dialogs parent to it (avoids second dock/taskbar icon)."""
        if hasattr(self, 'window') and callable(self.window):
            w = self.window()
            if w is not None:
                return w
        return getattr(self, 'parent_window', self)

    def _reveal_lineage_analysis(self) -> None:
        """Expand Lineage analysis and scroll it into view after a successful run."""
        section = getattr(self, '_lineage_analysis_section', None)
        if section is not None:
            section.set_expanded(True)
        QTimer.singleShot(0, self._scroll_lineage_analysis_into_view)

    def _scroll_lineage_analysis_into_view(self) -> None:
        section = getattr(self, '_lineage_analysis_section', None)
        scroll = getattr(self, '_tracking_left_scroll', None)
        if section is None or scroll is None:
            return
        try:
            scroll.ensureWidgetVisible(section, 24, 24)
            if section.is_expanded():
                content = section.content()
                if content is not None:
                    scroll.ensureWidgetVisible(content, 24, 24)
        except Exception:
            pass

    def attach_module_preview_mount(self, mount: Optional[QWidget]) -> None:
        """Wire module-shell right pane for embedded trajectory preview after Preview tracking."""
        self._module_preview_mount = mount
        self._preview_surface = None
        if mount is not None:
            parent = mount.parent()
            if isinstance(parent, PreviewSurface):
                self._preview_surface = parent
                parent.set_preview_context('tracking')
                parent.set_pick_callback(self._on_pick_mask_from_preview_list)
        analysis = getattr(self, '_lineage_analysis', None)
        if analysis is not None:
            analysis.attach_preview_mount(mount, self._preview_surface)
        self.refresh_tracking_candidate_list()
        if mount is not None:
            self._restore_last_tracking_preview()

    def _restore_last_tracking_preview(self) -> None:
        """Recreate the last in-memory preview after returning to the Tracking module."""
        result = getattr(self, '_last_tracking_preview_result', None)
        if result is None:
            return
        if getattr(self, '_embedded_tracking_preview', None) is not None:
            return
        if getattr(self, '_module_preview_mount', None) is None:
            return
        try:
            self._show_tracking_preview_result(result)
        except RuntimeError:
            return

    def _schedule_tracking_candidate_refresh(self, _text: str='') -> None:
        self._tracking_candidate_timer.start(200)

    def refresh_tracking_candidate_list(self, *, selected: Optional[str]=None) -> None:
        """Populate the right-hand mask dropdown from the current path."""
        surf = self._preview_surface
        if surf is None:
            return
        raw = (self.file_path.text() or '').strip()
        if not raw:
            surf.set_candidate_files([])
            return
        po = Path(raw)
        if not po.exists():
            surf.set_candidate_files([])
            return
        pick: Optional[str] = selected
        if po.is_dir():
            paths = self._collect_mask_files(po)[:_TRACKING_PREVIEW_TIFF_CAP]
            paths_str = [str(x) for x in paths]
            embedded_open = getattr(self, '_embedded_tracking_preview', None) is not None
            if pick is None and embedded_open and paths_str:
                pick = paths_str[0]
            surf.set_candidate_files(paths_str, selected=pick, advance_placeholder=embedded_open)
            return
        if po.is_file():
            parent = po.parent
            if parent.is_dir():
                paths = self._collect_mask_files(parent)[:_TRACKING_PREVIEW_TIFF_CAP]
                paths_str = [str(x) for x in paths]
                if is_cell_mask_tiff(po):
                    try:
                        default_sel = str(po.resolve())
                    except OSError:
                        default_sel = str(po)
                else:
                    match = self._find_mask_for_original(po, paths)
                    default_sel = str(match.resolve()) if match else paths_str[0] if paths_str else None
                surf.set_candidate_files(paths_str, selected=pick or default_sel)
            else:
                try:
                    one = str(po.resolve())
                except OSError:
                    one = str(po)
                surf.set_candidate_files([one], selected=pick or one)

    def _on_pick_mask_from_preview_list(self, path: str) -> None:
        """When user picks a mask from the module-shell dropdown, sync the path field."""
        if not path:
            return
        edit = self.file_path
        if edit.text() == path:
            return
        edit.blockSignals(True)
        try:
            edit.setText(path)
        finally:
            edit.blockSignals(False)
        self._update_tracking_button_state()
        self._update_lineage_tracking_button_state()
        self._update_filter_ranges_from_trajectory()

    def _dismiss_embedded_tracking_preview(self) -> None:
        prev = getattr(self, '_embedded_tracking_preview', None)
        if prev is not None:
            self._restore_tracking_preview_panels()
            try:
                prev.close()
                prev.deleteLater()
            except Exception:
                pass
            self._embedded_tracking_preview = None
        surf = self._preview_surface
        if surf is not None:
            try:
                surf.clear_mount()
            except Exception:
                pass

    def _build_run_tracking_section(self, parent_layout: QVBoxLayout) -> None:
        """Path row + preview / run actions (same bordered panel as segmentation)."""
        file_font = QFont()
        file_font.setPointSize(12)
        path_panel = new_module_path_action_panel(self)
        path_lay = module_path_action_outer_layout(path_panel)
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText('Select mask TIFF file or folder… (or drag and drop)')
        self.file_path.setMinimumWidth(0)
        self.file_path.setFont(file_font)
        self.file_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        file_btn = QPushButton('Browse File...')
        file_btn.clicked.connect(self.select_mask_file)
        folder_btn = QPushButton('Browse Folder...')
        folder_btn.clicked.connect(self.select_folder)
        path_lay.addWidget(module_path_drop_row(path_panel, self.handle_dropped_path, line_edit=self.file_path, buttons=(file_btn, folder_btn), info_icon=create_info_icon(f'Path to a mask TIFF, original TIFF, or parent folder.\n\nMasks (*_masks.tif, *_mask_filtered.tif) are tracked; matching phase TIFFs are loaded automatically for Trackastra and calibration. Folders are scanned for _segmentation_* and filtered_masks/ subfolders.\n\nFull runs write to _tracking/{{stem}}_tracking/. Preview tracking uses the first {TRACKING_PREVIEW_FRAME_LIMIT} frames only, in memory (nothing written to disk).', self)))
        actions_layout = module_path_action_buttons_layout()
        self._preview_btn_idle = 'Preview tracking'
        self._movement_btn_idle = 'Run tracking'
        self._tracking_busy_action: Optional[str] = None
        self.preview_btn = ProgressCancelButton(self._preview_btn_idle)
        self.preview_btn.setEnabled(False)
        style_module_primary_button(self.preview_btn)
        self.preview_btn.setToolTip(f'Opens a setup dialog: pick the mask and either a quick preview (first {TRACKING_PREVIEW_FRAME_LIMIT} frames) or the full time lapse.\n\nIn-memory only (downsampled for speed); nothing written to disk.\n\nUse Run tracking for full-sequence exports to disk.')
        self.preview_btn.clicked.connect(lambda : self._run_tracking_with_action('preview'))
        self.movement_btn = ProgressCancelButton(self._movement_btn_idle)
        self.movement_btn.setEnabled(False)
        style_module_primary_button(self.movement_btn)
        self.movement_btn.setToolTip('Track the full mask sequence with Trackastra (movement / linking only).\n\nWrites CSV, tracked masks, and report workbook under _tracking/{stem}_tracking/.\n\nWhen Create visualization is checked, also builds per-frame overlay and *_trajectories.tif (slow on long movies; this is not done during preview).\n\nFor divisions and lineage tables, use Run lineage tracking below.')
        self.movement_btn.clicked.connect(lambda : self._run_tracking_with_action('movement'))
        for btn in (self.preview_btn, self.movement_btn):
            actions_layout.addWidget(btn)
        path_lay.addLayout(actions_layout)
        parent_layout.addWidget(path_panel)

    def _build_run_lineage_tracking_section(self, parent_layout: QVBoxLayout) -> None:
        """Path row + preview / run lineage actions (same layout as Run tracking)."""
        file_font = QFont()
        file_font.setPointSize(12)
        path_panel = new_module_path_action_panel(self)
        path_lay = module_path_action_outer_layout(path_panel)
        self.lineage_file_path = QLineEdit()
        self.lineage_file_path.setPlaceholderText('Select mask TIFF file or folder… (or drag and drop)')
        self.lineage_file_path.setMinimumWidth(0)
        self.lineage_file_path.setFont(file_font)
        self.lineage_file_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lineage_file_btn = QPushButton('Browse File...')
        lineage_file_btn.clicked.connect(self.select_lineage_mask_file)
        lineage_folder_btn = QPushButton('Browse Folder...')
        lineage_folder_btn.clicked.connect(self.select_lineage_folder)
        path_lay.addWidget(module_path_drop_row(path_panel, self.handle_lineage_dropped_path, line_edit=self.lineage_file_path, buttons=(lineage_file_btn, lineage_folder_btn), info_icon=create_info_icon('Path to a mask TIFF, original TIFF, or parent folder for lineage tracking.\n\nSame mask discovery as Run tracking (_segmentation_*, filtered_masks/, etc.).\n\nExports lineage edges, summary workbook, and optional lineage TIF under _tracking/{stem}_tracking/. greedy_nodiv is switched to greedy automatically.', self)))
        actions_layout = module_path_action_buttons_layout()
        self._preview_lineage_btn_idle = 'Preview lineage QC'
        self._lineage_btn_idle = 'Run lineage tracking'
        self.preview_lineage_btn = ProgressCancelButton(self._preview_lineage_btn_idle)
        self.preview_lineage_btn.setEnabled(False)
        style_module_primary_button(self.preview_lineage_btn)
        self.preview_lineage_btn.setToolTip(f'Quick in-memory lineage QC: founder-colored overlay, division candidates, and a compact review panel (greedy_nodiv is switched to greedy).\n\nPick mask and quick ({TRACKING_PREVIEW_FRAME_LIMIT} frames) or full time lapse; nothing written to disk. Use Run lineage tracking for the full Excel/report.')
        self.preview_lineage_btn.clicked.connect(lambda : self._run_tracking_with_action('preview_lineage'))
        self.lineage_btn = ProgressCancelButton(self._lineage_btn_idle)
        self.lineage_btn.setEnabled(False)
        style_module_primary_button(self.lineage_btn)
        self.lineage_btn.setToolTip('Full lineage tracking: tables, Excel report, lineage TIF, and PNG figures.\n\nIf Mode is greedy_nodiv, it is switched to greedy so divisions can be recorded.\n\nOutputs live under _tracking/{stem}_tracking/ alongside trajectory files.')
        self.lineage_btn.clicked.connect(lambda : self._run_tracking_with_action('lineage'))
        for btn in (self.preview_lineage_btn, self.lineage_btn):
            actions_layout.addWidget(btn)
        path_lay.addLayout(actions_layout)
        parent_layout.addWidget(path_panel)

    def _build_filter_tracks_panel(self) -> QWidget:
        """Calibration + filter ranges (2 columns, one compact row per metric)."""
        filter_wrap = QWidget()
        grid = QGridLayout(filter_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(FILTER_GRID_H_SPACING)
        grid.setVerticalSpacing(FILTER_GRID_V_SPACING)
        for col in range(FILTER_GRID_COLUMNS):
            grid.setColumnStretch(col, 1)
        self.filter_pixel_size = QDoubleSpinBox()
        self.filter_pixel_size.setRange(0.0, 10.0)
        self.filter_pixel_size.setDecimals(3)
        self.filter_pixel_size.setValue(0.0)
        self.filter_pixel_size.setSingleStep(0.001)
        self.filter_pixel_size.setSpecialValueText('px only')
        self.filter_pixel_size.setMinimum(0.0)
        flex_control(self.filter_pixel_size, FILTER_CAL_SPIN_WIDTH, combo_style=False)
        self.filter_time_per_frame = QDoubleSpinBox()
        self.filter_time_per_frame.setRange(0.0, 1000000.0)
        self.filter_time_per_frame.setDecimals(6)
        self.filter_time_per_frame.setSingleStep(1e-06)
        self.filter_time_per_frame.setValue(0.0)
        self.filter_time_per_frame.setSpecialValueText('frames only')
        self.filter_time_per_frame.setMinimum(0.0)
        flex_control(self.filter_time_per_frame, FILTER_CAL_TIME_SPIN_WIDTH, combo_style=False)
        self.filter_time_per_frame_unit = QComboBox()
        self.filter_time_per_frame_unit.addItems(['min', 'h', 's', 'ms'])
        self.filter_time_per_frame_unit.setCurrentText('min')
        flex_control(self.filter_time_per_frame_unit, 76)
        self.create_vis_cb = QCheckBox('Create visualization')
        self.create_vis_cb.setChecked(True)
        self.create_vis_cb.setVisible(False)
        self.create_vis_cb.setToolTip(f'Full Run tracking only: build *_overlay_tracked.tif and *_trajectories.tif when a phase stack is found (per-frame overlay with labels — slow on long movies).\n\nUncheck to skip visualization and save only tracks CSV, tracked masks, and the report workbook.\n\nPreview tracking never writes files and uses a separate light in-memory overlay (first {TRACKING_PREVIEW_FRAME_LIMIT} frames only).')
        self.overlay_track_data_cb = QCheckBox('Metrics table')
        self.overlay_track_data_cb.setChecked(False)
        self.overlay_track_data_cb.setToolTip('Preview only: table of whole-track metrics for every track in the preview run.\n\nOpens below Filter tracks in the preview panel (the settings area scrolls to it).\n\nPass and other columns update immediately when you change filters (no image redraw). Use «Redraw filtered tracks» under the frame slider to refresh the trajectory image.\n\nDouble-click a row to focus that track (yellow path + cell highlight).')
        cal_host = QWidget()
        cal_line = QHBoxLayout(cal_host)
        cal_line.setContentsMargins(0, 0, 0, 0)
        cal_line.setSpacing(6)
        cal_line.addWidget(QLabel('Pixel (µm/px):'))
        cal_line.addWidget(self.filter_pixel_size)
        cal_line.addWidget(QLabel('Time / frame:'))
        cal_line.addWidget(self.filter_time_per_frame)
        cal_line.addWidget(self.filter_time_per_frame_unit)
        cal_line.addWidget(self.create_vis_cb)
        cal_line.addWidget(self.overlay_track_data_cb)
        cal_line.addWidget(create_info_icon('Calibration for physical filters (filled from mask/original TIFF metadata when available). Pixel 0 = «px only». Time / frame 0 = «frames only» (no µm/s). Create visualization: disk output for Run tracking. Metrics table: per-track stats in the preview panel (scrollable).', self, compact=True))
        cal_line.addStretch(1)
        grid.addWidget(cal_host, 0, 0, 1, FILTER_GRID_COLUMNS)

        def _metric_spin(spin: QDoubleSpinBox | QSpinBox) -> None:
            if isinstance(spin, QDoubleSpinBox) and spin.decimals() > 0:
                w = FILTER_METRIC_DEC_SPIN_WIDTH
            else:
                w = FILTER_METRIC_SPIN_WIDTH
            flex_control(spin, w, combo_style=False)
        self._filter_metric_title_labels: dict[str, QLabel] = {}

        def _metric_row_widget() -> tuple[QHBoxLayout, QWidget]:
            lay = QHBoxLayout()
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(4)
            w = QWidget()
            w.setLayout(lay)
            w.setMinimumHeight(FILTER_METRIC_CELL_MIN_HEIGHT)
            w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return (lay, w)

        def _range_cell(title: str, min_spin: QDoubleSpinBox | QSpinBox, max_spin: QDoubleSpinBox | QSpinBox, min_default: float | int, max_default: float | int, tip: str, *, label_key: str | None=None) -> QWidget:
            min_spin.setMinimum(0)
            min_spin.setMaximum(999999)
            min_spin.setValue(min_default)
            _metric_spin(min_spin)
            max_spin.setMinimum(0)
            max_spin.setMaximum(999999)
            max_spin.setValue(max_default)
            _metric_spin(max_spin)
            (lay, w) = _metric_row_widget()
            title_lbl = QLabel(title)
            title_lbl.setWordWrap(False)
            if label_key:
                self._filter_metric_title_labels[label_key] = title_lbl
            lay.addWidget(title_lbl, 0)
            lay.addSpacing(4)
            lay.addWidget(min_spin, 0)
            dash = QLabel('–')
            dash.setFixedWidth(10)
            dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(dash, 0)
            lay.addWidget(max_spin, 0)
            lay.addWidget(create_info_icon(tip, self, compact=True), 0)
            return w

        def _single_cell(title: str, spin: QDoubleSpinBox, default: float, tip: str, *, label_key: str | None=None) -> QWidget:
            spin.setMinimum(0)
            spin.setMaximum(999999)
            spin.setValue(default)
            _metric_spin(spin)
            (lay, w) = _metric_row_widget()
            title_lbl = QLabel(title)
            if label_key:
                self._filter_metric_title_labels[label_key] = title_lbl
            lay.addWidget(title_lbl, 0)
            lay.addSpacing(4)
            lay.addWidget(spin, 0)
            lay.addWidget(create_info_icon(tip, self, compact=True), 0)
            return w
        self.filter_min_length = QSpinBox()
        self.filter_max_length = QSpinBox()
        self.filter_min_duration = QSpinBox()
        self.filter_max_duration = QSpinBox()
        self.filter_min_gaps = QSpinBox()
        self.filter_max_gaps = QSpinBox()
        self.filter_min_displacement = QDoubleSpinBox()
        self.filter_max_displacement = QDoubleSpinBox()
        self.filter_min_displacement.setDecimals(0)
        self.filter_max_displacement.setDecimals(0)
        self.filter_min_path_length = QDoubleSpinBox()
        self.filter_max_path_length = QDoubleSpinBox()
        self.filter_min_path_length.setDecimals(0)
        self.filter_max_path_length.setDecimals(0)
        self.filter_min_speed = QDoubleSpinBox()
        self.filter_max_speed = QDoubleSpinBox()
        self.filter_min_speed.setDecimals(2)
        self.filter_max_speed.setDecimals(2)
        self.filter_min_straightness = QDoubleSpinBox()
        self.filter_min_straightness.setRange(0.0, 1.0)
        self.filter_min_straightness.setDecimals(2)
        self.filter_min_straightness.setSingleStep(0.05)
        self.filter_min_straightness.setValue(0.0)
        cells = [_range_cell('Detections', self.filter_min_length, self.filter_max_length, 2, 100, _FILTER_TIP_DETECTIONS, label_key='detections'), _range_cell('Duration', self.filter_min_duration, self.filter_max_duration, 1, 100, _FILTER_TIP_DURATION, label_key='duration'), _range_cell('Gaps', self.filter_min_gaps, self.filter_max_gaps, 0, 9999, _FILTER_TIP_GAPS, label_key='gaps'), _range_cell('Displacement', self.filter_min_displacement, self.filter_max_displacement, 0, 100, _FILTER_TIP_DISPLACEMENT, label_key='displacement'), _range_cell('Path length', self.filter_min_path_length, self.filter_max_path_length, 0, 100, _FILTER_TIP_PATH_LENGTH, label_key='path_length'), _range_cell('Mean speed', self.filter_min_speed, self.filter_max_speed, 0, 100, _FILTER_TIP_MEAN_SPEED, label_key='mean_speed'), _single_cell('Straight. ≥', self.filter_min_straightness, 0.0, _FILTER_TIP_STRAIGHTNESS, label_key='straightness')]
        cols = FILTER_GRID_COLUMNS
        for (i, cell) in enumerate(cells):
            grid.addWidget(cell, 1 + i // cols, i % cols)
        self._update_filter_metric_unit_labels()
        self.filter_pixel_size.valueChanged.connect(self._update_filter_metric_unit_labels)
        self.filter_time_per_frame.valueChanged.connect(self._update_filter_metric_unit_labels)
        self.filter_time_per_frame_unit.currentTextChanged.connect(lambda _t: self._update_filter_metric_unit_labels())
        return filter_wrap

    def _update_filter_metric_unit_labels(self) -> None:
        """Refresh filter row titles when calibration toggles µm vs px units."""
        labels = getattr(self, '_filter_metric_title_labels', None)
        if not labels:
            return
        use_um = getattr(self, 'filter_pixel_size', None) is not None and float(self.filter_pixel_size.value()) > 0
        use_speed_physical = False
        if hasattr(self, 'filter_min_speed'):
            params = self._get_filter_params()
            use_speed_physical = bool(params.get('use_speed_physical'))
        unit_suffix_um = 'µm' if use_um else 'px'
        speed_suffix = 'µm/s' if use_speed_physical else 'px/fr'
        unit_map = {'detections': 'Detections (#)', 'duration': 'Duration (frames)', 'gaps': 'Gaps (frames)', 'displacement': f'Displacement ({unit_suffix_um})', 'path_length': f'Path length ({unit_suffix_um})', 'mean_speed': f'Mean speed ({speed_suffix})', 'straightness': 'Min straightness (≥)'}
        for (key, text) in unit_map.items():
            lbl = labels.get(key)
            if lbl is not None:
                lbl.setText(text)

    def _get_filter_params(self) -> dict[str, Any]:
        """Current calibration and filter parameters from widgets for track filtering.
        Time per frame is converted to minutes using the unit combo."""
        pixel = getattr(self, 'filter_pixel_size', None)
        time_ = getattr(self, 'filter_time_per_frame', None)
        time_unit = getattr(self, 'filter_time_per_frame_unit', None)
        pixel_size = float(pixel.value()) if pixel is not None else 0.0
        time_val = float(time_.value()) if time_ is not None else 0.0
        unit_key = time_unit.currentText() if time_unit is not None else 'min'
        time_per_frame = time_val * TIME_UNIT_TO_MINUTES.get(unit_key, 1.0)
        params = {'pixel_size': pixel_size, 'time_per_frame': time_per_frame, 'use_physical_units': pixel_size > 0, 'use_speed_physical': pixel_size > 0 and time_per_frame > 0, 'min_length': getattr(self, 'filter_min_length', None) and self.filter_min_length.value(), 'max_length': getattr(self, 'filter_max_length', None) and self.filter_max_length.value(), 'min_duration': getattr(self, 'filter_min_duration', None) and self.filter_min_duration.value(), 'max_duration': getattr(self, 'filter_max_duration', None) and self.filter_max_duration.value(), 'min_gaps': getattr(self, 'filter_min_gaps', None) and self.filter_min_gaps.value(), 'max_gaps': getattr(self, 'filter_max_gaps', None) and self.filter_max_gaps.value(), 'min_displacement': getattr(self, 'filter_min_displacement', None) and self.filter_min_displacement.value(), 'max_displacement': getattr(self, 'filter_max_displacement', None) and self.filter_max_displacement.value(), 'min_path_length': getattr(self, 'filter_min_path_length', None) and self.filter_min_path_length.value(), 'max_path_length': getattr(self, 'filter_max_path_length', None) and self.filter_max_path_length.value(), 'min_speed': getattr(self, 'filter_min_speed', None) and self.filter_min_speed.value(), 'max_speed': getattr(self, 'filter_max_speed', None) and self.filter_max_speed.value(), 'min_straightness': getattr(self, 'filter_min_straightness', None) and self.filter_min_straightness.value()}
        return params

    def _build_tracking_run_metadata(self, mask_file: Path, *, model_name: str, mode: str, device: str, track_action: str, original_file: str | None, excel_file: str | None, frame_limit: int | None) -> dict[str, Any]:
        from datetime import datetime, timezone
        params = self._get_filter_params()
        return {'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'mask_file': str(Path(mask_file).resolve()), 'original_file': str(Path(original_file).resolve()) if original_file else '', 'analysis_excel': str(Path(excel_file).resolve()) if excel_file else '', 'trackastra_model': model_name, 'linking_mode': mode, 'device': device, 'track_action': track_action, 'frame_limit': frame_limit if frame_limit is not None else '', 'preview_frame_cap': TRACKING_PREVIEW_FRAME_LIMIT, 'create_visualization': bool(getattr(self, 'create_vis_cb', None) and self.create_vis_cb.isChecked()), 'filter_params': params}

    @staticmethod
    def _preview_display_xy(y: float, x: float, frame_shape: tuple[int, ...]) -> tuple[float, float]:
        """Map track (y, x) in preview frame space to displayed image coordinates."""
        from bactomate.gui.image_utils import PREVIEW_MAX_SIZE
        (h, w) = (int(frame_shape[0]), int(frame_shape[1]))
        if max(h, w) <= PREVIEW_MAX_SIZE:
            return (float(x), float(y))
        scale = PREVIEW_MAX_SIZE / max(h, w)
        return (float(x) * scale, float(y) * scale)

    def _build_tracking_params(self) -> QWidget:
        """Model + mode on one row (preview uses TRACKING_PREVIEW_FRAME_LIMIT frames)."""
        tracking_group = QGroupBox('')
        tracking_group.setFlat(True)
        tracking_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        row = QHBoxLayout()
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        model_label = QLabel('Model:')
        model_label.setMinimumHeight(34)
        row.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(['general_2d', 'general_2d_w_SAM2_features'])
        self.model_combo.setCurrentText('general_2d')
        self.model_combo.setMinimumHeight(34)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self.model_combo, 1)
        model_info = create_info_icon("Pretrained Trackastra model for cell tracking.\n\n- general_2d: default, fast general-purpose 2D tracking (always available)\n- general_2d_w_SAM2_features: optional SAM2 features — requires pip install 'trackastra[etultra]' in your BactoMate venv; slower, often better on bacteria", parent_widget=self, compact=True)
        row.addWidget(model_info)
        mode_label = QLabel('Mode:')
        mode_label.setMinimumHeight(34)
        row.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['greedy', 'greedy_nodiv', 'ilp'])
        self.mode_combo.setCurrentText('greedy')
        self.mode_combo.setMinimumHeight(34)
        self.mode_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self.mode_combo, 1)
        mode_info = create_info_icon('greedy: recommended. greedy_nodiv: no divisions (lineage auto-switches). ilp: global linking, slower, needs Gurobi or SCIP.', parent_widget=self, compact=True)
        row.addWidget(mode_info)
        row.addStretch(0)
        wrap = QWidget()
        wrap.setLayout(row)
        wrap.setMinimumHeight(54)
        outer = QVBoxLayout(tracking_group)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.addWidget(wrap)
        self._refresh_trackastra_model_combo()
        return tracking_group

    def _refresh_trackastra_model_combo(self) -> None:
        """Keep SAM2 model in the list; annotate if the optional dependency is missing."""
        if not hasattr(self, 'model_combo'):
            return
        sam2_ok = trackastra_pretrained_feats_installed()
        labels = [TRACKASTRA_DEFAULT_MODEL]
        if sam2_ok:
            labels.append(TRACKASTRA_SAM2_MODEL)
        else:
            labels.append(f'{TRACKASTRA_SAM2_MODEL} (not installed)')
        current = self.model_combo.currentText().split(' (')[0].strip()
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            self.model_combo.addItems(labels)
            if current == TRACKASTRA_SAM2_MODEL and (not sam2_ok):
                self.model_combo.setCurrentIndex(0)
            else:
                idx = self.model_combo.findText(current)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
                elif current == TRACKASTRA_SAM2_MODEL and sam2_ok:
                    idx = self.model_combo.findText(TRACKASTRA_SAM2_MODEL)
                    if idx >= 0:
                        self.model_combo.setCurrentIndex(idx)
        finally:
            self.model_combo.blockSignals(False)

    def _selected_trackastra_model(self) -> str:
        combo = getattr(self, 'model_combo', None)
        if not self._qt_widget_alive(combo):
            return TRACKASTRA_DEFAULT_MODEL
        try:
            text = (combo.currentText() or TRACKASTRA_DEFAULT_MODEL).strip()
        except RuntimeError:
            return TRACKASTRA_DEFAULT_MODEL
        return text.split(' (')[0].strip()

    def _ensure_trackastra_model_available(self, model_name: str, *, title: str='Tracking') -> bool:
        (ok, err) = validate_trackastra_model_choice(model_name)
        if ok:
            return True
        show_message_dialog(self._main_window(), title, err or 'Model not available.', kind='warning')
        return False

    def _resolve_device(self) -> str:
        """Prefer CUDA, then Apple MPS, else CPU (no user toggle)."""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
        return 'cpu'

    @staticmethod
    def _legacy_outputs_enabled() -> bool:
        """Legacy tabular artifacts are emitted only in legacy/both profile."""
        profile = str(os.getenv('BACTOMATE_TRACKING_OUTPUT_PROFILE', 'minimal')).strip().lower()
        return profile in {'legacy', 'both'}

    def _get_trajectory_files(self) -> list[Path]:
        """Return existing *_trajectories.tif under _tracking and _preview for the current Path."""
        raw = self.file_path.text() and self.file_path.text().strip()
        if not raw:
            return []
        p = Path(raw).resolve()
        if not p.exists():
            return []
        if p.is_file():
            base_dir = base_dir_for_mask(p)
            stem = p.stem
            found: list[Path] = []
            for folder_name in ('_tracking', '_preview'):
                root = base_dir / folder_name
                if not root.is_dir():
                    continue
                for sub in root.iterdir():
                    if sub.is_dir():
                        t = sub / f'{stem}_trajectories.tif'
                        if t.is_file():
                            found.append(t)
            return sorted(found)
        base = p.parent if p.name.startswith('_segmentation') else p
        found: list[Path] = []
        for folder_name in ('_tracking', '_preview'):
            root = base / folder_name
            if root.is_dir():
                found.extend(sorted(root.glob('*/*_trajectories.tif')))
        return sorted(set(found))

    def _get_one_trajectory_path(self) -> Optional[Path]:
        """Return one trajectory file path from the current Path's _tracking output, or None."""
        tracking = self._get_trajectory_files()
        return tracking[0] if tracking else None

    def _resolve_mask_path_for_calibration(self) -> Optional[Path]:
        """Mask TIFF used for filter calibration (single file, or first mask in a folder)."""
        raw = (self.file_path.text() or '').strip()
        if not raw:
            return None
        try:
            p = Path(raw).resolve()
        except OSError:
            p = Path(raw)
        if not p.exists():
            return None
        if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
            if is_cell_mask_tiff(p):
                return p
            masks = self._collect_mask_files(p.parent)
            return self._find_mask_for_original(p, masks)
        if p.is_dir():
            masks = self._collect_mask_files(p)
            return masks[0] if masks else None
        return None

    def _resolve_calibration_for_filters(self) -> tuple[float, float, Path | None, str, Path | None]:
        """Pixel size (µm), loop time (min), metadata source path, source kind, resolved original."""
        mask_path = self._resolve_mask_path_for_calibration()
        if mask_path is not None:
            (ps, tp, src, kind) = resolve_calibration_for_mask(mask_path)
            original = resolve_original_for_mask(mask_path)
            if ps > 0 or tp > 0:
                return (ps, tp, src, kind, original)
            if original is None:
                return (0.0, 0.0, None, 'none', None)
            return (0.0, 0.0, None, 'none', original)
        traj = self._get_one_trajectory_path()
        if traj is not None:
            (ps, tp) = resolve_calibration_from_trajectories_path(traj)
            return (ps, tp, traj if ps > 0 or tp > 0 else None, 'trajectory' if ps > 0 or tp > 0 else 'none', None)
        parent = getattr(self, 'parent_window', None)
        sm = getattr(parent, 'shared_metadata', None) if parent is not None else None
        if isinstance(sm, dict):
            try:
                ps = float(sm.get('pixel_size') or 0)
            except (TypeError, ValueError):
                ps = 0.0
            try:
                tp = float(sm.get('time_per_frame_min') or 0)
            except (TypeError, ValueError):
                tp = 0.0
            return (ps, tp, None, 'shared' if ps > 0 or tp > 0 else 'none', None)
        return (0.0, 0.0, None, 'none', None)

    def _alert_calibration_metadata(self, mask_path: Path, *, kind: str, original: Path | None) -> None:
        """Warn when physical calibration is missing or only on the mask (not the phase original)."""
        key = str(mask_path.resolve())
        if getattr(self, '_calibration_alert_key', None) == key:
            return
        self._calibration_alert_key = key
        from bactomate.gui.analysis_paths import base_dir_for_mask
        base = base_dir_for_mask(mask_path)
        stem = stem_from_mask_path(mask_path)
        expected = [base / f'{stem}.tif', base.parent / f'{stem}.tif']
        searched = ', '.join((str(d) for d in [mask_path.parent, base, base.parent] if d.is_dir()))
        if kind == 'mask':
            body = f'Pixel size was read only from the mask file:\n{mask_path}\n\nThe original phase TIFF was not found (expected e.g. {expected[0]} or {expected[1]}).\n\nFilters and speeds in µm depend on the correct pixel size. Set Pixel (µm/px) manually or place the raw {stem}.tif where the pipeline can find it.'
        else:
            body = f'No pixel size or time-per-frame metadata was found for:\n{mask_path}\n\nSearched: {searched}\n'
            if original and original.is_file():
                body += f'\nFound phase image but it has no calibration tags:\n{original}\n'
            else:
                body += f'\nOriginal phase TIFF not found (expected e.g. {expected[0]} or {expected[1]}).\n'
            body += '\nPixel (µm/px) is set to «px only» (0). Enter calibration manually before using µm filters or µm/s speed.'
        show_message_dialog(self._main_window(), 'Calibration metadata missing', body, kind='warning')

    def _get_trajectory_frame_count(self, path: Path) -> Optional[int]:
        """Return number of frames in a trajectory TIFF, or None if unreadable."""
        counts: list[int] = []
        try:
            meta = read_metadata_from_path(path)
            n = meta.get('frame_count')
            if n is not None and int(n) > 0:
                counts.append(int(n))
        except Exception:
            pass
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                if tif.series:
                    series = tif.series[0]
                    shape = tuple((int(v) for v in getattr(series, 'shape', ()) or ()))
                    axes = str(getattr(series, 'axes', '') or '').upper()
                    if 'T' in axes and len(shape) > axes.index('T'):
                        counts.append(int(shape[axes.index('T')]))
                    elif len(shape) >= 3:
                        if axes.endswith('C') and shape[-1] in (3, 4):
                            counts.append(int(shape[0]) if len(shape) > 3 else 1)
                        else:
                            counts.append(int(shape[0]))
                if tif.pages:
                    counts.append(int(len(tif.pages)))
        except Exception:
            pass
        if counts:
            return max((c for c in counts if c > 0))
        return None

    def _update_filter_ranges_from_trajectory(self) -> None:
        """Update filter max length/duration and calibration from mask/original or trajectory metadata."""
        if not hasattr(self, 'filter_max_length') or not hasattr(self, 'filter_max_duration'):
            return
        raw = (self.file_path.text() or '').strip()
        if not raw:
            return
        frame_path = self._resolve_mask_path_for_calibration() or self._get_one_trajectory_path()
        if frame_path is not None:
            num_frames = self._get_trajectory_frame_count(frame_path)
            if num_frames is not None and num_frames > 0:
                for spin in (self.filter_max_length, self.filter_max_duration, self.filter_max_gaps):
                    spin.setMaximum(max(spin.minimum(), num_frames))
                    if spin.value() > num_frames:
                        spin.setValue(num_frames)
                if self.filter_min_length.value() > num_frames:
                    self.filter_min_length.setValue(max(0, num_frames - 1))
                if self.filter_min_duration.value() > num_frames:
                    self.filter_min_duration.setValue(max(1, num_frames))
                if self.filter_max_gaps.value() > num_frames:
                    self.filter_max_gaps.setValue(num_frames)
        (ps, tp_min, _src, kind, original) = self._resolve_calibration_for_filters()
        mask_path = self._resolve_mask_path_for_calibration()
        if ps > 0:
            self.filter_pixel_size.setValue(ps)
        elif mask_path is not None:
            self.filter_pixel_size.setValue(0.0)
        self._apply_filter_time_per_frame_from_metadata(tp_min)
        self._update_filter_metric_unit_labels()
        if mask_path is not None and (ps <= 0 or kind == 'mask'):
            self._alert_calibration_metadata(mask_path, kind=kind, original=original)
        elif ps > 0 and kind == 'original':
            self._calibration_alert_key = None
        self._update_run_button_idle_labels()

    def _apply_filter_time_per_frame_from_metadata(self, tp_min: float) -> None:
        """Set time-per-frame spin from metadata (minutes). Uses seconds when < 0.1 min for readability."""
        if tp_min <= 0 or not hasattr(self, 'filter_time_per_frame'):
            return
        spin = self.filter_time_per_frame
        unit = getattr(self, 'filter_time_per_frame_unit', None)
        if tp_min < 0.1:
            spin.setValue(tp_min * 60.0)
            if unit is not None:
                unit.setCurrentText('s')
        else:
            spin.setValue(tp_min)
            if unit is not None:
                unit.setCurrentText('min')

    def _path_edit_for_action(self, track_action: str) -> QLineEdit:
        if track_action in ('lineage', 'preview_lineage'):
            return self.lineage_file_path
        return self.file_path

    def _primary_mask_for_path_edit(self, path_edit: QLineEdit) -> Optional[Path]:
        """Best-effort mask path for frame-count labels from a path field."""
        raw = (path_edit.text() or '').strip()
        if not raw:
            return None
        p = Path(raw).resolve()
        if not p.exists():
            return None
        if p.is_file() and is_cell_mask_tiff(p):
            return p
        if p.is_file():
            masks = self._collect_mask_files(p.parent)
            return self._find_mask_for_original(p, masks)
        if p.is_dir():
            masks = self._collect_mask_files(p)
            return masks[0] if masks else None
        return None

    def _primary_mask_for_run_labels(self) -> Optional[Path]:
        """Best-effort mask path for frame-count labels on Run tracking buttons."""
        return self._primary_mask_for_path_edit(self.file_path)

    def _primary_lineage_mask_for_run_labels(self) -> Optional[Path]:
        """Best-effort mask path for frame-count labels on lineage buttons."""
        return self._primary_mask_for_path_edit(self.lineage_file_path)

    def _mask_stack_frame_count(self, mask_path: Path) -> Optional[int]:
        if not mask_path.is_file():
            return None
        return self._get_trajectory_frame_count(mask_path)

    def _prompt_preview_tracking_setup(self, mask_files: list[Path], *, dialog_title: str='Preview tracking', intro_text: str | None=None, preferred_path_edit: QLineEdit | None=None) -> tuple[Path, Optional[int]] | None:
        """Mask + frame scope for preview. Returns (mask_path, frame_limit); None limit = full stack."""
        from bactomate.gui.layout_helpers import style_compact_action_button
        if not mask_files:
            return None
        cap = TRACKING_PREVIEW_FRAME_LIMIT
        prefer_full = bool(getattr(self, '_preview_prefer_full_stack', False))
        dialog = QDialog(self._main_window())
        dialog.setWindowTitle(dialog_title)
        dialog.setMinimumSize(560, 460)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        lay = QVBoxLayout(dialog)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)
        intro = QLabel(intro_text or 'Track in memory only — results appear in the preview panel. Nothing is written to disk.\nChoose the mask and how many frames to include.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color: #dddddd; font-size: 11pt;')
        lay.addWidget(intro)
        lay.addWidget(QLabel('Mask file:'))
        file_list = QListWidget()
        file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        file_list.setMinimumHeight(140)
        preferred_row = 0
        current_mask = self._primary_mask_for_path_edit(preferred_path_edit or self.file_path)
        for (i, p) in enumerate(mask_files):
            mp = Path(p).resolve()
            n_fr = self._mask_stack_frame_count(mp)
            suffix = f' — {n_fr} frames' if n_fr else ''
            item = QListWidgetItem(f'{mp.name}{suffix}')
            item.setData(Qt.ItemDataRole.UserRole, mp)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            file_list.addItem(item)
            if current_mask is not None and mp == current_mask.resolve():
                preferred_row = i
        if file_list.count() > 0:
            file_list.item(preferred_row).setCheckState(Qt.CheckState.Checked)
        lay.addWidget(file_list, 1)
        scope_box = QGroupBox('Frames to track')
        scope_box.setFlat(True)
        scope_box.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        scope_lay = QVBoxLayout(scope_box)
        scope_lay.setContentsMargins(0, 6, 0, 0)
        scope_lay.setSpacing(6)
        radio_quick = QRadioButton()
        radio_full = QRadioButton()
        scope_group = QButtonGroup(dialog)
        scope_group.addButton(radio_quick)
        scope_group.addButton(radio_full)
        scope_lay.addWidget(radio_quick)
        scope_lay.addWidget(radio_full)
        note = QLabel('Quick preview is recommended for tuning filters. Full time lapse can take several minutes and uses more memory; the preview image is downsampled either way.')
        note.setWordWrap(True)
        note.setStyleSheet('color: #aaaaaa; font-size: 10pt;')
        scope_lay.addWidget(note)
        lay.addWidget(scope_box)

        def _selected_mask() -> Path | None:
            for i in range(file_list.count()):
                item = file_list.item(i)
                if item is not None and item.checkState() == Qt.CheckState.Checked:
                    return item.data(Qt.ItemDataRole.UserRole)
            return None

        def _on_mask_item_changed(changed: QListWidgetItem) -> None:
            if file_list.signalsBlocked():
                return
            if changed.checkState() != Qt.CheckState.Checked:
                if _selected_mask() is None:
                    file_list.blockSignals(True)
                    changed.setCheckState(Qt.CheckState.Checked)
                    file_list.blockSignals(False)
                _refresh_scope_labels()
                return
            file_list.blockSignals(True)
            for i in range(file_list.count()):
                other = file_list.item(i)
                if other is not None and other is not changed:
                    other.setCheckState(Qt.CheckState.Unchecked)
            file_list.blockSignals(False)
            _refresh_scope_labels()

        def _refresh_scope_labels() -> None:
            mp = _selected_mask()
            n_total = self._mask_stack_frame_count(mp) if mp is not None else None
            if n_total is not None and n_total > 0:
                n_quick = min(int(n_total), cap)
                radio_quick.setText(f"Quick preview — first {n_quick} of {n_total} frame{('s' if n_total != 1 else '')} (recommended)")
                radio_full.setText(f"Full time lapse — all {n_total} frame{('s' if n_total != 1 else '')}")
                radio_full.setEnabled(True)
            else:
                radio_quick.setText(f'Quick preview — first {cap} frames (recommended)')
                radio_full.setText('Full time lapse — entire mask stack')
                radio_full.setEnabled(True)
        file_list.itemChanged.connect(_on_mask_item_changed)
        if prefer_full:
            radio_full.setChecked(True)
        else:
            radio_quick.setChecked(True)
        _refresh_scope_labels()
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        style_dialog_button_box(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        lay.addWidget(button_box)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        mask_path = _selected_mask()
        if mask_path is None:
            return None
        n_total = self._mask_stack_frame_count(mask_path)
        use_full = radio_full.isChecked()
        self._preview_prefer_full_stack = use_full
        if use_full:
            if n_total is not None and n_total > TRACKING_PREVIEW_FULL_STACK_WARN_FRAMES:
                if not show_confirm_dialog(self._main_window(), title='Full time lapse preview', message=f'This mask has {n_total} frames. In-memory preview on the full stack can take a long time and use substantial RAM (stacks are downsampled for display, but tracking still runs on all frames).\n\nContinue with full time lapse preview?', confirm_label='Yes', cancel_label='No', default_confirm=False):
                    return None
            return (mask_path, None)
        return (mask_path, cap)

    def _update_run_button_idle_labels(self) -> None:
        """Set Preview / Run button text from mask frame count (idle state only)."""
        if getattr(self, '_tracking_busy_action', None):
            return
        mask = self._primary_mask_for_run_labels()
        n_total = self._mask_stack_frame_count(mask) if mask is not None else None
        self._preview_btn_idle = 'Preview tracking'
        if n_total is not None and n_total > 0:
            self._movement_btn_idle = f'Run tracking ({int(n_total)} frames)'
        else:
            self._movement_btn_idle = 'Run tracking'
        for (attr, idle) in (('preview_btn', self._preview_btn_idle), ('movement_btn', self._movement_btn_idle)):
            btn = getattr(self, attr, None)
            if isinstance(btn, ProgressCancelButton) and (not getattr(btn, '_progress_active', False)):
                btn.setText(idle)
        lineage_mask = self._primary_lineage_mask_for_run_labels()
        n_lineage = self._mask_stack_frame_count(lineage_mask) if lineage_mask is not None else None
        self._preview_lineage_btn_idle = 'Preview lineage QC'
        if n_lineage is not None and n_lineage > 0:
            self._lineage_btn_idle = f'Run lineage tracking ({int(n_lineage)} frames)'
        else:
            self._lineage_btn_idle = 'Run lineage tracking'
        for (attr, idle) in (('preview_lineage_btn', self._preview_lineage_btn_idle), ('lineage_btn', self._lineage_btn_idle)):
            btn = getattr(self, attr, None)
            if isinstance(btn, ProgressCancelButton) and (not getattr(btn, '_progress_active', False)):
                btn.setText(idle)

    def _set_path_action_buttons_enabled(self, path_edit: QLineEdit, buttons: tuple[Optional[ProgressCancelButton], ...]) -> None:
        raw = path_edit.text() and path_edit.text().strip()
        if not raw:
            for b in buttons:
                if b is not None:
                    b.setEnabled(False)
            return
        p = Path(raw).resolve()
        if not p.exists():
            for b in buttons:
                if b is not None:
                    b.setEnabled(False)
            return
        if p.is_file():
            ok = True
        else:
            ok = len(self._collect_mask_files(p)) > 0
        for b in buttons:
            if b is not None:
                b.setEnabled(ok)

    def _update_tracking_button_state(self) -> None:
        """Enable run buttons when path is a valid file or folder; refresh frame-count labels."""
        self._set_path_action_buttons_enabled(self.file_path, (getattr(self, 'preview_btn', None), getattr(self, 'movement_btn', None)))
        self._update_run_button_idle_labels()

    def _update_lineage_tracking_button_state(self) -> None:
        """Enable lineage preview/run when lineage path is valid."""
        self._set_path_action_buttons_enabled(self.lineage_file_path, (getattr(self, 'preview_lineage_btn', None), getattr(self, 'lineage_btn', None)))
        self._update_run_button_idle_labels()

    def select_mask_file(self):
        (path, _) = QFileDialog.getOpenFileName(self, 'Select Mask File', get_default_browse_dir(), 'TIFF Files (*.tif *.tiff);;All Files (*)')
        if path:
            self.file_path.setText(path)
            self._update_tracking_button_state()

    def select_folder(self):
        """Open folder dialog to select a folder (e.g. _segmentation_bact_phase_omni, _segmentation_unet)."""
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder Containing Mask Files', get_default_browse_dir())
        if folder_path:
            self.file_path.setText(folder_path)
            self._update_tracking_button_state()

    def select_lineage_mask_file(self) -> None:
        (path, _) = QFileDialog.getOpenFileName(self, 'Select Mask File for Lineage Tracking', get_default_browse_dir(), 'TIFF Files (*.tif *.tiff);;All Files (*)')
        if path:
            self.lineage_file_path.setText(path)
            self._update_lineage_tracking_button_state()

    def select_lineage_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder for Lineage Tracking', get_default_browse_dir())
        if folder_path:
            self.lineage_file_path.setText(folder_path)
            self._update_lineage_tracking_button_state()

    def handle_dropped_path(self, dropped_path: str) -> None:
        """Handle dropped file or folder path for tracking tab."""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ('.tif', '.tiff'):
            self.file_path.setText(dropped_path)
        elif path_obj.is_dir():
            self.file_path.setText(dropped_path)
        else:
            show_message_dialog(self, 'Invalid Drop', 'Please drop a TIFF mask file or a folder containing mask files.', kind='warning')
        self._update_tracking_button_state()

    def handle_lineage_dropped_path(self, dropped_path: str) -> None:
        """Handle dropped file or folder for lineage tracking path."""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ('.tif', '.tiff'):
            self.lineage_file_path.setText(dropped_path)
        elif path_obj.is_dir():
            self.lineage_file_path.setText(dropped_path)
        else:
            show_message_dialog(self, 'Invalid Drop', 'Please drop a TIFF mask file or a folder containing mask files.', kind='warning')
        self._update_lineage_tracking_button_state()

    def _is_folder_mode(self) -> bool:
        """Return True if the current path is a folder."""
        if not self.file_path.text() or not self.file_path.text().strip():
            return False
        p = Path(self.file_path.text())
        return p.exists() and p.is_dir()

    @staticmethod
    def _tracking_output_dir(mask_path: Path, track_action: str='movement', preview_frame_limit: Optional[int]=None) -> Path:
        """Full runs: ``_tracking/{stem}_tracking``. Preview: ``_preview/{stem}_n{N}`` (no _tracking folder)."""
        base = base_dir_for_mask(mask_path)
        stem = mask_path.stem
        if track_action == 'preview' and preview_frame_limit is not None and (int(preview_frame_limit) > 0):
            return base / '_preview' / f'{stem}_n{int(preview_frame_limit)}'
        return base / '_tracking' / f'{stem}_tracking'

    def _ensure_lineage_linking_mode(self) -> None:
        """greedy_nodiv cannot represent divisions; switch to greedy and notify."""
        if self._linking_mode_text() != 'greedy_nodiv':
            return
        self._set_linking_mode_text('greedy')
        show_message_dialog(self._main_window(), 'Lineage QC / export', 'greedy_nodiv does not model cell divisions. Mode was switched to greedy for this run.\n\nBoth greedy and ilp support divisions; ilp is often more accurate on difficult lineages but is slower and needs an ILP solver (e.g. Gurobi or SCIP).', kind='info')

    def _run_action_buttons(self) -> tuple[ProgressCancelButton, ...]:
        out: list[ProgressCancelButton] = []
        for name in ('preview_btn', 'movement_btn', 'preview_lineage_btn', 'lineage_btn'):
            btn = getattr(self, name, None)
            if isinstance(btn, ProgressCancelButton):
                out.append(btn)
        return tuple(out)

    def _set_run_buttons_busy(self, busy: bool, *, action: Optional[str]=None, progress_label: Optional[str]=None) -> None:
        """Disable run buttons while a worker runs; blue in-button fill on the active action."""
        idle_by_action = {'preview': getattr(self, '_preview_btn_idle', 'Preview tracking'), 'preview_lineage': getattr(self, '_preview_lineage_btn_idle', 'Preview lineage QC'), 'movement': getattr(self, '_movement_btn_idle', 'Run tracking (movement)'), 'lineage': getattr(self, '_lineage_btn_idle', 'Run lineage tracking')}
        btn_by_action = {'preview': getattr(self, 'preview_btn', None), 'preview_lineage': getattr(self, 'preview_lineage_btn', None), 'movement': getattr(self, 'movement_btn', None), 'lineage': getattr(self, 'lineage_btn', None)}
        if busy:
            self._tracking_busy_action = action
            for b in self._run_action_buttons():
                b.set_progress_active(False)
                b.setEnabled(False)
            active = btn_by_action.get(action or '')
            if isinstance(active, ProgressCancelButton):
                idle = idle_by_action.get(action or '', '')
                active.setText(progress_label or idle)
                active.set_progress_subtitle('')
                active.setEnabled(True)
                active.set_progress_active(True)
                active.set_progress(0)
            return
        self._tracking_busy_action = None
        for (act, btn) in btn_by_action.items():
            if isinstance(btn, ProgressCancelButton):
                btn.set_progress_active(False)
                btn.set_progress_subtitle('')
                btn.setText(idle_by_action.get(act, btn.text()))
        self._update_tracking_button_state()
        self._update_lineage_tracking_button_state()

    def _update_busy_action_progress(self, value: int, message: str='') -> None:
        """Update the active run button in-button progress fill (segmentation-style)."""
        import re
        action = getattr(self, '_tracking_busy_action', None)
        pct = max(0, min(100, int(value)))
        btn_by_action = {'preview': getattr(self, 'preview_btn', None), 'preview_lineage': getattr(self, 'preview_lineage_btn', None), 'movement': getattr(self, 'movement_btn', None), 'lineage': getattr(self, 'lineage_btn', None)}
        active = btn_by_action.get(action or '')
        if not isinstance(active, ProgressCancelButton):
            return
        if not getattr(active, '_progress_active', False):
            return
        active.set_progress(pct)
        if action in ('preview', 'preview_lineage', 'movement', 'lineage'):
            subtitle = ''
            m = re.search('(\\d+)\\s*/\\s*(\\d+)', message or '')
            msg_l = (message or '').lower()
            if m and ('visualization' in msg_l or 'overlay' in msg_l or 'creating overlay' in msg_l):
                subtitle = f'({m.group(1)}/{m.group(2)} frames)'
            active.set_progress_subtitle(subtitle)
        if action == 'movement':
            tasks = getattr(self, 'folder_tracking_tasks', None)
            if tasks and len(tasks) > 1:
                idx = getattr(self, 'current_task_index', 0) + 1
                active.setText(f'{self._movement_btn_idle}… ({idx}/{len(tasks)})')

    @classmethod
    def _collect_mask_files(cls, folder: Path) -> list[Path]:
        """Collect *_masks / *_mask_filtered TIFFs under a folder and typical pipeline subfolders."""
        from bactomate.gui.artifact_paths import discover_mask_tiffs_for_workspace
        return discover_mask_tiffs_for_workspace(Path(folder))

    @staticmethod
    def _find_mask_for_original(original: Path, candidates: list[Path]) -> Optional[Path]:
        """Best mask TIFF for a phase/original stack in the same sample folder."""
        stem = original.stem
        for mask in candidates:
            if stem_from_mask_path(mask) == stem:
                return mask
        return candidates[0] if len(candidates) == 1 else None

    def _mask_files_for_tracking_input(self, path: Path) -> list[Path]:
        """Mask files to offer for tracking/preview (never raw phase TIFFs)."""
        from bactomate.gui.artifact_paths import discover_mask_tiffs_for_workspace
        return discover_mask_tiffs_for_workspace(Path(path))

    @staticmethod
    def _resolve_original_for_mask(mask_path: Path) -> Path | None:
        """Find the original (phase) TIFF for this mask (delegates to tiff_metadata)."""
        return resolve_original_for_mask(mask_path)

    @staticmethod
    def _resolve_from_trajectories_path(traj_path: Path) -> tuple[Path | None, Path | None, Path | None]:
        """From a *_trajectories.tif path return (tracks_csv_path, overlay_path, original_path).
        Tracks CSV and overlay are in the same output dir; original (phase) is found from the mask."""
        if not traj_path.is_file():
            return (None, None, None)
        out_dir = traj_path.parent
        stem = traj_path.stem.replace('_trajectories', '')
        tracks_csv = out_dir / f'{stem}_tracks.csv'
        overlay_path = out_dir / f'{stem}_overlay_tracked.tif'
        mask_stem = stem
        search_bases = [out_dir.parent.parent, out_dir.parent]
        mask_path: Path | None = None
        for base in search_bases:
            if not base.is_dir():
                continue
            for f in base.iterdir():
                if f.is_file() and f.suffix.lower() in ('.tif', '.tiff') and (f.stem == mask_stem):
                    mask_path = f
                    break
            if mask_path is not None:
                break
            seg = base / '_segmentation'
            if seg.is_dir():
                for f in seg.iterdir():
                    if f.is_file() and f.suffix.lower() in ('.tif', '.tiff') and (f.stem == mask_stem):
                        mask_path = f
                        break
            if mask_path is not None:
                break
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                for f in sub.iterdir():
                    if f.is_file() and f.suffix.lower() in ('.tif', '.tiff') and (f.stem == mask_stem):
                        mask_path = f
                        break
                if mask_path is not None:
                    break
            if mask_path is not None:
                break
            for pattern in (f'*{mask_stem}.tif', f'*{mask_stem}.tiff'):
                for f in base.glob(pattern):
                    if f.is_file():
                        mask_path = f
                        break
                if mask_path is None:
                    for f in (base / '_segmentation').glob(pattern) if (base / '_segmentation').is_dir() else []:
                        if f.is_file():
                            mask_path = f
                            break
                if mask_path is not None:
                    break
            if mask_path is not None:
                break
        original_path = TrackingTab._resolve_original_for_mask(mask_path) if mask_path else None
        return (tracks_csv, overlay_path if overlay_path.is_file() else None, original_path)

    @staticmethod
    def _compute_track_metrics(tracks_df: pd.DataFrame, pixel_size_um: float, time_per_frame_min: float) -> pd.DataFrame:
        """Compute per-track metrics. Requires columns: track_id, frame, y, x.
        Returns a DataFrame with one row per track_id and columns:
        track_id, trajectory_length, duration_frames, gap_frames, duration_min, first_frame, last_frame,
        displacement_px, displacement_um, path_length_px, path_length_um,
        mean_speed_px_frame, mean_speed_um_s, straight_line_speed_px_frame, straight_line_speed_um_s, straightness."""
        if tracks_df.empty or 'track_id' not in tracks_df.columns or 'frame' not in tracks_df.columns:
            return _pd().DataFrame()
        need_xy = 'y' in tracks_df.columns and 'x' in tracks_df.columns
        rows = []
        for (tid, g) in tracks_df.groupby('track_id'):
            g = g.sort_values('frame')
            n = len(g)
            first_frame = int(g['frame'].iloc[0])
            last_frame = int(g['frame'].iloc[-1])
            duration = max(1, int(last_frame - first_frame + 1))
            elapsed_intervals = max(1, int(last_frame - first_frame))
            displacement_px = 0.0
            path_length_px = 0.0
            if need_xy and n >= 2:
                ys = g['y'].values
                xs = g['x'].values
                displacement_px = float(np.hypot(ys[-1] - ys[0], xs[-1] - xs[0]))
                path_length_px = float(np.sum(np.hypot(np.diff(ys), np.diff(xs))))
            displacement_um = displacement_px * pixel_size_um if pixel_size_um > 0 else displacement_px
            path_length_um = path_length_px * pixel_size_um if pixel_size_um > 0 else path_length_px
            path_safe = path_length_px if path_length_px > 0 else 1.0
            straightness = float(displacement_px / path_safe) if path_safe > 0 else 0.0
            mean_speed_px_frame = path_length_px / elapsed_intervals if elapsed_intervals > 0 else 0.0
            duration_sec = elapsed_intervals * time_per_frame_min * 60.0 if time_per_frame_min > 0 else 0.0
            if pixel_size_um > 0 and duration_sec > 0:
                mean_speed_um_s = path_length_um / duration_sec
            else:
                mean_speed_um_s = mean_speed_px_frame
            duration_min = duration * time_per_frame_min if time_per_frame_min > 0 else 0.0
            straight_line_speed_px_frame = displacement_px / elapsed_intervals if elapsed_intervals > 0 else 0.0
            if duration_sec > 0 and pixel_size_um > 0:
                straight_line_speed_um_s = displacement_um / duration_sec
            else:
                straight_line_speed_um_s = straight_line_speed_px_frame
            gap_frames = max(0, int(duration) - int(n))
            rows.append({'track_id': tid, 'trajectory_length': n, 'duration_frames': duration, 'gap_frames': gap_frames, 'duration_min': duration_min, 'first_frame': first_frame, 'last_frame': last_frame, 'displacement_px': displacement_px, 'displacement_um': displacement_um, 'path_length_px': path_length_px, 'path_length_um': path_length_um, 'mean_speed_px_frame': mean_speed_px_frame, 'mean_speed_um_s': mean_speed_um_s, 'straight_line_speed_px_frame': straight_line_speed_px_frame, 'straight_line_speed_um_s': straight_line_speed_um_s, 'straightness': straightness})
        return _pd().DataFrame(rows)

    @staticmethod
    def _apply_track_filters(metrics_df: pd.DataFrame, params: dict[str, Any]) -> set[int]:
        from bactomate.gui.tracking_filters import apply_track_filters
        return apply_track_filters(metrics_df, params)

    @staticmethod
    def _frame_index_for_track_focus(tracks_df: Any, track_id: int, *, prefer: int | None=None) -> int:
        """Frame index to show for a track (prefer current frame if present, else middle detection)."""
        if not _is_pandas_dataframe(tracks_df) or 'frame' not in tracks_df.columns:
            return int(prefer or 0)
        frames = tracks_df.loc[tracks_df['track_id'] == int(track_id), 'frame']
        if frames.empty:
            return int(prefer or 0)
        frame_vals = sorted((int(f) for f in frames.unique()))
        if prefer is not None and int(prefer) in frame_vals:
            return int(prefer)
        return frame_vals[len(frame_vals) // 2]

    @staticmethod
    def _track_centroid_at_frame(tracks_df: Any, track_id: int, frame_idx: int) -> tuple[int, int] | None:
        if not _is_pandas_dataframe(tracks_df):
            return None
        need = {'track_id', 'frame', 'y', 'x'}
        if not need.issubset(tracks_df.columns):
            return None
        pts = tracks_df[(tracks_df['track_id'] == int(track_id)) & (tracks_df['frame'] == int(frame_idx))]
        if pts.empty:
            return None
        return (int(pts.iloc[0]['y']), int(pts.iloc[0]['x']))

    @staticmethod
    def _mask_region_at_centroid(mask_frame: np.ndarray, y: int, x: int) -> np.ndarray | None:
        """Label mask connected to the track centroid (not mask value == track_id)."""
        (h, w) = mask_frame.shape[:2]
        yi = int(np.clip(int(y), 0, h - 1))
        xi = int(np.clip(int(x), 0, w - 1))
        label = int(mask_frame[yi, xi])
        if label <= 0:
            return None
        return mask_frame == label

    @staticmethod
    def _mask_region_for_track_at_frame(masks_tracked: np.ndarray | None, tracks_df: Any, frame_idx: int, track_id: int) -> np.ndarray | None:
        centroid = TrackingTab._track_centroid_at_frame(tracks_df, track_id, frame_idx)
        if centroid is None:
            return None
        (y, x) = centroid
        if masks_tracked is None or getattr(masks_tracked, 'ndim', 0) != 3:
            return None
        if not 0 <= frame_idx < masks_tracked.shape[0]:
            return None
        return TrackingTab._mask_region_at_centroid(masks_tracked[frame_idx], y, x)

    @staticmethod
    def _draw_polyline_on_frame(frame_rgb: np.ndarray, points_yx: np.ndarray, *, color: tuple[int, int, int]=(255, 255, 80), thickness: int=2) -> None:
        """Draw a thick polyline (y, x) in place on an RGB uint8 frame."""
        if points_yx.shape[0] < 2:
            return
        (h, w) = (frame_rgb.shape[0], frame_rgb.shape[1])
        pts = np.asarray(points_yx, dtype=int)
        pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
        for i in range(len(pts) - 1):
            (y1, x1) = (int(pts[i, 0]), int(pts[i, 1]))
            (y2, x2) = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
            steps = max(abs(y2 - y1), abs(x2 - x1), 1)
            step_size = max(1, steps // 60)
            for (y, x) in zip(np.linspace(y1, y2, steps // step_size + 1, dtype=int), np.linspace(x1, x2, steps // step_size + 1, dtype=int)):
                for dy in range(-thickness, thickness + 1):
                    for dx in range(-thickness, thickness + 1):
                        (yy, xx) = (int(y) + dy, int(x) + dx)
                        if 0 <= yy < h and 0 <= xx < w:
                            frame_rgb[yy, xx, 0] = color[0]
                            frame_rgb[yy, xx, 1] = color[1]
                            frame_rgb[yy, xx, 2] = color[2]

    @staticmethod
    def _apply_selected_track_halo(frame_rgb: np.ndarray, frame_idx: int, track_id: int, masks_tracked: np.ndarray | None, tracks_df: Any) -> np.ndarray:
        """Highlight selected track on frames where it is detected (trajectory + cell halo)."""
        from scipy.ndimage import binary_dilation
        out = np.ascontiguousarray(frame_rgb.copy())
        (h, w) = (out.shape[0], out.shape[1])
        tid = int(track_id)
        if TrackingTab._track_centroid_at_frame(tracks_df, tid, int(frame_idx)) is None:
            return out
        if _is_pandas_dataframe(tracks_df) and {'track_id', 'frame', 'y', 'x'}.issubset(tracks_df.columns):
            traj = tracks_df[(tracks_df['track_id'] == tid) & (tracks_df['frame'] <= int(frame_idx))].sort_values('frame')
            if len(traj) >= 2:
                TrackingTab._draw_polyline_on_frame(out, traj[['y', 'x']].values.astype(int), color=(255, 255, 90), thickness=2)
            elif len(traj) == 1:
                (y, x) = (int(traj.iloc[0]['y']), int(traj.iloc[0]['x']))
                if 0 <= y < h and 0 <= x < w:
                    out[y, x] = (255, 255, 90)
        region = TrackingTab._mask_region_for_track_at_frame(masks_tracked, tracks_df, frame_idx, tid)
        if region is None or not np.any(region):
            centroid = TrackingTab._track_centroid_at_frame(tracks_df, tid, frame_idx)
            if centroid is not None:
                (y, x) = centroid
                region = np.zeros((h, w), dtype=bool)
                (y0, y1) = (max(0, y - 4), min(h, y + 5))
                (x0, x1) = (max(0, x - 4), min(w, x + 5))
                region[y0:y1, x0:x1] = True
        if region is None or not np.any(region):
            return out
        halo = binary_dilation(region, iterations=2)
        ring = halo & ~region
        out[region, 0] = np.clip(out[region, 0] * 0.2 + 255 * 0.8, 0, 255).astype(np.uint8)
        out[region, 1] = np.clip(out[region, 1] * 0.2 + 255 * 0.8, 0, 255).astype(np.uint8)
        out[region, 2] = np.clip(out[region, 2] * 0.2 + 120 * 0.8, 0, 255).astype(np.uint8)
        out[ring, 0] = 255
        out[ring, 1] = 240
        out[ring, 2] = 80
        return out

    @staticmethod
    def _populate_preview_metrics_table(table: QTableWidget, frame_idx: int, tracks_df: Any, metrics_df: Any, passing_ids: set[int], *, use_um: bool, use_speed_physical: bool, selected_track_id: int | None=None) -> None:
        """Fill preview table with whole-track metrics for every track (optional in-frame flag)."""
        disp_hdr = 'Displacement (µm)' if use_um else 'Displacement (px)'
        path_hdr = 'Path length (µm)' if use_um else 'Path length (px)'
        speed_hdr = 'Mean speed (µm/s)' if use_speed_physical else 'Mean speed (px/frame)'
        headers = ['Pass', 'Track ID', 'Detections', 'Duration (frames)', 'Gaps', disp_hdr, path_hdr, speed_hdr, 'Straightness', 'In frame']
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(0)
        if not _is_pandas_dataframe(metrics_df) or metrics_df.empty:
            return
        in_frame_ids: set[int] = set()
        if _is_pandas_dataframe(tracks_df) and (not tracks_df.empty) and ('frame' in tracks_df.columns):
            in_frame_ids = {int(tid) for tid in tracks_df.loc[tracks_df['frame'] == frame_idx, 'track_id'].unique()}
        subset = metrics_df.copy()
        subset['_pass_order'] = subset['track_id'].map(lambda tid: 0 if int(tid) in passing_ids else 1)
        subset = subset.sort_values(['_pass_order', 'track_id']).drop(columns=['_pass_order'])
        pass_brush = QBrush(QColor(40, 90, 55))
        fail_brush = QBrush(QColor(55, 45, 45))
        table.setRowCount(len(subset))
        select_row = -1
        for (row_i, (_, m)) in enumerate(subset.iterrows()):
            tid = int(m['track_id'])
            if selected_track_id is not None and tid == int(selected_track_id):
                select_row = row_i
            passes = tid in passing_ids
            disp = float(m['displacement_um'] if use_um else m['displacement_px'])
            path = float(m['path_length_um'] if use_um else m['path_length_px'])
            spd = float(m['mean_speed_um_s'] if use_speed_physical else m['mean_speed_px_frame'])
            values = ['Yes' if passes else 'No', str(tid), str(int(m['trajectory_length'])), str(int(m['duration_frames'])), str(int(m['gap_frames'])), f'{disp:.2f}' if use_um else f'{disp:.0f}', f'{path:.2f}' if use_um else f'{path:.0f}', f'{spd:.3f}' if use_speed_physical else f'{spd:.2f}', f"{float(m['straightness']):.2f}", 'Yes' if tid in in_frame_ids else 'No']
            brush = pass_brush if passes else fail_brush
            for (col_i, text) in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(brush)
                if col_i == 1:
                    item.setData(Qt.ItemDataRole.UserRole, tid)
                table.setItem(row_i, col_i, item)
        if select_row >= 0:
            table.selectRow(select_row)
            id_item = table.item(select_row, 1)
            if id_item is not None:
                table.scrollToItem(id_item)
        _fit_preview_metrics_table_columns(table)
        QTimer.singleShot(0, lambda t=table: _fit_preview_metrics_table_columns(t))

    @staticmethod
    def _draw_filtered_trajectory_overlay(images: np.ndarray, tracks_df: Any, masks_tracked: np.ndarray | None=None) -> np.ndarray:
        """Overlay: phase image + colored segmentation masks per track (+ thin trajectory lines)."""
        import matplotlib.pyplot as plt
        if images.ndim != 3:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        (num_frames, height, width) = (images.shape[0], images.shape[1], images.shape[2])
        img_min = images.min(axis=(1, 2), keepdims=True)
        img_max = images.max(axis=(1, 2), keepdims=True)
        img_range = img_max - img_min
        img_range[img_range == 0] = 1
        images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)
        if not _is_pandas_dataframe(tracks_df) or tracks_df.empty or 'track_id' not in tracks_df.columns:
            return np.stack([np.stack([images_norm[i], images_norm[i], images_norm[i]], axis=-1) for i in range(num_frames)], axis=0)
        unique_tracks = tracks_df['track_id'].unique()
        num_tracks = len(unique_tracks)
        cmap = plt.get_cmap('gist_ncar')
        track_colors: dict[int, tuple[int, int, int]] = {}
        for (i, track_id) in enumerate(unique_tracks):
            color = cmap(i / max(num_tracks, 1))
            track_colors[int(track_id)] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        use_masks = masks_tracked is not None and getattr(masks_tracked, 'ndim', 0) == 3 and (masks_tracked.shape[0] >= num_frames) and (masks_tracked.shape[1] == height) and (masks_tracked.shape[2] == width)
        need_xy = 'y' in tracks_df.columns and 'x' in tracks_df.columns and ('frame' in tracks_df.columns)
        visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for frame_idx in range(num_frames):
            vis_frame = np.stack([images_norm[frame_idx], images_norm[frame_idx], images_norm[frame_idx]], axis=-1).copy()
            mask_frame = masks_tracked[frame_idx] if use_masks else None
            active = tracks_df[tracks_df['frame'] == frame_idx]['track_id'].unique()
            for track_id_int in active:
                track_id_int = int(track_id_int)
                if track_id_int not in track_colors:
                    continue
                color = track_colors[track_id_int]
                if use_masks and mask_frame is not None:
                    centroid = TrackingTab._track_centroid_at_frame(tracks_df, track_id_int, frame_idx)
                    region = None
                    if centroid is not None:
                        region = TrackingTab._mask_region_at_centroid(mask_frame, centroid[0], centroid[1])
                    if region is not None and np.any(region):
                        vis_frame[region, 0] = np.clip(vis_frame[region, 0] * 0.45 + color[0] * 0.55, 0, 255).astype(np.uint8)
                        vis_frame[region, 1] = np.clip(vis_frame[region, 1] * 0.45 + color[1] * 0.55, 0, 255).astype(np.uint8)
                        vis_frame[region, 2] = np.clip(vis_frame[region, 2] * 0.45 + color[2] * 0.55, 0, 255).astype(np.uint8)
            if need_xy:
                for track_id_int in tracks_df[tracks_df['frame'] <= frame_idx]['track_id'].unique():
                    track_id_int = int(track_id_int)
                    if track_id_int not in track_colors:
                        continue
                    track_points = tracks_df[(tracks_df['track_id'] == track_id_int) & (tracks_df['frame'] <= frame_idx)]
                    if len(track_points) < 2:
                        continue
                    color = track_colors[track_id_int]
                    points = track_points[['y', 'x']].values.astype(int)
                    points[:, 0] = np.clip(points[:, 0], 0, height - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, width - 1)
                    for i in range(len(points) - 1):
                        (y1, x1) = (int(points[i, 0]), int(points[i, 1]))
                        (y2, x2) = (int(points[i + 1, 0]), int(points[i + 1, 1]))
                        steps = max(abs(y2 - y1), abs(x2 - x1), 1)
                        step_size = max(1, steps // 40)
                        for (y, x) in zip(np.linspace(y1, y2, steps // step_size + 1, dtype=int), np.linspace(x1, x2, steps // step_size + 1, dtype=int)):
                            if 0 <= y < height and 0 <= x < width:
                                vis_frame[y, x, 0] = np.clip(vis_frame[y, x, 0] * 0.6 + color[0] * 0.4, 0, 255).astype(np.uint8)
                                vis_frame[y, x, 1] = np.clip(vis_frame[y, x, 1] * 0.6 + color[1] * 0.4, 0, 255).astype(np.uint8)
                                vis_frame[y, x, 2] = np.clip(vis_frame[y, x, 2] * 0.6 + color[2] * 0.4, 0, 255).astype(np.uint8)
            visualization[frame_idx] = vis_frame
        return visualization

    def run_tracking(self) -> None:
        """Workflow / session replay: full movement tracking (same as Run tracking (movement))."""
        self._run_tracking_with_action('movement')

    def _run_single_tracking_preview(self, mask_file: Path, original_file: Optional[str], model_name: str, mode: str, device: str, *, frame_limit: Optional[int], keep_preview_open: bool=False, busy_action: str='preview') -> None:
        """In-memory Trackastra preview (no _preview/ or _tracking/ writes)."""
        self._last_preview_in_memory = False
        self._last_track_action = busy_action
        preview_title = 'Preview lineage QC' if busy_action == 'preview_lineage' else 'Preview tracking'
        if not self._ensure_trackastra_model_available(model_name, title=preview_title):
            return
        if not keep_preview_open:
            self._dismiss_embedded_tracking_preview()
        self._set_run_buttons_busy(True, action=busy_action)
        excel_file = None
        preview_lineage = busy_action == 'preview_lineage'
        if preview_lineage:
            wb = resolve_analysis_workbook_for_mask(mask_file)
            if wb is not None:
                excel_file = str(wb)
        self.worker = TrackingWorker(mask_file, original_file, excel_file, model_name, mode, device, self, output_dir=None, track_action='preview', frame_limit=frame_limit, write_outputs=False, preview_lineage=preview_lineage)
        self.worker.preview_ready.connect(self._on_tracking_preview_ready)
        self.worker.finished.connect(self.on_tracking_preview_finished)
        self.worker.progress.connect(self.on_tracking_progress)
        self.worker.start()
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        n_mask = self._mask_stack_frame_count(Path(mask_file))
        if frame_limit is None:
            scope = f"full time lapse ({n_mask or '?'} frames)"
        else:
            n_use = min(int(frame_limit), int(n_mask)) if n_mask else int(frame_limit)
            scope = f"first {n_use} frame{('s' if n_use != 1 else '')}"
        self.parent_window.log_output(f'{preview_title} (in memory): {Path(mask_file).name} — {scope}; no files will be written.\n')

    def _on_tracking_preview_ready(self, payload: object) -> None:
        if not isinstance(payload, TrackingPreviewResult):
            return
        self._last_preview_in_memory = True
        self._last_tracking_preview_result = payload
        if getattr(payload, 'is_lineage_preview', False):
            self._last_lineage_preview_result = payload
        updater = getattr(self, '_preview_state_updater', None)
        if callable(updater):
            try:
                updater(payload)
            finally:
                self._preview_state_updater = None
            return
        self._show_tracking_preview_result(payload)

    def _run_preview_retrack(self, preview_state: dict[str, Any], *, on_updated: Callable[[], None]) -> None:
        """Re-run Trackastra in memory for the open preview (model/mode from UI)."""
        mask_path = preview_state.get('mask_path')
        if mask_path is None:
            show_message_dialog(self._main_window(), 'Re-track', 'No mask path for this preview.', kind='warning')
            return
        mask_file = Path(mask_path)
        orig = preview_state.get('original_path')
        if orig is None or not Path(orig).is_file():
            orig = self._resolve_original_for_mask(mask_file)
        original_file = str(orig) if orig and Path(orig).is_file() else None

        def _apply_payload(result: TrackingPreviewResult) -> None:
            self._last_tracking_preview_result = result
            if getattr(result, 'is_lineage_preview', False):
                self._last_lineage_preview_result = result
            full_df = result.tracks_df.copy() if _is_pandas_dataframe(result.tracks_df) else result.tracks_df
            preview_state['tracks_df_full'] = full_df
            preview_state['tracks_df'] = full_df
            preview_state['frames'] = np.asarray(result.frames_rgb, dtype=np.uint8)
            preview_state['original_path'] = result.original_path
            if getattr(result, 'stack_images', None) is not None:
                preview_state['stack_images'] = np.asarray(result.stack_images)
            if getattr(result, 'masks_input', None) is not None:
                preview_state['masks_input'] = np.asarray(result.masks_input)
            if getattr(result, 'masks_tracked', None) is not None:
                preview_state['masks_tracked'] = np.asarray(result.masks_tracked)
            preview_state['edges_df'] = getattr(result, 'edges_df', None)
            preview_state['founder_by_track'] = getattr(result, 'founder_by_track', None) or {}
            preview_state['lineage_tables'] = getattr(result, 'lineage_tables', None) or {}
            preview_state['is_lineage_preview'] = bool(getattr(result, 'is_lineage_preview', False))
            preview_state.pop('metrics_df', None)
            on_updated()
        model_name = self._selected_trackastra_model()
        if not self._ensure_trackastra_model_available(model_name, title='Re-track'):
            return
        frame_limit = preview_state.get('preview_frame_limit', TRACKING_PREVIEW_FRAME_LIMIT)
        self._preview_state_updater = lambda p: _apply_payload(p) if isinstance(p, TrackingPreviewResult) else None
        self._run_single_tracking_preview(mask_file, original_file, model_name, self.mode_combo.currentText(), self._resolve_device(), frame_limit=frame_limit, keep_preview_open=True, busy_action='preview_lineage' if bool(preview_state.get('is_lineage_preview')) else 'preview')

    def on_tracking_preview_finished(self, success: bool, message: str) -> None:
        """Preview-only completion (results already shown via preview_ready)."""
        self.parent_window.progress_bar.setValue(100)
        action = getattr(self, '_tracking_busy_action', None) or 'preview'
        btn_by_action = {'preview': getattr(self, 'preview_btn', None), 'preview_lineage': getattr(self, 'preview_lineage_btn', None)}
        active = btn_by_action.get(action)
        if isinstance(active, ProgressCancelButton) and getattr(active, '_progress_active', False):
            active.set_progress(100)
        if success:
            label = 'Preview lineage QC' if action == 'preview_lineage' else 'Preview tracking'
            self.parent_window.log_output(f'✓ {label} finished.\n')
        elif 'trackastra_pretrained_feats' in (message or '') or 'pretrained features' in (message or '').lower():
            (_, install_msg) = validate_trackastra_model_choice(TRACKASTRA_SAM2_MODEL)
            show_message_dialog(self._main_window(), 'Preview failed', install_msg or format_trackastra_model_error(ImportError(message or '')), kind='warning')
        else:
            show_message_dialog(self._main_window(), 'Preview failed', message, kind='error')
        self._preview_state_updater = None
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.wait()
            self.worker = None
        QTimer.singleShot(2000, lambda : self.parent_window.progress_bar.setVisible(False))
        self._set_run_buttons_busy(False)

    def _show_tracking_preview_result(self, result: TrackingPreviewResult) -> None:
        """Embed trajectory preview with parameters/filters in the right pane (segmentation-style)."""
        n_tracks = result.n_tracks
        n_fr = int(result.frames_rgb.shape[0]) if getattr(result.frames_rgb, 'shape', None) else 0
        fl = getattr(self, '_last_preview_frame_limit', TRACKING_PREVIEW_FRAME_LIMIT)
        scope = f'{n_fr} frames' if fl is None else f'{n_fr}/{fl} frames'
        label = f'{result.mask_path.name} — preview ({n_tracks} tracks, {scope})'
        self._open_tracking_preview_dialog(frames_rgb=result.frames_rgb, tracks_df=result.tracks_df, original_path=result.original_path, display_name=label, mask_path=result.mask_path, stack_images=getattr(result, 'stack_images', None), masks_input=getattr(result, 'masks_input', None), masks_tracked=getattr(result, 'masks_tracked', None), edges_df=getattr(result, 'edges_df', None), founder_by_track=getattr(result, 'founder_by_track', None), lineage_tables=getattr(result, 'lineage_tables', None), is_lineage_preview=getattr(result, 'is_lineage_preview', False))

    def _lineage_preflight_check(self, mask_file: Path, *, title: str, always_confirm: bool = False) -> bool:
        """Warn about mask/analysis mismatch and linker before lineage jobs."""
        from bactomate.modules.lineage_integration import preflight_lineage_run, preflight_summary_lines, resolve_lineage_mask_path
        is_full_lineage = title == 'Run lineage tracking'
        pf = preflight_lineage_run(
            mask_file,
            linker_mode=self.mode_combo.currentText(),
            model_name=self._selected_trackastra_model(),
            require_analysis_workbook=is_full_lineage,
            strict_lineage=is_full_lineage,
        )
        if not pf.ok:
            show_message_dialog(self._main_window(), title, 'Cannot start lineage tracking:\n\n' + '\n'.join((f'• {e}' for e in pf.errors)), kind='warning')
            return False
        summary = preflight_summary_lines(pf)
        if not always_confirm and not pf.warnings:
            return True
        msg = 'Lineage preflight:\n\n' + '\n'.join(summary)
        if pf.warnings:
            msg += '\n\nWarnings:\n' + '\n'.join((f'• {w}' for w in pf.warnings))
        resolved = pf.resolved_mask_path or resolve_lineage_mask_path(mask_file)
        if resolved != Path(mask_file):
            msg += f'\n\nWill use mask:\n{resolved}'
        excel_hint = (
            '\n\nReminder: analysis Excel frames are 1-based; tracking frames are 0-based. '
            'Use the same mask file as Analyze Masks (filtered if you analyzed filtered).'
        )
        if is_full_lineage or pf.analysis_workbook is not None:
            msg += excel_hint
        if not show_confirm_dialog(self._main_window(), title=title, message=msg + '\n\nContinue?', confirm_label='Yes', cancel_label='No', default_confirm=True):
            return False
        return True

    def _open_lineage_inspector(self) -> None:
        from bactomate.gui.lineage_inspector import open_lineage_inspector_from_output_dir, open_lineage_inspector_from_preview
        last = getattr(self, '_last_lineage_preview_result', None)
        if last is not None and getattr(last, 'is_lineage_preview', False):
            open_lineage_inspector_from_preview(self._main_window(), last)
            return
        out = getattr(self, '_last_tracking_output_dir', None)
        if out is not None and Path(out).is_dir():
            try:
                open_lineage_inspector_from_output_dir(self._main_window(), Path(out))
                return
            except Exception as exc:
                show_message_dialog(self._main_window(), 'Lineage review', f'Could not load lineage tables from output folder:\n{exc}', kind='warning')
                return
        show_message_dialog(self._main_window(), 'Lineage review', 'Run Preview lineage QC first, or complete lineage tracking with outputs under _tracking/{stem}_tracking/.', kind='info')

    def _run_tracking_with_action(self, track_action: str) -> None:
        if track_action in ('lineage', 'preview_lineage'):
            self._ensure_lineage_linking_mode()
        path_edit = self._path_edit_for_action(track_action)
        model_name = self._selected_trackastra_model()
        title = {'preview': 'Preview tracking', 'preview_lineage': 'Preview lineage QC', 'movement': 'Run tracking', 'lineage': 'Run lineage tracking'}.get(track_action, 'Tracking')
        if not self._ensure_trackastra_model_available(model_name, title=title):
            return
        if not path_edit.text() or not path_edit.text().strip():
            show_message_dialog(self._main_window(), 'Missing Input', 'Please select a mask file or folder.', kind='warning')
            return
        path = Path(path_edit.text()).resolve()
        if not path.exists():
            show_message_dialog(self._main_window(), 'Not Found', f'Path not found:\n{path}', kind='warning')
            return
        mask_files = self._mask_files_for_tracking_input(path)
        if not mask_files:
            show_message_dialog(self._main_window(), 'No Mask Files Found', 'No mask TIFF files found for this path.\n\nSelect a *_masks.tif / *_mask_filtered.tif file, or a folder that contains them (e.g. _segmentation_* or a parent folder with Tif_* subfolders).', kind='warning')
            return
        if path.is_file() and (not is_cell_mask_tiff(path)) and (len(mask_files) == 1):
            self.parent_window.log_output(f'Using mask {mask_files[0].name} for {path.name} (original is for features/metadata only).\n')
        if track_action in ('preview', 'preview_lineage'):
            is_lineage_preview = track_action == 'preview_lineage'
            setup = self._prompt_preview_tracking_setup(mask_files, dialog_title='Preview lineage QC' if is_lineage_preview else 'Preview tracking', intro_text='Quick lineage QC in memory — check division candidates and track families before full lineage tracking. Linking allows divisions (greedy_nodiv is switched to greedy). Nothing is written to disk.\nChoose the mask and how many frames to include.' if is_lineage_preview else None, preferred_path_edit=self.lineage_file_path if is_lineage_preview else self.file_path)
            if setup is None:
                label = 'Preview lineage QC' if is_lineage_preview else 'Preview tracking'
                self.parent_window.log_output(f'{label} cancelled.\n')
                return
            (mask_file, frame_limit) = setup
            mask_file = mask_file.resolve()
            if is_lineage_preview and (not self._lineage_preflight_check(mask_file, title='Preview lineage QC')):
                return
            path_edit.blockSignals(True)
            try:
                path_edit.setText(str(mask_file))
            finally:
                path_edit.blockSignals(False)
            if is_lineage_preview:
                self._update_lineage_tracking_button_state()
            else:
                self._update_tracking_button_state()
                self._update_filter_ranges_from_trajectory()
                try:
                    sel = str(mask_file.resolve())
                except OSError:
                    sel = str(mask_file)
                self.refresh_tracking_candidate_list(selected=sel)
            from bactomate.modules.lineage_integration import resolve_lineage_mask_path
            run_mask = resolve_lineage_mask_path(mask_file) if is_lineage_preview else mask_file
            original_path = self._resolve_original_for_mask(run_mask)
            original_file = str(original_path) if original_path else None
            self._last_preview_frame_limit = frame_limit
            self._run_single_tracking_preview(run_mask, original_file, model_name, self.mode_combo.currentText(), self._resolve_device(), frame_limit=frame_limit, busy_action=track_action)
            return
        pick_msg = f'Select mask files to track (all {len(mask_files)} selected by default):'
        selected = show_file_selection_dialog(self._main_window(), mask_files, title='Select Mask Files for Tracking', message=pick_msg)
        if selected is None:
            self.parent_window.log_output('Tracking cancelled (no mask files selected).\n')
            return
        if len(selected) == 0:
            show_message_dialog(self._main_window(), 'No Files Selected', 'Please select at least one mask file to track.', kind='warning')
            return
        selected = [p for p in selected if is_cell_mask_tiff(p)]
        if not selected:
            show_message_dialog(self._main_window(), 'Invalid selection', 'Please select mask TIFFs (*_masks.tif or *_mask_filtered.tif), not the original phase stack.', kind='warning')
            return
        self._batch_lineage_summary_items: list = []
        if len(selected) == 1:
            mask_file = selected[0].resolve()
            output_dir = self._tracking_output_dir(mask_file, track_action, None)
            base_dir = base_dir_for_mask(mask_file)
            original_path = self._resolve_original_for_mask(mask_file)
            original_file = str(original_path) if original_path else None
            excel_path = find_analysis_excel_for_mask(mask_file, base_dir)
            excel_file = str(excel_path) if excel_path else None
            mode = self.mode_combo.currentText()
            device = self._resolve_device()
            self._run_single_tracking(mask_file, output_dir, original_file, excel_file, model_name, mode, device, track_action=track_action, frame_limit=None)
            return
        tasks = []
        for mask_path in selected:
            mask_path = mask_path.resolve()
            base_dir = base_dir_for_mask(mask_path)
            output_dir = self._tracking_output_dir(mask_path, track_action, None)
            original_path = self._resolve_original_for_mask(mask_path)
            excel_path = find_analysis_excel_for_mask(mask_path, base_dir)
            tasks.append((mask_path, original_path, excel_path, output_dir))
        mode = self.mode_combo.currentText()
        device = self._resolve_device()
        if device == 'cuda':
            debug_info = ['=== CUDA Debugging Information ===']
            try:
                import torch
                debug_info.append(f'✓ PyTorch version: {torch.__version__}')
                cuda_available = torch.cuda.is_available()
                debug_info.append(f'✓ torch.cuda.is_available(): {cuda_available}')
                if not cuda_available:
                    debug_info.append('✗ CUDA is not available')
                    error_msg = '\n'.join(debug_info) + '\n\nCUDA GPU is not available. Continue with CPU?'
                    if not show_confirm_dialog(self._main_window(), title='CUDA Not Available', message=error_msg, confirm_label='Yes', cancel_label='No', default_confirm=True):
                        return
                    device = 'cpu'
            except ImportError:
                show_message_dialog(self._main_window(), 'PyTorch Not Found', 'PyTorch required for CUDA. Switching to CPU.', kind='warning')
                device = 'cpu'
        elif device == 'mps':
            try:
                import torch
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    error_msg = 'MPS (Apple Silicon GPU) is not available. Continue with CPU?'
                    if not show_confirm_dialog(self._main_window(), title='MPS Not Available', message=error_msg, confirm_label='Yes', cancel_label='No', default_confirm=True):
                        return
                    device = 'cpu'
            except ImportError:
                show_message_dialog(self._main_window(), 'PyTorch Not Found', 'PyTorch required for MPS. Switching to CPU.', kind='warning')
                device = 'cpu'
        self.folder_tracking_tasks = tasks
        self.folder_tracking_results = []
        self.current_task_index = 0
        self._batch_model_name = model_name
        self._batch_mode = mode
        self._batch_device = device
        self._batch_track_action = track_action
        self._batch_frame_limit = None
        batch_action = getattr(self, '_batch_track_action', 'movement')
        batch_idle = {'preview': self._preview_btn_idle, 'movement': self._movement_btn_idle, 'lineage': self._lineage_btn_idle}.get(batch_action, self._movement_btn_idle)
        self._set_run_buttons_busy(True, action=batch_action, progress_label=f'{batch_idle}… (0/{len(tasks)})')
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f'Starting batch tracking for {len(tasks)} file(s).\n')
        self._process_next_tracking_task()
        return

    def _write_visualization_for_mask(self, mask_file: Path, track_action: str) -> bool:
        """Always write tracking disk visualizations (except preview runs)."""
        from bactomate.gui.workers.tracking_io import should_write_tracking_visualization

        return should_write_tracking_visualization(
            mask_file,
            user_enabled=True,
            track_action=track_action,
        )

    def _run_single_tracking(self, mask_file: Path, output_dir: Path, original_file: Optional[str], excel_file: Optional[str], model_name: str, mode: str, device: str, track_action: str='movement', frame_limit: Optional[int]=None) -> None:
        """Run device check and start tracking worker for a single mask file."""
        if device == 'cuda':
            debug_info = []
            debug_info.append('=== CUDA Debugging Information ===')
            try:
                import torch
                debug_info.append(f'✓ PyTorch version: {torch.__version__}')
                cuda_available = torch.cuda.is_available()
                debug_info.append(f'✓ torch.cuda.is_available(): {cuda_available}')
                if not cuda_available:
                    debug_info.append('✗ CUDA is not available')
                    debug_info.append(f"  - CUDA version (compiled): {(torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A')}")
                    debug_info.append(f"  - cuDNN version: {(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A')}")
                    try:
                        import subprocess
                        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
                        if result.returncode == 0:
                            debug_info.append(f'  - nvcc found: {result.stdout.strip()}')
                        else:
                            debug_info.append('  - nvcc not found in PATH')
                    except:
                        pass
                    error_msg = '\n'.join(debug_info)
                    error_msg += '\n\nCUDA GPU is not available on this system.'
                    error_msg += '\n\nPossible reasons:'
                    error_msg += '\n- No NVIDIA GPU installed'
                    error_msg += '\n- CUDA drivers not installed'
                    error_msg += '\n- PyTorch was installed without CUDA support'
                    error_msg += '\n- GPU drivers are outdated'
                    if not show_confirm_dialog(self, title='CUDA Not Available', message=error_msg + '\n\nWould you like to continue with CPU mode?', confirm_label='Yes', cancel_label='No', default_confirm=True):
                        return
                    device = 'cpu'
                    self.parent_window.log_output('\n'.join(debug_info) + '\n')
                else:
                    debug_info.append('✓ CUDA is available!')
                    debug_info.append(f"  - CUDA version (compiled): {(torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A')}")
                    debug_info.append(f'  - Number of GPUs: {torch.cuda.device_count()}')
                    for i in range(torch.cuda.device_count()):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_props = torch.cuda.get_device_properties(i)
                        debug_info.append(f'  - GPU {i}: {gpu_name}')
                        debug_info.append(f'    - Memory: {gpu_props.total_memory / 1024 ** 3:.2f} GB')
                        debug_info.append(f'    - Compute Capability: {gpu_props.major}.{gpu_props.minor}')
                    debug_info.append('\nTesting CUDA functionality...')
                    try:
                        test_tensor = torch.tensor([1.0], device='cuda')
                        debug_info.append('  ✓ Tensor creation: SUCCESS')
                        result = test_tensor * 2.0
                        debug_info.append('  ✓ Tensor operation: SUCCESS')
                        test_large = torch.zeros(1000, 1000, device='cuda')
                        debug_info.append('  ✓ Memory allocation: SUCCESS')
                        del test_tensor, result, test_large
                        torch.cuda.empty_cache()
                        debug_info.append('  ✓ Memory cleanup: SUCCESS')
                        debug_info.append('\n✓ CUDA is working correctly!')
                        self.parent_window.log_output('\n'.join(debug_info) + '\n')
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        debug_info.append(f'  ✗ CUDA test failed: {str(e)}')
                        debug_info.append(f'\nError details:\n{error_details}')
                        error_msg = '\n'.join(debug_info)
                        error_msg += '\n\nCUDA is available but encountered an error during testing.'
                        error_msg += '\nThis might indicate:'
                        error_msg += '\n- GPU driver issues'
                        error_msg += '\n- CUDA runtime problems'
                        error_msg += '\n- Memory allocation issues'
                        if not show_confirm_dialog(self, title='CUDA Error', message=error_msg + '\n\nWould you like to continue with CPU mode?', confirm_label='Yes', cancel_label='No', default_confirm=True):
                            return
                        device = 'cpu'
                        self.parent_window.log_output('\n'.join(debug_info) + '\n')
            except ImportError:
                debug_info.append('✗ PyTorch is not installed')
                error_msg = '\n'.join(debug_info)
                error_msg += '\n\nPyTorch is required for CUDA support.'
                error_msg += '\nInstall with: pip install torch torchvision'
                error_msg += '\nFor CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118'
                show_message_dialog(self._main_window(), 'PyTorch Not Found', error_msg + '\n\nSwitching to CPU mode.', kind='warning')
                device = 'cpu'
                self.parent_window.log_output('\n'.join(debug_info) + '\n')
            except Exception as e:
                import traceback
                debug_info.append(f'✗ Unexpected error: {str(e)}')
                debug_info.append(f'\n{traceback.format_exc()}')
                error_msg = '\n'.join(debug_info)
                show_message_dialog(self._main_window(), 'CUDA Check Error', error_msg + '\n\nSwitching to CPU mode.', kind='warning')
                device = 'cpu'
                self.parent_window.log_output('\n'.join(debug_info) + '\n')
        elif device == 'mps':
            debug_info = []
            debug_info.append('=== GPU (Apple Silicon) Debugging Information ===')
            try:
                import torch
                debug_info.append(f'✓ PyTorch version: {torch.__version__}')
                if hasattr(torch.backends, 'mps'):
                    mps_available = torch.backends.mps.is_available()
                    debug_info.append(f'✓ torch.backends.mps.is_available(): {mps_available}')
                    if not mps_available:
                        debug_info.append('✗ GPU is not available')
                        debug_info.append("  - This usually means you're not on Apple Silicon (M1/M2/M3)")
                        debug_info.append('  - Or PyTorch was built without GPU support')
                        error_msg = '\n'.join(debug_info)
                        error_msg += '\n\nMPS (Apple Silicon GPU) is not available.'
                        error_msg += '\n\nPossible reasons:'
                        error_msg += '\n- Not running on Apple Silicon (M1/M2/M3 chip)'
                        error_msg += '\n- PyTorch was installed without MPS support'
                        error_msg += '\n- macOS version too old (needs macOS 12.3+)'
                        if not show_confirm_dialog(self._main_window(), title='MPS Not Available', message=error_msg + '\n\nWould you like to continue with CPU mode?', confirm_label='Yes', cancel_label='No', default_confirm=True):
                            return
                        device = 'cpu'
                        self.parent_window.log_output('\n'.join(debug_info) + '\n')
                    else:
                        debug_info.append('✓ GPU is available!')
                        debug_info.append('  - Device: Apple Silicon GPU')
                        debug_info.append('\nTesting GPU functionality...')
                        try:
                            test_tensor = torch.tensor([1.0], device='mps')
                            debug_info.append('  ✓ Tensor creation: SUCCESS')
                            result = test_tensor * 2.0
                            debug_info.append('  ✓ Tensor operation: SUCCESS')
                            test_large = torch.zeros(1000, 1000, device='mps')
                            debug_info.append('  ✓ Memory allocation: SUCCESS')
                            del test_tensor, result, test_large
                            torch.mps.empty_cache()
                            debug_info.append('  ✓ Memory cleanup: SUCCESS')
                            debug_info.append('\n✓ GPU is working correctly!')
                            self.parent_window.log_output('\n'.join(debug_info) + '\n')
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            debug_info.append(f'  ✗ MPS test failed: {str(e)}')
                            debug_info.append(f'\nError details:\n{error_details}')
                            error_msg = '\n'.join(debug_info)
                            error_msg += '\n\nGPU is available but encountered an error during testing.'
                            error_msg += '\nThis might indicate:'
                            error_msg += '\n- macOS version compatibility issues'
                            error_msg += '\n- PyTorch MPS backend problems'
                            error_msg += '\n- Memory allocation issues'
                            if not show_confirm_dialog(self._main_window(), title='MPS Error', message=error_msg + '\n\nWould you like to continue with CPU mode?', confirm_label='Yes', cancel_label='No', default_confirm=True):
                                return
                            device = 'cpu'
                            self.parent_window.log_output('\n'.join(debug_info) + '\n')
                else:
                    debug_info.append('✗ MPS backend not available in this PyTorch version')
                    error_msg = '\n'.join(debug_info)
                    error_msg += '\n\nMPS backend is not available.'
                    error_msg += '\nThis PyTorch version may not support MPS.'
                    if not show_confirm_dialog(self, title='MPS Not Supported', message=error_msg + '\n\nWould you like to continue with CPU mode?', confirm_label='Yes', cancel_label='No', default_confirm=True):
                        return
                    device = 'cpu'
                    self.parent_window.log_output('\n'.join(debug_info) + '\n')
            except ImportError:
                debug_info.append('✗ PyTorch is not installed')
                error_msg = '\n'.join(debug_info)
                error_msg += '\n\nPyTorch is required for MPS support.'
                show_message_dialog(self, 'PyTorch Not Found', error_msg + '\n\nSwitching to CPU mode.', kind='warning')
                device = 'cpu'
                self.parent_window.log_output('\n'.join(debug_info) + '\n')
            except Exception as e:
                import traceback
                debug_info.append(f'✗ Unexpected error: {str(e)}')
                debug_info.append(f'\n{traceback.format_exc()}')
                error_msg = '\n'.join(debug_info)
                show_message_dialog(self, 'MPS Check Error', error_msg + '\n\nSwitching to CPU mode.', kind='warning')
                device = 'cpu'
                self.parent_window.log_output('\n'.join(debug_info) + '\n')
        from bactomate.modules.lineage_integration import resolve_lineage_mask_path
        if track_action == 'lineage':
            mask_file = resolve_lineage_mask_path(Path(mask_file))
            if excel_file is None:
                wb = resolve_analysis_workbook_for_mask(mask_file)
                excel_file = str(wb) if wb else None
            if not self._lineage_preflight_check(mask_file, title='Run lineage tracking', always_confirm=True):
                return
        self._last_tracking_output_dir = output_dir
        self._last_track_action = track_action
        self._last_preview_in_memory = False
        busy_action = 'lineage' if track_action == 'lineage' else 'movement'
        self._set_run_buttons_busy(True, action=busy_action)
        lineage_curation_path = None
        if track_action == 'lineage':
            from bactomate.modules.lineage_integration import resolve_lineage_curation_path

            lineage_curation_path = resolve_lineage_curation_path(Path(mask_file))
            if lineage_curation_path.is_file():
                self.parent_window.log_output(
                    f'Found saved lineage curation — will apply after tracking: {lineage_curation_path}\n'
                )
        self.worker = TrackingWorker(mask_file, original_file, excel_file, model_name, mode, device, self, output_dir=output_dir, track_action=track_action, frame_limit=frame_limit, run_metadata=self._build_tracking_run_metadata(mask_file, model_name=model_name, mode=mode, device=device, track_action=track_action, original_file=original_file, excel_file=excel_file, frame_limit=frame_limit), filter_params=self._get_filter_params(), write_visualization=self._write_visualization_for_mask(mask_file, track_action), lineage_curation_path=lineage_curation_path)
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.start()
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f'Starting tracking for: {mask_file.name}\n')
        vis_note = 'masks → overlay → trajectory visualization' if getattr(self, 'create_vis_cb', None) and self.create_vis_cb.isChecked() else 'masks → tracks (visualization skipped — Create visualization off)'
        self.parent_window.log_output(f"Tracking runs several steps ({vis_note}). Do not close the app until you see '✓ Tracking completed successfully!' below.\nAt 55%, feature extraction and linking can take several minutes; 'Still running…' messages in the log mean the app is working.\n")
        self.worker.progress.connect(self.on_tracking_progress)

    def on_tracking_progress(self, value, message) -> None:
        """Update progress bar, active button label, and log message."""
        self.parent_window.progress_bar.setValue(value)
        if getattr(self, '_tracking_busy_action', None):
            self._update_busy_action_progress(value, message=str(message or ''))
        if message:
            self.parent_window.log_output(f'[{value}%] {message}\n')

    def _maybe_auto_open_preview_window(self, out_path: Path) -> None:
        """After a successful Preview run, open the trajectory viewer if a TIFF was written."""
        if not out_path.is_dir():
            return
        traj_files = sorted(out_path.glob('*_trajectories.tif'))
        if not traj_files:
            self.parent_window.log_output('Preview: no *_trajectories.tif (enable Create visualization and ensure the phase/original stack is found).\n')
            return
        self._open_trajectory_preview_window(traj_files[0], index=0)

    def on_tracking_finished(self, success, message):
        self.parent_window.progress_bar.setValue(100)
        action = getattr(self, '_tracking_busy_action', None)
        btn_by_action = {'preview': getattr(self, 'preview_btn', None), 'preview_lineage': getattr(self, 'preview_lineage_btn', None), 'movement': getattr(self, 'movement_btn', None), 'lineage': getattr(self, 'lineage_btn', None)}
        active = btn_by_action.get(action or '')
        if isinstance(active, ProgressCancelButton) and getattr(active, '_progress_active', False):
            active.set_progress(100)
        if success:
            self.parent_window.log_output('✓ Tracking completed successfully!\n')
            out_dir = getattr(self, '_last_tracking_output_dir', None)
            if getattr(self, '_last_track_action', None) == 'lineage' and out_dir:
                analysis = getattr(self, '_lineage_analysis', None)
                if analysis is not None:
                    try:
                        self._dismiss_embedded_tracking_preview()
                        analysis.set_source_path(Path(out_dir), load=True, show_workspace=True)
                        self._reveal_lineage_analysis()
                    except Exception:
                        pass
            if getattr(self, '_last_track_action', None) in ('preview', 'preview_lineage') and out_dir and (not getattr(self, '_last_preview_in_memory', False)):
                self._maybe_auto_open_preview_window(Path(out_dir))
            if not getattr(self, '_last_preview_in_memory', False):
                from bactomate.gui.widgets import show_task_complete_dialog
                show_task_complete_dialog(self._main_window(), title='Tracking Complete', message='Tracking completed successfully.', output_dir=out_dir)
        else:
            show_message_dialog(self._main_window(), 'Error', f'Tracking failed:\n{message}', kind='error')
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda : self.parent_window.progress_bar.setVisible(False))
        self._update_tracking_button_state()
        self._update_lineage_tracking_button_state()
        if success:
            self._performed_this_session = True
        self.workflow_step_finished.emit(success)

    def _process_next_tracking_task(self) -> None:
        """Run the next task in batch tracking, or finish batch if none left."""
        tasks = getattr(self, 'folder_tracking_tasks', [])
        if self.current_task_index >= len(tasks):
            n_ok = sum((1 for r in getattr(self, 'folder_tracking_results', []) if r))
            n_fail = len(tasks) - n_ok
            self._set_run_buttons_busy(False)
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f'\n✓ Batch tracking complete: {n_ok}/{len(tasks)} succeeded.' + (f' {n_fail} failed — check log above.\n' if n_fail else '\n'))
            self._performed_this_session = True
            self.workflow_step_finished.emit(True)
            msg = f'Completed: {n_ok}/{len(tasks)} file(s) tracked successfully.'
            if n_fail:
                msg += f'\n{n_fail} file(s) failed — see Log for errors.'
            first_output = Path(tasks[0][3]) if tasks and len(tasks[0]) > 3 else None
            from bactomate.gui.widgets import show_task_complete_dialog
            if getattr(self, '_batch_track_action', None) == 'lineage':
                try:
                    from bactomate.modules.lineage_integration import MASTER_LINEAGE_SUMMARY_NAME, write_batch_summary_lineage
                    items = getattr(self, '_batch_lineage_summary_items', []) or []
                    if items and first_output is not None:
                        summary_path = Path(first_output).parent.parent / MASTER_LINEAGE_SUMMARY_NAME
                        write_batch_summary_lineage(summary_path, items)
                        self.parent_window.log_output(f'Wrote batch lineage summary: {summary_path}\n')
                except Exception:
                    pass
                results = getattr(self, 'folder_tracking_results', [])
                last_out = None
                for i in range(len(tasks) - 1, -1, -1):
                    if i < len(results) and results[i] and len(tasks[i]) > 3:
                        last_out = Path(tasks[i][3])
                        break
                analysis = getattr(self, '_lineage_analysis', None)
                if last_out is not None and analysis is not None:
                    try:
                        self._dismiss_embedded_tracking_preview()
                        analysis.set_source_path(last_out, load=True, show_workspace=True)
                        self._reveal_lineage_analysis()
                    except Exception:
                        pass
            show_task_complete_dialog(self._main_window(), title='Batch Tracking Complete', message=msg, output_dir=first_output)
            self._update_tracking_button_state()
            self._update_lineage_tracking_button_state()
            return
        (mask_path, original_path, excel_path, output_dir) = tasks[self.current_task_index]
        model_name = getattr(self, '_batch_model_name', self._selected_trackastra_model())
        mode = getattr(self, '_batch_mode', self.mode_combo.currentText())
        device = getattr(self, '_batch_device', self._resolve_device())
        original_file = str(original_path) if original_path else None
        excel_file = str(excel_path) if excel_path else None
        self.parent_window.log_output(f'\nProcessing {self.current_task_index + 1}/{len(tasks)}: {mask_path.name}\n')
        if self.current_task_index == 0:
            self.parent_window.log_output("Do not close the app until you see 'Batch tracking complete' (or all files finished).\n")
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        batch_action = getattr(self, '_batch_track_action', 'movement')
        lineage_curation_path = None
        if batch_action == 'lineage':
            from bactomate.modules.lineage_integration import resolve_lineage_curation_path

            lineage_curation_path = resolve_lineage_curation_path(Path(mask_path))
        self.worker = TrackingWorker(mask_path, original_file, excel_file, model_name, mode, device, self, output_dir=output_dir, track_action=batch_action, frame_limit=getattr(self, '_batch_frame_limit', None), run_metadata=self._build_tracking_run_metadata(mask_path, model_name=model_name, mode=mode, device=device, track_action=batch_action, original_file=original_file, excel_file=excel_file, frame_limit=getattr(self, '_batch_frame_limit', None)), filter_params=self._get_filter_params(), write_visualization=self._write_visualization_for_mask(mask_path, batch_action), lineage_curation_path=lineage_curation_path)
        self.worker.finished.connect(self.on_tracking_task_finished)
        self.worker.progress.connect(self._on_batch_tracking_progress)
        self.worker.start()

    def _on_batch_tracking_progress(self, value: int, message: str) -> None:
        """Update progress bar for batch: overall 0–100 across all tasks."""
        tasks = getattr(self, 'folder_tracking_tasks', [])
        if not tasks:
            return
        total = len(tasks)
        overall = (self.current_task_index + value / 100.0) / total * 100
        overall_i = int(min(99, overall))
        self.parent_window.progress_bar.setValue(overall_i)
        if getattr(self, '_tracking_busy_action', None):
            self._update_busy_action_progress(overall_i, message=str(message or ''))
        if message:
            self.parent_window.log_output(f'[{self.current_task_index + 1}/{total}] {message}\n')

    def on_tracking_task_finished(self, success: bool, message: str) -> None:
        """Called when one batch task finishes; record result and start next or finish batch."""
        self.folder_tracking_results.append(success)
        if not success:
            self.parent_window.log_output(f'✗ Failed: {message}\n')
        elif getattr(self, '_batch_track_action', None) == 'lineage':
            try:
                import pandas as pd
                tasks = getattr(self, 'folder_tracking_tasks', [])
                idx = max(0, int(self.current_task_index))
                if idx < len(tasks):
                    mask_path = Path(tasks[idx][0])
                    out_dir = Path(tasks[idx][3])
                    summary_p = out_dir / 'tables' / 'lineage_summary.parquet'
                    if not summary_p.is_file():
                        summary_p = out_dir / 'tables' / 'lineage_summary.csv.gz'
                    if summary_p.is_file():
                        df = pd.read_parquet(summary_p) if summary_p.suffix == '.parquet' else pd.read_csv(summary_p)
                        getattr(self, '_batch_lineage_summary_items', []).append((df, stem_from_mask_path(mask_path)))
            except Exception:
                pass
        self.current_task_index += 1
        n = len(self.folder_tracking_tasks)
        self.parent_window.progress_bar.setValue(int(self.current_task_index / n * 100))
        self._process_next_tracking_task()

    def _export_filtered_tracks_for_paths(self, traj_paths: list[Path], silent: bool=False) -> int:
        """Export filtered track Excel for each trajectory path. Returns number of files written. If silent, no progress UI or final message."""
        if not traj_paths:
            return 0
        n_files = len(traj_paths)
        if not silent:
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, max(1, n_files))
            self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f'Exporting filtered tracks ({n_files} file(s))...\n')
        params = self._get_filter_params()
        pixel_size = params['pixel_size']
        time_per_frame = params['time_per_frame']
        done = 0
        errors = []
        for (idx, traj_path) in enumerate(traj_paths):
            if not silent:
                self.parent_window.progress_bar.setValue(idx)
            self.parent_window.log_output(f'  [{idx + 1}/{n_files}] {traj_path.name}\n')
            QApplication.processEvents()
            out_dir = traj_path.parent
            stem = traj_path.stem.replace('_trajectories', '')
            tracks_csv = out_dir / f'{stem}_tracks.csv'
            if not tracks_csv.is_file():
                errors.append(f'{traj_path.name}: no {tracks_csv.name} found')
                continue
            try:
                tracks_df = _pd().read_csv(tracks_csv)
            except Exception as e:
                errors.append(f'{traj_path.name}: could not read CSV ({e})')
                continue
            for col in ('track_id', 'frame', 'y', 'x'):
                if col not in tracks_df.columns:
                    errors.append(f"{traj_path.name}: CSV missing column '{col}'")
                    break
            else:
                metrics_df = self._compute_track_metrics(tracks_df, pixel_size, time_per_frame)
                if metrics_df.empty:
                    errors.append(f'{traj_path.name}: no tracks in CSV')
                    continue
                filtered_ids = self._apply_track_filters(metrics_df, params)
                filtered_metrics = metrics_df[metrics_df['track_id'].isin(filtered_ids)].copy()
                cell_id_per_track = None
                if 'Cell_ID' in tracks_df.columns and (not filtered_metrics.empty):
                    first_cell = tracks_df.groupby('track_id')['Cell_ID'].first()
                    cell_id_per_track = filtered_metrics['track_id'].map(first_cell)
                export_df = _pd().DataFrame()
                if not filtered_metrics.empty:
                    export_df['Cell_ID'] = cell_id_per_track if cell_id_per_track is not None else filtered_metrics['track_id']
                    export_df['TrackID'] = filtered_metrics['track_id'].values
                    export_df['trajectory_length'] = filtered_metrics['trajectory_length'].values.astype(int)
                    export_df['duration_frames'] = filtered_metrics['duration_frames'].values.astype(int)
                    export_df['gap_frames'] = filtered_metrics['gap_frames'].values.astype(int)
                    export_df['duration_min'] = np.round(filtered_metrics['duration_min'].values, 2)
                    export_df['first_frame'] = filtered_metrics['first_frame'].values.astype(int)
                    export_df['last_frame'] = filtered_metrics['last_frame'].values.astype(int)
                    export_df['displacement_px'] = np.round(filtered_metrics['displacement_px'].values, 0).astype(int)
                    export_df['displacement_um'] = np.round(filtered_metrics['displacement_um'].values, 0).astype(int)
                    export_df['path_length_px'] = np.round(filtered_metrics['path_length_px'].values, 0).astype(int)
                    export_df['path_length_um'] = np.round(filtered_metrics['path_length_um'].values, 0).astype(int)
                    mean_um_s = np.round(filtered_metrics['mean_speed_um_s'].values, 4)
                    straight_um_s = np.round(filtered_metrics['straight_line_speed_um_s'].values, 4)
                    export_df['mean_speed_px_frame'] = np.round(filtered_metrics['mean_speed_px_frame'].values, 2)
                    export_df['mean_speed_um_s'] = mean_um_s
                    export_df['mean_speed_um_min'] = np.round(mean_um_s * 60.0, 2)
                    export_df['straight_line_speed_px_frame'] = np.round(filtered_metrics['straight_line_speed_px_frame'].values, 2)
                    export_df['straight_line_speed_um_s'] = straight_um_s
                    export_df['straight_line_speed_um_min'] = np.round(straight_um_s * 60.0, 2)
                    export_df['straightness'] = np.round(filtered_metrics['straightness'].values, 2)
                else:
                    export_df = _pd().DataFrame(columns=['Cell_ID', 'TrackID', 'trajectory_length', 'duration_frames', 'gap_frames', 'duration_min', 'first_frame', 'last_frame', 'displacement_px', 'displacement_um', 'path_length_px', 'path_length_um', 'mean_speed_px_frame', 'mean_speed_um_s', 'mean_speed_um_min', 'straight_line_speed_px_frame', 'straight_line_speed_um_s', 'straight_line_speed_um_min', 'straightness'])
                out_excel = out_dir / f'{stem}_filtered_tracks.xlsx'
                try:
                    with _pd().ExcelWriter(out_excel, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name='Filtered tracks', index=False)
                    done += 1
                    if hasattr(self, 'parent_window') and hasattr(self.parent_window, 'log_output'):
                        self.parent_window.log_output(f'✓ Exported {len(export_df)} tracks to {out_excel.name}\n')
                except Exception as e:
                    errors.append(f'{traj_path.name}: could not write Excel ({e})')
        if not silent:
            self.parent_window.progress_bar.setValue(n_files)
            QTimer.singleShot(2000, lambda : self.parent_window.progress_bar.setVisible(False))
        self.parent_window.log_output(f'✓ Export complete: {done}/{n_files} file(s) written.\n')
        if not silent:
            msg = f'Exported {done} file(s).'
            if errors:
                msg += '\n\nErrors:\n' + '\n'.join(errors[:15])
                if len(errors) > 15:
                    msg += f'\n... and {len(errors) - 15} more.'
            show_message_dialog(self._main_window(), 'Export filtered tracks', msg, kind='info')
        return done

    def _open_trajectory_preview_window(self, traj_path: Path | None, index: int=0, *, frames_rgb: np.ndarray | None=None, tracks_df: Any=None, original_path: Path | None=None, display_name: str | None=None, mask_path: Path | None=None, stack_images: np.ndarray | None=None, masks_tracked: np.ndarray | None=None) -> None:
        """Load saved trajectories (if needed) and open the shared tracking preview dialog."""
        tracks_csv: Path | None = None
        if frames_rgb is not None:
            frames_arr = np.asarray(frames_rgb, dtype=np.uint8)
        elif traj_path is not None:
            try:
                stack = safe_imread(str(traj_path))
            except Exception as e:
                show_message_dialog(self._main_window(), 'Load Error', f'Could not load trajectories file:\n{traj_path}\n\n{str(e)}', kind='error')
                return
            frames_arr = self._stack_to_frames_rgb(stack)
            if frames_arr is None:
                show_message_dialog(self._main_window(), 'Preview', f'Expected 3D (T, Y, X) or 4D overlay, got {stack.ndim} dimensions.', kind='warning')
                return
            (tracks_csv, _overlay_path, original_path) = self._resolve_from_trajectories_path(traj_path)
            if mask_path is None:
                mask_path = traj_path
        else:
            show_message_dialog(self._main_window(), 'Preview', 'No trajectory data to display.', kind='warning')
            return
        if traj_path is not None:
            (ps, tp) = resolve_calibration_from_trajectories_path(traj_path)
            if ps > 0 and hasattr(self, 'filter_pixel_size'):
                self.filter_pixel_size.setValue(ps)
            if tp > 0:
                self._apply_filter_time_per_frame_from_metadata(tp)
        self._open_tracking_preview_dialog(frames_rgb=frames_arr, tracks_df=tracks_df, original_path=original_path, display_name=display_name or (traj_path.name if traj_path else 'Preview'), mask_path=mask_path, traj_path=traj_path, tracks_csv=tracks_csv, initial_frame=index, stack_images=stack_images, masks_tracked=masks_tracked)

    def _open_tracking_preview_dialog(self, *, frames_rgb: np.ndarray, tracks_df: Any=None, original_path: Path | None=None, display_name: str='Tracking preview', mask_path: Path | None=None, traj_path: Path | None=None, tracks_csv: Path | None=None, initial_frame: int=0, stack_images: np.ndarray | None=None, masks_input: np.ndarray | None=None, masks_tracked: np.ndarray | None=None, edges_df: Any=None, founder_by_track: dict[int, int] | None=None, lineage_tables: dict[str, Any] | None=None, is_lineage_preview: bool=False) -> None:
        """Embedded or floating preview: canvas + frame nav + parameters/filters + redraw overlay."""
        mount = getattr(self, '_module_preview_mount', None)
        embed = mount is not None
        self._dismiss_embedded_tracking_preview()
        frames = np.asarray(frames_rgb, dtype=np.uint8)
        n_frames = int(frames.shape[0])
        tracks_full = None
        if _is_pandas_dataframe(tracks_df):
            tracks_full = tracks_df.copy()
        preview_state: dict[str, Any] = {'frames': frames, 'tracks_df': tracks_full if tracks_full is not None else tracks_df, 'tracks_df_full': tracks_full if tracks_full is not None else tracks_df, 'original_path': original_path, 'mask_path': mask_path, 'traj_path': traj_path, 'tracks_csv': tracks_csv, 'stack_images': np.asarray(stack_images) if stack_images is not None and getattr(stack_images, 'ndim', 0) == 3 else None, 'masks_input': np.asarray(masks_input) if masks_input is not None and getattr(masks_input, 'ndim', 0) == 3 else None, 'masks_tracked': np.asarray(masks_tracked) if masks_tracked is not None and getattr(masks_tracked, 'ndim', 0) == 3 else None, 'preview_frame_limit': getattr(self, '_last_preview_frame_limit', TRACKING_PREVIEW_FRAME_LIMIT), 'edges_df': edges_df, 'founder_by_track': founder_by_track or {}, 'lineage_tables': lineage_tables or {}, 'is_lineage_preview': bool(is_lineage_preview), 'selected_lineage_track_id': None, 'overlay_layers': {'masks': True, 'paths': True, 'ids': False, 'centers': False, 'untracked': bool(is_lineage_preview)}}
        reapply_btn = QPushButton('Redraw filtered tracks')
        style_module_primary_button(reapply_btn)
        reapply_btn.setMinimumHeight(32)
        reapply_btn.setToolTip('Redraw the trajectory image so only tracks that pass the current filters are shown (colored masks and paths).\n\nChanging filter spins updates the metrics table immediately but does not redraw the image — use this when you are happy with the filter settings.\n\nCan take a few seconds on long previews; tune filters via the table first, then redraw.\n\nDoes not re-run Trackastra — use Re-track after changing Model or Mode.')
        retrack_btn = QPushButton('Re-track')
        style_module_primary_button(retrack_btn)
        retrack_btn.setMinimumHeight(32)
        retrack_btn.setToolTip('Re-run Trackastra with the current Model and Mode on this mask (in memory, no files written).\n\nAfter re-tracking, click «Redraw filtered tracks» to refresh the image with your filters.')
        open_inspector_btn: QPushButton | None = None
        lineage_frame_label: QLabel | None = None
        if bool(is_lineage_preview):
            open_inspector_btn = QPushButton('Lineage review')
            style_module_primary_button(open_inspector_btn)
            open_inspector_btn.setMinimumHeight(32)
            open_inspector_btn.setToolTip('Open Lineage review for division review, sortable tables, plots, and manual correction of orange untracked cells.')
        nav = QWidget()
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(0, 0, 0, 0)
        nav_lay.setSpacing(8)
        nav_lay.addWidget(QLabel('Frame:'))
        frame_slider = QSlider(Qt.Orientation.Horizontal)
        frame_slider.setMinimum(0)
        frame_slider.setMaximum(max(0, n_frames - 1))
        frame_slider.setValue(max(0, min(initial_frame, n_frames - 1)))
        frame_label = QLabel('0')
        frame_label.setFixedWidth(40)
        loop_btn = QPushButton('Play')
        loop_btn.setMinimumHeight(32)
        loop_btn.setCheckable(True)
        loop_btn.setToolTip('Loop through preview frames. Use the slider to scrub manually.')
        frame_loop_timer = QTimer(self)
        frame_loop_timer.setInterval(180)
        nav_lay.addWidget(frame_slider, 1)
        nav_lay.addWidget(frame_label)
        nav_lay.addWidget(loop_btn)
        nav_lay.addWidget(retrack_btn)
        nav_lay.addWidget(reapply_btn)
        if open_inspector_btn is not None:
            nav_lay.addWidget(open_inspector_btn)
        if bool(is_lineage_preview):
            lineage_frame_label = QLabel('')
            lineage_frame_label.setStyleSheet('color: #8fbfff; font-weight: 600;')
            lineage_frame_label.setMinimumWidth(280)
            lineage_frame_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            nav_lay.addWidget(lineage_frame_label)
        settings_host = QWidget()
        settings_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        settings_lay = QVBoxLayout(settings_host)
        settings_lay.setContentsMargins(0, 0, 0, 0)
        settings_lay.setSpacing(3)
        tracking_params_section = CollapsibleSection('Tracking parameters', self._tracking_params_panel, parent=settings_host, expanded=False, compact=True)
        tracking_params_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        settings_lay.addWidget(tracking_params_section)
        filter_tracks_section = CollapsibleSection('Filter tracks', self._filter_tracks_panel, parent=settings_host, expanded=False, compact=True)
        filter_tracks_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        settings_lay.addWidget(filter_tracks_section)
        overlay_layers_host = QWidget()
        overlay_layers_lay = QHBoxLayout(overlay_layers_host)
        overlay_layers_lay.setContentsMargins(10, 6, 10, 6)
        overlay_layers_lay.setSpacing(12)
        overlay_layers_lay.addWidget(QLabel('Overlay:'))
        overlay_layer_checks: dict[str, QCheckBox] = {}
        for key, label, tip in (
            ('masks', 'Masks', 'Blue/founder-color cell mask overlay. This is the main lineage/track color cue.'),
            ('paths', 'Paths', 'Thin trajectory history up to the current frame.'),
            ('ids', 'Track IDs', 'Debug labels from Trackastra. They are arbitrary trajectory handles, not cell counts, lineage order, or biological IDs.'),
            ('centers', 'Centers', 'Small green centroid markers: Trackastra detection coordinates, not lineage identity.'),
            ('untracked', 'Untracked', 'Orange segmented objects that Trackastra did not link.'),
        ):
            cb = QCheckBox(label)
            cb.setChecked(bool(preview_state['overlay_layers'].get(key, False)))
            cb.setToolTip(tip)
            if key == 'untracked' and not bool(is_lineage_preview):
                cb.setVisible(False)
            overlay_layer_checks[key] = cb
            overlay_layers_lay.addWidget(cb)
        overlay_layers_lay.addStretch(1)
        settings_lay.addWidget(overlay_layers_host)
        preview_state['overlay_layer_checks'] = overlay_layer_checks
        metrics_table_host = QWidget()
        metrics_table_lay = QVBoxLayout(metrics_table_host)
        metrics_table_lay.setContentsMargins(0, 0, 0, 0)
        metrics_table_lay.setSpacing(4)
        metrics_table_hint = QLabel('Pass and metrics here update live when you change filters (the image above is not redrawn). Click «Redraw filtered tracks» under the frame slider to update the trajectory overlay.\n\nFilters apply to each full trajectory. In frame = whether that track appears on the frame you are viewing. Double-click a row to jump to that track (yellow path and cell highlight on the current overlay).')
        metrics_table_hint.setWordWrap(True)
        metrics_table_lay.addWidget(metrics_table_hint)
        metrics_table = QTableWidget()
        metrics_table.setAlternatingRowColors(True)
        metrics_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        metrics_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        metrics_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        metrics_table.setMinimumHeight(100)
        metrics_table.setMaximumHeight(180)
        metrics_table.setMinimumWidth(0)
        metrics_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preview_state['metrics_table_column_fit_filter'] = _MetricsTableColumnFitFilter(metrics_table, host=metrics_table_host)
        _col_fit = preview_state['metrics_table_column_fit_filter']
        metrics_table.installEventFilter(_col_fit)
        metrics_table_host.installEventFilter(_col_fit)
        metrics_table_lay.addWidget(metrics_table, 1)
        metrics_table_section = CollapsibleSection('Track metrics (live Pass)', metrics_table_host, parent=settings_host, expanded=True, compact=True)
        metrics_table_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        metrics_table_section.setVisible(False)

        def _refit_metrics_columns(*_a: object) -> None:
            _fit_preview_metrics_table_columns(metrics_table)
        try:
            metrics_table_section.toggled.connect(lambda _exp: _refit_metrics_columns())
        except Exception:
            pass
        settings_lay.addWidget(metrics_table_section)
        preview_state['metrics_table'] = metrics_table
        preview_state['metrics_table_section'] = metrics_table_section
        preview_state['refit_metrics_columns'] = _refit_metrics_columns
        preview_state['highlight_track_id'] = None
        self._tracking_params_panel.show()
        self._filter_tracks_panel.show()
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        settings_scroll.setMinimumHeight(130 if embed else 180)
        settings_scroll.setMaximumHeight(420 if embed else 520)
        settings_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_scroll.setWidget(settings_host)
        preview_state['settings_scroll'] = settings_scroll

        def _scroll_preview_settings_to_metrics() -> None:
            """Scroll the preview settings panel so the metrics table is visible."""
            scroll = preview_state.get('settings_scroll')
            section = preview_state.get('metrics_table_section')
            if scroll is None or section is None or (not section.isVisible()):
                return
            section.set_expanded(True)
            scroll.ensureWidgetVisible(section, 24, 24)
            refit = preview_state.get('refit_metrics_columns')
            if callable(refit):
                QTimer.singleShot(0, refit)

        def _tracks_for_filtering() -> Any:
            full = preview_state.get('tracks_df_full')
            if _is_pandas_dataframe(full) and (not full.empty):
                return full
            td = preview_state.get('tracks_df')
            if _is_pandas_dataframe(td) and (not td.empty):
                return td
            csv_p = preview_state.get('tracks_csv')
            if csv_p is not None and Path(csv_p).is_file():
                return _pd().read_csv(csv_p)
            return _pd().DataFrame()

        def _track_color_map(tracks_df_in: Any) -> dict[int, tuple[int, int, int]]:
            import matplotlib.pyplot as plt

            if not _is_pandas_dataframe(tracks_df_in) or 'track_id' not in tracks_df_in.columns:
                return {}
            tids = [int(t) for t in tracks_df_in['track_id'].dropna().unique()]
            if bool(preview_state.get('is_lineage_preview')) and preview_state.get('founder_by_track'):
                cmap = plt.get_cmap('tab20')
                founder_by = {int(k): int(v) for k, v in dict(preview_state.get('founder_by_track') or {}).items()}
                founders = sorted(set(founder_by.get(t, t) for t in tids))
                founder_colors = {}
                for i, fid in enumerate(founders):
                    c = cmap((i % 20) / 20.0)
                    founder_colors[int(fid)] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                return {t: founder_colors.get(founder_by.get(t, t), (80, 160, 220)) for t in tids}
            cmap = plt.get_cmap('gist_ncar')
            return {
                t: tuple(int(v * 255) for v in cmap(i / max(len(tids), 1))[:3])
                for i, t in enumerate(tids)
            }

        def _lineage_meta_for_track(track_id: int) -> dict[str, Any]:
            tables = preview_state.get('lineage_tables') or {}
            meta = tables.get('lineage_tracks_meta') if isinstance(tables, dict) else None
            if _is_pandas_dataframe(meta) and 'track_id' in meta.columns:
                sub = meta[meta['track_id'].astype(int) == int(track_id)]
                if len(sub):
                    return dict(sub.iloc[0])
            return {}

        def _build_tracking_hover_data(frame_idx: int, frame_shape: tuple[int, ...], tracks_df_in: Any, masks_input_arr: Any, masks_tracked_arr: Any) -> list[tuple[float, float, str, float]]:
            hover: list[tuple[float, float, str, float]] = []
            if _is_pandas_dataframe(tracks_df_in) and {'track_id', 'frame', 'y', 'x'}.issubset(tracks_df_in.columns):
                current = tracks_df_in[tracks_df_in['frame'].astype(int) == int(frame_idx)]
                for row in current.itertuples(index=False):
                    tid = int(getattr(row, 'track_id'))
                    y = float(getattr(row, 'y'))
                    x = float(getattr(row, 'x'))
                    dx, dy = TrackingTab._preview_display_xy(y, x, frame_shape)
                    meta = _lineage_meta_for_track(tid)
                    tooltip = [
                        f'Track {tid}',
                        f'Frame {int(frame_idx) + 1}',
                        f'Centroid: y={y:.1f}, x={x:.1f}',
                    ]
                    if meta:
                        if meta.get('lineage_id') is not None:
                            tooltip.append(f"Lineage: {meta.get('lineage_id')}")
                        if meta.get('generation') is not None:
                            tooltip.append(f"Generation: {meta.get('generation')}")
                        if meta.get('parent_track_id') is not None:
                            tooltip.append(f"Parent: {meta.get('parent_track_id')}")
                    tooltip.append('Color = track/founder lineage color')
                    hover.append((float(dx), float(dy), '\n'.join(tooltip), 18.0))
            layers = preview_state.get('overlay_layers') or {}
            if (
                bool(layers.get('untracked', False))
                and masks_input_arr is not None
                and masks_tracked_arr is not None
                and getattr(masks_input_arr, 'ndim', 0) == 3
                and getattr(masks_tracked_arr, 'ndim', 0) == 3
                and frame_idx < int(masks_input_arr.shape[0])
                and frame_idx < int(masks_tracked_arr.shape[0])
                and masks_input_arr.shape[1:] == masks_tracked_arr.shape[1:]
            ):
                input_frame = masks_input_arr[frame_idx]
                tracked_frame = masks_tracked_arr[frame_idx]
                for raw_label in np.unique(input_frame):
                    raw_label = int(raw_label)
                    if raw_label <= 0:
                        continue
                    region = input_frame == raw_label
                    if np.any(region & (tracked_frame > 0)):
                        continue
                    ys, xs = np.nonzero(region)
                    if len(ys) == 0:
                        continue
                    y = float(np.mean(ys))
                    x = float(np.mean(xs))
                    dx, dy = TrackingTab._preview_display_xy(y, x, frame_shape)
                    tooltip = (
                        f'Untracked segmented object\n'
                        f'Frame {int(frame_idx) + 1}, mask label {raw_label}\n'
                        f'Area: {int(region.sum())} px\n'
                        'Orange = present in mask, not linked by Trackastra'
                    )
                    hover.append((float(dx), float(dy), tooltip, 20.0))
            return hover

        def _compose_preview_frame(frame_idx: int, tracks_df_in: Any) -> tuple[np.ndarray, list[tuple[float, float, str, float]]]:
            layers = preview_state.get('overlay_layers') or {}
            images = preview_state.get('stack_images')
            masks_tr = preview_state.get('masks_tracked')
            masks_in = preview_state.get('masks_input')
            if images is None or getattr(images, 'ndim', 0) != 3 or frame_idx >= int(images.shape[0]):
                return np.asarray(preview_state['frames'][frame_idx], dtype=np.uint8), []
            img = np.asarray(images[frame_idx], dtype=np.float32)
            finite = img[np.isfinite(img)]
            if finite.size == 0:
                base = np.zeros(img.shape, dtype=np.uint8)
            else:
                lo, hi = float(finite.min()), float(finite.max())
                if hi <= lo:
                    hi = lo + 1.0
                base = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            vis_frame = np.stack([base, base, base], axis=-1)
            height, width = base.shape[:2]
            colors = _track_color_map(tracks_df_in)
            use_masks = masks_tr is not None and getattr(masks_tr, 'ndim', 0) == 3 and frame_idx < int(masks_tr.shape[0]) and masks_tr.shape[1] == height and masks_tr.shape[2] == width
            tracked_region_frame = np.zeros((height, width), dtype=bool)
            current = tracks_df_in[tracks_df_in['frame'].astype(int) == int(frame_idx)] if _is_pandas_dataframe(tracks_df_in) and {'track_id', 'frame'}.issubset(tracks_df_in.columns) else _pd().DataFrame()
            if use_masks and bool(layers.get('masks', True)):
                mask_frame = masks_tr[frame_idx]
                for row in current.itertuples(index=False):
                    tid = int(getattr(row, 'track_id'))
                    color = colors.get(tid, (80, 160, 220))
                    region = None
                    if hasattr(row, 'y') and hasattr(row, 'x'):
                        region = TrackingTab._mask_region_at_centroid(mask_frame, int(getattr(row, 'y')), int(getattr(row, 'x')))
                    if region is None:
                        region = mask_frame == tid
                    if region is None or not np.any(region):
                        continue
                    tracked_region_frame |= region
                    vis_frame[region, 0] = np.clip(vis_frame[region, 0] * 0.5 + color[0] * 0.5, 0, 255).astype(np.uint8)
                    vis_frame[region, 1] = np.clip(vis_frame[region, 1] * 0.5 + color[1] * 0.5, 0, 255).astype(np.uint8)
                    vis_frame[region, 2] = np.clip(vis_frame[region, 2] * 0.5 + color[2] * 0.5, 0, 255).astype(np.uint8)
            elif use_masks:
                mask_frame = masks_tr[frame_idx]
                for row in current.itertuples(index=False):
                    if hasattr(row, 'y') and hasattr(row, 'x'):
                        region = TrackingTab._mask_region_at_centroid(mask_frame, int(getattr(row, 'y')), int(getattr(row, 'x')))
                        if region is not None:
                            tracked_region_frame |= region
            if bool(layers.get('untracked', False)) and masks_in is not None and use_masks and getattr(masks_in, 'ndim', 0) == 3 and frame_idx < int(masks_in.shape[0]) and masks_in.shape[1] == height and masks_in.shape[2] == width:
                input_frame = masks_in[frame_idx]
                for raw_label in np.unique(input_frame):
                    raw_label = int(raw_label)
                    if raw_label <= 0:
                        continue
                    raw_region = input_frame == raw_label
                    if np.any(raw_region & tracked_region_frame):
                        continue
                    vis_frame[raw_region, 0] = np.clip(vis_frame[raw_region, 0] * 0.25 + 255 * 0.75, 0, 255).astype(np.uint8)
                    vis_frame[raw_region, 1] = np.clip(vis_frame[raw_region, 1] * 0.25 + 150 * 0.75, 0, 255).astype(np.uint8)
                    vis_frame[raw_region, 2] = np.clip(vis_frame[raw_region, 2] * 0.25, 0, 255).astype(np.uint8)
            labels_to_draw: list[tuple[str, int, int]] = []
            if _is_pandas_dataframe(tracks_df_in) and {'track_id', 'frame', 'y', 'x'}.issubset(tracks_df_in.columns):
                if bool(layers.get('paths', True)):
                    for tid in tracks_df_in.loc[tracks_df_in['frame'].astype(int) <= int(frame_idx), 'track_id'].dropna().unique():
                        tid = int(tid)
                        points_df = tracks_df_in[(tracks_df_in['track_id'].astype(int) == tid) & (tracks_df_in['frame'].astype(int) <= int(frame_idx))].sort_values('frame')
                        if len(points_df) < 2:
                            continue
                        pts = points_df[['y', 'x']].values.astype(int)
                        pts[:, 0] = np.clip(pts[:, 0], 0, height - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, width - 1)
                        TrackingTab._draw_polyline_on_frame(vis_frame, pts, color=colors.get(tid, (80, 160, 220)), thickness=1)
                for row in current.itertuples(index=False):
                    tid = int(getattr(row, 'track_id'))
                    y = int(getattr(row, 'y')) if hasattr(row, 'y') else None
                    x = int(getattr(row, 'x')) if hasattr(row, 'x') else None
                    if y is None or x is None or not (0 <= y < height and 0 <= x < width):
                        continue
                    color = colors.get(tid, (80, 160, 220))
                    if bool(layers.get('centers', False)):
                        s = 2
                        y0, y1 = max(0, y - s), min(height, y + s + 1)
                        x0, x1 = max(0, x - s), min(width, x + s + 1)
                        vis_frame[y0:y1, x0:x1, :] = (0, 255, 120)
                    if bool(layers.get('ids', True)):
                        labels_to_draw.append((str(tid), x + 5, y - 5))
            if labels_to_draw:
                try:
                    from PIL import Image
                    from bactomate.modules.label_render import draw_hq_labels

                    pil_img = Image.fromarray(vis_frame)
                    pil_img = draw_hq_labels(pil_img, labels_to_draw, font_size=12, scale=2, stroke_width=2)
                    vis_frame = np.asarray(pil_img)
                except Exception:
                    pass
            hover = _build_tracking_hover_data(frame_idx, vis_frame.shape, tracks_df_in, masks_in, masks_tr)
            return vis_frame, hover

        def _refresh_preview_metrics() -> None:
            tracks_in = _tracks_for_filtering()
            if not _is_pandas_dataframe(tracks_in) or tracks_in.empty:
                preview_state['metrics_df'] = _pd().DataFrame()
                return
            params = self._get_filter_params()
            preview_state['metrics_df'] = self._compute_track_metrics(tracks_in, params['pixel_size'], params['time_per_frame'])
        _refresh_preview_metrics()

        def _update_metrics_table(frame_idx: int) -> None:
            section = preview_state.get('metrics_table_section')
            table = preview_state.get('metrics_table')
            if section is None or table is None:
                return
            show = getattr(self, 'overlay_track_data_cb', None) is not None and self.overlay_track_data_cb.isChecked()
            section.setVisible(show)
            if not show:
                return
            metrics_df = preview_state.get('metrics_df')
            if not _is_pandas_dataframe(metrics_df) or metrics_df.empty:
                _refresh_preview_metrics()
                metrics_df = preview_state.get('metrics_df')
            tracks_in = _tracks_for_filtering()
            if not _is_pandas_dataframe(metrics_df) or metrics_df.empty:
                table.setRowCount(0)
                return
            params = self._get_filter_params()
            passing = self._apply_track_filters(metrics_df, params)
            self._populate_preview_metrics_table(table, frame_idx, tracks_in, metrics_df, passing, use_um=params['use_physical_units'], use_speed_physical=params['use_speed_physical'], selected_track_id=preview_state.get('highlight_track_id'))

        def _focus_track_in_preview(track_id: int) -> None:
            tracks_in = _tracks_for_filtering()
            prefer = int(frame_slider.value())
            target = TrackingTab._frame_index_for_track_focus(tracks_in, int(track_id), prefer=prefer)
            preview_state['highlight_track_id'] = int(track_id)
            frame_slider.blockSignals(True)
            frame_slider.setValue(target)
            frame_slider.blockSignals(False)
            frame_label.setText(str(target))
            dlg.set_frame_idx(target)
            _update_metrics_table(target)
            _update_lineage_panel_frame(target)
            dlg.refresh()
            centroid = TrackingTab._track_centroid_at_frame(tracks_in, int(track_id), target)
            if centroid is not None and preview_state.get('frames') is not None:
                frames_arr = np.asarray(preview_state['frames'])
                if frames_arr.ndim >= 3 and 0 <= target < frames_arr.shape[0]:
                    (y, x) = centroid
                    (dx, dy) = TrackingTab._preview_display_xy(float(y), float(x), frames_arr[target].shape)
                    dlg.canvas.center_on_image_point(dx, dy)

        def _on_metrics_table_double_clicked(row: int, _col: int) -> None:
            table = preview_state.get('metrics_table')
            if table is None or row < 0:
                return
            id_item = table.item(row, 1)
            if id_item is None:
                return
            tid = id_item.data(Qt.ItemDataRole.UserRole)
            if tid is None:
                try:
                    tid = int(id_item.text())
                except (TypeError, ValueError):
                    return
            preview_state['selected_lineage_track_id'] = int(tid)
            _focus_track_in_preview(int(tid))
            _refresh_lineage_plot()
        metrics_table.cellDoubleClicked.connect(_on_metrics_table_double_clicked)
        lineage_side_panel: QWidget | None = None
        lineage_table: QTableWidget | None = None
        lineage_metric_combo: QComboBox | None = None
        lineage_plot_notice: QLabel | None = None
        lineage_ax = None
        lineage_plot_canvas: Any = None

        def _lineage_points_table() -> Any:
            tables = preview_state.get('lineage_tables') or {}
            pts = tables.get('lineage_points') if isinstance(tables, dict) else None
            if _is_pandas_dataframe(pts):
                return pts
            return _pd().DataFrame()

        def _lineage_growth_table() -> Any:
            tables = preview_state.get('lineage_tables') or {}
            ts = tables.get('lineage_growth_timeseries') if isinstance(tables, dict) else None
            if _is_pandas_dataframe(ts):
                return ts
            return _pd().DataFrame()

        def _selected_lineage_id_from_preview() -> int | None:
            selected_tid = preview_state.get('selected_lineage_track_id')
            pts = _lineage_points_table()
            if selected_tid is None or not _is_pandas_dataframe(pts) or pts.empty:
                return None
            tid = int(selected_tid)
            if 'lineage_id' in pts.columns:
                row0 = pts[pts['track_id'] == tid].head(1)
                if len(row0):
                    lineage_id = row0.iloc[0].get('lineage_id')
                    if lineage_id is not None and (not _pd().isna(lineage_id)):
                        return int(lineage_id)
            return tid

        def _has_biological_lineage_measurement(pts: Any, growth_ts: Any) -> bool:
            if _is_pandas_dataframe(growth_ts) and (not growth_ts.empty):
                return True
            if not _is_pandas_dataframe(pts) or pts.empty:
                return False
            for col in pts.columns:
                cl = str(col).lower()
                if col in ('Area (µm²)', 'area', 'Area') or 'fi' in cl or 'foci' in cl or ('coloc' in cl):
                    return True
            return False

        def _set_lineage_plot_notice(visible: bool, text: str='') -> None:
            if lineage_plot_notice is None:
                return
            lineage_plot_notice.setText(text)
            lineage_plot_notice.setVisible(bool(visible))

        def _refresh_lineage_plot() -> None:
            if lineage_ax is None or lineage_plot_canvas is None:
                return
            lineage_ax.clear()
            choice = lineage_metric_combo.currentText() if lineage_metric_combo is not None else 'Lineage total area'
            pts = _lineage_points_table()
            growth_ts = _lineage_growth_table()
            selected_tid = preview_state.get('selected_lineage_track_id')
            lineage_id = _selected_lineage_id_from_preview()
            if choice == 'Lineage total area':
                if not _is_pandas_dataframe(growth_ts) or growth_ts.empty or 'total_area' not in growth_ts.columns:
                    lineage_ax.set_title('No lineage area timeseries')
                    _set_lineage_plot_notice(False)
                    lineage_plot_canvas.draw()
                    return
                sub = growth_ts.copy()
                if lineage_id is not None and 'lineage_id' in sub.columns:
                    filtered = sub[sub['lineage_id'] == int(lineage_id)]
                    if not filtered.empty:
                        sub = filtered
                for (lid, group) in sub.groupby('lineage_id'):
                    group = group.sort_values('frame')
                    lineage_ax.plot(group['frame'], group['total_area'], marker='o', markersize=2.5, label=f'lineage {lid}')
                lineage_ax.set_xlabel('Frame')
                lineage_ax.set_ylabel('Total area')
                title_suffix = f'lineage {lineage_id}' if lineage_id is not None else 'all lineages'
                lineage_ax.set_title(f'Lineage total area — {title_suffix}')
                if sub['lineage_id'].nunique() <= 8:
                    lineage_ax.legend(fontsize=7, loc='best')
                _set_lineage_plot_notice(False)
                lineage_plot_canvas.draw()
                return
            if not _is_pandas_dataframe(pts) or pts.empty or 'frame' not in pts.columns:
                lineage_ax.set_title('No lineage measurements')
                _set_lineage_plot_notice(False)
                lineage_plot_canvas.draw()
                return
            sub = pts.copy()
            if selected_tid is not None and 'track_id' in sub.columns:
                tid = int(selected_tid)
                if 'lineage_id' in sub.columns:
                    row0 = sub[sub['track_id'] == tid].head(1)
                    row_lineage_id = row0.iloc[0].get('lineage_id') if len(row0) else None
                    if row_lineage_id is not None and (not _pd().isna(row_lineage_id)):
                        sub = sub[sub['lineage_id'] == row_lineage_id]
                    else:
                        sub = sub[sub['track_id'] == tid]
                else:
                    sub = sub[sub['track_id'] == tid]
            ycol = None
            if choice in sub.columns:
                ycol = choice
            elif choice.startswith('y'):
                ycol = 'y' if 'y' in sub.columns else None
            elif choice == 'Area (µm²)':
                ycol = next((c for c in ('Area (µm²)', 'area', 'Area') if c in sub.columns), None)
            if ycol is None:
                ycol = 'y' if 'y' in sub.columns else None
            if ycol is None or sub.empty:
                lineage_ax.set_title('No plottable lineage column')
                _set_lineage_plot_notice(False)
                lineage_plot_canvas.draw()
                return
            position_only = ycol == 'y'
            for (tid, group) in sub.groupby('track_id'):
                group = group.sort_values('frame')
                lineage_ax.plot(group['frame'], group[ycol], marker='o', markersize=2.5, label=str(tid))
            lineage_ax.set_xlabel('Frame')
            lineage_ax.set_ylabel('Y position (px)' if position_only else str(ycol))
            lineage_ax.set_title(f"Track position — track {selected_tid or 'all'}" if position_only else f"Track/lineage {selected_tid or 'all'}")
            if 'track_id' in sub.columns and sub['track_id'].nunique() <= 8:
                lineage_ax.legend(fontsize=7, loc='best')
            if position_only:
                _set_lineage_plot_notice(True, 'Position preview only — not growth data. Join an Analyze workbook or run lineage tracking for area/FI plots.')
            else:
                _set_lineage_plot_notice(False)
            lineage_plot_canvas.draw()

        def _populate_lineage_side_panel(table: QTableWidget) -> None:
            tables = preview_state.get('lineage_tables') or {}
            meta = tables.get('lineage_tracks_meta') if isinstance(tables, dict) else None
            if not _is_pandas_dataframe(meta) or meta.empty:
                table.setRowCount(0)
                return
            preferred = ['track_id', 'lineage_id', 'parent_track_id', 'n_daughters', 'generation', 'start_frame', 'end_frame', 'n_points']
            cols = [c for c in preferred if c in meta.columns]
            if not cols:
                cols = list(meta.columns[:8])
            table.setSortingEnabled(False)
            table.setColumnCount(len(cols))
            table.setHorizontalHeaderLabels([str(c) for c in cols])
            table.setRowCount(len(meta))
            for (ri, row) in enumerate(meta[cols].itertuples(index=False)):
                track_id = None
                if 'track_id' in meta.columns:
                    try:
                        track_id = int(meta.iloc[ri]['track_id'])
                    except Exception:
                        track_id = None
                for (ci, col) in enumerate(cols):
                    item = SortableTrackingTableItem(getattr(row, col, ''))
                    if track_id is not None:
                        item.setData(Qt.ItemDataRole.UserRole, track_id)
                    table.setItem(ri, ci, item)
            table.setSortingEnabled(True)
            try:
                table.resizeColumnsToContents()
            except Exception:
                pass

        def _update_lineage_panel_frame(frame_idx: int) -> None:
            if lineage_frame_label is None:
                return
            pts = _lineage_points_table()
            n_tracks = 0
            if _is_pandas_dataframe(pts) and (not pts.empty) and ('frame' in pts.columns) and ('track_id' in pts.columns):
                n_tracks = int(pts.loc[pts['frame'].astype(int) == int(frame_idx), 'track_id'].nunique())
            edges = preview_state.get('edges_df')
            n_div = 0
            if _is_pandas_dataframe(edges) and {'parent_track_id', 'division_frame'}.issubset(edges.columns):
                try:
                    frame_edges = edges.loc[edges['division_frame'].astype(int) == int(frame_idx)]
                    n_div = int(frame_edges[['parent_track_id', 'division_frame']].drop_duplicates().shape[0])
                except Exception:
                    n_div = 0
            div_text = f" · {n_div} division event{'s' if n_div != 1 else ''}" if n_div else ' · no division events'
            lineage_frame_label.setText(f'Frame {int(frame_idx) + 1}: {n_tracks} tracks{div_text}')
        if bool(preview_state.get('is_lineage_preview')):
            _update_lineage_panel_frame(frame_slider.value())
            def _open_full_lineage_inspector() -> None:
                from bactomate.gui.lineage_inspector import LineageInspectorDialog, _sync_preview_result_from_inspector
                from bactomate.modules.lineage_integration import resolve_lineage_curation_path
                frames = preview_state.get('frames')
                if frames is None:
                    return
                mask_path = preview_state.get('mask_path')
                curation_path = resolve_lineage_curation_path(mask_path) if mask_path is not None else None
                dlg = LineageInspectorDialog(self._main_window(), frames_rgb=np.asarray(frames, dtype=np.uint8), tracks_df=preview_state.get('tracks_df'), edges_df=preview_state.get('edges_df'), lineage_tables=preview_state.get('lineage_tables') or {}, stack_images=preview_state.get('stack_images'), masks_input=preview_state.get('masks_input'), masks_tracked=preview_state.get('masks_tracked'), founder_by_track=preview_state.get('founder_by_track') or {}, curation_path=curation_path, mask_path=mask_path, original_path=preview_state.get('original_path'), title='Lineage review')
                dlg.exec()
                tables = preview_state.get('lineage_tables') or {}
                curated_tracks = tables.pop('_tracks_df_curated', None)
                curated_edges = tables.pop('_edges_df_curated', None)
                if curated_tracks is not None:
                    preview_state['tracks_df'] = curated_tracks
                    preview_state['tracks_df_full'] = curated_tracks
                if curated_edges is not None:
                    preview_state['edges_df'] = curated_edges
                last = getattr(self, '_last_lineage_preview_result', None)
                if last is not None and getattr(last, 'is_lineage_preview', False):
                    _sync_preview_result_from_inspector(last, dlg)
            if open_inspector_btn is not None:
                open_inspector_btn.clicked.connect(_open_full_lineage_inspector)

        def apply_filters() -> None:
            tracks_in = _tracks_for_filtering()
            if not _is_pandas_dataframe(tracks_in) or tracks_in.empty:
                show_message_dialog(self._main_window(), 'Redraw filtered tracks', 'No track data to filter.', kind='warning')
                return
            orig = preview_state.get('original_path')
            if orig is None or not Path(orig).is_file():
                mp = preview_state.get('mask_path')
                if mp is not None:
                    orig = self._resolve_original_for_mask(Path(mp))
                    preview_state['original_path'] = orig
            for col in ('track_id', 'frame', 'y', 'x'):
                if col not in tracks_in.columns:
                    show_message_dialog(self._main_window(), 'Invalid tracks', f'Missing column: {col}', kind='warning')
                    return
            params = self._get_filter_params()
            metrics_df = preview_state.get('metrics_df')
            if not _is_pandas_dataframe(metrics_df) or metrics_df.empty:
                metrics_df = self._compute_track_metrics(tracks_in, params['pixel_size'], params['time_per_frame'])
                preview_state['metrics_df'] = metrics_df
            if metrics_df.empty:
                return
            filtered_ids = self._apply_track_filters(metrics_df, params)
            filtered_df = tracks_in[tracks_in['track_id'].isin(filtered_ids)].copy()
            images = preview_state.get('stack_images')
            if images is None or getattr(images, 'ndim', 0) != 3:
                if orig is None or not Path(orig).is_file():
                    show_message_dialog(self._main_window(), 'Redraw filtered tracks', 'Original phase/stack image not found — needed to redraw filtered trajectories.', kind='warning')
                    return
                images = safe_imread(str(orig))
            else:
                images = np.asarray(images)
            if getattr(images, 'ndim', 0) != 3:
                return
            n_show = int(preview_state['frames'].shape[0])
            if images.shape[0] > n_show:
                images = images[:n_show]
            masks_tr = preview_state.get('masks_tracked')
            if masks_tr is not None:
                masks_tr = np.asarray(masks_tr)
                if masks_tr.shape[0] > n_show:
                    masks_tr = masks_tr[:n_show]
            preview_state['frames'] = self._draw_filtered_trajectory_overlay(images, filtered_df, masks_tracked=masks_tr)
            preview_state['tracks_df'] = filtered_df
            preview_state['metrics_df'] = metrics_df
            n_tr = filtered_df['track_id'].nunique() if not filtered_df.empty else 0
            dlg.info_label.setText(f'{display_name} — {n_tr} tracks after filter (colored by track ID on masks)')
            nf = preview_state['frames'].shape[0]
            frame_slider.setMaximum(max(0, nf - 1))
            if frame_slider.value() >= nf:
                frame_slider.setValue(max(0, nf - 1))
            _update_metrics_table(frame_slider.value())
            dlg.refresh()

        def retrack_and_filter() -> None:
            if getattr(self, 'worker', None) is not None and self.worker.isRunning():
                show_message_dialog(self._main_window(), 'Re-track', 'Tracking is already running — wait for it to finish.', kind='info')
                return

            def _after_retrack() -> None:
                nf = int(preview_state['frames'].shape[0])
                frame_slider.setMaximum(max(0, nf - 1))
                if frame_slider.value() >= nf:
                    frame_slider.setValue(max(0, nf - 1))
                if lineage_table is not None:
                    _populate_lineage_side_panel(lineage_table)
                    _update_lineage_panel_frame(frame_slider.value())
                    _refresh_lineage_plot()
                apply_filters()
            self._run_preview_retrack(preview_state, on_updated=_after_retrack)

        def _render(_state) -> RenderResult:
            fi = max(0, min(int(_state.frame_idx), preview_state['frames'].shape[0] - 1))
            _update_metrics_table(fi)
            tracks_in = _tracks_for_filtering()
            frame, hover_data = _compose_preview_frame(fi, tracks_in)
            highlight_tid = preview_state.get('highlight_track_id')
            if highlight_tid is not None:
                masks_tr = preview_state.get('masks_tracked')
                frame = TrackingTab._apply_selected_track_halo(frame, fi, int(highlight_tid), np.asarray(masks_tr) if masks_tr is not None else None, tracks_in)
            (scaled, _) = scale_for_preview(frame, frame)
            pixmap = to_qpixmap(scaled)
            n_tr = 0
            tdf = preview_state.get('tracks_df')
            if _is_pandas_dataframe(tdf) and (not tdf.empty) and ('track_id' in tdf.columns):
                n_tr = int(tdf['track_id'].nunique())
            info = f'{n_tr} tracks in overlay'
            if getattr(self, 'overlay_track_data_cb', None) and self.overlay_track_data_cb.isChecked():
                metrics_df = preview_state.get('metrics_df')
                if _is_pandas_dataframe(metrics_df) and (not metrics_df.empty):
                    params = self._get_filter_params()
                    passing = self._apply_track_filters(metrics_df, params)
                    n_total = int(metrics_df['track_id'].nunique())
                    in_frame = 0
                    tdf_full = _tracks_for_filtering()
                    if _is_pandas_dataframe(tdf_full) and 'frame' in tdf_full.columns:
                        in_frame = int(tdf_full.loc[tdf_full['frame'] == fi, 'track_id'].nunique())
                    info += f' · table: {n_total} tracks ({in_frame} in this frame), {len(passing)} pass / {n_total - len(passing)} fail'
            info += ' · colors = track ID on masks'
            if highlight_tid is not None:
                in_this = False
                tdf_full = _tracks_for_filtering()
                if _is_pandas_dataframe(tdf_full) and 'frame' in tdf_full.columns:
                    in_this = bool(((tdf_full['track_id'] == int(highlight_tid)) & (tdf_full['frame'] == fi)).any())
                info += f' · selected track {int(highlight_tid)}'
                if not in_this:
                    info += ' (not in this frame)'
            if preview_state.get('stack_images') is not None:
                info += ' · preview resolution (downsampled)'
            return RenderResult(pixmap=pixmap, image_size=(int(scaled.shape[1]), int(scaled.shape[0])), title_text=f"{display_name} — frame {fi + 1}/{preview_state['frames'].shape[0]}", info_text=info, foci_hover_data=hover_data)

        def _on_overlay_track_data_toggled(_checked: bool) -> None:
            section = preview_state.get('metrics_table_section')
            if section is not None:
                section.setVisible(bool(_checked))
                if _checked:
                    section.set_expanded(True)
            _update_metrics_table(frame_slider.value())
            if _checked:
                QTimer.singleShot(0, _scroll_preview_settings_to_metrics)
                QTimer.singleShot(80, _scroll_preview_settings_to_metrics)
            dlg.refresh()

        def _on_filter_spin_changed() -> None:
            preview_state.pop('metrics_df', None)
            _refresh_preview_metrics()
            _update_metrics_table(frame_slider.value())
        if hasattr(self, 'overlay_track_data_cb'):
            self.overlay_track_data_cb.stateChanged.connect(_on_overlay_track_data_toggled)
        for _spin in (self.filter_min_length, self.filter_max_length, self.filter_min_duration, self.filter_max_duration, self.filter_min_gaps, self.filter_max_gaps, self.filter_min_displacement, self.filter_max_displacement, self.filter_min_path_length, self.filter_max_path_length, self.filter_min_speed, self.filter_max_speed, self.filter_min_straightness, self.filter_pixel_size, self.filter_time_per_frame):
            _spin.valueChanged.connect(lambda _v: _on_filter_spin_changed())
        self.filter_time_per_frame_unit.currentTextChanged.connect(lambda _t: _on_filter_spin_changed())
        reapply_btn.clicked.connect(apply_filters)
        retrack_btn.clicked.connect(retrack_and_filter)
        dlg_parent = mount if embed else self._main_window()
        dlg = PreviewDialogBase(dlg_parent, title=display_name, post_canvas_controls=nav, navigation_controls=None, top_controls=settings_scroll, top_controls_title='', top_controls_expanded=False, lut_expanded=False, primary_action_label='Redraw filtered tracks', primary_action_callback=apply_filters, render_callback=_render, minimum_size=(200, 200) if embed else (900, 700), side_controls=None, embedded=embed, show_close_button=not embed, hide_title_and_builtin_info=embed, info_row_above_navigation=False)
        for _key, _cb in (preview_state.get('overlay_layer_checks') or {}).items():
            _cb.stateChanged.connect(lambda _state, key=_key, cb=_cb: (preview_state['overlay_layers'].__setitem__(key, cb.isChecked()), dlg.refresh()))
        dlg.set_primary_action(apply_filters, label='Redraw filtered tracks', visible=False)
        try:
            dlg.lut_section.setVisible(False)
        except Exception:
            pass
        dlg.set_frame_idx(frame_slider.value())
        dlg.info_label.setText(f'{display_name} — cell outlines colored by track ID (from tracked masks, not raw mask labels)')

        def _on_frame_changed(idx: int) -> None:
            dlg.set_frame_idx(int(idx))
            frame_label.setText(str(idx))
            _update_metrics_table(int(idx))
            _update_lineage_panel_frame(int(idx))
            dlg.refresh()
        frame_slider.valueChanged.connect(_on_frame_changed)
        _on_frame_changed(frame_slider.value())

        def _advance_frame_loop() -> None:
            nf = max(1, int(preview_state['frames'].shape[0]))
            frame_slider.setValue((int(frame_slider.value()) + 1) % nf)

        def _toggle_frame_loop(checked: bool) -> None:
            if checked:
                loop_btn.setText('Pause')
                frame_loop_timer.start()
            else:
                loop_btn.setText('Play')
                frame_loop_timer.stop()
        frame_loop_timer.timeout.connect(_advance_frame_loop)
        loop_btn.toggled.connect(_toggle_frame_loop)

        def _on_closed(_result: int=0) -> None:
            frame_loop_timer.stop()
            loop_btn.setChecked(False)
            self._embedded_tracking_preview = None
            self._restore_tracking_preview_panels()
            surf = self._preview_surface
            surf = self._preview_surface
            if embed and surf is not None:
                try:
                    surf.clear_mount()
                except Exception:
                    pass
        dlg.finished.connect(_on_closed)
        self._embedded_tracking_preview = dlg
        if embed:
            mount_embedded_preview(mount, dlg, preview_surface=self._preview_surface)
            surf = self._preview_surface
            if surf is not None:
                try:
                    surf.set_preview_chrome_visible(False)
                except Exception:
                    pass
        else:
            dlg.show()

    @staticmethod
    def _stack_to_frames_rgb(stack: np.ndarray) -> np.ndarray | None:
        """Normalize a trajectory or overlay TIFF stack to (T, H, W, 3) uint8."""
        if stack.ndim == 3:
            (n_frames, h, w) = stack.shape
            frames_rgb = np.stack([np.stack([stack[i], stack[i], stack[i]], axis=-1) for i in range(n_frames)], axis=0)
        elif stack.ndim == 4:
            n_frames = stack.shape[0]
            frames_rgb = []
            for i in range(n_frames):
                frame = stack[i]
                if frame.ndim == 3:
                    if frame.shape[-1] in (1, 3, 4):
                        f = frame[:, :, :3] if frame.shape[-1] >= 3 else np.stack([frame.squeeze(-1)] * 3, axis=-1)
                    elif frame.shape[0] in (1, 3, 4):
                        frame = np.transpose(frame, (1, 2, 0))
                        f = frame[:, :, :3] if frame.shape[-1] >= 3 else np.stack([frame.squeeze(-1)] * 3, axis=-1)
                    else:
                        f = np.stack([frame] * 3, axis=-1)
                else:
                    f = np.stack([frame] * 3, axis=-1)
                frames_rgb.append(f)
            frames_rgb = np.array(frames_rgb, dtype=np.uint8)
        else:
            return None
        return frames_rgb

    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (includes path for workflow replay)."""
        d = get_from_widgets([('model', self.model_combo), ('mode', self.mode_combo), ('create_vis', self.create_vis_cb), ('overlay_track_data', self.overlay_track_data_cb), ('filter_pixel_size', self.filter_pixel_size), ('filter_time_per_frame', self.filter_time_per_frame), ('filter_time_per_frame_unit', self.filter_time_per_frame_unit)])
        path_txt = (self.file_path.text() or '').strip()
        if path_txt:
            d['file_path'] = path_txt
        d['performed'] = getattr(self, '_performed_this_session', False)
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (includes path for workflow replay)."""
        apply_to_widgets(data, [('model', self.model_combo), ('mode', self.mode_combo), ('create_vis', self.create_vis_cb), ('overlay_track_data', self.overlay_track_data_cb), ('filter_pixel_size', self.filter_pixel_size), ('filter_time_per_frame', self.filter_time_per_frame), ('filter_time_per_frame_unit', self.filter_time_per_frame_unit)])
        if 'file_path' in data and data.get('file_path'):
            self.file_path.setText(str(data['file_path']).strip())
        if 'performed' in data:
            self._performed_this_session = bool(data['performed'])
