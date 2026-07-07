"""
ND2 / LIF / CZI conversion tab: file selection, conversion parameters, preview, convert to TIFF.
"""
from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget
from bactomate.modules.bioio_reader import get_info as bioio_get_info, is_available as bioio_is_available, load_data as bioio_load_data
from bactomate.modules.tiff_metadata import (
    TiffMetadataOverrides,
    get_tiff_file_info,
    normalize_tiff_array_to_tcyx,
    read_metadata_from_path,
    validate_tiff_metadata_overrides,
)
from bactomate.gui.constants import CONVERT_ND2_SCRIPT, LOGS_DIR, PROJECT_ROOT
from bactomate.gui.conversion_progress import ConversionProgressDialog
from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.module_shell import PreviewSurface
from bactomate.gui.preferences import LAST_DIR_ND2, get_default_browse_dir
from bactomate.gui.image_utils import to_qpixmap, scale_for_preview
from bactomate.gui.session_params import apply_to_widgets, get_from_widgets
from bactomate.gui.layout_helpers import style_compact_action_button, style_module_primary_button, vbox_from_rows
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET, MODULE_PANEL_COMBO_STYLESHEET
from bactomate.gui.utils import safe_imread
from bactomate.gui.widgets import DragDropGroupBox, create_info_icon, show_confirm_dialog, show_file_selection_dialog, show_message_dialog, show_task_complete_dialog, style_dialog_button_box, tif_files_output_dir
from bactomate.modules.conversion_crop import RoiTuple, apply_yx_crop, format_roi_status
from bactomate.gui.workers import WorkerThread
from bactomate.gui.workspace_scanner import list_convertible_microscopy_files
from bactomate.gui.preview.preview_window import CollapsibleSection, LUT_HIST_CANVAS_HEIGHT_PX, LUT_HIST_FIGSIZE_INCHES, apply_lut_histogram_figure_margins
_RIGHT_PANEL_LIST_CAP = 2500
_FOLDER_CONVERTIBLE_CAP_FOR_LOAD = 400
_METADATA_SOURCE_SUFFIXES = frozenset({'.nd2', '.lif', '.czi', '.tif', '.tiff'})

def _nd2_build_index(axis_keys: tuple[str, ...], t: int, c: int, z: int, position: int=0) -> tuple:
    """Index tuple for an ND2 ndarray where axis_keys matches ndarray.shape order (see nd2.ND2File.sizes)."""
    idx: list = []
    for k in axis_keys:
        if k == 'T':
            idx.append(int(t))
        elif k == 'C':
            idx.append(int(c))
        elif k == 'Z':
            idx.append(int(z))
        elif k == 'P':
            idx.append(int(position))
        elif k in ('Y', 'X', 'S'):
            idx.append(slice(None))
        else:
            idx.append(0)
    return tuple(idx)

def _nd2_slice_yx(arr: np.ndarray, axis_keys: tuple[str, ...], t: int, c: int, z: int, position: int=0) -> np.ndarray:
    """Single-plane (Y, X) slice from ND2 4D/5D array."""
    plane = np.asarray(arr[_nd2_build_index(axis_keys, t, c, z, position)])
    if plane.ndim > 2:
        plane = np.squeeze(plane)
    return plane
_LUT_PREVIEW_AUTO_LO_PCT = 0.01
_LUT_PREVIEW_AUTO_HI_PCT = 99.99

def _nd2_best_z_for_channel(arr: np.ndarray, axis_keys: tuple[str, ...], t: int, c: int, num_z: int) -> int:
    """Choose Z with strongest signal (99.5th percentile). Returns 0 if all slices are empty."""
    if 'Z' not in axis_keys or num_z <= 1:
        return 0
    (best_z, best_score) = (0, -1.0)
    for z in range(num_z):
        plane = _nd2_slice_yx(arr, axis_keys, t, c, z)
        if plane.size == 0:
            continue
        sc = float(np.percentile(plane, 99.5))
        if sc > best_score:
            best_score = sc
            best_z = z
    return best_z if best_score > 0 else 0

class ND2ConversionTab(QWidget):
    """Tab for converting microscopy files (ND2, LIF, CZI) to TIFF with metadata extraction"""
    workflow_step_finished = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget]=None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.file_info = None
        self._performed_this_session = False
        self._conversion_crop_roi: Optional[RoiTuple] = None
        self._module_preview_mount: Optional[QWidget] = None
        self._preview_surface: Optional[PreviewSurface] = None
        self._embed_fc_preview_widget: Optional[Any] = None
        self.init_ui()
        self._candidate_list_timer = QTimer(self)
        self._candidate_list_timer.setSingleShot(True)
        self._candidate_list_timer.timeout.connect(self.refresh_conversion_candidate_list)
        self.file_path.textChanged.connect(self._schedule_conversion_candidate_refresh)

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        scroll_widget.setMinimumWidth(0)
        scroll_widget.setStyleSheet('QWidget { font-size: 12pt; }QLabel { font-size: 12pt; }QLineEdit { font-size: 13pt; }QGroupBox { font-size: 12pt; font-weight: 600; }QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #aeb6c2; font-weight: 600; }QPushButton { font-size: 12pt; }QCheckBox { font-size: 12pt; }QSpinBox, QDoubleSpinBox { font-size: 12pt; }' + MODULE_PANEL_COMBO_STYLESHEET)
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(12)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        file_group = self._build_file_group()
        scroll_layout.addWidget(CollapsibleSection('File and path', file_group, expanded=True))
        params_group = self._build_conversion_params_group()
        self._conversion_params_section = CollapsibleSection('Conversion parameters and channels', params_group, expanded=False)
        scroll_layout.addWidget(self._conversion_params_section)
        metadata_group = QGroupBox('')
        metadata_group.setFlat(True)
        metadata_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        metadata_layout = QVBoxLayout()
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.metadata_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.metadata_text.setMinimumHeight(120)
        self.metadata_text.setMaximumHeight(200)
        self.metadata_text.setPlaceholderText('Metadata will be displayed here after loading microscopy file...')
        metadata_font = QFont()
        metadata_font.setPointSize(13)
        self.metadata_text.setFont(metadata_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.metadata_text.setSizePolicy(size_policy)
        metadata_layout.addWidget(self.metadata_text)
        metadata_btn_row = QHBoxLayout()
        self.edit_tiff_metadata_btn = QPushButton('Edit output metadata')
        self.edit_tiff_metadata_btn.setToolTip(
            'Manually correct pixel size, timing, channel names, and acquisition notes, '
            'then write them into generated TIFF files. Source files are not modified.'
        )
        style_compact_action_button(self.edit_tiff_metadata_btn, horizontal_padding=32)
        self.edit_tiff_metadata_btn.setMinimumWidth(self.edit_tiff_metadata_btn.width())
        self.edit_tiff_metadata_btn.setMaximumWidth(16777215)
        self.edit_tiff_metadata_btn.clicked.connect(self._open_tiff_metadata_editor)
        self.edit_tiff_metadata_btn.setEnabled(False)
        self.edit_tiff_metadata_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        metadata_btn_row.addWidget(self.edit_tiff_metadata_btn)
        metadata_layout.addLayout(metadata_btn_row)
        metadata_group.setLayout(metadata_layout)
        scroll_layout.addWidget(CollapsibleSection('Extracted metadata', metadata_group, expanded=False))
        self.preview_btn = QPushButton('Preview File')
        self.preview_btn.clicked.connect(self.preview_file)
        self.preview_btn.setEnabled(False)
        self.preview_btn.setVisible(False)
        scroll_layout.addStretch(1)
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        self.convert_btn = QPushButton('Convert to TIFF')
        self.convert_btn.setMinimumHeight(42)
        self.convert_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        style_module_primary_button(self.convert_btn)
        self.convert_btn.clicked.connect(self.convert_nd2)
        self.convert_btn.setEnabled(False)
        layout.addWidget(self.convert_btn, 0)
        self.setLayout(layout)

    def attach_module_preview_mount(self, mount: Optional[QWidget]) -> None:
        """Right-hand preview mount from ModuleShell (embedded file preview)."""
        self._module_preview_mount = mount
        self._preview_surface = None
        if mount is not None:
            parent = mount.parent()
            if isinstance(parent, PreviewSurface):
                self._preview_surface = parent
                parent.set_pick_callback(self._on_pick_from_preview_list)
        self.refresh_conversion_candidate_list()

    def _schedule_conversion_candidate_refresh(self, _text: str='') -> None:
        self._candidate_list_timer.start(200)

    def refresh_conversion_candidate_list(self) -> None:
        """Populate the module shell dropdown when the path is a folder, or siblings when it is a single file."""
        surf = self._preview_surface
        if surf is None:
            return
        p = (self.file_path.text() or '').strip()
        if not p:
            surf.set_candidate_files([])
            return
        po = Path(p)
        selected: Optional[str] = None
        if po.is_dir():
            paths = [str(x) for x in list_convertible_microscopy_files(po, max_files=_RIGHT_PANEL_LIST_CAP)]
            if self.file_info and Path(str(self.file_info.get('path') or '')).is_file():
                try:
                    fr = Path(str(self.file_info['path'])).resolve()
                    pr = po.resolve()
                    if pr in fr.parents:
                        selected = str(fr)
                except (OSError, ValueError, TypeError):
                    pass
            surf.set_candidate_files(paths, selected=selected)
            return
        if po.is_file():
            parent = po.parent
            if parent.is_dir():
                paths = [str(x) for x in list_convertible_microscopy_files(parent, max_files=_RIGHT_PANEL_LIST_CAP)]
                surf.set_candidate_files(paths, selected=str(po.resolve()))
            else:
                surf.set_candidate_files([])
            return
        surf.set_candidate_files([])

    def _on_pick_from_preview_list(self, path: str) -> None:
        """Load preview/metadata for this file without replacing the folder path on the left."""
        self.load_file_info(path)

    def _show_file_conversion_preview(self, **kwargs: Any) -> Any:
        kwargs.setdefault('embed_parent', self._module_preview_mount)
        from bactomate.gui.preview.file_conversion_preview import show_file_conversion_preview
        prev = self._embed_fc_preview_widget
        if prev is not None:
            prev.deleteLater()
            self._embed_fc_preview_widget = None
            if self._preview_surface is not None:
                self._preview_surface.set_preview_chrome_visible(True)
        dlg = show_file_conversion_preview(self, **kwargs)
        self._embed_fc_preview_widget = dlg
        emb = kwargs.get('embed_parent', self._module_preview_mount)
        if self._preview_surface is not None and emb is not None:
            try:
                if dlg.parent() is emb:
                    self._preview_surface.set_preview_chrome_visible(False)
            except Exception:
                pass
        return dlg

    def _refresh_embedded_conversion_preview(self) -> None:
        """If a file is selected for preview and the right panel is available, render the preview in place."""
        if not self._module_preview_mount:
            return
        p_line = (self.file_path.text() or '').strip()
        if not p_line or not Path(p_line).exists():
            if self._preview_surface is not None:
                self._preview_surface.set_preview_chrome_visible(True)
            return
        po_line = Path(p_line)
        preview_path: Optional[Path] = None
        if self.file_info:
            meta_p = Path(str(self.file_info.get('path') or ''))
            if meta_p.is_file():
                preview_path = meta_p
        if preview_path is None and po_line.is_file():
            preview_path = po_line
        if preview_path is None or not preview_path.exists():
            if self._preview_surface is not None:
                self._preview_surface.set_preview_chrome_visible(True)
            return
        if not self.file_info:
            if self._preview_surface is not None:
                self._preview_surface.set_preview_chrome_visible(True)
            return
        ft = str(self.file_info.get('file_type') or 'nd2').lower()
        try:
            self._preview_by_type(ft, preview_path)
        except Exception:
            if self._preview_surface is not None:
                self._preview_surface.set_preview_chrome_visible(True)

    def _build_file_group(self) -> QWidget:
        """Build the microscopy file selection group (Path + info, same layout as segmentation tabs)."""
        from bactomate.gui.layout_helpers import build_path_drop_group
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText('Select a microscopy file or folder... (or drag and drop)')
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        file_btn = QPushButton('Browse File...')
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton('Browse Folder...')
        folder_btn.clicked.connect(self.select_folder)
        return build_path_drop_group(self, self.handle_dropped_path, line_edit=self.file_path, buttons=(file_btn, folder_btn), info_icon=create_info_icon("Path to the microscopy file or folder to convert (ND2, LIF, CZI, or TIFF).\n\nWhen this path is a folder, use the preview dropdown on the right to open files — the path here stays on the folder so Convert to TIFF still applies to the whole folder.\n\nConverted TIFFs are saved in a single folder in the same directory: 'Tif_files' (reused if it already exists). For a single file, that folder is next to the file; for a folder, inside it.", self))

    def _build_conversion_params_group(self) -> QGroupBox:
        """Build the Conversion Parameters group (frames, channels, corrections)."""
        params_group = QGroupBox('')
        params_group.setFlat(True)
        params_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        params_layout = QVBoxLayout()
        params_layout.setSpacing(6)
        self.frame_extract_mode = QComboBox()
        self.frame_extract_mode.addItems(['First N frames', 'Single frame only'])
        self.frame_extract_mode.setToolTip('First N frames: extract frames 1 through N (e.g. 50 → frames 1–50).\nSingle frame only: extract only one time point (all channels) by frame number (e.g. frame 50).')
        frame_info_icon = create_info_icon('Number of time frames to extract from the microscopy file.\n\nFirst N frames: extract from the start (e.g. 50 = frames 1–50).\nSingle frame only: extract only one time point by its frame number (1-based); all channels of that frame are included.\n\nThe maximum value updates when a file is loaded.', parent_widget=self)
        self.frame_extract_mode.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.frame_extract_mode.setMaximumWidth(148)
        extract_row = QHBoxLayout()
        extract_row.addWidget(QLabel('Extract:'))
        extract_row.addWidget(self.frame_extract_mode, 0)
        extract_row.addWidget(frame_info_icon)
        self.frame_label = QLabel('Frames:')
        self.frame_label.setToolTip("Used when 'First N frames' is selected.")
        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 10000)
        self.max_frames.setValue(100)
        self.max_frames.setMaximumWidth(88)
        self.single_frame_label = QLabel('Frame (1-based):')
        self.single_frame_label.setToolTip("Used when 'Single frame only' is selected. Extracts only this time point (all channels).")
        self.single_frame_spin = QSpinBox()
        self.single_frame_spin.setRange(1, 10000)
        self.single_frame_spin.setValue(1)
        self.single_frame_spin.setMaximumWidth(88)
        extract_row.addWidget(self.frame_label)
        extract_row.addWidget(self.max_frames)
        extract_row.addWidget(self.single_frame_label)
        extract_row.addWidget(self.single_frame_spin)
        self.extract_crop_cb = QCheckBox('Extract crop')
        self.extract_crop_cb.setToolTip('When enabled, only the crop region drawn in the preview is converted. Right-drag on the preview image to set the crop; middle-drag pans, wheel zooms.')
        extract_row.addWidget(self.extract_crop_cb)
        crop_info_icon = create_info_icon(
            'Extract crop converts only the selected region from the preview.\n\n'
            'How to draw it:\n'
            '- Enable Extract crop.\n'
            '- In the preview image, right-drag with the mouse to draw a crop box.\n'
            '- Drag the crop handles to resize or move the box.\n'
            '- Use Clear crop to remove the selection.\n\n'
            'Navigation while cropping:\n'
            '- Mouse wheel zooms.\n'
            '- Middle-drag pans the image.\n\n'
            'With X/Y drift correction enabled, BactoMate corrects drift on the full frame first and applies the crop afterward. '
            'This avoids crop-edge artifacts from the registration step.',
            parent_widget=self,
        )
        extract_row.addWidget(crop_info_icon)
        self.clear_crop_btn = QPushButton('Clear crop')
        style_compact_action_button(self.clear_crop_btn)
        self.clear_crop_btn.setEnabled(False)
        self.clear_crop_btn.clicked.connect(self._clear_conversion_crop)
        extract_row.addWidget(self.clear_crop_btn)
        extract_row.addStretch(1)
        self.frame_extract_mode.currentIndexChanged.connect(self._on_frame_extract_mode_changed)
        self._on_frame_extract_mode_changed(self.frame_extract_mode.currentIndex())
        params_layout.addLayout(extract_row)
        crop_row = QHBoxLayout()
        self.crop_status_label = QLabel(format_roi_status(None))
        self.crop_status_label.setStyleSheet('color: #9E9E9E; font-size: 10pt;')
        self.crop_status_label.setWordWrap(True)
        crop_row.addWidget(self.crop_status_label, 1)
        self.extract_crop_cb.toggled.connect(self._on_extract_crop_toggled)
        params_layout.addLayout(crop_row)
        self.channel_options_group = QGroupBox('')
        self.channel_options_group.setFlat(True)
        self.channel_options_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        self.channel_options_layout = QVBoxLayout()
        self.channel_options_group.setLayout(self.channel_options_layout)
        self.channel_options_group.setVisible(False)
        self.fi_channel_checks: list[QCheckBox] = []
        self.foci_channel_checks: list[QCheckBox] = []
        self.zproject_channel_checks: list[QCheckBox] = []
        self.bleach_channel_checks: list[QCheckBox] = []
        self.channel_name_edits: list[QLineEdit] = []
        self._channel_names_source_hint: str = ''
        params_layout.addWidget(self.channel_options_group)
        corrections_group = QGroupBox('')
        corrections_group.setFlat(True)
        corrections_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        corrections_outer = QVBoxLayout()
        corrections_outer.setSpacing(10)
        corrections_outer.setContentsMargins(10, 10, 10, 10)
        _correction_row_min_h = 34

        def _add_correction_row(row_layout: QHBoxLayout) -> None:
            row_layout.setSpacing(6)
            row_layout.setContentsMargins(0, 0, 0, 0)
            wrap = QWidget()
            wrap.setLayout(row_layout)
            wrap.setMinimumHeight(_correction_row_min_h)
            wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            corrections_outer.addWidget(wrap)
        separate_row = QHBoxLayout()
        self.separate_channels = QCheckBox('Save channels separately')
        self.separate_channels.setChecked(True)
        separate_row.addWidget(self.separate_channels, 0)
        separate_row.addStretch(1)
        _add_correction_row(separate_row)
        drift_row = QHBoxLayout()
        self.drift_correct = QCheckBox('Apply drift correction (X,Y registration)')
        self.drift_correct.setChecked(False)
        drift_info_icon = create_info_icon('Apply X,Y drift correction to compensate for stage movement during time-lapse imaging.\n\nDefault: Disabled (unchecked)\nSource: Optional correction step\n\nMethod: Phase cross-correlation (frame-by-frame alignment)\nAlgorithm: Each frame is aligned to the PREVIOUS frame (incremental correction).\nThis approach is more robust to large cumulative drifts because:\n- Smaller shifts between consecutive frames improve alignment accuracy\n- Phase cross-correlation works best with small shifts\n- Naturally handles gradual drift over time\n\nUse when: You notice the sample drifting over time.\nNote: Increases processing time but improves temporal analysis accuracy.\nThe first frame remains unchanged; subsequent frames are incrementally aligned.', parent_widget=self)
        drift_row.addWidget(self.drift_correct, 0)
        drift_row.addWidget(drift_info_icon, 0)
        drift_row.addStretch(1)
        _add_correction_row(drift_row)
        bleach_row = QHBoxLayout()
        self.bleach_correct = QCheckBox('Apply photobleaching correction')
        self.bleach_correct.setChecked(False)
        self.bleach_method_combo = QComboBox()
        self.bleach_method_combo.addItems(['Exponential (intensity preservation)', 'Linear decay', 'Histogram matching', 'Per-frame min-max (visualization)'])
        self.bleach_method_combo.setCurrentIndex(0)
        self.bleach_method_combo.setToolTip('Method used for bleach correction. See info icon for details.')
        bleach_info_icon = create_info_icon("Photobleaching correction (inspired by CorrectBleach: https://github.com/fiji/CorrectBleach).\n\nWhich channels: Open the 'Channel' dropdown and tick the 'Bleach' checkbox for each fluorescence channel (e.g. GFP, mScarlet); bleach correction is applied only to those. Leave phase/transmission unticked.\n\nMethods:\n• Exponential: I(t)=I₀×exp(-t/τ), normalizes to first frame. Use for intensity quantification.\n• Linear: Linear decay fit, then scale to first frame.\n• Histogram matching: Match each frame's histogram to frame 0. Good for segmentation; not for quantification.\n• Per-frame min-max: Stretch each frame so its min→0 and max→full range of the current bit depth (e.g. 65535 for 16-bit). No RGB conversion; dtype and bit depth are preserved. Visualization only.", parent_widget=self)
        bleach_row.addWidget(self.bleach_correct, 0)
        bleach_row.addWidget(QLabel('Method:'), 0)
        bleach_row.addWidget(self.bleach_method_combo, 0)
        bleach_row.addWidget(bleach_info_icon, 0)
        bleach_row.addStretch(1)
        _add_correction_row(bleach_row)
        zproj_row = QHBoxLayout()
        self.max_z_projection = QCheckBox('Max intensity projection for multiple Z stack')
        self.max_z_projection.setChecked(False)
        z_projection_info_icon = create_info_icon('Apply maximum intensity projection across Z-slices when Z > 1.\n\nWhen enabled: For each pixel, takes the maximum intensity value across all Z-slices. This creates a single 2D image from a 3D Z-stack.\n\nWhen disabled: Uses only the first Z-slice (Z=1).\n\nUse when: You have Z-stack data and want to combine all slices into one image.', parent_widget=self)
        zproj_row.addWidget(self.max_z_projection, 0)
        zproj_row.addWidget(z_projection_info_icon, 0)
        zproj_row.addStretch(1)
        _add_correction_row(zproj_row)
        corrections_group.setLayout(corrections_outer)
        params_layout.addWidget(corrections_group)
        params_group.setLayout(params_layout)
        return params_group

    def _on_frame_extract_mode_changed(self, index: int) -> None:
        """Show/hide Number of Frames vs Single frame spinbox based on extract mode."""
        is_single = index == 1
        self.frame_label.setVisible(not is_single)
        self.max_frames.setVisible(not is_single)
        self.single_frame_label.setVisible(is_single)
        self.single_frame_spin.setVisible(is_single)

    def _expand_conversion_params_section(self) -> None:
        """Reveal conversion settings once a file/folder path has been loaded."""
        section = getattr(self, '_conversion_params_section', None)
        if section is not None and hasattr(section, 'set_expanded'):
            section.set_expanded(True)

    def _on_extract_crop_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._clear_conversion_crop()
        self._sync_conversion_crop_overlay()

    def _clear_conversion_crop(self) -> None:
        self._conversion_crop_roi = None
        self._update_crop_status_label()
        self._sync_conversion_crop_overlay()

    def _update_crop_status_label(self) -> None:
        if hasattr(self, 'crop_status_label'):
            self.crop_status_label.setText(format_roi_status(self._conversion_crop_roi))
        if hasattr(self, 'clear_crop_btn'):
            self.clear_crop_btn.setEnabled(self._conversion_crop_roi is not None)

    def set_conversion_crop_roi(self, roi: Optional[RoiTuple]) -> None:
        """Store crop ROI in native image coordinates (from preview)."""
        self._conversion_crop_roi = roi
        if roi is not None and hasattr(self, 'extract_crop_cb'):
            self.extract_crop_cb.setChecked(True)
        self._update_crop_status_label()
        self._sync_conversion_crop_overlay()

    def _sync_conversion_crop_overlay(self) -> None:
        dlg = getattr(self, '_embed_fc_preview_widget', None)
        if dlg is None or not hasattr(dlg, 'sync_crop_overlay'):
            return
        try:
            dlg.sync_crop_overlay()
        except Exception:
            pass

    def extract_crop_enabled(self) -> bool:
        return bool(getattr(self, 'extract_crop_cb', None) and self.extract_crop_cb.isChecked())

    def _validate_crop_before_conversion(self) -> bool:
        if not self.extract_crop_enabled():
            return True
        if self._conversion_crop_roi is not None:
            return True
        show_message_dialog(self, 'Crop region required', 'Extract crop is enabled, but no crop region is set.\n\nOpen the preview, right-drag a box on the image, or disable Extract crop.', kind='warning')
        return False

    def _append_crop_cli_args(self, cmd_args: list) -> None:
        if not self.extract_crop_enabled() or self._conversion_crop_roi is None:
            return
        (x0, y0, x1, y1) = self._conversion_crop_roi
        cmd_args.extend(['--crop', str(x0), str(y0), str(x1), str(y1)])

    def _apply_conversion_crop_array(self, arr: np.ndarray, *, context: str='') -> np.ndarray:
        if not self.extract_crop_enabled() or self._conversion_crop_roi is None:
            return arr
        try:
            return apply_yx_crop(arr, self._conversion_crop_roi)
        except ValueError as exc:
            raise ValueError(f'{context}{exc}') from exc

    def _maybe_crop_for_conversion(self, arr: np.ndarray, *, label: str) -> Optional[np.ndarray]:
        if not self.extract_crop_enabled() or self._conversion_crop_roi is None:
            return arr
        try:
            cropped = self._apply_conversion_crop_array(arr)
        except ValueError as exc:
            self.parent_window.log_output(f'  ✗ Crop failed for {label}: {exc}\n')
            return None
        self.parent_window.log_output(f"  Applying crop ROI for {label}: {format_roi_status(self._conversion_crop_roi).replace('Crop: ', '')}\n")
        return cropped

    def _frame_selection_changes_output(self, num_time: int) -> bool:
        """True when the selected frame mode would write fewer/different frames than the input."""
        total = max(1, int(num_time or 1))
        (start, end) = self._get_frame_range(total)
        return start != 0 or end != total

    def _tiff_conversion_has_work(self, *, num_time: int, num_channels: int) -> bool:
        """Return True if TIFF conversion would create an output different from the input."""
        return int(num_channels or 1) >= 2 or self.extract_crop_enabled() or self._frame_selection_changes_output(num_time) or bool(getattr(self, 'drift_correct', None) and self.drift_correct.isChecked()) or bool(getattr(self, 'bleach_correct', None) and self.bleach_correct.isChecked())

    def _show_tiff_noop_dialog(self) -> None:
        show_message_dialog(
            self,
            'TIFF already ready',
            'The selected file is already a single-channel TIFF and no conversion-changing options are enabled.\n\n'
            'Enable Extract crop, choose a frame subset, or enable drift/bleach correction if you want BactoMate to write a new TIFF.',
            kind='info',
        )

    def _get_frame_range(self, num_time: int) -> tuple[int, int]:
        """Return (start_0based, end_0based) for conversion. end is exclusive."""
        if num_time <= 0:
            return (0, 0)
        if self.frame_extract_mode.currentIndex() == 1:
            one_based = self.single_frame_spin.value()
            start = max(0, min(one_based - 1, num_time - 1))
            return (start, start + 1)
        n = min(self.max_frames.value(), num_time)
        return (0, n)

    def select_file(self):
        start_dir = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_ND2)
        (file_path, _) = QFileDialog.getOpenFileName(self, 'Select Microscopy File', start_dir, 'Microscopy Files (*.nd2 *.ND2 *.lif *.LIF *.czi *.CZI *.tif *.tiff *.TIF *.TIFF);;ND2 Files (*.nd2 *.ND2);;LIF Files (*.lif *.LIF);;CZI Files (*.czi *.CZI);;TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)')
        if file_path:
            if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_ND2, file_path)
            self.file_path.setText(file_path)
            self.load_file_info(file_path)

    def select_folder(self):
        start_dir = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_ND2)
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder with Microscopy Files', start_dir)
        if folder_path:
            if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_ND2, folder_path)
            self.file_path.setText(folder_path)
            folder = Path(folder_path)
            supported_files = list_convertible_microscopy_files(folder, max_files=_FOLDER_CONVERTIBLE_CAP_FOR_LOAD)
            if supported_files:
                self.load_folder_info(folder_path, supported_files)
            else:
                self.metadata_text.setText('No supported microscopy files (ND2, LIF, CZI, TIFF) found in selected folder')
                self.metadata_text.setStyleSheet('background-color: #2b2b2b; color: #ffffff; font-size: 12pt;')
                self.file_info = None
                self._update_tiff_metadata_button_state()
                self.refresh_conversion_candidate_list()

    def _apply_session_to_path(self) -> None:
        """Delegate to main window (Re-run lives on the dashboard). Kept for compatibility."""
        fn = getattr(self.parent_window, 'open_rerun_session_dialog', None)
        if callable(fn):
            fn()

    def handle_dropped_path(self, dropped_path):
        """Handle dropped file or folder path"""
        path_obj = Path(dropped_path)
        self.file_path.setText(dropped_path)
        if path_obj.is_file():
            self.load_file_info(dropped_path)
        elif path_obj.is_dir():
            folder = path_obj
            supported_files = list_convertible_microscopy_files(folder, max_files=_FOLDER_CONVERTIBLE_CAP_FOR_LOAD)
            if supported_files:
                self.load_folder_info(str(folder), supported_files)
            else:
                self.metadata_text.setText('No supported microscopy files (ND2, LIF, CZI, TIFF) found in dropped folder')
                self.metadata_text.setStyleSheet('background-color: #2b2b2b; color: #ffffff; font-size: 12pt;')
                self.file_info = None
                self._update_tiff_metadata_button_state()
                self.refresh_conversion_candidate_list()

    def _get_effective_channel_names(self, num_channels: int) -> list[str]:
        """Names used for output filenames (editable fields override loaded metadata)."""
        n = max(1, int(num_channels))
        if getattr(self, 'channel_name_edits', None) and len(self.channel_name_edits) >= n:
            out: list[str] = []
            for i in range(n):
                text = (self.channel_name_edits[i].text() or '').strip()
                out.append(text or f'Channel_{i + 1}')
            return out
        if self.file_info and self.file_info.get('channel_names'):
            base = list(self.file_info['channel_names'])
            if len(base) >= n:
                return base[:n]
        return [f'Channel_{i + 1}' for i in range(n)]

    def _conversion_fi_foci_index_sets(self, num_channels: int) -> tuple[set[int], set[int]]:
        fi_indices: set[int] = set()
        foci_indices: set[int] = set()
        if getattr(self, 'fi_channel_checks', None) and len(self.fi_channel_checks) >= num_channels:
            fi_indices = {i for i, cb in enumerate(self.fi_channel_checks) if cb.isChecked()}
        if getattr(self, 'foci_channel_checks', None) and len(self.foci_channel_checks) >= num_channels:
            foci_indices = {i for i, cb in enumerate(self.foci_channel_checks) if cb.isChecked()}
        return fi_indices, foci_indices

    def _conversion_channel_output_path(
        self,
        output_base: Path,
        stem: str,
        ch: int,
        channel_names: list[str],
        *,
        phase_used: bool,
        fi_indices: set[int],
        foci_indices: set[int],
    ) -> tuple[Path, bool]:
        from bactomate.modules.conversion_channel_names import split_channel_output_name

        ch_name = channel_names[ch] if ch < len(channel_names) else f'Channel_{ch + 1}'
        out_name, phase_used = split_channel_output_name(
            stem,
            ch_name,
            in_fi=ch in fi_indices,
            in_foci=ch in foci_indices,
            phase_used=phase_used,
        )
        return output_base / out_name, phase_used

    def _output_metadata_for_tiff(self, frames: int, *, axes: str='TYX') -> dict:
        """ImageJ metadata for generated TIFFs, including manual output overrides."""
        from bactomate.modules.tiff_metadata import build_metadata_dict

        info = self.file_info or {}
        pixel_size = info.get('pixel_size')
        time_per_frame = info.get('time_per_frame_min')
        metadata = build_metadata_dict(
            pixel_size_um=float(pixel_size) if pixel_size is not None and float(pixel_size) > 0 else None,
            time_per_frame_min=float(time_per_frame) if time_per_frame is not None and float(time_per_frame) > 0 else None,
            frames=max(1, int(frames)),
            images=max(1, int(frames)),
            axes=axes,
        )
        overrides = info.get('metadata_overrides') if isinstance(info.get('metadata_overrides'), dict) else {}
        info_lines: list[str] = []
        if overrides.get('objective_name') or overrides.get('objective_magnification'):
            obj = str(overrides.get('objective_name') or '').strip()
            mag = overrides.get('objective_magnification')
            suffix = f' {float(mag):.1f}x' if mag is not None and float(mag) > 0 else ''
            info_lines.append(f'Objective: {obj}{suffix}'.strip())
        if pixel_size is not None and float(pixel_size) > 0:
            info_lines.append(f'Pixel size: {float(pixel_size):.4f} µm')
        if time_per_frame is not None and float(time_per_frame) > 0:
            info_lines.append(f'Time per frame: {float(time_per_frame) * 60.0:.4f} s')
        if overrides.get('exposure_ms') is not None:
            info_lines.append(f"Exposure: {float(overrides['exposure_ms']):.3f} ms")
        if overrides.get('optics'):
            info_lines.append(f"Optics: {overrides['optics']}")
        if overrides.get('capturing'):
            info_lines.append(f"Capturing: {overrides['capturing']}")
        if overrides.get('acquisition_notes'):
            info_lines.append(str(overrides['acquisition_notes']))
        if info_lines:
            metadata['Info'] = '\n'.join(info_lines)
        return metadata

    def _append_output_metadata_cli_args(self, cmd_args: list) -> None:
        """Pass manually edited output metadata to the ND2 conversion script."""
        info = self.file_info or {}
        if info.get('pixel_size'):
            cmd_args.extend(['--pixel-size', str(info['pixel_size'])])
        if info.get('time_per_frame_min'):
            cmd_args.extend(['--time-per-frame', str(info['time_per_frame_min'])])
        num_channels = int(info.get('num_channels') or 0)
        if num_channels > 1:
            names = self._get_effective_channel_names(num_channels)
            if len(names) == num_channels:
                cmd_args.append('--channel-names')
                cmd_args.extend(names)

    def _rebuild_channel_options(self, channel_names, num_channels, *, names_source: str=''):
        """Populate Channel options panel: one row per channel with FI / Foci / Z-channel / Bleach correct checkboxes. Shown when num_channels > 1."""
        self.fi_channel_checks.clear()
        self.foci_channel_checks.clear()
        self.zproject_channel_checks.clear()
        self.bleach_channel_checks.clear()
        self.channel_name_edits.clear()
        self._channel_names_source_hint = (names_source or '').strip().lower()
        while self.channel_options_layout.count():
            item = self.channel_options_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if num_channels <= 1:
            self.channel_options_group.setVisible(False)
            return
        src_hint = {'labels': 'Channel names from ImageJ/Fiji Labels in the file.', 'ome': 'Channel names from OME-TIFF metadata.', 'info': 'Channel names from ImageJ Info / ImageDescription text.', 'luts': 'No names in file — inferred from display LUT colors (edit to match your dyes, e.g. dsRed, GFP).', 'default': 'No channel names in file metadata — edit names below (used in output filenames).'}.get(self._channel_names_source_hint, '')
        info_label = QLabel('Tick per channel: FI adds "_fi" in filename (Analyze Masks); Foci adds "_foci" for foci analysis; Z-channel = max Z-projection; Bleach correct = photobleaching correction.' + (f'\n\n{src_hint}' if src_hint else ''))
        info_label.setWordWrap(True)
        info_label.setStyleSheet('color: #a0a0a0; font-size: 10pt;')
        self.channel_options_layout.addWidget(info_label)
        self.channel_options_layout.addSpacing(12)
        for (i, name) in enumerate(channel_names):
            row_widget = QWidget()
            row = QHBoxLayout()
            row.setContentsMargins(0, 2, 0, 2)
            row_widget.setLayout(row)
            ch_label = QLabel(f'Channel {i + 1}:')
            ch_label.setMinimumWidth(72)
            name_edit = QLineEdit(str(name))
            name_edit.setPlaceholderText(f'Channel_{i + 1}')
            name_edit.setToolTip('Used in output filenames (e.g. stem_fi_GFP.tif). Edit if metadata has no dye names.')
            name_edit.setMinimumWidth(120)
            self.channel_name_edits.append(name_edit)
            row.addWidget(ch_label)
            row.addWidget(name_edit, 1)
            row.addSpacing(8)
            fi_cb = QCheckBox('FI')
            fi_cb.setToolTip('Save with "_fi" in filename (e.g. stem_fi.tif) for Analyze Masks batch')
            foci_cb = QCheckBox('Foci')
            foci_cb.setToolTip('Save with "_foci" in filename for Analyze Masks foci analysis')
            z_cb = QCheckBox('Z-channel')
            z_cb.setToolTip('Apply max Z-projection for this channel (when Z stack > 1)')
            bleach_cb = QCheckBox('Bleach correct')
            bleach_cb.setToolTip('Apply bleach correction to this channel')
            row.addWidget(fi_cb)
            row.addWidget(foci_cb)
            row.addWidget(z_cb)
            row.addWidget(bleach_cb)
            row.addStretch()
            self.channel_options_layout.addWidget(row_widget)
            self.fi_channel_checks.append(fi_cb)
            self.foci_channel_checks.append(foci_cb)
            self.zproject_channel_checks.append(z_cb)
            self.bleach_channel_checks.append(bleach_cb)
            z_cb.stateChanged.connect(self._sync_z_projection_from_channels)
            bleach_cb.stateChanged.connect(self._sync_bleach_from_channels)
        self.channel_options_group.setVisible(True)

    def _sync_z_projection_from_channels(self):
        """When any Z-channel is ticked, enable Max intensity projection; when all unticked, disable it."""
        if not getattr(self, 'zproject_channel_checks', None):
            return
        any_checked = any((cb.isChecked() for cb in self.zproject_channel_checks))
        if getattr(self, 'max_z_projection', None) is not None:
            self.max_z_projection.setChecked(any_checked)

    def _sync_bleach_from_channels(self):
        """When any Bleach correct is ticked, enable Apply photobleaching correction; when all unticked, disable it."""
        if not getattr(self, 'bleach_channel_checks', None):
            return
        any_checked = any((cb.isChecked() for cb in self.bleach_channel_checks))
        if getattr(self, 'bleach_correct', None) is not None:
            self.bleach_correct.setChecked(any_checked)

    def load_folder_info(self, folder_path, supported_files):
        """Load and display information for all files in a folder"""
        try:
            folder = Path(folder_path)
            all_files_info = []
            errors = []
            self.parent_window.log_output(f'Scanning folder: {folder_path}\n')
            self.parent_window.log_output(f'Found {len(supported_files)} file(s) to analyze\n')
            for file_path in supported_files:
                try:
                    file_ext = file_path.suffix.lower()
                    file_info = None
                    if file_ext == '.nd2':
                        file_info = self._get_nd2_info(str(file_path))
                    elif file_ext == '.lif':
                        file_info = self._get_lif_info(str(file_path))
                    elif file_ext == '.czi':
                        file_info = self._get_czi_info(str(file_path))
                    elif file_ext in ['.tif', '.tiff']:
                        file_info = self._get_tiff_info(str(file_path))
                    if file_info:
                        all_files_info.append((file_path.name, file_info))
                        self.parent_window.log_output(f"✓ {file_path.name}: {file_info['num_time']} frames, {file_info['num_channels']} channels\n")
                    else:
                        errors.append(f'{file_path.name}: Could not read file info')
                except Exception as e:
                    errors.append(f'{file_path.name}: {str(e)}')
                    self.parent_window.log_output(f'✗ {file_path.name}: Error - {str(e)}\n')
            if all_files_info:
                self._display_folder_summary(all_files_info, errors)
                first_file_path = str(supported_files[0])
                first_file_info = all_files_info[0][1]
                self.file_info = {'path': first_file_path, 'folder_path': str(folder_path), 'file_type': first_file_info['file_type'], 'num_time': first_file_info['num_time'], 'num_channels': first_file_info['num_channels'], 'num_z': first_file_info['num_z'], 'channel_names': first_file_info['channel_names'], 'pixel_size': first_file_info.get('pixel_size'), 'time_per_frame_min': first_file_info.get('time_per_frame_min')}
                n_t = first_file_info['num_time']
                self.max_frames.setMaximum(n_t)
                self.max_frames.setValue(n_t)
                self.single_frame_spin.setMaximum(max(1, n_t))
                if self.single_frame_spin.value() > n_t:
                    self.single_frame_spin.setValue(n_t)
                self._rebuild_channel_options(first_file_info['channel_names'], first_file_info['num_channels'])
                self.preview_btn.setEnabled(True)
                self.convert_btn.setEnabled(True)
                self._expand_conversion_params_section()
                if first_file_info.get('pixel_size'):
                    self.parent_window.shared_metadata['pixel_size'] = first_file_info['pixel_size']
                else:
                    self.parent_window.shared_metadata['pixel_size'] = None
                if first_file_info.get('time_per_frame_min'):
                    self.parent_window.shared_metadata['time_per_frame_min'] = first_file_info['time_per_frame_min']
                else:
                    self.parent_window.shared_metadata['time_per_frame_min'] = None
                self.parent_window.update_all_tabs_with_metadata()
                QTimer.singleShot(100, self._refresh_embedded_conversion_preview)
                self._update_tiff_metadata_button_state()
            else:
                self.metadata_text.setText('No files could be read from the selected folder.\n' + '\n'.join(errors) if errors else '')
                self.metadata_text.setStyleSheet('background-color: #2b2b2b; color: #ff6b6b; font-size: 12pt;')
                self.file_info = None
                self._update_tiff_metadata_button_state()
        finally:
            self.refresh_conversion_candidate_list()
            self._update_tiff_metadata_button_state()

    def _display_folder_summary(self, all_files_info, errors):
        """Display a summary of all files in the folder"""
        metadata_content = f'Folder contains {len(all_files_info)} file(s):\n'
        metadata_content += '=' * 60 + '\n\n'
        for (i, (filename, info)) in enumerate(all_files_info, 1):
            metadata_content += f'File {i}: {filename}\n'
            metadata_content += f"  Type: {info['file_type'].upper()}\n"
            metadata_content += f"  Dimensions: T={info['num_time']} frames, C={info['num_channels']} channels, Z={info['num_z']} slices\n"
            if info['num_channels'] > 1:
                metadata_content += f"  Channels: {', '.join(info['channel_names'])}\n"
            if info['pixel_size']:
                metadata_content += f"  Pixel Size: {info['pixel_size']:.4f} µm\n"
            else:
                metadata_content += f'  Pixel Size: Not found in metadata\n'
            if info.get('time_per_frame_min') is not None and info['time_per_frame_min'] > 0:
                tp = info['time_per_frame_min']
                metadata_content += f'  Time per frame: {tp:.6f} min ({tp * 60:.4f} s)\n'
            else:
                metadata_content += f'  Time per frame: Not found in metadata\n'
            if i > 1:
                first_info = all_files_info[0][1]
                differences = []
                if info['num_time'] != first_info['num_time']:
                    differences.append(f"frames ({info['num_time']} vs {first_info['num_time']})")
                if info['num_channels'] != first_info['num_channels']:
                    differences.append(f"channels ({info['num_channels']} vs {first_info['num_channels']})")
                if info['pixel_size'] and first_info['pixel_size']:
                    if abs(info['pixel_size'] - first_info['pixel_size']) > 0.0001:
                        differences.append(f"pixel size ({info['pixel_size']:.4f} vs {first_info['pixel_size']:.4f} µm)")
                if differences:
                    metadata_content += f"  ⚠ Different from first file: {', '.join(differences)}\n"
            metadata_content += '\n'
        if errors:
            metadata_content += f'\nErrors ({len(errors)} file(s)):\n'
            for error in errors:
                metadata_content += f'  ✗ {error}\n'
        self.metadata_text.setText(metadata_content)
        self.metadata_text.setStyleSheet('background-color: #2b2b2b; color: #ffffff; font-size: 10pt;')
        self._update_tiff_metadata_button_state()

    def _get_nd2_info(self, file_path):
        """Extract ND2 file info without updating UI (for folder scanning). Tries BioIO first, then tiff_metadata."""
        info = bioio_get_info(Path(file_path))
        if info and info.get('file_type') == 'nd2':
            out = {k: v for (k, v) in info.items() if k in ('file_type', 'num_time', 'num_channels', 'num_z', 'channel_names', 'pixel_size', 'scenes')}
            try:
                from bactomate.modules.tiff_metadata import read_metadata_from_path
                meta = read_metadata_from_path(Path(file_path))
                if meta.get('time_per_frame_min') is not None and float(meta['time_per_frame_min']) > 0:
                    out['time_per_frame_min'] = float(meta['time_per_frame_min'])
                for key in ('channel_wavelengths', 'objective_name', 'objective_magnification', 'zoom_magnification', 'exposure_ms', 'channel_exposure_ms', 'bits_per_component'):
                    if meta.get(key) is not None:
                        out[key] = meta[key]
            except Exception:
                pass
            return out
        try:
            from bactomate.modules.tiff_metadata import read_metadata_from_path
            meta = read_metadata_from_path(Path(file_path))
            num_time = meta.get('frame_count') or 1
            num_channels = meta.get('channel_count') or 1
            num_z = meta.get('z_count') or 1
            channel_names = meta.get('channel_names') or []
            if len(channel_names) != num_channels:
                channel_names = [f'Channel_{i + 1}' for i in range(num_channels)]
            return {'file_type': 'nd2', 'num_time': num_time, 'num_channels': num_channels, 'num_z': num_z, 'channel_names': channel_names, 'pixel_size': meta.get('pixel_size_um'), 'time_per_frame_min': meta.get('time_per_frame_min')}
        except Exception:
            return None

    @staticmethod
    def _safe_lif_pixel_size(reader):
        """Get pixel size from LIF reader without assuming physical_pixel_sizes shape (avoids dimension errors)."""
        try:
            pps = getattr(reader, 'physical_pixel_sizes', None)
            if pps is None:
                return None
            for attr in ('X', 'Y', 'Z'):
                v = getattr(pps, attr, None)
                if v is not None:
                    return float(v)
            for idx in (2, 1, 0):
                try:
                    v = pps[idx]
                    if v is not None:
                        return float(v)
                except (TypeError, IndexError, ValueError, KeyError):
                    continue
        except Exception:
            pass
        return None

    @staticmethod
    def _readlif_best_image_index(lif_file, log_cb=None):
        """Return the image index that looks like microscopy (XY > 1). Leica LIF can have Image 0 = 'Job' (sensor/log, e.g. 1x8983) and Image 1 = 'Series001' (1024x1024, 50T, 3C)."""
        n = getattr(lif_file, 'num_images', 0)
        if n <= 0:
            return 0
        for i in range(n):
            try:
                img = lif_file.get_image(i)
                info = getattr(img, 'info', None) or {}
                dims = info.get('dims') or getattr(img, 'dims', None)
                dims_n = getattr(img, 'dims_n', None) or {}
                x = 1
                y = 1
                if dims is not None:
                    x = int(getattr(dims, 'x', 1) or 1)
                    y = int(getattr(dims, 'y', 1) or 1)
                if x <= 1 or y <= 1:
                    x = int(dims_n.get(1, x) or x)
                    y = int(dims_n.get(2, y) or y)
                if x <= 1 or y <= 1:
                    try:
                        frame = img.get_frame(z=0, t=0, c=0)
                        if hasattr(frame, 'shape') and len(frame.shape) >= 2:
                            (y, x) = (frame.shape[0], frame.shape[1])
                    except Exception:
                        pass
                if x > 1 and y > 1:
                    if log_cb and i > 0:
                        log_cb(f'readlif: using image index {i} (XY={x}x{y}); index 0 looked like metadata/sensor.\n')
                    return i
            except Exception:
                continue
        return 0

    @staticmethod
    def _readlif_dims(image, log_cb=None):
        """Get (num_time, num_z, num_channels) from readlif LifImage.
        Uses image.info when present (channels, dims.t, dims.z), else dims/dims_n.
        When only dims_n[4] is set (no 5) and info says channels=1, probe get_frame(t=0,c=1) to detect real channel count;
        and when total planes looks like T*C*Z (e.g. 8983 ~ 50*2*90), infer T,C,Z."""
        (num_time, num_z, num_channels) = (1, 1, 1)
        info = getattr(image, 'info', None)
        dims_n = getattr(image, 'dims_n', None) or {}
        dims = getattr(image, 'dims', None)
        if log_cb:
            log_cb(f"readlif raw: info={info!r}\ndims_n={dims_n} dims={dims} nt={getattr(image, 'nt', None)} nz={getattr(image, 'nz', None)}\n")
        try:
            if isinstance(info, dict):
                ch = info.get('channels')
                if ch is not None and int(ch) >= 1:
                    num_channels = int(ch)
                d = info.get('dims')
                if d is not None:
                    num_time = int(getattr(d, 't', 1) or 1)
                    num_z = int(getattr(d, 'z', 1) or 1)
        except (AttributeError, TypeError, KeyError):
            pass
        if num_time <= 0 or num_z <= 0 or (isinstance(info, dict) and info.get('dims') is None):
            try:
                num_time = int(getattr(dims, 't', 1) or dims_n.get(4, 1))
                num_z = int(getattr(dims, 'z', 1) or dims_n.get(3, 1))
                num_channels = int(getattr(dims, 'c', None) or dims_n.get(5, 1))
            except (AttributeError, TypeError):
                pass
        v4 = int(dims_n.get(4, 0) or 0)
        v5 = int(dims_n.get(5, 0) or 0)
        if v4 >= 1 and v5 >= 1:
            (low, high) = (min(v4, v5), max(v4, v5))
            if 2 <= low <= 20 and 10 <= high <= 10000:
                num_channels = low
                num_time = high
        if num_channels <= 0:
            num_channels = 1
        if num_z <= 0:
            num_z = 1
        if num_time <= 0:
            num_time = 1
        return (num_time, num_z, num_channels)

    def _get_lif_info(self, file_path):
        """Extract LIF file info without updating UI (for folder scanning). Tries BioIO first, then aicsimageio/readlif."""
        try:
            info = bioio_get_info(Path(file_path))
            if info and info.get('file_type') == 'lif':
                return {k: v for (k, v) in info.items() if k in ('file_type', 'num_time', 'num_channels', 'num_z', 'channel_names', 'pixel_size', 'scenes')}
        except Exception:
            pass
        try:
            import numpy as np
            from aicsimageio import AICSImage
            lif_reader = AICSImage(file_path)
            try:
                dims = lif_reader.dims
                if hasattr(dims, 'order'):
                    dims_str = dims.order
                elif isinstance(dims, str):
                    dims_str = dims
                elif isinstance(dims, (list, tuple)):
                    dims_str = ''.join((str(d) for d in dims))
                else:
                    dims_str = str(dims)
                try:
                    if hasattr(lif_reader, 'get_xarray_data'):
                        xdata = lif_reader.get_xarray_data()
                        num_time = int(xdata.sizes.get('T', 1))
                        num_channels = int(xdata.sizes.get('C', 1))
                        num_z = int(xdata.sizes.get('Z', 1))
                    else:
                        shape = lif_reader.shape
                        if not isinstance(shape, (tuple, list)):
                            shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                        num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                        num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                        num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                except (KeyError, IndexError, AttributeError, TypeError) as e:
                    shape = lif_reader.shape
                    if isinstance(shape, dict):
                        num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                        num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                        num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                    else:
                        shape_list = list(shape) if hasattr(shape, '__iter__') else [shape]
                        num_time = shape_list[dims_str.index('T')] if 'T' in dims_str and len(shape_list) > dims_str.index('T') else 1
                        num_channels = shape_list[dims_str.index('C')] if 'C' in dims_str and len(shape_list) > dims_str.index('C') else 1
                        num_z = shape_list[dims_str.index('Z')] if 'Z' in dims_str and len(shape_list) > dims_str.index('Z') else 1
                channel_names = []
                try:
                    if hasattr(lif_reader, 'channel_names'):
                        channel_names = lif_reader.channel_names
                except:
                    pass
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i + 1}' for i in range(num_channels)]
                pixel_size = self._safe_lif_pixel_size(lif_reader)
                return {'file_type': 'lif', 'num_time': num_time, 'num_channels': num_channels, 'num_z': num_z, 'channel_names': channel_names, 'pixel_size': pixel_size}
            finally:
                if hasattr(lif_reader, 'close'):
                    try:
                        lif_reader.close()
                    except:
                        pass
        except Exception:
            pass
        try:
            from readlif.reader import LifFile
            lif_file = LifFile(file_path)
            if getattr(lif_file, 'num_images', 0) <= 0:
                return None
            best_idx = self._readlif_best_image_index(lif_file)
            image = lif_file.get_image(best_idx)
            (num_time, num_z, num_channels) = self._readlif_dims(image)
            pixel_size = None
            if hasattr(image, 'scale') and image.scale and (len(image.scale) >= 2):
                try:
                    sx = float(image.scale[0])
                    if sx > 0:
                        pixel_size = 1.0 / sx
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            return {'file_type': 'lif', 'num_time': num_time, 'num_channels': num_channels, 'num_z': num_z, 'channel_names': [f'Channel_{i + 1}' for i in range(num_channels)], 'pixel_size': pixel_size}
        except Exception:
            return None

    def _get_tiff_info(self, file_path):
        """Extract TIFF file info without updating UI (for folder scanning). Tries BioIO first, then tifffile."""
        path = Path(file_path)
        info = bioio_get_info(path)
        if info and info.get('file_type') == 'tiff':
            out = {k: v for (k, v) in info.items() if k in ('file_type', 'num_time', 'num_channels', 'num_z', 'channel_names', 'pixel_size', 'scenes')}
            tif_info = get_tiff_file_info(path)
            if tif_info:
                out['num_time'] = tif_info['num_time']
                out['num_channels'] = tif_info['num_channels']
                out['num_z'] = tif_info['num_z']
                if tif_info.get('channel_names'):
                    out['channel_names'] = tif_info['channel_names']
                if tif_info.get('pixel_size') is not None and float(tif_info['pixel_size']) > 0:
                    out['pixel_size'] = float(tif_info['pixel_size'])
            return out
        return get_tiff_file_info(path)

    def _czi_dims_from_axes(self, axes, shape):
        """Return (num_time, num_channels, num_z) from czifile axes string and shape.
        CZI dimension order is file-dependent (e.g. TZCYX, TCZYX, BSTCZYX0); we read T,C,Z from axes."""
        if not axes or not shape or len(axes) != len(shape):
            return (1, 1, 1)
        try:
            num_time = int(shape[axes.index('T')]) if 'T' in axes else 1
            num_channels = int(shape[axes.index('C')]) if 'C' in axes else 1
            num_z = int(shape[axes.index('Z')]) if 'Z' in axes else 1
            return (num_time, num_channels, num_z)
        except (ValueError, IndexError, TypeError):
            return (1, 1, 1)

    @staticmethod
    def _czi_pixel_size_from_metadata(czi_file):
        """Extract pixel size (µm) from CZI metadata XML. Returns None if not found."""
        try:
            meta = czi_file.metadata(raw=True) if callable(getattr(czi_file, 'metadata', None)) else getattr(czi_file, 'metadata', None)
            if not meta or not isinstance(meta, str):
                return None
            import xml.etree.ElementTree as ET
            root = ET.fromstring(meta)
            for elem in root.iter():
                tag = (elem.tag or '').split('}')[-1]
                if tag != 'Distance':
                    continue
                did = (elem.get('Id') or elem.get('id') or '').strip()
                if did != 'X':
                    continue
                val_el = elem.find('Value')
                if val_el is None:
                    val_el = next((c for c in elem if (c.tag or '').split('}')[-1] == 'Value'), None)
                if val_el is not None and getattr(val_el, 'text', None):
                    try:
                        scale_x = float(val_el.text.strip())
                        if scale_x > 0 and scale_x < 0.0001:
                            scale_x *= 1000000.0
                        return scale_x if scale_x > 0 else None
                    except ValueError:
                        pass
            return None
        except Exception:
            return None

    @staticmethod
    def _czi_slice_to_t_y_x(arr, axes, z_index=0, c_index=0):
        """Reduce CZI array to (T, Y, X) by taking channel c_index and z_index. axes matches arr.shape.
        Preserves time dimension (never squeezes to 0 frames). Transposes to (T, Y, X) so preview is never a thin line."""
        if not axes or len(axes) != len(arr.shape):
            return arr
        indices = []
        for (i, ax) in enumerate(axes):
            if ax == 'C' and arr.shape[i] > 0:
                ci = min(int(c_index), arr.shape[i] - 1) if c_index is not None else 0
                indices.append(max(0, ci))
            elif ax == 'Z' and arr.shape[i] > 0:
                zi = min(int(z_index), arr.shape[i] - 1) if z_index is not None else 0
                indices.append(max(0, zi))
            else:
                indices.append(slice(None))
        out = arr[tuple(indices)]
        remaining_axes = [axes[i] for i in range(len(axes)) if axes[i] not in ('C', 'Z')]
        if len(remaining_axes) >= 3 and 'T' in remaining_axes and ('Y' in remaining_axes) and ('X' in remaining_axes):
            t_pos = remaining_axes.index('T')
            y_pos = remaining_axes.index('Y')
            x_pos = remaining_axes.index('X')
            other = [i for i in range(len(remaining_axes)) if remaining_axes[i] not in ('T', 'Y', 'X')]
            perm = [t_pos, y_pos, x_pos] + other
            out = np.transpose(out, perm)
        while out.ndim > 3 and out.shape[-1] in (1, 3, 4):
            out = out[..., 0]
        while out.ndim > 3:
            squeezed = out.squeeze()
            if squeezed.ndim < 3:
                break
            out = squeezed
        if out.ndim == 2:
            out = out[np.newaxis, ...]
        return out

    @staticmethod
    def _czi_find_first_nonempty_plane(arr, axes):
        """Find the (channel, z) plane that is best for preview: first plane with any pixel > 0.
        If that would be too dim, uses the plane with maximum total intensity (most informative slice).
        Loops over Z then C. Returns (c_index, z_index); (0, 0) if none or no C/Z in axes."""
        if not axes or len(axes) != len(arr.shape):
            return (0, 0)
        c_axis = axes.index('C') if 'C' in axes else -1
        z_axis = axes.index('Z') if 'Z' in axes else -1
        if c_axis < 0 and z_axis < 0:
            return (0, 0)
        n_c = arr.shape[c_axis] if c_axis >= 0 else 1
        n_z = arr.shape[z_axis] if z_axis >= 0 else 1
        (best_c, best_z) = (0, 0)
        best_sum = -1.0
        for z in range(n_z):
            for c in range(n_c):
                indices = [slice(None)] * len(arr.shape)
                if c_axis >= 0:
                    indices[c_axis] = c
                if z_axis >= 0:
                    indices[z_axis] = z
                plane = np.asarray(arr[tuple(indices)])
                if plane.size == 0:
                    continue
                s = float(np.sum(plane))
                if s > best_sum:
                    best_sum = s
                    (best_c, best_z) = (c, z)
        return (best_c, best_z)

    @staticmethod
    def _czi_slice_to_t_c_y_x(arr, axes, max_frames=None, frame_start=None, frame_end=None):
        """Reduce CZI array to (T, C, Y, X) by taking first Z only. axes matches arr.shape.
        Returns (array, dims_str). Limits T: use [frame_start:frame_end) if frame_start is set, else first max_frames."""
        if not axes or len(axes) != len(arr.shape):
            return (arr, '')
        indices = []
        t_idx = axes.index('T') if 'T' in axes else None
        for (i, ax) in enumerate(axes):
            if ax == 'Z' and arr.shape[i] > 0:
                indices.append(0)
            elif ax == 'T' and t_idx is not None:
                size = arr.shape[i]
                if frame_start is not None:
                    end = frame_end if frame_end is not None else frame_start + 1
                    indices.append(slice(max(0, min(frame_start, size)), min(end, size)))
                elif max_frames is not None:
                    indices.append(slice(0, min(int(size), max_frames)))
                else:
                    indices.append(slice(None))
            else:
                indices.append(slice(None))
        out = arr[tuple(indices)]
        while out.ndim > 4 and out.shape[-1] in (1, 3, 4):
            out = out[..., 0] if out.shape[-1] == 1 else out[..., 0]
        dims_str = ''.join((ax for ax in axes if ax not in ('Z', '0')))
        return (out, dims_str if dims_str else 'TCYX')

    @staticmethod
    def _czi_slice_to_t_c_z_y_x(arr, axes, max_frames=None, frame_start=None, frame_end=None):
        """Reduce CZI array to (T, C, Z, Y, X) keeping Z dimension. axes matches arr.shape.
        Returns (array, dims_str, z_axis). Limits T: use [frame_start:frame_end) if frame_start is set, else first max_frames."""
        if not axes or len(axes) != len(arr.shape):
            return (arr, '', -1)
        indices = []
        t_idx = axes.index('T') if 'T' in axes else None
        z_idx_in_axes = axes.index('Z') if 'Z' in axes else -1
        for (i, ax) in enumerate(axes):
            if ax == 'T' and t_idx is not None:
                size = arr.shape[i]
                if frame_start is not None:
                    end = frame_end if frame_end is not None else frame_start + 1
                    indices.append(slice(max(0, min(frame_start, size)), min(end, size)))
                elif max_frames is not None:
                    indices.append(slice(0, min(int(size), max_frames)))
                else:
                    indices.append(slice(None))
            else:
                indices.append(slice(None))
        out = arr[tuple(indices)]
        while out.ndim > 5 and out.shape[-1] in (1, 3, 4):
            out = out[..., 0] if out.shape[-1] == 1 else out[..., 0]
        dims_str = ''.join((ax for ax in axes if ax not in ('0',)))
        if 'Z' not in dims_str:
            return (out, dims_str, -1)
        z_axis = dims_str.index('Z')
        return (out, dims_str, z_axis)

    def _get_czi_info_via_czifile(self, file_path):
        """Extract CZI file info using the czifile library (https://pypi.org/project/czifile/).
        Uses the file's axes string so T, C, Z are interpreted correctly regardless of dimension order."""
        try:
            import czifile
            with czifile.CziFile(file_path) as czi_file:
                shape = getattr(czi_file, 'shape', None)
                axes = getattr(czi_file, 'axes', None)
                if shape and axes and (len(axes) == len(shape)):
                    (num_time, num_channels, num_z) = self._czi_dims_from_axes(axes, shape)
                else:
                    (num_time, num_channels, num_z) = (1, 1, 1)
                    if shape and len(shape) >= 3:
                        num_time = int(shape[0])
                        num_channels = int(shape[1]) if len(shape) > 1 else 1
                        num_z = int(shape[2]) if len(shape) > 2 else 1
                channel_names = [f'Channel_{i + 1}' for i in range(num_channels)]
                pixel_size = self._czi_pixel_size_from_metadata(czi_file)
                return {'file_type': 'czi', 'num_time': num_time, 'num_channels': num_channels, 'num_z': num_z, 'channel_names': channel_names, 'pixel_size': pixel_size}
        except Exception:
            return None

    def _get_czi_info(self, file_path):
        """Extract CZI file info without updating UI (for folder scanning). Tries BioIO first, then czifile."""
        info = bioio_get_info(Path(file_path))
        if info and info.get('file_type') == 'czi':
            return {k: v for (k, v) in info.items() if k in ('file_type', 'num_time', 'num_channels', 'num_z', 'channel_names', 'pixel_size', 'scenes')}
        return self._get_czi_info_via_czifile(file_path)

    def load_file_info(self, file_path):
        """Load and display microscopy file information (ND2, LIF, CZI, or TIFF)"""
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        try:
            if file_ext == '.nd2':
                self._load_nd2_info(file_path)
            elif file_ext == '.lif':
                self._load_lif_info(file_path)
            elif file_ext == '.czi':
                self._load_czi_info(file_path)
            elif file_ext in ['.tif', '.tiff']:
                self._load_tiff_info(file_path)
            else:
                show_message_dialog(self, 'Unsupported Format', f"File format '{file_ext}' is not supported.\nSupported formats: ND2, LIF, CZI, TIFF", kind='warning')
                return
        except Exception as e:
            show_message_dialog(self, 'Error Loading File', f'Could not load file information:\n{str(e)}\n\n{traceback.format_exc()}', kind='error')
            self.metadata_text.setText(f'Error loading file: {str(e)}')
            self.metadata_text.setStyleSheet('background-color: #2b2b2b; color: #ffffff; font-size: 10pt;')
            self._update_tiff_metadata_button_state()
        finally:
            self.refresh_conversion_candidate_list()

    def _load_nd2_info(self, file_path):
        """Load and display ND2 file information. All metadata from tiff_metadata (single source of truth)."""
        try:
            from pathlib import Path
            from bactomate.modules.tiff_metadata import read_metadata_from_path, format_metadata_for_display
            self.parent_window.log_output(f'Loading ND2 file info: {file_path}\n')
            meta = read_metadata_from_path(Path(file_path))
            num_time = meta.get('frame_count') or 1
            num_channels = meta.get('channel_count') or 1
            num_z = meta.get('z_count') or 1
            channel_names = meta.get('channel_names') or []
            if len(channel_names) != num_channels:
                channel_names = [f'Channel_{i + 1}' for i in range(num_channels)]
            pixel_size = meta.get('pixel_size_um')
            time_per_frame_min = meta.get('time_per_frame_min')
            metadata_str = format_metadata_for_display(meta)
            self._update_file_info_ui(file_path, 'nd2', num_time, num_channels, num_z, channel_names, pixel_size, metadata_str, time_per_frame_min=time_per_frame_min)
            return
        except ImportError:
            show_message_dialog(self, 'ND2 Library Missing', "The 'nd2' library is not installed.\n\nInstall it with: pip install nd2", kind='warning')
            return
        except Exception as e:
            show_message_dialog(self, 'Error Loading ND2', f'Could not load ND2 file:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')
            return

    def _load_lif_info(self, file_path):
        """Load and display LIF file information. Uses BioIO when available, then aicsimageio/readlif."""
        try:
            self.parent_window.log_output(f'Loading LIF file info: {file_path}\n')
            try:
                info = bioio_get_info(Path(file_path))
                if info and info.get('file_type') == 'lif':
                    self._update_file_info_ui(file_path, 'lif', info['num_time'], info['num_channels'], info['num_z'], info['channel_names'], info.get('pixel_size'))
                    return
            except Exception:
                pass
            aics_success = False
            try:
                import numpy as np
                from aicsimageio import AICSImage
                lif_reader = AICSImage(file_path)
                try:
                    dims = lif_reader.dims
                    if hasattr(dims, 'order'):
                        dims_str = dims.order
                    elif isinstance(dims, str):
                        dims_str = dims
                    elif isinstance(dims, (list, tuple)):
                        dims_str = ''.join((str(d) for d in dims))
                    else:
                        dims_str = str(dims)
                    try:
                        if hasattr(lif_reader, 'get_xarray_data'):
                            xdata = lif_reader.get_xarray_data()
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            shape = lif_reader.shape
                            if not isinstance(shape, (tuple, list)):
                                shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        shape = lif_reader.shape
                        if isinstance(shape, dict):
                            num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                            num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                            num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                        else:
                            shape_list = list(shape) if hasattr(shape, '__iter__') else [shape]
                            num_time = shape_list[dims_str.index('T')] if 'T' in dims_str and len(shape_list) > dims_str.index('T') else 1
                            num_channels = shape_list[dims_str.index('C')] if 'C' in dims_str and len(shape_list) > dims_str.index('C') else 1
                            num_z = shape_list[dims_str.index('Z')] if 'Z' in dims_str and len(shape_list) > dims_str.index('Z') else 1
                    channel_names = []
                    try:
                        if hasattr(lif_reader, 'channel_names'):
                            channel_names = lif_reader.channel_names
                    except:
                        pass
                    if not channel_names or len(channel_names) != num_channels:
                        channel_names = [f'Channel_{i + 1}' for i in range(num_channels)]
                    pixel_size = self._safe_lif_pixel_size(lif_reader)
                    self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z, channel_names, pixel_size)
                    aics_success = True
                finally:
                    if hasattr(lif_reader, 'close'):
                        try:
                            lif_reader.close()
                        except:
                            pass
            except ImportError:
                pass
            except Exception:
                self.parent_window.log_output('AICSImage failed for this LIF, trying readlif...\n')
                pass
            if aics_success:
                return
            try:
                from readlif.reader import LifFile
                lif_file = LifFile(file_path)
                if getattr(lif_file, 'num_images', 0) <= 0:
                    raise ValueError('No image series found in LIF file')
                log_cb = lambda msg: self.parent_window.log_output(msg)
                best_idx = self._readlif_best_image_index(lif_file, log_cb=log_cb)
                image = lif_file.get_image(best_idx)
                (num_time, num_z, num_channels) = self._readlif_dims(image, log_cb=log_cb)
                channel_names = [f'Channel_{i + 1}' for i in range(num_channels)]
                pixel_size = None
                if hasattr(image, 'scale') and image.scale and (len(image.scale) >= 2):
                    try:
                        (sx, sy) = (float(image.scale[0]), float(image.scale[1]))
                        if sx > 0:
                            pixel_size = 1.0 / sx
                        elif sy > 0:
                            pixel_size = 1.0 / sy
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z, channel_names, pixel_size)
            except ImportError:
                show_message_dialog(self, 'LIF Library Missing', "The 'aicsimageio' or 'readlif' library is not installed.\n\nInstall with: pip install aicsimageio readlif", kind='warning')
            except Exception as e:
                show_message_dialog(self, 'Error Loading LIF', f'Could not load LIF file:\n{str(e)}', kind='error')
                self.parent_window.log_output(f'Error: {str(e)}\n')
        except Exception as e:
            show_message_dialog(self, 'Error Loading LIF', f'Could not load LIF file:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')

    def _load_czi_info(self, file_path):
        """Load and display CZI file information. Uses BioIO when available, then czifile."""
        try:
            self.parent_window.log_output(f'Loading CZI file info: {file_path}\n')
            info = bioio_get_info(Path(file_path))
            if info and info.get('file_type') == 'czi':
                self._update_file_info_ui(file_path, 'czi', info['num_time'], info['num_channels'], info['num_z'], info['channel_names'], info.get('pixel_size'))
                return
            import czifile
        except ImportError:
            show_message_dialog(self, 'CZI Library Missing', "CZI support requires the 'bioio' and 'bioio-czi' packages, or the 'czifile' package.\n\nInstall with: pip install bioio bioio-czi\nOr fallback: pip install czifile", kind='warning')
            return
        try:
            info = self._get_czi_info_via_czifile(file_path)
            if info is None:
                raise RuntimeError('Could not read CZI file')
            self._update_file_info_ui(file_path, 'czi', info['num_time'], info['num_channels'], info['num_z'], info['channel_names'], info['pixel_size'])
        except Exception as e:
            show_message_dialog(self, 'Error Loading CZI', f'Could not load CZI file:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')

    def _load_tiff_info(self, file_path):
        """Load and display TIFF file information. Uses BioIO when available, then centralized tiff_metadata."""
        try:
            from bactomate.modules.tiff_metadata import read_calibration_from_path, read_metadata_from_path
            self.parent_window.log_output(f'Loading TIFF file info: {file_path}\n')
            path = Path(file_path)
            info = get_tiff_file_info(path)
            if info is None:
                raise RuntimeError('Could not read TIFF file')
            meta = read_metadata_from_path(path)
            pixel_size = info.get('pixel_size')
            if meta.get('pixel_size_um') is not None and float(meta['pixel_size_um']) > 0:
                pixel_size = float(meta['pixel_size_um'])
            if pixel_size is None:
                (ps, _tp) = read_calibration_from_path(path)
                if ps > 0:
                    pixel_size = ps
            time_per_frame_min = None
            if meta.get('time_per_frame_min') is not None and float(meta.get('time_per_frame_min', 0)) > 0:
                time_per_frame_min = float(meta['time_per_frame_min'])
            channel_names = meta.get('channel_names') or info.get('channel_names')
            if not channel_names or len(channel_names) != info.get('num_channels', 1):
                channel_names = info.get('channel_names') or [f'Channel_{i + 1}' for i in range(info.get('num_channels', 1))]
            ch_src = meta.get('channel_names_source') or info.get('channel_names_source') or ''
            self._update_file_info_ui(file_path, 'tiff', info['num_time'], info['num_channels'], info['num_z'], channel_names, pixel_size, time_per_frame_min=time_per_frame_min, channel_names_source=ch_src)
        except Exception as e:
            show_message_dialog(self, 'Error Loading TIFF', f'Could not load TIFF file:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')

    def _update_file_info_ui(self, file_path, file_type, num_time, num_channels, num_z, channel_names, pixel_size, metadata_str=None, time_per_frame_min=None, channel_names_source: str=''):
        """Update UI with file information (common for all file types)."""
        self.file_info = {'path': file_path, 'file_type': file_type, 'num_time': num_time, 'num_channels': num_channels, 'num_z': num_z, 'channel_names': channel_names, 'pixel_size': pixel_size, 'time_per_frame_min': time_per_frame_min}
        line_txt = (self.file_path.text() or '').strip()
        try:
            lp = Path(line_txt)
            if lp.is_dir():
                self.file_info['folder_path'] = line_txt
            else:
                self.file_info.pop('folder_path', None)
        except Exception:
            self.file_info.pop('folder_path', None)
        n_t = num_time
        self.max_frames.setMaximum(n_t)
        self.max_frames.setValue(n_t)
        self.single_frame_spin.setMaximum(max(1, n_t))
        if self.single_frame_spin.value() > n_t:
            self.single_frame_spin.setValue(n_t)
        self._rebuild_channel_options(channel_names, num_channels, names_source=channel_names_source)
        metadata_content = f'File: {Path(file_path).name} ({file_type.upper()})\n'
        metadata_content += f'Dimensions: T={num_time} frames, C={num_channels} channels, Z={num_z} slices\n'
        if num_channels > 1:
            metadata_content += f"Channels: {', '.join(channel_names)}\n"
            if channel_names_source:
                metadata_content += f'Channel names source: {channel_names_source}\n'
        metadata_content += '\n'
        if metadata_str is None:
            metadata_content += f'Metadata:\n  File Type: {file_type.upper()}\n'
            if pixel_size:
                metadata_content += f'  Pixel Size: {pixel_size:.4f} µm\n'
            else:
                metadata_content += '  Pixel Size: Not found in metadata\n'
            if time_per_frame_min is not None and time_per_frame_min > 0:
                metadata_content += f'  Time per frame: {time_per_frame_min:.6f} min ({time_per_frame_min * 60:.4f} s)\n'
            else:
                metadata_content += '  Time per frame: Not found in metadata\n'
        else:
            metadata_content += metadata_str
        self.metadata_text.setText(metadata_content)
        self.metadata_text.setStyleSheet('background-color: #2b2b2b; color: #ffffff; font-size: 10pt;')
        self.parent_window.shared_metadata['pixel_size'] = pixel_size if pixel_size else None
        self.parent_window.shared_metadata['time_per_frame_min'] = time_per_frame_min if time_per_frame_min is not None and time_per_frame_min > 0 else None
        self.parent_window.update_all_tabs_with_metadata()
        self.preview_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
        self._expand_conversion_params_section()
        QTimer.singleShot(100, self._refresh_embedded_conversion_preview)
        self.parent_window.log_output(f'✓ Loaded: {num_time} frames, {num_channels} channels\n')
        if pixel_size:
            self.parent_window.log_output(f'✓ Pixel size extracted: {pixel_size:.4f} µm (updated in all tabs)\n')
        self._update_tiff_metadata_button_state()

    def _metadata_dialog_parent(self) -> QWidget:
        """Top-level parent so the editor dialog is not hidden behind the module shell."""
        parent_window = getattr(self, 'parent_window', None)
        if parent_window is not None:
            return parent_window
        win = self.window()
        return win if win is not None else self

    def _preview_selected_file_path(self) -> Optional[Path]:
        surf = self._preview_surface
        if surf is None:
            return None
        combo = getattr(surf, '_file_combo', None)
        if combo is None:
            return None
        raw = combo.currentData()
        if not raw:
            return None
        try:
            path = Path(str(raw))
        except (TypeError, ValueError):
            return None
        if path.is_file() and path.suffix.lower() in _METADATA_SOURCE_SUFFIXES:
            return path
        return None

    def _coerce_metadata_source_path(self, raw: str | Path) -> Optional[Path]:
        try:
            path = Path(raw)
        except (TypeError, ValueError):
            return None
        if path.is_file() and path.suffix.lower() in _METADATA_SOURCE_SUFFIXES:
            return path
        return None

    def _active_metadata_source_path(self) -> Optional[Path]:
        """File whose output metadata can be edited (preview pick, loaded file_info, or path line)."""
        for candidate in (
            self._preview_selected_file_path(),
            (self.file_info or {}).get('path') if isinstance(self.file_info, dict) else None,
            (self.file_path.text() or '').strip(),
        ):
            if not candidate:
                continue
            path = self._coerce_metadata_source_path(candidate)
            if path is not None:
                return path
        return None

    def _update_tiff_metadata_button_state(self) -> None:
        btn = getattr(self, 'edit_tiff_metadata_btn', None)
        if btn is None:
            return
        btn.setEnabled(self._active_metadata_source_path() is not None)

    def _open_tiff_metadata_editor(self) -> None:
        source_path = self._active_metadata_source_path()
        if source_path is None:
            show_message_dialog(
                self._metadata_dialog_parent(),
                'Output metadata',
                'Load a microscopy file (or pick one from the preview list when using a folder) to edit output metadata.',
                kind='warning',
            )
            return
        try:
            self._show_output_metadata_editor_dialog(source_path)
        except Exception as exc:
            self.parent_window.log_output(f'Error opening metadata editor: {exc}\n')
            show_message_dialog(
                self._metadata_dialog_parent(),
                'Output metadata',
                f'Could not open the metadata editor:\n{exc}',
                kind='error',
            )

    def _show_output_metadata_editor_dialog(self, source_path: Path) -> None:
        info = self.file_info if isinstance(self.file_info, dict) else {}
        meta = read_metadata_from_path(source_path)
        suffix = source_path.suffix.lower()
        if suffix in ('.tif', '.tiff'):
            file_type = 'tiff'
        elif suffix == '.nd2':
            file_type = 'nd2'
        elif suffix == '.lif':
            file_type = 'lif'
        elif suffix == '.czi':
            file_type = 'czi'
        else:
            file_type = str(info.get('file_type') or suffix.lstrip('.')).lower()
        is_tiff = suffix in ('.tif', '.tiff')
        num_channels = max(1, int(meta.get('channel_count') or info.get('num_channels') or 1))
        channel_names = list(meta.get('channel_names') or info.get('channel_names') or [f'Channel_{i + 1}' for i in range(num_channels)])
        if len(channel_names) < num_channels:
            channel_names.extend([f'Channel_{i + 1}' for i in range(len(channel_names), num_channels)])
        channel_names = channel_names[:num_channels]

        parent = self._metadata_dialog_parent()
        dlg = QDialog(parent)
        dlg.setWindowTitle('Edit output metadata')
        dlg.setMinimumWidth(560)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setStyleSheet(
            'QDialog { background-color: #1e1e1e; }'
            'QLabel { color: #e8eef6; font-size: 11pt; }'
            'QLineEdit, QDoubleSpinBox, QTextEdit { color: #e8eef6; background-color: #2b2b2b; }'
        )
        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(16, 14, 16, 12)
        outer.setSpacing(10)

        intro_text = (
            f'Edit output metadata for {source_path.name} ({file_type.upper()}). '
            'The source file is not modified; these values are written into generated TIFF files.'
        )
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(8)

        pixel_spin = QDoubleSpinBox()
        pixel_spin.setDecimals(6)
        pixel_spin.setRange(0.0, 100.0)
        pixel_spin.setSingleStep(0.0001)
        pixel_spin.setSpecialValueText('Not set')
        ps = meta.get('pixel_size_um') or info.get('pixel_size')
        if ps is not None and float(ps) > 0:
            pixel_spin.setValue(float(ps))
        else:
            pixel_spin.setValue(0.0)
        form.addRow('Pixel size (µm/px):', pixel_spin)

        time_min_spin = QDoubleSpinBox()
        time_min_spin.setDecimals(8)
        time_min_spin.setRange(0.0, 1e6)
        time_min_spin.setSingleStep(0.0001)
        time_min_spin.setSpecialValueText('Not set')
        tp = meta.get('time_per_frame_min') or info.get('time_per_frame_min')
        if tp is not None and float(tp) > 0:
            time_min_spin.setValue(float(tp))
        else:
            time_min_spin.setValue(0.0)
        form.addRow('Time per frame (min):', time_min_spin)

        time_sec_spin = QDoubleSpinBox()
        time_sec_spin.setDecimals(6)
        time_sec_spin.setRange(0.0, 1e8)
        time_sec_spin.setSingleStep(0.01)
        time_sec_spin.setSpecialValueText('Not set')
        if tp is not None and float(tp) > 0:
            time_sec_spin.setValue(float(tp) * 60.0)
        else:
            time_sec_spin.setValue(0.0)

        def _sync_sec_from_min() -> None:
            if time_min_spin.value() > 0:
                time_sec_spin.blockSignals(True)
                time_sec_spin.setValue(time_min_spin.value() * 60.0)
                time_sec_spin.blockSignals(False)

        def _sync_min_from_sec() -> None:
            if time_sec_spin.value() > 0:
                time_min_spin.blockSignals(True)
                time_min_spin.setValue(time_sec_spin.value() / 60.0)
                time_min_spin.blockSignals(False)

        time_min_spin.valueChanged.connect(lambda _v: _sync_sec_from_min())
        time_sec_spin.valueChanged.connect(lambda _v: _sync_min_from_sec())
        form.addRow('Time per frame (s):', time_sec_spin)

        channel_edits: list[QLineEdit] = []
        for (i, name) in enumerate(channel_names):
            if num_channels <= 1:
                break
            edit = QLineEdit(str(name))
            channel_edits.append(edit)
            form.addRow(f'Channel {i + 1}:', edit)

        obj_name_edit = QLineEdit(str(meta.get('objective_name') or ''))
        form.addRow('Objective:', obj_name_edit)

        obj_mag_spin = QDoubleSpinBox()
        obj_mag_spin.setDecimals(1)
        obj_mag_spin.setRange(0.0, 500.0)
        obj_mag_spin.setSpecialValueText('Not set')
        obj_mag = meta.get('objective_magnification')
        obj_mag_spin.setValue(float(obj_mag) if obj_mag is not None and float(obj_mag) > 0 else 0.0)
        form.addRow('Objective magnification:', obj_mag_spin)

        exposure_spin = QDoubleSpinBox()
        exposure_spin.setDecimals(3)
        exposure_spin.setRange(0.0, 1e7)
        exposure_spin.setSpecialValueText('Not set')
        exp_ms = meta.get('exposure_ms')
        exposure_spin.setValue(float(exp_ms) if exp_ms is not None and float(exp_ms) > 0 else 0.0)
        form.addRow('Exposure (ms):', exposure_spin)

        optics_edit = QLineEdit(str(meta.get('optics') or ''))
        form.addRow('Optics:', optics_edit)

        capturing_edit = QLineEdit(str(meta.get('capturing') or ''))
        form.addRow('Capturing:', capturing_edit)
        outer.addLayout(form)

        notes_edit = QTextEdit()
        notes_edit.setPlaceholderText('Additional acquisition notes (optional)')
        notes_edit.setMaximumHeight(90)
        outer.addWidget(QLabel('Acquisition notes:'))
        outer.addWidget(notes_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        save_btn = button_box.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setText('Use for output TIFFs')
        style_dialog_button_box(button_box, horizontal_padding=36)
        button_box.rejected.connect(dlg.reject)
        outer.addWidget(button_box)

        def _collect_overrides() -> TiffMetadataOverrides:
            names = [edit.text().strip() for edit in channel_edits] if channel_edits else None
            return TiffMetadataOverrides(
                pixel_size_um=float(pixel_spin.value()) if pixel_spin.value() > 0 else None,
                time_per_frame_min=float(time_min_spin.value()) if time_min_spin.value() > 0 else None,
                channel_names=names,
                objective_name=obj_name_edit.text().strip() or None,
                objective_magnification=float(obj_mag_spin.value()) if obj_mag_spin.value() > 0 else None,
                exposure_ms=float(exposure_spin.value()) if exposure_spin.value() > 0 else None,
                optics=optics_edit.text().strip() or None,
                capturing=capturing_edit.text().strip() or None,
                acquisition_notes=notes_edit.toPlainText().strip() or None,
            )

        def _on_write() -> None:
            overrides = _collect_overrides()
            if is_tiff:
                ok, err = validate_tiff_metadata_overrides(source_path, overrides)
            else:
                names = overrides.channel_names
                ok = True
                err = ''
                if names is not None and len(names) != num_channels:
                    ok = False
                    err = f'Channel name count ({len(names)}) must match file channel count ({num_channels}).'
            if not ok:
                show_message_dialog(self, 'Invalid metadata', err, kind='warning')
                return
            self._apply_output_metadata_overrides(overrides)
            dlg.accept()
            self.parent_window.log_output(f'✓ Using edited metadata for output TIFFs from {source_path.name}\n')
            show_message_dialog(
                self,
                'Output metadata saved',
                'These values will be written into TIFF files generated from this input. The source file was not modified.',
                kind='info',
            )

        button_box.accepted.connect(_on_write)
        dlg.adjustSize()
        try:
            center = parent.frameGeometry().center()
            dlg.move(center.x() - dlg.width() // 2, center.y() - dlg.height() // 2)
        except Exception:
            pass
        dlg.raise_()
        dlg.activateWindow()
        dlg.exec()

    def _apply_output_metadata_overrides(self, overrides: TiffMetadataOverrides) -> None:
        """Store manual metadata values for generated TIFF output without changing the source file."""
        if not isinstance(self.file_info, dict):
            return
        if overrides.pixel_size_um is not None:
            self.file_info['pixel_size'] = float(overrides.pixel_size_um)
            self.parent_window.shared_metadata['pixel_size'] = float(overrides.pixel_size_um)
        if overrides.time_per_frame_min is not None:
            self.file_info['time_per_frame_min'] = float(overrides.time_per_frame_min)
            self.parent_window.shared_metadata['time_per_frame_min'] = float(overrides.time_per_frame_min)
        if overrides.channel_names is not None:
            names = [str(name).strip() for name in overrides.channel_names]
            self.file_info['channel_names'] = names
            if getattr(self, 'channel_name_edits', None) and len(self.channel_name_edits) >= len(names):
                for i, name in enumerate(names):
                    self.channel_name_edits[i].setText(name)
        self.file_info['metadata_overrides'] = overrides.to_dict()
        self.parent_window.update_all_tabs_with_metadata()

        lines = []
        if overrides.pixel_size_um is not None:
            lines.append(f'  Pixel Size: {float(overrides.pixel_size_um):.4f} µm')
        if overrides.time_per_frame_min is not None:
            tp = float(overrides.time_per_frame_min)
            lines.append(f'  Time per frame: {tp:.6f} min ({tp * 60:.4f} s)')
        if overrides.channel_names is not None:
            lines.append(f'  Channel names: {", ".join(overrides.channel_names)}')
        if overrides.objective_name:
            lines.append(f'  Objective: {overrides.objective_name}')
        if overrides.objective_magnification is not None:
            lines.append(f'  Objective magnification: {float(overrides.objective_magnification):.1f}x')
        if overrides.exposure_ms is not None:
            lines.append(f'  Exposure: {float(overrides.exposure_ms):.3f} ms')
        if overrides.optics:
            lines.append(f'  Optics: {overrides.optics}')
        if overrides.capturing:
            lines.append(f'  Capturing: {overrides.capturing}')
        if overrides.acquisition_notes:
            lines.append(f'  Notes: {overrides.acquisition_notes}')
        if lines:
            self.metadata_text.append('\nManual output metadata overrides:\n' + '\n'.join(lines))

    def preview_file(self):
        """Preview microscopy file in a dialog with frame navigation"""
        if not self.file_path.text() or not Path(self.file_path.text()).exists():
            show_message_dialog(self, 'No File', 'Please select a valid microscopy file first.', kind='warning')
            return
        file_path_obj = Path(self.file_path.text())
        if file_path_obj.is_dir():
            folder = file_path_obj
            supported_files = list_convertible_microscopy_files(folder, max_files=_FOLDER_CONVERTIBLE_CAP_FOR_LOAD)
            if not supported_files:
                show_message_dialog(self, 'No Files Found', 'No supported microscopy files (ND2, LIF, CZI, TIFF) found in the selected folder.\nOnly files directly in this folder are listed (not inside subfolders).', kind='warning')
                return
            if len(supported_files) == 1:
                selected_file = supported_files[0]
            else:
                file_dialog = QDialog(self)
                file_dialog.setWindowTitle('Select file to preview')
                file_dialog.setMinimumSize(500, 400)
                layout = QVBoxLayout()
                label = QLabel('Files in folder (select one to preview; subfolders are not listed):')
                label.setStyleSheet('color: #ffffff; font-size: 11pt; padding: 10px;')
                label.setWordWrap(True)
                layout.addWidget(label)
                file_list = QListWidget()
                file_list.setStyleSheet('QListWidget { background-color: #2b2b2b; color: #ffffff; font-size: 11pt; border: 1px solid #555555; border-radius: 4px; padding: 5px; }QListWidget::item { padding: 8px; border-bottom: 1px solid #3a3a3a; }QListWidget::item:selected { background-color: #5D6D7E; color: #ffffff; }QListWidget::item:hover { background-color: #3a3a3a; }')
                for p in supported_files:
                    file_list.addItem(p.name)
                file_list.setCurrentRow(0)

                def on_double_click():
                    file_dialog.accept()
                file_list.itemDoubleClicked.connect(on_double_click)
                layout.addWidget(file_list)
                btn_layout = QHBoxLayout()
                btn_layout.addStretch()
                cancel_btn = QPushButton('Cancel')
                cancel_btn.clicked.connect(file_dialog.reject)
                ok_btn = QPushButton('Preview')
                ok_btn.clicked.connect(file_dialog.accept)
                btn_layout.addWidget(cancel_btn)
                btn_layout.addWidget(ok_btn)
                layout.addLayout(btn_layout)
                file_dialog.setLayout(layout)
                if file_dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                idx = file_list.currentRow()
                if idx < 0 or idx >= len(supported_files):
                    return
                selected_file = supported_files[idx]
            file_type = self._detect_preview_file_type(selected_file)
            if file_type is None:
                show_message_dialog(self, 'Unsupported Format', f'Preview not yet implemented for {selected_file.suffix} files.', kind='warning')
                return
            self._preview_by_type(file_type, selected_file)
            return
        if not self.file_info:
            show_message_dialog(self, 'File Not Loaded', 'Please load file information first.', kind='warning')
            return
        file_type = self.file_info.get('file_type', 'nd2')
        self._preview_by_type(file_type, file_path_obj)

    def _detect_preview_file_type(self, file_path: Path) -> Optional[str]:
        ext = file_path.suffix.lower()
        if ext == '.nd2':
            return 'nd2'
        if ext == '.lif':
            return 'lif'
        if ext == '.czi':
            return 'czi'
        if ext in ('.tif', '.tiff'):
            return 'tiff'
        return None

    def _preview_by_type(self, file_type: str, file_path: Path) -> None:
        ft = (file_type or '').strip().lower()
        if ft == 'nd2':
            self._preview_nd2(file_path=file_path)
            return
        if ft == 'lif':
            self._preview_lif(file_path=file_path)
            return
        if ft == 'czi':
            self._preview_czi(file_path=file_path)
            return
        if ft == 'tiff':
            self._preview_tiff(file_path=file_path)
            return
        show_message_dialog(self, 'Unsupported Format', f'Preview not yet implemented for {file_type} files.', kind='warning')

    def _get_bleach_preview_kwargs(self):
        """Return kwargs for preview bleach correction (from current tab settings)."""
        bleach_apply = self.bleach_correct.isChecked()
        bleach_method = ['exponential', 'linear', 'histogram_matching', 'per_frame_minmax'][self.bleach_method_combo.currentIndex()]
        bleach_channel_indices = None
        if getattr(self, 'bleach_channel_checks', None):
            indices = [i for (i, cb) in enumerate(self.bleach_channel_checks) if cb.isChecked()]
            if indices and len(indices) < len(self.bleach_channel_checks):
                bleach_channel_indices = indices
        return dict(bleach_apply=bleach_apply, bleach_method=bleach_method, bleach_channel_indices=bleach_channel_indices)

    def _create_unified_preview_ui(self, preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path, bleach_apply=False, bleach_method='exponential', bleach_channel_indices=None, nd2_axis_keys: Optional[tuple[str, ...]]=None):
        """Create unified preview UI with improved histogram layout (used by all file types).
        When bleach_apply is True, optional checkbox 'Show bleach correction' toggles live preview of correction.

        Callers must pass all_frames that own their data (e.g. .copy() taken while reader was open);
        otherwise later .copy() in this UI or in bleach correction can segfault."""
        import numpy as np
        import sys
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QKeySequence, QShortcut
        all_frames = np.ascontiguousarray(all_frames)
        if not all_frames.flags.owndata:
            all_frames = all_frames.copy()
        raw_frames = all_frames
        display_frames_ref = {'data': all_frames, 'view_label': 'Raw'}
        preview_dialog.current_z = 0
        preview_runtime_timers: list[QTimer] = []
        preview_runtime_cleaned = False
        title_label = QLabel(f'Timepoint 1 of {num_time}  |  {Path(file_path).name}')
        title_label.setStyleSheet('font-weight: bold; color: #ddd;')
        layout.addWidget(title_label)
        image_canvas = ImageCanvas()
        image_canvas.setToolTip('Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).')
        image_canvas.setMinimumSize(400, 300)
        image_canvas.setStyleSheet('background-color: #1e1e1e;')
        image_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(image_canvas, 1)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel('Frame:'))
        frame_spin = QSpinBox()
        frame_spin.setRange(1, num_time)
        frame_spin.setValue(1)
        frame_spin.setToolTip(f'Frame 1–{num_time}. Updates after a short pause or press Enter to jump immediately.')
        nav_layout.addWidget(frame_spin)
        nav_layout.addWidget(QLabel(f'of {num_time}'))
        if num_channels > 1:
            nav_layout.addWidget(QLabel('Channel:'))
            channel_spin = QSpinBox()
            channel_spin.setRange(1, num_channels)
            channel_spin.setValue(1)
            channel_spin.setToolTip(f'Channel 1–{num_channels}. Updates after a short pause or press Enter to switch immediately.')
            nav_layout.addWidget(channel_spin)
        else:
            channel_spin = None
        z_spin = None
        if num_z > 1 and nd2_axis_keys is not None and ('Z' in nd2_axis_keys):
            nav_layout.addWidget(QLabel('Z:'))
            z_spin = QSpinBox()
            z_spin.setRange(1, num_z)
            z_spin.setValue(1)
            z_spin.setToolTip(f"Z-slice 1–{num_z}. ND2 axis order: {', '.join(nd2_axis_keys)}. When you change channel, Z jumps to the slice with strongest signal if useful.")
            nav_layout.addWidget(z_spin)
            zb0 = _nd2_best_z_for_channel(raw_frames, nd2_axis_keys, 0, 0, num_z)
            z_spin.setValue(zb0 + 1)
            preview_dialog.current_z = zb0

        def current_z_display() -> int:
            return z_spin.value() if z_spin is not None else 1
        show_drift_check = None
        show_bleach_check = None
        bleach_status_label = None
        _bleach_method_display = {'exponential': 'Exponential', 'linear': 'Linear', 'histogram_matching': 'Histogram matching', 'per_frame_minmax': 'Per-frame min-max'}
        method_display_name = _bleach_method_display.get(bleach_method, bleach_method)
        if num_time > 1:
            show_drift_check = QCheckBox('Show X/Y drift corrected')
            show_drift_check.setToolTip('Preview X/Y drift correction. Computed only when checked, then cached for this preview.')
            show_drift_check.setChecked(False)
            nav_layout.addWidget(show_drift_check)
            show_bleach_check = QCheckBox('Show bleach correction')
            show_bleach_check.setToolTip('Preview the bleach correction effect. Uses the same method and fluorescence channel settings as on the conversion tab (or defaults). If X/Y drift preview is checked, bleach correction is applied after drift correction.')
            show_bleach_check.setChecked(False)
            nav_layout.addWidget(show_bleach_check)
            bleach_status_label = QLabel('View: Raw')
            bleach_status_label.setStyleSheet('color: #9E9E9E; font-size: 10pt;')
            bleach_status_label.setToolTip('Current preview: raw data or bleach-corrected. Changing frame updates the image; correction is applied to all frames.')
            nav_layout.addWidget(bleach_status_label)
        nav_layout.addStretch()
        frame_info = QLabel(f'Timepoint 1 of {num_time} | Channel: 1 | Z-slice: {current_z_display()}')
        frame_info.setStyleSheet('font-weight: bold; color: #2196F3;')
        nav_layout.addWidget(frame_info)
        layout.addLayout(nav_layout)
        lut_group = QGroupBox('')
        lut_group.setFlat(True)
        lut_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        lut_layout = QVBoxLayout()
        lut_layout.setSpacing(8)
        hist_fig = Figure(figsize=LUT_HIST_FIGSIZE_INCHES)
        hist_canvas = FigureCanvas(hist_fig)
        hist_canvas.setMinimumHeight(LUT_HIST_CANVAS_HEIGHT_PX)
        hist_canvas.setMaximumHeight(LUT_HIST_CANVAS_HEIGHT_PX)
        hist_ax = hist_fig.add_subplot(111)
        hist_fig.patch.set_facecolor('#2b2b2b')
        hist_ax.set_facecolor('#2b2b2b')
        hist_ax.tick_params(colors='white', labelsize=8)
        hist_ax.spines['bottom'].set_color('white')
        hist_ax.spines['top'].set_color('white')
        hist_ax.spines['left'].set_color('white')
        hist_ax.spines['right'].set_color('white')
        hist_ax.xaxis.label.set_color('white')
        hist_ax.yaxis.label.set_color('white')
        apply_lut_histogram_figure_margins(hist_fig)
        hist_canvas.setStyleSheet('background-color: #2b2b2b;')
        _fc_lut_tip = 'Drag the red (min) and teal (max) dashed lines to adjust display range.'
        hist_canvas.setToolTip(_fc_lut_tip)
        lut_layout.addWidget(hist_canvas)
        lut_min_slider = QDoubleSpinBox()
        lut_min_slider.setRange(0, 65535)
        lut_min_slider.setDecimals(1)
        lut_min_slider.setSingleStep(10)
        lut_max_slider = QDoubleSpinBox()
        lut_max_slider.setRange(0, 65535)
        lut_max_slider.setDecimals(1)
        lut_max_slider.setSingleStep(10)
        lut_group.setLayout(lut_layout)
        lut_group.setMaximumHeight(200)
        layout.addWidget(lut_group)
        current_frame_data = {'frame': None, 'vmin': None, 'vmax': None, 'frame_min': None, 'frame_max': None, 'histogram': None}
        is_updating_lut = False
        hist_vmin_line = None
        hist_vmax_line = None
        hist_plot = None
        dragging_line = None
        current_vmin = None
        current_vmax = None

        def refresh_image_display():
            """Redraw main image from current_frame_data (frame + vmin/vmax) into ImageCanvas."""
            frame = current_frame_data.get('frame')
            if frame is None:
                return
            if frame.ndim > 2:
                frame = np.asarray(frame).squeeze()
                if frame.ndim != 2:
                    return
            vmin = current_frame_data.get('vmin')
            vmax = current_frame_data.get('vmax')
            if vmin is None or vmax is None:
                return
            (vmin, vmax) = (float(vmin), float(vmax))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmax = max(vmin + 1e-06, float(np.nanmax(frame)) if np.any(np.isfinite(frame)) else vmin + 1)
            vrange = max(vmax - vmin, 1e-06)
            display = ((frame.astype(np.float64) - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
            dummy_mask = np.zeros(display.shape[:2], dtype=np.int32)
            (display_scaled, _) = scale_for_preview(display, dummy_mask)
            pixmap = to_qpixmap(display_scaled)
            image_canvas.set_pixmap(pixmap, (display_scaled.shape[1], display_scaled.shape[0]), reset_view=False)

        def update_histogram(frame, vmin, vmax):
            """Update histogram plot with min/max lines"""
            nonlocal hist_vmin_line, hist_vmax_line, hist_plot
            hist_ax.clear()
            flat_frame = frame.flatten()
            flat_frame = flat_frame[flat_frame > 0]
            if len(flat_frame) == 0:
                flat_frame = frame.flatten()
            frame_min = float(frame.min())
            frame_max = float(frame.max())
            bins = min(256, int(frame_max - frame_min + 1))
            (hist, bin_edges) = np.histogram(flat_frame, bins=bins, range=(frame_min, frame_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_plot = hist_ax.bar(bin_centers, np.log1p(hist), width=bin_edges[1] - bin_edges[0], color='#4A90E2', alpha=0.8, edgecolor='none')
            hist_vmin_line = hist_ax.axvline(vmin, color='#FF6B6B', linewidth=1.5, linestyle='--', label=f'Min: {vmin:.1f}')
            hist_vmax_line = hist_ax.axvline(vmax, color='#4ECDC4', linewidth=1.5, linestyle='--', label=f'Max: {vmax:.1f}')
            hist_ax.axvspan(vmin, vmax, alpha=0.1, color='yellow')
            hist_ax.set_xlabel('Intensity', color='white', fontsize=8)
            hist_ax.set_ylabel('Log(Count+1)', color='white', fontsize=8)
            hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='#5a5a5a', labelcolor='white', fontsize=7, framealpha=0.85, bbox_to_anchor=(0.99, 0.99))
            hist_ax.grid(True, alpha=0.25, color='white', linestyle='-', linewidth=0.5)
            hist_ax.margins(x=0.02, y=0.06)
            apply_lut_histogram_figure_margins(hist_fig)
            hist_canvas.draw()
            current_frame_data['histogram'] = (hist, bin_edges, frame_min, frame_max)

        def on_histogram_press(event):
            """Handle mouse press on histogram to start dragging"""
            if event.inaxes != hist_ax:
                return
            x = event.xdata
            if x is None:
                return
            nonlocal dragging_line, current_vmin, current_vmax
            if current_vmin is not None and current_vmax is not None:
                frame_min = current_frame_data.get('frame_min', 0)
                frame_max = current_frame_data.get('frame_max', 65535)
                range_size = frame_max - frame_min
                if range_size > 0:
                    threshold = max(range_size * 0.05, (frame_max - frame_min) * 0.01)
                else:
                    threshold = 100
                dist_to_min = abs(x - current_vmin)
                dist_to_max = abs(x - current_vmax)
                if dist_to_min < threshold and dist_to_min < dist_to_max:
                    dragging_line = 'min'
                elif dist_to_max < threshold:
                    dragging_line = 'max'
                else:
                    midpoint = (current_vmin + current_vmax) / 2
                    if x < midpoint:
                        dragging_line = 'min'
                    else:
                        dragging_line = 'max'

        def on_histogram_motion(event):
            """Handle mouse motion to drag lines"""
            if event.inaxes != hist_ax:
                return
            x = event.xdata
            if x is None or dragging_line is None:
                return
            nonlocal current_vmin, current_vmax, is_updating_lut
            if is_updating_lut:
                return
            is_updating_lut = True
            try:
                frame_min = current_frame_data.get('frame_min', 0)
                frame_max = current_frame_data.get('frame_max', 65535)
                if dragging_line == 'min':
                    new_min = max(frame_min, min(x, (current_vmax or frame_max) - 1))
                    current_vmin = new_min
                    lut_min_slider.blockSignals(True)
                    lut_min_slider.setValue(new_min)
                    lut_min_slider.blockSignals(False)
                elif dragging_line == 'max':
                    new_max = min(frame_max, max(x, (current_vmin or frame_min) + 1))
                    current_vmax = new_max
                    lut_max_slider.blockSignals(True)
                    lut_max_slider.setValue(new_max)
                    lut_max_slider.blockSignals(False)
                vmin = current_vmin
                vmax = current_vmax
                current_frame_data['vmin'] = vmin
                current_frame_data['vmax'] = vmax
                if current_frame_data['frame'] is not None:
                    refresh_image_display()
                    current_channel = channel_spin.value() if channel_spin is not None else 1
                    _view_suffix = f" | View: {display_frames_ref.get('view_label', 'Raw')}"
                    frame_info.setText(f'Timepoint {preview_dialog.current_frame + 1} of {num_time} | Channel: {current_channel} | Z-slice: {current_z_display()} | LUT: [{vmin:.1f}, {vmax:.1f}]{_view_suffix}')
                    if hist_vmin_line is not None:
                        hist_vmin_line.set_xdata([vmin, vmin])
                    if hist_vmax_line is not None:
                        hist_vmax_line.set_xdata([vmax, vmax])
                    for patch in list(hist_ax.patches):
                        if hasattr(patch, 'get_alpha') and patch.get_alpha() == 0.2:
                            patch.remove()
                    hist_ax.axvspan(vmin, vmax, alpha=0.2, color='yellow')
                    if hist_vmin_line is not None and hist_vmax_line is not None:
                        hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                    hist_canvas.draw()
            finally:
                is_updating_lut = False

        def on_histogram_release(event):
            """Handle mouse release to stop dragging"""
            nonlocal dragging_line
            dragging_line = None
        hist_canvas.mpl_connect('button_press_event', on_histogram_press)
        hist_canvas.mpl_connect('motion_notify_event', on_histogram_motion)
        hist_canvas.mpl_connect('button_release_event', on_histogram_release)

        def update_frame(frame_idx, channel_idx=0, vmin=None, vmax=None, update_sliders=True):
            """Update the displayed frame (frame_idx is 0-indexed internally, channel_idx is 0-indexed)"""
            frames_source = display_frames_ref['data']
            if channel_spin is not None:
                channel_idx = channel_spin.value() - 1
            else:
                channel_idx = 0
            if len(frames_source.shape) == 5:
                if nd2_axis_keys is not None and len(nd2_axis_keys) == len(frames_source.shape):
                    z_i = 0
                    if 'Z' in nd2_axis_keys:
                        z_i = max(0, min(int(getattr(preview_dialog, 'current_z', 0)), num_z - 1))
                    frame = _nd2_slice_yx(frames_source, nd2_axis_keys, frame_idx, channel_idx, z_i)
                elif channel_idx < frames_source.shape[1]:
                    frame = frames_source[frame_idx, channel_idx, 0, :, :]
                else:
                    frame = frames_source[frame_idx, 0, 0, :, :]
            elif len(frames_source.shape) == 4:
                if nd2_axis_keys is not None and len(nd2_axis_keys) == len(frames_source.shape):
                    z_i = 0
                    if 'Z' in nd2_axis_keys:
                        z_i = max(0, min(int(getattr(preview_dialog, 'current_z', 0)), num_z - 1))
                    frame = _nd2_slice_yx(frames_source, nd2_axis_keys, frame_idx, channel_idx, z_i)
                elif num_channels > 1:
                    if channel_idx < frames_source.shape[1]:
                        frame = frames_source[frame_idx, channel_idx, :, :]
                    else:
                        frame = frames_source[frame_idx, 0, :, :]
                else:
                    frame = frames_source[frame_idx, 0, :, :]
            elif len(frames_source.shape) == 3:
                if frames_source.shape[0] == num_time:
                    frame = frames_source[frame_idx, :, :]
                elif num_channels > 1:
                    if channel_idx < frames_source.shape[0]:
                        frame = frames_source[channel_idx, :, :]
                    else:
                        frame = frames_source[0, :, :]
                else:
                    frame = frames_source[frame_idx, :, :]
            elif len(frames_source.shape) == 2:
                frame = frames_source
            else:
                frame = frames_source[frame_idx] if frame_idx < len(frames_source) else frames_source[0]
                if len(frame.shape) > 2 and num_channels > 1:
                    if channel_idx < frame.shape[0]:
                        frame = frame[channel_idx]
                    else:
                        frame = frame[0]
            current_frame_data['frame'] = frame
            frame_min = float(frame.min())
            frame_max = float(frame.max())
            current_frame_data['frame_min'] = frame_min
            current_frame_data['frame_max'] = frame_max
            nonlocal current_vmin, current_vmax
            if vmin is not None and vmax is not None:
                current_vmin = vmin
                current_vmax = vmax
            elif current_vmin is None or current_vmax is None:
                current_vmin = float(np.percentile(frame, _LUT_PREVIEW_AUTO_LO_PCT))
                current_vmax = float(np.percentile(frame, _LUT_PREVIEW_AUTO_HI_PCT))
            vmin = current_vmin
            vmax = current_vmax
            current_frame_data['vmin'] = vmin
            current_frame_data['vmax'] = vmax
            if update_sliders:
                nonlocal is_updating_lut
                is_updating_lut = True
                try:
                    lut_min_slider.blockSignals(True)
                    lut_max_slider.blockSignals(True)
                    current_min_range = lut_min_slider.minimum()
                    current_max_range = lut_min_slider.maximum()
                    new_min_range = min(current_min_range, frame_min) if current_min_range > 0 else frame_min
                    new_max_range = max(current_max_range, frame_max)
                    lut_min_slider.setRange(new_min_range, new_max_range)
                    lut_max_slider.setRange(new_min_range, new_max_range)
                    current_min_val = lut_min_slider.value()
                    current_max_val = lut_max_slider.value()
                    if vmin is not None and vmax is not None:
                        lut_min_slider.setValue(max(new_min_range, min(new_max_range, vmin)))
                        lut_max_slider.setValue(max(new_min_range, min(new_max_range, vmax)))
                    else:
                        if new_min_range <= current_min_val <= new_max_range:
                            pass
                        else:
                            lut_min_slider.setValue(max(new_min_range, min(new_max_range, current_min_val)))
                        if new_min_range <= current_max_val <= new_max_range:
                            pass
                        else:
                            lut_max_slider.setValue(max(new_min_range, min(new_max_range, current_max_val)))
                    lut_min_slider.blockSignals(False)
                    lut_max_slider.blockSignals(False)
                finally:
                    is_updating_lut = False
            update_histogram(frame, vmin, vmax)
            _title = f'Timepoint {frame_idx + 1} of {num_time}  |  {Path(file_path).name}'
            _view_label = display_frames_ref.get('view_label', 'Raw')
            if _view_label != 'Raw':
                _title += f'  |  {_view_label}'
            title_label.setText(_title)
            refresh_image_display()
            current_channel = channel_spin.value() if channel_spin is not None else 1
            view_suffix = f" | View: {display_frames_ref.get('view_label', 'Raw')}"
            frame_info.setText(f'Timepoint {frame_idx + 1} of {num_time} | Channel: {current_channel} | Z-slice: {current_z_display()} | LUT: [{vmin:.1f}, {vmax:.1f}]{view_suffix}')

        def update_lut_from_sliders():
            """Update display when sliders are moved"""
            nonlocal is_updating_lut, current_vmin, current_vmax
            if is_updating_lut:
                return
            is_updating_lut = True
            try:
                vmin = lut_min_slider.value()
                vmax = lut_max_slider.value()
                if vmin >= vmax:
                    vmax = vmin + 1.0
                    lut_max_slider.blockSignals(True)
                    lut_max_slider.setValue(vmax)
                    lut_max_slider.blockSignals(False)
                current_vmin = vmin
                current_vmax = vmax
                current_frame_data['vmin'] = vmin
                current_frame_data['vmax'] = vmax
                if current_frame_data['frame'] is not None:
                    refresh_image_display()
                    current_channel = channel_spin.value() if channel_spin is not None else 1
                    _view_suff = f" | View: {display_frames_ref.get('view_label', 'Raw')}"
                    frame_info.setText(f'Timepoint {preview_dialog.current_frame + 1} of {num_time} | Channel: {current_channel} | Z-slice: {current_z_display()} | LUT: [{vmin:.1f}, {vmax:.1f}]{_view_suff}')
                    if hist_vmin_line is not None and hist_vmax_line is not None:
                        hist_vmin_line.set_xdata([vmin, vmin])
                        hist_vmax_line.set_xdata([vmax, vmax])
                        for patch in list(hist_ax.patches):
                            if hasattr(patch, 'get_alpha') and patch.get_alpha() == 0.2:
                                patch.remove()
                        hist_ax.axvspan(vmin, vmax, alpha=0.2, color='yellow')
                        hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                        hist_canvas.draw()
                    else:
                        update_histogram(current_frame_data['frame'], vmin, vmax)
                else:
                    update_frame(preview_dialog.current_frame, vmin=vmin, vmax=vmax, update_sliders=False)
            finally:
                is_updating_lut = False
        lut_update_timer = QTimer()
        lut_update_timer.setSingleShot(True)
        lut_update_timer.timeout.connect(update_lut_from_sliders)
        preview_runtime_timers.append(lut_update_timer)

        def on_lut_value_changed():
            """Debounced handler for valueChanged signal"""
            if not is_updating_lut:
                lut_update_timer.start(50)
        lut_min_slider.valueChanged.connect(on_lut_value_changed)
        lut_max_slider.valueChanged.connect(on_lut_value_changed)

        def on_lut_editing_finished():
            """Handle when user finishes editing (Enter key or focus loss)"""
            if not is_updating_lut:
                update_lut_from_sliders()
        lut_min_slider.editingFinished.connect(on_lut_editing_finished)
        lut_max_slider.editingFinished.connect(on_lut_editing_finished)
        preview_dialog.current_frame = 0

        def go_to_frame(idx):
            if 0 <= idx < num_time:
                preview_dialog.current_frame = idx
                frame_spin.blockSignals(True)
                frame_spin.setValue(idx + 1)
                frame_spin.blockSignals(False)
                try:
                    vmin = lut_min_slider.value()
                    vmax = lut_max_slider.value()
                    update_frame(idx, vmin=vmin, vmax=vmax, update_sliders=False)
                except Exception as e:
                    show_message_dialog(preview_dialog, 'Preview Error', f'Could not display frame {idx + 1}:\n{str(e)}', kind='warning')
        _DEBOUNCE_MS = 400
        if channel_spin is not None:

            def on_channel_changed():
                """Update frame when channel changes - auto-adjust LUT to new channel"""
                if preview_dialog.current_frame is not None:
                    nonlocal current_vmin, current_vmax
                    current_vmin = None
                    current_vmax = None
                    if z_spin is not None and nd2_axis_keys is not None and ('Z' in nd2_axis_keys):
                        ci = channel_spin.value() - 1
                        zb = _nd2_best_z_for_channel(raw_frames, nd2_axis_keys, preview_dialog.current_frame, ci, num_z)
                        z_spin.blockSignals(True)
                        z_spin.setValue(zb + 1)
                        z_spin.blockSignals(False)
                        preview_dialog.current_z = zb
                    update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)
            channel_debounce = QTimer(preview_dialog)
            channel_debounce.setSingleShot(True)
            channel_debounce.timeout.connect(on_channel_changed)
            preview_runtime_timers.append(channel_debounce)
            preview_dialog._channel_debounce_timer = channel_debounce

            def on_channel_value_changed():
                channel_debounce.stop()
                channel_debounce.start(_DEBOUNCE_MS)
            channel_spin.valueChanged.connect(on_channel_value_changed)
            channel_spin.editingFinished.connect(lambda : (channel_debounce.stop(), on_channel_changed()))
        if z_spin is not None:

            def on_z_changed():
                if preview_dialog.current_frame is not None:
                    preview_dialog.current_z = z_spin.value() - 1
                    vmin = lut_min_slider.value()
                    vmax = lut_max_slider.value()
                    update_frame(preview_dialog.current_frame, vmin=vmin, vmax=vmax, update_sliders=False)
            z_debounce = QTimer(preview_dialog)
            z_debounce.setSingleShot(True)
            z_debounce.timeout.connect(on_z_changed)
            preview_runtime_timers.append(z_debounce)

            def on_z_value_changed():
                z_debounce.stop()
                z_debounce.start(_DEBOUNCE_MS)
            z_spin.valueChanged.connect(on_z_value_changed)
            z_spin.editingFinished.connect(lambda : (z_debounce.stop(), on_z_changed()))
        if show_bleach_check is not None and bleach_status_label is not None:

            def _correction_base_stack() -> np.ndarray:
                data = np.asarray(raw_frames)
                if data.ndim == 5:
                    if nd2_axis_keys is not None and 'Z' in nd2_axis_keys:
                        data = np.take(data, 0, axis=nd2_axis_keys.index('Z')).copy()
                    else:
                        data = data[:, :, 0, :, :].copy()
                elif data.ndim == 4 and nd2_axis_keys is not None and ('Z' in nd2_axis_keys):
                    data = np.take(data, 0, axis=nd2_axis_keys.index('Z')).copy()
                elif data.ndim == 3 and num_channels == 1:
                    data = data[:, np.newaxis, :, :]
                return np.ascontiguousarray(data)

            def compute_display_stack():
                """Compute the requested preview corrections lazily."""
                data = _correction_base_stack()
                labels: list[str] = []
                if show_drift_check is not None and show_drift_check.isChecked():
                    try:
                        from bactomate.scripts.convert_nd2_to_tif import drift_correct_stack
                    except ImportError:
                        return (None, 'Could not import drift_correct_stack for preview.', None, None)
                    cached = getattr(preview_dialog, '_drift_corrected_frames', None)
                    if cached is None:
                        cached = drift_correct_stack(data, method='frame_by_frame')
                        preview_dialog._drift_corrected_frames = cached
                    data = np.asarray(cached)
                    labels.append('X/Y drift corrected')
                kwargs = self._get_bleach_preview_kwargs()
                cur_method = kwargs['bleach_method']
                cur_channels = kwargs.get('bleach_channel_indices')
                if show_bleach_check.isChecked():
                    try:
                        from bactomate.scripts.convert_nd2_to_tif import bleach_correct_stack
                    except ImportError:
                        return (None, 'Could not import bleach_correct_stack for preview.', None, None)
                    data = bleach_correct_stack(np.ascontiguousarray(data), method=cur_method, channel_indices=cur_channels)
                    disp_name = _bleach_method_display.get(cur_method, cur_method)
                    labels.append(f'Bleach corrected ({disp_name})')
                view_label = ' + '.join(labels) if labels else 'Raw'
                return (np.ascontiguousarray(data), None, view_label, (cur_method, cur_channels))

            def apply_preview_corrections(reset_lut: bool=True) -> None:
                nonlocal current_vmin, current_vmax
                result, error, view_label, bleach_state = compute_display_stack()
                if result is None:
                    display_frames_ref['data'] = raw_frames
                    display_frames_ref['view_label'] = 'Raw'
                    bleach_status_label.setText('View: Raw')
                    bleach_status_label.setStyleSheet('color: #9E9E9E; font-size: 10pt;')
                    if show_drift_check is not None:
                        show_drift_check.blockSignals(True)
                        show_drift_check.setChecked(False)
                        show_drift_check.blockSignals(False)
                    show_bleach_check.blockSignals(True)
                    show_bleach_check.setChecked(False)
                    show_bleach_check.blockSignals(False)
                    show_message_dialog(preview_dialog, 'Preview', str(error or 'Could not compute corrected preview.'), kind='warning')
                    update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)
                    return
                display_frames_ref['data'] = result
                display_frames_ref['view_label'] = str(view_label or 'Raw')
                if bleach_state is not None:
                    cur_method, cur_channels = bleach_state
                    preview_dialog._bleach_method_used = cur_method
                    preview_dialog._bleach_channel_indices_used = tuple(sorted(cur_channels or [])) if cur_channels is not None else ()
                bleach_status_label.setText(f"View: {display_frames_ref['view_label']}")
                bleach_status_label.setStyleSheet(
                    'color: #4CAF50; font-size: 10pt;'
                    if display_frames_ref['view_label'] != 'Raw'
                    else 'color: #9E9E9E; font-size: 10pt;'
                )
                if reset_lut:
                    current_vmin = None
                    current_vmax = None
                    update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)
                else:
                    update_frame(preview_dialog.current_frame, vmin=current_vmin, vmax=current_vmax, update_sliders=False)

            def on_preview_correction_toggled(_checked):
                try:
                    apply_preview_corrections(reset_lut=True)
                except Exception as e:
                    display_frames_ref['data'] = raw_frames
                    display_frames_ref['view_label'] = 'Raw'
                    bleach_status_label.setText('View: Raw')
                    bleach_status_label.setStyleSheet('color: #9E9E9E; font-size: 10pt;')
                    show_message_dialog(preview_dialog, 'Preview', f'Could not update corrected preview:\n{str(e)}', kind='warning')
                    update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)

            if show_drift_check is not None:
                show_drift_check.toggled.connect(on_preview_correction_toggled)
            show_bleach_check.toggled.connect(on_preview_correction_toggled)

            def poll_main_window_settings():
                if not show_bleach_check.isChecked():
                    return
                kwargs = self._get_bleach_preview_kwargs()
                cur_method = kwargs['bleach_method']
                cur_channels = kwargs.get('bleach_channel_indices')
                cur_channels_tuple = tuple(sorted(cur_channels or [])) if cur_channels is not None else ()
                used_method = getattr(preview_dialog, '_bleach_method_used', None)
                used_channels = getattr(preview_dialog, '_bleach_channel_indices_used', None)
                if cur_method != used_method or cur_channels_tuple != (used_channels or ()):
                    apply_preview_corrections(reset_lut=False)
            sync_timer = QTimer(preview_dialog)
            sync_timer.timeout.connect(poll_main_window_settings)
            sync_timer.start(500)
            preview_runtime_timers.append(sync_timer)
            preview_dialog._bleach_sync_timer = sync_timer

        def on_frame_changed():
            user_frame = frame_spin.value()
            idx = user_frame - 1
            if 0 <= idx < num_time:
                go_to_frame(idx)
            else:
                frame_spin.blockSignals(True)
                frame_spin.setValue(preview_dialog.current_frame + 1)
                frame_spin.blockSignals(False)
        frame_debounce = QTimer(preview_dialog)
        frame_debounce.setSingleShot(True)
        frame_debounce.timeout.connect(on_frame_changed)
        preview_runtime_timers.append(frame_debounce)
        preview_dialog._frame_debounce_timer = frame_debounce

        def on_frame_value_changed():
            frame_debounce.stop()
            frame_debounce.start(_DEBOUNCE_MS)
        frame_spin.valueChanged.connect(on_frame_value_changed)
        frame_spin.editingFinished.connect(lambda : (frame_debounce.stop(), QTimer.singleShot(0, on_frame_changed)))
        close_btn = QPushButton('Close')
        close_btn.setMinimumWidth(100)
        close_btn.setAutoDefault(False)
        close_btn.setDefault(False)
        close_btn.clicked.connect(preview_dialog.close)
        layout.addWidget(close_btn)
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, preview_dialog)
        close_shortcut.activated.connect(preview_dialog.close)
        minimize_shortcut = QShortcut(QKeySequence('Ctrl+M'), preview_dialog)
        minimize_shortcut.activated.connect(preview_dialog.showMinimized)
        preview_dialog.current_frame = 0
        update_frame(0)

        def _cleanup_preview_runtime(*_args) -> None:
            nonlocal preview_runtime_cleaned
            if preview_runtime_cleaned:
                return
            preview_runtime_cleaned = True
            for t in preview_runtime_timers:
                try:
                    t.stop()
                except Exception:
                    pass
        try:
            preview_dialog.finished.connect(_cleanup_preview_runtime)
        except Exception:
            pass
        try:
            preview_dialog.destroyed.connect(_cleanup_preview_runtime)
        except Exception:
            pass
        preview_dialog.setLayout(layout)
        preview_dialog.show()

    def _preview_nd2(self, file_path: Optional[Path]=None):
        """Preview ND2 file in a dialog with frame navigation"""
        try:
            import numpy as np
            import nd2
            file_path = str(file_path) if file_path is not None else self.file_path.text()
            with nd2.ND2File(file_path) as nd2_reader:
                axis_keys = tuple(nd2_reader.sizes.keys())
                shp = nd2_reader.shape
                if len(axis_keys) != len(shp):
                    num_time = nd2_reader.sizes.get('T', 1)
                    num_channels = nd2_reader.sizes.get('C', 1)
                    num_z = nd2_reader.sizes.get('Z', 1)
                    axis_keys = None
                else:
                    num_time = int(shp[axis_keys.index('T')]) if 'T' in axis_keys else 1
                    num_channels = int(shp[axis_keys.index('C')]) if 'C' in axis_keys else 1
                    num_z = int(shp[axis_keys.index('Z')]) if 'Z' in axis_keys else 1
                all_frames = np.ascontiguousarray(nd2_reader.asarray()).copy()
            self._show_file_conversion_preview(all_frames=all_frames, num_time=num_time, num_channels=num_channels, num_z=num_z, file_path=file_path, nd2_axis_keys=axis_keys)
        except ImportError as e:
            error_msg = str(e)
            if 'torch.testing' in error_msg:
                show_message_dialog(self, 'Preview Unavailable', 'Preview requires torch.testing module.\n\nThis is a known issue in some environments.\nTry running from source: python main.py', kind='warning')
            else:
                show_message_dialog(self, 'Preview Unavailable', f'Preview requires matplotlib:\n\npip install matplotlib', kind='warning')
        except Exception as e:
            error_msg = str(e)
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                show_message_dialog(self, 'Preview Error', 'PyTorch initialization error.\n\nThis is a known issue with PyTorch in some environments.\nTry restarting the application or running from source.', kind='warning')
            else:
                show_message_dialog(self, 'Preview Error', f'Could not preview file:\n{error_msg}', kind='error')
            import traceback
            traceback.print_exc()

    def _preview_lif(self, file_path: Optional[Path]=None):
        """Preview LIF file in a dialog with frame navigation. Uses BioIO when available, then aicsimageio."""
        try:
            import numpy as np
            file_path = str(file_path) if file_path is not None else self.file_path.text()
            max_f = self.max_frames.value() if hasattr(self, 'max_frames') else 500
            try:
                data = bioio_load_data(Path(file_path), max_frames=max_f, use_first_z_only=True)
                if data is not None and data.size > 0:
                    (num_time, num_channels) = (data.shape[0], data.shape[1])
                    num_z = 1
                    all_frames = np.ascontiguousarray(data).copy()
                    self._show_file_conversion_preview(all_frames=all_frames, num_time=num_time, num_channels=num_channels, num_z=num_z, file_path=file_path, nd2_axis_keys=None)
                    return
            except Exception:
                pass
            aics_preview_ok = False
            try:
                from aicsimageio import AICSImage
                lif_reader = AICSImage(file_path)
                try:
                    dims = lif_reader.dims
                    dims_str = dims.order if hasattr(dims, 'order') else dims if isinstance(dims, str) else str(dims)
                    if hasattr(lif_reader, 'get_xarray_data'):
                        xdata = lif_reader.get_xarray_data()
                        all_frames = np.array(xdata)
                        num_time = int(xdata.sizes.get('T', 1))
                        num_channels = int(xdata.sizes.get('C', 1))
                        num_z = int(xdata.sizes.get('Z', 1))
                    else:
                        all_frames = np.array(lif_reader.data)
                        shape = all_frames.shape
                        num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                        num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                        num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                    if len(all_frames.shape) > 3:
                        indices = [slice(None)] * len(all_frames.shape)
                        if 'C' in dims_str:
                            indices[dims_str.index('C')] = 0
                        if 'Z' in dims_str:
                            indices[dims_str.index('Z')] = 0
                        all_frames = all_frames[tuple(indices)]
                    if len(all_frames.shape) == 2:
                        all_frames = all_frames[np.newaxis, ...]
                    elif len(all_frames.shape) == 3 and all_frames.shape[0] > 1000:
                        all_frames = all_frames.transpose(2, 0, 1)
                    all_frames = np.ascontiguousarray(all_frames).copy()
                    self._show_file_conversion_preview(all_frames=all_frames, num_time=num_time, num_channels=num_channels, num_z=num_z, file_path=file_path, nd2_axis_keys=None)
                    aics_preview_ok = True
                finally:
                    if hasattr(lif_reader, 'close'):
                        try:
                            lif_reader.close()
                        except Exception:
                            pass
            except Exception:
                pass
            if aics_preview_ok:
                return
            try:
                from readlif.reader import LifFile
                lif_file = LifFile(file_path)
                if getattr(lif_file, 'num_images', 0) <= 0:
                    raise ValueError('No image series in LIF file')
                log_cb = lambda msg: self.parent_window.log_output(msg)
                best_idx = self._readlif_best_image_index(lif_file, log_cb=log_cb)
                image = lif_file.get_image(best_idx)
                (num_time, num_z, num_channels) = self._readlif_dims(image, log_cb=log_cb)
                n_load = min(max_f, num_time)
                n_ch = max(1, num_channels)
                frames_list = []
                ref_shape = None
                for t in range(n_load):
                    row = []
                    for c in range(n_ch):
                        try:
                            pil_img = image.get_frame(z=0, t=t, c=c)
                            arr = np.array(pil_img)
                            if ref_shape is None:
                                ref_shape = arr.shape
                            row.append(arr)
                        except Exception:
                            row.append(np.zeros(ref_shape or (1024, 1024), dtype=np.uint16))
                    if len(row) == 1:
                        frames_list.append(row[0])
                    else:
                        frames_list.append(np.stack(row, axis=0))
                if n_ch > 1:
                    all_frames = np.ascontiguousarray(np.stack(frames_list, axis=0))
                else:
                    all_frames = np.ascontiguousarray(np.stack(frames_list, axis=0))
                preview_dialog = QDialog(self)
                preview_dialog.setWindowTitle('File Preview (readlif)')
                preview_dialog.setMinimumSize(900, 700)
                preview_dialog.setModal(False)
                layout = QVBoxLayout()
                self._show_file_conversion_preview(all_frames=all_frames, num_time=n_load, num_channels=n_ch, num_z=num_z, file_path=file_path, nd2_axis_keys=None)
                return
            except Exception as e:
                import traceback
                self.parent_window.log_output(f'Readlif preview failed: {type(e).__name__}: {e}\n')
                self.parent_window.log_output(traceback.format_exc())
            show_message_dialog(self, 'Error Previewing LIF', 'Could not preview LIF with BioIO, AICSImageIO, or readlif.', kind='error')
        except ImportError:
            show_message_dialog(self, 'Library Missing', "LIF preview requires 'bioio' and 'bioio-lif', or 'aicsimageio', or 'readlif'.\n\nInstall with: pip install aicsimageio readlif", kind='warning')
        except Exception as e:
            show_message_dialog(self, 'Error Previewing LIF', f'Could not preview LIF file:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')

    def _preview_czi(self, file_path: Optional[Path]=None):
        """Preview CZI file in a dialog with frame navigation. Uses BioIO when available, then czifile."""
        try:
            import numpy as np
            file_path = str(file_path) if file_path is not None else self.file_path.text()
            if not file_path:
                show_message_dialog(self, 'No File', 'Please select a CZI file or folder first.', kind='warning')
                return
            path = Path(file_path)
            if path.is_dir():
                czi_in_dir = sorted(path.glob('*.czi')) + sorted(path.glob('*.CZI'))
                if not czi_in_dir:
                    show_message_dialog(self, 'CZI Preview', 'No CZI files found in the selected folder.', kind='warning')
                    return
                path = czi_in_dir[0]
                file_path = str(path)
            max_f = self.max_frames.value() if hasattr(self, 'max_frames') else 500
            max_z_proj = self.max_z_projection.isChecked() if hasattr(self, 'max_z_projection') else False
            data = bioio_load_data(path, max_frames=max_f, use_first_z_only=not max_z_proj, max_z_projection=max_z_proj)
            if data is not None and data.size > 0:
                (num_time, num_channels) = (data.shape[0], data.shape[1])
                num_z = 1
                all_frames = np.ascontiguousarray(data).copy()
                self._show_file_conversion_preview(all_frames=all_frames, num_time=num_time, num_channels=num_channels, num_z=num_z, file_path=file_path, nd2_axis_keys=None)
                return
            import czifile
        except ImportError:
            show_message_dialog(self, 'Library Missing', "CZI preview requires 'bioio' and 'bioio-czi', or 'czifile'.\n\nInstall with: pip install bioio bioio-czi\nOr: pip install czifile", kind='warning')
            return
        try:
            all_frames = None
            num_time = num_channels = num_z = 1
            with czifile.CziFile(file_path) as czi_file:
                axes = getattr(czi_file, 'axes', None)
                shape = getattr(czi_file, 'shape', None)
                if axes and shape and (len(axes) == len(shape)):
                    (num_time, num_channels, num_z) = self._czi_dims_from_axes(axes, shape)
                arr = czi_file.asarray() if hasattr(czi_file, 'asarray') else None
                if arr is not None and arr.size > 0:
                    arr = np.ascontiguousarray(arr).copy()
            if arr is not None and arr.size > 0:
                if axes and len(axes) == len(arr.shape):
                    (c_preview, z_preview) = self._czi_find_first_nonempty_plane(arr, axes)
                    all_frames = self._czi_slice_to_t_y_x(arr, axes, z_index=z_preview, c_index=c_preview)
                else:
                    sh = arr.shape
                    if len(sh) >= 5:
                        all_frames = arr[:, 0, 0, :, :]
                    elif len(sh) == 4:
                        all_frames = arr[:, 0, :, :]
                    elif len(sh) == 3:
                        all_frames = arr
                    else:
                        all_frames = arr
                    if axes and shape and (len(axes) == len(shape)):
                        (num_time, num_channels, num_z) = self._czi_dims_from_axes(axes, shape)
            if all_frames is not None and getattr(all_frames, 'size', 0) == 0:
                all_frames = None
            if all_frames is None or all_frames.size == 0:
                show_message_dialog(self, 'CZI Preview', 'Could not load CZI image data.\n\nInstall: pip install czifile', kind='warning')
                return
            if len(all_frames.shape) == 2:
                all_frames = all_frames[np.newaxis, ...]
            all_frames = np.ascontiguousarray(all_frames).copy()
            self._show_file_conversion_preview(all_frames=all_frames, num_time=num_time, num_channels=num_channels, num_z=num_z, file_path=file_path, nd2_axis_keys=None)
        except Exception as e:
            show_message_dialog(self, 'Error Previewing CZI', f'Could not preview CZI file:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')

    def _resolve_tiff_path_for_operation(self, purpose: str='convert') -> Optional[Path]:
        """Resolve a TIFF file path when the line edit may be a folder (folder scan / embedded preview)."""
        line = Path((self.file_path.text() or '').strip())
        if line.is_file() and line.suffix.lower() in ('.tif', '.tiff'):
            return line
        if self.file_info:
            meta_p = Path(str(self.file_info.get('path') or ''))
            if meta_p.is_file() and meta_p.suffix.lower() in ('.tif', '.tiff'):
                return meta_p
        if line.is_dir():
            tiffs = sorted([p for p in list_convertible_microscopy_files(line, max_files=2500) if p.suffix.lower() in ('.tif', '.tiff')], key=lambda p: p.name.lower())
            if not tiffs:
                return None
            if len(tiffs) == 1:
                return tiffs[0]
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle('Select TIFF file')
            file_dialog.setMinimumSize(480, 360)
            layout = QVBoxLayout()
            layout.addWidget(QLabel(f'Select which TIFF to {purpose}:'))
            file_list = QListWidget()
            for p in tiffs:
                file_list.addItem(p.name)
            file_list.setCurrentRow(0)
            file_list.itemDoubleClicked.connect(file_dialog.accept)
            layout.addWidget(file_list)
            btn_row = QHBoxLayout()
            btn_row.addStretch()
            cancel_btn = QPushButton('Cancel')
            cancel_btn.clicked.connect(file_dialog.reject)
            ok_btn = QPushButton('OK')
            ok_btn.clicked.connect(file_dialog.accept)
            btn_row.addWidget(cancel_btn)
            btn_row.addWidget(ok_btn)
            layout.addLayout(btn_row)
            file_dialog.setLayout(layout)
            if file_dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            idx = file_list.currentRow()
            if idx < 0 or idx >= len(tiffs):
                return None
            return tiffs[idx]
        return None

    def _show_conversion_progress_window(self, *, total_files: int, current_file: Optional[Any]=None, action: str='Converting to TIFF') -> None:
        dlg = getattr(self, '_conversion_progress_dialog', None)
        if dlg is None:
            dlg = ConversionProgressDialog(self)
            self._conversion_progress_dialog = dlg
            dlg.destroyed.connect(lambda *_: setattr(self, '_conversion_progress_dialog', None))
            dlg.show()
        detail = f'File 1/{max(1, int(total_files))}'
        if current_file is not None:
            detail += f': {Path(current_file).name}'
        dlg.set_progress(0, action, detail)

    def _set_conversion_progress_window(self, percent: int, step: str, detail: str='') -> None:
        dlg = getattr(self, '_conversion_progress_dialog', None)
        if dlg is None:
            return
        try:
            dlg.set_progress(percent, step, detail)
        except RuntimeError:
            self._conversion_progress_dialog = None

    def _close_conversion_progress_window(self, *, complete: bool=False) -> None:
        dlg = getattr(self, '_conversion_progress_dialog', None)
        self._conversion_progress_dialog = None
        if dlg is None:
            return
        try:
            if complete:
                dlg.finish_and_close()
            else:
                dlg.close()
        except RuntimeError:
            pass

    def _advance_to_segmentation_after_conversion(self) -> None:
        """Jump to Segmentation and use the converted Tif_files folder as input."""
        try:
            out_dir = getattr(self, '_last_conversion_output_dir', None)
            if out_dir is None:
                raw = (self.file_path.text() or '').strip()
                if not raw:
                    return
                out_dir = tif_files_output_dir(raw)
            out_path = Path(out_dir).resolve()
            if not out_path.exists():
                return
            pw = getattr(self, 'parent_window', None)
            seg = getattr(pw, 'segmentation_tab', None) if pw is not None else None
            if seg is None:
                return
            if hasattr(seg, '_sync_segmentation_input_path'):
                seg._sync_segmentation_input_path(str(out_path))
            try:
                seg.segmentation_section.set_expanded(True)
                seg.filtering_section.set_expanded(False)
                seg.analyze_section.set_expanded(False)
            except Exception:
                pass
            if hasattr(seg, 'refresh_segmentation_candidate_list'):
                seg.refresh_segmentation_candidate_list()
            if pw is not None and hasattr(pw, '_go_module_from_dashboard'):
                pw._go_module_from_dashboard('segmentation')
            if pw is not None and hasattr(pw, 'log_output'):
                pw.log_output(f'Segmentation input set to converted TIFF folder: {out_path}\n')
        except Exception:
            pw = getattr(self, 'parent_window', None)
            if pw is not None and hasattr(pw, 'log_output'):
                pw.log_output('Could not jump to Segmentation after conversion.\n')

    def _preview_tiff(self, file_path: Optional[Path]=None):
        """Preview TIFF file in a dialog with frame navigation. Uses BioIO when available, then tifffile."""
        try:
            import tifffile
            if file_path is not None:
                tiff_path = Path(file_path)
            else:
                tiff_path = self._resolve_tiff_path_for_operation(purpose='preview')
            if tiff_path is None or not tiff_path.is_file():
                show_message_dialog(self, 'No File', 'Please select a TIFF file.', kind='warning')
                return
            file_path_str = str(tiff_path)
            max_f = self.max_frames.value() if hasattr(self, 'max_frames') else 500
            info = get_tiff_file_info(tiff_path) or {}
            num_time = int(info.get('num_time') or self.file_info.get('num_time', 1) if self.file_info else 1)
            num_channels = int(info.get('num_channels') or (self.file_info.get('num_channels', 1) if self.file_info else 1))
            num_z = int(info.get('num_z') or (self.file_info.get('num_z', 1) if self.file_info else 1))
            series_axes = str(info.get('series_axes') or '')
            data = bioio_load_data(tiff_path, max_frames=max_f, use_first_z_only=True)
            if data is not None and data.size > 0:
                stack = normalize_tiff_array_to_tcyx(data, num_time=num_time, num_channels=num_channels, num_z=num_z, series_axes=series_axes)
            else:
                raw = safe_imread(file_path_str)
                stack = normalize_tiff_array_to_tcyx(raw, num_time=num_time, num_channels=num_channels, num_z=num_z, series_axes=series_axes)
            num_time = int(stack.shape[0])
            num_channels = int(stack.shape[1])
            all_frames = np.ascontiguousarray(stack).copy()
            self._show_file_conversion_preview(all_frames=all_frames, num_time=num_time, num_channels=num_channels, num_z=1, file_path=file_path_str, nd2_axis_keys=None)
        except Exception as e:
            show_message_dialog(self, 'Error Previewing TIFF', f'Could not preview TIFF file:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')

    def convert_nd2(self):
        """Convert microscopy file(s) to TIFF"""
        if not self.file_path.text():
            show_message_dialog(self, 'No File Selected', 'Please select a microscopy file or folder first.', kind='warning')
            return
        if not self.file_info:
            show_message_dialog(self, 'File Not Loaded', 'Please load file information first.', kind='warning')
            return
        if not self._validate_crop_before_conversion():
            return
        file_type = self.file_info.get('file_type', 'nd2')
        if file_type == 'nd2':
            self._convert_nd2()
        elif file_type == 'lif':
            self._convert_lif()
        elif file_type == 'czi':
            self._convert_czi()
        elif file_type == 'tiff':
            num_channels = self.file_info.get('num_channels', 1)
            num_time = self.file_info.get('num_time', 1)
            if self._tiff_conversion_has_work(num_time=num_time, num_channels=num_channels):
                self._convert_tiff()
            else:
                self._show_tiff_noop_dialog()
        else:
            show_message_dialog(self, 'Unsupported Format', f'Conversion not yet implemented for {file_type} files.', kind='warning')

    def _convert_nd2(self):
        """Convert file(s) to TIFF"""
        file_path = Path(self.file_path.text())
        if file_path.is_dir():
            all_micro = list_convertible_microscopy_files(file_path, max_files=2500)
            supported_files = sorted([p for p in all_micro if p.suffix.lower() == '.nd2'], key=lambda p: p.name.lower())
            if not supported_files:
                show_message_dialog(self, 'No ND2 Files', 'No ND2 files found in the selected folder.', kind='warning')
                return
            selected_files = show_file_selection_dialog(self, supported_files, title='Select Files to Convert', message=f'Select which files to convert to TIFF (all {len(supported_files)} selected by default):')
            if selected_files is None:
                return
            if len(selected_files) == 0:
                show_message_dialog(self, 'No Files Selected', 'Please select at least one file to convert.', kind='warning')
                return
            files_to_convert = selected_files
        else:
            files_to_convert = [file_path]
        nd2_paths = [Path(fp) for fp in files_to_convert if Path(fp).suffix.lower() == '.nd2']
        z_stack_infos: list[tuple[Path, int]] = []
        for fp in nd2_paths:
            try:
                inf = self._get_nd2_info(str(fp))
                if inf and int(inf.get('num_z', 1) or 1) > 1:
                    z_stack_infos.append((fp, int(inf.get('num_z', 1))))
            except Exception:
                continue
        if not z_stack_infos and self.file_info and (int(self.file_info.get('num_z', 1) or 1) > 1):
            p0 = self.file_info.get('path')
            if p0 and str(p0).lower().endswith('.nd2'):
                z_stack_infos.append((Path(str(p0)), int(self.file_info['num_z'])))
        if z_stack_infos:
            if not self.max_z_projection.isChecked():
                max_z = max((z for (_, z) in z_stack_infos))
                n_files = len(z_stack_infos)
                default_z = (max_z + 1) // 2
                dlg = QDialog(self)
                dlg.setWindowTitle('Z-Stack Detected')
                v = QVBoxLayout(dlg)
                if n_files == 1:
                    z_one = z_stack_infos[0][1]
                    intro = f'This ND2 has Z={z_one} slices.\n\nChoose how to handle Z during conversion:'
                else:
                    intro = f'{n_files} ND2 file(s) in this batch have Z-stacks (up to {max_z} slices in one file).\n\nThe same choice applies to all of them (single slice index is clamped per file if needed).\n\nChoose how to handle Z during conversion:'
                v.addWidget(QLabel(intro))
                rb_mip = QRadioButton('Enable max intensity projection (combine all Z slices)')
                rb_slice = QRadioButton('Use a single Z-slice')
                rb_slice.setChecked(True)
                v.addWidget(rb_mip)
                v.addWidget(rb_slice)
                slice_row = QHBoxLayout()
                slice_row.addWidget(QLabel('Z-slice:'))
                z_spin = QSpinBox()
                z_spin.setRange(1, max_z)
                z_spin.setValue(min(default_z, max_z))
                z_spin.setToolTip(f'Pick which slice to use (1–{max_z}); applied per file, clamped if that file has fewer slices.')
                slice_row.addWidget(z_spin)
                slice_row.addStretch(1)
                v.addLayout(slice_row)

                def _sync_enabled() -> None:
                    z_spin.setEnabled(rb_slice.isChecked())
                rb_mip.toggled.connect(_sync_enabled)
                rb_slice.toggled.connect(_sync_enabled)
                _sync_enabled()
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                style_dialog_button_box(buttons)
                buttons.accepted.connect(dlg.accept)
                buttons.rejected.connect(dlg.reject)
                v.addWidget(buttons)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                if rb_mip.isChecked():
                    self.max_z_projection.setChecked(True)
                    self._selected_z_slice_for_conversion = None
                else:
                    self._selected_z_slice_for_conversion = int(z_spin.value())
            else:
                self._selected_z_slice_for_conversion = None
        else:
            self._selected_z_slice_for_conversion = None
        from bactomate.gui.constants import CONVERT_ND2_SCRIPT
        if not CONVERT_ND2_SCRIPT.exists():
            show_message_dialog(self, 'Script Not Found', f'Conversion script not found.\n\nExpected: {CONVERT_ND2_SCRIPT}\n\nInstall BactoMate correctly or run from the project root.', kind='warning')
            return
        convert_script = CONVERT_ND2_SCRIPT
        self._conversion_queue = files_to_convert
        self._conversion_current_index = 0
        first_path = self._conversion_queue[0]
        cmd_args = [str(first_path)]
        if self.frame_extract_mode.currentIndex() == 1:
            cmd_args.extend(['--single-frame', str(self.single_frame_spin.value())])
        else:
            cmd_args.extend(['--frames', str(self.max_frames.value())])
        if self.separate_channels.isChecked():
            cmd_args.append('--separate-channels')
        else:
            cmd_args.append('--combine-channels')
        if self.max_z_projection.isChecked():
            cmd_args.append('--max-z-projection')
        else:
            z_sel = getattr(self, '_selected_z_slice_for_conversion', None)
            if z_sel is not None:
                cmd_args.extend(['--z-slice', str(int(z_sel))])
        if self.drift_correct.isChecked():
            cmd_args.append('--drift-correct')
        if self.bleach_correct.isChecked():
            cmd_args.append('--bleach-correct')
            bleach_methods = ['exponential', 'linear', 'histogram_matching', 'per_frame_minmax']
            cmd_args.append('--bleach-method')
            cmd_args.append(bleach_methods[self.bleach_method_combo.currentIndex()])
            if getattr(self, 'bleach_channel_checks', None):
                indices = [i for (i, cb) in enumerate(self.bleach_channel_checks) if cb.isChecked()]
                if indices and len(indices) < len(self.bleach_channel_checks):
                    cmd_args.append('--bleach-channels')
                    cmd_args.extend((str(i) for i in indices))
        if self.separate_channels.isChecked() and getattr(self, 'fi_channel_checks', None):
            fi_indices = [i for (i, cb) in enumerate(self.fi_channel_checks) if cb.isChecked()]
            if fi_indices:
                cmd_args.append('--fi-channels')
                cmd_args.extend((str(i) for i in fi_indices))
        if self.separate_channels.isChecked() and getattr(self, 'foci_channel_checks', None):
            foci_indices = [i for (i, cb) in enumerate(self.foci_channel_checks) if cb.isChecked()]
            if foci_indices:
                cmd_args.append('--foci-channels')
                cmd_args.extend((str(i) for i in foci_indices))
        self._append_output_metadata_cli_args(cmd_args)
        self._append_crop_cli_args(cmd_args)
        self.convert_btn.setEnabled(False)
        self.convert_btn.setText('Converting...')
        self.parent_window.log_output(f'Starting ND2 conversion: {first_path.name}' + (f' (file 1/{len(files_to_convert)})' if len(files_to_convert) > 1 else '') + '\n')
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        self.conversion_progress = {'total_files': max(1, len(files_to_convert)), 'current_file': 0, 'file_progress': 0}
        self._last_conversion_output_dir = tif_files_output_dir(file_path)
        self._show_conversion_progress_window(total_files=len(files_to_convert), current_file=first_path, action='Converting ND2 to TIFF')
        self.worker = WorkerThread(str(convert_script), cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self._update_conversion_progress)
        self.worker.start()

    def _update_conversion_progress(self, message):
        """Parse conversion output and update progress bar"""
        self.parent_window.log_output(message + '\n')
        msg_lower = message.lower()
        import re
        if 'found' in msg_lower:
            match = re.search('found (\\d+)', msg_lower, re.IGNORECASE)
            if match:
                self.conversion_progress['total_files'] = int(match.group(1))
                self.conversion_progress['current_file'] = 0
                self.conversion_progress['file_progress'] = 0
        if 'processing:' in msg_lower or ('processing' in msg_lower and '.nd2' in msg_lower):
            if self.conversion_progress['file_progress'] >= 90:
                if self.conversion_progress['total_files'] > 0:
                    self.conversion_progress['current_file'] = min(self.conversion_progress['current_file'] + 1, self.conversion_progress['total_files'] - 1)
            self.conversion_progress['file_progress'] = 5
        if 'reading frames' in msg_lower or 'reading' in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 20)
        if 'extracting frames' in msg_lower or 'extracting' in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 25)
        if 'dimensions:' in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 15)
        if 'applying drift correction' in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 40)
        if 'drift correction complete' in msg_lower or '✓ drift' in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 45)
        if 'applying bleach correction' in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 50)
        if 'bleach correction complete' in msg_lower or '✓ bleach' in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 55)
        if 'saving' in msg_lower:
            channel_match = re.search('channel (\\d+)/(\\d+)', msg_lower)
            if channel_match:
                current_ch = int(channel_match.group(1))
                total_ch = int(channel_match.group(2))
                save_progress = 60 + int(35 * (current_ch / total_ch))
                self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], save_progress)
            elif self.conversion_progress['file_progress'] < 60:
                self.conversion_progress['file_progress'] = 60
        if 'complete' in msg_lower or '✓' in message or 'successfully' in msg_lower:
            self.conversion_progress['file_progress'] = 100
        if len(message.strip()) > 5:
            if self.conversion_progress['file_progress'] < 95:
                self.conversion_progress['file_progress'] = min(95, self.conversion_progress['file_progress'] + 1)
        if self.conversion_progress['total_files'] > 0:
            completed_files = self.conversion_progress['current_file']
            current_file_progress = self.conversion_progress['file_progress'] / 100.0
            overall_progress = int((completed_files + current_file_progress) / self.conversion_progress['total_files'] * 100)
        else:
            overall_progress = self.conversion_progress['file_progress']
        overall_progress = max(0, min(100, overall_progress))
        if not self.parent_window.progress_bar.isVisible():
            self.parent_window.progress_bar.setVisible(True)
        if self.parent_window.progress_bar.maximum() != 100:
            self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(overall_progress)
        queue = getattr(self, '_conversion_queue', None) or []
        current_idx = int(getattr(self, '_conversion_current_index', 0) or 0)
        current_name = ''
        if queue and 0 <= current_idx < len(queue):
            current_name = Path(queue[current_idx]).name
        file_detail = f"File {current_idx + 1}/{max(1, int(self.conversion_progress.get('total_files', 1)))}"
        if current_name:
            file_detail += f': {current_name}'
        message_detail = str(message or '').strip()
        if message_detail:
            file_detail += f'\n{message_detail}'
        self._set_conversion_progress_window(overall_progress, 'Converting ND2 to TIFF', file_detail)
        QApplication.processEvents()

    def on_finished(self, success, message):
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        self.parent_window.log_output(message)
        if not success:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText('Convert to TIFF')
            self.parent_window.progress_bar.setVisible(False)
            self._close_conversion_progress_window()
            show_message_dialog(self, 'Error', f'Conversion failed:\n{message}', kind='error')
            self.workflow_step_finished.emit(False)
            return
        queue = getattr(self, '_conversion_queue', None)
        idx = getattr(self, '_conversion_current_index', 0)
        if queue and idx + 1 < len(queue):
            self._conversion_current_index = idx + 1
            next_path = queue[self._conversion_current_index]
            from bactomate.gui.constants import CONVERT_ND2_SCRIPT
            convert_script = CONVERT_ND2_SCRIPT
            cmd_args = [str(next_path)]
            if self.frame_extract_mode.currentIndex() == 1:
                cmd_args.extend(['--single-frame', str(self.single_frame_spin.value())])
            else:
                cmd_args.extend(['--frames', str(self.max_frames.value())])
            if self.separate_channels.isChecked():
                cmd_args.append('--separate-channels')
            else:
                cmd_args.append('--combine-channels')
            if self.max_z_projection.isChecked():
                cmd_args.append('--max-z-projection')
            else:
                z_sel = getattr(self, '_selected_z_slice_for_conversion', None)
                if z_sel is not None:
                    cmd_args.extend(['--z-slice', str(int(z_sel))])
            if self.drift_correct.isChecked():
                cmd_args.append('--drift-correct')
            if self.bleach_correct.isChecked():
                cmd_args.append('--bleach-correct')
                bleach_methods = ['exponential', 'linear', 'histogram_matching', 'per_frame_minmax']
                cmd_args.append('--bleach-method')
                cmd_args.append(bleach_methods[self.bleach_method_combo.currentIndex()])
                if getattr(self, 'bleach_channel_checks', None):
                    indices = [i for (i, cb) in enumerate(self.bleach_channel_checks) if cb.isChecked()]
                    if indices and len(indices) < len(self.bleach_channel_checks):
                        cmd_args.append('--bleach-channels')
                        cmd_args.extend((str(i) for i in indices))
            if self.separate_channels.isChecked() and getattr(self, 'fi_channel_checks', None):
                fi_indices = [i for (i, cb) in enumerate(self.fi_channel_checks) if cb.isChecked()]
                if fi_indices:
                    cmd_args.append('--fi-channels')
                    cmd_args.extend((str(i) for i in fi_indices))
            if self.separate_channels.isChecked() and getattr(self, 'foci_channel_checks', None):
                foci_indices = [i for (i, cb) in enumerate(self.foci_channel_checks) if cb.isChecked()]
                if foci_indices:
                    cmd_args.append('--foci-channels')
                    cmd_args.extend((str(i) for i in foci_indices))
            self._append_output_metadata_cli_args(cmd_args)
            self._append_crop_cli_args(cmd_args)
            self.parent_window.log_output(f'Starting ND2 conversion: {next_path.name} (file {self._conversion_current_index + 1}/{len(queue)})\n')
            self.conversion_progress['current_file'] = self._conversion_current_index
            self.conversion_progress['file_progress'] = 0
            self._set_conversion_progress_window(int(self._conversion_current_index / max(1, len(queue)) * 100), 'Converting ND2 to TIFF', f'File {self._conversion_current_index + 1}/{len(queue)}: {Path(next_path).name}')
            self.worker = WorkerThread(str(convert_script), cmd_args, parent=self)
            self.worker.finished.connect(self.on_finished)
            self.worker.progress.connect(self._update_conversion_progress)
            self.worker.start()
            return
        self.convert_btn.setEnabled(True)
        self.convert_btn.setText('Convert to TIFF')
        self.parent_window.progress_bar.setVisible(False)
        self._close_conversion_progress_window(complete=True)
        n = len(queue) if queue else 1
        self.parent_window.log_output(f'✓ ND2 conversion completed successfully ({n} file(s))!\n')
        if queue:
            self._conversion_queue = None
            self._conversion_current_index = 0
        self._performed_this_session = True
        show_task_complete_dialog(self, title='Conversion Complete', message=f'Successfully converted {n} file(s) to TIFF.', output_dir=getattr(self, '_last_conversion_output_dir', None), on_ok=self._advance_to_segmentation_after_conversion)
        self.workflow_step_finished.emit(True)

    def _convert_lif(self):
        """Convert LIF file(s) to TIFF"""
        try:
            import numpy as np
            import tifffile
            from aicsimageio import AICSImage
            file_path = Path(self.file_path.text())
            if file_path.is_dir():
                all_lif = sorted(list(file_path.glob('*.lif')) + list(file_path.glob('*.LIF')))
                if not all_lif:
                    show_message_dialog(self, 'No LIF Files', 'No LIF files found in the selected folder.', kind='warning')
                    return
                selected = show_file_selection_dialog(self, all_lif, title='Select Files to Convert', message=f'Select which files to convert to TIFF (all {len(all_lif)} selected by default):')
                if selected is None:
                    return
                if len(selected) == 0:
                    show_message_dialog(self, 'No Files Selected', 'Please select at least one file to convert.', kind='warning')
                    return
                lif_files = selected
                total_files = len(lif_files)
                self.parent_window.log_output(f'Found {total_files} LIF file(s) to convert\n')
            else:
                lif_files = [file_path]
                total_files = 1
            output_base = (file_path.parent if file_path.is_file() else file_path) / 'Tif_files'
            output_base.mkdir(parents=True, exist_ok=True)
            self._last_conversion_output_dir = output_base
            separate_channels = self.separate_channels.isChecked()
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText('Converting...')
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)
            self._show_conversion_progress_window(total_files=total_files, current_file=lif_files[0] if lif_files else None, action='Converting LIF to TIFF')
            for (file_idx, lif_file) in enumerate(lif_files):
                try:
                    self.parent_window.log_output(f'Processing: {lif_file.name}\n')
                    progress = int(file_idx / total_files * 100)
                    self.parent_window.progress_bar.setValue(progress)
                    self._set_conversion_progress_window(progress, 'Converting LIF to TIFF', f'File {file_idx + 1}/{total_files}: {lif_file.name}')
                    QApplication.processEvents()
                    num_time_for_range = 1
                    try:
                        info = bioio_get_info(Path(lif_file))
                        if info:
                            num_time_for_range = info.get('num_time', 1) or 1
                    except Exception:
                        pass
                    (frame_start, frame_end) = self._get_frame_range(num_time_for_range)
                    frame_suffix = f'_f{frame_start + 1}' if frame_end - frame_start == 1 else ''
                    data = None
                    try:
                        data = bioio_load_data(Path(lif_file), frame_start=frame_start, frame_end=frame_end, use_first_z_only=True)
                    except Exception:
                        pass
                    if data is not None and data.size > 0:
                        all_frames = np.ascontiguousarray(data)
                        cropped = self._maybe_crop_for_conversion(all_frames, label=lif_file.name)
                        if cropped is None:
                            continue
                        all_frames = cropped
                        (num_time, num_channels) = (data.shape[0], data.shape[1])
                        num_z = 1
                        current_shape = all_frames.shape
                        channel_dim = 1 if num_channels > 1 else None
                        time_dim = 0
                        stem_used = lif_file.stem + frame_suffix
                        output_file = output_base / f'{stem_used}.tif'
                        if separate_channels and num_channels > 1 and (channel_dim is not None):
                            channel_names = self._get_effective_channel_names(num_channels)
                            fi_indices, foci_indices = self._conversion_fi_foci_index_sets(num_channels)
                            phase_used = False
                            max_channels = min(num_channels, current_shape[channel_dim])
                            for ch in range(max_channels):
                                indices = [slice(None)] * len(current_shape)
                                indices[channel_dim] = ch
                                channel_data = all_frames[tuple(indices)]
                                axes_to_squeeze = [i for (i, s) in enumerate(channel_data.shape) if s == 1 and (time_dim is None or i != time_dim)]
                                if axes_to_squeeze:
                                    channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                ch_output, phase_used = self._conversion_channel_output_path(
                                    output_base,
                                    stem_used,
                                    ch,
                                    channel_names,
                                    phase_used=phase_used,
                                    fi_indices=fi_indices,
                                    foci_indices=foci_indices,
                                )
                                tifffile.imwrite(
                                    str(ch_output),
                                    channel_data,
                                    photometric='minisblack',
                                    metadata=self._output_metadata_for_tiff(channel_data.shape[0] if channel_data.ndim >= 3 else 1),
                                    imagej=True,
                                )
                                self.parent_window.log_output(f'  Saved channel {ch + 1}: {ch_output.name}\n')
                                ch_pct = int((file_idx + (ch + 1) / max(1, max_channels)) / total_files * 100)
                                self._set_conversion_progress_window(ch_pct, 'Saving LIF channels', f'File {file_idx + 1}/{total_files}: {lif_file.name}\nChannel {ch + 1}/{max_channels}')
                            self.parent_window.log_output(f'✓ Converted: {lif_file.name}\n')
                        else:
                            combined = all_frames
                            axes_to_squeeze = [i for (i, s) in enumerate(combined.shape) if s == 1 and (time_dim is None or i != time_dim)]
                            if axes_to_squeeze:
                                combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                            if len(combined.shape) == 2:
                                combined = combined[np.newaxis, ...]
                            tifffile.imwrite(
                                str(output_file),
                                combined,
                                photometric='minisblack',
                                metadata=self._output_metadata_for_tiff(combined.shape[0] if combined.ndim >= 3 else 1),
                                imagej=True,
                            )
                            self.parent_window.log_output(f'  Saved: {output_file.name}\n')
                            self.parent_window.log_output(f'✓ Converted: {lif_file.name}\n')
                        continue
                    aics_converted = False
                    try:
                        lif_reader = AICSImage(str(lif_file))
                        try:
                            dims = lif_reader.dims
                            if hasattr(dims, 'order'):
                                dims_str = dims.order
                            elif isinstance(dims, str):
                                dims_str = dims
                            else:
                                dims_str = str(dims)
                            if hasattr(lif_reader, 'get_xarray_data'):
                                xdata = lif_reader.get_xarray_data()
                                all_frames = np.array(xdata)
                                num_time = int(xdata.sizes.get('T', 1))
                                num_channels = int(xdata.sizes.get('C', 1))
                                num_z = int(xdata.sizes.get('Z', 1))
                            else:
                                all_frames = np.array(lif_reader.data)
                                shape = all_frames.shape
                                num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                                num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                                num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                            if 'T' in dims_str:
                                t_idx = dims_str.index('T')
                                indices = [slice(None)] * len(all_frames.shape)
                                indices[t_idx] = slice(frame_start, min(frame_end, num_time))
                                all_frames = all_frames[tuple(indices)]
                            num_time = frame_end - frame_start
                            stem_used = lif_file.stem + (f'_f{frame_start + 1}' if num_time == 1 else '')
                            if num_z > 1 and 'Z' in dims_str:
                                z_idx = dims_str.index('Z')
                                if z_idx < len(all_frames.shape):
                                    indices = [slice(None)] * len(all_frames.shape)
                                    indices[z_idx] = 0
                                    all_frames = all_frames[tuple(indices)]
                                    num_z = 1
                            if len(all_frames.shape) > 4:
                                time_dim_idx = None
                                if 'T' in dims_str:
                                    time_dim_idx = dims_str.index('T')
                                while len(all_frames.shape) > 4:
                                    dim_to_remove = None
                                    for i in range(len(all_frames.shape) - 1, -1, -1):
                                        if time_dim_idx is None or i != time_dim_idx:
                                            if all_frames.shape[i] == 1:
                                                dim_to_remove = i
                                                break
                                    if dim_to_remove is not None:
                                        all_frames = np.squeeze(all_frames, axis=dim_to_remove)
                                        if dim_to_remove < len(dims_str):
                                            dims_str = dims_str[:dim_to_remove] + dims_str[dim_to_remove + 1:]
                                            if time_dim_idx is not None and dim_to_remove < time_dim_idx:
                                                time_dim_idx -= 1
                                    else:
                                        for i in range(len(all_frames.shape) - 1, -1, -1):
                                            if time_dim_idx is None or i != time_dim_idx:
                                                indices = [slice(None)] * len(all_frames.shape)
                                                indices[i] = 0
                                                all_frames = all_frames[tuple(indices)]
                                                if i < len(dims_str):
                                                    dims_str = dims_str[:i] + dims_str[i + 1:]
                                                    if time_dim_idx is not None and i < time_dim_idx:
                                                        time_dim_idx -= 1
                                                break
                            cropped = self._maybe_crop_for_conversion(all_frames, label=lif_file.name)
                            if cropped is None:
                                aics_converted = False
                                continue
                            all_frames = cropped
                            output_file = output_base / f'{stem_used}.tif'
                            current_shape = all_frames.shape
                            channel_dim = None
                            if num_channels > 1:
                                for (i, size) in enumerate(current_shape):
                                    if size == num_channels:
                                        channel_dim = i
                                        break
                            time_dim = None
                            if 'T' in dims_str:
                                t_idx = dims_str.index('T')
                                for (i, size) in enumerate(current_shape):
                                    if size == num_time and num_time > 1:
                                        time_dim = i
                                        break
                                if time_dim is None and t_idx < len(current_shape):
                                    time_dim = t_idx
                            if separate_channels and num_channels > 1 and (channel_dim is not None):
                                channel_names = self._get_effective_channel_names(num_channels)
                                fi_indices, foci_indices = self._conversion_fi_foci_index_sets(num_channels)
                                phase_used = False
                                max_channels = min(num_channels, current_shape[channel_dim])
                                for ch in range(max_channels):
                                    indices = [slice(None)] * len(current_shape)
                                    indices[channel_dim] = ch
                                    channel_data = all_frames[tuple(indices)]
                                    axes_to_squeeze = []
                                    for (i, size) in enumerate(channel_data.shape):
                                        if size == 1 and (time_dim is None or i != time_dim):
                                            axes_to_squeeze.append(i)
                                    if axes_to_squeeze:
                                        channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                    if len(channel_data.shape) == 2:
                                        channel_data = channel_data[np.newaxis, ...]
                                    ch_output, phase_used = self._conversion_channel_output_path(
                                        output_base,
                                        stem_used,
                                        ch,
                                        channel_names,
                                        phase_used=phase_used,
                                        fi_indices=fi_indices,
                                        foci_indices=foci_indices,
                                    )
                                    tifffile.imwrite(
                                        str(ch_output),
                                        channel_data,
                                        photometric='minisblack',
                                        metadata=self._output_metadata_for_tiff(channel_data.shape[0] if channel_data.ndim >= 3 else 1),
                                        imagej=True,
                                    )
                                    self.parent_window.log_output(f'  Saved channel {ch + 1}: {ch_output.name} ({channel_data.shape})\n')
                                ch_pct = int((file_idx + (ch + 1) / max(1, max_channels)) / total_files * 100)
                                self._set_conversion_progress_window(ch_pct, 'Saving LIF channels', f'File {file_idx + 1}/{total_files}: {lif_file.name}\nChannel {ch + 1}/{max_channels}')
                            else:
                                if channel_dim is not None and num_channels > 1 and (current_shape[channel_dim] > 1):
                                    combined = np.max(all_frames, axis=channel_dim)
                                else:
                                    combined = all_frames
                                axes_to_squeeze = []
                                for (i, size) in enumerate(combined.shape):
                                    if size == 1 and (time_dim is None or i != time_dim):
                                        axes_to_squeeze.append(i)
                                if axes_to_squeeze:
                                    combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                                if len(combined.shape) == 2:
                                    combined = combined[np.newaxis, ...]
                                self.parent_window.log_output(f'  Saving with shape: {combined.shape}\n')
                                tifffile.imwrite(
                                    str(output_file),
                                    combined,
                                    photometric='minisblack',
                                    metadata=self._output_metadata_for_tiff(combined.shape[0] if combined.ndim >= 3 else 1),
                                    imagej=True,
                                )
                                self.parent_window.log_output(f'  Saved: {output_file.name}\n')
                            self.parent_window.log_output(f'✓ Converted: {lif_file.name}\n')
                            aics_converted = True
                        finally:
                            if hasattr(lif_reader, 'close'):
                                try:
                                    lif_reader.close()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if aics_converted:
                        continue
                    try:
                        from readlif.reader import LifFile
                        lif_file_obj = LifFile(str(lif_file))
                        if getattr(lif_file_obj, 'num_images', 0) <= 0:
                            raise ValueError('No image series in LIF file')
                        best_idx = self._readlif_best_image_index(lif_file_obj)
                        image = lif_file_obj.get_image(best_idx)
                        (num_time_file, num_z, num_channels) = self._readlif_dims(image)
                        (frame_start_rl, frame_end_rl) = self._get_frame_range(num_time_file)
                        num_time = frame_end_rl - frame_start_rl
                        n_ch = max(1, num_channels)
                        frames_list = []
                        for t in range(num_time):
                            row = []
                            for c in range(n_ch):
                                try:
                                    row.append(np.array(image.get_frame(z=0, t=frame_start_rl + t, c=c)))
                                except Exception:
                                    if row:
                                        row.append(np.zeros_like(row[0]))
                                    else:
                                        row.append(np.zeros((1024, 1024), dtype=np.uint16))
                            if n_ch == 1:
                                frames_list.append(row[0])
                            else:
                                frames_list.append(np.stack(row, axis=0))
                        all_frames = np.ascontiguousarray(np.stack(frames_list, axis=0))
                        cropped = self._maybe_crop_for_conversion(all_frames, label=lif_file.name)
                        if cropped is None:
                            continue
                        all_frames = cropped
                        stem_used_rl = lif_file.stem + (f'_f{frame_start_rl + 1}' if num_time == 1 else '')
                        output_file = output_base / f'{stem_used_rl}.tif'
                        if separate_channels and n_ch > 1:
                            channel_names = self._get_effective_channel_names(n_ch)
                            fi_indices, foci_indices = self._conversion_fi_foci_index_sets(n_ch)
                            phase_used = False
                            for ch in range(n_ch):
                                channel_data = all_frames[:, ch, :, :]
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                ch_output, phase_used = self._conversion_channel_output_path(
                                    output_base,
                                    stem_used_rl,
                                    ch,
                                    channel_names,
                                    phase_used=phase_used,
                                    fi_indices=fi_indices,
                                    foci_indices=foci_indices,
                                )
                                tifffile.imwrite(
                                    str(ch_output),
                                    channel_data,
                                    photometric='minisblack',
                                    metadata=self._output_metadata_for_tiff(channel_data.shape[0] if channel_data.ndim >= 3 else 1),
                                    imagej=True,
                                )
                                self.parent_window.log_output(f'  Saved channel {ch + 1}: {ch_output.name}\n')
                                ch_pct = int((file_idx + (ch + 1) / max(1, n_ch)) / total_files * 100)
                                self._set_conversion_progress_window(ch_pct, 'Saving LIF channels', f'File {file_idx + 1}/{total_files}: {lif_file.name}\nChannel {ch + 1}/{n_ch}')
                            self.parent_window.log_output(f'✓ Converted (readlif): {lif_file.name}\n')
                        else:
                            if all_frames.ndim == 2:
                                all_frames = all_frames[np.newaxis, ...]
                            tifffile.imwrite(
                                str(output_file),
                                all_frames,
                                photometric='minisblack',
                                metadata=self._output_metadata_for_tiff(all_frames.shape[0] if all_frames.ndim >= 3 else 1),
                                imagej=True,
                            )
                            self.parent_window.log_output(f'  Saved: {output_file.name}\n')
                            self.parent_window.log_output(f'✓ Converted (readlif): {lif_file.name}\n')
                        continue
                    except Exception as e2:
                        self.parent_window.log_output(f'✗ Error converting {lif_file.name}: {str(e2)}\n')
                        show_message_dialog(self, 'Conversion Error', f'Error converting {lif_file.name}:\n{str(e2)}', kind='warning')
                except Exception as e:
                    self.parent_window.log_output(f'✗ Error converting {lif_file.name}: {str(e)}\n')
                    show_message_dialog(self, 'Conversion Error', f'Error converting {lif_file.name}:\n{str(e)}', kind='warning')
            self.parent_window.progress_bar.setValue(100)
            self._close_conversion_progress_window(complete=True)
            show_task_complete_dialog(self, title='Conversion Complete', message=f'Successfully converted {total_files} LIF file(s) to TIFF.', output_dir=getattr(self, '_last_conversion_output_dir', None), on_ok=self._advance_to_segmentation_after_conversion)
            self._performed_this_session = True
            self.workflow_step_finished.emit(True)
        except ImportError:
            self._close_conversion_progress_window()
            show_message_dialog(self, 'Library Missing', "LIF conversion requires 'bioio' and 'bioio-lif', or 'aicsimageio', and 'tifffile'.\n\nInstall with: pip install bioio bioio-lif tifffile\nOr: pip install aicsimageio tifffile", kind='warning')
        except Exception as e:
            self._close_conversion_progress_window()
            show_message_dialog(self, 'Conversion Error', f'LIF conversion failed:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText('Convert to TIFF')
            self.parent_window.progress_bar.setVisible(False)
            if getattr(self, '_conversion_progress_dialog', None) is not None:
                self._close_conversion_progress_window()

    def _convert_tiff(self):
        """Split a multi-channel TIFF into separate per-channel TIFFs (BactoMate convention: stem_fi_<name>.tif, stem_phase.tif)."""
        try:
            import tifffile
            file_path = self._resolve_tiff_path_for_operation(purpose='split')
            if file_path is None or not file_path.is_file():
                show_message_dialog(self, 'No File', 'Please select a TIFF file (or a folder containing TIFFs).', kind='warning')
                return
            tiff_info = get_tiff_file_info(file_path) or {}
            num_time = int(tiff_info.get('num_time') or self.file_info.get('num_time', 1))
            num_channels = int(tiff_info.get('num_channels') or self.file_info.get('num_channels', 1))
            channel_names = self._get_effective_channel_names(num_channels)
            series_axes = str(tiff_info.get('series_axes') or '')
            tiff_has_work = self._tiff_conversion_has_work(num_time=num_time, num_channels=num_channels)
            single_channel_transform = bool(num_channels < 2)
            if not tiff_has_work:
                self._show_tiff_noop_dialog()
                return
            output_base = file_path.parent / 'Tif_files'
            output_base.mkdir(parents=True, exist_ok=True)
            self._last_conversion_output_dir = output_base
            fi_indices: set[int] = set()
            foci_indices: set[int] = set()
            if getattr(self, 'fi_channel_checks', None) and len(self.fi_channel_checks) >= num_channels:
                fi_indices = {i for (i, cb) in enumerate(self.fi_channel_checks) if cb.isChecked()}
            if getattr(self, 'foci_channel_checks', None) and len(self.foci_channel_checks) >= num_channels:
                foci_indices = {i for (i, cb) in enumerate(self.foci_channel_checks) if cb.isChecked()}
            (frame_start_tiff, frame_end_tiff) = self._get_frame_range(num_time)
            n_t = frame_end_tiff - frame_start_tiff
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText('Converting...' if single_channel_transform else 'Splitting...')
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)
            self._show_conversion_progress_window(total_files=1, current_file=file_path, action='Converting TIFF' if single_channel_transform else 'Splitting TIFF channels')
            if single_channel_transform:
                self.parent_window.log_output(f'Converting single-channel TIFF: {file_path.name} ({num_time} frames)\n')
            else:
                self.parent_window.log_output(f'Splitting multi-channel TIFF: {file_path.name} ({num_time} frames, {num_channels} channels)\n')
            with tifffile.TiffFile(file_path) as tif:
                first_page = tif.pages[0]
                shape = first_page.shape
                num_pages = len(tif.pages)
                if len(shape) == 2 and num_pages > 1:
                    stack = tif.asarray()
                    if stack.shape[0] != num_pages:
                        stack = np.array([p.asarray() for p in tif.pages])
                    n_total = stack.shape[0]
                    n_t = min(n_t, n_total // num_channels)
                    start_ix = frame_start_tiff * num_channels
                    end_ix = (frame_start_tiff + n_t) * num_channels
                    stack = stack[start_ix:end_ix]
                    data = stack.reshape(n_t, num_channels, stack.shape[1], stack.shape[2])
                else:
                    raw = tif.asarray()
                    data = normalize_tiff_array_to_tcyx(raw, num_time=num_time, num_channels=num_channels, num_z=int(tiff_info.get('num_z') or 1), series_axes=series_axes)
                    data = data[frame_start_tiff:frame_end_tiff]
                    n_t = data.shape[0]
            if self.drift_correct.isChecked():
                from bactomate.scripts.convert_nd2_to_tif import drift_correct_stack
                self.parent_window.log_output('  Applying drift correction (frame-by-frame method)...\n')
                data = drift_correct_stack(np.ascontiguousarray(data), method='frame_by_frame')
                self.parent_window.log_output('  ✓ Drift correction complete\n')
            cropped_data = self._maybe_crop_for_conversion(np.ascontiguousarray(data), label=file_path.name)
            if cropped_data is None:
                return
            data = cropped_data
            if self.bleach_correct.isChecked():
                from bactomate.scripts.convert_nd2_to_tif import bleach_correct_stack
                bleach_methods = ['exponential', 'linear', 'histogram_matching', 'per_frame_minmax']
                bleach_method = bleach_methods[self.bleach_method_combo.currentIndex()]
                bleach_channel_indices = None
                if getattr(self, 'bleach_channel_checks', None):
                    indices = [i for (i, cb) in enumerate(self.bleach_channel_checks) if cb.isChecked()]
                    if indices and len(indices) < len(self.bleach_channel_checks):
                        bleach_channel_indices = indices
                if bleach_channel_indices is None:
                    self.parent_window.log_output(f'  Applying bleach correction ({bleach_method}) to all channels...\n')
                else:
                    self.parent_window.log_output(f'  Applying bleach correction ({bleach_method}) to channels: {bleach_channel_indices}...\n')
                data = bleach_correct_stack(np.ascontiguousarray(data), method=bleach_method, channel_indices=bleach_channel_indices)
                self.parent_window.log_output('  ✓ Bleach correction complete\n')
            from bactomate.modules.tiff_metadata import build_metadata_dict, read_calibration_from_path
            from bactomate.modules.conversion_channel_names import split_channel_output_name

            stem_suffix_tiff = f'_f{frame_start_tiff + 1}' if n_t == 1 else ''
            stem_used_tiff = file_path.stem + stem_suffix_tiff
            (file_ps, file_tp) = read_calibration_from_path(file_path)
            split_ps = self.file_info.get('pixel_size') if self.file_info and self.file_info.get('pixel_size') else file_ps
            split_tp = self.file_info.get('time_per_frame_min') if self.file_info and self.file_info.get('time_per_frame_min') else file_tp
            phase_used = False
            for ch in range(num_channels):
                pct = int(ch / max(1, num_channels) * 100)
                self.parent_window.progress_bar.setValue(pct)
                self._set_conversion_progress_window(pct, 'Converting TIFF' if single_channel_transform else 'Splitting TIFF channels', f'File 1/1: {file_path.name}\nChannel {ch + 1}/{num_channels}')
                channel_data = np.ascontiguousarray(data[:, ch])
                if channel_data.ndim == 2:
                    channel_data = channel_data[np.newaxis, ...]
                name = channel_names[ch] if ch < len(channel_names) else f'Channel_{ch + 1}'
                out_name, phase_used = split_channel_output_name(
                    stem_used_tiff,
                    name,
                    in_fi=ch in fi_indices,
                    in_foci=ch in foci_indices,
                    phase_used=phase_used,
                )
                out_path = output_base / out_name
                n_frames = int(channel_data.shape[0]) if channel_data.ndim >= 3 else 1
                write_meta = build_metadata_dict(pixel_size_um=split_ps if split_ps and split_ps > 0 else None, time_per_frame_min=split_tp if split_tp and split_tp > 0 else None, frames=n_frames, images=n_frames, axes='TYX')
                tifffile.imwrite(str(out_path), channel_data, photometric='minisblack', metadata=write_meta, imagej=True)
                self.parent_window.log_output(f'  Saved: {out_path.name}\n')
            self.parent_window.progress_bar.setValue(100)
            self._close_conversion_progress_window(complete=True)
            show_task_complete_dialog(self, title='Conversion Complete', message=f'Converted {file_path.name}.' if single_channel_transform else f'Split {file_path.name} into {num_channels} channel TIFF(s).', output_dir=getattr(self, '_last_conversion_output_dir', None), on_ok=self._advance_to_segmentation_after_conversion)
            self._performed_this_session = True
            self.workflow_step_finished.emit(True)
        except Exception as e:
            self._close_conversion_progress_window()
            self.parent_window.log_output(f'Error: {str(e)}\n')
            show_message_dialog(self, 'Split failed', f'Could not split TIFF:\n{str(e)}', kind='error')
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText('Convert to TIFF')
            self.parent_window.progress_bar.setVisible(False)
            if getattr(self, '_conversion_progress_dialog', None) is not None:
                self._close_conversion_progress_window()

    def _convert_czi(self):
        """Convert CZI file(s) to TIFF using czifile (https://pypi.org/project/czifile/)."""
        try:
            import numpy as np
            import tifffile
            import czifile
            file_path = Path(self.file_path.text())
            if file_path.is_dir():
                all_czi = sorted(list(file_path.glob('*.czi')) + list(file_path.glob('*.CZI')))
                if not all_czi:
                    show_message_dialog(self, 'No CZI Files', 'No CZI files found in the selected folder.', kind='warning')
                    return
                selected = show_file_selection_dialog(self, all_czi, title='Select Files to Convert', message=f'Select which files to convert to TIFF (all {len(all_czi)} selected by default):')
                if selected is None:
                    return
                if len(selected) == 0:
                    show_message_dialog(self, 'No Files Selected', 'Please select at least one file to convert.', kind='warning')
                    return
                czi_files = selected
                total_files = len(czi_files)
                self.parent_window.log_output(f'Found {total_files} CZI file(s) to convert\n')
            else:
                czi_files = [file_path]
                total_files = 1
            output_base = (file_path.parent if file_path.is_file() else file_path) / 'Tif_files'
            output_base.mkdir(parents=True, exist_ok=True)
            self._last_conversion_output_dir = output_base
            separate_channels = self.separate_channels.isChecked()
            max_z_projection = self.max_z_projection.isChecked()
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText('Converting...')
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)
            self._show_conversion_progress_window(total_files=total_files, current_file=czi_files[0] if czi_files else None, action='Converting CZI to TIFF')
            for (file_idx, czi_file) in enumerate(czi_files):
                try:
                    self.parent_window.log_output(f'Processing: {czi_file.name}\n')
                    progress = int(file_idx / total_files * 100)
                    self.parent_window.progress_bar.setValue(progress)
                    self._set_conversion_progress_window(progress, 'Converting CZI to TIFF', f'File {file_idx + 1}/{total_files}: {czi_file.name}')
                    QApplication.processEvents()
                    num_time_czi = 1
                    try:
                        info = bioio_get_info(Path(czi_file))
                        if info:
                            num_time_czi = info.get('num_time', 1) or 1
                    except Exception:
                        pass
                    (frame_start_czi, frame_end_czi) = self._get_frame_range(num_time_czi)
                    frame_suffix_czi = f'_f{frame_start_czi + 1}' if frame_end_czi - frame_start_czi == 1 else ''
                    data = bioio_load_data(Path(czi_file), frame_start=frame_start_czi, frame_end=frame_end_czi, use_first_z_only=not max_z_projection, max_z_projection=max_z_projection)
                    if data is not None and data.size > 0:
                        all_frames = np.ascontiguousarray(data)
                        cropped = self._maybe_crop_for_conversion(all_frames, label=czi_file.name)
                        if cropped is None:
                            continue
                        all_frames = cropped
                        (num_time, num_channels) = (data.shape[0], data.shape[1])
                        num_z = 1
                        current_shape = all_frames.shape
                        channel_dim = 1 if num_channels > 1 else None
                        time_dim = 0
                        dims_str = 'TCYX'
                        stem_used_czi = czi_file.stem + frame_suffix_czi
                        output_file = output_base / f'{stem_used_czi}.tif'
                        if separate_channels and num_channels > 1 and (channel_dim is not None):
                            channel_names = self._get_effective_channel_names(num_channels)
                            fi_indices, foci_indices = self._conversion_fi_foci_index_sets(num_channels)
                            phase_used = False
                            max_ch = min(num_channels, current_shape[channel_dim])
                            for ch in range(max_ch):
                                indices = [slice(None)] * len(current_shape)
                                indices[channel_dim] = ch
                                channel_data = all_frames[tuple(indices)]
                                axes_to_squeeze = [i for (i, s) in enumerate(channel_data.shape) if s == 1 and (time_dim is None or i != time_dim)]
                                if axes_to_squeeze:
                                    channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                ch_output, phase_used = self._conversion_channel_output_path(
                                    output_base,
                                    stem_used_czi,
                                    ch,
                                    channel_names,
                                    phase_used=phase_used,
                                    fi_indices=fi_indices,
                                    foci_indices=foci_indices,
                                )
                                tifffile.imwrite(
                                    str(ch_output),
                                    channel_data,
                                    photometric='minisblack',
                                    metadata=self._output_metadata_for_tiff(channel_data.shape[0] if channel_data.ndim >= 3 else 1),
                                    imagej=True,
                                )
                                self.parent_window.log_output(f'  Saved channel {ch + 1}: {ch_output.name}\n')
                                ch_pct = int((file_idx + (ch + 1) / max(1, max_ch)) / total_files * 100)
                                self._set_conversion_progress_window(ch_pct, 'Saving CZI channels', f'File {file_idx + 1}/{total_files}: {czi_file.name}\nChannel {ch + 1}/{max_ch}')
                            self.parent_window.log_output(f'✓ Converted: {czi_file.name}\n')
                        else:
                            combined = all_frames
                            axes_to_squeeze = [i for (i, s) in enumerate(combined.shape) if s == 1 and (time_dim is None or i != time_dim)]
                            if axes_to_squeeze:
                                combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                            if len(combined.shape) == 2:
                                combined = combined[np.newaxis, ...]
                            tifffile.imwrite(
                                str(output_file),
                                combined,
                                photometric='minisblack',
                                metadata=self._output_metadata_for_tiff(combined.shape[0] if combined.ndim >= 3 else 1),
                                imagej=True,
                            )
                            self.parent_window.log_output(f'  Saved: {output_file.name}\n')
                            self.parent_window.log_output(f'✓ Converted: {czi_file.name}\n')
                        continue
                    with czifile.CziFile(str(czi_file)) as czi_f:
                        axes = getattr(czi_f, 'axes', None)
                        shape = getattr(czi_f, 'shape', None)
                        if axes and shape and (len(axes) == len(shape)):
                            (num_time, num_channels, num_z) = self._czi_dims_from_axes(axes, shape)
                        else:
                            (num_time, num_channels, num_z) = (1, 1, 1)
                        arr = czi_f.asarray() if hasattr(czi_f, 'asarray') else np.array(czi_f)
                    if arr is None or arr.size == 0:
                        raise RuntimeError('czifile returned empty array')
                    if axes and len(axes) == len(arr.shape):
                        num_time = frame_end_czi - frame_start_czi
                        if num_z > 1 and max_z_projection:
                            (all_frames, dims_str, z_axis) = self._czi_slice_to_t_c_z_y_x(arr, axes, frame_start=frame_start_czi, frame_end=frame_end_czi)
                            if z_axis >= 0:
                                all_frames = np.max(all_frames, axis=z_axis).astype(all_frames.dtype)
                            dims_str = ''.join((a for a in dims_str if a != 'Z')) or 'TCYX'
                        else:
                            (all_frames, dims_str) = self._czi_slice_to_t_c_y_x(arr, axes, frame_start=frame_start_czi, frame_end=frame_end_czi)
                        all_frames = np.ascontiguousarray(all_frames)
                    else:
                        sh = arr.shape
                        dims_str = ''
                        if len(sh) >= 5:
                            num_time = frame_end_czi - frame_start_czi
                            num_channels = int(sh[1])
                            num_z = int(sh[2])
                            all_frames = arr[frame_start_czi:frame_end_czi, :, :, :, :]
                            if max_z_projection and num_z > 1:
                                all_frames = np.max(all_frames, axis=2).astype(all_frames.dtype)
                            else:
                                all_frames = all_frames[:, :, 0, :, :]
                            all_frames = np.ascontiguousarray(all_frames)
                            dims_str = 'TCYX'
                        elif len(sh) == 4:
                            num_time = frame_end_czi - frame_start_czi
                            num_channels = int(sh[1])
                            num_z = 1
                            all_frames = np.ascontiguousarray(arr[frame_start_czi:frame_end_czi, :, :, :])
                            dims_str = 'TCYX'
                        elif len(sh) == 3:
                            num_time = frame_end_czi - frame_start_czi
                            num_channels = 1
                            num_z = 1
                            all_frames = np.ascontiguousarray(arr[frame_start_czi:frame_end_czi, :, :][:, np.newaxis, :, :])
                            dims_str = 'TCYX'
                        else:
                            raise RuntimeError(f'Unexpected CZI shape: {sh}')
                    if all_frames is None or all_frames.size == 0:
                        raise RuntimeError('No image data loaded')
                    if len(all_frames.shape) > 4:
                        time_dim_idx = None
                        if 'T' in dims_str:
                            time_dim_idx = dims_str.index('T')
                        while len(all_frames.shape) > 4:
                            dim_to_remove = None
                            for i in range(len(all_frames.shape) - 1, -1, -1):
                                if time_dim_idx is None or i != time_dim_idx:
                                    if all_frames.shape[i] == 1:
                                        dim_to_remove = i
                                        break
                            if dim_to_remove is not None:
                                all_frames = np.squeeze(all_frames, axis=dim_to_remove)
                                if dim_to_remove < len(dims_str):
                                    dims_str = dims_str[:dim_to_remove] + dims_str[dim_to_remove + 1:]
                                    if time_dim_idx is not None and dim_to_remove < time_dim_idx:
                                        time_dim_idx -= 1
                            else:
                                for i in range(len(all_frames.shape) - 1, -1, -1):
                                    if time_dim_idx is None or i != time_dim_idx:
                                        indices = [slice(None)] * len(all_frames.shape)
                                        indices[i] = 0
                                        all_frames = all_frames[tuple(indices)]
                                        if i < len(dims_str):
                                            dims_str = dims_str[:i] + dims_str[i + 1:]
                                            if time_dim_idx is not None and i < time_dim_idx:
                                                time_dim_idx -= 1
                                        break
                    cropped = self._maybe_crop_for_conversion(all_frames, label=czi_file.name)
                    if cropped is None:
                        continue
                    all_frames = cropped
                    stem_used_czi = czi_file.stem + frame_suffix_czi
                    output_file = output_base / f'{stem_used_czi}.tif'
                    current_shape = all_frames.shape
                    channel_dim = None
                    if num_channels > 1:
                        for (i, size) in enumerate(current_shape):
                            if size == num_channels:
                                channel_dim = i
                                break
                    time_dim = None
                    if 'T' in dims_str:
                        t_idx = dims_str.index('T')
                        for (i, size) in enumerate(current_shape):
                            if size == num_time and num_time > 1:
                                time_dim = i
                                break
                        if time_dim is None and t_idx < len(current_shape):
                            time_dim = t_idx
                    if separate_channels and num_channels > 1 and (channel_dim is not None):
                        channel_names = self._get_effective_channel_names(num_channels)
                        fi_indices, foci_indices = self._conversion_fi_foci_index_sets(num_channels)
                        phase_used = False
                        max_channels = min(num_channels, current_shape[channel_dim])
                        for ch in range(max_channels):
                            indices = [slice(None)] * len(current_shape)
                            indices[channel_dim] = ch
                            channel_data = all_frames[tuple(indices)]
                            axes_to_squeeze = []
                            for (i, size) in enumerate(channel_data.shape):
                                if size == 1 and (time_dim is None or i != time_dim):
                                    axes_to_squeeze.append(i)
                            if axes_to_squeeze:
                                channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                            if len(channel_data.shape) == 2:
                                channel_data = channel_data[np.newaxis, ...]
                            ch_output, phase_used = self._conversion_channel_output_path(
                                output_base,
                                stem_used_czi,
                                ch,
                                channel_names,
                                phase_used=phase_used,
                                fi_indices=fi_indices,
                                foci_indices=foci_indices,
                            )
                            tifffile.imwrite(
                                str(ch_output),
                                channel_data,
                                photometric='minisblack',
                                metadata=self._output_metadata_for_tiff(channel_data.shape[0] if channel_data.ndim >= 3 else 1),
                                imagej=True,
                            )
                            self.parent_window.log_output(f'  Saved channel {ch + 1}: {ch_output.name} ({channel_data.shape})\n')
                            ch_pct = int((file_idx + (ch + 1) / max(1, max_channels)) / total_files * 100)
                            self._set_conversion_progress_window(ch_pct, 'Saving CZI channels', f'File {file_idx + 1}/{total_files}: {czi_file.name}\nChannel {ch + 1}/{max_channels}')
                    else:
                        if channel_dim is not None and num_channels > 1 and (current_shape[channel_dim] > 1):
                            combined = np.max(all_frames, axis=channel_dim)
                        else:
                            combined = all_frames
                        axes_to_squeeze = []
                        for (i, size) in enumerate(combined.shape):
                            if size == 1 and (time_dim is None or i != time_dim):
                                axes_to_squeeze.append(i)
                        if axes_to_squeeze:
                            combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                        if len(combined.shape) == 2:
                            combined = combined[np.newaxis, ...]
                        self.parent_window.log_output(f'  Saving with shape: {combined.shape}\n')
                        tifffile.imwrite(
                            str(output_file),
                            combined,
                            photometric='minisblack',
                            metadata=self._output_metadata_for_tiff(combined.shape[0] if combined.ndim >= 3 else 1),
                            imagej=True,
                        )
                        self.parent_window.log_output(f'  Saved: {output_file.name}\n')
                    self.parent_window.log_output(f'✓ Converted: {czi_file.name}\n')
                except Exception as e:
                    self.parent_window.log_output(f'✗ Error converting {czi_file.name}: {str(e)}\n')
                    show_message_dialog(self, 'Conversion Error', f'Error converting {czi_file.name}:\n{str(e)}', kind='warning')
            self.parent_window.progress_bar.setValue(100)
            self._close_conversion_progress_window(complete=True)
            show_task_complete_dialog(self, title='Conversion Complete', message=f'Successfully converted {total_files} CZI file(s) to TIFF.', output_dir=getattr(self, '_last_conversion_output_dir', None), on_ok=self._advance_to_segmentation_after_conversion)
            self._performed_this_session = True
            self.workflow_step_finished.emit(True)
        except ImportError:
            self._close_conversion_progress_window()
            show_message_dialog(self, 'Library Missing', "CZI conversion requires 'bioio' and 'bioio-czi', or 'tifffile' and 'czifile'.\n\nInstall with: pip install bioio bioio-czi tifffile\nOr: pip install tifffile czifile", kind='warning')
        except Exception as e:
            self._close_conversion_progress_window()
            show_message_dialog(self, 'Conversion Error', f'CZI conversion failed:\n{str(e)}', kind='error')
            self.parent_window.log_output(f'Error: {str(e)}\n')
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText('Convert to TIFF')
            self.parent_window.progress_bar.setVisible(False)
            if getattr(self, '_conversion_progress_dialog', None) is not None:
                self._close_conversion_progress_window()

    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (includes path for workflow replay)."""
        out = get_from_widgets([('frame_extract_mode', self.frame_extract_mode), ('max_frames', self.max_frames), ('single_frame', self.single_frame_spin), ('separate_channels', self.separate_channels), ('max_z_projection', self.max_z_projection), ('drift_correct', self.drift_correct), ('bleach_correct', self.bleach_correct)])
        if hasattr(self, 'bleach_method_combo'):
            out['bleach_method'] = self.bleach_method_combo.currentIndex()
        if getattr(self, 'bleach_channel_checks', None):
            out['bleach_fluorescence_channels'] = [cb.isChecked() for cb in self.bleach_channel_checks]
        if getattr(self, 'fi_channel_checks', None):
            out['fi_channels'] = [cb.isChecked() for cb in self.fi_channel_checks]
        if getattr(self, 'foci_channel_checks', None):
            out['foci_channels'] = [cb.isChecked() for cb in self.foci_channel_checks]
        if getattr(self, 'zproject_channel_checks', None):
            out['zproject_channels'] = [cb.isChecked() for cb in self.zproject_channel_checks]
        path_txt = (self.file_path.text() or '').strip()
        if path_txt:
            out['file_path'] = path_txt
        out['performed'] = getattr(self, '_performed_this_session', False)
        return out

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values. Set file path and load file info first so channel options (FI, Z, Foci, bleach) exist, then apply their checkboxes."""
        apply_to_widgets(data, [('frame_extract_mode', self.frame_extract_mode), ('max_frames', self.max_frames), ('single_frame', self.single_frame_spin), ('separate_channels', self.separate_channels), ('max_z_projection', self.max_z_projection), ('drift_correct', self.drift_correct), ('bleach_correct', self.bleach_correct)])
        if 'bleach_method' in data and hasattr(self, 'bleach_method_combo'):
            try:
                idx = int(data['bleach_method'])
                if 0 <= idx < self.bleach_method_combo.count():
                    self.bleach_method_combo.setCurrentIndex(idx)
            except (TypeError, ValueError):
                pass
        if 'file_path' in data and data.get('file_path'):
            path = str(data['file_path']).strip()
            self.file_path.setText(path)
            if path and Path(path).exists():
                if Path(path).is_file():
                    self.load_file_info(path)
                else:
                    folder = Path(path)
                    supported = sorted(list(folder.glob('*.nd2')) + list(folder.glob('*.ND2')) + list(folder.glob('*.lif')) + list(folder.glob('*.LIF')) + list(folder.glob('*.czi')) + list(folder.glob('*.CZI')) + list(folder.glob('*.tif')) + list(folder.glob('*.tiff')) + list(folder.glob('*.TIF')) + list(folder.glob('*.TIFF')))
                    if supported:
                        self.load_folder_info(path, supported)
        if 'bleach_fluorescence_channels' in data and getattr(self, 'bleach_channel_checks', None):
            try:
                states = data['bleach_fluorescence_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.bleach_channel_checks):
                            self.bleach_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if 'fi_channels' in data and getattr(self, 'fi_channel_checks', None):
            try:
                states = data['fi_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.fi_channel_checks):
                            self.fi_channel_checks[i].setChecked(bool(checked))
            except Exception:
                pass
        if 'foci_channels' in data and getattr(self, 'foci_channel_checks', None):
            try:
                states = data['foci_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.foci_channel_checks):
                            self.foci_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if 'zproject_channels' in data and getattr(self, 'zproject_channel_checks', None):
            try:
                states = data['zproject_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.zproject_channel_checks):
                            self.zproject_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if 'performed' in data:
            self._performed_this_session = bool(data['performed'])

    def apply_channel_options_from_session(self, data: dict) -> None:
        """Apply only FI/Foci/bleach/Z-project channel checkboxes from session data.
        Use after loading a different file (e.g. on Re-run) so the new file's channel list
        gets the same FI/Foci selection as the original session."""
        if not isinstance(data, dict):
            return
        if 'bleach_fluorescence_channels' in data and getattr(self, 'bleach_channel_checks', None):
            try:
                states = data['bleach_fluorescence_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.bleach_channel_checks):
                            self.bleach_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if 'fi_channels' in data and getattr(self, 'fi_channel_checks', None):
            try:
                states = data['fi_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.fi_channel_checks):
                            self.fi_channel_checks[i].setChecked(bool(checked))
            except Exception:
                pass
        if 'foci_channels' in data and getattr(self, 'foci_channel_checks', None):
            try:
                states = data['foci_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.foci_channel_checks):
                            self.foci_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if 'zproject_channels' in data and getattr(self, 'zproject_channel_checks', None):
            try:
                states = data['zproject_channels']
                if isinstance(states, list):
                    for (i, checked) in enumerate(states):
                        if i < len(self.zproject_channel_checks):
                            self.zproject_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
