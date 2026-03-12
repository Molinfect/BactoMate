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
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.modules.bioio_reader import get_info as bioio_get_info, is_available as bioio_is_available, load_data as bioio_load_data
from bactimate.gui.constants import CONVERT_ND2_SCRIPT, LOGS_DIR, PROJECT_ROOT
from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.preferences import LAST_DIR_ND2, get_default_browse_dir
from bactimate.gui.image_utils import to_qpixmap, scale_for_preview
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.utils import safe_imread
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog
from bactimate.gui.workers import WorkerThread


class ND2ConversionTab(QWidget):
    """Tab for converting microscopy files (ND2, LIF, CZI) to TIFF with metadata extraction"""
    workflow_step_finished = pyqtSignal(bool)  # True = success; used for workflow replay chain

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.file_info = None  # Store file info
        self._performed_this_session = False  # True if user ran conversion this session (for Re-run only same steps)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        file_group = self._build_file_group()
        scroll_layout.addWidget(file_group)

        params_group = self._build_conversion_params_group()
        scroll_layout.addWidget(params_group)

        # Metadata info
        metadata_group = QGroupBox("Extracted Metadata")
        metadata_layout = QVBoxLayout()
        
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMinimumHeight(120)  # Increased for better visibility
        self.metadata_text.setMaximumHeight(200)  # Prevent excessive expansion
        self.metadata_text.setPlaceholderText("Metadata will be displayed here after loading microscopy file...")
        # Set larger, more readable font
        metadata_font = QFont()
        metadata_font.setPointSize(12)  # Increased font size for better readability
        self.metadata_text.setFont(metadata_font)
        # Prevent vertical expansion - only allow horizontal
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.metadata_text.setSizePolicy(size_policy)
        metadata_layout.addWidget(self.metadata_text)
        
        metadata_group.setLayout(metadata_layout)
        scroll_layout.addWidget(metadata_group)
        
        # Preview button (moved here to match Single File tab layout)
        self.preview_btn = QPushButton("Preview File")
        # Modern styling applied via global stylesheet
        self.preview_btn.clicked.connect(self.preview_file)
        self.preview_btn.setEnabled(False)
        scroll_layout.addWidget(self.preview_btn)
        
        # Convert button
        self.convert_btn = QPushButton("Convert to TIFF")
        # Modern styling applied via global stylesheet
        self.convert_btn.clicked.connect(self.convert_nd2)
        self.convert_btn.setEnabled(False)
        scroll_layout.addWidget(self.convert_btn)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_file_group(self) -> QWidget:
        """Build the microscopy file selection group (Path + info, same layout as segmentation tabs)."""
        file_group = DragDropGroupBox(
            "Run convert microscopy file",
            parent=self,
            on_drop_callback=self.handle_dropped_path,
        )
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Path:"))
        file_layout.addWidget(create_info_icon(
            "Path to the microscopy file or folder to convert (ND2, LIF, CZI, or TIFF).\n\n"
            "Converted TIFFs are saved in a single folder in the same directory: 'Tif_files' "
            "(reused if it already exists). For a single file, that folder is next to the file; for a folder, inside it.",
            self,
        ))
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a microscopy file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(200)
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        self.file_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)
        self.re_run_analysis_btn = QPushButton("Re-run analysis")
        self.re_run_analysis_btn.setFixedWidth(130)
        self.re_run_analysis_btn.clicked.connect(self._apply_session_to_path)
        self.re_run_analysis_btn.setEnabled(False)
        re_run_info_icon = create_info_icon(
            "Re-run analysis: Set the path above to your new file or folder (can be different from the session log). Choose a session log (JSON). The app applies all parameters from that session (FI, z-stack, foci, model, filters, etc.) and runs the full pipeline on this path: conversion → segment → analyze → tracking. Outputs go next to the input (Tif_files, _segmentation_omni, etc.). Same analysis type, different data.",
            self,
        )
        self.file_path.textChanged.connect(self._update_re_run_analysis_enabled)
        file_layout.addWidget(self.file_path, 1)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        file_layout.addWidget(self.re_run_analysis_btn, 0)
        file_layout.addWidget(re_run_info_icon, 0)
        file_group.setLayout(file_layout)
        return file_group

    def _build_conversion_params_group(self) -> QGroupBox:
        """Build the Conversion Parameters group (frames, channels, corrections)."""
        params_group = QGroupBox("Conversion Parameters")
        params_layout = QVBoxLayout()
        top_row = QHBoxLayout()
        self.frame_extract_mode = QComboBox()
        self.frame_extract_mode.addItems(["First N frames", "Single frame only"])
        self.frame_extract_mode.setToolTip(
            "First N frames: extract frames 1 through N (e.g. 50 → frames 1–50).\n"
            "Single frame only: extract only one time point (all channels) by frame number (e.g. frame 50)."
        )
        top_row.addWidget(QLabel("Extract:"))
        top_row.addWidget(self.frame_extract_mode)
        frame_info_icon = create_info_icon(
            "Number of time frames to extract from the microscopy file.\n\n"
            "First N frames: extract from the start (e.g. 50 = frames 1–50).\n"
            "Single frame only: extract only one time point by its frame number (1-based); all channels of that frame are included.\n\n"
            "The maximum value updates when a file is loaded.",
            parent_widget=self
        )
        self.frame_label = QLabel("Number of Frames:")
        self.frame_label.setToolTip("Used when 'First N frames' is selected.")
        top_row.addWidget(self.frame_label)
        top_row.addWidget(frame_info_icon)
        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 10000)
        self.max_frames.setValue(100)
        top_row.addWidget(self.max_frames)
        self.single_frame_label = QLabel("Frame (1-based):")
        self.single_frame_label.setToolTip("Used when 'Single frame only' is selected. Extracts only this time point (all channels).")
        self.single_frame_spin = QSpinBox()
        self.single_frame_spin.setRange(1, 10000)
        self.single_frame_spin.setValue(1)
        top_row.addWidget(self.single_frame_label)
        top_row.addWidget(self.single_frame_spin)
        self.single_frame_label.setVisible(False)
        self.single_frame_spin.setVisible(False)
        self.frame_extract_mode.currentIndexChanged.connect(self._on_frame_extract_mode_changed)
        self._on_frame_extract_mode_changed(self.frame_extract_mode.currentIndex())
        top_row.addSpacing(16)
        channel_label = QLabel("Channel Handling:")
        top_row.addWidget(channel_label)
        self.separate_channels = QCheckBox("Save channels separately")
        self.separate_channels.setChecked(True)
        top_row.addWidget(self.separate_channels)
        top_row.addStretch()
        params_layout.addLayout(top_row)
        # Channel options: visible when file has multiple channels; one row per channel with FI / Z / Bleach checkboxes
        self.channel_options_group = QGroupBox("Channel options")
        self.channel_options_layout = QVBoxLayout()
        self.channel_options_group.setLayout(self.channel_options_layout)
        self.channel_options_group.setVisible(False)
        self.fi_channel_checks: list[QCheckBox] = []
        self.foci_channel_checks: list[QCheckBox] = []
        self.zproject_channel_checks: list[QCheckBox] = []
        self.bleach_channel_checks: list[QCheckBox] = []
        params_layout.addWidget(self.channel_options_group)
        corrections_group = QGroupBox("Image Corrections")
        # Override padding so layout margins actually move the content (global style uses padding-top: 14px)
        corrections_group.setStyleSheet(
            "QGroupBox {"
            " border: 1px solid #2d2d2d; border-radius: 8px; margin-top: 10px; padding-top: 6px;"
            " background-color: #1a1a1a; font-weight: 600; font-size: 11pt; color: #e0e0e0;"
            "}"
            " QGroupBox::title {"
            " subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 2px 8px;"
            " background-color: #1a1a1a; color: #4a9eff; border-radius: 4px;"
            " }"
        )
        corrections_layout = QHBoxLayout()
        self.drift_correct = QCheckBox("Apply drift correction (X,Y registration)")
        self.drift_correct.setChecked(False)
        drift_info_icon = create_info_icon(
            "Apply X,Y drift correction to compensate for stage movement during time-lapse imaging.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: Optional correction step\n\n"
            "Method: Phase cross-correlation (frame-by-frame alignment)\n"
            "Algorithm: Each frame is aligned to the PREVIOUS frame (incremental correction).\n"
            "This approach is more robust to large cumulative drifts because:\n"
            "- Smaller shifts between consecutive frames improve alignment accuracy\n"
            "- Phase cross-correlation works best with small shifts\n"
            "- Naturally handles gradual drift over time\n\n"
            "Use when: You notice the sample drifting over time.\n"
            "Note: Increases processing time but improves temporal analysis accuracy.\n"
            "The first frame remains unchanged; subsequent frames are incrementally aligned.",
            parent_widget=self
        )
        corrections_layout.addWidget(self.drift_correct)
        corrections_layout.addWidget(drift_info_icon)
        corrections_layout.addSpacing(38)
        self.bleach_correct = QCheckBox("Apply photobleaching correction")
        self.bleach_correct.setChecked(False)
        self.bleach_method_combo = QComboBox()
        self.bleach_method_combo.addItems([
            "Exponential (intensity preservation)",
            "Linear decay",
            "Histogram matching",
            "Per-frame min-max (visualization)",
        ])
        self.bleach_method_combo.setCurrentIndex(0)
        self.bleach_method_combo.setToolTip("Method used for bleach correction. See info icon for details.")
        bleach_info_icon = create_info_icon(
            "Photobleaching correction (inspired by CorrectBleach: https://github.com/fiji/CorrectBleach).\n\n"
            "Which channels: Open the 'Channel' dropdown and tick the 'Bleach' checkbox for each fluorescence channel (e.g. GFP, mScarlet); "
            "bleach correction is applied only to those. Leave phase/transmission unticked.\n\n"
            "Methods:\n"
            "• Exponential: I(t)=I₀×exp(-t/τ), normalizes to first frame. Use for intensity quantification.\n"
            "• Linear: Linear decay fit, then scale to first frame.\n"
            "• Histogram matching: Match each frame's histogram to frame 0. Good for segmentation; not for quantification.\n"
            "• Per-frame min-max: Stretch each frame so its min→0 and max→full range of the current bit depth (e.g. 65535 for 16-bit). No RGB conversion; dtype and bit depth are preserved. Visualization only.",
            parent_widget=self
        )
        corrections_layout.addWidget(self.bleach_correct)
        corrections_layout.addSpacing(10)
        corrections_layout.addWidget(QLabel("Method:"))
        corrections_layout.addWidget(self.bleach_method_combo)
        corrections_layout.addWidget(bleach_info_icon)
        corrections_layout.addSpacing(24)
        self.max_z_projection = QCheckBox("Max intensity projection for multiple Z stack")
        self.max_z_projection.setChecked(False)
        corrections_layout.addWidget(self.max_z_projection)
        z_projection_info_icon = create_info_icon(
            "Apply maximum intensity projection across Z-slices when Z > 1.\n\n"
            "When enabled: For each pixel, takes the maximum intensity value across all Z-slices. "
            "This creates a single 2D image from a 3D Z-stack.\n\n"
            "When disabled: Uses only the first Z-slice (Z=1).\n\n"
            "Use when: You have Z-stack data and want to combine all slices into one image.",
            parent_widget=self
        )
        corrections_layout.addWidget(z_projection_info_icon)
        corrections_layout.addStretch()
        corrections_group.setLayout(corrections_layout)
        params_layout.addWidget(corrections_group)
        params_group.setLayout(params_layout)
        return params_group

    def _on_frame_extract_mode_changed(self, index: int) -> None:
        """Show/hide Number of Frames vs Single frame spinbox based on extract mode."""
        is_single = index == 1  # "Single frame only"
        self.frame_label.setVisible(not is_single)
        self.max_frames.setVisible(not is_single)
        self.single_frame_label.setVisible(is_single)
        self.single_frame_spin.setVisible(is_single)

    def _get_frame_range(self, num_time: int) -> tuple[int, int]:
        """Return (start_0based, end_0based) for conversion. end is exclusive."""
        if num_time <= 0:
            return 0, 0
        if self.frame_extract_mode.currentIndex() == 1:  # Single frame only
            one_based = self.single_frame_spin.value()
            start = max(0, min(one_based - 1, num_time - 1))
            return start, start + 1
        n = min(self.max_frames.value(), num_time)
        return 0, n

    def select_file(self):
        start_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_ND2)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Microscopy File", start_dir,
            "Microscopy Files (*.nd2 *.ND2 *.lif *.LIF *.czi *.CZI *.tif *.tiff *.TIF *.TIFF);;"
            "ND2 Files (*.nd2 *.ND2);;"
            "LIF Files (*.lif *.LIF);;"
            "CZI Files (*.czi *.CZI);;"
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;"
            "All Files (*)"
        )
        if file_path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_ND2, file_path)
            self.file_path.setText(file_path)
            self.load_file_info(file_path)

    def select_folder(self):
        start_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_ND2)
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Microscopy Files", start_dir)
        if folder_path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_ND2, folder_path)
            self.file_path.setText(folder_path)
            # Load info from all supported files in the folder
            folder = Path(folder_path)
            supported_files = sorted(
                list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
            )
            if supported_files:
                self.load_folder_info(folder_path, supported_files)
            else:
                self.metadata_text.setText("No supported microscopy files (ND2, LIF, CZI, TIFF) found in selected folder")
                self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 12pt;")

    def _update_re_run_analysis_enabled(self) -> None:
        """Enable Re-run analysis only when a path is entered."""
        if hasattr(self, "re_run_analysis_btn"):
            self.re_run_analysis_btn.setEnabled(bool((self.file_path.text() or "").strip()))

    def _apply_session_to_path(self) -> None:
        """Let user choose a session log; apply it and run the full workflow (conversion → segment → analyze → tracking) for all steps that have a path in the log."""
        log_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose session log",
            str(LOGS_DIR) if LOGS_DIR.exists() else get_default_browse_dir(),
            "JSON (*.json);;All files (*)",
        )
        if not log_path:
            return
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Session log",
                f"Could not load session file:\n{e}",
            )
            return
        run_fn = getattr(self.parent_window, "run_workflow_from_session", None)
        if callable(run_fn):
            run_fn(data)
        else:
            QMessageBox.warning(
                self,
                "Re-run analysis",
                "Workflow replay is not available.",
            )

    def handle_dropped_path(self, dropped_path):
        """Handle dropped file or folder path"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        
        # Check if it's a file or folder and load accordingly
        if path_obj.is_file():
            # Single file
            self.load_file_info(dropped_path)
        elif path_obj.is_dir():
            # Folder - load folder info
            folder = path_obj
            supported_files = sorted(
                list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
            )
            if supported_files:
                self.load_folder_info(str(folder), supported_files)
            else:
                self.metadata_text.setText("No supported microscopy files (ND2, LIF, CZI, TIFF) found in dropped folder")
                self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 12pt;")
    
    def _rebuild_channel_options(self, channel_names, num_channels):
        """Populate Channel options panel: one row per channel with FI / Foci / Z-channel / Bleach correct checkboxes. Shown when num_channels > 1."""
        self.fi_channel_checks.clear()
        self.foci_channel_checks.clear()
        self.zproject_channel_checks.clear()
        self.bleach_channel_checks.clear()
        while self.channel_options_layout.count():
            item = self.channel_options_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if num_channels <= 1:
            self.channel_options_group.setVisible(False)
            return
        info_label = QLabel(
            "Tick per channel: FI adds \"_fi\" in filename (Analyze Masks); Foci adds \"_foci\" for foci analysis; "
            "Z-channel = max Z-projection; Bleach correct = photobleaching correction."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #a0a0a0; font-size: 10pt;")
        self.channel_options_layout.addWidget(info_label)
        self.channel_options_layout.addSpacing(12)
        for i, name in enumerate(channel_names):
            row_widget = QWidget()
            row = QHBoxLayout()
            row.setContentsMargins(0, 2, 0, 2)
            row_widget.setLayout(row)
            ch_label = QLabel(f"Channel {i+1}: {name}")
            ch_label.setFixedWidth(200)
            row.addWidget(ch_label)
            row.addSpacing(12)
            fi_cb = QCheckBox("FI")
            fi_cb.setToolTip("Save with \"_fi\" in filename (e.g. stem_fi.tif) for Analyze Masks batch")
            foci_cb = QCheckBox("Foci")
            foci_cb.setToolTip("Save with \"_foci\" in filename for Analyze Masks foci analysis")
            z_cb = QCheckBox("Z-channel")
            z_cb.setToolTip("Apply max Z-projection for this channel (when Z stack > 1)")
            bleach_cb = QCheckBox("Bleach correct")
            bleach_cb.setToolTip("Apply bleach correction to this channel")
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
        if not getattr(self, "zproject_channel_checks", None):
            return
        any_checked = any(cb.isChecked() for cb in self.zproject_channel_checks)
        if getattr(self, "max_z_projection", None) is not None:
            self.max_z_projection.setChecked(any_checked)

    def _sync_bleach_from_channels(self):
        """When any Bleach correct is ticked, enable Apply photobleaching correction; when all unticked, disable it."""
        if not getattr(self, "bleach_channel_checks", None):
            return
        any_checked = any(cb.isChecked() for cb in self.bleach_channel_checks)
        if getattr(self, "bleach_correct", None) is not None:
            self.bleach_correct.setChecked(any_checked)

    def load_folder_info(self, folder_path, supported_files):
        """Load and display information for all files in a folder"""
        folder = Path(folder_path)
        all_files_info = []
        errors = []
        
        self.parent_window.log_output(f"Scanning folder: {folder_path}\n")
        self.parent_window.log_output(f"Found {len(supported_files)} file(s) to analyze\n")
        
        # Load info from each file
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
                    errors.append(f"{file_path.name}: Could not read file info")
                    
            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")
                self.parent_window.log_output(f"✗ {file_path.name}: Error - {str(e)}\n")
        
        # Display summary of all files
        if all_files_info:
            self._display_folder_summary(all_files_info, errors)
            # Store info from first file for conversion parameters (use first as default)
            first_file_info = all_files_info[0][1]
            self.file_info = {
                'path': folder_path,
                'file_type': first_file_info['file_type'],
                'num_time': first_file_info['num_time'],
                'num_channels': first_file_info['num_channels'],
                'num_z': first_file_info['num_z'],
                'channel_names': first_file_info['channel_names'],
                'pixel_size': first_file_info.get('pixel_size'),
                'time_per_frame_min': first_file_info.get('time_per_frame_min'),
            }
            # Update UI with first file's parameters (user can adjust)
            n_t = first_file_info['num_time']
            self.max_frames.setMaximum(n_t)
            self.max_frames.setValue(n_t)  # Automatically set to maximum
            self.single_frame_spin.setMaximum(max(1, n_t))
            if self.single_frame_spin.value() > n_t:
                self.single_frame_spin.setValue(n_t)
            # Channel options panel (FI / Z / Bleach per channel)
            self._rebuild_channel_options(first_file_info['channel_names'], first_file_info['num_channels'])
            self.preview_btn.setEnabled(True)
            self.convert_btn.setEnabled(True)
            
            # Push pixel size and time per frame to shared metadata (used by Tracking, Analyze Masks, etc.)
            if first_file_info.get('pixel_size'):
                self.parent_window.shared_metadata['pixel_size'] = first_file_info['pixel_size']
            else:
                self.parent_window.shared_metadata['pixel_size'] = None
            if first_file_info.get('time_per_frame_min'):
                self.parent_window.shared_metadata['time_per_frame_min'] = first_file_info['time_per_frame_min']
            else:
                self.parent_window.shared_metadata['time_per_frame_min'] = None
            self.parent_window.update_all_tabs_with_metadata()
        else:
            self.metadata_text.setText("No files could be read from the selected folder.\n" + 
                                     "\n".join(errors) if errors else "")
            self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ff6b6b; font-size: 12pt;")
    
    def _display_folder_summary(self, all_files_info, errors):
        """Display a summary of all files in the folder"""
        metadata_content = f"Folder contains {len(all_files_info)} file(s):\n"
        metadata_content += "=" * 60 + "\n\n"
        
        for i, (filename, info) in enumerate(all_files_info, 1):
            metadata_content += f"File {i}: {filename}\n"
            metadata_content += f"  Type: {info['file_type'].upper()}\n"
            metadata_content += f"  Dimensions: T={info['num_time']} frames, C={info['num_channels']} channels, Z={info['num_z']} slices\n"
            if info['num_channels'] > 1:
                metadata_content += f"  Channels: {', '.join(info['channel_names'])}\n"
            if info['pixel_size']:
                metadata_content += f"  Pixel Size: {info['pixel_size']:.4f} µm\n"
            else:
                metadata_content += f"  Pixel Size: Not found in metadata\n"
            if info.get('time_per_frame_min') is not None and info['time_per_frame_min'] > 0:
                tp = info['time_per_frame_min']
                metadata_content += f"  Time per frame: {tp:.6f} min ({tp * 60:.4f} s)\n"
            else:
                metadata_content += f"  Time per frame: Not found in metadata\n"
            
            # Show differences if files have different properties
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
            
            metadata_content += "\n"
        
        if errors:
            metadata_content += f"\nErrors ({len(errors)} file(s)):\n"
            for error in errors:
                metadata_content += f"  ✗ {error}\n"
        
        self.metadata_text.setText(metadata_content)
        self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 10pt;")
    
    def _get_nd2_info(self, file_path):
        """Extract ND2 file info without updating UI (for folder scanning). Tries BioIO first, then tiff_metadata."""
        info = bioio_get_info(Path(file_path))
        if info and info.get("file_type") == "nd2":
            out = {k: v for k, v in info.items() if k in ("file_type", "num_time", "num_channels", "num_z", "channel_names", "pixel_size", "scenes")}
            # BioIO does not provide time_per_frame, objective, wavelengths, exposure; get full ND2 metadata and merge
            try:
                from bactimate.modules.tiff_metadata import read_metadata_from_path
                meta = read_metadata_from_path(Path(file_path))
                if meta.get("time_per_frame_min") is not None and float(meta["time_per_frame_min"]) > 0:
                    out["time_per_frame_min"] = float(meta["time_per_frame_min"])
                for key in ("channel_wavelengths", "objective_name", "objective_magnification", "zoom_magnification",
                            "exposure_ms", "channel_exposure_ms", "bits_per_component"):
                    if meta.get(key) is not None:
                        out[key] = meta[key]
            except Exception:
                pass
            return out
        try:
            from bactimate.modules.tiff_metadata import read_metadata_from_path
            meta = read_metadata_from_path(Path(file_path))
            num_time = meta.get("frame_count") or 1
            num_channels = meta.get("channel_count") or 1
            num_z = meta.get("z_count") or 1
            channel_names = meta.get("channel_names") or []
            if len(channel_names) != num_channels:
                channel_names = [f"Channel_{i+1}" for i in range(num_channels)]
            return {
                "file_type": "nd2",
                "num_time": num_time,
                "num_channels": num_channels,
                "num_z": num_z,
                "channel_names": channel_names,
                "pixel_size": meta.get("pixel_size_um"),
                "time_per_frame_min": meta.get("time_per_frame_min"),
            }
        except Exception:
            return None
    
    @staticmethod
    def _safe_lif_pixel_size(reader):
        """Get pixel size from LIF reader without assuming physical_pixel_sizes shape (avoids dimension errors)."""
        try:
            pps = getattr(reader, "physical_pixel_sizes", None)
            if pps is None:
                return None
            for attr in ("X", "Y", "Z"):
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
        n = getattr(lif_file, "num_images", 0)
        if n <= 0:
            return 0
        for i in range(n):
            try:
                img = lif_file.get_image(i)
                info = getattr(img, "info", None) or {}
                dims = info.get("dims") or getattr(img, "dims", None)
                dims_n = getattr(img, "dims_n", None) or {}
                x = 1
                y = 1
                if dims is not None:
                    x = int(getattr(dims, "x", 1) or 1)
                    y = int(getattr(dims, "y", 1) or 1)
                if x <= 1 or y <= 1:
                    x = int(dims_n.get(1, x) or x)
                    y = int(dims_n.get(2, y) or y)
                if x <= 1 or y <= 1:
                    try:
                        frame = img.get_frame(z=0, t=0, c=0)
                        if hasattr(frame, "shape") and len(frame.shape) >= 2:
                            y, x = frame.shape[0], frame.shape[1]
                    except Exception:
                        pass
                if x > 1 and y > 1:
                    if log_cb and i > 0:
                        log_cb(f"readlif: using image index {i} (XY={x}x{y}); index 0 looked like metadata/sensor.\n")
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
        num_time, num_z, num_channels = 1, 1, 1
        info = getattr(image, "info", None)
        dims_n = getattr(image, "dims_n", None) or {}
        dims = getattr(image, "dims", None)
        if log_cb:
            log_cb(f"readlif raw: info={info!r}\ndims_n={dims_n} dims={dims} nt={getattr(image,'nt',None)} nz={getattr(image,'nz',None)}\n")
        try:
            if isinstance(info, dict):
                ch = info.get("channels")
                if ch is not None and int(ch) >= 1:
                    num_channels = int(ch)
                d = info.get("dims")
                if d is not None:
                    num_time = int(getattr(d, "t", 1) or 1)
                    num_z = int(getattr(d, "z", 1) or 1)
        except (AttributeError, TypeError, KeyError):
            pass
        if num_time <= 0 or num_z <= 0 or (isinstance(info, dict) and info.get("dims") is None):
            try:
                num_time = int(getattr(dims, "t", 1) or dims_n.get(4, 1))
                num_z = int(getattr(dims, "z", 1) or dims_n.get(3, 1))
                num_channels = int(getattr(dims, "c", None) or dims_n.get(5, 1))
            except (AttributeError, TypeError):
                pass
        # readlif: 4=time, 5=channels. Some LIF files have both in dims_n with (2,50) or (50,2) order.
        v4 = int(dims_n.get(4, 0) or 0)
        v5 = int(dims_n.get(5, 0) or 0)
        if v4 >= 1 and v5 >= 1:
            low, high = min(v4, v5), max(v4, v5)
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
            if info and info.get("file_type") == "lif":
                return {k: v for k, v in info.items() if k in ("file_type", "num_time", "num_channels", "num_z", "channel_names", "pixel_size", "scenes")}
        except Exception:
            pass  # BioIO/bioio-lif can raise e.g. "too many values to unpack" on some LIF files
        try:
            import numpy as np
            from aicsimageio import AICSImage
            # AICSImage doesn't support context manager protocol, use direct instantiation
            lif_reader = AICSImage(file_path)
            try:
                # Get dimensions - handle both old and new aicsimageio API
                dims = lif_reader.dims
                
                # Try to get dimension order as string
                if hasattr(dims, 'order'):
                    dims_str = dims.order
                elif isinstance(dims, str):
                    dims_str = dims
                elif isinstance(dims, (list, tuple)):
                    dims_str = ''.join(str(d) for d in dims)
                else:
                    dims_str = str(dims)
                
                # Get shape - try different methods depending on aicsimageio version
                try:
                    # Try accessing as dictionary-like (newer versions)
                    if hasattr(lif_reader, 'get_xarray_data'):
                        xdata = lif_reader.get_xarray_data()
                        num_time = int(xdata.sizes.get('T', 1))
                        num_channels = int(xdata.sizes.get('C', 1))
                        num_z = int(xdata.sizes.get('Z', 1))
                    else:
                        # Fallback to shape attribute
                        shape = lif_reader.shape
                        # Convert shape to tuple/list if needed
                        if not isinstance(shape, (tuple, list)):
                            shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                        
                        # Access dimensions by index in dims string
                        num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                        num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                        num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                except (KeyError, IndexError, AttributeError, TypeError) as e:
                    # If dimension access fails, try to get from shape directly
                    shape = lif_reader.shape
                    if isinstance(shape, dict):
                        # Shape is a dictionary - access by dimension name
                        num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                        num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                        num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                    else:
                        # Fallback: assume standard order and try indexing
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
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                pixel_size = self._safe_lif_pixel_size(lif_reader)
                
                return {
                    'file_type': 'lif',
                    'num_time': num_time,
                    'num_channels': num_channels,
                    'num_z': num_z,
                    'channel_names': channel_names,
                    'pixel_size': pixel_size
                }
            finally:
                # Clean up if needed (AICSImage may have resources to close)
                if hasattr(lif_reader, 'close'):
                    try:
                        lif_reader.close()
                    except:
                        pass
        except Exception:
            pass
        # Fallback: readlif (use microscopy series: Leica often has Image 0 = Job/sensor, Image 1 = Series001)
        try:
            from readlif.reader import LifFile
            lif_file = LifFile(file_path)
            if getattr(lif_file, "num_images", 0) <= 0:
                return None
            best_idx = self._readlif_best_image_index(lif_file)
            image = lif_file.get_image(best_idx)
            num_time, num_z, num_channels = self._readlif_dims(image)
            pixel_size = None
            if hasattr(image, "scale") and image.scale and len(image.scale) >= 2:
                try:
                    sx = float(image.scale[0])
                    if sx > 0:
                        pixel_size = 1.0 / sx
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            return {
                "file_type": "lif",
                "num_time": num_time,
                "num_channels": num_channels,
                "num_z": num_z,
                "channel_names": [f"Channel_{i+1}" for i in range(num_channels)],
                "pixel_size": pixel_size,
            }
        except Exception:
            return None
    
    def _get_tiff_info(self, file_path):
        """Extract TIFF file info without updating UI (for folder scanning). Tries BioIO first, then tifffile."""
        info = bioio_get_info(Path(file_path))
        if info and info.get("file_type") == "tiff":
            return {k: v for k, v in info.items() if k in ("file_type", "num_time", "num_channels", "num_z", "channel_names", "pixel_size", "scenes")}
        try:
            import tifffile
            import numpy as np
            
            # Read TIFF file to get dimensions
            with tifffile.TiffFile(file_path) as tif:
                # Get shape from first page
                first_page = tif.pages[0]
                shape = first_page.shape
                
                # Try to get metadata
                metadata = {}
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                elif hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    # OME metadata parsing would be more complex
                    pass
                
                # Determine dimensions from shape
                # Common formats: (T, Y, X), (T, C, Y, X), (C, Y, X), (Y, X), (T, Z, C, Y, X)
                ndim = len(shape)
                
                if ndim == 2:
                    # Single 2D image
                    num_time = 1
                    num_channels = 1
                    num_z = 1
                elif ndim == 3:
                    # Could be (T, Y, X) or (C, Y, X) or (Z, Y, X)
                    # Check if first dimension is small (likely channels or z-slices)
                    if shape[0] < 10:
                        num_time = 1
                        num_channels = shape[0]
                        num_z = 1
                    else:
                        # Likely (T, Y, X)
                        num_time = shape[0]
                        num_channels = 1
                        num_z = 1
                elif ndim == 4:
                    # Could be (T, C, Y, X) or (T, Z, Y, X) or (Z, C, Y, X)
                    if shape[1] < 10:
                        # Likely (T, C, Y, X)
                        num_time = shape[0]
                        num_channels = shape[1]
                        num_z = 1
                    elif shape[0] < 10:
                        # Likely (Z, C, Y, X)
                        num_time = 1
                        num_channels = shape[1]
                        num_z = shape[0]
                    else:
                        # Assume (T, Z, Y, X)
                        num_time = shape[0]
                        num_channels = 1
                        num_z = shape[1]
                else:
                    # 5D or more - assume (T, Z, C, Y, X)
                    num_time = shape[0] if len(shape) > 0 else 1
                    num_z = shape[1] if len(shape) > 1 else 1
                    num_channels = shape[2] if len(shape) > 2 else 1
                
                # Check if multi-page TIFF (time series or multi-channel)
                num_pages = len(tif.pages)
                if num_pages > 1 and ndim == 2:
                    # Prefer ImageJ metadata for channels (e.g. "ImageJ=1.51 images=16 channels=2 slices=8")
                    ij_meta = getattr(tif, "imagej_metadata", None) or {}
                    ij_channels = ij_meta.get("channels") if isinstance(ij_meta, dict) else None
                    ij_frames = ij_meta.get("frames") or ij_meta.get("images")
                    if ij_channels is not None and int(ij_channels) >= 2:
                        num_channels = int(ij_channels)
                        num_time = (num_pages // num_channels) if num_channels else num_pages
                        num_z = 1
                        if ij_frames is not None:
                            num_time = int(ij_frames)
                    else:
                        # Multi-page 2D without channel info - treat as time series
                        num_time = num_pages
                        num_channels = 1
                        num_z = 1
                
                # Get pixel size and channel names from metadata (ImageJ, OME, spacing=)
                pixel_size = None
                if 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                elif 'unit' in metadata and 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
            # Prefer tiff_metadata (read_metadata_from_path) for channel names and pixel size
            try:
                from bactimate.modules.tiff_metadata import read_metadata_from_path
                meta = read_metadata_from_path(Path(file_path))
                if meta.get("channel_names") and len(meta["channel_names"]) == num_channels:
                    channel_names = list(meta["channel_names"])
                if meta.get("pixel_size_um") is not None and float(meta["pixel_size_um"]) > 0:
                    pixel_size = float(meta["pixel_size_um"])
            except Exception:
                pass
            return {
                'file_type': 'tiff',
                'num_time': num_time,
                'num_channels': num_channels,
                'num_z': num_z,
                'channel_names': channel_names,
                'pixel_size': pixel_size
            }
        except Exception:
            return None
    
    def _czi_dims_from_axes(self, axes, shape):
        """Return (num_time, num_channels, num_z) from czifile axes string and shape.
        CZI dimension order is file-dependent (e.g. TZCYX, TCZYX, BSTCZYX0); we read T,C,Z from axes."""
        if not axes or not shape or len(axes) != len(shape):
            return 1, 1, 1
        try:
            num_time = int(shape[axes.index("T")]) if "T" in axes else 1
            num_channels = int(shape[axes.index("C")]) if "C" in axes else 1
            num_z = int(shape[axes.index("Z")]) if "Z" in axes else 1
            return num_time, num_channels, num_z
        except (ValueError, IndexError, TypeError):
            return 1, 1, 1

    @staticmethod
    def _czi_pixel_size_from_metadata(czi_file):
        """Extract pixel size (µm) from CZI metadata XML. Returns None if not found."""
        try:
            meta = czi_file.metadata(raw=True) if callable(getattr(czi_file, "metadata", None)) else getattr(czi_file, "metadata", None)
            if not meta or not isinstance(meta, str):
                return None
            import xml.etree.ElementTree as ET
            root = ET.fromstring(meta)
            # CZI uses namespaces; match by local name
            for elem in root.iter():
                tag = (elem.tag or "").split("}")[-1]
                if tag != "Distance":
                    continue
                did = (elem.get("Id") or elem.get("id") or "").strip()
                if did != "X":
                    continue
                val_el = elem.find("Value")
                if val_el is None:
                    val_el = next((c for c in elem if (c.tag or "").split("}")[-1] == "Value"), None)
                if val_el is not None and getattr(val_el, "text", None):
                    try:
                        scale_x = float(val_el.text.strip())
                        if scale_x > 0 and scale_x < 1e-4:
                            scale_x *= 1e6
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
        for i, ax in enumerate(axes):
            if ax == "C" and arr.shape[i] > 0:
                ci = min(int(c_index), arr.shape[i] - 1) if c_index is not None else 0
                indices.append(max(0, ci))
            elif ax == "Z" and arr.shape[i] > 0:
                zi = min(int(z_index), arr.shape[i] - 1) if z_index is not None else 0
                indices.append(max(0, zi))
            else:
                indices.append(slice(None))
        out = arr[tuple(indices)]
        # Which axes remain (in order)? We dropped C and Z.
        remaining_axes = [axes[i] for i in range(len(axes)) if axes[i] not in ("C", "Z")]
        # Transpose to (T, Y, X) so the preview always gets [frame_idx, :, :] = (Y, X), never a thin strip.
        if len(remaining_axes) >= 3 and "T" in remaining_axes and "Y" in remaining_axes and "X" in remaining_axes:
            t_pos = remaining_axes.index("T")
            y_pos = remaining_axes.index("Y")
            x_pos = remaining_axes.index("X")
            # Put T first, Y second, X third; any other dims (S, R, etc.) at end for squeezing
            other = [i for i in range(len(remaining_axes)) if remaining_axes[i] not in ("T", "Y", "X")]
            perm = [t_pos, y_pos, x_pos] + other
            out = np.transpose(out, perm)
        # Remove trailing sample dim (e.g. RGB)
        while out.ndim > 3 and out.shape[-1] in (1, 3, 4):
            out = out[..., 0]
        # Squeeze only non-leading singleton dims so we keep (T, Y, X) with T>=1
        while out.ndim > 3:
            squeezed = out.squeeze()
            if squeezed.ndim < 3:
                break
            out = squeezed
        # Ensure at least (1, Y, X)
        if out.ndim == 2:
            out = out[np.newaxis, ...]
        return out

    @staticmethod
    def _czi_find_first_nonempty_plane(arr, axes):
        """Find the (channel, z) plane that is best for preview: first plane with any pixel > 0.
        If that would be too dim, uses the plane with maximum total intensity (most informative slice).
        Loops over Z then C. Returns (c_index, z_index); (0, 0) if none or no C/Z in axes."""
        if not axes or len(axes) != len(arr.shape):
            return 0, 0
        c_axis = axes.index("C") if "C" in axes else -1
        z_axis = axes.index("Z") if "Z" in axes else -1
        if c_axis < 0 and z_axis < 0:
            return 0, 0
        n_c = arr.shape[c_axis] if c_axis >= 0 else 1
        n_z = arr.shape[z_axis] if z_axis >= 0 else 1
        best_c, best_z = 0, 0
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
                    best_c, best_z = c, z
        return best_c, best_z

    @staticmethod
    def _czi_slice_to_t_c_y_x(arr, axes, max_frames=None, frame_start=None, frame_end=None):
        """Reduce CZI array to (T, C, Y, X) by taking first Z only. axes matches arr.shape.
        Returns (array, dims_str). Limits T: use [frame_start:frame_end) if frame_start is set, else first max_frames."""
        if not axes or len(axes) != len(arr.shape):
            return arr, ""
        indices = []
        t_idx = axes.index("T") if "T" in axes else None
        for i, ax in enumerate(axes):
            if ax == "Z" and arr.shape[i] > 0:
                indices.append(0)
            elif ax == "T" and t_idx is not None:
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
        dims_str = "".join(ax for ax in axes if ax not in ("Z", "0"))
        return out, dims_str if dims_str else "TCYX"

    @staticmethod
    def _czi_slice_to_t_c_z_y_x(arr, axes, max_frames=None, frame_start=None, frame_end=None):
        """Reduce CZI array to (T, C, Z, Y, X) keeping Z dimension. axes matches arr.shape.
        Returns (array, dims_str, z_axis). Limits T: use [frame_start:frame_end) if frame_start is set, else first max_frames."""
        if not axes or len(axes) != len(arr.shape):
            return arr, "", -1
        indices = []
        t_idx = axes.index("T") if "T" in axes else None
        z_idx_in_axes = axes.index("Z") if "Z" in axes else -1
        for i, ax in enumerate(axes):
            if ax == "T" and t_idx is not None:
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
        dims_str = "".join(ax for ax in axes if ax not in ("0",))
        if "Z" not in dims_str:
            return out, dims_str, -1
        z_axis = dims_str.index("Z")
        return out, dims_str, z_axis

    def _get_czi_info_via_czifile(self, file_path):
        """Extract CZI file info using the czifile library (https://pypi.org/project/czifile/).
        Uses the file's axes string so T, C, Z are interpreted correctly regardless of dimension order."""
        try:
            import czifile
            with czifile.CziFile(file_path) as czi_file:
                shape = getattr(czi_file, "shape", None)
                axes = getattr(czi_file, "axes", None)
                if shape and axes and len(axes) == len(shape):
                    num_time, num_channels, num_z = self._czi_dims_from_axes(axes, shape)
                else:
                    num_time, num_channels, num_z = 1, 1, 1
                    if shape and len(shape) >= 3:
                        num_time = int(shape[0])
                        num_channels = int(shape[1]) if len(shape) > 1 else 1
                        num_z = int(shape[2]) if len(shape) > 2 else 1
                channel_names = [f"Channel_{i+1}" for i in range(num_channels)]
                pixel_size = self._czi_pixel_size_from_metadata(czi_file)
                return {
                    "file_type": "czi",
                    "num_time": num_time,
                    "num_channels": num_channels,
                    "num_z": num_z,
                    "channel_names": channel_names,
                    "pixel_size": pixel_size,
                }
        except Exception:
            return None

    def _get_czi_info(self, file_path):
        """Extract CZI file info without updating UI (for folder scanning). Tries BioIO first, then czifile."""
        info = bioio_get_info(Path(file_path))
        if info and info.get("file_type") == "czi":
            return {k: v for k, v in info.items() if k in ("file_type", "num_time", "num_channels", "num_z", "channel_names", "pixel_size", "scenes")}
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
                QMessageBox.warning(self, "Unsupported Format", 
                                  f"File format '{file_ext}' is not supported.\n"
                                  "Supported formats: ND2, LIF, CZI, TIFF")
                return
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", 
                               f"Could not load file information:\n{str(e)}\n\n{traceback.format_exc()}")
            self.metadata_text.setText(f"Error loading file: {str(e)}")
            self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 10pt;")
    
    def _load_nd2_info(self, file_path):
        """Load and display ND2 file information. All metadata from tiff_metadata (single source of truth)."""
        try:
            from pathlib import Path
            from bactimate.modules.tiff_metadata import read_metadata_from_path, format_metadata_for_display

            self.parent_window.log_output(f"Loading ND2 file info: {file_path}\n")
            meta = read_metadata_from_path(Path(file_path))
            num_time = meta.get("frame_count") or 1
            num_channels = meta.get("channel_count") or 1
            num_z = meta.get("z_count") or 1
            channel_names = meta.get("channel_names") or []
            if len(channel_names) != num_channels:
                channel_names = [f"Channel_{i+1}" for i in range(num_channels)]
            pixel_size = meta.get("pixel_size_um")
            time_per_frame_min = meta.get("time_per_frame_min")
            metadata_str = format_metadata_for_display(meta)
            self._update_file_info_ui(
                file_path, "nd2", num_time, num_channels, num_z,
                channel_names, pixel_size, metadata_str, time_per_frame_min=time_per_frame_min,
            )
            return
        except ImportError:
            QMessageBox.warning(
                self, "ND2 Library Missing",
                "The 'nd2' library is not installed.\n\nInstall it with: pip install nd2",
            )
            return
        except Exception as e:
            QMessageBox.critical(self, "Error Loading ND2", f"Could not load ND2 file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
            return
    def _load_lif_info(self, file_path):
        """Load and display LIF file information. Uses BioIO when available, then aicsimageio/readlif."""
        try:
            self.parent_window.log_output(f"Loading LIF file info: {file_path}\n")
            # Try BioIO first; on any error (e.g. bioio-lif "too many values to unpack") fall back to aicsimageio
            try:
                info = bioio_get_info(Path(file_path))
                if info and info.get("file_type") == "lif":
                    self._update_file_info_ui(
                        file_path, "lif",
                        info["num_time"], info["num_channels"], info["num_z"],
                        info["channel_names"], info.get("pixel_size"),
                    )
                    return
            except Exception:
                # BioIO/bioio-lif can raise e.g. "too many values to unpack (expected 4)" on some LIF files
                pass
            # Fallback: aicsimageio then readlif (aicsimageio can raise "too many values to unpack" on some LIFs)
            aics_success = False
            try:
                import numpy as np
                from aicsimageio import AICSImage
                lif_reader = AICSImage(file_path)
                try:
                    # Get dimensions - handle both old and new aicsimageio API
                    dims = lif_reader.dims
                    
                    # Try to get dimension order as string
                    if hasattr(dims, 'order'):
                        dims_str = dims.order
                    elif isinstance(dims, str):
                        dims_str = dims
                    elif isinstance(dims, (list, tuple)):
                        dims_str = ''.join(str(d) for d in dims)
                    else:
                        dims_str = str(dims)
                    
                    # Get shape - try different methods depending on aicsimageio version
                    try:
                        # Try accessing as dictionary-like (newer versions)
                        if hasattr(lif_reader, 'get_xarray_data'):
                            xdata = lif_reader.get_xarray_data()
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            # Fallback to shape attribute
                            shape = lif_reader.shape
                            # Convert shape to tuple/list if needed
                            if not isinstance(shape, (tuple, list)):
                                shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                            
                            # Access dimensions by index in dims string
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        # If dimension access fails, try to get from shape directly
                        shape = lif_reader.shape
                        if isinstance(shape, dict):
                            # Shape is a dictionary - access by dimension name
                            num_time = shape.get('T', 1) if isinstance(shape.get('T'), (int, np.integer)) else 1
                            num_channels = shape.get('C', 1) if isinstance(shape.get('C'), (int, np.integer)) else 1
                            num_z = shape.get('Z', 1) if isinstance(shape.get('Z'), (int, np.integer)) else 1
                        else:
                            # Fallback: assume standard order and try indexing
                            shape_list = list(shape) if hasattr(shape, '__iter__') else [shape]
                            num_time = shape_list[dims_str.index('T')] if 'T' in dims_str and len(shape_list) > dims_str.index('T') else 1
                            num_channels = shape_list[dims_str.index('C')] if 'C' in dims_str and len(shape_list) > dims_str.index('C') else 1
                            num_z = shape_list[dims_str.index('Z')] if 'Z' in dims_str and len(shape_list) > dims_str.index('Z') else 1
                    
                    # Get channel names if available
                    channel_names = []
                    try:
                        if hasattr(lif_reader, 'channel_names'):
                            channel_names = lif_reader.channel_names
                    except:
                        pass
                    
                    if not channel_names or len(channel_names) != num_channels:
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                    
                    pixel_size = self._safe_lif_pixel_size(lif_reader)
                    
                    self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z, 
                                            channel_names, pixel_size)
                    aics_success = True
                finally:
                    # Clean up if needed (AICSImage may have resources to close)
                    if hasattr(lif_reader, 'close'):
                        try:
                            lif_reader.close()
                        except:
                            pass
            except ImportError:
                pass  # aicsimageio not installed, try readlif
            except Exception:
                # aicsimageio can raise e.g. "too many values to unpack (expected 4)" on some LIF files
                self.parent_window.log_output("AICSImage failed for this LIF, trying readlif...\n")
                pass
            
            if aics_success:
                return
            
            # Fallback to readlif (use microscopy series; Image 0 may be Job/sensor)
            try:
                from readlif.reader import LifFile
                lif_file = LifFile(file_path)
                if getattr(lif_file, "num_images", 0) <= 0:
                    raise ValueError("No image series found in LIF file")
                log_cb = lambda msg: self.parent_window.log_output(msg)
                best_idx = self._readlif_best_image_index(lif_file, log_cb=log_cb)
                image = lif_file.get_image(best_idx)
                num_time, num_z, num_channels = self._readlif_dims(image, log_cb=log_cb)
                channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                pixel_size = None
                if hasattr(image, "scale") and image.scale and len(image.scale) >= 2:
                    try:
                        sx, sy = float(image.scale[0]), float(image.scale[1])
                        if sx > 0:
                            pixel_size = 1.0 / sx
                        elif sy > 0:
                            pixel_size = 1.0 / sy
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z,
                                        channel_names, pixel_size)
            except ImportError:
                QMessageBox.warning(self, "LIF Library Missing",
                                  "The 'aicsimageio' or 'readlif' library is not installed.\n\n"
                                  "Install with: pip install aicsimageio readlif")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading LIF", f"Could not load LIF file:\n{str(e)}")
                self.parent_window.log_output(f"Error: {str(e)}\n")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading LIF", f"Could not load LIF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _load_czi_info(self, file_path):
        """Load and display CZI file information. Uses BioIO when available, then czifile."""
        try:
            self.parent_window.log_output(f"Loading CZI file info: {file_path}\n")
            info = bioio_get_info(Path(file_path))
            if info and info.get("file_type") == "czi":
                self._update_file_info_ui(
                    file_path, "czi",
                    info["num_time"], info["num_channels"], info["num_z"],
                    info["channel_names"], info.get("pixel_size"),
                )
                return
            import czifile
        except ImportError:
            QMessageBox.warning(
                self, "CZI Library Missing",
                "CZI support requires the 'bioio' and 'bioio-czi' packages, or the 'czifile' package.\n\n"
                "Install with: pip install bioio bioio-czi\n"
                "Or fallback: pip install czifile"
            )
            return
        try:
            info = self._get_czi_info_via_czifile(file_path)
            if info is None:
                raise RuntimeError("Could not read CZI file")
            self._update_file_info_ui(
                file_path, "czi",
                info["num_time"], info["num_channels"], info["num_z"],
                info["channel_names"], info["pixel_size"]
            )
        except Exception as e:
            QMessageBox.critical(self, "Error Loading CZI", f"Could not load CZI file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _load_tiff_info(self, file_path):
        """Load and display TIFF file information. Uses BioIO when available, then tifffile; tiff_metadata for calibration and channel names."""
        try:
            from bactimate.modules.tiff_metadata import read_calibration_from_path, read_metadata_from_path
            self.parent_window.log_output(f"Loading TIFF file info: {file_path}\n")
            path = Path(file_path)
            # Try BioIO first for consistent dimension reading
            info = bioio_get_info(path)
            if info and info.get("file_type") == "tiff":
                meta = read_metadata_from_path(path)
                pixel_size = None
                if meta.get("pixel_size_um") is not None and float(meta["pixel_size_um"]) > 0:
                    pixel_size = float(meta["pixel_size_um"])
                if pixel_size is None and info.get("pixel_size") is not None and float(info["pixel_size"]) > 0:
                    pixel_size = float(info["pixel_size"])
                time_per_frame_min = None
                if meta.get("time_per_frame_min") is not None and float(meta.get("time_per_frame_min", 0)) > 0:
                    time_per_frame_min = float(meta["time_per_frame_min"])
                channel_names = meta.get("channel_names") or info.get("channel_names")
                if not channel_names or len(channel_names) != info.get("num_channels", 1):
                    channel_names = info.get("channel_names") or [f"Channel_{i+1}" for i in range(info.get("num_channels", 1))]
                self._update_file_info_ui(
                    file_path, "tiff",
                    info["num_time"], info["num_channels"], info["num_z"],
                    channel_names,
                    pixel_size,
                    time_per_frame_min=time_per_frame_min,
                )
                return
            import tifffile
            import numpy as np
            pixel_size, time_per_frame_min = read_calibration_from_path(path)
            with tifffile.TiffFile(file_path) as tif:
                # Get shape from first page
                first_page = tif.pages[0]
                shape = first_page.shape
                
                # Determine dimensions from shape
                ndim = len(shape)
                num_pages = len(tif.pages)
                
                if ndim == 2:
                    # Single 2D image or multi-page (possibly multi-channel from ImageJ)
                    if num_pages > 1:
                        ij_meta = getattr(tif, "imagej_metadata", None) or {}
                        ij_channels = ij_meta.get("channels") if isinstance(ij_meta, dict) else None
                        ij_frames = ij_meta.get("frames") or ij_meta.get("images")
                        if ij_channels is not None and int(ij_channels) >= 2:
                            num_channels = int(ij_channels)
                            num_time = (num_pages // num_channels) if num_channels else num_pages
                            num_z = 1
                            if ij_frames is not None:
                                num_time = int(ij_frames)
                        else:
                            num_time = num_pages
                            num_channels = 1
                            num_z = 1
                    else:
                        num_time = 1
                        num_channels = 1
                        num_z = 1
                elif ndim == 3:
                    # Could be (T, Y, X) or (C, Y, X) or (Z, Y, X)
                    if shape[0] < 10:
                        num_time = 1
                        num_channels = shape[0]
                        num_z = 1
                    else:
                        num_time = shape[0]
                        num_channels = 1
                        num_z = 1
                elif ndim == 4:
                    # Could be (T, C, Y, X) or (T, Z, Y, X) or (Z, C, Y, X)
                    if shape[1] < 10:
                        num_time = shape[0]
                        num_channels = shape[1]
                        num_z = 1
                    elif shape[0] < 10:
                        num_time = 1
                        num_channels = shape[1]
                        num_z = shape[0]
                    else:
                        num_time = shape[0]
                        num_channels = 1
                        num_z = shape[1]
                else:
                    # 5D or more - assume (T, Z, C, Y, X)
                    num_time = shape[0] if len(shape) > 0 else 1
                    num_z = shape[1] if len(shape) > 1 else 1
                    num_channels = shape[2] if len(shape) > 2 else 1
                
                channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                # Prefer channel names and calibration from tiff_metadata (ImageJ Labels, OME, spacing=, etc.)
                meta = read_metadata_from_path(path)
                if meta.get("channel_names") and len(meta["channel_names"]) == num_channels:
                    channel_names = list(meta["channel_names"])
                if meta.get("pixel_size_um") and float(meta["pixel_size_um"]) > 0:
                    pixel_size = float(meta["pixel_size_um"])
                if meta.get("time_per_frame_min") is not None and float(meta.get("time_per_frame_min", 0)) > 0:
                    time_per_frame_min = float(meta["time_per_frame_min"])
                self._update_file_info_ui(
                    file_path, 'tiff', num_time, num_channels, num_z,
                    channel_names, pixel_size if pixel_size > 0 else None,
                    time_per_frame_min=time_per_frame_min if time_per_frame_min > 0 else None
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error Loading TIFF", f"Could not load TIFF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _update_file_info_ui(self, file_path, file_type, num_time, num_channels, num_z,
                            channel_names, pixel_size, metadata_str=None, time_per_frame_min=None):
        """Update UI with file information (common for all file types)."""
        # Store info
        self.file_info = {
            'path': file_path,
            'file_type': file_type,
            'num_time': num_time,
            'num_channels': num_channels,
            'num_z': num_z,
            'channel_names': channel_names,
            'pixel_size': pixel_size,
            'time_per_frame_min': time_per_frame_min,
        }
        
        # Update frame selector - automatically set to max frames
        n_t = num_time
        self.max_frames.setMaximum(n_t)
        self.max_frames.setValue(n_t)  # Automatically set to maximum
        self.single_frame_spin.setMaximum(max(1, n_t))
        if self.single_frame_spin.value() > n_t:
            self.single_frame_spin.setValue(n_t)

        # Channel options panel (FI / Z / Bleach per channel)
        self._rebuild_channel_options(channel_names, num_channels)
        
        # Update metadata display (now includes file name and dimensions)
        metadata_content = f"File: {Path(file_path).name} ({file_type.upper()})\n"
        metadata_content += f"Dimensions: T={num_time} frames, C={num_channels} channels, Z={num_z} slices\n"
        if num_channels > 1:
            metadata_content += f"Channels: {', '.join(channel_names)}\n"
        metadata_content += "\n"
        
        if metadata_str is None:
            metadata_content += f"Metadata:\n  File Type: {file_type.upper()}\n"
            if pixel_size:
                metadata_content += f"  Pixel Size: {pixel_size:.4f} µm\n"
            else:
                metadata_content += "  Pixel Size: Not found in metadata\n"
            if time_per_frame_min is not None and time_per_frame_min > 0:
                metadata_content += f"  Time per frame: {time_per_frame_min:.6f} min ({time_per_frame_min * 60:.4f} s)\n"
            else:
                metadata_content += "  Time per frame: Not found in metadata\n"
        else:
            metadata_content += metadata_str
        
        self.metadata_text.setText(metadata_content)
        self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 10pt;")
        
        self.parent_window.shared_metadata['pixel_size'] = pixel_size if pixel_size else None
        self.parent_window.shared_metadata['time_per_frame_min'] = time_per_frame_min if (time_per_frame_min is not None and time_per_frame_min > 0) else None
        self.parent_window.update_all_tabs_with_metadata()
        
        self.preview_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
        
        self.parent_window.log_output(f"✓ Loaded: {num_time} frames, {num_channels} channels\n")
        if pixel_size:
            self.parent_window.log_output(f"✓ Pixel size extracted: {pixel_size:.4f} µm (updated in all tabs)\n")
    
    def preview_file(self):
        """Preview microscopy file in a dialog with frame navigation"""
        if not self.file_path.text() or not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "No File", "Please select a valid microscopy file first.")
            return
        
        file_path_obj = Path(self.file_path.text())
        
        # Check if it's a folder - if so, show file selection dialog
        if file_path_obj.is_dir():
            folder = file_path_obj
            # Find all supported files in the folder
            supported_files = sorted(
                list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
            )
            
            if not supported_files:
                QMessageBox.warning(self, "No Files Found", 
                                  "No supported microscopy files (ND2, LIF, CZI, TIFF) found in the selected folder.")
                return
            
            # Show file selection dialog
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select File for Preview")
            file_dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Label
            label = QLabel("Select a file to preview:")
            label.setStyleSheet("color: #ffffff; font-size: 11pt; padding: 10px;")
            layout.addWidget(label)
            
            # List widget with file names
            file_list = QListWidget()
            file_list.setStyleSheet(
                "QListWidget {"
                "background-color: #2b2b2b;"
                "color: #ffffff;"
                "font-size: 11pt;"
                "border: 1px solid #555555;"
                "border-radius: 4px;"
                "padding: 5px;"
                "}"
                "QListWidget::item {"
                "padding: 8px;"
                "border-bottom: 1px solid #3a3a3a;"
                "}"
                "QListWidget::item:selected {"
                "background-color: #5D6D7E;"
                "color: #ffffff;"
                "}"
                "QListWidget::item:hover {"
                "background-color: #3a3a3a;"
                "}"
            )
            
            # Add file names to list
            for f in supported_files:
                file_list.addItem(f.name)
            
            # Select first item by default
            if file_list.count() > 0:
                file_list.setCurrentRow(0)
            
            # Enable double-click to accept
            def on_item_double_clicked(item):
                file_dialog.accept()
            
            file_list.itemDoubleClicked.connect(on_item_double_clicked)
            
            layout.addWidget(file_list)
            
            # Buttons
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setStyleSheet(
                "QPushButton {"
                "background-color: #5D6D7E;"
                "color: white;"
                "font-weight: bold;"
                "padding: 6px 12px;"
                "border: none;"
                "border-radius: 4px;"
                "}"
                "QPushButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            cancel_btn.clicked.connect(file_dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            ok_btn = QPushButton("Preview")
            ok_btn.setStyleSheet(
                "QPushButton {"
                "background-color: #2196F3;"
                "color: white;"
                "font-weight: bold;"
                "padding: 6px 12px;"
                "border: none;"
                "border-radius: 4px;"
                "}"
                "QPushButton:hover {"
                "background-color: #1976D2;"
                "}"
            )
            ok_btn.clicked.connect(file_dialog.accept)
            button_layout.addWidget(ok_btn)
            
            layout.addLayout(button_layout)
            file_dialog.setLayout(layout)
            
            # Show dialog and get selected file
            if file_dialog.exec() == QDialog.DialogCode.Accepted:
                current_item = file_list.currentItem()
                if current_item:
                    selected_file_name = current_item.text()
                    selected_file = folder / selected_file_name
                    
                    # Load info for the selected file (so we know file_type, channels, etc.) but do NOT change the path
                    original_path = self.file_path.text()
                    self.file_path.setText(str(selected_file))
                    self.load_file_info(str(selected_file))
                    
                    if self.file_info:
                        file_type = self.file_info.get('file_type', 'nd2')
                        if file_type == 'nd2':
                            self._preview_nd2()
                        elif file_type == 'lif':
                            self._preview_lif()
                        elif file_type == 'czi':
                            self._preview_czi()
                        elif file_type == 'tiff':
                            self._preview_tiff()
                        else:
                            QMessageBox.warning(self, "Unsupported Format", f"Preview not yet implemented for {file_type} files.")
                    
                    # Restore path to folder so Convert still runs on the folder (preview choice must not change path)
                    self.file_path.setText(original_path)
            return
        
        # Single file mode - check if file info is loaded
        if not self.file_info:
            QMessageBox.warning(self, "File Not Loaded", "Please load file information first.")
            return
        
        file_type = self.file_info.get('file_type', 'nd2')
        
        if file_type == 'nd2':
            self._preview_nd2()
        elif file_type == 'lif':
            self._preview_lif()
        elif file_type == 'czi':
            self._preview_czi()
        elif file_type == 'tiff':
            self._preview_tiff()
        else:
            QMessageBox.warning(self, "Unsupported Format", f"Preview not yet implemented for {file_type} files.")
    
    def _get_bleach_preview_kwargs(self):
        """Return kwargs for preview bleach correction (from current tab settings)."""
        bleach_apply = self.bleach_correct.isChecked()
        bleach_method = ['exponential', 'linear', 'histogram_matching', 'per_frame_minmax'][
            self.bleach_method_combo.currentIndex()
        ]
        bleach_channel_indices = None
        if getattr(self, 'bleach_channel_checks', None):
            indices = [i for i, cb in enumerate(self.bleach_channel_checks) if cb.isChecked()]
            if indices and len(indices) < len(self.bleach_channel_checks):
                bleach_channel_indices = indices
        return dict(bleach_apply=bleach_apply, bleach_method=bleach_method, bleach_channel_indices=bleach_channel_indices)
    
    def _create_unified_preview_ui(
        self, preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
        bleach_apply=False, bleach_method='exponential', bleach_channel_indices=None,
    ):
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

        # Ensure we own a contiguous copy so we don't hold references to reader memory (e.g. ND2/CZI
        # after reader.close()). Otherwise later .copy() (e.g. in bleach correction) can segfault.
        all_frames = np.ascontiguousarray(all_frames)
        if not all_frames.flags.owndata:
            all_frames = all_frames.copy()
        raw_frames = all_frames
        display_frames_ref = {'data': all_frames}  # update_frame reads from this; we can swap to corrected

        # Title (same style as Segmentation Preview)
        title_label = QLabel(f"Timepoint 1 of {num_time}  |  {Path(file_path).name}")
        title_label.setStyleSheet("font-weight: bold; color: #ddd;")
        layout.addWidget(title_label)

        # Qt-native image canvas (same as Segmentation Preview: zoom wheel, pan left/middle-drag)
        image_canvas = ImageCanvas()
        image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        image_canvas.setMinimumSize(400, 300)
        image_canvas.setStyleSheet("background-color: #1e1e1e;")
        image_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(image_canvas, 1)

        # Frame navigation (same layout pattern as Segmentation Preview)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Frame:"))
        frame_spin = QSpinBox()
        frame_spin.setRange(1, num_time)
        frame_spin.setValue(1)
        frame_spin.setToolTip(f"Frame 1–{num_time}. Updates after a short pause or press Enter to jump immediately.")
        nav_layout.addWidget(frame_spin)
        nav_layout.addWidget(QLabel(f"of {num_time}"))
        
        # Channel selection (if multiple channels)
        if num_channels > 1:
            nav_layout.addWidget(QLabel("Channel:"))
            channel_spin = QSpinBox()
            channel_spin.setRange(1, num_channels)
            channel_spin.setValue(1)
            channel_spin.setToolTip(f"Channel 1–{num_channels}. Updates after a short pause or press Enter to switch immediately.")
            nav_layout.addWidget(channel_spin)
        else:
            channel_spin = None
        
        # Show bleach correction in preview whenever there are multiple frames (so user can preview effect)
        show_bleach_check = None
        bleach_status_label = None  # "View: Raw" or "View: Bleach corrected (Method)" - updated on toggle
        _bleach_method_display = {
            'exponential': 'Exponential',
            'linear': 'Linear',
            'histogram_matching': 'Histogram matching',
            'per_frame_minmax': 'Per-frame min-max',
        }
        method_display_name = _bleach_method_display.get(bleach_method, bleach_method)
        if num_time > 1:
            show_bleach_check = QCheckBox("Show bleach correction")
            show_bleach_check.setToolTip(
                "Preview the bleach correction effect. Uses the same method and fluorescence channel "
                "settings as on the conversion tab (or defaults). Check \"Apply photobleaching correction\" "
                "and tick fluorescence channels on the tab to match what will happen on Convert to TIFF."
            )
            show_bleach_check.setChecked(False)
            nav_layout.addWidget(show_bleach_check)
            bleach_status_label = QLabel("View: Raw")
            bleach_status_label.setStyleSheet("color: #9E9E9E; font-size: 10pt;")
            bleach_status_label.setToolTip("Current preview: raw data or bleach-corrected. Changing frame updates the image; correction is applied to all frames.")
            nav_layout.addWidget(bleach_status_label)
        
        nav_layout.addStretch()
        
        # Frame info label (will include View: Raw / View: Bleach corrected when we update it)
        frame_info = QLabel(f"Timepoint 1 of {num_time} | Channel: 1 | Z-slice: 1")
        frame_info.setStyleSheet("font-weight: bold; color: #2196F3;")
        nav_layout.addWidget(frame_info)
        
        layout.addLayout(nav_layout)
        
        # LUT controls with histogram - IMPROVED LAYOUT
        lut_group = QGroupBox("LUT (Lookup Table) Controls")
        lut_layout = QVBoxLayout()
        lut_layout.setSpacing(8)  # Reduce spacing between elements
        
        # Histogram plot - optimized size and appearance
        hist_fig = Figure(figsize=(10, 3.0))  # Increased height for better visibility
        hist_canvas = FigureCanvas(hist_fig)
        hist_canvas.setMinimumHeight(180)  # Increased to prevent cutoff
        hist_canvas.setMaximumHeight(180)  # Increased to prevent cutoff
        hist_ax = hist_fig.add_subplot(111)
        hist_fig.patch.set_facecolor('#2b2b2b')  # Dark background
        hist_ax.set_facecolor('#2b2b2b')
        hist_ax.tick_params(colors='white', labelsize=9)  # Slightly reduced for cleaner look
        hist_ax.spines['bottom'].set_color('white')
        hist_ax.spines['top'].set_color('white')
        hist_ax.spines['left'].set_color('white')
        hist_ax.spines['right'].set_color('white')
        hist_ax.xaxis.label.set_color('white')
        hist_ax.yaxis.label.set_color('white')
        # Set subplot parameters to ensure nothing gets cut off
        hist_fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
        hist_canvas.setStyleSheet("background-color: #2b2b2b;")
        lut_layout.addWidget(hist_canvas)

        # LUT min/max kept as internal widgets (histogram drag + code still use them); not shown in UI
        lut_min_slider = QDoubleSpinBox()
        lut_min_slider.setRange(0, 65535)
        lut_min_slider.setDecimals(1)
        lut_min_slider.setSingleStep(10)
        lut_max_slider = QDoubleSpinBox()
        lut_max_slider.setRange(0, 65535)
        lut_max_slider.setDecimals(1)
        lut_max_slider.setSingleStep(10)

        lut_group.setLayout(lut_layout)
        lut_group.setMaximumHeight(280)  # Increased to match new histogram height
        layout.addWidget(lut_group)

        # Store current frame data and LUT settings
        current_frame_data = {'frame': None, 'vmin': None, 'vmax': None, 'frame_min': None, 'frame_max': None, 'histogram': None}
        is_updating_lut = False  # Flag to prevent recursive updates during LUT adjustment
        hist_vmin_line = None  # Vertical line for min value in histogram
        hist_vmax_line = None  # Vertical line for max value in histogram
        hist_plot = None  # Histogram bar plot
        dragging_line = None  # Track which line is being dragged
        
        # Initialize LUT values (will be set from first frame)
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
            vmin, vmax = float(vmin), float(vmax)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmax = max(vmin + 1e-6, float(np.nanmax(frame)) if np.any(np.isfinite(frame)) else vmin + 1)
            vrange = max(vmax - vmin, 1e-6)
            display = ((frame.astype(np.float64) - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
            dummy_mask = np.zeros(display.shape[:2], dtype=np.int32)
            display_scaled, _ = scale_for_preview(display, dummy_mask)
            pixmap = to_qpixmap(display_scaled)
            image_canvas.set_pixmap(pixmap, (display_scaled.shape[1], display_scaled.shape[0]), reset_view=False)

        def update_histogram(frame, vmin, vmax):
            """Update histogram plot with min/max lines"""
            nonlocal hist_vmin_line, hist_vmax_line, hist_plot
            
            # Clear axes instead of trying to remove lines
            hist_ax.clear()
            
            # Calculate histogram
            flat_frame = frame.flatten()
            flat_frame = flat_frame[flat_frame > 0]  # Exclude zeros for better visualization
            if len(flat_frame) == 0:
                flat_frame = frame.flatten()
            
            frame_min = float(frame.min())
            frame_max = float(frame.max())
            
            # Create histogram
            bins = min(256, int(frame_max - frame_min + 1))
            hist, bin_edges = np.histogram(flat_frame, bins=bins, range=(frame_min, frame_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot histogram (log scale for better visualization)
            hist_plot = hist_ax.bar(bin_centers, np.log1p(hist), width=bin_edges[1] - bin_edges[0], 
                                   color='#4A90E2', alpha=0.8, edgecolor='none')
            
            # Draw min/max lines (cleaner appearance)
            hist_vmin_line = hist_ax.axvline(vmin, color='#FF6B6B', linewidth=2, linestyle='--', label=f'Min: {vmin:.1f}')
            hist_vmax_line = hist_ax.axvline(vmax, color='#4ECDC4', linewidth=2, linestyle='--', label=f'Max: {vmax:.1f}')
            
            # Subtle fill region between min and max (lighter, less intrusive)
            hist_ax.axvspan(vmin, vmax, alpha=0.15, color='yellow')
            
            hist_ax.set_xlabel('Intensity', color='white', fontsize=10)
            hist_ax.set_ylabel('Log(Count+1)', color='white', fontsize=10)
            hist_ax.set_title('Click/drag lines to adjust min/max', color='white', fontsize=9, pad=10)
            hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
            hist_ax.grid(True, alpha=0.25, color='white', linestyle='-', linewidth=0.5)
            
            # Ensure proper margins to prevent cutoff
            hist_ax.margins(x=0.02, y=0.08)
            
            # Reapply tight layout after drawing to prevent cutoff
            hist_fig.tight_layout(pad=2.0)
            
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
            
            # Determine which line is closest (within 5% of range for easier grabbing)
            if current_vmin is not None and current_vmax is not None:
                frame_min = current_frame_data.get('frame_min', 0)
                frame_max = current_frame_data.get('frame_max', 65535)
                range_size = frame_max - frame_min
                if range_size > 0:
                    threshold = max(range_size * 0.05, (frame_max - frame_min) * 0.01)  # 5% of range, or 1% minimum
                else:
                    threshold = 100  # Default threshold
                
                dist_to_min = abs(x - current_vmin)
                dist_to_max = abs(x - current_vmax)
                
                # Check if clicking near a line
                if dist_to_min < threshold and dist_to_min < dist_to_max:
                    dragging_line = 'min'
                elif dist_to_max < threshold:
                    dragging_line = 'max'
                else:
                    # If not near a line, create a new line at click position
                    # Determine which side of the midpoint to adjust
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
                    # Block signals to prevent recursive updates
                    lut_min_slider.blockSignals(True)
                    lut_min_slider.setValue(new_min)
                    lut_min_slider.blockSignals(False)
                elif dragging_line == 'max':
                    new_max = min(frame_max, max(x, (current_vmin or frame_min) + 1))
                    current_vmax = new_max
                    # Block signals to prevent recursive updates
                    lut_max_slider.blockSignals(True)
                    lut_max_slider.setValue(new_max)
                    lut_max_slider.blockSignals(False)
                
                # Update display directly without calling update_lut_from_sliders (to avoid recursion)
                vmin = current_vmin
                vmax = current_vmax
                current_frame_data['vmin'] = vmin
                current_frame_data['vmax'] = vmax

                # Update image display (Qt-native canvas)
                if current_frame_data['frame'] is not None:
                    refresh_image_display()

                    # Update frame info (include view mode; use stored display name if synced)
                    current_channel = (channel_spin.value() if channel_spin is not None else 1)
                    _showing_corrected = (display_frames_ref['data'] is not raw_frames)
                    _dn = getattr(preview_dialog, '_bleach_display_name_used', method_display_name) if _showing_corrected else None
                    _view_suffix = f" | View: Bleach corrected ({_dn})" if _showing_corrected else " | View: Raw"
                    frame_info.setText(f"Timepoint {preview_dialog.current_frame + 1} of {num_time} | Channel: {current_channel} | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]{_view_suffix}")

                    # Update histogram lines
                    if hist_vmin_line is not None:
                        hist_vmin_line.set_xdata([vmin, vmin])
                    if hist_vmax_line is not None:
                        hist_vmax_line.set_xdata([vmax, vmax])
                    # Update fill region
                    for patch in list(hist_ax.patches):
                        if hasattr(patch, 'get_alpha') and patch.get_alpha() == 0.2:
                            patch.remove()
                    hist_ax.axvspan(vmin, vmax, alpha=0.2, color='yellow')
                    # Update legend
                    if hist_vmin_line is not None and hist_vmax_line is not None:
                        hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                    hist_canvas.draw()
            finally:
                is_updating_lut = False
        
        def on_histogram_release(event):
            """Handle mouse release to stop dragging"""
            nonlocal dragging_line
            dragging_line = None
        
        # Connect histogram interaction handlers
        hist_canvas.mpl_connect('button_press_event', on_histogram_press)
        hist_canvas.mpl_connect('motion_notify_event', on_histogram_motion)
        hist_canvas.mpl_connect('button_release_event', on_histogram_release)
        
        # Function to update displayed frame (reads from display_frames_ref so raw/corrected can be switched)
        def update_frame(frame_idx, channel_idx=0, vmin=None, vmax=None, update_sliders=True):
            """Update the displayed frame (frame_idx is 0-indexed internally, channel_idx is 0-indexed)"""
            frames_source = display_frames_ref['data']
            # Get current channel selection if available
            if channel_spin is not None:
                channel_idx = channel_spin.value() - 1  # Convert to 0-indexed
            else:
                channel_idx = 0  # Default to first channel
            
            # Handle different frames_source shapes directly (more robust)
            # ND2 files can have various shapes: (T, C, Z, Y, X), (T, C, Y, X), (T, Y, X), etc.
            if len(frames_source.shape) == 5:  # TCZYX
                # Extract frame, channel, and first Z-slice
                if channel_idx < frames_source.shape[1]:
                    frame = frames_source[frame_idx, channel_idx, 0, :, :]  # T=frame_idx, C=channel_idx, Z=0
                else:
                    frame = frames_source[frame_idx, 0, 0, :, :]  # Fallback
            elif len(frames_source.shape) == 4:
                # TCYX (e.g. bleach-corrected) or TZYX: always take 2D slice so frame is (Y, X)
                if num_channels > 1:
                    # TCYX: extract frame and channel
                    if channel_idx < frames_source.shape[1]:
                        frame = frames_source[frame_idx, channel_idx, :, :]
                    else:
                        frame = frames_source[frame_idx, 0, :, :]
                else:
                    # Single channel: second dim is either C=1 or Z; take index 0 to get (Y, X)
                    frame = frames_source[frame_idx, 0, :, :]
            elif len(frames_source.shape) == 3:
                # Could be TYX (single channel, single Z) or CYX (single frame, multi-channel)
                if frames_source.shape[0] == num_time:
                    # TYX: single channel, single Z
                    frame = frames_source[frame_idx, :, :]
                elif num_channels > 1:
                    # CYX: single frame, multi-channel - extract channel
                    if channel_idx < frames_source.shape[0]:
                        frame = frames_source[channel_idx, :, :]
                    else:
                        frame = frames_source[0, :, :]
                else:
                    # Fallback: assume TYX
                    frame = frames_source[frame_idx, :, :]
            elif len(frames_source.shape) == 2:
                # YX: single frame, single channel
                frame = frames_source
            else:
                # Fallback: try to extract frame
                frame = frames_source[frame_idx] if frame_idx < len(frames_source) else frames_source[0]
                # If still multi-dimensional, try to extract channel
                if len(frame.shape) > 2 and num_channels > 1:
                    if channel_idx < frame.shape[0]:
                        frame = frame[channel_idx]
                    else:
                        frame = frame[0]
            
            # Store frame data and actual min/max
            current_frame_data['frame'] = frame
            frame_min = float(frame.min())
            frame_max = float(frame.max())
            current_frame_data['frame_min'] = frame_min
            current_frame_data['frame_max'] = frame_max
            
            # Determine LUT values - preserve current values or use defaults
            nonlocal current_vmin, current_vmax
            if vmin is not None and vmax is not None:
                # Explicit values provided
                current_vmin = vmin
                current_vmax = vmax
            elif current_vmin is None or current_vmax is None:
                # First time - use percentile-based defaults
                current_vmin = float(np.percentile(frame, 1))
                current_vmax = float(np.percentile(frame, 99))
            # Otherwise, keep current_vmin and current_vmax (preserve user settings)
            
            vmin = current_vmin
            vmax = current_vmax
            
            current_frame_data['vmin'] = vmin
            current_frame_data['vmax'] = vmax
            
            # Update sliders (block signals to prevent recursive calls)
            if update_sliders:
                nonlocal is_updating_lut
                is_updating_lut = True
                try:
                    lut_min_slider.blockSignals(True)
                    lut_max_slider.blockSignals(True)
                    
                    # Update range based on current frame (but don't change values!)
                    # Only expand range if needed, never shrink it
                    current_min_range = lut_min_slider.minimum()
                    current_max_range = lut_min_slider.maximum()
                    new_min_range = min(current_min_range, frame_min) if current_min_range > 0 else frame_min
                    new_max_range = max(current_max_range, frame_max)
                    
                    lut_min_slider.setRange(new_min_range, new_max_range)
                    lut_max_slider.setRange(new_min_range, new_max_range)
                    
                    # DON'T change the values - preserve what user set!
                    # Only clamp if absolutely necessary (out of new range)
                    current_min_val = lut_min_slider.value()
                    current_max_val = lut_max_slider.value()
                    
                    if vmin is not None and vmax is not None:
                        # Use provided values
                        lut_min_slider.setValue(max(new_min_range, min(new_max_range, vmin)))
                        lut_max_slider.setValue(max(new_min_range, min(new_max_range, vmax)))
                    else:
                        # Preserve current values if they're still valid
                        if new_min_range <= current_min_val <= new_max_range:
                            # Keep current value
                            pass
                        else:
                            # Clamp to new range
                            lut_min_slider.setValue(max(new_min_range, min(new_max_range, current_min_val)))
                        
                        if new_min_range <= current_max_val <= new_max_range:
                            # Keep current value
                            pass
                        else:
                            # Clamp to new range
                            lut_max_slider.setValue(max(new_min_range, min(new_max_range, current_max_val)))
                    
                    lut_min_slider.blockSignals(False)
                    lut_max_slider.blockSignals(False)
                finally:
                    is_updating_lut = False
            
            # Update histogram
            update_histogram(frame, vmin, vmax)

            # Update Qt-native canvas (same pipeline as Segmentation Preview)
            _title = f"Timepoint {frame_idx + 1} of {num_time}  |  {Path(file_path).name}"
            if display_frames_ref['data'] is not raw_frames:
                _dn = getattr(preview_dialog, '_bleach_display_name_used', method_display_name)
                _title += f"  |  Bleach corrected ({_dn})"
            title_label.setText(_title)
            refresh_image_display()

            # Update frame info (1-indexed for display); show whether view is raw or bleach-corrected (use stored display name if updated by sync)
            current_channel = (channel_spin.value() if channel_spin is not None else 1)
            showing_corrected = (display_frames_ref['data'] is not raw_frames)
            disp_name = getattr(preview_dialog, '_bleach_display_name_used', method_display_name) if showing_corrected else None
            view_suffix = f" | View: Bleach corrected ({disp_name})" if showing_corrected else " | View: Raw"
            frame_info.setText(f"Timepoint {frame_idx + 1} of {num_time} | Channel: {current_channel} | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]{view_suffix}")
        
        # Manual LUT adjustment from sliders
        def update_lut_from_sliders():
            """Update display when sliders are moved"""
            # Prevent recursive calls
            nonlocal is_updating_lut, current_vmin, current_vmax
            if is_updating_lut:
                return
            
            is_updating_lut = True
            try:
                vmin = lut_min_slider.value()
                vmax = lut_max_slider.value()
                
                # Ensure min < max for valid display
                if vmin >= vmax:
                    vmax = vmin + 1.0
                    lut_max_slider.blockSignals(True)
                    lut_max_slider.setValue(vmax)
                    lut_max_slider.blockSignals(False)
                
                # Update stored values
                current_vmin = vmin
                current_vmax = vmax
                current_frame_data['vmin'] = vmin
                current_frame_data['vmax'] = vmax

                # Update the image display (Qt-native canvas)
                if current_frame_data['frame'] is not None:
                    refresh_image_display()

                    # Update frame info (include view mode; use stored display name if synced)
                    current_channel = (channel_spin.value() if channel_spin is not None else 1)
                    _showing_corr = (display_frames_ref['data'] is not raw_frames)
                    _dn2 = getattr(preview_dialog, '_bleach_display_name_used', method_display_name) if _showing_corr else None
                    _view_suff = f" | View: Bleach corrected ({_dn2})" if _showing_corr else " | View: Raw"
                    frame_info.setText(f"Timepoint {preview_dialog.current_frame + 1} of {num_time} | Channel: {current_channel} | Z-slice: 1 | LUT: [{vmin:.1f}, {vmax:.1f}]{_view_suff}")

                    # Update histogram lines (only update lines, don't redraw entire histogram)
                    if hist_vmin_line is not None and hist_vmax_line is not None:
                        # Update line positions
                        hist_vmin_line.set_xdata([vmin, vmin])
                        hist_vmax_line.set_xdata([vmax, vmax])
                        # Update the fill region - clear old and add new
                        for patch in list(hist_ax.patches):
                            if hasattr(patch, 'get_alpha') and patch.get_alpha() == 0.2:
                                patch.remove()
                        hist_ax.axvspan(vmin, vmax, alpha=0.2, color='yellow')
                        # Update legend
                        hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                        hist_canvas.draw()
                    else:
                        # If lines don't exist yet, update full histogram
                        update_histogram(current_frame_data['frame'], vmin, vmax)
                else:
                    # Fallback: full update if image handle doesn't exist
                    update_frame(preview_dialog.current_frame, vmin=vmin, vmax=vmax, update_sliders=False)
            finally:
                is_updating_lut = False
        
        # Debounce timer for valueChanged signals (to avoid too many updates while dragging)
        lut_update_timer = QTimer()
        lut_update_timer.setSingleShot(True)
        lut_update_timer.timeout.connect(update_lut_from_sliders)
        
        def on_lut_value_changed():
            """Debounced handler for valueChanged signal"""
            # Only trigger if not currently updating (to avoid conflicts)
            if not is_updating_lut:
                lut_update_timer.start(50)  # 50ms debounce
        
        # Connect LUT sliders
        # Use debounced valueChanged for smooth real-time updates when using arrows
        lut_min_slider.valueChanged.connect(on_lut_value_changed)
        lut_max_slider.valueChanged.connect(on_lut_value_changed)
        # Use editingFinished for immediate update when user finishes typing (Enter key)
        # Make sure it doesn't get reset by blocking signals during update
        def on_lut_editing_finished():
            """Handle when user finishes editing (Enter key or focus loss)"""
            if not is_updating_lut:
                update_lut_from_sliders()
        
        lut_min_slider.editingFinished.connect(on_lut_editing_finished)
        lut_max_slider.editingFinished.connect(on_lut_editing_finished)
        
        # Store current frame
        preview_dialog.current_frame = 0
        
        # Connect navigation (matching Preview Segmentation style)
        def go_to_frame(idx):
            if 0 <= idx < num_time:
                preview_dialog.current_frame = idx
                # Block signals to prevent recursive calls
                frame_spin.blockSignals(True)
                frame_spin.setValue(idx + 1)
                frame_spin.blockSignals(False)
                try:
                    # Always preserve current slider values when navigating
                    # User's manual adjustments persist across frames
                    # Get current values from sliders (user's settings)
                    vmin = lut_min_slider.value()
                    vmax = lut_max_slider.value()
                    # Update frame with user's values - don't recalculate!
                    update_frame(idx, vmin=vmin, vmax=vmax, update_sliders=False)
                except Exception as e:
                    QMessageBox.warning(preview_dialog, "Preview Error", 
                                      f"Could not display frame {idx + 1}:\n{str(e)}")
        
        # Debounce delay for frame/channel spinboxes (update after this many ms of no change, or on Enter)
        _DEBOUNCE_MS = 400
        # Connect channel selection (if available); debounced or on Enter
        if channel_spin is not None:
            def on_channel_changed():
                """Update frame when channel changes - auto-adjust LUT to new channel"""
                if preview_dialog.current_frame is not None:
                    # Reset LUT values so they recalculate for the new channel
                    nonlocal current_vmin, current_vmax
                    current_vmin = None
                    current_vmax = None
                    # This will auto-adjust min/max based on the new channel's intensity range
                    update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)
            channel_debounce = QTimer(preview_dialog)
            channel_debounce.setSingleShot(True)
            channel_debounce.timeout.connect(on_channel_changed)
            preview_dialog._channel_debounce_timer = channel_debounce
            def on_channel_value_changed():
                channel_debounce.stop()
                channel_debounce.start(_DEBOUNCE_MS)
            channel_spin.valueChanged.connect(on_channel_value_changed)
            channel_spin.editingFinished.connect(lambda: (channel_debounce.stop(), on_channel_changed()))
        
        # Connect "Show bleach correction" checkbox: always use current main-window settings when computing
        if show_bleach_check is not None and bleach_status_label is not None:
            def compute_corrected_from_tab():
                """Read current method/channels from main window (self) and compute corrected stack."""
                kwargs = self._get_bleach_preview_kwargs()
                cur_method = kwargs['bleach_method']
                cur_channels = kwargs.get('bleach_channel_indices')
                try:
                    from bactimate.scripts.convert_nd2_to_tif import bleach_correct_stack
                except ImportError:
                    return None, None, None
                data = np.asarray(raw_frames)
                if data.ndim == 5:
                    data = data[:, :, 0, :, :].copy()  # TCZYX -> TCYX
                elif data.ndim == 3 and num_channels == 1:
                    data = data[:, np.newaxis, :, :]  # TYX -> T1YX for stack API
                corrected = bleach_correct_stack(data, method=cur_method, channel_indices=cur_channels)
                display_name = _bleach_method_display.get(cur_method, cur_method)
                return corrected, cur_method, cur_channels, display_name

            def on_show_bleach_toggled(checked):
                nonlocal current_vmin, current_vmax
                if checked:
                    result = compute_corrected_from_tab()
                    corrected, cur_method, cur_channels, disp_name = result[0], result[1], result[2], result[3]
                    if corrected is None:
                        QMessageBox.warning(preview_dialog, "Preview", "Could not import bleach_correct_stack for preview.")
                        show_bleach_check.setChecked(False)
                        return
                    preview_dialog._bleach_corrected_frames = corrected
                    preview_dialog._bleach_method_used = cur_method
                    preview_dialog._bleach_channel_indices_used = tuple(sorted(cur_channels or [])) if cur_channels is not None else ()
                    preview_dialog._bleach_display_name_used = disp_name
                    display_frames_ref['data'] = preview_dialog._bleach_corrected_frames
                    bleach_status_label.setText(f"View: Bleach corrected ({disp_name})")
                    bleach_status_label.setStyleSheet("color: #4CAF50; font-size: 10pt;")
                else:
                    display_frames_ref['data'] = raw_frames
                    bleach_status_label.setText("View: Raw")
                    bleach_status_label.setStyleSheet("color: #9E9E9E; font-size: 10pt;")
                current_vmin = None
                current_vmax = None
                try:
                    update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)
                except Exception as e:
                    if checked:
                        display_frames_ref['data'] = raw_frames
                        bleach_status_label.setText("View: Raw")
                        bleach_status_label.setStyleSheet("color: #9E9E9E; font-size: 10pt;")
                        show_bleach_check.setChecked(False)
                    QMessageBox.warning(
                        preview_dialog, "Preview",
                        f"Could not update preview after changing bleach correction:\n{str(e)}"
                    )
                    try:
                        update_frame(preview_dialog.current_frame, vmin=None, vmax=None, update_sliders=True)
                    except Exception:
                        pass

            show_bleach_check.toggled.connect(on_show_bleach_toggled)

            # Poll main window: if user changes method/channels while preview is open and "Show bleach correction" is on, recompute
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
                    result = compute_corrected_from_tab()
                    corrected, _, _, disp_name = result[0], result[1], result[2], result[3]
                    if corrected is not None:
                        preview_dialog._bleach_corrected_frames = corrected
                        preview_dialog._bleach_method_used = cur_method
                        preview_dialog._bleach_channel_indices_used = cur_channels_tuple
                        preview_dialog._bleach_display_name_used = disp_name
                        display_frames_ref['data'] = preview_dialog._bleach_corrected_frames
                        bleach_status_label.setText(f"View: Bleach corrected ({disp_name})")
                        current_vmin_outer = current_vmin
                        current_vmax_outer = current_vmax
                        update_frame(preview_dialog.current_frame, vmin=current_vmin_outer, vmax=current_vmax_outer, update_sliders=False)

            sync_timer = QTimer(preview_dialog)
            sync_timer.timeout.connect(poll_main_window_settings)
            sync_timer.start(500)  # Check every 0.5 s
            preview_dialog._bleach_sync_timer = sync_timer  # keep reference
        
        # Debounced frame change: update after ~400 ms of no typing, or immediately on Enter
        def on_frame_changed():
            user_frame = frame_spin.value()  # 1-based: 1 = first frame
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
        preview_dialog._frame_debounce_timer = frame_debounce
        def on_frame_value_changed():
            frame_debounce.stop()
            frame_debounce.start(_DEBOUNCE_MS)
        frame_spin.valueChanged.connect(on_frame_value_changed)
        frame_spin.editingFinished.connect(lambda: (frame_debounce.stop(), QTimer.singleShot(0, on_frame_changed)))
        
        # Close button (same as other preview windows). Disable as default so Enter in spinbox doesn't close dialog.
        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(100)
        close_btn.setAutoDefault(False)
        close_btn.setDefault(False)
        close_btn.clicked.connect(preview_dialog.close)
        layout.addWidget(close_btn)
        
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, preview_dialog)
        close_shortcut.activated.connect(preview_dialog.close)
        minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), preview_dialog)  # Cmd+M on macOS
        minimize_shortcut.activated.connect(preview_dialog.showMinimized)
        
        # Display first frame (with auto LUT applied by default)
        # Initialize current_frame to ensure it's set before channel changes
        preview_dialog.current_frame = 0
        update_frame(0)
        
        preview_dialog.setLayout(layout)
        preview_dialog.show()  # Use show() instead of exec_() for non-modal dialog
    
    def _preview_nd2(self):
        """Preview ND2 file in a dialog with frame navigation"""
        
        try:
            import numpy as np
            import nd2
            
            file_path = self.file_path.text()
            
            # Create preview dialog (non-modal so user can access main window)
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)  # Make non-modal so main window is accessible
            
            layout = QVBoxLayout()
            
            # Load ND2 file info first
            with nd2.ND2File(file_path) as nd2_reader:
                sizes = nd2_reader.sizes
                num_time = sizes.get('T', 1)
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                # Load all frames and take an owned copy while reader is still open.
                # nd2_reader.asarray() can return a view into reader-backed memory; after the
                # context exits the reader closes and that memory is invalid. Any later .copy()
                # (e.g. in preview or bleach correction) would then segfault.
                all_frames = np.ascontiguousarray(nd2_reader.asarray()).copy()
            
            # Use unified preview UI (with optional bleach preview from tab settings)
            self._create_unified_preview_ui(
                preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                **self._get_bleach_preview_kwargs()
            )
            
        except ImportError as e:
            error_msg = str(e)
            if 'torch.testing' in error_msg:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  "Preview requires torch.testing module.\n\n"
                                  "This is a known issue in some environments.\n"
                                  "Try running from source: python main.py")
            else:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  f"Preview requires matplotlib:\n\npip install matplotlib")
        except Exception as e:
            error_msg = str(e)
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "PyTorch initialization error.\n\n"
                                  "This is a known issue with PyTorch in some environments.\n"
                                  "Try restarting the application or running from source.")
            else:
                QMessageBox.critical(self, "Preview Error", f"Could not preview file:\n{error_msg}")
            import traceback
            traceback.print_exc()
    
    def _preview_lif(self):
        """Preview LIF file in a dialog with frame navigation. Uses BioIO when available, then aicsimageio."""
        try:
            import numpy as np
            file_path = self.file_path.text()
            max_f = self.max_frames.value() if hasattr(self, "max_frames") else 500
            # Try BioIO first; on error (e.g. bioio-lif "too many values to unpack") fall back to aicsimageio
            try:
                data = bioio_load_data(Path(file_path), max_frames=max_f, use_first_z_only=True)
                if data is not None and data.size > 0:
                    preview_dialog = QDialog(self)
                    preview_dialog.setWindowTitle("File Preview")
                    preview_dialog.setMinimumSize(900, 700)
                    preview_dialog.setModal(False)
                    layout = QVBoxLayout()
                    num_time, num_channels = data.shape[0], data.shape[1]
                    num_z = 1
                    # For unified UI: (T, C, Y, X) is supported; take first channel for 3D display if needed by UI
                    all_frames = np.ascontiguousarray(data).copy()
                    self._create_unified_preview_ui(
                        preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                        **self._get_bleach_preview_kwargs()
                    )
                    preview_dialog.setLayout(layout)
                    preview_dialog.show()
                    return
            except Exception:
                pass
            # Try aicsimageio
            aics_preview_ok = False
            try:
                from aicsimageio import AICSImage
                preview_dialog = QDialog(self)
                preview_dialog.setWindowTitle("File Preview")
                preview_dialog.setMinimumSize(900, 700)
                preview_dialog.setModal(False)
                layout = QVBoxLayout()
                lif_reader = AICSImage(file_path)
                try:
                    dims = lif_reader.dims
                    dims_str = dims.order if hasattr(dims, 'order') else (dims if isinstance(dims, str) else str(dims))
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
                    self._create_unified_preview_ui(
                        preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                        **self._get_bleach_preview_kwargs()
                    )
                    preview_dialog.setLayout(layout)
                    preview_dialog.show()
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
            # Fallback: readlif (use microscopy series; Image 0 may be Job/sensor)
            try:
                from readlif.reader import LifFile
                lif_file = LifFile(file_path)
                if getattr(lif_file, "num_images", 0) <= 0:
                    raise ValueError("No image series in LIF file")
                log_cb = lambda msg: self.parent_window.log_output(msg)
                best_idx = self._readlif_best_image_index(lif_file, log_cb=log_cb)
                image = lif_file.get_image(best_idx)
                num_time, num_z, num_channels = self._readlif_dims(image, log_cb=log_cb)
                n_load = min(max_f, num_time)
                n_ch = max(1, num_channels)
                # Load (T, C, Y, X) so the preview channel selector works
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
                    all_frames = np.ascontiguousarray(np.stack(frames_list, axis=0))  # (T, C, Y, X)
                else:
                    all_frames = np.ascontiguousarray(np.stack(frames_list, axis=0))  # (T, Y, X)
                preview_dialog = QDialog(self)
                preview_dialog.setWindowTitle("File Preview (readlif)")
                preview_dialog.setMinimumSize(900, 700)
                preview_dialog.setModal(False)
                layout = QVBoxLayout()
                self._create_unified_preview_ui(
                    preview_dialog, layout, all_frames, n_load, n_ch, num_z, file_path,
                    **self._get_bleach_preview_kwargs()
                )
                preview_dialog.setLayout(layout)
                preview_dialog.show()
                return
            except Exception as e:
                import traceback
                self.parent_window.log_output(f"Readlif preview failed: {type(e).__name__}: {e}\n")
                self.parent_window.log_output(traceback.format_exc())
            QMessageBox.critical(self, "Error Previewing LIF", "Could not preview LIF with BioIO, AICSImageIO, or readlif.")
        except ImportError:
            QMessageBox.warning(self, "Library Missing",
                              "LIF preview requires 'bioio' and 'bioio-lif', or 'aicsimageio', or 'readlif'.\n\n"
                              "Install with: pip install aicsimageio readlif")
        except Exception as e:
            QMessageBox.critical(self, "Error Previewing LIF", f"Could not preview LIF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _preview_czi(self):
        """Preview CZI file in a dialog with frame navigation. Uses BioIO when available, then czifile."""
        try:
            import numpy as np
            file_path = self.file_path.text()
            if not file_path:
                QMessageBox.warning(self, "No File", "Please select a CZI file or folder first.")
                return
            path = Path(file_path)
            if path.is_dir():
                czi_in_dir = sorted(path.glob("*.czi")) + sorted(path.glob("*.CZI"))
                if not czi_in_dir:
                    QMessageBox.warning(self, "CZI Preview", "No CZI files found in the selected folder.")
                    return
                path = czi_in_dir[0]
                file_path = str(path)
            max_f = self.max_frames.value() if hasattr(self, "max_frames") else 500
            max_z_proj = self.max_z_projection.isChecked() if hasattr(self, "max_z_projection") else False
            data = bioio_load_data(path, max_frames=max_f, use_first_z_only=not max_z_proj, max_z_projection=max_z_proj)
            if data is not None and data.size > 0:
                preview_dialog = QDialog(self)
                preview_dialog.setWindowTitle("File Preview")
                preview_dialog.setMinimumSize(900, 700)
                preview_dialog.setModal(False)
                layout = QVBoxLayout()
                num_time, num_channels = data.shape[0], data.shape[1]
                num_z = 1
                all_frames = np.ascontiguousarray(data).copy()
                self._create_unified_preview_ui(
                    preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                    **self._get_bleach_preview_kwargs()
                )
                preview_dialog.setLayout(layout)
                preview_dialog.show()
                return
            import czifile
        except ImportError:
            QMessageBox.warning(
                self, "Library Missing",
                "CZI preview requires 'bioio' and 'bioio-czi', or 'czifile'.\n\n"
                "Install with: pip install bioio bioio-czi\nOr: pip install czifile"
            )
            return
        try:
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            layout = QVBoxLayout()
            all_frames = None
            num_time = num_channels = num_z = 1
            with czifile.CziFile(file_path) as czi_file:
                axes = getattr(czi_file, "axes", None)
                shape = getattr(czi_file, "shape", None)
                if axes and shape and len(axes) == len(shape):
                    num_time, num_channels, num_z = self._czi_dims_from_axes(axes, shape)
                arr = czi_file.asarray() if hasattr(czi_file, "asarray") else None
                if arr is not None and arr.size > 0:
                    arr = np.ascontiguousarray(arr).copy()
            if arr is not None and arr.size > 0:
                if axes and len(axes) == len(arr.shape):
                    c_preview, z_preview = self._czi_find_first_nonempty_plane(arr, axes)
                    all_frames = self._czi_slice_to_t_y_x(arr, axes, z_index=z_preview, c_index=c_preview)
                else:
                    # Fallback when axes don't match array shape
                    sh = arr.shape
                    if len(sh) >= 5:
                        all_frames = arr[:, 0, 0, :, :]
                    elif len(sh) == 4:
                        all_frames = arr[:, 0, :, :]
                    elif len(sh) == 3:
                        all_frames = arr
                    else:
                        all_frames = arr
                    if axes and shape and len(axes) == len(shape):
                        num_time, num_channels, num_z = self._czi_dims_from_axes(axes, shape)
            if all_frames is not None and getattr(all_frames, "size", 0) == 0:
                all_frames = None

            if all_frames is None or all_frames.size == 0:
                QMessageBox.warning(
                    self, "CZI Preview",
                    "Could not load CZI image data.\n\nInstall: pip install czifile"
                )
                return
            if len(all_frames.shape) == 2:
                all_frames = all_frames[np.newaxis, ...]
            all_frames = np.ascontiguousarray(all_frames).copy()
            self._create_unified_preview_ui(
                preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                **self._get_bleach_preview_kwargs()
            )
            preview_dialog.setLayout(layout)
            preview_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "Error Previewing CZI", f"Could not preview CZI file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _preview_tiff(self):
        """Preview TIFF file in a dialog with frame navigation. Uses BioIO when available, then tifffile."""
        try:
            import numpy as np
            file_path = self.file_path.text()
            max_f = self.max_frames.value() if hasattr(self, "max_frames") else 500
            data = bioio_load_data(Path(file_path), max_frames=max_f, use_first_z_only=True)
            if data is not None and data.size > 0:
                preview_dialog = QDialog(self)
                preview_dialog.setWindowTitle("File Preview")
                preview_dialog.setMinimumSize(900, 700)
                preview_dialog.setModal(False)
                layout = QVBoxLayout()
                num_time, num_channels = data.shape[0], data.shape[1]
                num_z = 1
                all_frames = np.ascontiguousarray(data).copy()
                self._create_unified_preview_ui(
                    preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                    **self._get_bleach_preview_kwargs()
                )
                preview_dialog.setLayout(layout)
                preview_dialog.show()
                return
            import tifffile
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            layout = QVBoxLayout()
            all_frames = safe_imread(file_path)
            
            # Determine dimensions
            ndim = len(all_frames.shape)
            num_pages = 1
            try:
                with tifffile.TiffFile(file_path) as tif:
                    num_pages = len(tif.pages)
            except:
                pass
            
            if ndim == 2:
                # Single 2D image or multi-page
                if num_pages > 1:
                    # Multi-page TIFF - load all pages
                    all_frames = []
                    with tifffile.TiffFile(file_path) as tif:
                        for page in tif.pages:
                            all_frames.append(page.asarray())
                    all_frames = np.array(all_frames)
                    num_time = len(all_frames)
                    num_channels = 1
                    num_z = 1
                else:
                    all_frames = all_frames[np.newaxis, ...]
                    num_time = 1
                    num_channels = 1
                    num_z = 1
            elif ndim == 3:
                # Could be (T, Y, X) or (C, Y, X)
                if all_frames.shape[0] < 10:
                    # Likely channels
                    num_time = 1
                    num_channels = all_frames.shape[0]
                    num_z = 1
                    # Take first channel for preview
                    all_frames = all_frames[0:1, ...]  # Keep time dimension
                else:
                    # Likely time series
                    num_time = all_frames.shape[0]
                    num_channels = 1
                    num_z = 1
            elif ndim == 4:
                # Could be (T, C, Y, X) or (T, Z, Y, X)
                if all_frames.shape[1] < 10:
                    # Likely (T, C, Y, X)
                    num_time = all_frames.shape[0]
                    num_channels = all_frames.shape[1]
                    num_z = 1
                    # Take first channel for preview
                    all_frames = all_frames[:, 0, ...]
                else:
                    # Likely (T, Z, Y, X)
                    num_time = all_frames.shape[0]
                    num_channels = 1
                    num_z = all_frames.shape[1]
                    # Take first z-slice for preview
                    all_frames = all_frames[:, 0, ...]
            else:
                # 5D or more - take first channel and z-slice
                num_time = all_frames.shape[0]
                num_channels = all_frames.shape[2] if len(all_frames.shape) > 2 else 1
                num_z = all_frames.shape[1] if len(all_frames.shape) > 1 else 1
                all_frames = all_frames[:, 0, 0, ...]  # Take first z and channel
            
            # Ensure we have (T, Y, X) shape
            if len(all_frames.shape) == 2:
                all_frames = all_frames[np.newaxis, ...]
            
            # Own a contiguous copy so preview/bleach code never touches file-backed memory (avoids segfault on .copy()).
            all_frames = np.ascontiguousarray(all_frames).copy()
            # Use unified preview UI with full LUT controls (and optional bleach preview)
            self._create_unified_preview_ui(
                preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                **self._get_bleach_preview_kwargs()
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error Previewing TIFF", f"Could not preview TIFF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def convert_nd2(self):
        """Convert microscopy file(s) to TIFF"""
        if not self.file_path.text():
            QMessageBox.warning(self, "No File Selected", "Please select a microscopy file or folder first.")
            return
        
        if not self.file_info:
            QMessageBox.warning(self, "File Not Loaded", "Please load file information first.")
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
            if num_channels >= 2:
                self._convert_tiff()
            else:
                QMessageBox.information(self, "TIFF File",
                                        "The selected file is already a single-channel TIFF.\n"
                                        "No conversion needed.")
        else:
            QMessageBox.warning(self, "Unsupported Format", f"Conversion not yet implemented for {file_type} files.")
    
    def _convert_nd2(self):
        """Convert file(s) to TIFF"""
        file_path = Path(self.file_path.text())

        # When folder is selected: show file selection dialog (same as Preview) so user can tick/untick files to convert
        if file_path.is_dir():
            supported_files = sorted(
                list(file_path.glob("*.nd2")) + list(file_path.glob("*.ND2"))
            )
            if not supported_files:
                QMessageBox.warning(self, "No ND2 Files", "No ND2 files found in the selected folder.")
                return
            selected_files = show_file_selection_dialog(
                self,
                supported_files,
                title="Select Files to Convert",
                message=f"Select which files to convert to TIFF (all {len(supported_files)} selected by default):"
            )
            if selected_files is None:
                return  # User cancelled
            if len(selected_files) == 0:
                QMessageBox.warning(self, "No Files Selected", "Please select at least one file to convert.")
                return
            files_to_convert = selected_files
        else:
            files_to_convert = [file_path]

        # Check for Z-stack and prompt user if max projection is not enabled
        if self.file_info and self.file_info.get('num_z', 1) > 1:
            if not self.max_z_projection.isChecked():
                reply = QMessageBox.information(
                    self, 
                    "Z-Stack Detected",
                    f"This file has Z={self.file_info['num_z']} slices.\n\n"
                    "By default, only the first Z-slice (Z=1) will be used.\n\n"
                    "To combine all Z-slices into one image, please check:\n"
                    "\"Max intensity projection for multiple Z stack\"\n\n"
                    "Would you like to enable max Z-projection now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.max_z_projection.setChecked(True)
        
        # Check if convert_nd2_to_tif.py exists (path from bactimate.gui.paths / constants)
        from bactimate.gui.constants import CONVERT_ND2_SCRIPT
        if not CONVERT_ND2_SCRIPT.exists():
            QMessageBox.warning(
                self, "Script Not Found",
                f"Conversion script not found.\n\nExpected: {CONVERT_ND2_SCRIPT}\n\n"
                "Install BactiMate correctly or run from the project root."
            )
            return
        convert_script = CONVERT_ND2_SCRIPT
        
        # Queue for multi-file conversion: process one file at a time, start next on finish
        self._conversion_queue = files_to_convert
        self._conversion_current_index = 0
        first_path = self._conversion_queue[0]

        # Build command arguments (path is the file/folder to convert; for queue we pass single file)
        cmd_args = [str(first_path)]
        if self.frame_extract_mode.currentIndex() == 1:  # Single frame only
            cmd_args.extend(['--single-frame', str(self.single_frame_spin.value())])
        else:
            cmd_args.extend(['--frames', str(self.max_frames.value())])
        
        if self.separate_channels.isChecked():
            cmd_args.append('--separate-channels')
        else:
            cmd_args.append('--combine-channels')
        
        # Max Z projection
        if self.max_z_projection.isChecked():
            cmd_args.append('--max-z-projection')
        
        # All channels are converted (no single-channel selection)
        if self.drift_correct.isChecked():
            cmd_args.append('--drift-correct')
        
        if self.bleach_correct.isChecked():
            cmd_args.append('--bleach-correct')
            bleach_methods = ['exponential', 'linear', 'histogram_matching', 'per_frame_minmax']
            cmd_args.append('--bleach-method')
            cmd_args.append(bleach_methods[self.bleach_method_combo.currentIndex()])
            # Only pass --bleach-channels when user flagged specific channels in Channel options
            if getattr(self, 'bleach_channel_checks', None):
                indices = [i for i, cb in enumerate(self.bleach_channel_checks) if cb.isChecked()]
                if indices and len(indices) < len(self.bleach_channel_checks):
                    cmd_args.append('--bleach-channels')
                    cmd_args.extend(str(i) for i in indices)
        # FI and Foci channels: only when saving channels separately
        if self.separate_channels.isChecked() and getattr(self, 'fi_channel_checks', None):
            fi_indices = [i for i, cb in enumerate(self.fi_channel_checks) if cb.isChecked()]
            if fi_indices:
                cmd_args.append('--fi-channels')
                cmd_args.extend(str(i) for i in fi_indices)
        if self.separate_channels.isChecked() and getattr(self, 'foci_channel_checks', None):
            foci_indices = [i for i, cb in enumerate(self.foci_channel_checks) if cb.isChecked()]
            if foci_indices:
                cmd_args.append('--foci-channels')
                cmd_args.extend(str(i) for i in foci_indices)
        
        # Pass pixel size and time per frame if extracted (for TIFF metadata)
        if self.file_info and self.file_info.get('pixel_size'):
            cmd_args.extend(['--pixel-size', str(self.file_info['pixel_size'])])
        if self.file_info and self.file_info.get('time_per_frame_min'):
            cmd_args.extend(['--time-per-frame', str(self.file_info['time_per_frame_min'])])
        
        # Run in thread
        self.convert_btn.setEnabled(False)
        self.convert_btn.setText("Converting...")
        self.parent_window.log_output(f"Starting ND2 conversion: {first_path.name}" + (
            f" (file 1/{len(files_to_convert)})" if len(files_to_convert) > 1 else "") + "\n")
        
        # Show progress bar with proper range
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        
        # Track progress state
        self.conversion_progress = {
            'total_files': len(files_to_convert),
            'current_file': 0,
            'file_progress': 0
        }
        # For completion popup "Open output folder" (TIFFs written next to ND2)
        self._last_conversion_output_dir = first_path.parent

        # Store reference to prevent garbage collection
        self.worker = WorkerThread(str(convert_script), cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self._update_conversion_progress)
        self.worker.start()
    
    def _update_conversion_progress(self, message):
        """Parse conversion output and update progress bar"""
        # Log the message
        self.parent_window.log_output(message + "\n")
        
        # Parse progress from message
        msg_lower = message.lower()
        import re
        
        # Detect total number of files - try multiple patterns
        if "found" in msg_lower:
            # Try different patterns: "Found 3 ND2 file(s) to process" or "Found 3 file(s)"
            match = re.search(r'found (\d+)', msg_lower, re.IGNORECASE)
            if match:
                self.conversion_progress['total_files'] = int(match.group(1))
                # Reset progress when we detect files
                self.conversion_progress['current_file'] = 0
                self.conversion_progress['file_progress'] = 0
        
        # Detect file processing start
        if "processing:" in msg_lower or ("processing" in msg_lower and ".nd2" in msg_lower):
            # New file starting - increment current_file index
            # This happens when we see "Processing: filename.nd2"
            if self.conversion_progress['file_progress'] >= 90:  # Previous file was nearly done
                # Move to next file
                if self.conversion_progress['total_files'] > 0:
                    self.conversion_progress['current_file'] = min(
                        self.conversion_progress['current_file'] + 1,
                        self.conversion_progress['total_files'] - 1
                    )
            self.conversion_progress['file_progress'] = 5  # Start new file at 5%
        
        # Estimate progress based on stage (use if statements to update progressively)
        # Each stage sets a minimum progress, so progress can only increase
        if "reading frames" in msg_lower or "reading" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 20)
        if "extracting frames" in msg_lower or "extracting" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 25)
        if "dimensions:" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 15)
        if "applying drift correction" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 40)
        if "drift correction complete" in msg_lower or "✓ drift" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 45)
        if "applying bleach correction" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 50)
        if "bleach correction complete" in msg_lower or "✓ bleach" in msg_lower:
            self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], 55)
        
        # Parse channel saving progress (e.g., "Saving channel 1/3" or "channel 1/3:")
        if "saving" in msg_lower:
            channel_match = re.search(r'channel (\d+)/(\d+)', msg_lower)
            if channel_match:
                current_ch = int(channel_match.group(1))
                total_ch = int(channel_match.group(2))
                # Saving is 60-95% of file progress
                save_progress = 60 + int(35 * (current_ch / total_ch))
                self.conversion_progress['file_progress'] = max(self.conversion_progress['file_progress'], save_progress)
            else:
                # General saving stage
                if self.conversion_progress['file_progress'] < 60:
                    self.conversion_progress['file_progress'] = 60
        
        # Detect completion
        if "complete" in msg_lower or "✓" in message or "successfully" in msg_lower:
            self.conversion_progress['file_progress'] = 100
        
        # Fallback: if we see any meaningful output, increment progress slightly
        # This ensures the bar moves even if pattern matching fails
        if len(message.strip()) > 5:
            # Increment by 1% for each meaningful message (capped at 95% to leave room for completion)
            if self.conversion_progress['file_progress'] < 95:
                self.conversion_progress['file_progress'] = min(
                    95, 
                    self.conversion_progress['file_progress'] + 1
                )
        
        # Calculate overall progress
        if self.conversion_progress['total_files'] > 0:
            # Progress = (completed files + current file progress) / total files
            # current_file is 0-indexed, so file 0 = first file
            completed_files = self.conversion_progress['current_file']  # Files already completed (0 = working on first)
            current_file_progress = self.conversion_progress['file_progress'] / 100.0
            
            # Overall = (files_done + current_file_progress) / total_files
            # If we're on file 0 at 50%, that's 0.5/3 = 16.7% overall (for 3 files)
            overall_progress = int(((completed_files + current_file_progress) / self.conversion_progress['total_files']) * 100)
        else:
            # Fallback: if we don't know total files, use file_progress directly
            # This ensures progress shows even if detection fails
            overall_progress = self.conversion_progress['file_progress']
        
        # Clamp to 0-100
        overall_progress = max(0, min(100, overall_progress))
        
        # Update progress bar (ensure it's visible and has correct range)
        if not self.parent_window.progress_bar.isVisible():
            self.parent_window.progress_bar.setVisible(True)
        if self.parent_window.progress_bar.maximum() != 100:
            self.parent_window.progress_bar.setRange(0, 100)
        
        # Update the value - always update even if 0 to ensure it's set
        self.parent_window.progress_bar.setValue(overall_progress)
        
        # Force GUI update to ensure the bar redraws
        QApplication.processEvents()
    
    def on_finished(self, success, message):
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        self.parent_window.log_output(message)
        if not success:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")
            self.workflow_step_finished.emit(False)
            return
        # Success: if we have a queue and more files to convert, run next
        queue = getattr(self, '_conversion_queue', None)
        idx = getattr(self, '_conversion_current_index', 0)
        if queue and idx + 1 < len(queue):
            self._conversion_current_index = idx + 1
            next_path = queue[self._conversion_current_index]
            from bactimate.gui.constants import CONVERT_ND2_SCRIPT
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
            if self.drift_correct.isChecked():
                cmd_args.append('--drift-correct')
            if self.bleach_correct.isChecked():
                cmd_args.append('--bleach-correct')
                cmd_args.append('--bleach-method')
                cmd_args.append(['exponential', 'linear', 'histogram_matching', 'per_frame_minmax'][self.bleach_method_combo.currentIndex()])
                if getattr(self, 'bleach_channel_checks', None):
                    indices = [i for i, cb in enumerate(self.bleach_channel_checks) if cb.isChecked()]
                    if indices and len(indices) < len(self.bleach_channel_checks):
                        cmd_args.append('--bleach-channels')
                        cmd_args.extend(str(i) for i in indices)
            if self.separate_channels.isChecked() and getattr(self, 'fi_channel_checks', None):
                fi_indices = [i for i, cb in enumerate(self.fi_channel_checks) if cb.isChecked()]
                if fi_indices:
                    cmd_args.append('--fi-channels')
                    cmd_args.extend(str(i) for i in fi_indices)
            if self.separate_channels.isChecked() and getattr(self, 'foci_channel_checks', None):
                foci_indices = [i for i, cb in enumerate(self.foci_channel_checks) if cb.isChecked()]
                if foci_indices:
                    cmd_args.append('--foci-channels')
                    cmd_args.extend(str(i) for i in foci_indices)
            if self.file_info and self.file_info.get('pixel_size'):
                cmd_args.extend(['--pixel-size', str(self.file_info['pixel_size'])])
            self.parent_window.log_output(f"Starting ND2 conversion: {next_path.name} (file {self._conversion_current_index + 1}/{len(queue)})\n")
            self.conversion_progress['current_file'] = self._conversion_current_index
            self.conversion_progress['file_progress'] = 0
            self.worker = WorkerThread(str(convert_script), cmd_args, parent=self)
            self.worker.finished.connect(self.on_finished)
            self.worker.progress.connect(self._update_conversion_progress)
            self.worker.start()
            return
        # All done
        self.convert_btn.setEnabled(True)
        self.convert_btn.setText("Convert to TIFF")
        self.parent_window.progress_bar.setVisible(False)
        n = len(queue) if queue else 1
        self.parent_window.log_output(f"✓ ND2 conversion completed successfully ({n} file(s))!\n")
        if queue:
            self._conversion_queue = None
            self._conversion_current_index = 0
        self._performed_this_session = True
        out_dir = getattr(self, "_last_conversion_output_dir", None)
        if out_dir is not None and Path(out_dir).exists():
            msg = QMessageBox(self)
            msg.setWindowTitle("Conversion Complete")
            msg.setText(f"Successfully converted {n} file(s) to TIFF.")
            msg.setIcon(QMessageBox.Icon.Information)
            open_btn = msg.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
            msg.addButton(QMessageBox.StandardButton.Ok)
            msg.exec()
            if msg.clickedButton() == open_btn:
                from PyQt6.QtCore import QUrl
                from PyQt6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
        else:
            QMessageBox.information(
                self,
                "Conversion Complete",
                f"Successfully converted {n} file(s) to TIFF.",
            )
        self.workflow_step_finished.emit(True)

    def _convert_lif(self):
        """Convert LIF file(s) to TIFF"""
        try:
            import numpy as np
            import tifffile
            from aicsimageio import AICSImage
            
            file_path = Path(self.file_path.text())
            
            # Check if it's a folder or single file
            if file_path.is_dir():
                # Folder mode: find all LIF files and show selection dialog (same as Preview / ND2)
                all_lif = sorted(
                    list(file_path.glob("*.lif")) + list(file_path.glob("*.LIF"))
                )
                if not all_lif:
                    QMessageBox.warning(self, "No LIF Files", 
                                      "No LIF files found in the selected folder.")
                    return
                selected = show_file_selection_dialog(
                    self, all_lif,
                    title="Select Files to Convert",
                    message=f"Select which files to convert to TIFF (all {len(all_lif)} selected by default):"
                )
                if selected is None:
                    return
                if len(selected) == 0:
                    QMessageBox.warning(self, "No Files Selected", "Please select at least one file to convert.")
                    return
                lif_files = selected
                total_files = len(lif_files)
                self.parent_window.log_output(f"Found {total_files} LIF file(s) to convert\n")
            else:
                # Single file mode
                lif_files = [file_path]
                total_files = 1
            
            # Output: Tif_files folder next to input file or in selected folder (same as ND2 / TIFF split)
            output_base = (file_path.parent if file_path.is_file() else file_path) / "Tif_files"
            output_base.mkdir(parents=True, exist_ok=True)

            # Get conversion parameters
            separate_channels = self.separate_channels.isChecked()

            # Show progress
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText("Converting...")
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)

            # Process each file
            for file_idx, lif_file in enumerate(lif_files):
                try:
                    self.parent_window.log_output(f"Processing: {lif_file.name}\n")
                    progress = int((file_idx / total_files) * 100)
                    self.parent_window.progress_bar.setValue(progress)
                    QApplication.processEvents()
                    
                    # Resolve frame range (first N or single frame) from file's num_time
                    num_time_for_range = 1
                    try:
                        info = bioio_get_info(Path(lif_file))
                        if info:
                            num_time_for_range = info.get("num_time", 1) or 1
                    except Exception:
                        pass
                    frame_start, frame_end = self._get_frame_range(num_time_for_range)
                    frame_suffix = f"_f{frame_start + 1}" if (frame_end - frame_start) == 1 else ""
                    
                    # Try BioIO first; on error (e.g. bioio-lif "too many values to unpack") fall back to AICSImage
                    data = None
                    try:
                        data = bioio_load_data(
                            Path(lif_file),
                            frame_start=frame_start,
                            frame_end=frame_end,
                            use_first_z_only=True,
                        )
                    except Exception:
                        pass
                    if data is not None and data.size > 0:
                        all_frames = np.ascontiguousarray(data)
                        num_time, num_channels = data.shape[0], data.shape[1]
                        num_z = 1
                        current_shape = all_frames.shape
                        channel_dim = 1 if num_channels > 1 else None
                        time_dim = 0
                        stem_used = lif_file.stem + frame_suffix
                        output_file = output_base / f"{stem_used}.tif"
                        if separate_channels and num_channels > 1 and channel_dim is not None:
                            max_channels = min(num_channels, current_shape[channel_dim])
                            for ch in range(max_channels):
                                indices = [slice(None)] * len(current_shape)
                                indices[channel_dim] = ch
                                channel_data = all_frames[tuple(indices)]
                                axes_to_squeeze = [i for i, s in enumerate(channel_data.shape) if s == 1 and (time_dim is None or i != time_dim)]
                                if axes_to_squeeze:
                                    channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                ch_output = output_base / f"{stem_used}_ch{ch+1}.tif"
                                tifffile.imwrite(str(ch_output), channel_data, photometric='minisblack')
                                self.parent_window.log_output(f"  Saved channel {ch+1}: {ch_output.name}\n")
                            self.parent_window.log_output(f"✓ Converted: {lif_file.name}\n")
                        else:
                            combined = all_frames
                            axes_to_squeeze = [i for i, s in enumerate(combined.shape) if s == 1 and (time_dim is None or i != time_dim)]
                            if axes_to_squeeze:
                                combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                            if len(combined.shape) == 2:
                                combined = combined[np.newaxis, ...]
                            tifffile.imwrite(str(output_file), combined, photometric='minisblack')
                            self.parent_window.log_output(f"  Saved: {output_file.name}\n")
                            self.parent_window.log_output(f"✓ Converted: {lif_file.name}\n")
                        continue
                    # Fallback: load with AICSImage (can raise "too many values to unpack" on some Leica LIFs)
                    aics_converted = False
                    try:
                        lif_reader = AICSImage(str(lif_file))
                        try:
                            # Get dimensions
                            dims = lif_reader.dims
                            if hasattr(dims, 'order'):
                                dims_str = dims.order
                            elif isinstance(dims, str):
                                dims_str = dims
                            else:
                                dims_str = str(dims)
                            
                            # Get data
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
                            
                            # Limit frames to selected range (first N or single frame)
                            if 'T' in dims_str:
                                t_idx = dims_str.index('T')
                                indices = [slice(None)] * len(all_frames.shape)
                                indices[t_idx] = slice(frame_start, min(frame_end, num_time))
                                all_frames = all_frames[tuple(indices)]
                            num_time = frame_end - frame_start
                            stem_used = lif_file.stem + (f"_f{frame_start + 1}" if num_time == 1 else "")
                            
                            # All channels converted (no single-channel selection)
                            # Take first z-slice if multiple z-slices
                            if num_z > 1 and 'Z' in dims_str:
                                z_idx = dims_str.index('Z')
                                if z_idx < len(all_frames.shape):
                                    indices = [slice(None)] * len(all_frames.shape)
                                    indices[z_idx] = 0
                                    all_frames = all_frames[tuple(indices)]
                                    num_z = 1
                            
                            # Reshape to (T, C, Y, X) or (T, Y, X)
                            # CRITICAL: Don't use all_frames[0] as it might remove time dimension!
                            # Find time dimension and avoid removing it
                            if len(all_frames.shape) > 4:
                                # Find time dimension index
                                time_dim_idx = None
                                if 'T' in dims_str:
                                    time_dim_idx = dims_str.index('T')
                                
                                # Remove dimensions that are not time, starting from the end
                                while len(all_frames.shape) > 4:
                                    # Find a dimension to remove (not time dimension)
                                    dim_to_remove = None
                                    for i in range(len(all_frames.shape) - 1, -1, -1):  # Start from end
                                        if time_dim_idx is None or i != time_dim_idx:
                                            if all_frames.shape[i] == 1:
                                                dim_to_remove = i
                                                break
                                    
                                    if dim_to_remove is not None:
                                        # Remove singleton dimension
                                        all_frames = np.squeeze(all_frames, axis=dim_to_remove)
                                        # Update dims_str if needed
                                        if dim_to_remove < len(dims_str):
                                            dims_str = dims_str[:dim_to_remove] + dims_str[dim_to_remove+1:]
                                            if time_dim_idx is not None and dim_to_remove < time_dim_idx:
                                                time_dim_idx -= 1
                                    else:
                                        # No singleton dimensions found, take first of a non-time dimension
                                        for i in range(len(all_frames.shape) - 1, -1, -1):
                                            if time_dim_idx is None or i != time_dim_idx:
                                                # Create indices to take first element of this dimension
                                                indices = [slice(None)] * len(all_frames.shape)
                                                indices[i] = 0
                                                all_frames = all_frames[tuple(indices)]
                                                # Update dims_str
                                                if i < len(dims_str):
                                                    dims_str = dims_str[:i] + dims_str[i+1:]
                                                    if time_dim_idx is not None and i < time_dim_idx:
                                                        time_dim_idx -= 1
                                                break
                            
                            # Create output filename (include _fN when single-frame extraction)
                            output_file = output_base / f"{stem_used}.tif"
                            
                            # Save as TIFF - use more robust dimension handling
                            # Find channel dimension by looking at actual shape after all slicing
                            current_shape = all_frames.shape
                            channel_dim = None
                            
                            # Try to find channel dimension - look for dimension matching num_channels
                            # But only if we still have multiple channels
                            if num_channels > 1:
                                for i, size in enumerate(current_shape):
                                    if size == num_channels:
                                        channel_dim = i
                                        break
                            
                            
                            # Find time dimension to preserve it
                            time_dim = None
                            if 'T' in dims_str:
                                t_idx = dims_str.index('T')
                                # After slicing, find which dimension corresponds to time
                                # Look for dimension with size matching num_time
                                for i, size in enumerate(current_shape):
                                    if size == num_time and num_time > 1:
                                        time_dim = i
                                        break
                                # If time dimension not found by size, try original position
                                if time_dim is None and t_idx < len(current_shape):
                                    time_dim = t_idx
                            
                            if separate_channels and num_channels > 1 and channel_dim is not None:
                                # Save each channel separately
                                max_channels = min(num_channels, current_shape[channel_dim])
                                for ch in range(max_channels):
                                    # Create indexing that selects one channel
                                    indices = [slice(None)] * len(current_shape)
                                    indices[channel_dim] = ch
                                    channel_data = all_frames[tuple(indices)]
                                    
                                    # Remove singleton dimensions but preserve time dimension
                                    # Only squeeze dimensions that are size 1 and not the time dimension
                                    axes_to_squeeze = []
                                    for i, size in enumerate(channel_data.shape):
                                        if size == 1 and (time_dim is None or i != time_dim):
                                            axes_to_squeeze.append(i)
                                    if axes_to_squeeze:
                                        channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                    
                                    # Ensure we have at least 2D (add time dimension if needed)
                                    if len(channel_data.shape) == 2:
                                        channel_data = channel_data[np.newaxis, ...]
                                    
                                    ch_output = output_base / f"{stem_used}_ch{ch+1}.tif"
                                    tifffile.imwrite(str(ch_output), channel_data, photometric='minisblack')
                                    self.parent_window.log_output(f"  Saved channel {ch+1}: {ch_output.name} ({channel_data.shape})\n")
                            else:
                                # Combine channels or single channel
                                if channel_dim is not None and num_channels > 1 and current_shape[channel_dim] > 1:
                                    # Use max projection across channels
                                    combined = np.max(all_frames, axis=channel_dim)
                                else:
                                    # Single channel or no channel dimension
                                    combined = all_frames
                                
                                # Remove singleton dimensions but preserve time dimension
                                # Only squeeze dimensions that are size 1 and not the time dimension
                                axes_to_squeeze = []
                                for i, size in enumerate(combined.shape):
                                    if size == 1 and (time_dim is None or i != time_dim):
                                        axes_to_squeeze.append(i)
                                if axes_to_squeeze:
                                    combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                                
                                # Ensure we have at least 2D (add time dimension if needed)
                                if len(combined.shape) == 2:
                                    combined = combined[np.newaxis, ...]
                                
                                # Log shape for debugging
                                self.parent_window.log_output(f"  Saving with shape: {combined.shape}\n")
                                
                                tifffile.imwrite(str(output_file), combined, photometric='minisblack')
                                self.parent_window.log_output(f"  Saved: {output_file.name}\n")
                            
                            self.parent_window.log_output(f"✓ Converted: {lif_file.name}\n")
                            aics_converted = True
                        finally:
                            if hasattr(lif_reader, 'close'):
                                try:
                                    lif_reader.close()
                                except Exception:
                                    pass
                    except Exception:
                        pass  # Try readlif fallback below
                    if aics_converted:
                        continue
                    # Fallback: readlif (when BioIO and AICSImage both fail, e.g. "too many values to unpack")
                    try:
                        from readlif.reader import LifFile
                        lif_file_obj = LifFile(str(lif_file))
                        if getattr(lif_file_obj, "num_images", 0) <= 0:
                            raise ValueError("No image series in LIF file")
                        best_idx = self._readlif_best_image_index(lif_file_obj)
                        image = lif_file_obj.get_image(best_idx)
                        num_time_file, num_z, num_channels = self._readlif_dims(image)
                        frame_start_rl, frame_end_rl = self._get_frame_range(num_time_file)
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
                        stem_used_rl = lif_file.stem + (f"_f{frame_start_rl + 1}" if num_time == 1 else "")
                        output_file = output_base / f"{stem_used_rl}.tif"
                        if separate_channels and n_ch > 1:
                            for ch in range(n_ch):
                                channel_data = all_frames[:, ch, :, :]
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                ch_output = output_base / f"{stem_used_rl}_ch{ch+1}.tif"
                                tifffile.imwrite(str(ch_output), channel_data, photometric='minisblack')
                                self.parent_window.log_output(f"  Saved channel {ch+1}: {ch_output.name}\n")
                            self.parent_window.log_output(f"✓ Converted (readlif): {lif_file.name}\n")
                        else:
                            if all_frames.ndim == 2:
                                all_frames = all_frames[np.newaxis, ...]
                            tifffile.imwrite(str(output_file), all_frames, photometric='minisblack')
                            self.parent_window.log_output(f"  Saved: {output_file.name}\n")
                            self.parent_window.log_output(f"✓ Converted (readlif): {lif_file.name}\n")
                        continue
                    except Exception as e2:
                        self.parent_window.log_output(f"✗ Error converting {lif_file.name}: {str(e2)}\n")
                        QMessageBox.warning(self, "Conversion Error",
                                            f"Error converting {lif_file.name}:\n{str(e2)}")
                except Exception as e:
                    self.parent_window.log_output(f"✗ Error converting {lif_file.name}: {str(e)}\n")
                    QMessageBox.warning(self, "Conversion Error",
                                        f"Error converting {lif_file.name}:\n{str(e)}")
            
            # Complete
            self.parent_window.progress_bar.setValue(100)
            QMessageBox.information(self, "Conversion Complete", 
                                  f"Successfully converted {total_files} LIF file(s) to TIFF.")
            
        except ImportError:
            QMessageBox.warning(self, "Library Missing",
                              "LIF conversion requires 'bioio' and 'bioio-lif', or 'aicsimageio', and 'tifffile'.\n\n"
                              "Install with: pip install bioio bioio-lif tifffile\nOr: pip install aicsimageio tifffile")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"LIF conversion failed:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
    
    def _convert_tiff(self):
        """Split a multi-channel TIFF into separate per-channel TIFFs (BactiMate convention: stem_fi_<name>.tif, stem_phase.tif)."""
        try:
            import tifffile
            file_path = Path(self.file_path.text())
            if not file_path.is_file():
                QMessageBox.warning(self, "No File", "Please select a TIFF file.")
                return
            num_time = self.file_info.get("num_time", 1)
            num_channels = self.file_info.get("num_channels", 1)
            channel_names = self.file_info.get("channel_names") or [f"Channel_{i+1}" for i in range(num_channels)]
            if num_channels < 2:
                QMessageBox.information(self, "TIFF", "This TIFF has only one channel. No split needed.")
                return
            # Output: Tif_files in same directory (like ND2)
            output_base = file_path.parent / "Tif_files"
            output_base.mkdir(parents=True, exist_ok=True)
            fi_indices = set()
            if getattr(self, "fi_channel_checks", None) and len(self.fi_channel_checks) >= num_channels:
                fi_indices = {i for i, cb in enumerate(self.fi_channel_checks) if cb.isChecked()}
            frame_start_tiff, frame_end_tiff = self._get_frame_range(num_time)
            n_t = frame_end_tiff - frame_start_tiff
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText("Splitting...")
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.log_output(f"Splitting multi-channel TIFF: {file_path.name} ({num_time} frames, {num_channels} channels)\n")
            with tifffile.TiffFile(file_path) as tif:
                first_page = tif.pages[0]
                shape = first_page.shape
                num_pages = len(tif.pages)
                if len(shape) == 2 and num_pages > 1:
                    # Multipage 2D: ImageJ order T*C (or C*T); use file_info num_time, num_channels
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
                    if raw.ndim == 2:
                        raw = raw[np.newaxis, np.newaxis, ...]
                    elif raw.ndim == 3:
                        if num_channels > 1 and raw.shape[0] == num_channels:
                            raw = raw[np.newaxis, ...]
                        else:
                            raw = raw[frame_start_tiff:frame_end_tiff, np.newaxis, ...]
                    if raw.ndim == 4:
                        raw = raw[frame_start_tiff:frame_end_tiff]
                        if num_channels == 1 and raw.shape[1] > 1:
                            raw = raw[:, 0:1, ...]
                    data = raw
                    n_t = data.shape[0]
            def _sanitize(s):
                return "".join(c if c.isalnum() or c in "_-" else "_" for c in (s or "channel")).strip("_") or "channel"
            stem_suffix_tiff = f"_f{frame_start_tiff + 1}" if n_t == 1 else ""
            stem_used_tiff = file_path.stem + stem_suffix_tiff
            phase_used = False
            for ch in range(num_channels):
                channel_data = np.ascontiguousarray(data[:, ch])
                if channel_data.ndim == 2:
                    channel_data = channel_data[np.newaxis, ...]
                name = (channel_names[ch] if ch < len(channel_names) else f"Channel_{ch+1}")
                safe = _sanitize(name)
                if ch in fi_indices:
                    out_name = f"{stem_used_tiff}_fi_{safe}.tif"
                else:
                    if not phase_used:
                        out_name = f"{stem_used_tiff}_phase.tif"
                        phase_used = True
                    else:
                        out_name = f"{stem_used_tiff}_{safe}.tif"
                out_path = output_base / out_name
                tifffile.imwrite(str(out_path), channel_data, photometric="minisblack")
                self.parent_window.log_output(f"  Saved: {out_path.name}\n")
            self.parent_window.progress_bar.setValue(100)
            QMessageBox.information(self, "Conversion Complete",
                                    f"Split {file_path.name} into {num_channels} channel TIFF(s) in:\n{output_base}")
        except Exception as e:
            self.parent_window.log_output(f"Error: {str(e)}\n")
            QMessageBox.critical(self, "Split failed", f"Could not split TIFF:\n{str(e)}")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
    
    def _convert_czi(self):
        """Convert CZI file(s) to TIFF using czifile (https://pypi.org/project/czifile/)."""
        try:
            import numpy as np
            import tifffile
            import czifile
            file_path = Path(self.file_path.text())
            
            # Check if it's a folder or single file
            if file_path.is_dir():
                # Folder mode: find all CZI files and show selection dialog (same as Preview / ND2)
                all_czi = sorted(
                    list(file_path.glob("*.czi")) + list(file_path.glob("*.CZI"))
                )
                if not all_czi:
                    QMessageBox.warning(self, "No CZI Files", 
                                      "No CZI files found in the selected folder.")
                    return
                selected = show_file_selection_dialog(
                    self, all_czi,
                    title="Select Files to Convert",
                    message=f"Select which files to convert to TIFF (all {len(all_czi)} selected by default):"
                )
                if selected is None:
                    return
                if len(selected) == 0:
                    QMessageBox.warning(self, "No Files Selected", "Please select at least one file to convert.")
                    return
                czi_files = selected
                total_files = len(czi_files)
                self.parent_window.log_output(f"Found {total_files} CZI file(s) to convert\n")
            else:
                # Single file mode
                czi_files = [file_path]
                total_files = 1
            
            # Output: next to input file or in selected folder (auto)
            output_base = file_path.parent if file_path.is_file() else file_path

            # Get conversion parameters
            separate_channels = self.separate_channels.isChecked()
            max_z_projection = self.max_z_projection.isChecked()

            # Show progress
            self.convert_btn.setEnabled(False)
            self.convert_btn.setText("Converting...")
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)

            # Process each file
            for file_idx, czi_file in enumerate(czi_files):
                try:
                    self.parent_window.log_output(f"Processing: {czi_file.name}\n")
                    progress = int((file_idx / total_files) * 100)
                    self.parent_window.progress_bar.setValue(progress)
                    QApplication.processEvents()
                    
                    # Resolve frame range from file's num_time
                    num_time_czi = 1
                    try:
                        info = bioio_get_info(Path(czi_file))
                        if info:
                            num_time_czi = info.get("num_time", 1) or 1
                    except Exception:
                        pass
                    frame_start_czi, frame_end_czi = self._get_frame_range(num_time_czi)
                    frame_suffix_czi = f"_f{frame_start_czi + 1}" if (frame_end_czi - frame_start_czi) == 1 else ""
                    
                    # Try BioIO first for robust reading
                    data = bioio_load_data(
                        Path(czi_file),
                        frame_start=frame_start_czi,
                        frame_end=frame_end_czi,
                        use_first_z_only=not max_z_projection,
                        max_z_projection=max_z_projection,
                    )
                    if data is not None and data.size > 0:
                        all_frames = np.ascontiguousarray(data)
                        num_time, num_channels = data.shape[0], data.shape[1]
                        num_z = 1
                        current_shape = all_frames.shape
                        channel_dim = 1 if num_channels > 1 else None
                        time_dim = 0
                        dims_str = "TCYX"
                        stem_used_czi = czi_file.stem + frame_suffix_czi
                        output_file = output_base / f"{stem_used_czi}.tif"
                        if separate_channels and num_channels > 1 and channel_dim is not None:
                            max_ch = min(num_channels, current_shape[channel_dim])
                            for ch in range(max_ch):
                                indices = [slice(None)] * len(current_shape)
                                indices[channel_dim] = ch
                                channel_data = all_frames[tuple(indices)]
                                axes_to_squeeze = [i for i, s in enumerate(channel_data.shape) if s == 1 and (time_dim is None or i != time_dim)]
                                if axes_to_squeeze:
                                    channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                                if len(channel_data.shape) == 2:
                                    channel_data = channel_data[np.newaxis, ...]
                                ch_output = output_base / f"{stem_used_czi}_ch{ch+1}.tif"
                                tifffile.imwrite(str(ch_output), channel_data, photometric='minisblack')
                                self.parent_window.log_output(f"  Saved channel {ch+1}: {ch_output.name}\n")
                            self.parent_window.log_output(f"✓ Converted: {czi_file.name}\n")
                        else:
                            combined = all_frames
                            axes_to_squeeze = [i for i, s in enumerate(combined.shape) if s == 1 and (time_dim is None or i != time_dim)]
                            if axes_to_squeeze:
                                combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                            if len(combined.shape) == 2:
                                combined = combined[np.newaxis, ...]
                            tifffile.imwrite(str(output_file), combined, photometric='minisblack')
                            self.parent_window.log_output(f"  Saved: {output_file.name}\n")
                            self.parent_window.log_output(f"✓ Converted: {czi_file.name}\n")
                        continue
                    # Fallback: load with czifile
                    with czifile.CziFile(str(czi_file)) as czi_f:
                        axes = getattr(czi_f, "axes", None)
                        shape = getattr(czi_f, "shape", None)
                        if axes and shape and len(axes) == len(shape):
                            num_time, num_channels, num_z = self._czi_dims_from_axes(axes, shape)
                        else:
                            num_time, num_channels, num_z = 1, 1, 1
                        arr = czi_f.asarray() if hasattr(czi_f, 'asarray') else np.array(czi_f)
                    if arr is None or arr.size == 0:
                        raise RuntimeError("czifile returned empty array")
                    if axes and len(axes) == len(arr.shape):
                        num_time = frame_end_czi - frame_start_czi
                        if num_z > 1 and max_z_projection:
                            # Keep Z, then max over Z so conversion respects "Max intensity projection" regardless of preview
                            all_frames, dims_str, z_axis = self._czi_slice_to_t_c_z_y_x(
                                arr, axes, frame_start=frame_start_czi, frame_end=frame_end_czi
                            )
                            if z_axis >= 0:
                                all_frames = np.max(all_frames, axis=z_axis).astype(all_frames.dtype)
                            dims_str = "".join(a for a in dims_str if a != "Z") or "TCYX"
                        else:
                            # First Z only (or single Z)
                            all_frames, dims_str = self._czi_slice_to_t_c_y_x(
                                arr, axes, frame_start=frame_start_czi, frame_end=frame_end_czi
                            )
                        all_frames = np.ascontiguousarray(all_frames)
                    else:
                        # Fallback when axes not available
                        sh = arr.shape
                        dims_str = ""
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
                            dims_str = "TCYX"
                        elif len(sh) == 4:
                            num_time = frame_end_czi - frame_start_czi
                            num_channels = int(sh[1])
                            num_z = 1
                            all_frames = np.ascontiguousarray(arr[frame_start_czi:frame_end_czi, :, :, :])
                            dims_str = "TCYX"
                        elif len(sh) == 3:
                            num_time = frame_end_czi - frame_start_czi
                            num_channels = 1
                            num_z = 1
                            all_frames = np.ascontiguousarray(
                                arr[frame_start_czi:frame_end_czi, :, :][:, np.newaxis, :, :]
                            )
                            dims_str = "TCYX"
                        else:
                            raise RuntimeError(f"Unexpected CZI shape: {sh}")
                    
                    if all_frames is None or all_frames.size == 0:
                        raise RuntimeError("No image data loaded")
                    
                    # Reshape to (T, C, Y, X) or (T, Y, X)
                    if len(all_frames.shape) > 4:
                        # Find time dimension first
                        time_dim_idx = None
                        if 'T' in dims_str:
                            time_dim_idx = dims_str.index('T')
                        
                        # Remove dimensions that are not time, starting from the end
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
                                    dims_str = dims_str[:dim_to_remove] + dims_str[dim_to_remove+1:]
                                    if time_dim_idx is not None and dim_to_remove < time_dim_idx:
                                        time_dim_idx -= 1
                            else:
                                for i in range(len(all_frames.shape) - 1, -1, -1):
                                    if time_dim_idx is None or i != time_dim_idx:
                                        indices = [slice(None)] * len(all_frames.shape)
                                        indices[i] = 0
                                        all_frames = all_frames[tuple(indices)]
                                        if i < len(dims_str):
                                            dims_str = dims_str[:i] + dims_str[i+1:]
                                            if time_dim_idx is not None and i < time_dim_idx:
                                                time_dim_idx -= 1
                                        break
                    
                    # Create output filename (include _fN when single-frame extraction)
                    stem_used_czi = czi_file.stem + frame_suffix_czi
                    output_file = output_base / f"{stem_used_czi}.tif"
                    
                    # Save as TIFF - use more robust dimension handling
                    current_shape = all_frames.shape
                    channel_dim = None
                    
                    if num_channels > 1:
                        for i, size in enumerate(current_shape):
                            if size == num_channels:
                                channel_dim = i
                                break
                    
                    time_dim = None
                    if 'T' in dims_str:
                        t_idx = dims_str.index('T')
                        for i, size in enumerate(current_shape):
                            if size == num_time and num_time > 1:
                                time_dim = i
                                break
                        if time_dim is None and t_idx < len(current_shape):
                            time_dim = t_idx
                    
                    if separate_channels and num_channels > 1 and channel_dim is not None:
                        max_channels = min(num_channels, current_shape[channel_dim])
                        for ch in range(max_channels):
                            indices = [slice(None)] * len(current_shape)
                            indices[channel_dim] = ch
                            channel_data = all_frames[tuple(indices)]
                            axes_to_squeeze = []
                            for i, size in enumerate(channel_data.shape):
                                if size == 1 and (time_dim is None or i != time_dim):
                                    axes_to_squeeze.append(i)
                            if axes_to_squeeze:
                                channel_data = np.squeeze(channel_data, axis=tuple(axes_to_squeeze))
                            if len(channel_data.shape) == 2:
                                channel_data = channel_data[np.newaxis, ...]
                            ch_output = output_base / f"{stem_used_czi}_ch{ch+1}.tif"
                            tifffile.imwrite(str(ch_output), channel_data, photometric='minisblack')
                            self.parent_window.log_output(f"  Saved channel {ch+1}: {ch_output.name} ({channel_data.shape})\n")
                    else:
                        if channel_dim is not None and num_channels > 1 and current_shape[channel_dim] > 1:
                            combined = np.max(all_frames, axis=channel_dim)
                        else:
                            combined = all_frames
                        axes_to_squeeze = []
                        for i, size in enumerate(combined.shape):
                            if size == 1 and (time_dim is None or i != time_dim):
                                axes_to_squeeze.append(i)
                        if axes_to_squeeze:
                            combined = np.squeeze(combined, axis=tuple(axes_to_squeeze))
                        if len(combined.shape) == 2:
                            combined = combined[np.newaxis, ...]
                        self.parent_window.log_output(f"  Saving with shape: {combined.shape}\n")
                        tifffile.imwrite(str(output_file), combined, photometric='minisblack')
                        self.parent_window.log_output(f"  Saved: {output_file.name}\n")
                    
                    self.parent_window.log_output(f"✓ Converted: {czi_file.name}\n")
                
                except Exception as e:
                    self.parent_window.log_output(f"✗ Error converting {czi_file.name}: {str(e)}\n")
                    QMessageBox.warning(self, "Conversion Error", f"Error converting {czi_file.name}:\n{str(e)}")
            
            # Complete
            self.parent_window.progress_bar.setValue(100)
            QMessageBox.information(self, "Conversion Complete", 
                                  f"Successfully converted {total_files} CZI file(s) to TIFF.")
            
        except ImportError:
            QMessageBox.warning(self, "Library Missing", 
                              "CZI conversion requires 'bioio' and 'bioio-czi', or 'tifffile' and 'czifile'.\n\n"
                              "Install with: pip install bioio bioio-czi tifffile\nOr: pip install tifffile czifile")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"CZI conversion failed:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (includes path for workflow replay)."""
        out = get_from_widgets([
            ("frame_extract_mode", self.frame_extract_mode),
            ("max_frames", self.max_frames),
            ("single_frame", self.single_frame_spin),
            ("separate_channels", self.separate_channels),
            ("max_z_projection", self.max_z_projection),
            ("drift_correct", self.drift_correct),
            ("bleach_correct", self.bleach_correct),
        ])
        if hasattr(self, "bleach_method_combo"):
            out["bleach_method"] = self.bleach_method_combo.currentIndex()
        if getattr(self, "bleach_channel_checks", None):
            out["bleach_fluorescence_channels"] = [cb.isChecked() for cb in self.bleach_channel_checks]
        if getattr(self, "fi_channel_checks", None):
            out["fi_channels"] = [cb.isChecked() for cb in self.fi_channel_checks]
        if getattr(self, "foci_channel_checks", None):
            out["foci_channels"] = [cb.isChecked() for cb in self.foci_channel_checks]
        if getattr(self, "zproject_channel_checks", None):
            out["zproject_channels"] = [cb.isChecked() for cb in self.zproject_channel_checks]
        path_txt = (self.file_path.text() or "").strip()
        if path_txt:
            out["file_path"] = path_txt
        out["performed"] = getattr(self, "_performed_this_session", False)
        return out

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values. Set file path and load file info first so channel options (FI, Z, Foci, bleach) exist, then apply their checkboxes."""
        apply_to_widgets(data, [
            ("frame_extract_mode", self.frame_extract_mode),
            ("max_frames", self.max_frames),
            ("single_frame", self.single_frame_spin),
            ("separate_channels", self.separate_channels),
            ("max_z_projection", self.max_z_projection),
            ("drift_correct", self.drift_correct),
            ("bleach_correct", self.bleach_correct),
        ])
        if "bleach_method" in data and hasattr(self, "bleach_method_combo"):
            try:
                idx = int(data["bleach_method"])
                if 0 <= idx < self.bleach_method_combo.count():
                    self.bleach_method_combo.setCurrentIndex(idx)
            except (TypeError, ValueError):
                pass
        # Load path and file info first so channel checkboxes (FI, Z, Foci, bleach) are created
        if "file_path" in data and data.get("file_path"):
            path = str(data["file_path"]).strip()
            self.file_path.setText(path)
            if path and Path(path).exists():
                if Path(path).is_file():
                    self.load_file_info(path)
                else:
                    folder = Path(path)
                    supported = sorted(
                        list(folder.glob("*.nd2")) + list(folder.glob("*.ND2")) +
                        list(folder.glob("*.lif")) + list(folder.glob("*.LIF")) +
                        list(folder.glob("*.czi")) + list(folder.glob("*.CZI")) +
                        list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) +
                        list(folder.glob("*.TIF")) + list(folder.glob("*.TIFF"))
                    )
                    if supported:
                        self.load_folder_info(path, supported)
        # Now apply per-channel selections (FI, Z-stack, Foci, bleach) from session log
        if "bleach_fluorescence_channels" in data and getattr(self, "bleach_channel_checks", None):
            try:
                states = data["bleach_fluorescence_channels"]
                if isinstance(states, list):
                    for i, checked in enumerate(states):
                        if i < len(self.bleach_channel_checks):
                            self.bleach_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if "fi_channels" in data and getattr(self, "fi_channel_checks", None):
            try:
                states = data["fi_channels"]
                if isinstance(states, list):
                    for i, checked in enumerate(states):
                        if i < len(self.fi_channel_checks):
                            self.fi_channel_checks[i].setChecked(bool(checked))
            except Exception:
                pass
        if "foci_channels" in data and getattr(self, "foci_channel_checks", None):
            try:
                states = data["foci_channels"]
                if isinstance(states, list):
                    for i, checked in enumerate(states):
                        if i < len(self.foci_channel_checks):
                            self.foci_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if "zproject_channels" in data and getattr(self, "zproject_channel_checks", None):
            try:
                states = data["zproject_channels"]
                if isinstance(states, list):
                    for i, checked in enumerate(states):
                        if i < len(self.zproject_channel_checks):
                            self.zproject_channel_checks[i].setChecked(bool(checked))
            except (TypeError, ValueError):
                pass
        if "performed" in data:
            self._performed_this_session = bool(data["performed"])

