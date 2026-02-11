"""
Processing tab: single file or folder segmentation with Omnipose (batch_process_tiff_sequence).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, QCoreApplication, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QPixmap, QImage, QPainter, QColor, QPen, QShortcut
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
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.constants import PROJECT_ROOT
from bactimate.gui.log_config import get_logger
from bactimate.gui.preferences import LAST_DIR_SEGMENTATION
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.utils import safe_imread
from bactimate.gui.widgets import (
    ClickableTooltip,
    TooltipClickFilter,
    DragDropGroupBox,
    create_info_icon,
    show_file_selection_dialog,
)
from bactimate.gui.workers import WorkerThread

class ProcessingTab(QWidget):
    """Unified tab for processing single files or folders"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        file_group = self._build_file_group()
        scroll_layout.addWidget(file_group)

        seg_group = self._build_segmentation_params()
        scroll_layout.addWidget(seg_group)

        # Connect file path changes to update overlay checkbox state
        self.file_path.textChanged.connect(self._update_overlay_checkbox_state)
        self._update_overlay_checkbox_state()

        # Preview button - modern styling applied via global stylesheet
        self.preview_btn = QPushButton("Preview Omnipose Segmentation")
        self.preview_btn.setEnabled(False)  # Disabled until file/folder is selected
        # Wrap preview_segmentation to catch all exceptions and prevent PyQt6 abort()
        def safe_preview_segmentation():
            try:
                self.preview_segmentation()
            except BaseException as e:
                # PyQt6 calls abort() on unhandled exceptions in slots
                get_logger().exception("Error in preview segmentation")
                import traceback
                error_msg = f"Error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                try:
                    QMessageBox.critical(self, "Preview Error", error_msg)
                except Exception:
                    get_logger().exception("Could not show preview error dialog")
                    print(f"CRITICAL ERROR: {error_msg}", file=sys.stderr)
                # Don't re-raise - PyQt6 will abort if we do
        self.preview_btn.clicked.connect(safe_preview_segmentation)
        scroll_layout.addWidget(self.preview_btn)

        # Process button
        self.process_btn = QPushButton("Process File or Folder")
        self.process_btn.setEnabled(False)  # Disabled until file/folder is selected
        # Modern styling applied via global stylesheet
        self.process_btn.clicked.connect(self.process_file_or_folder)
        scroll_layout.addWidget(self.process_btn)

        # Connect file path changes to update button states (after buttons are created)
        self.file_path.textChanged.connect(self._update_button_states)
        self._update_button_states()

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_file_group(self) -> QWidget:
        """Build the file/folder input group with path line edit and browse buttons."""
        file_group = DragDropGroupBox(
            "Input File/Folder",
            parent=self,
            on_drop_callback=self.handle_dropped_path_segmentation
        )
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.file_path, 0)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        file_group.setLayout(file_layout)
        return file_group

    def _build_segmentation_params(self) -> QGroupBox:
        """Build the segmentation parameters group (basic params + options + reset)."""
        seg_group = QGroupBox("Segmentation Parameters")
        seg_container_layout = QHBoxLayout()

        # Left column: Basic parameters
        basic_params_group = QGroupBox("Basic Parameters")
        basic_params_layout = QVBoxLayout()
        seg_layout = basic_params_layout

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_info_icon = create_info_icon(
            "Omnipose model type for cell segmentation.\n\n"
            "Default: 'bact_phase_omni'\n"
            "Models are downloaded from: https://github.com/kevinjohncutler/omnipose-models\n"
            "Documentation: https://omnipose.readthedocs.io/models.html\n\n"
            "Available Omnipose models:\n"
            "- bact_phase_omni: Bacterial phase contrast (recommended for bacteria)\n"
            "- bact_fluor_omni: Bacterial fluorescence\n"
            "- bact_phase_affinity: Bacterial phase contrast (affinity-based)\n\n"
            "Tip: Choose based on your cell type and imaging modality."
        , parent_widget=self)
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_info_icon)
        # Model status indicator (on the left, before combo box)
        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet("color: #7F8C8D; font-size: 9pt; font-style: italic;")
        self.model_status_label.setFixedWidth(150)  # Fixed width for status text
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        model_layout.addWidget(self.model_status_label)
        model_layout.addStretch()  # Add stretch before combo box to align to right
        self.model_combo = QComboBox()
        # Validate and add only available models
        available_models = self._get_available_models()
        self.model_combo.addItems(available_models)
        if available_models:
            # Set default to bact_phase_omni if available, otherwise first available
            if 'bact_phase_omni' in available_models:
                self.model_combo.setCurrentText('bact_phase_omni')
            else:
                self.model_combo.setCurrentText(available_models[0])
        model_layout.addWidget(self.model_combo)
        
        # Check model availability when changed
        def check_model_availability():
            """Check if selected model is available (cached or downloadable)"""
            from pathlib import Path
            try:
                from cellpose_omni.models import MODEL_DIR
                model_type = self.model_combo.currentText()
                
                # Check if model exists in cache
                torch_str = 'torch'
                basename = f'{model_type}{torch_str}_0'
                cache_model_file = Path(MODEL_DIR) / basename
                
                if cache_model_file.exists():
                    self.model_status_label.setText("✓ Cached")
                    self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                else:
                    self.model_status_label.setText("Will download on first use")
                    self.model_status_label.setStyleSheet("color: #FF9800; font-size: 9pt;")
            except Exception:
                get_logger().debug("Model availability check failed, hiding status", exc_info=True)
                self.model_status_label.setText("")
        
        self.model_combo.currentTextChanged.connect(check_model_availability)
        # Check initial model
        check_model_availability()
        self.model_combo.setToolTip(
            "Omnipose model type for cell segmentation.\n\n"
            "Default: 'bact_phase_omni'\n\n"
            "Available models:\n"
            "- bact_phase_omni: Bacterial phase contrast (default, best for bacteria)\n"
            "- bact_fluor_omni: Bacterial fluorescence\n"
            "- bact_phase_affinity: Bacterial phase contrast (affinity-based)\n\n"
            "Tip: Choose based on your cell type and imaging modality."
        )
        seg_layout.addLayout(model_layout)
        
        # Diameter
        diam_layout = QHBoxLayout()
        diam_label = QLabel("Diameter (0=auto):")
        diam_info_icon = create_info_icon(
            "Expected cell diameter in pixels.\n\n"
            "Default: 0 (auto-detect)\n"
            "Source: Omnipose default (auto-detection)\n\n"
            "Set to 0 for automatic detection (recommended for first try).\n"
            "Specify a value (e.g., 20-50 pixels for bacteria) for more consistent results.\n\n"
            "Typical values:\n"
            "- Bacteria: 20-50 pixels\n"
            "- Mammalian cells: 30-100 pixels\n"
            "- Nuclei: 15-30 pixels\n\n"
            "Tip: Auto-detection works well, but specifying diameter often gives better results."
        , parent_widget=self)
        diam_layout.addWidget(diam_label)
        diam_layout.addWidget(diam_info_icon)
        diam_layout.addStretch()  # Add stretch before spin box to align to right
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setValue(0)
        self.diameter.setDecimals(1)
        self.diameter.setFixedWidth(100)  # Fixed width for uniform appearance
        # Tooltip removed - info available via (?) icon only
        diam_layout.addWidget(self.diameter)
        seg_layout.addLayout(diam_layout)
        
        # Flow threshold
        flow_layout = QHBoxLayout()
        flow_label = QLabel("Flow Threshold:")
        flow_info_icon = create_info_icon(
            "Flow error threshold for cell detection.\n\n"
            "Default: 0.4\n"
            "Source: Omnipose default value\n\n"
            "Controls sensitivity of cell detection:\n"
            "- Lower values (0.2-0.3): More sensitive, detects more cells (may include false positives)\n"
            "- Higher values (0.5-0.7): More strict, fewer cells (may miss some cells)\n\n"
            "Range: 0.0 to 1.0\n"
            "Tip: If missing cells, try lowering to 0.3. If too many false positives, raise to 0.5."
        , parent_widget=self)
        flow_layout.addWidget(flow_label)
        flow_layout.addWidget(flow_info_icon)
        flow_layout.addStretch()  # Add stretch before spin box to align to right
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 1.0)
        self.flow_threshold.setValue(0.4)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setDecimals(2)
        self.flow_threshold.setFixedWidth(100)  # Fixed width for uniform appearance
        # Tooltip removed - info available via (?) icon only
        flow_layout.addWidget(self.flow_threshold)
        seg_layout.addLayout(flow_layout)
        
        # Mask threshold
        mask_layout = QHBoxLayout()
        mask_label = QLabel("Mask threshold:")
        mask_info_icon = create_info_icon(
            "Mask probability threshold for cell detection.\n\n"
            "Default: 0.0\n"
            "Source: Omnipose/Cellpose default\n\n"
            "Lower values: More permissive (more cells, possible false positives).\n"
            "Higher values: Stricter (fewer cells).\n\n"
            "Range: 0.0 to 1.0\n"
            "Tip: Usually 0.0 is fine; increase if masks look too loose.",
            parent_widget=self
        )
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(mask_info_icon)
        mask_layout.addStretch()
        self.mask_threshold = QDoubleSpinBox()
        self.mask_threshold.setRange(0.0, 1.0)
        self.mask_threshold.setValue(0.0)
        self.mask_threshold.setSingleStep(0.1)
        self.mask_threshold.setDecimals(2)
        self.mask_threshold.setFixedWidth(100)
        mask_layout.addWidget(self.mask_threshold)
        seg_layout.addLayout(mask_layout)
        
        basic_params_group.setLayout(basic_params_layout)
        basic_params_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        seg_container_layout.addWidget(basic_params_group)
        
        # Right column: Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # GPU option
        gpu_layout = QHBoxLayout()
        self.use_gpu = QCheckBox("Use GPU (if available)")
        self.use_gpu.setChecked(True)
        gpu_info_icon = create_info_icon(
            "Use GPU acceleration for faster processing.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Recommended for performance\n\n"
            "When enabled: Uses GPU if available (10-100x faster than CPU).\n"
            "When disabled: Forces CPU processing (slower but works without GPU).\n\n"
            "Note: Automatically falls back to CPU if no GPU is available.\n"
            "Tip: Disable if you encounter out-of-memory errors.",
            parent_widget=self
        )
        gpu_layout.addWidget(self.use_gpu)
        gpu_layout.addWidget(gpu_info_icon)
        gpu_layout.addStretch()
        options_layout.addLayout(gpu_layout)
        
        # DBSCAN clustering option
        cluster_layout = QHBoxLayout()
        self.cluster = QCheckBox("Use DBSCAN clustering")
        self.cluster.setChecked(False)
        cluster_info_icon = create_info_icon(
            "Use DBSCAN clustering to help segment thin/elongated cells.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: Optional Omnipose feature\n\n"
            "When enabled: Applies DBSCAN clustering to improve segmentation of:\n"
            "- Thin, elongated cells (e.g., bacteria, worms)\n"
            "- Cells with complex shapes\n"
            "- Overlapping cells\n\n"
            "Use when: Standard segmentation struggles with elongated or thin structures.\n"
            "Note: Slightly increases processing time.",
            parent_widget=self
        )
        cluster_layout.addWidget(self.cluster)
        cluster_layout.addWidget(cluster_info_icon)
        cluster_layout.addStretch()
        options_layout.addLayout(cluster_layout)
        
        # Overlay visualization option
        overlay_layout = QHBoxLayout()
        self.create_overlay = QCheckBox("Create overlay visualization")
        self.create_overlay.setChecked(True)
        overlay_info_icon = create_info_icon(
            "Create overlay visualization showing masks overlaid on original images.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Optional output file\n\n"
            "When enabled: Creates a colored overlay TIFF file showing segmentation masks overlaid on the original images.\n"
            "When disabled: Skips overlay creation (faster processing, saves disk space).\n\n"
            "Saved as: filename_overlay.tif\n\n"
            "Tip: Disable to speed up processing when you only need masks.",
            parent_widget=self
        )
        overlay_layout.addWidget(self.create_overlay)
        overlay_layout.addWidget(overlay_info_icon)
        overlay_layout.addStretch()
        options_layout.addLayout(overlay_layout)
        
        options_group.setLayout(options_layout)
        options_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        seg_container_layout.addWidget(options_group)
        
        # Connect file path changes to update overlay checkbox state
        self.file_path.textChanged.connect(self._update_overlay_checkbox_state)
        # Initialize state
        self._update_overlay_checkbox_state()
        
        # Reset to defaults button for segmentation parameters
        reset_seg_btn = QPushButton("Reset to Default")
        reset_seg_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #7D8C9E;"
            "color: white;"
            "font-weight: bold;"
            "padding: 6px;"
            "border: none;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "background-color: #6C7B8C;"
            "}"
        )
        reset_seg_btn.setToolTip("Reset all segmentation parameters to their default values")
        reset_seg_btn.clicked.connect(self.reset_segmentation_defaults)
        seg_container_layout.addWidget(reset_seg_btn)
        
        seg_group.setLayout(seg_container_layout)
        return seg_group

    def _update_button_states(self):
        """Update button enabled states based on whether file/folder is selected"""
        has_path = bool(self.file_path.text() and self.file_path.text().strip())
        if hasattr(self, 'preview_btn'):
            self.preview_btn.setEnabled(has_path)
        if hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(has_path)
    
    def _update_overlay_checkbox_state(self):
        """Update overlay checkbox tooltip based on folder vs file mode"""
        is_folder = self._is_folder_mode()
        # Overlay is now supported in both modes, so always enable it
        self.create_overlay.setEnabled(True)
        if is_folder:
            self.create_overlay.setToolTip(
                "Create overlay visualization showing masks overlaid on original images.\n\n"
                "Default: Enabled (checked)\n"
                "When enabled: Creates a colored overlay TIFF file for each file in the folder.\n"
                "When disabled: Skips overlay creation (faster processing).\n\n"
                "Saved as: filename_overlay.tif (in each file's output directory)"
            )
        else:
            self.create_overlay.setToolTip(
                "Create overlay visualization showing masks overlaid on original images.\n\n"
                "Default: Enabled (checked)\n"
                "When enabled: Creates a colored overlay TIFF file.\n"
                "When disabled: Skips overlay creation (faster processing).\n\n"
                "Saved as: filename_overlay.tif"
            )
    
    def select_file(self):
        """Open file dialog to select a TIFF file"""
        try:
            # Get the current file path if one exists, to use as starting directory
            start_dir = ""
            if self.file_path.text() and Path(self.file_path.text()).exists():
                start_dir = str(Path(self.file_path.text()).parent)
            elif self.file_path.text():
                start_dir = str(Path(self.file_path.text()).parent)
            else:
                start_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_SEGMENTATION) or str(Path.home())

            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select TIFF File",
                start_dir,
                "TIFF Files (*.tif *.tiff);;All Files (*)"
            )
            if file_path:
                if getattr(self.parent_window, "set_last_dir_for_tab", None):
                    self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, file_path)
                self.file_path.setText(file_path)
                self._update_overlay_checkbox_state()
                if self.parent_window and hasattr(self.parent_window, 'log_output'):
                    self.parent_window.log_output(f"Selected file: {file_path}\n")
        except Exception as e:
            QMessageBox.critical(
                self, 
                "File Selection Error", 
                f"Could not open file dialog:\n{str(e)}\n\n{traceback.format_exc()}"
            )
            if self.parent_window and hasattr(self.parent_window, 'log_output'):
                self.parent_window.log_output(f"Error selecting file: {str(e)}\n{traceback.format_exc()}\n")
    
    def select_folder(self):
        """Open folder dialog to select a folder"""
        start_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_SEGMENTATION) or str(Path.home())
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if folder_path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, folder_path)
            self.file_path.setText(folder_path)
            self._update_overlay_checkbox_state()
    
    def handle_dropped_path_segmentation(self, dropped_path):
        """Handle dropped file or folder path for segmentation tab"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        # Update overlay checkbox state
        self._update_overlay_checkbox_state()
        self._update_button_states()
    
    def _is_folder_mode(self):
        """Check if current path is a folder"""
        if not self.file_path.text():
            return False
        path = Path(self.file_path.text())
        return path.exists() and path.is_dir()
    
    def _get_available_models(self):
        """
        Return list of available Omnipose models only (no Cellpose-only models).
        """
        return [
            'bact_phase_omni',  # Recommended for bacteria
            'bact_fluor_omni',
            'bact_phase_affinity'
        ]
    
    def reset_segmentation_defaults(self):
        """Reset all segmentation parameters to their default values"""
        # Set default model, fallback to first available if bact_phase_omni not available
        if 'bact_phase_omni' in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText('bact_phase_omni')
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        self.diameter.setValue(0)
        self.flow_threshold.setValue(0.4)
        self.mask_threshold.setValue(0.0)
        self.use_gpu.setChecked(True)
        self.cluster.setChecked(False)
        self.create_overlay.setChecked(True)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        return get_from_widgets([
            ("model", self.model_combo),
            ("diameter", self.diameter),
            ("flow_threshold", self.flow_threshold),
            ("mask_threshold", self.mask_threshold),
            ("use_gpu", self.use_gpu),
            ("cluster", self.cluster),
            ("create_overlay", self.create_overlay),
        ])

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("model", self.model_combo),
            ("diameter", self.diameter),
            ("flow_threshold", self.flow_threshold),
            ("mask_threshold", self.mask_threshold),
            ("use_gpu", self.use_gpu),
            ("cluster", self.cluster),
            ("create_overlay", self.create_overlay),
        ])
    
    def preview_segmentation(self):
        """Preview segmentation on a single frame or file from folder"""
        # Note: This method is called from safe_preview_segmentation() wrapper
        # which catches all exceptions to prevent PyQt6 from aborting
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a TIFF file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: show file selection dialog first
            folder = Path(self.file_path.text())
            # Look for TIFF files (not mask files - this is for processing, not analyzing)
            # Use set to avoid duplicates from multiple glob patterns
            tiff_files_set = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                            set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out mask files (we want original files for segmentation preview)
            tiff_files = sorted([f for f in tiff_files_set if '_masks' not in f.name.lower()], 
                               key=lambda x: x.name.lower())
            
            if not tiff_files:
                QMessageBox.warning(self, "No TIFF Files Found", 
                                  "No TIFF files found in the selected folder.\n\n"
                                  "Please select a folder containing TIFF files to process.")
                return
            
            # Let user choose which file to preview - use custom dialog
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select File for Preview")
            file_dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Label
            label = QLabel("Select a file to preview segmentation:")
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
            for f in tiff_files:
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
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(file_dialog.accept)
            button_box.rejected.connect(file_dialog.reject)
            button_box.setStyleSheet(
                "QPushButton {"
                "background-color: #5D6D7E;"
                "color: white;"
                "font-weight: bold;"
                "padding: 8px 20px;"
                "border: none;"
                "border-radius: 4px;"
                "min-width: 80px;"
                "}"
                "QPushButton:hover {"
                "background-color: #34495E;"
                "}"
            )
            layout.addWidget(button_box)
            
            file_dialog.setLayout(layout)
            
            # Show dialog
            if file_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            # Get selected file
            selected_item = file_list.currentItem()
            if not selected_item:
                return
            
            selected_name = selected_item.text()
            selected_file = None
            for f in tiff_files:
                if f.name == selected_name:
                    selected_file = f
                    break
            
            if not selected_file:
                return
            
            # Use selected file for preview
            file_path = str(selected_file)
        else:
            # Single file mode: use the selected file directly
            file_path = self.file_path.text()
        
        # Show progress dialog while loading
        loading_progress = QProgressDialog("Loading file and initializing preview...", None, 0, 0, self)
        loading_progress.setWindowTitle("Loading")
        loading_progress.setAutoClose(False)
        loading_progress.setAutoReset(False)
        loading_progress.setCancelButton(None)
        loading_progress.show()
        QApplication.processEvents()
        
        # Load TIFF file
        try:
            loading_progress.setLabelText("Loading TIFF file...")
            QApplication.processEvents()
            
            import tifffile
            from bactimate.gui.image_utils import to_qpixmap, build_segmentation_overlay, scale_for_preview
            from bactimate.gui.image_canvas import ImageCanvas

            # CRITICAL: Patch omnipose.core.OMNI_MODELS before cellpose_omni imports it
            # This fixes "cannot import name 'OMNI_MODELS' from 'omnipose.core'" error in frozen builds
            # Also ensure get_masks patch is applied
            try:
                import omnipose.core
                # Re-apply patch in case it wasn't applied earlier (omnipose imported late)
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                if is_frozen:
                    _patch_omnipose_get_masks()
                if not hasattr(omnipose.core, 'OMNI_MODELS'):
                    # Construct OMNI_MODELS from available model constants
                    OMNI_MODELS = []
                    if hasattr(omnipose.core, 'C1_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_MODELS)
                    if hasattr(omnipose.core, 'C2_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_MODELS)
                    if hasattr(omnipose.core, 'C1_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_BD_MODELS)
                    if hasattr(omnipose.core, 'C2_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_BD_MODELS)
                    # Remove duplicates
                    seen = set()
                    OMNI_MODELS = [x for x in OMNI_MODELS if not (x in seen or seen.add(x))]
                    # Default fallback if still empty
                    if not OMNI_MODELS:
                        OMNI_MODELS = ['bact_omni', 'bact_phase_omni',
                                      'bact_fluor_omni']
                    omnipose.core.OMNI_MODELS = OMNI_MODELS
            except Exception:
                get_logger().debug("Omnipose get_masks patch failed, using fallback", exc_info=True)
            
            from cellpose_omni import models
            from cellpose_omni.io import imread
            import numpy as np
            
            images = safe_imread(file_path)
            
            loading_progress.setLabelText("Processing image data...")
            QApplication.processEvents()
            
            # Convert to float32 for MPS compatibility (Apple Silicon GPU doesn't support float64)
            if images.dtype != np.float32:
                images = images.astype(np.float32)
            
            # Handle different input formats
            if images.ndim == 2:
                images = images[np.newaxis, ...]
            elif images.ndim == 3:
                if images.shape[-1] < 10:  # Likely channels last
                    images = images.transpose(2, 0, 1)
                    images = images[np.newaxis, ...]
            elif images.ndim == 4:
                if images.shape[-1] < 10:  # Channels last
                    images = images.transpose(0, 3, 1, 2)
                elif images.shape[0] < 10:  # Channels first
                    images = images.transpose(1, 0, 2, 3)
            
            # Check if we have any frames
            if images.size == 0 or images.shape[0] == 0:
                QMessageBox.critical(preview_dialog, "Empty File", 
                                   "The TIFF file appears to be empty or has no frames.")
                return
            
            num_frames = images.shape[0]
            
            # Create preview dialog as top-level window (parent=None) so it opens in front
            # on all platforms; non-modal so user can access main window
            preview_dialog = QDialog(None)
            preview_dialog.setWindowTitle("Segmentation Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            preview_dialog.setWindowFlags(
                preview_dialog.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
            )
            
            layout = QVBoxLayout()

            # Title label (frame / cells / filename)
            title_label = QLabel("")
            title_label.setStyleSheet("font-weight: bold; color: #ddd;")
            layout.addWidget(title_label)

            # Qt-native image canvas (zoom: wheel; pan: left-drag or middle-drag)
            image_canvas = ImageCanvas()
            image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
            image_canvas.setMinimumSize(400, 300)
            image_canvas.setStyleSheet("background-color: #1e1e1e;")
            image_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            layout.addWidget(image_canvas, 1)

            # Store data for navigation
            preview_dialog.images = images
            preview_dialog.current_frame = 0
            preview_dialog.masks = None
            preview_dialog.model = None
            
            # Status label for model loading messages (defined early so load_model can use it)
            status_label = QLabel("Ready")
            status_label.setStyleSheet("color: #7F8C8D; font-size: 9pt; font-style: italic;")
            
            def find_model_file(model_type, model_index=0, use_torch=True):
                """
                Find model file in the default cache location.
                Models are cached in ~/.cellpose/models and persist between sessions.
                Returns the path to the model file if found, None otherwise.
                """
                from pathlib import Path
                from cellpose_omni.models import MODEL_DIR
                
                # Construct model filename
                torch_str = 'torch' if use_torch else ''
                basename = f'{model_type}{torch_str}_{model_index}'
                
                # Check default cache location
                cache_model_file = Path(MODEL_DIR) / basename
                
                if cache_model_file.exists():
                    return str(cache_model_file)
                
                # Not found
                return None
            
            def load_model():
                """Load Omnipose model (only reloads if model type or GPU setting changed)"""
                # Get current parameters
                params = get_current_parameters()
                current_model_type = params['model_type']
                current_use_gpu = params['use_gpu']
                
                # Check if model needs to be reloaded (only when model type or GPU changes)
                needs_reload = (preview_dialog.model is None or 
                               (hasattr(preview_dialog, '_last_model_type') and preview_dialog._last_model_type != current_model_type) or
                               (hasattr(preview_dialog, '_last_use_gpu') and preview_dialog._last_use_gpu != current_use_gpu))
                
                if needs_reload:
                    # Clear cache when model changes (old results are invalid)
                    preview_dialog.segmentation_cache.clear()
                
                if needs_reload:
                    try:
                        from cellpose_omni.models import model_path, MODEL_DIR
                        import os
                        from urllib.error import HTTPError
                        from pathlib import Path
                        
                        # Verify MODEL_DIR is set correctly (should be local models directory if env var was set)
                        # This is just for debugging - MODEL_DIR is set when module is imported
                        expected_model_dir = PROJECT_ROOT / 'models'
                        if expected_model_dir.exists() and str(MODEL_DIR) != str(expected_model_dir):
                            # Environment variable might not have been set before import
                            # This shouldn't happen, but log it if it does
                            pass
                        
                        # Check for model in local directory first, then cache
                        model_file = find_model_file(current_model_type, 0, True)
                        
                        if model_file and os.path.exists(model_file):
                            # Model found in cache
                            status_label.setText("Loading model from cache...")
                            QApplication.processEvents()
                        else:
                            # Model not found - need to download
                            status_label.setText("Downloading model (this may take a moment)...")
                            QApplication.processEvents()
                            # Call model_path - it will download if needed
                            try:
                                model_file = model_path(current_model_type, 0, True)
                                # Verify the model was actually downloaded
                                if not model_file or not os.path.exists(model_file):
                                    # model_path returned a path but file doesn't exist - try to download explicitly
                                    status_label.setText("Model not found, downloading...")
                                    QApplication.processEvents()
                                    # Force download by calling model_path again (it should download if file doesn't exist)
                                    model_file = model_path(current_model_type, 0, True)
                                    # Wait a moment for download to complete
                                    import time
                                    time.sleep(0.5)
                                    # Verify again
                                    if not model_file or not os.path.exists(model_file):
                                        raise FileNotFoundError(f"Model file not found after download attempt: {model_file}")
                                status_label.setText("Model downloaded, loading...")
                                QApplication.processEvents()
                            except HTTPError as http_err:
                                # Handle 404 errors (model not found in repository)
                                if http_err.code == 404:
                                    # Check if model exists in cache
                                    model_file = find_model_file(current_model_type, 0, True)
                                    if model_file and os.path.exists(model_file):
                                        status_label.setText("Using cached model (not in repo)...")
                                        QApplication.processEvents()
                                    else:
                                        # Model not available - show helpful error
                                        QMessageBox.warning(
                                            preview_dialog,
                                            "Model Not Available",
                                            f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                            f"Available Omnipose models:\n"
                                            f"  • bact_phase_omni (recommended for bacteria)\n"
                                            f"  • bact_fluor_omni\n"
                                            f"  • bact_phase_affinity\n\n"
                                            f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                            f"Please select a different model from the dropdown."
                                        )
                                        status_label.setText("Model not available")
                                        return False
                                else:
                                    # Other HTTP errors - check local directory one more time
                                    model_file = find_model_file(current_model_type, 0, True)
                                    if model_file and os.path.exists(model_file):
                                        status_label.setText("Download failed, using local/cached model...")
                                        QApplication.processEvents()
                                    else:
                                        # Re-raise to be handled by outer exception handler
                                        raise
                            except Exception as download_err:
                                # Check one more time if model exists (might have been downloaded in background)
                                model_file = find_model_file(current_model_type, 0, True)
                                if model_file and os.path.exists(model_file):
                                    status_label.setText("Using model from cache...")
                                    QApplication.processEvents()
                                else:
                                    # Re-raise to be handled by outer exception handler
                                    raise
                        
                        # Final verification: ensure model file exists before loading
                        if not model_file or not os.path.exists(model_file):
                            raise FileNotFoundError(
                                f"Model file not found: {model_file}\n\n"
                                f"Model '{current_model_type}' may not be available or download failed.\n"
                                f"Please check your internet connection and try again."
                            )
                        
                        status_label.setText("Loading model...")
                        QApplication.processEvents()
                        
                        preview_dialog.model = models.Cellpose(
                            gpu=current_use_gpu, 
                            model_type=current_model_type, 
                            omni=True
                        )
                        
                        # CRITICAL: Patch omnipose.core.get_masks to handle bd=None
                        # In frozen builds, the boundary field (bd) can be None if model files
                        # aren't found, causing "None > -1" errors. This patch provides a fallback.
                        try:
                            import omnipose.core as oc
                            import sys
                            import numpy as np
                            
                            # First try the global patch function
                            is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                            if is_frozen:
                                _patch_omnipose_get_masks()
                            
                            # Also apply local patch if global one didn't work
                            if is_frozen and not hasattr(oc.get_masks, '_bactimate_patched'):
                                _orig_get_masks = oc.get_masks
                                
                                def get_masks_safe(*args, **kwargs):
                                    """
                                    Patched get_masks that handles bd=None in frozen builds.
                                    The boundary field (bd) can be None, causing "None > -1" errors.
                                    This patch provides a fallback array.
                                    """
                                    # bd is typically the second positional argument (index 1)
                                    if len(args) >= 2:
                                        bd = args[1]
                                        if bd is None:
                                            # Need to create a fallback array with the right shape
                                            # Try to get shape from dist (3rd arg), mask (4th arg), or p (1st arg)
                                            shape = None
                                            
                                            # Try dist field (dt_pad) - 3rd argument
                                            if len(args) >= 3 and args[2] is not None and hasattr(args[2], 'shape'):
                                                shape = args[2].shape
                                            # Try mask field (iscell_pad) - 4th argument  
                                            elif len(args) >= 4 and args[3] is not None and hasattr(args[3], 'shape'):
                                                shape = args[3].shape
                                            # Try p field - 1st argument
                                            elif len(args) >= 1 and args[0] is not None and hasattr(args[0], 'shape'):
                                                shape = args[0].shape
                                            
                                            # ALWAYS create a fallback if bd is None, even if shape detection fails
                                            if shape is not None:
                                                # Create fallback array with values < -1 so (bd > -1) is False everywhere
                                                # This effectively disables boundary-based suppression
                                                bd = np.full(shape, -2, dtype=np.float32)
                                            else:
                                                # Last resort: try to infer shape from any available array
                                                # This should rarely happen, but better than crashing
                                                for arg in args:
                                                    if arg is not None and hasattr(arg, 'shape') and len(arg.shape) >= 2:
                                                        shape = arg.shape
                                                        bd = np.full(shape, -2, dtype=np.float32)
                                                        break
                                                
                                                # If still no shape, create a minimal fallback (shouldn't happen in practice)
                                                if bd is None:
                                                    # This is a last resort - create a small array
                                                    # The original function will likely fail anyway, but at least we tried
                                                    bd = np.array([[-2.0]], dtype=np.float32)
                                            
                                            # Replace bd in args - CRITICAL: must do this even if shape detection failed
                                            args = list(args)
                                            args[1] = bd
                                            args = tuple(args)
                                    
                                    # Final safety check: ensure bd is never None before calling original function
                                    if len(args) >= 2 and args[1] is None:
                                        # This should never happen, but if it does, create a minimal fallback
                                        args = list(args)
                                        args[1] = np.array([[-2.0]], dtype=np.float32)
                                        args = tuple(args)
                                    
                                    return _orig_get_masks(*args, **kwargs)
                                
                                # Mark as patched to avoid recursion
                                get_masks_safe._bactimate_patched = True
                                oc.get_masks = get_masks_safe
                        except Exception:
                            get_logger().debug("Patching get_masks failed, continuing", exc_info=True)
                        
                        preview_dialog._last_model_type = current_model_type
                        preview_dialog._last_use_gpu = current_use_gpu
                        status_label.setText("Model loaded")
                        QApplication.processEvents()
                    except RuntimeError as e:
                        error_msg = str(e)
                        # Check if it's a model architecture mismatch
                        if "size mismatch" in error_msg or "state_dict" in error_msg or "checkpoint" in error_msg.lower() or "Unexpected key" in error_msg:
                            # Check if there's a corrupted or wrong model file (e.g., cp3 symlinked as omni)
                            from cellpose_omni.models import model_path, MODEL_DIR
                            import os
                            from pathlib import Path
                            expected_basename = f'{current_model_type}torch_0'
                            model_file_path = Path(MODEL_DIR) / expected_basename
                            
                            # Check if model file is a symlink pointing to wrong model (e.g., cp3 instead of omni)
                            if model_file_path.exists() and model_file_path.is_symlink():
                                try:
                                    target = model_file_path.readlink()
                                    # If symlink points to cp3 model but we need omni, remove it
                                    if '_cp3' in str(target) and '_omni' in current_model_type:
                                        model_file_path.unlink()
                                        status_label.setText("Removed incorrect model symlink...")
                                        QApplication.processEvents()
                                except Exception:
                                    get_logger().debug("Could not remove incorrect model symlink", exc_info=True)
                            
                            # Try to fix by clearing model cache and retrying
                            try:
                                model_file = model_path(current_model_type, 0, True)
                                if os.path.exists(model_file):
                                    status_label.setText("Fixing model architecture mismatch...")
                                    QApplication.processEvents()
                                    # Backup the corrupted model
                                    backup_file = model_file + ".corrupted"
                                    if os.path.exists(backup_file):
                                        os.remove(backup_file)
                                    os.rename(model_file, backup_file)
                                    # Retry loading model (will download if needed)
                                    status_label.setText("Re-loading model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_type = current_model_type
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Model reloaded successfully!")
                                    QApplication.processEvents()
                                else:
                                    raise RuntimeError("Model file not found after clearing cache")
                            except Exception as retry_error:
                                error_details = f"Original error: {error_msg}\nRetry error: {retry_error}"
                                # Provide helpful error message
                                help_text = (
                                    f"The model checkpoint has a different architecture than expected.\n\n"
                                    f"{error_details}\n\n"
                                )
                                
                                # Check if user has cp3 models that might be causing confusion
                                script_dir = PROJECT_ROOT
                                local_models_dir = script_dir / 'models'
                                has_cp3 = (local_models_dir / 'bact_phase_cp3').exists() if local_models_dir.exists() else False
                                
                                if has_cp3 and '_omni' in current_model_type:
                                    help_text += (
                                        f"⚠️ IMPORTANT: You have 'bact_phase_cp3' in your models directory, "
                                        f"but this is a Cellpose model, NOT an Omnipose model.\n"
                                        f"They have different architectures and cannot be used interchangeably.\n\n"
                                    )
                                
                                help_text += (
                                    f"Please try:\n"
                                    f"1. Wait for the model server to come back online (currently HTTP 500 error)\n"
                                    f"2. Try a different model type\n"
                                    f"3. If you have cp3 models, use them with Cellpose model types, not Omnipose\n"
                                    f"4. Delete corrupted models: rm -rf {MODEL_DIR}/*.corrupted\n"
                                    f"5. Restart the application"
                                )
                                
                                QMessageBox.critical(
                                    preview_dialog, 
                                    "Model Architecture Mismatch",
                                    help_text
                                )
                                status_label.setText("Model load failed")
                                return False
                        else:
                            QMessageBox.critical(preview_dialog, "Model Error", 
                                               f"Could not load model:\n{error_msg}")
                            status_label.setText("Model load failed")
                        return False
                    except HTTPError as http_err:
                        # Handle HTTP errors (network issues when downloading models)
                        error_msg = str(http_err)
                        
                        # Special handling for 404 errors (model not found)
                        if http_err.code == 404:
                            model_file = find_model_file(current_model_type, 0, True)
                            if model_file and os.path.exists(model_file):
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Using cached model (not in repo)...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_type = current_model_type
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Model Not Available",
                                        f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                        f"Available Omnipose models:\n"
                                        f"  • bact_phase_omni (recommended for bacteria)\n"
                                        f"  • bact_fluor_omni\n"
                                        f"  • bact_phase_affinity\n\n"
                                        f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                        f"Please select a different model from the dropdown."
                                    )
                                    status_label.setText("Model not available")
                                    return False
                            else:
                                # Model not available and not cached
                                QMessageBox.warning(
                                    preview_dialog, 
                                    "Model Not Available",
                                    f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                    f"Available Omnipose models:\n"
                                    f"  • bact_phase_omni (recommended for bacteria)\n"
                                    f"  • bact_fluor_omni\n"
                                    f"  • bact_phase_affinity\n\n"
                                    f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                    f"Please select a different model from the dropdown."
                                )
                                status_label.setText("Model not available")
                                return False
                        
                        # Check for model in local directory or cache without triggering another download
                        model_file = find_model_file(current_model_type, 0, True)
                        cached_model_exists = (model_file is not None and os.path.exists(model_file))
                        
                        if cached_model_exists:
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Trying cached model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_type = current_model_type
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Network Error & Cache Issue",
                                        f"Could not download model (server error) and cached model also failed:\n\n"
                                        f"Download error: {error_msg}\n"
                                        f"Cache error: {cache_error}\n\n"
                                        f"Please try:\n"
                                        f"1. Wait a few minutes and try again (server may be temporarily down)\n"
                                        f"2. Try a different model type\n"
                                        f"3. Check your internet connection\n"
                                        f"4. If persistent, clear cache: rm -rf ~/.cellpose/models/"
                                    )
                                    status_label.setText("Network & cache error")
                                    return False
                        else:
                            # No cached model available
                            QMessageBox.warning(
                                preview_dialog, 
                                "Network Error - Model Server Unavailable",
                                f"Could not download model due to network error:\n\n{error_msg}\n\n"
                                f"The model server appears to be temporarily unavailable (HTTP 500).\n\n"
                                f"Please try:\n"
                                f"1. Wait 5-10 minutes and try again (server may be temporarily down)\n"
                                f"2. Check your internet connection\n"
                                f"3. Try a different model type (may already be cached)\n"
                                f"4. Check the Cellpose website status: https://www.cellpose.org\n\n"
                                f"Note: Once downloaded, models are cached locally and won't need to be re-downloaded."
                            )
                            status_label.setText("Server unavailable - try again later")
                            return False
                    except Exception as e:
                        error_msg = str(e)
                        # Handle other HTTP/network errors that might be wrapped
                        if "HTTP" in error_msg or "500" in error_msg or "connection" in error_msg.lower() or "network" in error_msg.lower() or "urllib" in error_msg.lower() or "download" in error_msg.lower():
                            # Check for model in local directory or cache without triggering another download
                            model_file = find_model_file(current_model_type, 0, True)
                            cached_model_exists = (model_file is not None)
                            
                            if cached_model_exists:
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Trying cached model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_type = current_model_type
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Network Error & Cache Issue",
                                        f"Could not download model (server error) and cached model also failed:\n\n"
                                        f"Download error: {error_msg}\n"
                                        f"Cache error: {cache_error}\n\n"
                                        f"Please try:\n"
                                        f"1. Wait a few minutes and try again (server may be temporarily down)\n"
                                        f"2. Try a different model type\n"
                                        f"3. Check your internet connection\n"
                                        f"4. If persistent, clear cache: rm -rf ~/.cellpose/models/"
                                    )
                                    status_label.setText("Network & cache error")
                                    return False
                            else:
                                # No cached model available
                                QMessageBox.warning(
                                    preview_dialog, 
                                    "Network Error - Model Server Unavailable",
                                    f"Could not download model due to network error:\n\n{error_msg}\n\n"
                                    f"The model server appears to be temporarily unavailable (HTTP 500).\n\n"
                                    f"Please try:\n"
                                    f"1. Wait 5-10 minutes and try again (server may be temporarily down)\n"
                                    f"2. Check your internet connection\n"
                                    f"3. Try a different model type (may already be cached)\n"
                                    f"4. Check the Cellpose website status: https://www.cellpose.org\n\n"
                                    f"Note: Once downloaded, models are cached locally and won't need to be re-downloaded."
                                )
                                status_label.setText("Server unavailable - try again later")
                                return False
                        else:
                            QMessageBox.critical(preview_dialog, "Model Error", 
                                               f"Could not load model:\n{error_msg}")
                            status_label.setText("Model load failed")
                        return False
                return True
            
            def get_current_parameters():
                """Get current parameters from main window (works because dialog is non-modal)"""
                # CRITICAL: In frozen builds, diameter=0 (auto-detect) causes None comparison errors
                # in omnipose/core.py. Use a default diameter instead of 0.
                import sys
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                
                diameter_value = self.diameter.value()
                model_type = self.model_combo.currentText()
                
                # If diameter is 0 and we're in a frozen build, use a model-specific default
                if diameter_value == 0 and is_frozen:
                    # Default diameters based on model type (common values for bacteria)
                    default_diameters = {
                        'bact_phase_omni': 30.0,
                        'bact_omni': 30.0,
                        'bact_fluor_omni': 30.0,
                    }
                    diameter_value = default_diameters.get(model_type, 30.0)
                
                return {
                    'diameter': diameter_value if diameter_value > 0 else 0,  # Use 0 instead of None
                    'flow_threshold': self.flow_threshold.value(),
                    'mask_threshold': self.mask_threshold.value(),
                    'cluster': self.cluster.isChecked(),
                    'model_type': model_type,
                    'use_gpu': self.use_gpu.isChecked(),
                }
            
            # Cache for segmentation results: key = (frame_idx, diameter, flow_threshold, mask_threshold, cluster, model_type, use_gpu)
            preview_dialog.segmentation_cache = {}
            
            def get_cache_key(frame_idx, params):
                """Generate cache key from frame index and parameters"""
                # Ensure diameter is always a number (0 if None) for cache key consistency
                diameter = params.get('diameter', 0)
                if diameter is None or not isinstance(diameter, (int, float)):
                    diameter = 0.0
                diameter = float(diameter)
                
                flow_threshold = params.get('flow_threshold', 0.4)
                if flow_threshold is None or not isinstance(flow_threshold, (int, float)):
                    flow_threshold = 0.4
                flow_threshold = float(flow_threshold)
                mask_threshold = params.get('mask_threshold', 0.0)
                if mask_threshold is None or not isinstance(mask_threshold, (int, float)):
                    mask_threshold = 0.0
                mask_threshold = float(mask_threshold)
                
                return (frame_idx, diameter, flow_threshold, mask_threshold,
                       bool(params.get('cluster', False)), 
                       str(params.get('model_type', 'bact_phase_omni')), 
                       bool(params.get('use_gpu', True)))
            
            # Segmentation worker thread class
            class SegmentationWorker(QThread):
                finished = pyqtSignal(object, object)  # mask, error
                
                def __init__(self, model, frame, diameter, flow_threshold, mask_threshold, cluster):
                    super().__init__()
                    self.model = model
                    self.frame = frame
                    self.diameter = diameter
                    self.flow_threshold = flow_threshold
                    self.mask_threshold = mask_threshold
                    self.cluster = cluster
                
                def run(self):
                    try:
                        # CRITICAL: Ensure all parameters are valid numbers, not None
                        # This prevents comparison errors in model.eval() which expects numbers
                        diameter = self.diameter
                        if diameter is None or not isinstance(diameter, (int, float)):
                            diameter = 0
                        diameter = float(diameter)
                        
                        flow_threshold = self.flow_threshold
                        if flow_threshold is None or not isinstance(flow_threshold, (int, float)):
                            flow_threshold = 0.4
                        flow_threshold = float(flow_threshold)
                        mask_threshold = self.mask_threshold
                        if mask_threshold is None or not isinstance(mask_threshold, (int, float)):
                            mask_threshold = 0.0
                        mask_threshold = float(mask_threshold)
                        
                        mask, flows, styles, diams = self.model.eval(
                            self.frame,
                            channels=[0, 0],
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            mask_threshold=mask_threshold,
                            cellprob_threshold=0.0,
                            do_3D=False,
                            omni=True,
                            cluster=self.cluster,
                            net_avg=False,  # Faster, omni models don't support net_avg anyway
                            resample=True,
                            verbose=False
                        )
                        self.finished.emit(mask, None)
                    except Exception as e:
                        self.finished.emit(None, str(e))
            
            def run_segmentation(frame_idx, use_cache=True):
                """Run segmentation on a specific frame with caching and background threading"""
                # Get current parameters
                params = get_current_parameters()
                
                # CRITICAL: Clean parameters IMMEDIATELY - ensure no None values anywhere
                # This must happen before cache key generation or any other operations
                if params.get('diameter') is None or not isinstance(params.get('diameter'), (int, float)):
                    params['diameter'] = 0.0
                else:
                    params['diameter'] = float(params['diameter'])
                
                if params.get('flow_threshold') is None or not isinstance(params.get('flow_threshold'), (int, float)):
                    params['flow_threshold'] = 0.4
                else:
                    params['flow_threshold'] = float(params['flow_threshold'])
                if params.get('mask_threshold') is None or not isinstance(params.get('mask_threshold'), (int, float)):
                    params['mask_threshold'] = 0.0
                else:
                    params['mask_threshold'] = float(params['mask_threshold'])
                
                # Now generate cache key with clean parameters
                cache_key = get_cache_key(frame_idx, params)
                
                # Check cache first (safely - handle any old cache entries with None values)
                if use_cache:
                    try:
                        if cache_key in preview_dialog.segmentation_cache:
                            return preview_dialog.segmentation_cache[cache_key]
                    except (TypeError, ValueError) as e:
                        # Cache key comparison failed (likely old entries with None values)
                        # Clean cache and continue
                        preview_dialog.segmentation_cache.clear()
                
                # Load model if needed (only when model type or GPU setting changes)
                if not load_model():
                    return None
                
                frame = images[frame_idx]
                # Handle multi-channel: use first channel if needed
                if frame.ndim == 3:
                    frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                
                # Ensure frame is float32 for MPS compatibility (Apple Silicon GPU doesn't support float64)
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32)
                
                # Extract parameters (already validated and converted above)
                diameter = params['diameter']
                flow_threshold = params['flow_threshold']
                mask_threshold = params['mask_threshold']
                cluster = params['cluster']
                
                try:
                    # Final validation before calling model.eval()
                    # Double-check all parameters are valid numbers (defensive programming)
                    assert isinstance(diameter, (int, float)) and diameter is not None, f"Invalid diameter: {diameter}"
                    assert isinstance(flow_threshold, (int, float)) and flow_threshold is not None, f"Invalid flow_threshold: {flow_threshold}"
                    assert isinstance(mask_threshold, (int, float)) and mask_threshold is not None, f"Invalid mask_threshold: {mask_threshold}"
                    
                    # Run segmentation (synchronously for now, but we can make it async later)
                    # Pass all parameters explicitly to match working batch_process_tiff_sequence.py
                    # Use a wrapper that catches the None comparison error and retries with workaround
                    import sys
                    is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                    
                    if is_frozen:
                        # In frozen builds, wrap the call to handle the omnipose bug
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                mask, flows, styles, diams = preview_dialog.model.eval(
                                    frame,
                                    channels=[0, 0],
                                    diameter=diameter if diameter > 0 else 30.0,  # Never use 0 in frozen builds
                                    flow_threshold=flow_threshold,
                                    mask_threshold=mask_threshold,
                                    cellprob_threshold=0.0,
                                    do_3D=False,
                                    omni=True,
                                    cluster=cluster,
                                    net_avg=False,
                                    resample=True,
                                    verbose=False
                                )
                                break  # Success, exit retry loop
                            except TypeError as e:
                                if "'>' not supported between instances of 'NoneType'" in str(e) or "'<' not supported between instances of 'NoneType'" in str(e):
                                    if attempt < max_retries - 1:
                                        # Try with a different diameter
                                        diameter = 30.0 + (attempt * 5.0)  # Try 30, 35, 40
                                        continue
                                    else:
                                        # Last attempt failed, raise the error
                                        raise
                                else:
                                    # Different TypeError, raise immediately
                                    raise
                    else:
                        # Not frozen build, use normal call
                        mask, flows, styles, diams = preview_dialog.model.eval(
                            frame,
                            channels=[0, 0],
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            mask_threshold=mask_threshold,
                            cellprob_threshold=0.0,
                            do_3D=False,
                            omni=True,
                            cluster=cluster,
                            net_avg=False,
                            resample=True,
                            verbose=False
                        )
                    # Cache the result
                    preview_dialog.segmentation_cache[cache_key] = mask
                    return mask
                except RecursionError as e:
                    # Handle recursion errors (e.g., from broken runtime hooks)
                    import traceback
                    error_details = f"Recursion error detected:\n{str(e)}\n\n"
                    error_details += f"This may be caused by a bug in the frozen build.\n"
                    error_details += f"Try rebuilding the application.\n\n"
                    error_details += f"Parameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\n"
                    error_details += f"Traceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error - Recursion", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
                except TypeError as e:
                    # Handle the specific "'>' not supported between instances of 'NoneType' and 'int'" error
                    # This is a known bug in omnipose when running in frozen builds
                    if "'>' not supported between instances of 'NoneType'" in str(e) or "'<' not supported between instances of 'NoneType'" in str(e):
                        import sys
                        is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                        
                        # This error occurs in frozen builds due to a bug in omnipose/core.py
                        # Try different diameter values as workaround
                        retry_diameters = [30.0, 25.0, 35.0, 20.0, 40.0] if is_frozen else []
                        
                        if retry_diameters:
                            for retry_diameter in retry_diameters:
                                if retry_diameter == diameter:
                                    continue  # Skip if we already tried this value
                                
                                try:
                                    QMessageBox.warning(preview_dialog, "Omnipose Bug Workaround", 
                                                      f"Encountered a known bug in omnipose (frozen build compatibility issue).\n"
                                                      f"Retrying with diameter={retry_diameter} pixels...")
                                    mask, flows, styles, diams = preview_dialog.model.eval(
                                        frame,
                                        channels=[0, 0],
                                        diameter=retry_diameter,
                                        flow_threshold=flow_threshold,
                                        mask_threshold=mask_threshold,
                                        cellprob_threshold=0.0,
                                        do_3D=False,
                                        omni=True,
                                        cluster=cluster,
                                        net_avg=False,
                                        resample=True,
                                        verbose=False
                                    )
                                    # Cache the result with the new diameter
                                    cache_key_retry = get_cache_key(frame_idx, {'diameter': retry_diameter, 'flow_threshold': flow_threshold,
                                                                              'mask_threshold': mask_threshold, 'cluster': cluster, 'model_type': params.get('model_type'),
                                                                              'use_gpu': params.get('use_gpu')})
                                    preview_dialog.segmentation_cache[cache_key_retry] = mask
                                    return mask
                                except (TypeError, Exception) as e2:
                                    # If this diameter also fails, try next one
                                    if retry_diameter == retry_diameters[-1]:
                                        # Last attempt failed, show error
                                        import traceback
                                        error_details = f"Known omnipose bug in frozen builds.\n\n"
                                        error_details += f"Original error: {str(e)}\n\n"
                                        error_details += f"Tried diameters: {[diameter] + retry_diameters}\n"
                                        error_details += f"All attempts failed. Last error: {str(e2)}\n\n"
                                        error_details += f"Workaround: Please set a specific diameter value (e.g., 30 pixels) instead of 0 (auto-detect).\n\n"
                                        error_details += f"Full traceback:\n{traceback.format_exc()}"
                                        QMessageBox.critical(preview_dialog, "Segmentation Error - Omnipose Bug", 
                                                           f"Could not segment frame:\n{error_details}")
                                        return None
                                    continue
                        else:
                            # Not a frozen build, or no retry diameters
                            import traceback
                            error_details = f"TypeError in omnipose: {str(e)}\n\n"
                            error_details += f"This may be a bug in omnipose. Please try setting a specific diameter value.\n\n"
                            error_details += f"Traceback:\n{traceback.format_exc()}"
                            QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                               f"Could not segment frame:\n{error_details}")
                            return None
                    else:
                        # Different TypeError, re-raise
                        raise
                except AssertionError as e:
                    # Parameter validation error
                    import traceback
                    error_details = f"Parameter validation error:\n{str(e)}\n\nParameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\nTraceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
                except Exception as e:
                    # Other errors - include full traceback for debugging
                    import traceback
                    error_details = f"{str(e)}\n\nParameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\nTraceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
            
            def update_display(frame_idx, show_progress=True, force_recalc=False):
                """Update the display with segmentation overlay"""
                # Check cache first
                params = get_current_parameters()
                cache_key = get_cache_key(frame_idx, params)
                has_cache = cache_key in preview_dialog.segmentation_cache and not force_recalc
                
                # Show progress dialog if we need to calculate
                progress_dialog = None
                if show_progress and not has_cache:
                    progress_dialog = QProgressDialog("Segmenting frame...", None, 0, 0, preview_dialog)
                    progress_dialog.setWindowTitle("Processing")
                    progress_dialog.setAutoClose(False)
                    progress_dialog.setAutoReset(False)
                    progress_dialog.setCancelButton(None)  # Don't allow cancellation
                    progress_dialog.show()
                    QApplication.processEvents()
                
                # Disable controls during processing
                if progress_dialog:
                    resegment_btn.setEnabled(False)
                    clear_cache_btn.setEnabled(False)
                    frame_spin.setEnabled(False)
                
                try:
                    frame = images[frame_idx]
                    # Handle multi-channel: use first channel if needed
                    if frame.ndim == 3:
                        display_frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                    else:
                        display_frame = frame
                    
                    # Update progress message
                    if progress_dialog:
                        progress_dialog.setLabelText("Running segmentation...")
                        QApplication.processEvents()
                    
                    # Run segmentation (with caching)
                    mask = run_segmentation(frame_idx, use_cache=not force_recalc)
                    if mask is None:
                        if progress_dialog:
                            progress_dialog.close()
                        status_label.setText("Segmentation failed")
                        status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
                        return
                    
                    preview_dialog.masks = mask
                    
                    # Update progress message and let UI breathe
                    if progress_dialog:
                        progress_dialog.setLabelText("Rendering display...")
                    QApplication.processEvents()

                    # Scale down for preview so overlay + pixmap + paint stay fast
                    disp_scaled, mask_scaled = scale_for_preview(display_frame, mask)
                    overlay = build_segmentation_overlay(disp_scaled, mask_scaled, alpha=0.5)
                    pixmap = to_qpixmap(overlay)
                    height, width = overlay.shape[:2]
                    image_canvas.set_pixmap(pixmap, (width, height), reset_view=False)
                    QApplication.processEvents()

                    file_name = Path(file_path).name
                    num_cells = len(np.unique(mask)) - 1
                    title_label.setText(
                        f"Frame {frame_idx + 1} of {num_frames} | Cells detected: {num_cells} | {file_name}"
                    )
                    
                    # Close progress dialog
                    if progress_dialog:
                        progress_dialog.close()
                    
                    # Re-enable controls
                    resegment_btn.setEnabled(True)
                    clear_cache_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    
                    # Update status
                    cache_size = len(preview_dialog.segmentation_cache)
                    if has_cache:
                        status_label.setText(f"Ready (from cache)")
                    else:
                        status_label.setText(f"Ready (cached: {cache_size})")
                    status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                    # Re-enable controls on error
                    resegment_btn.setEnabled(True)
                    clear_cache_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    status_label.setText(f"Error: {str(e)}")
                    status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
                    QMessageBox.warning(preview_dialog, "Preview Error", f"Could not update display:\n{str(e)}")
            
            # Frame navigation (spinbox only; no prev/next buttons)
            nav_layout = QHBoxLayout()
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_frames)
            frame_spin.setValue(1)
            nav_layout.addWidget(QLabel("Frame:"))
            nav_layout.addWidget(frame_spin)
            nav_layout.addWidget(QLabel(f"of {num_frames}"))
            
            # Add status label to navigation layout
            nav_layout.addStretch()
            nav_layout.addWidget(status_label)
            
            # Track if we're currently updating to prevent concurrent updates
            is_updating_frame = False
            
            def go_to_frame(idx, force_recalc=False):
                nonlocal is_updating_frame
                # Prevent concurrent frame updates
                if is_updating_frame:
                    return
                
                if not (0 <= idx < num_frames):
                    # Invalid frame index - reset to current frame
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
                    return
                
                is_updating_frame = True
                preview_dialog.current_frame = idx
                # Block signals to prevent recursive calls
                frame_spin.blockSignals(True)
                frame_spin.setValue(idx + 1)
                frame_spin.blockSignals(False)
                
                try:
                    # Check if we have cached result for this frame with current parameters
                    params = get_current_parameters()
                    cache_key = get_cache_key(idx, params)
                    has_cache = cache_key in preview_dialog.segmentation_cache
                    
                    # Only show progress if we need to recalculate
                    update_display(idx, show_progress=(not has_cache or force_recalc))
                except Exception as e:
                    import traceback
                    error_msg = f"Could not display frame {idx + 1}:\n{str(e)}"
                    if hasattr(e, '__traceback__'):
                        error_msg += f"\n\n{traceback.format_exc()}"
                    QMessageBox.warning(preview_dialog, "Preview Error", error_msg)
                finally:
                    is_updating_frame = False
            
            # editingFinished: when user presses Enter or leaves field. valueChanged + 300ms debounce: for arrow keys.
            def on_frame_changed():
                if hasattr(frame_spin, '_change_timer'):
                    frame_spin._change_timer.stop()
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    go_to_frame(frame_num)
                else:
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)

            frame_spin.editingFinished.connect(on_frame_changed)
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)

            def on_value_changed(v):
                if 1 <= v <= num_frames:
                    frame_spin._change_timer.start(300)
                else:
                    frame_spin._change_timer.stop()
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
            frame_spin.valueChanged.connect(on_value_changed)
            
            nav_layout.addStretch()
            
            # Re-segment button (to update with current parameters from main window)
            resegment_btn = QPushButton("Re-segment")
            resegment_btn.setToolTip(
                "Re-run segmentation on the current frame using the current parameters from the main window.\n"
                "You can change parameters in the main window (it stays accessible) and click this button to see the effect.\n"
                "Results are cached - changing parameters and clicking again will use cached results if parameters match."
            )
            def on_resegment():
                # Clear cache for current frame to force recalculation
                params = get_current_parameters()
                cache_key = get_cache_key(preview_dialog.current_frame, params)
                if cache_key in preview_dialog.segmentation_cache:
                    del preview_dialog.segmentation_cache[cache_key]
                # Force recalculation with progress
                update_display(preview_dialog.current_frame, show_progress=True, force_recalc=True)
            resegment_btn.clicked.connect(on_resegment)
            nav_layout.addWidget(resegment_btn)
            
            # Clear cache button
            clear_cache_btn = QPushButton("Clear Cache")
            clear_cache_btn.setToolTip(
                "Clear all cached segmentation results.\n"
                "Use this if you want to force recalculation of all frames."
            )
            def on_clear_cache():
                preview_dialog.segmentation_cache.clear()
                status_label.setText("Cache cleared")
                status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                QTimer.singleShot(2000, lambda: status_label.setText("Ready") or status_label.setStyleSheet("color: #00FF00; font-weight: bold;"))
            clear_cache_btn.clicked.connect(on_clear_cache)
            nav_layout.addWidget(clear_cache_btn)
            
            layout.addLayout(nav_layout)

            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)

            # Add Cmd+W (or Ctrl+W) shortcut to close dialog
            close_shortcut = QShortcut(QKeySequence("Ctrl+W"), preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            if sys.platform == 'darwin':
                close_shortcut_mac = QShortcut(QKeySequence("Meta+W"), preview_dialog)
                close_shortcut_mac.activated.connect(preview_dialog.close)

            preview_dialog.setLayout(layout)

            # Show first frame immediately (image only), scaled for fast display
            frame0 = images[0]
            if frame0.ndim == 3:
                display_frame0 = frame0[0] if frame0.shape[0] < 10 else frame0[:, :, 0]
            else:
                display_frame0 = frame0
            empty_mask = np.zeros(display_frame0.shape, dtype=np.int32)
            disp0, _ = scale_for_preview(display_frame0, empty_mask)
            empty_scaled = np.zeros(disp0.shape, dtype=np.int32)
            initial_overlay = build_segmentation_overlay(disp0, empty_scaled, alpha=0.0)
            initial_pixmap = to_qpixmap(initial_overlay)
            h0, w0 = initial_overlay.shape[:2]
            image_canvas.set_pixmap(initial_pixmap, (w0, h0), reset_view=True)
            file_name = Path(file_path).name
            title_label.setText(f"Frame 1 of {num_frames} | Segmenting... | {file_name}")
            status_label.setText("Segmenting first frame...")
            loading_progress.close()
            preview_dialog.show()
            preview_dialog.raise_()
            preview_dialog.activateWindow()

            def start_preview():
                preview_dialog.raise_()
                preview_dialog.activateWindow()
                update_display(0, show_progress=True)
                # Clear "stay on top" after first display so the window doesn't stay on top forever
                def clear_stay_on_top():
                    preview_dialog.setWindowFlags(
                        preview_dialog.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint
                    )
                    preview_dialog.show()  # needed after changing flags
                QTimer.singleShot(1500, clear_stay_on_top)
            QTimer.singleShot(0, start_preview)

            # Show cache info in status
            def update_cache_status():
                cache_size = len(preview_dialog.segmentation_cache)
                if cache_size > 0:
                    status_label.setText(f"Ready (cached: {cache_size})")
                else:
                    status_label.setText("Ready")

            QTimer.singleShot(100, update_cache_status)
            
        except ImportError as e:
            loading_progress.close()
            error_msg = str(e)
            # Handle specific import errors
            if 'torch.testing' in error_msg:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  "Preview requires torch.testing module.\n\n"
                                  "This is a known issue with frozen builds.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try running from source: python main.py")
            else:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  f"Preview requires additional libraries:\n\n{error_msg}\n\n"
                                  f"Install with: pip install cellpose-omni")
        except Exception as e:
            loading_progress.close()
            error_msg = str(e)
            # Handle PyTorch RpcBackendOptions duplicate initialization error
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "PyTorch initialization error detected.\n\n"
                                  "This is a known issue with frozen builds and PyTorch.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try:\n"
                                  "1. Restart the application\n"
                                  "2. Run from source: python main.py\n"
                                  "3. Use the processing features instead of preview")
            # Handle pbr versioning error (mgen package version detection issue)
            elif 'Versioning for this project requires' in error_msg or 'pbr.version.VersionInfo' in error_msg or 'mgen' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "Package versioning error detected.\n\n"
                                  "This is a known issue with frozen builds and the mgen package.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try:\n"
                                  "1. Restart the application\n"
                                  "2. Run from source: python main.py\n"
                                  "3. Use the processing features instead of preview\n\n"
                                  "This error is harmless and doesn't affect actual processing.")
            # Handle TorchScript source access error
            elif 'TorchScript requires source access' in error_msg or "Can't get source for" in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "TorchScript compilation error detected.\n\n"
                                  "This is a known issue with frozen builds and PyTorch's TorchScript.\n"
                                  "TorchScript requires access to original source files which aren't\n"
                                  "available in packaged applications.\n\n"
                                  "Try:\n"
                                  "1. Restart the application (the runtime hook should fix this)\n"
                                  "2. Run from source: python main.py\n"
                                  "3. Use the processing features instead of preview\n\n"
                                  "Note: This has been fixed in the latest build. Please rebuild the app.")
            else:
                QMessageBox.critical(self, "Preview Error", f"Could not preview segmentation:\n{error_msg}")
            import traceback
            traceback.print_exc()
        except BaseException as e:
            # PyQt6 is strict about exceptions in slots - catch ALL exceptions (including SystemExit, KeyboardInterrupt)
            # This outer try-except catches any exceptions not handled by inner handlers
            import traceback
            error_msg = f"Unexpected error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Preview Error", error_msg)
            # Don't re-raise - PyQt6 will abort if we do
        except BaseException as e:
            # Final catch-all for any exceptions that slip through
            get_logger().exception("Critical error in preview segmentation")
            import traceback
            error_msg = f"Critical error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            try:
                QMessageBox.critical(self, "Preview Error", error_msg)
            except Exception:
                get_logger().exception("Could not show preview error dialog")
                print(f"CRITICAL ERROR in preview_segmentation: {error_msg}", file=sys.stderr)
            # Don't re-raise - PyQt6 will abort if we do
    
    def _validate_for_run(self) -> list[str]:
        """Return a list of error messages; if non-empty, do not start the job."""
        errors: list[str] = []
        if not self.file_path.text():
            errors.append("Please select a TIFF file or folder first.")
            return errors
        path = Path(self.file_path.text())
        if not path.exists():
            errors.append(f"The selected path does not exist: {path}")
            return errors
        script_path = PROJECT_ROOT / "scripts" / "batch_process_tiff_sequence.py"
        if not script_path.exists():
            errors.append(f"Segmentation script not found: {script_path}")
        if self.diameter.value() < 0:
            errors.append("Diameter must be ≥ 0 (use 0 for auto-detect).")
        return errors

    def process_file_or_folder(self):
        errors = self._validate_for_run()
        if errors:
            QMessageBox.warning(
                self, "Cannot Start",
                "Please fix the following:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return

        script_dir = PROJECT_ROOT
        is_folder = self._is_folder_mode()
        
        if not is_folder:
            # Single file mode
            script_path = str(script_dir / "scripts" / "batch_process_tiff_sequence.py")
            input_file = Path(self.file_path.text())
            
            # Determine base directory (go up one level if input is in Tif_* folder)
            base_dir = input_file.parent
            if base_dir.name.startswith('Tif_'):
                base_dir = base_dir.parent
            
            # Create output directory at base level with _segmentation suffix
            output_dir = base_dir / f"{input_file.stem}_segmentation"
            
            cmd_args = [
                self.file_path.text(),
                '--output', str(output_dir),
                '--model', self.model_combo.currentText(),
                '--diameter', str(self.diameter.value()),
                '--flow-threshold', str(self.flow_threshold.value()),
                '--mask-threshold', str(self.mask_threshold.value()),
                '--batch-size', '32',  # Increased batch size for faster GPU processing (adjust if OOM errors occur)
            ]
            
            if not self.use_gpu.isChecked():
                cmd_args.append('--no-gpu')
            if self.cluster.isChecked():
                cmd_args.append('--cluster')
            if not self.create_overlay.isChecked():
                cmd_args.append('--no-overlay')
            
            self.process_btn.setEnabled(False)
            self.process_btn.setText("Processing...")
            self.parent_window.log_output(f"Starting single file processing: {self.file_path.text()}\n")
            self.parent_window.log_output(f"Output directory: {output_dir}\n")
        else:
            # Folder mode - show file selection dialog first
            folder = Path(self.file_path.text())
            # Find TIFF files (not mask files)
            tiff_files_set = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                            set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out mask files and output files
            tiff_files = sorted([f for f in tiff_files_set 
                                if '_masks' not in f.name.lower() 
                                and '_overlay' not in f.name.lower()
                                and '_filter_visualization' not in f.name.lower()], 
                               key=lambda x: x.name.lower())
            
            if not tiff_files:
                QMessageBox.warning(self, "No TIFF Files Found", 
                                  "No TIFF files found in the selected folder.\n\n"
                                  "Please select a folder containing TIFF files to process.")
                return
            
            # Show file selection dialog
            selected_files = show_file_selection_dialog(
                self,
                tiff_files,
                title="Select Files for Segmentation",
                message=f"Select files to process (all {len(tiff_files)} files selected by default):"
            )
            
            if selected_files is None:
                # User cancelled or no files selected
                return
            
            if len(selected_files) == 0:
                QMessageBox.warning(self, "No Files Selected", 
                                  "Please select at least one file to process.")
                return
            
            # Process selected files - use batch_process_folder.py but we'll need to filter files
            # For now, let's process each file individually (simpler approach)
            script_path = str(script_dir / "scripts" / "batch_process_tiff_sequence.py")
            
            # Store files and current index for sequential processing
            self.folder_files = selected_files
            self.current_file_index = 0
            # Track progress within current file (for batch/frame progress)
            self.current_file_progress = 0.0  # 0.0 to 1.0
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range for percentage display
            self.parent_window.progress_bar.setValue(0)
            
            self.process_btn.setEnabled(False)
            self.process_btn.setText(f"Processing... (0/{len(selected_files)})")
            self.parent_window.log_output(f"Starting batch folder processing: {len(selected_files)} file(s) selected from {self.file_path.text()}\n")
            
            # Process first file
            self._process_next_segmentation_file(script_path)
            return  # Early return, processing continues in _process_next_segmentation_file
        
        # Show progress bar
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range for percentage display
        self.parent_window.progress_bar.setValue(0)
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()
    
    def _process_next_segmentation_file(self, script_path):
        """Process next file in folder mode (segmentation)"""
        if self.current_file_index >= len(self.folder_files):
            # All files processed
            self.process_btn.setEnabled(True)
            is_folder = self._is_folder_mode()
            if is_folder:
                self.process_btn.setText("Process All Files in Folder")
            else:
                self.process_btn.setText("Process File")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f"\n✓ Segmentation complete for all {len(self.folder_files)} file(s)!\n")
            QMessageBox.information(self, "Processing Complete", 
                                  f"Successfully processed {len(self.folder_files)} file(s).")
            return
        
        tiff_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f"\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {tiff_file.name}\n")
        
        # Determine base directory (go up one level if input is in Tif_* folder)
        base_dir = tiff_file.parent
        if base_dir.name.startswith('Tif_'):
            base_dir = base_dir.parent
        
        # Create output directory at base level with _segmentation suffix
        output_dir = base_dir / f"{tiff_file.stem}_segmentation"
        
        # Build command arguments for this file
        cmd_args = [
            str(tiff_file),
            '--output', str(output_dir),
            '--model', self.model_combo.currentText(),
            '--diameter', str(self.diameter.value()),
            '--flow-threshold', str(self.flow_threshold.value()),
            '--mask-threshold', str(self.mask_threshold.value()),
            '--batch-size', '32',  # Increased batch size for faster GPU processing (adjust if OOM errors occur)
        ]
        
        if not self.use_gpu.isChecked():
            cmd_args.append('--no-gpu')
        if self.cluster.isChecked():
            cmd_args.append('--cluster')
        if not self.create_overlay.isChecked():
            cmd_args.append('--no-overlay')
        
        self.parent_window.log_output(f"Output directory: {output_dir}\n")
        
        # Reset current file progress when starting a new file
        self.current_file_progress = 0.0
        
        # Update progress bar to show file start
        file_progress = int((self.current_file_index / len(self.folder_files)) * 100)
        self.parent_window.progress_bar.setValue(file_progress)
        self.process_btn.setText(f"Processing... ({self.current_file_index + 1}/{len(self.folder_files)})")
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(lambda success, msg: self._on_segmentation_file_finished(success, msg, script_path))
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()
    
    def _on_segmentation_file_finished(self, success, message, script_path):
        """Handle completion of a single file in folder mode"""
        self.parent_window.log_output(message)
        if success:
            self.parent_window.log_output(f"✓ File {self.current_file_index + 1}/{len(self.folder_files)} processed successfully!\n")
            # Mark file as 100% complete
            self.current_file_progress = 1.0
        else:
            self.parent_window.log_output(f"✗ File {self.current_file_index + 1}/{len(self.folder_files)} failed!\n")
            # Still mark as complete (even if failed) for progress tracking
            self.current_file_progress = 1.0
        
        # Update progress bar to show file completion
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            file_portion = 100.0 / len(self.folder_files)
            file_progress_base = ((self.current_file_index + 1) / len(self.folder_files)) * 100
            self.parent_window.progress_bar.setValue(int(min(100, file_progress_base)))
        
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        
        # Move to next file
        self.current_file_index += 1
        self._process_next_segmentation_file(script_path)
    
    def _parse_segmentation_progress(self, message):
        """Parse progress messages from segmentation script and update progress bar"""
        import re
        # Log the message
        self.parent_window.log_output(message)
        
        # Update overall progress based on file, batch, and frame progress
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            # Initialize current_file_progress if not set
            if not hasattr(self, 'current_file_progress'):
                self.current_file_progress = 0.0
            
            # Handle model download progress (small portion of first file, ~5%)
            if 'Downloading:' in message or 'to /' in message and '.cellpose/models/' in message:
                download_match = re.search(r'(\d+)%', message)
                if download_match:
                    try:
                        download_pct = int(download_match.group(1))
                        # Model download is ~5% of first file's progress
                        if self.current_file_index == 0:
                            self.current_file_progress = (download_pct / 100.0) * 0.05
                    except (ValueError, AttributeError):
                        pass  # ignore progress parse errors
            
            # Try to parse batch completion: "✓ Batch 2/6 complete: 64/181 frames (35%)"
            batch_complete_match = re.search(r'Batch (\d+)/(\d+) complete', message)
            if batch_complete_match:
                try:
                    batch_num = int(batch_complete_match.group(1))
                    total_batches = int(batch_complete_match.group(2))
                    # Update progress: batch_num/total_batches (e.g., 2/6 = 0.33)
                    self.current_file_progress = batch_num / total_batches
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
            
            # Try to parse batch start: "Processing batch 3/6: frames 65-96 of 181"
            batch_start_match = re.search(r'Processing batch (\d+)/(\d+):', message)
            if batch_start_match and not batch_complete_match:
                try:
                    batch_num = int(batch_start_match.group(1))
                    total_batches = int(batch_start_match.group(2))
                    # Progress at start of batch (e.g., batch 3/6 means we're at 2/6 = 0.33)
                    # But if this is the first batch and we haven't started processing yet, set to 0.05 (after model load)
                    if batch_num == 1 and self.current_file_progress < 0.05:
                        self.current_file_progress = 0.05
                    else:
                        self.current_file_progress = (batch_num - 1) / total_batches
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
            
            # Try to parse frame progress within batch: "12%|#2 | 4/32 [00:19<02:18, 4.95s/it]"
            frame_progress_match = re.search(r'(\d+)%\s*\|\s*\d+/(\d+)', message)
            if frame_progress_match:
                try:
                    frame_pct = int(frame_progress_match.group(1))
                    # Find which batch we're in
                    batch_info = re.search(r'batch (\d+)/(\d+)', message)
                    if batch_info:
                        batch_num = int(batch_info.group(1))
                        total_batches = int(batch_info.group(2))
                        # Progress = (batch-1)/total + (frame_pct/100)/total
                        batch_progress = (batch_num - 1) / total_batches
                        frame_progress = (frame_pct / 100.0) / total_batches
                        self.current_file_progress = batch_progress + frame_progress
                    else:
                        # Fallback: use frame percentage directly
                        self.current_file_progress = frame_pct / 100.0
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
            
            # Calculate overall progress across all files
            file_portion = 100.0 / len(self.folder_files)
            # Base progress from completed files
            file_progress_base = (self.current_file_index / len(self.folder_files)) * 100
            # Current file's progress scaled to its portion
            file_progress_current = self.current_file_progress * file_portion
            # Total progress
            total_progress = int(min(100, file_progress_base + file_progress_current))
            self.parent_window.progress_bar.setValue(total_progress)
        else:
            # Single file mode - just use percentage from message
            progress_match = re.search(r'(\d+)%', message)
            if progress_match:
                try:
                    progress_pct = int(progress_match.group(1))
                    self.parent_window.progress_bar.setValue(progress_pct)
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
    
    def on_finished(self, success, message):
        self.process_btn.setEnabled(True)
        is_folder = self._is_folder_mode()
        if is_folder:
            self.process_btn.setText("Process All Files in Folder")
        else:
            self.process_btn.setText("Process File")
        self.parent_window.log_output(message)
        # Set progress to 100% on completion
        if success:
            self.parent_window.progress_bar.setValue(100)
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()  # Wait for thread to finish
            self.worker = None
        # Hide progress bar
        self.parent_window.progress_bar.setVisible(False)
        if success:
            if is_folder:
                self.parent_window.log_output("✓ Folder processed successfully!\n")
            else:
                self.parent_window.log_output("✓ File processed successfully!\n")
        else:
            QMessageBox.critical(self, "Error", f"Processing failed:\n{message}")


