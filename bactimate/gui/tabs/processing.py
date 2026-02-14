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
from bactimate.gui.log_config import get_logger, append_omnipose_debug
from bactimate.gui.paths import get_user_models_dir
from bactimate.gui.patches import _patch_omnipose_get_masks
from bactimate.gui.preferences import LAST_DIR_SEGMENTATION
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.mask_utils import (
    OMNIPOSE_MASKS_SUBDIR,
    convert_masks_in_folder,
    ensure_omnipose_masks_has_images,
    prepare_omnipose_masks_from_pairs,
)
from bactimate.gui.omnipose_utils import ensure_omni_in_model_path, inspect_omnipose_checkpoint
from bactimate.gui.utils import safe_imread
from bactimate.gui.widgets import (
    ClickableTooltip,
    TooltipClickFilter,
    DragDropGroupBox,
    create_info_icon,
    show_file_selection_dialog,
)
from bactimate.gui.workers import WorkerThread

# Built-in Omnipose model types (no Cellpose-only). User-trained models are listed separately.
BUILTIN_OMNIPOSE_MODELS = [
    "bact_phase_omni",
    "bact_fluor_omni",
    "bact_phase_affinity",
]
# Extension for user-trained model files saved in get_user_models_dir()
USER_MODEL_EXTENSION = ".pth"

# Transfer learning: only offer models that exist in the main tab (BUILTIN_OMNIPOSE_MODELS).
# (nchan, nclasses, dim) per Omnipose models table; used to lock nchan/nclasses when selected.
PRETRAINED_NCHAN_NCLASSES_DIM = {
    "bact_phase_omni": (2, 3, 2),
    "bact_fluor_omni": (2, 3, 2),
    "bact_phase_affinity": (1, 2, 2),
}

# When user selects "New model (same architecture)", we initialize from this so the result is
# identical to the built-in model (same architecture, same init) but trained on their data.
CANONICAL_BACTERIAL_MODEL = "bact_phase_omni"


def _get_pretrained_training_options():
    """From scratch (None), from bact_phase_omni (New model), or from another built-in by name. Same architecture: nchan=2, nclasses=3."""
    return [
        ("From scratch", "None"),  # Omnipose paper: --pretrained_model None --nclasses 3
        ("New model (from bact_phase_omni)", None),  # transfer from built-in
    ] + [(name, name) for name in BUILTIN_OMNIPOSE_MODELS if name in PRETRAINED_NCHAN_NCLASSES_DIM and name != CANONICAL_BACTERIAL_MODEL]


class ProcessingTab(QWidget):
    """Unified tab for processing single files or folders"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._trained_models_this_session: list[str] = []  # for session log when user trains a model
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

        # Process and Cancel buttons
        self.process_btn = QPushButton("Process File or Folder")
        self.process_btn.setEnabled(False)  # Disabled until file/folder is selected
        # Modern styling applied via global stylesheet
        self.process_btn.clicked.connect(self.process_file_or_folder)
        scroll_layout.addWidget(self.process_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the current segmentation job.")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self._user_cancelled_processing = False
        def on_cancel_processing():
            if hasattr(self, "worker") and self.worker is not None and self.worker.isRunning():
                self._user_cancelled_processing = True
                self.worker.terminate_process()
                self.parent_window.log_output("Cancelled by user.\n")
        self.cancel_btn.clicked.connect(on_cancel_processing)
        scroll_layout.addWidget(self.cancel_btn)

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
        basic_params_group = QGroupBox("Model Parameters")
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
            """Check if selected model is available (user model file or built-in cached/downloadable)."""
            try:
                display = self.model_combo.currentText()
                model_type, pretrained_path = self._resolve_model_selection(display)
                if pretrained_path is not None:
                    self.model_status_label.setText("✓ User model")
                    self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                    flow_sb = getattr(self, "flow_threshold", None)
                    if flow_sb is not None:
                        flow_sb.setValue(0.0)  # User models: use 0 per Omnipose recommendation
                    # No mask_threshold pre-selection; 0 works for models trained with checkpoint metadata
                    return
                from cellpose_omni.models import MODEL_DIR
                torch_str = "torch"
                basename = f"{model_type}{torch_str}_0"
                cache_model_file = Path(MODEL_DIR) / basename
                if cache_model_file.exists():
                    self.model_status_label.setText("✓ Cached")
                    self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                else:
                    self.model_status_label.setText("Will download on first use")
                    self.model_status_label.setStyleSheet("color: #FF9800; font-size: 9pt;")
                flow_sb = getattr(self, "flow_threshold", None)
                if flow_sb is not None:
                    flow_sb.setValue(0.4)  # Built-in models: default 0.4
            except Exception:
                get_logger().debug("Model availability check failed, hiding status", exc_info=True)
                self.model_status_label.setText("")
        
        self.model_combo.currentTextChanged.connect(check_model_availability)
        check_model_availability()
        self.model_combo.setToolTip(
            "Omnipose model type for cell segmentation.\n\n"
            "Default: 'bact_phase_omni'\n\n"
            "Available models:\n"
            "- bact_phase_omni: Bacterial phase contrast (default, best for bacteria)\n"
            "- bact_fluor_omni: Bacterial fluorescence\n"
            "- bact_phase_affinity: Bacterial phase contrast (affinity-based)\n\n"
            "User-trained models appear here after training via 'Train model…'."
        )
        fix_metadata_btn = QPushButton("Fix metadata")
        fix_metadata_btn.setToolTip(
            "If a user model does not detect cells, it may lack nchan/nclasses in the file. "
            "Click to write them into the selected user model so it loads correctly."
        )
        fix_metadata_btn.setEnabled(False)

        def update_fix_metadata_enabled():
            display = self.model_combo.currentText()
            _, pretrained_path = self._resolve_model_selection(display)
            fix_metadata_btn.setEnabled(pretrained_path is not None)

        def on_fix_metadata():
            display = self.model_combo.currentText()
            _, pretrained_path = self._resolve_model_selection(display)
            if not pretrained_path or not pretrained_path.is_file():
                QMessageBox.warning(self, "Fix metadata", "Select a user model first.")
                return
            from bactimate.gui.omnipose_utils import fix_user_model_metadata
            ok, msg = fix_user_model_metadata(pretrained_path)
            if ok:
                QMessageBox.information(self, "Fix metadata", msg)
            else:
                QMessageBox.warning(self, "Fix metadata", msg)

        fix_metadata_btn.clicked.connect(on_fix_metadata)
        self.model_combo.currentTextChanged.connect(update_fix_metadata_enabled)
        update_fix_metadata_enabled()
        model_layout.addWidget(fix_metadata_btn)
        train_model_btn = QPushButton("Train model")
        train_model_btn.setToolTip("Train a new Omnipose model from your images and ground-truth masks.")
        train_model_btn.clicked.connect(self._open_train_model_dialog)
        model_layout.addWidget(train_model_btn)
        seg_layout.addLayout(model_layout)
        
        # Diameter
        diam_layout = QHBoxLayout()
        diam_label = QLabel("Diameter (0=auto):")
        diam_info_icon = create_info_icon(
            "Expected cell diameter in pixels (only diameter field; Pixel size (µm) below is for scale bar).\n\n"
            "Default: 0 (auto-detect). Set to 0 for automatic detection (recommended for first try).\n"
            "Specify a value (e.g., 20-50 pixels for bacteria) for more consistent results.\n"
            "If you know size in µm: diameter_px = diameter_µm ÷ Pixel size (µm).\n\n"
            "Typical values:\n"
            "- Bacteria: 20-50 pixels\n"
            "- Mammalian cells: 30-100 pixels\n"
            "- Nuclei: 15-30 pixels\n\n"
            "Tip: Auto-detection works well, but specifying diameter often gives better results.\n\n"
            "Custom/user model: try Flow threshold 0, Mask threshold -2, Diameter 0 or cell size (px). "
            "See docs/CUSTOM_MODEL_TROUBLESHOOTING.md."
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
        
        # Pixel size (µm) for scale bar and calibration only; diameter is set above in pixels
        cal_layout = QHBoxLayout()
        cal_layout.addWidget(QLabel("Pixel size (µm):"))
        pixel_info = create_info_icon(
            "Pixel size in µm (for scale bar and calibration). Diameter above is always in pixels.\n\n"
            "To get diameter in px from µm: diameter_px = diameter_µm ÷ pixel_size_µm.\n"
            "Common: 100× ~0.108, 60× ~0.18, 40× ~0.27. Can be filled from file metadata when available.",
            parent_widget=self,
        )
        cal_layout.addWidget(pixel_info)
        cal_layout.addStretch()  # Align spin box to the right like other rows
        self.pixel_size_um = QDoubleSpinBox()
        self.pixel_size_um.setRange(0, 10)
        self.pixel_size_um.setValue(0)
        self.pixel_size_um.setDecimals(3)
        self.pixel_size_um.setSingleStep(0.01)
        self.pixel_size_um.setFixedWidth(100)  # Same width as other spin boxes
        self.pixel_size_um.setSpecialValueText("0")  # 0 = no µm scale (pixels only); see (?) tooltip
        cal_layout.addWidget(self.pixel_size_um)
        seg_layout.addLayout(cal_layout)
        
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
        # Sync flow threshold to current model (user -> 0, built-in -> 0.4) now that widget exists
        try:
            display = self.model_combo.currentText()
            _, pretrained_path = self._resolve_model_selection(display)
            self.flow_threshold.setValue(0.0 if pretrained_path is not None else 0.4)
            self.mask_threshold.setValue(0.0)  # Same default for all models
        except Exception:
            pass
        
        # Mask threshold
        mask_layout = QHBoxLayout()
        mask_label = QLabel("Mask threshold:")
        mask_info_icon = create_info_icon(
            "Distance/cell probability threshold for cell detection.\n\n"
            "Default: 0.0. Lower values: more permissive. Higher values: stricter.\n"
            "Range: -5.0 to 1.0 (negative can help if a model outputs shifted distance values).",
            parent_widget=self
        )
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(mask_info_icon)
        mask_layout.addStretch()
        self.mask_threshold = QDoubleSpinBox()
        self.mask_threshold.setRange(-5.0, 1.0)
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
        options_group = QGroupBox("Run Parameters")
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
        """Update button enabled states: path must exist and be a file or directory."""
        raw = self.file_path.text() and self.file_path.text().strip()
        has_path = False
        if raw:
            p = Path(raw)
            has_path = p.exists() and (p.is_file() or p.is_dir())
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
        Return display names for the model dropdown: built-in Omnipose models first,
        then user-trained models (filename stem) from get_user_models_dir().
        """
        names = list(BUILTIN_OMNIPOSE_MODELS)
        try:
            user_dir = get_user_models_dir()
            for p in sorted(user_dir.glob(f"*{USER_MODEL_EXTENSION}")):
                if p.is_file():
                    names.append(p.stem)
        except Exception:
            pass
        return names

    def _resolve_model_selection(self, display_name: str) -> tuple[Optional[str], Optional[Path]]:
        """
        Resolve combo display name to either built-in model_type or user pretrained path.
        Returns (model_type, pretrained_path): one is set, the other None.
        """
        if display_name in BUILTIN_OMNIPOSE_MODELS:
            return (display_name, None)
        try:
            user_path = get_user_models_dir() / f"{display_name}{USER_MODEL_EXTENSION}"
            if user_path.is_file():
                return (None, user_path)
        except Exception:
            pass
        return (None, None)

    def _on_train_finished_background(self, success: bool, name: str, msg: str) -> None:
        """Called when training finishes after the user closed the Train model dialog."""
        self._train_worker = None
        self.model_combo.clear()
        self.model_combo.addItems(self._get_available_models())
        if success:
            if self.model_combo.findText(name) >= 0:
                self.model_combo.setCurrentText(name)
            QMessageBox.information(self, "Training finished", f"Model '{name}' was saved and is available in the dropdown.")
        else:
            QMessageBox.warning(self, "Training failed", msg)

    def _open_train_model_dialog(self):
        """Open dialog to train a new Omnipose model from user data (from-scratch)."""
        from bactimate.gui.paths import get_user_models_dir
        d = QDialog(self)
        d.setWindowTitle("Train Omnipose model")
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, d)
        close_shortcut.activated.connect(d.reject)
        layout = QVBoxLayout()
        last_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_SEGMENTATION) or str(Path.home())
        # Training directory (drag-and-drop supported)
        train_dir_edit = QLineEdit()
        train_dir_edit.setPlaceholderText("Absolute path to folder with image/mask pairs (or drag and drop)")
        train_dir_edit.setAcceptDrops(False)  # let parent DragDropGroupBox receive drops
        def handle_dropped_training_path(dropped_path):
            path_obj = Path(dropped_path)
            if path_obj.is_dir():
                train_dir_edit.setText(dropped_path)
            elif path_obj.is_file():
                train_dir_edit.setText(str(path_obj.parent))
        dir_group = DragDropGroupBox(
            "Training directory (images + masks)",
            d,
            on_drop_callback=handle_dropped_training_path,
        )
        dir_layout = QHBoxLayout()
        path_label = QLabel("Path:")
        path_label.setAcceptDrops(False)
        dir_layout.addWidget(path_label)
        dir_info = create_info_icon(
            "Folder containing your training data: one image file and one mask file per pair.\n\n"
            "Omnipose expects: for each image 'NAME.tif', a mask file 'NAME<Mask filter>.tif' (e.g. NAME_masks.tif). Same base name, mask = base + Mask filter + extension.\n\n"
            "Mask format: must be instance labels (see (?) next to 'Convert masks for Omnipose').",
            d,
        )
        dir_layout.addWidget(dir_info)
        dir_layout.addWidget(train_dir_edit)
        def browse_dir():
            path = QFileDialog.getExistingDirectory(d, "Select training directory", last_dir)
            if path:
                train_dir_edit.setText(path)
        browse_btn = QPushButton("Browse…")
        browse_btn.setAcceptDrops(False)
        browse_btn.clicked.connect(browse_dir)
        dir_layout.addWidget(browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        # Filters
        filter_label = QLabel("File name filters:")
        filter_info = create_info_icon(
            "Suffix that identifies image vs mask files.\n\n"
            "Image filter: leave empty if filenames are e.g. A.tif; use _img if you have A_img.tif.\n"
            "Mask filter: suffix on mask files. With _masks, mask for 'cell1.tif' must be 'cell1_masks.tif'. With _segmentation, use 'cell1_segmentation.tif'. Must match your filenames exactly.",
            d,
        )
        filter_row = QHBoxLayout()
        filter_row.addWidget(filter_label)
        filter_row.addWidget(filter_info)
        filter_row.addStretch()
        layout.addLayout(filter_row)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Image filter:"))
        img_filter_edit = QLineEdit()
        img_filter_edit.setPlaceholderText("empty or _img")
        img_filter_edit.setMaximumWidth(120)
        filter_layout.addWidget(img_filter_edit)
        filter_layout.addWidget(QLabel("Mask filter:"))
        mask_filter_edit = QLineEdit()
        mask_filter_edit.setText("_mask")
        mask_filter_edit.setMaximumWidth(120)
        filter_layout.addWidget(mask_filter_edit)
        make_masks_btn = QPushButton("Convert masks for Omnipose")
        make_masks_btn.setFixedWidth(260)
        make_masks_info = create_info_icon(
            "Mask format: Omnipose needs instance label matrices, not binary. Background = 0; each cell = a unique integer (1, 2, 3, …). Do not use one value (e.g. 255) for all cells—Omnipose needs per-cell IDs to compute flow fields.\n\n"
            "This button: Converts binary masks (e.g. 0 and 255 from Ilastik) to instance labels. Output (and image copies when naming matches) goes to subfolder 'omnipose_masks'; originals are never overwritten. When you Start training, we use omnipose_masks automatically.",
            d,
        )
        filter_layout.addWidget(make_masks_btn)
        filter_layout.addSpacing(8)
        filter_layout.addWidget(make_masks_info)
        def on_make_masks():
            folder_str = train_dir_edit.text().strip()
            mask_filter = mask_filter_edit.text().strip()
            img_filter = img_filter_edit.text().strip()
            if not folder_str:
                QMessageBox.warning(d, "No folder", "Enter a training directory first.")
                return
            folder = Path(folder_str)
            converted, skipped, errors, out_dir = convert_masks_in_folder(
                folder, mask_filter, img_filter=img_filter
            )
            total = converted + skipped
            if total == 0 and not errors:
                QMessageBox.information(
                    d,
                    "No masks found",
                    f"No files matching *{mask_filter}.tif or *{mask_filter}.tiff in:\n{folder}",
                )
                return
            msg = (
                f"Converted masks + image copies: {out_dir}\n\n"
                f"Converted: {converted}, already OK: {skipped}."
            )
            if errors:
                err_text = "\n".join(errors[:10])
                if len(errors) > 10:
                    err_text += f"\n… and {len(errors) - 10} more."
                QMessageBox.warning(d, "Mask conversion", f"{msg}\n\nErrors:\n{err_text}")
                return
            msg += (
                "\n\nKeep the training directory as is. When you click Start training, "
                "we will use the omnipose_masks subfolder automatically (and copy images there if needed)."
            )
            box = QMessageBox(d)
            box.setWindowTitle("Masks ready")
            box.setText(msg)
            box.setIcon(QMessageBox.Icon.Information)
            open_btn = box.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Ok)
            box.exec()
            if box.clickedButton() == open_btn:
                try:
                    if sys.platform == "darwin":
                        subprocess.run(["open", str(out_dir)], check=False)
                    elif sys.platform == "win32":
                        subprocess.run(["explorer", str(out_dir)], check=False)
                    else:
                        subprocess.run(["xdg-open", str(out_dir)], check=False)
                except Exception:
                    pass
        make_masks_btn.clicked.connect(on_make_masks)
        layout.addLayout(filter_layout)
        # Model name (required)
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Model name:"))
        user_models_path = get_user_models_dir()
        name_info = create_info_icon(
            "A short name for your model (letters, numbers, underscores). "
            "It will appear in the model dropdown and is used to save the .pth file. "
            "Example: my_bacteria_v1. Do not use path separators (no / or \\).\n\n"
            "BactiMate automatically adds '_omni' to the filename if your name does not already contain 'omni' (e.g. my_model → my_model_omni.pth). "
            "This is required: Omnipose/Cellpose only enable correct cell detection when 'omni' is in the filename; without it, the model would load with wrong settings and fail to detect cells.\n\n"
            "Models are saved in BactiMate's user models folder. You can also copy .pth files (trained with Omnipose elsewhere) into this folder—they will appear in the dropdown on refresh; name them with '_omni' in the filename (e.g. my_model_omni.pth) so they work correctly.\n\n"
            f"User models folder:\n{user_models_path}",
            d,
        )
        name_layout.addWidget(name_info)
        model_name_edit = QLineEdit()
        model_name_edit.setPlaceholderText("e.g. my_bacteria_v1")
        name_layout.addWidget(model_name_edit)
        layout.addLayout(name_layout)
        # Start from (transfer learning)
        start_from_layout = QHBoxLayout()
        start_from_layout.addWidget(QLabel("Start from:"))
        pretrained_combo = QComboBox()
        pretrained_combo.setMinimumWidth(200)
        for display_name, _ in _get_pretrained_training_options():
            pretrained_combo.addItem(display_name)
        start_from_info = create_info_icon(
            "New model: BactiMate initializes from bact_phase_omni so the result is identical to the built-in model (same architecture, same init) but trained on your data. "
            "Other options: fine-tune from that pretrained model.\n\n"
            "Channels and classes are set automatically: bact_phase_affinity → 1 channel, 2 classes; "
            "other bacterial models (bact_phase_omni, bact_fluor_omni, New model) → 2 channels, 3 classes.",
            d,
        )
        start_from_layout.addWidget(pretrained_combo)
        start_from_layout.addWidget(start_from_info)
        check_compat_btn = QPushButton("Check compatibility")
        check_compat_btn.setFixedWidth(260)
        start_from_layout.addWidget(check_compat_btn)
        start_from_layout.addSpacing(12)
        check_compat_info = create_info_icon(
            "Validates image/mask pairs in your training folder and that image dimensions match the selected model (e.g. 2D for bact_phase_omni). Use after choosing 'Start from' and setting the training directory.",
            d,
        )
        start_from_layout.addWidget(check_compat_info)
        layout.addLayout(start_from_layout)
        # Hyperparameters
        layout.addWidget(QLabel("Training parameters:"))
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Epochs:"))
        epochs_info = create_info_icon(
            "Number of full passes over the training set.\n\n"
            "Omnipose often benefits from more epochs (e.g. 4000 for bacteria).\n"
            "Fewer epochs (500–1000) for quick tests; 4–20 for a fast sanity check.",
            d,
        )
        params_layout.addWidget(epochs_info)
        n_epochs_spin = QSpinBox()
        n_epochs_spin.setRange(1, 50000)
        n_epochs_spin.setSingleStep(100)
        n_epochs_spin.setValue(4000)
        params_layout.addWidget(n_epochs_spin)
        params_layout.addWidget(QLabel("Learning rate:"))
        lr_info = create_info_icon(
            "Step size for the optimizer (default 0.1).\n\n"
            "Larger values train faster but can be unstable; smaller values are slower but more stable.",
            d,
        )
        params_layout.addWidget(lr_info)
        lr_spin = QDoubleSpinBox()
        lr_spin.setRange(0.001, 1.0)
        lr_spin.setValue(0.1)
        lr_spin.setDecimals(3)
        params_layout.addWidget(lr_spin)
        params_layout.addWidget(QLabel("Batch size:"))
        batch_info = create_info_icon(
            "Number of image crops per training step.\n\n"
            "Larger batches (e.g. 16) train faster if you have enough GPU memory;\n"
            "smaller batches (e.g. 4–8) use less memory.",
            d,
        )
        params_layout.addWidget(batch_info)
        batch_spin = QSpinBox()
        batch_spin.setRange(1, 128)
        batch_spin.setValue(16)
        params_layout.addWidget(batch_spin)
        use_gpu_cb = QCheckBox("Use GPU")
        use_gpu_cb.setChecked(True)
        gpu_info = create_info_icon(
            "Use GPU (CUDA or Apple MPS) if available for much faster training.\n"
            "Uncheck to train on CPU only.",
            d,
        )
        params_layout.addWidget(gpu_info)
        params_layout.addWidget(use_gpu_cb)
        params_layout.addStretch()
        layout.addLayout(params_layout)
        # nchan/nclasses are fixed by "Start from"; hidden spinboxes updated by update_pretrained_lock
        nchan_spin = QSpinBox(d)
        nchan_spin.setRange(1, 8)
        nchan_spin.setValue(2)
        nchan_spin.setVisible(False)
        nclasses_spin = QSpinBox(d)
        nclasses_spin.setRange(2, 4)
        nclasses_spin.setValue(3)
        nclasses_spin.setVisible(False)

        _locked_style = "QSpinBox:disabled { background-color: palette(mid); color: palette(placeholder-text); }"
        _normal_style = ""

        def update_pretrained_lock():
            idx = pretrained_combo.currentIndex()
            options = _get_pretrained_training_options()
            cli_value = options[idx][1] if 0 <= idx < len(options) else None
            # Same architecture as paper: nchan=2, nclasses=3 for bacterial (From scratch / New model / bact_phase_omni)
            model_for_arch = CANONICAL_BACTERIAL_MODEL if (cli_value is None or cli_value == "None") else cli_value
            if model_for_arch in PRETRAINED_NCHAN_NCLASSES_DIM:
                nchan, nclass, _ = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_arch]
                nchan_spin.setValue(nchan)
                nclasses_spin.setValue(nclass)
                nchan_spin.setEnabled(False)
                nclasses_spin.setEnabled(False)
                nchan_spin.setStyleSheet(_locked_style)
                nclasses_spin.setStyleSheet(_locked_style)
                lock_tip = (
                    f"Locked to match {model_for_arch} (same architecture as built-in). "
                    "Required so the new model works at inference."
                )
                nchan_spin.setToolTip(lock_tip)
                nclasses_spin.setToolTip(lock_tip)
            else:
                nchan_spin.setEnabled(True)
                nclasses_spin.setEnabled(True)
                nchan_spin.setStyleSheet(_normal_style)
                nclasses_spin.setStyleSheet(_normal_style)
                nchan_spin.setToolTip("")
                nclasses_spin.setToolTip("")
        pretrained_combo.currentIndexChanged.connect(update_pretrained_lock)
        update_pretrained_lock()

        # Log output
        log_label = QLabel("Training log:")
        layout.addWidget(log_label)
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.setMaximumHeight(120)
        layout.addWidget(log_text)
        # Buttons
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        bb.setOrientation(Qt.Orientation.Horizontal)
        start_btn = bb.button(QDialogButtonBox.StandardButton.Ok)
        start_btn.setText("Start training")
        bb.rejected.connect(d.reject)

        class TrainWorker(QThread):
            log_line = pyqtSignal(str)
            finished_with = pyqtSignal(bool, str, str)  # success, model_name, message

            def __init__(self, cmd, env, cwd, save_dir, dest_file, model_name, nchan, nclasses):
                super().__init__()
                self.cmd = cmd
                self.env = env
                self.cwd = cwd
                self.save_dir = save_dir
                self.dest_file = dest_file
                self.model_name = model_name
                self.nchan = int(nchan)
                self.nclasses = int(nclasses)

            def _write_checkpoint_metadata(self):
                """Persist the same nchan/nclasses we passed to training into the saved .pth so the loader can load it (Omnipose may not write them). No architecture change."""
                try:
                    import torch
                    ckpt = torch.load(self.dest_file, map_location="cpu", weights_only=False)
                    if not isinstance(ckpt, dict):
                        ckpt = {"state_dict": ckpt, "nchan": self.nchan, "nclasses": self.nclasses}
                        torch.save(ckpt, self.dest_file)
                        return
                    if "nchan" in ckpt and "nclasses" in ckpt:
                        return
                    state = ckpt.get("state_dict") or ckpt
                    ckpt = dict(ckpt)
                    ckpt["state_dict"] = state
                    ckpt["nchan"] = self.nchan
                    ckpt["nclasses"] = self.nclasses
                    torch.save(ckpt, self.dest_file)
                except Exception:
                    pass  # non-fatal; loading will still infer from state_dict

            def run(self):
                import shutil
                try:
                    proc = subprocess.Popen(
                        self.cmd,
                        env=self.env,
                        cwd=self.cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in proc.stdout:
                        self.log_line.emit(line.rstrip())
                    proc.wait()
                    if proc.returncode != 0:
                        self.finished_with.emit(False, self.model_name, f"Omnipose exited with code {proc.returncode}.")
                        return
                    # Omnipose saves to train_dir/models/; filename often has no .pth extension
                    pth_files = list(Path(self.save_dir).glob("*.pth"))
                    alt = Path(self.cwd) / "models"
                    if not pth_files and alt.is_dir():
                        pth_files = list(alt.glob("*.pth"))
                    if not pth_files and alt.is_dir():
                        # Fallback: Omnipose may save without .pth; use newest file in models/
                        all_files = [p for p in alt.iterdir() if p.is_file()]
                        if all_files:
                            newest = max(all_files, key=lambda p: p.stat().st_mtime)
                            pth_files = [newest]
                    if not pth_files:
                        self.finished_with.emit(False, self.model_name, "No .pth checkpoint found after training.")
                        return
                    newest = max(pth_files, key=lambda p: p.stat().st_mtime)
                    shutil.copy2(newest, self.dest_file)
                    self._write_checkpoint_metadata()
                    shutil.rmtree(self.save_dir, ignore_errors=True)
                    self.finished_with.emit(True, self.model_name, f"Saved to {self.dest_file}")
                except Exception as e:
                    self.finished_with.emit(False, self.model_name, str(e))

        def _training_pairs(train_path, img_filter, mask_filter):
            """Return (n_pairs, first_paired_image_path, image_candidates) for compatibility check and start_training."""
            exts = (".tif", ".tiff", ".png", ".npy", ".npz")
            mask_suffixes = ("_cp_masks", "_cp_output", "_flows", mask_filter)

            def base_name(p, suf):
                return p.stem.replace(suf, "") if suf else p.stem

            candidates = [
                f for f in train_path.iterdir()
                if f.is_file() and f.suffix.lower() in exts
                and not any(f.stem.endswith(s) or (f.stem + s == f.stem) for s in mask_suffixes)
            ]
            n, first = 0, None
            for f in candidates:
                base_val = base_name(f, img_filter)
                if not base_val or base_val == ".":
                    continue
                if any((train_path / f"{base_val}{mask_filter}{e}").exists() for e in exts):
                    n += 1
                    if first is None:
                        first = f
            return n, first, candidates

        def on_check_compatibility():
            train_dir = train_dir_edit.text().strip()
            if not train_dir or not Path(train_dir).is_dir():
                QMessageBox.warning(d, "Invalid directory", "Please choose an absolute path to a training directory.")
                return
            mask_filter = mask_filter_edit.text().strip() or "_mask"
            img_filter = img_filter_edit.text().strip()
            train_path = ensure_omnipose_masks_has_images(Path(train_dir).resolve(), mask_filter, img_filter=img_filter)
            n_pairs, first_img, _ = _training_pairs(train_path, img_filter, mask_filter)
            if n_pairs == 0:
                QMessageBox.warning(
                    d, "Compatibility",
                    "No image/mask pairs found. Set Image and Mask filters so each image has a matching mask file.",
                )
                return
            idx = pretrained_combo.currentIndex()
            options = _get_pretrained_training_options()
            cli_value = options[idx][1] if 0 <= idx < len(options) else None
            model_for_arch = cli_value or CANONICAL_BACTERIAL_MODEL
            detail_parts = []
            if model_for_arch in PRETRAINED_NCHAN_NCLASSES_DIM:
                _, _, dim = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_arch]
                if dim == 2 and first_img:
                    try:
                        arr = safe_imread(str(first_img))
                    except Exception:
                        arr = None
                    if arr is not None and arr.ndim == 3 and arr.shape[0] > 4:
                        QMessageBox.warning(
                            d, "Compatibility",
                            f"Found {n_pairs} pair(s), but the first image has shape {tuple(arr.shape)} (likely a 3D stack). "
                            "The selected model is 2D. Use 2D slices or a 3D model (e.g. plant_omni).",
                        )
                        return
                    if arr is not None:
                        detail_parts.append(
                            f"Checked first image dimensions: shape {tuple(arr.shape)} — 2D, compatible with the selected 2D model ({model_for_arch})."
                        )
                    else:
                        detail_parts.append("Could not load first image to verify dimensions; pair count and filters are valid.")
                elif dim == 2:
                    detail_parts.append("Pair count and filters valid. No image could be sampled for dimension check.")
                else:
                    detail_parts.append(f"Pair count and filters valid. Selected model is 3D ({model_for_arch}); dimension check skipped.")
            else:
                detail_parts.append(
                    "Filters and pair naming are valid. Select a pretrained model in 'Start from' to also verify image dimensions (2D vs 3D)."
                )
            msg = f"Found {n_pairs} image/mask pair(s).\n\n" + "\n\n".join(detail_parts)
            QMessageBox.information(d, "Compatibility", msg)

        check_compat_btn.clicked.connect(on_check_compatibility)

        def start_training():
            train_dir = train_dir_edit.text().strip()
            model_name = model_name_edit.text().strip()
            if not train_dir or not Path(train_dir).is_dir():
                QMessageBox.warning(d, "Invalid directory", "Please choose an absolute path to a training directory.")
                return
            if not model_name:
                QMessageBox.warning(d, "Invalid name", "Please enter a model name.")
                return
            if "/" in model_name or "\\" in model_name:
                QMessageBox.warning(d, "Invalid name", "Model name must not contain path separators.")
                return
            # Cellpose/Omnipose only enable omni dynamics when "omni" is in the filename; without it, detection fails.
            # So we always save with _omni in the stem (e.g. my_model → my_model_omni.pth) so the model works at inference.
            final_stem = model_name if "omni" in model_name.lower() else f"{model_name}_omni"
            dest_file = get_user_models_dir() / f"{final_stem}{USER_MODEL_EXTENSION}"
            if dest_file.exists():
                if QMessageBox.question(d, "Model already exists", f"A model named '{final_stem}' already exists. Overwrite it? This cannot be undone.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                    return
            train_dir_abs = Path(train_dir).resolve()
            img_filter = img_filter_edit.text().strip()
            mask_filter = mask_filter_edit.text().strip() or "_mask"
            # Populate omnipose_masks from the same pair discovery as _training_pairs so we always
            # use converted masks when pairs exist (avoids glob mismatch: e.g. default _mask vs *_masks.tif).
            parent_dir = train_dir_abs.parent if train_dir_abs.name == OMNIPOSE_MASKS_SUBDIR else train_dir_abs
            train_path, n_from_pairs = prepare_omnipose_masks_from_pairs(
                parent_dir, img_filter, mask_filter
            )
            if n_from_pairs == 0:
                train_path = ensure_omnipose_masks_has_images(
                    parent_dir, mask_filter, img_filter=img_filter
                )
            train_dir_abs = str(train_path)
            # Pre-check: Omnipose needs one mask file per image (base name + mask_filter + ext). Catch missing masks or bad names early.
            pairs_ok, first_img_path, image_candidates = _training_pairs(train_path, img_filter, mask_filter)

            def base_name(p, suf):
                return p.stem.replace(suf, "") if suf else p.stem

            bad_base = [f.name for f in image_candidates if not base_name(f, img_filter) or base_name(f, img_filter) == "."]

            # If Image filter is empty and no pairs found, try to infer it from filenames (e.g. 634_phase.tif + 634_masks.tif → _phase)
            if pairs_ok == 0 and not img_filter and image_candidates:
                mask_files = [f for f in train_path.iterdir() if f.is_file() and f.stem.endswith(mask_filter)]
                for img_f in image_candidates:
                    stem = img_f.stem
                    for mf in mask_files:
                        base = mf.stem[: -len(mask_filter)] if mask_filter else mf.stem
                        if base and stem.startswith(base) and stem != base:
                            inferred = stem[len(base):]
                            if _training_pairs(train_path, inferred, mask_filter)[0] > 0:
                                img_filter = inferred
                                img_filter_edit.setText(inferred)
                                pairs_ok, first_img_path, image_candidates = _training_pairs(train_path, img_filter, mask_filter)
                                bad_base = [f.name for f in image_candidates if not base_name(f, img_filter) or base_name(f, img_filter) == "."]
                                break
                    if pairs_ok > 0:
                        break

            if bad_base:
                QMessageBox.warning(
                    d, "Invalid training files",
                    f"Some image filenames give an empty base name (Omnipose will skip them or fail): {bad_base[:5]}{'…' if len(bad_base) > 5 else ''}. "
                    "Use names like 'image.tif' with a matching 'image_masks.tif' (no leading dot).",
                )
                return
            if pairs_ok == 0:
                folder_used = getattr(train_path, "name", str(train_path))
                if folder_used == OMNIPOSE_MASKS_SUBDIR or str(train_path).endswith(os.path.sep + OMNIPOSE_MASKS_SUBDIR):
                    hint = (
                        f"The folder '{train_path}' contains only mask files (no images). "
                        f"Images are copied from the parent folder when they match the mask names: for each mask 'NAME{mask_filter}.tif' we need an image 'NAME{img_filter or ''}.tif' (or .tiff/.png) in the parent. "
                        "Check that the parent folder has image files with matching base names and that Image filter matches your image filenames."
                    )
                else:
                    hint = (
                        f"For each image we need a mask: base name + '{mask_filter}' + extension. "
                        f"Image filter is '{img_filter or '(empty)'}' — the base is taken from the image by removing this suffix. "
                        f"Example: for image '634_phase.tif' and mask '634_masks.tif', set Image filter to '_phase' so the base is '634'."
                    )
                QMessageBox.warning(
                    d, "No image/mask pairs found",
                    f"No matching mask files found. {hint}",
                )
                return
            # Compatibility with pretrained model: 2D models expect 2D images (not 3D stacks)
            pretrained_idx = pretrained_combo.currentIndex()
            options = _get_pretrained_training_options()
            pretrained_cli = options[pretrained_idx][1] if 0 <= pretrained_idx < len(options) else None
            model_for_dim = CANONICAL_BACTERIAL_MODEL if (pretrained_cli is None or pretrained_cli == "None") else pretrained_cli
            if model_for_dim in PRETRAINED_NCHAN_NCLASSES_DIM:
                _, _, dim = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_dim]
                if dim == 2:
                    _, first_img_path, _ = _training_pairs(train_path, img_filter, mask_filter)
                    if first_img_path:
                        try:
                            arr = safe_imread(str(first_img_path))
                        except Exception:
                            arr = None
                        if arr is not None and arr.ndim == 3 and arr.shape[0] > 4:
                            if QMessageBox.question(
                                d, "Compatibility",
                                f"First image has shape {tuple(arr.shape)} (likely a 3D stack). The selected model is 2D. Continue anyway?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No,
                            ) != QMessageBox.StandardButton.Yes:
                                return
            start_btn.setEnabled(False)
            log_text.clear()
            n_epochs = n_epochs_spin.value()
            log_text.append(f"Found {pairs_ok} image/mask pair(s). All will be used every epoch for {n_epochs} epochs.")
            # Pass through to Omnipose: same args as paper (--nchan 2 --nclasses 3). No architecture change.
            pretrained_arg = pretrained_cli if pretrained_cli is not None else CANONICAL_BACTERIAL_MODEL
            if pretrained_cli == "None":
                log_text.append("From scratch: --pretrained_model None, nchan=2, nclasses=3 (Omnipose paper defaults).")
            elif pretrained_cli:
                log_text.append(f"Transfer learning: starting from {pretrained_cli} (nchan/nclasses match that model).")
            else:
                log_text.append(f"New model: from {CANONICAL_BACTERIAL_MODEL} (same architecture), training on your data.")
            log_text.append("Starting Omnipose training (this may take a long time)…")
            # Same as Omnipose paper: --train --diameter 0 --n_epochs 4000 --learning_rate 0.1 --batch_size 16 --RAdam --nclasses 3
            # (From scratch: --pretrained_model None; transfer: --pretrained_model bact_phase_omni or other). We pass --nchan/--nclasses through; no architecture change.
            cmd = [
                sys.executable, "-m", "omnipose", "--train",
                "--dir", train_dir_abs,
                "--mask_filter", mask_filter,
                "--pretrained_model", str(pretrained_arg),
                "--diameter", "0",
                "--n_epochs", str(n_epochs_spin.value()),
                "--learning_rate", str(lr_spin.value()),
                "--batch_size", str(batch_spin.value()),
                "--nchan", str(nchan_spin.value()),
                "--nclasses", str(nclasses_spin.value()),
                "--RAdam",
            ]
            if img_filter:
                cmd.extend(["--img_filter", img_filter])
            if use_gpu_cb.isChecked():
                cmd.append("--use_gpu")
            # Channel axis: 0 = channels first (CYX). Only relevant when nchan > 1.
            if nchan_spin.value() >= 2:
                cmd.extend(["--channel_axis", "0"])
            import tempfile
            save_dir = tempfile.mkdtemp(prefix="bactimate_train_")
            env = os.environ.copy()
            env["CELLPOSE_LOCAL_MODELS_PATH"] = save_dir

            worker = TrainWorker(
                cmd, env, train_dir_abs, save_dir, dest_file, final_stem,
                nchan_spin.value(), nclasses_spin.value(),
            )
            self._train_worker = worker  # Keep ref so thread isn't destroyed if dialog closes

            def on_log(line):
                if log_text:
                    log_text.append(line)

            def on_finished(success, name, msg):
                start_btn.setEnabled(True)
                if log_text:
                    log_text.append(f"\n{msg}")
                if success:
                    training_record = {
                        "model_name": name,
                        "n_epochs": n_epochs_spin.value(),
                        "learning_rate": lr_spin.value(),
                        "batch_size": batch_spin.value(),
                        "nchan": nchan_spin.value(),
                        "nclasses": nclasses_spin.value(),
                        "use_gpu": use_gpu_cb.isChecked(),
                        "mask_filter": mask_filter_edit.text().strip() or "_mask",
                        "img_filter": img_filter_edit.text().strip() or "",
                        "train_dir": train_dir_edit.text().strip(),
                        "pretrained_model": pretrained_arg,
                    }
                    self._trained_models_this_session.append(training_record)
                    if getattr(self.parent_window, "_save_session_log", None):
                        self.parent_window._save_session_log()
                    done_msg = (
                        f"Model '{name}' saved to:\n{dest_file}\n\n"
                        "It will appear in the model dropdown. "
                        "When using it: if no cells are detected, set Diameter (e.g. 30 px) in Processing; custom models do not support auto-diameter."
                    )
                    done_msg = f"(Initialized from {pretrained_arg})\n\n" + done_msg
                    QMessageBox.information(d, "Done", done_msg)
                    current = self.model_combo.currentText()
                    self.model_combo.clear()
                    self.model_combo.addItems(self._get_available_models())
                    if self.model_combo.findText(name) >= 0:
                        self.model_combo.setCurrentText(name)
                    else:
                        self.model_combo.setCurrentText(current or "bact_phase_omni")
                else:
                    QMessageBox.warning(d, "Training failed", msg)
                self._train_worker = None

            worker.log_line.connect(on_log)
            worker.finished_with.connect(on_finished)
            worker.start()

            def on_dialog_closed():
                # If user closes dialog while training is running, keep worker alive and handle finish on the tab
                w = getattr(self, "_train_worker", None)
                if w is not None and w.isRunning():
                    try:
                        w.log_line.disconnect()
                    except Exception:
                        pass
                    try:
                        w.finished_with.disconnect()
                    except Exception:
                        pass
                    w.finished_with.connect(self._on_train_finished_background)

            d.finished.connect(on_dialog_closed)

        start_btn.clicked.connect(start_training)
        layout.addWidget(bb)
        # Enforce minimum width so dialog cannot be resized narrower (avoids (?) overlapping button)
        content = QWidget()
        content.setLayout(layout)
        _min_width = 650
        content.setMinimumWidth(_min_width)
        d.setMinimumWidth(_min_width)
        d_layout = QVBoxLayout()
        d_layout.setContentsMargins(0, 0, 0, 0)
        d_layout.addWidget(content)
        d.setLayout(d_layout)
        d.exec()

    def reset_segmentation_defaults(self):
        """Reset all segmentation parameters to their default values"""
        # Set default model, fallback to first available if bact_phase_omni not available
        if 'bact_phase_omni' in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText('bact_phase_omni')
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        self.diameter.setValue(0)
        self.pixel_size_um.setValue(0)
        self.flow_threshold.setValue(0.4)
        self.mask_threshold.setValue(0.0)
        self.use_gpu.setChecked(True)
        self.cluster.setChecked(False)
        self.create_overlay.setChecked(True)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        out = get_from_widgets([
            ("model", self.model_combo),
            ("diameter", self.diameter),
            ("pixel_size_um", self.pixel_size_um),
            ("flow_threshold", self.flow_threshold),
            ("mask_threshold", self.mask_threshold),
            ("use_gpu", self.use_gpu),
            ("cluster", self.cluster),
            ("create_overlay", self.create_overlay),
        ])
        out["trained_models_this_session"] = list(self._trained_models_this_session)
        return out

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("model", self.model_combo),
            ("diameter", self.diameter),
            ("pixel_size_um", self.pixel_size_um),
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
            from bactimate.gui.image_utils import to_qpixmap, build_segmentation_overlay, scale_for_preview, draw_scale_bar
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
                """Load Omnipose model (only reloads if model type or GPU setting changed)."""
                params = get_current_parameters()
                current_model_type = params.get("model_type")
                pretrained_path = params.get("pretrained_model")
                current_use_gpu = params["use_gpu"]
                current_key = pretrained_path or current_model_type or ""
                needs_reload = (
                    preview_dialog.model is None
                    or (getattr(preview_dialog, "_last_model_key", None) != current_key)
                    or (getattr(preview_dialog, "_last_use_gpu", None) != current_use_gpu)
                )
                if needs_reload:
                    preview_dialog.segmentation_cache.clear()
                if needs_reload:
                    try:
                        import os
                        from urllib.error import HTTPError  # required for except HTTPError below (must be in scope for all paths)
                        # User-trained model: load by path
                        if pretrained_path and os.path.isfile(pretrained_path):
                            status_label.setText("Loading user model...")
                            QApplication.processEvents()
                            # Debug: inspect checkpoint before load (see Logs/omnipose_debug.log)
                            try:
                                inspect_info = inspect_omnipose_checkpoint(pretrained_path)
                                append_omnipose_debug(
                                    f"[BactiMate USER_MODEL pre-load] path={pretrained_path!r} "
                                    f"inspect: nchan={inspect_info.get('nchan')} nclasses={inspect_info.get('nclasses')} "
                                    f"first_layer_shape={inspect_info.get('first_layer_shape')} keys={inspect_info.get('keys', [])[:15]!r}"
                                )
                            except Exception:
                                pass
                            # cellpose_omni only sets omni=True when "omni" is in the filename
                            path_for_loader = ensure_omni_in_model_path(pretrained_path)
                            preview_dialog.model = models.Cellpose(
                                gpu=current_use_gpu,
                                pretrained_model=path_for_loader,
                                omni=True,
                            )
                            # Debug: what did the patched loader set on model.cp?
                            try:
                                cp = getattr(preview_dialog.model, "cp", None)
                                nchan = getattr(cp, "nchan", None)
                                nclasses = getattr(cp, "nclasses", None)
                                append_omnipose_debug(
                                    f"[BactiMate USER_MODEL post-load] model.cp.nchan={nchan} model.cp.nclasses={nclasses}"
                                )
                                get_logger().info("User model loaded: nchan=%s nclasses=%s", nchan, nclasses)
                                # If size model failed to load, diameter=0 won't rescale -> often 0 cells
                                if getattr(preview_dialog.model, "pretrained_size", None) is None:
                                    append_omnipose_debug(
                                        "[BactiMate USER_MODEL] size model not loaded (pretrained_size is None). "
                                        "diameter=0 will not rescale -> often 0 cells. "
                                        "Run from project root: venv/bin/python3 setup/apply_all_patches.py then restart BactiMate."
                                    )
                            except Exception:
                                pass
                            preview_dialog._last_model_key = current_key
                            preview_dialog._last_use_gpu = current_use_gpu
                            return True
                        from cellpose_omni.models import model_path, MODEL_DIR
                        from pathlib import Path
                        expected_model_dir = PROJECT_ROOT / "models"
                        if expected_model_dir.exists() and str(MODEL_DIR) != str(expected_model_dir):
                            pass
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
                        # Debug: built-in model load (see Logs/omnipose_debug.log)
                        try:
                            cp = getattr(preview_dialog.model, "cp", None)
                            nchan = getattr(cp, "nchan", None)
                            nclasses = getattr(cp, "nclasses", None)
                            append_omnipose_debug(
                                f"[BactiMate BUILTIN post-load] model_type={current_model_type!r} "
                                f"model.cp.nchan={nchan} model.cp.nclasses={nclasses}"
                            )
                        except Exception:
                            pass

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
                        
                        preview_dialog._last_model_key = current_key
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
                                    preview_dialog._last_model_key = current_key
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
                                    preview_dialog._last_model_key = current_key
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
                                    preview_dialog._last_model_key = current_key
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
                                    preview_dialog._last_model_key = current_key
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
                """Get current parameters from main window (works because dialog is non-modal)."""
                import sys
                is_frozen = getattr(sys, "frozen", False) or hasattr(sys, "__compiled__")
                diameter_value = self._get_effective_diameter_pixels()
                display = self.model_combo.currentText()
                model_type, pretrained_path = self._resolve_model_selection(display)
                # Built-in: only default to 30 when frozen (frozen build workaround).
                if diameter_value == 0 and is_frozen and not pretrained_path:
                    default_diameters = {
                        "bact_phase_omni": 30.0,
                        "bact_omni": 30.0,
                        "bact_fluor_omni": 30.0,
                    }
                    diameter_value = default_diameters.get(model_type or display, 30.0)
                # User models: diameter=0 triggers slow two-pass (estimate then segment). Use 30 so we run once; result is the same.
                if pretrained_path is not None and (diameter_value is None or diameter_value <= 0):
                    diameter_value = 30.0
                return {
                    "diameter": diameter_value if diameter_value and diameter_value > 0 else 0,
                    "flow_threshold": self.flow_threshold.value(),
                    "mask_threshold": self._get_effective_mask_threshold(),
                    "cluster": self.cluster.isChecked(),
                    "model_type": model_type,
                    "pretrained_model": str(pretrained_path) if pretrained_path else None,
                    "use_gpu": self.use_gpu.isChecked(),
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
                       bool(params.get("cluster", False)),
                       str(params.get("model_type") or params.get("pretrained_model") or "bact_phase_omni"),
                       bool(params.get("use_gpu", True)))
            
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
                
                # Optional disk cache (session): same key as in-memory, stored under temp dir
                import hashlib
                import tempfile
                _cache_dir = Path(tempfile.gettempdir()) / "bactimate_preview_cache"
                _disk_key = hashlib.sha256(f"{file_path}|{cache_key}".encode()).hexdigest()[:24]
                _disk_path = _cache_dir / f"{_disk_key}.npy"
                
                # Check cache first (safely - handle any old cache entries with None values)
                if use_cache:
                    try:
                        if cache_key in preview_dialog.segmentation_cache:
                            return preview_dialog.segmentation_cache[cache_key]
                        # Check disk cache
                        if _disk_path.exists():
                            try:
                                mask_disk = np.load(_disk_path)
                                preview_dialog.segmentation_cache[cache_key] = mask_disk
                                return mask_disk
                            except Exception:
                                pass
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
                    # Debug: log eval result for user vs built-in (see Logs/omnipose_debug.log)
                    try:
                        kind = "USER_MODEL" if params.get("pretrained_model") else "BUILTIN"
                        model_id = params.get("pretrained_model") or params.get("model_type") or "?"
                        n_cells = len(np.unique(mask)) - 1
                        append_omnipose_debug(
                            f"[BactiMate eval] kind={kind} model={model_id!r} "
                            f"diameter={diameter} flow_thr={flow_threshold} mask_thr={mask_threshold} "
                            f"frame_shape={getattr(frame, 'shape', None)} channels=[0,0] -> "
                            f"n_cells={n_cells} mask_min={int(mask.min())} mask_max={int(mask.max())}"
                        )
                    except Exception:
                        pass
                    # Cache the result (memory and optional disk)
                    preview_dialog.segmentation_cache[cache_key] = mask
                    try:
                        _cache_dir.mkdir(parents=True, exist_ok=True)
                        np.save(_disk_path, mask)
                    except Exception:
                        pass
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
                                                                              'mask_threshold': mask_threshold, 'cluster': cluster, 'model_type': params.get('model_type'), 'pretrained_model': params.get('pretrained_model'),
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
                    pixel_size_um = self.pixel_size_um.value()
                    if pixel_size_um > 0:
                        scale_factor = disp_scaled.shape[1] / max(1, display_frame.shape[1])
                        draw_scale_bar(overlay, pixel_size_um, scale_factor=scale_factor)
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
                    if num_cells == 0:
                        params = get_current_parameters()
                        if params.get("pretrained_model"):
                            status_label.setText("No cells detected. Try Flow 0, Mask -2, Diameter 0 or cell size (px). See docs/CUSTOM_MODEL_TROUBLESHOOTING.md.")
                            status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                        else:
                            if has_cache:
                                status_label.setText("Ready (from cache) — no cells")
                            else:
                                status_label.setText(f"Ready (cached: {cache_size}) — no cells")
                            status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
                    else:
                        if has_cache:
                            status_label.setText("Ready (from cache)")
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
            scale_label = QLabel("")
            if self.pixel_size_um.value() > 0:
                scale_label.setText(f"Scale: {self.pixel_size_um.value():.3f} µm/px")
                scale_label.setToolTip("Pixel size from Processing tab (scale bar drawn on image).")
            nav_layout.addWidget(scale_label)
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

            close_shortcut = QShortcut(QKeySequence.StandardKey.Close, preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)

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
            pixel_size_um_0 = self.pixel_size_um.value()
            if pixel_size_um_0 > 0:
                scale_factor_0 = disp0.shape[1] / max(1, display_frame0.shape[1])
                draw_scale_bar(initial_overlay, pixel_size_um_0, scale_factor=scale_factor_0)
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
    
    def _get_effective_diameter_pixels(self) -> float:
        """Return diameter in pixels (the single Diameter field)."""
        return self.diameter.value()

    def _get_run_diameter_and_flow(self) -> tuple[float, float]:
        """
        Return (diameter, flow_threshold) to pass to the batch script when running segmentation.
        Custom (user-trained) models: diameter=0 would give no cells, so use 30; flow_threshold
        must be 0 (permissive). Built-in models: use GUI values as-is.
        """
        display = self.model_combo.currentText()
        model_type, pretrained_path = self._resolve_model_selection(display)
        diameter = self._get_effective_diameter_pixels()
        flow = float(self.flow_threshold.value())
        if pretrained_path is not None:
            # User-trained model: no size model, so diameter=0 fails at inference
            if diameter is None or diameter <= 0:
                diameter = 30.0
            # Custom models need permissive flow (0); 0.4 often yields no cells
            flow = 0.0
        return (float(diameter), flow)

    def _get_effective_mask_threshold(self) -> float:
        """Return mask_threshold to pass to model.eval(). Use UI value as-is (no special case for user models)."""
        return float(self.mask_threshold.value())

    def _open_in_file_manager(self, path: Path) -> None:
        """Open the given path (file or directory) in the OS default file manager."""
        path = Path(path)
        if not path.exists():
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif sys.platform == "win32":
                subprocess.run(["explorer", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            get_logger().debug("Could not open path in file manager", exc_info=True)

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
        if self._get_effective_diameter_pixels() < 0:
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
            
            _mt, _path = self._resolve_model_selection(self.model_combo.currentText())
            if _path is not None:
                model_arg = ensure_omni_in_model_path(_path)
            else:
                model_arg = _mt or self.model_combo.currentText()
            run_diameter, run_flow = self._get_run_diameter_and_flow()
            if _path is not None and (self._get_effective_diameter_pixels() <= 0 or self.flow_threshold.value() != 0):
                self.parent_window.log_output(
                    "Custom model: using diameter=30 and flow_threshold=0 (set Diameter in Processing for your cell size if needed).\n"
                )
            cmd_args = [
                self.file_path.text(),
                '--output', str(output_dir),
                '--model', model_arg,
                '--diameter', str(run_diameter),
                '--flow-threshold', str(run_flow),
                '--mask-threshold', str(self._get_effective_mask_threshold()),
                '--batch-size', '32',  # Increased batch size for faster GPU processing (adjust if OOM errors occur)
            ]
            
            if not self.use_gpu.isChecked():
                cmd_args.append('--no-gpu')
            if self.cluster.isChecked():
                cmd_args.append('--cluster')
            if not self.create_overlay.isChecked():
                cmd_args.append('--no-overlay')
            
            self._last_output_dir = output_dir
            self.process_btn.setEnabled(False)
            self.process_btn.setText("Processing...")
            self.cancel_btn.setVisible(True)
            self.cancel_btn.setEnabled(True)
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
            self.cancel_btn.setVisible(True)
            self.cancel_btn.setEnabled(True)
            self.parent_window.log_output(f"Starting batch folder processing: {len(selected_files)} file(s) selected from {self.file_path.text()}\n")
            
            self._last_output_dir = folder  # folder containing all _segmentation outputs
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
            self.cancel_btn.setVisible(False)
            self.cancel_btn.setEnabled(False)
            self.process_btn.setEnabled(True)
            is_folder = self._is_folder_mode()
            if is_folder:
                self.process_btn.setText("Process All Files in Folder")
            else:
                self.process_btn.setText("Process File")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f"\n✓ Segmentation complete for all {len(self.folder_files)} file(s)!\n")
            msg = QMessageBox(self)
            msg.setWindowTitle("Processing Complete")
            msg.setText(
                f"Successfully processed {len(self.folder_files)} file(s).\n\n"
                "Output: one _segmentation subfolder per file (masks and overlay)."
            )
            msg.setIcon(QMessageBox.Icon.Information)
            open_btn = msg.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
            msg.addButton(QMessageBox.StandardButton.Ok)
            msg.exec()
            if msg.clickedButton() == open_btn and getattr(self, "_last_output_dir", None):
                self._open_in_file_manager(Path(self._last_output_dir))
            return
        
        tiff_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f"\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {tiff_file.name}\n")
        
        # Determine base directory (go up one level if input is in Tif_* folder)
        base_dir = tiff_file.parent
        if base_dir.name.startswith('Tif_'):
            base_dir = base_dir.parent
        
        # Create output directory at base level with _segmentation suffix
        output_dir = base_dir / f"{tiff_file.stem}_segmentation"
        
        _mt, _path = self._resolve_model_selection(self.model_combo.currentText())
        if _path is not None:
            model_arg = ensure_omni_in_model_path(_path)
        else:
            model_arg = _mt or self.model_combo.currentText()
        run_diameter, run_flow = self._get_run_diameter_and_flow()
        if _path is not None and (self._get_effective_diameter_pixels() <= 0 or self.flow_threshold.value() != 0):
            self.parent_window.log_output(
                "Custom model: using diameter=30 and flow_threshold=0 (set Diameter in Processing for your cell size if needed).\n"
            )
        cmd_args = [
            str(tiff_file),
            '--output', str(output_dir),
            '--model', model_arg,
            '--diameter', str(run_diameter),
            '--flow-threshold', str(run_flow),
            '--mask-threshold', str(self._get_effective_mask_threshold()),
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
        
        # If user cancelled, stop the queue and reset UI
        if getattr(self, "_user_cancelled_processing", False):
            self._user_cancelled_processing = False
            self.cancel_btn.setVisible(False)
            self.cancel_btn.setEnabled(False)
            self.process_btn.setEnabled(True)
            self.process_btn.setText("Process All Files in Folder" if self._is_folder_mode() else "Process File")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output("Processing cancelled.\n")
            return
        
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
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(False)
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
            # Offer to open output folder (single-file mode only; folder mode has its own dialog)
            if not is_folder and getattr(self, "_last_output_dir", None):
                out_path = Path(self._last_output_dir)
                if out_path.exists():
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Processing Complete")
                    msg.setText("File processed successfully.")
                    msg.setIcon(QMessageBox.Icon.Information)
                    open_btn = msg.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
                    msg.addButton(QMessageBox.StandardButton.Ok)
                    msg.exec()
                    if msg.clickedButton() == open_btn:
                        self._open_in_file_manager(out_path)
        else:
            if getattr(self, "_user_cancelled_processing", False):
                self._user_cancelled_processing = False
                self.parent_window.log_output("Processing was cancelled.\n")
            else:
                QMessageBox.critical(self, "Error", f"Processing failed:\n{message}")


