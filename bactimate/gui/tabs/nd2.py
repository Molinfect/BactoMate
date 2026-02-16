"""
ND2 / LIF / CZI conversion tab: file selection, conversion parameters, preview, convert to TIFF.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
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

from bactimate.gui.constants import CONVERT_ND2_SCRIPT, PROJECT_ROOT
from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.preferences import LAST_DIR_ND2
from bactimate.gui.image_utils import to_qpixmap, scale_for_preview
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.utils import safe_imread
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog
from bactimate.gui.workers import WorkerThread

class ND2ConversionTab(QWidget):
    """Tab for converting microscopy files (ND2, LIF, CZI) to TIFF with metadata extraction"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.file_info = None  # Store file info
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

        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QVBoxLayout()
        
        output_select_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Auto: creates 'Tif_1-N' folder in input directory")
        self.output_dir.setMinimumWidth(500)  # Increased for better visibility
        # Set larger font for better readability
        output_font = QFont()
        output_font.setPointSize(12)
        self.output_dir.setFont(output_font)
        # Fix size to minimum - prevent expansion
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.output_dir.setSizePolicy(size_policy)
        output_btn = QPushButton("Browse...")
        output_btn.setFixedWidth(120)  # Fixed width for buttons
        output_btn.clicked.connect(self.select_output_dir)
        output_select_layout.addWidget(self.output_dir, 0)  # No stretch - keep at minimum size
        output_select_layout.addWidget(output_btn, 0)  # No stretch for buttons
        output_layout.addLayout(output_select_layout)
        
        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)
        
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
        """Build the microscopy file selection group."""
        file_group = DragDropGroupBox(
            "Microscopy File Selection",
            parent=self,
            on_drop_callback=self.handle_dropped_path
        )
        file_layout = QVBoxLayout()
        file_select_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a microscopy file or folder... (or drag and drop)")
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
        file_select_layout.addWidget(self.file_path, 0)
        file_select_layout.addWidget(file_btn, 0)
        file_select_layout.addWidget(folder_btn, 0)
        file_layout.addLayout(file_select_layout)
        file_group.setLayout(file_layout)
        return file_group

    def _build_conversion_params_group(self) -> QGroupBox:
        """Build the Conversion Parameters group (frames, channels, corrections)."""
        params_group = QGroupBox("Conversion Parameters")
        params_layout = QVBoxLayout()
        frame_layout = QHBoxLayout()
        frame_label = QLabel("Number of Frames:")
        frame_info_icon = create_info_icon(
            "Number of time frames to extract from the microscopy file.\n\n"
            "Source: Hardcoded default value\n\n"
            "The value will automatically update to the maximum number of frames when a file is loaded.\n\n"
            "Note: Extracting more frames takes longer and uses more disk space.",
            parent_widget=self
        )
        frame_layout.addWidget(frame_label)
        frame_layout.addWidget(frame_info_icon)
        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 10000)
        self.max_frames.setValue(100)
        frame_layout.addWidget(self.max_frames)
        frame_layout.addStretch()
        params_layout.addLayout(frame_layout)
        channel_layout = QHBoxLayout()
        channel_label = QLabel("Channel Handling:")
        channel_layout.addWidget(channel_label)
        self.separate_channels = QCheckBox("Save channels separately (recommended)")
        self.separate_channels.setChecked(True)
        channel_layout.addWidget(self.separate_channels)
        self.max_z_projection = QCheckBox("Max intensity projection for multiple Z stack")
        self.max_z_projection.setChecked(False)
        channel_layout.addWidget(self.max_z_projection)
        z_projection_info_icon = create_info_icon(
            "Apply maximum intensity projection across Z-slices when Z > 1.\n\n"
            "Default: Disabled (unchecked)\n"
            "Source: User selection\n\n"
            "When enabled: For each pixel, takes the maximum intensity value across all Z-slices.\n"
            "This creates a single 2D image from a 3D Z-stack.\n\n"
            "When disabled: Uses only the first Z-slice (Z=1).\n\n"
            "Use when: You have Z-stack data and want to combine all slices into one image.",
            parent_widget=self
        )
        channel_layout.addWidget(z_projection_info_icon)
        channel_layout.addStretch()
        params_layout.addLayout(channel_layout)
        self.channel_select = QComboBox()
        self.channel_select.addItem("All channels")
        self.channel_select.setEnabled(False)
        params_layout.addWidget(self.channel_select)
        corrections_group = QGroupBox("Image Corrections")
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
            "Which channels: Tick only the channels that are fluorescence (e.g. GFP, mScarlet); "
            "bleach correction is applied only to those. Leave phase/transmission unticked. "
            "Checkboxes appear here after loading a multi-channel file.\n\n"
            "Methods:\n"
            "• Exponential: I(t)=I₀×exp(-t/τ), normalizes to first frame. Use for intensity quantification.\n"
            "• Linear: Linear decay fit, then scale to first frame.\n"
            "• Histogram matching: Match each frame's histogram to frame 0. Good for segmentation; not for quantification.\n"
            "• Per-frame min-max: Stretch each frame so its min→0 and max→full range of the current bit depth (e.g. 65535 for 16-bit). No RGB conversion; dtype and bit depth are preserved. Visualization only.",
            parent_widget=self
        )
        corrections_layout.addWidget(self.bleach_correct)
        corrections_layout.addWidget(QLabel("Method:"))
        corrections_layout.addWidget(self.bleach_method_combo)
        corrections_layout.addWidget(bleach_info_icon)
        self.bleach_channel_widget = QWidget()
        self.bleach_channel_layout = QHBoxLayout()
        self.bleach_channel_layout.setContentsMargins(0, 0, 0, 0)
        self.bleach_channel_checks = []
        self.bleach_channel_widget.setLayout(self.bleach_channel_layout)
        self.bleach_channel_widget.setVisible(False)
        corrections_layout.addWidget(self.bleach_channel_widget)
        corrections_layout.addStretch()
        corrections_group.setLayout(corrections_layout)
        params_layout.addWidget(corrections_group)
        params_group.setLayout(params_layout)
        return params_group

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
    
    def select_output_dir(self):
        start_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_ND2)
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", start_dir)
        if dir_path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_ND2, dir_path)
            self.output_dir.setText(dir_path)
    
    def _rebuild_bleach_channel_checks(self, channel_names, num_channels):
        """Populate fluorescence channel checkboxes (which channels get bleach correction). All checked by default."""
        # Remove old checkboxes
        for cb in self.bleach_channel_checks:
            self.bleach_channel_layout.removeWidget(cb)
            cb.deleteLater()
        self.bleach_channel_checks.clear()
        if num_channels <= 1:
            self.bleach_channel_widget.setVisible(False)
            return
        for i, name in enumerate(channel_names):
            cb = QCheckBox(f"Ch{i+1}: {name}")
            cb.setChecked(False)  # Default: none selected; user ticks fluorescence channels
            cb.setToolTip("Tick if this channel is fluorescence and should get bleach correction")
            self.bleach_channel_layout.addWidget(cb)
            self.bleach_channel_checks.append(cb)
        self.bleach_channel_widget.setVisible(True)
    
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
                'pixel_size': first_file_info['pixel_size']
            }
            # Update UI with first file's parameters (user can adjust)
            self.max_frames.setMaximum(first_file_info['num_time'])
            self.max_frames.setValue(first_file_info['num_time'])  # Automatically set to maximum
            self.channel_select.clear()
            self.channel_select.addItem("All channels")
            for i, name in enumerate(first_file_info['channel_names']):
                self.channel_select.addItem(f"Channel {i+1}: {name}")
            self.channel_select.setEnabled(first_file_info['num_channels'] > 1)
            # Fluorescence channel checkboxes: one per channel, all checked by default (bleach applies to all unless user unticks)
            self._rebuild_bleach_channel_checks(first_file_info['channel_names'], first_file_info['num_channels'])
            self.preview_btn.setEnabled(True)
            self.convert_btn.setEnabled(True)
            
            # Use pixel size from first file if available
            if first_file_info['pixel_size']:
                self.parent_window.shared_metadata['pixel_size'] = first_file_info['pixel_size']
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
        """Extract ND2 file info without updating UI (for folder scanning)"""
        try:
            import nd2
            with nd2.ND2File(file_path) as nd2_reader:
                sizes = nd2_reader.sizes
                try:
                    shape = nd2_reader.shape
                    num_time = shape[0] if len(shape) > 0 else sizes.get('T', 1)
                except Exception:
                    num_time = sizes.get('T', 1)
                
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                channel_names = []
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        if hasattr(meta, 'channels') and meta.channels:
                            channel_names = [ch.channel.name if hasattr(ch, 'channel') and hasattr(ch.channel, 'name') 
                                           else f'Channel_{i+1}' 
                                           for i, ch in enumerate(meta.channels)]
                except Exception:
                    pass
                
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                pixel_size = None
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        if hasattr(meta, 'channels') and meta.channels:
                            first_channel = meta.channels[0]
                            if hasattr(first_channel, 'volume') and first_channel.volume:
                                vol = first_channel.volume
                                if hasattr(vol, 'axesCalibration') and vol.axesCalibration:
                                    if len(vol.axesCalibration) >= 2:
                                        pixel_size = vol.axesCalibration[0]
                except Exception:
                    pass
                
                return {
                    'file_type': 'nd2',
                    'num_time': num_time,
                    'num_channels': num_channels,
                    'num_z': num_z,
                    'channel_names': channel_names,
                    'pixel_size': pixel_size
                }
        except Exception:
            return None
    
    def _get_lif_info(self, file_path):
        """Extract LIF file info without updating UI (for folder scanning)"""
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
                
                pixel_size = None
                try:
                    if hasattr(lif_reader, 'physical_pixel_sizes'):
                        if lif_reader.physical_pixel_sizes and len(lif_reader.physical_pixel_sizes) >= 3:
                            pixel_size = float(lif_reader.physical_pixel_sizes[2])
                except:
                    pass
                
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
            return None
    
    def _get_tiff_info(self, file_path):
        """Extract TIFF file info without updating UI (for folder scanning)"""
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
                
                # Check if multi-page TIFF (time series)
                num_pages = len(tif.pages)
                if num_pages > 1 and ndim == 2:
                    # Multi-page 2D images - treat as time series
                    num_time = num_pages
                    num_channels = 1
                    num_z = 1
                
                # Get pixel size from metadata if available
                pixel_size = None
                if 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                elif 'unit' in metadata and 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                
                channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
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
    
    def _get_czi_info(self, file_path):
        """Extract CZI file info without updating UI (for folder scanning)"""
        try:
            import numpy as np
            from aicsimageio import AICSImage
            # AICSImage doesn't support context manager protocol, use direct instantiation
            czi_reader = AICSImage(file_path)
            try:
                # Get dimensions - handle both old and new aicsimageio API
                dims = czi_reader.dims
                
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
                    if hasattr(czi_reader, 'get_xarray_data'):
                        xdata = czi_reader.get_xarray_data()
                        num_time = int(xdata.sizes.get('T', 1))
                        num_channels = int(xdata.sizes.get('C', 1))
                        num_z = int(xdata.sizes.get('Z', 1))
                    else:
                        # Fallback to shape attribute
                        shape = czi_reader.shape
                        # Convert shape to tuple/list if needed
                        if not isinstance(shape, (tuple, list)):
                            shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                        
                        # Access dimensions by index in dims string
                        num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                        num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                        num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                except (KeyError, IndexError, AttributeError, TypeError) as e:
                    # If dimension access fails, try to get from shape directly
                    shape = czi_reader.shape
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
                    if hasattr(czi_reader, 'channel_names'):
                        channel_names = czi_reader.channel_names
                except:
                    pass
                
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                pixel_size = None
                try:
                    if hasattr(czi_reader, 'physical_pixel_sizes'):
                        if czi_reader.physical_pixel_sizes and len(czi_reader.physical_pixel_sizes) >= 3:
                            pixel_size = float(czi_reader.physical_pixel_sizes[2])
                except:
                    pass
                
                return {
                    'file_type': 'czi',
                    'num_time': num_time,
                    'num_channels': num_channels,
                    'num_z': num_z,
                    'channel_names': channel_names,
                    'pixel_size': pixel_size
                }
            finally:
                # Clean up if needed (AICSImage may have resources to close)
                if hasattr(czi_reader, 'close'):
                    try:
                        czi_reader.close()
                    except:
                        pass
        except Exception:
            return None
    
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
        """Load and display ND2 file information"""
        try:
            # Lazy import to avoid loading if not needed
            import nd2
            
            self.parent_window.log_output(f"Loading ND2 file info: {file_path}\n")
            
            with nd2.ND2File(file_path) as nd2_reader:
                sizes = nd2_reader.sizes
                # Use sizes dictionary directly - it properly maps dimension names to counts
                # Don't assume dimension order from shape[0] as ND2 files can have different orders
                num_time = sizes.get('T', 1)
                num_channels = sizes.get('C', 1)
                num_z = sizes.get('Z', 1)
                
                # Verify by checking actual data if needed (but trust sizes first)
                # Sometimes ND2 files report dimensions incorrectly, so double-check with shape
                try:
                    shape = nd2_reader.shape
                    # Get dimension order from ND2 reader
                    if hasattr(nd2_reader, 'sizes') and hasattr(nd2_reader.sizes, 'keys'):
                        # Check if shape matches reported sizes
                        # If shape[0] matches channels instead of time, sizes might be wrong
                        # But we'll trust sizes.get() as it's the proper way
                        pass
                except Exception:
                    pass
                
                # Get channel names
                channel_names = []
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        # Metadata is an object, not a dict - access channels attribute
                        if hasattr(meta, 'channels') and meta.channels:
                            channel_names = [ch.channel.name if hasattr(ch, 'channel') and hasattr(ch.channel, 'name') 
                                           else f'Channel_{i+1}' 
                                           for i, ch in enumerate(meta.channels)]
                except Exception as e:
                    # Silently fail - will use default names
                    pass
                
                if not channel_names or len(channel_names) != num_channels:
                    channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                # Extract pixel size from metadata
                pixel_size = None
                metadata_str = "Metadata:\n"
                
                try:
                    if hasattr(nd2_reader, 'metadata') and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        
                        # Pixel size is in volume.axesCalibration (first element is X, second is Y)
                        # Both should be the same for square pixels
                        if hasattr(meta, 'channels') and meta.channels:
                            # Get pixel size from first channel's volume
                            first_channel = meta.channels[0]
                            if hasattr(first_channel, 'volume') and first_channel.volume:
                                vol = first_channel.volume
                                if hasattr(vol, 'axesCalibration') and vol.axesCalibration:
                                    # axesCalibration is a list: [X, Y, Z] pixel sizes
                                    if len(vol.axesCalibration) >= 2:
                                        # Use X pixel size (should be same as Y for square pixels)
                                        pixel_size = vol.axesCalibration[0]
                        
                        # Format metadata for display
                        if pixel_size:
                            metadata_str += f"  Pixel Size: {pixel_size:.4f} µm\n"
                        else:
                            metadata_str += "  Pixel Size: Not found in metadata\n"
                        
                        # Add other metadata if available
                        if hasattr(meta, 'contents') and meta.contents:
                            if hasattr(meta.contents, 'frameCount'):
                                metadata_str += f"  Frame Count: {meta.contents.frameCount}\n"
                            if hasattr(meta.contents, 'channelCount'):
                                metadata_str += f"  Channel Count: {meta.contents.channelCount}\n"
                        
                except Exception as e:
                    metadata_str += f"  Error reading metadata: {str(e)}\n"
                
                # Use common UI update function
                self._update_file_info_ui(file_path, 'nd2', num_time, num_channels, num_z, 
                                        channel_names, pixel_size, metadata_str)
                
        except ImportError:
            QMessageBox.warning(self, "ND2 Library Missing", 
                              "The 'nd2' library is not installed.\n\n"
                              "Install it with: pip install nd2")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading ND2", f"Could not load ND2 file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _load_lif_info(self, file_path):
        """Load and display LIF file information"""
        try:
            # Try using aicsimageio for LIF files
            try:
                import numpy as np
                from aicsimageio import AICSImage
                self.parent_window.log_output(f"Loading LIF file info: {file_path}\n")
                
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
                    
                    # Get channel names if available
                    channel_names = []
                    try:
                        if hasattr(lif_reader, 'channel_names'):
                            channel_names = lif_reader.channel_names
                    except:
                        pass
                    
                    if not channel_names or len(channel_names) != num_channels:
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                    
                    # Extract pixel size
                    pixel_size = None
                    try:
                        if hasattr(lif_reader, 'physical_pixel_sizes'):
                            # physical_pixel_sizes is typically (Z, Y, X)
                            if lif_reader.physical_pixel_sizes and len(lif_reader.physical_pixel_sizes) >= 3:
                                pixel_size = float(lif_reader.physical_pixel_sizes[2])  # X dimension
                    except:
                        pass
                    
                    self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z, 
                                            channel_names, pixel_size)
                finally:
                    # Clean up if needed (AICSImage may have resources to close)
                    if hasattr(lif_reader, 'close'):
                        try:
                            lif_reader.close()
                        except:
                            pass
                    
            except ImportError:
                # Fallback to readlif if aicsimageio not available
                try:
                    from readlif.reader import LifFile
                    self.parent_window.log_output(f"Loading LIF file info: {file_path}\n")
                    
                    lif_file = LifFile(file_path)
                    # Get first image series
                    if len(lif_file.get_image_count()) > 0:
                        image = lif_file.get_image(0)
                        num_time = image.get_dims()['t'] if 't' in image.get_dims() else 1
                        num_channels = image.get_dims()['c'] if 'c' in image.get_dims() else 1
                        num_z = image.get_dims()['z'] if 'z' in image.get_dims() else 1
                        
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                        pixel_size = None  # readlif doesn't easily provide pixel size
                        
                        self._update_file_info_ui(file_path, 'lif', num_time, num_channels, num_z,
                                                channel_names, pixel_size)
                    else:
                        raise ValueError("No image series found in LIF file")
                except ImportError:
                    QMessageBox.warning(self, "LIF Library Missing",
                                      "The 'aicsimageio' or 'readlif' library is not installed.\n\n"
                                      "Install with: pip install aicsimageio readlif")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading LIF", f"Could not load LIF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _load_czi_info(self, file_path):
        """Load and display CZI file information"""
        try:
            # Try using aicsimageio for CZI files
            try:
                import numpy as np
                from aicsimageio import AICSImage
                self.parent_window.log_output(f"Loading CZI file info: {file_path}\n")
                
                # AICSImage doesn't support context manager protocol, use direct instantiation
                czi_reader = AICSImage(file_path)
                try:
                    # Get dimensions - handle both old and new aicsimageio API
                    dims = czi_reader.dims
                    
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
                        if hasattr(czi_reader, 'get_xarray_data'):
                            xdata = czi_reader.get_xarray_data()
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            # Fallback to shape attribute
                            shape = czi_reader.shape
                            # Convert shape to tuple/list if needed
                            if not isinstance(shape, (tuple, list)):
                                shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)
                            
                            # Access dimensions by index in dims string
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        # If dimension access fails, try to get from shape directly
                        shape = czi_reader.shape
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
                        if hasattr(czi_reader, 'channel_names'):
                            channel_names = czi_reader.channel_names
                    except:
                        pass
                    
                    if not channel_names or len(channel_names) != num_channels:
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                    
                    # Extract pixel size
                    pixel_size = None
                    try:
                        if hasattr(czi_reader, 'physical_pixel_sizes'):
                            if czi_reader.physical_pixel_sizes and len(czi_reader.physical_pixel_sizes) >= 3:
                                pixel_size = float(czi_reader.physical_pixel_sizes[2])  # X dimension
                    except:
                        pass
                    
                    self._update_file_info_ui(file_path, 'czi', num_time, num_channels, num_z,
                                            channel_names, pixel_size)
                finally:
                    # Clean up if needed (AICSImage may have resources to close)
                    if hasattr(czi_reader, 'close'):
                        try:
                            czi_reader.close()
                        except:
                            pass
                    
            except ImportError:
                # Fallback to czifile if aicsimageio not available
                try:
                    import czifile
                    self.parent_window.log_output(f"Loading CZI file info: {file_path}\n")
                    
                    with czifile.CziFile(file_path) as czi_file:
                        # Get metadata
                        metadata = czi_file.metadata()
                        # Parse dimensions from metadata
                        # This is simplified - czifile metadata parsing can be complex
                        num_time = 1
                        num_channels = 1
                        num_z = 1
                        channel_names = ['Channel_1']
                        pixel_size = None
                        
                        # Try to extract from shape
                        if hasattr(czi_file, 'shape'):
                            shape = czi_file.shape
                            # CZI shape is typically (T, C, Z, Y, X) or similar
                            if len(shape) >= 3:
                                num_time = shape[0] if len(shape) > 0 else 1
                                num_channels = shape[1] if len(shape) > 1 else 1
                                num_z = shape[2] if len(shape) > 2 else 1
                        
                        channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                        
                        self._update_file_info_ui(file_path, 'czi', num_time, num_channels, num_z,
                                                channel_names, pixel_size)
                except ImportError:
                    QMessageBox.warning(self, "CZI Library Missing",
                                      "The 'aicsimageio' or 'czifile' library is not installed.\n\n"
                                      "Install with: pip install aicsimageio czifile")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading CZI", f"Could not load CZI file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _load_tiff_info(self, file_path):
        """Load and display TIFF file information"""
        try:
            import tifffile
            import numpy as np
            self.parent_window.log_output(f"Loading TIFF file info: {file_path}\n")
            
            # Read TIFF file to get dimensions
            with tifffile.TiffFile(file_path) as tif:
                # Get shape from first page
                first_page = tif.pages[0]
                shape = first_page.shape
                
                # Try to get metadata
                metadata = {}
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    metadata = tif.imagej_metadata
                
                # Determine dimensions from shape
                ndim = len(shape)
                num_pages = len(tif.pages)
                
                if ndim == 2:
                    # Single 2D image or multi-page
                    if num_pages > 1:
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
                
                # Get pixel size from metadata if available
                pixel_size = None
                if 'pixelSize' in metadata:
                    pixel_size = float(metadata['pixelSize'])
                
                channel_names = [f'Channel_{i+1}' for i in range(num_channels)]
                
                self._update_file_info_ui(file_path, 'tiff', num_time, num_channels, num_z,
                                        channel_names, pixel_size)
                
        except Exception as e:
            QMessageBox.critical(self, "Error Loading TIFF", f"Could not load TIFF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _update_file_info_ui(self, file_path, file_type, num_time, num_channels, num_z, 
                            channel_names, pixel_size, metadata_str=None):
        """Update UI with file information (common for all file types)"""
        # Store info
        self.file_info = {
            'path': file_path,
            'file_type': file_type,
            'num_time': num_time,
            'num_channels': num_channels,
            'num_z': num_z,
            'channel_names': channel_names,
            'pixel_size': pixel_size
        }
        
        # Update frame selector - automatically set to max frames
        self.max_frames.setMaximum(num_time)
        self.max_frames.setValue(num_time)  # Automatically set to maximum
        
        # Update channel selector
        self.channel_select.clear()
        self.channel_select.addItem("All channels")
        for i, name in enumerate(channel_names):
            self.channel_select.addItem(f"Channel {i+1}: {name}")
        self.channel_select.setEnabled(num_channels > 1)
        self._rebuild_bleach_channel_checks(channel_names, num_channels)
        
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
        else:
            metadata_content += metadata_str
        
        self.metadata_text.setText(metadata_content)
        self.metadata_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-size: 10pt;")
        
        if pixel_size:
            # Update shared metadata and propagate to all tabs
            self.parent_window.shared_metadata['pixel_size'] = pixel_size
            self.parent_window.update_all_tabs_with_metadata()
        else:
            # Clear shared metadata
            self.parent_window.shared_metadata['pixel_size'] = None
        
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
        When bleach_apply is True, optional checkbox 'Show bleach correction' toggles live preview of correction."""
        import numpy as np
        import sys
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QKeySequence, QShortcut

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
        frame_spin.setToolTip(f"Jump to frame 1–{num_time}. Enter a number and press Enter.")
        nav_layout.addWidget(frame_spin)
        nav_layout.addWidget(QLabel(f"of {num_time}"))
        
        # Channel selection (if multiple channels)
        if num_channels > 1:
            nav_layout.addWidget(QLabel("Channel:"))
            channel_spin = QSpinBox()
            channel_spin.setRange(1, num_channels)
            channel_spin.setValue(1)
            channel_spin.setToolTip(f"Select channel to preview (1-{num_channels})")
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
        
        # Connect channel selection (if available)
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
            channel_spin.valueChanged.connect(on_channel_changed)
        
        # Connect "Show bleach correction" checkbox: always use current main-window settings when computing
        if show_bleach_check is not None and bleach_status_label is not None:
            def compute_corrected_from_tab():
                """Read current method/channels from main window (self) and compute corrected stack."""
                kwargs = self._get_bleach_preview_kwargs()
                cur_method = kwargs['bleach_method']
                cur_channels = kwargs.get('bleach_channel_indices')
                try:
                    from scripts.convert_nd2_to_tif import bleach_correct_stack
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
        
        # editingFinished + debounced valueChanged (300ms) for frame spinbox
        def on_frame_changed():
            user_frame = frame_spin.value()  # 1-based: 1 = first frame
            idx = user_frame - 1
            if 0 <= idx < num_time:
                go_to_frame(idx)
            else:
                frame_spin.blockSignals(True)
                frame_spin.setValue(preview_dialog.current_frame + 1)
                frame_spin.blockSignals(False)

        frame_spin.editingFinished.connect(lambda: QTimer.singleShot(0, on_frame_changed))
        frame_spin._change_timer = QTimer()
        frame_spin._change_timer.setSingleShot(True)
        frame_spin._change_timer.timeout.connect(on_frame_changed)

        def on_value_changed(v):
            if 1 <= v <= num_time:
                frame_spin._change_timer.start(300)
            else:
                frame_spin._change_timer.stop()
                frame_spin.blockSignals(True)
                frame_spin.setValue(preview_dialog.current_frame + 1)
                frame_spin.blockSignals(False)
        frame_spin.valueChanged.connect(on_value_changed)
        
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
                
                # Load all frames (for navigation)
                all_frames = nd2_reader.asarray()
            
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
                                  "This is a known issue with frozen builds.\n"
                                  "Try running from source: python main.py")
            else:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  f"Preview requires matplotlib:\n\npip install matplotlib")
        except Exception as e:
            error_msg = str(e)
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "PyTorch initialization error.\n\n"
                                  "This is a known issue with frozen builds and PyTorch.\n"
                                  "Try restarting the application or running from source.")
            else:
                QMessageBox.critical(self, "Preview Error", f"Could not preview file:\n{error_msg}")
            import traceback
            traceback.print_exc()
    
    def _preview_lif(self):
        """Preview LIF file in a dialog with frame navigation"""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            from aicsimageio import AICSImage
            
            file_path = self.file_path.text()
            
            # Create preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            
            layout = QVBoxLayout()
            
            # Load LIF file
            lif_reader = AICSImage(file_path)
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
                
                # Reshape to (T, Y, X) for preview (take first channel and z-slice)
                if len(all_frames.shape) > 3:
                    # Take first channel and first z-slice
                    if 'C' in dims_str and 'Z' in dims_str:
                        c_idx = dims_str.index('C')
                        z_idx = dims_str.index('Z')
                        t_idx = dims_str.index('T') if 'T' in dims_str else 0
                        # Create indexing tuple
                        indices = [slice(None)] * len(all_frames.shape)
                        indices[c_idx] = 0  # First channel
                        indices[z_idx] = 0  # First z-slice
                        all_frames = all_frames[tuple(indices)]
                    elif 'C' in dims_str:
                        c_idx = dims_str.index('C')
                        indices = [slice(None)] * len(all_frames.shape)
                        indices[c_idx] = 0
                        all_frames = all_frames[tuple(indices)]
                    elif 'Z' in dims_str:
                        z_idx = dims_str.index('Z')
                        indices = [slice(None)] * len(all_frames.shape)
                        indices[z_idx] = 0
                        all_frames = all_frames[tuple(indices)]
                
                # Ensure we have (T, Y, X) shape
                if len(all_frames.shape) == 2:
                    all_frames = all_frames[np.newaxis, ...]
                elif len(all_frames.shape) == 3 and all_frames.shape[0] > 1000:
                    # Likely (Y, X, T) - transpose
                    all_frames = all_frames.transpose(2, 0, 1)
                
                # Use unified preview UI with full LUT controls (and optional bleach preview)
                self._create_unified_preview_ui(
                    preview_dialog, layout, all_frames, num_time, num_channels, num_z, file_path,
                    **self._get_bleach_preview_kwargs()
                )
                
            finally:
                if hasattr(lif_reader, 'close'):
                    try:
                        lif_reader.close()
                    except:
                        pass
                        
        except ImportError:
            QMessageBox.warning(self, "Library Missing", 
                              "The 'aicsimageio' library is required for LIF preview.\n\n"
                              "Install with: pip install aicsimageio")
        except Exception as e:
            QMessageBox.critical(self, "Error Previewing LIF", f"Could not preview LIF file:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
    
    def _preview_czi(self):
        """Preview CZI file (stub - to be implemented)"""
        QMessageBox.information(self, "CZI Preview", 
                              "CZI file preview is not yet fully implemented.\n"
                              "Please use ND2, LIF, or TIFF files for preview.")
    
    def _preview_tiff(self):
        """Preview TIFF file in a dialog with frame navigation"""
        try:
            import tifffile
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            file_path = self.file_path.text()
            
            # Create preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("File Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            
            layout = QVBoxLayout()
            
            # Load TIFF file
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
            QMessageBox.information(self, "TIFF File", 
                                  "The selected file is already a TIFF file.\n"
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
        cmd_args = [
            str(first_path),
            '--frames', str(self.max_frames.value()),
        ]
        
        if self.output_dir.text():
            cmd_args.extend(['--output', self.output_dir.text()])
        
        if self.separate_channels.isChecked():
            cmd_args.append('--separate-channels')
        else:
            cmd_args.append('--combine-channels')
        
        # Max Z projection
        if self.max_z_projection.isChecked():
            cmd_args.append('--max-z-projection')
        
        # Channel selection
        selected_channel = self.channel_select.currentIndex() - 1  # -1 means all
        if selected_channel >= 0:
            cmd_args.extend(['--channel', str(selected_channel)])
        
        if self.drift_correct.isChecked():
            cmd_args.append('--drift-correct')
        
        if self.bleach_correct.isChecked():
            cmd_args.append('--bleach-correct')
            bleach_methods = ['exponential', 'linear', 'histogram_matching', 'per_frame_minmax']
            cmd_args.append('--bleach-method')
            cmd_args.append(bleach_methods[self.bleach_method_combo.currentIndex()])
            # Only pass --bleach-channels when user has flagged specific fluorescence channels
            if getattr(self, 'bleach_channel_checks', None):
                indices = [i for i, cb in enumerate(self.bleach_channel_checks) if cb.isChecked()]
                if indices and len(indices) < len(self.bleach_channel_checks):
                    # Not all channels selected: pass only the flagged ones
                    cmd_args.append('--bleach-channels')
                    cmd_args.extend(str(i) for i in indices)
        
        # Pass pixel size if extracted
        if self.file_info and self.file_info.get('pixel_size'):
            cmd_args.extend(['--pixel-size', str(self.file_info['pixel_size'])])
        
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
            return
        # Success: if we have a queue and more files to convert, run next
        queue = getattr(self, '_conversion_queue', None)
        idx = getattr(self, '_conversion_current_index', 0)
        if queue and idx + 1 < len(queue):
            self._conversion_current_index = idx + 1
            next_path = queue[self._conversion_current_index]
            from bactimate.gui.constants import CONVERT_ND2_SCRIPT
            convert_script = CONVERT_ND2_SCRIPT
            cmd_args = [
                str(next_path),
                '--frames', str(self.max_frames.value()),
            ]
            if self.output_dir.text():
                cmd_args.extend(['--output', self.output_dir.text()])
            if self.separate_channels.isChecked():
                cmd_args.append('--separate-channels')
            else:
                cmd_args.append('--combine-channels')
            if self.max_z_projection.isChecked():
                cmd_args.append('--max-z-projection')
            selected_channel = self.channel_select.currentIndex() - 1
            if selected_channel >= 0:
                cmd_args.extend(['--channel', str(selected_channel)])
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
            
            # Determine output directory
            if self.output_dir.text():
                output_base = Path(self.output_dir.text())
            else:
                output_base = file_path.parent if file_path.is_file() else file_path
            
            # Get conversion parameters
            max_frames = self.max_frames.value()
            separate_channels = self.separate_channels.isChecked()
            selected_channel = self.channel_select.currentIndex() - 1  # -1 means all
            
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
                    
                    # Load LIF file
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
                        
                        # Limit frames
                        num_time = min(num_time, max_frames)
                        if 'T' in dims_str:
                            t_idx = dims_str.index('T')
                            indices = [slice(None)] * len(all_frames.shape)
                            indices[t_idx] = slice(0, num_time)
                            all_frames = all_frames[tuple(indices)]
                        
                        # Handle channel selection - validate bounds first
                        if selected_channel >= 0 and num_channels > 1:
                            if 'C' in dims_str:
                                c_idx = dims_str.index('C')
                                # Validate bounds before accessing
                                if c_idx < len(all_frames.shape) and selected_channel < all_frames.shape[c_idx]:
                                    indices = [slice(None)] * len(all_frames.shape)
                                    indices[c_idx] = selected_channel
                                    all_frames = all_frames[tuple(indices)]
                                    num_channels = 1
                                else:
                                    self.parent_window.log_output(f"  Warning: Selected channel {selected_channel + 1} is out of bounds (shape: {all_frames.shape}, channel dim: {c_idx}). Using all channels.\n")
                        
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
                        
                        # Create output filename
                        output_file = output_base / f"{lif_file.stem}.tif"
                        
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
                                
                                ch_output = output_base / f"{lif_file.stem}_ch{ch+1}.tif"
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
                        
                    finally:
                        if hasattr(lif_reader, 'close'):
                            try:
                                lif_reader.close()
                            except:
                                pass
                
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
                              "LIF conversion requires 'aicsimageio' and 'tifffile' libraries.\n\n"
                              "Install with: pip install aicsimageio tifffile")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"LIF conversion failed:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
    
    def _convert_czi(self):
        """Convert CZI file(s) to TIFF"""
        try:
            import numpy as np
            import tifffile
            from aicsimageio import AICSImage
            
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
            
            # Determine output directory
            if self.output_dir.text():
                output_base = Path(self.output_dir.text())
            else:
                output_base = file_path.parent if file_path.is_file() else file_path
            
            # Get conversion parameters
            max_frames = self.max_frames.value()
            separate_channels = self.separate_channels.isChecked()
            selected_channel = self.channel_select.currentIndex() - 1  # -1 means all
            
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
                    
                    # Load CZI file
                    czi_reader = AICSImage(str(czi_file))
                    try:
                        # Get dimensions
                        dims = czi_reader.dims
                        if hasattr(dims, 'order'):
                            dims_str = dims.order
                        elif isinstance(dims, str):
                            dims_str = dims
                        else:
                            dims_str = str(dims)
                        
                        # Get data
                        if hasattr(czi_reader, 'get_xarray_data'):
                            xdata = czi_reader.get_xarray_data()
                            all_frames = np.array(xdata)
                            num_time = int(xdata.sizes.get('T', 1))
                            num_channels = int(xdata.sizes.get('C', 1))
                            num_z = int(xdata.sizes.get('Z', 1))
                        else:
                            all_frames = np.array(czi_reader.data)
                            shape = all_frames.shape
                            num_time = shape[dims_str.index('T')] if 'T' in dims_str else 1
                            num_channels = shape[dims_str.index('C')] if 'C' in dims_str else 1
                            num_z = shape[dims_str.index('Z')] if 'Z' in dims_str else 1
                        
                        # Limit frames
                        num_time = min(num_time, max_frames)
                        if 'T' in dims_str:
                            t_idx = dims_str.index('T')
                            indices = [slice(None)] * len(all_frames.shape)
                            indices[t_idx] = slice(0, num_time)
                            all_frames = all_frames[tuple(indices)]
                        
                        # Handle channel selection - validate bounds first
                        if selected_channel >= 0 and num_channels > 1:
                            if 'C' in dims_str:
                                c_idx = dims_str.index('C')
                                # Validate bounds before accessing
                                if c_idx < len(all_frames.shape) and selected_channel < all_frames.shape[c_idx]:
                                    indices = [slice(None)] * len(all_frames.shape)
                                    indices[c_idx] = selected_channel
                                    all_frames = all_frames[tuple(indices)]
                                    num_channels = 1
                                else:
                                    self.parent_window.log_output(f"  Warning: Selected channel {selected_channel + 1} is out of bounds (shape: {all_frames.shape}, channel dim: {c_idx}). Using all channels.\n")
                        
                        # Take first z-slice if multiple z-slices
                        if num_z > 1 and 'Z' in dims_str:
                            z_idx = dims_str.index('Z')
                            if z_idx < len(all_frames.shape):
                                indices = [slice(None)] * len(all_frames.shape)
                                indices[z_idx] = 0
                                all_frames = all_frames[tuple(indices)]
                                num_z = 1
                        
                        # Reshape to (T, C, Y, X) or (T, Y, X)
                        # Don't use all_frames[0] as it might remove time dimension!
                        # Instead, find and remove only non-time singleton dimensions
                        if len(all_frames.shape) > 4:
                            # Find time dimension first
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
                        
                        # Create output filename
                        output_file = output_base / f"{czi_file.stem}.tif"
                        
                        # Save as TIFF - use more robust dimension handling
                        current_shape = all_frames.shape
                        channel_dim = None
                        
                        # Try to find channel dimension - look for dimension matching num_channels
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
                                
                                ch_output = output_base / f"{czi_file.stem}_ch{ch+1}.tif"
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
                        
                        self.parent_window.log_output(f"✓ Converted: {czi_file.name}\n")
                        
                    finally:
                        if hasattr(czi_reader, 'close'):
                            try:
                                czi_reader.close()
                            except:
                                pass
                
                except Exception as e:
                    self.parent_window.log_output(f"✗ Error converting {czi_file.name}: {str(e)}\n")
                    QMessageBox.warning(self, "Conversion Error", 
                                      f"Error converting {czi_file.name}:\n{str(e)}")
            
            # Complete
            self.parent_window.progress_bar.setValue(100)
            QMessageBox.information(self, "Conversion Complete", 
                                  f"Successfully converted {total_files} CZI file(s) to TIFF.")
            
        except ImportError:
            QMessageBox.warning(self, "Library Missing", 
                              "CZI conversion requires 'aicsimageio' and 'tifffile' libraries.\n\n"
                              "Install with: pip install aicsimageio tifffile")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"CZI conversion failed:\n{str(e)}")
            self.parent_window.log_output(f"Error: {str(e)}\n")
        finally:
            self.convert_btn.setEnabled(True)
            self.convert_btn.setText("Convert to TIFF")
            self.parent_window.progress_bar.setVisible(False)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        out = get_from_widgets([
            ("max_frames", self.max_frames),
            ("separate_channels", self.separate_channels),
            ("max_z_projection", self.max_z_projection),
            ("drift_correct", self.drift_correct),
            ("bleach_correct", self.bleach_correct),
        ])
        if hasattr(self, "bleach_method_combo"):
            out["bleach_method"] = self.bleach_method_combo.currentIndex()
        if getattr(self, "bleach_channel_checks", None):
            out["bleach_fluorescence_channels"] = [cb.isChecked() for cb in self.bleach_channel_checks]
        return out

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("max_frames", self.max_frames),
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
        if "bleach_fluorescence_channels" in data and getattr(self, "bleach_channel_checks", None):
            try:
                states = data["bleach_fluorescence_channels"]
                if isinstance(states, list) and len(states) == len(self.bleach_channel_checks):
                    for cb, checked in zip(self.bleach_channel_checks, states):
                        cb.setChecked(bool(checked))
            except (TypeError, ValueError):
                pass

