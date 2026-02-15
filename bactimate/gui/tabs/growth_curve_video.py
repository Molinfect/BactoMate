"""
Growth curve video tab: create side-by-side growth curve graph and video from TIFF + Excel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.preferences import get_default_browse_dir
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog

from bactimate.gui.workers import GrowthCurveVideoWorker

class GrowthCurveVideoTab(QWidget):
    """Tab for creating growth curve videos with side-by-side graph and video"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        video_group = self._build_video_input_group()
        scroll_layout.addWidget(video_group)

        excel_time_widget = self._build_excel_time_section()
        scroll_layout.addWidget(excel_time_widget)

        settings_container = self._build_graph_and_video_settings()
        scroll_layout.addWidget(settings_container)

        # Process button
        self.process_btn = QPushButton("Create Growth Curve Video")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.create_video)
        scroll_layout.addWidget(self.process_btn)

        self.video_path.textChanged.connect(self._update_button_state)
        self.excel_path.textChanged.connect(self._update_button_state)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self._update_button_state()

    def _build_video_input_group(self) -> QWidget:
        """Build the video input (TIFF sequence) group."""
        video_group = DragDropGroupBox(
            "Video Input (TIFF Sequence)",
            parent=self,
            on_drop_callback=self.handle_dropped_video_path
        )
        video_layout = QVBoxLayout()
        file_folder_layout = QHBoxLayout()
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Select TIFF file or folder containing TIFF sequence... (or drag and drop)")
        self.video_path.setMinimumWidth(500)
        file_font = QFont()
        file_font.setPointSize(11)
        self.video_path.setFont(file_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.video_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_video_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(120)
        folder_btn.clicked.connect(self.select_video_folder)
        file_folder_layout.addWidget(self.video_path, 0)
        file_folder_layout.addWidget(file_btn, 0)
        file_folder_layout.addWidget(folder_btn, 0)
        video_layout.addLayout(file_folder_layout)
        video_group.setLayout(video_layout)
        return video_group

    def _build_excel_time_section(self) -> QWidget:
        """Build Excel file + time settings side-by-side."""
        file_font = QFont()
        file_font.setPointSize(11)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        excel_group = DragDropGroupBox(
            "Excel Analysis File",
            parent=self,
            on_drop_callback=self.handle_dropped_excel_path
        )
        excel_layout = QHBoxLayout()
        self.excel_path = QLineEdit()
        self.excel_path.setPlaceholderText("Select Excel file from Analyze Masks tab (e.g., *_masks_analysis.xlsx)... (or drag and drop)")
        self.excel_path.setMinimumWidth(500)
        self.excel_path.setFont(file_font)
        self.excel_path.setSizePolicy(size_policy)
        excel_btn = QPushButton("Browse File...")
        excel_btn.setFixedWidth(120)
        excel_btn.clicked.connect(self.select_excel_file)
        excel_info_icon = create_info_icon(
            "Excel file from the Analyze Masks tab containing growth data.\n\n"
            "Required columns:\n"
            "- Frame: Frame number (1, 2, 3, ...)\n"
            "- Total Cell Area (Relative): Normalized cell area (first frame = 1.0)",
            parent_widget=self
        )
        excel_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        excel_layout.addWidget(self.excel_path, 0)
        excel_layout.addWidget(excel_btn, 0)
        excel_layout.addWidget(excel_info_icon, 0)
        excel_group.setLayout(excel_layout)
        
        # Time interval input (next to Excel file)
        time_group = QGroupBox("Time Settings")
        time_layout = QHBoxLayout()
        time_label = QLabel("Time per frame (min.):")
        self.time_interval = QDoubleSpinBox()
        self.time_interval.setMinimum(0.0)
        self.time_interval.setMaximum(10000.0)
        self.time_interval.setValue(10.0)
        self.time_interval.setDecimals(2)
        self.time_interval.setSuffix(" min")
        time_info_icon = create_info_icon(
            "Time difference between consecutive frames.\n\n"
            "This is used to convert frame numbers to time (hours) for the x-axis of the growth curve.\n\n"
            "Example: If images were taken every 10 minutes, enter 10.0",
            parent_widget=self
        )
        time_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.time_interval.setFixedWidth(100)  # Fixed width for uniform appearance
        time_layout.addWidget(time_label)
        time_layout.addWidget(time_info_icon)
        time_layout.addStretch()  # Add stretch before spin box to align to right
        time_layout.addWidget(self.time_interval)
        time_group.setLayout(time_layout)
        
        # Excel and Time Settings side-by-side
        excel_time_container = QHBoxLayout()
        excel_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        time_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        excel_time_container.addWidget(excel_group)
        excel_time_container.addWidget(time_group)
        excel_time_widget = QWidget()
        excel_time_widget.setLayout(excel_time_container)
        return excel_time_widget

    def _build_graph_and_video_settings(self) -> QWidget:
        """Build graph settings + video settings + output formats side-by-side."""
        settings_container_layout = QHBoxLayout()
        
        # Graph settings
        graph_group = QGroupBox("Graph Settings")
        graph_layout = QVBoxLayout()
        
        # Y-axis min/max
        yaxis_layout = QHBoxLayout()
        ymin_label = QLabel("Y-axis min:")
        self.yaxis_min = QDoubleSpinBox()
        self.yaxis_min.setMinimum(0.0)
        self.yaxis_min.setMaximum(100.0)
        self.yaxis_min.setValue(0.0)
        self.yaxis_min.setDecimals(2)
        self.yaxis_min.setSpecialValueText("Auto")
        
        ymax_label = QLabel("Y-axis max:")
        self.yaxis_max = QDoubleSpinBox()
        self.yaxis_max.setMinimum(0.0)
        self.yaxis_max.setMaximum(100.0)
        self.yaxis_max.setValue(0.0)
        self.yaxis_max.setDecimals(2)
        self.yaxis_max.setSpecialValueText("Auto")
        
        graph_info_icon = create_info_icon(
            "Configure the Y-axis range for the growth curve graph.\n\n"
            "Y-axis represents: Relative cell area (normalized to first frame = 1.0)\n\n"
            "Y-axis min/max:\n"
            "- Set to 'Auto' (0.0) to automatically scale based on data\n"
            "- Set custom values to fix the Y-axis range\n"
            "- Useful for comparing multiple videos with consistent scaling\n\n"
            "Example: Set min=0.0, max=3.0 to show growth from 1x to 3x initial area",
            parent_widget=self
        )
        graph_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        yaxis_layout.addWidget(ymin_label)
        yaxis_layout.addWidget(self.yaxis_min)
        yaxis_layout.addWidget(ymax_label)
        yaxis_layout.addWidget(self.yaxis_max)
        yaxis_layout.addWidget(graph_info_icon)
        yaxis_layout.addStretch()
        graph_layout.addLayout(yaxis_layout)
        
        graph_group.setLayout(graph_layout)
        graph_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(graph_group)
        
        # Video settings
        video_settings_group = QGroupBox("Video Settings")
        video_settings_layout = QHBoxLayout()
        fps_label = QLabel("Frames per second (FPS):")
        self.fps = QDoubleSpinBox()
        self.fps.setMinimum(0.1)
        self.fps.setMaximum(60.0)
        self.fps.setValue(5.0)
        self.fps.setDecimals(1)
        self.fps.setSuffix(" fps")
        fps_info_icon = create_info_icon(
            "Playback speed for the output video.\n\n"
            "Higher values = faster playback\n"
            "Lower values = slower playback\n\n"
            "Default: 5.0 fps\n"
            "Typical range: 1-10 fps for time-lapse videos",
            parent_widget=self
        )
        fps_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        video_settings_layout.addWidget(fps_label)
        video_settings_layout.addWidget(self.fps)
        video_settings_layout.addWidget(fps_info_icon)
        video_settings_layout.addStretch()
        video_settings_group.setLayout(video_settings_layout)
        video_settings_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(video_settings_group)
        
        # Output format selection (in same row)
        format_group = QGroupBox("Output Formats")
        format_layout = QVBoxLayout()
        
        # Format checkboxes - all in a single row with "Select All" first
        checkbox_layout = QHBoxLayout()
        
        # "Select All" checkbox first
        self.format_all = QCheckBox("Select All")
        self.format_all.stateChanged.connect(self._on_select_all_changed)
        checkbox_layout.addWidget(self.format_all)
        
        # Add a separator (spacing) between "Select All" and format options
        checkbox_layout.addSpacing(20)
        
        # Format checkboxes
        self.format_tif = QCheckBox("TIFF")
        self.format_tif.setChecked(True)  # Default enabled
        self.format_avi = QCheckBox("AVI")
        self.format_avi.setChecked(True)  # Default enabled
        self.format_gif = QCheckBox("GIF")
        self.format_mp4 = QCheckBox("MP4")
        
        # Connect format checkboxes to update "Select All" state
        self.format_tif.stateChanged.connect(self._on_format_changed)
        self.format_avi.stateChanged.connect(self._on_format_changed)
        self.format_gif.stateChanged.connect(self._on_format_changed)
        self.format_mp4.stateChanged.connect(self._on_format_changed)
        
        checkbox_layout.addWidget(self.format_tif)
        checkbox_layout.addWidget(self.format_avi)
        checkbox_layout.addWidget(self.format_gif)
        checkbox_layout.addWidget(self.format_mp4)
        checkbox_layout.addStretch()
        format_layout.addLayout(checkbox_layout)
        
        format_group.setLayout(format_layout)
        format_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(format_group)
        
        settings_container = QWidget()
        settings_container.setLayout(settings_container_layout)
        return settings_container

    def _update_button_state(self):
        """Update process button state based on input fields"""
        has_video = bool(self.video_path.text())
        has_excel = bool(self.excel_path.text())
        has_format = (self.format_tif.isChecked() or self.format_avi.isChecked() or 
                     self.format_gif.isChecked() or self.format_mp4.isChecked())
        self.process_btn.setEnabled(has_video and has_excel and has_format)
    
    def _on_select_all_changed(self, state):
        """Handle 'Select All' checkbox change"""
        checked = state == 2  # Qt.Checked = 2
        self.format_tif.setChecked(checked)
        self.format_avi.setChecked(checked)
        self.format_gif.setChecked(checked)
        self.format_mp4.setChecked(checked)
    
    def _on_format_changed(self):
        """Handle individual format checkbox change"""
        all_checked = (self.format_tif.isChecked() and self.format_avi.isChecked() and 
                      self.format_gif.isChecked() and self.format_mp4.isChecked())
        self.format_all.blockSignals(True)
        self.format_all.setChecked(all_checked)
        self.format_all.blockSignals(False)
        self._update_button_state()
    
    def select_video_file(self):
        """Select TIFF file for video input"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF File",
            get_default_browse_dir(),
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)"
        )
        if file_path:
            self.video_path.setText(file_path)
    
    def select_video_folder(self):
        """Select folder containing TIFF sequence"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with TIFF Sequence",
            get_default_browse_dir()
        )
        if folder_path:
            self.video_path.setText(folder_path)
    
    def handle_dropped_video_path(self, dropped_path):
        """Handle dropped file or folder path for growth curve video tab"""
        path_obj = Path(dropped_path)
        # Update video path
        self.video_path.setText(dropped_path)

    def handle_dropped_excel_path(self, dropped_path):
        """Handle dropped path for Excel analysis file (file or folder with Excel files)."""
        path_obj = Path(dropped_path)
        if not path_obj.exists():
            return
        if path_obj.is_file():
            self.excel_path.setText(dropped_path)
        elif path_obj.is_dir():
            xlsx_files = list(path_obj.glob("*.xlsx")) + list(path_obj.glob("*.xls"))
            if xlsx_files:
                self.excel_path.setText(str(xlsx_files[0]))

    def select_excel_file(self):
        """Select Excel file from Analyze Masks tab"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel Analysis File",
            get_default_browse_dir(),
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.excel_path.setText(file_path)
    
    def _is_folder_mode(self):
        """Check if video input is a folder"""
        path = Path(self.video_path.text())
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
    
    def create_video(self):
        """Create growth curve video with side-by-side graph and video"""
        if not self.video_path.text() or not self.excel_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select both video input and Excel file.")
            return
        
        video_path = Path(self.video_path.text())
        excel_path = Path(self.excel_path.text())
        
        if not video_path.exists():
            QMessageBox.warning(self, "File Not Found", f"Video input not found:\n{video_path}")
            return
        
        if not excel_path.exists():
            QMessageBox.warning(self, "File Not Found", f"Excel file not found:\n{excel_path}")
            return
        
        # Determine output directory (go up one level if input is in Tif_* or _segmentation folder)
        if video_path.is_file():
            base_dir = video_path.parent
            input_stem = video_path.stem
        else:
            base_dir = video_path
            input_stem = video_path.name
        
        # Go up one level if in segmentation or Tif_ folder (same logic as other tabs)
        if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
            base_dir = base_dir.parent
        
        # Create output directory at base level
        output_dir = base_dir / f"{input_stem}_growth_curve_video"
        output_dir.mkdir(exist_ok=True)
        
        # Show progress
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Creating Video...")
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        
        # Get selected formats
        formats = []
        if self.format_tif.isChecked():
            formats.append('tif')
        if self.format_avi.isChecked():
            formats.append('avi')
        if self.format_gif.isChecked():
            formats.append('gif')
        if self.format_mp4.isChecked():
            formats.append('mp4')
        
        # Run in worker thread
        self.worker = GrowthCurveVideoWorker(
            video_path,
            excel_path,
            output_dir,
            self.time_interval.value(),
            self.yaxis_min.value() if self.yaxis_min.value() > 0 else None,
            self.yaxis_max.value() if self.yaxis_max.value() > 0 else None,
            self.fps.value(),
            formats,
            parent=self
        )
        self.worker.finished.connect(self.on_video_finished)
        self.worker.progress.connect(self.on_video_progress)
        self.worker.start()
    
    def on_video_progress(self, value, message):
        """Handle progress updates"""
        self.parent_window.progress_bar.setValue(value)
        if message:
            self.parent_window.log_output(message)
    
    def on_video_finished(self, success, message):
        """Handle video creation completion"""
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Create Growth Curve Video")
        self.parent_window.progress_bar.setVisible(False)
        
        if success:
            # Just log the message instead of showing a popup
            self.parent_window.log_output(f"\n✓ {message}\n")
        else:
            QMessageBox.warning(self, "Error", message)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        return get_from_widgets([
            ("time_interval", self.time_interval),
            ("yaxis_min", self.yaxis_min),
            ("yaxis_max", self.yaxis_max),
            ("fps", self.fps),
            ("format_tif", self.format_tif),
            ("format_avi", self.format_avi),
            ("format_gif", self.format_gif),
            ("format_mp4", self.format_mp4),
        ])

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("time_interval", self.time_interval),
            ("yaxis_min", self.yaxis_min),
            ("yaxis_max", self.yaxis_max),
            ("fps", self.fps),
            ("format_tif", self.format_tif),
            ("format_avi", self.format_avi),
            ("format_gif", self.format_gif),
            ("format_mp4", self.format_mp4),
        ])


