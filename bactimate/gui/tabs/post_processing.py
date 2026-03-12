"""
Post-processing tab: growth curve video, scale bar on image, and montage.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QColorDialog,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.image_utils import to_qpixmap, scale_for_preview
from bactimate.gui.preferences import get_default_browse_dir
from bactimate.modules.tiff_metadata import read_calibration_from_path, read_metadata_from_path
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.utils import safe_imread
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog

from bactimate.gui.workers import GrowthCurveVideoWorker

class PostProcessingTab(QWidget):
    """Tab for post-processing: growth curve video, scale bar on image, and montage."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.montage_channel_color_overrides: dict = {}  # (path_stem, ch_idx) -> "auto"|"gray"|"red"|"green"|...
        self.montage_tile_color_overrides: Dict[int, Tuple[float, float, float]] = {}  # tile_idx -> (r,g,b) 0-1
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # Order: Montage → Scale bar → Growth curve video (each in its own frame)
        montage_group = self._build_montage_group()
        scroll_layout.addWidget(montage_group)

        scalebar_group = self._build_scalebar_group()
        scroll_layout.addWidget(scalebar_group)

        growth_curve_group = self._build_growth_curve_video_group()
        scroll_layout.addWidget(growth_curve_group)

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

    def _build_growth_curve_video_group(self) -> QWidget:
        """Build the 'Growth curve video' frame: TIFF input, Excel + time, settings, and Create button."""
        group = DragDropGroupBox(
            "Growth curve video",
            parent=self,
            on_drop_callback=self.handle_dropped_video_path,
        )
        group_layout = QVBoxLayout()
        video_group = self._build_video_input_group()
        group_layout.addWidget(video_group)
        excel_time_widget = self._build_excel_time_section()
        group_layout.addWidget(excel_time_widget)
        settings_container = self._build_graph_and_video_settings()
        group_layout.addWidget(settings_container)
        self.process_btn = QPushButton("Create Growth Curve Video")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.create_video)
        group_layout.addWidget(self.process_btn)
        group.setLayout(group_layout)
        return group

    def _build_scalebar_group(self) -> QWidget:
        """Build the 'Add scale bar to image' group."""
        group = DragDropGroupBox(
            "Add scale bar to image",
            parent=self,
            on_drop_callback=lambda p: self.scalebar_path.setText(p) if getattr(self, "scalebar_path", None) else None,
        )
        layout = QVBoxLayout()
        # Image path
        path_row = QHBoxLayout()
        self.scalebar_path = QLineEdit()
        self.scalebar_path.setPlaceholderText("Image path (TIFF, PNG, or other)...")
        self.scalebar_path.setMinimumWidth(400)
        path_btn = QPushButton("Browse...")
        path_btn.clicked.connect(self._select_scalebar_image)
        self.scalebar_path.textChanged.connect(self._try_update_scalebar_pixel_size_from_path)
        path_row.addWidget(self.scalebar_path)
        path_row.addWidget(path_btn)
        layout.addLayout(path_row)
        # Pixel size and scale length
        opt_row1 = QHBoxLayout()
        opt_row1.addWidget(QLabel("Pixel size (µm/px):"))
        self.scalebar_pixel_size = QDoubleSpinBox()
        self.scalebar_pixel_size.setRange(0.001, 100.0)
        self.scalebar_pixel_size.setValue(0.108)
        self.scalebar_pixel_size.setDecimals(3)
        opt_row1.addWidget(self.scalebar_pixel_size)
        opt_row1.addWidget(QLabel("Scale bar length (µm):"))
        self.scalebar_length_um = QDoubleSpinBox()
        self.scalebar_length_um.setRange(0.1, 1000.0)
        self.scalebar_length_um.setValue(10.0)
        self.scalebar_length_um.setDecimals(1)
        opt_row1.addWidget(self.scalebar_length_um)
        opt_row1.addStretch()
        layout.addLayout(opt_row1)
        # Position and style
        opt_row2 = QHBoxLayout()
        opt_row2.addWidget(QLabel("Position:"))
        self.scalebar_position = QComboBox()
        self.scalebar_position.addItems(["Bottom-right", "Bottom-left", "Top-right", "Top-left"])
        opt_row2.addWidget(self.scalebar_position)
        opt_row2.addWidget(QLabel("Color:"))
        self.scalebar_color = QComboBox()
        self.scalebar_color.addItems(["White", "Black"])
        opt_row2.addWidget(self.scalebar_color)
        opt_row2.addWidget(QLabel("Thickness (px):"))
        self.scalebar_thickness = QSpinBox()
        self.scalebar_thickness.setRange(1, 20)
        self.scalebar_thickness.setValue(5)
        opt_row2.addWidget(self.scalebar_thickness)
        opt_row2.addWidget(QLabel("Font size:"))
        self.scalebar_font_size = QSpinBox()
        self.scalebar_font_size.setRange(8, 72)
        self.scalebar_font_size.setValue(20)
        opt_row2.addWidget(self.scalebar_font_size)
        self.scalebar_show_label = QCheckBox("Show label (e.g. 10 µm)")
        self.scalebar_show_label.setChecked(True)
        opt_row2.addWidget(self.scalebar_show_label)
        opt_row2.addStretch()
        layout.addLayout(opt_row2)
        # Buttons (50/50)
        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview scale bar")
        preview_btn.clicked.connect(self._preview_scalebar)
        save_btn = QPushButton("Save with scale bar")
        save_btn.clicked.connect(self._save_scalebar)
        btn_row.addWidget(preview_btn, 1)
        btn_row.addWidget(save_btn, 1)
        layout.addLayout(btn_row)
        group.setLayout(layout)
        return group

    def _build_montage_group(self) -> QWidget:
        """Build the Montage group: one image or folder → Phase first, then FI channels, then merged."""
        group = DragDropGroupBox(
            "Montage",
            parent=self,
            on_drop_callback=lambda p: self.montage_path.setText(p) if getattr(self, "montage_path", None) else None,
        )
        layout = QVBoxLayout()

        # —— Input ———
        path_row = QHBoxLayout()
        self.montage_path = QLineEdit()
        self.montage_path.setPlaceholderText("One ND2 or TIFF file, or a folder of images")
        self.montage_path.setMinimumWidth(400)
        folder_btn = QPushButton("Browse folder…")
        folder_btn.clicked.connect(self._select_montage_folder)
        files_btn = QPushButton("Add files…")
        files_btn.clicked.connect(self._select_montage_files)
        path_row.addWidget(self.montage_path)
        path_row.addWidget(folder_btn)
        path_row.addWidget(files_btn)
        layout.addLayout(path_row)
        montage_hint = QLabel("Output: Phase contrast first, then fluorescence channels (with colors), then merged image. Uses first frame only.")
        montage_hint.setStyleSheet("color: gray; font-size: 11px;")
        montage_hint.setWordWrap(True)
        layout.addWidget(montage_hint)

        # —— Layout (grid) ———
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Rows:"))
        self.montage_rows = QSpinBox()
        self.montage_rows.setRange(1, 50)
        self.montage_rows.setValue(1)
        self.montage_rows.setToolTip("Number of rows. Default 1 = single row of tiles.")
        grid_row.addWidget(self.montage_rows)
        grid_row.addWidget(QLabel("Cols:"))
        self.montage_cols = QSpinBox()
        self.montage_cols.setRange(0, 50)
        self.montage_cols.setValue(0)
        self.montage_cols.setSpecialValueText("Auto")
        self.montage_cols.setToolTip("Auto = derived from number of tiles and rows.")
        grid_row.addWidget(self.montage_cols)
        grid_row.addWidget(QLabel("Spacing (px):"))
        self.montage_spacing = QSpinBox()
        self.montage_spacing.setRange(0, 100)
        self.montage_spacing.setValue(4)
        grid_row.addWidget(self.montage_spacing)
        grid_row.addWidget(QLabel("Border (px):"))
        self.montage_border = QSpinBox()
        self.montage_border.setRange(0, 100)
        self.montage_border.setValue(8)
        grid_row.addWidget(self.montage_border)
        grid_row.addWidget(QLabel("Order:"))
        self.montage_order = QComboBox()
        self.montage_order.addItems(["Row-major (left→right, top→bottom)", "Column-major (top→bottom, left→right)"])
        grid_row.addWidget(self.montage_order)
        grid_row.addStretch()
        layout.addLayout(grid_row)

        # —— Buttons (50/50) ———
        btn_row = QHBoxLayout()
        preview_montage_btn = QPushButton("Preview montage")
        preview_montage_btn.clicked.connect(self._preview_montage)
        save_montage_btn = QPushButton("Save montage")
        save_montage_btn.clicked.connect(self._save_montage)
        btn_row.addWidget(preview_montage_btn, 1)
        btn_row.addWidget(save_montage_btn, 1)
        layout.addLayout(btn_row)
        group.setLayout(layout)
        return group

    def _edit_montage_channel_colors(self) -> None:
        """Legacy: Open dialog for ND2 channel colors. Prefer _edit_montage_tile_colors when in montage preview."""
        paths = self._get_montage_image_paths()
        nd2_path = None
        channel_names: List[str] = []
        for p in paths:
            if p.suffix.lower() == ".nd2":
                meta = read_metadata_from_path(p)
                nc = meta.get("channel_count") or 1
                names = meta.get("channel_names") or [f"Ch{i+1}" for i in range(nc)]
                if nc >= 1:
                    nd2_path = p
                    channel_names = names[:nc]
                    break
        if not nd2_path or not channel_names:
            QMessageBox.information(
                self,
                "Channel colors",
                "No multi-channel ND2 in the current montage paths. Add an ND2 file and try again, or use \"Edit channel colors\" in the montage preview to set color per tile.",
            )
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Channel display colors")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"File: {nd2_path.name}\nSet color per channel (Auto = from metadata):"))
        form = QFormLayout()
        combos: List[tuple] = []
        choices = ["Auto", "Grayscale", "Red", "Orange", "Green", "Blue", "Yellow", "Cyan", "Magenta"]
        for i, name in enumerate(channel_names):
            cb = QComboBox()
            cb.addItems(choices)
            key = (nd2_path.stem, i)
            current = self.montage_channel_color_overrides.get(key, "auto")
            idx = next((j for j, c in enumerate(choices) if c.lower() == current), 0)
            cb.setCurrentIndex(idx)
            form.addRow(f"Ch{i+1} ({name}):", cb)
            combos.append((key, cb))
        layout.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            for key, cb in combos:
                val = cb.currentText().lower()
                if val == "auto":
                    self.montage_channel_color_overrides.pop(key, None)
                else:
                    self.montage_channel_color_overrides[key] = val

    def _edit_montage_tile_colors(
        self, layout: dict, refresh_callback: Optional[Callable[[], None]] = None
    ) -> None:
        """Open dialog to set display color per tile (by tile index). Works for any montage source (ND2, TIFFs, folder)."""
        tiles = layout.get("tiles", [])
        labels = layout.get("tile_labels", [])
        preset_names = ["Default", "Grayscale", "Red", "Orange", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Pick…"]
        preset_to_rgb = {
            "Default": None,
            "Grayscale": (1.0, 1.0, 1.0),
            "Red": (204 / 255.0, 0.0, 0.0),
            "Orange": (242 / 255.0, 137 / 255.0, 26 / 255.0), 
            "Green": (86 / 255.0, 207 / 255.0, 33 / 255.0),
            "Blue": (0 / 255.0, 0 / 255.0, 225 / 255.0),
            "Yellow": (231 / 255.0, 213 / 255.0, 32 / 255.0),
            "Cyan": (0.0, 1.0, 1.0),
            "Magenta": (1.0, 0.0, 1.0),
        }

        rows: List[tuple] = []  # (tile_idx, label, color_or_none, combo, current_rgb)
        for idx in range(len(tiles)):
            if idx >= len(tiles) or tiles[idx][2] is None:
                continue
            default_color = tiles[idx][2]
            override = self.montage_tile_color_overrides.get(idx)
            current_rgb = override if override is not None else default_color
            label = labels[idx] if idx < len(labels) else f"Tile {idx}"
            combo = QComboBox()
            combo.addItems(preset_names)
            if override is None:
                combo.setCurrentIndex(0)
            else:
                key = next((k for k, v in preset_to_rgb.items() if v == override), None)
                if key and key != "Pick…":
                    combo.setCurrentIndex(preset_names.index(key))
                else:
                    combo.addItem("Custom")
                    combo.setCurrentText("Custom")
            rows.append((idx, label, default_color, combo, current_rgb))

        if not rows:
            QMessageBox.information(
                self,
                "Tile colors",
                "No tiles with a single display color (e.g. merged tile has no per-tile color).",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Tile display colors")
        vlayout = QVBoxLayout(dlg)
        vlayout.addWidget(QLabel("Set display color per tile. Default = use current/auto color."))
        form = QFormLayout()
        for idx, label, _default, combo, _current in rows:
            form.addRow(f"Tile {idx} — {label[:50]}{'…' if len(label) > 50 else ''}:", combo)
        vlayout.addLayout(form)

        def on_combo_change(row_idx: int) -> None:
            r = rows[row_idx]
            idx, _label, default_color, combo, current_rgb = r
            text = combo.currentText()
            if text == "Default":
                new_rgb = default_color
            elif text == "Pick…":
                qc = QColorDialog.getColor(
                    QColor(int(current_rgb[0] * 255), int(current_rgb[1] * 255), int(current_rgb[2] * 255)),
                    self,
                    "Pick color for tile",
                )
                if qc.isValid():
                    new_rgb = (qc.red() / 255.0, qc.green() / 255.0, qc.blue() / 255.0)
                else:
                    new_rgb = current_rgb
                if combo.findText("Custom") < 0:
                    combo.addItem("Custom")
                combo.setCurrentText("Custom")
            elif text == "Custom":
                new_rgb = current_rgb
            else:
                new_rgb = preset_to_rgb.get(text) or default_color
            rows[row_idx] = (idx, r[1], default_color, combo, new_rgb)

        for i, row in enumerate(rows):
            row[3].currentIndexChanged.connect(lambda *_, ri=i: on_combo_change(ri))

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vlayout.addWidget(bb)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.montage_tile_color_overrides.clear()
            for idx, _label, default_rgb, _combo, chosen_rgb in rows:
                if chosen_rgb != default_rgb:
                    self.montage_tile_color_overrides[idx] = chosen_rgb
            if refresh_callback:
                refresh_callback()

    def _try_update_scalebar_pixel_size_from_path(self) -> None:
        """When scale bar image path is set to an existing file, try to read pixel size from metadata and update the spinbox."""
        path = self.scalebar_path.text().strip()
        if not path:
            return
        p = Path(path)
        if not p.is_file():
            return
        try:
            ps, _ = read_calibration_from_path(p)
            if ps > 0 and getattr(self, "scalebar_pixel_size", None) is not None:
                self.scalebar_pixel_size.setValue(ps)
        except Exception:
            pass

    def _select_scalebar_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", get_default_browse_dir(),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All (*)"
        )
        if path:
            self.scalebar_path.setText(path)

    def _select_montage_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder with images", get_default_browse_dir())
        if path:
            self.montage_path.setText(path)

    def _select_montage_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select images", get_default_browse_dir(),
            "Images (*.tif *.tiff *.nd2 *.png *.jpg *.jpeg);;ND2 (*.nd2);;TIFF (*.tif *.tiff);;All (*)"
        )
        if paths:
            self.montage_path.setText("|".join(paths))

    def _get_scalebar_image_array(self) -> Optional[np.ndarray]:
        """Load image for scale bar; return (H, W, 3) uint8 or None. Ensures (H, W) or (H, W, C) layout."""
        path = self.scalebar_path.text().strip()
        if not path or not Path(path).exists():
            return None
        try:
            img = safe_imread(path)
        except Exception:
            return None
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            # (H, W, C) has last dim in 1,2,3,4; (T, H, W) or (C, H, W) has first dim small
            if img.shape[-1] in (1, 2, 3, 4):
                pass  # (H, W, C)
            elif img.shape[0] in (1, 2, 3, 4) or img.shape[0] < img.shape[-1]:
                img = img[0]  # (T/C, H, W) -> (H, W)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 2:
            img = np.concatenate([img, img[..., :1]], axis=-1)
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3].copy()
        if img.dtype != np.uint8:
            if img.max() > img.min():
                img = ((img.astype(np.float64) - img.min()) / (img.max() - img.min()) * 255).clip(0, 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        return np.ascontiguousarray(img)

    def _get_scalebar_image_raw(self) -> Optional[tuple]:
        """Load image for scale bar; return (img_float, data_min, data_max) or None. img_float is (H, W, 3) float64."""
        path = self.scalebar_path.text().strip()
        if not path or not Path(path).exists():
            return None
        try:
            img = safe_imread(path)
        except Exception:
            return None
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[-1] in (1, 2, 3, 4):
                pass
            elif img.shape[0] in (1, 2, 3, 4) or img.shape[0] < img.shape[-1]:
                img = img[0]
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 2:
            img = np.concatenate([img, img[..., :1]], axis=-1)
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3].copy()
        img = img.astype(np.float64)
        data_min = float(np.nanmin(img))
        data_max = float(np.nanmax(img))
        if not np.isfinite(data_min):
            data_min = 0.0
        if not np.isfinite(data_max) or data_max <= data_min:
            data_max = data_min + 1.0
        return np.ascontiguousarray(img), data_min, data_max

    def _draw_scalebar_on_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Draw scale bar on a copy of img (H, W, 3) uint8. Position auto-adjusts so bar + label stay inside image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            QMessageBox.warning(self, "Scale bar", "Pillow (PIL) is required for drawing the scale bar.")
            return None
        h, w = img.shape[:2]
        pixel_size = self.scalebar_pixel_size.value()
        length_um = self.scalebar_length_um.value()
        length_px = max(1, int(round(length_um / pixel_size)))
        thickness = self.scalebar_thickness.value()
        margin = max(12, thickness + 6)
        gap = 4
        pos_name = self.scalebar_position.currentText().lower()
        color_name = self.scalebar_color.currentText().lower()
        color = (255, 255, 255) if color_name == "white" else (0, 0, 0)
        font = None
        font_size = self.scalebar_font_size.value()
        for path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "arial.ttf",
        ]:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        show_label = self.scalebar_show_label.isChecked()
        label = ""
        label_w, label_h = 0, 0
        if show_label:
            label = f"{length_um:.0f}" if length_um >= 1 else f"{length_um}"
            label += " µm"
            try:
                bbox = font.getbbox(label)
                label_w = bbox[2] - bbox[0]
                label_h = bbox[3] - bbox[1]
            except AttributeError:
                label_w = len(label) * (font_size // 2)
                label_h = font_size

        # Vertical position so bar + label fit (label above bar for bottom, below bar for top)
        if "bottom" in pos_name:
            y0_max = h - margin - thickness
            y0 = (y0_max - (label_h + gap)) if show_label else y0_max
            # Keep label inside image: label top at ly = y0 - gap - label_h must be >= margin
            if show_label:
                y0 = max(y0, margin + gap + label_h)
            y0 = max(margin, y0)
        else:
            y0 = margin
            if show_label and (y0 + thickness + gap + label_h > h - margin):
                y0 = max(margin, h - margin - thickness - gap - label_h)
        y1 = y0 + thickness

        # Horizontal: prefer bar at corner, clamp so label stays in [margin, w-margin]
        half_bar = length_px // 2
        half_label = (label_w + 1) // 2
        lx_min = margin + half_label
        lx_max = (w - margin - half_label) if w > 2 * margin + label_w else lx_min
        if "right" in pos_name:
            lx = min(w - margin - half_bar, lx_max)
            lx = max(lx, lx_min)
        else:
            lx = max(margin + half_bar, lx_min)
            lx = min(lx, lx_max)
        x0 = int(lx - half_bar)
        x0 = max(margin, min(x0, w - margin - length_px))
        x1 = x0 + length_px
        lx = x0 + half_bar

        pil_img = Image.fromarray(img)
        draw = ImageDraw.ImageDraw(pil_img)
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=color)
        if show_label:
            if "top" in pos_name:
                ly = y1 + gap  # label top just below bar
            else:
                ly = y0 - gap - label_h  # label bottom just above bar (anchor "mt" = top of text at ly)
            draw.text((lx, ly), label, fill=color, font=font, anchor="mt")
        return np.array(pil_img)

    def _build_scalebar_preview_image(self, raw: np.ndarray, vmin: float, vmax: float) -> Optional[np.ndarray]:
        """Apply LUT (vmin, vmax) to raw float image and draw scale bar. Returns uint8 (H, W, 3) or None."""
        vrange = max(float(vmax) - float(vmin), 1e-9)
        display = ((raw.astype(np.float64) - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
        return self._draw_scalebar_on_image(display)

    def _preview_scalebar(self) -> None:
        raw_result = self._get_scalebar_image_raw()
        if raw_result is None:
            QMessageBox.warning(self, "Scale bar", "Could not load image. Check path and format.")
            return
        raw, data_min, data_max = raw_result
        self._show_scalebar_preview_dialog(raw, data_min, data_max)

    def _save_scalebar(self) -> None:
        img = self._get_scalebar_image_array()
        if img is None:
            QMessageBox.warning(self, "Scale bar", "Could not load image. Check path and format.")
            return
        out = self._draw_scalebar_on_image(img)
        if out is None:
            return
        default_path = Path(self.scalebar_path.text().strip())
        default_path = default_path.parent / f"{default_path.stem}_scalebar.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save with scale bar", str(default_path),
            "PNG (*.png);;TIFF (*.tif *.tiff);;All (*)"
        )
        if not path:
            return
        try:
            from PIL import Image
            Image.fromarray(out).save(path)
            self.parent_window.log_output(f"Scale bar image saved: {path}\n")
        except Exception as e:
            QMessageBox.warning(self, "Scale bar", f"Save failed: {e}")

    _MONTAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2")

    # Channel name -> (R, G, B) in 0-1 for false color. Phase → gray; common fluorophores → semantic colors.
    _CHANNEL_COLOR_MAP = {
        "gray": (1.0, 1.0, 1.0),
        "grayscale": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "orange": (1.0, 86 / 255.0, 0.0),  # RGB (255, 86, 0) — dsRED / mScarlet / 561 nm
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
    }

    @staticmethod
    def _channel_name_to_rgb(name: str) -> tuple[float, float, float]:
        """Map channel name (from metadata) to display color (R,G,B) 0-1. Phase→gray; mScarlet/RFP→red; GFP→green; YFP→yellow; etc."""
        if not name:
            return (0.0, 1.0, 0.0)  # default green
        n = name.strip().lower()
        if "phase" in n or "bf" == n or "brightfield" in n or "trans" in n or "transmission" in n:
            return (1.0, 1.0, 1.0)  # grayscale
        if "mscarlet" in n or "dsred" in n or "561" in n or "orange" in n:
            return (1.0, 86 / 255.0, 0.0)  # orange RGB (255, 86, 0) — dsRED / mScarlet / 561 nm
        if "mcherry" in n or "rfp" in n or "red" in n or "cy3" in n or "texas" in n:
            return (1.0, 0.0, 0.0)
        if "gfp" in n or "egfp" in n or "fitc" in n or "green" in n:
            return (0.0, 1.0, 0.0)
        if "yfp" in n or "mvenus" in n or "venus" in n or "yellow" in n:
            return (1.0, 1.0, 0.0)
        if "cfp" in n or "cerulean" in n or "cyan" in n:
            return (0.0, 1.0, 1.0)
        if "cy5" in n or "magenta" in n:
            return (1.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)  # default green

    # Default colors for multiple single-channel (non-ND2) images: cycle so each file gets a different color.
    _CYCLE_COLORS = ((1.0, 0.0, 0.0), (1.0, 86 / 255.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0))  # R, orange (255,86,0), G, B, Y, C, M

    def _get_channel_colors_for_path(
        self, path: Path, n_channels: int, entry_index: Optional[int] = None
    ) -> List[tuple]:
        """Return list of (r,g,b) 0-1 per channel. ND2: from metadata + overrides. Non-ND2 single-channel: cycle by entry_index."""
        if path.suffix.lower() != ".nd2" and n_channels == 1 and entry_index is not None:
            return [self._CYCLE_COLORS[entry_index % len(self._CYCLE_COLORS)]]
        overrides = getattr(self, "montage_channel_color_overrides", None) or {}
        meta = read_metadata_from_path(path) if path.suffix.lower() == ".nd2" else {}
        names = meta.get("channel_names") or [f"Ch{i+1}" for i in range(n_channels)]
        wavelengths = meta.get("channel_wavelengths") or []
        result = []
        for i in range(n_channels):
            key = (path.stem, i)
            if key in overrides:
                choice = overrides[key].lower()
                if choice == "auto":
                    pass  # fall through to name-based
                else:
                    result.append(self._CHANNEL_COLOR_MAP.get(choice, (0.0, 1.0, 0.0)))
                    continue
            # Auto: use phase flag or name
            if i < len(wavelengths) and wavelengths[i].get("phase_contrast"):
                result.append((1.0, 1.0, 1.0))
            else:
                result.append(self._channel_name_to_rgb(names[i] if i < len(names) else ""))
        return result

    # TIFF filename suffixes → channel name for grouping (stem must end with one of these).
    _CHANNEL_SUFFIXES = [
        ("_phase", "phase"), ("_Phase", "phase"),
        ("_phase60x", "phase"), ("_Phase60X", "phase"), ("_phase40x", "phase"), ("_phase100x", "phase"),  # e.g. EM13748_AnTC_004_Phase60X
        ("_fi", "FI"), ("_FI", "FI"),  # generic fluorescence (e.g. untitled_fi.tif + untitled_phase.tif → one merged tile)
        ("_gfp", "GFP"), ("_GFP", "GFP"), ("_egfp", "GFP"),
        ("_mscarlet", "mScarlet"), ("_mScarlet", "mScarlet"), ("_dsred", "dsRED"), ("_dsRed", "dsRED"),
        ("_mcherry", "mCherry"), ("_rfp", "RFP"),
        ("_561", "561nm"), ("_orange", "Orange"),
        ("_yfp", "YFP"), ("_YFP", "YFP"), ("_venus", "Venus"),
        ("_cfp", "CFP"), ("_cyan", "Cyan"),
        ("_red", "red"), ("_green", "green"), ("_blue", "blue"),
    ]

    @classmethod
    def _infer_channel_from_stem(cls, stem: str) -> Optional[tuple[str, str]]:
        """If stem looks like 'base_channel', return (base, channel_name). E.g. exp_phase -> (exp, phase); exp_fi_GFP -> (exp, GFP)."""
        if "_fi_" in stem:
            base, _, ch = stem.partition("_fi_")
            return (base.strip("_"), ch.strip("_") or "FI")
        stem_lower = stem.lower()
        for suffix, ch_name in cls._CHANNEL_SUFFIXES:
            if stem_lower.endswith(suffix.lower()):
                base = stem[: -len(suffix)].strip("_")
                return (base, ch_name)
        return None

    @staticmethod
    def _channel_sort_key(channel_name: str) -> tuple:
        """Sort key: phase first (0), then red(1), orange(2), green(3), blue(4), yellow(5), cyan(6), magenta(7), else (8, name)."""
        n = channel_name.strip().lower()
        if "phase" in n or n in ("bf", "brightfield", "trans"):
            return (0, n)
        if "mcherry" in n or "rfp" in n or "red" in n or "cy3" in n:
            return (1, n)
        if "mscarlet" in n or "dsred" in n or "561" in n or "orange" in n:
            return (2, n)
        if "green" in n or "gfp" in n or "fitc" in n:
            return (3, n)
        if "blue" in n:
            return (4, n)
        if "yellow" in n or "yfp" in n or "venus" in n:
            return (5, n)
        if "cyan" in n or "cfp" in n:
            return (6, n)
        if "magenta" in n or "cy5" in n:
            return (7, n)
        return (8, n)

    def _get_montage_datasets(self) -> List[dict]:
        """
        Return list of datasets. Each dataset yields tiles in order: Phase, FI channels (with colors), Merged.
        - Single ND2 or single multi-channel TIFF: one dataset (channels from metadata, frame 0).
        - Folder of ND2s: one dataset per ND2.
        - Folder of TIFFs: group by base name (e.g. exp_phase.tif + exp_GFP.tif → one dataset); channels from filenames.
        """
        paths = self._get_montage_image_paths()
        if not paths:
            return []
        nd2_paths = [p for p in paths if p.suffix.lower() == ".nd2"]
        tiff_paths = [p for p in paths if p.suffix.lower() in (".tif", ".tiff")]

        # Single file: one dataset
        if len(paths) == 1:
            p = paths[0]
            if p.suffix.lower() == ".nd2":
                return [{"type": "nd2", "path": p}]
            # Single TIFF: treat as one channel unless multi-channel metadata
            meta = read_metadata_from_path(p)
            nc = meta.get("channel_count") or 1
            if nc > 1:
                return [{"type": "nd2_like", "path": p, "channel_count": nc}]  # same load path as ND2
            return [{"type": "single_tiff", "path": p}]

        # Multiple files: ND2s each one dataset; TIFFs grouped by base
        datasets: List[dict] = []
        for p in nd2_paths:
            datasets.append({"type": "nd2", "path": p})

        if tiff_paths:
            # Two or more TIFFs: one group → tile per file + one Merged tile (colors from filename when detected, else user can set in montage)
            if len(tiff_paths) >= 2:
                channel_list: List[tuple[Path, str]] = []
                for p in tiff_paths:
                    inferred = self._infer_channel_from_stem(p.stem)
                    ch_name = inferred[1] if inferred else p.stem
                    channel_list.append((p, ch_name))
                channel_list.sort(key=lambda x: self._channel_sort_key(x[1]))
                base = channel_list[0][0].stem if channel_list else "montage"
                datasets.append({"type": "tiff_group", "base": base, "channels": channel_list})
            else:
                # Single TIFF: one tile only (no merge)
                datasets.append({"type": "single_tiff", "path": tiff_paths[0]})

        return datasets

    def _get_montage_image_paths(self) -> List[Path]:
        """Resolve montage input to a sorted list of image paths. Folder: filter by dropdown (All / ND2 only / TIFF only)."""
        raw = self.montage_path.text().strip()
        if not raw:
            return []
        if "|" in raw:
            return [Path(p.strip()) for p in raw.split("|") if Path(p.strip()).exists()]
        p = Path(raw)
        if not p.exists():
            return []
        if p.is_file():
            return [p] if p.suffix.lower() in self._MONTAGE_EXTS else []
        # Folder: include all ND2 and TIFF (deduplicate by resolved path so each file appears once)
        paths = []
        seen = set()
        for ext in self._MONTAGE_EXTS:
            for x in p.glob(f"*{ext}"):
                if x.is_file():
                    r = x.resolve()
                    if r not in seen:
                        seen.add(r)
                        paths.append(x)
        return sorted(paths, key=lambda x: x.name)

    def _build_montage_array(
        self,
        per_tile_lut: Optional[Dict[int, Tuple[float, float]]] = None,
        per_tile_color_override: Optional[Dict[int, Tuple[float, float, float]]] = None,
        return_layout: bool = False,
    ) -> Optional[np.ndarray]:
        """Build montage: Phase first, then FI channels (with colors), then merged. One frame only (first timepoint).
        If per_tile_lut is set, use (vmin, vmax) per tile index; else use per-tile percentile.
        If per_tile_color_override is set, use (r,g,b) per tile index for colored tiles.
        If return_layout is True, return (array, layout_dict) with layout_dict containing rows, cols, tiles, etc. for preview."""
        datasets = self._get_montage_datasets()
        if not datasets:
            return None
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            QMessageBox.warning(self, "Montage", "Pillow (PIL) is required for montage.")
            return None
        use_color_merge = True  # Always: phase + FI channels + merged
        use_global_lut = False  # Per-tile LUT (or per_tile_lut from preview)
        order_col_major = "Column" in self.montage_order.currentText()
        rows = self.montage_rows.value()
        cols = self.montage_cols.value()
        spacing = self.montage_spacing.value()
        border = self.montage_border.value()
        show_labels = False
        font_size = 12

        def _normalize_plane(plane: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
            """Robust 0-1 normalization using percentiles so weak channels don't go black."""
            plane = np.asarray(plane, dtype=np.float64)
            valid = plane[np.isfinite(plane)]
            if valid.size == 0:
                return np.zeros_like(plane)
            vmin = float(np.percentile(valid, p_low))
            vmax = float(np.percentile(valid, p_high))
            if vmax <= vmin:
                vmax = vmin + 1.0
            out = (plane - vmin) / (vmax - vmin)
            return np.clip(out, 0.0, 1.0)

        def _normalize_plane_with_lut(plane: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
            """Normalize plane to 0-1 using (vmin, vmax) if provided, else percentile."""
            plane = np.asarray(plane, dtype=np.float64)
            valid = plane[np.isfinite(plane)]
            if valid.size == 0:
                return np.zeros_like(plane)
            if vmin is not None and vmax is not None and vmax > vmin:
                out = (plane - vmin) / (vmax - vmin)
            else:
                vmin = float(np.percentile(valid, 2.0))
                vmax = float(np.percentile(valid, 98.0))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                out = (plane - vmin) / (vmax - vmin)
            return np.clip(out, 0.0, 1.0)

        def _compose_merged(arr_hwc: np.ndarray, colors: List[tuple]) -> np.ndarray:
            """Merge: phase contrast (gray) as visible base + FI channels in their colors. Percentile LUT so weak channels aren't black."""
            return _compose_merged_with_lut(arr_hwc, colors, {})

        def _compose_merged_with_lut(arr_hwc: np.ndarray, colors: List[tuple], per_ch_lut: dict) -> np.ndarray:
            """Like _compose_merged but use per_ch_lut[channel_index] = (vmin, vmax) when present."""
            h, w = arr_hwc.shape[0], arr_hwc.shape[1]
            C = arr_hwc.shape[-1]
            phase = _normalize_plane_with_lut(arr_hwc[..., 0], *(per_ch_lut.get(0) or (None, None))) if C >= 1 else np.zeros((h, w), dtype=np.float64)
            fi_rgb = np.zeros((h, w, 3), dtype=np.float64)
            for c in range(1, min(C, len(colors))):
                vmin, vmax = per_ch_lut.get(c) or (None, None)
                plane = _normalize_plane_with_lut(arr_hwc[..., c], vmin, vmax)
                r, g, b = colors[c]
                fi_rgb[..., 0] += plane * r
                fi_rgb[..., 1] += plane * g
                fi_rgb[..., 2] += plane * b
            if fi_rgb.max() > 1e-9:
                fi_rgb = fi_rgb / fi_rgb.max()
            phase_weight = 0.45
            out = np.zeros((h, w, 3), dtype=np.float64)
            out[..., 0] = phase_weight * phase + (1 - phase_weight) * fi_rgb[..., 0]
            out[..., 1] = phase_weight * phase + (1 - phase_weight) * fi_rgb[..., 1]
            out[..., 2] = phase_weight * phase + (1 - phase_weight) * fi_rgb[..., 2]
            out = out / max(out.max(), 1e-9)
            return (out * 255).clip(0, 255).astype(np.uint8)

        def load_nd2_or_multichannel_tiff(path: Path, frame_index: int = 0) -> Optional[tuple[np.ndarray, List[str], List[tuple], int]]:
            """Load first frame from ND2 or multi-channel TIFF. Return (arr (H,W,C), channel_names, colors, n_timepoints) with channels sorted phase first. n_timepoints is 1 for TIFF; for ND2 it is sizes['T']."""
            try:
                suf = path.suffix.lower()
                meta = read_metadata_from_path(path)
                nC = meta.get("channel_count") or 1
                names = meta.get("channel_names") or [f"Ch{i+1}" for i in range(nC)]
                nT, nZ = 1, 1
                if suf == ".nd2":
                    import nd2
                    with nd2.ND2File(path) as r:
                        arr = np.ascontiguousarray(r.asarray())
                        sizes = getattr(r, "sizes", None) or {}
                    nT = int(sizes.get("T", 1))
                    nZ = int(sizes.get("Z", 1))
                    nC_file = int(sizes.get("C", 1))
                    # ND2 uses T C Z Y X (or T C Y X when Z=1)
                    if arr.ndim == 5:
                        arr = arr[frame_index] if nT > 1 else arr[0]  # (C, Z, Y, X)
                        arr = arr[:, 0, :, :]  # (C, Y, X)
                    elif arr.ndim == 4:
                        # (T, C, Y, X) or (T, Y, X) when C=1
                        arr = arr[frame_index] if nT > 1 else arr[0]
                        if arr.ndim == 2:
                            arr = arr[np.newaxis, ...]  # (1, Y, X)
                    elif arr.ndim == 3 and nT > 1:
                        arr = arr[frame_index]  # (Y, X)
                        arr = arr[..., np.newaxis]  # (Y, X, 1)
                    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))  # (Y, X, C)
                    elif arr.ndim == 3 and arr.shape[-1] not in (1, 2, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
                else:
                    import tifffile
                    arr = tifffile.imread(str(path), key=frame_index)
                    arr = np.asarray(arr)
                    if arr.ndim == 2:
                        arr = arr[:, :, np.newaxis]
                    elif arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 2:
                    arr = np.concatenate([arr, np.zeros((*arr.shape[:2], 1), dtype=arr.dtype)], axis=-1)
                nCh = arr.shape[-1]
                # Use file channel count when we have multi-channel ND2 so names/colors match
                if suf == ".nd2" and nC_file > 1 and len(names) < nCh:
                    names = (names + [f"Ch{i+1}" for i in range(len(names), nCh)])[:nCh]
                order = sorted(range(nCh), key=lambda i: self._channel_sort_key(names[i] if i < len(names) else f"Ch{i+1}"))
                arr = np.asarray(arr[..., order])
                names = [names[i] if i < len(names) else f"Ch{i+1}" for i in order]
                colors = self._get_channel_colors_for_path(path, nCh)
                colors = [colors[i] for i in order]
                return np.asarray(arr), names, colors, nT
            except Exception:
                return None

        def load_single_plane(path: Path, frame_index: int = 0) -> Optional[np.ndarray]:
            """Load one 2D image (first frame/page)."""
            try:
                if path.suffix.lower() in (".tif", ".tiff"):
                    import tifffile
                    try:
                        arr = tifffile.imread(str(path))
                    except Exception:
                        arr = tifffile.imread(str(path), key=frame_index)
                    arr = np.asarray(arr)
                    # Single 2D: keep. (Y,X,C) C<=4: take first channel. (T,Y,X) or (T,C,Y,X): take first frame
                    if arr.ndim == 3:
                        if arr.shape[-1] in (1, 2, 3, 4):
                            arr = arr[..., 0]
                        else:
                            arr = arr[frame_index]
                    elif arr.ndim == 4:
                        arr = arr[frame_index]
                        if arr.shape[-1] in (1, 2, 3, 4):
                            arr = arr[..., 0]
                        elif arr.shape[0] in (1, 2, 3, 4):
                            arr = np.transpose(arr, (1, 2, 0))[..., 0]
                else:
                    arr = safe_imread(str(path))
                    if arr.ndim >= 3 and arr.shape[0] <= 4:
                        arr = arr[0]
                    arr = np.asarray(arr)
                if arr.ndim > 2:
                    arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
                return np.asarray(arr)
            except Exception:
                return None

        tiles: List[tuple] = []  # (arr or None, label, color_or_None)
        all_arrays_for_global: List[np.ndarray] = []

        for ds in datasets:
            ds_type = ds["type"]
            base_label = ds.get("base", Path(ds["path"]).stem if "path" in ds else "?")

            if ds_type in ("nd2", "nd2_like"):
                path = ds["path"]
                result = load_nd2_or_multichannel_tiff(path, 0)
                if result is None:
                    tiles.append((None, base_label, None))
                    continue
                arr_hwc, ch_names, colors, n_timepoints = result
                nCh = arr_hwc.shape[-1] if arr_hwc.ndim == 3 else 1
                # Single channel but multiple timepoints → one tile per frame (real montage)
                if nCh == 1 and n_timepoints > 1 and path.suffix.lower() == ".nd2":
                    try:
                        import nd2
                        with nd2.ND2File(path) as r:
                            arr_full = np.ascontiguousarray(r.asarray())
                        # arr_full is (T, Y, X) or (T, C, Y, X)
                        if arr_full.ndim == 3:
                            frames = arr_full  # (T, Y, X)
                        else:
                            frames = arr_full[:, 0, :, :]  # (T, Y, X)
                        n_show = min(frames.shape[0], 30)  # cap at 30 frames
                        gray = (1.0, 1.0, 1.0)
                        for t in range(n_show):
                            plane = np.asarray(frames[t])
                            tiles.append((plane, f"{base_label} t{t}", gray))
                            if use_global_lut:
                                all_arrays_for_global.append(np.asarray(plane, dtype=np.float64))
                    except Exception:
                        single = arr_hwc[..., 0] if arr_hwc.ndim == 3 else arr_hwc
                        col = self._get_channel_colors_for_path(path, 1, entry_index=0)[0]
                        tiles.append((single, base_label, col))
                        if use_global_lut:
                            all_arrays_for_global.append(np.asarray(single, dtype=np.float64))
                    continue
                if not use_color_merge or nCh < 2:
                    single = arr_hwc[..., 0] if arr_hwc.ndim == 3 else arr_hwc
                    col = self._get_channel_colors_for_path(path, 1, entry_index=0)[0]
                    tiles.append((single, base_label, col))
                    if use_global_lut:
                        all_arrays_for_global.append(np.asarray(single, dtype=np.float64))
                    continue
                for c in range(arr_hwc.shape[-1]):
                    plane = np.asarray(arr_hwc[..., c])
                    label = ch_names[c] if c < len(ch_names) else f"Ch{c+1}"
                    tiles.append((plane, f"{base_label} {label}", colors[c]))
                    if use_global_lut:
                        all_arrays_for_global.append(np.asarray(plane, dtype=np.float64))
                merged = _compose_merged(arr_hwc, colors)
                tiles.append((merged, f"{base_label} Merged", None))
                continue

            if ds_type == "tiff_group":
                base = ds["base"]
                channel_list: List[tuple[Path, str]] = ds["channels"]
                planes: List[np.ndarray] = []
                color_list: List[tuple] = []
                for path, ch_name in channel_list:
                    plane = load_single_plane(path, 0)
                    if plane is not None:
                        planes.append(plane)
                        color_list.append(self._channel_name_to_rgb(ch_name))
                        if use_global_lut:
                            all_arrays_for_global.append(np.asarray(plane, dtype=np.float64))
                for i, (_, ch_name) in enumerate(channel_list):
                    if i < len(planes):
                        tiles.append((planes[i], f"{base} {ch_name}", color_list[i]))
                if len(planes) > 1:
                    h = max(p.shape[0] for p in planes)
                    w = max(p.shape[1] for p in planes)
                    resized = []
                    for p in planes:
                        if p.shape[0] == h and p.shape[1] == w:
                            resized.append(np.asarray(p, dtype=np.float64))
                        else:
                            # Resize to common size
                            p_u8 = ((p.astype(np.float64) - np.nanmin(p)) / max(np.nanmax(p) - np.nanmin(p), 1e-9) * 255).clip(0, 255).astype(np.uint8)
                            img = Image.fromarray(p_u8).resize((w, h), Image.Resampling.LANCZOS)
                            resized.append(np.array(img).astype(np.float64))
                    arr_hwc = np.stack(resized, axis=-1)
                    merged = _compose_merged(arr_hwc, color_list)
                    tiles.append((merged, f"{base} Merged", None))
                continue

            if ds_type == "single_tiff":
                path = ds["path"]
                arr = load_single_plane(path, 0)
                if arr is not None:
                    tiles.append((arr, base_label, (1.0, 1.0, 1.0)))
                    if use_global_lut:
                        all_arrays_for_global.append(np.asarray(arr, dtype=np.float64))
                else:
                    tiles.append((None, base_label, None))
                continue

        # Global LUT (only when no per_tile_lut): use percentiles across all tiles
        gmin, gmax = None, None
        if use_global_lut and all_arrays_for_global and not per_tile_lut:
            all_pixels = np.concatenate([a.ravel() for a in all_arrays_for_global])
            all_pixels = all_pixels[np.isfinite(all_pixels)]
            if all_pixels.size > 0:
                gmin = float(np.percentile(all_pixels, 2.0))
                gmax = float(np.percentile(all_pixels, 98.0))
            if gmin is None:
                gmin, gmax = 0.0, 255.0
            if gmax <= gmin:
                gmax = gmin + 1.0

        n_tiles = len(tiles)
        if n_tiles == 0:
            return None
        if rows <= 0 and cols <= 0:
            cols = int(np.ceil(np.sqrt(n_tiles)))
            rows = max(1, (n_tiles + cols - 1) // cols)
        elif rows <= 0:
            rows = max(1, (n_tiles + cols - 1) // cols)
        elif cols <= 0:
            cols = max(1, (n_tiles + rows - 1) // rows)

        pil_images = []
        max_w, max_h = 0, 0
        placeholder_size = 100
        for idx, (arr, label, color_or_none) in enumerate(tiles):
            if arr is None:
                placeholder = np.full((placeholder_size, placeholder_size, 3), 128, dtype=np.uint8)
                pil_images.append((Image.fromarray(placeholder), label + "?"))
                max_w = max(max_w, placeholder_size)
                max_h = max(max_h, placeholder_size)
                continue
            if per_tile_color_override and idx in per_tile_color_override and color_or_none is not None:
                color_or_none = per_tile_color_override[idx]
            # Rebuild merged tile from preceding channel tiles when user adjusted color and/or LUT on any of them
            has_lut = per_tile_lut and any(i in per_tile_lut for i in range(idx))
            has_color_override = per_tile_color_override and any(i in per_tile_color_override for i in range(idx))
            if label.endswith(" Merged") and color_or_none is None and idx > 0 and (has_lut or has_color_override):
                try:
                    ch_arrays = [np.asarray(tiles[i][0]).squeeze() for i in range(idx)]
                    if all(a.ndim == 2 for a in ch_arrays) and len(ch_arrays) >= 1:
                        h_max = max(a.shape[0] for a in ch_arrays)
                        w_max = max(a.shape[1] for a in ch_arrays)
                        resized = []
                        for a in ch_arrays:
                            if a.shape[0] == h_max and a.shape[1] == w_max:
                                resized.append(np.asarray(a, dtype=np.float64))
                            else:
                                a_u8 = ((a.astype(np.float64) - np.nanmin(a)) / max(np.nanmax(a) - np.nanmin(a), 1e-9) * 255).clip(0, 255).astype(np.uint8)
                                resized.append(np.array(Image.fromarray(a_u8).resize((w_max, h_max), Image.Resampling.LANCZOS)).astype(np.float64))
                        arr_hwc = np.stack(resized, axis=-1)
                        colors_ch = [
                            per_tile_color_override[i] if (per_tile_color_override and i in per_tile_color_override) else tiles[i][2]
                            for i in range(idx)
                        ]
                        per_ch_lut = {i: per_tile_lut[i] for i in range(idx) if i in per_tile_lut} if per_tile_lut else {}
                        arr = _compose_merged_with_lut(arr_hwc, colors_ch, per_ch_lut)
                except Exception:
                    pass
            # Resolve (vmin, vmax) for this tile: per_tile_lut override, else global, else percentile
            vmin_override, vmax_override = None, None
            if per_tile_lut and idx in per_tile_lut:
                vmin_override, vmax_override = per_tile_lut[idx]
                if vmax_override <= vmin_override:
                    vmax_override = vmin_override + 1e-9
            # Single-channel with semantic color (r,g,b)
            if color_or_none is not None and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)):
                ch = arr[..., 0] if arr.ndim == 3 else arr
                ch = ch.astype(np.float64)
                if vmin_override is not None and vmax_override is not None:
                    ch = (ch - vmin_override) / max(vmax_override - vmin_override, 1e-9)
                elif use_global_lut and gmin is not None and gmax is not None:
                    ch = (ch - gmin) / max(gmax - gmin, 1e-9)
                else:
                    valid = ch[np.isfinite(ch)]
                    if valid.size > 0:
                        vmin = float(np.percentile(valid, 2.0))
                        vmax = float(np.percentile(valid, 98.0))
                        if vmax <= vmin:
                            vmax = vmin + 1.0
                        ch = (ch - vmin) / (vmax - vmin)
                    ch = np.clip(ch, 0.0, 1.0)
                r, g, b = color_or_none
                arr = np.stack([ch * r, ch * g, ch * b], axis=-1)
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            # Pre-composed merged RGB (from _compose_merged)
            elif color_or_none is None and arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype == np.uint8:
                if vmin_override is not None and vmax_override is not None:
                    arr = ((arr.astype(np.float64) - vmin_override) / max(vmax_override - vmin_override, 1e-9) * 255).clip(0, 255).astype(np.uint8)
            # Multi-channel RGB from load_one (not expanded)
            elif use_color_merge and arr.ndim == 3 and arr.shape[-1] >= 2:
                if arr.shape[-1] == 2:
                    arr = np.concatenate([arr, np.zeros((*arr.shape[:2], 1), dtype=arr.dtype)], axis=-1)
                if arr.shape[-1] >= 3:
                    arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    if vmin_override is not None and vmax_override is not None:
                        arr = ((arr.astype(np.float64) - vmin_override) / max(vmax_override - vmin_override, 1e-9) * 255).clip(0, 255).astype(np.uint8)
                    else:
                        per_ch = [(arr[..., c].astype(np.float64) - arr[..., c].min()) / max(arr[..., c].max() - arr[..., c].min(), 1e-9) for c in range(3)]
                        arr = (np.stack(per_ch, axis=-1) * 255).clip(0, 255).astype(np.uint8)
            else:
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    arr = arr.mean(axis=-1, keepdims=True)
                    arr = np.concatenate([arr, arr, arr], axis=-1)
                if arr.dtype != np.uint8:
                    if vmin_override is not None and vmax_override is not None:
                        arr = ((arr.astype(np.float64) - vmin_override) / max(vmax_override - vmin_override, 1e-9) * 255).clip(0, 255).astype(np.uint8)
                    elif use_global_lut and gmin is not None and gmax is not None:
                        arr = ((arr.astype(np.float64) - gmin) / max(gmax - gmin, 1e-9) * 255).clip(0, 255).astype(np.uint8)
                    else:
                        arr = ((arr.astype(np.float64) - arr.min()) / max(arr.max() - arr.min(), 1e-9) * 255).clip(0, 255).astype(np.uint8)
            pil_images.append((Image.fromarray(arr), label))
            max_w = max(max_w, arr.shape[1])
            max_h = max(max_h, arr.shape[0])
        if not pil_images:
            return None
        cell_w = max_w
        cell_h = max_h + (font_size + 4) if show_labels else max_h
        out_w = border * 2 + cols * cell_w + (cols - 1) * spacing
        out_h = border * 2 + rows * cell_h + (rows - 1) * spacing
        font = None
        for path in ["/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"]:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        out_pil = Image.new("RGB", (out_w, out_h), (255, 255, 255))
        draw = ImageDraw.ImageDraw(out_pil)
        for idx, (pil_img, stem) in enumerate(pil_images):
            r, c = idx // cols, idx % cols
            x0 = border + c * (cell_w + spacing)
            y0 = border + r * (cell_h + spacing)
            # Resize to fit cell (max_w x max_h)
            pil_resized = pil_img.resize((max_w, max_h), Image.Resampling.LANCZOS)
            out_pil.paste(pil_resized, (x0, y0))
            if show_labels:
                ly = y0 + max_h + 2
                draw.text((x0 + max_w // 2, ly), stem[:32] + ("…" if len(stem) > 32 else ""), fill=(0, 0, 0), font=font, anchor="mt")
        out_arr = np.array(out_pil)
        if return_layout:
            layout = {
                "rows": rows,
                "cols": cols,
                "border": border,
                "spacing": spacing,
                "cell_w": cell_w,
                "cell_h": cell_h,
                "max_w": max_w,
                "max_h": max_h,
                "out_w": out_w,
                "out_h": out_h,
                "n_tiles": n_tiles,
                "tiles": tiles,
                "tile_labels": [label for _, label, _ in tiles],
            }
            return (out_arr, layout)
        return out_arr

    def _preview_montage(self) -> None:
        paths = self._get_montage_image_paths()
        if not paths:
            QMessageBox.warning(
                self, "Montage",
                "No images found. Enter one ND2 or TIFF file, or a folder."
            )
            return
        datasets = self._get_montage_datasets()
        if not datasets:
            QMessageBox.warning(
                self, "Montage",
                "No datasets to show. For a folder of TIFFs, use names like sample_phase.tif and sample_GFP.tif so channels can be grouped."
            )
            return
        result = self._build_montage_array(
            return_layout=True,
            per_tile_color_override=getattr(self, "montage_tile_color_overrides", None) or {},
        )
        if result is None:
            QMessageBox.warning(self, "Montage", "Could not build montage (check image formats and that Pillow is installed).")
            return
        out, layout = result
        self._show_montage_preview_dialog(out, layout)

    def _save_montage(self) -> None:
        paths = self._get_montage_image_paths()
        if not paths:
            QMessageBox.warning(
                self, "Montage",
                "No images found. Enter one ND2 or TIFF file, or a folder."
            )
            return
        if not self._get_montage_datasets():
            QMessageBox.warning(
                self, "Montage",
                "No datasets to include. For a folder of TIFFs, use names like sample_phase.tif and sample_GFP.tif."
            )
            return
        out = self._build_montage_array(
            per_tile_color_override=getattr(self, "montage_tile_color_overrides", None) or {},
        )
        if out is None:
            QMessageBox.warning(self, "Montage", "Could not build montage (check formats and Pillow).")
            return
        default_dir = paths[0].parent
        path, _ = QFileDialog.getSaveFileName(
            self, "Save montage", str(default_dir / "montage.png"),
            "PNG (*.png);;TIFF (*.tif *.tiff);;All (*)"
        )
        if not path:
            return
        try:
            from PIL import Image
            Image.fromarray(out).save(path)
            self.parent_window.log_output(f"Montage saved: {path}\n")
        except Exception as e:
            QMessageBox.warning(self, "Montage", f"Save failed: {e}")

    def _show_montage_preview_dialog(self, initial_montage: np.ndarray, layout: dict) -> None:
        """Non-modal montage preview: click a tile to select it, adjust LUT via histogram; Update, Edit colors, Save, Close."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        dlg = QDialog(self)
        dlg.setWindowTitle("Montage preview")
        dlg.setModal(False)
        dlg.setMinimumSize(900, 800)
        main_layout = QVBoxLayout(dlg)
        dlg._layout = layout
        dlg._per_tile_lut = {}
        dlg._current_displayed = initial_montage.copy()

        selected_label = QLabel("Click an image tile to adjust its display range (min/max). Middle-drag to pan, wheel to zoom.")
        selected_label.setWordWrap(True)
        selected_label.setStyleSheet("color: #aaa; font-size: 11px;")
        main_layout.addWidget(selected_label)

        lut_group = QGroupBox("LUT — selected tile: drag the red/green lines to set display range")
        lut_layout = QVBoxLayout()
        hist_fig = Figure(figsize=(8, 2.6))
        hist_canvas = FigureCanvas(hist_fig)
        hist_canvas.setMinimumHeight(170)
        hist_canvas.setMaximumHeight(200)
        hist_ax = hist_fig.add_subplot(111)
        hist_fig.patch.set_facecolor("#2b2b2b")
        hist_ax.set_facecolor("#2b2b2b")
        hist_ax.tick_params(colors="white", labelsize=10)
        for spine in ["bottom", "top", "left", "right"]:
            hist_ax.spines[spine].set_color("white")
        hist_ax.xaxis.label.set_color("white")
        hist_ax.yaxis.label.set_color("white")
        hist_fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.18)
        hist_canvas.setStyleSheet("background-color: #2b2b2b;")
        hist_canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        lut_layout.addWidget(hist_canvas)
        min_max_row = QHBoxLayout()
        min_max_row.addWidget(QLabel("Min:"))
        min_spin = QDoubleSpinBox()
        min_spin.setRange(0, 65535)
        min_spin.setValue(0)
        min_spin.setDecimals(0)
        min_spin.setMinimumWidth(90)
        min_max_row.addWidget(min_spin)
        min_max_row.addWidget(QLabel("Max:"))
        max_spin = QDoubleSpinBox()
        max_spin.setRange(0, 65535)
        max_spin.setValue(65535)
        max_spin.setDecimals(0)
        max_spin.setMinimumWidth(90)
        min_max_row.addWidget(max_spin)
        min_max_row.addStretch()
        lut_layout.addLayout(min_max_row)
        lut_group.setLayout(lut_layout)
        main_layout.addWidget(lut_group)

        image_canvas = ImageCanvas()
        image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        image_canvas.enable_drag_selection = True
        image_canvas.click_button = Qt.MouseButton.LeftButton
        image_canvas.selection_button = Qt.MouseButton.RightButton  # so left-click selects tile, not drag
        image_canvas.pan_button = Qt.MouseButton.MiddleButton
        main_layout.addWidget(image_canvas, 1)

        selected_index: List[Optional[int]] = [None]
        current_vmin: List[Optional[float]] = [None]
        current_vmax: List[Optional[float]] = [None]
        hist_vmin_line: List[Any] = [None]
        hist_vmax_line: List[Any] = [None]
        hist_vspan: List[Any] = [None]
        dragging_line: List[Optional[str]] = [None]
        is_updating_lut: List[bool] = [False]

        def tile_at_xy(x: int, y: int) -> Optional[int]:
            layout = dlg._layout
            rows, cols = layout["rows"], layout["cols"]
            border = layout["border"]
            spacing = layout["spacing"]
            cell_w, cell_h = layout["cell_w"], layout["cell_h"]
            if x < border or y < border:
                return None
            c = (x - border) // (cell_w + spacing)
            r = (y - border) // (cell_h + spacing)
            if c < 0 or c >= cols or r < 0 or r >= rows:
                return None
            idx = r * cols + c
            if idx >= layout["n_tiles"]:
                return None
            return idx

        def get_tile_raw_for_hist(tile_idx: int) -> Optional[tuple[np.ndarray, np.dtype]]:
            """Return (flat float array, original dtype) for histogram. dtype used for full intensity range."""
            layout = dlg._layout
            tiles_list = layout["tiles"]
            if tile_idx < 0 or tile_idx >= len(tiles_list):
                return None
            arr, _label, _color = tiles_list[tile_idx]
            if arr is None:
                return None
            if arr.ndim == 2:
                return arr.astype(np.float64).ravel(), arr.dtype
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                if arr.shape[-1] == 1:
                    return arr[..., 0].astype(np.float64).ravel(), arr.dtype
                return arr.mean(axis=-1).astype(np.float64).ravel(), arr.dtype
            return None

        def full_range_for_dtype(dtype: np.dtype, data_min: float, data_max: float) -> tuple[float, float]:
            """Full possible intensity range for consistent histogram across images (e.g. 0–65535 for 16-bit).
            When dtype is uint8/uint16 use that range. When unknown (e.g. float), infer from data: if values
            fit in 0–255 or 0–65535 use that full range so montages can share the same scale; else use data range.
            """
            if dtype == np.uint8:
                return (0.0, 255.0)
            if dtype == np.uint16:
                return (0.0, 65535.0)
            # Unknown or float: infer from data so we still get a stable range when data is really 8/16-bit
            if data_max <= 255 and data_min >= 0:
                return (0.0, 255.0)
            if data_max <= 65535 and data_min >= 0:
                return (0.0, 65535.0)
            return (data_min, data_max)

        hist_range: List[tuple[float, float]] = [(0.0, 1.0)]  # (hmin, hmax) for current tile; used by spinboxes and drag

        def update_histogram_for_tile(tile_idx: Optional[int]) -> None:
            hist_ax.clear()
            if tile_idx is None:
                lut_group.setTitle("LUT — selected tile: drag the red/green lines to set display range")
                hist_ax.text(0.5, 0.5, "Click a tile to select", ha="center", va="center", color="white", fontsize=13, transform=hist_ax.transAxes)
                hist_ax.set_xlim(0, 1)
                hist_ax.set_ylim(0, 1)
                min_spin.setEnabled(False)
                max_spin.setEnabled(False)
                hist_canvas.draw()
                return
            labels = dlg._layout.get("tile_labels", [])
            title = labels[tile_idx] if tile_idx < len(labels) else f"Tile {tile_idx}"
            result = get_tile_raw_for_hist(tile_idx)
            if result is None:
                hist_ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white", transform=hist_ax.transAxes)
                hist_canvas.draw()
                return
            flat, dtype = result
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                hist_canvas.draw()
                return
            fmin, fmax = float(np.min(flat)), float(np.max(flat))
            if fmax <= fmin:
                fmax = fmin + 1.0
            hmin, hmax = full_range_for_dtype(dtype, fmin, fmax)
            hist_range[0] = (hmin, hmax)
            n_bins = 256 if hmax - hmin <= 256 else min(1024, max(256, int(hmax - hmin + 1)))
            hist, bin_edges = np.histogram(flat, bins=n_bins, range=(hmin, hmax))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            log_counts = np.log1p(hist)
            vmin = current_vmin[0] if current_vmin[0] is not None else float(np.percentile(flat, 2.0))
            vmax = current_vmax[0] if current_vmax[0] is not None else float(np.percentile(flat, 98.0))
            vmin = max(hmin, min(vmin, hmax - 1e-9))
            vmax = min(hmax, max(vmax, hmin + 1e-9))
            if vmax <= vmin:
                vmax = vmin + 1.0
            current_vmin[0], current_vmax[0] = vmin, vmax
            hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.12, color="yellow", zorder=0)
            hist_ax.bar(bin_centers, log_counts, width=bin_edges[1] - bin_edges[0], color="#4A90E2", alpha=0.85, edgecolor="none", zorder=1)
            hist_vmin_line[0] = hist_ax.axvline(vmin, color="#FF6B6B", linewidth=2, linestyle="--", label=f"Min: {vmin:.1f}", zorder=2)
            hist_vmax_line[0] = hist_ax.axvline(vmax, color="#4ECDC4", linewidth=2, linestyle="--", label=f"Max: {vmax:.1f}", zorder=2)
            hist_ax.set_xlim(hmin, hmax)
            y_max = float(np.max(log_counts)) if log_counts.size > 0 else 1.0
            hist_ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
            hist_ax.set_xlabel("Intensity", color="white", fontsize=10)
            hist_ax.set_ylabel("Log(Count+1)", color="white", fontsize=10)
            hist_ax.set_title(f"Selected: tile {tile_idx} — {title}", color="white", fontsize=10, pad=4)
            hist_ax.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="white", labelcolor="white", fontsize=9, framealpha=0.9)
            hist_ax.grid(True, alpha=0.25, color="white")
            hist_ax.margins(x=0.02, y=0.08)
            hist_fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.18)
            hist_canvas.draw()
            # Sync min/max spinboxes to current range and limits (block signals to avoid feedback)
            min_spin.blockSignals(True)
            max_spin.blockSignals(True)
            min_spin.setRange(hmin, hmax)
            max_spin.setRange(hmin, hmax)
            min_spin.setValue(vmin)
            max_spin.setValue(vmax)
            min_spin.setDecimals(0 if dtype in (np.uint8, np.uint16) else 2)
            max_spin.setDecimals(0 if dtype in (np.uint8, np.uint16) else 2)
            min_spin.blockSignals(False)
            max_spin.blockSignals(False)
            min_spin.setEnabled(True)
            max_spin.setEnabled(True)

        def refresh_preview(reset_view: bool = True) -> None:
            out = self._build_montage_array(
                per_tile_lut=dlg._per_tile_lut,
                per_tile_color_override=getattr(self, "montage_tile_color_overrides", None) or {},
            )
            if out is None:
                return
            dlg._current_displayed = out
            dummy_mask = np.zeros(out.shape[:2], dtype=np.int32)
            img_scaled, _ = scale_for_preview(out, dummy_mask)
            pixmap = to_qpixmap(img_scaled)
            w, h = img_scaled.shape[1], img_scaled.shape[0]
            dlg._display_w, dlg._display_h = w, h
            image_canvas.set_pixmap(pixmap, (w, h), reset_view=reset_view)
            image_canvas.update()
            image_canvas.repaint()

        def on_canvas_click(x: int, y: int) -> None:
            layout = dlg._layout
            ow, oh = layout.get("out_w"), layout.get("out_h")
            dw, dh = getattr(dlg, "_display_w", None), getattr(dlg, "_display_h", None)
            if ow is not None and oh is not None and dw and dh:
                x = int(round(x * ow / dw))
                y = int(round(y * oh / dh))
            idx = tile_at_xy(x, y)
            if idx is None:
                return
            selected_index[0] = idx
            if idx in dlg._per_tile_lut:
                current_vmin[0], current_vmax[0] = dlg._per_tile_lut[idx]
            else:
                current_vmin[0], current_vmax[0] = None, None
            labels = layout.get("tile_labels", [])
            label = labels[idx] if idx < len(labels) else f"Tile {idx}"
            selected_label.setText(f"Selected: tile {idx} — {label}. Drag the histogram lines to set display range.")
            update_histogram_for_tile(idx)

        def apply_min_max_from_spinboxes() -> None:
            if selected_index[0] is None:
                return
            hmin, hmax = hist_range[0]
            vmin = min_spin.value()
            vmax = max_spin.value()
            vmin = max(hmin, min(vmin, hmax - 1e-9))
            vmax = min(hmax, max(vmax, hmin + 1e-9))
            if vmax <= vmin:
                vmax = vmin + 1e-9
            current_vmin[0], current_vmax[0] = vmin, vmax
            dlg._per_tile_lut[selected_index[0]] = (vmin, vmax)
            min_spin.setValue(vmin)
            max_spin.setValue(vmax)
            if hist_vmin_line[0] is not None:
                hist_vmin_line[0].set_xdata([vmin, vmin])
            if hist_vmax_line[0] is not None:
                hist_vmax_line[0].set_xdata([vmax, vmax])
            if hist_vspan[0] is not None:
                hist_vspan[0].remove()
            hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.15, color="yellow")
            hist_canvas.draw()
            refresh_preview(reset_view=False)

        def on_spin_editing_finished() -> None:
            """Apply LUT when user finishes editing (Enter or focus leave). Lets you type full number without preview jumping on each digit."""
            if selected_index[0] is None or is_updating_lut[0]:
                return
            is_updating_lut[0] = True
            try:
                apply_min_max_from_spinboxes()
            finally:
                is_updating_lut[0] = False

        min_spin.editingFinished.connect(on_spin_editing_finished)
        max_spin.editingFinished.connect(on_spin_editing_finished)

        def on_hist_press(event):
            if event.inaxes != hist_ax or event.xdata is None or selected_index[0] is None:
                return
            x = event.xdata
            result = get_tile_raw_for_hist(selected_index[0])
            if result is None or result[0].size == 0:
                return
            hmin, hmax = hist_range[0]
            rng = max(hmax - hmin, 1e-9)
            thresh = max(rng * 0.02, 1e-6)
            vmin, vmax = current_vmin[0] or hmin, current_vmax[0] or hmax
            if abs(x - vmin) < thresh and abs(x - vmin) <= abs(x - vmax):
                dragging_line[0] = "min"
            elif abs(x - vmax) < thresh:
                dragging_line[0] = "max"
            else:
                mid = (vmin + vmax) / 2
                dragging_line[0] = "min" if x < mid else "max"

        def on_hist_motion(event):
            if event.inaxes != hist_ax or event.xdata is None or dragging_line[0] is None or is_updating_lut[0] or selected_index[0] is None:
                return
            x = event.xdata
            hmin, hmax = hist_range[0]
            is_updating_lut[0] = True
            try:
                vmin, vmax = current_vmin[0] or hmin, current_vmax[0] or hmax
                if dragging_line[0] == "min":
                    new_min = max(hmin, min(x, (current_vmax[0] or hmax) - 1e-9))
                    current_vmin[0] = new_min
                else:
                    new_max = min(hmax, max(x, (current_vmin[0] or hmin) + 1e-9))
                    current_vmax[0] = new_max
                vmin, vmax = current_vmin[0], current_vmax[0]
                dlg._per_tile_lut[selected_index[0]] = (vmin, vmax)
                min_spin.blockSignals(True)
                min_spin.setValue(vmin)
                min_spin.setRange(hmin, hmax)
                min_spin.blockSignals(False)
                max_spin.blockSignals(True)
                max_spin.setValue(vmax)
                max_spin.setRange(hmin, hmax)
                max_spin.blockSignals(False)
                refresh_preview(reset_view=False)
                if hist_vmin_line[0] is not None:
                    hist_vmin_line[0].set_xdata([vmin, vmin])
                if hist_vmax_line[0] is not None:
                    hist_vmax_line[0].set_xdata([vmax, vmax])
                if hist_vspan[0] is not None:
                    hist_vspan[0].remove()
                hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.15, color="yellow")
                hist_canvas.draw()
            finally:
                is_updating_lut[0] = False

        def on_hist_release(event):
            dragging_line[0] = None

        image_canvas.clicked_image.connect(on_canvas_click)
        hist_canvas.mpl_connect("button_press_event", on_hist_press)
        hist_canvas.mpl_connect("motion_notify_event", on_hist_motion)
        hist_canvas.mpl_connect("button_release_event", on_hist_release)

        update_histogram_for_tile(None)

        def on_update() -> None:
            # Rebuild with current spacing/border/paths but keep existing per-tile LUT and color overrides
            result = self._build_montage_array(
                return_layout=True,
                per_tile_lut=dlg._per_tile_lut,
                per_tile_color_override=getattr(self, "montage_tile_color_overrides", None) or {},
            )
            if result is None:
                return
            out, new_layout = result
            dlg._layout = new_layout
            dlg._current_displayed = out
            selected_index[0] = None
            current_vmin[0], current_vmax[0] = None, None
            selected_label.setText("Click an image tile to adjust its display range (min/max). Middle-drag to pan, wheel to zoom.")
            update_histogram_for_tile(None)
            refresh_preview(reset_view=True)

        def on_edit_colors() -> None:
            self._edit_montage_tile_colors(dlg._layout, refresh_callback=lambda: refresh_preview(reset_view=False))

        def on_save() -> None:
            if dlg._current_displayed is None:
                QMessageBox.warning(dlg, "Montage", "Nothing to save. Update preview first.")
                return
            paths = self._get_montage_image_paths()
            default_dir = paths[0].parent if paths else Path(".")
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Save montage", str(default_dir / "montage.png"),
                "PNG (*.png);;TIFF (*.tif *.tiff);;All (*)"
            )
            if not path:
                return
            try:
                from PIL import Image
                Image.fromarray(dlg._current_displayed).save(path)
                self.parent_window.log_output(f"Montage saved: {path}\n")
                QMessageBox.information(dlg, "Montage", f"Saved to:\n{path}")
            except Exception as e:
                QMessageBox.warning(dlg, "Montage", f"Save failed: {e}")

        refresh_preview(reset_view=True)

        btn_row = QHBoxLayout()
        update_btn = QPushButton("Update")
        update_btn.setToolTip("Rebuild montage from main-window settings (paths, grid). Resets per-tile LUT.")
        update_btn.setAutoDefault(False)
        update_btn.setDefault(False)
        update_btn.clicked.connect(on_update)
        edit_colors_btn = QPushButton("Edit channel colors…")
        edit_colors_btn.setToolTip("Set display color per tile (phase, GFP, RFP, etc.). Default = auto from source.")
        edit_colors_btn.setAutoDefault(False)
        edit_colors_btn.setDefault(False)
        edit_colors_btn.clicked.connect(on_edit_colors)
        save_btn = QPushButton("Save montage")
        save_btn.setAutoDefault(False)
        save_btn.setDefault(False)
        save_btn.clicked.connect(on_save)
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.setDefault(False)
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(update_btn)
        btn_row.addWidget(edit_colors_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        main_layout.addLayout(btn_row)

        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, dlg)
        close_shortcut.activated.connect(dlg.close)
        minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), dlg)
        minimize_shortcut.activated.connect(dlg.showMinimized)
        dlg.setLayout(main_layout)
        dlg.show()

    def _show_scalebar_preview_dialog(
        self, raw: np.ndarray, data_min: float, data_max: float
    ) -> None:
        """Non-modal scale bar preview with histogram LUT (drag min/max), Update, Save, Cmd+W."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        dlg = QDialog(self)
        dlg.setWindowTitle("Scale bar preview")
        dlg.setModal(False)
        dlg.setMinimumSize(900, 750)
        layout = QVBoxLayout(dlg)

        # LUT: histogram with draggable min/max (like file conversion preview)
        lut_group = QGroupBox("LUT — drag the red/green lines to set display range; save when happy")
        lut_layout = QVBoxLayout()
        hist_fig = Figure(figsize=(10, 2.8))
        hist_canvas = FigureCanvas(hist_fig)
        hist_canvas.setMinimumHeight(160)
        hist_canvas.setMaximumHeight(160)
        hist_ax = hist_fig.add_subplot(111)
        hist_fig.patch.set_facecolor("#2b2b2b")
        hist_ax.set_facecolor("#2b2b2b")
        hist_ax.tick_params(colors="white", labelsize=9)
        for spine in ["bottom", "top", "left", "right"]:
            hist_ax.spines[spine].set_color("white")
        hist_ax.xaxis.label.set_color("white")
        hist_ax.yaxis.label.set_color("white")
        hist_fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.18)
        hist_canvas.setStyleSheet("background-color: #2b2b2b;")
        lut_layout.addWidget(hist_canvas)
        lut_group.setLayout(lut_layout)
        layout.addWidget(lut_group)

        image_canvas = ImageCanvas()
        image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        layout.addWidget(image_canvas, 1)

        dlg._raw = raw.copy()
        dlg._data_min = data_min
        dlg._data_max = data_max
        dlg._current_displayed = None
        current_vmin = [data_min]
        current_vmax = [data_max]
        hist_vmin_line = [None]
        hist_vmax_line = [None]
        hist_vspan = [None]
        is_updating_lut = [False]
        dragging_line = [None]

        def update_histogram(vmin: float, vmax: float) -> None:
            frame = dlg._raw
            flat = np.asarray(frame).astype(np.float64).ravel()
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                flat = np.asarray([dlg._data_min, dlg._data_max])
            fmin, fmax = float(np.min(flat)), float(np.max(flat))
            if fmax <= fmin:
                fmax = fmin + 1.0
            bins = min(256, max(2, int(fmax - fmin + 1)))
            hist, bin_edges = np.histogram(flat, bins=bins, range=(fmin, fmax))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_ax.clear()
            hist_ax.bar(bin_centers, np.log1p(hist), width=bin_edges[1] - bin_edges[0], color="#4A90E2", alpha=0.8, edgecolor="none")
            hist_vmin_line[0] = hist_ax.axvline(vmin, color="#FF6B6B", linewidth=2, linestyle="--", label=f"Min: {vmin:.1f}")
            hist_vmax_line[0] = hist_ax.axvline(vmax, color="#4ECDC4", linewidth=2, linestyle="--", label=f"Max: {vmax:.1f}")
            hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.15, color="yellow")
            hist_ax.set_xlabel("Intensity", color="white", fontsize=10)
            hist_ax.set_ylabel("Log(Count+1)", color="white", fontsize=10)
            hist_ax.set_title("Click/drag the lines to adjust min/max", color="white", fontsize=9, pad=8)
            hist_ax.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="white", labelcolor="white", fontsize=8, framealpha=0.9)
            hist_ax.grid(True, alpha=0.25, color="white")
            hist_ax.margins(x=0.02, y=0.08)
            hist_fig.tight_layout(pad=2.0)
            hist_canvas.draw()

        def refresh_preview(reset_view: bool = False) -> None:
            vmin = current_vmin[0]
            vmax = current_vmax[0]
            if vmax <= vmin:
                vmax = vmin + 1e-9
            out = self._build_scalebar_preview_image(dlg._raw, vmin, vmax)
            if out is None:
                return
            dlg._current_displayed = out
            dummy_mask = np.zeros(out.shape[:2], dtype=np.int32)
            img_scaled, _ = scale_for_preview(out, dummy_mask)
            pixmap = to_qpixmap(img_scaled)
            w, h = img_scaled.shape[1], img_scaled.shape[0]
            dlg._last_pixmap = pixmap
            dlg._last_size = (w, h)
            image_canvas.set_pixmap(pixmap, (w, h), reset_view=reset_view)
            image_canvas.update()
            image_canvas.repaint()

        def on_hist_press(event):
            if event.inaxes != hist_ax or event.xdata is None:
                return
            x = event.xdata
            fmin, fmax = dlg._data_min, dlg._data_max
            rng = max(fmax - fmin, 1e-9)
            thresh = max(rng * 0.05, 1e-6)
            if abs(x - current_vmin[0]) < thresh and abs(x - current_vmin[0]) <= abs(x - current_vmax[0]):
                dragging_line[0] = "min"
            elif abs(x - current_vmax[0]) < thresh:
                dragging_line[0] = "max"
            else:
                mid = (current_vmin[0] + current_vmax[0]) / 2
                dragging_line[0] = "min" if x < mid else "max"

        def on_hist_motion(event):
            if event.inaxes != hist_ax or event.xdata is None or dragging_line[0] is None or is_updating_lut[0]:
                return
            x = event.xdata
            fmin, fmax = dlg._data_min, dlg._data_max
            is_updating_lut[0] = True
            try:
                if dragging_line[0] == "min":
                    new_min = max(fmin, min(x, current_vmax[0] - 1e-9))
                    current_vmin[0] = new_min
                else:
                    new_max = min(fmax, max(x, current_vmin[0] + 1e-9))
                    current_vmax[0] = new_max
                vmin, vmax = current_vmin[0], current_vmax[0]
                refresh_preview(reset_view=False)
                if hist_vmin_line[0] is not None:
                    hist_vmin_line[0].set_xdata([vmin, vmin])
                if hist_vmax_line[0] is not None:
                    hist_vmax_line[0].set_xdata([vmax, vmax])
                if hist_vspan[0] is not None:
                    hist_vspan[0].remove()
                hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.15, color="yellow")
                hist_canvas.draw()
            finally:
                is_updating_lut[0] = False

        def on_hist_release(event):
            dragging_line[0] = None

        hist_canvas.mpl_connect("button_press_event", on_hist_press)
        hist_canvas.mpl_connect("motion_notify_event", on_hist_motion)
        hist_canvas.mpl_connect("button_release_event", on_hist_release)

        def on_update() -> None:
            raw_result = self._get_scalebar_image_raw()
            if raw_result is None:
                QMessageBox.warning(dlg, "Scale bar", "Could not reload image. Check path and format.")
                return
            dlg._raw, dlg._data_min, dlg._data_max = raw_result
            current_vmin[0] = dlg._data_min
            current_vmax[0] = dlg._data_max
            update_histogram(current_vmin[0], current_vmax[0])
            refresh_preview(reset_view=True)

        def on_save() -> None:
            if dlg._current_displayed is None:
                QMessageBox.warning(dlg, "Scale bar", "Nothing to save. Update preview first.")
                return
            default_path = Path(self.scalebar_path.text().strip())
            default_path = default_path.parent / f"{default_path.stem}_scalebar.png"
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Save with scale bar", str(default_path),
                "PNG (*.png);;TIFF (*.tif *.tiff);;All (*)"
            )
            if not path:
                return
            try:
                from PIL import Image
                Image.fromarray(dlg._current_displayed).save(path)
                self.parent_window.log_output(f"Scale bar image saved: {path}\n")
                QMessageBox.information(dlg, "Scale bar", f"Saved to:\n{path}")
            except Exception as e:
                QMessageBox.warning(dlg, "Scale bar", f"Save failed: {e}")

        update_histogram(current_vmin[0], current_vmax[0])
        refresh_preview(reset_view=True)

        btn_row = QHBoxLayout()
        update_btn = QPushButton("Update")
        update_btn.setToolTip(
            "Reload image and redraw scale bar using current main-window settings (pixel size, length, position, font size, etc.). Resets zoom/pan to fit."
        )
        update_btn.clicked.connect(on_update)
        save_btn = QPushButton("Save with scale bar")
        save_btn.clicked.connect(on_save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(update_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, dlg)
        close_shortcut.activated.connect(dlg.close)
        minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), dlg)
        minimize_shortcut.activated.connect(dlg.showMinimized)

        dlg.setLayout(layout)
        dlg.show()

    def _show_image_dialog(self, img: np.ndarray, title: str) -> None:
        """Show a non-modal preview dialog with ImageCanvas (zoom, pan), Reset view, Cmd+W, and Ctrl+M."""
        dummy_mask = np.zeros(img.shape[:2], dtype=np.int32)
        img_scaled, _ = scale_for_preview(img, dummy_mask)
        pixmap = to_qpixmap(img_scaled)
        w, h = img_scaled.shape[1], img_scaled.shape[0]

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setModal(False)
        dlg.setMinimumSize(900, 700)
        layout = QVBoxLayout(dlg)

        image_canvas = ImageCanvas()
        image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        image_canvas.set_pixmap(pixmap, (w, h), reset_view=True)
        layout.addWidget(image_canvas, 1)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset view")
        reset_btn.setToolTip("Reset zoom and pan to fit image.")
        def on_reset_view():
            image_canvas.set_pixmap(pixmap, (w, h), reset_view=True)
        reset_btn.clicked.connect(on_reset_view)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, dlg)
        close_shortcut.activated.connect(dlg.close)
        minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), dlg)
        minimize_shortcut.activated.connect(dlg.showMinimized)

        dlg.setLayout(layout)
        dlg.show()

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
        
        # For completion popup "Open output folder"
        self._last_video_output_dir = output_dir
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
            self.parent_window.log_output(f"\n✓ {message}\n")
            out_dir = getattr(self, "_last_video_output_dir", None)
            if out_dir is not None and Path(out_dir).exists():
                msg = QMessageBox(self)
                msg.setWindowTitle("Video Complete")
                msg.setText(message if message else "Growth curve video(s) created successfully.")
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
                    "Video Complete",
                    message if message else "Growth curve video(s) created successfully.",
                )
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


