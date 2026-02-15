"""
Analyze masks tab: run analyze_cell_areas on mask TIFFs, combine summaries, morphological filters.
"""
from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile
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
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.constants import PROJECT_ROOT
from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.preferences import get_default_browse_dir
from bactimate.gui.image_utils import to_qpixmap, scale_for_preview
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.utils import safe_imread
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog
from bactimate.gui.workers import WorkerThread

class AnalyzeMasksTab(QWidget):
    """Tab for analyzing existing masks"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # Shared font/size policy for path line edits (used in _build_* and FI section below)
        file_font = QFont()
        file_font.setPointSize(12)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        file_group = self._build_file_group()
        scroll_layout.addWidget(file_group)

        combine_group = self._build_combine_group()
        scroll_layout.addWidget(combine_group)

        # Analysis parameters - side-by-side layout (compact to avoid horizontal scroll)
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_container_layout = QHBoxLayout()
        analysis_container_layout.setSpacing(8)
        analysis_container_layout.setContentsMargins(6, 6, 6, 6)
        
        # First box: Basic and Morphological filters (fused into one)
        morph_params_group = QGroupBox("Morphological Filters")
        morph_params_layout = QVBoxLayout()
        
        pixel_layout = QHBoxLayout()
        pixel_label = QLabel("Pixel Size (µm):")
        pixel_info_icon = create_info_icon(
            "Pixel size in micrometers (µm) for area calculations.\n\n"
            "Default: 0.108 µm (or from metadata if available)\n"
            "Source: Hardcoded default, or extracted from metadata\n\n"
            "This value is used to convert pixel areas to µm².\n"
            "If you loaded a file with pixel size metadata, this will be auto-filled.\n\n"
            "Tip: Check your microscope settings or ND2 metadata for the correct value."
        , parent_widget=self)
        pixel_layout.addWidget(pixel_label)
        pixel_layout.addWidget(pixel_info_icon)
        # Add indicator label for auto-filled values or warning (before spin box, on the left)
        self.pixel_size_indicator = QLabel("⚠ Check value!")
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self.pixel_size_indicator.setFixedWidth(100)  # Fixed width for warning text
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size_indicator.setToolTip("Default value - verify this matches your metadata!")
        pixel_layout.addWidget(self.pixel_size_indicator)
        pixel_layout.addStretch()  # Add stretch before spin box to align to right
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 10.0)
        self.pixel_size.setDecimals(3)
        self.pixel_size.setSingleStep(0.001)
        # Always default to 0.108 on initialization (metadata will update it if a file is loaded)
        # Use explicit float to avoid any precision issues
        default_pixel_size = float(0.108)
        self.pixel_size.setValue(default_pixel_size)
        self.pixel_size.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        # valueChanged will only fire when user finishes editing (Enter or focus loss)
        self.pixel_size.setKeyboardTracking(False)
        # Track if metadata was loaded to show warning
        self.pixel_size_metadata_loaded = False
        # Connect value change to hide warning when user manually changes value
        self.pixel_size.valueChanged.connect(self._on_pixel_size_changed)
        # Debug: Verify the value was set correctly (can be removed later)
        # print(f"DEBUG: Pixel size set to: {self.pixel_size.value()}")
        pixel_layout.addWidget(self.pixel_size)
        morph_params_layout.addLayout(pixel_layout)
        
        # Morphological filters (now part of the same group)
        morph_layout = QVBoxLayout()
        
        aspect_min_layout = QHBoxLayout()
        aspect_min_label = QLabel("Min Aspect Ratio:")
        aspect_min_info_icon = create_info_icon(
            "Minimum aspect ratio (length/width) for rod-shaped cell filtering.\n\n"
            "Default: 1.5\n"
            "Source: Hardcoded default for rod-shaped cell detection\n\n"
            "Aspect ratio = major_axis_length / minor_axis_length\n"
            "Filters out circular or nearly circular objects.\n\n"
            "Typical values:\n"
            "- Rod-shaped bacteria: 2.0-5.0\n"
            "- Very elongated cells: 3.0-10.0\n"
            "- Less strict: 1.5-2.0\n\n"
            "Range: 1.0 to 20.0\n"
            "Tip: Rod-shaped cells typically have aspect ratio > 2.0"
        , parent_widget=self)
        aspect_min_layout.addWidget(aspect_min_label)
        aspect_min_layout.addWidget(aspect_min_info_icon)
        aspect_min_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_aspect_ratio = QDoubleSpinBox()
        self.min_aspect_ratio.setRange(1.0, 20.0)
        self.min_aspect_ratio.setValue(1.5)
        self.min_aspect_ratio.setDecimals(1)
        self.min_aspect_ratio.setSingleStep(0.1)
        self.min_aspect_ratio.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_aspect_ratio.setKeyboardTracking(False)
        aspect_min_layout.addWidget(self.min_aspect_ratio)
        morph_layout.addLayout(aspect_min_layout)
        
        aspect_max_layout = QHBoxLayout()
        aspect_max_label = QLabel("Max Aspect Ratio:")
        aspect_max_info_icon = create_info_icon(
            "Maximum aspect ratio (length/width) for rod-shaped cell filtering.\n\n"
            "Default: 10.0\n"
            "Source: Hardcoded default to filter out extremely elongated artifacts\n\n"
            "Filters out objects that are too elongated (likely artifacts or debris chains).\n\n"
            "Range: 1.0 to 50.0\n"
            "Tip: Most rod-shaped cells have aspect ratio < 10.0. Very high values may indicate artifacts."
        , parent_widget=self)
        aspect_max_layout.addWidget(aspect_max_label)
        aspect_max_layout.addWidget(aspect_max_info_icon)
        aspect_max_layout.addStretch()  # Add stretch before spin box to align to right
        self.max_aspect_ratio = QDoubleSpinBox()
        self.max_aspect_ratio.setRange(1.0, 50.0)
        self.max_aspect_ratio.setValue(10.0)
        self.max_aspect_ratio.setDecimals(1)
        self.max_aspect_ratio.setSingleStep(0.1)
        self.max_aspect_ratio.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.max_aspect_ratio.setKeyboardTracking(False)
        aspect_max_layout.addWidget(self.max_aspect_ratio)
        morph_layout.addLayout(aspect_max_layout)
        
        ecc_layout = QHBoxLayout()
        ecc_label = QLabel("Min Eccentricity:")
        ecc_info_icon = create_info_icon(
            "Minimum eccentricity for rod-shaped cell filtering.\n\n"
            "Default: 0.5\n"
            "Source: Hardcoded default for rod-shaped cell detection\n\n"
            "Eccentricity measures how elongated an object is:\n"
            "- 0.0 = perfect circle\n"
            "- 1.0 = line (infinitely elongated)\n\n"
            "Filters out circular or nearly circular objects.\n\n"
            "Range: 0.0 to 1.0\n"
            "Typical values: 0.5-0.9 for rod-shaped cells\n"
            "Tip: Higher values (0.7-0.9) are more strict, filtering out rounder objects."
        , parent_widget=self)
        ecc_layout.addWidget(ecc_label)
        ecc_layout.addWidget(ecc_info_icon)
        ecc_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_eccentricity = QDoubleSpinBox()
        self.min_eccentricity.setRange(0.0, 1.0)
        self.min_eccentricity.setValue(0.5)
        self.min_eccentricity.setDecimals(2)
        self.min_eccentricity.setSingleStep(0.05)
        self.min_eccentricity.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_eccentricity.setKeyboardTracking(False)
        ecc_layout.addWidget(self.min_eccentricity)
        morph_layout.addLayout(ecc_layout)
        
        sol_layout = QHBoxLayout()
        sol_label = QLabel("Min Solidity:")
        sol_info_icon = create_info_icon(
            "Minimum solidity for rod-shaped cell filtering.\n\n"
            "Default: 0.7\n"
            "Source: Hardcoded default for filtering irregular shapes\n\n"
            "Solidity = area / convex_hull_area\n"
            "Measures how 'solid' or 'filled' an object is:\n"
            "- 1.0 = perfectly solid (no indentations)\n"
            "- Lower values = more irregular, indented shapes\n\n"
            "Filters out irregular shapes, debris, and poorly segmented objects.\n\n"
            "Range: 0.0 to 1.0\n"
            "Tip: Higher values (0.8-0.9) filter out more irregular shapes and debris."
        , parent_widget=self)
        sol_layout.addWidget(sol_label)
        sol_layout.addWidget(sol_info_icon)
        sol_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_solidity = QDoubleSpinBox()
        self.min_solidity.setRange(0.0, 1.0)
        self.min_solidity.setValue(0.7)
        self.min_solidity.setDecimals(2)
        self.min_solidity.setSingleStep(0.05)
        self.min_solidity.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_solidity.setKeyboardTracking(False)
        sol_layout.addWidget(self.min_solidity)
        morph_layout.addLayout(sol_layout)
        
        # Add morphological filters to the main layout
        morph_params_layout.addLayout(morph_layout)
        
        morph_params_group.setLayout(morph_params_layout)
        morph_params_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        morph_params_group.setContentsMargins(4, 6, 4, 6)
        analysis_container_layout.addWidget(morph_params_group)
        
        # Second box: Size filters (side-by-side)
        size_group = QGroupBox("Size Filters")
        size_layout = QVBoxLayout()
        
        min_area_layout = QHBoxLayout()
        min_area_label = QLabel("Min Area (µm², 0=no limit):")
        min_area_info_icon = create_info_icon(
            "Minimum cell area in square micrometers (µm²).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Filters out cells smaller than this value (likely debris or noise).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for bacteria: 0.5-5 µm²\n"
            "Range: 0 to 10000 µm²\n\n"
            "Tip: Use to filter out undersized objects that are likely artifacts."
        , parent_widget=self)
        min_area_layout.addWidget(min_area_label)
        min_area_layout.addWidget(min_area_info_icon)
        min_area_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_area = QDoubleSpinBox()
        self.min_area.setRange(0, 10000)
        self.min_area.setValue(0)
        self.min_area.setDecimals(1)
        self.min_area.setSingleStep(0.5)
        self.min_area.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_area.setKeyboardTracking(False)
        min_area_layout.addWidget(self.min_area)
        size_layout.addLayout(min_area_layout)
        
        max_area_layout = QHBoxLayout()
        max_area_label = QLabel("Max Area (µm², 0=no limit):")
        max_area_info_icon = create_info_icon(
            "Maximum cell area in square micrometers (µm²).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Filters out cells larger than this value (likely mis-segmented debris or clumps).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for bacteria: 10-50 µm²\n"
            "Range: 0 to 10000 µm²\n\n"
            "Tip: Use to filter out oversized objects that are likely artifacts."
        , parent_widget=self)
        max_area_layout.addWidget(max_area_label)
        max_area_layout.addWidget(max_area_info_icon)
        max_area_layout.addStretch()  # Add stretch before spin box to align to right
        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(0, 10000)
        self.max_area.setValue(0)
        self.max_area.setDecimals(1)
        self.max_area.setSingleStep(1.0)
        self.max_area.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.max_area.setKeyboardTracking(False)
        max_area_layout.addWidget(self.max_area)
        size_layout.addLayout(max_area_layout)
        
        min_width_layout = QHBoxLayout()
        min_width_label = QLabel("Min Width (µm, 0=no limit):")
        min_width_info_icon = create_info_icon(
            "Minimum cell width in micrometers (µm).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Width is measured as the minor axis (perpendicular to length).\n"
            "Filters out cells narrower than this value (likely debris or noise).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for rod-shaped bacteria: 0.3-0.5 µm\n"
            "Range: 0 to 100 µm\n\n"
            "Tip: Use to filter out objects that are too narrow to be single cells."
        , parent_widget=self)
        min_width_layout.addWidget(min_width_label)
        min_width_layout.addWidget(min_width_info_icon)
        min_width_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_width = QDoubleSpinBox()
        self.min_width.setRange(0, 100)
        self.min_width.setValue(0)
        self.min_width.setDecimals(1)
        self.min_width.setSingleStep(0.1)
        self.min_width.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_width.setKeyboardTracking(False)
        min_width_layout.addWidget(self.min_width)
        size_layout.addLayout(min_width_layout)
        
        max_width_layout = QHBoxLayout()
        max_width_label = QLabel("Max Width (µm, 0=no limit):")
        max_width_info_icon = create_info_icon(
            "Maximum cell width in micrometers (µm).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Width is measured as the minor axis (perpendicular to length).\n"
            "Filters out cells wider than this value (likely mis-segmented debris or clumps).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for rod-shaped bacteria: 0.5-2.0 µm\n"
            "Range: 0 to 100 µm\n\n"
            "Tip: Use to filter out objects that are too wide to be single cells."
        , parent_widget=self)
        max_width_layout.addWidget(max_width_label)
        max_width_layout.addWidget(max_width_info_icon)
        max_width_layout.addStretch()  # Add stretch before spin box to align to right
        self.max_width = QDoubleSpinBox()
        self.max_width.setRange(0, 100)
        self.max_width.setValue(0)
        self.max_width.setDecimals(1)
        self.max_width.setSingleStep(0.1)
        self.max_width.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.max_width.setKeyboardTracking(False)
        max_width_layout.addWidget(self.max_width)
        size_layout.addLayout(max_width_layout)
        
        # Min Length filter (5th size filter to match left side)
        min_length_layout = QHBoxLayout()
        min_length_label = QLabel("Min Length (µm, 0=no limit):")
        min_length_info_icon = create_info_icon(
            "Minimum cell length in micrometers (µm).\n\n"
            "Default: 0 (no limit)\n"
            "Source: Optional filter, disabled by default\n\n"
            "Length is measured as the major axis (longest dimension).\n"
            "Filters out cells shorter than this value (likely debris or noise).\n\n"
            "Set to 0 to disable this filter.\n"
            "Typical values for rod-shaped bacteria: 1.0-3.0 µm\n"
            "Range: 0 to 100 µm\n\n"
            "Tip: Use to filter out objects that are too short to be single cells.",
            parent_widget=self
        )
        min_length_layout.addWidget(min_length_label)
        min_length_layout.addWidget(min_length_info_icon)
        min_length_layout.addStretch()  # Add stretch before spin box to align to right
        self.min_length = QDoubleSpinBox()
        self.min_length.setRange(0, 100)
        self.min_length.setValue(0)
        self.min_length.setDecimals(1)
        self.min_length.setSingleStep(0.1)
        self.min_length.setFixedWidth(100)  # Fixed width for uniform appearance
        # Disable keyboard tracking to reduce event processing overhead
        self.min_length.setKeyboardTracking(False)
        min_length_layout.addWidget(self.min_length)
        size_layout.addLayout(min_length_layout)
        
        size_group.setLayout(size_layout)
        size_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        size_group.setContentsMargins(4, 6, 4, 6)
        analysis_container_layout.addWidget(size_group)
        
        # Reset to defaults button for analysis parameters (at end of horizontal row)
        reset_analysis_btn = QPushButton("Reset to Default")
        reset_analysis_btn.setStyleSheet(
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
        reset_analysis_btn.setToolTip("Reset all analysis parameters to their default values")
        reset_analysis_btn.clicked.connect(self.reset_analysis_defaults)
        analysis_container_layout.addWidget(reset_analysis_btn)
        
        analysis_group.setLayout(analysis_container_layout)
        scroll_layout.addWidget(analysis_group)
        
        # Options and Analyze FI (side by side)
        options_container_layout = QHBoxLayout()
        
        # Left box: Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Row 1: Include detailed per-cell data + Create filter visualization
        options_row1 = QHBoxLayout()
        detailed_layout = QHBoxLayout()
        self.detailed = QCheckBox("Include detailed per-cell data")
        self.detailed.setChecked(True)  # Default to checked
        detailed_info_icon = create_info_icon(
            "Include detailed per-cell measurements in the Excel output.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Optional output mode\n\n"
            "When enabled: Excel file includes a separate sheet with measurements for each cell:\n"
            "- Cell ID, area, width, length, aspect ratio, eccentricity, solidity, etc.\n\n"
            "When disabled: Only summary data (total area per frame) is included.\n\n"
            "Use when: You need individual cell measurements for further analysis.\n"
            "Note: Creates larger Excel files but provides more detailed information.",
            parent_widget=self
        )
        detailed_layout.addWidget(self.detailed)
        detailed_layout.addWidget(detailed_info_icon)
        detailed_layout.addStretch()
        options_row1.addLayout(detailed_layout)
        
        vis_layout = QHBoxLayout()
        self.create_vis = QCheckBox("Create filter visualization")
        self.create_vis.setChecked(True)
        vis_info_icon = create_info_icon(
            "Create visualization TIFF showing which cells were accepted/rejected.\n\n"
            "Default: Enabled (checked)\n"
            "Source: Recommended for quality control\n\n"
            "When enabled: Creates a TIFF file where:\n"
            "- Green = accepted cells (passed all filters)\n"
            "- Red = rejected cells (failed one or more filters)\n\n"
            "Saved as: filename_filter_visualization.tif\n\n"
            "Use to: Verify that filters are working correctly and adjust parameters if needed.\n"
            "Note: Increases processing time but very useful for validation.",
            parent_widget=self
        )
        vis_layout.addWidget(self.create_vis)
        vis_layout.addWidget(vis_info_icon)
        vis_layout.addStretch()
        options_row1.addLayout(vis_layout)
        options_layout.addLayout(options_row1)
        
        # Row 2: same structure as row 1 (two sub-layouts with stretch) so spacing matches
        options_row2 = QHBoxLayout()
        combined_summary_layout = QHBoxLayout()
        self.create_combined_summary = QCheckBox("Create combined summary")
        self.create_combined_summary.setChecked(False)
        combined_summary_info_icon = create_info_icon(
            "When analyzing a folder of mask files, create one extra Excel that combines all Summary sheets.\n\n"
            "Output: {folder_name}_combined_summary.xlsx with one sheet:\n"
            "- 'Combined Summary (wide)': one row per frame, one set of columns per file.\n\n"
            "Use for: Downstream plotting, stats, or comparing many samples in one table.",
            parent_widget=self
        )
        combined_summary_layout.addWidget(self.create_combined_summary)
        combined_summary_layout.addWidget(combined_summary_info_icon)
        combined_summary_layout.addStretch()
        options_row2.addLayout(combined_summary_layout)
        exclude_border_layout = QHBoxLayout()
        self.exclude_border_cb = QCheckBox("Exclude masks at borders")
        self.exclude_border_cb.setChecked(False)
        exclude_border_info_icon = create_info_icon(
            "When enabled, any mask region that touches the top, bottom, left or right edge of the image "
            "is excluded from counts and area (useful to avoid partial cells at the border).",
            parent_widget=self
        )
        exclude_border_layout.addWidget(self.exclude_border_cb)
        exclude_border_layout.addWidget(exclude_border_info_icon)
        exclude_border_layout.addStretch()
        options_row2.addLayout(exclude_border_layout)
        options_layout.addLayout(options_row2)
        
        options_group.setLayout(options_layout)
        options_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        options_container_layout.addWidget(options_group)
        
        # Right box: Analyze FI (with drag-and-drop support)
        fi_group = DragDropGroupBox(
            "Analyze FI",
            parent=self,
            on_drop_callback=self.handle_dropped_fi_file
        )
        fi_layout = QVBoxLayout()
        
        fi_checkbox_layout = QHBoxLayout()
        self.analyze_fi = QCheckBox("Analyze FI")
        self.analyze_fi.setChecked(False)
        fi_info_icon = create_info_icon(
            "Analyze fluorescence intensity (FI) from a separate FI channel image.\n\n"
            "When enabled: Requires selecting an FI channel TIFF file.\n"
            "The analysis will always add:\n"
            "- Mean FI intensity (average pixel intensity within each cell)\n\n"
            "Optional (if 'Contour FI' is enabled):\n"
            "- Contour FI intensity (average intensity at cell boundary)\n"
            "- FI ratio (contour FI / mean FI)\n\n"
            "Use when: You have a separate fluorescence channel image and want to quantify\n"
            "fluorescence intensity per cell.",
            parent_widget=self
        )
        fi_checkbox_layout.addWidget(self.analyze_fi)
        fi_checkbox_layout.addWidget(fi_info_icon)
        
        # Optional contour FI analysis toggle + contour thickness (same line)
        self.analyze_fi_contour = QCheckBox("Contour FI")
        # Keep it OFF until Analyze FI is enabled (prevents confusing "checked but disabled" state)
        self.analyze_fi_contour.setChecked(False)
        self.analyze_fi_contour.setEnabled(False)  # Disabled until Analyze FI is enabled
        contour_fi_info_icon = create_info_icon(
            "Compute contour-based FI metrics (boundary intensity + FI ratio).\n\n"
            "When disabled: Only Mean FI Intensity will be calculated.\n"
            "Contour thickness is ignored.",
            parent_widget=self
        )
        fi_checkbox_layout.addWidget(self.analyze_fi_contour)
        fi_checkbox_layout.addWidget(contour_fi_info_icon)

        contour_thickness_label = QLabel("Contour thickness (pixels):")
        contour_thickness_info_icon = create_info_icon(
            "Thickness of the contour region used for FI intensity calculation.\n\n"
            "Default: 1 pixel (single-pixel boundary)\n"
            "Range: 1-10 pixels\n\n"
            "The contour is the region at the cell boundary.\n"
            "Increasing thickness includes more pixels near the boundary.\n\n"
            "Note: This affects both the preview visualization and the final analysis.",
            parent_widget=self
        )
        fi_checkbox_layout.addWidget(contour_thickness_label)
        fi_checkbox_layout.addWidget(contour_thickness_info_icon)
        self.fi_contour_thickness = QSpinBox()
        self.fi_contour_thickness.setRange(1, 10)
        self.fi_contour_thickness.setValue(1)  # Default: 1 pixel (current behavior)
        self.fi_contour_thickness.setEnabled(False)  # Disabled by default
        self.fi_contour_thickness.setFixedWidth(80)
        fi_checkbox_layout.addWidget(self.fi_contour_thickness)
        fi_checkbox_layout.addStretch()
        fi_layout.addLayout(fi_checkbox_layout)
        
        # FI file browse (only enabled when checkbox is checked) with drag-and-drop
        fi_file_layout = QHBoxLayout()
        self.fi_file_path = QLineEdit()
        self.fi_file_path.setPlaceholderText("Select FI channel TIFF file... (or drag and drop)")
        self.fi_file_path.setMinimumWidth(400)  # Slightly smaller since it's in a side box
        self.fi_file_path.setFont(file_font)
        self.fi_file_path.setSizePolicy(size_policy)
        self.fi_file_path.setEnabled(False)  # Disabled by default
        self.fi_file_btn = QPushButton("Browse FI File...")
        self.fi_file_btn.setFixedWidth(120)
        self.fi_file_btn.setEnabled(False)  # Disabled by default
        self.fi_file_btn.clicked.connect(self.select_fi_file)
        fi_file_layout.addWidget(self.fi_file_path, 0)
        fi_file_layout.addWidget(self.fi_file_btn, 0)
        fi_layout.addLayout(fi_file_layout)
        
        # Connect checkbox(es) to enable/disable FI file selection and contour options
        self.analyze_fi.toggled.connect(self._update_fi_file_state)
        self.analyze_fi_contour.toggled.connect(self._update_fi_contour_state)
        
        fi_group.setLayout(fi_layout)
        fi_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        options_container_layout.addWidget(fi_group)
        
        scroll_layout.addLayout(options_container_layout)
        
        # Preview and Analyze buttons
        # Preview button - modern styling applied via global stylesheet
        self.preview_btn = QPushButton("Preview Filter Effects")
        self.preview_btn.setToolTip("Preview how current filter parameters would affect the selected mask file")
        self.preview_btn.setEnabled(False)  # Disabled until file/folder is selected
        self.preview_btn.clicked.connect(self.preview_filter_effects)
        scroll_layout.addWidget(self.preview_btn)
        
        self.analyze_btn = QPushButton("Analyze Masks")
        self.analyze_btn.setEnabled(False)  # Disabled until file/folder is selected
        # Modern styling applied via global stylesheet
        self.analyze_btn.clicked.connect(self.analyze_masks)
        scroll_layout.addWidget(self.analyze_btn)
        
        # Connect file path changes to update button states
        self.file_path.textChanged.connect(self._update_analyze_button_state)
        # Initialize state
        self._update_analyze_button_state()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_file_group(self) -> QWidget:
        """Build the mask file/folder input group with path line edit and browse buttons."""
        file_group = DragDropGroupBox(
            "Mask File/Folder",
            parent=self,
            on_drop_callback=self.handle_dropped_path_analyze
        )
        file_font = QFont()
        file_font.setPointSize(12)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a mask TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)
        self.file_path.setFont(file_font)
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

    def _build_combine_group(self) -> QGroupBox:
        """Build the combine summaries from folder group."""
        file_font = QFont()
        file_font.setPointSize(12)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        combine_group = QGroupBox("Combine existing summaries from folder")
        combine_layout = QHBoxLayout()
        self.combine_folder_path = QLineEdit()
        self.combine_folder_path.setPlaceholderText("Folder with *_analysis.xlsx (subfolders)...")
        self.combine_folder_path.setMinimumWidth(500)
        self.combine_folder_path.setFont(file_font)
        self.combine_folder_path.setSizePolicy(size_policy)
        combine_browse_btn = QPushButton("Browse...")
        combine_browse_btn.setFixedWidth(120)
        combine_browse_btn.clicked.connect(self._browse_combine_folder)
        combine_btn = QPushButton("Combine")
        combine_btn.setFixedWidth(130)
        combine_btn.setToolTip("Read all Summary sheets from analysis Excel files in the selected folder and its subfolders, combine into one wide-format table, and save as {folder_name}_combined_summary.xlsx in the selected folder.")
        combine_btn.clicked.connect(self._combine_summaries_from_folder)
        combine_layout.addWidget(self.combine_folder_path, 0)
        combine_layout.addWidget(combine_browse_btn, 0)
        combine_layout.addWidget(combine_btn, 0)
        combine_group.setLayout(combine_layout)
        return combine_group
    
    def _update_analyze_button_state(self):
        """Update analyze and preview button enabled state based on whether file/folder is selected"""
        has_path = bool(self.file_path.text() and self.file_path.text().strip())
        if hasattr(self, 'analyze_btn'):
            self.analyze_btn.setEnabled(has_path)
        if hasattr(self, 'preview_btn'):
            self.preview_btn.setEnabled(has_path)
    
    def select_file(self):
        """Open file dialog to select a mask TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask TIFF File", get_default_browse_dir(), "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self._update_analyze_button_state()
    
    def select_folder(self):
        """Open folder dialog to select a folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Mask Files", get_default_browse_dir()
        )
        if folder_path:
            self.file_path.setText(folder_path)
            self._update_analyze_button_state()
    
    def handle_dropped_path_analyze(self, dropped_path):
        """Handle dropped file or folder path for analyze masks tab"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        # Update button states
        self._update_analyze_button_state()
    
    def select_fi_file(self):
        """Open file dialog to select an FI channel TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select FI Channel TIFF File", get_default_browse_dir(), "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.fi_file_path.setText(file_path)
    
    def handle_dropped_fi_file(self, dropped_path):
        """Handle dropped FI file for analyze masks tab"""
        path_obj = Path(dropped_path)
        # Check if it's a valid TIFF file
        if path_obj.is_file() and path_obj.suffix.lower() in ['.tif', '.tiff']:
            # Enable the analyze_fi checkbox if not already enabled
            if not self.analyze_fi.isChecked():
                self.analyze_fi.setChecked(True)
            # Set the file path
            self.fi_file_path.setText(dropped_path)
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid TIFF file (.tif or .tiff).")
    
    def _update_fi_file_state(self, checked):
        """Enable/disable FI file selection and contour thickness based on checkbox state"""
        self.fi_file_path.setEnabled(checked)
        if hasattr(self, 'fi_file_btn'):
            self.fi_file_btn.setEnabled(checked)
        if hasattr(self, 'analyze_fi_contour'):
            self.analyze_fi_contour.setEnabled(checked)
            # When FI is disabled, force contour FI off to avoid a confusing checked state
            if not checked:
                self.analyze_fi_contour.setChecked(False)
            # When FI is enabled, default contour FI to ON only if user hasn't chosen yet
            elif checked and not self.analyze_fi_contour.isChecked() and not getattr(self, "_fi_contour_user_set", False):
                self.analyze_fi_contour.setChecked(True)
        self._update_fi_contour_state()

    def _update_fi_contour_state(self, *_args):
        """Enable/disable contour thickness based on Analyze FI + Contour FI toggles."""
        # Track explicit user interaction so we don't re-force the default later
        if getattr(self, "analyze_fi_contour", None) is not None:
            self._fi_contour_user_set = True
        is_fi_enabled = bool(getattr(self, "analyze_fi", None) and self.analyze_fi.isChecked())
        is_contour_enabled = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
        if hasattr(self, 'fi_contour_thickness'):
            self.fi_contour_thickness.setEnabled(is_fi_enabled and is_contour_enabled)
    
    def _is_folder_mode(self):
        """Check if current path is a folder"""
        if not self.file_path.text():
            return False
        path = Path(self.file_path.text())
        return path.exists() and path.is_dir()
    
    def _browse_combine_folder(self):
        """Open folder dialog for 'Combine summaries from folder' path."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Analysis Excel Files", get_default_browse_dir()
        )
        if folder_path:
            self.combine_folder_path.setText(folder_path)
    
    def _build_combined_summary_excel(self, excel_paths, output_path):
        """Build long + wide combined summary from a list of analysis Excel paths. Returns True on success."""
        import pandas as pd
        if not excel_paths:
            return False
        frames_long = []
        for p in excel_paths:
            p = Path(p)
            if not p.exists():
                continue
            try:
                df = pd.read_excel(p, sheet_name="Summary")
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df["File"] = p.stem.replace("_analysis", "").replace("_masks_analysis", "")
            frames_long.append(df)
        if not frames_long:
            return False
        long_df = pd.concat(frames_long, ignore_index=True)
        # Ensure column order: File, Frame, then rest
        cols = [c for c in ["File", "Frame"] if c in long_df.columns]
        cols += [c for c in long_df.columns if c not in cols]
        long_df = long_df[cols]
        # Wide format: one row per frame, one set of columns per file
        wide_df = None
        try:
            frames = sorted(long_df["Frame"].dropna().unique().astype(int).tolist())
            wide_data = {"Frame": frames}
            for f in long_df["File"].unique().tolist():
                sub = long_df[long_df["File"] == f].set_index("Frame")
                for col in ["Total Cell Area (µm²)", "Total Cell Area (Relative)", "Number of Cells"]:
                    if col in sub.columns:
                        wide_data[f"{f}_{col}"] = sub.reindex(frames)[col].values
            wide_df = pd.DataFrame(wide_data)
        except Exception:
            wide_df = None
        if wide_df is None or wide_df.empty:
            return False
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                wide_df.to_excel(writer, sheet_name="Combined Summary (wide)", index=False)
        except Exception:
            return False
        return True
    
    def _combine_summaries_from_folder(self):
        """Combine Summary sheets from all analysis Excel files in the folder given by combine_folder_path."""
        path_text = self.combine_folder_path.text().strip()
        if not path_text:
            QMessageBox.warning(self, "No Folder", "Please select a folder (Browse...) that contains analysis Excel files.")
            return
        folder = Path(path_text)
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "Invalid Folder", "The selected path is not an existing folder.")
            return
        # Search in folder and all subfolders (one Excel per subfolder layout)
        excel_paths = list(folder.rglob("*_analysis.xlsx")) + list(folder.rglob("*_masks_analysis.xlsx"))
        excel_paths = sorted(set(excel_paths), key=lambda p: (str(p).lower(), p.name.lower()))
        if not excel_paths:
            QMessageBox.warning(self, "No Analysis Files", 
                "No *_analysis.xlsx or *_masks_analysis.xlsx files found in the selected folder or its subfolders.")
            return
        out_path = folder / f"{folder.name}_combined_summary.xlsx"
        if self._build_combined_summary_excel([str(p) for p in excel_paths], str(out_path)):
            self.parent_window.log_output(f"Combined summary saved to: {out_path}\n")
            QMessageBox.information(self, "Combine Complete", 
                f"Combined {len(excel_paths)} Summary sheet(s) into:\n{out_path.name}\n\nSheet: 'Combined Summary (wide)'.")
        else:
            QMessageBox.warning(self, "Combine Failed", "Could not read any Summary sheets from the Excel files in the selected folder.")
    
    def _on_pixel_size_changed(self):
        """Hide warning when user manually changes pixel size (they've verified it)"""
        if not self.pixel_size_metadata_loaded:
            # User manually changed the value, hide the warning
            self.pixel_size_indicator.setText("")
            self.pixel_size_indicator.setFixedWidth(0)
            self.pixel_size_indicator.hide()
    
    def reset_analysis_defaults(self):
        """Reset all analysis parameters to their default values"""
        # Pixel size: always reset to 0.108 (default value)
        default_pixel_size = float(0.108)
        self.pixel_size.setValue(default_pixel_size)
        self.pixel_size_metadata_loaded = False  # Reset metadata flag
        self.pixel_size_indicator.setText("⚠ Check value!")  # Show warning
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self.pixel_size_indicator.setFixedWidth(100)  # Keep width for warning
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size_indicator.setToolTip("Default value - verify this matches your microscope settings!\n\nIf you converted an ND2 file, pixel size should be auto-filled.\nOtherwise, check your microscope/objective settings.")
        self.pixel_size_indicator.show()
        
        # Morphological filters
        self.min_aspect_ratio.setValue(1.5)
        self.max_aspect_ratio.setValue(10.0)
        self.min_eccentricity.setValue(0.5)
        self.min_solidity.setValue(0.7)
        
        # Size filters
        self.min_area.setValue(0)
        self.max_area.setValue(0)
        self.min_width.setValue(0)
        self.max_width.setValue(0)
        self.min_length.setValue(0)
        
        # Options: "Include detailed per-cell data" stays checked by default (user must actively deselect)
        self.detailed.setChecked(True)
        self.create_vis.setChecked(True)
        if hasattr(self, "create_combined_summary"):
            self.create_combined_summary.setChecked(False)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        d = get_from_widgets([
            ("pixel_size", self.pixel_size),
            ("min_aspect_ratio", self.min_aspect_ratio),
            ("max_aspect_ratio", self.max_aspect_ratio),
            ("min_eccentricity", self.min_eccentricity),
            ("min_solidity", self.min_solidity),
            ("min_area", self.min_area),
            ("max_area", self.max_area),
            ("min_width", self.min_width),
            ("max_width", self.max_width),
            ("min_length", self.min_length),
            ("detailed", self.detailed),
            ("create_vis", self.create_vis),
            ("exclude_border", self.exclude_border_cb),
        ])
        if hasattr(self, "create_combined_summary"):
            d["create_combined_summary"] = self.create_combined_summary.isChecked()
        if hasattr(self, "analyze_fi"):
            d["analyze_fi"] = self.analyze_fi.isChecked()
        if hasattr(self, "analyze_fi_contour"):
            d["analyze_fi_contour"] = self.analyze_fi_contour.isChecked()
        if hasattr(self, "fi_contour_thickness"):
            d["fi_contour_thickness"] = self.fi_contour_thickness.value()
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only). Ignores unknown keys."""
        apply_to_widgets(data, [
            ("pixel_size", self.pixel_size),
            ("min_aspect_ratio", self.min_aspect_ratio),
            ("max_aspect_ratio", self.max_aspect_ratio),
            ("min_eccentricity", self.min_eccentricity),
            ("min_solidity", self.min_solidity),
            ("min_area", self.min_area),
            ("max_area", self.max_area),
            ("min_width", self.min_width),
            ("max_width", self.max_width),
            ("min_length", self.min_length),
            ("detailed", self.detailed),
            ("create_vis", self.create_vis),
            ("exclude_border", self.exclude_border_cb),
        ])
        if "create_combined_summary" in data and hasattr(self, "create_combined_summary"):
            self.create_combined_summary.setChecked(bool(data["create_combined_summary"]))
        if "analyze_fi" in data and hasattr(self, "analyze_fi"):
            self.analyze_fi.setChecked(bool(data["analyze_fi"]))
        if "analyze_fi_contour" in data and hasattr(self, "analyze_fi_contour"):
            self.analyze_fi_contour.setChecked(bool(data["analyze_fi_contour"]))
        if "fi_contour_thickness" in data and hasattr(self, "fi_contour_thickness"):
            try:
                self.fi_contour_thickness.setValue(int(data["fi_contour_thickness"]))
            except (TypeError, ValueError):
                pass
    
    def preview_filter_effects(self):
        """Preview how current filter parameters would affect mask segmentation"""
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a mask file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: show file selection dialog first
            folder = Path(self.file_path.text())
            # Look for mask files - use set to avoid duplicates
            mask_files_set = set()
            
            # First, collect all files with _masks in the name (preferred)
            mask_files_set.update(folder.glob("*_masks.tif"))
            mask_files_set.update(folder.glob("*_masks.tiff"))
            mask_files_set.update(folder.glob("*_masks.TIF"))
            mask_files_set.update(folder.glob("*_masks.TIFF"))
            
            # Then, collect other .tif files that don't have _masks in the name
            all_tif_files = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                           set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out files that already have _masks in name and filter visualization files
            for f in all_tif_files:
                if "_masks" not in f.name.lower() and "_filter_visualization" not in f.name.lower():
                    mask_files_set.add(f)
            
            # Convert to sorted list
            mask_files = sorted(mask_files_set, key=lambda x: x.name.lower())
            
            if not mask_files:
                QMessageBox.warning(self, "No Mask Files Found", 
                                  "No mask files found in the selected folder.\n\n"
                                  "Please select a folder containing mask TIFF files.")
                return
            
            # Let user choose which file to preview - use custom dialog
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select Mask File for Preview")
            file_dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Label
            label = QLabel("Select a mask file to preview filter effects:")
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
            for f in mask_files:
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
            for f in mask_files:
                if f.name == selected_name:
                    selected_file = f
                    break
            
            if not selected_file:
                return
            
            # Show preview dialog for selected file
            self._show_filter_preview(str(selected_file))
        else:
            # Single file mode: show preview directly
            self._show_filter_preview(self.file_path.text())
    
    def _show_filter_preview(self, mask_path):
        """Show preview dialog with filter effects"""
        # Show loading progress
        loading_progress = QProgressDialog("Loading mask file and initializing preview...", None, 0, 100, self)
        loading_progress.setWindowTitle("Loading")
        loading_progress.setAutoClose(False)
        loading_progress.setAutoReset(False)
        loading_progress.setCancelButton(None)
        loading_progress.setValue(0)
        loading_progress.show()
        QApplication.processEvents()
        
        try:
            loading_progress.setLabelText("Loading mask file... (20%)")
            loading_progress.setValue(20)
            QApplication.processEvents()
            
            import tifffile
            from skimage.measure import regionprops
            from skimage.morphology import remove_small_objects, binary_dilation
            from skimage.segmentation import find_boundaries

            # Load masks
            masks = safe_imread(mask_path)
            
            loading_progress.setLabelText("Processing mask data... (50%)")
            loading_progress.setValue(50)
            QApplication.processEvents()
            if masks.ndim == 2:
                masks = masks[np.newaxis, :, :]
            num_frames = masks.shape[0]
            
            # Load FI images if FI analysis is enabled
            fi_images = None
            fi_contour_thickness = 1
            do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
            if self.analyze_fi.isChecked() and self.fi_file_path.text():
                try:
                    fi_path = Path(self.fi_file_path.text())
                    if fi_path.exists():
                        loading_progress.setLabelText("Loading FI images... (60%)")
                        loading_progress.setValue(60)
                        QApplication.processEvents()
                        fi_images = safe_imread(str(fi_path))
                        # Handle different dimensions (same logic as in analyze_cell_areas.py)
                        if fi_images.ndim == 2:
                            fi_images = fi_images[np.newaxis, :, :]
                        elif fi_images.ndim == 3 and fi_images.shape[0] != num_frames:
                            # Might be (C, Y, X) or (Y, X, C) - try to handle
                            if fi_images.shape[2] < 10:  # Likely (Y, X, C)
                                fi_images = fi_images[np.newaxis, :, :, 0]  # Take first channel
                            else:  # Likely (C, Y, X)
                                fi_images = fi_images[0, np.newaxis, :, :]  # Take first channel
                        # Ensure frame count matches
                        if fi_images.shape[0] != num_frames:
                            QMessageBox.warning(preview_dialog, "Frame Mismatch", 
                                              f"FI image has {fi_images.shape[0]} frames but mask has {num_frames} frames.")
                            fi_images = None
                        if do_fi_contour:
                            fi_contour_thickness = self.fi_contour_thickness.value()
                except Exception as e:
                    QMessageBox.warning(preview_dialog, "FI Load Error", 
                                      f"Could not load FI image:\n{str(e)}")
                    fi_images = None
            
            loading_progress.setLabelText("Initializing preview dialog... (80%)")
            loading_progress.setValue(80)
            QApplication.processEvents()
            
            # Create preview dialog (non-modal so user can access main window)
            loading_progress.setLabelText("Ready! (100%)")
            loading_progress.setValue(100)
            QApplication.processEvents()
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle("Filter Effects Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)  # Make non-modal so main window is accessible
            
            layout = QVBoxLayout()

            # Title (same style as Segmentation Preview)
            title_label = QLabel(f"File: {Path(mask_path).name}")
            title_label.setStyleSheet("font-weight: bold; color: #ddd;")
            layout.addWidget(title_label)

            # Qt-native image canvas (same as Segmentation Preview: zoom wheel, pan left/middle-drag)
            image_canvas = ImageCanvas()
            image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
            image_canvas.setMinimumSize(400, 300)
            image_canvas.setStyleSheet("background-color: #1e1e1e;")
            image_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(image_canvas, 1)

            # Frame navigation (same layout as Segmentation Preview)
            nav_layout = QHBoxLayout()
            nav_layout.addWidget(QLabel("Frame:"))
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_frames)
            frame_spin.setValue(1)
            nav_layout.addWidget(frame_spin)
            nav_layout.addWidget(QLabel(f"of {num_frames}"))
            nav_layout.addStretch()
            
            # Info label will be added to nav_layout after reapply button
            info_label = QLabel("")
            info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            
            # Import filter functions from analyze_cell_areas
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.analyze_cell_areas import calculate_cell_properties, filter_rod_shaped_cells, filter_by_size
            
            # Store current frame
            preview_dialog.current_frame = 0
            
            # Info label (will be added to nav_layout after reapply button)
            info_label = QLabel("")
            info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            
            def get_current_parameters():
                """Get current filter parameters from main window (works because dialog is non-modal)"""
                return {
                    'pixel_size': self.pixel_size.value(),
                    'min_aspect_ratio': self.min_aspect_ratio.value(),
                    'max_aspect_ratio': self.max_aspect_ratio.value(),
                    'min_eccentricity': self.min_eccentricity.value(),
                    'min_solidity': self.min_solidity.value(),
                    'min_area': self.min_area.value() if self.min_area.value() > 0 else None,
                    'max_area': self.max_area.value() if self.max_area.value() > 0 else None,
                    'min_width': self.min_width.value() if self.min_width.value() > 0 else None,
                    'max_width': self.max_width.value() if self.max_width.value() > 0 else None,
                    'exclude_border': self.exclude_border_cb.isChecked(),
                    'fi_contour_thickness': self.fi_contour_thickness.value() if hasattr(self, 'fi_contour_thickness') else 1,
                }
            
            # Track if we're currently showing the warning dialog to prevent duplicates
            preview_dialog._showing_warning = False
            
            # Function to update preview
            def update_preview(frame_idx, show_progress=True):
                if frame_idx < 0 or frame_idx >= num_frames:
                    return
                
                # Prevent multiple simultaneous calls
                if hasattr(preview_dialog, '_updating_preview') and preview_dialog._updating_preview:
                    return
                preview_dialog._updating_preview = True
                
                # Get current FI contour thickness (may have changed)
                nonlocal fi_contour_thickness
                do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                if do_fi_contour and hasattr(self, 'fi_contour_thickness'):
                    fi_contour_thickness = self.fi_contour_thickness.value()
                
                # Reload FI images if FI file path changed or wasn't loaded before
                nonlocal fi_images
                if self.analyze_fi.isChecked() and self.fi_file_path.text():
                    fi_path = Path(self.fi_file_path.text())
                    if fi_path.exists() and (fi_images is None or str(fi_path) != str(Path(mask_path).parent / "fi_temp")):
                        try:
                            fi_images = safe_imread(str(fi_path))
                            # Handle different dimensions (same logic as above)
                            if fi_images.ndim == 2:
                                fi_images = fi_images[np.newaxis, :, :]
                            elif fi_images.ndim == 3 and fi_images.shape[0] != num_frames:
                                if fi_images.shape[2] < 10:
                                    fi_images = fi_images[np.newaxis, :, :, 0]
                                else:
                                    fi_images = fi_images[0, np.newaxis, :, :]
                            if fi_images.shape[0] != num_frames:
                                fi_images = None
                        except Exception:
                            fi_images = None
                elif not self.analyze_fi.isChecked():
                    fi_images = None
                
                # Show progress dialog
                progress_dialog = None
                cancelled = False
                if show_progress:
                    progress_dialog = QProgressDialog("Applying filters...", "Cancel", 0, 100, preview_dialog)
                    progress_dialog.setWindowTitle("Processing")
                    progress_dialog.setAutoClose(False)
                    progress_dialog.setAutoReset(False)
                    progress_dialog.setValue(0)
                    progress_dialog.show()
                    QApplication.processEvents()
                    
                    def check_cancelled():
                        nonlocal cancelled
                        if progress_dialog.wasCanceled():
                            cancelled = True
                    progress_dialog.canceled.connect(check_cancelled)
                
                # Disable controls during processing
                if progress_dialog:
                    reapply_btn.setEnabled(False)
                    frame_spin.setEnabled(False)
                
                try:
                    preview_dialog.current_frame = frame_idx
                    mask = masks[frame_idx]
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Cleaning mask... (10%)")
                        progress_dialog.setValue(10)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    # Remove small objects (more aggressive pre-filtering for performance)
                    mask_cleaned = remove_small_objects(mask, min_size=10)
                    
                    # Check number of unique labels (excluding background)
                    unique_labels = np.unique(mask_cleaned)
                    num_objects = len(unique_labels) - (1 if 0 in unique_labels else 0)
                    
                    # Warn if too many objects (performance/memory concern)
                    MAX_OBJECTS_WARNING = 5000
                    MAX_OBJECTS_VIZ = 10000  # Skip visualization if more than this
                    skip_visualization = False
                    
                    if num_objects > MAX_OBJECTS_WARNING:
                        # Prevent showing multiple warning dialogs simultaneously
                        if hasattr(preview_dialog, '_showing_warning') and preview_dialog._showing_warning:
                            # Already showing a warning, skip this one
                            skip_visualization = True
                        else:
                            if progress_dialog:
                                progress_dialog.close()
                            # Mark that we're showing a warning
                            preview_dialog._showing_warning = True
                            
                            # Use modal message box to prevent multiple instances
                            msg = QMessageBox(preview_dialog)
                            msg.setWindowTitle("Many Objects Detected")
                            msg.setModal(True)  # Make it modal to prevent multiple instances
                            msg.setText(f"This frame contains {num_objects:,} objects.")
                            msg.setInformativeText(
                                f"Processing may take a long time and could use significant memory.\n\n"
                                f"Options:\n"
                                f"• 'Yes' = Continue with full processing (may be slow)\n"
                                f"• 'No' = Skip visualization, only show statistics (faster)\n"
                                f"• 'Cancel' = Skip this frame entirely"
                            )
                            yes_btn = msg.addButton("Yes", QMessageBox.ButtonRole.YesRole)
                            no_btn = msg.addButton("No (Stats Only)", QMessageBox.ButtonRole.NoRole)
                            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                            msg.setDefaultButton(no_btn)
                            msg.exec()
                            
                            # Clear the flag after dialog is closed
                            preview_dialog._showing_warning = False
                        
                        if msg.clickedButton() == cancel_btn:
                            # Re-enable controls
                            reapply_btn.setEnabled(True)
                            frame_spin.setEnabled(True)
                            return
                        elif msg.clickedButton() == no_btn:
                            skip_visualization = True
                        
                        # Recreate progress dialog if user continues
                        if show_progress:
                            progress_dialog = QProgressDialog("Applying filters...", "Cancel", 0, 100, preview_dialog)
                            progress_dialog.setWindowTitle("Processing")
                            progress_dialog.setAutoClose(False)
                            progress_dialog.setAutoReset(False)
                            progress_dialog.setValue(10)
                            progress_dialog.show()
                            progress_dialog.canceled.connect(check_cancelled)
                            QApplication.processEvents()
                    
                    # Auto-skip visualization for extremely large datasets
                    if num_objects > MAX_OBJECTS_VIZ:
                        skip_visualization = True
                    
                    # Get current parameters from main window
                    params = get_current_parameters()
                    
                    if progress_dialog:
                        progress_dialog.setLabelText(f"Calculating cell properties... ({num_objects:,} objects) (30%)")
                        progress_dialog.setValue(30)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    # All non-zero labels in mask (for visualization: show every cell as accepted or rejected)
                    all_labels = set(np.unique(mask_cleaned)) - {0}
                    # Calculate properties; when exclude_border=True, border-touching cells are omitted from properties
                    properties = calculate_cell_properties(
                        mask_cleaned,
                        pixel_size=params['pixel_size'],
                        exclude_border=params.get('exclude_border', False),
                    )
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Applying morphological filters... (50%)")
                        progress_dialog.setValue(50)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    # Apply filters with current parameters
                    rod_shaped = filter_rod_shaped_cells(
                        properties,
                        min_aspect_ratio=params['min_aspect_ratio'],
                        max_aspect_ratio=params['max_aspect_ratio'],
                        min_eccentricity=params['min_eccentricity'],
                        min_solidity=params['min_solidity']
                    )
                    rod_shaped_labels = {cell['label'] for cell in rod_shaped}
                    
                    if progress_dialog:
                        progress_dialog.setLabelText("Applying size filters... (70%)")
                        progress_dialog.setValue(70)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    filtered = filter_by_size(
                        rod_shaped,
                        min_area_um2=params['min_area'],
                        max_area_um2=params['max_area'],
                        min_width_um=params['min_width'],
                        max_width_um=params['max_width']
                    )
                    accepted_labels = {cell['label'] for cell in filtered}
                    rejected_labels = all_labels - accepted_labels
                    
                    if progress_dialog:
                        if skip_visualization:
                            progress_dialog.setLabelText("Skipping visualization (too many objects)... (85%)")
                        else:
                            progress_dialog.setLabelText("Creating visualization... (85%)")
                        progress_dialog.setValue(85)
                        QApplication.processEvents()
                        if cancelled:
                            return
                    
                    # Create visualization using vectorized numpy operations (much faster)
                    # Skip visualization for very large datasets to prevent crashes
                    if skip_visualization:
                        # Show statistics only
                        fig.clear()
                        ax = fig.add_subplot(111)
                        ax.text(0.5, 0.5, 
                               f"Visualization skipped due to large number of objects ({num_objects:,})\n\n"
                               f"Statistics:\n"
                               f"• Total objects: {len(all_labels):,}\n"
                               f"• Accepted: {len(accepted_labels):,}\n"
                               f"• Rejected: {len(rejected_labels):,}\n"
                               f"• Acceptance rate: {len(accepted_labels)/len(all_labels)*100:.1f}%",
                               ha='center', va='center', fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}: Statistics Only (Visualization Skipped)")
                        ax.axis('off')
                        canvas.draw()
                    else:
                        height, width = mask.shape
                        vis = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        # Convert label sets to arrays for vectorized operations
                        if accepted_labels:
                            accepted_array = np.array(list(accepted_labels))
                            # Use numpy.isin for fast membership testing
                            accepted_mask = np.isin(mask, accepted_array)
                            vis[accepted_mask, 1] = 255  # Green channel
                        
                        if rejected_labels:
                            rejected_array = np.array(list(rejected_labels))
                            rejected_mask = np.isin(mask, rejected_array)
                            vis[rejected_mask, 0] = 255  # Red channel
                        
                        # Add FI contours if enabled: show the contour *region* (pixels used for contour FI)
                        # so the user can see which area is averaged and judge vs mean intensity.
                        do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                        if fi_images is not None and self.analyze_fi.isChecked() and do_fi_contour:
                            try:
                                # Get FI image for this frame
                                fi_image = fi_images[frame_idx]
                                # Handle multi-channel FI images
                                if fi_image.ndim > 2:
                                    fi_image = fi_image[:, :, 0] if fi_image.shape[2] < 10 else fi_image[0, :, :]
                                
                                # Create mask with only accepted cells
                                accepted_mask_full = np.zeros_like(mask, dtype=bool)
                                if accepted_labels:
                                    accepted_array = np.array(list(accepted_labels))
                                    accepted_mask_full = np.isin(mask, accepted_array)
                                
                                # Find contour *region* (same logic as analyze_cell_areas: inner boundary, then dilate within cell)
                                if np.any(accepted_mask_full):
                                    temp_mask = np.zeros_like(mask)
                                    temp_mask[accepted_mask_full] = mask[accepted_mask_full]
                                    boundaries = find_boundaries(temp_mask, mode='inner')
                                    # Restrict to accepted cells so contour stays inside cells (matches script)
                                    contour_mask = boundaries & accepted_mask_full
                                    if fi_contour_thickness > 1:
                                        for _ in range(fi_contour_thickness - 1):
                                            contour_mask = binary_dilation(contour_mask) & accepted_mask_full
                                    
                                    # Use FI image as background so user sees actual intensity in contour region
                                    fi_norm = fi_image.astype(np.float32)
                                    if fi_norm.max() > fi_norm.min():
                                        fi_norm = (fi_norm - fi_norm.min()) / (fi_norm.max() - fi_norm.min()) * 255
                                    else:
                                        fi_norm = np.zeros_like(fi_norm)
                                    vis_fi = np.stack([fi_norm, fi_norm, fi_norm], axis=-1).astype(np.uint8)
                                    # Overlay accepted cells in light green, rejected in light red (so we keep context)
                                    vis_fi[accepted_mask_full, 0] = np.clip(vis_fi[accepted_mask_full, 0].astype(np.int32) - 40, 0, 255).astype(np.uint8)
                                    vis_fi[accepted_mask_full, 1] = 255
                                    vis_fi[accepted_mask_full, 2] = np.clip(vis_fi[accepted_mask_full, 2].astype(np.int32) - 40, 0, 255).astype(np.uint8)
                                    if rejected_labels:
                                        rejected_array = np.array(list(rejected_labels))
                                        rejected_mask = np.isin(mask, rejected_array)
                                        vis_fi[rejected_mask, 0] = 255
                                        vis_fi[rejected_mask, 1] = np.clip(vis_fi[rejected_mask, 1].astype(np.int32) - 40, 0, 255).astype(np.uint8)
                                        vis_fi[rejected_mask, 2] = np.clip(vis_fi[rejected_mask, 2].astype(np.int32) - 40, 0, 255).astype(np.uint8)
                                    # Contour region: magenta overlay so it's clearly distinct from green (accepted)
                                    # and easy to see which pixels are used for contour FI
                                    alpha = 0.6
                                    vis_fi[contour_mask, 0] = (vis_fi[contour_mask, 0].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                                    vis_fi[contour_mask, 1] = (vis_fi[contour_mask, 1].astype(np.float32) * (1 - alpha) + 0 * alpha).astype(np.uint8)
                                    vis_fi[contour_mask, 2] = (vis_fi[contour_mask, 2].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                                    # Replace vis with FI-based view when contour is on (so contour area and intensity are visible)
                                    vis[:] = vis_fi[:]
                            except Exception as e:
                                # If contour drawing fails, continue without contours
                                print(f"Warning: Could not draw FI contours: {str(e)}")
                        
                        if progress_dialog:
                            progress_dialog.setLabelText("Rendering display... (100%)")
                            progress_dialog.setValue(100)
                            QApplication.processEvents()
                            if cancelled:
                                return

                        # Update Qt-native canvas (same pipeline as Segmentation Preview)
                        do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                        title = f"Frame {frame_idx + 1}/{num_frames}"
                        legend = f"Green=Accepted ({len(accepted_labels)})  Red=Rejected ({len(rejected_labels)})"
                        if fi_images is not None and self.analyze_fi.isChecked() and do_fi_contour:
                            legend += f"  Magenta=Contour (t={fi_contour_thickness})"
                        title_label.setText(f"{Path(mask_path).name}  |  {title}  |  {legend}")
                        # vis is (H, W, 3) uint8; scale for large images then display
                        dummy_mask = np.zeros(vis.shape[:2], dtype=np.int32)
                        vis_scaled, _ = scale_for_preview(vis, dummy_mask)
                        pixmap = to_qpixmap(vis_scaled)
                        image_canvas.set_pixmap(pixmap, (vis_scaled.shape[1], vis_scaled.shape[0]), reset_view=False)
                    
                    info_label.setText(f"Accepted: {len(accepted_labels)} | Rejected: {len(rejected_labels)} | Total: {len(all_labels)}")
                    
                    # Close progress dialog
                    if progress_dialog:
                        progress_dialog.setValue(100)
                        QApplication.processEvents()
                        progress_dialog.close()
                    
                    # Re-enable controls
                    if progress_dialog:
                        reapply_btn.setEnabled(True)
                        frame_spin.setEnabled(True)
                    
                    # Clear the updating flag
                    preview_dialog._updating_preview = False
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                    # Re-enable controls on error
                    reapply_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    # Clear flags on error
                    if hasattr(preview_dialog, '_updating_preview'):
                        preview_dialog._updating_preview = False
                    if hasattr(preview_dialog, '_showing_warning'):
                        preview_dialog._showing_warning = False
                    QMessageBox.warning(preview_dialog, "Preview Error", f"Could not update preview:\n{str(e)}")
            
            def go_to_frame(idx):
                if 0 <= idx < num_frames:
                    if hasattr(frame_spin, '_change_timer'):
                        frame_spin._change_timer.stop()
                    preview_dialog.current_frame = idx
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(idx + 1)  # Display is 1-based, index is 0-based
                    frame_spin.blockSignals(False)
                    try:
                        # Only show progress if it's a new frame (not cached)
                        update_preview(idx, show_progress=False)  # Filter preview is usually fast
                        # Ensure spinbox still shows correct value after update_preview
                        if preview_dialog.current_frame == idx:
                            frame_spin.blockSignals(True)
                            frame_spin.setValue(idx + 1)
                            frame_spin.blockSignals(False)
                    except Exception as e:
                        QMessageBox.warning(preview_dialog, "Preview Error", 
                                          f"Could not display frame {idx + 1}:\n{str(e)}")
            
            # editingFinished + debounced valueChanged (300ms) for frame spinbox
            def on_frame_changed():
                if hasattr(frame_spin, '_change_timer'):
                    frame_spin._change_timer.stop()
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    if frame_num != preview_dialog.current_frame:
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
            
            # Re-apply filters button (moved to match segmentation preview layout)
            reapply_btn = QPushButton("Re-apply Filters with Current Parameters")
            reapply_btn.setToolTip(
                "Re-apply filters on the current frame using the current parameters from the main window.\n"
                "You can change filter parameters in the main window (it stays accessible) and click this button to see the effect."
            )
            def on_reapply():
                update_preview(preview_dialog.current_frame, show_progress=True)
            reapply_btn.clicked.connect(on_reapply)
            nav_layout.addWidget(reapply_btn)
            
            # Info label (moved after reapply button to match segmentation preview)
            nav_layout.addWidget(info_label)

            layout.addLayout(nav_layout)

            # Close loading progress dialog now that preview dialog is ready
            loading_progress.close()
            
            # Initial preview with progress
            update_preview(0, show_progress=True)
            
            close_shortcut = QShortcut(QKeySequence.StandardKey.Close, preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)
            
            preview_dialog.setLayout(layout)
            preview_dialog.show()  # Use show() instead of exec_() to make it non-modal
            
        except Exception as e:
            loading_progress.close()
            QMessageBox.critical(self, "Preview Error", f"Could not create preview:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def analyze_masks(self):
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a mask file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        script_dir = PROJECT_ROOT
        script_path = str(PROJECT_ROOT / "scripts" / "analyze_cell_areas.py")
        
        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: process all mask files in the folder
            folder = Path(self.file_path.text())
            # Look for mask files - use set to avoid duplicates
            mask_files_set = set()
            
            # First, collect all files with _masks in the name (preferred)
            mask_files_set.update(folder.glob("*_masks.tif"))
            mask_files_set.update(folder.glob("*_masks.tiff"))
            mask_files_set.update(folder.glob("*_masks.TIF"))
            mask_files_set.update(folder.glob("*_masks.TIFF"))
            
            # Then, collect other .tif files that don't have _masks in the name
            all_tif_files = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                           set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out files that already have _masks in name and filter visualization files
            for f in all_tif_files:
                if "_masks" not in f.name.lower() and "_filter_visualization" not in f.name.lower():
                    mask_files_set.add(f)
            
            # Convert to sorted list
            mask_files = sorted(mask_files_set, key=lambda x: x.name.lower())
            
            if not mask_files:
                QMessageBox.warning(self, "No Mask Files Found", 
                                  "No mask files found in the selected folder.\n\n"
                                  "Please select a folder containing mask TIFF files.")
                return
            
            # Show file selection dialog
            selected_files = show_file_selection_dialog(
                self,
                mask_files,
                title="Select Mask Files for Analysis",
                message=f"Select mask files to analyze (all {len(mask_files)} files selected by default):"
            )
            
            if selected_files is None:
                # User cancelled or no files selected
                return
            
            if len(selected_files) == 0:
                QMessageBox.warning(self, "No Files Selected", 
                                  "Please select at least one file to analyze.")
                return
            
            # Process each file sequentially
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText(f"Analyzing... (0/{len(selected_files)})")
            self.parent_window.log_output(f"Starting mask analysis for {len(selected_files)} file(s) selected from folder: {folder}\n")
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, len(selected_files))
            self.parent_window.progress_bar.setValue(0)
            
            # Store files, current index, and output paths (for combined summary)
            self.folder_files = selected_files
            self.current_file_index = 0
            self.folder_results = []
            self.folder_output_paths = []
            
            # Process first file
            self._process_next_folder_file(script_path)
        else:
            # Single file mode
            input_file = Path(self.file_path.text())
            
            # Determine base directory (go up one level if input is in Tif_* or _segmentation folder)
            base_dir = input_file.parent
            if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
                base_dir = base_dir.parent
            
            # Create output path at base level with _analysis suffix
            output_path = base_dir / f"{input_file.stem}_analysis.xlsx"
            
            # Build command arguments
            cmd_args = [
                self.file_path.text(),
                '--output', str(output_path),
                '--pixel-size', str(self.pixel_size.value()),
                '--min-aspect-ratio', str(self.min_aspect_ratio.value()),
                '--max-aspect-ratio', str(self.max_aspect_ratio.value()),
                '--min-eccentricity', str(self.min_eccentricity.value()),
                '--min-solidity', str(self.min_solidity.value()),
            ]
            
            if self.min_area.value() > 0:
                cmd_args.extend(['--min-area', str(self.min_area.value())])
            if self.max_area.value() > 0:
                cmd_args.extend(['--max-area', str(self.max_area.value())])
            if self.min_width.value() > 0:
                cmd_args.extend(['--min-width', str(self.min_width.value())])
            if self.max_width.value() > 0:
                cmd_args.extend(['--max-width', str(self.max_width.value())])
            if self.min_length.value() > 0:
                cmd_args.extend(['--min-length', str(self.min_length.value())])
            if self.detailed.isChecked():
                cmd_args.append('--detailed')
            if self.create_vis.isChecked():
                cmd_args.append('--visualization')
            if self.exclude_border_cb.isChecked():
                cmd_args.append('--exclude-border')
            
            # Add FI image path if FI analysis is enabled
            if self.analyze_fi.isChecked() and self.fi_file_path.text():
                fi_path = Path(self.fi_file_path.text())
                if fi_path.exists():
                    cmd_args.extend(['--fi-image', str(fi_path)])
                    do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                    if do_fi_contour:
                        fi_contour_thickness = self.fi_contour_thickness.value()
                        cmd_args.extend(['--fi-contour-thickness', str(fi_contour_thickness)])
                    else:
                        cmd_args.append('--no-fi-contour')
                else:
                    QMessageBox.warning(self, "FI File Not Found", 
                                      f"The selected FI file does not exist:\n{fi_path}")
                    return
            
            # Run in thread
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Analyzing...")
            self.parent_window.log_output(f"Starting mask analysis: {self.file_path.text()}\n")
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range
            self.parent_window.progress_bar.setValue(0)
            
            # Store reference to prevent garbage collection
            self.worker = WorkerThread(script_path, cmd_args, parent=self)
            self.worker.finished.connect(self.on_finished)
            self.worker.progress.connect(self._parse_analyze_progress)
            self.worker.start()
    
    def _parse_analyze_progress(self, message):
        """Parse progress messages from analyze script and update progress bar"""
        import re
        # Log the message
        self.parent_window.log_output(message)
        
        # Try to parse tqdm-style progress: "Creating visualization: 25%|"
        # Or percentage in message
        progress_match = re.search(r'(\d+)%', message)
        if progress_match:
            try:
                progress_pct = int(progress_match.group(1))
                # Clamp to 0-100
                progress_pct = max(0, min(100, progress_pct))
                self.parent_window.progress_bar.setValue(progress_pct)
            except:
                pass
        # Also check for "X/Y" patterns which might indicate progress
        elif 'Creating visualization' in message or 'Processing' in message:
            # If we see these messages, increment progress slightly
            current = self.parent_window.progress_bar.value()
            if current < 90:
                self.parent_window.progress_bar.setValue(min(90, current + 1))
    
    def _process_next_folder_file(self, script_path):
        """Process next file in folder mode"""
        if self.current_file_index >= len(self.folder_files):
            # All files processed
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Analyze Masks")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f"\n✓ Analysis complete for all {len(self.folder_files)} file(s)!\n")
            # Create combined summary if option was enabled and we have output paths
            if getattr(self, "create_combined_summary", None) and getattr(self, "create_combined_summary").isChecked():
                paths = getattr(self, "folder_output_paths", [])
                if paths and self.folder_files:
                    folder = self.folder_files[0].parent
                    out_path = folder / f"{folder.name}_combined_summary.xlsx"
                    if self._build_combined_summary_excel(paths, str(out_path)):
                        self.parent_window.log_output(f"Combined summary saved to: {out_path}\n")
                        QMessageBox.information(self, "Analysis Complete", 
                            f"Successfully analyzed {len(self.folder_files)} mask file(s).\n\n"
                            f"Combined summary saved to:\n{out_path.name}")
                        return
            QMessageBox.information(self, "Analysis Complete", 
                                  f"Successfully analyzed {len(self.folder_files)} mask file(s).")
            return
        
        mask_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f"\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {mask_file.name}\n")
        
        # Determine base directory (go up one level if input is in Tif_* or _segmentation folder)
        base_dir = mask_file.parent
        if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
            base_dir = base_dir.parent
        
        # Create output path at base level with _analysis suffix
        output_path = base_dir / f"{mask_file.stem}_analysis.xlsx"
        self.folder_output_paths.append(str(output_path))
        
        # Build command arguments
        cmd_args = [
            str(mask_file),
            '--output', str(output_path),
            '--pixel-size', str(self.pixel_size.value()),
            '--min-aspect-ratio', str(self.min_aspect_ratio.value()),
            '--max-aspect-ratio', str(self.max_aspect_ratio.value()),
            '--min-eccentricity', str(self.min_eccentricity.value()),
            '--min-solidity', str(self.min_solidity.value()),
        ]
        
        if self.min_area.value() > 0:
            cmd_args.extend(['--min-area', str(self.min_area.value())])
        if self.max_area.value() > 0:
            cmd_args.extend(['--max-area', str(self.max_area.value())])
        if self.min_width.value() > 0:
            cmd_args.extend(['--min-width', str(self.min_width.value())])
        if self.max_width.value() > 0:
            cmd_args.extend(['--max-width', str(self.max_width.value())])
        if self.min_length.value() > 0:
            cmd_args.extend(['--min-length', str(self.min_length.value())])
        if self.detailed.isChecked():
            cmd_args.append('--detailed')
        if self.create_vis.isChecked():
            cmd_args.append('--visualization')
        if self.exclude_border_cb.isChecked():
            cmd_args.append('--exclude-border')
        
        # Add FI image path if FI analysis is enabled
        if self.analyze_fi.isChecked() and self.fi_file_path.text():
            fi_path = Path(self.fi_file_path.text())
            if fi_path.exists():
                cmd_args.extend(['--fi-image', str(fi_path)])
                do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                if do_fi_contour:
                    fi_contour_thickness = self.fi_contour_thickness.value()
                    cmd_args.extend(['--fi-contour-thickness', str(fi_contour_thickness)])
                else:
                    cmd_args.append('--no-fi-contour')
            else:
                QMessageBox.warning(self, "FI File Not Found", 
                                  f"The selected FI file does not exist:\n{fi_path}")
                return
        
        self.parent_window.log_output(f"Output file: {output_path}\n")
        
        # Update progress (for folder mode, show file progress)
        file_progress = int((self.current_file_index / len(self.folder_files)) * 100)
        self.parent_window.progress_bar.setValue(file_progress)
        self.analyze_btn.setText(f"Analyzing... ({self.current_file_index + 1}/{len(self.folder_files)})")
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(lambda success, msg: self._on_folder_file_finished(success, msg, script_path))
        self.worker.progress.connect(self.parent_window.log_output)
        self.worker.start()
    
    def _on_folder_file_finished(self, success, message, script_path):
        """Handle completion of a single file in folder mode"""
        self.parent_window.log_output(message)
        if success:
            self.folder_results.append(True)
            self.parent_window.log_output(f"✓ File {self.current_file_index + 1} completed successfully\n")
        else:
            self.folder_results.append(False)
            self.parent_window.log_output(f"✗ File {self.current_file_index + 1} failed: {message}\n")
        
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        
        # Process next file
        self.current_file_index += 1
        self._process_next_folder_file(script_path)
    
    def on_finished(self, success, message):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Masks")
        self.parent_window.log_output(message)
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()  # Wait for thread to finish
            self.worker = None
        # Hide progress bar
        self.parent_window.progress_bar.setVisible(False)
        if success:
            self.parent_window.log_output("✓ Analysis completed successfully!\n")
        else:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{message}")


