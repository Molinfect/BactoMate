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
    QButtonGroup,
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
    QRadioButton,
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
from bactimate.tiff_metadata import read_calibration_from_path_or_original

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

        # Analyze FI — mode, optional single file button, and feedback (no path line edit)
        self._single_fi_path: Optional[Path] = None
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
            "• One FI file for all masks: use the same FI stack for every mask in the run.\n"
            "• Match FI by name: one FI file per mask, found by name next to each mask (e.g. stem_fi.tif).\n\n"
            "The analysis adds Mean FI intensity (and optionally Contour FI + FI ratio if 'Contour FI' is on).",
            parent_widget=self
        )
        fi_checkbox_layout.addWidget(self.analyze_fi)
        fi_checkbox_layout.addWidget(fi_info_icon)
        self.analyze_fi_contour = QCheckBox("Contour FI")
        self.analyze_fi_contour.setChecked(False)
        self.analyze_fi_contour.setEnabled(False)
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
            "Thickness of the contour region used for FI intensity calculation (1-10 pixels).",
            parent_widget=self
        )
        fi_checkbox_layout.addWidget(contour_thickness_label)
        fi_checkbox_layout.addWidget(contour_thickness_info_icon)
        self.fi_contour_thickness = QSpinBox()
        self.fi_contour_thickness.setRange(1, 10)
        self.fi_contour_thickness.setValue(1)
        self.fi_contour_thickness.setEnabled(False)
        self.fi_contour_thickness.setFixedWidth(80)
        fi_checkbox_layout.addWidget(self.fi_contour_thickness)
        fi_checkbox_layout.addStretch()
        fi_layout.addLayout(fi_checkbox_layout)

        # FI mode and content (visible when Analyze FI is on)
        self.fi_content = QWidget()
        fi_content_layout = QVBoxLayout(self.fi_content)
        fi_content_layout.setContentsMargins(0, 6, 0, 6)
        self.fi_mode_one_file = QRadioButton("One FI file for all masks")
        self.fi_mode_match_by_name = QRadioButton("Match FI channel(s) by name")
        self.fi_mode_match_by_name.setChecked(True)
        self.fi_mode_group = QButtonGroup(self)
        self.fi_mode_group.addButton(self.fi_mode_one_file)
        self.fi_mode_group.addButton(self.fi_mode_match_by_name)
        fi_mode_row = QHBoxLayout()
        fi_mode_row.addWidget(self.fi_mode_one_file)
        fi_mode_row.addWidget(self.fi_mode_match_by_name)
        self.fi_naming_hint = QLabel('(Match "_fi" and "_foci" files with the same stem name to the mask, in the mask folder or its parent)')
        self.fi_naming_hint.setStyleSheet("color: #888; font-size: 9pt;")
        self.fi_naming_hint.setWordWrap(False)
        fi_mode_row.addWidget(self.fi_naming_hint)
        fi_mode_row.addStretch()
        fi_content_layout.addLayout(fi_mode_row)
        self.fi_select_btn = QPushButton("Select FI file…")
        self.fi_select_btn.setFixedWidth(140)
        self.fi_select_btn.clicked.connect(self._select_fi_file)
        fi_content_layout.addWidget(self.fi_select_btn)
        self.fi_feedback = QTextEdit()
        self.fi_feedback.setReadOnly(True)
        self.fi_feedback.setMaximumHeight(140)
        self.fi_feedback.setPlaceholderText("Set mask path and choose FI mode to see what will be analyzed.")
        self.fi_feedback.setStyleSheet(
            "font-size: 10pt; color: #b0b0b0; background-color: #2b2b2b; border: 1px solid #555; border-radius: 4px; padding: 6px;"
            " QTextEdit QScrollBar:vertical { width: 10px; background: #5a5a5a; border-radius: 5px; margin: 0; }"
            " QTextEdit QScrollBar::handle:vertical { min-height: 24px; background: #404040; border-radius: 5px; }"
            " QTextEdit QScrollBar::handle:vertical:hover { background: #505050; }"
            " QTextEdit QScrollBar::add-line:vertical, QTextEdit QScrollBar::sub-line:vertical { height: 0; }"
        )
        fi_content_layout.addWidget(self.fi_feedback)
        # Same vertical gap as between Analyze FI checkbox and One FI file (fi_content top margin)
        fi_content_layout.addSpacing(12)
        # Foci analysis (run detection on FI channel, add foci count and Foci sheet)
        run_foci_row = QHBoxLayout()
        self.run_foci = QCheckBox("Run foci analysis")
        self.run_foci.setChecked(False)
        run_foci_row.addWidget(self.run_foci)
        run_foci_row.addWidget(create_info_icon(
            "Detect foci (bright spots) in the FI channel per frame; add foci count and Foci sheet to Excel.",
            parent_widget=self
        ))
        run_foci_row.addSpacing(12)
        self.foci_exclude_outside_cells = QCheckBox("Exclude foci outside / rejected cells")
        self.foci_exclude_outside_cells.setChecked(False)
        run_foci_row.addWidget(self.foci_exclude_outside_cells)
        run_foci_row.addWidget(create_info_icon(
            "When checked: only foci inside accepted cells (cells that passed size/width filters) are "
            "included in the Foci sheet and per-cell foci counts. Foci in the background and foci "
            "inside rejected cells are excluded.\n\n"
            "When unchecked: all detected foci are included (inside and outside cells).",
            parent_widget=self
        ))
        run_foci_row.addStretch()
        fi_content_layout.addLayout(run_foci_row)
        foci_params_row = QHBoxLayout()
        foci_params_row.addWidget(QLabel("Min relative intensity:"))
        foci_params_row.addWidget(create_info_icon(
            "Peaks whose intensity is below (median × this value) are excluded.\n\n"
            "The median is computed from the raw FI image (no preprocessing), so the threshold "
            "is in the same units as your image: threshold = raw median × min relative intensity. "
            "Example: raw median 982 and min rel 10 → only peaks with intensity ≥ 9820 are kept. "
            "The preview shows the raw median FI for the current frame so you can judge the setting.\n\n"
            "Per frame: median over all pixels in the raw FI image; peaks with raw intensity "
            "< median × this value are discarded. Peak detection still uses internal preprocessing "
            "for localization; only the min-rel filter uses raw values.",
            parent_widget=self
        ))
        self.foci_min_rel = QDoubleSpinBox()
        self.foci_min_rel.setRange(0.5, 20.0)
        self.foci_min_rel.setValue(2.0)
        self.foci_min_rel.setDecimals(1)
        foci_params_row.addWidget(self.foci_min_rel)
        foci_params_row.addWidget(QLabel("Min separation (px):"))
        foci_params_row.addWidget(create_info_icon(
            "Minimum distance in pixels between two detected foci.\n\n"
            "Peaks closer than this are merged (the stronger one is kept). "
            "Increases with this value to avoid multiple detections on the same spot. "
            "Use ~2 for dense foci; increase if you get duplicate detections.",
            parent_widget=self
        ))
        self.foci_min_separation = QSpinBox()
        self.foci_min_separation.setRange(1, 20)
        self.foci_min_separation.setValue(2)
        foci_params_row.addWidget(self.foci_min_separation)
        foci_params_row.addWidget(QLabel("Min size (px):"))
        foci_params_row.addWidget(create_info_icon(
            "Minimum focus size in pixels (connected region around the peak).\n\n"
            "For each detected peak, the size is measured as the number of connected pixels above "
            "a fraction of the peak intensity. Peaks with size below this are excluded. "
            "Use 1 to allow single-pixel foci; increase to filter out tiny noise.",
            parent_widget=self
        ))
        self.foci_min_size = QSpinBox()
        self.foci_min_size.setRange(1, 50)
        self.foci_min_size.setValue(1)
        foci_params_row.addWidget(self.foci_min_size)
        foci_params_row.addWidget(QLabel("Max size (px):"))
        foci_params_row.addWidget(create_info_icon(
            "Maximum focus size in pixels. Peaks with a larger connected region are excluded.\n\n"
            "Set to 0 for no maximum (all sizes allowed). Use a positive value to ignore very large "
            "blobs that are not true foci.",
            parent_widget=self
        ))
        self.foci_max_size = QSpinBox()
        self.foci_max_size.setRange(0, 500)
        self.foci_max_size.setValue(0)
        self.foci_max_size.setSpecialValueText("0")
        foci_params_row.addWidget(self.foci_max_size)
        foci_params_row.addStretch()
        fi_content_layout.addLayout(foci_params_row)
        self.foci_params_widgets = [self.foci_min_rel, self.foci_min_separation, self.foci_min_size, self.foci_max_size, self.foci_exclude_outside_cells]
        self.fi_content.setVisible(False)
        fi_layout.addWidget(self.fi_content)

        self.analyze_fi.toggled.connect(self._update_fi_ui_state)
        self.run_foci.toggled.connect(self._update_fi_ui_state)
        self.analyze_fi_contour.toggled.connect(self._update_fi_contour_state)
        self.fi_mode_one_file.toggled.connect(self._update_fi_ui_state)
        self.fi_mode_match_by_name.toggled.connect(self._update_fi_ui_state)
        self.file_path.textChanged.connect(self._update_fi_feedback)
        fi_group.setLayout(fi_layout)
        fi_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        scroll_layout.addWidget(fi_group)
        self._update_fi_ui_state()

        # Analysis Parameters — below Analyze FI (Morphological, Size, Options, Reset) with spacing between blocks
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_container_layout = QHBoxLayout()
        analysis_container_layout.setSpacing(12)
        analysis_container_layout.setContentsMargins(6, 6, 6, 6)

        # First box: Morphological filters
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
        analysis_container_layout.addSpacing(16)

        # Second box: Size filters
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
        analysis_container_layout.addSpacing(16)

        # Third box: Options (4 tick boxes stacked vertically)
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_layout.setSpacing(4)
        detailed_row = QHBoxLayout()
        self.detailed = QCheckBox("Detailed per-cell data")
        self.detailed.setChecked(True)
        detailed_info_icon = create_info_icon(
            "Include detailed per-cell measurements in the Excel output (sheet 'Per-Cell Data').\n\n"
            "When enabled, each row is one cell with:\n"
            "• Frame, Cell ID, Cell Label\n"
            "• Area (µm²), Length (µm), Width (µm)\n"
            "• Aspect ratio, Eccentricity, Solidity\n"
            "• If Analyze FI is used: Mean FI intensity, Contour FI intensity, FI ratio (Contour/Mean)\n\n"
            "When disabled: Only the Summary sheet (total area and cell count per frame) is included.",
            parent_widget=self
        )
        detailed_row.addWidget(self.detailed)
        detailed_row.addWidget(detailed_info_icon)
        detailed_row.addStretch()
        options_layout.addLayout(detailed_row)
        vis_row = QHBoxLayout()
        self.create_vis = QCheckBox("Create filter visualization")
        self.create_vis.setChecked(True)
        vis_info_icon = create_info_icon(
            "Create visualization TIFF (green = accepted, red = rejected). Saved as filename_filter_visualization.tif.",
            parent_widget=self
        )
        vis_row.addWidget(self.create_vis)
        vis_row.addWidget(vis_info_icon)
        vis_row.addStretch()
        options_layout.addLayout(vis_row)
        combined_summary_row = QHBoxLayout()
        self.create_combined_summary = QCheckBox("Combine summary")
        self.create_combined_summary.setChecked(False)
        combined_summary_info_icon = create_info_icon(
            "When analyzing a folder: create one extra Excel combining all Summary sheets ({folder_name}_combined_summary.xlsx).",
            parent_widget=self
        )
        combined_summary_row.addWidget(self.create_combined_summary)
        combined_summary_row.addWidget(combined_summary_info_icon)
        combined_summary_row.addStretch()
        options_layout.addLayout(combined_summary_row)
        exclude_border_row = QHBoxLayout()
        self.exclude_border_cb = QCheckBox("Exclude masks at borders")
        self.exclude_border_cb.setChecked(False)
        exclude_border_info_icon = create_info_icon(
            "Exclude mask regions touching the image border from counts and area.",
            parent_widget=self
        )
        exclude_border_row.addWidget(self.exclude_border_cb)
        exclude_border_row.addWidget(exclude_border_info_icon)
        exclude_border_row.addStretch()
        options_layout.addLayout(exclude_border_row)
        options_group.setLayout(options_layout)
        options_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        options_group.setContentsMargins(4, 6, 4, 6)
        analysis_container_layout.addWidget(options_group)
        
        # Reset to default — same blue as other BactiMate buttons (no custom grey style)
        reset_analysis_btn = QPushButton("Reset to Default")
        reset_analysis_btn.setToolTip("Reset all analysis parameters to their default values")
        reset_analysis_btn.clicked.connect(self.reset_analysis_defaults)
        analysis_container_layout.addWidget(reset_analysis_btn)
        
        analysis_group.setLayout(analysis_container_layout)
        scroll_layout.addWidget(analysis_group)

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
        
        # Connect file path changes to update button states and try to load metadata from TIFF
        self.file_path.textChanged.connect(self._update_analyze_button_state)
        self.file_path.textChanged.connect(lambda: self._update_fi_ui_state(None))
        self.file_path.textChanged.connect(self._try_fill_metadata_from_path)
        # Initialize state
        self._update_analyze_button_state()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_file_group(self) -> QWidget:
        """Build the mask file/folder input group: Path + info (same as other tabs), path line edit, browse buttons."""
        file_group = DragDropGroupBox(
            "Run analysis on masks",
            parent=self,
            on_drop_callback=self.handle_dropped_path_analyze
        )
        file_font = QFont()
        file_font.setPointSize(12)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Path:"))
        file_layout.addWidget(create_info_icon(
            "Path to a mask TIFF file or folder of mask TIFFs to analyze.\n\n"
            "Single file: one *_analysis.xlsx and optional filter visualization TIFF.\n"
            "Folder: one analysis Excel per mask file (and optional combined summary if enabled in Options).\n\n"
            "You can select the _segmentation_omni or _segmentation_unet folder even if it contains an overlays/ "
            "subfolder or *_filter_visualization.tif files — those are ignored; only *_masks.tif (and similar) are used.",
            self,
        ))
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a mask TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(200)
        self.file_path.setFont(file_font)
        self.file_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.file_path, 1)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        # Combine uses the same folder as Path (from Browse Folder); add info + button as last in block
        file_layout.addWidget(create_info_icon(
            "Combine: merge Summary sheets from analysis Excel files in the folder currently in Path.\n\n"
            "Use Browse Folder to select a folder that contains *_analysis.xlsx (or subfolders). "
            "Saves {folder_name}_combined_summary.xlsx in that folder with sheet 'Combined Summary (wide)'.",
            self,
        ))
        combine_btn = QPushButton("Combine")
        combine_btn.setFixedWidth(100)
        combine_btn.setToolTip("Combine Summary sheets from the folder in Path (select a folder with Browse Folder first).")
        combine_btn.clicked.connect(self._combine_summaries_from_folder)
        file_layout.addWidget(combine_btn, 0)
        file_group.setLayout(file_layout)
        return file_group
    
    def _update_analyze_button_state(self):
        """Update analyze and preview button enabled state based on whether file/folder is selected"""
        has_path = bool(self.file_path.text() and self.file_path.text().strip())
        if hasattr(self, 'analyze_btn'):
            self.analyze_btn.setEnabled(has_path)
        if hasattr(self, 'preview_btn'):
            self.preview_btn.setEnabled(has_path)

    def _try_fill_metadata_from_path(self):
        """When path is a TIFF file (or folder with a TIFF), read calibration from tiff_metadata and fill pixel size."""
        path_text = (self.file_path.text() or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        tiff_path = None
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            tiff_path = path
        elif path.is_dir():
            # Prefer mask/source TIFFs over trajectory TIFFs so calibration comes from acquisition metadata
            all_tifs = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
            for f in all_tifs:
                if not f.is_file():
                    continue
                if f.name.endswith("_trajectories.tif"):
                    continue
                tiff_path = f
                break
            else:
                tiff_path = next((f for f in all_tifs if f.is_file()), None)
        if tiff_path is None:
            return
        try:
            ps, tp = read_calibration_from_path_or_original(tiff_path)
            if ps > 0 or tp > 0:
                if ps > 0:
                    self.pixel_size.setValue(ps)
                    self.pixel_size_indicator.setText("(from file)")
                    self.pixel_size_indicator.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                    self.pixel_size_indicator.setFixedWidth(100)
                    self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.pixel_size_indicator.show()
                    self.pixel_size_metadata_loaded = True
                if hasattr(self, "parent_window") and getattr(self.parent_window, "shared_metadata", None) is not None:
                    if ps > 0:
                        self.parent_window.shared_metadata["pixel_size"] = ps
                    if tp > 0:
                        self.parent_window.shared_metadata["time_per_frame_min"] = tp
                    self.parent_window.update_all_tabs_with_metadata()
        except Exception:
            pass

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
    
    def _select_fi_file(self):
        """Open file dialog and set single FI file for all masks."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select FI Channel TIFF File", get_default_browse_dir(), "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self._single_fi_path = Path(file_path)
            self._update_fi_feedback()

    def handle_dropped_fi_file(self, dropped_path):
        """Handle dropped FI file: enable Analyze FI, switch to one-file mode, set path."""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in (".tif", ".tiff"):
            if not self.analyze_fi.isChecked():
                self.analyze_fi.setChecked(True)
            self.fi_mode_one_file.setChecked(True)
            self._single_fi_path = path_obj
            self._update_fi_feedback()
        else:
            QMessageBox.warning(self, "Invalid File",
                                "Please drop a valid TIFF file (.tif or .tiff).")

    def _is_match_by_name_mode(self) -> bool:
        """True when 'Match FI by name' is selected."""
        return bool(getattr(self, "fi_mode_match_by_name", None) and self.fi_mode_match_by_name.isChecked())

    def _get_mask_files_for_feedback(self) -> list:
        """Return list of mask paths (one for single file, many for folder) for feedback."""
        path_text = self.file_path.text().strip()
        if not path_text:
            return []
        path = Path(path_text)
        if not path.exists():
            return []
        if path.is_file():
            return [path.resolve()]
        folder = path
        mask_set = set()
        mask_set.update(folder.glob("*_masks.tif"))
        mask_set.update(folder.glob("*_masks.tiff"))
        mask_set.update(folder.glob("*_masks.TIF"))
        mask_set.update(folder.glob("*_masks.TIFF"))
        all_tif = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
        for f in all_tif:
            name_lower = f.name.lower()
            if "_masks" not in name_lower and "_filter_visualization" not in name_lower and "_overlay" not in name_lower:
                mask_set.add(f)
        return sorted(mask_set, key=lambda p: p.name.lower())

    def _update_fi_feedback(self) -> None:
        """Update the read-only feedback text to show what will be analyzed."""
        if not getattr(self, "fi_feedback", None):
            return
        if not self.analyze_fi.isChecked():
            self.fi_feedback.setPlainText("Analyze FI is off. Enable it and choose a mode to see FI pairings.")
            return
        if self._is_match_by_name_mode():
            masks = self._get_mask_files_for_feedback()
            if not masks:
                self.fi_feedback.setPlainText("Set mask path (file or folder) to see FI pairings.")
                return
            fi_lines = []
            total_fi_pairings = 0
            for mask_path in masks:
                fi_paths = self._resolve_fi_paths_for_mask(mask_path)
                n_fi = len(fi_paths)
                total_fi_pairings += n_fi
                if fi_paths:
                    fi_names = ", ".join(p.name for p in fi_paths)
                    fi_lines.append(f"• {mask_path.name} → {fi_names}")
                else:
                    fi_lines.append(f"• {mask_path.name} → (no FI found)")
            # Summary: one mask file = one segmentation, used once per FI channel (same mask, multiple analyses)
            if total_fi_pairings == 0:
                header = f"{len(masks)} mask file(s); no FI files matched.\n\n"
            elif total_fi_pairings == len(masks):
                header = f"{len(masks)} mask file(s). Same segmentation used for 1 FI channel each.\n\n"
            else:
                header = f"{len(masks)} mask file(s) → {total_fi_pairings} analyses (same segmentation per file, used once per FI channel below).\n\n"
            body = "Analyze FI (mask → FI channel(s)):\n" + "\n".join(fi_lines)
            run_foci_on = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
            if run_foci_on:
                foci_lines = []
                for mask_path in masks:
                    fi_paths = self._resolve_fi_paths_for_mask(mask_path)
                    foci_paths = [p for p in fi_paths if "_foci" in p.stem]
                    if foci_paths:
                        foci_names = ", ".join(p.name for p in foci_paths)
                        foci_lines.append(f"• {mask_path.name} → {foci_names}")
                    else:
                        foci_lines.append(f"• {mask_path.name} → (no _foci matched)")
                body += "\n\nRun foci (mask → _foci channel(s) used for foci detection):\n" + "\n".join(foci_lines)
            self.fi_feedback.setPlainText(header + body)
        else:
            if self._single_fi_path and self._single_fi_path.exists():
                self.fi_feedback.setPlainText(f"FI file: {self._single_fi_path.name} will be used for all masks.")
            else:
                self.fi_feedback.setPlainText("Select one FI file (button above) to use for all masks.")

    def _update_fi_ui_state(self, _checked=None) -> None:
        """Show/hide FI content, enable contour, show/hide Select button and naming hint.
        Always derive state from Analyze FI checkbox (not from the signal sender) so that
        toggling Run foci does not hide the FI section."""
        analyze_fi_on = bool(getattr(self, "analyze_fi", None) and self.analyze_fi.isChecked())
        if hasattr(self, "fi_content"):
            self.fi_content.setVisible(analyze_fi_on)
        one_file = bool(getattr(self, "fi_mode_one_file", None) and self.fi_mode_one_file.isChecked())
        if hasattr(self, "fi_select_btn"):
            self.fi_select_btn.setVisible(analyze_fi_on and one_file)
        if hasattr(self, "fi_naming_hint"):
            self.fi_naming_hint.setVisible(analyze_fi_on and self._is_match_by_name_mode())
        if hasattr(self, "analyze_fi_contour"):
            self.analyze_fi_contour.setEnabled(analyze_fi_on)
            if not analyze_fi_on:
                self.analyze_fi_contour.setChecked(False)
            elif analyze_fi_on and not self.analyze_fi_contour.isChecked() and not getattr(self, "_fi_contour_user_set", False):
                self.analyze_fi_contour.setChecked(True)
        run_foci_on = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
        if hasattr(self, "run_foci"):
            self.run_foci.setEnabled(analyze_fi_on)
            if not analyze_fi_on:
                self.run_foci.setChecked(False)
        for w in getattr(self, "foci_params_widgets", []):
            w.setEnabled(analyze_fi_on and run_foci_on)
        self._update_fi_contour_state()
        self._update_fi_feedback()

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

    def _resolve_fi_paths_for_mask(self, mask_path: Path) -> list[Path]:
        """For batch FI: find all FI files for this mask.
        Tries: stem_fi / stem_fi_* (e.g. EM13748_AnTC_003_Phase60X_fi.tif).
        If none found, tries base stem: base_fi / base_fi_* where base = stem without last segment
        (e.g. EM13748_AnTC_003_Phase60X -> base EM13748_AnTC_003 -> EM13748_AnTC_003_fi_GFP.tif).
        Search dir: mask's parent, or parent's parent if mask is in _segmentation*.
        """
        parent = mask_path.parent
        if parent.name.startswith("_segmentation"):
            search_dir = parent.parent
        else:
            search_dir = parent
        stem = mask_path.stem.replace("_masks", "")

        def matches_stem(name: str, s: str) -> bool:
            if name == f"{s}_fi":
                return True
            if name.startswith(f"{s}_fi_"):
                return True
            return False

        found: list[Path] = []
        for f in search_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                continue
            if matches_stem(f.stem, stem):
                found.append(f)
        if not found and "_" in stem:
            base = stem.rsplit("_", 1)[0]
            for f in search_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if matches_stem(f.stem, base):
                    found.append(f)
        return sorted(found, key=lambda p: p.name.lower())

    def _resolve_phase_path_for_mask(self, mask_path: Path) -> Path | None:
        """For batch FI: find the non-FI (phase/original) TIFF that matches this mask, if any.
        Looks for stem.tif or base.tif in the same search dir as FI, excluding any file with _fi in the name.
        """
        parent = mask_path.parent
        if parent.name.startswith("_segmentation"):
            search_dir = parent.parent
        else:
            search_dir = parent
        stem = mask_path.stem.replace("_masks", "")
        for f in search_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                continue
            if "_fi" in f.stem.lower():
                continue
            if f.stem == stem:
                return f
        if "_" in stem:
            base = stem.rsplit("_", 1)[0]
            for f in search_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if "_fi" in f.stem.lower():
                    continue
                if f.stem == base:
                    return f
        return None

    def _fi_output_suffix(self, fi_path: Path | None, stem: str) -> str:
        """Return suffix for analysis output filename: '' for single stem_fi, '_ChannelName' for stem_fi_ChannelName.tif.
        Also supports base stem (e.g. stem=EM13748_AnTC_003_Phase60X, fi=EM13748_AnTC_003_fi_GFP -> _GFP).
        """
        if fi_path is None:
            return ""
        name = fi_path.stem
        if name == f"{stem}_fi":
            return ""
        prefix = f"{stem}_fi_"
        if name.startswith(prefix):
            return "_" + name[len(prefix):]
        base = stem.rsplit("_", 1)[0] if "_" in stem else stem
        if name == f"{base}_fi":
            return "_fi"
        if name.startswith(f"{base}_fi_"):
            return "_" + name[len(base) + 4:]
        return "_fi"
    
    def _match_log_from_folder_tasks(self) -> tuple[int, int, list[str]]:
        """Build (n_masks, n_tif, lines) from folder_fi_tasks and folder_results for the Logs column."""
        tasks = getattr(self, "folder_fi_tasks", [])
        results = getattr(self, "folder_results", [])
        if len(results) != len(tasks):
            return 0, 0, []
        phase_per_mask = getattr(self, "folder_phase_per_mask", {})
        seen_stem: set[str] = set()
        lines: list[str] = []
        mask_stems: set[str] = set()
        n_tif = 0
        for i, (mask_file, fi_path, _out) in enumerate(tasks):
            if not results[i]:
                continue
            stem = mask_file.stem.replace("_masks", "")
            if fi_path is not None:
                tif_name = fi_path.name
                n_tif += 1
            else:
                phase_path = phase_per_mask.get(mask_file)
                tif_name = phase_path.name if phase_path else "—"
                if phase_path:
                    n_tif += 1
            mask_stems.add(stem)
            if stem not in seen_stem:
                seen_stem.add(stem)
                lines.append(f"{stem}_masks -> {tif_name}")
            else:
                lines.append(f"{stem}_masks with {tif_name}")
        return len(mask_stems), n_tif, lines

    def _build_combined_summary_excel(self, excel_paths, output_path, match_log=None):
        """Build long + wide combined summary from a list of analysis Excel paths. Returns True on success.
        If match_log is (n_masks, n_tif, lines), writes Logs in column G (G1=Logs, G2=count, G3+=lines).
        If output_path exists and already has Logs in G, appends the new lines to column G instead of overwriting.
        """
        import pandas as pd
        from openpyxl import load_workbook
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
        cols = [c for c in ["File", "Frame"] if c in long_df.columns]
        cols += [c for c in long_df.columns if c not in cols]
        long_df = long_df[cols]
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
        out_path = Path(output_path)
        existing_log_lines: list[str] = []
        if out_path.exists() and match_log is not None:
            try:
                wb = load_workbook(output_path, data_only=True)
                sheet_name = "Combined Summary (wide)"
                if sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    for r in range(2, min(ws.max_row + 1, 5000)):
                        val = ws.cell(row=r, column=7).value
                        if val is None or not str(val).strip():
                            continue
                        s = str(val).strip()
                        # Keep only pair lines (mask -> tif / mask with tif), not count lines
                        if " -> " in s or " with " in s:
                            existing_log_lines.append(s)
                wb.close()
            except Exception:
                pass
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                wide_df.to_excel(writer, sheet_name="Combined Summary (wide)", index=False)
            if match_log is not None:
                n_masks, n_tif, lines = match_log
                wb = load_workbook(output_path)
                ws = wb["Combined Summary (wide)"]
                ws["G1"] = "Logs"
                ws["G2"] = f"{n_masks} mask(s) and {n_tif} tif file(s) matched"
                # Append: existing pair lines (previous runs) then this run's pairs
                row = 3
                for line in existing_log_lines:
                    ws.cell(row=row, column=7, value=line)
                    row += 1
                for line in lines:
                    ws.cell(row=row, column=7, value=line)
                    row += 1
                wb.save(output_path)
                wb.close()
        except Exception:
            return False
        return True

    def _write_logs_to_individual_excel(self, output_path: Path, mask_file: Path, fi_path: Path | None, display_tif_name: str | None = None) -> None:
        """Write Logs to column G of the Summary sheet (G1=Logs, G2=mask -> tif). Use display_tif_name when fi_path is None (e.g. phase file name)."""
        if not output_path.exists():
            return
        try:
            from openpyxl import load_workbook
            wb = load_workbook(output_path)
            if "Summary" not in wb.sheetnames:
                wb.close()
                return
            ws = wb["Summary"]
            stem = mask_file.stem.replace("_masks", "")
            tif_name = fi_path.name if fi_path else (display_tif_name if display_tif_name else "—")
            ws["G1"] = "Logs"
            ws["G2"] = f"{stem}_masks -> {tif_name}"
            wb.save(output_path)
            wb.close()
        except Exception:
            pass

    def _merge_fi_channels_into_one_excel(self) -> None:
        """After folder analysis: one Excel per mask with all FI columns and Logs in Summary.
        - 1 FI channel: copy that file to stem_analysis.xlsx, add Logs, remove suffixed file.
        - 2+ FI channels: merge Per-Cell Data (morphology + FI columns), one Summary with Logs, remove per-channel files.
        """
        import shutil
        import pandas as pd
        tasks = getattr(self, "folder_fi_tasks", [])
        results = getattr(self, "folder_results", [])
        if len(results) != len(tasks):
            return
        path_replacements: dict[str, str] = {}  # per-channel path -> merged path (for folder_output_paths)
        by_stem: dict[str, list[tuple[Path, Path, str]]] = {}
        for i, (mask_file, fi_path, output_path) in enumerate(tasks):
            if not results[i] or not output_path.exists():
                continue
            stem = mask_file.stem.replace("_masks", "")
            if stem not in by_stem:
                by_stem[stem] = []
            channel_label = "FI"
            if fi_path is not None:
                suffix = self._fi_output_suffix(fi_path, stem)
                channel_label = (suffix[1:] if suffix.startswith("_") else "FI") or "FI"
            by_stem[stem].append((output_path, fi_path, channel_label))
        phase_per_mask = getattr(self, "folder_phase_per_mask", {})
        for stem, group in by_stem.items():
            phase_entries = [(p, fp, lbl) for p, fp, lbl in group if fp is None]
            fi_entries = [(p, fp, lbl) for p, fp, lbl in group if fp is not None]
            if not fi_entries:
                continue
            analysis_dir = group[0][0].parent
            merged_path = analysis_dir / f"{stem}_analysis.xlsx"
            if len(fi_entries) == 1 and not phase_entries:
                # Single FI: copy to stem_analysis.xlsx, add Logs, remove suffixed file
                single_path, single_fp, _ = fi_entries[0]
                if single_path != merged_path:
                    try:
                        shutil.copy2(single_path, merged_path)
                        self._write_logs_to_individual_excel(merged_path, Path(stem + "_masks.tif"), single_fp)
                        path_replacements[str(single_path)] = str(merged_path)
                        single_path.unlink()
                        self.parent_window.log_output(f"Single FI channel -> {merged_path.name}\n")
                    except Exception as e:
                        get_logger().debug("Single-FI consolidate failed: %s", e, exc_info=True)
                else:
                    try:
                        from openpyxl import load_workbook
                        wb = load_workbook(merged_path)
                        if "Summary" in wb.sheetnames:
                            ws = wb["Summary"]
                            ws["G1"] = "Logs"
                            ws["G2"] = f"{stem}_masks -> {single_fp.name}"
                            wb.save(merged_path)
                            wb.close()
                    except Exception:
                        pass
                continue
            # 2+ FI channels (or 1 phase + 1+ FI): merge into one Excel
            try:
                summary_df = None
                merged = None
                first_excel = phase_entries[0][0] if phase_entries else fi_entries[0][0]
                key_cols = ["Frame", "Cell_ID"] if "Cell_ID" in pd.read_excel(first_excel, sheet_name="Per-Cell Data", nrows=0).columns else ["Frame", "Cell Label"]
                fi_col_names = ["Mean FI Intensity", "Contour FI Intensity", "FI Ratio (Contour/Mean)"]
                if phase_entries:
                    phase_path = phase_entries[0][0]
                    df_phase = pd.read_excel(phase_path, sheet_name="Per-Cell Data")
                    summary_df = pd.read_excel(phase_path, sheet_name="Summary")
                    if df_phase is not None and not df_phase.empty:
                        merged = df_phase.copy()
                for idx, (excel_path, _fp, channel_label) in enumerate(fi_entries):
                    df = pd.read_excel(excel_path, sheet_name="Per-Cell Data")
                    if df is None or df.empty:
                        continue
                    if summary_df is None:
                        summary_df = pd.read_excel(excel_path, sheet_name="Summary")
                    has_fi = [c for c in fi_col_names if c in df.columns]
                    if not has_fi:
                        continue
                    rename = {c: f"{c} ({channel_label})" for c in has_fi}
                    sub = df.rename(columns=rename)
                    if merged is None:
                        merged = sub.copy()
                    else:
                        merge_on = [c for c in key_cols if c in merged.columns and c in sub.columns]
                        fi_new = [c for c in sub.columns if c not in merge_on and "FI" in c and c not in merged.columns]
                        if merge_on and fi_new:
                            merged = merged.merge(sub[merge_on + fi_new], on=merge_on, how="outer")
                if summary_df is not None and merged is not None and not merged.empty and "Frame" in merged.columns:
                    with pd.ExcelWriter(merged_path, engine="openpyxl") as writer:
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)
                        merged.to_excel(writer, sheet_name="Per-Cell Data", index=False)
                    n_matched = len(fi_entries) + (1 if phase_entries else 0)
                    phase_path_display = next(
                        (phase_per_mask[m].name for m in phase_per_mask if m.stem.replace("_masks", "") == stem and phase_per_mask.get(m)),
                        None,
                    )
                    try:
                        from openpyxl import load_workbook
                        wb = load_workbook(merged_path)
                        if "Summary" in wb.sheetnames:
                            ws = wb["Summary"]
                            ws["G1"] = "Logs"
                            ws["G2"] = f"1 mask(s) and {n_matched} tif file(s) matched"
                            row = 3
                            if phase_path_display:
                                ws.cell(row=row, column=7, value=f"{stem}_masks -> {phase_path_display}")
                                row += 1
                            for i, (_p, fp, _lbl) in enumerate(fi_entries):
                                line = f"{stem}_masks -> {fp.name}" if (row == 3) else f"{stem}_masks with {fp.name}"
                                ws.cell(row=row, column=7, value=line)
                                row += 1
                            wb.save(merged_path)
                        wb.close()
                    except Exception:
                        pass
                    self.parent_window.log_output(f"Merged {n_matched} run(s) into {merged_path.name}\n")
                    for excel_path, _fp, _lbl in phase_entries + fi_entries:
                        if excel_path != merged_path and excel_path.exists():
                            path_replacements[str(excel_path)] = str(merged_path)
                            try:
                                excel_path.unlink()
                                self.parent_window.log_output(f"  Removed {excel_path.name}\n")
                            except Exception:
                                pass
            except Exception as e:
                get_logger().debug("Merge FI channels failed: %s", e, exc_info=True)
        # So combined summary and completion message use the merged path(s)
        if path_replacements and getattr(self, "folder_output_paths", None):
            self.folder_output_paths = list(dict.fromkeys([path_replacements.get(p, p) for p in self.folder_output_paths]))

    def _combine_summaries_from_folder(self):
        """Combine Summary sheets from the folder in Path (same as Browse Folder)."""
        path_text = self.file_path.text().strip()
        if not path_text:
            QMessageBox.warning(
                self, "No Folder",
                "Select a folder in Path first (use Browse Folder). Combine uses that folder.",
            )
            return
        folder = Path(path_text)
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(
                self, "Invalid Folder",
                "Path must be a folder for Combine. Use Browse Folder to select a folder.",
            )
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
        if hasattr(self, "fi_mode_match_by_name"):
            d["fi_match_by_name"] = self.fi_mode_match_by_name.isChecked()
        if getattr(self, "_single_fi_path", None) is not None:
            d["single_fi_path"] = str(self._single_fi_path)
        if hasattr(self, "analyze_fi_contour"):
            d["analyze_fi_contour"] = self.analyze_fi_contour.isChecked()
        if hasattr(self, "fi_contour_thickness"):
            d["fi_contour_thickness"] = self.fi_contour_thickness.value()
        if hasattr(self, "run_foci"):
            d["run_foci"] = self.run_foci.isChecked()
        if hasattr(self, "foci_min_rel"):
            d["foci_min_rel"] = self.foci_min_rel.value()
        if hasattr(self, "foci_min_separation"):
            d["foci_min_separation"] = self.foci_min_separation.value()
        if hasattr(self, "foci_min_size"):
            d["foci_min_size"] = self.foci_min_size.value()
        if hasattr(self, "foci_max_size"):
            d["foci_max_size"] = self.foci_max_size.value()
        if hasattr(self, "foci_exclude_outside_cells"):
            d["foci_exclude_outside_cells"] = self.foci_exclude_outside_cells.isChecked()
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
        # Support both new (fi_match_by_name) and old (batch_fi) session keys
        fi_match = data.get("fi_match_by_name", data.get("batch_fi", True))
        if hasattr(self, "fi_mode_match_by_name"):
            self.fi_mode_match_by_name.setChecked(bool(fi_match))
            self.fi_mode_one_file.setChecked(not bool(fi_match))
        if "single_fi_path" in data and data["single_fi_path"]:
            try:
                self._single_fi_path = Path(data["single_fi_path"])
            except Exception:
                self._single_fi_path = None
        elif fi_match:
            self._single_fi_path = None
        if "analyze_fi_contour" in data and hasattr(self, "analyze_fi_contour"):
            self.analyze_fi_contour.setChecked(bool(data["analyze_fi_contour"]))
        if "fi_contour_thickness" in data and hasattr(self, "fi_contour_thickness"):
            try:
                self.fi_contour_thickness.setValue(int(data["fi_contour_thickness"]))
            except (TypeError, ValueError):
                pass
        if "run_foci" in data and hasattr(self, "run_foci"):
            self.run_foci.setChecked(bool(data["run_foci"]))
        if "foci_min_rel" in data and hasattr(self, "foci_min_rel"):
            try:
                self.foci_min_rel.setValue(float(data["foci_min_rel"]))
            except (TypeError, ValueError):
                pass
        if "foci_min_separation" in data and hasattr(self, "foci_min_separation"):
            try:
                self.foci_min_separation.setValue(int(data["foci_min_separation"]))
            except (TypeError, ValueError):
                pass
        if "foci_min_size" in data and hasattr(self, "foci_min_size"):
            try:
                self.foci_min_size.setValue(int(data["foci_min_size"]))
            except (TypeError, ValueError):
                pass
        if "foci_max_size" in data and hasattr(self, "foci_max_size"):
            try:
                self.foci_max_size.setValue(int(data["foci_max_size"]))
            except (TypeError, ValueError):
                pass
        if "foci_exclude_outside_cells" in data and hasattr(self, "foci_exclude_outside_cells"):
            self.foci_exclude_outside_cells.setChecked(bool(data["foci_exclude_outside_cells"]))
        if hasattr(self, "_update_fi_feedback"):
            self._update_fi_feedback()
    
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
            
            # Filter out files that already have _masks in name and exclude visualization/overlay files (not masks)
            for f in all_tif_files:
                name_lower = f.name.lower()
                if "_masks" not in name_lower and "_filter_visualization" not in name_lower and "_overlay" not in name_lower:
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
            
            # Load FI images if FI analysis is enabled (resolve path: single file or match-by-name for selected mask)
            fi_images = None
            fi_contour_thickness = 1
            do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
            if self.analyze_fi.isChecked():
                fi_path = None
                if self._is_match_by_name_mode():
                    fi_paths = self._resolve_fi_paths_for_mask(Path(mask_path))
                    fi_path = fi_paths[0] if fi_paths else None
                else:
                    fi_path = getattr(self, "_single_fi_path", None)
                if fi_path is not None and fi_path.exists():
                    try:
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
                        if do_fi_contour and hasattr(self, 'fi_contour_thickness'):
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
                if self.analyze_fi.isChecked() and getattr(self, "_single_fi_path", None) and self._single_fi_path.exists():
                    fi_path = self._single_fi_path
                    if fi_images is None or str(fi_path) != str(Path(mask_path).parent / "fi_temp"):
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
                        
                        # Foci overlay: when Run foci analysis is on, detect foci on current FI frame and draw on vis
                        n_foci_total = 0
                        raw_median_fi = None  # raw FI median (used for min rel threshold); shown in legend
                        run_foci = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
                        if run_foci and fi_images is not None and self.analyze_fi.isChecked():
                            try:
                                from bactimate.foci_utils import detect_peaks_skimage as _detect_foci
                            except ImportError:
                                _detect_foci = None
                            if _detect_foci is not None:
                                fi_image = fi_images[frame_idx]
                                if fi_image.ndim > 2:
                                    fi_image = fi_image[:, :, 0] if fi_image.shape[2] < 10 else fi_image[0, :, :]
                                if fi_image.shape[:2] == vis.shape[:2]:
                                    raw_median_fi = float(np.median(np.asarray(fi_image, dtype=np.float64)))
                                    min_rel = self.foci_min_rel.value() if hasattr(self, "foci_min_rel") else 2.0
                                    min_sep = self.foci_min_separation.value() if hasattr(self, "foci_min_separation") else 2
                                    min_sz = self.foci_min_size.value() if hasattr(self, "foci_min_size") else 1
                                    max_sz = self.foci_max_size.value() if hasattr(self, "foci_max_size") else 0
                                    max_sz = max_sz if max_sz > 0 else None
                                    peaks = _detect_foci(
                                        fi_image,
                                        min_rel=min_rel,
                                        min_distance=min_sep,
                                        min_size=min_sz,
                                        max_size=max_sz,
                                    )
                                    exclude_outside = bool(getattr(self, "foci_exclude_outside_cells", None) and self.foci_exclude_outside_cells.isChecked())
                                    if exclude_outside:
                                        peaks = [p for p in peaks if 0 <= int(p["y"]) < mask.shape[0] and 0 <= int(p["x"]) < mask.shape[1] and mask[int(p["y"]), int(p["x"])] in accepted_labels]
                                    n_foci_total = len(peaks)
                                    # Draw up to 500 foci as orange circles (radius 3 px)
                                    FOCI_CAP = 500
                                    orange = np.array([255, 165, 0], dtype=np.uint8)
                                    r = 3
                                    h, w = vis.shape[:2]
                                    for i, p in enumerate(peaks):
                                        if i >= FOCI_CAP:
                                            break
                                        yi, xi = int(p["y"]), int(p["x"])
                                        for dy in range(-r, r + 1):
                                            for dx in range(-r, r + 1):
                                                if dy * dy + dx * dx <= r * r:
                                                    yk, xk = yi + dy, xi + dx
                                                    if 0 <= yk < h and 0 <= xk < w:
                                                        vis[yk, xk] = orange
                        
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
                        if run_foci:
                            legend += f"  Orange=foci ({n_foci_total})"
                            if raw_median_fi is not None:
                                legend += f"  [raw median FI: {raw_median_fi:.0f}  (× min rel = threshold)]"
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
            minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), preview_dialog)  # Cmd+M on macOS
            minimize_shortcut.activated.connect(preview_dialog.showMinimized)
            
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
        
        # Run the package script as a module so we always use bactimate.scripts.analyze_cell_areas
        script_path = "-m"
        script_module = "bactimate.scripts.analyze_cell_areas"
        self._analyze_script_module = script_module
        script_dir = PROJECT_ROOT

        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: resolve to absolute path so the subprocess finds files regardless of cwd
            folder = Path(self.file_path.text()).resolve()
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
            
            # Filter out files that already have _masks in name and exclude visualization/overlay files (not masks)
            for f in all_tif_files:
                name_lower = f.name.lower()
                if "_masks" not in name_lower and "_filter_visualization" not in name_lower and "_overlay" not in name_lower:
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
                # User cancelled the file selection dialog
                self.parent_window.log_output("Analysis cancelled (no mask files selected).\n")
                return

            if len(selected_files) == 0:
                QMessageBox.warning(self, "No Files Selected", 
                                  "Please select at least one file to analyze.")
                return
            if self.analyze_fi.isChecked() and not self._is_match_by_name_mode() and not (getattr(self, "_single_fi_path", None) and self._single_fi_path.exists()):
                QMessageBox.warning(self, "FI File Required",
                                    "Analyze FI is on with 'One FI file for all masks' but no file is selected.\n\n"
                                    "Click 'Select FI file…' or switch to 'Match FI by name' to have FI files found automatically.")
                return
            
            # Process each file sequentially
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText(f"Analyzing... (0/{len(selected_files)})")
            self.parent_window.log_output(f"Starting mask analysis for {len(selected_files)} file(s) selected from folder: {folder}\n")
            
            # Show progress bar (range = number of tasks, which can be > selected_files when Batch FI has multiple FI per mask)
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)
            
            # Build list of tasks: (mask_path, fi_path or None, output_path). One task per (mask, FI) pair; multiple FI per mask = multiple tasks.
            self.folder_files = selected_files
            self.folder_results = []
            self.folder_output_paths = []
            batch_fi_on = self.analyze_fi.isChecked() and self._is_match_by_name_mode()
            single_fi_path = getattr(self, "_single_fi_path", None) if (self.analyze_fi.isChecked() and not batch_fi_on) else None
            if single_fi_path and not single_fi_path.exists():
                single_fi_path = None
            self.folder_fi_tasks: list[tuple[Path, Path | None, Path]] = []
            self.folder_phase_per_mask: dict[Path, Path | None] = {}  # for logging (mask -> phase.tif)
            for mask_path in selected_files:
                # Use resolved paths so the subprocess finds files regardless of current working directory
                mask_path = mask_path.resolve()
                base_dir = mask_path.parent
                if base_dir.name.startswith("Tif_") or "_segmentation" in base_dir.name:
                    base_dir = base_dir.parent
                stem = mask_path.stem.replace("_masks", "")
                analysis_dir = base_dir / "_analysis"
                if batch_fi_on:
                    fi_paths = self._resolve_fi_paths_for_mask(mask_path)
                    if not fi_paths:
                        self.folder_fi_tasks.append((mask_path, None, analysis_dir / f"{stem}_analysis.xlsx"))
                        self.folder_phase_per_mask[mask_path] = None
                        self.parent_window.log_output(f"  No FI file for {mask_path.name} (expected e.g. {stem}_fi.tif or {stem}_fi_*.tif in parent folder)\n")
                    else:
                        phase_path = self._resolve_phase_path_for_mask(mask_path)
                        self.folder_phase_per_mask[mask_path] = phase_path
                        if phase_path is not None:
                            self.folder_fi_tasks.append((mask_path, None, analysis_dir / f"{stem}_analysis_phase.xlsx"))
                        for fi_path in fi_paths:
                            suffix = self._fi_output_suffix(fi_path, stem)
                            out = analysis_dir / f"{stem}_analysis{suffix}.xlsx"
                            self.folder_fi_tasks.append((mask_path, fi_path, out))
                else:
                    out = analysis_dir / f"{stem}_analysis.xlsx"
                    self.folder_fi_tasks.append((mask_path, single_fi_path, out))
            if not self.folder_fi_tasks:
                self.parent_window.log_output("No tasks to run (no mask files or FI pairs).\n")
                self.analyze_btn.setEnabled(True)
                self.analyze_btn.setText("Analyze Masks")
                return
            self.current_task_index = 0
            self.parent_window.log_output(f"Running {len(self.folder_fi_tasks)} task(s) (script: {script_module})\n")
            # Process first task
            self._process_next_folder_file(script_path)
        else:
            # Single file mode (use resolved path so subprocess finds the file)
            input_file = Path(self.file_path.text()).resolve()

            # Determine base directory (go up one level if input is in Tif_* or _segmentation folder)
            base_dir = input_file.parent
            if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
                base_dir = base_dir.parent
            
            # Create output path in single _analysis folder
            output_path = base_dir / "_analysis" / f"{input_file.stem}_analysis.xlsx"
            
            # Build command arguments (use resolved input path)
            cmd_args = [
                str(input_file),
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
            if self.analyze_fi.isChecked() and getattr(self, "_single_fi_path", None) and self._single_fi_path.exists():
                fi_path = self._single_fi_path
                cmd_args.extend(['--fi-image', str(fi_path)])
                do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                if do_fi_contour:
                    fi_contour_thickness = self.fi_contour_thickness.value()
                    cmd_args.extend(['--fi-contour-thickness', str(fi_contour_thickness)])
                else:
                    cmd_args.append('--no-fi-contour')
                if getattr(self, "run_foci", None) and self.run_foci.isChecked():
                    cmd_args.append('--run-foci')
                    cmd_args.extend(['--foci-min-rel', str(self.foci_min_rel.value())])
                    cmd_args.extend(['--foci-min-separation', str(self.foci_min_separation.value())])
                    cmd_args.extend(['--foci-min-size', str(self.foci_min_size.value())])
                    if self.foci_max_size.value() > 0:
                        cmd_args.extend(['--foci-max-size', str(self.foci_max_size.value())])
                    if getattr(self, "foci_exclude_outside_cells", None) and self.foci_exclude_outside_cells.isChecked():
                        cmd_args.append('--foci-exclude-outside-cells')
            
            # Run in thread
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Analyzing...")
            self.parent_window.log_output(f"Starting mask analysis: {self.file_path.text()}\n")
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range
            self.parent_window.progress_bar.setValue(0)
            run_args = [getattr(self, "_analyze_script_module", "bactimate.scripts.analyze_cell_areas")] + cmd_args if script_path == "-m" else cmd_args
            # Store reference to prevent garbage collection
            self.worker = WorkerThread(script_path, run_args, parent=self)
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
        """Process next task in folder mode (one task = one mask + one FI file, or mask without FI)."""
        if self.current_task_index >= len(self.folder_fi_tasks):
            # All tasks processed
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Analyze Masks")
            self.parent_window.progress_bar.setVisible(False)
            n_tasks = len(self.folder_fi_tasks)
            n_masks = len(self.folder_files)
            n_ok = sum(1 for r in getattr(self, "folder_results", []) if r)
            n_fail = n_tasks - n_ok
            # Script writes to output_path.parent (single _analysis folder) or parent / stem_analysis
            success_dirs = []
            for i, (mask_file, _fi, output_path) in enumerate(self.folder_fi_tasks):
                if getattr(self, "folder_results", []) and i < len(self.folder_results) and self.folder_results[i]:
                    actual_dir = output_path.parent if output_path.parent.name == "_analysis" else output_path.parent / f"{Path(mask_file).stem}_analysis"
                    success_dirs.append(str(actual_dir))
            success_dirs = list(dict.fromkeys(success_dirs))  # deduplicate (e.g. all in _analysis)
            self.parent_window.log_output(f"\n✓ Analysis complete: {n_ok}/{n_tasks} task(s) succeeded ({n_masks} mask file(s)).\n")
            if success_dirs:
                self.parent_window.log_output("Output folder(s):\n  " + "\n  ".join(success_dirs) + "\n")
            if n_fail > 0:
                self.parent_window.log_output(f"  {n_fail} task(s) failed — check log above for errors.\n")
            # Merge multiple FI channel outputs into one Excel per mask (one table with two FI columns)
            self._merge_fi_channels_into_one_excel()
            if getattr(self, "create_combined_summary", None) and getattr(self, "create_combined_summary").isChecked() and n_ok > 0:
                paths = getattr(self, "folder_output_paths", [])
                if paths and self.folder_files:
                    folder = Path(self.folder_files[0]).parent
                    if folder.name.startswith("_segmentation"):
                        folder = folder.parent
                    out_path = folder / "_analysis" / f"{folder.name}_combined_summary.xlsx"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    match_log = self._match_log_from_folder_tasks()
                    if self._build_combined_summary_excel(paths, str(out_path), match_log=match_log):
                        self.parent_window.log_output(f"Combined summary saved to: {out_path}\n")
            msg = f"Completed: {n_ok}/{n_tasks} task(s) succeeded."
            if n_fail > 0:
                msg += f"\n{n_fail} task(s) failed — see Log for errors."
            if success_dirs:
                msg += "\n\nExcel files are in these folders:"
                for d in success_dirs[:5]:
                    msg += f"\n  • {Path(d).name}"
                if len(success_dirs) > 5:
                    msg += f"\n  ... and {len(success_dirs) - 5} more (see Log)."
            QMessageBox.information(self, "Analysis Complete", msg)
            return

        mask_file, fi_path, output_path = self.folder_fi_tasks[self.current_task_index]
        fi_label = fi_path.stem if fi_path else "no FI"
        self.parent_window.log_output(f"\nProcessing task {self.current_task_index + 1}/{len(self.folder_fi_tasks)}: {mask_file.name} + {fi_label}\n")
        if fi_path is not None:
            self.parent_window.log_output(f"  FI image: {fi_path}\n")
        else:
            self.parent_window.log_output(f"  No FI image for this task (morphology only). Enable Batch and use stem_fi.tif or stem_fi_Channel.tif in the mask folder's parent.\n")
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
        
        # Add FI image path if FI analysis is enabled (from current task)
        if self.analyze_fi.isChecked():
            if fi_path is not None:
                cmd_args.extend(['--fi-image', str(fi_path)])
                do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                if do_fi_contour:
                    fi_contour_thickness = self.fi_contour_thickness.value()
                    cmd_args.extend(['--fi-contour-thickness', str(fi_contour_thickness)])
                else:
                    cmd_args.append('--no-fi-contour')
                if getattr(self, "run_foci", None) and self.run_foci.isChecked():
                    cmd_args.append('--run-foci')
                    cmd_args.extend(['--foci-min-rel', str(self.foci_min_rel.value())])
                    cmd_args.extend(['--foci-min-separation', str(self.foci_min_separation.value())])
                    cmd_args.extend(['--foci-min-size', str(self.foci_min_size.value())])
                    if self.foci_max_size.value() > 0:
                        cmd_args.extend(['--foci-max-size', str(self.foci_max_size.value())])
                    if getattr(self, "foci_exclude_outside_cells", None) and self.foci_exclude_outside_cells.isChecked():
                        cmd_args.append('--foci-exclude-outside-cells')
        
        self.parent_window.log_output(f"Output file: {output_path}\n")
        file_progress = int((self.current_task_index / len(self.folder_fi_tasks)) * 100)
        self.parent_window.progress_bar.setValue(file_progress)
        self.analyze_btn.setText(f"Analyzing... ({self.current_task_index + 1}/{len(self.folder_fi_tasks)})")
        run_args = [getattr(self, "_analyze_script_module", "bactimate.scripts.analyze_cell_areas")] + cmd_args if script_path == "-m" else cmd_args
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, run_args, parent=self)
        self.worker.finished.connect(lambda success, msg: self._on_folder_file_finished(success, msg, script_path))
        self.worker.progress.connect(self.parent_window.log_output)
        self.worker.start()
    
    def _on_folder_file_finished(self, success, message, script_path):
        """Handle completion of a single task in folder mode"""
        self.parent_window.log_output(message)
        if success:
            self.folder_results.append(True)
            self.parent_window.log_output(f"✓ Task {self.current_task_index + 1}/{len(self.folder_fi_tasks)} completed successfully\n")
            # Write Logs to this task's Excel (Summary sheet column G)
            mask_file, fi_path, output_path = self.folder_fi_tasks[self.current_task_index]
            display_tif = None
            if fi_path is None:
                phase_path = getattr(self, "folder_phase_per_mask", {}).get(mask_file)
                if phase_path is not None:
                    display_tif = phase_path.name
            self._write_logs_to_individual_excel(output_path, mask_file, fi_path, display_tif_name=display_tif)
        else:
            self.folder_results.append(False)
            self.parent_window.log_output(f"✗ Task {self.current_task_index + 1}/{len(self.folder_fi_tasks)} failed: {message}\n")
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        self.current_task_index += 1
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


