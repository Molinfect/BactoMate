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
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
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
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.analyze_masks_preview_vis import build_preview_vis
from bactimate.gui.analysis_paths import analysis_dir_from_base, base_dir_for_mask, stem_from_mask_path
from bactimate.gui.constants import PROJECT_ROOT
from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.preferences import get_default_browse_dir
from bactimate.gui.image_utils import to_qpixmap, scale_for_preview
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.utils import safe_imread
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog
from bactimate.gui.workers import WorkerThread
from bactimate.gui.log_config import get_logger, write_crash_log, write_gui_log_line
from bactimate.modules.tiff_metadata import read_calibration_from_path_or_original
from bactimate.modules.validation import paths_exist, mask_fi_spatial_shape_match

class AnalyzeMasksTab(QWidget):
    """Tab for analyzing existing masks"""
    workflow_step_finished = pyqtSignal(bool)  # for workflow replay chain

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._performed_this_session = False  # True if user ran analysis this session (for Re-run only same steps)
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
        self.fi_naming_hint = QLabel('(Match stem_fi / stem_foci and stem_fi_* / stem_foci_* to the mask stem; search in mask folder and parent folder)')
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
        run_foci_row.addSpacing(8)
        self.foci_exclude_outside_cells = QCheckBox("Exclude foci outside / rejected cells")
        self.foci_exclude_outside_cells.setChecked(False)
        run_foci_row.addWidget(self.foci_exclude_outside_cells)
        run_foci_row.addSpacing(12)
        run_foci_row.addWidget(QLabel("Detection:"))
        self.foci_detection_mode = QComboBox()
        self.foci_detection_mode.addItems(["Manual parameters", "Auto-detect (data-driven)", "Spotiflow (deep learning)"])
        self.foci_detection_mode.setCurrentIndex(0)
        self.foci_detection_mode.setMinimumContentsLength(18)
        self.foci_detection_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.foci_detection_mode.currentIndexChanged.connect(self._update_foci_mode_visibility)
        run_foci_row.addWidget(self.foci_detection_mode, 0)
        run_foci_row.addSpacing(8)
        run_foci_row.addWidget(create_info_icon(
            "**Run foci:** Detect bright spots in the FI channel per frame; add foci count and a **Foci** sheet to Excel "
            "(foci_id, cell_id, frame, y, x, intensity, Area (px/µm²), Mean intensity, Rel intensity, in_cell).\n\n"
            "**Exclude foci outside / rejected cells:** When checked, only foci inside accepted cells are included; "
            "when unchecked, all detected foci are included.\n\n"
            "**Detection:**\n"
            "• **Manual:** Cell-first: we start from the mask and each cell. Inside each cell we find local maxima, "
            "build the focus blob (connected pixels ≥ 80% of peak, same cell), then apply your filters (min peak, min/max area, min focus mean). "
            "The per-cell factor (e.g. 1.6) compares the **focus blob mean** to the **mean of the rest of the cell** (cell minus focus); "
            "a focus is kept only if focus mean ≥ factor × rest-of-cell mean. The focus never contributes to that baseline.\n"
            "• **Auto-detect:** Cell-first, no manual thresholds; region-growing from each cell's brightest pixel.\n"
            "• **Spotiflow:** Deep learning–based spot detection; very robust and threshold-free. Uses PyTorch; GPU recommended for speed.",
            parent_widget=self
        ))
        run_foci_row.addStretch()
        fi_content_layout.addLayout(run_foci_row)
        # Manual parameters — shown only when Detection = Manual
        self.foci_manual_params_container = QWidget()
        foci_manual_layout = QVBoxLayout(self.foci_manual_params_container)
        foci_manual_layout.setContentsMargins(0, 0, 0, 0)
        foci_params_row = QHBoxLayout()
        self.foci_min_rel_label = QLabel("Min relative intensity:")
        foci_params_row.addWidget(self.foci_min_rel_label)
        self.foci_min_rel = QDoubleSpinBox()
        self.foci_min_rel.setRange(0.5, 100.0)
        self.foci_min_rel.setValue(2.0)
        self.foci_min_rel.setDecimals(1)
        self.foci_min_rel.setSingleStep(0.1)
        foci_params_row.addWidget(self.foci_min_rel)
        self.foci_min_rel_mode = QComboBox()
        self.foci_min_rel_mode.addItems(["per cell", "per frame", "both"])
        self.foci_min_rel_mode.setCurrentIndex(0)  # default: per cell (first in list)
        self.foci_min_rel_mode.setToolTip("Per frame: frame median × factor. Per cell: (focus blob mean) / (mean of cell excluding focus) ≥ factor. Both: apply frame and cell factors.")
        self.foci_min_rel_mode.currentTextChanged.connect(self._update_foci_mode_visibility)
        foci_params_row.addWidget(self.foci_min_rel_mode)
        # Second factor (cell) — visible only when mode is "both"
        self.foci_min_rel_cell_label = QLabel("Cell factor:")
        self.foci_min_rel_cell = QDoubleSpinBox()
        self.foci_min_rel_cell.setRange(0.5, 100.0)
        self.foci_min_rel_cell.setValue(2.0)
        self.foci_min_rel_cell.setDecimals(1)
        self.foci_min_rel_cell.setSingleStep(0.1)
        self.foci_min_rel_cell.setToolTip("When mode is Both: minimum (focus blob mean) / (mean of cell excluding focus) for in-cell foci.")
        foci_params_row.addWidget(self.foci_min_rel_cell_label)
        foci_params_row.addWidget(self.foci_min_rel_cell)
        self.foci_min_rel_cell_label.setVisible(False)
        self.foci_min_rel_cell.setVisible(False)
        foci_params_row.addWidget(create_info_icon(
            "Minimum intensity threshold for foci (factor). Use the dropdown to choose how it is applied.\n\n"
            "• Per frame: Peaks whose raw intensity is below (frame median × this value) are excluded. "
            "The median is over the whole raw FI image (all pixels, including background). "
            "So if the image has a lot of dark background, the frame median is low and the threshold (median × factor) can be much lower than factor × cell mean — you may see more foci than expected. "
            "Example: frame median 500, factor 2 → threshold 1000 (raw); cell mean 1500 → 2× would be 3000, but we use 1000. "
            "Good when you want one global cutoff per image.\n\n"
            "• Per cell: For each cell, only foci where (focus blob mean) / (mean of rest of cell) ≥ this value are kept. "
            "The focus blob is the connected region we built; the rest of the cell excludes it, so the focus never contributes to the baseline. "
            "A focus must be at least this many times brighter than the cytoplasm. Good when cells differ in brightness.\n\n"
            "• Both: First exclude peaks below (frame median × Frame factor). "
            "Then, for in-cell foci, keep only those with (focus blob mean) / (mean of cell excluding focus) ≥ Cell factor.",
            parent_widget=self
        ))
        foci_params_row.addWidget(QLabel("Min separation (px):"))
        foci_params_row.addWidget(create_info_icon(
            "Minimum distance in pixels between two detected foci.\n\n"
            "Peaks closer than this are merged (the stronger one is kept). "
            "Increases with this value to avoid multiple detections on the same spot. "
            "Use ~2 for dense foci; increase if you get duplicate detections.",
            parent_widget=self
        ))
        self.foci_min_separation = QSpinBox()
        self.foci_min_separation.setRange(1, 200)
        self.foci_min_separation.setValue(2)
        foci_params_row.addWidget(self.foci_min_separation)
        foci_params_row.addWidget(QLabel("Min area (µm²):"))
        foci_params_row.addWidget(create_info_icon(
            "Minimum focus area in µm² (converted to pixels using Pixel Size (µm)).\n\n"
            "Focus area is the connected region around the peak (pixels above half peak intensity, within the same cell). "
            "If you see fewer foci than expected, try: lower Min separation (e.g. 1 px), lower Min relative intensity, or set Min area to 0.001.\n\n"
            "The Excel Foci sheet reports Area (px) and Area (µm²) per focus.",
            parent_widget=self
        ))
        self.foci_min_area_um2 = QDoubleSpinBox()
        self.foci_min_area_um2.setRange(0.001, 6.0)
        self.foci_min_area_um2.setDecimals(3)
        self.foci_min_area_um2.setSingleStep(0.01)
        self.foci_min_area_um2.setValue(0.100)
        self.foci_min_area_um2.setFixedWidth(80)
        foci_params_row.addWidget(self.foci_min_area_um2)
        foci_params_row.addWidget(QLabel("Max area (µm²):"))
        foci_params_row.addWidget(create_info_icon(
            "Maximum focus area in µm². Peaks with a larger connected region are excluded. Set to 0 for no maximum.",
            parent_widget=self
        ))
        self.foci_max_area_um2 = QDoubleSpinBox()
        self.foci_max_area_um2.setRange(0.0, 6.0)
        self.foci_max_area_um2.setDecimals(3)
        self.foci_max_area_um2.setSingleStep(0.01)
        self.foci_max_area_um2.setValue(0.400)
        self.foci_max_area_um2.setSpecialValueText("0 (no max)")
        self.foci_max_area_um2.setFixedWidth(80)
        foci_params_row.addWidget(self.foci_max_area_um2)
        foci_params_row.addStretch()
        foci_manual_layout.addLayout(foci_params_row)
        # Second row: optional brightness filters (below min rel intensity row)
        foci_params_row2 = QHBoxLayout()
        foci_params_row2.addWidget(QLabel("Min peak (raw):"))
        foci_params_row2.addWidget(create_info_icon(
            "Minimum raw intensity at the focus peak (0 = off).\n\n"
            "Only foci with peak pixel value ≥ this value are kept. Use to require a minimum brightness "
            "and keep obvious bright foci while dropping dim noise.",
            parent_widget=self
        ))
        self.foci_min_peak_raw = QDoubleSpinBox()
        self.foci_min_peak_raw.setRange(0, 100000)
        self.foci_min_peak_raw.setValue(0)
        self.foci_min_peak_raw.setDecimals(0)
        self.foci_min_peak_raw.setSingleStep(100)
        self.foci_min_peak_raw.setSpecialValueText("0 (off)")
        foci_params_row2.addWidget(self.foci_min_peak_raw)
        foci_params_row2.addWidget(QLabel("Min peak mean:"))
        foci_params_row2.addWidget(create_info_icon(
            "Minimum mean intensity of the focus blob (0 = off).\n\n"
            "Mean is taken over the connected pixels above half the peak (same region as size). "
            "Bright compact foci have high mean; diffuse dim spots have low mean. Use to keep obvious foci and drop weak/diffuse ones.",
            parent_widget=self
        ))
        self.foci_min_focus_mean = QDoubleSpinBox()
        self.foci_min_focus_mean.setRange(0, 100000)
        self.foci_min_focus_mean.setValue(0)
        self.foci_min_focus_mean.setDecimals(0)
        self.foci_min_focus_mean.setSingleStep(100)
        self.foci_min_focus_mean.setSpecialValueText("0 (off)")
        foci_params_row2.addWidget(self.foci_min_focus_mean)
        foci_params_row2.addStretch()
        foci_manual_layout.addLayout(foci_params_row2)
        fi_content_layout.addWidget(self.foci_manual_params_container)
        # Manual-only widgets (Exclude foci outside is not here — it applies to all detection modes)
        self.foci_params_widgets = [self.foci_min_rel, self.foci_min_rel_mode, self.foci_min_rel_cell_label, self.foci_min_rel_cell, self.foci_min_separation, self.foci_min_area_um2, self.foci_max_area_um2, self.foci_min_peak_raw, self.foci_min_focus_mean]
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
            "Pixel size in µm per pixel for area and distance calculations.\n\n"
            "Used for: cell area (µm²), size filters, and focus Area (µm²) in the Foci sheet (Area (µm²) = Area (px) × pixel_size²). "
            "Source: Read from file metadata when available (TIFF ImageDescription pixelSize= or spacing=, "
            "ImageJ metadata, OME PhysicalSizeX/Y, ND2 calibration, or TIFF Resolution). "
            "If not found, the field is left at 0.108 µm (default); the indicator shows '(from file)' when metadata was loaded, "
            "or '⚠ Check value!' when using the default.\n\n"
            "Used to convert pixel areas to µm² (e.g. cell area, focus area = area_px × pixel_size²)."
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
            "• If Analyze FI is used: Mean FI intensity, Contour FI intensity, FI ratio (Contour/Mean)\n"
            "• If Run foci is on: Number of Foci, Foci Mean Intensity, Foci Total Intensity, Foci Max Intensity, "
            "Foci Mean Area (px), Foci Total Area (px), Foci Mean Intensity Ratio\n\n"
            "Run foci also adds a **Foci** sheet: one row per focus with foci_id, cell_id, frame, y, x, intensity, "
            "Area (px), Area (µm²), Mean intensity (focus blob), Rel intensity (peak/frame median), in_cell.\n\n"
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
        export_csv_row = QHBoxLayout()
        self.export_csv = QCheckBox("Also save CSV")
        self.export_csv.setChecked(False)
        export_csv_info_icon = create_info_icon(
            "Also write Summary, Per-Cell Data, and Foci (if present) as CSV files next to the Excel file.\n\n"
            "Useful for R, Python, or Prism; filenames: stem_summary.csv, stem_per_cell.csv, stem_foci.csv.",
            parent_widget=self
        )
        export_csv_row.addWidget(self.export_csv)
        export_csv_row.addWidget(export_csv_info_icon)
        export_csv_row.addStretch()
        options_layout.addLayout(export_csv_row)
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
        
        analyze_btn_row = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Masks")
        self.analyze_btn.setEnabled(False)  # Disabled until file/folder is selected
        self.analyze_btn.clicked.connect(self.analyze_masks)
        analyze_btn_row.addWidget(self.analyze_btn, 1)  # stretch so same full width as Preview button
        self.analyze_cancel_btn = QPushButton("Cancel")
        self.analyze_cancel_btn.setToolTip("Stop the current analysis run.")
        self.analyze_cancel_btn.setVisible(False)
        self.analyze_cancel_btn.clicked.connect(self._on_analyze_cancel)
        analyze_btn_row.addWidget(self.analyze_cancel_btn)
        scroll_layout.addLayout(analyze_btn_row)

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
            "You can select a segmentation folder (e.g. _segmentation_bact_phase_omni, _segmentation_unet) even if it contains an overlays/ "
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

    def _foci_area_um2_to_px(self, pixel_size_um: Optional[float] = None):
        """Convert foci min/max area (µm²) to pixel counts using Pixel Size (µm). Returns (min_sz, max_sz_or_none).
        Uses floor for max so no focus with displayed area > user max is included; ceil for min so displayed area >= user min."""
        ps = float(pixel_size_um if pixel_size_um is not None else (getattr(self, "pixel_size", None) and self.pixel_size.value() or 0.108))
        if ps <= 0:
            ps = 0.108
        ps2 = ps * ps
        min_um2 = getattr(self, "foci_min_area_um2", None) and self.foci_min_area_um2.value() or 0.2
        max_um2 = getattr(self, "foci_max_area_um2", None) and self.foci_max_area_um2.value() or 0.0
        # Ceil for min: e.g. 0.2 µm² -> require at least 18 px so displayed area is >= 0.2
        min_sz = max(1, int(np.ceil(min_um2 / ps2)))
        # Floor for max: e.g. 0.8 µm² -> allow at most 68 px so no focus displays > 0.8 µm²
        max_sz = int(max_um2 / ps2) if max_um2 > 0 else None
        return min_sz, max_sz

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
            # Summary: one mask = one segmentation, used for FI, Foci (if checked), and phase (if found)
            run_foci_on = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
            desc = "Segmentation used for their FI channel(s)"
            if run_foci_on:
                desc += ", Foci (Run foci checked)"
            desc += ", and phase (if found) each."
            if total_fi_pairings == 0:
                header = f"{len(masks)} mask file(s); no FI files matched.\n\n"
            elif total_fi_pairings == len(masks):
                header = f"{len(masks)} mask file(s). {desc}\n\n"
            else:
                header = f"{len(masks)} mask file(s) → {total_fi_pairings} analyses ({desc})\n\n"
            body = "Analyze FI (mask → FI channel(s)):\n" + "\n".join(fi_lines)
            if run_foci_on:
                foci_lines = []
                for mask_path in masks:
                    foci_paths = self._resolve_foci_paths_for_mask(mask_path)
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
        manual_foci = self._is_foci_manual_mode()
        for w in getattr(self, "foci_params_widgets", []):
            w.setEnabled(analyze_fi_on and run_foci_on and manual_foci)
        self._update_fi_contour_state()
        if hasattr(self, "_update_foci_mode_visibility"):
            self._update_foci_mode_visibility()

    def _is_foci_manual_mode(self) -> bool:
        """True if foci detection mode is Manual parameters (manual sliders apply)."""
        if not getattr(self, "foci_detection_mode", None):
            return True
        return self.foci_detection_mode.currentIndex() == 0

    def _update_foci_mode_visibility(self) -> None:
        """Show/hide manual params block; show Frame/Cell factor labels when mode is 'both'; enable widgets when Detection = Manual."""
        manual_foci = self._is_foci_manual_mode()
        run_foci_on = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
        analyze_fi_on = bool(getattr(self, "analyze_fi", None) and self.analyze_fi.isChecked())
        if getattr(self, "foci_manual_params_container", None) is not None:
            self.foci_manual_params_container.setVisible(analyze_fi_on and run_foci_on and manual_foci)
        if getattr(self, "foci_min_rel_mode", None):
            is_both = self.foci_min_rel_mode.currentText().strip().lower() == "both"
            if getattr(self, "foci_min_rel_label", None):
                self.foci_min_rel_label.setText("Frame factor:" if is_both else "Min relative intensity:")
            if getattr(self, "foci_min_rel_cell_label", None):
                self.foci_min_rel_cell_label.setVisible(is_both)
            if getattr(self, "foci_min_rel_cell", None):
                self.foci_min_rel_cell.setVisible(is_both)
        for w in getattr(self, "foci_params_widgets", []):
            w.setEnabled(analyze_fi_on and run_foci_on and manual_foci)
        # Exclude foci outside / rejected cells applies to all detection modes (Manual, Auto-detect, Spotiflow)
        if hasattr(self, "foci_exclude_outside_cells"):
            self.foci_exclude_outside_cells.setEnabled(analyze_fi_on and run_foci_on)
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
        """For Analyze FI: find FI channel files (name must contain _fi). stem_fi / stem_fi_* only.
        _foci-only files are not included (use _resolve_foci_paths_for_mask for Run foci).
        If none found, tries base stem. Search dir: mask folder then parent folder.
        """
        parent = mask_path.parent
        if parent.name.startswith("_segmentation"):
            search_dir = parent.parent
        else:
            search_dir = parent
        stem = mask_path.stem.replace("_masks", "")

        def matches_fi(name: str, s: str) -> bool:
            # Only _fi files for Analyze FI (mean/contour). _foci-only files use _resolve_foci_paths_for_mask.
            return name == f"{s}_fi" or name.startswith(f"{s}_fi_")

        def search_in(directory: Path) -> list[Path]:
            out: list[Path] = []
            if not directory.exists() or not directory.is_dir():
                return out
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if matches_fi(f.stem, stem):
                    out.append(f)
            return out

        def try_base(directory: Path, s: str) -> list[Path]:
            out = []
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if matches_fi(f.stem, s):
                    out.append(f)
            return out

        found: list[Path] = search_in(search_dir)
        if not found and "_" in stem:
            base = stem.rsplit("_", 1)[0]
            found = try_base(search_dir, base)
        # Fallback: stem like EM19763_NoPhage__Phase60X vs FI file EM19763_NoPhage_fi_foci_mScarlet — use part before "__"
        if not found and "__" in stem:
            base_double = stem.split("__")[0]
            if base_double:
                found = try_base(search_dir, base_double)
        if not found and search_dir.parent != search_dir:
            found = search_in(search_dir.parent)
            if not found and "_" in stem:
                found = try_base(search_dir.parent, stem.rsplit("_", 1)[0])
            if not found and "__" in stem:
                base_double = stem.split("__")[0]
                if base_double:
                    found = try_base(search_dir.parent, base_double)
        return sorted(set(found), key=lambda p: p.name.lower())

    def _resolve_foci_paths_for_mask(self, mask_path: Path) -> list[Path]:
        """For Run foci: find _foci channel files. stem_foci / stem_foci_* or stem_fi_* with _foci in name.
        Search dir: same as FI. Used only when Run foci analysis is checked."""
        parent = mask_path.parent
        if parent.name.startswith("_segmentation"):
            search_dir = parent.parent
        else:
            search_dir = parent
        stem = mask_path.stem.replace("_masks", "")

        def matches_foci(name: str, s: str) -> bool:
            if name == f"{s}_foci" or name.startswith(f"{s}_foci_"):
                return True
            if name.startswith(f"{s}_fi_") and "_foci" in name:
                return True
            return False

        def search_in(directory: Path) -> list[Path]:
            out: list[Path] = []
            if not directory.exists() or not directory.is_dir():
                return out
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if matches_foci(f.stem, stem):
                    out.append(f)
            return out

        def try_base_foci(directory: Path, s: str) -> list[Path]:
            out = []
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if matches_foci(f.stem, s):
                    out.append(f)
            return out

        found_foci: list[Path] = search_in(search_dir)
        if not found_foci and "_" in stem:
            base = stem.rsplit("_", 1)[0]
            found_foci = try_base_foci(search_dir, base)
        # Fallback: stem like EM19763_NoPhage__Phase60X vs FI/foci file EM19763_NoPhage_fi_foci_mScarlet — use part before "__"
        if not found_foci and "__" in stem:
            base_double = stem.split("__")[0]
            if base_double:
                found_foci = try_base_foci(search_dir, base_double)
        if not found_foci and search_dir.parent != search_dir:
            found_foci = search_in(search_dir.parent)
            if not found_foci and "_" in stem:
                found_foci = try_base_foci(search_dir.parent, stem.rsplit("_", 1)[0])
            if not found_foci and "__" in stem:
                base_double = stem.split("__")[0]
                if base_double:
                    found_foci = try_base_foci(search_dir.parent, base_double)
        return sorted(set(found_foci), key=lambda p: p.name.lower())

    def _resolve_phase_path_for_mask(self, mask_path: Path) -> Path | None:
        """For batch FI: find the non-FI (phase/original) TIFF that matches this mask, if any.
        Looks for stem.tif or base.tif in the same search dir as FI (and parent folder if not found), excluding _fi/_foci files.
        """
        parent = mask_path.parent
        if parent.name.startswith("_segmentation"):
            search_dir = parent.parent
        else:
            search_dir = parent
        stem = mask_path.stem.replace("_masks", "")

        def find_in(directory: Path) -> Path | None:
            if not directory.exists() or not directory.is_dir():
                return None
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if "_fi" in f.stem.lower() or "_foci" in f.stem.lower():
                    continue
                if f.stem == stem:
                    return f
            if "_" in stem:
                base = stem.rsplit("_", 1)[0]
                for f in directory.iterdir():
                    if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                        continue
                    if "_fi" in f.stem.lower() or "_foci" in f.stem.lower():
                        continue
                    if f.stem == base:
                        return f
            return None

        p = find_in(search_dir)
        if p is not None:
            return p
        if search_dir.parent != search_dir:
            return find_in(search_dir.parent)
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

    def _build_combined_summary_excel(self, excel_paths, output_path, match_log=None, log_callback=None):
        """Build long + wide combined summary from a list of analysis Excel paths. Returns True on success.
        If match_log is (n_masks, n_tif, lines), writes Logs in column G (G1=Logs, G2=count, G3+=lines).
        If output_path exists and already has Logs in G, appends the new lines to column G instead of overwriting.
        If log_callback is set (e.g. parent_window.log_output), logs why the build failed when it returns False.
        """
        def _log(msg: str) -> None:
            if log_callback:
                log_callback(msg)

        import re
        import pandas as pd
        from openpyxl import load_workbook
        from openpyxl.utils.dataframe import dataframe_to_rows

        def _sanitize_sheet_name(name: str, max_len: int = 31) -> str:
            # Excel: no \ / ? * [ ]
            s = re.sub(r'[\\/:*?\[\]]', '_', str(name))
            return s[:max_len] if len(s) > max_len else s

        if not excel_paths:
            _log("Combined summary: no paths provided.\n")
            return False
        frames_long = []
        missing = []
        no_summary = []
        per_cell_cache: dict[Path, pd.DataFrame | None] = {}  # path -> Per-Cell Data (or None); filled when we open each file once
        for p in excel_paths:
            p = Path(p)
            if not p.exists():
                missing.append(p.name)
                continue
            try:
                with pd.ExcelFile(p) as xl:
                    df = pd.read_excel(xl, sheet_name="Summary")
                    per_cell = None
                    try:
                        per_cell = pd.read_excel(xl, sheet_name="Per-Cell Data")
                    except Exception:
                        pass
                    per_cell_cache[p] = per_cell
            except Exception as e:
                no_summary.append((p.name, str(e)))
                continue
            if df is None or df.empty:
                no_summary.append((p.name, "empty Summary sheet"))
                continue
            df = df.copy()
            df["File"] = p.stem.replace("_analysis", "").replace("_masks_analysis", "")
            frames_long.append(df)
        if missing:
            _log(f"Combined summary: {len(missing)} path(s) not found: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}\n")
        if no_summary and not frames_long:
            _log(f"Combined summary: no Summary sheet could be read. First failure: {no_summary[0][0]} — {no_summary[0][1]}\n")
        if not frames_long:
            return False
        long_df = pd.concat(frames_long, ignore_index=True)
        cols = [c for c in ["File", "Frame"] if c in long_df.columns]
        cols += [c for c in long_df.columns if c not in cols]
        long_df = long_df[cols]
        wide_df = None
        overview_df = None  # Per-file overview for single-time-point analyses
        wide_err = None
        try:
            if "Frame" not in long_df.columns:
                wide_err = "Summary sheets have no 'Frame' column"
            else:
                frames = sorted(long_df["Frame"].dropna().unique().astype(int).tolist())
                if not frames:
                    wide_err = "Summary sheets have no Frame values"
                else:
                    # Single time point (e.g. one frame per file): build per-file overview instead of Frame-centric wide
                    single_time_point = len(frames) == 1
                    if single_time_point:
                        overview_rows = []
                        for f in long_df["File"].unique().tolist():
                            sub_summary = long_df[long_df["File"] == f].iloc[0]
                            row = {"File": f}
                            for c in ["Number of Cells", "Total Cell Area (µm²)", "Total Cell Area (Relative)"]:
                                if c in sub_summary.index:
                                    row[c] = sub_summary[c]
                            # Enrich from Per-Cell Data for this file
                            path_for_file = next((Path(p) for p in excel_paths if Path(p).stem.replace("_analysis", "").replace("_masks_analysis", "") == f), None)
                            if path_for_file is None:
                                path_for_file = next((Path(p) for p in excel_paths if f in Path(p).stem), None)
                            if path_for_file is not None:
                                per_cell = per_cell_cache.get(path_for_file)
                                if per_cell is not None and not per_cell.empty:
                                    n_cells = len(per_cell)
                                    row["Number of Cells"] = n_cells
                                    if "Total Cell Area (µm²)" not in row and "Area (µm²)" in per_cell.columns:
                                        row["Total Cell Area (µm²)"] = per_cell["Area (µm²)"].sum()
                                    # Cells with foci
                                    foci_col = None
                                    for cand in ("Number of Foci", "GFP_foci_count", "FI_foci_count"):
                                        if cand in per_cell.columns:
                                            foci_col = cand
                                            break
                                    if foci_col is not None:
                                        n_with_foci = (per_cell[foci_col].fillna(0) > 0).sum()
                                        row["Cells with foci"] = int(n_with_foci)
                                    # Mean FI (whole cell)
                                    for fi_col in ("Mean FI Intensity (whole cell)", "Mean FI Intensity"):
                                        if fi_col in per_cell.columns:
                                            row["Mean FI (cells)"] = per_cell[fi_col].mean()
                                            break
                                    # Mean Foci intensity (over cells that have foci, or all)
                                    if "Foci Mean Intensity" in per_cell.columns:
                                        row["Mean Foci intensity"] = per_cell["Foci Mean Intensity"].mean()
                            overview_rows.append(row)
                        overview_df = pd.DataFrame(overview_rows)
                        # Prefer column order for readability
                        pref = ["File", "Number of Cells", "Total Cell Area (µm²)", "Total Cell Area (Relative)", "Cells with foci", "Mean FI (cells)", "Mean Foci intensity"]
                        extra = [c for c in overview_df.columns if c not in pref]
                        overview_df = overview_df[[c for c in pref if c in overview_df.columns] + extra]
                    # Frame-centric wide table (for growth/video, multi-frame); add per-frame stats from Per-Cell when available
                    wide_data = {"Frame": frames}
                    # Per (file, frame) aggregates from Per-Cell Data for extra columns
                    per_file_frame_agg: dict = {}
                    for p in excel_paths:
                        p = Path(p)
                        per_cell = per_cell_cache.get(p)
                        if per_cell is None or per_cell.empty or "Frame" not in per_cell.columns:
                            continue
                        f_name = p.stem.replace("_analysis", "").replace("_masks_analysis", "")
                        per_file_frame_agg[f_name] = {}
                        foci_col = next((c for c in ("Number of Foci", "GFP_foci_count", "FI_foci_count") if c in per_cell.columns), None)
                        fi_col = next((c for c in ("Mean FI Intensity (whole cell)", "Mean FI Intensity") if c in per_cell.columns), None)
                        foci_mean_col = "Foci Mean Intensity" if "Foci Mean Intensity" in per_cell.columns else None
                        for frame_val in per_cell["Frame"].dropna().unique():
                            fr = int(frame_val) if frame_val == frame_val else None
                            if fr is None:
                                continue
                            pc = per_cell[per_cell["Frame"] == frame_val]
                            agg = {}
                            if foci_col is not None:
                                agg["cells_with_foci"] = int((pc[foci_col].fillna(0) > 0).sum())
                            if fi_col is not None:
                                agg["mean_fi"] = pc[fi_col].mean()
                            if foci_mean_col is not None:
                                agg["mean_foci"] = pc[foci_mean_col].mean()
                            per_file_frame_agg[f_name][fr] = agg
                    for f in long_df["File"].unique().tolist():
                        sub = long_df[long_df["File"] == f].drop_duplicates(subset=["Frame"], keep="first").set_index("Frame")
                        for col in ["Total Cell Area (µm²)", "Total Cell Area (Relative)", "Number of Cells"]:
                            if col in sub.columns:
                                wide_data[f"{f}_{col}"] = sub.reindex(frames)[col].values
                        # Per-frame stats from Per-Cell (cells with foci, mean FI, mean foci)
                        agg_by_frame = per_file_frame_agg.get(f, {})
                        wide_data[f"{f}_Cells with foci"] = [agg_by_frame.get(fr, {}).get("cells_with_foci", None) for fr in frames]
                        wide_data[f"{f}_Mean FI (cells)"] = [agg_by_frame.get(fr, {}).get("mean_fi", None) for fr in frames]
                        wide_data[f"{f}_Mean Foci intensity"] = [agg_by_frame.get(fr, {}).get("mean_foci", None) for fr in frames]
                    wide_df = pd.DataFrame(wide_data)
        except Exception as e:
            wide_err = str(e)
            wide_df = None
            overview_df = None
        # Use overview as main sheet when single time point, else wide
        sheet_df = overview_df if (overview_df is not None and not overview_df.empty) else wide_df
        if sheet_df is None or sheet_df.empty:
            if wide_err:
                _log(f"Combined summary: wide table build failed — {wide_err}\n")
            return False
        out_path = Path(output_path)
        existing_log_lines: list[str] = []
        if out_path.exists() and match_log is not None:
            try:
                wb_existing = load_workbook(output_path, data_only=True)
                sheet_name_wide = "Combined Summary (wide)"
                if sheet_name_wide in wb_existing.sheetnames:
                    ws_existing = wb_existing[sheet_name_wide]
                    for r in range(2, min(ws_existing.max_row + 1, 5000)):
                        val = ws_existing.cell(row=r, column=7).value
                        if val is None or not str(val).strip():
                            continue
                        s = str(val).strip()
                        if " -> " in s or " with " in s:
                            existing_log_lines.append(s)
                wb_existing.close()
            except Exception:
                pass
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            sheet_title = "Combined Summary (overview)" if (overview_df is not None and not overview_df.empty) else "Combined Summary (wide)"
            ws.title = sheet_title
            for row in dataframe_to_rows(sheet_df, index=False, header=True):
                ws.append(row)
            if match_log is not None:
                n_masks, n_tif, lines = match_log
                ws["G1"] = "Logs"
                ws["G2"] = f"{n_masks} mask(s) and {n_tif} tif file(s) matched"
                row = 3
                for line in existing_log_lines:
                    ws.cell(row=row, column=7, value=line)
                    row += 1
                for line in lines:
                    ws.cell(row=row, column=7, value=line)
                    row += 1
            seen_sheet_names = set(wb.sheetnames)
            for p in excel_paths:
                p = Path(p)
                if not p.exists():
                    continue
                per_cell = per_cell_cache.get(p)
                if per_cell is None:
                    continue
                if per_cell.empty:
                    continue
                base_name = p.stem.replace("_analysis", "").replace("_masks_analysis", "")
                sheet_name = _sanitize_sheet_name(base_name)
                if sheet_name in seen_sheet_names:
                    stem = sheet_name[:28] if len(sheet_name) > 28 else sheet_name
                    n = 1
                    while f"{stem}_{n}" in seen_sheet_names:
                        n += 1
                    sheet_name = f"{stem}_{n}"
                seen_sheet_names.add(sheet_name)
                new_ws = wb.create_sheet(title=sheet_name)
                for row in dataframe_to_rows(per_cell, index=False, header=True):
                    new_ws.append(row)
            wb.save(output_path)
        except Exception as e:
            _log(f"Combined summary: write failed — {e}\n")
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
                    # FI columns to rename (optional; merge still runs if only foci columns present)
                    has_fi = [c for c in fi_col_names if c in df.columns]
                    rename = {c: f"{c} ({channel_label})" for c in has_fi}
                    sub = df.rename(columns=rename)
                    merge_on = [c for c in key_cols if c in merged.columns and c in sub.columns]
                    # Include FI and Foci columns from this file (not just "FI" in name)
                    def is_fi_or_foci_col(c: str) -> bool:
                        if c in key_cols:
                            return False
                        up = c.upper()
                        if "FI" in c or "FOCI" in up or c == "Number of Foci":
                            return True
                        if c.startswith("foci_ids_") or "_foci_count" in c:
                            return True
                        return False
                    fi_new = [c for c in sub.columns if c not in merge_on and is_fi_or_foci_col(c) and c not in merged.columns]
                    if merged is None:
                        merged = sub.copy()
                    else:
                        if merge_on and fi_new:
                            merged = merged.merge(sub[merge_on + fi_new], on=merge_on, how="outer")
                        elif merge_on and has_fi:
                            # Renamed FI columns (e.g. "Mean FI Intensity (mScarlet)")
                            fi_cols_to_add = [rename[c] for c in has_fi if rename[c] in sub.columns and rename[c] not in merged.columns]
                            if fi_cols_to_add:
                                merged = merged.merge(sub[merge_on + fi_cols_to_add], on=merge_on, how="outer")
                if summary_df is not None and merged is not None and not merged.empty and "Frame" in merged.columns:
                    with pd.ExcelWriter(merged_path, engine="openpyxl") as writer:
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)
                        merged.to_excel(writer, sheet_name="Per-Cell Data", index=False)
                        # Include Foci sheet from first FI Excel that has one (one row per focus, link by cell_id)
                        for excel_path, _fp, _lbl in fi_entries:
                            try:
                                df_foci = pd.read_excel(excel_path, sheet_name="Foci")
                                if df_foci is not None and not df_foci.empty:
                                    df_foci.to_excel(writer, sheet_name="Foci", index=False)
                                    break
                            except Exception:
                                pass
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
        if self._build_combined_summary_excel(
            [str(p) for p in excel_paths], str(out_path), log_callback=self.parent_window.log_output
        ):
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
        # Foci area (µm²)
        if hasattr(self, "foci_min_area_um2"):
            self.foci_min_area_um2.setValue(0.200)
        if hasattr(self, "foci_max_area_um2"):
            self.foci_max_area_um2.setValue(0.800)
    
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
            ("export_csv", self.export_csv),
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
        if hasattr(self, "foci_min_rel_mode"):
            d["foci_min_rel_mode"] = self.foci_min_rel_mode.currentText()
        if hasattr(self, "foci_min_rel_cell"):
            d["foci_min_rel_cell"] = self.foci_min_rel_cell.value()
        if hasattr(self, "foci_min_separation"):
            d["foci_min_separation"] = self.foci_min_separation.value()
        if hasattr(self, "foci_min_area_um2"):
            d["foci_min_area_um2"] = self.foci_min_area_um2.value()
        if hasattr(self, "foci_max_area_um2"):
            d["foci_max_area_um2"] = self.foci_max_area_um2.value()
        if hasattr(self, "foci_min_peak_raw"):
            d["foci_min_peak_raw"] = self.foci_min_peak_raw.value()
        if hasattr(self, "foci_min_focus_mean"):
            d["foci_min_focus_mean"] = self.foci_min_focus_mean.value()
        if hasattr(self, "foci_exclude_outside_cells"):
            d["foci_exclude_outside_cells"] = self.foci_exclude_outside_cells.isChecked()
        if hasattr(self, "foci_detection_mode"):
            d["foci_detection_mode"] = self.foci_detection_mode.currentIndex()
        path_txt = (self.file_path.text() or "").strip()
        if path_txt:
            d["file_path"] = path_txt
        d["performed"] = getattr(self, "_performed_this_session", False)
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
            ("export_csv", self.export_csv),
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
        if "foci_min_rel_mode" in data and hasattr(self, "foci_min_rel_mode"):
            txt = str(data.get("foci_min_rel_mode", "per cell")).strip().lower()
            if txt == "both":
                idx = self.foci_min_rel_mode.findText("both")
                if idx >= 0:
                    self.foci_min_rel_mode.setCurrentIndex(idx)
            elif "cell" in txt:
                idx = self.foci_min_rel_mode.findText("per cell")
                if idx >= 0:
                    self.foci_min_rel_mode.setCurrentIndex(idx)
            else:
                idx = self.foci_min_rel_mode.findText("per frame")
                if idx >= 0:
                    self.foci_min_rel_mode.setCurrentIndex(idx)
        if "foci_min_rel_cell" in data and hasattr(self, "foci_min_rel_cell"):
            try:
                self.foci_min_rel_cell.setValue(float(data["foci_min_rel_cell"]))
            except (TypeError, ValueError):
                pass
        if "foci_min_rel_mode" in data or "foci_min_rel_cell" in data:
            if hasattr(self, "_update_foci_mode_visibility"):
                self._update_foci_mode_visibility()
        if "foci_min_separation" in data and hasattr(self, "foci_min_separation"):
            try:
                self.foci_min_separation.setValue(int(data["foci_min_separation"]))
            except (TypeError, ValueError):
                pass
        if "foci_min_area_um2" in data and hasattr(self, "foci_min_area_um2"):
            try:
                self.foci_min_area_um2.setValue(float(data["foci_min_area_um2"]))
            except (TypeError, ValueError):
                pass
        elif "foci_min_size" in data and hasattr(self, "foci_min_area_um2"):
            # Legacy: convert px to µm² using current pixel size
            try:
                ps = float(self.pixel_size.value() or 0.108)
                if ps > 0:
                    area_px = int(data["foci_min_size"])
                    self.foci_min_area_um2.setValue(round(area_px * (ps * ps), 3))
            except (TypeError, ValueError):
                pass
        if "foci_max_area_um2" in data and hasattr(self, "foci_max_area_um2"):
            try:
                self.foci_max_area_um2.setValue(float(data["foci_max_area_um2"]))
            except (TypeError, ValueError):
                pass
        elif "foci_max_size" in data and hasattr(self, "foci_max_area_um2"):
            try:
                ps = float(self.pixel_size.value() or 0.108)
                if ps > 0:
                    area_px = int(data["foci_max_size"])
                    self.foci_max_area_um2.setValue(round(area_px * (ps * ps), 3) if area_px > 0 else 0.0)
            except (TypeError, ValueError):
                pass
        if "foci_min_peak_raw" in data and hasattr(self, "foci_min_peak_raw"):
            try:
                self.foci_min_peak_raw.setValue(float(data["foci_min_peak_raw"]))
            except (TypeError, ValueError):
                pass
        if "foci_min_focus_mean" in data and hasattr(self, "foci_min_focus_mean"):
            try:
                self.foci_min_focus_mean.setValue(float(data["foci_min_focus_mean"]))
            except (TypeError, ValueError):
                pass
        if "foci_exclude_outside_cells" in data and hasattr(self, "foci_exclude_outside_cells"):
            self.foci_exclude_outside_cells.setChecked(bool(data["foci_exclude_outside_cells"]))
        if "foci_detection_mode" in data and hasattr(self, "foci_detection_mode"):
            try:
                idx = int(data["foci_detection_mode"])
                if 0 <= idx < self.foci_detection_mode.count():
                    self.foci_detection_mode.setCurrentIndex(idx)
            except (TypeError, ValueError):
                pass
        if "file_path" in data and data.get("file_path"):
            self.file_path.setText(str(data["file_path"]).strip())
        if "performed" in data:
            self._performed_this_session = bool(data["performed"])
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
        # If a previous preview dialog still exists and its worker is running, wait for it then close it.
        # Otherwise overwriting _filter_preview_dialog would leave the old dialog to be destroyed with a running QThread when the tab is destroyed.
        existing = getattr(self, "_filter_preview_dialog", None)
        if existing is not None:
            w = getattr(existing, "_preview_worker", None)
            if w is not None and w.isRunning():
                loading_wait = QProgressDialog("Waiting for previous preview to finish...", None, 0, 0, self)
                loading_wait.setWindowTitle("Please wait")
                loading_wait.setCancelButton(None)
                loading_wait.show()
                QApplication.processEvents()
                while w.isRunning():
                    QApplication.processEvents()
                    w.wait(200)
                loading_wait.close()
            try:
                existing.close()
            except Exception:
                pass
            self._filter_preview_dialog = None

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
                            QMessageBox.warning(self, "Frame Mismatch",
                                              f"FI image has {fi_images.shape[0]} frames but mask has {num_frames} frames.")
                            fi_images = None
                        if do_fi_contour and hasattr(self, 'fi_contour_thickness'):
                            fi_contour_thickness = self.fi_contour_thickness.value()
                    except Exception as e:
                        QMessageBox.warning(self, "FI Load Error",
                                          f"Could not load FI image:\n{str(e)}")
                        fi_images = None
            
            loading_progress.setLabelText("Initializing preview dialog... (80%)")
            loading_progress.setValue(80)
            QApplication.processEvents()
            
            # Create preview dialog (non-modal so user can access main window)
            loading_progress.setLabelText("Ready! (100%)")
            loading_progress.setValue(100)
            QApplication.processEvents()

            class PreviewDialog(QDialog):
                """Waits for filter preview worker to finish before closing; never destroy QThread while running."""
                def closeEvent(self, event):
                    w = getattr(self, "_preview_worker", None)
                    s = getattr(self, "_preview_worker_slot", None)
                    if w is not None and w.isRunning():
                        # Don't block event loop or destroy thread: hide and close when worker finishes
                        write_gui_log_line("Preview", "Filter Effects Preview: worker still running, deferring close until it finishes (dialog hidden)")
                        self._dialog_closed = True
                        self._close_pending = True
                        self.hide()
                        event.ignore()
                        return
                    self._dialog_closed = True
                    if w is not None and s is not None:
                        try:
                            w.finished.disconnect(s)
                        except Exception:
                            pass
                    self._preview_worker = None
                    self._preview_worker_slot = None
                    parent_tab = getattr(self, "_parent_tab", None)
                    if parent_tab is not None:
                        parent_tab._filter_preview_dialog = None
                    super().closeEvent(event)

            preview_dialog = PreviewDialog(self)
            preview_dialog._parent_tab = self  # so main window can find and wait for worker on app close
            self._filter_preview_dialog = preview_dialog
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

            # Stack: page 0 = canvas (shown from start; one progress dialog used for initial load and Re-apply)
            preview_stack = QStackedWidget()
            preview_stack.addWidget(image_canvas)  # index 0
            preview_stack.setCurrentIndex(0)  # show canvas immediately; only the small "Processing" dialog indicates progress
            preview_dialog._preview_stack = preview_stack
            layout.addWidget(preview_stack, 1)

            # Frame navigation (same layout as Segmentation Preview)
            nav_layout = QHBoxLayout()
            nav_layout.addWidget(QLabel("Frame:"))
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_frames)
            frame_spin.setValue(1)
            frame_spin.setToolTip(f"Frame 1–{num_frames}. Type a number and press Enter to jump.")
            nav_layout.addWidget(frame_spin)
            nav_layout.addWidget(QLabel(f"of {num_frames}"))
            nav_layout.addStretch()
            
            # Info label will be added to nav_layout after reapply button
            info_label = QLabel("")
            info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            
            # Import filter functions from analyze_cell_areas
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from bactimate.scripts.analyze_cell_areas import calculate_cell_properties, filter_rod_shaped_cells, filter_by_size
            
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
            preview_dialog._preview_request_id = 0
            preview_dialog._preview_current_requested_frame = 0
            preview_dialog._preview_worker = None
            preview_dialog._preview_progress_dialog = None
            preview_dialog._preview_worker_slot = None
            preview_dialog._dialog_closed = False
            preview_dialog._close_pending = False

            # Worker: runs filter pipeline + builds overlay in background; main thread only scales and displays
            class AnalyzeMasksPreviewWorker(QThread):
                progress = pyqtSignal(int, str)  # percent 0-100, message for progress dialog
                # finished: rid, fid, acc_list, rej_list, all_list, n_obj, err, vis, peaks_for_hover, n_foci, raw_median_fi, min_rel, mode_txt
                finished = pyqtSignal(int, int, list, list, list, int, str, object, object, int, object, object, str)

                def __init__(
                    self,
                    request_id: int,
                    frame_idx: int,
                    mask_cleaned: np.ndarray,
                    params: dict,
                    fi_image: Optional[np.ndarray],
                    display_params: dict,
                    skip_vis: bool,
                ):
                    super().__init__()
                    self._request_id = request_id
                    self._frame_idx = frame_idx
                    self._mask_cleaned = np.asarray(mask_cleaned).copy()
                    self._params = dict(params)
                    self._fi_image = np.asarray(fi_image).copy() if fi_image is not None else None
                    self._display_params = dict(display_params) if display_params else {}
                    self._skip_vis = bool(skip_vis)

                def run(self) -> None:
                    err = ""
                    accepted_list, rejected_list, all_list, num_objects = [], [], [], 0
                    vis, peaks_for_hover = None, None
                    n_foci_total, raw_median_fi, min_rel, mode_txt = 0, None, None, ""
                    try:
                        from bactimate.scripts.analyze_cell_areas import (
                            calculate_cell_properties,
                            filter_rod_shaped_cells,
                            filter_by_size,
                        )
                        params = self._params
                        self.progress.emit(20, "Computing cell properties... (20%)")
                        properties = calculate_cell_properties(
                            self._mask_cleaned,
                            pixel_size=params["pixel_size"],
                            exclude_border=params.get("exclude_border", False),
                        )
                        self.progress.emit(45, "Filtering shape... (45%)")
                        rod_shaped = filter_rod_shaped_cells(
                            properties,
                            min_aspect_ratio=params["min_aspect_ratio"],
                            max_aspect_ratio=params["max_aspect_ratio"],
                            min_eccentricity=params["min_eccentricity"],
                            min_solidity=params["min_solidity"],
                        )
                        self.progress.emit(70, "Applying size filter... (70%)")
                        filtered = filter_by_size(
                            rod_shaped,
                            min_area_um2=params.get("min_area"),
                            max_area_um2=params.get("max_area"),
                            min_width_um=params.get("min_width"),
                            max_width_um=params.get("max_width"),
                        )
                        accepted_labels = {cell["label"] for cell in filtered}
                        all_labels = set(np.unique(self._mask_cleaned)) - {0}
                        rejected_labels = all_labels - accepted_labels
                        accepted_list = list(accepted_labels)
                        rejected_list = list(rejected_labels)
                        all_list = list(all_labels)
                        num_objects = len(all_labels)
                        self.progress.emit(85, "Building overlay... (85%)")
                        if not self._skip_vis:
                            dp = self._display_params
                            progress_cb = lambda p, m: self.progress.emit(p, m)
                            vis, peaks_for_hover, n_foci_total, raw_median_fi, min_rel, mode_txt = build_preview_vis(
                                self._mask_cleaned,
                                set(accepted_list),
                                set(rejected_list),
                                self._fi_image,
                                dp.get("do_fi_contour", False),
                                dp.get("fi_contour_thickness_val", 1),
                                dp.get("run_foci", False),
                                dp.get("foci_snapshot"),
                                progress_callback=progress_cb,
                            )
                        self.progress.emit(90, "Ready to draw... (90%)")
                    except Exception as e:
                        err = str(e)
                    self.finished.emit(
                        self._request_id,
                        self._frame_idx,
                        accepted_list,
                        rejected_list,
                        all_list,
                        num_objects,
                        err,
                        vis,
                        peaks_for_hover if peaks_for_hover is not None else [],
                        n_foci_total,
                        raw_median_fi,
                        min_rel,
                        mode_txt,
                    )

            def _apply_filter_result_to_ui(frame_idx, accepted_list, rejected_list, all_list, num_objects, skip_vis, foci_snapshot=None):
                """Apply filter result to display (called from worker slot). Re-enables controls. foci_snapshot: dict with min_rel, mode, etc. from when Re-apply was clicked."""
                # Hide loading overlay once first frame is ready
                stack = getattr(preview_dialog, "_preview_stack", None)
                if stack is not None and stack.currentIndex() == 1:
                    stack.setCurrentIndex(0)
                mask = masks[frame_idx]
                accepted_labels = set(accepted_list)
                rejected_labels = set(rejected_list)
                all_labels = set(all_list)
                preview_dialog.current_frame = frame_idx
                if skip_vis:
                    try:
                        from matplotlib.figure import Figure
                        from matplotlib.backends.backend_agg import FigureCanvasAgg
                        import io
                        from PyQt6.QtGui import QImage, QPixmap
                        fig = Figure(figsize=(8, 5), facecolor="#2b2b2b")
                        ax = fig.add_subplot(111)
                        ax.set_facecolor("#2b2b2b")
                        ax.tick_params(colors="white")
                        rate = (len(accepted_labels) / len(all_labels) * 100) if all_labels else 0
                        ax.text(0.5, 0.5,
                               f"Visualization skipped due to large number of objects ({num_objects:,})\n\n"
                               f"Statistics:\n"
                               f"• Total objects: {len(all_labels):,}\n"
                               f"• Accepted: {len(accepted_labels):,}\n"
                               f"• Rejected: {len(rejected_labels):,}\n"
                               f"• Acceptance rate: {rate:.1f}%",
                               ha='center', va='center', fontsize=12, color="white",
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        ax.set_title(f"Frame {frame_idx + 1}/{num_frames}: Statistics Only (Visualization Skipped)", color="white")
                        ax.axis('off')
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#2b2b2b", edgecolor="none")
                        buf.seek(0)
                        img = QImage()
                        img.loadFromData(buf.getvalue())
                        pixmap = QPixmap.fromImage(img)
                        image_canvas.set_pixmap(pixmap, (pixmap.width(), pixmap.height()), reset_view=False)
                        image_canvas.set_foci_hover_data(None)
                    except Exception:
                        title_label.setText(f"{Path(mask_path).name}  |  Frame {frame_idx + 1}/{num_frames}  |  Stats only (vis skipped)")
                        info_label.setText(f"Accepted: {len(accepted_labels)} | Rejected: {len(rejected_labels)} | Total: {len(all_labels)}")
                else:
                    peaks_for_hover = []  # filled when run_foci and peaks are drawn; used for tooltips
                    height, width = mask.shape
                    vis = np.zeros((height, width, 3), dtype=np.uint8)
                    if accepted_labels:
                        accepted_array = np.array(list(accepted_labels))
                        accepted_mask = np.isin(mask, accepted_array)
                        vis[accepted_mask, 1] = 255
                    if rejected_labels:
                        rejected_array = np.array(list(rejected_labels))
                        rejected_mask = np.isin(mask, rejected_array)
                        vis[rejected_mask, 0] = 255
                    do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                    fi_contour_thickness_val = self.fi_contour_thickness.value() if hasattr(self, "fi_contour_thickness") else 1
                    run_foci = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
                    # When Run foci is on (or FI contour), show FI as background so representation reflects reality
                    if fi_images is not None and self.analyze_fi.isChecked() and (do_fi_contour or run_foci):
                        try:
                            from skimage.segmentation import find_boundaries
                            from skimage.morphology import binary_dilation
                            fi_image = fi_images[frame_idx]
                            if fi_image.ndim > 2:
                                fi_image = fi_image[:, :, 0] if fi_image.shape[2] < 10 else fi_image[0, :, :]
                            accepted_mask_full = np.zeros_like(mask, dtype=bool)
                            if accepted_labels:
                                accepted_mask_full = np.isin(mask, np.array(list(accepted_labels)))
                            fi_norm = np.asarray(fi_image, dtype=np.float32)
                            if fi_norm.size > 0:
                                # Linear min–max stretch (no percentile); full image range maps to 0–255
                                p_low, p_high = float(np.min(fi_norm)), float(np.max(fi_norm))
                                if p_high > p_low:
                                    fi_norm = (fi_norm - p_low) / (p_high - p_low) * 255
                                    fi_norm = np.clip(fi_norm, 0, 255).astype(np.float32)
                                else:
                                    fi_norm = np.zeros_like(fi_norm)
                            else:
                                fi_norm = np.zeros_like(fi_norm)
                            vis_fi = np.stack([fi_norm, fi_norm, fi_norm], axis=-1).astype(np.uint8)
                            # Green tint accepted, red tint rejected (blend so FI remains visible)
                            if np.any(accepted_mask_full):
                                alpha_g = 0.55
                                vis_fi[accepted_mask_full, 0] = (vis_fi[accepted_mask_full, 0].astype(np.float32) * (1 - alpha_g)).astype(np.uint8)
                                vis_fi[accepted_mask_full, 1] = (vis_fi[accepted_mask_full, 1].astype(np.float32) * (1 - alpha_g) + 255 * alpha_g).astype(np.uint8)
                                vis_fi[accepted_mask_full, 2] = (vis_fi[accepted_mask_full, 2].astype(np.float32) * (1 - alpha_g)).astype(np.uint8)
                            if rejected_labels:
                                rej_arr = np.array(list(rejected_labels))
                                rej_mask = np.isin(mask, rej_arr)
                                if np.any(rej_mask):
                                    alpha_r = 0.55
                                    vis_fi[rej_mask, 0] = (vis_fi[rej_mask, 0].astype(np.float32) * (1 - alpha_r) + 255 * alpha_r).astype(np.uint8)
                                    vis_fi[rej_mask, 1] = (vis_fi[rej_mask, 1].astype(np.float32) * (1 - alpha_r)).astype(np.uint8)
                                    vis_fi[rej_mask, 2] = (vis_fi[rej_mask, 2].astype(np.float32) * (1 - alpha_r)).astype(np.uint8)
                            # Draw boundaries between cells so multiple accepted cells are visible
                            if accepted_labels and np.any(accepted_mask_full):
                                for lab in accepted_labels:
                                    cell_binary = (mask == lab)
                                    if not np.any(cell_binary):
                                        continue
                                    boundaries = find_boundaries(cell_binary, mode="outer")
                                    if np.any(boundaries):
                                        vis_fi[boundaries, 0] = 0
                                        vis_fi[boundaries, 1] = 0
                                        vis_fi[boundaries, 2] = 0
                            if do_fi_contour and np.any(accepted_mask_full):
                                temp_mask = np.zeros_like(mask)
                                temp_mask[accepted_mask_full] = mask[accepted_mask_full]
                                boundaries = find_boundaries(temp_mask, mode="inner")
                                contour_mask = boundaries & accepted_mask_full
                                if fi_contour_thickness_val > 1:
                                    for _ in range(fi_contour_thickness_val - 1):
                                        contour_mask = binary_dilation(contour_mask) & accepted_mask_full
                                alpha = 0.6
                                vis_fi[contour_mask, 0] = (vis_fi[contour_mask, 0].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                                vis_fi[contour_mask, 1] = (vis_fi[contour_mask, 1].astype(np.float32) * (1 - alpha) + 0 * alpha).astype(np.uint8)
                                vis_fi[contour_mask, 2] = (vis_fi[contour_mask, 2].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                            vis[:] = vis_fi[:]
                        except Exception:
                            pass
                    n_foci_total = 0
                    raw_median_fi = None
                    min_rel = None  # used in legend; set inside foci block when frame filter is applied
                    mode_txt = ""   # foci mode for legend; set inside foci block
                    if run_foci and fi_images is not None and self.analyze_fi.isChecked():
                        try:
                            from bactimate.modules.foci_utils import detect_peaks_skimage as _detect_foci, get_focus_blob_mask
                        except ImportError:
                            _detect_foci = None
                            get_focus_blob_mask = None
                        if _detect_foci is not None:
                            try:
                                fi_image = fi_images[frame_idx]
                                if fi_image.ndim > 2:
                                    fi_image = fi_image[:, :, 0] if fi_image.shape[2] < 10 else fi_image[0, :, :]
                                if fi_image.shape[:2] == vis.shape[:2]:
                                    raw_median_fi = float(np.median(np.asarray(fi_image, dtype=np.float64)))
                                    # Use foci_snapshot passed from callback (set when Re-apply was clicked) so value is correct
                                    sn = foci_snapshot or {}
                                    mode_txt = (sn.get("mode") or (self.foci_min_rel_mode.currentText().strip().lower() if getattr(self, "foci_min_rel_mode", None) else "") or "per_cell")
                                    if mode_txt in ("per_cell", "per cell"):
                                        min_rel = None
                                    else:
                                        min_rel = sn.get("min_rel")
                                        if min_rel is None:
                                            min_rel = self.foci_min_rel.value() if hasattr(self, "foci_min_rel") else 2.0
                                    min_sep = sn.get("min_sep")
                                    if min_sep is None:
                                        min_sep = self.foci_min_separation.value() if hasattr(self, "foci_min_separation") else 2
                                    min_sz = sn.get("min_sz")
                                    max_sz = sn.get("max_sz")
                                    if min_sz is None or max_sz is None:
                                        _min_px, _max_px = self._foci_area_um2_to_px()
                                        if min_sz is None:
                                            min_sz = _min_px
                                        if max_sz is None:
                                            max_sz = _max_px
                                    min_peak_raw = sn.get("min_peak_raw")
                                    if min_peak_raw is None:
                                        min_peak_raw = self.foci_min_peak_raw.value() if hasattr(self, "foci_min_peak_raw") else 0
                                    min_focus_mean = sn.get("min_focus_mean")
                                    if min_focus_mean is None:
                                        min_focus_mean = self.foci_min_focus_mean.value() if hasattr(self, "foci_min_focus_mean") else 0
                                    peaks = _detect_foci(
                                        fi_image,
                                        min_rel=min_rel,
                                        min_distance=min_sep,
                                        min_size=min_sz,
                                        max_size=max_sz,
                                        min_peak_raw=min_peak_raw if min_peak_raw and float(min_peak_raw) > 0 else None,
                                        min_focus_mean=min_focus_mean if min_focus_mean and float(min_focus_mean) > 0 else None,
                                        labeled_mask=mask,
                                    )
                                    exclude_outside = sn.get("exclude_outside")
                                    if exclude_outside is None:
                                        exclude_outside = bool(getattr(self, "foci_exclude_outside_cells", None) and self.foci_exclude_outside_cells.isChecked())
                                    if exclude_outside:
                                        peaks = [p for p in peaks if 0 <= int(p["y"]) < mask.shape[0] and 0 <= int(p["x"]) < mask.shape[1] and mask[int(p["y"]), int(p["x"])] in accepted_labels]
                                    # Apply per-cell filter in preview so counts match Run/Excel
                                    if mode_txt and "cell" in mode_txt and accepted_labels:
                                        cell_factor = sn.get("cell_factor") or sn.get("min_rel") or (self.foci_min_rel.value() if hasattr(self, "foci_min_rel") else 2.0)
                                        try:
                                            fi_flat = np.asarray(fi_image, dtype=np.float64)
                                            if fi_flat.ndim > 2:
                                                fi_flat = fi_flat[:, :, 0] if fi_flat.shape[2] < 10 else fi_flat[0, :, :]
                                            label_to_mean_fi = {}
                                            for lab in accepted_labels:
                                                cell_mask = (mask == lab)
                                                if np.any(cell_mask) and cell_mask.shape == fi_flat.shape[:2]:
                                                    cell_pixels = fi_flat[cell_mask]
                                                    mu = float(np.mean(cell_pixels))
                                                    if np.isfinite(mu) and mu > 0:
                                                        label_to_mean_fi[lab] = mu
                                            peaks_filtered = []
                                            for p in peaks:
                                                yi, xi = int(p["y"]), int(p["x"])
                                                if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
                                                    continue
                                                lab = int(mask[yi, xi])
                                                mean_fi = label_to_mean_fi.get(lab)
                                                if mean_fi is None or mean_fi <= 0:
                                                    continue
                                                intensity = p.get("intensity")
                                                if intensity is None or not np.isfinite(intensity):
                                                    continue
                                                if float(intensity) / float(mean_fi) >= float(cell_factor):
                                                    peaks_filtered.append(p)
                                            peaks = peaks_filtered
                                        except Exception:
                                            pass
                                    n_foci_total = len(peaks)
                                    FOCI_CAP = 500
                                    orange = np.array([255, 165, 0], dtype=np.uint8)
                                    h, w = vis.shape[:2]
                                    peaks_to_draw = peaks  # same list we counted
                                    if get_focus_blob_mask is not None:
                                        from skimage.segmentation import find_boundaries
                                        for i, p in enumerate(peaks_to_draw):
                                            if i >= FOCI_CAP:
                                                break
                                            yi, xi = int(p["y"]), int(p["x"])
                                            blob = get_focus_blob_mask(fi_image, yi, xi, threshold_frac=0.8, max_radius=5, labeled_mask=mask)
                                            if np.any(blob):
                                                vis[blob] = orange
                                        # Thin contour around each focus so adjacent foci look like two
                                        for i, p in enumerate(peaks_to_draw):
                                            if i >= FOCI_CAP:
                                                break
                                            yi, xi = int(p["y"]), int(p["x"])
                                            blob = get_focus_blob_mask(fi_image, yi, xi, threshold_frac=0.8, max_radius=5, labeled_mask=mask)
                                            if np.any(blob):
                                                boundaries = find_boundaries(blob, mode="outer")
                                                if np.any(boundaries):
                                                    vis[boundaries, 0] = 0
                                                    vis[boundaries, 1] = 0
                                                    vis[boundaries, 2] = 0
                                    else:
                                        for i, p in enumerate(peaks_to_draw):
                                            if i >= FOCI_CAP:
                                                break
                                            yi, xi = int(p["y"]), int(p["x"])
                                            r = max(2, min(5, int(round((max(1, p.get("size_px", 1)) / np.pi) ** 0.5))))
                                            for dy in range(-r, r + 1):
                                                for dx in range(-r, r + 1):
                                                    if dy * dy + dx * dx <= r * r:
                                                        yk, xk = yi + dy, xi + dx
                                                        if 0 <= yk < h and 0 <= xk < w:
                                                            vis[yk, xk] = orange
                                    peaks_for_hover.clear()
                                    peaks_for_hover.extend(peaks_to_draw[:FOCI_CAP])
                            except Exception:
                                pass
                    title = f"Frame {frame_idx + 1}/{num_frames}"
                    legend = f"Green=Accepted ({len(accepted_labels)} cells)  Red=Rejected ({len(rejected_labels)})"
                    if fi_images is not None and self.analyze_fi.isChecked() and do_fi_contour:
                        legend += f"  Magenta=Contour (t={fi_contour_thickness_val})"
                    if run_foci:
                        legend += f"  |  Orange=foci ({n_foci_total})"
                        if raw_median_fi is not None and min_rel is not None:
                            legend += f"  [raw median FI: {raw_median_fi:.0f}  (× {min_rel} = threshold)]"
                        elif raw_median_fi is not None and mode_txt and "cell" in mode_txt:
                            legend += f"  [per-cell filter applied]"
                    title_label.setText(f"{Path(mask_path).name}  |  {title}  |  {legend}")
                    dummy_mask = np.zeros(vis.shape[:2], dtype=np.int32)
                    vis_scaled, _ = scale_for_preview(vis, dummy_mask)
                    pixmap = to_qpixmap(vis_scaled)
                    image_canvas.set_pixmap(pixmap, (vis_scaled.shape[1], vis_scaled.shape[0]), reset_view=False)
                    # Hover tooltips for foci: map peak coords to display coords and pass to canvas
                    if run_foci and peaks_for_hover:
                        h_vis, w_vis = vis.shape[0], vis.shape[1]
                        h_s, w_s = vis_scaled.shape[0], vis_scaled.shape[1]
                        scale_x = w_s / w_vis if w_vis else 1
                        scale_y = h_s / h_vis if h_vis else 1
                        # Cell mean FI per label for tooltip (when FI available)
                        label_to_mean_fi = {}
                        if fi_images is not None and frame_idx < len(fi_images):
                            fi_flat = np.asarray(fi_images[frame_idx], dtype=np.float64)
                            if fi_flat.ndim > 2:
                                fi_flat = fi_flat[:, :, 0] if fi_flat.shape[2] < 10 else fi_flat[0, :, :]
                            if fi_flat.shape[:2] == mask.shape[:2]:
                                for lab in accepted_labels:
                                    cell_mask = (mask == lab)
                                    if np.any(cell_mask):
                                        cell_pixels = fi_flat[cell_mask]
                                        label_to_mean_fi[lab] = float(np.mean(cell_pixels)) if cell_pixels.size > 0 else np.nan
                        mode_txt = ((foci_snapshot or {}).get("mode") or "per_cell").strip().lower()
                        hover_list = []
                        for p in peaks_for_hover:
                            yi, xi = int(p["y"]), int(p["x"])
                            x_disp = xi * scale_x
                            y_disp = yi * scale_y
                            cell_mean = None
                            area_px = max(1, int(p.get("size_px", 0)))
                            # Per-focus hover radius in display coords so tooltip only when cursor is on/near this focus
                            r_img = (area_px / np.pi) ** 0.5
                            r_disp = max(4.0, min(20.0, r_img * min(scale_x, scale_y)))
                            parts = ["Peak at (%d, %d)" % (xi, yi), "peak raw intensity: %.0f" % p.get("intensity", 0)]
                            try:
                                ps = float(self.pixel_size.value()) if hasattr(self, "pixel_size") else 0
                                if ps > 0 and area_px >= 0:
                                    area_um2 = area_px * (ps ** 2)
                                    parts.append("area (µm²): %.3f" % area_um2)
                            except Exception:
                                pass
                            if "mean_intensity" in p:
                                parts.append("mean_intensity: %.0f" % p["mean_intensity"])
                            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                                cell_id = int(mask[yi, xi])
                                cell_mean = label_to_mean_fi.get(cell_id)
                                if cell_mean is not None and np.isfinite(cell_mean):
                                    parts.append("cell_id: %d  cell mean FI: %.0f" % (cell_id, cell_mean))
                            if "cell" in mode_txt and cell_mean is not None and cell_mean > 0:
                                peak_val = p.get("intensity", 0) or 0
                                rel_cell = peak_val / cell_mean
                                parts.append("rel_intensity: %.2f (peak / cell mean FI)" % rel_cell)
                            elif "rel_intensity" in p:
                                parts.append("rel_intensity: %.2f (peak / frame median)" % p["rel_intensity"])
                            hover_list.append((x_disp, y_disp, "\n".join(parts), r_disp))
                        image_canvas.set_foci_hover_data(hover_list)
                    else:
                        image_canvas.set_foci_hover_data(None)
                info_label.setText(f"Accepted: {len(accepted_labels)} | Rejected: {len(rejected_labels)} | Total: {len(all_labels)}")
                reapply_btn.setEnabled(True)
                frame_spin.setEnabled(True)
                preview_dialog._updating_preview = False

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
                    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                    progress_dialog.show()
                    progress_dialog.raise_()
                    progress_dialog.activateWindow()
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
                            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                            progress_dialog.show()
                            progress_dialog.raise_()
                            progress_dialog.activateWindow()
                            progress_dialog.canceled.connect(check_cancelled)
                            QApplication.processEvents()
                    
                    # Auto-skip visualization for extremely large datasets
                    if num_objects > MAX_OBJECTS_VIZ:
                        skip_visualization = True
                    
                    # Snapshot foci params when Re-apply is clicked (force commit then read so spinbox value is current)
                    fw = QApplication.focusWidget()
                    if fw is not None:
                        fw.clearFocus()
                    QApplication.processEvents()
                    _mode = (self.foci_min_rel_mode.currentText().strip().lower() if getattr(self, "foci_min_rel_mode", None) else "") or "per_cell"
                    ps = float(self.pixel_size.value() or 0.108)
                    min_sz, max_sz = self._foci_area_um2_to_px(ps)
                    detection_mode = self.foci_detection_mode.currentIndex() if getattr(self, "foci_detection_mode", None) else 0
                    preview_dialog._preview_foci_snapshot = {
                        "detection_mode": detection_mode,
                        "min_rel": self.foci_min_rel.value() if hasattr(self, "foci_min_rel") else 2.0,
                        "mode": _mode,
                        "cell_factor": (self.foci_min_rel_cell.value() if _mode == "both" and hasattr(self, "foci_min_rel_cell") else self.foci_min_rel.value()) if hasattr(self, "foci_min_rel") else 2.0,
                        "min_sep": self.foci_min_separation.value() if hasattr(self, "foci_min_separation") else 2,
                        "min_sz": min_sz,
                        "max_sz": max_sz,
                        "min_peak_raw": self.foci_min_peak_raw.value() if hasattr(self, "foci_min_peak_raw") else 0,
                        "min_focus_mean": self.foci_min_focus_mean.value() if hasattr(self, "foci_min_focus_mean") else 0,
                        "exclude_outside": bool(getattr(self, "foci_exclude_outside_cells", None) and self.foci_exclude_outside_cells.isChecked()),
                    }
                    
                    # Get current parameters from main window
                    params = get_current_parameters()
                    existing = getattr(preview_dialog, "_preview_worker", None)
                    if existing is not None and existing.isRunning():
                        preview_dialog._preview_request_id += 1
                        preview_dialog._preview_current_requested_frame = frame_idx
                        info_label.setText("Updating preview... (requested frame %s)" % (frame_idx + 1))
                        preview_dialog._updating_preview = False
                        return
                    preview_dialog._preview_request_id += 1
                    preview_dialog._preview_current_requested_frame = frame_idx
                    preview_dialog._preview_skip_vis = skip_visualization
                    if progress_dialog:
                        progress_dialog.setLabelText("Updating preview...")
                        QApplication.processEvents()
                        preview_dialog._preview_progress_dialog = progress_dialog

                    def _on_preview_worker_finished(rid, fid, acc_list, rej_list, all_list, n_obj, err, vis, peaks_for_hover, n_foci_total, raw_median_fi, min_rel, mode_txt):
                        try:
                            if getattr(preview_dialog, "_close_pending", False):
                                write_gui_log_line("Preview", "Filter Effects Preview: worker finished, closing dialog (was deferred)")
                                w = getattr(preview_dialog, "_preview_worker", None)
                                s = getattr(preview_dialog, "_preview_worker_slot", None)
                                if w is not None and s is not None:
                                    try:
                                        w.finished.disconnect(s)
                                    except Exception:
                                        pass
                                preview_dialog._preview_worker = None
                                preview_dialog._preview_worker_slot = None
                                preview_dialog.close()
                                return
                            if getattr(preview_dialog, "_dialog_closed", False):
                                return
                            if rid != getattr(preview_dialog, "_preview_request_id", -1):
                                update_preview(getattr(preview_dialog, "_preview_current_requested_frame", 0))
                                return
                            prog = getattr(preview_dialog, "_preview_progress_dialog", None)
                            preview_dialog._preview_worker = None
                            if err:
                                if prog is not None:
                                    try:
                                        prog.close()
                                    except Exception:
                                        pass
                                    preview_dialog._preview_progress_dialog = None
                                stack = getattr(preview_dialog, "_preview_stack", None)
                                if stack is not None and stack.currentIndex() == 1:
                                    stack.setCurrentIndex(0)
                                info_label.setText("Error: %s" % (err[:80]))
                                reapply_btn.setEnabled(True)
                                frame_spin.setEnabled(True)
                                preview_dialog._updating_preview = False
                                QMessageBox.warning(preview_dialog, "Preview Error", "Could not update preview:\n%s" % err)
                                return
                            foci_snapshot = getattr(preview_dialog, "_preview_foci_snapshot", None)
                            if prog is not None:
                                try:
                                    prog.setLabelText("Drawing preview... (95%)")
                                    prog.setValue(95)
                                    QApplication.processEvents()
                                except Exception:
                                    pass
                            if vis is not None:
                                # Fast path: overlay already built in worker; main thread only scales and displays
                                accepted_labels = set(acc_list)
                                rejected_labels = set(rej_list)
                                preview_dialog.current_frame = fid
                                stack = getattr(preview_dialog, "_preview_stack", None)
                                if stack is not None and stack.currentIndex() == 1:
                                    stack.setCurrentIndex(0)
                                dummy_mask = np.zeros(vis.shape[:2], dtype=np.int32)
                                vis_scaled, _ = scale_for_preview(vis, dummy_mask)
                                pixmap = to_qpixmap(vis_scaled)
                                image_canvas.set_pixmap(pixmap, (vis_scaled.shape[1], vis_scaled.shape[0]), reset_view=False)
                                do_fi_contour = bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked())
                                fi_contour_thickness_val = self.fi_contour_thickness.value() if hasattr(self, "fi_contour_thickness") else 1
                                run_foci = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
                                title = f"Frame {fid + 1}/{num_frames}"
                                legend = f"Green=Accepted ({len(accepted_labels)} cells)  Red=Rejected ({len(rejected_labels)})"
                                if fi_images is not None and self.analyze_fi.isChecked() and do_fi_contour:
                                    legend += f"  Magenta=Contour (t={fi_contour_thickness_val})"
                                if run_foci:
                                    legend += f"  |  Orange=foci ({n_foci_total})"
                                    if raw_median_fi is not None and min_rel is not None:
                                        legend += f"  [raw median FI: {raw_median_fi:.0f}  (× {min_rel} = threshold)]"
                                    elif raw_median_fi is not None and mode_txt and "cell" in mode_txt:
                                        legend += f"  [per-cell filter applied]"
                                title_label.setText(f"{Path(mask_path).name}  |  {title}  |  {legend}")
                                if peaks_for_hover and run_foci:
                                    h_vis, w_vis = vis.shape[0], vis.shape[1]
                                    h_s, w_s = vis_scaled.shape[0], vis_scaled.shape[1]
                                    scale_x = w_s / w_vis if w_vis else 1
                                    scale_y = h_s / h_vis if h_vis else 1
                                    mask = masks[fid]
                                    label_to_mean_fi = {}
                                    if fi_images is not None and fid < len(fi_images):
                                        fi_flat = np.asarray(fi_images[fid], dtype=np.float64)
                                        if fi_flat.ndim > 2:
                                            fi_flat = fi_flat[:, :, 0] if fi_flat.shape[2] < 10 else fi_flat[0, :, :]
                                        if fi_flat.shape[:2] == mask.shape[:2]:
                                            for lab in accepted_labels:
                                                cell_mask = (mask == lab)
                                                if np.any(cell_mask):
                                                    cell_pixels = fi_flat[cell_mask]
                                                    label_to_mean_fi[lab] = float(np.mean(cell_pixels)) if cell_pixels.size > 0 else np.nan
                                    mode_txt_hover = ((foci_snapshot or {}).get("mode") or "per_cell").strip().lower()
                                    ps = float(self.pixel_size.value()) if hasattr(self, "pixel_size") else 0
                                    hover_list = []
                                    for p in peaks_for_hover:
                                        yi, xi = int(p["y"]), int(p["x"])
                                        x_disp = xi * scale_x
                                        y_disp = yi * scale_y
                                        cell_mean = None
                                        area_px = max(1, int(p.get("size_px", 0)))
                                        r_img = (area_px / np.pi) ** 0.5
                                        r_disp = max(4.0, min(20.0, r_img * min(scale_x, scale_y)))
                                        parts = ["Peak at (%d, %d)" % (xi, yi), "peak raw intensity: %.0f" % p.get("intensity", 0)]
                                        if ps > 0 and area_px >= 0:
                                            parts.append("area (µm²): %.3f" % (area_px * (ps ** 2)))
                                        if "mean_intensity" in p:
                                            parts.append("mean_intensity: %.0f" % p["mean_intensity"])
                                        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                                            cell_id = int(mask[yi, xi])
                                            cell_mean = label_to_mean_fi.get(cell_id)
                                            if cell_mean is not None and np.isfinite(cell_mean):
                                                parts.append("cell_id: %d  cell mean FI: %.0f" % (cell_id, cell_mean))
                                        if "cell" in mode_txt_hover and cell_mean is not None and cell_mean > 0:
                                            peak_val = p.get("intensity", 0) or 0
                                            parts.append("rel_intensity: %.2f (peak / cell mean FI)" % (peak_val / cell_mean))
                                        elif "rel_intensity" in p:
                                            parts.append("rel_intensity: %.2f (peak / frame median)" % p["rel_intensity"])
                                        hover_list.append((x_disp, y_disp, "\n".join(parts), r_disp))
                                    image_canvas.set_foci_hover_data(hover_list)
                                else:
                                    image_canvas.set_foci_hover_data(None)
                                info_label.setText(f"Accepted: {len(accepted_labels)} | Rejected: {len(rejected_labels)} | Total: {len(all_list)}")
                                reapply_btn.setEnabled(True)
                                frame_spin.setEnabled(True)
                                preview_dialog._updating_preview = False
                            else:
                                _apply_filter_result_to_ui(
                                    fid, acc_list, rej_list, all_list, n_obj,
                                    True,
                                    foci_snapshot=foci_snapshot,
                                )
                            if prog is not None:
                                try:
                                    prog.setLabelText("Done. (100%)")
                                    prog.setValue(100)
                                    QApplication.processEvents()
                                    prog.close()
                                except Exception:
                                    pass
                                preview_dialog._preview_progress_dialog = None
                        except RuntimeError:
                            pass  # Dialog or widgets already destroyed (e.g. user closed preview)
                        except Exception as e:
                            reapply_btn.setEnabled(True)
                            frame_spin.setEnabled(True)
                            if hasattr(preview_dialog, "_updating_preview"):
                                preview_dialog._updating_preview = False
                            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                            log_msg = f"\n--- Filter preview re-apply failed (frame {fid}) ---\n{tb}"
                            write_crash_log(log_msg)
                            # Always print to stderr so traceback is visible when running from terminal
                            sys.stderr.write(log_msg)
                            sys.stderr.flush()
                            try:
                                QMessageBox.warning(
                                    preview_dialog, "Preview Error",
                                    "Re-apply failed. Full traceback was written to the debug log (Logs folder).\n\nError: %s" % (str(e)[:300])
                                )
                            except Exception:
                                pass

                    def _on_preview_progress(percent: int, msg: str):
                        p = getattr(preview_dialog, "_preview_progress_dialog", None)
                        if p is not None and not cancelled:
                            try:
                                p.setValue(percent)
                                p.setLabelText(msg)
                                QApplication.processEvents()
                            except Exception:
                                pass

                    fi_image_for_worker = None
                    if fi_images is not None and frame_idx < len(fi_images):
                        fi_img = fi_images[frame_idx]
                        if fi_img is not None:
                            fi_image_for_worker = np.asarray(fi_img)
                            if fi_image_for_worker.ndim > 2:
                                fi_image_for_worker = fi_image_for_worker[:, :, 0] if fi_image_for_worker.shape[2] < 10 else fi_image_for_worker[0, :, :]
                    display_params = {
                        "do_fi_contour": bool(getattr(self, "analyze_fi_contour", None) and self.analyze_fi_contour.isChecked()),
                        "fi_contour_thickness_val": self.fi_contour_thickness.value() if hasattr(self, "fi_contour_thickness") else 1,
                        "run_foci": bool(getattr(self, "run_foci", None) and self.run_foci.isChecked()),
                        "foci_snapshot": getattr(preview_dialog, "_preview_foci_snapshot", None),
                    }
                    worker = AnalyzeMasksPreviewWorker(
                        preview_dialog._preview_request_id,
                        frame_idx,
                        mask_cleaned,
                        params,
                        fi_image_for_worker,
                        display_params,
                        skip_visualization,
                    )
                    preview_dialog._preview_worker_slot = _on_preview_worker_finished
                    worker.progress.connect(_on_preview_progress)
                    worker.finished.connect(_on_preview_worker_finished)
                    preview_dialog._preview_worker = worker
                    worker.start()
                    preview_dialog._updating_preview = False
                    return
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                    reapply_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
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
            
            # Commit on Enter/focus only (no live valueChanged) so typing "55" doesn't jump to 5
            def on_frame_changed():
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    if frame_num != preview_dialog.current_frame:
                        go_to_frame(frame_num)
                else:
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)

            frame_spin.editingFinished.connect(on_frame_changed)
            
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
            tb = traceback.format_exc()
            log_msg = f"\n--- Filter preview failed (Show preview clicked) ---\n{tb}"
            write_crash_log(log_msg)
            sys.stderr.write(log_msg)
            sys.stderr.flush()
            QMessageBox.critical(
                self, "Preview Error",
                f"Could not create preview.\n\nError: {str(e)[:200]}\n\nFull traceback was written to the debug log (Logs folder) and to stderr if you run from terminal."
            )
    
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
            if getattr(self, "analyze_cancel_btn", None) is not None:
                self.analyze_cancel_btn.setVisible(True)
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
                base_dir = base_dir_for_mask(mask_path)
                stem = stem_from_mask_path(mask_path)
                analysis_dir = analysis_dir_from_base(base_dir)
                if batch_fi_on:
                    fi_paths = self._resolve_fi_paths_for_mask(mask_path)
                    run_foci_on = bool(getattr(self, "run_foci", None) and self.run_foci.isChecked())
                    if not fi_paths:
                        # No _fi file: if Run foci is on, use _foci file so foci detection can run
                        foci_paths = self._resolve_foci_paths_for_mask(mask_path) if run_foci_on else []
                        if run_foci_on and foci_paths:
                            self.folder_phase_per_mask[mask_path] = self._resolve_phase_path_for_mask(mask_path)
                            for fp in foci_paths:
                                suf = self._fi_output_suffix(fp, stem)
                                self.folder_fi_tasks.append((mask_path, fp, analysis_dir / f"{stem}_analysis{suf}.xlsx"))
                        else:
                            self.folder_fi_tasks.append((mask_path, None, analysis_dir / f"{stem}_analysis.xlsx"))
                            self.folder_phase_per_mask[mask_path] = None
                            self.parent_window.log_output(f"  No FI/foci file for {mask_path.name} (expected e.g. {stem}_fi.tif or {stem}_foci.tif)\n")
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

            # Pre-run validation: paths exist and mask/FI shape match for each task
            validation_errors = []
            for mask_path, fi_path, _ in self.folder_fi_tasks:
                ok, missing = paths_exist([mask_path])
                if not ok:
                    validation_errors.append(f"{mask_path.name}: mask file not found.")
                    continue
                if fi_path is not None:
                    ok, missing = paths_exist([fi_path])
                    if not ok:
                        validation_errors.append(f"{mask_path.name}: FI file not found.")
                        continue
                    ok, err = mask_fi_spatial_shape_match(mask_path, fi_path)
                    if not ok and err:
                        validation_errors.append(f"{mask_path.name}: {err}")
            if validation_errors:
                QMessageBox.warning(
                    self,
                    "Validation failed",
                    "Cannot start analysis:\n\n" + "\n\n".join(validation_errors[:10])
                    + ("\n\n… and more." if len(validation_errors) > 10 else ""),
                )
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

            # Pre-run validation: FI path and mask/FI shape match
            if self.analyze_fi.isChecked() and getattr(self, "_single_fi_path", None) and self._single_fi_path.exists():
                fi_path = self._single_fi_path
                ok, err = mask_fi_spatial_shape_match(input_file, fi_path)
                if not ok and err:
                    QMessageBox.warning(
                        self,
                        "Validation failed",
                        f"Cannot start analysis:\n\n{err}",
                    )
                    return

            base_dir = base_dir_for_mask(input_file)
            output_path = analysis_dir_from_base(base_dir) / f"{input_file.stem}_analysis.xlsx"
            self._last_analysis_output_dir = output_path.parent  # for completion popup "Open output folder"

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
            if getattr(self, 'export_csv', None) and self.export_csv.isChecked():
                cmd_args.append('--export-csv')
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
                    cmd_args.extend(['--foci-detection-mode', str(self.foci_detection_mode.currentIndex() if getattr(self, 'foci_detection_mode', None) else 0)])
                    cmd_args.extend(['--foci-min-rel', str(self.foci_min_rel.value())])
                    mode_txt = self.foci_min_rel_mode.currentText().strip().lower() if getattr(self, 'foci_min_rel_mode', None) else ''
                    if mode_txt == 'both':
                        cmd_args.extend(['--foci-min-rel-mode', 'both'])
                        cmd_args.extend(['--foci-min-rel-cell', str(self.foci_min_rel_cell.value())])
                    elif 'cell' in mode_txt:
                        cmd_args.extend(['--foci-min-rel-mode', 'per_cell'])
                    else:
                        cmd_args.extend(['--foci-min-rel-mode', 'per_frame'])
                    cmd_args.extend(['--foci-min-separation', str(self.foci_min_separation.value())])
                    min_sz, max_sz = self._foci_area_um2_to_px()
                    cmd_args.extend(['--foci-min-size', str(min_sz)])
                    if max_sz is not None and max_sz > 0:
                        cmd_args.extend(['--foci-max-size', str(max_sz)])
                    if getattr(self, "foci_min_peak_raw", None) and self.foci_min_peak_raw.value() > 0:
                        cmd_args.extend(['--foci-min-peak-raw', str(int(self.foci_min_peak_raw.value()))])
                    if getattr(self, "foci_min_focus_mean", None) and self.foci_min_focus_mean.value() > 0:
                        cmd_args.extend(['--foci-min-focus-mean', str(int(self.foci_min_focus_mean.value()))])
                    if getattr(self, "foci_exclude_outside_cells", None) and self.foci_exclude_outside_cells.isChecked():
                        cmd_args.append('--foci-exclude-outside-cells')

            # Run in thread
            self.analyze_btn.setEnabled(False)
            self.analyze_btn.setText("Analyzing...")
            if getattr(self, "analyze_cancel_btn", None) is not None:
                self.analyze_cancel_btn.setVisible(True)
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
            # All tasks processed (or user cancelled)
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("Analyze Masks")
            if getattr(self, "analyze_cancel_btn", None) is not None:
                self.analyze_cancel_btn.setVisible(False)
            self.parent_window.progress_bar.setVisible(False)
            if not getattr(self, "_folder_run_cancelled", False):
                self._performed_this_session = True
            self.workflow_step_finished.emit(True)
            if getattr(self, "_folder_run_cancelled", False):
                self._folder_run_cancelled = False
                self.parent_window.log_output("Analysis cancelled.\n")
                return
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
            combined_summary_path = None
            if getattr(self, "create_combined_summary", None) and getattr(self, "create_combined_summary").isChecked() and n_ok > 0:
                paths = getattr(self, "folder_output_paths", [])
                if paths and self.folder_files:
                    folder = base_dir_for_mask(Path(self.folder_files[0]))
                    out_path = analysis_dir_from_base(folder) / f"{folder.name}_combined_summary.xlsx"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    match_log = self._match_log_from_folder_tasks()
                    if self._build_combined_summary_excel(
                        paths, str(out_path), match_log=match_log, log_callback=self.parent_window.log_output
                    ):
                        combined_summary_path = out_path
                        self.parent_window.log_output(f"Combined summary saved to: {out_path}\n")
                        if Path(self.folder_files[0]).parent != folder:
                            self.parent_window.log_output(
                                f"  (Masks are in a subfolder; combined summary is in the parent folder's _analysis.)\n"
                            )
                    else:
                        self.parent_window.log_output(
                            f"Combined summary not created (no Summary sheets or build failed). Paths used: {len(paths)}\n"
                        )
                elif not paths:
                    self.parent_window.log_output("Combined summary skipped: no output paths.\n")
            msg = f"Completed: {n_ok}/{n_tasks} task(s) succeeded."
            if combined_summary_path is not None:
                msg += f"\n\nCombined summary: {combined_summary_path.name}\n(in {combined_summary_path.parent})"
            if n_fail > 0:
                msg += f"\n{n_fail} task(s) failed — see Log for errors."
            if success_dirs:
                msg += "\n\nExcel files are in these folders:"
                for d in success_dirs[:5]:
                    msg += f"\n  • {Path(d).name}"
                if len(success_dirs) > 5:
                    msg += f"\n  ... and {len(success_dirs) - 5} more (see Log)."
            # Unified: always offer "Open output folder" when we have a path
            first_dir = Path(success_dirs[0]) if success_dirs else None
            if first_dir is not None and first_dir.exists():
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Analysis Complete")
                dlg.setText(msg)
                dlg.setIcon(QMessageBox.Icon.Information)
                open_btn = dlg.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
                dlg.addButton(QMessageBox.StandardButton.Ok)
                dlg.exec()
                if dlg.clickedButton() == open_btn:
                    from PyQt6.QtCore import QUrl
                    from PyQt6.QtGui import QDesktopServices
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(first_dir)))
            else:
                QMessageBox.information(self, "Analysis Complete", msg)
            return

        mask_file, fi_path, output_path = self.folder_fi_tasks[self.current_task_index]
        fi_label = fi_path.stem if fi_path else "no FI"
        self.parent_window.log_output(f"\nProcessing task {self.current_task_index + 1}/{len(self.folder_fi_tasks)}: {mask_file.name} + {fi_label}\n")
        if fi_path is not None:
            self.parent_window.log_output(f"  FI image: {fi_path}\n")
        else:
            # Phase-only task (morphology from phase image) when output is _analysis_phase.xlsx; otherwise no FI was found
            if output_path.name.endswith("_analysis_phase.xlsx"):
                self.parent_window.log_output(f"  Phase-only analysis (morphology from phase image). FI channel(s) will be analyzed in the next task(s).\n")
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
        if getattr(self, 'export_csv', None) and self.export_csv.isChecked():
            cmd_args.append('--export-csv')
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
                    cmd_args.extend(['--foci-detection-mode', str(self.foci_detection_mode.currentIndex() if getattr(self, 'foci_detection_mode', None) else 0)])
                    cmd_args.extend(['--foci-min-rel', str(self.foci_min_rel.value())])
                    mode_txt = self.foci_min_rel_mode.currentText().strip().lower() if getattr(self, 'foci_min_rel_mode', None) else ''
                    if mode_txt == 'both':
                        cmd_args.extend(['--foci-min-rel-mode', 'both'])
                        cmd_args.extend(['--foci-min-rel-cell', str(self.foci_min_rel_cell.value())])
                    elif 'cell' in mode_txt:
                        cmd_args.extend(['--foci-min-rel-mode', 'per_cell'])
                    else:
                        cmd_args.extend(['--foci-min-rel-mode', 'per_frame'])
                    cmd_args.extend(['--foci-min-separation', str(self.foci_min_separation.value())])
                    min_sz, max_sz = self._foci_area_um2_to_px()
                    cmd_args.extend(['--foci-min-size', str(min_sz)])
                    if max_sz is not None and max_sz > 0:
                        cmd_args.extend(['--foci-max-size', str(max_sz)])
                    if getattr(self, "foci_min_peak_raw", None) and self.foci_min_peak_raw.value() > 0:
                        cmd_args.extend(['--foci-min-peak-raw', str(int(self.foci_min_peak_raw.value()))])
                    if getattr(self, "foci_min_focus_mean", None) and self.foci_min_focus_mean.value() > 0:
                        cmd_args.extend(['--foci-min-focus-mean', str(int(self.foci_min_focus_mean.value()))])
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
        if getattr(self, "_analyze_cancelled", False):
            self._analyze_cancelled = False
            self._folder_run_cancelled = True
            self.current_task_index = len(self.folder_fi_tasks)  # Skip to "all done"
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.wait()
                self.worker = None
            self._process_next_folder_file(script_path)
            return
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
    
    def _on_analyze_cancel(self) -> None:
        """Stop the current analysis run (single file or current task in folder mode)."""
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self.worker.terminate_process()
            self._analyze_cancelled = True  # So folder mode stops after this task
            self.parent_window.log_output("Cancelling analysis...\n")

    def on_finished(self, success, message):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Masks")
        if getattr(self, "analyze_cancel_btn", None) is not None:
            self.analyze_cancel_btn.setVisible(False)
        self.parent_window.log_output(message)
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()  # Wait for thread to finish
            self.worker = None
        # Hide progress bar
        self.parent_window.progress_bar.setVisible(False)
        if success:
            self.parent_window.log_output("✓ Analysis completed successfully!\n")
            self._performed_this_session = True
            # Completion popup (unified with Omnipose, Tracking, etc.)
            out_dir = getattr(self, "_last_analysis_output_dir", None)
            if out_dir is not None and Path(out_dir).exists():
                msg = QMessageBox(self)
                msg.setWindowTitle("Analysis Complete")
                msg.setText("Analysis completed successfully.")
                msg.setInformativeText(f"Output: {out_dir}")
                msg.setIcon(QMessageBox.Icon.Information)
                open_btn = msg.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
                msg.addButton(QMessageBox.StandardButton.Ok)
                msg.exec()
                if msg.clickedButton() == open_btn:
                    from PyQt6.QtCore import QUrl
                    from PyQt6.QtGui import QDesktopServices
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
            else:
                QMessageBox.information(self, "Analysis Complete", "Analysis completed successfully.")
        else:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{message}")
        self.workflow_step_finished.emit(success)


