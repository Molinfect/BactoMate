"""
Tracking tab: run trackastra on mask TIFFs.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.image_utils import scale_for_preview, to_qpixmap
from bactimate.gui.preferences import get_default_browse_dir
from bactimate.gui.utils import safe_imread
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog
from bactimate.modules.tiff_metadata import read_metadata_from_path, resolve_calibration_from_trajectories_path

from bactimate.gui.workers import TrackingWorker

# Time per frame unit conversion to minutes (for speed in µm/min)
TIME_UNIT_TO_MINUTES = {"min": 1.0, "h": 60.0, "s": 1.0 / 60.0, "ms": 1.0 / 60000.0}


class TrackingTab(QWidget):
    """Tab for cell tracking using trackastra"""
    workflow_step_finished = pyqtSignal(bool)  # for workflow replay chain

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._performed_this_session = False  # True if user ran tracking this session (for Re-run only same steps)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        self._build_input_groups(scroll_layout)

        tracking_group = self._build_tracking_params()
        scroll_layout.addWidget(tracking_group)

        self.process_btn = QPushButton("Run Tracking")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.run_tracking)
        scroll_layout.addWidget(self.process_btn)

        self._build_preview_section(scroll_layout)

        self.file_path.textChanged.connect(self._update_tracking_button_state)
        self.file_path.textChanged.connect(self._update_export_filtered_button_state)
        self.file_path.textChanged.connect(self._update_filter_ranges_from_trajectory)
        self.preview_path.textChanged.connect(self._update_preview_button_state)
        self.preview_path.textChanged.connect(self._update_export_filtered_button_state)
        self.preview_path.textChanged.connect(self._update_filter_ranges_from_trajectory)
        self.batch_cb.toggled.connect(self._update_tracking_button_state)
        self._update_tracking_button_state()
        self._update_export_filtered_button_state()
        QTimer.singleShot(0, self._update_filter_ranges_from_trajectory)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_input_groups(self, scroll_layout: QVBoxLayout) -> None:
        """Build Path group only (mask file or folder). Original/Excel are resolved automatically."""
        file_font = QFont()
        file_font.setPointSize(11)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        path_group = DragDropGroupBox(
            "Run tracking with TrackAstra",
            parent=self,
            on_drop_callback=self.handle_dropped_path
        )
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        path_layout.addWidget(create_info_icon(
            "Path to a mask TIFF file or folder of mask TIFFs to track.\n\n"
            "Single file: leave Batch unchecked and select a mask TIFF (Browse File).\n"
            "Folder: check Batch in Tracking Parameters and select a folder (e.g. _segmentation_omni or _segmentation_unet); "
            "tracking runs for each mask in the folder (one or more).\n\n"
            "Outputs are always written to a _tracking folder at the same level as the segmentation folder.",
            self,
        ))
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select mask TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)
        self.file_path.setFont(file_font)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_mask_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)
        path_layout.addWidget(self.file_path, 1)
        path_layout.addWidget(file_btn, 0)
        path_layout.addWidget(folder_btn, 0)
        path_group.setLayout(path_layout)
        scroll_layout.addWidget(path_group)

    def _build_preview_section(self, scroll_layout: QVBoxLayout) -> None:
        """Build Preview trajectories subsection: path, Browse File/Folder, then button."""
        preview_group = DragDropGroupBox(
            "Preview trajectories",
            parent=self,
            on_drop_callback=self._handle_dropped_preview_path,
        )
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Path:"))
        preview_layout.addWidget(create_info_icon(
            "Path to a folder or to a single *_trajectories.tif file.\n\n"
            "Click 'Preview trajectories' to open the file; if multiple files are found, you can choose which one to view.",
            self,
        ))
        self.preview_path = QLineEdit()
        self.preview_path.setPlaceholderText("Select folder or trajectory file... (or drag and drop)")
        self.preview_path.setMinimumWidth(500)
        preview_font = QFont()
        preview_font.setPointSize(11)
        self.preview_path.setFont(preview_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.preview_path.setSizePolicy(size_policy)
        preview_file_btn = QPushButton("Browse File...")
        preview_file_btn.setFixedWidth(120)
        preview_file_btn.clicked.connect(self.select_preview_file)
        preview_folder_btn = QPushButton("Browse Folder...")
        preview_folder_btn.setFixedWidth(130)
        preview_folder_btn.clicked.connect(self.select_preview_folder)
        preview_layout.addWidget(self.preview_path, 1)
        preview_layout.addWidget(preview_file_btn, 0)
        preview_layout.addWidget(preview_folder_btn, 0)
        preview_group.setLayout(preview_layout)
        scroll_layout.addWidget(preview_group)

        preview_btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview trajectories")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self.preview_trajectories)
        preview_btn_row.addWidget(self.preview_btn, 1)
        preview_btn_row.addStretch()
        preview_btn_row.addWidget(create_info_icon(
            "Find *_trajectories.tif at the path above; if multiple, you can choose which to view. "
            "Calibration (pixel size and time per frame) is auto-filled from TIFF metadata when available.",
            self,
        ))
        scroll_layout.addLayout(preview_btn_row)

        self._build_filter_tracks_section(scroll_layout)

    def _build_filter_tracks_section(self, scroll_layout: QVBoxLayout) -> None:
        """Build Filter tracks subsection: calibration row + 2-column grid so filters fill horizontal space; (?) next to each spin set."""
        filter_group = QGroupBox("Filter tracks")
        grid = QGridLayout()
        label_width = 118  # Min width so "Min straightness:" fits; keeps spinboxes aligned in both columns

        # Calibration row (full width): pixel size, time per frame + unit, info at end
        self.filter_pixel_size = QDoubleSpinBox()
        self.filter_pixel_size.setRange(0.0, 10.0)
        self.filter_pixel_size.setDecimals(3)
        self.filter_pixel_size.setValue(0.108)
        self.filter_pixel_size.setSingleStep(0.001)
        self.filter_pixel_size.setSpecialValueText("px only")
        self.filter_pixel_size.setMinimum(0.0)
        self.filter_time_per_frame = QDoubleSpinBox()
        self.filter_time_per_frame.setRange(0.0, 1e6)
        self.filter_time_per_frame.setDecimals(2)
        self.filter_time_per_frame.setValue(1.0)
        self.filter_time_per_frame.setSpecialValueText("frames only")
        self.filter_time_per_frame.setMinimum(0.0)
        self.filter_time_per_frame_unit = QComboBox()
        self.filter_time_per_frame_unit.addItems(["min", "h", "s", "ms"])
        self.filter_time_per_frame_unit.setCurrentText("min")
        GAP_PX = 10  # same gap: spin–dash, dash–spin, spin–info (and calibration row → info)
        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("Pixel size:"))
        cal_row.addWidget(self.filter_pixel_size)
        cal_row.addSpacing(GAP_PX)
        cal_row.addWidget(QLabel("Time/frame:"))
        cal_row.addWidget(self.filter_time_per_frame)
        cal_row.addWidget(self.filter_time_per_frame_unit)
        cal_row.addSpacing(GAP_PX)
        cal_row.addWidget(create_info_icon(
            "Calibration for physical units. Pixel size: µm/px (0 = pixels only). Time/frame: interval between frames; unit min/h/s/ms. Speed in µm/min when both set. Filled from trajectory metadata when you set a path.",
            self,
        ))
        cal_row.addStretch()
        grid.addLayout(cal_row, 0, 0, 1, 2)

        SPINBOX_WIDTH = 98
        STRETCH_SMALL = 1
        STRETCH_HUGE = 10

        def make_filter_cell(
            label: str,
            min_spin: QDoubleSpinBox | QSpinBox,
            max_spin: QDoubleSpinBox | QSpinBox | None,
            min_default: float | int,
            max_default: float | int,
            info_text: str,
        ) -> QHBoxLayout:
            min_spin.setMinimum(0)
            min_spin.setMaximum(999999)
            min_spin.setValue(min_default)
            min_spin.setFixedWidth(SPINBOX_WIDTH)
            if max_spin is not None:
                max_spin.setMinimum(0)
                max_spin.setMaximum(999999)
                max_spin.setValue(max_default)
                max_spin.setFixedWidth(SPINBOX_WIDTH)
            cell = QHBoxLayout()
            cell.setContentsMargins(0, 2, 0, 2)
            cell.setSpacing(0)
            lab = QLabel(label)
            lab.setMinimumWidth(label_width)
            cell.addWidget(lab)
            cell.addStretch(STRETCH_SMALL)
            cell.addWidget(min_spin)
            if max_spin is not None:
                cell.addSpacing(GAP_PX)
                dash = QLabel("–")
                dash.setFixedWidth(6)
                dash.setContentsMargins(0, 0, 0, 0)
                dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.addWidget(dash)
                cell.addSpacing(GAP_PX)
                cell.addWidget(max_spin)
            cell.addSpacing(GAP_PX)
            cell.addWidget(create_info_icon(info_text, self))
            cell.addStretch(STRETCH_HUGE)
            return cell

        # Row 1: Length | Duration
        self.filter_min_length = QSpinBox()
        self.filter_max_length = QSpinBox()
        self.filter_min_duration = QSpinBox()
        self.filter_max_duration = QSpinBox()
        grid.addLayout(
            make_filter_cell(
                "Length (frames):",
                self.filter_min_length,
                self.filter_max_length,
                2, 100,
                "Number of detections of frames per track (min–max). Max capped from trajectory frame count when path is set.",
            ), 1, 0)
        grid.addLayout(
            make_filter_cell(
                "Duration (fr):",
                self.filter_min_duration,
                self.filter_max_duration,
                1, 100,
                "Frame span (last − first + 1). Max capped from trajectory frame count when path is set.",
            ), 1, 1)

        # Row 2: Displacement | Path length (integer values: µm or px)
        self.filter_min_displacement = QDoubleSpinBox()
        self.filter_max_displacement = QDoubleSpinBox()
        self.filter_min_displacement.setDecimals(0)
        self.filter_max_displacement.setDecimals(0)
        self.filter_min_displacement.setRange(0, 1e9)
        self.filter_max_displacement.setRange(0, 1e9)
        self.filter_min_path_length = QDoubleSpinBox()
        self.filter_max_path_length = QDoubleSpinBox()
        self.filter_min_path_length.setDecimals(0)
        self.filter_max_path_length.setDecimals(0)
        self.filter_min_path_length.setRange(0, 1e9)
        self.filter_max_path_length.setRange(0, 1e9)
        grid.addLayout(
            make_filter_cell(
                "Displacement:",
                self.filter_min_displacement,
                self.filter_max_displacement,
                0, 100,
                "Net distance from first to last detection (µm if pixel size set, else px).",
            ), 2, 0)
        grid.addLayout(
            make_filter_cell(
                "Path length:",
                self.filter_min_path_length,
                self.filter_max_path_length,
                0, 100,
                "Total distance along the trajectory (sum of step lengths).",
            ), 2, 1)

        # Row 3: Mean speed | Min straightness (speed: 2 decimals, e.g. µm/min)
        self.filter_min_speed = QDoubleSpinBox()
        self.filter_max_speed = QDoubleSpinBox()
        self.filter_min_speed.setDecimals(2)
        self.filter_max_speed.setDecimals(2)
        self.filter_min_speed.setRange(0, 1e9)
        self.filter_max_speed.setRange(0, 1e9)
        self.filter_min_straightness = QDoubleSpinBox()
        self.filter_min_straightness.setRange(0.0, 1.0)
        self.filter_min_straightness.setDecimals(2)
        self.filter_min_straightness.setSingleStep(0.05)
        self.filter_min_straightness.setValue(0.0)
        grid.addLayout(
            make_filter_cell(
                "Mean speed:",
                self.filter_min_speed,
                self.filter_max_speed,
                0, 100,
                "Path length / duration. µm/min when calibration set; otherwise px/frame.",
            ), 3, 0)
        straight_cell = QHBoxLayout()
        straight_cell.setContentsMargins(0, 2, 0, 2)
        straight_cell.setSpacing(0)
        straight_lab = QLabel("Min Straightness:")
        straight_lab.setMinimumWidth(label_width)
        straight_cell.addWidget(straight_lab)
        straight_cell.addStretch(STRETCH_SMALL)
        self.filter_min_straightness.setFixedWidth(SPINBOX_WIDTH)
        straight_cell.addWidget(self.filter_min_straightness)
        straight_cell.addSpacing(GAP_PX)
        straight_cell.addWidget(create_info_icon(
            "Straightness = displacement / path length. 0 = any; 1 = perfectly straight (ballistic).",
            self,
        ))
        # Extra fixed width so STRETCH_SMALL gets same size as in Path length row → spinbox aligns; gap is right of info
        straight_cell.addSpacing(GAP_PX + 6 + GAP_PX + SPINBOX_WIDTH)
        straight_cell.addStretch(STRETCH_HUGE)
        grid.addLayout(straight_cell, 3, 1)

        filter_group.setLayout(grid)
        scroll_layout.addWidget(filter_group)

        export_row = QHBoxLayout()
        self.export_filtered_btn = QPushButton("Run Tracking with filters")
        self.export_filtered_btn.clicked.connect(self._run_export_filtered_tracks)
        export_row.addWidget(self.export_filtered_btn, 1)
        export_row.addStretch()
        export_row.addWidget(create_info_icon(
            "Find all *_trajectories.tif files (from Preview path or Tracking path), let you select which to process, "
            "then apply the current filter settings and write one Excel per file with one row per track that passes the filters. "
            "Columns: Cell_ID (if from analysis), TrackID, length, duration (frames and min), displacement, path length, "
            "mean speed, straight-line speed, straightness, first/last frame.",
            self,
        ))
        scroll_layout.addLayout(export_row)

    def _get_filter_params(self) -> dict[str, Any]:
        """Current calibration and filter parameters from widgets for track filtering.
        Time per frame is converted to minutes using the unit combo."""
        pixel = getattr(self, "filter_pixel_size", None)
        time_ = getattr(self, "filter_time_per_frame", None)
        time_unit = getattr(self, "filter_time_per_frame_unit", None)
        pixel_size = float(pixel.value()) if pixel is not None else 0.0
        time_val = float(time_.value()) if time_ is not None else 0.0
        unit_key = time_unit.currentText() if time_unit is not None else "min"
        time_per_frame = time_val * TIME_UNIT_TO_MINUTES.get(unit_key, 1.0)
        params = {
            "pixel_size": pixel_size,
            "time_per_frame": time_per_frame,
            "use_physical_units": pixel_size > 0,
            "use_speed_physical": pixel_size > 0 and time_per_frame > 0,
            "min_length": getattr(self, "filter_min_length", None) and self.filter_min_length.value(),
            "max_length": getattr(self, "filter_max_length", None) and self.filter_max_length.value(),
            "min_duration": getattr(self, "filter_min_duration", None) and self.filter_min_duration.value(),
            "max_duration": getattr(self, "filter_max_duration", None) and self.filter_max_duration.value(),
            "min_displacement": getattr(self, "filter_min_displacement", None) and self.filter_min_displacement.value(),
            "max_displacement": getattr(self, "filter_max_displacement", None) and self.filter_max_displacement.value(),
            "min_path_length": getattr(self, "filter_min_path_length", None) and self.filter_min_path_length.value(),
            "max_path_length": getattr(self, "filter_max_path_length", None) and self.filter_max_path_length.value(),
            "min_speed": getattr(self, "filter_min_speed", None) and self.filter_min_speed.value(),
            "max_speed": getattr(self, "filter_max_speed", None) and self.filter_max_speed.value(),
            "min_straightness": getattr(self, "filter_min_straightness", None) and self.filter_min_straightness.value(),
        }
        return params

    def _build_tracking_params(self) -> QGroupBox:
        """Build the Tracking Parameters group: one row with Model, Mode, and all checkboxes + info icons."""
        tracking_group = QGroupBox("Tracking Parameters")
        row = QHBoxLayout()

        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['general_2d', 'general_2d_w_SAM2_features'])
        self.model_combo.setCurrentText('general_2d')
        model_info_icon = create_info_icon(
            "Pretrained Trackastra model for cell tracking.\n\n"
            "Default: 'general_2d'\n"
            "Source: Trackastra pretrained models (https://github.com/weigertlab/trackastra)\n\n"
            "Available models:\n"
            "- general_2d: General-purpose 2D tracking model (default)\n"
            "  • Fast and works well for most cell types\n"
            "  • Good balance between speed and accuracy\n"
            "  • Recommended for general use cases\n\n"
            "- general_2d_w_SAM2_features: Enhanced model with SAM2 features\n"
            "  • Uses Segment Anything Model 2 (SAM2) features for improved tracking\n"
            "  • Better performance on certain data types, especially bacteria\n"
            "  • Slower than general_2d (requires GPU for reasonable speed)\n"
            "  • Best for challenging tracking scenarios\n\n"
            "Note: Trackastra won the 7th Cell Tracking Challenge (ISBI 2024).\n"
            "The models use transformer-based architecture for cell linking.",
            parent_widget=self
        )
        model_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(model_info_icon)
        row.addLayout(model_layout)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['greedy', 'greedy_nodiv', 'ilp'])
        self.mode_combo.setCurrentText('greedy')
        mode_info_icon = create_info_icon(
            "Linking mode for connecting cells across time frames.\n\n"
            "Default: 'greedy'\n"
            "Source: Trackastra linking algorithms (https://github.com/weigertlab/trackastra)\n\n"
            "Available modes:\n"
            "- greedy: Greedy linking with division support (recommended)\n"
            "  • Fast algorithm, no additional dependencies\n"
            "  • Handles cell divisions (splitting)\n"
            "  • Good for most tracking scenarios\n"
            "  • Best balance of speed and accuracy\n\n"
            "- greedy_nodiv: Greedy linking without division support\n"
            "  • Fast algorithm, no additional dependencies\n"
            "  • Cells cannot split (no division tracking)\n"
            "  • Use when cells don't divide or you want faster processing\n"
            "  • Simpler tracking logic\n\n"
            "- ilp: Integer Linear Programming (optimal linking)\n"
            "  • More accurate than greedy methods\n"
            "  • Solves global optimization problem\n"
            "  • Slower than greedy methods\n"
            "  • Requires additional dependencies (motile, Gurobi or SCIP solver)\n"
            "  • Best for challenging cases where accuracy is critical\n"
            "  • Note: Academic Gurobi licenses are free\n\n"
            "Tip: Start with 'greedy' for most cases. Use 'ilp' if you need maximum accuracy.",
            parent_widget=self
        )
        mode_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(mode_info_icon)
        row.addLayout(mode_layout)

        gpu_layout = QHBoxLayout()
        self.use_gpu_cb = QCheckBox("Use GPU")
        self.use_gpu_cb.setChecked(True)
        gpu_info_icon = create_info_icon(
            "Use GPU acceleration for tracking (CUDA or Apple Silicon MPS).\n\n"
            "Default: Checked. Uncheck to force CPU (slower).\n"
            "Falls back to CPU if GPU is not available or fails.",
            parent_widget=self,
        )
        gpu_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        gpu_layout.addWidget(self.use_gpu_cb)
        gpu_layout.addWidget(gpu_info_icon)
        row.addLayout(gpu_layout)

        create_vis_layout = QHBoxLayout()
        self.create_vis_cb = QCheckBox("Create visualization")
        self.create_vis_cb.setChecked(True)
        create_vis_info = create_info_icon(
            "When checked, trajectory overlay and visualization are created if an original TIF aligned to the mask is found (e.g. same name in the folder).\n\n"
            "Output is saved in the _tracking folder (e.g. *_trajectories.tif). Uncheck to skip visualization (faster).",
            parent_widget=self,
        )
        create_vis_info.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        create_vis_layout.addWidget(self.create_vis_cb)
        create_vis_layout.addWidget(create_vis_info)
        row.addLayout(create_vis_layout)

        batch_layout = QHBoxLayout()
        self.batch_cb = QCheckBox("Batch")
        self.batch_cb.setChecked(False)
        batch_info = create_info_icon(
            "When checked, Path must be a folder; tracking runs for each mask in that folder (one or more).\n\n"
            "When unchecked, Path must be a single mask file (single-file tracking). Auto-ticked when Path is a folder.",
            parent_widget=self,
        )
        batch_info.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        batch_layout.addWidget(self.batch_cb)
        batch_layout.addWidget(batch_info)
        row.addLayout(batch_layout)
        row.addStretch()

        tracking_group.setLayout(row)
        return tracking_group

    def _resolve_device(self) -> str:
        """Resolve device from Use GPU checkbox: 'cuda' or 'mps' if checked and available, else 'cpu'."""
        if not self.use_gpu_cb.isChecked():
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_trajectory_files(self) -> list[Path]:
        """Return list of existing *_trajectories.tif paths under the current Path's _tracking output."""
        raw = self.file_path.text() and self.file_path.text().strip()
        if not raw:
            return []
        p = Path(raw).resolve()
        if not p.exists():
            return []
        if p.is_file():
            out_dir = self._tracking_output_dir(p)
            traj = out_dir / f"{p.stem}_trajectories.tif"
            return [traj] if traj.exists() else []
        # Folder: find all _tracking/*/*_trajectories.tif
        base = p.parent if p.name.startswith("_segmentation") else p
        tracking_base = base / "_tracking"
        if not tracking_base.is_dir():
            return []
        return sorted(tracking_base.glob("*/*_trajectories.tif"))

    def _get_one_trajectory_path(self) -> Optional[Path]:
        """Return one trajectory file path from preview path or tracking path, or None."""
        preview = self._get_preview_tracked_files()
        if preview:
            return preview[0]
        tracking = self._get_trajectory_files()
        if tracking:
            return tracking[0]
        return None

    def _get_trajectory_frame_count(self, path: Path) -> Optional[int]:
        """Return number of frames in a trajectory TIFF, or None if unreadable."""
        meta = read_metadata_from_path(path)
        n = meta.get("frame_count")
        if n is not None and n > 0:
            return int(n)
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                if tif.pages:
                    return len(tif.pages)
        except Exception:
            pass
        return None

    def _update_filter_ranges_from_trajectory(self) -> None:
        """Update filter max length/duration from trajectory frame count and calibration from metadata."""
        if not hasattr(self, "filter_max_length") or not hasattr(self, "filter_max_duration"):
            return
        path = self._get_one_trajectory_path()
        if path is None:
            return
        # Cap length and duration max from frame count
        num_frames = self._get_trajectory_frame_count(path)
        if num_frames is not None and num_frames > 0:
            for spin in (self.filter_max_length, self.filter_max_duration):
                spin.setMaximum(max(spin.minimum(), num_frames))
                if spin.value() > num_frames:
                    spin.setValue(num_frames)
            if self.filter_min_length.value() > num_frames:
                self.filter_min_length.setValue(max(0, num_frames - 1))
            if self.filter_min_duration.value() > num_frames:
                self.filter_min_duration.setValue(max(1, num_frames))
        # Fill calibration from trajectory pipeline metadata
        ps, tp_min = resolve_calibration_from_trajectories_path(path)
        if ps > 0:
            self.filter_pixel_size.setValue(ps)
        if tp_min > 0:
            self.filter_time_per_frame.setValue(tp_min)
            self.filter_time_per_frame_unit.setCurrentText("min")

    def _update_tracking_button_state(self):
        """Enable Run when path matches Batch mode (file vs folder). Auto-tick Batch when path is folder."""
        raw = self.file_path.text() and self.file_path.text().strip()
        if not raw:
            if hasattr(self, "process_btn"):
                self.process_btn.setEnabled(False)
            return
        p = Path(raw)
        # Auto-tick Batch when path is a folder, untick when path is a file
        if p.exists() and hasattr(self, "batch_cb"):
            if p.is_dir():
                self.batch_cb.setChecked(True)
            elif p.is_file():
                self.batch_cb.setChecked(False)
        batch = getattr(self, "batch_cb", None) and self.batch_cb.isChecked()
        if hasattr(self, "process_btn"):
            if batch:
                self.process_btn.setEnabled(p.exists() and p.is_dir())
            else:
                self.process_btn.setEnabled(p.exists() and p.is_file())

    def _update_preview_button_state(self) -> None:
        """Enable Preview trajectories when the preview path is non-empty."""
        if not hasattr(self, "preview_btn"):
            return
        raw = self.preview_path.text() and self.preview_path.text().strip()
        self.preview_btn.setEnabled(bool(raw))

    def _update_export_filtered_button_state(self) -> None:
        """Enable Run Tracking with filters when any trajectory files are found (preview or main path)."""
        if not hasattr(self, "export_filtered_btn"):
            return
        has_files = bool(self._get_preview_tracked_files() or self._get_trajectory_files())
        self.export_filtered_btn.setEnabled(has_files)

    def select_mask_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask File", get_default_browse_dir(),
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if path:
            self.file_path.setText(path)
            self._update_tracking_button_state()

    def select_folder(self):
        """Open folder dialog to select a folder (e.g. _segmentation_omni or _segmentation_unet)."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Mask Files", get_default_browse_dir()
        )
        if folder_path:
            self.file_path.setText(folder_path)
            self._update_tracking_button_state()

    def _get_preview_tracked_files(self) -> list[Path]:
        """Find *_trajectories.tif under the preview path (file or folder)."""
        raw = self.preview_path.text() and self.preview_path.text().strip()
        if not raw:
            return []
        p = Path(raw).resolve()
        if not p.exists():
            return []
        if p.is_file():
            if p.name.endswith("_trajectories.tif"):
                return [p]
            # Search same directory and one level down
            parent = p.parent
            found = list(parent.glob("*_trajectories.tif"))
            for sub in parent.iterdir():
                if sub.is_dir():
                    found.extend(sub.glob("*_trajectories.tif"))
            return sorted(set(found))
        # Folder: search in folder and two levels down (e.g. _tracking and _tracking/*_tracking/)
        found = list(p.glob("*_trajectories.tif"))
        for sub in p.iterdir():
            if sub.is_dir():
                found.extend(sub.glob("*_trajectories.tif"))
                for sub2 in sub.iterdir():
                    if sub2.is_dir():
                        found.extend(sub2.glob("*_trajectories.tif"))
        return sorted(set(found))

    def select_preview_file(self) -> None:
        """Browse for a single *_trajectories.tif file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trajectories file to preview", get_default_browse_dir(),
            "Trajectories TIFF (*_trajectories.tif);;TIFF (*.tif *.tiff);;All (*)"
        )
        if path:
            self.preview_path.setText(path)
            self._update_preview_button_state()

    def select_preview_folder(self) -> None:
        """Browse for a folder containing *_trajectories.tif files."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select folder containing trajectory files", get_default_browse_dir()
        )
        if folder_path:
            self.preview_path.setText(folder_path)
            self._update_preview_button_state()

    def _handle_dropped_preview_path(self, dropped_path: str) -> None:
        """Handle dropped file or folder on the Preview path group."""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in (".tif", ".tiff"):
            self.preview_path.setText(dropped_path)
        elif path_obj.is_dir():
            self.preview_path.setText(dropped_path)
        else:
            QMessageBox.warning(
                self, "Invalid Drop",
                "Please drop a TIFF file or a folder for the preview path."
            )
        self._update_preview_button_state()

    def handle_dropped_path(self, dropped_path: str) -> None:
        """Handle dropped file or folder path for tracking tab."""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in (".tif", ".tiff"):
            self.file_path.setText(dropped_path)
        elif path_obj.is_dir():
            self.file_path.setText(dropped_path)
        else:
            QMessageBox.warning(
                self, "Invalid Drop",
                "Please drop a TIFF mask file or a folder containing mask files."
            )
        self._update_tracking_button_state()

    def _is_folder_mode(self) -> bool:
        """Return True if the current path is a folder."""
        if not self.file_path.text() or not self.file_path.text().strip():
            return False
        p = Path(self.file_path.text())
        return p.exists() and p.is_dir()

    @staticmethod
    def _tracking_output_dir(mask_path: Path) -> Path:
        """Compute output directory for tracking: base_dir / _tracking / {stem}_tracking.
        base_dir is one level above if mask is inside Tif_*, _segmentation*, or _analysis."""
        base = mask_path.parent
        if base.name.startswith("Tif_") or "_segmentation" in base.name or base.name == "_analysis":
            base = base.parent
        return base / "_tracking" / f"{mask_path.stem}_tracking"

    @staticmethod
    def _collect_mask_files(folder: Path) -> list[Path]:
        """Collect mask TIFF paths from a folder (same logic as Analyze Masks)."""
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
        return sorted(mask_set, key=lambda x: x.name.lower())

    @staticmethod
    def _resolve_original_for_mask(mask_path: Path) -> Path | None:
        """Find the original (phase) TIFF for this mask: stem.tif or base.tif in mask dir, parent, or subfolders."""
        parent = mask_path.parent
        stem = mask_path.stem.replace("_masks", "")
        # Search in mask's directory and optionally one level up (e.g. mask in _segmentation -> look in parent too).
        search_dirs: list[Path] = [parent]
        if parent.name.startswith("_segmentation"):
            search_dirs.append(parent.parent)
        else:
            search_dirs.append(parent.parent)
        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
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
            # One level down (e.g. Tif_001) in case original lives in a subfolder
            for sub in search_dir.iterdir():
                if not sub.is_dir():
                    continue
                for f in sub.iterdir():
                    if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                        continue
                    if "_fi" in f.stem.lower():
                        continue
                    base = stem.rsplit("_", 1)[0] if "_" in stem else stem
                    if f.stem == stem or f.stem == base:
                        return f
        return None

    @staticmethod
    def _resolve_from_trajectories_path(traj_path: Path) -> tuple[Path | None, Path | None, Path | None]:
        """From a *_trajectories.tif path return (tracks_csv_path, overlay_path, original_path).
        Tracks CSV and overlay are in the same output dir; original (phase) is found from the mask."""
        if not traj_path.is_file():
            return (None, None, None)
        out_dir = traj_path.parent
        stem = traj_path.stem.replace("_trajectories", "")
        tracks_csv = out_dir / f"{stem}_tracks.csv"
        overlay_path = out_dir / f"{stem}_overlay_tracked.tif"
        mask_stem = stem
        # Search for mask: same stem in base, _segmentation, or Tif_* / other subfolders (tracking input can be anywhere).
        search_bases = [out_dir.parent.parent, out_dir.parent]
        mask_path: Path | None = None
        for base in search_bases:
            if not base.is_dir():
                continue
            # Direct children
            for f in base.iterdir():
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff") and f.stem == mask_stem:
                    mask_path = f
                    break
            if mask_path is not None:
                break
            # _segmentation subfolder
            seg = base / "_segmentation"
            if seg.is_dir():
                for f in seg.iterdir():
                    if f.is_file() and f.suffix.lower() in (".tif", ".tiff") and f.stem == mask_stem:
                        mask_path = f
                        break
            if mask_path is not None:
                break
            # Common subfolders (e.g. Tif_001, Tif_002) — one level down
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                for f in sub.iterdir():
                    if f.is_file() and f.suffix.lower() in (".tif", ".tiff") and f.stem == mask_stem:
                        mask_path = f
                        break
                if mask_path is not None:
                    break
            if mask_path is not None:
                break
            # Glob anywhere under base (two levels max to avoid slow scan)
            for pattern in (f"*{mask_stem}.tif", f"*{mask_stem}.tiff"):
                for f in base.glob(pattern):
                    if f.is_file():
                        mask_path = f
                        break
                if mask_path is None:
                    for f in (base / "_segmentation").glob(pattern) if (base / "_segmentation").is_dir() else []:
                        if f.is_file():
                            mask_path = f
                            break
                if mask_path is not None:
                    break
            if mask_path is not None:
                break
        original_path = TrackingTab._resolve_original_for_mask(mask_path) if mask_path else None
        return (tracks_csv, overlay_path if overlay_path.is_file() else None, original_path)

    @staticmethod
    def _compute_track_metrics(
        tracks_df: pd.DataFrame,
        pixel_size_um: float,
        time_per_frame_min: float,
    ) -> pd.DataFrame:
        """Compute per-track metrics. Requires columns: track_id, frame, y, x.
        Returns a DataFrame with one row per track_id and columns:
        track_id, trajectory_length, duration_frames, duration_min, first_frame, last_frame,
        displacement_px, displacement_um, path_length_px, path_length_um,
        mean_speed_px_frame, mean_speed_um_min, straight_line_speed_px_frame, straight_line_speed_um_min, straightness."""
        if tracks_df.empty or "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return pd.DataFrame()
        need_xy = "y" in tracks_df.columns and "x" in tracks_df.columns
        rows = []
        for tid, g in tracks_df.groupby("track_id"):
            g = g.sort_values("frame")
            n = len(g)
            first_frame = int(g["frame"].iloc[0])
            last_frame = int(g["frame"].iloc[-1])
            duration = max(1, int(last_frame - first_frame + 1))
            displacement_px = 0.0
            path_length_px = 0.0
            if need_xy and n >= 2:
                ys = g["y"].values
                xs = g["x"].values
                displacement_px = float(np.hypot(ys[-1] - ys[0], xs[-1] - xs[0]))
                path_length_px = float(np.sum(np.hypot(np.diff(ys), np.diff(xs))))
            displacement_um = displacement_px * pixel_size_um if pixel_size_um > 0 else displacement_px
            path_length_um = path_length_px * pixel_size_um if pixel_size_um > 0 else path_length_px
            path_safe = path_length_px if path_length_px > 0 else 1.0
            straightness = float(displacement_px / path_safe) if path_safe > 0 else 0.0
            mean_speed_px_frame = path_length_px / duration if duration > 0 else 0.0
            if pixel_size_um > 0 and time_per_frame_min > 0:
                mean_speed_um_min = path_length_um / (duration * time_per_frame_min)
            else:
                mean_speed_um_min = mean_speed_px_frame
            duration_min = duration * time_per_frame_min if time_per_frame_min > 0 else 0.0
            straight_line_speed_px_frame = displacement_px / duration if duration > 0 else 0.0
            straight_line_speed_um_min = displacement_um / (duration * time_per_frame_min) if (duration > 0 and time_per_frame_min > 0 and pixel_size_um > 0) else straight_line_speed_px_frame
            rows.append({
                "track_id": tid,
                "trajectory_length": n,
                "duration_frames": duration,
                "duration_min": duration_min,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "displacement_px": displacement_px,
                "displacement_um": displacement_um,
                "path_length_px": path_length_px,
                "path_length_um": path_length_um,
                "mean_speed_px_frame": mean_speed_px_frame,
                "mean_speed_um_min": mean_speed_um_min,
                "straight_line_speed_px_frame": straight_line_speed_px_frame,
                "straight_line_speed_um_min": straight_line_speed_um_min,
                "straightness": straightness,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _apply_track_filters(metrics_df: pd.DataFrame, params: dict[str, Any]) -> set[int]:
        """Return set of track_id that pass all filters. params keys: min_length, max_length, min_duration, max_duration,
        min_displacement, max_displacement, min_path_length, max_path_length, min_speed, max_speed, min_straightness.
        Uses µm/µm/min when pixel_size and time_per_frame are set (params can use use_physical and we use _um / _um_min columns)."""
        if metrics_df.empty:
            return set()
        use_um = params.get("use_physical_units", True)
        use_speed_physical = params.get("use_speed_physical", use_um)
        disp_col = "displacement_um" if use_um else "displacement_px"
        path_col = "path_length_um" if use_um else "path_length_px"
        speed_col = "mean_speed_um_min" if use_speed_physical else "mean_speed_px_frame"
        mask = np.ones(len(metrics_df), dtype=bool)
        m = metrics_df
        if "min_length" in params and params["min_length"] is not None:
            mask &= (m["trajectory_length"] >= params["min_length"])
        if "max_length" in params and params["max_length"] is not None:
            mask &= (m["trajectory_length"] <= params["max_length"])
        if "min_duration" in params and params["min_duration"] is not None:
            mask &= (m["duration_frames"] >= params["min_duration"])
        if "max_duration" in params and params["max_duration"] is not None:
            mask &= (m["duration_frames"] <= params["max_duration"])
        if "min_displacement" in params and params["min_displacement"] is not None:
            mask &= (m[disp_col] >= params["min_displacement"])
        if "max_displacement" in params and params["max_displacement"] is not None:
            mask &= (m[disp_col] <= params["max_displacement"])
        if "min_path_length" in params and params["min_path_length"] is not None:
            mask &= (m[path_col] >= params["min_path_length"])
        if "max_path_length" in params and params["max_path_length"] is not None:
            mask &= (m[path_col] <= params["max_path_length"])
        if "min_speed" in params and params["min_speed"] is not None:
            mask &= (m[speed_col] >= params["min_speed"])
        if "max_speed" in params and params["max_speed"] is not None:
            mask &= (m[speed_col] <= params["max_speed"])
        if "min_straightness" in params and params["min_straightness"] is not None:
            mask &= (m["straightness"] >= params["min_straightness"])
        return set(metrics_df.loc[mask, "track_id"].astype(int).tolist())

    @staticmethod
    def _draw_filtered_trajectory_overlay(images: np.ndarray, tracks_df: pd.DataFrame) -> np.ndarray:
        """Build (T, H, W, 3) uint8 overlay: normalized images + trajectory lines and dots for given tracks only."""
        import matplotlib.pyplot as plt
        if images.ndim != 3:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        num_frames, height, width = images.shape[0], images.shape[1], images.shape[2]
        img_min = images.min(axis=(1, 2), keepdims=True)
        img_max = images.max(axis=(1, 2), keepdims=True)
        img_range = img_max - img_min
        img_range[img_range == 0] = 1
        images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)
        if tracks_df.empty or "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            out = np.stack([np.stack([images_norm[i], images_norm[i], images_norm[i]], axis=-1) for i in range(num_frames)], axis=0)
            return out
        need_xy = "y" in tracks_df.columns and "x" in tracks_df.columns
        unique_tracks = tracks_df["track_id"].unique()
        num_tracks = len(unique_tracks)
        cmap = plt.get_cmap("gist_ncar")
        track_colors = {}
        for i, track_id in enumerate(unique_tracks):
            color = cmap(i / max(num_tracks, 1))
            track_colors[int(track_id)] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        for frame_idx in range(num_frames):
            vis_frame = np.stack([images_norm[frame_idx], images_norm[frame_idx], images_norm[frame_idx]], axis=-1).copy()
            if need_xy:
                tracks_in_frame = tracks_df[
                    (tracks_df["frame"] <= frame_idx) & (tracks_df["track_id"].isin(unique_tracks))
                ]["track_id"].unique()
                for track_id_int in tracks_in_frame:
                    track_id_int = int(track_id_int)
                    if track_id_int not in track_colors:
                        continue
                    track_points = tracks_df[
                        (tracks_df["track_id"] == track_id_int) & (tracks_df["frame"] <= frame_idx)
                    ]
                    color = track_colors[track_id_int]
                    if len(track_points) > 1:
                        points = track_points[["y", "x"]].values.astype(int)
                        points[:, 0] = np.clip(points[:, 0], 0, height - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, width - 1)
                        for i in range(len(points) - 1):
                            y1, x1 = points[i]
                            y2, x2 = points[i + 1]
                            dy, dx = abs(y2 - y1), abs(x2 - x1)
                            if dx == 0 and dy == 0:
                                continue
                            max_steps = max(dx, dy)
                            step_size = max(1, max_steps // 50) if max_steps > 50 else 1
                            num_steps = max_steps // step_size + 1
                            if dx > dy:
                                x_coords = np.linspace(x1, x2, num_steps, dtype=int)
                                y_coords = np.round(np.linspace(y1, y2, num_steps)).astype(int)
                            else:
                                y_coords = np.linspace(y1, y2, num_steps, dtype=int)
                                x_coords = np.round(np.linspace(x1, x2, num_steps)).astype(int)
                            y_coords = np.clip(y_coords, 1, height - 2)
                            x_coords = np.clip(x_coords, 1, width - 2)
                            for y, x in zip(y_coords, x_coords):
                                ys, ye = y - 1, y + 2
                                xs_, xe = x - 1, x + 2
                                vis_frame[ys:ye, xs_:xe, 0] = (vis_frame[ys:ye, xs_:xe, 0] * 0.3 + color[0] * 0.7).astype(np.uint8)
                                vis_frame[ys:ye, xs_:xe, 1] = (vis_frame[ys:ye, xs_:xe, 1] * 0.3 + color[1] * 0.7).astype(np.uint8)
                                vis_frame[ys:ye, xs_:xe, 2] = (vis_frame[ys:ye, xs_:xe, 2] * 0.3 + color[2] * 0.7).astype(np.uint8)
                    current_track = track_points[track_points["frame"] == frame_idx]
                    if len(current_track) > 0:
                        y = int(current_track.iloc[0]["y"])
                        x = int(current_track.iloc[0]["x"])
                        if 0 <= y < height and 0 <= x < width:
                            dot_size = 4
                            color = track_colors[track_id_int]
                            y0 = max(0, y - dot_size)
                            y1 = min(height, y + dot_size + 1)
                            x0 = max(0, x - dot_size)
                            x1 = min(width, x + dot_size + 1)
                            vis_frame[y0:y1, x0:x1, :] = color
            visualization[frame_idx] = vis_frame
        return visualization

    @staticmethod
    def _resolve_analysis_excel_for_mask(mask_path: Path, base_dir: Path) -> Path | None:
        """Find analysis Excel for this mask in base_dir / _analysis (e.g. {stem}_analysis.xlsx)."""
        stem = mask_path.stem.replace("_masks", "")
        analysis_dir = base_dir / "_analysis"
        if not analysis_dir.is_dir():
            return None
        for name in (f"{stem}_analysis.xlsx", f"{stem}_masks_analysis.xlsx"):
            p = analysis_dir / name
            if p.is_file():
                return p
        return None

    def run_tracking(self):
        # Expand Status & Session logs so the user can see progress during long-running tracking
        if hasattr(self.parent_window, "expand_status_log"):
            self.parent_window.expand_status_log()
        if not self.file_path.text() or not self.file_path.text().strip():
            QMessageBox.warning(self, "Missing Input", "Please select a mask file or folder.")
            return

        path = Path(self.file_path.text()).resolve()
        if not path.exists():
            QMessageBox.warning(self, "Not Found", f"Path not found:\n{path}")
            return

        batch = self.batch_cb.isChecked()
        if path.is_dir() and not batch:
            QMessageBox.warning(
                self, "Batch Required",
                "Path is a folder. Check 'Batch' in Tracking Parameters to run tracking on the mask(s) in the folder."
            )
            return
        if path.is_file() and batch:
            QMessageBox.warning(
                self, "Single File",
                "Path is a file. Uncheck 'Batch' in Tracking Parameters for single-file tracking."
            )
            return

        # Build list of mask files (one for single-file, many for folder) and show selection dialog
        if path.is_file():
            mask_files = [path]
        else:
            mask_files = self._collect_mask_files(path)
            if not mask_files:
                QMessageBox.warning(
                    self, "No Mask Files Found",
                    "No mask TIFF files found in the selected folder.\n\n"
                    "Please select a folder containing *_masks.tif (or similar) files."
                )
                return
        selected = show_file_selection_dialog(
            self,
            mask_files,
            title="Select Mask Files for Tracking",
            message=f"Select mask files to track (all {len(mask_files)} selected by default):",
        )
        if selected is None:
            self.parent_window.log_output("Tracking cancelled (no mask files selected).\n")
            return
        if len(selected) == 0:
            QMessageBox.warning(self, "No Files Selected", "Please select at least one mask file to track.")
            return

        if len(selected) == 1:
            # Single-file tracking
            mask_file = selected[0].resolve()
            output_dir = self._tracking_output_dir(mask_file)
            base_dir = mask_file.parent
            if base_dir.name.startswith("Tif_") or "_segmentation" in base_dir.name:
                base_dir = base_dir.parent
            original_path = self._resolve_original_for_mask(mask_file) if self.create_vis_cb.isChecked() else None
            original_file = str(original_path) if original_path else None
            excel_path = self._resolve_analysis_excel_for_mask(mask_file, base_dir)
            excel_file = str(excel_path) if excel_path else None
            model_name = self.model_combo.currentText()
            mode = self.mode_combo.currentText()
            device = self._resolve_device()
            # Device check and single worker block follows below (after the batch block)
            self._run_single_tracking(mask_file, output_dir, original_file, excel_file, model_name, mode, device)
            return

        # Batch: multiple files selected
        tasks = []
        for mask_path in selected:
            mask_path = mask_path.resolve()
            base_dir = mask_path.parent
            if base_dir.name.startswith("Tif_") or "_segmentation" in base_dir.name:
                base_dir = base_dir.parent
            output_dir = self._tracking_output_dir(mask_path)
            original_path = self._resolve_original_for_mask(mask_path) if self.create_vis_cb.isChecked() else None
            excel_path = self._resolve_analysis_excel_for_mask(mask_path, base_dir)
            tasks.append((mask_path, original_path, excel_path, output_dir))
        model_name = self.model_combo.currentText()
        mode = self.mode_combo.currentText()
        device = self._resolve_device()
        # Device check (same as single-file path)
        if device == "cuda":
            debug_info = ["=== CUDA Debugging Information ==="]
            try:
                import torch
                debug_info.append(f"✓ PyTorch version: {torch.__version__}")
                cuda_available = torch.cuda.is_available()
                debug_info.append(f"✓ torch.cuda.is_available(): {cuda_available}")
                if not cuda_available:
                    debug_info.append("✗ CUDA is not available")
                    error_msg = "\n".join(debug_info) + "\n\nCUDA GPU is not available. Continue with CPU?"
                    reply = QMessageBox.question(
                        self, "CUDA Not Available", error_msg,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = "cpu"
                    self.use_gpu_cb.setChecked(False)
                # else: CUDA ok, continue
            except ImportError:
                QMessageBox.warning(self, "PyTorch Not Found", "PyTorch required for CUDA. Switching to CPU.")
                device = "cpu"
                self.use_gpu_cb.setChecked(False)
        elif device == "mps":
            try:
                import torch
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    error_msg = "MPS (Apple Silicon GPU) is not available. Continue with CPU?"
                    reply = QMessageBox.question(
                        self, "MPS Not Available", error_msg,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = "cpu"
                    self.use_gpu_cb.setChecked(False)
            except ImportError:
                QMessageBox.warning(self, "PyTorch Not Found", "PyTorch required for MPS. Switching to CPU.")
                device = "cpu"
                self.use_gpu_cb.setChecked(False)
        self.folder_tracking_tasks = tasks
        self.folder_tracking_results = []
        self.current_task_index = 0
        self._batch_model_name = model_name
        self._batch_mode = mode
        self._batch_device = device
        self.process_btn.setEnabled(False)
        self.process_btn.setText(f"Tracking... (0/{len(tasks)})")
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f"Starting batch tracking for {len(tasks)} file(s).\n")
        self._process_next_tracking_task()
        return

    def _run_single_tracking(
        self,
        mask_file: Path,
        output_dir: Path,
        original_file: Optional[str],
        excel_file: Optional[str],
        model_name: str,
        mode: str,
        device: str,
    ) -> None:
        """Run device check and start tracking worker for a single mask file."""
        # Check device availability with detailed debugging
        if device == 'cuda':
            debug_info = []
            debug_info.append("=== CUDA Debugging Information ===")
            
            try:
                import torch
                debug_info.append(f"✓ PyTorch version: {torch.__version__}")
                
                # Check CUDA availability
                cuda_available = torch.cuda.is_available()
                debug_info.append(f"✓ torch.cuda.is_available(): {cuda_available}")
                
                if not cuda_available:
                    debug_info.append("✗ CUDA is not available")
                    debug_info.append(f"  - CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
                    debug_info.append(f"  - cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
                    
                    # Check if CUDA libraries are found
                    try:
                        import subprocess
                        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
                        if result.returncode == 0:
                            debug_info.append(f"  - nvcc found: {result.stdout.strip()}")
                        else:
                            debug_info.append("  - nvcc not found in PATH")
                    except:
                        pass
                    
                    error_msg = "\n".join(debug_info)
                    error_msg += "\n\nCUDA GPU is not available on this system."
                    error_msg += "\n\nPossible reasons:"
                    error_msg += "\n- No NVIDIA GPU installed"
                    error_msg += "\n- CUDA drivers not installed"
                    error_msg += "\n- PyTorch was installed without CUDA support"
                    error_msg += "\n- GPU drivers are outdated"
                    
                    reply = QMessageBox.question(
                        self, "CUDA Not Available",
                        error_msg + "\n\nWould you like to continue with CPU mode?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = 'cpu'
                    self.use_gpu_cb.setChecked(False)
                    self.parent_window.log_output("\n".join(debug_info) + "\n")
                else:
                    # CUDA is available, get detailed info
                    debug_info.append("✓ CUDA is available!")
                    debug_info.append(f"  - CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
                    debug_info.append(f"  - Number of GPUs: {torch.cuda.device_count()}")
                    
                    for i in range(torch.cuda.device_count()):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_props = torch.cuda.get_device_properties(i)
                        debug_info.append(f"  - GPU {i}: {gpu_name}")
                        debug_info.append(f"    - Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
                        debug_info.append(f"    - Compute Capability: {gpu_props.major}.{gpu_props.minor}")
                    
                    # Test CUDA with a simple operation
                    debug_info.append("\nTesting CUDA functionality...")
                    try:
                        # Test basic tensor creation
                        test_tensor = torch.tensor([1.0], device='cuda')
                        debug_info.append("  ✓ Tensor creation: SUCCESS")
                        
                        # Test simple operation
                        result = test_tensor * 2.0
                        debug_info.append("  ✓ Tensor operation: SUCCESS")
                        
                        # Test memory allocation
                        test_large = torch.zeros(1000, 1000, device='cuda')
                        debug_info.append("  ✓ Memory allocation: SUCCESS")
                        
                        del test_tensor, result, test_large
                        torch.cuda.empty_cache()
                        debug_info.append("  ✓ Memory cleanup: SUCCESS")
                        
                        debug_info.append("\n✓ CUDA is working correctly!")
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                        
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        debug_info.append(f"  ✗ CUDA test failed: {str(e)}")
                        debug_info.append(f"\nError details:\n{error_details}")
                        
                        error_msg = "\n".join(debug_info)
                        error_msg += "\n\nCUDA is available but encountered an error during testing."
                        error_msg += "\nThis might indicate:"
                        error_msg += "\n- GPU driver issues"
                        error_msg += "\n- CUDA runtime problems"
                        error_msg += "\n- Memory allocation issues"
                        
                        reply = QMessageBox.question(
                            self, "CUDA Error",
                            error_msg + "\n\nWould you like to continue with CPU mode?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.No:
                            return
                        device = 'cpu'
                        self.use_gpu_cb.setChecked(False)
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                        
            except ImportError:
                debug_info.append("✗ PyTorch is not installed")
                error_msg = "\n".join(debug_info)
                error_msg += "\n\nPyTorch is required for CUDA support."
                error_msg += "\nInstall with: pip install torch torchvision"
                error_msg += "\nFor CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                
                QMessageBox.warning(
                    self, "PyTorch Not Found",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.use_gpu_cb.setChecked(False)
                self.parent_window.log_output("\n".join(debug_info) + "\n")
            except Exception as e:
                import traceback
                debug_info.append(f"✗ Unexpected error: {str(e)}")
                debug_info.append(f"\n{traceback.format_exc()}")
                error_msg = "\n".join(debug_info)
                
                QMessageBox.warning(
                    self, "CUDA Check Error",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.use_gpu_cb.setChecked(False)
                self.parent_window.log_output("\n".join(debug_info) + "\n")
        elif device == 'mps':
            # Check MPS (Apple Silicon) availability
            debug_info = []
            debug_info.append("=== GPU (Apple Silicon) Debugging Information ===")
            
            try:
                import torch
                debug_info.append(f"✓ PyTorch version: {torch.__version__}")
                
                # Check MPS availability
                if hasattr(torch.backends, 'mps'):
                    mps_available = torch.backends.mps.is_available()
                    debug_info.append(f"✓ torch.backends.mps.is_available(): {mps_available}")
                    
                    if not mps_available:
                        debug_info.append("✗ GPU is not available")
                        debug_info.append("  - This usually means you're not on Apple Silicon (M1/M2/M3)")
                        debug_info.append("  - Or PyTorch was built without GPU support")
                        
                        error_msg = "\n".join(debug_info)
                        error_msg += "\n\nMPS (Apple Silicon GPU) is not available."
                        error_msg += "\n\nPossible reasons:"
                        error_msg += "\n- Not running on Apple Silicon (M1/M2/M3 chip)"
                        error_msg += "\n- PyTorch was installed without MPS support"
                        error_msg += "\n- macOS version too old (needs macOS 12.3+)"
                        
                        reply = QMessageBox.question(
                            self, "MPS Not Available",
                            error_msg + "\n\nWould you like to continue with CPU mode?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.No:
                            return
                        device = 'cpu'
                        self.use_gpu_cb.setChecked(False)
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                    else:
                        # MPS is available, test it
                        debug_info.append("✓ GPU is available!")
                        debug_info.append("  - Device: Apple Silicon GPU")
                        
                        # Test MPS with a simple operation
                        debug_info.append("\nTesting GPU functionality...")
                        try:
                            # Test basic tensor creation
                            test_tensor = torch.tensor([1.0], device='mps')
                            debug_info.append("  ✓ Tensor creation: SUCCESS")
                            
                            # Test simple operation
                            result = test_tensor * 2.0
                            debug_info.append("  ✓ Tensor operation: SUCCESS")
                            
                            # Test memory allocation
                            test_large = torch.zeros(1000, 1000, device='mps')
                            debug_info.append("  ✓ Memory allocation: SUCCESS")
                            
                            del test_tensor, result, test_large
                            torch.mps.empty_cache()
                            debug_info.append("  ✓ Memory cleanup: SUCCESS")
                            
                            debug_info.append("\n✓ GPU is working correctly!")
                            self.parent_window.log_output("\n".join(debug_info) + "\n")
                            
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            debug_info.append(f"  ✗ MPS test failed: {str(e)}")
                            debug_info.append(f"\nError details:\n{error_details}")
                            
                            error_msg = "\n".join(debug_info)
                            error_msg += "\n\nGPU is available but encountered an error during testing."
                            error_msg += "\nThis might indicate:"
                            error_msg += "\n- macOS version compatibility issues"
                            error_msg += "\n- PyTorch MPS backend problems"
                            error_msg += "\n- Memory allocation issues"
                            
                            reply = QMessageBox.question(
                                self, "MPS Error",
                                error_msg + "\n\nWould you like to continue with CPU mode?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.Yes
                            )
                            if reply == QMessageBox.StandardButton.No:
                                return
                            device = 'cpu'
                            self.use_gpu_cb.setChecked(False)
                            self.parent_window.log_output("\n".join(debug_info) + "\n")
                else:
                    debug_info.append("✗ MPS backend not available in this PyTorch version")
                    error_msg = "\n".join(debug_info)
                    error_msg += "\n\nMPS backend is not available."
                    error_msg += "\nThis PyTorch version may not support MPS."
                    
                    reply = QMessageBox.question(
                        self, "MPS Not Supported",
                        error_msg + "\n\nWould you like to continue with CPU mode?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = 'cpu'
                    self.use_gpu_cb.setChecked(False)
                    self.parent_window.log_output("\n".join(debug_info) + "\n")
                    
            except ImportError:
                debug_info.append("✗ PyTorch is not installed")
                error_msg = "\n".join(debug_info)
                error_msg += "\n\nPyTorch is required for MPS support."
                
                QMessageBox.warning(
                    self, "PyTorch Not Found",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.use_gpu_cb.setChecked(False)
                self.parent_window.log_output("\n".join(debug_info) + "\n")
            except Exception as e:
                import traceback
                debug_info.append(f"✗ Unexpected error: {str(e)}")
                debug_info.append(f"\n{traceback.format_exc()}")
                error_msg = "\n".join(debug_info)
                
                QMessageBox.warning(
                    self, "MPS Check Error",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.use_gpu_cb.setChecked(False)
                self.parent_window.log_output("\n".join(debug_info) + "\n")
        
        # Create worker thread (output_dir = base / _tracking / {stem}_tracking)
        self.worker = TrackingWorker(
            mask_file, original_file, excel_file,
            model_name, mode, device, self,
            output_dir=output_dir,
        )
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.start()
        
        # Disable button during processing
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f"Starting tracking for: {mask_file.name}\n")
        self.parent_window.log_output(
            "Tracking runs several steps (masks → overlay → trajectory visualization, etc.). "
            "Do not close the app until you see '✓ Tracking completed successfully!' below.\n"
            "At 55%, feature extraction and linking can take several minutes; 'Still running…' messages in the log mean the app is working.\n"
        )
        # Connect progress signal
        self.worker.progress.connect(self.on_tracking_progress)
    
    def on_tracking_progress(self, value, message):
        """Update progress bar and log message"""
        self.parent_window.progress_bar.setValue(value)
        if message:
            self.parent_window.log_output(f"[{value}%] {message}\n")
    
    def on_tracking_finished(self, success, message):
        self.parent_window.progress_bar.setValue(100)
        if success:
            self.parent_window.log_output("✓ Tracking completed successfully!\n")
        else:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{message}")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.parent_window.progress_bar.setVisible(False))
        self._update_tracking_button_state()
        if success:
            self._performed_this_session = True
        self.workflow_step_finished.emit(success)

    def _process_next_tracking_task(self) -> None:
        """Run the next task in batch tracking, or finish batch if none left."""
        tasks = getattr(self, "folder_tracking_tasks", [])
        if self.current_task_index >= len(tasks):
            # Batch complete
            n_ok = sum(1 for r in getattr(self, "folder_tracking_results", []) if r)
            n_fail = len(tasks) - n_ok
            self.process_btn.setEnabled(True)
            self.process_btn.setText("Run Tracking")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(
                f"\n✓ Batch tracking complete: {n_ok}/{len(tasks)} succeeded."
                + (f" {n_fail} failed — check log above.\n" if n_fail else "\n")
            )
            self._performed_this_session = True
            self.workflow_step_finished.emit(True)
            msg = f"Completed: {n_ok}/{len(tasks)} file(s) tracked successfully."
            if n_fail:
                msg += f"\n{n_fail} file(s) failed — see Log for errors."
            QMessageBox.information(self, "Batch Tracking Complete", msg)
            self._update_tracking_button_state()
            return
        mask_path, original_path, excel_path, output_dir = tasks[self.current_task_index]
        model_name = getattr(self, "_batch_model_name", self.model_combo.currentText())
        mode = getattr(self, "_batch_mode", self.mode_combo.currentText())
        device = getattr(self, "_batch_device", self._resolve_device())
        original_file = str(original_path) if original_path else None
        excel_file = str(excel_path) if excel_path else None
        self.parent_window.log_output(
            f"\nProcessing {self.current_task_index + 1}/{len(tasks)}: {mask_path.name}\n"
        )
        if self.current_task_index == 0:
            self.parent_window.log_output(
                "Do not close the app until you see 'Batch tracking complete' (or all files finished).\n"
            )
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        self.worker = TrackingWorker(
            mask_path, original_file, excel_file,
            model_name, mode, device, self,
            output_dir=output_dir,
        )
        self.worker.finished.connect(self.on_tracking_task_finished)
        self.worker.progress.connect(self._on_batch_tracking_progress)
        self.worker.start()

    def _on_batch_tracking_progress(self, value: int, message: str) -> None:
        """Update progress bar for batch: overall 0–100 across all tasks."""
        tasks = getattr(self, "folder_tracking_tasks", [])
        if not tasks:
            return
        total = len(tasks)
        # Current task contributes value/100 of one slot
        overall = (self.current_task_index + value / 100.0) / total * 100
        self.parent_window.progress_bar.setValue(int(min(99, overall)))
        if message:
            self.parent_window.log_output(f"[{self.current_task_index + 1}/{total}] {message}\n")

    def on_tracking_task_finished(self, success: bool, message: str) -> None:
        """Called when one batch task finishes; record result and start next or finish batch."""
        self.folder_tracking_results.append(success)
        if not success:
            self.parent_window.log_output(f"✗ Failed: {message}\n")
        self.current_task_index += 1
        n = len(self.folder_tracking_tasks)
        self.process_btn.setText(f"Tracking... ({self.current_task_index}/{n})")
        self.parent_window.progress_bar.setValue(int(self.current_task_index / n * 100))
        self._process_next_tracking_task()

    def preview_trajectories(self) -> None:
        """Find *_trajectories.tif at preview path; if multiple, show selection; open chosen file."""
        files = self._get_preview_tracked_files()
        if not files:
            QMessageBox.warning(
                self, "No files found",
                "No *_trajectories.tif files found at the given path.\n\n"
                "Set the Preview path to a folder or file and try again."
            )
            return
        parent = self.window() if hasattr(self, "window") and callable(self.window) else self
        selected = show_file_selection_dialog(
            parent,
            files,
            title="Select file to view",
            message="Select which trajectory file to view:",
        )
        if selected is None or not selected:
            return
        # Open one preview window per selected file (all non-modal, slightly offset so multiple are visible)
        for i, traj_path in enumerate(selected):
            self._open_trajectory_preview_window(traj_path, index=i)

    def _run_export_filtered_tracks(self) -> None:
        """Find trajectory files, let user select which to process, apply current filters and export Excel per file."""
        files = self._get_preview_tracked_files()
        if not files:
            files = self._get_trajectory_files()
        if not files:
            QMessageBox.warning(
                self,
                "No trajectory files",
                "No *_trajectories.tif files found.\n\n"
                "Set the Preview path (folder or file) or the main Tracking path (folder with _tracking output), then try again."
            )
            return
        parent = self.window() if hasattr(self, "window") and callable(self.window) else self
        selected = show_file_selection_dialog(
            parent,
            files,
            title="Export filtered tracks",
            message="Select which trajectory files to export (filtered track data will be written as Excel):",
        )
        if selected is None or not selected:
            return
        n_files = len(selected)
        if hasattr(self.parent_window, "expand_status_log"):
            self.parent_window.expand_status_log()
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, max(1, n_files))
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f"Exporting filtered tracks ({n_files} file(s))...\n")
        params = self._get_filter_params()
        pixel_size = params["pixel_size"]
        time_per_frame = params["time_per_frame"]
        done = 0
        errors = []
        for idx, traj_path in enumerate(selected):
            self.parent_window.progress_bar.setValue(idx)
            self.parent_window.log_output(f"  [{idx + 1}/{n_files}] {traj_path.name}\n")
            QApplication.processEvents()
            out_dir = traj_path.parent
            stem = traj_path.stem.replace("_trajectories", "")
            tracks_csv = out_dir / f"{stem}_tracks.csv"
            if not tracks_csv.is_file():
                errors.append(f"{traj_path.name}: no {tracks_csv.name} found")
                continue
            try:
                tracks_df = pd.read_csv(tracks_csv)
            except Exception as e:
                errors.append(f"{traj_path.name}: could not read CSV ({e})")
                continue
            for col in ("track_id", "frame", "y", "x"):
                if col not in tracks_df.columns:
                    errors.append(f"{traj_path.name}: CSV missing column '{col}'")
                    break
            else:
                metrics_df = self._compute_track_metrics(tracks_df, pixel_size, time_per_frame)
                if metrics_df.empty:
                    errors.append(f"{traj_path.name}: no tracks in CSV")
                    continue
                filtered_ids = self._apply_track_filters(metrics_df, params)
                filtered_metrics = metrics_df[metrics_df["track_id"].isin(filtered_ids)].copy()
                cell_id_per_track = None
                if "Cell_ID" in tracks_df.columns and not filtered_metrics.empty:
                    first_cell = tracks_df.groupby("track_id")["Cell_ID"].first()
                    cell_id_per_track = filtered_metrics["track_id"].map(first_cell)
                export_df = pd.DataFrame()
                if not filtered_metrics.empty:
                    export_df["Cell_ID"] = cell_id_per_track if cell_id_per_track is not None else filtered_metrics["track_id"]
                    export_df["TrackID"] = filtered_metrics["track_id"].values
                    export_df["trajectory_length"] = filtered_metrics["trajectory_length"].values.astype(int)
                    export_df["duration_frames"] = filtered_metrics["duration_frames"].values.astype(int)
                    export_df["duration_min"] = np.round(filtered_metrics["duration_min"].values, 2)
                    export_df["first_frame"] = filtered_metrics["first_frame"].values.astype(int)
                    export_df["last_frame"] = filtered_metrics["last_frame"].values.astype(int)
                    # Displacement and path length: full values, integer (µm or px)
                    export_df["displacement_px"] = np.round(filtered_metrics["displacement_px"].values, 0).astype(int)
                    export_df["displacement_um"] = np.round(filtered_metrics["displacement_um"].values, 0).astype(int)
                    export_df["path_length_px"] = np.round(filtered_metrics["path_length_px"].values, 0).astype(int)
                    export_df["path_length_um"] = np.round(filtered_metrics["path_length_um"].values, 0).astype(int)
                    # Speeds: 2 decimals; add µm/s (from µm/min) for alignment with e.g. bacteria µm/s
                    mean_um_min = np.round(filtered_metrics["mean_speed_um_min"].values, 2)
                    straight_um_min = np.round(filtered_metrics["straight_line_speed_um_min"].values, 2)
                    export_df["mean_speed_px_frame"] = np.round(filtered_metrics["mean_speed_px_frame"].values, 2)
                    export_df["mean_speed_um_min"] = mean_um_min
                    export_df["mean_speed_um_s"] = np.round(mean_um_min / 60.0, 2)
                    export_df["straight_line_speed_px_frame"] = np.round(filtered_metrics["straight_line_speed_px_frame"].values, 2)
                    export_df["straight_line_speed_um_min"] = straight_um_min
                    export_df["straight_line_speed_um_s"] = np.round(straight_um_min / 60.0, 2)
                    export_df["straightness"] = np.round(filtered_metrics["straightness"].values, 2)
                else:
                    export_df = pd.DataFrame(columns=[
                        "Cell_ID", "TrackID", "trajectory_length", "duration_frames", "duration_min",
                        "first_frame", "last_frame", "displacement_px", "displacement_um",
                        "path_length_px", "path_length_um", "mean_speed_px_frame", "mean_speed_um_min", "mean_speed_um_s",
                        "straight_line_speed_px_frame", "straight_line_speed_um_min", "straight_line_speed_um_s", "straightness",
                    ])
                out_excel = out_dir / f"{stem}_filtered_tracks.xlsx"
                try:
                    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
                        export_df.to_excel(writer, sheet_name="Filtered tracks", index=False)
                    done += 1
                    if hasattr(self, "parent_window") and hasattr(self.parent_window, "log_output"):
                        self.parent_window.log_output(f"✓ Exported {len(export_df)} tracks to {out_excel.name}\n")
                except Exception as e:
                    errors.append(f"{traj_path.name}: could not write Excel ({e})")
        self.parent_window.progress_bar.setValue(n_files)
        self.parent_window.log_output(f"✓ Export complete: {done}/{n_files} file(s) written.\n")
        QTimer.singleShot(2000, lambda: self.parent_window.progress_bar.setVisible(False))
        msg = f"Exported {done} file(s)."
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors[:15])
            if len(errors) > 15:
                msg += f"\n... and {len(errors) - 15} more."
        QMessageBox.information(self, "Export filtered tracks", msg)

    def _open_trajectory_preview_window(self, traj_path: Path, index: int = 0) -> None:
        """Open a single non-modal preview window for one *_trajectories.tif; supports Re-apply with current settings."""
        try:
            stack = safe_imread(str(traj_path))
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Could not load trajectories file:\n{traj_path}\n\n{str(e)}"
            )
            return
        # Normalize to (T, H, W, 3) for consistent display
        if stack.ndim == 3:
            n_frames, h, w = stack.shape
            frames_rgb = np.stack([np.stack([stack[i], stack[i], stack[i]], axis=-1) for i in range(n_frames)], axis=0)
        elif stack.ndim == 4:
            n_frames = stack.shape[0]
            frames_rgb = []
            for i in range(n_frames):
                frame = stack[i]
                if frame.ndim == 3:
                    if frame.shape[-1] in (1, 3, 4):
                        f = frame[:, :, :3] if frame.shape[-1] >= 3 else np.stack([frame.squeeze(-1)] * 3, axis=-1)
                    elif frame.shape[0] in (1, 3, 4):
                        frame = np.transpose(frame, (1, 2, 0))
                        f = frame[:, :, :3] if frame.shape[-1] >= 3 else np.stack([frame.squeeze(-1)] * 3, axis=-1)
                    else:
                        f = np.stack([frame] * 3, axis=-1)
                else:
                    f = np.stack([frame] * 3, axis=-1)
                frames_rgb.append(f)
            frames_rgb = np.array(frames_rgb, dtype=np.uint8)
        else:
            QMessageBox.warning(
                self, "Preview",
                f"Expected 3D (T, Y, X) or 4D overlay, got {stack.ndim} dimensions."
            )
            return
        tracks_csv, _overlay_path, original_path = self._resolve_from_trajectories_path(traj_path)
        ps, tp = resolve_calibration_from_trajectories_path(traj_path)
        if ps > 0 and hasattr(self, "filter_pixel_size"):
            self.filter_pixel_size.setValue(ps)
        if tp > 0 and hasattr(self, "filter_time_per_frame"):
            self.filter_time_per_frame.setValue(tp)
            if hasattr(self, "filter_time_per_frame_unit"):
                self.filter_time_per_frame_unit.setCurrentText("min")
        main_win = self.window() if hasattr(self, "window") and callable(self.window) else self
        dialog = TrajectoryPreviewDialog(
            self, traj_path, frames_rgb, tracks_csv, original_path, index=index, parent=main_win
        )
        dialog.show()

    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (includes path for workflow replay)."""
        d = get_from_widgets([
            ("model", self.model_combo),
            ("mode", self.mode_combo),
            ("use_gpu", self.use_gpu_cb),
            ("create_vis", self.create_vis_cb),
            ("batch", self.batch_cb),
            ("filter_pixel_size", self.filter_pixel_size),
            ("filter_time_per_frame", self.filter_time_per_frame),
            ("filter_time_per_frame_unit", self.filter_time_per_frame_unit),
        ])
        path_txt = (self.file_path.text() or "").strip()
        if path_txt:
            d["file_path"] = path_txt
        d["performed"] = getattr(self, "_performed_this_session", False)
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (includes path for workflow replay)."""
        apply_to_widgets(data, [
            ("model", self.model_combo),
            ("mode", self.mode_combo),
            ("use_gpu", self.use_gpu_cb),
            ("create_vis", self.create_vis_cb),
            ("batch", self.batch_cb),
            ("filter_pixel_size", self.filter_pixel_size),
            ("filter_time_per_frame", self.filter_time_per_frame),
            ("filter_time_per_frame_unit", self.filter_time_per_frame_unit),
        ])
        if "file_path" in data and data.get("file_path"):
            self.file_path.setText(str(data["file_path"]).strip())
        if "performed" in data:
            self._performed_this_session = bool(data["performed"])


class TrajectoryPreviewDialog(QDialog):
    """Non-modal trajectory preview: shows trajectory TIF; Re-apply with current settings applies filter and redraws."""

    def __init__(
        self,
        tab: "TrackingTab",
        traj_path: Path,
        frames_rgb: np.ndarray,
        tracks_csv: Path | None,
        original_path: Path | None,
        index: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tab = tab
        self._traj_path = traj_path
        self._frames = frames_rgb  # (T, H, W, 3)
        self._tracks_csv = tracks_csv
        self._original_path = original_path
        self.setWindowTitle(f"Trajectories — {traj_path.name}")
        self.setMinimumSize(900, 700)
        self.setModal(False)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        layout = QVBoxLayout()
        self._title_label = QLabel(f"Trajectories — {traj_path.name}")
        self._title_label.setObjectName("traj_preview_title")
        self._title_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 6px;")
        layout.addWidget(self._title_label)
        self._canvas = ImageCanvas()
        self._canvas.setMinimumSize(400, 300)
        self._canvas.setStyleSheet("background-color: #1e1e1e;")
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas, 1)
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        n_frames = self._frames.shape[0]
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._frame_slider.setTickInterval(max(1, n_frames // 10))
        self._frame_label = QLabel("0")
        self._frame_label.setFixedWidth(40)
        slider_row.addWidget(self._frame_slider, 1)
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)
        reapply_btn = QPushButton("Re-apply with current settings")
        reapply_btn.clicked.connect(self._on_reapply)
        layout.addWidget(reapply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, self)
        close_shortcut.activated.connect(self.close)
        self.setLayout(layout)

        def show_frame(idx: int) -> None:
            if idx < 0 or idx >= self._frames.shape[0]:
                return
            frame = self._frames[idx]
            scaled, _ = scale_for_preview(frame, frame)
            pixmap = to_qpixmap(scaled)
            self._canvas.set_pixmap(pixmap, (scaled.shape[1], scaled.shape[0]), reset_view=(idx == 0))
            self._frame_label.setText(str(idx))

        self._frame_slider.valueChanged.connect(show_frame)
        show_frame(0)
        offset_px = 45
        main_win = self._tab.window() if hasattr(self._tab, "window") and callable(self._tab.window) else None
        if main_win:
            top_left = main_win.mapToGlobal(main_win.rect().topLeft())
            self.move(
                top_left.x() + 50 + index * offset_px,
                top_left.y() + 50 + index * offset_px,
            )

    def _on_reapply(self) -> None:
        """Re-read filter params from tab, reload CSV, filter, redraw overlay, update preview."""
        if not self._tracks_csv or not self._tracks_csv.is_file():
            where = f"Looked for: {self._tracks_csv}" if self._tracks_csv else "Path could not be determined."
            QMessageBox.warning(
                self,
                "Re-apply filters",
                "Tracks CSV not found.\n\n"
                "BactiMate looks for the tracks file automatically: same folder as the trajectory file, "
                "same name with _tracks.csv (e.g. stem_trajectories.tif → stem_tracks.csv).\n\n"
                f"{where}\n\nRun tracking first to generate it."
            )
            return
        if not self._original_path or not self._original_path.is_file():
            where = f"Looked for original next to mask: {self._original_path}" if self._original_path else "Original path could not be determined (mask not found)."
            QMessageBox.warning(
                self,
                "Re-apply filters",
                "Original image not found.\n\n"
                "Re-apply needs the original phase/stack image to draw filtered trajectories. "
                "It is looked up from the mask file name (same stem as the trajectory).\n\n"
                f"{where}"
            )
            return
        try:
            tracks_df = pd.read_csv(self._tracks_csv)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Could not read tracks CSV:\n{e}")
            return
        for col in ("track_id", "frame", "y", "x"):
            if col not in tracks_df.columns:
                QMessageBox.warning(self, "Invalid CSV", f"Missing column: {col}")
                return
        params = self._tab._get_filter_params()
        pixel_size = params["pixel_size"]
        time_per_frame = params["time_per_frame"]
        metrics_df = self._tab._compute_track_metrics(tracks_df, pixel_size, time_per_frame)
        if metrics_df.empty:
            return
        filtered_ids = self._tab._apply_track_filters(metrics_df, params)
        filtered_df = tracks_df[tracks_df["track_id"].isin(filtered_ids)].copy()
        try:
            images = safe_imread(str(self._original_path))
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Could not load original stack:\n{e}")
            return
        if images.ndim != 3:
            return
        self._frames = self._tab._draw_filtered_trajectory_overlay(images, filtered_df)
        n_frames = self._frames.shape[0]
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._frame_slider.setTickInterval(max(1, n_frames // 10))
        n_tracks = filtered_df["track_id"].nunique() if not filtered_df.empty else 0
        self._title_label.setText(f"Trajectories — {self._traj_path.name} ({n_tracks} tracks)")
        idx = self._frame_slider.value()
        if idx >= n_frames:
            idx = max(0, n_frames - 1)
            self._frame_slider.setValue(idx)
        frame = self._frames[idx]
        scaled, _ = scale_for_preview(frame, frame)
        pixmap = to_qpixmap(scaled)
        self._canvas.set_pixmap(pixmap, (scaled.shape[1], scaled.shape[0]), reset_view=False)
        self._frame_label.setText(str(idx))


