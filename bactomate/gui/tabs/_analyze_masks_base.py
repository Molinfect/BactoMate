"""
Shared base widget for the Filtering and Analyze stages of the BactoMate pipeline.

This module intentionally contains **no** segmentation or foci-detection UI. Those
belong to Section 1 (Segmentation preview / tuning).

Subclasses:
- `FilteringTab`: mask + foci CSV filtering (writes `filtered_masks/*_mask_filtered.tif`, `foci_filtered/*.csv`)
- `AnalyzeTab`: FI/contour/colocalization reporting (writes `*_analysis*.xlsx`)
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QDoubleSpinBox,
)

from bactomate.gui.preferences import get_default_browse_dir
from bactomate.gui.session_params import apply_to_widgets, get_from_widgets
from bactomate.gui.layout_helpers import (
    module_path_action_buttons_layout,
    module_path_action_outer_layout,
    module_path_drop_row,
    new_module_path_action_panel,
    stacked_path_block,
    style_module_primary_button,
)
from bactomate.gui.styles import PATH_DROP_GROUPBOX_STYLESHEET
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET
from bactomate.gui.widgets import DragDropGroupBox, create_info_icon
from bactomate.gui.log_config import write_gui_log_line
from bactomate.modules.tiff_metadata import (
    read_calibration_from_path_or_original,
    resolve_calibration_for_mask,
)


class AnalyzeMasksBase(QWidget):
    """Shared UI + helpers for FilteringTab and AnalyzeTab."""

    workflow_step_finished = pyqtSignal(bool)  # for workflow replay chain

    # ---- Construction -------------------------------------------------
    def __init__(self, parent: Optional[QWidget] = None, *, embedded: bool = False) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._embedded = bool(embedded)
        self._performed_this_session = False
        self.pixel_size_metadata_loaded = False
        self._analysis_settings_stash: Optional[QWidget] = None
        self._analysis_settings_stash_layout = None
        self._filtering_preview_controls_muted = False
        self._filtering_preview_noop_controls: list[QWidget] = []
        self._filter_preview_dialog = None
        self._filter_preview_mask_path: Optional[Path] = None
        self._module_preview_mount: Optional[QWidget] = None
        self._preview_surface = None
        self._init_ui()

    def attach_module_preview_mount(self, mount: Optional[QWidget]) -> None:
        """Store module-shell right pane mount for embedded filter/analyze previews."""
        self._module_preview_mount = mount
        self._preview_surface = None
        if mount is not None:
            parent = mount.parent()
            while parent is not None:
                if parent.__class__.__name__ == "PreviewSurface":
                    self._preview_surface = parent
                    break
                parent = parent.parent()
        self._sync_embedded_preview_button_visibility()

    def _sync_embedded_preview_button_visibility(self) -> None:
        in_shell = self._module_preview_mount is not None
        if hasattr(self, "preview_action_btn"):
            self.preview_action_btn.setVisible(not in_shell)

    # ---- Required subclass hooks -------------------------------------
    def _stage_key(self) -> str:
        raise NotImplementedError

    def _file_group_title(self) -> str:
        return "Run analysis on masks"

    def _preview_button_label(self) -> str:
        return "Preview"

    def _preview_button_tooltip(self) -> str:
        return "Open preview and adjust settings"

    def _primary_button_label(self) -> str:
        return "Run"

    def _primary_button_tooltip(self) -> str:
        return ""

    def _build_stage_controls(self, scroll_layout: QVBoxLayout) -> None:
        """Insert stage-specific tuning controls into the main layout (optional)."""

    def _build_stage_analysis_subpanel(self) -> Optional[QWidget]:
        """Return optional widget appended under the shared Analysis settings."""
        return None

    def _wire_stage_buttons(self) -> None:
        """Connect preview/analyze buttons to subclass actions."""
        raise NotImplementedError

    def _stage_note_text(self) -> str:
        return "Adjust settings in Preview, then run."

    def _uses_combined_path_actions_group(self) -> bool:
        """When True, path row and Preview/Run buttons share one bordered group."""
        return False

    def _show_preview_action_in_combined_group(self) -> bool:
        return True

    def _extend_combined_path_actions_group(self, outer: QVBoxLayout, _parent: QWidget) -> None:
        """Hook for rows between the path row and action buttons in the combined group."""

    # ---- UI -----------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        # In embedded mode, the host (SegmentationTab) owns outer padding.
        if self._embedded:
            scroll_layout.setContentsMargins(0, 0, 0, 0)
        else:
            scroll_layout.setContentsMargins(8, 4, 8, 8)

        combined_path_actions = self._uses_combined_path_actions_group()
        if combined_path_actions:
            scroll_layout.addWidget(self._build_combined_path_actions_group())
        else:
            scroll_layout.addWidget(self._build_file_group())

        # Stage-specific controls (e.g. filtering peak filters or analyze FI panel)
        self._build_stage_controls(scroll_layout)

        # Shared analysis settings block (morphology/size/options)
        analysis_section = self._build_analysis_settings_section()
        scroll_layout.addWidget(analysis_section)
        # Filtering stage: keep the main tab clean; parameter controls live in the Preview.
        try:
            if self._stage_key() == "filtering":
                analysis_section.hide()
        except Exception:
            pass

        # Historically, Filtering used the preview window as the tuning surface and moved these
        # widgets out of the main UI. For the cleaned-up single-scroll overview, keep controls
        # stable and visible in the main section.
        self._analysis_settings_stash = None
        self._analysis_settings_stash_layout = None

        if not combined_path_actions:
            settings_note = QLabel(self._stage_note_text())
            settings_note.setWordWrap(True)
            settings_note.setStyleSheet("color: #9e9e9e;")
            scroll_layout.addWidget(settings_note)
            scroll_layout.addWidget(self._build_run_actions_section())

        self._wire_stage_buttons()

        self.file_path.textChanged.connect(self._update_action_button_state)
        self.file_path.textChanged.connect(self._try_fill_metadata_from_path)
        self._update_action_button_state()

        if self._embedded:
            # Embedded mode: no nested scroll area; caller owns scrolling.
            self._scroll_area = None
            layout.addWidget(scroll_widget)
        else:
            scroll = QScrollArea()
            scroll.setWidget(scroll_widget)
            scroll.setWidgetResizable(True)
            self._scroll_area = scroll
            layout.addWidget(scroll)
        self.setLayout(layout)

    def _new_path_drop_group(self) -> DragDropGroupBox:
        file_group = DragDropGroupBox(
            "",
            parent=self,
            on_drop_callback=self.handle_dropped_path_analyze,
            base_stylesheet=FLAT_GROUPBOX_FRAME_STYLESHEET,
        )
        file_group.setFlat(True)
        file_group.setStyleSheet(PATH_DROP_GROUPBOX_STYLESHEET)
        file_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        return file_group

    def _init_path_inputs(self) -> tuple[QPushButton, QPushButton]:
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a mask TIFF file or folder… (or drag and drop)")
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        self.file_path.setAcceptDrops(False)

        file_btn = QPushButton("Browse File...")
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.clicked.connect(self.select_folder)
        return file_btn, folder_btn

    def _path_row_widget(self, parent: QWidget, file_btn: QPushButton, folder_btn: QPushButton) -> QWidget:
        return module_path_drop_row(
            parent,
            self.handle_dropped_path_analyze,
            line_edit=self.file_path,
            buttons=(file_btn, folder_btn),
            info_icon=create_info_icon(
                "Path to a mask TIFF file or folder of mask TIFFs.\n\n"
                "Folder selection may point at a segmentation output folder; "
                "overlays/visualizations are ignored.",
                self,
            ),
        )

    def _populate_run_action_buttons(self, run_layout: QHBoxLayout | QVBoxLayout) -> None:
        self.preview_action_btn = QPushButton(self._preview_button_label())
        self.preview_action_btn.setToolTip(self._preview_button_tooltip())
        self.preview_action_btn.setEnabled(False)
        style_module_primary_button(self.preview_action_btn)
        if self._show_preview_action_in_combined_group():
            run_layout.addWidget(self.preview_action_btn)
        else:
            self.preview_action_btn.setVisible(False)

        if self._stage_key() in ("analyze", "filtering"):
            from bactomate.gui.widgets import ProgressCancelButton

            self.primary_action_btn = ProgressCancelButton(self._primary_button_label())
        else:
            self.primary_action_btn = QPushButton(self._primary_button_label())
        self.primary_action_btn.setToolTip(self._primary_button_tooltip())
        self.primary_action_btn.setEnabled(False)
        style_module_primary_button(self.primary_action_btn)
        run_layout.addWidget(self.primary_action_btn)
        self.preview_btn = self.preview_action_btn
        self.analyze_btn = self.primary_action_btn

    def _run_actions_layout(self) -> QHBoxLayout | QVBoxLayout:
        try:
            is_filtering_stage = self._stage_key() == "filtering"
        except Exception:
            is_filtering_stage = False
        if self._embedded or not is_filtering_stage:
            run_layout: QHBoxLayout | QVBoxLayout = QVBoxLayout()
        else:
            run_layout = QHBoxLayout()
        run_layout.setSpacing(6)
        run_layout.setContentsMargins(0, 0, 0, 0)
        return run_layout

    def _build_run_actions_section(self) -> QGroupBox:
        run_section = QGroupBox("")
        run_section.setFlat(True)
        run_section.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        run_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        run_layout = self._run_actions_layout()
        self._populate_run_action_buttons(run_layout)
        run_section.setLayout(run_layout)
        return run_section

    def _build_combined_path_actions_group(self) -> QWidget:
        file_group = new_module_path_action_panel(self)
        file_btn, folder_btn = self._init_path_inputs()
        outer = module_path_action_outer_layout(file_group)
        outer.addWidget(self._path_row_widget(file_group, file_btn, folder_btn))
        self._extend_combined_path_actions_group(outer, file_group)
        run_layout = self._run_actions_layout()
        self._populate_run_action_buttons(run_layout)
        outer.addLayout(run_layout)
        return file_group

    def _build_file_group(self) -> QWidget:
        file_group = self._new_path_drop_group()
        file_btn, folder_btn = self._init_path_inputs()
        outer = QVBoxLayout(file_group)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._path_row_widget(file_group, file_btn, folder_btn))
        file_group.setLayout(outer)
        return file_group

    def _build_analysis_settings_section(self) -> QGroupBox:
        """
        Build the shared "Analysis settings" group.

        Filtering stage uses a wide H-row [Morph | Size | Options | Reset] with the
        pixel-size row above. Analyze stage uses [Pixel size | Outputs | Reset];
        embedded mode stacks those rows vertically. Morph/size/border/CSV controls
        are hidden but still created so session params and tests keep working.
        """
        is_analyze = (self._stage_key() == "analyze")

        analysis_section = QGroupBox("")
        analysis_section.setFlat(True)
        analysis_section.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        analysis_section_layout = QVBoxLayout()
        analysis_section_layout.setSpacing(6)
        analysis_section_layout.setContentsMargins(8, 4, 8, 8)

        # ---- Widgets (always created so session params stay stable) -------
        pixel_info_icon = create_info_icon(
            "Pixel size in µm per pixel for area and distance calculations.\n\n"
            "Auto-filled from TIFF metadata when possible; otherwise defaults to 0.108 µm.",
            parent_widget=self,
        )
        self.pixel_size_indicator = QLabel("⚠ Check value!")
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.pixel_size_indicator.setToolTip("Default value - verify this matches your metadata!")

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 10.0)
        self.pixel_size.setDecimals(3)
        self.pixel_size.setSingleStep(0.001)
        self.pixel_size.setValue(float(0.108))
        self.pixel_size.setFixedWidth(100)
        self.pixel_size.setKeyboardTracking(False)
        self.pixel_size.valueChanged.connect(self._on_pixel_size_changed)

        # Morphology / size widgets (filtering-stage only in the visible UI;
        # always instantiated so session params and tests keep working).
        morph_group = QGroupBox("")
        morph_group.setFlat(True)
        morph_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        morph_layout = QVBoxLayout()
        self.min_aspect_ratio = QDoubleSpinBox()
        self.min_aspect_ratio.setRange(1.0, 20.0)
        self.min_aspect_ratio.setValue(1.5)
        self.min_aspect_ratio.setKeyboardTracking(False)
        self.max_aspect_ratio = QDoubleSpinBox()
        self.max_aspect_ratio.setRange(1.0, 50.0)
        self.max_aspect_ratio.setValue(10.0)
        self.max_aspect_ratio.setKeyboardTracking(False)
        self.min_eccentricity = QDoubleSpinBox()
        self.min_eccentricity.setRange(0.0, 1.0)
        self.min_eccentricity.setValue(0.5)
        self.min_eccentricity.setDecimals(2)
        self.min_eccentricity.setKeyboardTracking(False)
        self.min_solidity = QDoubleSpinBox()
        self.min_solidity.setRange(0.0, 1.0)
        self.min_solidity.setValue(0.7)
        self.min_solidity.setDecimals(2)
        self.min_solidity.setKeyboardTracking(False)

        def _row(target: QVBoxLayout, lbl: str, w: QWidget) -> None:
            r = QHBoxLayout()
            r.addWidget(QLabel(lbl))
            r.addStretch()
            r.addWidget(w)
            target.addLayout(r)

        _row(morph_layout, "Min Aspect Ratio:", self.min_aspect_ratio)
        _row(morph_layout, "Max Aspect Ratio:", self.max_aspect_ratio)
        _row(morph_layout, "Min Eccentricity:", self.min_eccentricity)
        _row(morph_layout, "Min Solidity:", self.min_solidity)
        morph_group.setLayout(morph_layout)

        size_group = QGroupBox("")
        size_group.setFlat(True)
        size_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        size_layout = QVBoxLayout()
        self.min_area = QDoubleSpinBox()
        self.min_area.setRange(0, 10000)
        self.min_area.setValue(0)
        self.min_area.setKeyboardTracking(False)
        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(0, 10000)
        self.max_area.setValue(0)
        self.max_area.setKeyboardTracking(False)
        self.min_width = QDoubleSpinBox()
        self.min_width.setRange(0, 100)
        self.min_width.setValue(0)
        self.min_width.setKeyboardTracking(False)
        self.max_width = QDoubleSpinBox()
        self.max_width.setRange(0, 100)
        self.max_width.setValue(0)
        self.max_width.setKeyboardTracking(False)
        self.min_length = QDoubleSpinBox()
        self.min_length.setRange(0, 100)
        self.min_length.setValue(0)
        self.min_length.setKeyboardTracking(False)

        _row(size_layout, "Min Area (µm²):", self.min_area)
        _row(size_layout, "Max Area (µm²):", self.max_area)
        _row(size_layout, "Min Width (µm):", self.min_width)
        _row(size_layout, "Max Width (µm):", self.max_width)
        _row(size_layout, "Min Length (µm):", self.min_length)
        size_group.setLayout(size_layout)

        # Output toggles
        self.detailed = QCheckBox("Per-cell data")
        self.detailed.setChecked(True)
        # Analyze stage relabels this; filtering keeps the historical name.
        self.create_vis = QCheckBox("Create visualization" if is_analyze else "Create filter visualization")
        self.create_vis.setChecked(True)
        self.create_vis.setVisible(False)
        if is_analyze:
            self.create_vis.setToolTip(
                "After analysis, save one multi-panel TIFF in _analysis/:\n"
                "Cells (phase + Cell_ID labels) | each foci channel (peaks + foci_id labels)."
            )
        self.export_csv = QCheckBox("Also save CSV")
        self.export_csv.setChecked(False)
        self.export_csv.setToolTip(
            "Also write Summary, Per-Cell Data, and Foci (if present) as CSV files "
            "next to each *_analysis.xlsx. Off by default to keep _analysis/ tidy."
        )
        self.create_combined_summary: QCheckBox | None = None
        if is_analyze:
            self.create_combined_summary = QCheckBox("Create combined summary")
            self.create_combined_summary.setChecked(False)
            self.create_combined_summary.setToolTip(
                "After analyzing a folder with multiple samples, combine all analysis exports "
                "into one workbook ({folder}_combined_summary.xlsx in _analysis/): stacked "
                "Summary, Per-Cell, and Foci sheets plus a wide time-series pivot.\n\n"
                "Off by default — turn on for multi-sample batches."
            )
        self.exclude_border_cb = QCheckBox("Exclude masks at borders")
        self.exclude_border_cb.setChecked(False)
        self._filtering_preview_noop_controls = [self.detailed, self.create_vis, self.export_csv]

        reset_btn: QPushButton | None = None
        if not is_analyze:
            reset_btn = QPushButton("Reset to Default")
            reset_btn.setToolTip("Reset all analysis parameters to their default values")
            reset_btn.clicked.connect(self.reset_analysis_defaults)

        # ---- Layout assembly ----------------------------------------------
        if is_analyze:
            pixel_grid = QGridLayout()
            pixel_grid.setHorizontalSpacing(10)
            pixel_grid.setVerticalSpacing(6)
            pixel_grid.setContentsMargins(0, 0, 0, 0)
            px_lbl = QLabel("Pixel size (µm):")
            px_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            pixel_grid.addWidget(px_lbl, 0, 0)
            self.pixel_size.setMinimumWidth(72)
            self.pixel_size.setMaximumWidth(100)
            self.pixel_size.setFixedHeight(32)
            self.pixel_size.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            pixel_grid.addWidget(self.pixel_size, 0, 1)
            self._sync_pixel_size_indicator_geometry()
            pixel_grid.addWidget(self.pixel_size_indicator, 0, 2)
            pixel_grid.addWidget(self.detailed, 1, 0, 1, 2)
            pixel_grid.addWidget(self.create_vis, 1, 2, 1, 2)
            pixel_grid.setColumnStretch(3, 1)
            analysis_section_layout.addLayout(pixel_grid)

            outputs_row = QHBoxLayout()
            outputs_row.setSpacing(12)
            outputs_row.setContentsMargins(0, 0, 0, 0)
            outputs_row.addWidget(self.export_csv)
            if self.create_combined_summary is not None:
                outputs_row.addWidget(self.create_combined_summary)
            outputs_row.addWidget(
                create_info_icon(
                    "Also save CSV — writes *_summary.csv, *_per_cell.csv, and *_foci.csv "
                    "beside each analysis Excel (optional; off by default).\n\n"
                    "Create combined summary — after a multi-sample folder batch, writes "
                    "{folder}_combined_summary.xlsx in _analysis/ with stacked sheets: "
                    "Combined Summary (long), Combined Summary (wide), Combined Per-Cell Data, "
                    "and Combined Foci (when foci analysis was run). Off by default.",
                    self,
                ),
                0,
            )
            outputs_row.addStretch(1)
            analysis_section_layout.addLayout(outputs_row)

            # Park hidden widgets in a hidden holder so they remain alive but never paint.
            hidden_holder = QWidget()
            hidden_holder.setVisible(False)
            hh = QHBoxLayout(hidden_holder)
            hh.setContentsMargins(0, 0, 0, 0)
            hh.addWidget(morph_group)
            hh.addWidget(size_group)
            hh.addWidget(self.exclude_border_cb)
            analysis_section_layout.addWidget(hidden_holder)
            self._analyze_hidden_settings_holder = hidden_holder
        else:
            # Filtering: pixel-size row above a wide [Morph | Size | Options | Reset] container.
            pixel_layout = QHBoxLayout()
            pixel_layout.addWidget(QLabel("Pixel Size (µm):"))
            pixel_layout.addWidget(pixel_info_icon)
            pixel_layout.addWidget(self.pixel_size_indicator)
            pixel_layout.addStretch()
            pixel_layout.addWidget(self.pixel_size)
            analysis_section_layout.addLayout(pixel_layout)

            options_group = QGroupBox("")
            options_group.setFlat(True)
            options_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
            options_layout = QVBoxLayout()
            for cb in (self.detailed, self.create_vis, self.export_csv, self.exclude_border_cb):
                options_layout.addWidget(cb)
            filtering_note = QLabel("Note: Detailed/CSV/Visualization options are Analyze-stage outputs only.")
            filtering_note.setWordWrap(True)
            filtering_note.setStyleSheet("color: #9e9e9e; font-size: 9pt;")
            options_layout.addWidget(filtering_note)
            options_group.setLayout(options_layout)

            container = QWidget()
            container_layout = QHBoxLayout()
            container_layout.setSpacing(12)
            container_layout.setContentsMargins(6, 6, 6, 6)
            container_layout.addWidget(morph_group)
            container_layout.addWidget(size_group)
            container_layout.addWidget(options_group)
            if reset_btn is not None:
                container_layout.addWidget(reset_btn)
            container.setLayout(container_layout)
            analysis_section_layout.addWidget(container)

        stage_panel = self._build_stage_analysis_subpanel()
        if stage_panel is not None:
            analysis_section_layout.addWidget(stage_panel)

        analysis_section.setLayout(analysis_section_layout)
        self.analysis_section = analysis_section
        return analysis_section

    def _sync_pixel_size_indicator_geometry(self) -> None:
        """Keep pixel-size warning label compact on the analyze settings row."""
        ind = self.pixel_size_indicator
        ind.setMinimumWidth(0)
        ind.setMaximumWidth(16777215)
        ind.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

    # ---- Preview settings stash helpers -------------------------------
    @staticmethod
    def _move_widget_between_layouts(widget: QWidget, target_layout) -> None:
        if widget is None or target_layout is None:
            return
        old_parent = widget.parentWidget()
        if old_parent is not None and old_parent.layout() is not None:
            old_parent.layout().removeWidget(widget)
        target_layout.addWidget(widget)

    def _place_analysis_settings_in_preview(self, target_layout) -> None:
        section = getattr(self, "analysis_section", None)
        if section is None:
            return
        self._move_widget_between_layouts(section, target_layout)
        self._set_filtering_preview_noop_controls_muted(True)

    def _restore_analysis_settings_to_stash(self) -> None:
        section = getattr(self, "analysis_section", None)
        stash_layout = getattr(self, "_analysis_settings_stash_layout", None)
        if section is None or stash_layout is None:
            return
        self._move_widget_between_layouts(section, stash_layout)
        self._set_filtering_preview_noop_controls_muted(False)

    def _set_filtering_preview_noop_controls_muted(self, muted: bool) -> None:
        """Filtering preview keeps non-functional output toggles visible but disabled with an explanation."""
        try:
            if self._stage_key() != "filtering":
                return
        except Exception:
            return
        self._filtering_preview_controls_muted = bool(muted)
        msg = "Analyze-only output option. It does not affect Filtering preview or filtering artifacts."
        for w in getattr(self, "_filtering_preview_noop_controls", []):
            if w is None:
                continue
            try:
                w.setEnabled(not muted)
                w.setToolTip(msg if muted else "")
            except Exception:
                pass

    # ---- Common behaviors --------------------------------------------
    def _update_action_button_state(self) -> None:
        has_path = bool(self.file_path.text() and self.file_path.text().strip())
        if hasattr(self, "primary_action_btn"):
            self.primary_action_btn.setEnabled(has_path)
        if hasattr(self, "preview_action_btn"):
            self.preview_action_btn.setEnabled(has_path)

    # Backward-compatible method alias.
    def _update_analyze_button_state(self) -> None:
        self._update_action_button_state()

    def _calibration_probe_path(self, path: Path) -> Optional[Path]:
        """Pick a TIFF to read calibration from (mask → original phase stack when possible)."""
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            return path
        if path.is_dir():
            masks = self._get_mask_files_for_feedback()
            if masks:
                return masks[0]
            all_tifs = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
            return next((f for f in all_tifs if f.is_file()), None)
        return None

    def _apply_pixel_size_from_calibration(self, ps: float, *, indicator: str = "(from file)") -> None:
        if ps <= 0:
            return
        self.pixel_size.setValue(float(ps))
        self.pixel_size_indicator.setText(indicator)
        self.pixel_size_indicator.setStyleSheet("color: #4CAF50; font-size: 9pt;")
        self._sync_pixel_size_indicator_geometry()
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size_indicator.show()
        self.pixel_size_metadata_loaded = True

    def _try_fill_metadata_from_path(self) -> None:
        path_text = (self.file_path.text() or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        tiff_path = self._calibration_probe_path(path)
        if tiff_path is None:
            return
        try:
            ps, tp, src, kind = resolve_calibration_for_mask(tiff_path)
            if ps <= 0:
                ps, tp = read_calibration_from_path_or_original(tiff_path)
            if ps <= 0:
                parent = getattr(self, "parent_window", None)
                sm = getattr(parent, "shared_metadata", None) if parent is not None else None
                if isinstance(sm, dict) and sm.get("pixel_size") is not None:
                    try:
                        ps = float(sm["pixel_size"])
                    except (TypeError, ValueError):
                        ps = 0.0
            if ps > 0:
                ind = "(from file)"
                if kind == "original" and src is not None:
                    ind = f"(from {src.name})"
                elif kind == "mask" and src is not None:
                    ind = "(from mask file)"
                self._apply_pixel_size_from_calibration(ps, indicator=ind)
            if tp > 0 and hasattr(self, "parent_window") and getattr(self.parent_window, "shared_metadata", None) is not None:
                self.parent_window.shared_metadata["time_per_frame_min"] = tp
        except Exception:
            pass

    def _on_pixel_size_changed(self) -> None:
        if not self.pixel_size_metadata_loaded:
            self.pixel_size_indicator.setText("")
            self.pixel_size_indicator.setFixedWidth(0)
            self.pixel_size_indicator.hide()

    def _open_output_folder(self, folder: Path | str | None) -> None:
        if folder is None:
            return
        try:
            p = Path(folder)
            if p.is_file():
                p = p.parent
            if p.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception:
            pass

    def _show_finished_dialog_with_output_folder(
        self,
        *,
        title: str,
        message: str,
        output_folder: Path | str | None = None,
        icon: str | None = None,
        on_ok=None,
    ) -> None:
        from bactomate.gui.widgets import show_task_complete_dialog

        _ = icon  # legacy icon hint; message text carries status
        show_task_complete_dialog(
            self,
            title=title,
            message=message,
            output_dir=output_folder,
            on_ok=on_ok,
        )

    def reset_analysis_defaults(self) -> None:
        self.pixel_size.setValue(float(0.108))
        self.pixel_size_metadata_loaded = False
        self.pixel_size_indicator.setText("⚠ Check value!")
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self._sync_pixel_size_indicator_geometry()
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size_indicator.show()
        self.min_aspect_ratio.setValue(1.5)
        self.max_aspect_ratio.setValue(10.0)
        self.min_eccentricity.setValue(0.5)
        self.min_solidity.setValue(0.7)
        self.min_area.setValue(0)
        self.max_area.setValue(0)
        self.min_width.setValue(0)
        self.max_width.setValue(0)
        self.min_length.setValue(0)
        self.detailed.setChecked(True)
        self.create_vis.setChecked(True)
        self.export_csv.setChecked(False)
        if getattr(self, 'create_combined_summary', None) is not None:
            self.create_combined_summary.setChecked(False)
        self.exclude_border_cb.setChecked(False)

    # ---- File/folder selection ---------------------------------------
    def select_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mask TIFF File",
            get_default_browse_dir(),
            "TIFF Files (*.tif *.tiff);;All Files (*)",
        )
        if file_path:
            self.file_path.setText(file_path)
            self._update_action_button_state()

    def select_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Mask Files",
            get_default_browse_dir(),
        )
        if folder_path:
            self.file_path.setText(folder_path)
            self._update_action_button_state()

    def handle_dropped_path_analyze(self, dropped_path: str) -> None:
        self.file_path.setText(str(dropped_path))
        self._update_action_button_state()

    def _is_folder_mode(self) -> bool:
        if not self.file_path.text():
            return False
        p = Path(self.file_path.text())
        return p.exists() and p.is_dir()

    def _get_mask_files_for_feedback(self) -> list[Path]:
        """Segmentation / filtered mask TIFFs only (not phase, FI, or foci stacks)."""
        from bactomate.gui.artifact_paths import discover_mask_tiffs_for_workspace

        path_text = (self.file_path.text() or "").strip()
        if not path_text:
            return []
        return discover_mask_tiffs_for_workspace(Path(path_text))

    # ---- Name matching + discovery helpers ---------------------------
    @staticmethod
    def _looks_non_channel_token(token: str) -> bool:
        t = (token or "").strip().lower()
        if not t:
            return True
        if t in {"fi", "foci", "mask", "masks", "phase", "bf", "dic", "seg", "segmentation"}:
            return True
        if t.startswith("t=") or t.startswith("z="):
            return True
        if t.endswith("x") and t[:-1].isdigit():
            return True
        return False

    def _candidate_match_stems(self, stem: str) -> list[str]:
        out: list[str] = []

        def _add(s: str) -> None:
            s = (s or "").strip("_")
            if s and s not in out:
                out.append(s)

        _add(stem)
        # Common pipeline suffixes that shouldn't participate in matching other artifacts.
        # Example dataset:
        #   mask:  G183A-004_Phase60x_mask_filtered.tif
        #   foci:  G183A-004_foci_dsRed.tif
        # so we must consider the shorter "G183A-004" stem as a match candidate.
        low = stem.lower()
        if low.endswith("_mask_filtered"):
            _add(stem[: -len("_mask_filtered")])
        if low.endswith("_mask"):
            _add(stem[: -len("_mask")])
        if "_phase" in low:
            try:
                i = low.index("_phase")
                _add(stem[:i])
            except Exception:
                pass
        if "_" in stem:
            _add(stem.rsplit("_", 1)[0])
        if "__" in stem:
            s2 = stem.split("__", 1)[0]
            _add(s2)
            if "_" in s2:
                _add(s2.rsplit("_", 1)[0])
        try:
            from bactomate.gui.artifact_paths import sample_identity_stem

            _add(sample_identity_stem(stem))
        except Exception:
            pass
        return out

    def _matches_fi_name(self, name: str, stem: str) -> bool:
        if name == f"{stem}_fi" or name.startswith(f"{stem}_fi_"):
            return True
        if name.startswith(f"{stem}_") and name.endswith("_fi"):
            return True
        # FI channels often share foci-style names, e.g. sample_fi_foci_GFP.tif
        nl, sl = name.lower(), stem.lower()
        if nl.startswith(sl):
            if len(nl) == len(sl):
                return nl in (f"{sl}_fi", f"{sl}_fi_foci")
            if nl[len(sl)] != "_":
                return False
            rest = nl[len(sl) :].lstrip("_")
            if rest.startswith("fi_foci") or (rest.startswith("fi") and "_foci" in rest):
                return True
        return False

    def _matches_foci_name(self, name: str, stem: str) -> bool:
        """Match ``{sample}_foci_*``, ``{sample}__fi_foci_*`` (file conversion), etc."""
        nl, sl = name.lower(), stem.lower()
        if not nl.startswith(sl):
            return False
        if len(nl) == len(sl):
            return nl in (f"{sl}_foci", f"{sl}_fi_foci")
        boundary = nl[len(sl)]
        if boundary not in ("_",):
            return False
        rest = nl[len(sl) :].lstrip("_")
        if not rest:
            return False
        if rest.startswith("foci"):
            return True
        if rest.startswith("fi") and "_foci" in rest:
            return True
        return "_foci" in rest

    def _extract_channel_tag_for_mask_stem(self, file_stem: str, mask_stem: str) -> str | None:
        for s in self._candidate_match_stems(mask_stem):
            if file_stem in (f"{s}_fi", f"{s}_foci", f"{s}_fi_foci", f"{s}_foci_fi"):
                return None
            if file_stem.startswith(f"{s}_fi_"):
                rem = file_stem[len(s) + 4 :].strip("_")
                if not rem or rem == "foci":
                    return None
                if rem.startswith("foci_"):
                    rem = rem[len("foci_") :].strip("_")
                if rem.endswith("_foci"):
                    rem = rem[: -len("_foci")].strip("_")
                return rem or None
            if file_stem.startswith(f"{s}_foci_"):
                rem = file_stem[len(s) + 6 :].strip("_")
                if rem.lower().startswith("overlay_"):
                    rem = rem[8:].strip("_")
                return rem or None
            # File conversion: sample__fi_foci_GFP.tif (double underscore after sample id)
            ls, ss = file_stem.lower(), s.lower()
            if ls.startswith(ss):
                rem2 = file_stem[len(s) :].lstrip("_")
                if rem2.lower().startswith("foci_"):
                    tag = rem2[5:].strip("_")
                    if tag.lower().startswith("overlay_"):
                        tag = tag[8:].strip("_")
                    return tag or None
                m = re.match(r"^(\d+_)?foci_(?P<ch>.+)$", rem2, flags=re.IGNORECASE)
                if m:
                    return (m.group("ch") or "").strip("_") or None
                if rem2.lower().startswith("fi_foci_"):
                    tag = rem2[8:].strip("_")
                    return tag or None
                m = re.match(r"^(\d+_)?fi_foci_(?P<ch>.+)$", rem2, flags=re.IGNORECASE)
                if m:
                    return (m.group("ch") or "").strip("_") or None
            if file_stem.startswith(f"{s}_") and file_stem.endswith("_foci"):
                rem = file_stem[len(s) + 1 : -len("_foci")].strip("_")
                return rem or None
            if file_stem.startswith(f"{s}_") and file_stem.endswith("_fi"):
                rem = file_stem[len(s) + 1 : -len("_fi")].strip("_")
                return rem or None
        if file_stem.endswith("_foci"):
            prefix = file_stem[: -len("_foci")].strip("_")
            if "_" in prefix:
                token = prefix.rsplit("_", 1)[-1].strip("_")
                if token and not self._looks_non_channel_token(token):
                    return token
        return None

    def _filter_artifact_paths_for_mask(self, mask_path: Path, paths: list[Path]) -> list[Path]:
        """Drop sibling replicates (e.g. cl1__ vs cl1_001_) that share a shortened match stem."""
        from bactomate.gui.filter_preview_utils import filter_paths_for_mask

        return filter_paths_for_mask(self, mask_path, paths)

    def _filtering_tab(self) -> Any | None:
        """Segmentation host tab that owns Filtering settings (if this widget is under it)."""
        host = self
        for _ in range(16):
            if host is None:
                break
            ft = getattr(host, "filtering_tab", None)
            if ft is not None:
                return ft
            try:
                host = host.parent()
            except Exception:
                break
        return None

    def _foci_filter_settings_for_source(self, foci_source: Path, mask_path: Path) -> dict[str, Any]:
        """Per-channel foci filter settings from the Filtering stage (preview / batch)."""
        from bactomate.gui.filter_preview_utils import default_foci_channel_settings

        ft = self._filtering_tab()
        if ft is not None and hasattr(ft, "get_settings_for_foci_source"):
            try:
                return dict(ft.get_settings_for_foci_source(Path(foci_source)))
            except Exception:
                pass
        return dict(default_foci_channel_settings())

    def _resolve_fi_paths_for_mask(self, mask_path: Path) -> list[Path]:
        from bactomate.gui.analysis_paths import artifact_search_dirs_for_mask

        stem = mask_path.stem.replace("_masks", "").replace("_mask_filtered", "")
        stems = self._candidate_match_stems(stem)

        def search_in(directory: Path) -> list[Path]:
            out: list[Path] = []
            if not directory.exists() or not directory.is_dir():
                return out
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                low = f.name.lower()
                if any((token in low for token in ("_masks", "_mask_filtered", "_overlay", "_analysis", "_filter_visualization"))):
                    continue
                if any(self._matches_fi_name(f.stem, s) for s in stems):
                    out.append(f)
            return out

        found: list[Path] = []
        seen_paths: set[str] = set()
        for d in artifact_search_dirs_for_mask(mask_path):
            for p in search_in(d):
                try:
                    key = str(p.resolve())
                except Exception:
                    key = str(p)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                found.append(p)
        return self._filter_artifact_paths_for_mask(mask_path, found)

    def _resolve_fi_inputs_for_mask(self, mask_path: Path) -> list[Path]:
        """FI stacks for Analyze FI: *_fi* names plus foci TIFFs for any missing channel."""
        from bactomate.gui.analysis_paths import merge_fi_input_paths, stem_from_mask_path

        mask_path = Path(mask_path)
        mask_stem = stem_from_mask_path(mask_path)

        def _channel_tag(p: Path) -> str:
            try:
                return str(self._extract_channel_tag_for_mask_stem(p.stem, mask_stem) or "")
            except Exception:
                return ""

        return merge_fi_input_paths(
            mask_path=mask_path,
            explicit_fi=list(self._resolve_fi_paths_for_mask(mask_path) or []),
            foci_stacks=list(self._resolve_foci_paths_for_mask(mask_path) or []),
            channel_tag_for_path=_channel_tag,
            filter_paths=self._filter_artifact_paths_for_mask,
        )

    def _resolve_foci_paths_for_mask(self, mask_path: Path) -> list[Path]:
        from bactomate.gui.analysis_paths import artifact_search_dirs_for_mask, stem_from_mask_path
        from bactomate.gui.artifact_paths import discover_foci_source_tiffs

        unique_dirs = artifact_search_dirs_for_mask(mask_path)
        stem = stem_from_mask_path(mask_path)
        stems = self._candidate_match_stems(stem)

        def search_in(directory: Path) -> list[Path]:
            out: list[Path] = []
            if not directory.exists() or not directory.is_dir():
                return out
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if any(self._matches_foci_name(f.stem, s) for s in stems):
                    out.append(f)
            return out

        found: list[Path] = []
        seen_paths: set[str] = set()
        for d in unique_dirs:
            for p in search_in(d):
                try:
                    key = str(p.resolve())
                except Exception:
                    key = str(p)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                found.append(p)
            for s in stems:
                for p in discover_foci_source_tiffs(s, d):
                    try:
                        key = str(p.resolve())
                    except Exception:
                        key = str(p)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    found.append(p)
        return self._filter_artifact_paths_for_mask(mask_path, found)

    def _preview_fi_path_candidates(self, mask_path: Path) -> list[Path]:
        """Preview helper: includes both FI and foci stacks (for background/points selection)."""
        candidates: list[Path] = []
        try:
            candidates.extend(self._resolve_fi_paths_for_mask(mask_path))
        except Exception:
            pass
        try:
            candidates.extend(self._resolve_foci_paths_for_mask(mask_path))
        except Exception:
            pass
        # De-dupe
        out: list[Path] = []
        seen: set[str] = set()
        for p in candidates:
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    def _foci_path_for_fi_path(self, mask_path: Path, fi_path: Path, foci_paths: list[Path]) -> Path | None:
        if not foci_paths:
            return None
        mask_stem = mask_path.stem.replace("_masks", "")
        fi_tag = self._extract_channel_tag_for_mask_stem(fi_path.stem, mask_stem)
        if fi_tag:
            tagged = [
                p
                for p in foci_paths
                if (self._extract_channel_tag_for_mask_stem(p.stem, mask_stem) or "").strip().lower()
                == fi_tag.strip().lower()
            ]
            if len(tagged) == 1:
                return tagged[0]
            if len(tagged) > 1:
                return sorted(tagged, key=lambda p: p.name.lower())[0]
        if len(foci_paths) == 1:
            return foci_paths[0]
        return None

    # ---- Session params ----------------------------------------------
    def get_session_params(self) -> dict[str, Any]:
        d = get_from_widgets(
            [
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
                *(
                    [("create_combined_summary", self.create_combined_summary)]
                    if getattr(self, "create_combined_summary", None) is not None
                    else []
                ),
                ("exclude_border", self.exclude_border_cb),
            ]
        )
        d["stage"] = self._stage_key()
        path_txt = (self.file_path.text() or "").strip()
        if path_txt:
            d["file_path"] = path_txt
        d["performed"] = getattr(self, "_performed_this_session", False)
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        stage = str(data.get("stage", "") or "").strip().lower()
        if stage and stage != self._stage_key():
            # Reject cross-stage application; this avoids silent misconfiguration.
            write_gui_log_line("Session", f"Skipping session params for stage={stage} on tab stage={self._stage_key()}")
            return
        apply_to_widgets(
            data,
            [
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
                *(
                    [("create_combined_summary", self.create_combined_summary)]
                    if getattr(self, "create_combined_summary", None) is not None
                    else []
                ),
                ("exclude_border", self.exclude_border_cb),
            ],
        )
        if "file_path" in data and data.get("file_path"):
            self.file_path.setText(str(data["file_path"]).strip())
        if "performed" in data:
            self._performed_this_session = bool(data["performed"])
