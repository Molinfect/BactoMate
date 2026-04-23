"""
Shared base widget for the Filtering and Analyze stages of the BactiMate pipeline.

This module intentionally contains **no** segmentation or foci-detection UI. Those
belong to Section 1 (Segmentation preview / tuning).

Subclasses:
- `FilteringTab`: mask + foci CSV filtering (writes `*_mask_filtered.tif`, `*_foci_filtered*.csv`)
- `AnalyzeTab`: FI/contour/colocalization reporting (writes `*_analysis*.xlsx`)
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QDoubleSpinBox,
)

from bactimate.gui.preferences import get_default_browse_dir
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon
from bactimate.gui.log_config import write_gui_log_line
from bactimate.modules.tiff_metadata import read_calibration_from_path_or_original


class AnalyzeMasksBase(QWidget):
    """Shared UI + helpers for FilteringTab and AnalyzeTab."""

    workflow_step_finished = pyqtSignal(bool)  # for workflow replay chain

    # ---- Construction -------------------------------------------------
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._performed_this_session = False
        self.pixel_size_metadata_loaded = False
        self._analysis_settings_stash: Optional[QWidget] = None
        self._analysis_settings_stash_layout = None
        self._filter_preview_dialog = None
        self._init_ui()

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

    # ---- UI -----------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(6)
        scroll_layout.setContentsMargins(8, 4, 8, 8)

        file_group = self._build_file_group()
        scroll_layout.addWidget(file_group)

        # Stage-specific controls (e.g. filtering peak filters or analyze FI panel)
        self._build_stage_controls(scroll_layout)

        # Shared analysis settings block (morphology/size/options)
        analysis_section = self._build_analysis_settings_section()
        scroll_layout.addWidget(analysis_section)

        # Stash analysis settings out of the main tab only for the Filtering stage,
        # where the preview window is the tuning surface. Analyze stage has no preview,
        # so these controls must remain visible on the tab.
        try:
            stash_settings = (self._stage_key() == "filtering")
        except Exception:
            stash_settings = False
        if stash_settings:
            self._analysis_settings_stash = QWidget(self)
            self._analysis_settings_stash.hide()
            stash_layout = QVBoxLayout(self._analysis_settings_stash)
            stash_layout.setContentsMargins(0, 0, 0, 0)
            stash_layout.setSpacing(0)
            stash_layout.addWidget(analysis_section)
            self._analysis_settings_stash_layout = stash_layout
        else:
            self._analysis_settings_stash = None
            self._analysis_settings_stash_layout = None

        settings_note = QLabel(self._stage_note_text())
        settings_note.setWordWrap(True)
        settings_note.setStyleSheet("color: #9e9e9e;")
        scroll_layout.addWidget(settings_note)

        # Preview and run
        run_section = QGroupBox("Preview and run")
        run_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        # Filtering stage wants side-by-side buttons for quick iteration.
        try:
            is_filtering_stage = (self._stage_key() == "filtering")
        except Exception:
            is_filtering_stage = False
        run_layout = QHBoxLayout() if is_filtering_stage else QVBoxLayout()
        run_layout.setSpacing(10 if is_filtering_stage else 6)
        run_layout.setContentsMargins(8, 8, 8, 8)

        self.preview_btn = QPushButton(self._preview_button_label())
        self.preview_btn.setToolTip(self._preview_button_tooltip())
        self.preview_btn.setEnabled(False)
        self.preview_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        run_layout.addWidget(self.preview_btn, 1 if is_filtering_stage else 0)

        self.analyze_btn = QPushButton(self._primary_button_label())
        self.analyze_btn.setToolTip(self._primary_button_tooltip())
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        run_layout.addWidget(self.analyze_btn, 1 if is_filtering_stage else 0)

        run_section.setLayout(run_layout)
        scroll_layout.addWidget(run_section)

        self._wire_stage_buttons()

        self.file_path.textChanged.connect(self._update_analyze_button_state)
        self.file_path.textChanged.connect(self._try_fill_metadata_from_path)
        self._update_analyze_button_state()

        scroll_widget.setLayout(scroll_layout)
        scroll_widget.setMinimumWidth(800)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        self._scroll_area = scroll
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_file_group(self) -> QWidget:
        file_group = DragDropGroupBox(
            self._file_group_title(),
            parent=self,
            on_drop_callback=self.handle_dropped_path_analyze,
        )
        file_font = QFont()
        file_font.setPointSize(12)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Path:"))
        file_layout.addWidget(
            create_info_icon(
                "Path to a mask TIFF file or folder of mask TIFFs.\n\n"
                "Folder selection may point at a segmentation output folder; overlays/ visualizations are ignored.",
                self,
            )
        )
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a mask TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(200)
        self.file_path.setFont(file_font)
        self.file_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.file_path.setAcceptDrops(False)

        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)

        file_layout.addWidget(self.file_path, 1)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        file_group.setLayout(file_layout)
        return file_group

    def _build_analysis_settings_section(self) -> QGroupBox:
        # This is essentially the existing “Analysis settings” UI: pixel size + morphology + size + options + reset.
        analysis_section = QGroupBox("Analysis settings")
        analysis_section_layout = QVBoxLayout()
        analysis_section_layout.setSpacing(6)
        analysis_section_layout.setContentsMargins(8, 4, 8, 8)

        container = QWidget()
        analysis_container_layout = QHBoxLayout()
        analysis_container_layout.setSpacing(12)
        analysis_container_layout.setContentsMargins(6, 6, 6, 6)

        # Pixel size
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("Pixel Size (µm):"))
        pixel_layout.addWidget(
            create_info_icon(
                "Pixel size in µm per pixel for area and distance calculations.\n\n"
                "Auto-filled from TIFF metadata when possible; otherwise defaults to 0.108 µm.",
                parent_widget=self,
            )
        )
        self.pixel_size_indicator = QLabel("⚠ Check value!")
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self.pixel_size_indicator.setFixedWidth(100)
        self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size_indicator.setToolTip("Default value - verify this matches your metadata!")
        pixel_layout.addWidget(self.pixel_size_indicator)
        pixel_layout.addStretch()
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.001, 10.0)
        self.pixel_size.setDecimals(3)
        self.pixel_size.setSingleStep(0.001)
        self.pixel_size.setValue(float(0.108))
        self.pixel_size.setFixedWidth(100)
        self.pixel_size.setKeyboardTracking(False)
        self.pixel_size.valueChanged.connect(self._on_pixel_size_changed)
        pixel_layout.addWidget(self.pixel_size)
        analysis_section_layout.addLayout(pixel_layout)

        # Morphology group (minimal; subclasses/tests care about widget existence, not layout cosmetics)
        morph_group = QGroupBox("Morphological Filters")
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
        analysis_container_layout.addWidget(morph_group)

        size_group = QGroupBox("Size Filters")
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
        analysis_container_layout.addWidget(size_group)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        self.detailed = QCheckBox("Detailed per-cell data")
        self.detailed.setChecked(True)
        # Label varies by stage: filtering writes a filter visualization, analyze writes analysis visualizations.
        self.create_vis = QCheckBox("Create filter visualization")
        self.create_vis.setChecked(True)
        self.export_csv = QCheckBox("Also save CSV")
        self.export_csv.setChecked(False)
        self.exclude_border_cb = QCheckBox("Exclude masks at borders")
        self.exclude_border_cb.setChecked(False)
        for cb in (self.detailed, self.create_vis, self.export_csv, self.exclude_border_cb):
            options_layout.addWidget(cb)
        options_group.setLayout(options_layout)
        analysis_container_layout.addWidget(options_group)

        # Analyze stage should not expose mask-filter tuning controls (those live in Filter Preview).
        # Keep widgets alive for session persistence and compatibility, but hide them in the UI.
        try:
            if self._stage_key() == "analyze":
                morph_group.hide()
                size_group.hide()
                self.exclude_border_cb.hide()
                self.create_vis.setText("Create visualization")
                # CSV sidecars are redundant with the Excel workbook in Analyze.
                self.export_csv.setChecked(False)
                self.export_csv.hide()
        except Exception:
            pass

        reset_btn = QPushButton("Reset to Default")
        reset_btn.setToolTip("Reset all analysis parameters to their default values")
        reset_btn.clicked.connect(self.reset_analysis_defaults)
        analysis_container_layout.addWidget(reset_btn)

        container.setLayout(analysis_container_layout)
        analysis_section_layout.addWidget(container)

        stage_panel = self._build_stage_analysis_subpanel()
        if stage_panel is not None:
            analysis_section_layout.addWidget(stage_panel)

        analysis_section.setLayout(analysis_section_layout)
        self.analysis_section = analysis_section
        return analysis_section

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

    def _restore_analysis_settings_to_stash(self) -> None:
        section = getattr(self, "analysis_section", None)
        stash_layout = getattr(self, "_analysis_settings_stash_layout", None)
        if section is None or stash_layout is None:
            return
        self._move_widget_between_layouts(section, stash_layout)

    # ---- Common behaviors --------------------------------------------
    def _update_analyze_button_state(self) -> None:
        has_path = bool(self.file_path.text() and self.file_path.text().strip())
        if hasattr(self, "analyze_btn"):
            self.analyze_btn.setEnabled(has_path)
        if hasattr(self, "preview_btn"):
            self.preview_btn.setEnabled(has_path)

    def _try_fill_metadata_from_path(self) -> None:
        path_text = (self.file_path.text() or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        tiff_path: Optional[Path] = None
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            tiff_path = path
        elif path.is_dir():
            all_tifs = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
            tiff_path = next((f for f in all_tifs if f.is_file()), None)
        if tiff_path is None:
            return
        try:
            ps, tp = read_calibration_from_path_or_original(tiff_path)
            if ps > 0:
                self.pixel_size.setValue(ps)
                self.pixel_size_indicator.setText("(from file)")
                self.pixel_size_indicator.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                self.pixel_size_indicator.setFixedWidth(100)
                self.pixel_size_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.pixel_size_indicator.show()
                self.pixel_size_metadata_loaded = True
            if tp > 0 and hasattr(self, "parent_window") and getattr(self.parent_window, "shared_metadata", None) is not None:
                self.parent_window.shared_metadata["time_per_frame_min"] = tp
        except Exception:
            pass

    def _on_pixel_size_changed(self) -> None:
        if not self.pixel_size_metadata_loaded:
            self.pixel_size_indicator.setText("")
            self.pixel_size_indicator.setFixedWidth(0)
            self.pixel_size_indicator.hide()

    def reset_analysis_defaults(self) -> None:
        self.pixel_size.setValue(float(0.108))
        self.pixel_size_metadata_loaded = False
        self.pixel_size_indicator.setText("⚠ Check value!")
        self.pixel_size_indicator.setStyleSheet("color: #FFA500; font-size: 9pt; font-weight: bold;")
        self.pixel_size_indicator.setFixedWidth(100)
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
            self._update_analyze_button_state()

    def select_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Mask Files",
            get_default_browse_dir(),
        )
        if folder_path:
            self.file_path.setText(folder_path)
            self._update_analyze_button_state()

    def handle_dropped_path_analyze(self, dropped_path: str) -> None:
        self.file_path.setText(str(dropped_path))
        self._update_analyze_button_state()

    def _is_folder_mode(self) -> bool:
        if not self.file_path.text():
            return False
        p = Path(self.file_path.text())
        return p.exists() and p.is_dir()

    def _get_mask_files_for_feedback(self) -> list[Path]:
        path_text = (self.file_path.text() or "").strip()
        if not path_text:
            return []
        path = Path(path_text)
        if not path.exists():
            return []
        if path.is_file():
            return [path.resolve()]
        folder = path
        mask_set: set[Path] = set()
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
        return out

    def _matches_fi_name(self, name: str, stem: str) -> bool:
        if name == f"{stem}_fi" or name.startswith(f"{stem}_fi_"):
            return True
        if name.startswith(f"{stem}_") and name.endswith("_fi"):
            return True
        return False

    def _matches_foci_name(self, name: str, stem: str) -> bool:
        if name == f"{stem}_foci" or name.startswith(f"{stem}_foci_"):
            return True
        if name.startswith(f"{stem}_fi_") and "_foci" in name:
            return True
        if name.startswith(f"{stem}_") and name.endswith("_foci"):
            return True
        return False

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
                return rem or None
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

    def _resolve_fi_paths_for_mask(self, mask_path: Path) -> list[Path]:
        parent = mask_path.parent
        search_dir = parent.parent if parent.name.startswith("_segmentation") else parent
        stem = mask_path.stem.replace("_masks", "")
        stems = self._candidate_match_stems(stem)

        def search_in(directory: Path) -> list[Path]:
            out: list[Path] = []
            if not directory.exists() or not directory.is_dir():
                return out
            for f in directory.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if any(self._matches_fi_name(f.stem, s) for s in stems):
                    out.append(f)
            return out

        found: list[Path] = search_in(search_dir)
        if not found and search_dir.parent != search_dir:
            found = search_in(search_dir.parent)
        return sorted(set(found), key=lambda p: p.name.lower())

    def _resolve_foci_paths_for_mask(self, mask_path: Path) -> list[Path]:
        parent = mask_path.parent
        search_dir = parent.parent if parent.name.startswith("_segmentation") else parent
        stem = mask_path.stem.replace("_masks", "")
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

        found: list[Path] = search_in(search_dir)
        if not found and search_dir.parent != search_dir:
            found = search_in(search_dir.parent)
        return sorted(set(found), key=lambda p: p.name.lower())

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
                ("exclude_border", self.exclude_border_cb),
            ],
        )
        if "file_path" in data and data.get("file_path"):
            self.file_path.setText(str(data["file_path"]).strip())
        if "performed" in data:
            self._performed_this_session = bool(data["performed"])

