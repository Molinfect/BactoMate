"""
Composed Segmentation tab: choose backend (Omnipose / Cellpose / U-Net), run segmentation,
then enable the Analyze Masks workflow below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.tabs.analyze_masks import AnalyzeTab, FilteringTab
from bactimate.gui.tabs.omnipose_seg import OmniposeInferenceWidget, OmniposeSegTab
from bactimate.gui.tabs.unet_seg import UNetInferenceWidget, UNetSegmentTab


class SegmentationTab(QWidget):
    """Top-to-bottom segmentation flow with backend selection + gated Analyze Masks section."""

    BACKENDS = ("Omnipose", "Cellpose", "U-Net")

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        omnipose_tab: OmniposeSegTab,
        unet_tab: UNetSegmentTab,
        filtering_tab: FilteringTab,
        analyze_tab: AnalyzeTab,
    ) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.omnipose_tab = omnipose_tab
        self.unet_tab = unet_tab
        self.filtering_tab = filtering_tab
        self.analyze_tab = analyze_tab
        self._last_segmentation_output_dir: Optional[Path] = None

        # Separate owner instance dedicated to Cellpose-SAM inference panel
        self._cellpose_owner = OmniposeSegTab(parent)

        self._init_ui()
        self._wire_signals()
        self._update_stage_gates()

    def _init_ui(self) -> None:
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        segmentation_group = QGroupBox("1) Segmentation")
        segmentation_layout = QVBoxLayout()
        segmentation_layout.setSpacing(6)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(list(self.BACKENDS))
        self.backend_combo.setCurrentIndex(0)
        segmentation_layout.addWidget(QLabel("Backend:"))
        segmentation_layout.addWidget(self.backend_combo)

        # Backend-specific inference panels
        self.stack = QStackedWidget()
        self._omni_panel = OmniposeInferenceWidget(self.omnipose_tab, parent=self)
        self._cellpose_panel = OmniposeInferenceWidget(
            self._cellpose_owner,
            parent=self,
            cellpose_sam_only=True,
        )
        self._unet_panel = UNetInferenceWidget(self.unet_tab, parent=self)
        self.stack.addWidget(self._omni_panel)
        self.stack.addWidget(self._cellpose_panel)
        self.stack.addWidget(self._unet_panel)
        segmentation_layout.addWidget(self.stack)
        segmentation_group.setLayout(segmentation_layout)
        layout.addWidget(segmentation_group)

        # 2) Filtering section (artifact-driven; user can set path manually)
        self.filtering_group = QGroupBox("2) Filtering")
        filtering_layout = QVBoxLayout()
        filtering_layout.setSpacing(6)
        filtering_layout.setContentsMargins(8, 8, 8, 8)
        self._filtering_hint = QLabel(
            "Run segmentation to populate outputs, or set a mask/segmentation folder path manually below."
        )
        self._filtering_hint.setWordWrap(True)
        self._filtering_hint.setStyleSheet("color: #9e9e9e;")
        filtering_layout.addWidget(self._filtering_hint)
        filtering_layout.addWidget(self.filtering_tab)
        self.filtering_group.setLayout(filtering_layout)
        layout.addWidget(self.filtering_group, 1)

        # 3) Analyze section (artifact-driven; user can set path manually)
        self.analyze_group = QGroupBox("3) Analyze")
        analyze_layout = QVBoxLayout()
        analyze_layout.setSpacing(6)
        analyze_layout.setContentsMargins(8, 8, 8, 8)
        self._analyze_hint = QLabel(
            "Run filtering/segmentation to produce artifacts, or set a path manually below."
        )
        self._analyze_hint.setWordWrap(True)
        self._analyze_hint.setStyleSheet("color: #9e9e9e;")
        analyze_layout.addWidget(self._analyze_hint)
        analyze_layout.addWidget(self.analyze_tab)
        self.analyze_group.setLayout(analyze_layout)
        layout.addWidget(self.analyze_group, 1)

        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll)
        self.setLayout(outer)

    def _wire_signals(self) -> None:
        self.backend_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)

        # Gate enablement: if user sets paths manually, enable their stage.
        try:
            self.filtering_tab.file_path.textChanged.connect(self._update_stage_gates)
            self.analyze_tab.file_path.textChanged.connect(self._update_stage_gates)
        except Exception:
            pass

        # When a backend completes segmentation successfully, populate downstream stage paths.
        self.omnipose_tab.workflow_step_finished.connect(lambda ok: self._on_backend_finished("omnipose", ok))
        self._cellpose_owner.workflow_step_finished.connect(lambda ok: self._on_backend_finished("cellpose", ok))
        self.unet_tab.workflow_step_finished.connect(lambda ok: self._on_backend_finished("unet", ok))

    def _on_backend_finished(self, backend: str, ok: bool) -> None:
        if not ok:
            self._update_analyze_gate()
            return
        out_dir = self._infer_output_dir_for_backend(backend)
        if out_dir is not None:
            self._last_segmentation_output_dir = out_dir
            try:
                self.filtering_tab.file_path.setText(str(out_dir))
                self.analyze_tab.file_path.setText(str(out_dir))
            except Exception:
                pass
        self._update_stage_gates()

    def _infer_output_dir_for_backend(self, backend: str) -> Optional[Path]:
        try:
            if backend in ("omnipose", "cellpose"):
                owner = self.omnipose_tab if backend == "omnipose" else self._cellpose_owner
                raw = (owner.file_path.text() or "").strip()
                if not raw:
                    return None
                p = Path(raw)
                folder_name = owner.get_segmentation_output_folder_name()
                if p.is_dir():
                    return p / folder_name
                if p.is_file():
                    # Best-effort: Omnipose writes next to file; if it lives in a Tif_* folder, output folder is inside that.
                    return p.parent / folder_name
                return None
            if backend == "unet":
                raw = (self.unet_tab.file_path.text() or "").strip()
                if not raw:
                    return None
                p = Path(raw)
                if p.is_dir():
                    return p / "_segmentation_unet"
                if p.is_file():
                    if p.parent.name.startswith("Tif_"):
                        return p.parent / "_segmentation_unet"
                    return p.parent / f"{p.stem}_unet_segmentation"
        except Exception:
            return None
        return None

    def _update_stage_gates(self) -> None:
        # Hints are visible until the stage has a valid path (seg outputs or manual).
        manual_filter = ""
        manual_analyze = ""
        try:
            manual_filter = (self.filtering_tab.file_path.text() or "").strip()
        except Exception:
            manual_filter = ""
        try:
            manual_analyze = (self.analyze_tab.file_path.text() or "").strip()
        except Exception:
            manual_analyze = ""

        seg_ok = bool(self._last_segmentation_output_dir) and self._last_segmentation_output_dir.exists()
        filter_ok = bool(manual_filter) and Path(manual_filter).exists()
        analyze_ok = bool(manual_analyze) and Path(manual_analyze).exists()
        self._filtering_hint.setVisible(not (filter_ok or seg_ok))
        self._analyze_hint.setVisible(not (analyze_ok or seg_ok))

        # Always keep stage widgets enabled so user can set paths pre-run.
        self.filtering_tab.setEnabled(True)
        self.analyze_tab.setEnabled(True)
        self.filtering_group.setEnabled(True)
        self.analyze_group.setEnabled(True)

