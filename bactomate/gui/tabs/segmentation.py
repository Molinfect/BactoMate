"""
Composed Segmentation tab: choose algorithm (Omnipose / Cellpose / U-Net), run segmentation,
then enable the Analyze Masks workflow below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.module_shell import PreviewSurface
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET
from bactomate.gui.widgets import create_info_icon
from bactomate.gui.preview.preview_window import CollapsibleSection
from bactomate.gui.tabs.analyze_masks import AnalyzeTab, FilteringTab
from bactomate.gui.tabs.omnipose_seg import OmniposeInferenceWidget, OmniposeSegTab
from bactomate.gui.tabs.unet_seg import UNetInferenceWidget, UNetSegmentTab
from bactomate.gui.workspace_scanner import list_tiff_files

_RIGHT_PANEL_TIFF_CAP = 2500

_SEGMENTATION_ALGORITHM_INFO = (
    "Segmentation algorithm for this step:\n\n"
    "• Omnipose — bacterial phase/fluorescence segmentation with pretrained Omnipose models "
    "and optional custom training.\n\n"
    "• Cellpose — Cellpose-SAM for general cell types and imaging modalities.\n\n"
    "• U-Net — patch-based deep learning using your own trained .keras models.\n\n"
    "If a preview is open, switching algorithm reopens it on the same TIFF with the new "
    "backend (Omnipose ↔ Cellpose ↔ U-Net)."
)


class SegmentationTab(QWidget):
    """Top-to-bottom segmentation flow with algorithm selection + gated Analyze Masks section."""

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
        self._module_preview_mount: Optional[QWidget] = None
        self._preview_surface: Optional[PreviewSurface] = None
        self._seg_candidate_timer = QTimer(self)
        self._seg_candidate_timer.setSingleShot(True)
        self._seg_candidate_timer.timeout.connect(self.refresh_segmentation_candidate_list)

        # Separate owner instance dedicated to Cellpose-SAM inference panel
        self._cellpose_owner = OmniposeSegTab(parent, inference_mode="cellpose_sam")

        self._init_ui()
        self._wire_signals()
        self._update_stage_gates()

    def _init_ui(self) -> None:
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setMinimumWidth(0)
        layout = QVBoxLayout(scroll_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        seg_content = QWidget()
        seg_layout = QVBoxLayout(seg_content)
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(6)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(list(self.BACKENDS))
        self.backend_combo.setCurrentIndex(0)
        self._backend_combo_prev_index = 0
        self.backend_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.stack = QStackedWidget()
        self._omni_panel = OmniposeInferenceWidget(
            self.omnipose_tab, parent=self, embedded_in_segmentation=True
        )
        self._cellpose_panel = OmniposeInferenceWidget(
            self._cellpose_owner,
            parent=self,
            cellpose_sam_only=True,
            embedded_in_segmentation=True,
        )
        self._unet_panel = UNetInferenceWidget(
            self.unet_tab, parent=self, embedded_in_segmentation=True
        )
        self.stack.addWidget(self._omni_panel)
        self.stack.addWidget(self._cellpose_panel)
        self.stack.addWidget(self._unet_panel)

        inference_group = QGroupBox("")
        inference_group.setFlat(True)
        inference_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        inference_layout = QVBoxLayout()
        inference_layout.setContentsMargins(0, 0, 0, 0)
        inference_layout.setSpacing(6)

        algorithm_row = QHBoxLayout()
        algorithm_row.addWidget(QLabel("Algorithm:"))
        algorithm_row.addWidget(create_info_icon(_SEGMENTATION_ALGORITHM_INFO, self))
        algorithm_row.addWidget(self.backend_combo, 1)
        inference_layout.addLayout(algorithm_row)
        inference_layout.addWidget(self.stack)
        inference_group.setLayout(inference_layout)
        seg_layout.addWidget(inference_group)

        self.segmentation_section = CollapsibleSection("Segmentation", seg_content, parent=self, expanded=True)
        layout.addWidget(self.segmentation_section, 0)

        filter_content = QWidget()
        filter_layout = QVBoxLayout(filter_content)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(6)
        filter_layout.addWidget(self.filtering_tab)

        self.filtering_section = CollapsibleSection("Filtering", filter_content, parent=self, expanded=False)
        layout.addWidget(self.filtering_section, 0)

        analyze_content = QWidget()
        analyze_layout = QVBoxLayout(analyze_content)
        analyze_layout.setContentsMargins(0, 0, 0, 0)
        analyze_layout.setSpacing(6)
        analyze_layout.addWidget(self.analyze_tab)

        self.analyze_section = CollapsibleSection("Analyze", analyze_content, parent=self, expanded=False)
        layout.addWidget(self.analyze_section, 0)

        layout.addStretch(1)

        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll)
        self.setLayout(outer)

    def attach_module_preview_mount(self, mount: Optional[QWidget]) -> None:
        """Wire module-shell preview (TIFF dropdown + toolbar) to the active segmentation backend."""
        self._module_preview_mount = mount
        self._preview_surface = None
        if mount is not None:
            parent = mount.parent()
            if isinstance(parent, PreviewSurface):
                self._preview_surface = parent
                parent.set_preview_context("segmentation")
                parent.set_pick_callback(self._on_pick_from_preview_list)
                parent.set_toolbar_widget(None)
        self.omnipose_tab.attach_module_preview_mount(mount)
        self._cellpose_owner.attach_module_preview_mount(mount)
        self.unet_tab.attach_module_preview_mount(mount)
        self.filtering_tab.attach_module_preview_mount(mount)
        self.analyze_tab.attach_module_preview_mount(mount)
        if mount is not None and self._preview_surface is not None:
            self._preview_surface.set_preview_context("segmentation")
        self.refresh_segmentation_candidate_list()
        self.refresh_filtering_candidate_list()

    def _segmentation_path_tabs(self) -> tuple:
        return (self.omnipose_tab, self._cellpose_owner, self.unet_tab)

    def _segmentation_owner_for_index(self, idx: int):
        if idx == 1:
            return self._cellpose_owner
        if idx == 2:
            return self.unet_tab
        return self.omnipose_tab

    def _active_segmentation_path_owner(self):
        return self._segmentation_owner_for_index(self.backend_combo.currentIndex())

    def _sync_segmentation_input_path(self, path: str, *, skip: object | None = None) -> None:
        """Keep the same input path across Omnipose / Cellpose / U-Net panels."""
        for tab in self._segmentation_path_tabs():
            if tab is skip:
                continue
            edit = tab.file_path
            if edit.text() == path:
                continue
            edit.blockSignals(True)
            try:
                edit.setText(path)
            finally:
                edit.blockSignals(False)
            try:
                tab._try_push_metadata_from_path()
            except Exception:
                pass
            try:
                tab._update_button_states()
            except Exception:
                pass
            try:
                tab._update_overlay_checkbox_state()
            except Exception:
                pass

    def _on_segmentation_path_edited(self, source_tab: object) -> None:
        path = source_tab.file_path.text()
        self._sync_segmentation_input_path(path, skip=source_tab)

    def _schedule_segmentation_candidate_refresh(self, _text: str = "") -> None:
        self._seg_candidate_timer.start(200)

    def refresh_segmentation_candidate_list(self) -> None:
        surf = self._preview_surface
        if surf is None:
            return
        owner = self._active_segmentation_path_owner()
        embedded_open = (
            getattr(owner, "_embedded_segmentation_preview", None) is not None
            or getattr(owner, "_embedded_unet_preview", None) is not None
        )
        p = (owner.file_path.text() or "").strip()
        if not p:
            surf.set_candidate_files([])
            return
        po = Path(p)

        detect_foci_on = bool(
            getattr(owner, "detect_foci", None) is not None and owner.detect_foci.isChecked()
        )
        foci_sibling_keys: set[str] = set()
        if not detect_foci_on and hasattr(owner, "_find_foci_sources_for_input"):
            anchor = po if po.is_file() else po
            try:
                for fp in owner._find_foci_sources_for_input(anchor):
                    try:
                        foci_sibling_keys.add(str(fp.resolve()))
                    except OSError:
                        foci_sibling_keys.add(str(fp))
            except Exception:
                pass

        def _filter_seg_inputs(paths: list[Path]) -> list[Path]:
            out: list[Path] = []
            for x in paths:
                try:
                    if hasattr(owner, "_is_segmentation_input_file") and not owner._is_segmentation_input_file(x):
                        continue
                except Exception:
                    pass
                if "_masks" in x.name.lower():
                    continue
                if foci_sibling_keys:
                    try:
                        if str(x.resolve()) in foci_sibling_keys:
                            continue
                    except OSError:
                        if str(x) in foci_sibling_keys:
                            continue
                out.append(x)
            return out

        selected: Optional[str] = None
        if embedded_open:
            pd = getattr(owner, "_embedded_segmentation_preview", None) or getattr(
                owner, "_embedded_unet_preview", None
            )
            if pd is not None:
                inp = getattr(pd, "_segmentation_input_path", None)
                if inp:
                    try:
                        selected = str(Path(inp).resolve())
                    except OSError:
                        selected = str(inp)
        if po.is_dir():
            paths = _filter_seg_inputs(
                list_tiff_files(po, max_files=_RIGHT_PANEL_TIFF_CAP)
            )
            paths_str = [str(x) for x in paths]
            pick = getattr(owner, "_folder_preview_pick", None)
            if pick and paths_str:
                try:
                    pr = str(Path(pick).resolve())
                    canon = {str(Path(x).resolve()) for x in paths}
                    if pr in canon:
                        selected = pr
                    else:
                        owner._folder_preview_pick = None
                except OSError:
                    owner._folder_preview_pick = None
            elif not paths_str:
                owner._folder_preview_pick = None
            # While preview is closed, leave the combo on the placeholder until the user picks a TIFF.
            if embedded_open and paths_str and selected is None:
                selected = paths_str[0]
                surf.set_candidate_files(
                    paths_str,
                    selected=selected,
                    advance_placeholder=embedded_open,
                )
                if embedded_open:
                    surf.set_preview_chrome_visible(False)
            return
        if po.is_file():
            parent = po.parent
            if parent.is_dir():
                paths = _filter_seg_inputs(
                    list_tiff_files(parent, max_files=_RIGHT_PANEL_TIFF_CAP)
                )
                paths_str = [str(x) for x in paths]
                try:
                    default_sel = str(po.resolve())
                except OSError:
                    default_sel = str(po)
                # If filters/caps omit the active input, still list it so the shell combo can match it.
                try:
                    canon_resolved = {str(Path(x).resolve()) for x in paths_str}
                except OSError:
                    canon_resolved = set(paths_str)
                if default_sel not in canon_resolved:
                    paths_str = [default_sel] + paths_str
                keep_embedded = selected
                selected = default_sel
                pick = getattr(owner, "_folder_preview_pick", None)
                if pick and paths_str:
                    try:
                        pr = str(Path(pick).resolve())
                        canon = {str(Path(x).resolve()) for x in paths_str}
                        if pr in canon:
                            selected = pr
                        else:
                            owner._folder_preview_pick = None
                    except OSError:
                        owner._folder_preview_pick = None
                if embedded_open and keep_embedded:
                    selected = keep_embedded
                surf.set_candidate_files(
                    paths_str,
                    selected=selected,
                    advance_placeholder=embedded_open,
                )
                if embedded_open:
                    surf.set_preview_chrome_visible(False)
            else:
                surf.set_candidate_files([])
            return
        surf.set_candidate_files([])

    def refresh_filtering_candidate_list(self) -> None:
        """Populate the right-hand file list from the Filtering mask path."""
        surf = self._preview_surface
        if surf is None:
            return
        raw = (self.filtering_tab.file_path.text() or "").strip()
        if not raw:
            surf.set_candidate_files([])
            return
        po = Path(raw)
        if not po.exists():
            surf.set_candidate_files([])
            return
        try:
            if po.is_file():
                paths = [str(po.resolve())]
            else:
                paths = [
                    str(p.resolve())
                    for p in self.filtering_tab._get_mask_files_for_feedback()
                ]
            if paths:
                # Do not auto-pick the first mask in a folder; keep placeholder until the
                # user chooses. Once a mask is active (dropdown pick or open preview), show it.
                sel = surf.current_candidate_path()
                preview_open = getattr(self.filtering_tab, "_filter_preview_dialog", None) is not None
                active_mask = getattr(self.filtering_tab, "_filter_preview_mask_path", None)
                if sel is None and preview_open and active_mask is not None:
                    try:
                        sel = str(Path(active_mask).resolve())
                    except OSError:
                        sel = str(active_mask)
                surf.set_candidate_files(
                    paths,
                    selected=sel,
                    advance_placeholder=False,
                )
            else:
                surf.set_candidate_files([])
        except Exception:
            surf.set_candidate_files([])

    def _on_pick_from_preview_list(self, path: str) -> None:
        p = Path(path)
        if "_masks" in p.name.lower():
            filter_dlg = getattr(self.filtering_tab, "_filter_preview_dialog", None)
            if filter_dlg is not None:
                try:
                    if self._preview_surface is not None:
                        self._preview_surface.set_preview_context("filtering")
                    self.filtering_tab.reload_filter_preview_for_path(p)
                except Exception:
                    pass
                return
            try:
                self.filtering_tab._filter_preview_mask_path = p
            except Exception:
                pass
            return

        owner = self._active_segmentation_path_owner()
        owner._folder_preview_pick = str(path)
        try:
            owner._try_push_metadata_from_path()
        except Exception:
            pass
        try:
            owner._update_button_states()
        except Exception:
            pass
        # Embedded preview was opened for a specific TIFF; changing the shell list must
        # rebuild preview so stacks, overlays, and segmentation match the new pick.
        prev_seg = getattr(owner, "_embedded_segmentation_preview", None)
        prev_unet = getattr(owner, "_embedded_unet_preview", None)
        if prev_seg is None and prev_unet is None:
            return
        try:
            if hasattr(owner, "preview_segmentation"):
                try:
                    owner._preview_switching_input_file = True
                except Exception:
                    pass
                owner.preview_segmentation()
            elif hasattr(owner, "_preview_segmentation"):
                owner._preview_segmentation()
        except Exception:
            pass

    def _dismiss_filtering_embedded_preview(self) -> None:
        prev = getattr(self.filtering_tab, "_filter_preview_dialog", None)
        if prev is not None:
            try:
                prev.close()
                prev.deleteLater()
            except Exception:
                pass
            self.filtering_tab._filter_preview_dialog = None
        surf = self._preview_surface
        if surf is not None:
            try:
                surf.clear_mount()
            except Exception:
                pass

    def _dismiss_fi_contour_embedded_preview(self) -> None:
        at = getattr(self, "analyze_tab", None)
        if at is None:
            return
        dlg = getattr(at, "_fi_contour_preview_dialog", None)
        if dlg is None:
            return
        try:
            dlg.close()
            dlg.deleteLater()
        except Exception:
            pass
        try:
            at._fi_contour_preview_dialog = None
        except Exception:
            pass

    def _dismiss_embedded_previews(self, *, except_owner: Optional[object] = None) -> None:
        """Close embedded previews so algorithm/model switches cannot reuse the wrong backend."""
        self._dismiss_fi_contour_embedded_preview()
        self._dismiss_filtering_embedded_preview()
        for tab in (self.omnipose_tab, self._cellpose_owner, self.unet_tab):
            if tab is except_owner:
                continue
            try:
                tab._restore_seg_tuning_to_left_panel()
            except Exception:
                pass
            prev = getattr(tab, "_embedded_segmentation_preview", None)
            if prev is not None:
                try:
                    prev.close()
                    prev.deleteLater()
                except Exception:
                    pass
                tab._embedded_segmentation_preview = None
            prev_u = getattr(tab, "_embedded_unet_preview", None)
            if prev_u is not None:
                try:
                    prev_u.close()
                    prev_u.deleteLater()
                except Exception:
                    pass
                tab._embedded_unet_preview = None

    def advance_to_filtering_stage(self) -> None:
        """Move the composed workflow from Segmentation to Filtering."""
        self._update_stage_gates()
        self.segmentation_section.set_expanded(False)
        self.filtering_section.set_expanded(True)
        self.analyze_section.set_expanded(False)
        surf = self._preview_surface
        if surf is not None:
            try:
                surf.set_preview_context("filtering")
                self.refresh_filtering_candidate_list()
            except Exception:
                pass

    def advance_to_analyze_stage(self) -> None:
        """Move the composed workflow from Filtering to Analyze."""
        self._update_stage_gates()
        self.segmentation_section.set_expanded(False)
        self.filtering_section.set_expanded(False)
        self.analyze_section.set_expanded(True)
        self._dismiss_filtering_embedded_preview()

    def _wire_signals(self) -> None:
        def _on_backend_changed(idx: int) -> None:
            prev_idx = getattr(self, "_backend_combo_prev_index", 0)
            if idx == prev_idx:
                return
            # currentIndexChanged fires after the combo already shows idx — use prev_idx for the
            # panel that had the open preview (Omnipose / Cellpose / U-Net).
            previous_owner = self._segmentation_owner_for_index(prev_idx)
            self._backend_combo_prev_index = idx
            path = previous_owner.file_path.text()
            preview_input = None
            preview_was_open = False
            prev_seg = getattr(previous_owner, "_embedded_segmentation_preview", None)
            prev_unet = getattr(previous_owner, "_embedded_unet_preview", None)
            if prev_seg is not None:
                preview_was_open = True
                preview_input = getattr(prev_seg, "_segmentation_input_path", None)
            elif prev_unet is not None:
                preview_was_open = True
                preview_input = getattr(prev_unet, "_segmentation_input_path", None)
            self._dismiss_embedded_previews()
            self.stack.setCurrentIndex(idx)
            self._sync_segmentation_input_path(path)
            self.refresh_segmentation_candidate_list()

            if not preview_was_open or not preview_input:
                return
            owner = self._active_segmentation_path_owner()
            backend_label = self.BACKENDS[idx] if 0 <= idx < len(self.BACKENDS) else "segmentation"
            reopen_path = str(preview_input)
            for tab in self._segmentation_path_tabs():
                try:
                    tab._folder_preview_pick = reopen_path
                    tab._preview_reopen_file = reopen_path
                except Exception:
                    pass

            def _reopen_preview_for_backend() -> None:
                try:
                    pw = getattr(self, "parent_window", None)
                    if pw is not None and hasattr(pw, "log_output"):
                        pw.log_output(
                            f"Algorithm switched to {backend_label} — reopening preview on "
                            f"{Path(reopen_path).name} (same file as before).\n"
                        )
                    owner._preview_reopen_file = reopen_path
                    owner._folder_preview_pick = reopen_path
                    owner._preview_switching_input_file = True
                    if hasattr(owner, "preview_segmentation"):
                        owner.preview_segmentation()
                    elif hasattr(owner, "_preview_segmentation"):
                        owner._preview_segmentation()
                except Exception:
                    pass
                finally:
                    if hasattr(owner, "_preview_switching_input_file"):
                        owner._preview_switching_input_file = False
                    if hasattr(owner, "_preview_reopen_file"):
                        try:
                            del owner._preview_reopen_file
                        except Exception:
                            owner._preview_reopen_file = None

            QTimer.singleShot(0, _reopen_preview_for_backend)

        self.backend_combo.currentIndexChanged.connect(_on_backend_changed)

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

        for tab in self._segmentation_path_tabs():
            try:
                tab.file_path.textChanged.connect(self._schedule_segmentation_candidate_refresh)
                tab.file_path.textChanged.connect(
                    lambda _text="", source=tab: self._on_segmentation_path_edited(source)
                )
            except Exception:
                pass
        try:
            self.filtering_tab.file_path.textChanged.connect(self.refresh_filtering_candidate_list)
        except Exception:
            pass

    def _on_backend_finished(self, backend: str, ok: bool) -> None:
        if not ok:
            self._update_stage_gates()
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
        # Always keep stage widgets enabled so user can set paths pre-run.
        self.filtering_tab.setEnabled(True)
        self.analyze_tab.setEnabled(True)
        self.segmentation_section.setEnabled(True)
        self.filtering_section.setEnabled(True)
        self.analyze_section.setEnabled(True)
