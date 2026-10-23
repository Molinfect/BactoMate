"""
Omnipose seg tab: train Omnipose models and run single file or folder segmentation (batch_process_tiff_sequence).
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import threading
import traceback
import csv
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import tifffile
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, QCoreApplication, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QPixmap, QImage, QPainter, QColor, QPen, QShortcut
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QProgressDialog, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget
from bactomate.gui.constants import BATCH_PROCESS_TIFF_SCRIPT, PROJECT_ROOT
from bactomate.gui.log_config import get_logger, append_omnipose_debug
from bactomate.gui.paths import get_omnipose_user_models_dir, get_python_executable_for_subprocess
from bactomate.gui.patches import _patch_omnipose_get_masks
from bactomate.gui.preferences import LAST_DIR_SEGMENTATION
from bactomate.gui.session_params import apply_to_widgets, get_from_widgets
from bactomate.gui.layout_helpers import ResponsivePanel, build_preview_settings_scroll, build_settings_card, flex_control, hbox_row_widget, module_path_action_buttons_layout, module_path_action_outer_layout, module_path_drop_row, new_module_path_action_panel, setting_cell, setting_field, stacked_path_block, style_compact_action_button, style_module_combo, style_module_primary_button, style_path_row_browse_button, style_preview_nav_combo, style_preview_nav_spinbox, vbox_from_rows
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET, MODULE_PANEL_COMBO_STYLESHEET, UI_THEME
from bactomate.gui.mask_utils import OMNIPOSE_MASKS_SUBDIR, OMNIPOSE_MASKS_TILED_SUBDIR, ensure_omnipose_masks_has_images, ensure_training_images_nchan, get_training_pairs, infer_training_filters, prepare_omnipose_masks_from_pairs, tile_training_data
from bactomate.gui.omnipose_utils import eval_uses_affinity_segmentation, format_omnipose_preview_import_error, omnipose_eval_first_mask, inspect_omnipose_checkpoint, is_affinity_builtin_model, load_bact_phase_affinity_model, load_cellpose_segmentation_model, prepare_user_pretrained_loader_path
from bactomate.gui.utils import safe_imread
from bactomate.gui.widgets import ClickableTooltip, ProgressCancelButton, TooltipClickFilter, DragDropGroupBox, PathDropWidget, create_info_icon, show_confirm_dialog, show_file_selection_dialog, show_linked_mask_selection_dialog, show_message_dialog, style_dialog_button_box
from bactomate.gui.workers import WorkerThread
from bactomate.gui.module_shell import detach_mount_widget
from bactomate.gui.workspace_scanner import list_tiff_files
from bactomate.modules.tiff_metadata import read_calibration_from_path_or_original
BUILTIN_OMNIPOSE_MODELS = ['bact_phase_omni', 'bact_fluor_omni', 'bact_phase_affinity']
CELLPOSE_SAM_DISPLAY = 'Cellpose-SAM'
CELLPOSE_SAM_MODEL = 'cellpose_sam'
USER_MODEL_EXTENSION = '.pth'
FOCI_PREVIEW_RING_RGB: tuple[tuple[int, int, int], ...] = ((255, 165, 0), (0, 220, 255), (255, 80, 220), (120, 255, 120), (255, 230, 80))
OMNIPOSE_SEG_PARAM_LABEL_STYLESHEET = 'color: #c8d0dc; font-size: 10pt; font-weight: 500;'

def _omnipose_seg_param_qlabel(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setStyleSheet(OMNIPOSE_SEG_PARAM_LABEL_STYLESHEET)
    lb.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    return lb
PRETRAINED_NCHAN_NCLASSES_DIM = {'bact_phase_omni': (2, 3, 2), 'bact_fluor_omni': (2, 3, 2), 'bact_phase_affinity': (1, 2, 2)}
CANONICAL_BACTERIAL_MODEL = 'bact_phase_omni'
SEGMENTATION_MODEL_DEFAULTS = {'bact_phase_omni': (30.0, 0.4, 0.0), 'bact_fluor_omni': (30.0, 0.4, 0.0), 'bact_phase_affinity': (30.0, 0.4, 0.0), CELLPOSE_SAM_MODEL: (30.0, 0.4, 0.0)}
SEGMENTATION_DEFAULT_FALLBACK = (30.0, 0.4, 0.0)
USER_MODEL_DEFAULTS = (30.0, 0.0, 0.0)
MASK_DISTANCE_OMNI_MIN = -20.0
MASK_DISTANCE_OMNI_MAX = 1.0
MASK_CELLPROB_SAM_MIN = -6.0
MASK_CELLPROB_SAM_MAX = 6.0
SPOTIFLOW_DETECTION_DEFAULTS: dict[str, Any] = {'model_name': 'general', 'model_path': '', 'use_optimized_threshold': True, 'prob_thresh': None, 'min_distance': 1, 'subpix': True, 'normalize': True, 'perc_low': 1.0, 'perc_high': 99.8, 'bg_subtract_radius': 0, 'blur_sigma': 0.0, 'auto_tiling': True, 'n_tiles_y': 1, 'n_tiles_x': 1, 'verbose': False, 'raise_on_error': False}
SPOTIFLOW_CHANNEL_KEYS = ('normalize', 'perc_low', 'perc_high', 'use_optimized_threshold', 'prob_thresh', 'min_distance')

class _OmniposeTrainWorker(QThread):
    """Runs Omnipose training subprocess in background so the GUI stays responsive."""
    log_line = pyqtSignal(str)
    finished_with = pyqtSignal(bool, str, str)

    def __init__(self, cmd, env, cwd, save_dir, dest_file, model_name, nchan, nclasses, keep_save_dir=False):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self.cwd = cwd
        self.save_dir = save_dir
        self.dest_file = dest_file
        self.model_name = model_name
        self.nchan = int(nchan)
        self.nclasses = int(nclasses)
        self.keep_save_dir = keep_save_dir

    def _write_checkpoint_metadata(self):
        """Save in same format as official Omnipose models (plain state_dict only)."""
        try:
            import torch
            ckpt = torch.load(self.dest_file, map_location='cpu', weights_only=False)
            state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(state, dict):
                return
            first_key = next(iter(state.keys()), '')
            if first_key.startswith('module.'):
                state = {k.replace('module.', '', 1): v for (k, v) in state.items()}
            torch.save(state, self.dest_file)
        except Exception:
            pass

    def run(self):
        import shutil
        try:
            proc = subprocess.Popen(self.cmd, env=self.env, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                self.finished_with.emit(False, self.model_name, f'Omnipose exited with code {proc.returncode}.')
                return
            pth_files = list(Path(self.save_dir).glob('*.pth'))
            alt = Path(self.cwd) / 'models'
            if not pth_files and alt.is_dir():
                pth_files = list(alt.glob('*.pth'))
            if not pth_files and alt.is_dir():
                all_files = [p for p in alt.iterdir() if p.is_file()]
                if all_files:
                    newest = max(all_files, key=lambda p: p.stat().st_mtime)
                    pth_files = [newest]
            if not pth_files:
                self.finished_with.emit(False, self.model_name, 'No .pth checkpoint found after training.')
                return
            newest = max(pth_files, key=lambda p: p.stat().st_mtime)
            shutil.copy2(newest, self.dest_file)
            self._write_checkpoint_metadata()
            if not self.keep_save_dir:
                shutil.rmtree(self.save_dir, ignore_errors=True)
            msg = f'Saved to {self.dest_file}'
            if self.keep_save_dir:
                msg += f'. Intermediate checkpoints are in: {self.save_dir}'
            self.finished_with.emit(True, self.model_name, msg)
        except Exception as e:
            self.finished_with.emit(False, self.model_name, str(e))

def _get_pretrained_training_options():
    """From scratch (None), or a built-in model name to initialize from. Same architecture: nchan=2, nclasses=3."""
    return [('From scratch', 'None'), (CANONICAL_BACTERIAL_MODEL, None)] + [(name, name) for name in BUILTIN_OMNIPOSE_MODELS if name in PRETRAINED_NCHAN_NCLASSES_DIM and name != CANONICAL_BACTERIAL_MODEL]

class OmniposeTrainWidget(QWidget):
    """Wrapper widget hosting the training section of an `OmniposeSegTab`.

    The underlying training behavior stays in `OmniposeSegTab`; this wrapper only
    re-parents the already-constructed training group into another tab.
    """

    def __init__(self, owner: 'OmniposeSegTab', parent: Optional[QWidget]=None) -> None:
        super().__init__(parent)
        self._owner = owner
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(owner.take_train_group())
        layout.addStretch(1)
        self.setLayout(layout)

class OmniposeInferenceWidget(QWidget):
    """Wrapper widget hosting the inference section of an `OmniposeSegTab`."""

    def __init__(self, owner: 'OmniposeSegTab', parent: Optional[QWidget]=None, *, cellpose_sam_only: bool=False, embedded_in_segmentation: bool=False) -> None:
        super().__init__(parent)
        self._owner = owner
        self._cellpose_sam_only = bool(cellpose_sam_only)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        seg_group = owner.take_segmentation_group()
        if embedded_in_segmentation and isinstance(seg_group, QGroupBox):
            seg_group.setStyleSheet('QGroupBox { border: none; margin: 0; padding: 0; background: transparent; }')
            owner._seg_params_in_preview = True
            tw = getattr(owner, '_seg_tuning_widget', None)
            if tw is not None:
                tw.hide()
            opts = getattr(owner, '_inference_options_panel', None)
            if opts is not None:
                opts.hide()
        else:
            owner._seg_params_in_preview = False
        layout.addWidget(seg_group)
        self.setLayout(layout)
        try:
            if self._cellpose_sam_only:
                owner.set_inference_mode_cellpose_sam_only()
            else:
                owner.set_inference_mode_omnipose_only()
        except Exception:
            pass

class OmniposeSegTab(QWidget):
    """Unified tab for Omnipose: train models and run single file or folder segmentation."""
    workflow_step_finished = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget]=None, *, inference_mode: str='omnipose') -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._module_preview_mount: Optional[QWidget] = None
        self._preview_surface: Optional[Any] = None
        self._embedded_segmentation_preview: Optional[QDialog] = None
        self._folder_preview_pick: Optional[str] = None
        self._folder_preview_controller: Optional[Any] = None
        self._trained_models_this_session: list[str] = []
        self._performed_this_session = False
        self._segment_filename_filter = ''
        self._inference_mode = 'cellpose_sam' if inference_mode == 'cellpose_sam' else 'omnipose'
        self._seg_params_in_preview = False
        self._spotiflow_detection_params: dict[str, Any] = dict(SPOTIFLOW_DETECTION_DEFAULTS)
        self._spotiflow_channel_params: dict[str, dict[str, Any]] = {}
        self._foci_csv_min_probability: float = 0.0
        self._seg_preview_detect_conn: Optional[Callable[..., None]] = None
        self.init_ui()

    def init_ui(self) -> None:
        self._omnipose_train_worker = None
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout()
        self._train_group_widget = self._build_omnipose_train_group()
        self._scroll_layout.addWidget(self._train_group_widget)
        self._seg_group_widget = self._build_segmentation_params()
        self._scroll_layout.addWidget(self._seg_group_widget)
        self.file_path.textChanged.connect(self._update_overlay_checkbox_state)
        self._update_overlay_checkbox_state()
        self.file_path.textChanged.connect(self._update_button_states)
        self.file_path.textChanged.connect(self._try_push_metadata_from_path)
        self._update_button_states()
        self._scroll_widget.setLayout(self._scroll_layout)
        scroll.setWidget(self._scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def take_train_group(self) -> QWidget:
        """Detach and return the training group widget for embedding elsewhere."""
        w = getattr(self, '_train_group_widget', None)
        if w is None:
            w = self._build_omnipose_train_group()
            self._train_group_widget = w
        try:
            if getattr(self, '_scroll_layout', None) is not None:
                self._scroll_layout.removeWidget(w)
        except Exception:
            pass
        w.setParent(None)
        return w

    def take_segmentation_group(self) -> QWidget:
        """Detach and return the segmentation/inference group widget for embedding elsewhere."""
        w = getattr(self, '_seg_group_widget', None)
        if w is None:
            w = self._build_segmentation_params()
            self._seg_group_widget = w
        try:
            if getattr(self, '_scroll_layout', None) is not None:
                self._scroll_layout.removeWidget(w)
        except Exception:
            pass
        w.setParent(None)
        return w

    def attach_module_preview_mount(self, mount: Optional[QWidget]) -> None:
        """Store right-panel mount for embedded segmentation preview (pick callback is owned by ``SegmentationTab``)."""
        self._module_preview_mount = mount
        self._preview_surface = None
        if mount is not None:
            parent = mount.parent()
            try:
                from bactomate.gui.module_shell import PreviewSurface
                if isinstance(parent, PreviewSurface):
                    self._preview_surface = parent
            except Exception:
                pass

    def _segmentation_shell_owner_tab(self):
        """The composed Segmentation tab that owns PreviewSurface / candidate combo.

        ``QWidget.parent()`` for ``OmniposeSegTab`` is typically MainWindow because inference widgets are reparented
        into ``OmniposeInferenceWidget``; parent-chain walks miss ``SegmentationTab``.
        """
        pw = getattr(self, 'parent_window', None)
        st = getattr(pw, 'segmentation_tab', None) if pw is not None else None
        if st is not None and st.__class__.__name__ == 'SegmentationTab':
            return st
        w = self.parent()
        while w is not None:
            if w.__class__.__name__ == 'SegmentationTab':
                return w
            w = w.parent()
        return None

    def _advance_segmentation_shell_to_filtering(self) -> None:
        seg_tab = self._segmentation_shell_owner_tab()
        if seg_tab is None or not hasattr(seg_tab, 'advance_to_filtering_stage'):
            return
        QTimer.singleShot(0, seg_tab.advance_to_filtering_stage)

    def _restore_seg_tuning_to_left_panel(self) -> None:
        """Put segmentation tuning controls back in the left panel after embedded preview closes."""
        tw = getattr(self, '_seg_tuning_widget', None)
        seg_group = getattr(self, '_seg_group_widget', None)
        if tw is None or seg_group is None:
            return
        try:
            from bactomate.gui.layout_helpers import _qt_widget_alive
            if not _qt_widget_alive(tw):
                return
        except Exception:
            pass
        lay = seg_group.layout()
        if lay is None:
            return
        if tw.parent() is seg_group:
            return
        idx = int(getattr(self, '_seg_tuning_layout_index', 2))
        tw.setParent(seg_group)
        lay.insertWidget(min(idx, lay.count()), tw)
        if getattr(self, '_seg_params_in_preview', False):
            tw.hide()
        else:
            tw.show()

    @staticmethod
    def _link_checkboxes(preview_cb: QCheckBox, main_cb: QCheckBox) -> None:
        """Two-way sync for embedded-preview duplicates (never reparent main widgets)."""
        preview_cb.setChecked(main_cb.isChecked())

        def _pv(checked: bool) -> None:
            main_cb.blockSignals(True)
            try:
                main_cb.setChecked(bool(checked))
            finally:
                main_cb.blockSignals(False)

        def _mv(checked: bool) -> None:
            preview_cb.blockSignals(True)
            try:
                preview_cb.setChecked(bool(checked))
            finally:
                preview_cb.blockSignals(False)
        preview_cb.toggled.connect(_pv)
        main_cb.toggled.connect(_mv)

    @staticmethod
    def _link_spinboxes(preview_spin: QDoubleSpinBox, main_spin: QDoubleSpinBox, *, fixed_width: int=80) -> None:
        """Two-way sync between preview-only and left-panel spinboxes."""
        preview_spin.setRange(main_spin.minimum(), main_spin.maximum())
        preview_spin.setDecimals(main_spin.decimals())
        preview_spin.setSingleStep(main_spin.singleStep())
        preview_spin.setFixedWidth(int(fixed_width))
        preview_spin.blockSignals(True)
        main_spin.blockSignals(True)
        try:
            preview_spin.setValue(main_spin.value())
        finally:
            preview_spin.blockSignals(False)
            main_spin.blockSignals(False)

        def _preview_changed(v: float) -> None:
            main_spin.blockSignals(True)
            try:
                main_spin.setValue(v)
            finally:
                main_spin.blockSignals(False)

        def _main_changed(v: float) -> None:
            preview_spin.blockSignals(True)
            try:
                preview_spin.setValue(v)
            finally:
                preview_spin.blockSignals(False)
        preview_spin.valueChanged.connect(_preview_changed)
        main_spin.valueChanged.connect(_main_changed)

    def _build_embedded_cell_model_params_scroll(self, parent: QWidget, *, preview_dialog: Optional[Any]=None) -> tuple[QWidget, QPushButton]:
        """Omnipose / Cellpose-SAM cell parameters card + Reset button (not embedded in the card).

        Reset/Re-detect live in the nav action bar so they clearly apply to cell + Spotiflow.
        """

        def linked_preview_spin(main_spin: QDoubleSpinBox) -> QDoubleSpinBox:
            pv = QDoubleSpinBox(parent)
            self._link_spinboxes(pv, main_spin, fixed_width=88)
            flex_control(pv, 74, combo_style=False, max_width=96)
            return pv
        pv_diameter = linked_preview_spin(self.diameter)
        pv_flow = linked_preview_spin(self.flow_threshold)
        pv_mask = QDoubleSpinBox(parent)
        self._sync_mask_threshold_spinbox_for_model()
        self._link_spinboxes(pv_mask, self.mask_threshold, fixed_width=88)
        flex_control(pv_mask, 74, combo_style=False, max_width=96)
        pv_cluster = QCheckBox('DBSCAN')
        pv_cluster.setToolTip('Use DBSCAN clustering for thin/elongated cells.')
        self._link_checkboxes(pv_cluster, self.cluster)
        reset_btn = QPushButton('Reset')
        reset_btn.setToolTip(
            'Reset cell segmentation and Spotiflow parameters to module defaults.\n'
            'Does not re-run detection — click Re-detect after Reset to refresh the preview.'
        )

        def _reset_preview_tuning() -> None:
            self.reset_segmentation_defaults()
            pv_diameter.setValue(self.diameter.value())
            pv_flow.setValue(self.flow_threshold.value())
            self._sync_mask_threshold_spinbox_for_model()
            pv_mask.setValue(self.mask_threshold.value())
            pv_cluster.setChecked(self.cluster.isChecked())
            self.use_gpu.setChecked(True)
            self.create_overlay.setChecked(True)
            if preview_dialog is not None:
                try:
                    key = self._active_spotiflow_channel_key(preview_dialog)
                    self._load_spotiflow_widgets_for_channel(preview_dialog, key)
                except Exception:
                    pass
        reset_btn.clicked.connect(_reset_preview_tuning)
        fields = [
            setting_field('Diameter', pv_diameter, min_width=74, max_width=96),
            setting_field('Flow', pv_flow, min_width=74, max_width=96),
            setting_field('Mask', pv_mask, min_width=74, max_width=96),
            setting_cell(pv_cluster),
        ]
        card = build_settings_card(
            'Cell segmentation',
            object_name='EmbeddedCellSegmentationControls',
            parent=parent,
            min_cell_width=150,
            label_min_width=70,
            control_max_width=96,
            h_spacing=12,
            fields=fields,
        )
        return card, reset_btn

    def _build_embedded_preview_action_bar(self, parent: QWidget, *, reset_btn: QPushButton, redetect_btn: QPushButton) -> QWidget:
        """Primary Reset + Re-detect actions for the embedded preview nav (filter-style Apply chrome)."""
        bar = QWidget(parent)
        bar.setObjectName('EmbeddedSegPreviewActions')
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for btn in (reset_btn, redetect_btn):
            style_module_primary_button(btn)
            # Enter in pHigh/pLow must not click Reset (QDialog promotes the first
            # autoDefault button — that was snapping perc_high back to 99.8).
            try:
                btn.setAutoDefault(False)
                btn.setDefault(False)
            except Exception:
                pass
            # Fixed width from text — Minimum/Expanding lets Qt crush "Re-detect" when the nav is crowded.
            fm = btn.fontMetrics()
            text = btn.text() or ''
            text_w = max(int(fm.horizontalAdvance(text)), int(fm.boundingRect(text).width()))
            btn.setFixedWidth(max(108, text_w + 36))
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lay.addWidget(btn, 0)
        bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return bar

    def _build_embedded_spotiflow_foci_strip(self, parent: QWidget, preview_dialog: Optional[Any]=None) -> QWidget:
        """Spotiflow foci enable strip — sits with cell params above the canvas."""
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.Shape.NoFrame)
        frame.setObjectName('SpotiflowFociStrip')
        try:
            frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        except Exception:
            pass
        frame.setStyleSheet('#SpotiflowFociStrip { background-color: #1a1a1a; border: 1px solid #303844; border-radius: 8px;}#SpotiflowFociStrip QLabel { background-color: transparent; border: none;}#SpotiflowFociStrip QCheckBox { spacing: 10px; padding: 6px 10px; color: #dce4ee; font-size: 10pt; background: transparent; border: none; outline: none;}#SpotiflowFociStrip QCheckBox::indicator { width: 22px; height: 22px; border-radius: 4px; border: 1px solid #3a4555; background-color: #2a3140;}#SpotiflowFociStrip QCheckBox::indicator:checked { background-color: #4a9eff; border: 1px solid #6bb3ff;}#SpotiflowFociStrip QCheckBox::indicator:disabled { background-color: #252a32; border: 1px solid #3a4555;}')
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(10, 8, 12, 8)
        lay.setSpacing(12)
        intro = QLabel('Spotiflow foci')
        intro.setStyleSheet('color: #d4dce8; font-size: 11pt; font-weight: 600;')
        sub = QLabel('Runs peak detection on paired foci TIFFs (filenames containing "_foci").')
        sub.setWordWrap(True)
        sub.setStyleSheet('color: #8b96a8; font-size: 10pt; background: transparent;')
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.addWidget(intro)
        text_col.addWidget(sub)
        lay.addLayout(text_col, 1)
        pv_foci = QCheckBox('Enable foci detection')
        pv_foci.setToolTip('When enabled: Spotiflow settings, foci View backgrounds, and peak rings.\nWhen off: phase/input only — no foci overlays or 488_nm-style View entries.\nIndependent of Omnipose vs Cellpose-SAM cell masks.')
        self._link_checkboxes(pv_foci, self.detect_foci)
        try:
            wa = getattr(Qt.WidgetAttribute, 'WA_MacShowFocusRect', None)
            if wa is not None:
                pv_foci.setAttribute(wa, False)
        except Exception:
            pass
        lay.addWidget(pv_foci, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        if preview_dialog is not None:
            try:
                preview_dialog._embedded_detect_foci_cb = pv_foci
            except Exception:
                pass
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return frame

    def _on_embedded_segmentation_preview_closed(self) -> None:
        pd = getattr(self, '_embedded_segmentation_preview', None)
        if pd is not None:
            timer = getattr(pd, '_detect_foci_apply_timer', None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
        fn_e = getattr(self, '_seg_preview_embed_detect_conn', None)
        if pd is not None and fn_e is not None:
            cb = getattr(pd, '_embedded_detect_foci_cb', None)
            if cb is not None:
                try:
                    cb.toggled.disconnect(fn_e)
                except Exception:
                    pass
        self._seg_preview_embed_detect_conn = None
        if pd is not None:
            worker = getattr(pd, '_seg_worker', None)
            if worker is not None:
                slot = getattr(pd, '_seg_worker_slot', None)
                if slot is not None:
                    try:
                        worker.finished.disconnect(slot)
                    except Exception:
                        pass
                try:
                    if worker.isRunning():
                        worker.requestInterruption()
                        worker.quit()
                        if not worker.wait(120000):
                            worker.terminate()
                            worker.wait(5000)
                except Exception:
                    pass
                try:
                    worker.deleteLater()
                except Exception:
                    pass
                try:
                    pd._seg_worker = None
                    pd._seg_worker_slot = None
                except Exception:
                    pass
            foci_worker = getattr(pd, '_foci_worker', None)
            if foci_worker is not None:
                foci_slot = getattr(pd, '_foci_worker_slot', None)
                if foci_slot is not None:
                    try:
                        foci_worker.finished.disconnect(foci_slot)
                    except Exception:
                        pass
                try:
                    if foci_worker.isRunning():
                        foci_worker.requestInterruption()
                        foci_worker.quit()
                        if not foci_worker.wait(120000):
                            foci_worker.terminate()
                            foci_worker.wait(5000)
                except Exception:
                    pass
                try:
                    foci_worker.deleteLater()
                except Exception:
                    pass
                try:
                    pd._foci_worker = None
                    pd._foci_worker_slot = None
                except Exception:
                    pass
        self._embedded_segmentation_preview = None
        fn = getattr(self, '_seg_preview_detect_conn', None)
        if fn is not None:
            try:
                self.detect_foci.toggled.disconnect(fn)
            except Exception:
                pass
            self._seg_preview_detect_conn = None
        self._restore_seg_tuning_to_left_panel()
        surf = getattr(self, '_preview_surface', None)
        if surf is not None:
            try:
                surf.set_preview_chrome_visible(True)
            except Exception:
                pass
        try:
            seg_tab = self._segmentation_shell_owner_tab()
            if seg_tab is not None and hasattr(seg_tab, 'refresh_segmentation_candidate_list'):
                seg_tab.refresh_segmentation_candidate_list()
        except Exception:
            pass

    def _redetect_open_segmentation_preview(self) -> None:
        """Re-run the already-open preview on its current TIFF after model/algorithm changes."""
        pd = getattr(self, '_embedded_segmentation_preview', None)
        if pd is None:
            return
        cb = getattr(pd, '_on_redetect', None)
        if callable(cb):
            try:
                cb()
                return
            except Exception:
                get_logger().exception('Auto re-detect failed')
        try:
            self._preview_switching_input_file = True
            self.preview_segmentation()
        finally:
            self._preview_switching_input_file = False

    def _schedule_open_segmentation_preview_redetect(self) -> None:
        """Debounce automatic preview re-runs from model/default parameter updates."""
        if getattr(self, '_embedded_segmentation_preview', None) is None:
            return
        timer = getattr(self, '_preview_model_redetect_timer', None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._redetect_open_segmentation_preview)
            self._preview_model_redetect_timer = timer
        timer.start(120)

    def _refresh_model_combo_for_inference_mode(self, *, prefer: str | None=None) -> None:
        """Rebuild model dropdown from ``_inference_mode`` (Omnipose vs Cellpose-SAM backend)."""
        if not hasattr(self, 'model_combo'):
            return
        models = self._get_available_models()
        current = prefer or self.model_combo.currentText()
        try:
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            self.model_combo.addItems(models)
            if current in models:
                self.model_combo.setCurrentText(current)
            elif self._inference_mode == 'cellpose_sam':
                self.model_combo.setCurrentText(CELLPOSE_SAM_DISPLAY)
            elif 'bact_phase_omni' in models:
                self.model_combo.setCurrentText('bact_phase_omni')
            elif models:
                self.model_combo.setCurrentIndex(0)
        finally:
            try:
                self.model_combo.blockSignals(False)
            except Exception:
                pass
        try:
            self._sync_mask_threshold_spinbox_for_model()
        except Exception:
            pass

    def set_inference_mode_omnipose_only(self) -> None:
        """Omnipose algorithm panel: built-in Omnipose models and user-trained .pth only."""
        self._inference_mode = 'omnipose'
        self._refresh_model_combo_for_inference_mode()

    def set_inference_mode_cellpose_sam_only(self) -> None:
        """Cellpose algorithm panel: Cellpose-SAM only (no Omnipose models in Model dropdown)."""
        self._inference_mode = 'cellpose_sam'
        self._refresh_model_combo_for_inference_mode(prefer=CELLPOSE_SAM_DISPLAY)

    def _segmentation_status_prefix(self) -> str:
        """Status bar label — matches Segmentation » Algorithm (not the shared OmniposeSegTab class name)."""
        if getattr(self, '_inference_mode', 'omnipose') == 'cellpose_sam':
            return 'Cellpose'
        return 'Omnipose'

    def _build_omnipose_train_group(self) -> QGroupBox:
        """Build 'Train Omnipose model' section (all controls inline, no dialog)."""
        from bactomate.gui.paths import get_omnipose_user_models_dir
        train_group = QGroupBox('')
        train_group.setFlat(True)
        train_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        layout = QVBoxLayout()
        last_dir = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_SEGMENTATION) or str(Path.home())

        def handle_dropped_training_path(dropped_path):
            path_obj = Path(dropped_path)
            if path_obj.is_dir():
                self.omnipose_train_dir_edit.setText(dropped_path)
            elif path_obj.is_file():
                self.omnipose_train_dir_edit.setText(str(path_obj.parent))
        dir_group = DragDropGroupBox('', self, on_drop_callback=handle_dropped_training_path, base_stylesheet=FLAT_GROUPBOX_FRAME_STYLESHEET)
        dir_group.setFlat(True)
        dir_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel('Path:'))
        dir_layout.addWidget(create_info_icon("Folder containing your training data: one image file and one mask file per pair.\n\nOmnipose expects: for each image 'NAME.tif', a mask file 'NAME<Mask filter>.tif' (e.g. NAME_masks.tif). Same base name, mask = base + Mask filter + extension.\n\nMask format: must be instance labels (background = 0; each cell = a unique integer). When you click Start training, an 'omnipose_masks' folder is created automatically and used for training.\n\nTip: For large or sparse images, tile to ~512×512 and drop mostly-empty tiles before training; set Crop size (tyx) to ~half tile size.", self))
        self.omnipose_train_dir_edit = QLineEdit()
        self.omnipose_train_dir_edit.setPlaceholderText('Absolute path to folder with image/mask pairs (or drag and drop)')
        self.omnipose_train_dir_edit.setAcceptDrops(False)
        dir_layout.addWidget(self.omnipose_train_dir_edit)

        def browse_train_dir():
            path = QFileDialog.getExistingDirectory(self, 'Select training directory', last_dir)
            if path:
                self.omnipose_train_dir_edit.setText(path)
                if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                    self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, path)
        browse_btn = QPushButton('Folder…')
        browse_btn.setAcceptDrops(False)
        browse_btn.setToolTip('')
        style_path_row_browse_button(browse_btn)
        browse_btn.clicked.connect(browse_train_dir)
        dir_layout.addWidget(browse_btn)
        dir_group.setLayout(dir_layout)
        self._omnipose_train_dir_group_widget = dir_group
        layout.addWidget(dir_group)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel('Image filter:'))
        self.omnipose_img_filter_edit = QLineEdit()
        self.omnipose_img_filter_edit.setPlaceholderText('empty or _img')
        self.omnipose_img_filter_edit.setMinimumWidth(100)
        filter_layout.addWidget(self.omnipose_img_filter_edit)
        filter_layout.addWidget(QLabel('Mask filter:'))
        self.omnipose_mask_filter_edit = QLineEdit()
        self.omnipose_mask_filter_edit.setText('_mask')
        self.omnipose_mask_filter_edit.setMinimumWidth(100)
        filter_layout.addWidget(self.omnipose_mask_filter_edit)
        auto_detect_btn = QPushButton('Auto-detect filters')
        auto_detect_btn.setToolTip('')
        style_compact_action_button(auto_detect_btn)
        auto_detect_btn.clicked.connect(self._on_omnipose_auto_detect_filters)
        filter_layout.addWidget(auto_detect_btn)
        filter_layout.addWidget(create_info_icon("Suffix that identifies image vs mask files.\n\nImage filter: leave empty if filenames are e.g. A.tif; use _img if you have A_img.tif.\nMask filter: suffix on mask files. With _masks, mask for 'cell1.tif' must be 'cell1_masks.tif'. Must match your filenames exactly.", self))
        filter_widget = hbox_row_widget(filter_layout, self)
        self._omnipose_train_filter_widget = filter_widget
        layout.addWidget(filter_widget)
        prep_row = QHBoxLayout()
        prep_row.setSpacing(6)
        make_masks_btn = QPushButton('Make masks instance labeled')
        make_masks_btn.setToolTip('')
        style_compact_action_button(make_masks_btn)
        make_masks_btn.clicked.connect(self._on_omnipose_make_masks_instance_labeled)
        prep_row.addWidget(make_masks_btn, 0)
        make_masks_info = create_info_icon("Omnipose needs instance-label masks (background = 0; each cell = a unique integer 1, 2, 3, …). When you click Start training, an 'omnipose_masks' folder is created automatically and used for training; no separate conversion step is needed.", self, compact=True)
        make_masks_info.setFixedSize(14, 14)
        prep_row.addWidget(make_masks_info, 0)
        prep_row.addSpacing(10)
        tile_btn = QPushButton('Tile and prepare for training')
        tile_btn.setToolTip('')
        style_compact_action_button(tile_btn)
        tile_btn.clicked.connect(self._on_omnipose_tile_and_prepare)
        prep_row.addWidget(tile_btn, 0)
        tile_info = create_info_icon('Split images into smaller tiles and drop mostly-empty tiles. Use when images are large or sparse to avoid the model collapsing to 0 cells. Output goes to a separate folder; set training directory to it and Crop size (tyx) to half the tile size. Each run clears the output folder first.', self, compact=True)
        tile_info.setFixedSize(14, 14)
        prep_row.addWidget(tile_info, 0)
        prep_row.addStretch(1)
        layout.addLayout(prep_row)
        self.omnipose_use_tiled_cb = QCheckBox('Use tiled/cropped images for training')
        self.omnipose_use_tiled_cb.setChecked(True)
        self.omnipose_use_tiled_cb.setToolTip('')
        tile_crop_row = QHBoxLayout()
        tile_crop_row.addWidget(self.omnipose_use_tiled_cb)
        tile_crop_row.addWidget(create_info_icon('Uncheck to train on full-size images instead of tiled crops. Use when your images are already small enough or you prefer no tiling.', self))
        tile_crop_row.addSpacing(16)
        self.omnipose_use_gpu_cb = QCheckBox('Use GPU')
        self.omnipose_use_gpu_cb.setChecked(True)
        self.omnipose_use_gpu_cb.setToolTip('')
        tile_crop_row.addWidget(self.omnipose_use_gpu_cb)
        tile_crop_row.addStretch()
        layout.addLayout(tile_crop_row)
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel('Model name:'))
        user_models_path = get_omnipose_user_models_dir()
        name_layout.addWidget(create_info_icon(f"A short name for your model (letters, numbers, underscores). It will appear in the model dropdown and is used to save the .pth file. Example: my_bacteria_v1. BactoMate automatically adds '_omni' to the filename if your name does not already contain 'omni'. User models folder:\n{user_models_path}", self))
        self.omnipose_model_name_edit = QLineEdit()
        self.omnipose_model_name_edit.setPlaceholderText('e.g. my_bacteria_v1')
        self.omnipose_model_name_edit.setMinimumWidth(220)
        name_layout.addWidget(self.omnipose_model_name_edit, 1)
        layout.addLayout(name_layout)
        start_from_layout = QHBoxLayout()
        start_from_layout.addWidget(QLabel('Start from:'))
        self.omnipose_pretrained_combo = QComboBox()
        self.omnipose_pretrained_combo.setMinimumWidth(260)
        style_module_combo(self.omnipose_pretrained_combo)
        for (display_name, _) in _get_pretrained_training_options():
            self.omnipose_pretrained_combo.addItem(display_name)
        if self.omnipose_pretrained_combo.count() >= 2:
            self.omnipose_pretrained_combo.setCurrentIndex(1)
        start_from_layout.addWidget(self.omnipose_pretrained_combo, 1)
        start_from_layout.addWidget(create_info_icon('Choose which checkpoint to start training from. From scratch: no pretrained model. bact_phase_omni / bact_fluor_omni / bact_phase_affinity: new model file initialized from that built-in; training runs on your data.', self))
        check_compat_btn = QPushButton('Check compatibility')
        check_compat_btn.setToolTip('')
        style_compact_action_button(check_compat_btn)
        check_compat_btn.clicked.connect(self._on_omnipose_check_compatibility)
        start_from_layout.addWidget(check_compat_btn)
        start_from_layout.addWidget(create_info_icon('Validates image/mask pairs in your training folder and that image dimensions match the selected model (e.g. 2D for bact_phase_omni).', self))
        layout.addLayout(start_from_layout)
        layout.addWidget(QLabel('Training parameters:'))
        params_row1 = QHBoxLayout()
        params_row1.addWidget(QLabel('Epochs:'))
        params_row1.addWidget(create_info_icon('Number of full passes over the training set. From scratch: Omnipose paper uses 4000. Fine-tuning: start with 50 epochs (default).', self))
        self.omnipose_n_epochs_spin = QSpinBox()
        self.omnipose_n_epochs_spin.setRange(1, 50000)
        self.omnipose_n_epochs_spin.setSingleStep(100)
        self.omnipose_n_epochs_spin.setValue(4000)
        flex_control(self.omnipose_n_epochs_spin, 72, combo_style=False)
        params_row1.addWidget(self.omnipose_n_epochs_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel('Save N epochs:'))
        params_row1.addWidget(create_info_icon('Save a checkpoint every N epochs (0 = only final model). Use e.g. 50 to compare epochs and pick the best.', self))
        self.omnipose_save_each_spin = QSpinBox()
        self.omnipose_save_each_spin.setRange(0, 1000)
        self.omnipose_save_each_spin.setValue(0)
        self.omnipose_save_each_spin.setSpecialValueText('final only')
        flex_control(self.omnipose_save_each_spin, 72, combo_style=False)
        params_row1.addWidget(self.omnipose_save_each_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel('LR:'))
        params_row1.addWidget(create_info_icon('Learning rate: Step size for the optimizer. From scratch: 0.1. Fine-tuning: use 0.01 or 0.005 so the pretrained model does not drift.', self))
        self.omnipose_lr_spin = QDoubleSpinBox()
        self.omnipose_lr_spin.setRange(0.001, 1.0)
        self.omnipose_lr_spin.setValue(0.1)
        self.omnipose_lr_spin.setDecimals(3)
        flex_control(self.omnipose_lr_spin, 72, combo_style=False)
        params_row1.addWidget(self.omnipose_lr_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel('Batch:'))
        params_row1.addWidget(create_info_icon('Number of image crops per training step. Larger (e.g. 16) trains faster if you have enough GPU memory; smaller (e.g. 4–8) uses less memory.', self))
        self.omnipose_batch_spin = QSpinBox()
        self.omnipose_batch_spin.setRange(1, 128)
        self.omnipose_batch_spin.setValue(16)
        flex_control(self.omnipose_batch_spin, 72, combo_style=False)
        params_row1.addWidget(self.omnipose_batch_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel('Crop size:'))
        params_row1.addWidget(create_info_icon('Training crop size (Y, X) for random crops each epoch. Default 256×256. Must be divisible by 8. Use ~half your image/tile size.', self))
        params_row1.addWidget(QLabel('Y:'))
        self.omnipose_tyx_y_spin = QSpinBox()
        self.omnipose_tyx_y_spin.setRange(64, 512)
        self.omnipose_tyx_y_spin.setSingleStep(8)
        self.omnipose_tyx_y_spin.setValue(256)
        flex_control(self.omnipose_tyx_y_spin, 64, combo_style=False)
        params_row1.addWidget(self.omnipose_tyx_y_spin)
        params_row1.addWidget(QLabel('X:'))
        self.omnipose_tyx_x_spin = QSpinBox()
        self.omnipose_tyx_x_spin.setRange(64, 512)
        self.omnipose_tyx_x_spin.setSingleStep(8)
        self.omnipose_tyx_x_spin.setValue(256)
        flex_control(self.omnipose_tyx_x_spin, 64, combo_style=False)
        params_row1.addWidget(self.omnipose_tyx_x_spin)
        params_row1.addStretch()
        layout.addLayout(params_row1)
        self.omnipose_nchan_spin = QSpinBox(self)
        self.omnipose_nchan_spin.setRange(1, 8)
        self.omnipose_nchan_spin.setValue(2)
        self.omnipose_nchan_spin.setVisible(False)
        self.omnipose_nclasses_spin = QSpinBox(self)
        self.omnipose_nclasses_spin.setRange(2, 4)
        self.omnipose_nclasses_spin.setValue(3)
        self.omnipose_nclasses_spin.setVisible(False)
        _locked_style = 'QSpinBox:disabled { background-color: palette(mid); color: palette(placeholder-text); }'
        _normal_style = ''

        def update_pretrained_lock():
            idx = self.omnipose_pretrained_combo.currentIndex()
            options = _get_pretrained_training_options()
            cli_value = options[idx][1] if 0 <= idx < len(options) else None
            model_for_arch = CANONICAL_BACTERIAL_MODEL if cli_value is None or cli_value == 'None' else cli_value
            if model_for_arch in PRETRAINED_NCHAN_NCLASSES_DIM:
                (nchan, nclass, _) = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_arch]
                self.omnipose_nchan_spin.setValue(nchan)
                self.omnipose_nclasses_spin.setValue(nclass)
                self.omnipose_nchan_spin.setEnabled(False)
                self.omnipose_nclasses_spin.setEnabled(False)
                self.omnipose_nchan_spin.setStyleSheet(_locked_style)
                self.omnipose_nclasses_spin.setStyleSheet(_locked_style)
            else:
                self.omnipose_nchan_spin.setEnabled(True)
                self.omnipose_nclasses_spin.setEnabled(True)
                self.omnipose_nchan_spin.setStyleSheet(_normal_style)
                self.omnipose_nclasses_spin.setStyleSheet(_normal_style)
            if idx == 0:
                self.omnipose_n_epochs_spin.setValue(4000)
                self.omnipose_lr_spin.setValue(0.1)
            else:
                self.omnipose_n_epochs_spin.setValue(50)
                self.omnipose_lr_spin.setValue(0.01)
        self.omnipose_pretrained_combo.currentIndexChanged.connect(update_pretrained_lock)
        update_pretrained_lock()
        layout.addWidget(QLabel('Training log:'))
        self.omnipose_training_log = QTextEdit()
        self.omnipose_training_log.setReadOnly(True)
        self.omnipose_training_log.setMaximumHeight(180)
        self.omnipose_training_log.setFont(QFont('Menlo', 10) if sys.platform == 'darwin' else QFont('Consolas', 9))
        layout.addWidget(self.omnipose_training_log)
        self.omnipose_start_training_btn = QPushButton('Start training')
        style_module_primary_button(self.omnipose_start_training_btn)
        self.omnipose_start_training_btn.clicked.connect(self._start_omnipose_training)
        layout.addWidget(self.omnipose_start_training_btn)
        train_group.setLayout(layout)
        return train_group

    def _training_pairs_for_train(self, train_path: Path, img_filter: str, mask_filter: str):
        """Return (n_pairs, first_paired_image_path, image_candidates) for compatibility check and start_training."""
        exts = ('.tif', '.tiff', '.png', '.npy', '.npz')
        mask_suffixes = ('_cp_masks', '_cp_output', '_flows', mask_filter)

        def base_name(p, suf):
            return p.stem.replace(suf, '') if suf else p.stem
        candidates = [f for f in train_path.iterdir() if f.is_file() and f.suffix.lower() in exts and (not any((f.stem.endswith(s) or f.stem + s == f.stem for s in mask_suffixes)))]
        (n, first) = (0, None)
        for f in candidates:
            base_val = base_name(f, img_filter)
            if not base_val or base_val == '.':
                continue
            if any(((train_path / f'{base_val}{mask_filter}{e}').exists() for e in exts)):
                n += 1
                if first is None:
                    first = f
        return (n, first, candidates)

    def _format_omnipose_training_log_line(self, raw: str) -> str:
        """Strip ANSI codes and reformat Omnipose 'Train epoch' lines for readability."""
        stripped = re.sub('\\x1b\\[[0-9;]*[a-zA-Z]', '', raw)
        stripped = re.sub('\\033\\[[0-9;]*[a-zA-Z]', '', stripped)
        stripped = re.sub('\\[0m|\\[38;5;\\d+m', '', stripped)
        stripped = stripped.strip()
        if not stripped:
            return ''
        if '\t' in stripped:
            parts = stripped.split('\t')
            last = parts[-1].strip() if parts else ''
            if last and ('/cellpose_' in last or 'cellpose_residual' in last) and ('epoch_' in last or re.search('_\\d{4}_\\d{2}_\\d{2}', last)):
                return ''
        if 'Train epoch:' in stripped and '\t' in stripped:
            parts = stripped.split('\t')
            msg = stripped
            for i in range(len(parts) - 1, -1, -1):
                if 'Train epoch:' in parts[i]:
                    msg = parts[i].strip()
                    break
        else:
            msg = stripped
        if 'Train epoch:' in msg and ' | ' in msg:
            segments = [s.strip() for s in msg.split(' | ')]
            segments = [s for s in segments if 'sec/batch' not in s.lower() and 'last batch loss' not in s.lower()]
            if not segments:
                return ''

            def _shorten(s):
                m = re.search('(Loss>:?\\s*)([\\d.]+)', s)
                if m:
                    try:
                        val = float(m.group(2))
                        return s[:m.start(2)] + f'{val:.2f}'
                    except ValueError:
                        pass
                return s
            segments = [_shorten(s) for s in segments]
            return '  ' + ' | '.join(segments)
        return msg

    def _on_omnipose_auto_detect_filters(self):
        folder_str = self.omnipose_train_dir_edit.text().strip()
        if not folder_str:
            show_message_dialog(self, 'No folder', 'Enter a training directory first.', kind='warning')
            return
        folder = Path(folder_str)
        if not folder.is_dir():
            show_message_dialog(self, 'Invalid folder', 'Training directory does not exist or is not a folder.', kind='warning')
            return
        (img_filter, mask_filter) = infer_training_filters(folder)
        if img_filter is None and mask_filter is None:
            show_message_dialog(self, 'Auto-detect filters', 'Could not detect image/mask pairs. Ensure the folder contains paired files (e.g. 634_phase.tif and 634_masks.tif) and try again, or set the filters manually.', kind='info')
            return
        self.omnipose_img_filter_edit.setText(img_filter or '')
        self.omnipose_mask_filter_edit.setText(mask_filter or '_mask')
        pairs = get_training_pairs(folder, img_filter or '', mask_filter or '_mask')
        show_message_dialog(self, 'Auto-detect filters', f'Detected filters. Found {len(pairs)} image/mask pair(s).', kind='info')

    def _on_omnipose_make_masks_instance_labeled(self):
        """Create omnipose_masks folder with instance-labeled masks from current training dir and filters."""
        train_dir = self.omnipose_train_dir_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            show_message_dialog(self, 'No folder', 'Set a valid training directory first.', kind='warning')
            return
        img_filter = self.omnipose_img_filter_edit.text().strip()
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or '_mask'
        parent_dir = Path(train_dir).resolve()
        try:
            (train_path, n_from_pairs) = prepare_omnipose_masks_from_pairs(parent_dir, img_filter, mask_filter)
        except Exception as e:
            show_message_dialog(self, 'Convert masks', f'Failed to prepare instance-labeled masks:\n{e}', kind='warning')
            return
        if n_from_pairs == 0:
            show_message_dialog(self, 'No pairs', 'No image/mask pairs found. Set Image and Mask filters so each image has a matching mask, or run Auto-detect filters.', kind='warning')
            return
        show_message_dialog(self, 'Masks prepared', f'Created instance-labeled masks for {n_from_pairs} pair(s) in:\n{train_path}\n\nStart training will use this folder automatically, or you can set Training directory to the path above.', kind='info')

    def _on_omnipose_tile_and_prepare(self):
        train_dir = self.omnipose_train_dir_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            show_message_dialog(self, 'No folder', 'Set a valid training directory first.', kind='warning')
            return
        img_filter = self.omnipose_img_filter_edit.text().strip()
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or '_mask'
        pairs = get_training_pairs(Path(train_dir), img_filter, mask_filter)
        if not pairs:
            show_message_dialog(self, 'No pairs', 'No image/mask pairs found. Set Image and Mask filters so each image has a matching mask.', kind='warning')
            return
        tile_dlg = QDialog(self)
        tile_dlg.setWindowTitle('Tile and prepare for training')
        tile_layout = QVBoxLayout(tile_dlg)
        tile_layout.addWidget(QLabel('Tile size (Y and X). Must be divisible by 8. Use 512 or 256.'))
        tile_size_spin = QSpinBox()
        tile_size_spin.setRange(64, 512)
        tile_size_spin.setSingleStep(8)
        tile_size_spin.setValue(512)
        tile_layout.addWidget(tile_size_spin)
        tile_layout.addWidget(QLabel('Minimum cells per tile (tiles with fewer cells are skipped):'))
        min_cells_spin = QSpinBox()
        min_cells_spin.setRange(0, 50)
        min_cells_spin.setValue(1)
        tile_layout.addWidget(min_cells_spin)
        tile_layout.addWidget(QLabel('Output folder (default: training_dir/omnipose_masks_tiled):'))
        out_edit = QLineEdit()
        out_edit.setText(str(Path(train_dir).resolve() / OMNIPOSE_MASKS_TILED_SUBDIR))
        out_edit.setMinimumWidth(400)
        tile_layout.addWidget(out_edit)
        bb_tile = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        style_dialog_button_box(bb_tile)
        bb_tile.accepted.connect(tile_dlg.accept)
        bb_tile.rejected.connect(tile_dlg.reject)
        tile_layout.addWidget(bb_tile)
        if tile_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        tile_size = max(8, tile_size_spin.value() // 8 * 8)
        min_cells = min_cells_spin.value()
        out_path = Path(out_edit.text().strip())
        if not out_path:
            out_path = Path(train_dir).resolve() / OMNIPOSE_MASKS_TILED_SUBDIR
        try:
            (n_tiles, n_imgs, errs) = tile_training_data(Path(train_dir).resolve(), img_filter, mask_filter, tile_size, min_cells, out_path)
        except Exception as e:
            show_message_dialog(self, 'Tiling failed', str(e), kind='warning')
            return
        if errs:
            msg = f'Written {n_tiles} tiles from {n_imgs} image(s) to:\n{out_path}\n\nWarnings:\n' + '\n'.join(errs[:8])
            if len(errs) > 8:
                msg += f'\n... and {len(errs) - 8} more.'
        else:
            msg = f'Written {n_tiles} tiles from {n_imgs} image(s) to:\n{out_path}'
        if n_tiles > 0:
            tyx_half = max(8, tile_size // 2 // 8 * 8)
            self.omnipose_tyx_y_spin.setValue(tyx_half)
            self.omnipose_tyx_x_spin.setValue(tyx_half)
            msg += f'\n\nCrop size (tyx) has been set to {tyx_half}×{tyx_half} (half the tile size).'
        msg += '\n\nNext: set Training directory to the folder above if not already, and keep the same Image/Mask filters.'
        show_message_dialog(self, 'Tiling done', msg, kind='info')
        if n_tiles > 0 and show_confirm_dialog(self, title='Set training directory?', message='Set Training directory to the tiled folder now?', confirm_label='Yes', cancel_label='No', default_confirm=True):
            self.omnipose_train_dir_edit.setText(str(out_path))

    def _on_omnipose_check_compatibility(self):
        train_dir = self.omnipose_train_dir_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            show_message_dialog(self, 'Invalid directory', 'Please choose an absolute path to a training directory.', kind='warning')
            return
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or '_mask'
        img_filter = self.omnipose_img_filter_edit.text().strip()
        train_path = ensure_omnipose_masks_has_images(Path(train_dir).resolve(), mask_filter, img_filter=img_filter)
        (n_pairs, first_img, _) = self._training_pairs_for_train(train_path, img_filter, mask_filter)
        if n_pairs == 0:
            show_message_dialog(self, 'Compatibility', 'No image/mask pairs found. Set Image and Mask filters so each image has a matching mask file.', kind='warning')
            return
        idx = self.omnipose_pretrained_combo.currentIndex()
        options = _get_pretrained_training_options()
        cli_value = options[idx][1] if 0 <= idx < len(options) else None
        model_for_arch = cli_value or CANONICAL_BACTERIAL_MODEL
        detail_parts = []
        if model_for_arch in PRETRAINED_NCHAN_NCLASSES_DIM:
            (_, _, dim) = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_arch]
            if dim == 2 and first_img:
                try:
                    arr = safe_imread(str(first_img))
                except Exception:
                    arr = None
                if arr is not None and arr.ndim == 3 and (arr.shape[0] > 4):
                    show_message_dialog(self, 'Compatibility', f'Found {n_pairs} pair(s), but the first image has shape {tuple(arr.shape)} (likely a 3D stack). The selected model is 2D. Use 2D slices or a 3D model (e.g. plant_omni).', kind='warning')
                    return
                if arr is not None:
                    detail_parts.append(f'Checked first image dimensions: shape {tuple(arr.shape)} — 2D, compatible with the selected 2D model ({model_for_arch}).')
                else:
                    detail_parts.append('Could not load first image to verify dimensions; pair count and filters are valid.')
            elif dim == 2:
                detail_parts.append('Pair count and filters valid. No image could be sampled for dimension check.')
            else:
                detail_parts.append(f'Pair count and filters valid. Selected model is 3D ({model_for_arch}); dimension check skipped.')
        else:
            detail_parts.append("Filters and pair naming are valid. Select a pretrained model in 'Start from' to also verify image dimensions (2D vs 3D).")
        msg = f'Found {n_pairs} image/mask pair(s).\n\n' + '\n\n'.join(detail_parts)
        show_message_dialog(self, 'Compatibility', msg, kind='info')

    def _start_omnipose_training(self):
        """Run Omnipose training from tab widgets (no dialog)."""
        train_dir = self.omnipose_train_dir_edit.text().strip()
        model_name = self.omnipose_model_name_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            show_message_dialog(self, 'Invalid directory', 'Please choose an absolute path to a training directory.', kind='warning')
            return
        if not model_name:
            show_message_dialog(self, 'Invalid name', 'Please enter a model name.', kind='warning')
            return
        if '/' in model_name or '\\' in model_name:
            show_message_dialog(self, 'Invalid name', 'Model name must not contain path separators.', kind='warning')
            return
        final_stem = model_name if 'omni' in model_name.lower() else f'{model_name}_omni'
        dest_file = get_omnipose_user_models_dir() / f'{final_stem}{USER_MODEL_EXTENSION}'
        if dest_file.exists():
            if not show_confirm_dialog(self, title='Model already exists', message=f"A model named '{final_stem}' already exists. Overwrite it? This cannot be undone.", confirm_label='Yes', cancel_label='No', default_confirm=False):
                return
        train_dir_abs = Path(train_dir).resolve()
        img_filter = self.omnipose_img_filter_edit.text().strip()
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or '_mask'
        use_tiled = self.omnipose_use_tiled_cb.isChecked()
        if train_dir_abs.name == OMNIPOSE_MASKS_TILED_SUBDIR and use_tiled:
            train_path = train_dir_abs
        else:
            parent_dir = train_dir_abs.parent if train_dir_abs.name == OMNIPOSE_MASKS_SUBDIR or train_dir_abs.name == OMNIPOSE_MASKS_TILED_SUBDIR else train_dir_abs
            (train_path, n_from_pairs) = prepare_omnipose_masks_from_pairs(parent_dir, img_filter, mask_filter)
            if n_from_pairs == 0:
                train_path = ensure_omnipose_masks_has_images(parent_dir, mask_filter, img_filter=img_filter)
        train_dir_abs = str(train_path)
        (pairs_ok, first_img_path, image_candidates) = self._training_pairs_for_train(train_path, img_filter, mask_filter)

        def base_name(p, suf):
            return p.stem.replace(suf, '') if suf else p.stem
        bad_base = [f.name for f in image_candidates if not base_name(f, img_filter) or base_name(f, img_filter) == '.']
        if pairs_ok == 0 and (not img_filter) and image_candidates:
            mask_files = [f for f in train_path.iterdir() if f.is_file() and f.stem.endswith(mask_filter)]
            for img_f in image_candidates:
                stem = img_f.stem
                for mf in mask_files:
                    base = mf.stem[:-len(mask_filter)] if mask_filter else mf.stem
                    if base and stem.startswith(base) and (stem != base):
                        inferred = stem[len(base):]
                        if self._training_pairs_for_train(train_path, inferred, mask_filter)[0] > 0:
                            img_filter = inferred
                            self.omnipose_img_filter_edit.setText(inferred)
                            (pairs_ok, first_img_path, image_candidates) = self._training_pairs_for_train(train_path, img_filter, mask_filter)
                            bad_base = [f.name for f in image_candidates if not base_name(f, img_filter) or base_name(f, img_filter) == '.']
                            break
                if pairs_ok > 0:
                    break
        if bad_base:
            show_message_dialog(self, 'Invalid training files', f"Some image filenames give an empty base name: {bad_base[:5]}{('…' if len(bad_base) > 5 else '')}. Use names like 'image.tif' with a matching 'image_masks.tif' (no leading dot).", kind='warning')
            return
        if pairs_ok == 0:
            folder_used = getattr(train_path, 'name', str(train_path))
            if folder_used == OMNIPOSE_MASKS_SUBDIR or str(train_path).endswith(os.path.sep + OMNIPOSE_MASKS_SUBDIR):
                hint = f"The folder '{train_path}' contains only mask files (no images). Images are copied from the parent folder when they match the mask names. Check that the parent folder has image files with matching base names and that Image filter matches your image filenames."
            else:
                hint = f"For each image we need a mask: base name + '{mask_filter}' + extension. Image filter is '{img_filter or '(empty)'}' — the base is taken from the image by removing this suffix."
            show_message_dialog(self, 'No image/mask pairs found', f'No matching mask files found. {hint}', kind='warning')
            return
        pretrained_idx = self.omnipose_pretrained_combo.currentIndex()
        options = _get_pretrained_training_options()
        pretrained_cli = options[pretrained_idx][1] if 0 <= pretrained_idx < len(options) else None
        model_for_dim = CANONICAL_BACTERIAL_MODEL if pretrained_cli is None or pretrained_cli == 'None' else pretrained_cli
        if model_for_dim in PRETRAINED_NCHAN_NCLASSES_DIM:
            (_, _, dim) = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_dim]
            if dim == 2 and first_img_path:
                try:
                    arr = safe_imread(str(first_img_path))
                except Exception:
                    arr = None
                if arr is not None and arr.ndim == 3 and (arr.shape[0] > 4):
                    if not show_confirm_dialog(self, title='Compatibility', message=f'First image has shape {tuple(arr.shape)} (likely a 3D stack). The selected model is 2D. Continue anyway?', confirm_label='Yes', cancel_label='No', default_confirm=False):
                        return
        n_epochs = self.omnipose_n_epochs_spin.value()
        if pretrained_cli == 'None' and n_epochs < 1000:
            if not show_confirm_dialog(self, title='From-scratch training', message=f'You chose "From scratch" with only {n_epochs} epochs. Omnipose from-scratch typically needs around 4000 epochs to detect cells. Continue anyway?', confirm_label='Yes', cancel_label='No', default_confirm=False):
                return
        if pretrained_cli and pretrained_cli != 'None' and (n_epochs > 500):
            if not show_confirm_dialog(self, title='Fine-tuning with many epochs', message=f'You are fine-tuning with {n_epochs} epochs. Many epochs often drift the output head and cause 0 cells. Recommended: start with 50 epochs. Continue anyway?', confirm_label='Yes', cancel_label='No', default_confirm=False):
                return
        nchan_val = self.omnipose_nchan_spin.value()
        (n_chan_updated, chan_errors) = ensure_training_images_nchan(Path(train_dir_abs), img_filter, mask_filter, nchan_val)
        self.omnipose_start_training_btn.setEnabled(False)
        self.omnipose_training_log.clear()
        self.omnipose_training_log.append(f'Found {pairs_ok} image/mask pair(s). All will be used every epoch for {n_epochs} epochs.')
        if first_img_path:
            try:
                arr = safe_imread(str(first_img_path))
                if arr is not None:
                    if arr.ndim == 3 and arr.shape[0] <= 8:
                        self.omnipose_training_log.append(f'Dataset dimensions (first image): Y={arr.shape[1]}, X={arr.shape[2]} (nchan={arr.shape[0]}).')
                    elif arr.ndim == 2:
                        self.omnipose_training_log.append(f'Dataset dimensions (first image): Y={arr.shape[0]}, X={arr.shape[1]}.')
            except Exception:
                pass
        tyx_y = max(8, self.omnipose_tyx_y_spin.value() // 8 * 8)
        tyx_x = max(8, self.omnipose_tyx_x_spin.value() // 8 * 8)
        self.omnipose_training_log.append(f'Crop size (tyx): {tyx_y}×{tyx_x}.')
        if use_tiled:
            self.omnipose_training_log.append('Using tiled images. Crop size is typically half the tile size so each epoch uses random crops (data augmentation).')
        if chan_errors:
            for err in chan_errors[:5]:
                self.omnipose_training_log.append(f'Warning: {err}')
            if len(chan_errors) > 5:
                self.omnipose_training_log.append(f'... and {len(chan_errors) - 5} more.')
        if n_chan_updated > 0:
            self.omnipose_training_log.append(f'Ensured {n_chan_updated} image(s) have {nchan_val} channels in CYX order for training.')
        pretrained_arg = pretrained_cli if pretrained_cli is not None else CANONICAL_BACTERIAL_MODEL
        if pretrained_cli == 'None':
            self.omnipose_training_log.append('From scratch: --pretrained_model None, nchan=2, nclasses=3.')
        elif pretrained_cli:
            self.omnipose_training_log.append(f'Transfer learning: starting from {pretrained_cli} (nchan/nclasses match that model).')
        else:
            self.omnipose_training_log.append(f'Initializing from {CANONICAL_BACTERIAL_MODEL} (same architecture), training on your data.')
        self.omnipose_training_log.append('Starting Omnipose training (this may take a while)…')
        cmd = [get_python_executable_for_subprocess(), '-m', 'omnipose', '--train', '--dir', train_dir_abs, '--mask_filter', mask_filter, '--pretrained_model', str(pretrained_arg), '--diameter', '0', '--n_epochs', str(self.omnipose_n_epochs_spin.value()), '--learning_rate', str(self.omnipose_lr_spin.value()), '--batch_size', str(self.omnipose_batch_spin.value()), '--nchan', str(self.omnipose_nchan_spin.value()), '--nclasses', str(self.omnipose_nclasses_spin.value()), '--residual_on', '1', '--style_on', '1', '--concatenation', '0', '--RAdam']
        if img_filter:
            cmd.extend(['--img_filter', img_filter])
        if self.omnipose_use_gpu_cb.isChecked():
            cmd.append('--use_gpu')
        if tyx_y != 224 or tyx_x != 224:
            cmd.extend(['--tyx', f'{tyx_y},{tyx_x}'])
        save_every_val = self.omnipose_save_each_spin.value()
        if save_every_val > 0:
            cmd.extend(['--save_every', str(save_every_val), '--save_each'])
        if self.omnipose_nchan_spin.value() >= 2:
            cmd.extend(['--channel_axis', '0', '--all_channels'])
        import tempfile
        if save_every_val > 0:
            save_dir_path = Path(train_dir_abs) / 'models'
            save_dir_path.mkdir(parents=True, exist_ok=True)
            save_dir = str(save_dir_path)
            keep_save_dir = True
        else:
            save_dir = tempfile.mkdtemp(prefix='bactomate_train_')
            keep_save_dir = False
        env = os.environ.copy()
        env['CELLPOSE_LOCAL_MODELS_PATH'] = save_dir
        worker = _OmniposeTrainWorker(cmd, env, train_dir_abs, save_dir, str(dest_file), final_stem, self.omnipose_nchan_spin.value(), self.omnipose_nclasses_spin.value(), keep_save_dir=keep_save_dir)
        self._omnipose_train_worker = worker

        def on_log(line):
            formatted = self._format_omnipose_training_log_line(line)
            if formatted:
                self.omnipose_training_log.append(formatted)

        def on_finished(success, name, msg):
            self.omnipose_start_training_btn.setEnabled(True)
            self.omnipose_training_log.append(f'\n{msg}')
            if success:
                training_record = {'model_name': name, 'n_epochs': self.omnipose_n_epochs_spin.value(), 'learning_rate': self.omnipose_lr_spin.value(), 'batch_size': self.omnipose_batch_spin.value(), 'nchan': self.omnipose_nchan_spin.value(), 'nclasses': self.omnipose_nclasses_spin.value(), 'use_gpu': self.omnipose_use_gpu_cb.isChecked(), 'mask_filter': self.omnipose_mask_filter_edit.text().strip() or '_mask', 'img_filter': self.omnipose_img_filter_edit.text().strip() or '', 'train_dir': self.omnipose_train_dir_edit.text().strip(), 'pretrained_model': pretrained_arg}
                self._trained_models_this_session.append(training_record)
                if getattr(self.parent_window, '_save_session_log', None):
                    self.parent_window._save_session_log()
                done_msg = f"Model '{name}' saved to:\n{dest_file}\n\nIt will appear in the model dropdown below. When using it: if no cells are detected, set Diameter (e.g. 30 px) in Segmentation parameters; custom models do not support auto-diameter."
                if save_every_val > 0:
                    done_msg += f"\n\nIntermediate checkpoints (every {save_every_val} epochs) are in:\n{Path(train_dir_abs) / 'models'}"
                done_msg = f'(Initialized from {pretrained_arg})\n\n' + done_msg
                show_message_dialog(self, 'Done', done_msg, kind='info')
                current = self.model_combo.currentText()
                self.model_combo.clear()
                self.model_combo.addItems(self._get_available_models())
                if self.model_combo.findText(name) >= 0:
                    self.model_combo.setCurrentText(name)
                else:
                    self.model_combo.setCurrentText(current or 'bact_phase_omni')
            else:
                show_message_dialog(self, 'Training failed', msg, kind='warning')
            self._omnipose_train_worker = None
        worker.log_line.connect(on_log)
        worker.finished_with.connect(on_finished)
        worker.start()

    def _build_file_group(self) -> QWidget:
        """Path row + preview/segment actions in one bordered panel (same as Filtering)."""
        file_group = new_module_path_action_panel(self)
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText('Select a TIFF file or folder… (or drag and drop)')
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        self.file_path.setAcceptDrops(False)
        file_btn = QPushButton('Browse File...')
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton('Browse Folder...')
        folder_btn.clicked.connect(self.select_folder)
        path_lay = module_path_action_outer_layout(file_group)
        path_lay.addWidget(module_path_drop_row(
            file_group,
            self.handle_dropped_path_segmentation,
            line_edit=self.file_path,
            buttons=(file_btn, folder_btn),
            info_icon=create_info_icon(
                'Path to a TIFF file or folder for segmentation.\n\n'
                'Folder: processes all compatible TIFFs in that directory. '
                'Drag and drop a file or folder onto this area.\n\n'
                'Foci pairing (when Detect foci is on): put the segmentation input and '
                'foci stacks in the same folder with matching sample IDs. Names use '
                'underscores between sample ID and channel tags — hyphen exports '
                'from other software are not paired automatically.\n\n'
                'Works:\n'
                '  EM20129_phase.tif  +  EM20129_foci_ssb.tif\n'
                '  sample_Phase.tif  +  sample_foci_GFP.tif\n\n'
                'Does not pair:\n'
                '  EM20129-Membrane.tif  +  EM20129-ssb_foci.tif\n\n'
                'Best: File Conversion with Foci ticked for fluorescence channels — '
                'writes sample_phase.tif and sample_foci_<channel>.tif.',
                self,
            ),
        ))
        actions_layout = module_path_action_buttons_layout()
        self._preview_btn_idle = 'Preview segmentation'
        self.preview_btn = ProgressCancelButton(self._preview_btn_idle)
        self.preview_btn.setEnabled(False)
        style_module_primary_button(self.preview_btn)

        def safe_preview_segmentation():
            try:
                self.preview_segmentation()
            except BaseException as e:
                get_logger().exception('Error in preview segmentation')
                import traceback
                try:
                    show_message_dialog(self, 'Preview Error', f'Error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}', kind='error')
                except Exception:
                    pass
            finally:
                if getattr(self, '_segmentation_busy_action', None) == 'preview':
                    self._set_segmentation_buttons_busy(False)
                    self._update_button_states()
        self.preview_btn.clicked.connect(safe_preview_segmentation)
        actions_layout.addWidget(self.preview_btn)
        self.folder_preview_btn = QPushButton('Preview folder')
        self.folder_preview_btn.setEnabled(False)
        self.folder_preview_btn.setToolTip(
            'Run current segmentation (+ Spotiflow foci if enabled) on up to 48 files '
            'and show a QC grid. Double-click a tile to enlarge; results are cached until settings change.'
        )
        style_module_primary_button(self.folder_preview_btn)
        self.folder_preview_btn.clicked.connect(self.preview_folder_segmentation)
        actions_layout.addWidget(self.folder_preview_btn)
        self.process_btn = ProgressCancelButton('Segment File/Folder')
        self.process_btn.setEnabled(False)
        style_module_primary_button(self.process_btn)
        self._segmentation_in_progress = False
        self._segmentation_busy_action: Optional[str] = None
        self.process_btn.clicked.connect(self._on_process_or_cancel_clicked)
        actions_layout.addWidget(self.process_btn)
        path_lay.addLayout(actions_layout)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self._user_cancelled_processing = False

        def on_cancel_processing():
            if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
                self._user_cancelled_processing = True
                self.worker.terminate_process()
                self.parent_window.log_output('Cancelled by user.\n')
        self.cancel_btn.clicked.connect(on_cancel_processing)
        return file_group

    def _build_segmentation_params(self) -> QGroupBox:
        """Build Run segmentation section (U-Net style: model full line, then file, params, options, buttons)."""
        seg_group = QGroupBox('')
        seg_group.setFlat(True)
        seg_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Model:'))
        model_row.addWidget(create_info_icon("Omnipose model type for cell segmentation.\n\nDefault: 'bact_phase_omni'\nModels are downloaded from: https://github.com/kevinjohncutler/omnipose-models\nDocumentation: https://omnipose.readthedocs.io/models.html\n\nAvailable Omnipose models:\n- bact_phase_omni: Bacterial phase contrast (recommended for bacteria)\n- bact_fluor_omni: Bacterial fluorescence\n- bact_phase_affinity: Bacterial phase contrast (affinity-based)\n\nTip: Choose based on your cell type and imaging modality. User-trained models appear here after training above.", self))
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        available_models = self._get_available_models()
        self.model_combo.addItems(available_models)
        if available_models:
            if 'bact_phase_omni' in available_models:
                self.model_combo.setCurrentText('bact_phase_omni')
            else:
                self.model_combo.setCurrentText(available_models[0])
        model_row.addWidget(self.model_combo, 1)
        style_module_combo(self.model_combo)
        self.model_status_label = QLabel('')
        self.model_status_label.setStyleSheet('color: #7F8C8D; font-size: 9pt; font-style: italic;')
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        model_row.addWidget(self.model_status_label)
        main_layout.addLayout(model_row)
        file_group = self._build_file_group()
        main_layout.addWidget(file_group, 0)
        main_layout.addSpacing(6)
        diameter_row = QHBoxLayout()
        diameter_row.addWidget(QLabel('Diameter:'))
        diameter_row.addWidget(create_info_icon('Expected cell diameter in pixels. 0 = auto-detect. Specify a value (e.g. 20–50 for bacteria) for more consistent results.', self))
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setValue(0)
        self.diameter.setDecimals(1)
        flex_control(self.diameter, 72, combo_style=False)
        diameter_row.addWidget(self.diameter)
        diameter_row.addStretch(1)
        flow_row = QHBoxLayout()
        flow_row.addWidget(_omnipose_seg_param_qlabel('Flow:'))
        flow_row.addWidget(create_info_icon('Flow error threshold. Default 0.4. Use 0.0 if you get 0 cells with low mask threshold.', self))
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 1.0)
        self.flow_threshold.setValue(0.4)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setDecimals(2)
        flex_control(self.flow_threshold, 72, combo_style=False)
        flow_row.addWidget(self.flow_threshold)
        flow_row.addStretch(1)
        mask_row = QHBoxLayout()
        mask_row.addWidget(_omnipose_seg_param_qlabel('Mask thresh:'))
        self.mask_threshold_info = create_info_icon('Threshold for mask generation. Meaning depends on the selected model (see tooltip after choosing a model).', self)
        mask_row.addWidget(self.mask_threshold_info)
        self.mask_threshold = QDoubleSpinBox()
        self.mask_threshold.setRange(-20.0, 1.0)
        self.mask_threshold.setValue(0.0)
        self.mask_threshold.setSingleStep(0.5)
        self.mask_threshold.setDecimals(2)
        flex_control(self.mask_threshold, 72, combo_style=False)
        mask_row.addWidget(self.mask_threshold)
        mask_row.addStretch(1)
        reset_seg_btn = QPushButton('Reset to Default')
        reset_seg_btn.setToolTip('Reset all segmentation parameters to their default values')
        reset_seg_btn.clicked.connect(self.reset_segmentation_defaults)
        check_pipeline_btn = QPushButton('Check pipeline')
        check_pipeline_btn.setToolTip('Run model diagnostic: check and compare models.')
        check_pipeline_btn.clicked.connect(self._open_check_pipeline_dialog)
        action_row = QHBoxLayout()
        action_row.addWidget(reset_seg_btn)
        action_row.addWidget(check_pipeline_btn)
        action_row.addStretch(1)
        self.cluster = QCheckBox('Use DBSCAN clustering')
        self.cluster.setChecked(False)
        cluster_info = create_info_icon('Use DBSCAN clustering to help segment thin/elongated cells (e.g. bacteria, worms). Slightly increases processing time.', self)
        cluster_row = QHBoxLayout()
        cluster_row.addWidget(self.cluster)
        cluster_row.addWidget(cluster_info)
        cluster_row.addStretch(1)
        self._seg_tuning_widget = ResponsivePanel(breakpoint=560)
        self._seg_tuning_widget.set_defer_reflow(True)
        self._seg_tuning_widget.add_widget(hbox_row_widget(diameter_row))
        self._seg_tuning_widget.add_widget(hbox_row_widget(flow_row))
        self._seg_tuning_widget.add_widget(hbox_row_widget(mask_row))
        self._seg_tuning_widget.add_widget(hbox_row_widget(cluster_row))
        self._seg_tuning_widget.add_widget(hbox_row_widget(action_row))
        self._seg_tuning_widget.set_defer_reflow(False)
        self._seg_tuning_widget.reflow()
        main_layout.addWidget(self._seg_tuning_widget)
        self._seg_tuning_layout_index = main_layout.indexOf(self._seg_tuning_widget)
        self.use_gpu = QCheckBox('Use GPU')
        self.use_gpu.setChecked(True)
        self.create_overlay = QCheckBox('Create overlay visualization')
        self.create_overlay.setChecked(True)
        self.create_overlay.setVisible(False)
        overlay_info = create_info_icon("Create overlay TIFF (masks over original). Saved in an 'overlays' subfolder so the main folder has only mask files for Analyze Masks. Disable to speed up when you only need masks.", self)
        self.assign_persistent_cellids = QCheckBox('Assign persistent Cell IDs')
        self.assign_persistent_cellids.setChecked(True)
        self.assign_persistent_cellids.setVisible(False)
        cellids_info = create_info_icon(
            'After segmentation, cells are linked across frames and persistent Cell IDs are written '
            'into {stem}_masks.tif (the file Analyze Masks and Tracking use). '
            'A separate *_masks_cellids.tif copy is no longer written.',
            self,
        )
        self.detect_foci = QCheckBox('Detect foci')
        self.detect_foci.setChecked(False)
        foci_info = create_info_icon('When enabled, BactoMate looks for `*_foci.tif` next to the input file/folder and runs spot detection.\n\nOutputs a raw foci CSV in the segmentation output folder (one per foci channel).\nTune Spotiflow per channel in the preview (Foci channel dropdown under Spotiflow).\nColumns: y, x, intensity, probability (and frame when stacks have multiple frames).\n\nSibling foci TIFFs must share the sample prefix with the segmentation input — see the Path info box for naming examples.', self)
        options_panel = ResponsivePanel(breakpoint=480)
        options_panel.set_defer_reflow(True)
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.use_gpu)
        gpu_row.addStretch(1)
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(self.create_overlay)
        overlay_row.addWidget(overlay_info)
        overlay_row.addStretch(1)
        cellids_row = QHBoxLayout()
        cellids_row.addWidget(self.assign_persistent_cellids)
        cellids_row.addWidget(cellids_info)
        cellids_row.addStretch(1)
        foci_row = QHBoxLayout()
        foci_row.addWidget(self.detect_foci)
        foci_row.addWidget(foci_info)
        foci_row.addStretch(1)
        options_panel.add_widget(hbox_row_widget(gpu_row))
        options_panel.add_widget(hbox_row_widget(overlay_row))
        options_panel.add_widget(hbox_row_widget(cellids_row))
        options_panel.add_widget(hbox_row_widget(foci_row))
        options_panel.set_defer_reflow(False)
        options_panel.reflow()
        self._inference_options_panel = options_panel
        main_layout.addWidget(options_panel)

        for _w in (self.diameter, self.flow_threshold, self.mask_threshold):
            try:
                _w.valueChanged.connect(self._mark_folder_preview_stale_if_needed)
            except Exception:
                pass
        for _w in (self.cluster, self.detect_foci):
            try:
                _w.toggled.connect(self._mark_folder_preview_stale_if_needed)
            except Exception:
                pass
        try:
            self.model_combo.currentIndexChanged.connect(self._mark_folder_preview_stale_if_needed)
        except Exception:
            pass

        def check_model_availability():
            try:
                display = self.model_combo.currentText()
                (model_type, pretrained_path) = self._resolve_model_selection(display)
                if pretrained_path is not None:
                    self.model_status_label.setText('✓ User model')
                    self.model_status_label.setStyleSheet('color: #4CAF50; font-size: 9pt;')
                    (d, f, m) = USER_MODEL_DEFAULTS
                    self.diameter.setValue(d)
                    self.flow_threshold.setValue(f)
                    self.mask_threshold.setValue(m)
                    return
                if model_type == CELLPOSE_SAM_MODEL:
                    self.model_status_label.setText('✓ Cellpose-SAM')
                    self.model_status_label.setStyleSheet('color: #4CAF50; font-size: 9pt;')
                    (d, f, m) = SEGMENTATION_MODEL_DEFAULTS.get(CELLPOSE_SAM_MODEL, SEGMENTATION_DEFAULT_FALLBACK)
                    self.diameter.setValue(d)
                    self.flow_threshold.setValue(f)
                    self.mask_threshold.setValue(m)
                    return
                from bactomate.gui.omnipose_pretrained_cache import try_get_bactomate_cached_builtin
                from cellpose_omni.models import MODEL_DIR
                basename = f'{model_type}torch_0'
                bact_cached = try_get_bactomate_cached_builtin(model_type) is not None
                cellpose_cached = (Path(MODEL_DIR) / basename).is_file()
                if bact_cached:
                    self.model_status_label.setText('✓ Cached (BactoMate)')
                    self.model_status_label.setStyleSheet('color: #4CAF50; font-size: 9pt;')
                elif cellpose_cached:
                    self.model_status_label.setText('✓ Cached (will copy to BactoMate on use)')
                    self.model_status_label.setStyleSheet('color: #4CAF50; font-size: 9pt;')
                else:
                    self.model_status_label.setText('Will download on first use')
                    self.model_status_label.setStyleSheet('color: #FF9800; font-size: 9pt;')
                (d, f, m) = SEGMENTATION_MODEL_DEFAULTS.get(model_type or display, SEGMENTATION_DEFAULT_FALLBACK)
                self.diameter.setValue(d)
                self.flow_threshold.setValue(f)
                self.mask_threshold.setValue(m)
            except Exception:
                get_logger().debug('Model availability check failed', exc_info=True)
                self.model_status_label.setText('')
            finally:
                try:
                    self._sync_mask_threshold_spinbox_for_model()
                except Exception:
                    get_logger().debug('Mask threshold spinbox sync failed', exc_info=True)
        self.model_combo.currentTextChanged.connect(check_model_availability)
        self.model_combo.currentTextChanged.connect(lambda _text: self._schedule_open_segmentation_preview_redetect())
        check_model_availability()
        seg_group.setLayout(main_layout)
        return seg_group

    def _process_btn_idle_label(self) -> str:
        raw = (self.file_path.text() or '').strip()
        if raw:
            p = Path(raw)
            if p.is_dir():
                return 'Segment All Files in Folder'
            if p.is_file():
                return 'Segment File'
        return 'Segment File/Folder'

    def _segmentation_run_buttons(self) -> tuple[ProgressCancelButton, ...]:
        out: list[ProgressCancelButton] = []
        for name in ('preview_btn', 'process_btn'):
            btn = getattr(self, name, None)
            if isinstance(btn, ProgressCancelButton):
                out.append(btn)
        return tuple(out)

    def _set_segmentation_buttons_busy(self, busy: bool, *, action: Optional[str]=None, progress_label: Optional[str]=None) -> None:
        """Blue in-button progress on the active segmentation action (preview load or batch run)."""
        idle_by_action = {'preview': getattr(self, '_preview_btn_idle', 'Preview segmentation'), 'process': self._process_btn_idle_label()}
        btn_by_action = {'preview': getattr(self, 'preview_btn', None), 'process': getattr(self, 'process_btn', None)}
        if busy:
            self._segmentation_busy_action = action
            for b in self._segmentation_run_buttons():
                b.set_progress_active(False)
                b.setEnabled(False)
            active = btn_by_action.get(action or '')
            if isinstance(active, ProgressCancelButton):
                active.setText(progress_label or idle_by_action.get(action or '', ''))
                active.set_progress_subtitle('')
                active.setEnabled(True)
                active.set_progress_active(True)
                active.set_progress(3)
            return
        self._segmentation_busy_action = None
        for (act, btn) in btn_by_action.items():
            if isinstance(btn, ProgressCancelButton):
                btn.set_progress_active(False)
                btn.set_progress_subtitle('')
                btn.setText(idle_by_action.get(act, btn.text()))
        self._update_button_states()

    def _update_segmentation_button_progress(self, value: int, message: str='') -> None:
        action = getattr(self, '_segmentation_busy_action', None)
        pct = max(3, min(100, int(value))) if int(value) >= 0 else 0
        btn_by_action = {'preview': getattr(self, 'preview_btn', None), 'process': getattr(self, 'process_btn', None)}
        active = btn_by_action.get(action or '')
        if not isinstance(active, ProgressCancelButton):
            return
        if not getattr(active, '_progress_active', False):
            return
        active.set_progress(pct)
        subtitle = ''
        m = re.search('(\\d+)\\s*/\\s*(\\d+)', message or '')
        if m and action == 'process':
            subtitle = f'({m.group(1)}/{m.group(2)} files)'
        active.set_progress_subtitle(subtitle)

    def _reset_process_btn_idle(self) -> None:
        self._set_segmentation_buttons_busy(False)

    def _set_process_btn_cancel_mode(self, progress: int=0) -> None:
        if getattr(self, '_segmentation_busy_action', None) != 'process':
            self._set_segmentation_buttons_busy(True, action='process', progress_label='Cancel')
        self._update_segmentation_button_progress(max(int(progress), 3), '')

    def _set_segmentation_progress(self, value: int, message: str='') -> None:
        value = max(0, min(100, int(value)))
        try:
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(value)
        except Exception:
            pass
        self._update_segmentation_button_progress(value, message)

    def _on_process_or_cancel_clicked(self) -> None:
        if bool(getattr(self, '_segmentation_in_progress', False)):
            try:
                if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
                    self._user_cancelled_processing = True
                    self.worker.terminate_process()
            except Exception:
                pass
            try:
                self.process_btn.setEnabled(False)
                self.process_btn.setText('Cancelling…')
            except Exception:
                pass
            return
        self.process_file_or_folder()

    def _no_foci_channels_found_message(self, input_paths: list[Path]) -> str:
        """User-facing text when Detect foci is on but no *_foci* TIFF siblings exist."""
        examples = '\n'.join((f'  • {p.name}' for p in input_paths[:3]))
        more = f'\n  • … ({len(input_paths) - 3} more)' if len(input_paths) > 3 else ''
        return f'No foci channel TIFFs were found for the selected input(s).\n\nSegmentation’s “Detect foci” looks for converted files whose names contain "_foci" (for example sample_foci_GFP.tif or sample_fi_foci_Channel_2.tif).\n\nCreate them in File Conversion: tick Foci for each fluorescence channel before splitting or converting a multi-channel TIFF, then run segmentation again.\n\nInput(s) checked:\n{examples}{more}'

    def _warn_no_foci_channels_if_enabled(self, input_paths: list[Path]) -> bool:
        """If Detect foci is on and no *_foci* stacks exist, show a dialog. Returns True when OK to continue."""
        if not input_paths:
            return True
        if not (getattr(self, 'detect_foci', None) is not None and self.detect_foci.isChecked()):
            return True
        found: list[Path] = []
        for p in input_paths:
            try:
                found.extend(self._find_foci_sources_for_input(Path(p)))
            except Exception:
                continue
        if found:
            return True
        show_message_dialog(self, 'No foci channels found', self._no_foci_channels_found_message([Path(x) for x in input_paths]), kind='info')
        return True

    def _find_foci_sources_for_input(self, input_image_path: Path) -> list[Path]:
        from bactomate.gui.artifact_paths import discover_foci_companions

        return discover_foci_companions(input_image_path)

    def _companions_for_segmentation_input(self, input_image_path: Path) -> dict[str, list]:
        """Linked TIFFs shown before folder-mode segmentation when Detect foci is enabled."""
        input_image_path = Path(input_image_path)
        companion_rows: list[tuple[Path, str]] = []
        try:
            for fp in self._find_foci_sources_for_input(input_image_path):
                companion_rows.append((Path(fp), 'spotiflow companion'))
        except Exception:
            pass
        return {
            'mask': [(input_image_path, 'segmentation input')],
            'phase': [],
            'foci': [],
            'fi': companion_rows,
        }

    @staticmethod
    def _is_segmentation_input_file(path: Path) -> bool:
        """Return True when a TIFF should be treated as a cell-segmentation input.

        Allowed: phase/BF stacks and (optionally) plain ``*_fi_*`` fluorescence for
        fluor-model workflows. Excluded: foci detection stacks (``*_foci*``,
        ``*_fi_foci*``), masks, overlays, and QC visualizations.
        """
        name = path.name.lower()
        if '_masks' in name or '_overlay' in name or '_filter_visualization' in name:
            return False
        # Foci companions are for Spotiflow peak detection, not cell outlines.
        if '_fi_foci' in name or '_foci' in name:
            return False
        return True

    @staticmethod
    def _looks_like_fi_or_foci_input(path: Path) -> bool:
        """True for fluorescence/foci TIFFs that should not be auto-selected over phase images."""
        name = Path(path).name.lower()
        return '_fi' in name or '_foci' in name

    @classmethod
    def _default_segmentation_selection_predicate(cls, tiff_files: list[Path]):
        """
        Prefer plain segmentation inputs by default.

        If at least one TIFF name contains neither ``_fi`` nor ``_foci``, auto-check only
        those plain files. If none exist, fall back to checking remaining listed files
        (typically plain ``*_fi_*`` fluorescence for bact_fluor-style runs — foci stacks
        are already excluded from the list).
        """
        files = [Path(p) for p in tiff_files]
        has_plain_input = any((not cls._looks_like_fi_or_foci_input(p) for p in files))
        if not has_plain_input:
            return None
        return lambda p: not cls._looks_like_fi_or_foci_input(Path(p))

    def _write_raw_foci_csv(self, *, output_csv: Path, rows: list[dict], include_frame: bool) -> None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['y', 'x', 'intensity', 'probability']
        if include_frame:
            fieldnames = ['frame'] + fieldnames
        with open(output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                out = {'y': int(r.get('y', 0)), 'x': int(r.get('x', 0)), 'intensity': float(r.get('intensity', 0.0) or 0.0), 'probability': '' if r.get('probability') is None else float(r.get('probability'))}
                if include_frame:
                    out['frame'] = int(r.get('frame', 0))
                w.writerow(out)

    def _detect_foci_for_stack_and_write_csv(self, *, foci_stack: np.ndarray, output_csv: Path, prob_thresh: Optional[float]=None, spotiflow_params: dict[str, Any] | None=None) -> tuple[bool, str]:
        """Run Spotiflow detection per frame and write raw CSV.

        Returns ``(ok, reason)`` where ``ok=False`` indicates Spotiflow/import/predict failure.
        """
        try:
            from bactomate.modules.image_stack_utils import coerce_stack_to_tyx, coerce_plane_to_2d
            from bactomate.modules.foci_utils import detect_peaks_spotiflow, load_spotiflow_model
        except Exception as e:
            return (False, f'imports failed: {e}')
        arr = coerce_stack_to_tyx(np.asarray(foci_stack))
        if arr is None or arr.size == 0:
            return (False, 'empty or invalid foci stack')
        include_frame = bool(arr.shape[0] > 1)
        rows: list[dict] = []
        if isinstance(spotiflow_params, dict) and spotiflow_params:
            params = {**self._get_spotiflow_detection_params(), **spotiflow_params}
        else:
            params = self._get_spotiflow_detection_params()
        if prob_thresh is not None:
            params['use_optimized_threshold'] = False
            params['prob_thresh'] = float(prob_thresh)
        try:
            spotiflow_model = load_spotiflow_model(model_name=str(params.get('model_name') or 'general'), model_path=str(params.get('model_path') or '') or None)
        except Exception as e:
            return (False, f'Spotiflow model load failed: {e}')
        min_prob_csv = float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0)
        for (fi, frame) in enumerate(arr):
            plane = np.asarray(coerce_plane_to_2d(np.asarray(frame)), dtype=np.float64)
            peaks = detect_peaks_spotiflow(plane, spotiflow_model=spotiflow_model, **params)
            if peaks is None:
                return (False, f'Spotiflow unavailable or prediction failed at frame {fi}')
            if min_prob_csv > 0:
                peaks = [p for p in peaks if p.get('probability') is None or float(p.get('probability') or 0.0) >= min_prob_csv]
            for p in peaks:
                q = dict(p)
                q['frame'] = fi
                rows.append(q)
        self._write_raw_foci_csv(output_csv=output_csv, rows=rows, include_frame=include_frame)
        return (True, '')

    def _update_button_states(self):
        """Update button enabled states: path must exist and be a file or directory."""
        raw = self.file_path.text() and self.file_path.text().strip()
        has_path = False
        if raw:
            p = Path(raw)
            has_path = p.exists() and (p.is_file() or p.is_dir())
        if hasattr(self, 'preview_btn'):
            if not getattr(self, '_segmentation_in_progress', False) and (not getattr(self, '_segmentation_busy_action', None)):
                self.preview_btn.setEnabled(has_path)
        if hasattr(self, 'folder_preview_btn'):
            if not getattr(self, '_segmentation_in_progress', False) and (not getattr(self, '_segmentation_busy_action', None)):
                n = len(self._list_folder_preview_candidates()) if has_path else 0
                from bactomate.gui.folder_preview.constants import folder_preview_enabled
                self.folder_preview_btn.setEnabled(has_path and folder_preview_enabled(n))
        if hasattr(self, 'process_btn'):
            if not getattr(self, '_segmentation_in_progress', False) and (not getattr(self, '_segmentation_busy_action', None)):
                self.process_btn.setEnabled(has_path)

    def _update_overlay_checkbox_state(self):
        """Update overlay checkbox tooltip based on folder vs file mode"""
        is_folder = self._is_folder_mode()
        self.create_overlay.setEnabled(True)
        if is_folder:
            self.create_overlay.setToolTip("Create overlay visualization showing masks overlaid on original images.\n\nDefault: Enabled (checked)\nWhen enabled: Creates a colored overlay TIFF file for each file in the folder.\nWhen disabled: Skips overlay creation (faster processing).\n\nSaved as: filename_overlay.tif (in each file's output directory)")
        else:
            self.create_overlay.setToolTip('Create overlay visualization showing masks overlaid on original images.\n\nDefault: Enabled (checked)\nWhen enabled: Creates a colored overlay TIFF file.\nWhen disabled: Skips overlay creation (faster processing).\n\nSaved as: filename_overlay.tif')

    def _try_push_metadata_from_path(self):
        """When path is a TIFF file (or folder with a TIFF), read calibration via tiff_metadata and push to shared_metadata."""
        raw = (self.file_path.text() or '').strip()
        if not raw:
            return
        path = Path(raw)
        tiff_path = None
        if path.is_file() and path.suffix.lower() in ('.tif', '.tiff'):
            tiff_path = path
        elif path.is_dir():
            pick = (getattr(self, '_folder_preview_pick', None) or '').strip()
            if pick:
                pp = Path(pick)
                if pp.is_file() and pp.suffix.lower() in ('.tif', '.tiff'):
                    tiff_path = pp
            if tiff_path is None:
                for ext in ('*.tif', '*.tiff'):
                    first = next(path.glob(ext), None)
                    if first is not None and first.is_file():
                        tiff_path = first
                        break
        if tiff_path is None:
            return
        try:
            (ps, tp) = read_calibration_from_path_or_original(tiff_path)
            if (ps > 0 or tp > 0) and hasattr(self, 'parent_window') and (getattr(self.parent_window, 'shared_metadata', None) is not None):
                if ps > 0:
                    self.parent_window.shared_metadata['pixel_size'] = ps
                if tp > 0:
                    self.parent_window.shared_metadata['time_per_frame_min'] = tp
                self.parent_window.update_all_tabs_with_metadata()
        except Exception:
            pass

    def select_file(self):
        """Open file dialog to select a TIFF file"""
        try:
            start_dir = ''
            if self.file_path.text() and Path(self.file_path.text()).exists():
                start_dir = str(Path(self.file_path.text()).parent)
            elif self.file_path.text():
                start_dir = str(Path(self.file_path.text()).parent)
            else:
                start_dir = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_SEGMENTATION) or str(Path.home())
            (file_path, _) = QFileDialog.getOpenFileName(self, 'Select TIFF File', start_dir, 'TIFF Files (*.tif *.tiff);;All Files (*)')
            if file_path:
                if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                    self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, file_path)
                self.file_path.setText(file_path)
                self._update_overlay_checkbox_state()
                if self.parent_window and hasattr(self.parent_window, 'log_output'):
                    self.parent_window.log_output(f'Selected file: {file_path}\n')
        except Exception as e:
            show_message_dialog(self, 'File Selection Error', f'Could not open file dialog:\n{str(e)}\n\n{traceback.format_exc()}', kind='error')
            if self.parent_window and hasattr(self.parent_window, 'log_output'):
                self.parent_window.log_output(f'Error selecting file: {str(e)}\n{traceback.format_exc()}\n')

    def select_folder(self):
        """Open folder dialog to select a folder"""
        start_dir = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_SEGMENTATION) or str(Path.home())
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', start_dir)
        if folder_path:
            if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, folder_path)
            self.file_path.setText(folder_path)
            self._update_overlay_checkbox_state()

    def handle_dropped_path_segmentation(self, dropped_path):
        """Handle dropped file or folder path for segmentation tab"""
        path_obj = Path(dropped_path)
        self.file_path.setText(dropped_path)
        self._update_overlay_checkbox_state()
        self._update_button_states()

    def _is_folder_mode(self):
        """Check if current path is a folder"""
        if not self.file_path.text():
            return False
        path = Path(self.file_path.text())
        return path.exists() and path.is_dir()

    def _get_available_models(self):
        """
        Return display names for the model dropdown for the active inference backend.

        Omnipose algorithm: built-in Omnipose models + user-trained .pth (no Cellpose-SAM).
        Cellpose algorithm: Cellpose-SAM only.
        """
        if getattr(self, '_inference_mode', 'omnipose') == 'cellpose_sam':
            return [CELLPOSE_SAM_DISPLAY]
        names = list(BUILTIN_OMNIPOSE_MODELS)
        try:
            user_dir = get_omnipose_user_models_dir()
            for p in sorted(user_dir.glob(f'*{USER_MODEL_EXTENSION}')):
                if p.is_file():
                    names.append(p.stem)
        except Exception:
            pass
        return names

    def _resolve_model_selection(self, display_name: str) -> tuple[Optional[str], Optional[Path]]:
        """
        Resolve combo display name to either built-in model_type or user pretrained path.
        Returns (model_type, pretrained_path): one is set, the other None.
        """
        if display_name in BUILTIN_OMNIPOSE_MODELS:
            return (display_name, None)
        if display_name == CELLPOSE_SAM_DISPLAY:
            return (CELLPOSE_SAM_MODEL, None)
        try:
            user_path = get_omnipose_user_models_dir() / f'{display_name}{USER_MODEL_EXTENSION}'
            if user_path.is_file():
                return (None, user_path)
        except Exception:
            pass
        return (None, None)

    def _sync_mask_threshold_spinbox_for_model(self) -> None:
        """Omnipose uses distance-map thresholds (-20..1). Cellpose-SAM uses cellprob_threshold (~-6..6)."""
        display = self.model_combo.currentText() or ''
        (model_type, _pretrained_path) = self._resolve_model_selection(display)
        v = float(self.mask_threshold.value())
        if model_type == CELLPOSE_SAM_MODEL:
            self.mask_threshold.setRange(MASK_CELLPROB_SAM_MIN, MASK_CELLPROB_SAM_MAX)
            self.mask_threshold.setSingleStep(0.1)
            v = max(MASK_CELLPROB_SAM_MIN, min(MASK_CELLPROB_SAM_MAX, v))
            self.mask_threshold.setValue(v)
            if getattr(self, 'mask_threshold_info', None) is not None:
                self.mask_threshold_info.setToolTip("Cellpose-SAM: this value is passed as cellprob_threshold (logit scale, roughly -6 to +6; default 0).\nLower values include more pixels in the dynamics step; higher values are stricter.\nThis is not the same as Omnipose's distance-map threshold.\n\nIf the app crashes when segmenting, try mask 0 and set environment variable OPENBLAS_NUM_THREADS=1 before launching BactoMate (known OpenBLAS/threading issue on some Macs).")
        else:
            self.mask_threshold.setRange(MASK_DISTANCE_OMNI_MIN, MASK_DISTANCE_OMNI_MAX)
            self.mask_threshold.setSingleStep(0.5)
            v = max(MASK_DISTANCE_OMNI_MIN, min(MASK_DISTANCE_OMNI_MAX, v))
            self.mask_threshold.setValue(v)
            if getattr(self, 'mask_threshold_info', None) is not None:
                self.mask_threshold_info.setToolTip("Omnipose: distance-map threshold (mask_threshold). If 'No cell pixels found', try -1, -5, or -10; also set Flow to 0. User-trained .pth models use the same semantics.")

    def _get_omnipose_segmentation_folder_name(self) -> str:
        """Return the segmentation output folder name from the current model (e.g. _segmentation_bact_phase_omni, _segmentation_cellpose_sam)."""
        display = self.model_combo.currentText() or ''
        (model_type, pretrained_path) = self._resolve_model_selection(display)
        if pretrained_path is not None:
            slug = pretrained_path.stem
        else:
            slug = (model_type or display).strip()
        slug = re.sub('[^a-z0-9_]', '_', slug.lower()).strip('_') if slug else 'omni'
        return f'_segmentation_{slug}'

    def get_segmentation_output_folder_name(self) -> str:
        """Public: return folder name for Re-run / main window (e.g. _segmentation_bact_phase_omni)."""
        return self._get_omnipose_segmentation_folder_name()

    @staticmethod
    def resolve_segmentation_output_dir_for_tiff(tiff_path: Path, folder_name: str) -> Path:
        """Return the output directory used for one TIFF (must match batch script invocation)."""
        tiff_path = Path(tiff_path)
        base_dir = tiff_path.parent
        if base_dir.name.startswith('Tif_'):
            return base_dir / str(folder_name)
        return base_dir / f'{tiff_path.stem}_omnipose_segmentation'

    def _get_spotiflow_detection_params(self, channel_key: str | None=None) -> dict[str, Any]:
        """Return Spotiflow parameters (defaults + global + optional per-channel overrides)."""
        merged = dict(SPOTIFLOW_DETECTION_DEFAULTS)
        custom = getattr(self, '_spotiflow_detection_params', None)
        if isinstance(custom, dict):
            merged.update(custom)
        from bactomate.gui.filter_preview_utils import normalize_foci_channel_key
        key = normalize_foci_channel_key(channel_key)
        if key:
            chmap = getattr(self, '_spotiflow_channel_params', None)
            if isinstance(chmap, dict):
                ch = chmap.get(key)
                if isinstance(ch, dict):
                    merged.update(ch)
        return merged

    def _ensure_spotiflow_channel_entries(self, foci_paths: list[Path]) -> None:
        """Create empty per-channel override entries for each foci stack.

        An empty entry means "inherit defaults + global params"; only values the user
        edits for a channel are stored (see ``_save_spotiflow_widgets_to_channel``).
        """
        chmap = getattr(self, '_spotiflow_channel_params', None)
        if not isinstance(chmap, dict):
            self._spotiflow_channel_params = {}
            chmap = self._spotiflow_channel_params
        for fp in foci_paths:
            tag = self._spotiflow_channel_key_for_path(Path(fp))
            if tag not in chmap:
                chmap[tag] = {}

    def _spotiflow_channel_key_for_path(self, foci_path: Path) -> str:
        """Canonical per-channel key for a foci stack; migrates a legacy-key entry (old sessions) once."""
        from bactomate.gui.filter_preview_utils import migrate_foci_channel_entry
        chmap = getattr(self, '_spotiflow_channel_params', None)
        if not isinstance(chmap, dict):
            self._spotiflow_channel_params = {}
            chmap = self._spotiflow_channel_params
        return migrate_foci_channel_entry(chmap, Path(foci_path))

    def _spotiflow_channel_key_for_preview_index(self, preview_dialog: Any, source_index: int) -> str:
        paths = getattr(preview_dialog, '_foci_source_paths', None) or []
        si = int(source_index)
        if 0 <= si < len(paths):
            return self._spotiflow_channel_key_for_path(Path(paths[si]))
        return '_default'

    @staticmethod
    def _strip_to_channel_overrides(entry: Any) -> dict[str, Any]:
        """Keep only per-channel UI keys (drops model/subpix/other global-only keys from legacy snapshots)."""
        if not isinstance(entry, dict):
            return {}
        return {k: v for (k, v) in entry.items() if k in SPOTIFLOW_CHANNEL_KEYS}

    def _spotiflow_slice_from_preview_widgets(self, dlg: QWidget) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            if getattr(dlg, '_sf_normalize_cb', None) is not None:
                out['normalize'] = bool(dlg._sf_normalize_cb.isChecked())
            if getattr(dlg, '_sf_perc_low', None) is not None:
                out['perc_low'] = float(dlg._sf_perc_low.value())
            if getattr(dlg, '_sf_perc_high', None) is not None:
                out['perc_high'] = float(dlg._sf_perc_high.value())
            if getattr(dlg, '_sf_use_optimized', None) is not None:
                out['use_optimized_threshold'] = bool(dlg._sf_use_optimized.isChecked())
            if getattr(dlg, '_sf_prob_thresh', None) is not None:
                out['prob_thresh'] = float(dlg._sf_prob_thresh.value())
            if getattr(dlg, '_sf_min_distance', None) is not None:
                out['min_distance'] = int(dlg._sf_min_distance.value())
        except Exception:
            get_logger().debug('read spotiflow widgets failed', exc_info=True)
        return out

    def _active_foci_source_index(self, preview_dialog: Any) -> int:
        combo = getattr(preview_dialog, '_sf_channel_combo', None) or getattr(preview_dialog, '_foci_source_combo', None)
        if combo is not None:
            idx = int(combo.currentIndex())
            sources = getattr(preview_dialog, '_foci_sources', None) or []
            if 0 <= idx < len(sources):
                return idx
        return 0

    def _invalidate_foci_peak_cache_for_source(self, preview_dialog: Any, source_index: int) -> None:
        """Remove cached peaks for one foci channel only (other channels keep their rings)."""
        try:
            si = int(source_index)
            cache = getattr(preview_dialog, '_foci_peaks_cache', None)
            if not isinstance(cache, dict):
                return
            for k in list(cache.keys()):
                if isinstance(k, tuple) and len(k) >= 1 and (int(k[0]) == si):
                    cache.pop(k, None)
            if self._active_foci_source_index(preview_dialog) == si:
                preview_dialog._foci_peaks_for_current = []
        except Exception:
            get_logger().debug('partial foci cache invalidate failed', exc_info=True)

    def _foci_peak_cache_key(self, source_index: int, frame_idx: int, channel_key: str, sf_local: dict[str, Any], csv_minp: float) -> tuple:
        return (int(source_index), int(frame_idx), str(channel_key), bool(sf_local.get('use_optimized_threshold', True)), float(sf_local.get('prob_thresh') if sf_local.get('prob_thresh') is not None else 0.5), int(sf_local.get('min_distance', 1) or 1), bool(sf_local.get('subpix', True)), bool(sf_local.get('normalize', True)), float(sf_local.get('perc_low', 1.0) or 1.0), float(sf_local.get('perc_high', 99.8) or 99.8), float(csv_minp))

    def _filter_foci_peaks_by_csv_threshold(self, peaks: list, csv_minp: float) -> list:
        thr = float(csv_minp or 0.0)
        if thr <= 0:
            return list(peaks)
        return [p for p in peaks if p.get('probability') is None or float(p.get('probability') or 0.0) >= thr]

    def _spotiflow_detect_peaks_on_plane(self, plane: np.ndarray, sf: dict[str, Any]) -> tuple[Optional[list], bool]:
        """Run Spotiflow on one 2D plane (thread-safe; no Qt). Returns (peaks, spotiflow_available)."""
        try:
            from bactomate.modules.foci_utils import detect_peaks_spotiflow, load_spotiflow_model
        except Exception as e:
            get_logger().warning('Spotiflow preview imports failed: %s', e, exc_info=True)
            return (None, False)
        _sf_opt = bool(sf.get('use_optimized_threshold', True))
        _sf_prob = float(sf.get('prob_thresh') if sf.get('prob_thresh') is not None else 0.5)
        _sf_min_dist = int(sf.get('min_distance', 1) or 1)
        _sf_subpix = bool(sf.get('subpix', True))
        _sf_norm = bool(sf.get('normalize', True))
        _sf_pl = float(sf.get('perc_low', 1.0) or 1.0)
        _sf_ph = float(sf.get('perc_high', 99.8) or 99.8)
        try:
            prob_kw = None if _sf_opt else float(_sf_prob)
            mn = str(sf.get('model_name') or 'general')
            mp = str(sf.get('model_path') or '').strip() or None
            spotiflow_model = load_spotiflow_model(model_name=mn, model_path=mp)
            det_kw = {k: v for (k, v) in sf.items() if k not in ('model_name', 'model_path', 'raise_on_error')}
            det_kw['use_optimized_threshold'] = bool(_sf_opt)
            det_kw['prob_thresh'] = prob_kw
            det_kw['min_distance'] = int(_sf_min_dist)
            det_kw['subpix'] = bool(_sf_subpix)
            det_kw['normalize'] = bool(_sf_norm)
            det_kw['perc_low'] = float(_sf_pl)
            det_kw['perc_high'] = float(_sf_ph)
            det_kw['raise_on_error'] = False
            result = detect_peaks_spotiflow(np.asarray(plane, dtype=np.float64), spotiflow_model=spotiflow_model, **det_kw)
        except Exception as e:
            get_logger().warning('Spotiflow preview detection raised: %s', e, exc_info=True)
            return (None, False)
        if result is None:
            return (None, False)
        return (list(result), True)

    def _active_spotiflow_channel_key(self, dlg: QWidget) -> str:
        combo = getattr(dlg, '_sf_channel_combo', None) or getattr(dlg, '_foci_source_combo', None)
        if combo is not None:
            try:
                data = combo.currentData()
                if isinstance(data, str) and data.strip():
                    return data.strip()
            except Exception:
                pass
            idx = int(combo.currentIndex())
            paths = getattr(dlg, '_foci_source_paths', None) or []
            if 0 <= idx < len(paths):
                return self._spotiflow_channel_key_for_preview_index(dlg, idx)
        paths = getattr(dlg, '_foci_source_paths', None) or []
        if paths:
            return self._spotiflow_channel_key_for_preview_index(dlg, 0)
        return '_default'

    def _save_spotiflow_widgets_to_channel(self, dlg: QWidget, channel_key: str) -> None:
        from bactomate.gui.filter_preview_utils import normalize_foci_channel_key
        key = normalize_foci_channel_key(channel_key) or '_default'
        slice_dict = self._spotiflow_slice_from_preview_widgets(dlg)
        if not isinstance(getattr(self, '_spotiflow_channel_params', None), dict):
            self._spotiflow_channel_params = {}
        # Per-channel overrides only; the global template is never mutated by a channel edit.
        # Store only values that differ from the effective global; keys equal to global stay inherited.
        global_params = self._get_spotiflow_detection_params()
        overrides = {k: v for (k, v) in self._strip_to_channel_overrides(slice_dict).items() if k not in global_params or global_params[k] != v}
        self._spotiflow_channel_params[key] = overrides

    def _load_spotiflow_widgets_for_channel(self, dlg: QWidget, channel_key: str) -> None:
        sf = self._get_spotiflow_detection_params(channel_key=channel_key)
        widgets: list[Any] = [
            getattr(dlg, '_sf_normalize_cb', None),
            getattr(dlg, '_sf_use_optimized', None),
            getattr(dlg, '_sf_perc_low', None),
            getattr(dlg, '_sf_perc_high', None),
            getattr(dlg, '_sf_prob_thresh', None),
            getattr(dlg, '_sf_min_distance', None),
        ]
        for w in widgets:
            if w is None:
                continue
            try:
                w.blockSignals(True)
            except Exception:
                pass
        try:
            if getattr(dlg, '_sf_normalize_cb', None) is not None:
                dlg._sf_normalize_cb.setChecked(bool(sf.get('normalize', True)))
            if getattr(dlg, '_sf_perc_low', None) is not None:
                plow = sf.get('perc_low', 1.0)
                dlg._sf_perc_low.setValue(float(1.0 if plow is None else plow))
            if getattr(dlg, '_sf_perc_high', None) is not None:
                phigh = sf.get('perc_high', 99.8)
                dlg._sf_perc_high.setValue(float(99.8 if phigh is None else phigh))
            if getattr(dlg, '_sf_use_optimized', None) is not None:
                dlg._sf_use_optimized.setChecked(bool(sf.get('use_optimized_threshold', True)))
            if getattr(dlg, '_sf_prob_thresh', None) is not None:
                _pt = sf.get('prob_thresh')
                dlg._sf_prob_thresh.setValue(float(_pt) if _pt is not None else 0.5)
                dlg._sf_prob_thresh.setEnabled(not dlg._sf_use_optimized.isChecked())
            if getattr(dlg, '_sf_min_distance', None) is not None:
                dlg._sf_min_distance.setValue(int(sf.get('min_distance', 1) or 1))
            # Keep pLow/pHigh enabled only when Normalize is on.
            try:
                norm_on = bool(getattr(dlg, '_sf_normalize_cb', None) and dlg._sf_normalize_cb.isChecked())
                for _spin in (getattr(dlg, '_sf_perc_low', None), getattr(dlg, '_sf_perc_high', None)):
                    if _spin is not None:
                        _spin.setEnabled(norm_on)
            except Exception:
                pass
        except Exception:
            get_logger().debug('load spotiflow widgets failed', exc_info=True)
        finally:
            for w in widgets:
                if w is None:
                    continue
                try:
                    w.blockSignals(False)
                except Exception:
                    pass
        try:
            dlg._sf_channel_edit_key = str(channel_key or '').strip() or '_default'
        except Exception:
            pass

    def _rebuild_preview_spotiflow_channel_combo(self, preview_dialog: Any) -> None:
        combo = getattr(preview_dialog, '_sf_channel_combo', None)
        if combo is None:
            return
        from bactomate.gui.filter_preview_utils import foci_channel_key_for_path, short_foci_stack_label
        paths = [Path(p) for p in getattr(preview_dialog, '_foci_source_paths', None) or []]
        self._ensure_spotiflow_channel_entries(paths)
        keep = getattr(preview_dialog, '_sf_channel_edit_key', None)
        combo.blockSignals(True)
        try:
            combo.clear()
            for fp in paths:
                tag = foci_channel_key_for_path(fp)
                combo.addItem(short_foci_stack_label(fp), tag)
                try:
                    combo.setItemData(combo.count() - 1, str(fp.resolve()), Qt.ItemDataRole.ToolTipRole)
                except OSError:
                    combo.setItemData(combo.count() - 1, str(fp), Qt.ItemDataRole.ToolTipRole)
            if keep:
                for ii in range(combo.count()):
                    if combo.itemData(ii) == keep:
                        combo.setCurrentIndex(ii)
                        break
            if combo.count() > 0 and combo.currentIndex() < 0:
                combo.setCurrentIndex(0)
            if combo.count() > 0:
                idx = int(combo.currentIndex())
                key = combo.itemData(idx)
                if isinstance(key, str):
                    self._load_spotiflow_widgets_for_channel(preview_dialog, key)
        finally:
            combo.blockSignals(False)

    def _sync_spotiflow_from_preview_dialog(self, dlg: QWidget) -> None:
        """Persist Spotiflow knobs from the preview into the active foci channel entry."""
        try:
            self._commit_spotiflow_preview_spins(dlg)
            self._save_spotiflow_widgets_to_channel(dlg, self._active_spotiflow_channel_key(dlg))
        except Exception:
            get_logger().debug('sync spotiflow from preview failed', exc_info=True)

    def _commit_spotiflow_preview_spins(self, dlg: Optional[QWidget]=None) -> None:
        """Apply in-progress Spotiflow spinbox text before reading values."""
        target = dlg if dlg is not None else getattr(self, '_embedded_segmentation_preview', None)
        if target is None:
            return
        for name in ('_sf_perc_low', '_sf_perc_high', '_sf_prob_thresh', '_sf_min_distance'):
            w = getattr(target, name, None)
            if w is None:
                continue
            try:
                w.interpretText()
            except Exception:
                pass

    def _get_segmentation_run_params(self, *, selected_files: list[Path] | None=None) -> dict[str, Any]:
        """Return the effective segmentation/foci parameters for the run about to start."""
        display_model = self.model_combo.currentText()
        (model_type, pretrained_path) = self._resolve_model_selection(display_model)
        if pretrained_path is not None:
            model_arg = prepare_user_pretrained_loader_path(pretrained_path)
        else:
            model_arg = model_type or display_model
        (run_diameter, run_flow) = self._get_run_diameter_and_flow()
        params: dict[str, Any] = {'input_path': (self.file_path.text() or '').strip(), 'selected_file_count': len(selected_files or []), 'selected_files': [Path(p).name for p in selected_files or []], 'model': display_model, 'model_arg': str(model_arg), 'diameter_ui': float(self.diameter.value()), 'diameter_effective': float(run_diameter), 'flow_threshold_ui': float(self.flow_threshold.value()), 'flow_threshold_effective': float(run_flow), 'mask_threshold': float(self._get_effective_mask_threshold()), 'batch_size': 32, 'use_gpu': bool(self.use_gpu.isChecked()), 'cluster': bool(self.cluster.isChecked()), 'create_overlay': bool(self.create_overlay.isChecked()), 'detect_foci': bool(getattr(self, 'detect_foci', None) and self.detect_foci.isChecked()), 'spotiflow': self._get_spotiflow_detection_params(), 'spotiflow_channels': dict(getattr(self, '_spotiflow_channel_params', None) or {})}
        return params

    def _log_segmentation_run_params(self, *, selected_files: list[Path] | None=None) -> None:
        params = self._get_segmentation_run_params(selected_files=selected_files)
        lines = ['Segmentation run parameters:']
        for (key, value) in params.items():
            lines.append(f'  {key}: {value}')
        if hasattr(self.parent_window, 'log_output'):
            self.parent_window.log_output('\n'.join(lines) + '\n')
        if getattr(self.parent_window, '_save_session_log', None):
            self.parent_window._save_session_log()

    def reset_segmentation_defaults(self):
        """Reset all segmentation parameters to their default values (model + diameter/flow/mask for that model)."""
        if 'bact_phase_omni' in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText('bact_phase_omni')
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        display = self.model_combo.currentText()
        (model_type, pretrained_path) = self._resolve_model_selection(display)
        if pretrained_path is not None:
            (d, f, m) = USER_MODEL_DEFAULTS
        else:
            (d, f, m) = SEGMENTATION_MODEL_DEFAULTS.get(model_type or display, SEGMENTATION_DEFAULT_FALLBACK)
        self.diameter.setValue(d)
        self.flow_threshold.setValue(f)
        self.mask_threshold.setValue(m)
        self.use_gpu.setChecked(True)
        self.cluster.setChecked(False)
        self.create_overlay.setChecked(True)
        self.assign_persistent_cellids.setChecked(True)
        self.detect_foci.setChecked(False)
        self._spotiflow_detection_params = dict(SPOTIFLOW_DETECTION_DEFAULTS)
        self._spotiflow_channel_params = {}
        self._foci_csv_min_probability = 0.0
        try:
            self._sync_mask_threshold_spinbox_for_model()
        except Exception:
            pass

    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        out = get_from_widgets([('model', self.model_combo), ('diameter', self.diameter), ('flow_threshold', self.flow_threshold), ('mask_threshold', self.mask_threshold), ('use_gpu', self.use_gpu), ('cluster', self.cluster), ('create_overlay', self.create_overlay), ('assign_persistent_cellids', self.assign_persistent_cellids), ('detect_foci', self.detect_foci)])
        spotiflow_params = self._get_spotiflow_detection_params()
        for (key, value) in spotiflow_params.items():
            out[f'spotiflow_{key}'] = value
        chmap = getattr(self, '_spotiflow_channel_params', None)
        if isinstance(chmap, dict) and chmap:
            out['spotiflow_channels'] = dict(chmap)
        out['foci_csv_min_probability'] = float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0)
        out['trained_models_this_session'] = list(self._trained_models_this_session)
        path_txt = (self.file_path.text() or '').strip()
        if path_txt:
            out['file_path'] = path_txt
        if hasattr(self, 'omnipose_train_dir_edit'):
            out_omni = get_from_widgets([('omnipose_train_dir', self.omnipose_train_dir_edit), ('omnipose_img_filter', self.omnipose_img_filter_edit), ('omnipose_mask_filter', self.omnipose_mask_filter_edit), ('omnipose_model_name', self.omnipose_model_name_edit), ('omnipose_pretrained', self.omnipose_pretrained_combo), ('omnipose_n_epochs', self.omnipose_n_epochs_spin), ('omnipose_lr', self.omnipose_lr_spin), ('omnipose_batch', self.omnipose_batch_spin), ('omnipose_tyx_y', self.omnipose_tyx_y_spin), ('omnipose_tyx_x', self.omnipose_tyx_x_spin), ('omnipose_save_each', self.omnipose_save_each_spin)])
            out_omni['omnipose_use_gpu'] = self.omnipose_use_gpu_cb.isChecked()
            out_omni['omnipose_use_tiled'] = self.omnipose_use_tiled_cb.isChecked()
            out.update(out_omni)
        out['performed'] = getattr(self, '_performed_this_session', False)
        out['segment_filename_filter'] = getattr(self, '_segment_filename_filter', '') or ''
        return out

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [('model', self.model_combo), ('diameter', self.diameter), ('flow_threshold', self.flow_threshold), ('mask_threshold', self.mask_threshold), ('use_gpu', self.use_gpu), ('cluster', self.cluster), ('create_overlay', self.create_overlay), ('assign_persistent_cellids', self.assign_persistent_cellids), ('detect_foci', self.detect_foci)])
        if hasattr(self, 'omnipose_train_dir_edit'):
            apply_to_widgets(data, [('omnipose_train_dir', self.omnipose_train_dir_edit), ('omnipose_img_filter', self.omnipose_img_filter_edit), ('omnipose_mask_filter', self.omnipose_mask_filter_edit), ('omnipose_model_name', self.omnipose_model_name_edit), ('omnipose_pretrained', self.omnipose_pretrained_combo), ('omnipose_n_epochs', self.omnipose_n_epochs_spin), ('omnipose_lr', self.omnipose_lr_spin), ('omnipose_batch', self.omnipose_batch_spin), ('omnipose_tyx_y', self.omnipose_tyx_y_spin), ('omnipose_tyx_x', self.omnipose_tyx_x_spin), ('omnipose_save_each', self.omnipose_save_each_spin)])
            if 'omnipose_use_gpu' in data:
                self.omnipose_use_gpu_cb.setChecked(bool(data['omnipose_use_gpu']))
            if 'omnipose_use_tiled' in data:
                self.omnipose_use_tiled_cb.setChecked(bool(data['omnipose_use_tiled']))
        if 'file_path' in data and data.get('file_path'):
            self.file_path.setText(str(data['file_path']).strip())
        if 'performed' in data:
            self._performed_this_session = bool(data['performed'])
        if 'segment_filename_filter' in data and data['segment_filename_filter'] is not None:
            self._segment_filename_filter = str(data['segment_filename_filter']).strip() or ''
        raw_ch = (data or {}).get('spotiflow_channels')
        if isinstance(raw_ch, dict) and raw_ch:
            # Legacy sessions stored full snapshots per channel; keep only per-channel keys.
            # Keys are casefolded; legacy per-sample/truncated keys migrate lazily per foci path.
            from bactomate.gui.filter_preview_utils import normalize_foci_channel_map
            self._spotiflow_channel_params = normalize_foci_channel_map({str(k): self._strip_to_channel_overrides(v) for (k, v) in raw_ch.items() if isinstance(v, dict)})
        if any((isinstance(k, str) and k.startswith('spotiflow_') for k in (data or {}).keys())):
            merged_sf = dict(SPOTIFLOW_DETECTION_DEFAULTS)
            for (k, v) in (data or {}).items():
                if isinstance(k, str) and k.startswith('spotiflow_') and (k != 'spotiflow_channels'):
                    merged_sf[k[len('spotiflow_'):]] = v
            self._spotiflow_detection_params = merged_sf
            if not self._spotiflow_channel_params:
                self._spotiflow_channel_params = {}
        if 'foci_csv_min_probability' in data:
            try:
                self._foci_csv_min_probability = float(data['foci_csv_min_probability'] or 0.0)
            except Exception:
                pass

    def _list_folder_preview_candidates(self) -> list[Path]:
        """TIFF inputs eligible for folder-wide segmentation QC preview."""
        raw = (self.file_path.text() or '').strip()
        if not raw:
            return []
        path = Path(raw)
        if path.is_file() and self._is_segmentation_input_file(path):
            return [path]
        if not path.is_dir():
            return []
        tiff_files = sorted(
            [f for f in list_tiff_files(path, max_files=5000) if self._is_segmentation_input_file(f)],
            key=lambda x: x.name.lower(),
        )
        pred = self._default_segmentation_selection_predicate(tiff_files)
        if pred is None:
            return tiff_files
        preferred = [p for p in tiff_files if pred(p)]
        return preferred or tiff_files

    def _folder_preview_seg_params(self) -> dict[str, Any]:
        self._commit_segmentation_parameter_spins()
        import sys
        is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
        diameter_value = self._get_effective_diameter_pixels()
        display = self.model_combo.currentText()
        (model_type, pretrained_path) = self._resolve_model_selection(display)
        try:
            flow_val = float(self.flow_threshold.value())
        except Exception:
            flow_val = 0.4
        try:
            cluster_on = bool(self.cluster.isChecked())
        except Exception:
            cluster_on = False
        use_gpu = True
        try:
            use_gpu = bool(self.use_gpu.isChecked())
        except Exception:
            pass
        return {
            'diameter': float(diameter_value) if diameter_value and diameter_value > 0 else 0.0,
            'flow_threshold': flow_val,
            'mask_threshold': float(self._get_effective_mask_threshold()),
            'cluster': cluster_on,
            'model_type': model_type,
            'pretrained_model': str(pretrained_path) if pretrained_path else None,
            'use_gpu': use_gpu,
            'is_cellpose_sam': model_type == CELLPOSE_SAM_MODEL,
            'affinity_seg': bool(
                eval_uses_affinity_segmentation(model_type=model_type, pretrained_path=pretrained_path)
            ),
            'is_frozen': bool(is_frozen),
            'detect_foci': bool(getattr(self, 'detect_foci', None) and self.detect_foci.isChecked()),
            'spotiflow': self._get_spotiflow_detection_params(),
            'spotiflow_channels': dict(getattr(self, '_spotiflow_channel_params', None) or {}),
            'foci_csv_min_probability': float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0),
        }

    def _mark_folder_preview_stale_if_needed(self, *_args) -> None:
        ctrl = getattr(self, '_folder_preview_controller', None)
        if ctrl is None or getattr(ctrl, 'grid', None) is None:
            return
        try:
            ctrl.mark_stale_if_params_changed(self._folder_preview_seg_params())
        except Exception:
            pass

    def _load_model_for_folder_preview(self, params: dict[str, Any]):
        """Load Omnipose / Cellpose-SAM model for folder preview (main thread)."""
        import os
        model_type = params.get('model_type')
        pretrained_path = params.get('pretrained_model')
        use_gpu = bool(params.get('use_gpu', True))
        if model_type == CELLPOSE_SAM_MODEL:
            from cellpose import models as cellpose_models
            return cellpose_models.CellposeModel(pretrained_model='cpsam', gpu=use_gpu)
        if pretrained_path and os.path.isfile(str(pretrained_path)):
            return load_cellpose_segmentation_model(
                gpu=use_gpu, loader_path=str(pretrained_path), pretrained_path=str(pretrained_path)
            )
        from bactomate.gui.omnipose_pretrained_cache import ensure_builtin_in_bactomate_cache
        bcp = ensure_builtin_in_bactomate_cache(model_type)
        model_file = str(bcp)
        if is_affinity_builtin_model(model_type):
            return load_bact_phase_affinity_model(gpu=use_gpu, checkpoint_path=model_file)
        return load_cellpose_segmentation_model(
            gpu=use_gpu, loader_path=model_file, model_type=model_type
        )

    def preview_folder_segmentation(self) -> None:
        """Closer-to-batch QC grid for up to 48 folder inputs."""
        from bactomate.gui.folder_preview.constants import folder_preview_enabled
        from bactomate.gui.folder_preview.controller import FolderPreviewController
        from bactomate.gui.folder_preview.seg_compute import compute_segmentation_tile

        candidates = self._list_folder_preview_candidates()
        if not folder_preview_enabled(len(candidates)):
            show_message_dialog(
                self,
                'Preview folder',
                'Need more than one TIFF in the source folder for folder preview.\n'
                'Use Preview segmentation for a single file.',
                kind='warning',
            )
            return
        mount = getattr(self, '_module_preview_mount', None)
        if mount is None:
            show_message_dialog(
                self,
                'Preview folder',
                'Folder preview needs the module right panel (open Segmentation from the main window).',
                kind='warning',
            )
            return
        # Dismiss single-file embedded preview if open.
        prev = getattr(self, '_embedded_segmentation_preview', None)
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass
            self._embedded_segmentation_preview = None
        params = self._folder_preview_seg_params()
        try:
            model = self._load_model_for_folder_preview(params)
        except Exception as e:
            show_message_dialog(self, 'Model load failed', f'Could not load model for folder preview:\n{e}', kind='error')
            return

        spotiflow_model = None
        if bool(params.get('detect_foci')):
            try:
                from bactomate.modules.foci_utils import load_spotiflow_model
                sf0 = dict(params.get('spotiflow') or {})
                spotiflow_model = load_spotiflow_model(
                    model_name=str(sf0.get('model_name') or 'general'),
                    model_path=str(sf0.get('model_path') or '').strip() or None,
                )
            except Exception as e:
                get_logger().warning('Folder preview: Spotiflow model load failed (%s); tiles will show cells only.', e)

        def compute_fn(path: Path, _idx: int):
            return compute_segmentation_tile(
                path,
                model,
                params,
                spotiflow_model=spotiflow_model,
                find_foci_sources=self._find_foci_sources_for_input,
            )

        ctrl = getattr(self, '_folder_preview_controller', None)
        if ctrl is None:
            ctrl = FolderPreviewController(
                self,
                preview_surface=getattr(self, '_preview_surface', None),
                mount=mount,
                on_tile_open=None,
                parent=self,
            )
            self._folder_preview_controller = ctrl
        try:
            ctrl.start(candidates, params, compute_fn)
        except Exception as e:
            show_message_dialog(self, 'Preview folder', f'Could not start folder preview:\n{e}', kind='error')

    def preview_segmentation(self):
        """Preview segmentation on a single frame or file from folder"""
        # If a folder grid is open, close it before single-file preview.
        ctrl = getattr(self, '_folder_preview_controller', None)
        if ctrl is not None and getattr(ctrl, 'grid', None) is not None:
            try:
                ctrl.close()
            except Exception:
                pass
            self._folder_preview_controller = None
        if not self.file_path.text():
            show_message_dialog(self, 'No Selection', 'Please select a TIFF file or folder first.', kind='warning')
            return
        if not Path(self.file_path.text()).exists():
            show_message_dialog(self, 'Not Found', 'The selected file or folder does not exist.', kind='warning')
            return
        reopen_file = (getattr(self, '_preview_reopen_file', None) or '').strip()
        if reopen_file:
            self._folder_preview_pick = reopen_file
        switching_input = bool(getattr(self, '_preview_switching_input_file', False)) or bool(reopen_file)
        self._preview_switching_input_file = False
        is_folder = self._is_folder_mode()
        folder_preview_files: list[Path] = []
        if is_folder:
            folder = Path(self.file_path.text())
            tiff_files = sorted([f for f in list_tiff_files(folder, max_files=5000) if self._is_segmentation_input_file(f)], key=lambda x: x.name.lower())
            folder_preview_files = list(tiff_files)
            if not tiff_files:
                show_message_dialog(self, 'No TIFF Files Found', 'No .tif / .tiff files found under the selected folder (recursive scan).\n\nChoose a folder that contains TIFF inputs for segmentation.', kind='warning')
                return
            file_path: str
            pick = (getattr(self, '_folder_preview_pick', None) or '').strip()
            chosen: Optional[Path] = None
            if pick:
                try:
                    pp = Path(pick)
                    if pp.is_file() and self._is_segmentation_input_file(pp):
                        rp = pp.resolve()
                        fr = folder.resolve()
                        if fr in rp.parents or rp.parent == fr:
                            chosen = pp
                except OSError:
                    chosen = None
            if not switching_input and len(tiff_files) > 1:
                chosen = None
            if chosen is None and switching_input and pick:
                try:
                    pp = Path(pick)
                    if pp.is_file():
                        chosen = pp
                except OSError:
                    pass
            if chosen is None and len(tiff_files) == 1:
                chosen = tiff_files[0]
            if chosen is None and switching_input:
                pred = self._default_segmentation_selection_predicate(tiff_files)
                if pred is not None:
                    preferred = [f for f in tiff_files if pred(f)]
                    if len(preferred) == 1:
                        chosen = preferred[0]
            if chosen is not None:
                file_path = str(chosen)
                self._folder_preview_pick = file_path
            elif switching_input and pick:
                file_path = str(Path(pick))
                self._folder_preview_pick = file_path
            else:
                file_dialog = QDialog(self)
                file_dialog.setWindowTitle('Select File for Preview')
                file_dialog.setMinimumSize(500, 400)
                layout = QVBoxLayout()
                label = QLabel('Select a file to preview segmentation:')
                label.setStyleSheet('color: #ffffff; font-size: 11pt; padding: 10px;')
                layout.addWidget(label)
                file_list = QListWidget()
                file_list.setStyleSheet('QListWidget {background-color: #2b2b2b;color: #ffffff;font-size: 11pt;border: 1px solid #555555;border-radius: 4px;padding: 5px;}QListWidget::item {padding: 8px;border-bottom: 1px solid #3a3a3a;}QListWidget::item:selected {background-color: #5D6D7E;color: #ffffff;}QListWidget::item:hover {background-color: #3a3a3a;}')
                for f in tiff_files:
                    file_list.addItem(f.name)
                if file_list.count() > 0:
                    selected_row = 0
                    if pick:
                        try:
                            pick_resolved = str(Path(pick).resolve())
                            for (row_i, f) in enumerate(tiff_files):
                                if str(Path(f).resolve()) == pick_resolved:
                                    selected_row = row_i
                                    break
                        except OSError:
                            pass
                    file_list.setCurrentRow(selected_row)

                def on_item_double_clicked(item):
                    file_dialog.accept()
                file_list.itemDoubleClicked.connect(on_item_double_clicked)
                layout.addWidget(file_list)
                button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                style_dialog_button_box(button_box)
                button_box.accepted.connect(file_dialog.accept)
                button_box.rejected.connect(file_dialog.reject)
                layout.addWidget(button_box)
                file_dialog.setLayout(layout)
                if file_dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                selected_item = file_list.currentItem()
                if not selected_item:
                    return
                selected_name = selected_item.text()
                selected_file = None
                for f in tiff_files:
                    if f.name == selected_name:
                        selected_file = f
                        break
                if not selected_file:
                    return
                file_path = str(selected_file)
                self._folder_preview_pick = file_path
        else:
            file_path = self.file_path.text()
            self._folder_preview_pick = file_path
        prev_embed = getattr(self, '_embedded_segmentation_preview', None)
        if prev_embed is not None and (not switching_input) and (getattr(self, '_module_preview_mount', None) is not None):
            try:
                cur_inp = getattr(prev_embed, '_segmentation_input_path', None)
                if cur_inp and Path(cur_inp).resolve() == Path(file_path).resolve():
                    pw = getattr(self, 'parent_window', None)
                    if pw is not None and hasattr(pw, 'log_output'):
                        pw.log_output('Preview is already open for this TIFF. Use the TIFF dropdown in the right pane to switch files in the folder, or «Re-detect» after changing diameter / flow / mask.\n')
                    return
            except OSError:
                pass
        self._set_segmentation_buttons_busy(True, action='preview', progress_label='Loading…')
        self._update_segmentation_button_progress(5, 'Preparing preview')
        loading_progress = QProgressDialog('Loading file and initializing preview...', None, 0, 0, self)
        loading_progress.setWindowTitle('Loading')
        loading_progress.setAutoClose(False)
        loading_progress.setAutoReset(False)
        loading_progress.setCancelButton(None)
        loading_progress.show()
        QApplication.processEvents()
        try:
            loading_progress.setLabelText('Loading TIFF file...')
            self._update_segmentation_button_progress(12, 'Loading TIFF')
            QApplication.processEvents()
            import tifffile
            from bactomate.gui.image_utils import to_qpixmap, build_segmentation_overlay, scale_for_preview, draw_scale_bar, downsample_for_preview_seg, upsample_mask_to_shape, PREVIEW_SEG_MAX_SIZE
            from bactomate.gui.image_canvas import ImageCanvas
            try:
                import omnipose.core
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                if is_frozen:
                    _patch_omnipose_get_masks()
                if not hasattr(omnipose.core, 'OMNI_MODELS'):
                    OMNI_MODELS = []
                    if hasattr(omnipose.core, 'C1_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_MODELS)
                    if hasattr(omnipose.core, 'C2_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_MODELS)
                    if hasattr(omnipose.core, 'C1_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_BD_MODELS)
                    if hasattr(omnipose.core, 'C2_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_BD_MODELS)
                    seen = set()
                    OMNI_MODELS = [x for x in OMNI_MODELS if not (x in seen or seen.add(x))]
                    if not OMNI_MODELS:
                        OMNI_MODELS = ['bact_omni', 'bact_phase_omni', 'bact_fluor_omni']
                    omnipose.core.OMNI_MODELS = OMNI_MODELS
            except Exception:
                get_logger().debug('Omnipose get_masks patch failed, using fallback', exc_info=True)
            from cellpose_omni import models
            from cellpose_omni.io import imread
            import numpy as np
            images = safe_imread(file_path)
            loading_progress.setLabelText('Processing image data...')
            self._update_segmentation_button_progress(28, 'Processing image')
            QApplication.processEvents()
            if images.dtype != np.float32:
                images = images.astype(np.float32)
            if images.ndim == 2:
                images = images[np.newaxis, ...]
            elif images.ndim == 3:
                if images.shape[-1] < 10:
                    images = images.transpose(2, 0, 1)
                    images = images[np.newaxis, ...]
            elif images.ndim == 4:
                if images.shape[-1] < 10:
                    images = images.transpose(0, 3, 1, 2)
                elif images.shape[0] < 10:
                    images = images.transpose(1, 0, 2, 3)
            if images.size == 0 or images.shape[0] == 0:
                loading_progress.close()
                self._set_segmentation_buttons_busy(False)
                show_message_dialog(self, 'Empty File', 'The TIFF file appears to be empty or has no frames.', kind='error')
                return
            num_frames = images.shape[0]
            main_win = self.window() if hasattr(self, 'window') and callable(self.window) else None
            mount = getattr(self, '_module_preview_mount', None)
            dlg_parent = mount if mount is not None else main_win
            embed_seg = mount is not None
            from bactomate.gui.preview.preview_window import PreviewDialogBase, WorkerGuard
            preview_dialog = PreviewDialogBase(dlg_parent, title='Segmentation Preview', primary_action_label='Re-detect', primary_action_callback=None, render_callback=None, worker_guard=None, minimum_size=(900, 700), embedded=embed_seg, show_close_button=not embed_seg)
            if embed_seg:
                try:
                    style_path_row_browse_button(preview_dialog.primary_button)
                except Exception:
                    pass
            layout = preview_dialog.layout()
            title_label = preview_dialog.title_label
            if embed_seg:
                try:
                    title_label.setWordWrap(True)
                    title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                except Exception:
                    pass
            preview_dialog._foci_sources = []
            preview_dialog._foci_source_paths = []
            preview_dialog._foci_peaks_cache = {}
            preview_dialog._foci_peaks_for_current = []
            preview_dialog._foci_peaks_request_id = 0
            preview_dialog._foci_worker = None
            preview_dialog._foci_worker_slot = None
            preview_dialog._foci_preview_hint = ''
            try:
                preview_dialog._segmentation_input_path = str(Path(file_path).resolve())
            except OSError:
                preview_dialog._segmentation_input_path = str(file_path)
            try:
                foci_sources = self._find_foci_sources_for_input(Path(file_path))
                if foci_sources:
                    from bactomate.modules.image_stack_utils import coerce_stack_to_tyx
                    loaded = []
                    loaded_paths = []
                    for fp in foci_sources:
                        try:
                            arr = coerce_stack_to_tyx(np.asarray(safe_imread(str(fp))))
                            if arr is None or arr.size == 0:
                                continue
                            loaded.append(arr)
                            loaded_paths.append(fp)
                        except Exception:
                            continue
                    preview_dialog._foci_sources = loaded
                    preview_dialog._foci_source_paths = loaded_paths
                    self._ensure_spotiflow_channel_entries(loaded_paths)
                detect_cb = getattr(self, 'detect_foci', None)
                if not preview_dialog._foci_source_paths:
                    preview_dialog._foci_preview_hint = f"No *_foci*.tif matching '{Path(file_path).stem}' (or its ND2 stem) found next to the input. Expected: {Path(file_path).stem}_foci*.tif or <nd2_stem>_foci_<channel>.tif."
                elif detect_cb is None or not detect_cb.isChecked():
                    preview_dialog._foci_preview_hint = 'Detect foci is off — peaks are not computed (foci stacks are still listed as backgrounds).'
            except Exception:
                preview_dialog._foci_sources = []
                preview_dialog._foci_source_paths = []
            image_canvas = preview_dialog.canvas
            lut_widget = preview_dialog.lut_widget

            def _mask_matching_display(mask, display_shape):
                """Return a mask with the same Y/X shape as the display frame."""
                arr = np.asarray(mask)
                target = tuple(int(v) for v in display_shape[:2])
                if arr.shape[:2] == target:
                    return arr
                if arr.ndim >= 2 and arr.shape[0] == target[1] and arr.shape[1] == target[0]:
                    get_logger().warning('Preview mask/display axes were swapped; transposing mask %s -> %s', arr.shape[:2], target)
                    return np.ascontiguousarray(arr.T)
                get_logger().warning('Ignoring stale preview mask with shape %s for display shape %s', arr.shape[:2], target)
                return np.zeros(target, dtype=arr.dtype if arr.dtype.kind in {'i', 'u'} else np.int32)

            def refresh_image_display() -> None:
                """Redraw overlay using current display_frame, mask, and shared histogram LUT."""
                display_frame = getattr(preview_dialog, 'current_display_frame', None)
                mask = getattr(preview_dialog, 'masks', None)
                if display_frame is None or mask is None:
                    return
                display_shape = tuple(np.asarray(display_frame).shape[:2])
                mask = _mask_matching_display(mask, display_shape)
                preview_dialog.masks = mask
                (vmin, vmax) = lut_widget.current_lut()
                if vmin is None or vmax is None or vmax <= vmin:
                    try:
                        fmin = float(np.nanmin(display_frame))
                        fmax = float(np.nanmax(display_frame))
                    except Exception:
                        (fmin, fmax) = (0.0, 255.0)
                    vmin = fmin
                    vmax = fmax if fmax > fmin else fmin + 1.0
                (vmin, vmax) = (float(vmin), float(vmax))
                vrange = max(vmax - vmin, 1e-06)
                display_uint8 = ((np.asarray(display_frame).astype(np.float64) - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
                (disp_scaled, mask_scaled) = scale_for_preview(display_uint8, mask, max_size=PREVIEW_SEG_MAX_SIZE)
                if bool(preview_dialog.state().extra.get('show_cells', True)):
                    overlay = build_segmentation_overlay(disp_scaled, mask_scaled, alpha=0.5)
                else:
                    overlay = np.stack([disp_scaled, disp_scaled, disp_scaled], axis=-1).astype(np.uint8)
                try:
                    _detect_on = bool(getattr(getattr(self, 'detect_foci', None), 'isChecked', lambda : False)())
                    if not _detect_on:
                        raise RuntimeError('foci off')
                    sources = getattr(preview_dialog, '_foci_sources', None) or []
                    paths = getattr(preview_dialog, '_foci_source_paths', None) or []
                    n_src = min(len(sources), len(paths))
                    if n_src <= 0:
                        raise RuntimeError('no foci sources')
                    visible: list[bool] = []
                    _nav = getattr(preview_dialog, '_nav_ctrl', None)
                    if _nav is not None and hasattr(_nav, 'foci_channel_overlay_visible'):
                        visible = list(_nav.foci_channel_overlay_visible() or [])
                    if not visible and (bool(preview_dialog.state().extra.get('show_foci', False)) or _detect_on):
                        visible = [True] * n_src
                    cache = getattr(preview_dialog, '_foci_peaks_cache', None)
                    if not isinstance(cache, dict):
                        cache = {}
                    frame_i = int(getattr(preview_dialog, 'current_frame', 0) or 0)
                    (h0, w0) = display_uint8.shape[:2]
                    (hs, ws) = (overlay.shape[0], overlay.shape[1])
                    sx = ws / w0 if w0 else 1.0
                    sy = hs / h0 if h0 else 1.0
                    _csv_minp = float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0)
                    for si in range(n_src):
                        if visible and si < len(visible) and (not visible[si]):
                            continue
                        ch_key = self._spotiflow_channel_key_for_preview_index(preview_dialog, si)
                        sf_si = self._get_spotiflow_detection_params(channel_key=ch_key)
                        _sf_opt = bool(sf_si.get('use_optimized_threshold', True))
                        _sf_prob = float(sf_si.get('prob_thresh') if sf_si.get('prob_thresh') is not None else 0.5)
                        _sf_min_dist = int(sf_si.get('min_distance', 1) or 1)
                        _sf_subpix = bool(sf_si.get('subpix', True))
                        _sf_norm = bool(sf_si.get('normalize', True))
                        _sf_pl = float(sf_si.get('perc_low', 1.0) or 1.0)
                        _sf_ph = float(sf_si.get('perc_high', 99.8) or 99.8)
                        key = (si, frame_i, ch_key, _sf_opt, _sf_prob, _sf_min_dist, _sf_subpix, _sf_norm, _sf_pl, _sf_ph, _csv_minp)
                        peaks = list(cache.get(key) or [])
                        if not peaks and si == 0:
                            peaks = list(getattr(preview_dialog, '_foci_peaks_for_current', None) or [])
                        rgb = FOCI_PREVIEW_RING_RGB[si % len(FOCI_PREVIEW_RING_RGB)]
                        for p in peaks:
                            yy = int(round(float(p.get('y', 0)) * sy))
                            xx = int(round(float(p.get('x', 0)) * sx))
                            if 0 <= yy < hs and 0 <= xx < ws:
                                r = 4
                                for dy in range(-r, r + 1):
                                    for dx in range(-r, r + 1):
                                        d2 = dx * dx + dy * dy
                                        if d2 < (r - 1) * (r - 1) or d2 > r * r:
                                            continue
                                        y2 = yy + dy
                                        x2 = xx + dx
                                        if 0 <= y2 < hs and 0 <= x2 < ws:
                                            overlay[y2, x2, 0] = int(rgb[0])
                                            overlay[y2, x2, 1] = int(rgb[1])
                                            overlay[y2, x2, 2] = int(rgb[2])
                except Exception:
                    pass
                pixmap = to_qpixmap(overlay)
                (h, w) = overlay.shape[:2]
                image_canvas.set_pixmap(pixmap, (w, h), reset_view=False)
            try:
                lut_widget.lut_changed.connect(lambda *_: refresh_image_display())
            except Exception:
                pass

            class FociPeaksPreviewWorker(QThread):
                """Spotiflow foci peak detection off the GUI thread (preview only)."""
                finished = pyqtSignal(int, object, bool, object)

                def __init__(self, request_id: int, tab: 'OmniposeSegTab', jobs: list[dict[str, Any]], parent=None):
                    super().__init__(parent)
                    self.setObjectName('FociPeaksPreviewWorker')
                    self._request_id = int(request_id)
                    self._tab = tab
                    self._jobs = list(jobs)

                def run(self) -> None:
                    cache_updates: dict[tuple, list] = {}
                    spotiflow_available = True
                    status_parts: list[str] = []
                    for job in self._jobs:
                        if self.isInterruptionRequested():
                            self.finished.emit(self._request_id, cache_updates, False, status_parts)
                            return
                        (peaks_raw, avail) = self._tab._spotiflow_detect_peaks_on_plane(job['plane'], job['sf'])
                        if not avail or peaks_raw is None:
                            spotiflow_available = False
                            break
                        peaks = self._tab._filter_foci_peaks_by_csv_threshold(list(peaks_raw), float(job['csv_minp']))
                        cache_updates[job['key']] = peaks
                        if job.get('status_label'):
                            status_parts.append(f"{job['status_label']}: {len(peaks)}")
                    self.finished.emit(self._request_id, cache_updates, spotiflow_available, status_parts)

            def _stop_foci_peaks_worker() -> None:
                worker = getattr(preview_dialog, '_foci_worker', None)
                slot = getattr(preview_dialog, '_foci_worker_slot', None)
                if worker is not None and slot is not None:
                    try:
                        worker.finished.disconnect(slot)
                    except Exception:
                        pass
                if worker is not None:
                    try:
                        if worker.isRunning():
                            worker.requestInterruption()
                            worker.quit()
                    except Exception:
                        pass
                    try:
                        worker.deleteLater()
                    except Exception:
                        pass
                preview_dialog._foci_worker = None
                preview_dialog._foci_worker_slot = None

            def _csv_minp_note() -> str:
                m = float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0)
                return f' (raw CSV min probability {m:g} applied)' if m > 0 else ''

            def _foci_status(text: str) -> None:
                """Show foci feedback in the Spotiflow section (visible in embedded mode) and the toolbar label."""
                lbl = getattr(preview_dialog, '_sf_foci_status_label', None)
                if lbl is not None:
                    try:
                        lbl.setText(text)
                        lbl.setVisible(bool(text))
                    except Exception:
                        pass
                try:
                    status_label.setText(text)
                except Exception:
                    pass

            def _finalize_foci_peaks_status(src_idx: int, frame_idx: int, *, spotiflow_available: bool, status_parts: list[str]) -> None:
                sf_cur = self._get_spotiflow_detection_params(channel_key=self._spotiflow_channel_key_for_preview_index(preview_dialog, src_idx))
                ch_key = self._spotiflow_channel_key_for_preview_index(preview_dialog, src_idx)
                csv_minp = float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0)
                cur_key = self._foci_peak_cache_key(src_idx, frame_idx, ch_key, sf_cur, csv_minp)
                cache = getattr(preview_dialog, '_foci_peaks_cache', None) or {}
                preview_dialog._foci_peaks_for_current = list(cache.get(cur_key) or [])
                if not spotiflow_available:
                    _foci_status(
                        'Foci: Spotiflow unavailable — check logs '
                        '(often wandb/protobuf mismatch on Windows). No peaks drawn.'
                    )
                elif status_parts:
                    _foci_status('Foci — ' + ', '.join(status_parts) + _csv_minp_note())
                else:
                    _foci_status(f'Foci: {len(preview_dialog._foci_peaks_for_current)} peak(s) on current frame.' + _csv_minp_note())
                refresh_image_display()

            def _compute_foci_peaks_for_current(frame_idx: int, *, allow_detect: bool=True, force_recalc: bool=False, background: bool=False) -> None:
                """Compute/cached foci peaks for the selected foci source + frame (preview only).

                Communicates the outcome through ``status_label`` so the user can
                distinguish the three "no markers drawn" cases that previously
                looked identical:

                1. No matching ``*_foci.tif`` next to the input.
                2. ``Detect foci`` is unchecked (Spotiflow intentionally skipped).
                3. Spotiflow not installed / raised during inference
                   (``detect_peaks_spotiflow`` returns ``None`` with
                   ``raise_on_error=False``).

                Case 4 — detection ran and found zero peaks — is also reported
                so the user knows the pipeline executed.

                When ``allow_detect=False`` and the cached peaks for the current
                parameter key are missing, the previously displayed markers are
                preserved so a parameter tweak does not visually wipe the
                overlay; the status line indicates the cache is stale and the
                user must click ``Re-detect`` to refresh.

                Set ``background=True`` to run Spotiflow off the GUI thread (used when
                enabling Detect foci so the checkbox responds immediately).
                """
                combo = getattr(preview_dialog, '_sf_channel_combo', None) or getattr(preview_dialog, '_foci_source_combo', None)
                sources = getattr(preview_dialog, '_foci_sources', None) or []

                _set_status = _foci_status

                def _clear_peaks() -> None:
                    preview_dialog._foci_peaks_for_current = []
                detect_cb = getattr(self, 'detect_foci', None)
                detect_on = detect_cb is not None and bool(detect_cb.isChecked())
                if not sources:
                    _clear_peaks()
                    if detect_on:
                        hint = getattr(preview_dialog, '_foci_preview_hint', '') or ''
                        if hint:
                            _set_status(hint)
                    return
                if not detect_on:
                    _clear_peaks()
                    if allow_detect and force_recalc:
                        _set_status('Foci: Detect foci is off — enable it on the left to run peak detection.')
                    return
                from bactomate.gui.filter_preview_utils import short_foci_stack_label
                from bactomate.modules.image_stack_utils import coerce_plane_to_2d
                paths = getattr(preview_dialog, '_foci_source_paths', None) or []
                src_idx = 0
                if combo is not None:
                    src_idx = int(combo.currentIndex())
                    if src_idx < 0 or src_idx >= len(sources):
                        src_idx = 0
                self._sync_spotiflow_from_preview_dialog(preview_dialog)
                _csv_minp = float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0)
                cache = getattr(preview_dialog, '_foci_peaks_cache', None)
                if not isinstance(cache, dict):
                    preview_dialog._foci_peaks_cache = {}
                    cache = preview_dialog._foci_peaks_cache

                def _cache_key(si: int, sf_local: dict[str, Any]) -> tuple:
                    ch_key = self._spotiflow_channel_key_for_preview_index(preview_dialog, si)
                    return self._foci_peak_cache_key(si, frame_idx, ch_key, sf_local, _csv_minp)
                pending_jobs: list[dict[str, Any]] = []
                for si in range(len(sources)):
                    ch_key = self._spotiflow_channel_key_for_preview_index(preview_dialog, si)
                    sf = self._get_spotiflow_detection_params(channel_key=ch_key)
                    key = _cache_key(si, sf)
                    if key in cache and (not force_recalc):
                        continue
                    if not allow_detect:
                        continue
                    arr = sources[si]
                    fi = max(0, min(int(frame_idx), int(arr.shape[0]) - 1))
                    plane = np.asarray(coerce_plane_to_2d(np.asarray(arr[fi])), dtype=np.float64)
                    lbl = ''
                    if si < len(paths):
                        lbl = short_foci_stack_label(Path(paths[si]))
                    pending_jobs.append({'key': key, 'plane': plane, 'sf': dict(sf), 'csv_minp': _csv_minp, 'status_label': lbl})
                if background and pending_jobs and allow_detect:
                    preview_dialog._foci_peaks_request_id = int(getattr(preview_dialog, '_foci_peaks_request_id', 0) or 0) + 1
                    request_id = int(preview_dialog._foci_peaks_request_id)
                    n_jobs = len(pending_jobs)
                    _set_status(f"Detecting foci… ({n_jobs} channel{('s' if n_jobs != 1 else '')})")
                    _stop_foci_peaks_worker()

                    def _on_foci_worker_finished(rid, cache_updates, spotiflow_ok, status_parts):
                        if int(rid) != int(getattr(preview_dialog, '_foci_peaks_request_id', 0) or 0):
                            return
                        if getattr(preview_dialog, '_dialog_closed', False):
                            return
                        try:
                            for (k, peaks) in (cache_updates or {}).items():
                                cache[k] = list(peaks)
                        except Exception:
                            pass
                        full_status: list[str] = []
                        try:
                            for si in range(len(sources)):
                                ch_key = self._spotiflow_channel_key_for_preview_index(preview_dialog, si)
                                sf_i = self._get_spotiflow_detection_params(channel_key=ch_key)
                                k_i = _cache_key(si, sf_i)
                                n_peaks = len(list(cache.get(k_i) or []))
                                if si < len(paths):
                                    lbl_i = short_foci_stack_label(Path(paths[si]))
                                    full_status.append(f'{lbl_i}: {n_peaks}')
                        except Exception:
                            full_status = list(status_parts or [])
                        _finalize_foci_peaks_status(src_idx, frame_idx, spotiflow_available=bool(spotiflow_ok), status_parts=full_status or list(status_parts or []))
                        preview_dialog._foci_worker = None
                        preview_dialog._foci_worker_slot = None
                    worker = FociPeaksPreviewWorker(request_id, self, pending_jobs, parent=preview_dialog)
                    preview_dialog._foci_worker_slot = _on_foci_worker_finished
                    worker.finished.connect(_on_foci_worker_finished)
                    preview_dialog._foci_worker = worker
                    worker.start()
                    return
                spotiflow_available = True
                status_parts: list[str] = []
                for si in range(len(sources)):
                    ch_key = self._spotiflow_channel_key_for_preview_index(preview_dialog, si)
                    sf = self._get_spotiflow_detection_params(channel_key=ch_key)
                    key = _cache_key(si, sf)
                    peaks: list
                    if key in cache and (not force_recalc):
                        peaks = list(cache.get(key) or [])
                    else:
                        if not allow_detect:
                            continue
                        arr = sources[si]
                        fi = max(0, min(int(frame_idx), int(arr.shape[0]) - 1))
                        plane = np.asarray(coerce_plane_to_2d(np.asarray(arr[fi])), dtype=np.float64)
                        (peaks_raw, avail) = self._spotiflow_detect_peaks_on_plane(plane, sf)
                        if not avail or peaks_raw is None:
                            spotiflow_available = False
                            peaks = []
                            break
                        peaks = list(peaks_raw)
                        if spotiflow_available:
                            cache[key] = list(peaks)
                    peaks = self._filter_foci_peaks_by_csv_threshold(list(peaks), _csv_minp)
                    cache[key] = peaks
                    if si < len(paths):
                        lbl = short_foci_stack_label(Path(paths[si]))
                        status_parts.append(f'{lbl}: {len(peaks)}')
                if not allow_detect and (not force_recalc):
                    sf_cur = self._get_spotiflow_detection_params(channel_key=self._spotiflow_channel_key_for_preview_index(preview_dialog, src_idx))
                    cur_key = _cache_key(src_idx, sf_cur)
                    cached_now = list(cache.get(cur_key) or [])
                    preview_dialog._foci_peaks_for_current = cached_now
                    act_lbl = short_foci_stack_label(Path(paths[src_idx])) if 0 <= src_idx < len(paths) else self._active_spotiflow_channel_key(preview_dialog)
                    if cur_key in cache:
                        _set_status(f'Foci “{act_lbl}”: {len(cached_now)} peak(s) on current frame.' + _csv_minp_note())
                    else:
                        _set_status(f'Spotiflow settings for “{act_lbl}” changed. Rings hidden until Re-detect; other channels unchanged.')
                    refresh_image_display()
                    return
                _finalize_foci_peaks_status(src_idx, frame_idx, spotiflow_available=spotiflow_available, status_parts=status_parts)
            preview_dialog.images = images
            preview_dialog.current_frame = 0
            preview_dialog.masks = None
            preview_dialog.model = None
            status_label = QLabel('Ready')
            status_label.setStyleSheet('color: #7F8C8D; font-size: 9pt; font-style: italic;')
            if embed_seg:
                status_label.hide()  # foci feedback is shown in preview_dialog._sf_foci_status_label instead

            def find_model_file(model_type, model_index=0, use_torch=True):
                """
                Find model file: BactoMate omnipose_pretrained mirror first, then ~/.cellpose/models.
                """
                from pathlib import Path
                from bactomate.gui.omnipose_pretrained_cache import try_get_bactomate_cached_builtin
                from cellpose_omni.models import MODEL_DIR
                b = try_get_bactomate_cached_builtin(model_type, model_index)
                if b is not None:
                    return str(b)
                torch_str = 'torch' if use_torch else ''
                basename = f'{model_type}{torch_str}_{model_index}'
                cache_model_file = Path(MODEL_DIR) / basename
                if cache_model_file.is_file():
                    return str(cache_model_file)
                return None

            def load_builtin_cellpose_from_find_cache(mt: str) -> bool:
                """Load built-in weights from BactoMate or cellpose cache via pretrained_model path."""
                mf = find_model_file(mt, 0, True)
                if not mf or not os.path.isfile(mf):
                    return False
                if is_affinity_builtin_model(mt):
                    preview_dialog.model = load_bact_phase_affinity_model(gpu=current_use_gpu, checkpoint_path=mf)
                else:
                    preview_dialog.model = load_cellpose_segmentation_model(gpu=current_use_gpu, loader_path=mf, model_type=mt)
                return True

            def _model_loader_family(model_type: Optional[str], pretrained_path: Optional[str]) -> str:
                if model_type == CELLPOSE_SAM_MODEL:
                    return 'cellpose_sam'
                if pretrained_path:
                    return 'user'
                return 'omnipose'

            def load_model():
                """Load Omnipose model (only reloads if model type or GPU setting changed)."""
                params = get_current_parameters()
                current_model_type = params.get('model_type')
                pretrained_path = params.get('pretrained_model')
                current_use_gpu = params['use_gpu']
                current_key = pretrained_path or current_model_type or ''
                family = _model_loader_family(current_model_type, pretrained_path)
                family_changed = getattr(preview_dialog, '_last_model_family', None) is not None and getattr(preview_dialog, '_last_model_family', None) != family
                needs_reload = preview_dialog.model is None or family_changed or getattr(preview_dialog, '_last_model_key', None) != current_key or (getattr(preview_dialog, '_last_use_gpu', None) != current_use_gpu)
                if needs_reload:
                    preview_dialog.model = None
                    preview_dialog.segmentation_cache.clear()
                preview_dialog._last_model_family = family

                def set_loaded_status():
                    """Set status label to show whether GPU is actually in use (CUDA / MPS / CPU)."""
                    import torch
                    if not current_use_gpu:
                        status_label.setText('Model loaded (CPU – Use GPU unchecked)')
                        return
                    if torch.cuda.is_available():
                        try:
                            name = torch.cuda.get_device_name(0) or 'CUDA'
                        except Exception:
                            name = 'CUDA'
                        status_label.setText(f'Model loaded (GPU: {name})')
                    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                        status_label.setText('Model loaded (GPU: Apple MPS)')
                    else:
                        status_label.setText('Model loaded (CPU – GPU requested but not available)')
                if needs_reload:
                    try:
                        import os
                        from urllib.error import HTTPError
                        if current_model_type == CELLPOSE_SAM_MODEL:
                            status_label.setText('Loading Cellpose-SAM...')
                            QApplication.processEvents()
                            try:
                                from cellpose import models as cellpose_models
                                preview_dialog.model = cellpose_models.CellposeModel(pretrained_model='cpsam', gpu=current_use_gpu)
                            except ImportError as e:
                                status_label.setText('Cellpose-SAM requires: pip install cellpose')
                                get_logger().exception('Cellpose-SAM preview: cellpose not installed')
                                return False
                            except Exception as e:
                                status_label.setText('Cellpose-SAM load failed: %s' % (str(e)[:60],))
                                get_logger().exception('Cellpose-SAM preview load failed')
                                return False
                            preview_dialog._last_model_key = current_key
                            preview_dialog._last_use_gpu = current_use_gpu
                            set_loaded_status()
                            return True
                        if pretrained_path and os.path.isfile(pretrained_path):
                            status_label.setText('Loading user model...')
                            QApplication.processEvents()
                            try:
                                inspect_info = inspect_omnipose_checkpoint(pretrained_path)
                                ckpt_nout = inspect_info.get('output_layer_out_channels')
                                append_omnipose_debug(f"[BactoMate USER_MODEL pre-load] path={pretrained_path!r} inspect: nchan={inspect_info.get('nchan')} nclasses={inspect_info.get('nclasses')} ckpt_output_channels={ckpt_nout} first_layer_shape={inspect_info.get('first_layer_shape')} keys={inspect_info.get('keys', [])[:15]!r}")
                            except Exception:
                                pass
                            preview_dialog.model = load_cellpose_segmentation_model(gpu=current_use_gpu, loader_path=pretrained_path, pretrained_path=pretrained_path)
                            try:
                                cp = getattr(preview_dialog.model, 'cp', None)
                                nchan = getattr(cp, 'nchan', None)
                                nclasses = getattr(cp, 'nclasses', None)
                                append_omnipose_debug(f'[BactoMate USER_MODEL post-load] model.cp.nchan={nchan} model.cp.nclasses={nclasses}')
                                if inspect_info.get('output_layer_out_channels') is not None and nclasses != inspect_info.get('output_layer_out_channels'):
                                    append_omnipose_debug(f"[BactoMate USER_MODEL] MISMATCH: checkpoint output head has {inspect_info['output_layer_out_channels']} channels but model.cp.nclasses={nclasses}. This can cause crash or 0 cells. See docs/CUSTOM_MODEL_TROUBLESHOOTING.md or run: python -m bactomate.scripts.inspect_omnipose_checkpoint <path>")
                                get_logger().info('User model loaded: nchan=%s nclasses=%s', nchan, nclasses)
                                if getattr(preview_dialog.model, 'pretrained_size', None) is None:
                                    append_omnipose_debug('[BactoMate USER_MODEL] size model not loaded (pretrained_size is None). diameter=0 will not rescale -> often 0 cells. Run from project root: venv/bin/python3 setup/apply_all_patches.py then restart BactoMate.')
                            except Exception:
                                pass
                            preview_dialog._last_model_key = current_key
                            preview_dialog._last_use_gpu = current_use_gpu
                            set_loaded_status()
                            return True
                        from bactomate.gui.omnipose_pretrained_cache import ensure_builtin_in_bactomate_cache
                        status_label.setText('Loading model (BactoMate cache / download if needed)...')
                        QApplication.processEvents()
                        try:
                            bcp = ensure_builtin_in_bactomate_cache(current_model_type)
                            model_file = str(bcp)
                        except Exception as download_err:
                            get_logger().exception('Built-in Omnipose model resolve failed')
                            show_message_dialog(preview_dialog, 'Model load failed', f"Could not load model '{current_model_type}':\n{download_err}\n\nCheck your connection or try another built-in model.", kind='warning')
                            status_label.setText('Model load failed')
                            return False
                        if not model_file or not os.path.exists(model_file):
                            raise FileNotFoundError(f"Model file not found: {model_file}\n\nModel '{current_model_type}' may not be available or download failed.\nPlease check your internet connection and try again.")
                        status_label.setText('Loading model...')
                        QApplication.processEvents()
                        if is_affinity_builtin_model(current_model_type):
                            preview_dialog.model = load_bact_phase_affinity_model(gpu=current_use_gpu, checkpoint_path=model_file)
                        else:
                            preview_dialog.model = load_cellpose_segmentation_model(gpu=current_use_gpu, loader_path=model_file, model_type=current_model_type)
                        try:
                            from bactomate.gui.omnipose_utils import BACTOMATE_AFFINITY_LOADER_ID
                            cp = getattr(preview_dialog.model, 'cp', preview_dialog.model)
                            nchan = getattr(cp, 'nchan', None)
                            nclasses = getattr(cp, 'nclasses', None)
                            append_omnipose_debug(f'[BactoMate BUILTIN post-load] model_type={current_model_type!r} loader={model_file!r} class={type(preview_dialog.model).__name__} nchan={nchan} nclasses={nclasses} affinity_loader={BACTOMATE_AFFINITY_LOADER_ID}')
                        except Exception:
                            pass
                        preview_dialog._last_model_key = current_key
                        preview_dialog._last_use_gpu = current_use_gpu
                        set_loaded_status()
                        QApplication.processEvents()
                    except RuntimeError as e:
                        error_msg = str(e)
                        if is_affinity_builtin_model(current_model_type):
                            show_message_dialog(preview_dialog, 'bact_phase_affinity load failed', f'Could not load the affinity model with a 3-class head.\n\n{error_msg}\n\nStart BactoMate from the project folder so you get the latest loader:\n  cd .../BactoMate && python3 -m bactomate\n\nOr run: python3 -m bactomate.scripts.verify_affinity_model', kind='error')
                            status_label.setText('Affinity model load failed')
                            return False
                        if 'size mismatch' in error_msg or 'state_dict' in error_msg or 'checkpoint' in error_msg.lower() or ('Unexpected key' in error_msg):
                            from cellpose_omni.models import model_path, MODEL_DIR
                            import os
                            expected_basename = f'{current_model_type}torch_0'
                            model_file_path = Path(MODEL_DIR) / expected_basename
                            if model_file_path.exists() and model_file_path.is_symlink():
                                try:
                                    target = model_file_path.readlink()
                                    if '_cp3' in str(target) and '_omni' in current_model_type:
                                        model_file_path.unlink()
                                        status_label.setText('Removed incorrect model symlink...')
                                        QApplication.processEvents()
                                except Exception:
                                    get_logger().debug('Could not remove incorrect model symlink', exc_info=True)
                            try:
                                from bactomate.gui.omnipose_pretrained_cache import clear_bactomate_builtin_cache_entry, ensure_builtin_in_bactomate_cache
                                clear_bactomate_builtin_cache_entry(current_model_type)
                                if model_file_path.is_file():
                                    status_label.setText('Fixing model architecture mismatch...')
                                    QApplication.processEvents()
                                    backup_file = str(model_file_path) + '.corrupted'
                                    if os.path.exists(backup_file):
                                        os.remove(backup_file)
                                    os.rename(model_file_path, backup_file)
                                status_label.setText('Re-loading model...')
                                QApplication.processEvents()
                                bcp = ensure_builtin_in_bactomate_cache(current_model_type)
                                preview_dialog.model = load_cellpose_segmentation_model(gpu=current_use_gpu, loader_path=bcp, model_type=current_model_type)
                                preview_dialog._last_model_key = current_key
                                preview_dialog._last_use_gpu = current_use_gpu
                                status_label.setText('Model reloaded successfully!')
                                QApplication.processEvents()
                            except Exception as retry_error:
                                error_details = f'Original error: {error_msg}\nRetry error: {retry_error}'
                                help_text = f'The model checkpoint has a different architecture than expected.\n\n{error_details}\n\n'
                                script_dir = PROJECT_ROOT
                                local_models_dir = script_dir / 'models'
                                has_cp3 = (local_models_dir / 'bact_phase_cp3').exists() if local_models_dir.exists() else False
                                if has_cp3 and '_omni' in current_model_type:
                                    help_text += f"⚠️ IMPORTANT: You have 'bact_phase_cp3' in your models directory, but this is a Cellpose model, NOT an Omnipose model.\nThey have different architectures and cannot be used interchangeably.\n\n"
                                if is_affinity_builtin_model(current_model_type):
                                    help_text += "bact_phase_affinity is a 3-channel affinity model (not a 4-channel flow Omnipose model). Do not rename it with '_omni' in the filename.\n\n"
                                help_text += f'Please try:\n1. Restart BactoMate after updating (affinity models load without _omni symlink)\n2. Try a different model type\n3. If you have cp3 models, use them with Cellpose model types, not Omnipose\n4. Delete corrupted models: rm -rf {MODEL_DIR}/*.corrupted\n5. Re-download: clear BactoMate omnipose_pretrained cache for this model'
                                show_message_dialog(preview_dialog, 'Model Architecture Mismatch', help_text, kind='error')
                                status_label.setText('Model load failed')
                                return False
                        else:
                            show_message_dialog(preview_dialog, 'Model Error', f'Could not load model:\n{error_msg}', kind='error')
                            status_label.setText('Model load failed')
                        return False
                    except HTTPError as http_err:
                        error_msg = str(http_err)
                        if http_err.code == 404:
                            model_file = find_model_file(current_model_type, 0, True)
                            if model_file and os.path.exists(model_file):
                                try:
                                    status_label.setText('Using cached model (not in repo)...')
                                    QApplication.processEvents()
                                    if not load_builtin_cellpose_from_find_cache(current_model_type):
                                        raise RuntimeError('Could not load from cache path')
                                    preview_dialog._last_model_key = current_key
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText('Using cached model')
                                    QApplication.processEvents()
                                    return True
                                except Exception as cache_error:
                                    show_message_dialog(preview_dialog, 'Model Not Available', f"Model '{current_model_type}' is not available in the repository (404 error).\n\nAvailable Omnipose models:\n  • bact_phase_omni (recommended for bacteria)\n  • bact_fluor_omni\n  • bact_phase_affinity\n\nSee: https://omnipose.readthedocs.io/models.html\n\nPlease select a different model from the dropdown.", kind='warning')
                                    status_label.setText('Model not available')
                                    return False
                            else:
                                show_message_dialog(preview_dialog, 'Model Not Available', f"Model '{current_model_type}' is not available in the repository (404 error).\n\nAvailable Omnipose models:\n  • bact_phase_omni (recommended for bacteria)\n  • bact_fluor_omni\n  • bact_phase_affinity\n\nSee: https://omnipose.readthedocs.io/models.html\n\nPlease select a different model from the dropdown.", kind='warning')
                                status_label.setText('Model not available')
                                return False
                        model_file = find_model_file(current_model_type, 0, True)
                        cached_model_exists = model_file is not None and os.path.exists(model_file)
                        if cached_model_exists:
                            try:
                                status_label.setText('Trying cached model...')
                                QApplication.processEvents()
                                if not load_builtin_cellpose_from_find_cache(current_model_type):
                                    raise RuntimeError('Could not load from cache path')
                                preview_dialog._last_model_key = current_key
                                preview_dialog._last_use_gpu = current_use_gpu
                                status_label.setText('Using cached model')
                                QApplication.processEvents()
                                return True
                            except Exception as cache_error:
                                show_message_dialog(preview_dialog, 'Network Error & Cache Issue', f'Could not download model (server error) and cached model also failed:\n\nDownload error: {error_msg}\nCache error: {cache_error}\n\nPlease try:\n1. Wait a few minutes and try again (server may be temporarily down)\n2. Try a different model type\n3. Check your internet connection\n4. If persistent, clear cache: rm -rf ~/.cellpose/models/', kind='warning')
                                status_label.setText('Network & cache error')
                                return False
                        else:
                            show_message_dialog(preview_dialog, 'Network Error - Model Server Unavailable', f"Could not download model due to network error:\n\n{error_msg}\n\nThe model server appears to be temporarily unavailable (HTTP 500).\n\nPlease try:\n1. Wait 5-10 minutes and try again (server may be temporarily down)\n2. Check your internet connection\n3. Try a different model type (may already be cached)\n4. Check the Cellpose website status: https://www.cellpose.org\n\nNote: Once downloaded, models are cached locally and won't need to be re-downloaded.", kind='warning')
                            status_label.setText('Server unavailable - try again later')
                            return False
                    except Exception as e:
                        error_msg = str(e)
                        if 'HTTP' in error_msg or '500' in error_msg or 'connection' in error_msg.lower() or ('network' in error_msg.lower()) or ('urllib' in error_msg.lower()) or ('download' in error_msg.lower()):
                            model_file = find_model_file(current_model_type, 0, True)
                            cached_model_exists = model_file is not None and os.path.isfile(model_file)
                            if cached_model_exists:
                                try:
                                    status_label.setText('Trying cached model...')
                                    QApplication.processEvents()
                                    if not load_builtin_cellpose_from_find_cache(current_model_type):
                                        raise RuntimeError('Could not load from cache path')
                                    preview_dialog._last_model_key = current_key
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText('Using cached model')
                                    QApplication.processEvents()
                                    return True
                                except Exception as cache_error:
                                    show_message_dialog(preview_dialog, 'Network Error & Cache Issue', f'Could not download model (server error) and cached model also failed:\n\nDownload error: {error_msg}\nCache error: {cache_error}\n\nPlease try:\n1. Wait a few minutes and try again (server may be temporarily down)\n2. Try a different model type\n3. Check your internet connection\n4. If persistent, clear cache: rm -rf ~/.cellpose/models/', kind='warning')
                                    status_label.setText('Network & cache error')
                                    return False
                            else:
                                show_message_dialog(preview_dialog, 'Network Error - Model Server Unavailable', f"Could not download model due to network error:\n\n{error_msg}\n\nThe model server appears to be temporarily unavailable (HTTP 500).\n\nPlease try:\n1. Wait 5-10 minutes and try again (server may be temporarily down)\n2. Check your internet connection\n3. Try a different model type (may already be cached)\n4. Check the Cellpose website status: https://www.cellpose.org\n\nNote: Once downloaded, models are cached locally and won't need to be re-downloaded.", kind='warning')
                                status_label.setText('Server unavailable - try again later')
                                return False
                        else:
                            show_message_dialog(preview_dialog, 'Model Error', f'Could not load model:\n{error_msg}', kind='error')
                            status_label.setText('Model load failed')
                        return False
                return True

            def get_current_parameters():
                """Get current parameters from main window (works because dialog is non-modal)."""
                self._commit_segmentation_parameter_spins()
                import sys
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                diameter_value = self._get_effective_diameter_pixels()
                display = self.model_combo.currentText()
                (model_type, pretrained_path) = self._resolve_model_selection(display)
                if diameter_value == 0 and is_frozen and (not pretrained_path):
                    default_diameters = {'bact_phase_omni': 30.0, 'bact_omni': 30.0, 'bact_fluor_omni': 30.0}
                    diameter_value = default_diameters.get(model_type or display, 30.0)
                try:
                    flow_val = float(self.flow_threshold.value())
                except RuntimeError:
                    flow_val = 0.4
                try:
                    cluster_on = bool(self.cluster.isChecked())
                except RuntimeError:
                    cluster_on = False
                use_gpu = True
                return {'diameter': diameter_value if diameter_value and diameter_value > 0 else 0, 'flow_threshold': flow_val, 'mask_threshold': self._get_effective_mask_threshold(), 'cluster': cluster_on, 'model_type': model_type, 'pretrained_model': str(pretrained_path) if pretrained_path else None, 'use_gpu': use_gpu}
            preview_dialog.segmentation_cache = {}
            preview_dialog.display_cache_key = None
            preview_dialog._seg_request_id = 0
            preview_dialog._seg_current_requested_frame = 0
            preview_dialog._seg_worker = None
            preview_dialog._dialog_closed = False
            preview_dialog._seg_progress_dialog = None
            preview_dialog._seg_worker_slot = None
            preview_dialog.set_extra('show_cells', True)
            preview_dialog.set_extra('show_foci', bool(getattr(self, 'detect_foci', None) and self.detect_foci.isChecked() and getattr(preview_dialog, '_foci_source_paths', [])))

            def _seg_is_running() -> bool:
                w = getattr(preview_dialog, '_seg_worker', None)
                return bool(w is not None and w.isRunning())

            def _seg_request_stop() -> None:
                w = getattr(preview_dialog, '_seg_worker', None)
                if w is None:
                    return
                try:
                    if w.isRunning():
                        w.requestInterruption()
                        w.quit()
                except Exception:
                    pass

            def _seg_wait(ms: int) -> bool:
                w = getattr(preview_dialog, '_seg_worker', None)
                if w is None:
                    return True
                try:
                    return bool(w.wait(int(ms)))
                except Exception:
                    return True

            def _seg_disconnect() -> None:
                """Detach and schedule deletion of the preview segmentation QThread (PyQt-safe)."""
                w = getattr(preview_dialog, '_seg_worker', None)
                s = getattr(preview_dialog, '_seg_worker_slot', None)
                if w is not None and s is not None:
                    try:
                        w.finished.disconnect(s)
                    except Exception:
                        pass
                if w is not None:
                    try:
                        w.deleteLater()
                    except Exception:
                        pass
                try:
                    preview_dialog._seg_worker = None
                except Exception:
                    pass
                try:
                    preview_dialog._seg_worker_slot = None
                except Exception:
                    pass
            try:
                preview_dialog.set_worker_guard(WorkerGuard(is_running=_seg_is_running, request_stop=_seg_request_stop, wait=_seg_wait, disconnect=_seg_disconnect, on_close_while_running='block_with_warning', warning_title='Segmentation in progress', warning_text='Please wait for the current segmentation to finish before closing the preview.'))
            except Exception:
                pass
            _PARAM_DECIMALS = 4

            def _normalize_source_key(source_key):
                """Normalize a source selection tuple to (kind, idx)."""
                if not isinstance(source_key, tuple) or len(source_key) < 2:
                    return ('input', 0)
                try:
                    kind = str(source_key[0] or 'input')
                except Exception:
                    kind = 'input'
                try:
                    idx = int(source_key[1])
                except Exception:
                    idx = 0
                return (kind, idx)

            def _source_token(source_key) -> str:
                (kind, idx) = _normalize_source_key(source_key)
                return f'{kind}:{idx}'

            def _source_label_from_token(token: str) -> str:
                t = str(token or 'input:0')
                if t.startswith('input:'):
                    return ''
                if t.startswith('foci:'):
                    return f" | src={t.replace('foci:', 'foci#')}"
                if t.startswith('sibling:'):
                    return f" | src={t.replace('sibling:', 'file#')}"
                return f' | src={t}'

            def format_omnipose_params_from_key(key):
                """Short param string from cache key for display/combo."""
                if not key or len(key) < 7:
                    return '?'
                try:
                    frame_idx = key[0]
                    diameter = key[1]
                    flow_threshold = key[2]
                    mask_threshold = key[3]
                    cluster = key[4]
                    model_str = key[5]
                except Exception:
                    return '?'
                model_short = Path(model_str).stem if model_str and getattr(Path(model_str), 'suffix', '') else model_str or '?'
                source_suffix = _source_label_from_token(key[7] if len(key) > 7 else 'input:0')
                return f"d={diameter:.2f} flow={flow_threshold:.2f} mask={mask_threshold:.2f} cluster={('on' if cluster else 'off')} {model_short}{source_suffix}"

            def format_omnipose_params_from_params(params):
                """Short param string from params dict for display."""
                d = params.get('diameter')
                d = 0.0 if d is None or not isinstance(d, (int, float)) else float(d)
                f = params.get('flow_threshold', 0.4)
                f = 0.4 if f is None or not isinstance(f, (int, float)) else float(f)
                m = params.get('mask_threshold', 0.0)
                m = 0.0 if m is None or not isinstance(m, (int, float)) else float(m)
                cluster = bool(params.get('cluster', False))
                model_str = params.get('model_type') or params.get('pretrained_model') or 'bact_phase_omni'
                model_short = Path(model_str).stem if model_str and getattr(Path(model_str), 'suffix', '') else model_str or '?'
                return f"d={d:.2f} flow={f:.2f} mask={m:.2f} cluster={('on' if cluster else 'off')} {model_short}"

            def get_cache_key(frame_idx, params, source_key=('input', 0)):
                """Generate cache key from frame index and parameters. Floats rounded to avoid 0.3 -> 0.30000000000004."""
                diameter = params.get('diameter', 0)
                if diameter is None or not isinstance(diameter, (int, float)):
                    diameter = 0.0
                diameter = round(float(diameter), _PARAM_DECIMALS)
                flow_threshold = params.get('flow_threshold', 0.4)
                if flow_threshold is None or not isinstance(flow_threshold, (int, float)):
                    flow_threshold = 0.4
                flow_threshold = round(float(flow_threshold), _PARAM_DECIMALS)
                mask_threshold = params.get('mask_threshold', 0.0)
                if mask_threshold is None or not isinstance(mask_threshold, (int, float)):
                    mask_threshold = 0.0
                mask_threshold = round(float(mask_threshold), _PARAM_DECIMALS)
                try:
                    frame_shape_token = tuple(int(x) for x in np.asarray(images[frame_idx]).shape[-2:])
                except Exception:
                    frame_shape_token = ()
                return (frame_idx, diameter, flow_threshold, mask_threshold, bool(params.get('cluster', False)), str(params.get('model_type') or params.get('pretrained_model') or 'bact_phase_omni'), bool(params.get('use_gpu', True)), _source_token(source_key), frame_shape_token)

            def _seg_disk_cache_path(cache_key):
                """Return the on-disk preview-cache path for an exact segmentation key."""
                import hashlib
                import tempfile
                cache_dir = Path(tempfile.gettempdir()) / 'bactomate_preview_cache'
                disk_key = hashlib.sha256(f'{file_path}|{cache_key}'.encode()).hexdigest()[:24]
                return cache_dir / f'{disk_key}.npy'

            def _cached_mask_for_key(cache_key):
                """Load an already-computed mask from memory/disk without running segmentation."""
                try:
                    if cache_key in preview_dialog.segmentation_cache:
                        return preview_dialog.segmentation_cache[cache_key]
                    disk_path = _seg_disk_cache_path(cache_key)
                    if disk_path.exists():
                        mask_disk = np.load(disk_path)
                        preview_dialog.segmentation_cache[cache_key] = mask_disk
                        return mask_disk
                except Exception:
                    pass
                return None

            def _preview_to_three_channel(img):
                """Convert image to 3-channel (H, W, 3) for Cellpose-SAM. Grayscale duplicated."""
                arr = np.asarray(img, dtype=np.float32)
                if arr.ndim == 2:
                    return np.stack([arr, arr, arr], axis=-1)
                if arr.ndim == 3:
                    if arr.shape[-1] == 1:
                        return np.concatenate([arr, arr, arr], axis=-1)
                    if arr.shape[-1] == 3:
                        return arr
                return np.stack([arr, arr, arr], axis=-1)

            class OmniposePreviewWorker(QThread):
                finished = pyqtSignal(int, int, tuple, object, str)

                def __init__(self, request_id, frame_idx, cache_key, model, seg_frame, diameter_seg, flow_threshold, mask_threshold, cluster, orig_h, orig_w, seg_scale, is_frozen, is_cellpose_sam=False, affinity_seg=False, parent=None):
                    super().__init__(parent)
                    self._request_id = request_id
                    self._frame_idx = frame_idx
                    self._cache_key = cache_key
                    self._model = model
                    self._seg_frame = np.asarray(seg_frame, dtype=np.float32).copy()
                    self._diameter_seg = diameter_seg
                    self._flow_threshold = flow_threshold
                    self._mask_threshold = mask_threshold
                    self._cluster = cluster
                    (self._orig_h, self._orig_w) = (orig_h, orig_w)
                    self._seg_scale = seg_scale
                    self._is_frozen = is_frozen
                    self._is_cellpose_sam = is_cellpose_sam
                    self._affinity_seg = bool(affinity_seg)

                def _omni_eval_kwargs(self, *, diameter):
                    return dict(channels=[0, 0], diameter=diameter, flow_threshold=self._flow_threshold, mask_threshold=self._mask_threshold, cellprob_threshold=0.0, do_3D=False, omni=True, affinity_seg=self._affinity_seg, cluster=self._cluster, net_avg=False, resample=False, verbose=False)

                def _eval_omni(self, seg_frame):
                    """Run cellpose_omni eval on the given seg_frame. Returns mask (no upsample)."""
                    if self._is_frozen:
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                diameter = self._diameter_seg if self._diameter_seg and self._diameter_seg > 0 else 30.0
                                return omnipose_eval_first_mask(self._model.eval(seg_frame, **self._omni_eval_kwargs(diameter=diameter)))
                            except TypeError as e:
                                if ("'>' not supported" in str(e) or "'<' not supported" in str(e)) and attempt < max_retries - 1:
                                    d = 30.0 + attempt * 5.0
                                    self._diameter_seg = d * self._seg_scale if self._seg_scale < 1.0 else d
                                    continue
                                raise
                    return omnipose_eval_first_mask(self._model.eval(seg_frame, **self._omni_eval_kwargs(diameter=self._diameter_seg)))

                def _crop_to_multiple_of(self, arr, divisor: int):
                    """Crop bottom/right rows so each side is a multiple of `divisor` (no resize, label-safe)."""
                    (h, w) = arr.shape[:2]
                    new_h = max(divisor, h // divisor * divisor)
                    new_w = max(divisor, w // divisor * divisor)
                    if new_h == h and new_w == w:
                        return arr
                    if arr.ndim == 2:
                        return np.ascontiguousarray(arr[:new_h, :new_w])
                    return np.ascontiguousarray(arr[:new_h, :new_w, ...])

                def _is_broadcast_shape_error(self, exc: BaseException) -> bool:
                    s = str(exc)
                    return 'could not broadcast' in s and 'shape' in s or 'operands could not be broadcast' in s

                def run(self):
                    (mask, error_str) = (None, '')
                    if self.isInterruptionRequested():
                        self.finished.emit(self._request_id, self._frame_idx, self._cache_key, None, 'cancelled')
                        return
                    try:
                        if self._is_cellpose_sam:
                            import os
                            os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
                            os.environ.setdefault('OMP_NUM_THREADS', '1')
                            img_3ch = _preview_to_three_channel(self._seg_frame)
                            result = self._model.eval([img_3ch], diameter=self._diameter_seg if self._diameter_seg and self._diameter_seg > 0 else None, flow_threshold=self._flow_threshold, cellprob_threshold=self._mask_threshold, batch_size=1)
                            if isinstance(result, tuple):
                                batch_masks = result[0]
                            else:
                                batch_masks = result
                            if isinstance(batch_masks, np.ndarray) and batch_masks.ndim == 3:
                                mask = batch_masks[0].astype(np.uint16)
                            else:
                                mask = np.asarray(batch_masks[0], dtype=np.uint16)
                            if self._seg_scale < 1.0 and mask is not None:
                                mask = upsample_mask_to_shape(mask, self._orig_h, self._orig_w)
                        else:
                            try:
                                mask = self._eval_omni(self._seg_frame)
                            except ValueError as e:
                                if not self._is_broadcast_shape_error(e):
                                    raise
                                (seg_h, seg_w) = self._seg_frame.shape[:2]
                                cropped = self._crop_to_multiple_of(self._seg_frame, 16)
                                if cropped.shape[:2] == (seg_h, seg_w):
                                    nh = max(16, seg_h - 16)
                                    nw = max(16, seg_w - 16)
                                    cropped = np.ascontiguousarray(self._seg_frame[:nh, :nw]) if self._seg_frame.ndim == 2 else np.ascontiguousarray(self._seg_frame[:nh, :nw, ...])
                                print(f'[preview] broadcast error on shape={self._seg_frame.shape} -> retry with cropped shape={cropped.shape} ({e})')
                                mask = self._eval_omni(cropped)
                                if mask is not None and mask.shape[:2] != (seg_h, seg_w):
                                    padded = np.zeros((seg_h, seg_w), dtype=mask.dtype)
                                    (mh, mw) = mask.shape[:2]
                                    padded[:mh, :mw] = mask
                                    mask = padded
                        if self._seg_scale < 1.0 and mask is not None:
                            mask = upsample_mask_to_shape(mask, self._orig_h, self._orig_w)
                    except Exception as e:
                        try:
                            print(f"[preview] segmentation failed: err={type(e).__name__}: {e} seg_frame_shape={getattr(self._seg_frame, 'shape', None)} orig=({self._orig_h},{self._orig_w}) diameter_seg={self._diameter_seg} seg_scale={self._seg_scale} is_cellpose_sam={self._is_cellpose_sam} is_frozen={self._is_frozen}")
                            print('[preview] traceback:')
                            traceback.print_exc()
                        except Exception:
                            pass
                        error_str = str(e)
                    self.finished.emit(self._request_id, self._frame_idx, self._cache_key, mask, error_str)

            def run_segmentation(frame_idx, use_cache=True, *, source_key=('input', 0), frame_override=None, preserve_user_lut=True):
                """Run segmentation on a specific frame with caching and background threading"""
                params = get_current_parameters()
                if params.get('diameter') is None or not isinstance(params.get('diameter'), (int, float)):
                    params['diameter'] = 0.0
                else:
                    params['diameter'] = float(params['diameter'])
                if params.get('flow_threshold') is None or not isinstance(params.get('flow_threshold'), (int, float)):
                    params['flow_threshold'] = 0.4
                else:
                    params['flow_threshold'] = float(params['flow_threshold'])
                if params.get('mask_threshold') is None or not isinstance(params.get('mask_threshold'), (int, float)):
                    params['mask_threshold'] = 0.0
                else:
                    params['mask_threshold'] = float(params['mask_threshold'])
                cache_key = get_cache_key(frame_idx, params, source_key=source_key)
                _disk_path = _seg_disk_cache_path(cache_key)
                if use_cache:
                    try:
                        mask_cached = _cached_mask_for_key(cache_key)
                        if mask_cached is not None:
                            return (mask_cached, None)
                    except (TypeError, ValueError) as e:
                        preview_dialog.segmentation_cache.clear()
                if not load_model():
                    return (None, 'Model load failed')
                frame = frame_override
                if frame is None:
                    frame = images[frame_idx]
                    if frame.ndim == 3:
                        frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                frame = np.asarray(frame)
                if frame.ndim != 2:
                    try:
                        from bactomate.modules.image_stack_utils import coerce_plane_to_2d
                        frame = np.asarray(coerce_plane_to_2d(frame))
                    except Exception:
                        frame = np.asarray(frame.squeeze())
                if frame.ndim != 2:
                    return (None, 'Selected image source is not a 2D frame')
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32)
                (seg_frame, seg_scale, (orig_h, orig_w)) = downsample_for_preview_seg(frame, max_size=PREVIEW_SEG_MAX_SIZE)
                if seg_scale < 1.0:
                    get_logger().info('Preview: downsampled frame for speed: %s -> %s (scale=%.3f)', (orig_h, orig_w), getattr(seg_frame, 'shape', None), seg_scale)
                diameter = params['diameter']
                flow_threshold = params['flow_threshold']
                mask_threshold = params['mask_threshold']
                cluster = params['cluster']
                default_diameters_preview = {'bact_phase_omni': 30.0, 'bact_omni': 30.0, 'bact_fluor_omni': 30.0, CELLPOSE_SAM_MODEL: 30.0}
                effective_diameter = diameter if diameter is not None and diameter > 0 else default_diameters_preview.get(params.get('model_type') or '', 30.0)
                diameter_seg = effective_diameter * seg_scale
                if diameter_seg < 1:
                    diameter_seg = max(1.0, diameter_seg)
                existing = getattr(preview_dialog, '_seg_worker', None)
                if existing is not None and existing.isRunning():
                    preview_dialog._seg_request_id += 1
                    preview_dialog._seg_current_requested_frame = frame_idx
                    if not hasattr(preview_dialog, '_seg_request_meta') or not isinstance(preview_dialog._seg_request_meta, dict):
                        preview_dialog._seg_request_meta = {}
                    preview_dialog._seg_request_meta[preview_dialog._seg_request_id] = {'preserve_user_lut': bool(preserve_user_lut)}
                    status_label.setText('Segmenting... (requested frame %s)' % (frame_idx + 1))
                    return (None, None)
                import sys
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                preview_dialog._seg_request_id += 1
                preview_dialog._seg_current_requested_frame = frame_idx
                if not hasattr(preview_dialog, '_seg_request_meta') or not isinstance(preview_dialog._seg_request_meta, dict):
                    preview_dialog._seg_request_meta = {}
                preview_dialog._seg_request_meta[preview_dialog._seg_request_id] = {'preserve_user_lut': bool(preserve_user_lut)}
                kind = 'USER_MODEL' if params.get('pretrained_model') else 'BUILTIN'
                get_logger().info('Preview: starting background worker (%s) frame_shape=%s diameter=%s flow_thr=%s', kind, getattr(seg_frame, 'shape', None), diameter_seg, flow_threshold)

                def _on_seg_worker_finished(rid, fid, ck, m, err):
                    worker = getattr(preview_dialog, '_seg_worker', None)
                    slot = getattr(preview_dialog, '_seg_worker_slot', None)
                    try:
                        if getattr(preview_dialog, '_dialog_closed', False):
                            return
                        if rid != getattr(preview_dialog, '_seg_request_id', -1):
                            pending_fid = int(getattr(preview_dialog, '_seg_current_requested_frame', 0))
                            update_display(pending_fid)

                            def _retry_pending_seg():
                                if getattr(preview_dialog, '_dialog_closed', False):
                                    return
                                try:
                                    run_segmentation(pending_fid, use_cache=True)
                                except Exception:
                                    pass
                            QTimer.singleShot(0, _retry_pending_seg)
                            return
                        prog = getattr(preview_dialog, '_seg_progress_dialog', None)
                        if prog is not None:
                            try:
                                prog.close()
                            except Exception:
                                pass
                            preview_dialog._seg_progress_dialog = None
                        if err:
                            status_label.setText('Segmentation failed: %s' % err[:80])
                            status_label.setStyleSheet('color: #FF0000; font-weight: bold;')
                            redetect_btn.setEnabled(True)
                            clear_cache_btn.setEnabled(True)
                            frame_spin.setEnabled(True)
                            return
                        preview_dialog.segmentation_cache[ck] = m
                        try:
                            _disk_path = _seg_disk_cache_path(ck)
                            _disk_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(_disk_path, m)
                        except Exception:
                            pass
                        meta = {}
                        try:
                            meta = (getattr(preview_dialog, '_seg_request_meta', {}) or {}).pop(rid, {}) or {}
                        except Exception:
                            meta = {}
                        _apply_mask_to_ui(fid, m, preserve_user_lut=bool(meta.get('preserve_user_lut', True)))
                        cache_size = len(preview_dialog.segmentation_cache)
                        status_label.setText(f'Ready (cached: {cache_size})')
                        status_label.setStyleSheet('color: #00FF00; font-weight: bold;')
                    except RuntimeError:
                        pass
                    finally:
                        if worker is not None:
                            if slot is not None:
                                try:
                                    worker.finished.disconnect(slot)
                                except Exception:
                                    pass
                            try:
                                worker.deleteLater()
                            except Exception:
                                pass
                        try:
                            preview_dialog._seg_worker = None
                            preview_dialog._seg_worker_slot = None
                        except Exception:
                            pass
                is_cellpose_sam = params.get('model_type') == CELLPOSE_SAM_MODEL
                affinity_seg = eval_uses_affinity_segmentation(model_type=params.get('model_type'), pretrained_path=params.get('pretrained_model'))
                worker = OmniposePreviewWorker(preview_dialog._seg_request_id, frame_idx, cache_key, preview_dialog.model, seg_frame, diameter_seg, flow_threshold, mask_threshold, cluster, orig_h, orig_w, seg_scale, is_frozen, is_cellpose_sam=is_cellpose_sam, affinity_seg=affinity_seg, parent=preview_dialog)
                preview_dialog._seg_worker_slot = _on_seg_worker_finished
                worker.finished.connect(_on_seg_worker_finished)
                preview_dialog._seg_worker = worker
                worker.start()
                return (None, None)

            def _apply_mask_to_ui(frame_idx, mask, *, preserve_user_lut=True):
                """Apply a segmentation mask to the display (used after cache hit or worker result). Re-enables controls."""
                display_frame = None
                try:
                    _set_display_frame_for_selection(frame_idx)
                    display_frame = getattr(preview_dialog, 'current_display_frame', None)
                except Exception:
                    display_frame = None
                if display_frame is None:
                    frame = images[frame_idx]
                    if frame.ndim == 3:
                        display_frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                    else:
                        display_frame = frame
                preview_dialog.masks = mask
                preview_dialog.current_display_frame = display_frame
                try:
                    lut_widget.set_frame(np.asarray(display_frame), preserve_user_lut=bool(preserve_user_lut))
                except Exception:
                    pass
                refresh_image_display()
                QApplication.processEvents()
                preview_dialog.display_cache_key = None
                file_name = Path(file_path).name
                num_cells = len(np.unique(mask)) - 1
                params = get_current_parameters()
                model_name = Path(params['pretrained_model']).stem if params.get('pretrained_model') else params.get('model_type', '—')
                params_str = format_omnipose_params_from_params(params)
                title_label.setText(f'Frame {frame_idx + 1} of {num_frames} | Cells: {num_cells} | Model: {model_name} | {file_name}\nParameters: {params_str}')
                redetect_btn.setEnabled(True)
                clear_cache_btn.setEnabled(True)
                frame_spin.setEnabled(True)
                cache_size = len(preview_dialog.segmentation_cache)
                if num_cells == 0:
                    if params.get('pretrained_model'):
                        status_label.setText('No cells detected. Try Flow 0, Mask -2, Diameter 0 or cell size (px). See docs/CUSTOM_MODEL_TROUBLESHOOTING.md.')
                    else:
                        status_label.setText(f'Ready (cached: {cache_size}) — no cells')
                    status_label.setStyleSheet('color: #FF9800; font-weight: bold;')
                else:
                    status_label.setText(f'Ready (cached: {cache_size})')
                    status_label.setStyleSheet('color: #00FF00; font-weight: bold;')
                refresh_cached_runs_combo()

            def update_display(frame_idx, show_progress=True, force_recalc=False, *, source_key=('input', 0), frame_override=None, preserve_user_lut=True):
                """Update the display with segmentation overlay"""
                params = get_current_parameters()
                cache_key = get_cache_key(frame_idx, params, source_key=source_key)
                has_cache = cache_key in preview_dialog.segmentation_cache and (not force_recalc)
                progress_dialog = None
                if show_progress and (not has_cache):
                    progress_dialog = QProgressDialog('Segmenting frame...', None, 0, 0, preview_dialog)
                    progress_dialog.setWindowTitle('Processing')
                    progress_dialog.setAutoClose(False)
                    progress_dialog.setAutoReset(False)
                    progress_dialog.setCancelButton(None)
                    progress_dialog.show()
                    QApplication.processEvents()
                    preview_dialog._seg_progress_dialog = progress_dialog
                if progress_dialog:
                    redetect_btn.setEnabled(False)
                    clear_cache_btn.setEnabled(False)
                    frame_spin.setEnabled(False)
                try:
                    if progress_dialog:
                        progress_dialog.setLabelText('Running segmentation...')
                        QApplication.processEvents()
                    (mask, err) = run_segmentation(frame_idx, use_cache=not force_recalc, source_key=source_key, frame_override=frame_override, preserve_user_lut=preserve_user_lut)
                    if err is not None:
                        if progress_dialog:
                            progress_dialog.close()
                            preview_dialog._seg_progress_dialog = None
                        status_label.setText('Segmentation failed' + (': ' + err[:60] if err else ''))
                        status_label.setStyleSheet('color: #FF0000; font-weight: bold;')
                        redetect_btn.setEnabled(True)
                        clear_cache_btn.setEnabled(True)
                        frame_spin.setEnabled(True)
                        return
                    if mask is not None:
                        if progress_dialog:
                            progress_dialog.close()
                            preview_dialog._seg_progress_dialog = None
                        _apply_mask_to_ui(frame_idx, mask, preserve_user_lut=preserve_user_lut)
                        return
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                        preview_dialog._seg_progress_dialog = None
                    redetect_btn.setEnabled(True)
                    clear_cache_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    status_label.setText(f'Error: {str(e)}')
                    status_label.setStyleSheet('color: #FF0000; font-weight: bold;')
                    show_message_dialog(preview_dialog, 'Preview Error', f'Could not update display:\n{str(e)}', kind='warning')

            def show_unsegmented_frame(frame_idx: int, *, preserve_user_lut=True, status_text: str | None=None):
                """Redraw the selected frame without starting segmentation."""
                display_frame = None
                try:
                    _set_display_frame_for_selection(frame_idx)
                    display_frame = getattr(preview_dialog, 'current_display_frame', None)
                except Exception:
                    display_frame = None
                if display_frame is None:
                    frame = images[frame_idx]
                    if frame.ndim == 3:
                        display_frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                    else:
                        display_frame = frame
                preview_dialog.masks = np.zeros(np.asarray(display_frame).shape[:2], dtype=np.int32)
                preview_dialog.current_display_frame = display_frame
                try:
                    lut_widget.set_frame(np.asarray(display_frame), preserve_user_lut=bool(preserve_user_lut))
                except Exception:
                    pass
                refresh_image_display()
                file_name = Path(file_path).name
                params = get_current_parameters()
                model_name = Path(params['pretrained_model']).stem if params.get('pretrained_model') else params.get('model_type', '—')
                params_str = format_omnipose_params_from_params(params)
                title_label.setText(f'Frame {frame_idx + 1} of {num_frames} | Not segmented | Model: {model_name} | {file_name}\nPending parameters: {params_str}')
                if status_text:
                    status_label.setText(status_text)
                    status_label.setStyleSheet('color: #2196F3; font-weight: bold;')
                refresh_cached_runs_combo()

            def show_current_cache_or_unsegmented(frame_idx: int, *, preserve_user_lut=True):
                """Display an exact cached current-parameter result, but never start model inference."""
                params = get_current_parameters()
                source_key = image_combo.currentData() if image_combo is not None else ('input', 0)
                cache_key = get_cache_key(frame_idx, params, source_key=source_key)
                mask = _cached_mask_for_key(cache_key)
                if mask is not None:
                    _apply_mask_to_ui(frame_idx, mask, preserve_user_lut=preserve_user_lut)
                    return
                show_unsegmented_frame(frame_idx, preserve_user_lut=preserve_user_lut, status_text='Preview parameters changed. Click Re-detect to update.')

            def show_cached_run(cache_key):
                """Display a cached segmentation (mask + params) without running model. Used when user picks a cached run from combo."""
                if cache_key not in preview_dialog.segmentation_cache:
                    return
                frame_idx = cache_key[0]
                mask = preview_dialog.segmentation_cache[cache_key]
                try:
                    _set_display_frame_for_selection(frame_idx)
                    display_frame = getattr(preview_dialog, 'current_display_frame', None)
                except Exception:
                    display_frame = None
                if display_frame is None:
                    frame = images[frame_idx]
                    if frame.ndim == 3:
                        display_frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                    else:
                        display_frame = frame
                preview_dialog.masks = mask
                preview_dialog.current_display_frame = display_frame
                try:
                    lut_widget.set_frame(np.asarray(display_frame), preserve_user_lut=True)
                except Exception:
                    pass
                refresh_image_display()
                num_cells = len(np.unique(mask)) - 1
                file_name = Path(file_path).name
                params_str = format_omnipose_params_from_key(cache_key)
                model_short = cache_key[5] if len(cache_key) > 5 else '—'
                if model_short and getattr(Path(model_short), 'suffix', None):
                    model_short = Path(model_short).stem
                title_label.setText(f'Frame {frame_idx + 1} of {num_frames} | Cells: {num_cells} | Model: {model_short} | {file_name}\nParameters: {params_str}')
            cached_runs_combo = QComboBox()
            cached_runs_combo.setToolTip('Compare segmentation runs for this frame.\n• Current parameters — matches sliders/checkboxes now.\n• Other entries — earlier previews cached with different settings.')
            if embed_seg:
                # Keep cache combo compact so Reset/Re-detect stay fully visible on one nav row.
                cached_runs_combo.setMinimumWidth(120)
                cached_runs_combo.setMaximumWidth(200)
                cached_runs_combo.setMinimumContentsLength(12)
                cached_runs_combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                try:
                    cached_runs_combo.setSizeAdjustPolicy(
                        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                    )
                except Exception:
                    pass
            else:
                cached_runs_combo.setMinimumWidth(200)
                cached_runs_combo.setMaximumWidth(420)
            try:
                from bactomate.gui.styles import PREVIEW_NAV_COMBO_STYLESHEET
                cached_runs_combo.setStyleSheet(PREVIEW_NAV_COMBO_STYLESHEET + ' QComboBox { font-size: 11pt; }')
            except Exception:
                pass
            clear_cache_btn = QPushButton('Clear Cache')
            clear_cache_btn.setToolTip('Clear all cached segmentation results.\nUse this if you want to force recalculation of all frames.')
            style_compact_action_button(clear_cache_btn, horizontal_padding=20, dense=True)

            def refresh_cached_runs_combo():
                cur_frame = preview_dialog.current_frame
                keys_for_frame = [k for k in preview_dialog.segmentation_cache if k[0] == cur_frame]
                keys_for_frame.sort(key=lambda k: (k[1], k[2], k[3]))
                cached_runs_combo.blockSignals(True)
                cached_runs_combo.clear()
                cached_runs_combo.addItem('Current parameters', None)
                for k in keys_for_frame:
                    cached_runs_combo.addItem(format_omnipose_params_from_key(k), k)
                idx = 0
                if preview_dialog.display_cache_key is not None:
                    try:
                        idx = 1 + keys_for_frame.index(preview_dialog.display_cache_key)
                    except ValueError:
                        pass
                cached_runs_combo.setCurrentIndex(idx)
                cached_runs_combo.blockSignals(False)

            def on_cached_run_selected(index: int):
                key = cached_runs_combo.currentData()
                if key is None:
                    preview_dialog.display_cache_key = None
                    show_current_cache_or_unsegmented(preview_dialog.current_frame)
                else:
                    preview_dialog.display_cache_key = key
                    show_cached_run(key)

            def on_clear_cache():
                preview_dialog.segmentation_cache.clear()
                preview_dialog.display_cache_key = None
                if not embed_seg:
                    status_label.setText('Cache cleared')
                    status_label.setStyleSheet('color: #2196F3; font-weight: bold;')

                    def _restore_ready_status() -> None:
                        status_label.setText('Ready')
                        status_label.setStyleSheet('color: #00FF00; font-weight: bold;')
                    QTimer.singleShot(2000, _restore_ready_status)
                refresh_cached_runs_combo()
            clear_cache_btn.clicked.connect(on_clear_cache)
            cached_runs_combo.currentIndexChanged.connect(on_cached_run_selected)
            redetect_btn = QPushButton('Re-detect')
            redetect_btn.setToolTip(
                'Re-run cell segmentation and Spotiflow foci detection on the current frame\n'
                'using the parameters above (cell + Spotiflow). Segmentation is cached per\n'
                'parameter set; foci are cached per (source, frame, Spotiflow params).'
            )
            if not embed_seg:
                style_compact_action_button(redetect_btn, horizontal_padding=28)

            def on_redetect():
                self._commit_segmentation_parameter_spins()
                params = get_current_parameters()
                source_key = image_combo.currentData() if image_combo is not None else ('input', 0)
                cache_key = get_cache_key(preview_dialog.current_frame, params, source_key=source_key)
                if cache_key in preview_dialog.segmentation_cache:
                    del preview_dialog.segmentation_cache[cache_key]
                update_display(preview_dialog.current_frame, show_progress=True, force_recalc=True, source_key=source_key)
                try:
                    _compute_foci_peaks_for_current(preview_dialog.current_frame, allow_detect=True, force_recalc=True, background=True)
                except Exception:
                    get_logger().exception('Re-detect: foci detection failed')
            redetect_btn.clicked.connect(on_redetect)
            try:
                preview_dialog._on_redetect = on_redetect
            except Exception:
                pass

            embed_settings_scroll: Optional[QWidget] = None
            embed_action_trailer: Optional[QWidget] = None
            if embed_seg:
                settings_host = QWidget(preview_dialog)
                settings_outer = QVBoxLayout(settings_host)
                settings_outer.setContentsMargins(0, 0, 0, 0)
                settings_outer.setSpacing(8)
                (cell_card, embed_reset_btn) = self._build_embedded_cell_model_params_scroll(
                    settings_host, preview_dialog=preview_dialog
                )
                foci_strip = self._build_embedded_spotiflow_foci_strip(settings_host, preview_dialog=preview_dialog)
                sf_mount = QWidget(settings_host)
                sf_mount.setObjectName('EmbeddedSpotiflowParamsMount')
                sf_mount_lay = QVBoxLayout(sf_mount)
                sf_mount_lay.setContentsMargins(0, 0, 0, 0)
                sf_mount_lay.setSpacing(0)
                settings_outer.addWidget(cell_card)
                settings_outer.addWidget(foci_strip)
                settings_outer.addWidget(sf_mount)
                try:
                    preview_dialog._sf_settings_mount = sf_mount
                    preview_dialog._sf_settings_mount_lay = sf_mount_lay
                except Exception:
                    pass
                embed_settings_scroll = build_preview_settings_scroll(settings_host, embedded=True)
                embed_action_trailer = self._build_embedded_preview_action_bar(
                    preview_dialog, reset_btn=embed_reset_btn, redetect_btn=redetect_btn
                )

            def _on_preview_cell_params_changed(*_args) -> None:
                """Keep Pending parameters in sync when diameter/flow/mask change."""
                if getattr(self, '_embedded_segmentation_preview', None) is not preview_dialog:
                    return
                if getattr(preview_dialog, '_dialog_closed', False):
                    return
                try:
                    self._commit_segmentation_parameter_spins()
                    show_current_cache_or_unsegmented(preview_dialog.current_frame)
                except Exception:
                    get_logger().debug('Preview param refresh failed', exc_info=True)

            for _spin in (self.diameter, self.flow_threshold, self.mask_threshold):
                try:
                    _spin.valueChanged.connect(_on_preview_cell_params_changed)
                except Exception:
                    pass
            try:
                self.cluster.toggled.connect(_on_preview_cell_params_changed)
            except Exception:
                pass
            nav_layout = QVBoxLayout()
            nav_layout.setContentsMargins(10, 10, 10, 2)
            nav_layout.setSpacing(6)
            sf_params_row = QHBoxLayout()
            sf_params_row.setContentsMargins(0, 0, 0, 0)
            sf_params_row.setSpacing(8)
            bottom_layout = QHBoxLayout()
            bottom_layout.setContentsMargins(0, 0, 0, 0)
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_frames)
            frame_spin.setValue(1)
            frame_spin.setToolTip(f'Frame 1–{num_frames}. Type a number and press Enter to jump.')
            try:
                frame_spin.setVisible(False)
            except Exception:
                pass
            image_combo = QComboBox()
            image_combo.setMinimumContentsLength(26)
            image_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            ch_count = 1
            try:
                fr0 = images[0]
                if isinstance(fr0, np.ndarray) and fr0.ndim == 3 and (fr0.shape[0] < 10):
                    ch_count = int(fr0.shape[0])
            except Exception:
                ch_count = 1
            for ci in range(max(1, ch_count)):
                if ch_count <= 1:
                    image_combo.addItem('Input image', ('input', ci))
                else:
                    image_combo.addItem(f'Input channel {ci + 1}', ('input', ci))
            foci_sources = getattr(preview_dialog, '_foci_source_paths', None) or []
            detect_cb = getattr(self, 'detect_foci', None)
            detect_foci_on = detect_cb is not None and bool(detect_cb.isChecked())
            if detect_foci_on:
                for (si, fp) in enumerate(foci_sources):
                    image_combo.addItem(f'Foci {fp.name}', ('foci', si))
            image_combo.setToolTip('Display layer under the cell overlay for the currently previewed TIFF:\n• Input image / Input channel N — planes inside the loaded stack\n• Foci <name> — matched *_foci* sibling stacks (Spotiflow), not separate *_fi* files')
            try:
                image_combo.setVisible(False)
            except Exception:
                pass
            nav_preview_spec_builder = None
            try:
                from bactomate.gui.preview.navigation_controls import PreviewNavSpec, PreviewNavState, PreviewSource, PreviewNavigationControls
            except Exception:
                PreviewNavigationControls = None
            nav_ctrl = None
            if PreviewNavigationControls is not None:

                def _nav_preview_spec(include_foci_items: bool) -> PreviewNavSpec:
                    from bactomate.gui.filter_preview_utils import short_foci_stack_label
                    srcs: list[PreviewSource] = []
                    for ci in range(max(1, ch_count)):
                        label = 'Input image' if ch_count <= 1 else f'Input channel {ci + 1}'
                        srcs.append(PreviewSource(key=('input', ci), label=label, kind='image', resets_lut=True))
                    fps_nav = getattr(preview_dialog, '_foci_source_paths', None) or []
                    foci_ch_labels: tuple[str, ...] = ()
                    if include_foci_items and fps_nav:
                        foci_ch_labels = tuple((f'Foci {short_foci_stack_label(Path(fp))}' for fp in fps_nav))
                    if include_foci_items:
                        for (si, fp) in enumerate(fps_nav):
                            srcs.append(PreviewSource(key=('foci', si), label=f'Foci {fp.name}', kind='image', resets_lut=True))
                    use_per_channel = bool(foci_ch_labels)
                    return PreviewNavSpec(frame_count=int(max(1, num_frames)), image_sources=srcs, show_image=True, show_channel=False, show_z=False, show_cell_overlay_toggle=True, show_foci_overlay_toggle=bool(include_foci_items and fps_nav and (not use_per_channel)), cell_overlay_default=True, foci_overlay_available=bool(include_foci_items and fps_nav), foci_overlay_default=bool(include_foci_items and fps_nav), foci_channel_overlay_labels=foci_ch_labels, foci_channel_overlay_defaults=tuple((True for _ in foci_ch_labels)), debounce_ms=400)
                nav_preview_spec_builder = _nav_preview_spec
                cache_nav_inline = None
                if embed_seg:
                    cache_nav_inline = QWidget()
                    cache_nav_inline.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                    cnl = QHBoxLayout(cache_nav_inline)
                    cnl.setContentsMargins(0, 0, 0, 2)
                    cnl.setSpacing(6)
                    cnl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
                    pipe_lbl = QLabel('|')
                    pipe_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    pipe_lbl.setStyleSheet('color: #6a7a90; font-size: 12pt; font-weight: 300; padding: 0 4px;')
                    cnl.addWidget(pipe_lbl, 0)
                    cnl.addWidget(cached_runs_combo, 0)
                nav_ctrl = PreviewNavigationControls(
                    spec=_nav_preview_spec(detect_foci_on),
                    inline_trailer=embed_action_trailer,
                    inline_end=cache_nav_inline,
                    trailing_end=clear_cache_btn if embed_seg else None,
                )
                try:
                    nav_ctrl.set_state(PreviewNavState(source_key=image_combo.currentData(), frame_idx=int(getattr(preview_dialog, 'current_frame', 0) or 0)), emit=False)
                except Exception:
                    pass
                bottom_layout.addWidget(nav_ctrl)
                try:
                    preview_dialog._nav_ctrl = nav_ctrl
                except Exception:
                    pass
            if foci_sources:
                _sf_lbl_style = f"color: {UI_THEME['muted_text']}; font-size: 10pt; font-weight: 500; background: transparent;"
                _sf_cb_style = f"QCheckBox {{ font-size: 10pt; color: {UI_THEME['text']}; spacing: 6px; background: transparent; }}"

                def _spot_param_label(text: str) -> QLabel:
                    lb = QLabel(text)
                    lb.setStyleSheet(_sf_lbl_style)
                    lb.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                    return lb

                def _spot_col_sep() -> QFrame:
                    fr = QFrame()
                    fr.setFixedWidth(1)
                    fr.setMaximumWidth(1)
                    fr.setMinimumHeight(22)
                    fr.setStyleSheet(f"QFrame {{ background-color: {UI_THEME['field_border']}; border: none; margin-left: 6px; margin-right: 6px; }}")
                    return fr
                from bactomate.gui.filter_preview_utils import foci_channel_key_for_path, short_foci_stack_label
                self._ensure_spotiflow_channel_entries([Path(p) for p in foci_sources])
                sf_channel_lbl = _spot_param_label('Foci channel')
                sf_channel_lbl.setToolTip('Normalize, percentiles, threshold and min distance on this row apply to the selected foci stack only (per-channel overrides). Model and other options are shared.\nSwitch channel to tune each reporter separately.')
                sf_channel_combo = QComboBox(preview_dialog)
                sf_channel_combo.setMinimumContentsLength(12)
                sf_channel_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
                sf_channel_combo.setMinimumWidth(120)
                sf_channel_combo.setMaximumWidth(220)
                style_preview_nav_combo(sf_channel_combo)
                sf_channel_combo.blockSignals(True)
                try:
                    for fp in foci_sources:
                        tag = foci_channel_key_for_path(Path(fp))
                        sf_channel_combo.addItem(short_foci_stack_label(Path(fp)), tag)
                    for (ii, fp) in enumerate(foci_sources):
                        try:
                            sf_channel_combo.setItemData(ii, str(Path(fp).resolve()), Qt.ItemDataRole.ToolTipRole)
                        except OSError:
                            sf_channel_combo.setItemData(ii, str(fp), Qt.ItemDataRole.ToolTipRole)
                finally:
                    sf_channel_combo.blockSignals(False)
                foci_combo = sf_channel_combo
                try:
                    preview_dialog._sf_channel_label = sf_channel_lbl
                except Exception:
                    pass
                _ch_vis_init = bool(detect_foci_on)
                sf_channel_lbl.setVisible(_ch_vis_init)
                sf_channel_combo.setVisible(_ch_vis_init)
                first_ch = foci_channel_key_for_path(Path(foci_sources[0])) if foci_sources else '_default'
                preview_dialog._sf_channel_edit_key = first_ch
                p_sf = self._get_spotiflow_detection_params(channel_key=first_ch)
                sf_params_row.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                sf_norm_cb = QCheckBox('Normalize')
                sf_norm_cb.setChecked(bool(p_sf.get('normalize', True)))
                sf_norm_cb.setToolTip(
                    'Percentile-normalize the foci plane before Spotiflow.\n'
                    'pLow / pHigh are only used when this is on — editing them no longer force-enables Normalize.'
                )
                sf_norm_cb.setStyleSheet(_sf_cb_style)
                sf_params_row.addWidget(sf_norm_cb)
                sf_params_row.addWidget(_spot_param_label('pLow'))
                sf_plow = QDoubleSpinBox()
                sf_plow.setRange(0.0, 49.0)
                sf_plow.setDecimals(1)
                sf_plow.setSingleStep(0.5)
                sf_plow.setValue(float(p_sf.get('perc_low', 1.0) or 1.0))
                sf_plow.setFixedWidth(70)
                try:
                    sf_plow.setKeyboardTracking(False)
                except Exception:
                    pass
                style_preview_nav_spinbox(sf_plow)
                sf_plow.setToolTip('Lower percentile for Spotiflow normalization (used only when Normalize is on).')
                sf_params_row.addWidget(sf_plow)
                sf_params_row.addWidget(_spot_param_label('pHigh'))
                sf_phigh = QDoubleSpinBox()
                sf_phigh.setRange(51.0, 100.0)
                sf_phigh.setDecimals(1)
                sf_phigh.setSingleStep(0.5)
                sf_phigh.setValue(float(p_sf.get('perc_high', 99.8) or 99.8))
                sf_phigh.setFixedWidth(74)
                try:
                    sf_phigh.setKeyboardTracking(False)
                except Exception:
                    pass
                style_preview_nav_spinbox(sf_phigh)
                sf_phigh.setToolTip('Upper percentile for Spotiflow normalization (used only when Normalize is on).')
                sf_params_row.addWidget(sf_phigh)
                sf_params_row.addWidget(_spot_col_sep())
                def _csv_tip() -> str:
                    m = float(getattr(self, '_foci_csv_min_probability', 0.0) or 0.0)
                    return f'\nNote: a session raw-CSV minimum probability of {m:g} is also applied.' if m > 0 else ''
                sf_use_opt = QCheckBox('Optimized threshold')
                sf_use_opt.setChecked(bool(p_sf.get('use_optimized_threshold', True)))
                sf_use_opt.setToolTip('Per-channel detection threshold mode. When on, Spotiflow uses its model-optimized threshold and Prob thr is ignored.\nPeaks below the detection threshold are never written to the raw foci CSV; the Filtering stage probability filter can only remove more peaks later.' + _csv_tip())
                sf_use_opt.setStyleSheet(_sf_cb_style)
                sf_params_row.addWidget(sf_use_opt)
                sf_params_row.addWidget(_spot_param_label('Prob thr'))
                sf_prob = QDoubleSpinBox()
                sf_prob.setToolTip('Per-channel Spotiflow detection threshold (used when Optimized threshold is off).\nPeaks below it are never written to the raw foci CSV; the Filtering stage probability filter can only remove more peaks later.' + _csv_tip())
                sf_prob.setRange(0.0, 1.0)
                sf_prob.setDecimals(2)
                sf_prob.setSingleStep(0.05)
                _pt = p_sf.get('prob_thresh')
                sf_prob.setValue(float(_pt) if _pt is not None else 0.5)
                sf_prob.setFixedWidth(70)
                style_preview_nav_spinbox(sf_prob)
                sf_prob.setEnabled(not sf_use_opt.isChecked())
                sf_params_row.addWidget(sf_prob)
                sf_params_row.addWidget(_spot_col_sep())
                sf_params_row.addWidget(_spot_param_label('Min distance'))
                sf_min_dist = QSpinBox()
                sf_min_dist.setRange(1, 50)
                sf_min_dist.setSingleStep(1)
                sf_min_dist.setValue(int(p_sf.get('min_distance', 1) or 1))
                sf_min_dist.setFixedWidth(64)
                style_preview_nav_spinbox(sf_min_dist)
                sf_params_row.addWidget(sf_min_dist)
                sf_params_row.addWidget(_spot_col_sep())
                _sf_v = Qt.AlignmentFlag.AlignVCenter
                sf_params_row.addWidget(sf_channel_lbl, 0, _sf_v)
                sf_params_row.addWidget(sf_channel_combo, 0, _sf_v)
                sf_params_row.addStretch(1)
                _sf_hint = QLabel('Each foci channel has its own Spotiflow settings (prob thr, optimized threshold, …). View in the nav bar only changes the background image. Click Re-detect (below the image) after edits.')
                _sf_hint.setWordWrap(True)
                _sf_hint.setStyleSheet(f"QLabel {{ color: {UI_THEME['muted_text']}; font-size: 9pt; padding: 0 2px 4px 2px; }}")
                nav_layout.addWidget(_sf_hint)
                nav_layout.addLayout(sf_params_row)
                _sf_foci_status = QLabel('')
                _sf_foci_status.setWordWrap(True)
                _sf_foci_status.setStyleSheet(f"QLabel {{ color: {UI_THEME['muted_text']}; font-size: 9pt; font-style: italic; padding: 0 2px 2px 2px; }}")
                _sf_foci_status.hide()
                nav_layout.addWidget(_sf_foci_status)
                preview_dialog._sf_foci_status_label = _sf_foci_status
                _foci_sf_row = (sf_channel_lbl, sf_channel_combo, sf_norm_cb, sf_plow, sf_phigh, sf_use_opt, sf_prob, sf_min_dist)
                for _w in _foci_sf_row:
                    try:
                        _w.setEnabled(bool(detect_foci_on))
                    except Exception:
                        pass
                try:
                    sf_prob.setEnabled(bool(detect_foci_on) and (not sf_use_opt.isChecked()))
                except Exception:
                    pass
            else:
                foci_combo = None
                sf_channel_combo = None
                sf_norm_cb = None
                sf_plow = None
                sf_phigh = None
                sf_use_opt = None
                sf_prob = None
                sf_min_dist = None
            scale_label = QLabel('')
            is_updating_frame = False

            def go_to_frame(idx, force_recalc=False):
                nonlocal is_updating_frame
                if is_updating_frame:
                    return
                if not 0 <= idx < num_frames:
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
                    return
                is_updating_frame = True
                preview_dialog.current_frame = idx
                preview_dialog.display_cache_key = None
                frame_spin.blockSignals(True)
                frame_spin.setValue(idx + 1)
                frame_spin.blockSignals(False)
                try:
                    if force_recalc:
                        update_display(idx, show_progress=True, force_recalc=True)
                    else:
                        show_current_cache_or_unsegmented(idx)
                    try:
                        _det_on = bool(getattr(getattr(self, 'detect_foci', None), 'isChecked', lambda : False)())
                        _compute_foci_peaks_for_current(preview_dialog.current_frame, allow_detect=bool(_det_on), force_recalc=False)
                        refresh_image_display()
                    except Exception:
                        pass
                    refresh_cached_runs_combo()
                except Exception as e:
                    import traceback
                    error_msg = f'Could not display frame {idx + 1}:\n{str(e)}'
                    if hasattr(e, '__traceback__'):
                        error_msg += f'\n\n{traceback.format_exc()}'
                    show_message_dialog(preview_dialog, 'Preview Error', error_msg, kind='warning')
                finally:
                    is_updating_frame = False

            def on_frame_changed():
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    go_to_frame(frame_num)
                else:
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
            frame_spin.editingFinished.connect(on_frame_changed)
            preview_dialog._image_combo = image_combo
            if foci_sources:
                preview_dialog._foci_source_combo = foci_combo
                preview_dialog._sf_channel_combo = sf_channel_combo
                preview_dialog._sf_normalize_cb = sf_norm_cb
                preview_dialog._sf_perc_low = sf_plow
                preview_dialog._sf_perc_high = sf_phigh
                preview_dialog._sf_use_optimized = sf_use_opt
                preview_dialog._sf_prob_thresh = sf_prob
                preview_dialog._sf_min_distance = sf_min_dist

            def _set_display_frame_for_selection(frame_idx: int) -> None:
                """Set preview_dialog.current_display_frame based on image_combo selection."""
                sel = image_combo.currentData()
                if not isinstance(sel, tuple) or len(sel) < 2:
                    sel = ('input', 0)
                (kind, idx) = (sel[0], int(sel[1]))
                if kind == 'foci':
                    sources = getattr(preview_dialog, '_foci_sources', None) or []
                    if 0 <= idx < len(sources):
                        arr = sources[idx]
                        fi = max(0, min(int(frame_idx), int(arr.shape[0]) - 1))
                        try:
                            from bactomate.modules.image_stack_utils import coerce_plane_to_2d
                            preview_dialog.current_display_frame = np.asarray(coerce_plane_to_2d(np.asarray(arr[fi])), dtype=np.float32)
                        except Exception:
                            preview_dialog.current_display_frame = np.asarray(arr[fi], dtype=np.float32)
                    else:
                        return
                else:
                    frame = images[frame_idx]
                    if isinstance(frame, np.ndarray) and frame.ndim == 3 and (frame.shape[0] < 10):
                        ci = max(0, min(int(idx), int(frame.shape[0]) - 1))
                        preview_dialog.current_display_frame = frame[ci]
                    else:
                        preview_dialog.current_display_frame = frame if frame.ndim == 2 else frame[:, :, 0]

            def _on_image_changed():
                """Image source changed: refresh background, auto-rescale LUT, redraw."""
                try:
                    _set_display_frame_for_selection(preview_dialog.current_frame)
                    _compute_foci_peaks_for_current(preview_dialog.current_frame, allow_detect=False)
                    try:
                        lut_widget.set_frame(np.asarray(preview_dialog.current_display_frame), preserve_user_lut=False)
                    except Exception:
                        pass
                    refresh_image_display()
                    _sync_module_shell_candidate_pick()
                except Exception:
                    pass

            def _sync_module_shell_candidate_pick() -> None:
                """Keep module-shell file dropdown aligned with the segmented TIFF (input stack).

                View/Foci backgrounds only change display under the overlay — they must not move the
                shell dropdown away from the TIFF chosen as segmentation input.
                """
                if not embed_seg:
                    return
                sel = image_combo.currentData() if image_combo is not None else None
                if isinstance(sel, tuple) and len(sel) >= 2 and (str(sel[0]) == 'foci'):
                    return
                viewed_path = getattr(preview_dialog, '_segmentation_input_path', None) or str(file_path)
                if not viewed_path:
                    return
                try:
                    self._folder_preview_pick = str(Path(viewed_path).resolve())
                except OSError:
                    self._folder_preview_pick = str(viewed_path)
                try:
                    seg_tab = self._segmentation_shell_owner_tab()
                    if seg_tab is None:
                        return
                    surf = getattr(seg_tab, '_preview_surface', None)
                    if surf is None:
                        return
                    seg_tab.refresh_segmentation_candidate_list()
                    try:
                        surf.set_selected_candidate(str(Path(viewed_path).resolve()))
                    except OSError:
                        surf.set_selected_candidate(str(viewed_path))
                except Exception:
                    pass
            image_combo.currentIndexChanged.connect(_on_image_changed)
            if foci_sources:
                def _sf_ui_changed(*_a):
                    try:
                        self._sync_spotiflow_from_preview_dialog(preview_dialog)
                        try:
                            sf_prob.setEnabled(bool(foci_sources) and (not sf_use_opt.isChecked()))
                        except Exception:
                            pass
                        self._invalidate_foci_peak_cache_for_source(preview_dialog, self._active_foci_source_index(preview_dialog))
                        _compute_foci_peaks_for_current(preview_dialog.current_frame, allow_detect=False)
                        refresh_image_display()
                    except Exception:
                        pass

                def _sf_prob_spin_changed(*_a):
                    try:
                        if sf_use_opt.isChecked():
                            sf_use_opt.blockSignals(True)
                            sf_use_opt.setChecked(False)
                            sf_use_opt.blockSignals(False)
                        sf_prob.setEnabled(True)
                    except Exception:
                        pass
                    _sf_ui_changed()

                def _sync_sf_channel_combo_tooltip() -> None:
                    try:
                        ix = int(sf_channel_combo.currentIndex())
                        pth = sf_channel_combo.itemData(ix, Qt.ItemDataRole.ToolTipRole)
                        tag = sf_channel_combo.itemData(ix)
                        prefix = f'Spotiflow settings apply to this foci stack only.\nChannel tag: {tag}\n\nFull path:\n'
                        sf_channel_combo.setToolTip(prefix + (str(pth) if pth else '(unknown)'))
                    except Exception:
                        pass

                def _on_sf_channel_combo_changed(*_a) -> None:
                    try:
                        old_key = getattr(preview_dialog, '_sf_channel_edit_key', None)
                        if old_key:
                            self._save_spotiflow_widgets_to_channel(preview_dialog, str(old_key))
                        new_key = self._active_spotiflow_channel_key(preview_dialog)
                        self._load_spotiflow_widgets_for_channel(preview_dialog, new_key)
                        _sync_sf_channel_combo_tooltip()
                        _compute_foci_peaks_for_current(preview_dialog.current_frame, allow_detect=False)
                        refresh_image_display()
                    except Exception:
                        pass
                _sync_sf_channel_combo_tooltip()
                sf_channel_combo.currentIndexChanged.connect(_on_sf_channel_combo_changed)

                def _sync_normalize_spin_enabled(*_a) -> None:
                    on = bool(sf_norm_cb.isChecked())
                    for w in (sf_plow, sf_phigh):
                        try:
                            w.setEnabled(on)
                        except Exception:
                            pass

                def _on_sf_normalize_toggled(*_a) -> None:
                    _sync_normalize_spin_enabled()
                    _sf_ui_changed()

                _sync_normalize_spin_enabled()
                sf_norm_cb.toggled.connect(_on_sf_normalize_toggled)
                # Do not auto-check Normalize when editing percentiles — that fought the user
                # (uncheck → edit pHigh → Normalize snapped back on).
                sf_plow.valueChanged.connect(_sf_ui_changed)
                sf_phigh.valueChanged.connect(_sf_ui_changed)
                sf_use_opt.toggled.connect(_sf_ui_changed)
                sf_prob.valueChanged.connect(_sf_prob_spin_changed)
                sf_min_dist.valueChanged.connect(_sf_ui_changed)
            toolbar_layout = QHBoxLayout()
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            toolbar_layout.addWidget(scale_label)
            toolbar_layout.addStretch(1)
            toolbar_layout.addWidget(status_label)
            if not embed_seg:
                toolbar_layout.addWidget(redetect_btn)
            try:
                preview_dialog.set_primary_action(on_redetect, label='Re-detect')
                if embed_seg:
                    try:
                        preview_dialog.primary_button.setVisible(False)
                    except Exception:
                        pass
            except Exception:
                pass
            if not embed_seg:
                toolbar_layout.addWidget(clear_cache_btn)
                cr_toolbar_lbl = QLabel('Cached segmentation:')
                cr_toolbar_lbl.setStyleSheet('color: #aeb6c2; font-size: 10pt; font-weight: 600;')
                toolbar_layout.addWidget(cr_toolbar_lbl)
                toolbar_layout.addWidget(cached_runs_combo, 0)
            preview_dialog.top_section = None
            preview_dialog._spotiflow_section_mounted = False

            def _mount_spotiflow_section_if_needed() -> None:
                """Insert Foci/Spotiflow controls the first time Detect foci is enabled."""
                if getattr(preview_dialog, '_spotiflow_section_mounted', False):
                    return
                if getattr(preview_dialog, '_sf_channel_combo', None) is None:
                    return
                top_widget = QWidget()
                top_widget.setLayout(nav_layout)
                sec: QWidget
                try:
                    from bactomate.gui.preview.preview_window import CollapsibleSection
                    preview_dialog.top_section = CollapsibleSection('Spotiflow', top_widget, expanded=True, compact=True)
                    sec = preview_dialog.top_section
                except Exception:
                    preview_dialog.top_section = None
                    sec = top_widget
                embed_mount_lay = getattr(preview_dialog, '_sf_settings_mount_lay', None) if embed_seg else None
                if embed_mount_lay is not None:
                    try:
                        embed_mount_lay.addWidget(sec)
                    except Exception:
                        try:
                            layout.insertWidget(1, sec)
                        except Exception:
                            layout.addWidget(sec)
                else:
                    try:
                        layout.insertWidget(1, sec)
                    except Exception:
                        layout.addWidget(sec)
                preview_dialog._spotiflow_section_mounted = True
            if embed_seg and embed_settings_scroll is not None:
                try:
                    # Match Filtering: settings above canvas (after optional title).
                    insert_at = 1 if layout.indexOf(preview_dialog.title_label) >= 0 else 0
                    layout.insertWidget(insert_at, embed_settings_scroll)
                except Exception:
                    try:
                        layout.insertWidget(0, embed_settings_scroll)
                    except Exception:
                        pass
            if foci_sources and detect_foci_on:
                _mount_spotiflow_section_if_needed()
            elif getattr(preview_dialog, 'top_section', None) is not None:
                try:
                    preview_dialog.top_section.setVisible(False)
                except Exception:
                    pass
            bottom_widget = QWidget()
            bottom_outer = QVBoxLayout(bottom_widget)
            bottom_outer.setContentsMargins(0, 0, 0, 10)
            bottom_outer.setSpacing(8)
            if embed_seg:
                try:
                    scale_label.hide()
                except Exception:
                    pass
                nav_wrap = QWidget()
                nav_wrap_lay = QHBoxLayout(nav_wrap)
                nav_wrap_lay.setContentsMargins(0, 0, 0, 0)
                nav_wrap_lay.setSpacing(0)
                nav_wrap_lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
                if nav_ctrl is not None:
                    nav_wrap_lay.addWidget(nav_ctrl, 1)
                bottom_outer.addWidget(nav_wrap)
            else:
                bottom_outer.addLayout(bottom_layout)
                bottom_outer.addLayout(toolbar_layout)
            try:
                preview_dialog.set_navigation_controls(bottom_widget)
            except Exception:
                layout.addWidget(bottom_widget)
            if nav_ctrl is not None:

                def _find_image_index_by_key(k) -> int:
                    try:
                        for ii in range(int(image_combo.count())):
                            if image_combo.itemData(ii) == k:
                                return int(ii)
                    except Exception:
                        return 0
                    return 0

                def _on_nav_changed(st, reason: str) -> None:
                    try:
                        if reason in ('frame', 'init'):
                            go_to_frame(int(getattr(st, 'frame_idx', 0) or 0))
                        if reason in ('source', 'init'):
                            key = getattr(st, 'source_key', None)
                            idx = _find_image_index_by_key(key)
                            image_combo.blockSignals(True)
                            try:
                                image_combo.setCurrentIndex(int(idx))
                            finally:
                                image_combo.blockSignals(False)
                            _on_image_changed()
                    except Exception:
                        return
                try:
                    nav_ctrl.nav_changed.connect(_on_nav_changed)
                except Exception:
                    pass

                def _on_overlay_changed(key: str, checked: bool) -> None:
                    try:
                        preview_dialog.set_extra(str(key), bool(checked))
                        refresh_image_display()
                    except Exception:
                        pass

                def _on_foci_channel_overlay_changed(_idx: int, _checked: bool) -> None:
                    try:
                        refresh_image_display()
                    except Exception:
                        pass
                try:
                    nav_ctrl.overlay_changed.connect(_on_overlay_changed)
                    nav_ctrl.foci_channel_overlay_changed.connect(_on_foci_channel_overlay_changed)
                except Exception:
                    pass

                def _apply_detect_foci_preview_toggle() -> None:
                    """Foci enable/disable: nav rebuild on main thread; Spotiflow runs in background."""
                    builder = nav_preview_spec_builder
                    if builder is None or nav_ctrl is None:
                        return
                    checked = bool(self.detect_foci.isChecked())
                    if checked:
                        inp = getattr(preview_dialog, '_segmentation_input_path', None)
                        existing_paths = getattr(preview_dialog, '_foci_source_paths', None) or []
                        existing_sources = getattr(preview_dialog, '_foci_sources', None) or []
                        if inp and (not (existing_paths and existing_sources)):
                            try:
                                found = self._find_foci_sources_for_input(Path(inp))
                                loaded: list = []
                                loaded_paths: list[Path] = []
                                if found:
                                    from bactomate.modules.image_stack_utils import coerce_stack_to_tyx
                                    for fp in found:
                                        try:
                                            arr = coerce_stack_to_tyx(np.asarray(safe_imread(str(fp))))
                                            if arr is None or arr.size == 0:
                                                get_logger().warning('Foci file found but stack empty after load: %s', fp.name)
                                                continue
                                            loaded.append(arr)
                                            loaded_paths.append(fp)
                                        except Exception:
                                            get_logger().warning('Failed to load foci TIFF for preview: %s', fp, exc_info=True)
                                            continue
                                preview_dialog._foci_sources = loaded
                                preview_dialog._foci_source_paths = loaded_paths
                                if not loaded_paths:
                                    inp_p = Path(inp)
                                    show_message_dialog(self, 'No foci channels found', self._no_foci_channels_found_message([inp_p]), kind='info')
                            except Exception:
                                pass
                    fps_t = getattr(preview_dialog, '_foci_source_paths', None) or []
                    if not (checked and len(fps_t) > 1):
                        preview_dialog.set_extra('show_foci', bool(checked and fps_t))
                    if not checked:
                        _stop_foci_peaks_worker()
                        preview_dialog._foci_peaks_for_current = []
                        if isinstance(getattr(preview_dialog, '_foci_peaks_cache', None), dict):
                            preview_dialog._foci_peaks_cache.clear()
                    image_combo.blockSignals(True)
                    try:
                        image_combo.clear()
                        for ci in range(max(1, ch_count)):
                            if ch_count <= 1:
                                image_combo.addItem('Input image', ('input', ci))
                            else:
                                image_combo.addItem(f'Input channel {ci + 1}', ('input', ci))
                        if checked:
                            for (si, fp) in enumerate(fps_t):
                                image_combo.addItem(f'Foci {fp.name}', ('foci', si))
                    finally:
                        image_combo.blockSignals(False)
                    try:
                        nav_ctrl.set_spec(builder(bool(checked)), emit_init=False)
                    except Exception:
                        return
                    if not checked:
                        try:
                            from bactomate.gui.preview.navigation_controls import PreviewNavState
                            nav_ctrl.set_state(PreviewNavState(source_key=('input', 0), frame_idx=int(getattr(preview_dialog, 'current_frame', 0) or 0)), emit=True)
                        except Exception:
                            try:
                                image_combo.blockSignals(True)
                                image_combo.setCurrentIndex(0)
                            finally:
                                image_combo.blockSignals(False)
                            _on_image_changed()
                    try:
                        nav_ctrl.nav_changed.emit(nav_ctrl.state(), 'source')
                    except Exception:
                        pass
                    if checked and fps_t:
                        _mount_spotiflow_section_if_needed()
                    if checked and fps_t:
                        self._rebuild_preview_spotiflow_channel_combo(preview_dialog)
                    if checked and fps_t and (nav_ctrl is not None):
                        try:
                            nav_ctrl.set_foci_channel_overlay_visible([True] * len(fps_t))
                        except Exception:
                            pass
                    _sf_ch_on = bool(checked and fps_t)
                    for _ch_w in (getattr(preview_dialog, '_sf_channel_label', None), getattr(preview_dialog, '_sf_channel_combo', None)):
                        if _ch_w is not None:
                            try:
                                _ch_w.setVisible(_sf_ch_on)
                            except Exception:
                                pass
                    for _name in ('_foci_source_combo', '_sf_channel_combo', '_sf_normalize_cb', '_sf_perc_low', '_sf_perc_high', '_sf_use_optimized', '_sf_prob_thresh', '_sf_min_distance'):
                        _w = getattr(preview_dialog, _name, None)
                        if _w is not None:
                            try:
                                _w.setEnabled(bool(checked and fps_t))
                            except Exception:
                                pass
                    try:
                        uo = getattr(preview_dialog, '_sf_use_optimized', None)
                        pr = getattr(preview_dialog, '_sf_prob_thresh', None)
                        if pr is not None and uo is not None:
                            pr.setEnabled(bool(checked and fps_t) and (not uo.isChecked()))
                    except Exception:
                        pass
                    try:
                        _compute_foci_peaks_for_current(int(getattr(preview_dialog, 'current_frame', 0) or 0), allow_detect=bool(checked), force_recalc=bool(checked and fps_t), background=bool(checked and fps_t))
                    except Exception:
                        get_logger().exception('Detect foci toggle: foci peak update failed')
                    if not (checked and fps_t):
                        refresh_image_display()
                    ts = getattr(preview_dialog, 'top_section', None)
                    if ts is not None:
                        vis = bool(checked and fps_t)
                        ts.setVisible(vis)
                        if vis:
                            try:
                                ts.set_expanded(True)
                            except Exception:
                                pass

                def _queue_detect_foci_preview_toggle(_checked: bool) -> None:
                    """Defer heavy toggle handling so the checkbox repaints before TIFF I/O."""
                    if getattr(preview_dialog, '_detect_foci_apply_busy', False):
                        preview_dialog._detect_foci_requeue = True
                        return
                    timer = getattr(preview_dialog, '_detect_foci_apply_timer', None)
                    if timer is None:
                        timer = QTimer(preview_dialog)
                        timer.setSingleShot(True)
                        preview_dialog._detect_foci_apply_timer = timer

                        def _run_deferred_detect_foci_toggle() -> None:
                            if getattr(preview_dialog, '_detect_foci_apply_busy', False):
                                preview_dialog._detect_foci_requeue = True
                                return
                            preview_dialog._detect_foci_apply_busy = True
                            try:
                                _apply_detect_foci_preview_toggle()
                            finally:
                                preview_dialog._detect_foci_apply_busy = False
                                if bool(getattr(preview_dialog, '_detect_foci_requeue', False)):
                                    preview_dialog._detect_foci_requeue = False
                                    timer.start(0)
                        timer.timeout.connect(_run_deferred_detect_foci_toggle)
                    timer.stop()
                    timer.start(0)
                prev_dn = getattr(self, '_seg_preview_detect_conn', None)
                if prev_dn is not None:
                    try:
                        self.detect_foci.toggled.disconnect(prev_dn)
                    except Exception:
                        pass
                self._seg_preview_detect_conn = _queue_detect_foci_preview_toggle
                try:
                    self.detect_foci.toggled.connect(self._seg_preview_detect_conn)
                except Exception:
                    pass
                embed_cb = getattr(preview_dialog, '_embedded_detect_foci_cb', None)
                if embed_cb is not None:
                    prev_emb = getattr(self, '_seg_preview_embed_detect_conn', None)
                    if prev_emb is not None:
                        try:
                            embed_cb.toggled.disconnect(prev_emb)
                        except Exception:
                            pass
                    self._seg_preview_embed_detect_conn = _queue_detect_foci_preview_toggle
                    try:
                        embed_cb.toggled.connect(self._seg_preview_embed_detect_conn)
                    except Exception:
                        pass
            close_shortcut = QShortcut(QKeySequence.StandardKey.Close, preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            minimize_shortcut = QShortcut(QKeySequence('Ctrl+M'), preview_dialog)
            minimize_shortcut.activated.connect(preview_dialog.showMinimized)
            frame0 = images[0]
            try:
                _set_display_frame_for_selection(0)
                display_frame0 = getattr(preview_dialog, 'current_display_frame', None)
            except Exception:
                display_frame0 = None
            if display_frame0 is None:
                if frame0.ndim == 3:
                    display_frame0 = frame0[0] if frame0.shape[0] < 10 else frame0[:, :, 0]
                else:
                    display_frame0 = frame0
            empty_mask = np.zeros(display_frame0.shape, dtype=np.int32)
            (disp0, _) = scale_for_preview(display_frame0, empty_mask, max_size=PREVIEW_SEG_MAX_SIZE)
            empty_scaled = np.zeros(disp0.shape, dtype=np.int32)
            initial_overlay = build_segmentation_overlay(disp0, empty_scaled, alpha=0.0)
            initial_pixmap = to_qpixmap(initial_overlay)
            (h0, w0) = initial_overlay.shape[:2]
            image_canvas.set_pixmap(initial_pixmap, (w0, h0), reset_view=True)
            file_name = Path(file_path).name
            params0 = get_current_parameters()
            model_name0 = Path(params0['pretrained_model']).stem if params0.get('pretrained_model') else params0.get('model_type', '—')
            params_str0 = format_omnipose_params_from_params(params0)
            title_label.setText(f'Frame 1 of {num_frames} | Segmenting... | Model: {model_name0} | {file_name}\nParameters: {params_str0}')
            status_label.setText('Segmenting first frame...')
            loading_progress.close()
            self._set_segmentation_buttons_busy(False)
            if not bool(self.detect_foci.isChecked()):
                try:
                    refresh_image_display()
                except Exception:
                    pass

            def _on_preview_finished(_result: int=0, pd=preview_dialog) -> None:
                self._on_embedded_segmentation_preview_closed()
                self._sync_spotiflow_from_preview_dialog(pd)
            preview_dialog.finished.connect(_on_preview_finished)
            self._restore_seg_tuning_to_left_panel()
            prev_embed = getattr(self, '_embedded_segmentation_preview', None)
            if prev_embed is not None and prev_embed is not preview_dialog:
                try:
                    prev_embed.close()
                    prev_embed.deleteLater()
                except Exception:
                    pass
            self._embedded_segmentation_preview = preview_dialog
            try:
                seg_tab = self._segmentation_shell_owner_tab()
                if seg_tab is not None and hasattr(seg_tab, '_dismiss_embedded_previews'):
                    seg_tab._dismiss_embedded_previews(except_owner=self)
            except Exception:
                pass
            if mount is not None:
                lay = mount.layout()
                if lay is None:
                    lay = QVBoxLayout(mount)
                    lay.setContentsMargins(0, 0, 0, 0)
                    mount.setLayout(lay)
                surf = getattr(self, '_preview_surface', None)
                ph = getattr(surf, '_placeholder', None) if surf is not None else None
                while lay.count():
                    item = lay.takeAt(0)
                    w = item.widget()
                    if w is None:
                        continue
                    if ph is not None and w is ph:
                        w.hide()
                        w.setParent(None)
                        continue
                    detach_mount_widget(w)
                lay.addWidget(preview_dialog, 1)
                if surf is not None:
                    try:
                        surf.set_preview_chrome_visible(len(folder_preview_files) > 1)
                    except Exception:
                        pass
                try:
                    seg_tab = self._segmentation_shell_owner_tab()
                    if seg_tab is not None:

                        def _refresh_shell_after_preview_open() -> None:
                            try:
                                seg_tab.refresh_segmentation_candidate_list()
                                surf2 = getattr(seg_tab, '_preview_surface', None)
                                if surf2 is not None:
                                    try:
                                        surf2.set_preview_chrome_visible(len(folder_preview_files) > 1)
                                    except Exception:
                                        pass
                                    try:
                                        surf2.set_selected_candidate(str(Path(file_path).resolve()))
                                    except OSError:
                                        surf2.set_selected_candidate(str(file_path))
                                _sync_module_shell_candidate_pick()
                            except Exception:
                                pass
                        QTimer.singleShot(0, _refresh_shell_after_preview_open)
                except Exception:
                    pass
            preview_dialog.show()
            if mount is None:
                preview_dialog.raise_()
                preview_dialog.activateWindow()

            def start_preview():
                if mount is None:
                    preview_dialog.raise_()
                    preview_dialog.activateWindow()
                update_display(0, show_progress=True)
                try:
                    _set_display_frame_for_selection(0)
                    _det_on = bool(getattr(getattr(self, 'detect_foci', None), 'isChecked', lambda : False)())
                    _has_foci = bool(getattr(preview_dialog, '_foci_source_paths', None) or [])
                    _compute_foci_peaks_for_current(0, allow_detect=bool(_det_on), force_recalc=bool(_det_on and _has_foci), background=bool(_det_on and _has_foci))
                    if not (_det_on and _has_foci):
                        refresh_image_display()
                except Exception:
                    pass
            QTimer.singleShot(80, start_preview)
        except ImportError as e:
            try:
                loading_progress.close()
            except Exception:
                pass
            self._set_segmentation_buttons_busy(False)
            show_message_dialog(self, 'Preview Unavailable', format_omnipose_preview_import_error(str(e)), kind='warning')
        except Exception as e:
            try:
                loading_progress.close()
            except Exception:
                pass
            self._set_segmentation_buttons_busy(False)
            error_msg = str(e)
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                show_message_dialog(self, 'Preview Error', 'PyTorch initialization error detected.\n\nThis is a known issue with PyTorch in some environments.\nThe preview feature may not work.\n\nTry:\n1. Restart the application\n2. Run from source: python main.py\n3. Use the processing features instead of preview', kind='warning')
            elif 'Versioning for this project requires' in error_msg or 'pbr.version.VersionInfo' in error_msg or 'mgen' in error_msg:
                show_message_dialog(self, 'Preview Error', "Package versioning error detected.\n\nThis is a known issue with the mgen package in some environments.\nThe preview feature may not work in the packaged application.\n\nTry:\n1. Restart the application\n2. Run from source: python main.py\n3. Use the processing features instead of preview\n\nThis error is harmless and doesn't affect actual processing.", kind='warning')
            elif 'TorchScript requires source access' in error_msg or "Can't get source for" in error_msg:
                show_message_dialog(self, 'Preview Error', "TorchScript compilation error detected.\n\nThis is a known issue with PyTorch's TorchScript in some environments.\nTorchScript requires access to original source files which aren't\navailable in packaged applications.\n\nTry:\n1. Restart the application (the runtime hook should fix this)\n2. Run from source: python main.py\n3. Use the processing features instead of preview\n\nNote: This has been fixed in the latest build. Please rebuild the app.", kind='warning')
            elif 'WinError 1114' in error_msg or 'c10.dll' in error_msg or 'DLL initialization routine failed' in error_msg or ('dynamic link library' in error_msg.lower()):
                show_message_dialog(self, 'Preview Error', 'PyTorch failed to load on Windows (DLL error).\n\nTo keep using the GPU:\n\n1. Install Microsoft Visual C++ Redistributable (64-bit, 2015–2022) from Microsoft.\n\n2. Reinstall PyTorch with the correct CUDA index for your driver.\n   Check version: nvidia-smi\n   Example for CUDA 12.8:\n   pip uninstall torch torchvision torchaudio -y\n   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n\n   For 12.4 use cu124, for 11.8 use cu118. See INSTALL_AND_LAUNCH_windows.md.', kind='warning')
            else:
                show_message_dialog(self, 'Preview Error', f'Could not preview segmentation:\n{error_msg}', kind='error')
            import traceback
            traceback.print_exc()
        except BaseException as e:
            import traceback
            error_msg = f'Unexpected error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}'
            show_message_dialog(self, 'Preview Error', error_msg, kind='error')
        except BaseException as e:
            get_logger().exception('Critical error in preview segmentation')
            import traceback
            error_msg = f'Critical error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}'
            try:
                show_message_dialog(self, 'Preview Error', error_msg, kind='error')
            except Exception:
                get_logger().exception('Could not show preview error dialog')
                print(f'CRITICAL ERROR in preview_segmentation: {error_msg}', file=sys.stderr)

    def _open_check_pipeline_dialog(self):
        """Open diagnostic dialog: check current model (or compare two), optional training dir, optional image. Can download built-in models if needed."""
        from PyQt6.QtWidgets import QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout
        from PyQt6.QtGui import QFont
        from bactomate.scripts.check_omnipose_pipeline import get_checkable_model_path, _check_model, _check_training_data, _check_inference, format_model_report, format_training_report, format_inference_report, run_checkpoint_comparison
        d = QDialog(self)
        d.setWindowTitle('Check Omnipose pipeline')
        d.setMinimumSize(620, 520)
        layout = QVBoxLayout()
        last_dir = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_SEGMENTATION) or str(Path.home())
        model_names = self._get_available_models()
        download_cb = QCheckBox('Download built-in models if needed')
        download_cb.setChecked(True)
        download_cb.setToolTip('When a built-in model (e.g. bact_phase_omni) is selected, download it into the cache first so it can be checked or compared.')
        layout.addWidget(download_cb)
        compare_cb = QCheckBox('Compare two models')
        compare_cb.setToolTip('Run model check on two models (Model A and Model B) and show both reports. Use to compare user vs user or user vs built-in.')
        layout.addWidget(compare_cb)
        single_group = QGroupBox('Single check (current tab model)')
        single_layout = QVBoxLayout()
        display = self.model_combo.currentText()
        (model_type, user_path) = self._resolve_model_selection(display)
        path = get_checkable_model_path(model_type, user_path, ensure_builtin_cached=False)
        model_label = QLabel(f'Current model: {display}')
        if path is None and (model_type or user_path):
            if model_type == CELLPOSE_SAM_MODEL:
                model_label.setText(f'Current model: {display} (uses official cellpose + cpsam)')
            elif model_type:
                model_label.setText(f"Current model: {display} (not cached — enable 'Download built-in…' and run, or run segmentation once)")
            else:
                model_label.setText(f'Current model: {display} (file not found)')
        model_label.setWordWrap(True)
        single_layout.addWidget(model_label)
        single_group.setLayout(single_layout)
        layout.addWidget(single_group)
        compare_group = QGroupBox('Compare two models')
        compare_layout = QHBoxLayout()
        compare_layout.addWidget(QLabel('Model A:'))
        combo_a = QComboBox()
        combo_a.addItems(model_names)
        compare_layout.addWidget(combo_a)
        compare_layout.addWidget(QLabel('Model B:'))
        combo_b = QComboBox()
        combo_b.addItems(model_names)
        compare_layout.addWidget(combo_b)
        compare_group.setLayout(compare_layout)
        layout.addWidget(compare_group)
        compare_group.setVisible(False)

        def on_compare_toggled(checked):
            compare_group.setVisible(checked)
            single_group.setVisible(not checked)
        compare_cb.toggled.connect(on_compare_toggled)
        single_group.setVisible(True)
        train_group = QGroupBox('Training data (optional)')
        train_layout = QVBoxLayout()
        train_dir_edit = QLineEdit()
        train_dir_edit.setPlaceholderText('Path to folder with image/mask pairs')
        train_row = QHBoxLayout()
        train_row.addWidget(train_dir_edit)
        browse_train_btn = QPushButton('Browse…')
        browse_train_btn.clicked.connect(lambda : _browse_train())
        train_row.addWidget(browse_train_btn)
        train_layout.addLayout(train_row)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel('Image filter:'))
        img_filter_edit = QLineEdit()
        img_filter_edit.setPlaceholderText('e.g. _phase')
        img_filter_edit.setFixedWidth(100)
        filter_row.addWidget(img_filter_edit)
        filter_row.addWidget(QLabel('Mask filter:'))
        mask_filter_edit = QLineEdit()
        mask_filter_edit.setText('_mask')
        mask_filter_edit.setFixedWidth(100)
        filter_row.addWidget(mask_filter_edit)
        train_layout.addLayout(filter_row)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        def _browse_train():
            path = QFileDialog.getExistingDirectory(d, 'Select training directory', last_dir)
            if path:
                train_dir_edit.setText(path)
        image_group = QGroupBox('Inference test image (optional)')
        image_layout = QHBoxLayout()
        image_edit = QLineEdit()
        image_edit.setPlaceholderText('Path to an image to test with current / Model A')
        image_layout.addWidget(image_edit)

        def browse_image():
            (path, _) = QFileDialog.getOpenFileName(d, 'Select image', last_dir, 'Images (*.tif *.tiff *.png);;All (*)')
            if path:
                image_edit.setText(path)
        browse_image_btn = QPushButton('Browse…')
        browse_image_btn.clicked.connect(browse_image)
        image_layout.addWidget(browse_image_btn)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        report = QTextEdit()
        report.setReadOnly(True)
        report.setFont(QFont('Monospace', 9))
        report.setPlaceholderText("Click 'Run diagnostic'. Single: checks current tab model. Compare: checks Model A and Model B. Enable 'Download built-in…' to cache Omnipose models first.")
        layout.addWidget(report)

        class DiagnosticWorker(QThread):
            result_ready = pyqtSignal(str)

            def __init__(self, tab, is_compare, name_a, name_b, do_download, train_dir_str, img_filter, mask_filter, image_str, single_display):
                super().__init__()
                self.tab = tab
                self.is_compare = is_compare
                self.name_a = name_a
                self.name_b = name_b
                self.do_download = do_download
                self.train_dir_str = train_dir_str
                self.img_filter = img_filter
                self.mask_filter = mask_filter
                self.image_str = image_str
                self.single_display = single_display

            def run(self):
                lines = []
                primary_model_path = None
                try:
                    if self.is_compare:
                        (type_a, path_a) = self.tab._resolve_model_selection(self.name_a)
                        (type_b, path_b) = self.tab._resolve_model_selection(self.name_b)
                        try:
                            model_path_a = get_checkable_model_path(type_a, path_a, ensure_builtin_cached=self.do_download and bool(type_a))
                        except Exception as e:
                            model_path_a = None
                            lines.append(f'Model A ({self.name_a}): failed to resolve — {e}')
                            lines.append('')
                        try:
                            model_path_b = get_checkable_model_path(type_b, path_b, ensure_builtin_cached=self.do_download and bool(type_b))
                        except Exception as e:
                            model_path_b = None
                            lines.append(f'Model B ({self.name_b}): failed to resolve — {e}')
                            lines.append('')
                        if model_path_a is None and (not any(('Model A' in ln for ln in lines))):
                            lines.append(f'Model A ({self.name_a}): no file (built-in not cached or user file missing).')
                            lines.append('')
                        elif model_path_a is not None:
                            try:
                                r = _check_model(Path(model_path_a))
                                lines.append(f'=== Model A: {self.name_a} ===')
                                lines.append(format_model_report(r))
                                lines.append('')
                            except Exception as e:
                                lines.append(f'Model A check error: {e}')
                                lines.append('')
                            import gc
                            gc.collect()
                        if model_path_b is None and (not any(('Model B' in ln for ln in lines))):
                            lines.append(f'Model B ({self.name_b}): no file (built-in not cached or user file missing).')
                            lines.append('')
                        elif model_path_b is not None:
                            try:
                                r = _check_model(Path(model_path_b))
                                lines.append(f'=== Model B: {self.name_b} ===')
                                lines.append(format_model_report(r))
                                lines.append('')
                            except Exception as e:
                                lines.append(f'Model B check error: {e}')
                                lines.append('')
                        primary_model_path = model_path_a if model_path_a is not None else model_path_b
                    else:
                        (model_type, user_path) = self.tab._resolve_model_selection(self.single_display)
                        try:
                            primary_model_path = get_checkable_model_path(model_type, user_path, ensure_builtin_cached=self.do_download and bool(model_type))
                        except Exception as e:
                            primary_model_path = None
                            lines.append(f'Model resolve error: {e}')
                            lines.append('')
                        if primary_model_path is not None:
                            try:
                                r = _check_model(Path(primary_model_path))
                                lines.append(format_model_report(r))
                                lines.append('')
                            except Exception as e:
                                lines.append(f'Model check error: {e}')
                                lines.append('')
                        elif model_type == CELLPOSE_SAM_MODEL:
                            lines.append('Cellpose-SAM: uses official cellpose package (cpsam). No .pth file to check.')
                            lines.append('')
                        elif not any(('Model resolve error' in ln for ln in lines)):
                            lines.append('Model check skipped: no model file (user model missing or built-in not cached).')
                            lines.append('')
                    compare_paths = []
                    compare_labels = []
                    if self.is_compare:
                        if model_path_a is not None and Path(model_path_a).is_file():
                            compare_paths.append(Path(model_path_a))
                            compare_labels.append(self.name_a)
                        if model_path_b is not None and Path(model_path_b).is_file():
                            compare_paths.append(Path(model_path_b))
                            compare_labels.append(self.name_b)
                    elif primary_model_path is not None and Path(primary_model_path).is_file():
                        compare_paths.append(Path(primary_model_path))
                        compare_labels.append(self.single_display)
                    if len(compare_paths) >= 1:
                        try:
                            lines.append(run_checkpoint_comparison(compare_paths, compare_labels, brief=True))
                            lines.append('')
                        except Exception as e:
                            lines.append(f'Checkpoint comparison error: {e}')
                            lines.append('')
                    if self.train_dir_str and Path(self.train_dir_str).is_dir():
                        try:
                            r = _check_training_data(Path(self.train_dir_str), img_filter=self.img_filter or '', mask_filter=self.mask_filter or '_mask', dry_run=True)
                            lines.append(format_training_report(r))
                            lines.append('')
                        except Exception as e:
                            lines.append(f'Training data check error: {e}')
                            lines.append('')
                    if self.image_str and Path(self.image_str).is_file():
                        if primary_model_path is None:
                            lines.append('Inference check skipped: no model file available.')
                        else:
                            try:
                                r = _check_inference(Path(primary_model_path), Path(self.image_str))
                                lines.append(format_inference_report(r))
                            except Exception as e:
                                lines.append(f'Inference check error: {e}')
                    elif self.image_str:
                        lines.append('Inference check skipped: image path is not a file.')
                        lines.append('')
                except Exception as e:
                    import traceback
                    lines.append(f'Diagnostic error: {e}')
                    lines.append('')
                    lines.append(traceback.format_exc())
                self.result_ready.emit('\n'.join(lines))
        worker = [None]

        def run_diagnostic():
            report.clear()
            report.append('Running diagnostic in background…')
            run_btn.setEnabled(False)
            w = DiagnosticWorker(self, compare_cb.isChecked(), combo_a.currentText(), combo_b.currentText(), download_cb.isChecked(), train_dir_edit.text().strip(), img_filter_edit.text().strip() or '', mask_filter_edit.text().strip() or '_mask', image_edit.text().strip(), self.model_combo.currentText())
            worker[0] = w

            def on_done(text):
                report.clear()
                report.setText(text)
                run_btn.setEnabled(True)
            w.result_ready.connect(on_done)
            w.finished.connect(lambda : run_btn.setEnabled(True))
            w.start()
        run_btn = QPushButton('Run diagnostic')
        run_btn.clicked.connect(run_diagnostic)
        layout.addWidget(run_btn)
        d.setLayout(layout)
        d.exec()

    def _commit_segmentation_parameter_spins(self) -> None:
        """Apply any in-progress QDoubleSpinBox line-edit text before reading values."""
        for w in (self.diameter, self.flow_threshold, self.mask_threshold):
            try:
                if w is not None:
                    w.interpretText()
            except RuntimeError:
                pass
        self._commit_spotiflow_preview_spins()

    def _get_effective_diameter_pixels(self) -> float:
        """Return diameter in pixels (the single Diameter field)."""
        try:
            return float(self.diameter.value())
        except RuntimeError:
            return 0.0

    def _get_run_diameter_and_flow(self) -> tuple[float, float]:
        """
        Return (diameter, flow_threshold) to pass to the batch script when running segmentation.
        Uses GUI values as-is; no overrides for custom models (Omnipose defaults/behavior unchanged).
        """
        display = self.model_combo.currentText()
        (model_type, pretrained_path) = self._resolve_model_selection(display)
        diameter = self._get_effective_diameter_pixels()
        try:
            flow = float(self.flow_threshold.value())
        except RuntimeError:
            flow = 0.4
        return (float(diameter), flow)

    def _get_effective_mask_threshold(self) -> float:
        """Return mask/cellprob threshold for model.eval(); clamp SAM to the documented logit range."""
        display = self.model_combo.currentText() or ''
        (model_type, _) = self._resolve_model_selection(display)
        try:
            v = float(self.mask_threshold.value())
        except RuntimeError:
            v = 0.0
        if model_type == CELLPOSE_SAM_MODEL:
            return max(MASK_CELLPROB_SAM_MIN, min(MASK_CELLPROB_SAM_MAX, v))
        return v

    def _open_in_file_manager(self, path: Path) -> None:
        """Open the given path (file or directory) in the OS default file manager."""
        path = Path(path)
        if not path.exists():
            return
        try:
            if sys.platform == 'darwin':
                subprocess.run(['open', str(path)], check=False)
            elif sys.platform == 'win32':
                subprocess.run(['explorer', str(path)], check=False)
            else:
                subprocess.run(['xdg-open', str(path)], check=False)
        except Exception:
            get_logger().debug('Could not open path in file manager', exc_info=True)

    def _validate_for_run(self) -> list[str]:
        """Return a list of error messages; if non-empty, do not start the job."""
        errors: list[str] = []
        if not self.file_path.text():
            errors.append('Please select a TIFF file or folder first.')
            return errors
        path = Path(self.file_path.text())
        if not path.exists():
            errors.append(f'The selected path does not exist: {path}')
            return errors
        script_path = BATCH_PROCESS_TIFF_SCRIPT
        if not script_path.exists():
            errors.append(f'Segmentation script not found: {script_path}')
        if self._get_effective_diameter_pixels() < 0:
            errors.append('Diameter must be ≥ 0 (use 0 for auto-detect).')
        return errors

    def process_file_or_folder(self):
        self._segmentation_in_progress = True
        errors = self._validate_for_run()
        if errors:
            show_message_dialog(self, 'Cannot Start', 'Please fix the following:\n\n' + '\n'.join((f'• {e}' for e in errors)), kind='warning')
            self._segmentation_in_progress = False
            return
        self._set_segmentation_buttons_busy(True, action='process', progress_label='Cancel')
        self._update_segmentation_button_progress(3, '')
        script_path_str = str(BATCH_PROCESS_TIFF_SCRIPT)
        is_folder = self._is_folder_mode()
        if not is_folder:
            script_path = script_path_str
            input_file = Path(self.file_path.text())
            self._warn_no_foci_channels_if_enabled([input_file])
            base_dir = input_file.parent
            if base_dir.name.startswith('Tif_'):
                output_dir = base_dir / self._get_omnipose_segmentation_folder_name()
            else:
                output_dir = base_dir / f'{input_file.stem}_omnipose_segmentation'
            (_mt, _path) = self._resolve_model_selection(self.model_combo.currentText())
            if _path is not None:
                model_arg = prepare_user_pretrained_loader_path(_path)
            else:
                model_arg = _mt or self.model_combo.currentText()
            (run_diameter, run_flow) = self._get_run_diameter_and_flow()
            default_batch_size = '16'
            cmd_args = [self.file_path.text(), '--output', str(output_dir), '--model', model_arg, '--diameter', str(run_diameter), '--flow-threshold', str(run_flow), '--mask-threshold', str(self._get_effective_mask_threshold()), '--batch-size', default_batch_size]
            if not self.use_gpu.isChecked():
                cmd_args.append('--no-gpu')
            if self.cluster.isChecked():
                cmd_args.append('--cluster')
            self._last_output_dir = output_dir
            self.parent_window.log_output(f'Starting single file processing: {self.file_path.text()}\n')
            self.parent_window.log_output(f'Output directory: {output_dir}\n')
            try:
                self.parent_window.log_output(f'Omnipose params: batch_size={default_batch_size} use_gpu={self.use_gpu.isChecked()} cluster={self.cluster.isChecked()} overlay={self.create_overlay.isChecked()}\n')
            except Exception:
                pass
            self._log_segmentation_run_params(selected_files=[input_file])
        else:
            folder = Path(self.file_path.text())
            tiff_files = sorted([f for f in list_tiff_files(folder, max_files=5000) if self._is_segmentation_input_file(f)], key=lambda x: x.name.lower())
            if not tiff_files:
                show_message_dialog(self, 'No TIFF Files Found', 'No .tif / .tiff files found under the selected folder (recursive scan).\n\nChoose the folder that contains your TIFF inputs (e.g. Tif_files/), or a parent folder that includes them in a subfolder.', kind='warning')
                self._segmentation_in_progress = False
                self._set_segmentation_buttons_busy(False)
                return
            detect_foci_on = bool(getattr(self, 'detect_foci', None) is not None and self.detect_foci.isChecked())
            if detect_foci_on:
                seg_message = (
                    f'Select TIFFs to segment ({len(tiff_files)} listed).\n\n'
                    'Detect foci is enabled: phase/input TIFFs are preferred. '
                    'Plain "*_fi_*" fluorescence files stay available for fluor models, '
                    'but "*_foci*" / "*_fi_foci*" stacks are omitted (Spotiflow companions, not cell inputs). '
                    'When a phase TIFF is present, "*_fi_*" files are not pre-selected.\n\n'
                    'The right side shows only currently selected inputs plus their Spotiflow companion TIFFs.'
                )
            else:
                seg_message = f'Select files to process (all {len(tiff_files)} files selected by default):'
            default_checked = self._default_segmentation_selection_predicate(tiff_files)
            if detect_foci_on:
                result = show_linked_mask_selection_dialog(
                    self,
                    tiff_files,
                    companion_for_mask=self._companions_for_segmentation_input,
                    title='Select Files for Segmentation',
                    message=seg_message,
                    left_header='Selected segmentation inputs',
                    right_header='Segmentation input + Spotiflow companions',
                    filter_label_text='Select inputs containing:',
                    initial_filename_filter=getattr(self, '_segment_filename_filter', '') or '',
                    return_filename_filter=True,
                    default_checked=default_checked,
                )
            else:
                result = show_file_selection_dialog(self, tiff_files, title='Select Files for Segmentation', message=seg_message, initial_filename_filter=getattr(self, '_segment_filename_filter', '') or '', return_filename_filter=True, default_checked=default_checked)
            (selected_files, filter_used) = result if isinstance(result, tuple) else (result, '')
            if filter_used is not None:
                self._segment_filename_filter = filter_used or ''
            if selected_files is None:
                self._segmentation_in_progress = False
                self._set_segmentation_buttons_busy(False)
                return
            if len(selected_files) == 0:
                show_message_dialog(self, 'No Files Selected', 'Please select at least one file to process.', kind='warning')
                self._segmentation_in_progress = False
                self._set_segmentation_buttons_busy(False)
                return
            self._warn_no_foci_channels_if_enabled([Path(p) for p in selected_files])
            script_path = script_path_str
            self.folder_files = selected_files
            self.current_file_index = 0
            self.current_file_progress = 0.0
            n_files = len(selected_files)
            first_name = Path(selected_files[0]).name if selected_files else '…'
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)
            self.parent_window.progress_bar.setValue(0)
            if hasattr(self.parent_window, 'set_status_message'):
                prefix = self._segmentation_status_prefix()
                self.parent_window.set_status_message(f'{prefix}: file 1 of {n_files} — {first_name}')
            self._update_segmentation_button_progress(3, f'file 1/{n_files}')
            self.parent_window.log_output(f'Starting batch folder processing: {len(selected_files)} file(s) selected from {self.file_path.text()}\n')
            self._log_segmentation_run_params(selected_files=list(selected_files))
            # Match where masks are actually written (TIFF parent, often Tif_files/ under a parent pick).
            try:
                first_tiff = Path(selected_files[0])
                self._last_output_dir = self.resolve_segmentation_output_dir_for_tiff(
                    first_tiff, self._get_omnipose_segmentation_folder_name()
                )
            except Exception:
                self._last_output_dir = folder / self._get_omnipose_segmentation_folder_name() if folder.name.startswith('Tif_') else folder
            self._process_next_segmentation_file(script_path)
            return
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        if hasattr(self.parent_window, 'set_status_message'):
            input_path = Path(self.file_path.text()) if self.file_path.text() else None
            name = input_path.name if input_path else '…'
            self.parent_window.set_status_message(f'{self._segmentation_status_prefix()}: {name}')
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()

    def _process_next_segmentation_file(self, script_path):
        """Process next file in folder mode (segmentation)"""
        if self.current_file_index >= len(self.folder_files):
            self._reset_process_btn_idle()
            self._segmentation_in_progress = False
            self.parent_window.progress_bar.setVisible(False)
            if hasattr(self.parent_window, 'set_status_message'):
                self.parent_window.set_status_message('')
            self.parent_window.log_output(f'\n✓ Segmentation complete for all {len(self.folder_files)} file(s)!\n')
            self._performed_this_session = True
            self.workflow_step_finished.emit(True)
            from bactomate.gui.widgets import show_task_complete_dialog
            seg_folder = self._get_omnipose_segmentation_folder_name()
            out_desc = f'Output: {seg_folder} folder (one mask file per image).' if self.folder_files and Path(self.folder_files[0]).parent.name.startswith('Tif_') else 'Output: one _omnipose_segmentation subfolder per file (masks and overlay).'
            show_task_complete_dialog(self, title='Processing Complete', message=f'Successfully processed {len(self.folder_files)} file(s).\n\n{out_desc}', output_dir=getattr(self, '_last_output_dir', None), on_ok=self._advance_segmentation_shell_to_filtering)
            return
        tiff_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f'\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {tiff_file.name}\n')
        base_dir = tiff_file.parent
        if base_dir.name.startswith('Tif_'):
            output_dir = base_dir / self._get_omnipose_segmentation_folder_name()
        else:
            output_dir = base_dir / f'{tiff_file.stem}_omnipose_segmentation'
        self._last_output_dir = output_dir
        (_mt, _path) = self._resolve_model_selection(self.model_combo.currentText())
        if _path is not None:
            model_arg = prepare_user_pretrained_loader_path(_path)
        else:
            model_arg = _mt or self.model_combo.currentText()
        (run_diameter, run_flow) = self._get_run_diameter_and_flow()
        default_batch_size = '16'
        cmd_args = [str(tiff_file), '--output', str(output_dir), '--model', model_arg, '--diameter', str(run_diameter), '--flow-threshold', str(run_flow), '--mask-threshold', str(self._get_effective_mask_threshold()), '--batch-size', default_batch_size]
        if not self.use_gpu.isChecked():
            cmd_args.append('--no-gpu')
        if self.cluster.isChecked():
            cmd_args.append('--cluster')
        self.parent_window.log_output(f'Output directory: {output_dir}\n')
        try:
            self.parent_window.log_output(f'Omnipose params: batch_size={default_batch_size} use_gpu={self.use_gpu.isChecked()} cluster={self.cluster.isChecked()} overlay={self.create_overlay.isChecked()}\n')
        except Exception:
            pass
        self.current_file_progress = 0.0
        n_files = len(self.folder_files)
        file_progress = int(self.current_file_index / n_files * 100)
        file_msg = f'file {self.current_file_index + 1}/{n_files}'
        self._set_segmentation_progress(file_progress, file_msg)
        if hasattr(self.parent_window, 'set_status_message'):
            prefix = self._segmentation_status_prefix()
            self.parent_window.set_status_message(f'{prefix}: file {self.current_file_index + 1} of {n_files} — {tiff_file.name}')
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(lambda success, msg: self._on_segmentation_file_finished(success, msg, script_path))
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()

    def _write_persistent_cellid_masks(self, mask_path: Path, *, phase_path: Path | None = None) -> None:
        from bactomate.modules.cell_id_linking import write_persistent_cellid_mask_file

        shared = getattr(self.parent_window, 'shared_metadata', None)
        if write_persistent_cellid_mask_file(
            mask_path,
            shared_metadata=shared,
            phase_path=phase_path,
            log=self.parent_window.log_output,
        ) is None:
            get_logger().debug('Persistent Cell ID assignment failed', exc_info=True)

    def _on_segmentation_file_finished(self, success, message, script_path):
        """Handle completion of a single file in folder mode"""
        self.parent_window.log_output(message)
        if success:
            self.parent_window.log_output(f'✓ File {self.current_file_index + 1}/{len(self.folder_files)} processed successfully!\n')
            self.current_file_progress = 1.0
            try:
                tiff_file = self.folder_files[self.current_file_index]
                base_dir = Path(tiff_file).parent
                output_dir = base_dir / self._get_omnipose_segmentation_folder_name() if base_dir.name.startswith('Tif_') else base_dir / f'{Path(tiff_file).stem}_omnipose_segmentation'
                mask_guess = output_dir / f'{Path(tiff_file).stem}_masks.tif'
                self._write_persistent_cellid_masks(mask_guess, phase_path=Path(tiff_file))
            except Exception:
                get_logger().debug('Folder-mode persistent Cell ID export failed', exc_info=True)
            try:
                if getattr(self, 'detect_foci', None) is not None and self.detect_foci.isChecked():
                    tiff_file = self.folder_files[self.current_file_index]
                    base_dir = Path(tiff_file).parent
                    output_dir = base_dir / self._get_omnipose_segmentation_folder_name() if base_dir.name.startswith('Tif_') else base_dir / f'{Path(tiff_file).stem}_omnipose_segmentation'
                    sources = self._find_foci_sources_for_input(Path(tiff_file))
                    if not sources:
                        self.parent_window.log_output(f'[Foci] No *_foci* TIFF found for {Path(tiff_file).name} — tick Foci in File Conversion or disable Detect foci.\n')
                    if sources:
                        from bactomate.gui.artifact_paths import infer_pipeline_dir_from_mask, foci_csv_path
                        mask_guess = output_dir / f'{Path(tiff_file).stem}_masks.tif'
                        pipeline_dir = infer_pipeline_dir_from_mask(mask_guess)
                        from bactomate.gui.filter_preview_utils import short_foci_stack_label
                        for src in sources:
                            try:
                                stack = safe_imread(str(src))
                            except Exception:
                                continue
                            csv_path = foci_csv_path(pipeline_dir, mask_path=mask_guess, foci_source_path=src)
                            ch_key = self._spotiflow_channel_key_for_path(src)
                            sf_ch = self._get_spotiflow_detection_params(channel_key=ch_key)
                            self.parent_window.log_output(f'Detecting foci (CSV): {src.name} → {csv_path.name} [channel {short_foci_stack_label(src)}]\n')
                            (ok_foci, reason) = self._detect_foci_for_stack_and_write_csv(foci_stack=stack, output_csv=csv_path, prob_thresh=None, spotiflow_params=sf_ch)
                            if not ok_foci:
                                self.parent_window.log_output(f'[Foci] Detection failed for {src.name}: {reason}. CSV not written.\n')
                                continue
                            try:
                                from bactomate.gui.foci_overlay_artifacts import foci_overlay_path_for_csv, read_foci_csv_rows, write_foci_overlay_tif
                                overlay_path = foci_overlay_path_for_csv(csv_path)
                                rows = read_foci_csv_rows(csv_path)
                                ok = write_foci_overlay_tif(foci_image_path=Path(src), output_path=overlay_path, rows=rows, mask_path=mask_guess if Path(mask_guess).exists() else None)
                                if ok:
                                    self.parent_window.log_output(f'Foci overlay: {Path(src).name} → {overlay_path.name}\n')
                            except Exception:
                                pass
            except Exception:
                get_logger().debug('Folder-mode foci export failed', exc_info=True)
        else:
            self.parent_window.log_output(f'✗ File {self.current_file_index + 1}/{len(self.folder_files)} failed!\n')
            self.current_file_progress = 1.0
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            file_portion = 100.0 / len(self.folder_files)
            file_progress_base = (self.current_file_index + 1) / len(self.folder_files) * 100
            n_files = len(self.folder_files)
            self._set_segmentation_progress(int(min(100, file_progress_base)), f'file {min(self.current_file_index + 1, n_files)}/{n_files}')
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        if getattr(self, '_user_cancelled_processing', False):
            self._user_cancelled_processing = False
            self._reset_process_btn_idle()
            self._segmentation_in_progress = False
            self.parent_window.progress_bar.setVisible(False)
            if hasattr(self.parent_window, 'set_status_message'):
                self.parent_window.set_status_message('')
            self.parent_window.log_output('Processing cancelled.\n')
            return
        self.current_file_index += 1
        self._process_next_segmentation_file(script_path)

    def _parse_segmentation_progress(self, message):
        """Parse progress messages from segmentation script and update progress bar"""
        import re
        self.parent_window.log_output(message)
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            if not hasattr(self, 'current_file_progress'):
                self.current_file_progress = 0.0
            if 'Downloading:' in message or ('to /' in message and '.cellpose/models/' in message):
                download_match = re.search('(\\d+)%', message)
                if download_match:
                    try:
                        download_pct = int(download_match.group(1))
                        if self.current_file_index == 0:
                            self.current_file_progress = download_pct / 100.0 * 0.05
                    except (ValueError, AttributeError):
                        pass
            batch_complete_match = re.search('Batch (\\d+)/(\\d+) complete', message)
            if batch_complete_match:
                try:
                    batch_num = int(batch_complete_match.group(1))
                    total_batches = int(batch_complete_match.group(2))
                    self.current_file_progress = batch_num / total_batches
                except (ValueError, AttributeError):
                    pass
            batch_start_match = re.search('Processing batch (\\d+)/(\\d+):', message)
            if batch_start_match and (not batch_complete_match):
                try:
                    batch_num = int(batch_start_match.group(1))
                    total_batches = int(batch_start_match.group(2))
                    if batch_num == 1 and self.current_file_progress < 0.05:
                        self.current_file_progress = 0.05
                    else:
                        self.current_file_progress = (batch_num - 1) / total_batches
                except (ValueError, AttributeError):
                    pass
            frame_progress_match = re.search('(\\d+)%\\s*\\|\\s*\\d+/(\\d+)', message)
            if frame_progress_match:
                try:
                    frame_pct = int(frame_progress_match.group(1))
                    batch_info = re.search('batch (\\d+)/(\\d+)', message)
                    if batch_info:
                        batch_num = int(batch_info.group(1))
                        total_batches = int(batch_info.group(2))
                        batch_progress = (batch_num - 1) / total_batches
                        frame_progress = frame_pct / 100.0 / total_batches
                        self.current_file_progress = batch_progress + frame_progress
                    else:
                        self.current_file_progress = frame_pct / 100.0
                except (ValueError, AttributeError):
                    pass
            file_portion = 100.0 / len(self.folder_files)
            file_progress_base = self.current_file_index / len(self.folder_files) * 100
            file_progress_current = self.current_file_progress * file_portion
            total_progress = int(min(100, file_progress_base + file_progress_current))
            file_msg = f'file {self.current_file_index + 1}/{len(self.folder_files)}'
            self._set_segmentation_progress(total_progress, file_msg)
        else:
            progress_match = re.search('(\\d+)%', message)
            if progress_match:
                try:
                    progress_pct = int(progress_match.group(1))
                    self._set_segmentation_progress(progress_pct, message)
                except (ValueError, AttributeError):
                    pass

    def on_finished(self, success, message):
        is_folder = self._is_folder_mode()
        self._reset_process_btn_idle()
        self._segmentation_in_progress = False
        self.parent_window.log_output(message)
        if success:
            self._set_segmentation_progress(100)
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        self.parent_window.progress_bar.setVisible(False)
        if hasattr(self.parent_window, 'set_status_message'):
            self.parent_window.set_status_message('')
        if success:
            if is_folder:
                self.parent_window.log_output('✓ Folder processed successfully!\n')
            else:
                self.parent_window.log_output('✓ File processed successfully!\n')
            try:
                if not is_folder and getattr(self, '_last_output_dir', None):
                    in_path = Path(self.file_path.text())
                    mask_guess = Path(self._last_output_dir) / f'{in_path.stem}_masks.tif'
                    self._write_persistent_cellid_masks(mask_guess, phase_path=in_path)
            except Exception:
                get_logger().debug('Persistent Cell ID export failed', exc_info=True)
            try:
                if not is_folder and getattr(self, 'detect_foci', None) is not None and self.detect_foci.isChecked() and getattr(self, '_last_output_dir', None):
                    in_path = Path(self.file_path.text())
                    out_dir = Path(self._last_output_dir)
                    sources = self._find_foci_sources_for_input(in_path)
                    if not sources:
                        self.parent_window.log_output(f'[Foci] No *_foci* TIFF found for {in_path.name} — tick Foci in File Conversion or disable Detect foci.\n')
                        show_message_dialog(self, 'No foci channels found', self._no_foci_channels_found_message([in_path]), kind='info')
                    if sources:
                        from bactomate.gui.artifact_paths import infer_pipeline_dir_from_mask, foci_csv_path
                        mask_guess = out_dir / f'{in_path.stem}_masks.tif'
                        pipeline_dir = infer_pipeline_dir_from_mask(mask_guess)
                        from bactomate.gui.filter_preview_utils import short_foci_stack_label
                        for src in sources:
                            try:
                                stack = safe_imread(str(src))
                            except Exception:
                                continue
                            csv_path = foci_csv_path(pipeline_dir, mask_path=mask_guess, foci_source_path=src)
                            ch_key = self._spotiflow_channel_key_for_path(src)
                            sf_ch = self._get_spotiflow_detection_params(channel_key=ch_key)
                            self.parent_window.log_output(f'Detecting foci (CSV): {src.name} → {csv_path.name} [channel {short_foci_stack_label(src)}]\n')
                            (ok_foci, reason) = self._detect_foci_for_stack_and_write_csv(foci_stack=stack, output_csv=csv_path, prob_thresh=None, spotiflow_params=sf_ch)
                            if not ok_foci:
                                self.parent_window.log_output(f'[Foci] Detection failed for {src.name}: {reason}. CSV not written.\n')
                                continue
                            try:
                                from bactomate.gui.foci_overlay_artifacts import foci_overlay_path_for_csv, read_foci_csv_rows, write_foci_overlay_tif
                                overlay_path = foci_overlay_path_for_csv(csv_path)
                                rows = read_foci_csv_rows(csv_path)
                                ok = write_foci_overlay_tif(foci_image_path=Path(src), output_path=overlay_path, rows=rows, mask_path=mask_guess if Path(mask_guess).exists() else None)
                                if ok:
                                    self.parent_window.log_output(f'Foci overlay: {Path(src).name} → {overlay_path.name}\n')
                            except Exception:
                                pass
            except Exception:
                get_logger().debug('Foci detection export failed', exc_info=True)
            if not is_folder and getattr(self, '_last_output_dir', None):
                out_path = Path(self._last_output_dir)
                if out_path.exists():
                    from bactomate.gui.widgets import show_task_complete_dialog
                    show_task_complete_dialog(self, title='Processing Complete', message='File processed successfully.', output_dir=out_path, on_ok=self._advance_segmentation_shell_to_filtering)
        elif getattr(self, '_user_cancelled_processing', False):
            self._user_cancelled_processing = False
            self.parent_window.log_output('Processing was cancelled.\n')
        else:
            show_message_dialog(self, 'Error', f'Processing failed:\n{message}', kind='error')
        if success:
            self._performed_this_session = True
        self.workflow_step_finished.emit(success)
