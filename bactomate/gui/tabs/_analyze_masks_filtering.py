from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import numpy as np
import tifffile
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QProgressDialog, QWidget, QVBoxLayout
from bactomate.gui.session_params import apply_to_widgets
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET
from bactomate.gui.utils import safe_imread
from bactomate.gui.tabs._analyze_masks_base import AnalyzeMasksBase
from bactomate.gui.widgets import show_file_selection_dialog, show_message_dialog

class FilteringTab(AnalyzeMasksBase):
    """Section 2: Filtering (mask morphology/size + foci CSV peak filtering)."""

    def _segmentation_tab(self):
        host = self
        while host is not None:
            if host.__class__.__name__ == 'SegmentationTab':
                return host
            try:
                host = host.parent()
            except Exception:
                return None
        return None

    def _propagate_paths_to_analyze_stage(self, folder: Path | None) -> None:
        """Point the Analyze stage path at the same folder after filtering (like post-segmentation)."""
        if folder is None:
            return
        path_txt = str(Path(folder))
        seg = self._segmentation_tab()
        if seg is not None and hasattr(seg, 'analyze_tab'):
            try:
                seg.analyze_tab.file_path.setText(path_txt)
            except Exception:
                pass
        mw = getattr(self, 'parent_window', None)
        if mw is not None and hasattr(mw, 'analyze_tab'):
            try:
                if seg is None or getattr(mw, 'analyze_tab', None) is not getattr(seg, 'analyze_tab', None):
                    mw.analyze_tab.file_path.setText(path_txt)
            except Exception:
                pass

    def _advance_segmentation_shell_to_analyze(self) -> None:
        seg = self._segmentation_tab()
        if seg is None or not hasattr(seg, 'advance_to_analyze_stage'):
            return
        QTimer.singleShot(0, seg.advance_to_analyze_stage)

    def _stage_key(self) -> str:
        return 'filtering'

    def _file_group_title(self) -> str:
        return 'Filter masks'

    def _preview_button_label(self) -> str:
        return 'Preview filters'

    def _preview_button_tooltip(self) -> str:
        return 'Open the filter preview in the right panel and adjust parameters there'

    def _primary_button_label(self) -> str:
        return 'Filter File/Folder'

    def _primary_button_tooltip(self) -> str:
        return 'Write filtered mask and foci CSV artifacts for the selected path. Uses the current Preview filters panel settings when the preview is open. If the filter preview is open, it is refreshed afterward (same as Apply current filters).'

    def _uses_combined_path_actions_group(self) -> bool:
        return True

    def _build_stage_controls(self, scroll_layout: QVBoxLayout) -> None:
        return

    def _sync_embedded_preview_button_visibility(self) -> None:
        if hasattr(self, 'preview_action_btn'):
            self.preview_action_btn.setVisible(True)

    def _wire_stage_buttons(self) -> None:
        self._filtering_in_progress = False
        self._cancel_filtering_requested = False
        self._filter_filename_filter = ''
        self._folder_preview_controller = None
        self.preview_action_btn.clicked.connect(self.preview_filter_effects)
        if hasattr(self, 'folder_preview_btn'):
            self.folder_preview_btn.clicked.connect(self.preview_folder_filters)
        self.primary_action_btn.clicked.connect(self._on_filter_or_cancel_clicked)

    def _folder_preview_filter_params(self) -> dict:
        fs = dict(self.get_filter_settings())
        # Map UI filter settings into analyze_cell_areas-style keys used by compute.
        out = {
            'pixel_size': float(self.pixel_size.value()) if hasattr(self, 'pixel_size') else float(fs.get('pixel_size') or 0.108),
            'min_area': fs.get('min_area_um2'),
            'max_area': fs.get('max_area_um2'),
            'min_width': fs.get('min_width_um'),
            'max_width': fs.get('max_width_um'),
            'min_length': fs.get('min_length_um'),
            'max_length': fs.get('max_length_um'),
            'min_aspect_ratio': float(fs.get('min_aspect_ratio') or 0.0),
            'max_aspect_ratio': fs.get('max_aspect_ratio'),
            'min_eccentricity': float(fs.get('min_eccentricity') or 0.0),
            'min_solidity': float(fs.get('min_solidity') or 0.0),
            'exclude_border': bool(fs.get('exclude_border', False)),
            'remove_small': True,
            'min_size_pixels': 10,
            'focus_filter_enabled': bool(fs.get('focus_filter_enabled', False)),
            'focus_modality': fs.get('focus_modality', 'phase'),
            'focus_min_fraction_of_median': float(fs.get('focus_min_fraction_of_median') or 0.5),
            '_filter_settings': fs,
        }
        return out

    def preview_folder_filters(self) -> None:
        """Closer-to-batch QC grid for up to 48 masks in the folder."""
        from bactomate.gui.folder_preview.constants import folder_preview_enabled
        from bactomate.gui.folder_preview.controller import FolderPreviewController
        from bactomate.gui.folder_preview.filter_compute import compute_filter_tile

        if not self.file_path.text():
            show_message_dialog(self, 'No Selection', 'Please select a mask file or folder first.', kind='warning')
            return
        masks = self._get_mask_files_for_feedback()
        if not folder_preview_enabled(len(masks)):
            show_message_dialog(
                self,
                'Preview folder',
                'Need more than one mask in the source location for folder preview.\n'
                'Use Preview filters for a single mask.',
                kind='warning',
            )
            return
        mount = getattr(self, '_module_preview_mount', None) or getattr(self, '_preview_mount', None)
        surf = getattr(self, '_preview_surface', None)
        if mount is None:
            # Filtering is embedded in SegmentationTab — ask segmentation for mount.
            seg = self._segmentation_tab() if hasattr(self, '_segmentation_tab') else None
            if seg is not None:
                mount = getattr(seg, '_module_preview_mount', None)
                surf = getattr(seg, '_preview_surface', None) or surf
        if mount is None:
            show_message_dialog(
                self,
                'Preview folder',
                'Folder preview needs the module right panel.',
                kind='warning',
            )
            return
        dlg = getattr(self, '_filter_preview_dialog', None)
        if dlg is not None:
            try:
                dlg.close()
            except Exception:
                pass
            self._filter_preview_dialog = None
        params = self._folder_preview_filter_params()

        def compute_fn(path, _idx):
            return compute_filter_tile(path, params)

        ctrl = getattr(self, '_folder_preview_controller', None)
        if ctrl is None:
            ctrl = FolderPreviewController(
                self,
                preview_surface=surf,
                mount=mount,
                on_tile_open=None,
                parent=self,
            )
            self._folder_preview_controller = ctrl
        try:
            ctrl.start(masks, params, compute_fn)
        except Exception as e:
            show_message_dialog(self, 'Preview folder', f'Could not start folder preview:\n{e}', kind='error')

    def _on_filter_or_cancel_clicked(self) -> None:
        if bool(getattr(self, '_filtering_in_progress', False)):
            self._cancel_filtering_requested = True
            try:
                self._update_filter_button_progress(100, 'Stopping…')
                self.primary_action_btn.setEnabled(False)
                self.primary_action_btn.setText('Cancelling…')
            except Exception:
                pass
            return
        self.filter_masks()

    def _set_filter_button_busy(self, busy: bool, *, progress: int=0, subtitle: str='') -> None:
        try:
            if not hasattr(self.primary_action_btn, 'set_progress_active'):
                return
            self.primary_action_btn.set_progress_active(bool(busy))
            self.primary_action_btn.set_progress(max(0, min(100, int(progress))))
            self.primary_action_btn.set_progress_subtitle(str(subtitle or ''))
        except Exception:
            pass

    def _update_filter_button_progress(self, value: int, step: str='') -> None:
        try:
            if not hasattr(self.primary_action_btn, 'set_progress'):
                return
            pct = max(0, min(100, int(value)))
            if bool(getattr(self, '_filtering_in_progress', False)) and pct < 100:
                pct = max(3, pct)
            self.primary_action_btn.set_progress(pct)
            self.primary_action_btn.set_progress_subtitle(str(step or '').strip())
        except Exception:
            pass

    def _build_filtering_peak_filters_widget(self) -> QGroupBox:
        peak_group = QGroupBox('')
        peak_group.setFlat(True)
        peak_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        peak_layout = QVBoxLayout()
        peak_layout.setSpacing(6)
        row = QHBoxLayout()
        self.filter_prob_enabled = QCheckBox('Filter by probability ≥')
        self.filter_prob_enabled.setChecked(False)
        self.filter_prob_thresh = QDoubleSpinBox()
        self.filter_prob_thresh.setRange(0.0, 1.0)
        self.filter_prob_thresh.setDecimals(3)
        self.filter_prob_thresh.setSingleStep(0.05)
        self.filter_prob_thresh.setValue(0.5)
        row.addWidget(self.filter_prob_enabled)
        row.addWidget(self.filter_prob_thresh)
        row.addStretch()
        peak_layout.addLayout(row)
        row2 = QHBoxLayout()
        self.filter_min_int_enabled = QCheckBox('Min raw intensity ≥')
        self.filter_min_int_enabled.setChecked(False)
        self.filter_min_int = QDoubleSpinBox()
        self.filter_min_int.setRange(0, 100000)
        self.filter_min_int.setDecimals(0)
        self.filter_min_int.setSingleStep(100)
        self.filter_min_int.setValue(0)
        row2.addWidget(self.filter_min_int_enabled)
        row2.addWidget(self.filter_min_int)
        row2.addStretch()
        peak_layout.addLayout(row2)
        row3 = QHBoxLayout()
        self.filter_per_cell_ratio_enabled = QCheckBox('Per-cell ratio ≥')
        self.filter_per_cell_ratio_enabled.setChecked(False)
        self.filter_per_cell_ratio = QDoubleSpinBox()
        self.filter_per_cell_ratio.setRange(0.5, 100.0)
        self.filter_per_cell_ratio.setDecimals(2)
        self.filter_per_cell_ratio.setSingleStep(0.1)
        self.filter_per_cell_ratio.setValue(2.0)
        row3.addWidget(self.filter_per_cell_ratio_enabled)
        row3.addWidget(self.filter_per_cell_ratio)
        row3.addWidget(QLabel('(focus mean / rest-of-cell mean)'))
        row3.addStretch()
        peak_layout.addLayout(row3)
        peak_group.setLayout(peak_layout)
        self.peak_filters_group = peak_group
        from bactomate.gui.layout_helpers import wire_spin_activates_checkbox
        wire_spin_activates_checkbox(self.filter_prob_thresh, self.filter_prob_enabled)
        wire_spin_activates_checkbox(self.filter_min_int, self.filter_min_int_enabled)
        wire_spin_activates_checkbox(self.filter_per_cell_ratio, self.filter_per_cell_ratio_enabled)
        return peak_group

    def _place_peak_filters_in_preview(self, target_layout) -> None:
        return

    def _restore_peak_filters_to_stash(self) -> None:
        return

    def _shell_selected_mask_path(self) -> Optional[Path]:
        surf = getattr(self, '_preview_surface', None)
        if surf is None:
            return None
        raw = surf.current_candidate_path()
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_file() else None

    def preview_filter_effects(self) -> None:
        ctrl = getattr(self, '_folder_preview_controller', None)
        if ctrl is not None:
            try:
                ctrl.close()
            except Exception:
                pass
            self._folder_preview_controller = None
        if not self.file_path.text():
            show_message_dialog(self, 'No Selection', 'Please select a mask file or folder first.', kind='warning')
            return
        raw_path = self.file_path.text().strip()
        if not Path(raw_path).exists():
            show_message_dialog(self, 'Not Found', 'The selected file or folder does not exist.', kind='warning')
            return
        from bactomate.gui.filter_preview_utils import pick_mask_path_for_preview
        raw_obj = Path(raw_path)
        shell_pick = self._shell_selected_mask_path()
        if raw_obj.is_dir():
            mask_path = pick_mask_path_for_preview(self, raw_path, force_dialog=True, default_path=shell_pick)
        else:
            mask_path = shell_pick or pick_mask_path_for_preview(self, raw_path)
        if mask_path is None or not mask_path.exists():
            show_message_dialog(self, 'No Masks Found', 'No mask files were found in the selected path.', kind='warning')
            return
        self._open_filter_preview_for_mask(mask_path)
        seg = self._segmentation_tab()
        if seg is not None:
            try:
                seg.refresh_filtering_candidate_list()
            except Exception:
                pass

    def reload_filter_preview_for_path(self, path: str | Path) -> None:
        """Reload the embedded filter preview when the shell mask dropdown changes."""
        mask_path = Path(path)
        if not mask_path.is_file():
            return
        current = getattr(self, '_filter_preview_mask_path', None)
        if current is not None:
            try:
                if Path(current).resolve() == mask_path.resolve():
                    return
            except OSError:
                if Path(current) == mask_path:
                    return
        dlg = getattr(self, '_filter_preview_dialog', None)
        if dlg is None:
            return
        try:
            dlg._preview_request_id = int(getattr(dlg, '_preview_request_id', 0)) + 1
            dlg._close_pending = True
        except Exception:
            pass
        try:
            dlg.close()
        except Exception:
            pass
        try:
            self._filter_preview_dialog = None
        except Exception:
            pass
        self._open_filter_preview_for_mask(mask_path)

    def _open_filter_preview_for_mask(self, mask_path: Path) -> None:
        from PyQt6.QtCore import QEventLoop, Qt, QThread
        from PyQt6.QtWidgets import QApplication, QProgressDialog
        from bactomate.gui.filter_preview_open_worker import FilterPreviewFociStacksWorker, FilterPreviewOpenWorker
        from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview
        mask_path = Path(mask_path)
        if not mask_path.is_file():
            show_message_dialog(self, 'Not Found', f'Mask file not found:\n{mask_path}', kind='warning')
            return
        try:
            self._filter_preview_mask_path = mask_path
        except Exception:
            pass
        btn = getattr(self, 'preview_action_btn', None)
        prev_label = btn.text() if btn is not None else None
        if btn is not None:
            try:
                btn.setEnabled(False)
                btn.setText('Opening preview…')
            except Exception:
                pass
        loading = QProgressDialog('Loading mask stack…', None, 0, 0, self)
        loading.setWindowTitle('Filter preview')
        loading.setWindowModality(Qt.WindowModality.ApplicationModal)
        loading.setAutoClose(False)
        loading.setAutoReset(False)
        loading.setCancelButton(None)
        loading.show()
        QApplication.processEvents()
        load_result: dict = {'masks': None, 'err': ''}
        loop = QEventLoop()
        worker = FilterPreviewOpenWorker(mask_path)
        thread = QThread(self)
        thread.setObjectName('FilterPreviewOpenThread')
        worker.moveToThread(thread)

        def _on_masks_loaded(masks_obj, err_msg: str) -> None:
            load_result['masks'] = masks_obj
            load_result['err'] = str(err_msg or '')
            try:
                loop.quit()
            except Exception:
                pass
        worker.finished.connect(_on_masks_loaded)
        worker.finished.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        worker.finished.connect(worker.deleteLater)
        thread.start()
        loop.exec()
        try:
            loading.close()
        except Exception:
            pass
        if btn is not None:
            try:
                btn.setEnabled(True)
                if prev_label is not None:
                    btn.setText(prev_label)
            except Exception:
                pass
        if load_result['err']:
            show_message_dialog(self, 'Load Error', f"Could not load mask:\n{load_result['err']}", kind='warning')
            return
        if load_result['masks'] is None:
            show_message_dialog(self, 'Load Error', 'Could not load mask (empty result).', kind='warning')
            return
        foci_preload: dict = {}
        from bactomate.gui.filter_preview_utils import filter_paths_for_mask
        foci_paths: list[Path] = []
        try:
            raw_foci = [Path(p) for p in self._resolve_foci_paths_for_mask(mask_path)]
            foci_paths = filter_paths_for_mask(self, mask_path, raw_foci)
        except Exception:
            foci_paths = []
        if foci_paths:
            try:
                loading.setLabelText('Loading foci channels…')
                QApplication.processEvents()
            except Exception:
                pass
            foci_result: dict = {'stacks': None, 'err': ''}
            foci_loop = QEventLoop()
            foci_worker = FilterPreviewFociStacksWorker(foci_paths)
            foci_thread = QThread(self)
            foci_thread.setObjectName('FilterPreviewFociStacksThread')
            foci_worker.moveToThread(foci_thread)

            def _on_foci_loaded(stacks_obj, err_msg: str) -> None:
                foci_result['stacks'] = stacks_obj
                foci_result['err'] = str(err_msg or '')
                try:
                    foci_loop.quit()
                except Exception:
                    pass
            foci_worker.finished.connect(_on_foci_loaded)
            foci_worker.finished.connect(foci_thread.quit)
            foci_thread.started.connect(foci_worker.run)
            foci_thread.finished.connect(foci_thread.deleteLater)
            foci_worker.finished.connect(foci_worker.deleteLater)
            foci_thread.start()
            foci_loop.exec()
            if isinstance(foci_result.get('stacks'), dict):
                foci_preload = dict(foci_result['stacks'])
        try:
            show_filter_preview(self, str(mask_path), masks_preloaded=load_result['masks'], mask_path=mask_path, foci_stacks_preloaded=foci_preload or None)
        except Exception as exc:
            show_message_dialog(self, 'Preview Error', f'Could not open filtering preview:\n{exc}', kind='warning')

    def _resolve_mask_path_for_preview(self, path_text: str) -> Optional[Path]:
        """Resolve mask TIFF for preview open (prompts when multiple masks in a folder)."""
        from bactomate.gui.filter_preview_utils import pick_mask_path_for_preview
        return pick_mask_path_for_preview(self, path_text)

    def _default_filter_settings(self) -> dict[str, Any]:
        return {'pixel_size': 0.108, 'min_aspect_ratio': 1.5, 'max_aspect_ratio': 10.0, 'min_eccentricity': 0.5, 'min_solidity': 0.7, 'min_area_um2': None, 'max_area_um2': None, 'min_width_um': None, 'max_width_um': None, 'min_length_um': None, 'max_length_um': None, 'exclude_border': False, 'focus_filter_enabled': False, 'focus_min_fraction_of_median': 0.5, 'focus_modality': 'phase', 'foci_prob_enabled': False, 'foci_prob_thresh': 0.5, 'foci_min_int_enabled': False, 'foci_min_int': 0.0, 'foci_per_cell_ratio_enabled': False, 'foci_per_cell_ratio': 2.0, 'foci_exclude_outside_cells': False, 'foci_exclude_outside_clear': False, 'foci_exclude_outside_touching': False, 'foci_blob_threshold_frac': 0.9, 'foci_blob_max_radius': 5, 'foci_min_area_px_enabled': False, 'foci_min_area_px': 1}

    def _ensure_filter_settings(self) -> dict[str, Any]:
        fs = getattr(self, '_filter_settings', None)
        if not isinstance(fs, dict):
            fs = self._default_filter_settings()
            self._filter_settings = dict(fs)
        for (k, v) in self._default_filter_settings().items():
            if k not in fs:
                fs[k] = v
        return fs

    def get_filter_settings(self) -> dict[str, Any]:
        """Return a stable snapshot used by both preview and batch filtering."""
        fs = dict(self._ensure_filter_settings())
        for k in ('min_area_um2', 'max_area_um2', 'min_width_um', 'max_width_um', 'min_length_um', 'max_length_um'):
            v = fs.get(k, None)
            try:
                if v is None:
                    fs[k] = None
                else:
                    fv = float(v)
                    fs[k] = fv if fv > 0 else None
            except Exception:
                fs[k] = None
        try:
            fs['exclude_border'] = bool(fs.get('exclude_border', False))
        except Exception:
            fs['exclude_border'] = False
        return fs

    def set_filter_settings(self, data: dict[str, Any]) -> None:
        fs = self._ensure_filter_settings()
        if not isinstance(data, dict):
            return
        for k in self._default_filter_settings().keys():
            if k in data:
                fs[k] = data.get(k)
        if isinstance(data.get('foci_channels'), dict):
            fs['foci_channels'] = dict(data['foci_channels'])

    def get_foci_channel_settings(self, channel_key: str) -> dict[str, Any]:
        from bactomate.gui.filter_preview_utils import default_foci_channel_settings, ensure_foci_channels_map
        fs = self._ensure_filter_settings()
        channels = ensure_foci_channels_map(fs)
        key = str(channel_key or '_default').strip() or '_default'
        ch = channels.get(key)
        if not isinstance(ch, dict):
            ch = dict(default_foci_channel_settings())
            channels[key] = ch
        return dict(ch)

    def set_foci_channel_settings(self, channel_key: str, data: dict[str, Any]) -> None:
        from bactomate.gui.filter_preview_utils import FOCI_CHANNEL_SETTING_KEYS, default_foci_channel_settings, ensure_foci_channels_map
        fs = self._ensure_filter_settings()
        channels = ensure_foci_channels_map(fs)
        key = str(channel_key or '_default').strip() or '_default'
        ch = channels.setdefault(key, dict(default_foci_channel_settings()))
        if not isinstance(data, dict):
            return
        for k in FOCI_CHANNEL_SETTING_KEYS:
            if k in data:
                ch[k] = data.get(k)

    def ensure_foci_channel_settings_for_sources(self, foci_sources: list[Path]) -> None:
        from bactomate.gui.filter_preview_utils import seed_foci_channel_settings
        fs = self._ensure_filter_settings()
        seed_foci_channel_settings(fs, [Path(p) for p in foci_sources])

    def get_settings_for_foci_source(self, foci_source: Path) -> dict[str, Any]:
        from bactomate.gui.filter_preview_utils import resolve_foci_settings_for_source
        return resolve_foci_settings_for_source(self.get_filter_settings(), Path(foci_source))

    def sync_filter_settings_from_preview_ui(self) -> bool:
        """Snapshot open preview widgets into tab settings (cell + active foci channel)."""
        from bactomate.gui.filter_preview_utils import sync_filter_settings_from_preview_dialog
        return sync_filter_settings_from_preview_dialog(self)

    def reset_filter_settings_defaults(self) -> None:
        self._filter_settings = self._default_filter_settings()

    def filter_masks(self) -> None:
        if bool(getattr(self, '_filtering_in_progress', False)):
            return
        raw_path = (self.file_path.text() or '').strip() if getattr(self, 'file_path', None) else ''
        if not raw_path:
            show_message_dialog(self, 'No Selection', 'Please select a mask file or folder first.', kind='warning')
            return
        p = Path(raw_path)
        if not p.exists():
            show_message_dialog(self, 'Not Found', 'The selected file or folder does not exist.', kind='warning')
            return
        if p.is_file():
            mask_files = [p]
        else:
            mask_files = self._get_mask_files_for_feedback()
            if not mask_files:
                show_message_dialog(self, 'No Masks Found', 'No mask files were found in the selected path.', kind='warning')
                return
            result = show_file_selection_dialog(self, mask_files, title='Select Mask Files for Filtering', message=f'Select mask files to filter ({len(mask_files)} listed). All files are selected by default; untick any you want to skip.', initial_filename_filter=getattr(self, '_filter_filename_filter', '') or '', return_filename_filter=True)
            (selected_files, filter_used) = result if isinstance(result, tuple) else (result, '')
            if filter_used is not None:
                self._filter_filename_filter = filter_used or ''
            if selected_files is None:
                return
            if len(selected_files) == 0:
                show_message_dialog(self, 'No Files Selected', 'Please select at least one mask file to filter.', kind='warning')
                return
            mask_files = list(selected_files)
        if self.sync_filter_settings_from_preview_ui():
            if hasattr(self.parent_window, 'log_output'):
                self.parent_window.log_output('[Filtering] Applied current Preview filter settings before batch write.\n')
        fs = self.get_filter_settings()
        self._filtering_in_progress = True
        self._cancel_filtering_requested = False
        try:
            self.preview_action_btn.setEnabled(False)
        except Exception:
            pass
        try:
            self.primary_action_btn.setText('Cancel Filtering')
            self.primary_action_btn.setToolTip('Stop the current filtering job.')
            self._set_filter_button_busy(True, progress=3)
        except Exception:
            pass
        from bactomate.gui.filter_apply_progress import FilterApplyProgressDialog, batch_filter_percent, batch_foci_step_title, format_filter_batch_detail
        progress = FilterApplyProgressDialog(self)
        progress.setWindowTitle('Filtering')
        progress.set_progress(0, 'Filtering masks', f'{len(mask_files)} file(s)')
        progress.show()
        QApplication.processEvents()

        def _request_cancel() -> None:
            self._cancel_filtering_requested = True

        def _close_progress_dialog() -> None:
            try:
                progress.canceled.disconnect(_request_cancel)
            except Exception:
                pass
            try:
                progress.hide()
                progress.setVisible(False)
                progress.close()
                progress.deleteLater()
            except Exception:
                pass
            try:
                QApplication.processEvents()
            except Exception:
                pass
        try:
            progress.canceled.connect(_request_cancel)
        except Exception:
            pass
        from bactomate.gui.artifact_paths import FOCI_DETECTION_DIRNAME, filtered_mask_path_for_mask, filtered_foci_csv_path, infer_pipeline_dir_from_mask, resolve_segmentation_foci_csv
        try:
            from bactomate.scripts.analyze_cell_areas import calculate_cell_properties, filter_rod_shaped_cells, filter_by_size
        except Exception as e:
            _close_progress_dialog()
            show_message_dialog(self, 'Filtering Error', f'Could not import filtering pipeline:\n{e}', kind='warning')
            self._filtering_in_progress = False
            try:
                self._set_filter_button_busy(False)
                self.primary_action_btn.setText(self._primary_button_label())
                self.primary_action_btn.setToolTip(self._primary_button_tooltip())
                self.primary_action_btn.setEnabled(True)
                self.preview_action_btn.setEnabled(True)
            except Exception:
                pass
            return
        try:
            from bactomate.gui.foci_filter_pipeline import process_foci_rows_for_mask, read_raw_foci_csv, write_filtered_foci_csv
            from bactomate.modules.image_stack_utils import coerce_stack_to_tyx
        except Exception:
            process_foci_rows_for_mask = None
            read_raw_foci_csv = None
            write_filtered_foci_csv = None
            coerce_stack_to_tyx = None
        helper_skip_logged = False

        def _set_progress(pct: int, step: str, detail: str='') -> None:
            try:
                progress.set_progress(int(pct), str(step), str(detail or ''))
            except Exception:
                pass
            self._update_filter_button_progress(int(pct), str(step))

        def _batch_detail(mask_path: Path, file_i: int, *, channel_path: Path | None=None, channel_i: int | None=None, channel_n: int | None=None, extra: str | None=None) -> str:
            return format_filter_batch_detail(file_index=file_i + 1, file_count=n_files, mask_name=mask_path.name, channel_name=Path(channel_path).name if channel_path is not None else None, channel_index=int(channel_i) + 1 if channel_i is not None else None, channel_count=channel_n, extra=extra)
        n_files = max(1, int(len(mask_files)))
        try:
            for (i, mask_path) in enumerate(mask_files):
                if bool(getattr(self, '_cancel_filtering_requested', False)):
                    break
                _set_progress(int(round(100.0 * float(i) / float(n_files))), 'Filtering masks', _batch_detail(mask_path, i))
                QApplication.processEvents()
                try:
                    _set_progress(int(round(100.0 * float(i) / float(n_files))), 'Loading mask', _batch_detail(mask_path, i))
                    masks = safe_imread(str(mask_path))
                    masks = np.asarray(masks)
                    if masks.ndim == 2:
                        masks = masks[np.newaxis, :, :]
                except Exception as e:
                    if hasattr(self.parent_window, 'log_output'):
                        self.parent_window.log_output(f'[Filtering] Failed to load mask {mask_path.name}: {e}\n')
                    continue
                out = np.zeros_like(masks, dtype=np.int32)
                n_frames = max(1, int(masks.shape[0]))
                phase_stack = None
                if fs.get('focus_filter_enabled'):
                    try:
                        from bactomate.gui.analysis_visualization import resolve_phase_image_for_mask
                        from bactomate.modules.image_stack_utils import coerce_stack_to_tyx

                        phase_path = resolve_phase_image_for_mask(mask_path)
                        if phase_path is not None and phase_path.is_file():
                            phase_stack = coerce_stack_to_tyx(np.asarray(safe_imread(str(phase_path))))
                    except Exception:
                        phase_stack = None
                for fi in range(masks.shape[0]):
                    if bool(getattr(self, '_cancel_filtering_requested', False)):
                        break
                    if fi == 0 or fi == n_frames - 1 or fi % max(1, n_frames // 25) == 0:
                        _set_progress(batch_filter_percent(file_index=i, file_count=n_files, cell_phase=float(fi) / float(max(1, n_frames - 1))), 'Filtering cells', _batch_detail(mask_path, i, extra=f'Frame {fi + 1}/{n_frames}'))
                        QApplication.processEvents()
                    frame_mask = np.asarray(masks[fi], dtype=np.int32)
                    try:
                        props = calculate_cell_properties(frame_mask, pixel_size=float(fs.get('pixel_size', 0.108) or 0.108), exclude_border=bool(fs.get('exclude_border', False)))
                        rod = filter_rod_shaped_cells(props, min_aspect_ratio=float(fs.get('min_aspect_ratio', 1.5) or 1.5), max_aspect_ratio=float(fs.get('max_aspect_ratio', 10.0) or 10.0), min_eccentricity=float(fs.get('min_eccentricity', 0.5) or 0.5), min_solidity=float(fs.get('min_solidity', 0.7) or 0.7))
                        filtered = filter_by_size(rod, min_area_um2=fs.get('min_area_um2', None), max_area_um2=fs.get('max_area_um2', None), min_width_um=fs.get('min_width_um', None), max_width_um=fs.get('max_width_um', None), min_length_um=fs.get('min_length_um', None), max_length_um=fs.get('max_length_um', None))
                        accepted = {int(c['label']) for c in filtered}
                        if accepted and fs.get('focus_filter_enabled') and phase_stack is not None:
                            try:
                                from bactomate.modules.cell_focus_score import (
                                    filter_labels_by_focus,
                                    score_cells_in_frame,
                                )

                                ch_idx = max(0, min(int(fi), int(phase_stack.shape[0]) - 1))
                                phase_plane = np.asarray(phase_stack[ch_idx], dtype=np.float64)
                                scores = score_cells_in_frame(
                                    phase_plane,
                                    frame_mask,
                                    accepted,
                                    modality=fs.get('focus_modality', 'phase'),
                                )
                                frac = float(fs.get('focus_min_fraction_of_median', 0.5) or 0.5)
                                accepted = filter_labels_by_focus(
                                    scores,
                                    min_fraction_of_median=frac if frac > 0 else None,
                                )
                            except Exception:
                                pass
                        if accepted:
                            out_frame = np.where(np.isin(frame_mask, np.array(list(accepted), dtype=np.int32)), frame_mask, 0)
                        else:
                            out_frame = np.zeros_like(frame_mask, dtype=np.int32)
                        out[fi] = out_frame
                    except Exception:
                        out[fi] = frame_mask
                out_path = filtered_mask_path_for_mask(mask_path)
                try:
                    _set_progress(batch_filter_percent(file_index=i, file_count=n_files, cell_phase=1.0), 'Writing filtered mask', _batch_detail(mask_path, i, extra=f'Output: {out_path.name}'))
                    QApplication.processEvents()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(out_path), out.astype(np.int32))
                    if hasattr(self.parent_window, 'log_output'):
                        self.parent_window.log_output(f'[Filtering] Wrote filtered mask: {out_path}\n')
                    try:
                        from bactomate.modules.pipeline_outputs import write_filter_stage_visualization

                        write_filter_stage_visualization(
                            mask_path=mask_path,
                            original_masks=masks,
                            filtered_masks=out,
                            pixel_size=float(fs.get('pixel_size', 0.108) or 0.108),
                            exclude_border=bool(fs.get('exclude_border', False)),
                            log=self.parent_window.log_output if hasattr(self.parent_window, 'log_output') else None,
                        )
                    except Exception as exc:
                        if hasattr(self.parent_window, 'log_output'):
                            self.parent_window.log_output(f'[Filtering] Filter visualization failed: {exc}\n')
                except Exception as e:
                    if hasattr(self.parent_window, 'log_output'):
                        self.parent_window.log_output(f'[Filtering] Failed to write filtered mask {out_path.name}: {e}\n')
                pipeline_dir = infer_pipeline_dir_from_mask(Path(mask_path))
                try:
                    foci_sources = self._resolve_foci_paths_for_mask(Path(mask_path))
                except Exception:
                    foci_sources = []
                if not foci_sources:
                    if hasattr(self.parent_window, 'log_output'):
                        self.parent_window.log_output(f'[Filtering] No foci image stack matched {mask_path.name}; skipped foci CSV filtering (mask filtering still applied).\n')
                    continue
                if process_foci_rows_for_mask is None or read_raw_foci_csv is None or write_filtered_foci_csv is None or (coerce_stack_to_tyx is None):
                    if not helper_skip_logged and hasattr(self.parent_window, 'log_output'):
                        self.parent_window.log_output('[Filtering] Skipping foci CSV filtering: required helper imports are unavailable.\n')
                        helper_skip_logged = True
                    continue
                n_src = max(1, int(len(foci_sources)))
                for (si, src) in enumerate(foci_sources):
                    if bool(getattr(self, '_cancel_filtering_requested', False)):
                        break
                    try:
                        src_p = Path(src)

                        def _foci_progress(phase: str, ch_phase: float, extra: str | None=None) -> None:
                            _set_progress(batch_filter_percent(file_index=i, file_count=n_files, channel_index=si, channel_count=n_src, channel_phase=ch_phase), batch_foci_step_title(src_p, si, n_src, phase), _batch_detail(mask_path, i, channel_path=src_p, channel_i=si, channel_n=n_src, extra=extra))
                            QApplication.processEvents()
                        _foci_progress('Starting', 0.0)
                        chosen_csv = resolve_segmentation_foci_csv(pipeline_dir, mask_path=mask_path, foci_source_path=src)
                        if chosen_csv is None:
                            if hasattr(self.parent_window, 'log_output'):
                                self.parent_window.log_output(f'[Filtering] No raw foci CSV for {mask_path.name} + {Path(src).name}; run segmentation with Detect foci enabled first.\n')
                            continue
                        raw_rows = read_raw_foci_csv(chosen_csv)
                        if not raw_rows:
                            continue
                        if chosen_csv.parent.name != FOCI_DETECTION_DIRNAME:
                            if hasattr(self.parent_window, 'log_output'):
                                self.parent_window.log_output(f'[Filtering] Using legacy foci CSV: {chosen_csv}\n')
                        _foci_progress('Loading stack', 0.12)
                        foci_stack = np.asarray(safe_imread(str(src)))
                        foci_tyx = coerce_stack_to_tyx(foci_stack)
                        if foci_tyx is None or foci_tyx.size == 0:
                            continue
                        include_frame = bool(foci_tyx.shape[0] > 1 or any((int(r.get('frame', 0)) != 0 for r in raw_rows)))
                        ch_fs = self.get_settings_for_foci_source(Path(src))
                        _foci_progress('Scoring filters', 0.35, extra=f'{len(raw_rows):,} peaks')
                        filtered_rows = process_foci_rows_for_mask(raw_rows, foci_tyx, np.asarray(out), ch_fs)
                        out_csv = filtered_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=src)
                        _foci_progress('Writing CSV', 0.62, extra=f'Output: {out_csv.name} ({len(filtered_rows):,} rows)')
                        write_filtered_foci_csv(out_csv, filtered_rows, include_frame=include_frame)
                        if hasattr(self.parent_window, 'log_output'):
                            self.parent_window.log_output(f'[Filtering] Wrote filtered foci CSV: {out_csv}\n')
                        try:
                            from bactomate.gui.foci_overlay_artifacts import write_filtered_foci_overlay_for_csv
                            _foci_progress('Writing overlay', 0.82)
                            (overlay_path, ok) = write_filtered_foci_overlay_for_csv(foci_image_path=Path(src), filtered_csv_path=out_csv, rows=filtered_rows, mask_path=out_path if Path(out_path).exists() else None, filter_settings=ch_fs)
                            if hasattr(self.parent_window, 'log_output'):
                                if ok:
                                    self.parent_window.log_output(f'[Filtering] Wrote filtered overlay: {overlay_path}\n')
                                else:
                                    self.parent_window.log_output(f'[Filtering] Failed to write filtered overlay for {Path(src).name}.\n')
                        except Exception as exc:
                            if hasattr(self.parent_window, 'log_output'):
                                self.parent_window.log_output(f'[Filtering] Filtered overlay error for {Path(src).name}: {exc}\n')
                        _foci_progress('Channel complete', 0.95)
                    except Exception as e:
                        if hasattr(self.parent_window, 'log_output'):
                            self.parent_window.log_output(f'[Filtering] Foci filtering failed for {src.name}: {e}\n')
                        continue
        finally:
            pass
        cancelled = bool(getattr(self, '_cancel_filtering_requested', False))

        def _resolve_output_folder():
            try:
                output_dirs = {Path(m).parent.resolve() for m in mask_files}
                if len(output_dirs) == 1:
                    return next(iter(output_dirs))
                selected_path = Path(raw_path)
                return selected_path if selected_path.is_dir() else selected_path.parent
            except Exception:
                return None

        def _restore_filtering_ui() -> None:
            self._filtering_in_progress = False
            try:
                self._set_filter_button_busy(False)
                self.primary_action_btn.setText(self._primary_button_label())
                self.primary_action_btn.setToolTip(self._primary_button_tooltip())
                self.primary_action_btn.setEnabled(True)
                self.preview_action_btn.setEnabled(True)
            except Exception:
                pass

        def _run_post_filter_finalize() -> None:
            output_folder = _resolve_output_folder()
            try:
                _set_progress(90, 'Finishing', 'Updating analysis workbooks…')
                QApplication.processEvents()
                if hasattr(self.parent_window, 'log_output'):
                    self.parent_window.log_output('[Filtering] Updating analysis workbooks from filtered foci…\n')
                self._propagate_paths_to_analyze_stage(output_folder)
                from bactomate.gui.analysis_workbook import refresh_workbooks_after_filtering
                wb_logs = refresh_workbooks_after_filtering(self, mask_files, prefer_filtered=True)
                if hasattr(self.parent_window, 'log_output'):
                    for line in wb_logs:
                        self.parent_window.log_output(line)
            except Exception as exc:
                if hasattr(self.parent_window, 'log_output'):
                    self.parent_window.log_output(f'[Filtering] Could not refresh analysis workbook(s): {exc}\n')
            from bactomate.gui.filter_preview_utils import request_filter_preview_apply
            try:
                _set_progress(96, 'Finishing', 'Refreshing open filter preview…')
                QApplication.processEvents()
            except Exception:
                pass
            if request_filter_preview_apply(self, blocking=True):
                if hasattr(self.parent_window, 'log_output'):
                    self.parent_window.log_output('[Filtering] Filter preview refreshed (Apply current filters).\n')
            try:
                _set_progress(100, 'Complete', '')
                QApplication.processEvents()
            except Exception:
                pass
            _close_progress_dialog()
            try:
                self.workflow_step_finished.emit(True)
            except Exception:
                pass
            _restore_filtering_ui()
            if not cancelled:
                self._show_finished_dialog_with_output_folder(title='Filtering complete', message='Filtering finished. Filtered artifacts were written next to the masks.\n\nThe Analyze path was updated to match. Existing *_analysis.xlsx files were refreshed from filtered foci CSVs when found.', output_folder=output_folder, on_ok=self._advance_segmentation_shell_to_analyze)
        if cancelled:
            _close_progress_dialog()
            if hasattr(self.parent_window, 'log_output'):
                self.parent_window.log_output('[Filtering] Cancelled by user.\n')
            _restore_filtering_ui()
            show_message_dialog(self, 'Filtering cancelled', 'Filtering was cancelled.', kind='info')
            return
        QTimer.singleShot(0, _run_post_filter_finalize)

    def get_session_params(self) -> dict[str, Any]:
        d = super().get_session_params()
        fs = self.get_filter_settings()
        d['filter_settings'] = dict(fs)
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        super().set_session_params(data)
        fs = data.get('filter_settings', None)
        if isinstance(fs, dict):
            self.set_filter_settings(fs)
            return
        legacy = {}
        for k in ('filter_prob_enabled', 'filter_prob_thresh', 'filter_min_int_enabled', 'filter_min_int', 'filter_per_cell_ratio_enabled', 'filter_per_cell_ratio'):
            if k in data:
                legacy[k] = data.get(k)
        if legacy:
            self.set_filter_settings({'foci_prob_enabled': bool(legacy.get('filter_prob_enabled', False)), 'foci_prob_thresh': float(legacy.get('filter_prob_thresh', 0.5) or 0.5), 'foci_min_int_enabled': bool(legacy.get('filter_min_int_enabled', False)), 'foci_min_int': float(legacy.get('filter_min_int', 0.0) or 0.0), 'foci_per_cell_ratio_enabled': bool(legacy.get('filter_per_cell_ratio_enabled', False)), 'foci_per_cell_ratio': float(legacy.get('filter_per_cell_ratio', 2.0) or 2.0)})
