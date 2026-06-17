"""
U-NET Segment tab: patch-based U-Net segmentation with binary masks.
Training and inference in one tab: configure training data and params, then train;
select a model and input to preview or process.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
import tifffile
_ANSI_ESCAPE = re.compile('\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])')
from PyQt6.QtCore import Qt, QEventLoop, QThread, QUrl, pyqtSignal, QTimer
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget
from bactomate.gui.layout_helpers import module_path_action_buttons_layout, module_path_action_outer_layout, module_path_drop_row, new_module_path_action_panel, stacked_path_block, style_compact_action_button, style_module_combo, style_module_primary_button, vbox_from_rows
from bactomate.gui.log_config import get_logger
from bactomate.gui.module_shell import detach_mount_widget
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET
from bactomate.gui.paths import get_unet_models_dir
from bactomate.gui.workers import WorkerThread
from bactomate.gui.preferences import LAST_DIR_UNET_SEGMENT
from bactomate.gui.mask_utils import UNET_MASKS_TILED_SUBDIR, get_training_pairs, infer_training_filters, prepare_unet_masks_binary, tile_training_data
from bactomate.gui.session_params import apply_to_widgets, get_from_widgets
from bactomate.gui.widgets import DragDropGroupBox, PathDropWidget, create_info_icon, show_confirm_dialog, show_message_dialog, style_dialog_button_box
from bactomate.gui.workspace_scanner import list_tiff_files
from bactomate.modules.tiff_metadata import read_calibration_from_path_or_original
from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.image_utils import build_segmentation_overlay, scale_for_preview, to_qpixmap

class UNetTrainWidget(QWidget):
    """Wrapper widget hosting the training section of a `UNetSegmentTab`."""

    def __init__(self, owner: 'UNetSegmentTab', parent: Optional[QWidget]=None) -> None:
        super().__init__(parent)
        self._owner = owner
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(owner.take_train_group())
        layout.addStretch(1)
        self.setLayout(layout)

class UNetInferenceWidget(QWidget):
    """Wrapper widget hosting the inference section of a `UNetSegmentTab`."""

    def __init__(self, owner: 'UNetSegmentTab', parent: Optional[QWidget]=None, *, embedded_in_segmentation: bool=False) -> None:
        super().__init__(parent)
        self._owner = owner
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        infer_group = owner.take_infer_group()
        if embedded_in_segmentation and isinstance(infer_group, QGroupBox):
            infer_group.setStyleSheet('QGroupBox { border: none; margin: 0; padding: 0; background: transparent; }')
        layout.addWidget(infer_group)
        self.setLayout(layout)

class UNetSegmentTab(QWidget):
    """Tab for U-Net: train a model (all controls inline) and run segmentation."""
    workflow_step_finished = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget]=None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._module_preview_mount: Optional[QWidget] = None
        self._preview_surface: Optional[Any] = None
        self._embedded_unet_preview: Optional[QDialog] = None
        self._folder_preview_pick: Optional[str] = None
        self._performed_this_session = False
        self.init_ui()

    def init_ui(self) -> None:
        self._unet_train_worker = None
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout()
        self._train_group_widget = QGroupBox('')
        self._train_group_widget.setFlat(True)
        self._train_group_widget.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        train_layout = QVBoxLayout()

        def handle_dropped_training_path(dropped_path: str) -> None:
            p = Path(dropped_path)
            if p.is_dir():
                self.train_dir_edit.setText(dropped_path)
            elif p.is_file():
                self.train_dir_edit.setText(str(p.parent))
        train_dir_group = DragDropGroupBox('', parent=self, on_drop_callback=handle_dropped_training_path, base_stylesheet=FLAT_GROUPBOX_FRAME_STYLESHEET)
        train_dir_group.setFlat(True)
        train_dir_group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        self._unet_train_dir_group_widget = train_dir_group
        self.train_dir_edit = QLineEdit()
        self.train_dir_edit.setPlaceholderText('Path to folder with image/mask pairs (or drag and drop)')
        self.train_dir_edit.setMinimumWidth(0)
        self.train_dir_edit.setAcceptDrops(False)
        browse_train_btn = QPushButton('Folder…')
        browse_train_btn.setAcceptDrops(False)
        browse_train_btn.clicked.connect(self._browse_training_dir)
        train_outer = QVBoxLayout(train_dir_group)
        train_outer.addWidget(stacked_path_block(line_edit=self.train_dir_edit, buttons=(browse_train_btn,), info_icon=create_info_icon("Folder containing your training data: one image file and one mask file per pair.\n\nU-Net expects binary masks: background = 0, cells = 255. For each image 'NAME.tif', provide a mask file 'NAME<Mask filter>.tif' (e.g. NAME_mask.tif). Use 'Auto-detect filters' to infer image/mask suffixes from filenames. Optionally run 'Make masks binary' to convert instance or grayscale masks to [0, 255] in a new subfolder.", self), parent=self))
        train_layout.addWidget(train_dir_group)
        img_filter_row = QHBoxLayout()
        img_filter_row.addWidget(QLabel('Image filter:'))
        self.img_filter_edit = QLineEdit()
        self.img_filter_edit.setPlaceholderText('empty or _img')
        self.img_filter_edit.setMinimumWidth(0)
        img_filter_row.addWidget(self.img_filter_edit, 1)
        mask_filter_row = QHBoxLayout()
        mask_filter_row.addWidget(QLabel('Mask filter:'))
        self.mask_filter_edit = QLineEdit()
        self.mask_filter_edit.setText('_mask')
        self.mask_filter_edit.setMinimumWidth(0)
        mask_filter_row.addWidget(self.mask_filter_edit, 1)
        auto_detect_btn = QPushButton('Auto-detect filters')
        style_compact_action_button(auto_detect_btn)
        auto_detect_btn.clicked.connect(self._on_auto_detect_filters)
        filter_widget = vbox_from_rows(img_filter_row, mask_filter_row, parent=self)
        self._unet_train_filter_widget = filter_widget
        train_layout.addWidget(filter_widget)
        train_layout.addWidget(auto_detect_btn)
        train_layout.addWidget(create_info_icon("Suffix that identifies image vs mask files.\n\nImage filter: leave empty if filenames are e.g. A.tif; use _img if you have A_img.tif.\nMask filter: suffix on mask files. With _mask, mask for 'cell1.tif' must be 'cell1_mask.tif'. With _segmentation use 'cell1_segmentation.tif'. Must match your filenames exactly.\n\n'Auto-detect filters' infers these suffixes from the folder and fills the fields, then reports how many image/mask pairs were found.", self))
        make_binary_btn = QPushButton('Make masks binary (U-Net)')
        style_compact_action_button(make_binary_btn)
        make_binary_btn.clicked.connect(self._on_make_binary)
        train_layout.addWidget(make_binary_btn)
        train_layout.addWidget(create_info_icon('Convert all masks to strict binary [0, 255] and copy images into unet_masks_binary/.\n\nLogic: 0 stays 0 (background); any value from 1 to whatever (instance IDs 1, 2, 3, … or grayscale) becomes 255 (foreground). No normalization or threshold.\n\nU-Net training expects: background = 0, cells = 255. Original files are not modified. After running, you can set Training directory to the new subfolder.', self))
        tile_btn = QPushButton('Tile and prepare for training')
        style_compact_action_button(tile_btn)
        tile_btn.clicked.connect(self._on_tile)
        train_layout.addWidget(tile_btn)
        train_layout.addWidget(create_info_icon('Split images into smaller tiles and drop mostly-empty tiles. Use when images are large or sparse.\n\nOutput goes to unet_masks_tiled/. Each run clears that folder. After tiling, set Training directory to the tiled folder if you want to train on tiles.', self))
        epochs_row = QHBoxLayout()
        epochs_row.addWidget(QLabel('Epochs:'))
        epochs_row.addWidget(create_info_icon('Number of full passes over the training set. Start with 50; increase (e.g. 100–200) if loss is still decreasing. Early stopping may end training earlier.', self))
        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setRange(1, 5000)
        self.n_epochs_spin.setValue(50)
        epochs_row.addWidget(self.n_epochs_spin)
        epochs_row.addStretch(1)
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel('Batch size:'))
        batch_row.addWidget(create_info_icon('Number of patches per training step. Larger (e.g. 16) trains faster with enough GPU memory; smaller (e.g. 4–8) uses less memory.', self))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        batch_row.addWidget(self.batch_spin)
        batch_row.addStretch(1)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel('Model name:'))
        name_row.addWidget(create_info_icon("Short name for your model (letters, numbers, underscores). Saved as name.keras in BactoMate's U-Net models folder and will appear in the Model dropdown below.", self))
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText('e.g. my_unet_v1')
        self.model_name_edit.setMinimumWidth(0)
        name_row.addWidget(self.model_name_edit, 1)
        self.use_gpu_cb = QCheckBox('Use GPU')
        self.use_gpu_cb.setChecked(True)
        train_layout.addWidget(vbox_from_rows(epochs_row, batch_row, name_row, parent=self))
        train_layout.addWidget(self.use_gpu_cb)
        log_label_row = QHBoxLayout()
        log_label_row.addWidget(QLabel('Training log:'))
        log_info = create_info_icon('Output from patch config analysis and training. Check here for errors or progress. When training finishes, the new model appears in the Model dropdown below.', self)
        log_label_row.addWidget(log_info)
        log_label_row.addStretch()
        train_layout.addLayout(log_label_row)
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(130)
        train_layout.addWidget(self.training_log)
        self.start_training_btn = QPushButton('Start training')
        style_module_primary_button(self.start_training_btn)
        self.start_training_btn.clicked.connect(self._start_training)
        train_layout.addWidget(self.start_training_btn)
        self._train_group_widget.setLayout(train_layout)
        self._scroll_layout.addWidget(self._train_group_widget)
        self._infer_group_widget = QGroupBox('')
        self._infer_group_widget.setFlat(True)
        self._infer_group_widget.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        infer_layout = QVBoxLayout()
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Model:'))
        model_info = create_info_icon("Select a trained U-Net model (.keras) to use for segmentation. Models are listed from BactoMate's U-Net models folder. Click Refresh after training a new model.", self)
        model_row.addWidget(model_info)
        self.model_combo = QComboBox()
        self._refresh_model_combo()
        self.model_combo.currentIndexChanged.connect(lambda _idx: QTimer.singleShot(0, self._rerun_open_unet_preview_after_model_change))
        model_row.addWidget(self.model_combo, 1)
        style_module_combo(self.model_combo)
        refresh_btn = QPushButton('Refresh')
        refresh_btn.setFixedWidth(80)
        style_compact_action_button(refresh_btn)
        refresh_btn.clicked.connect(self._refresh_model_combo)
        model_row.addWidget(refresh_btn)
        models_folder_btn = QPushButton('Models Folder')
        style_compact_action_button(models_folder_btn)
        models_folder_btn.setToolTip("Open BactoMate's U-Net models folder (.keras files and patch configs).")
        models_folder_btn.clicked.connect(self._open_unet_models_folder)
        model_row.addWidget(models_folder_btn)
        infer_layout.addLayout(model_row)
        file_group = new_module_path_action_panel(self)
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText('Select a TIFF file or folder to segment... (or drag and drop)')
        self.file_path.setMinimumWidth(0)
        self.file_path.setAcceptDrops(False)
        file_btn = QPushButton('Browse File...')
        file_btn.clicked.connect(self._select_file)
        folder_btn = QPushButton('Browse Folder...')
        folder_btn.clicked.connect(self._select_folder)
        file_outer = module_path_action_outer_layout(file_group)
        file_outer.addWidget(module_path_drop_row(file_group, self._handle_dropped_path, line_edit=self.file_path, buttons=(file_btn, folder_btn), info_icon=create_info_icon('Path to a TIFF file or folder for U-Net segmentation.', self)))
        action_layout = module_path_action_buttons_layout()
        self.preview_btn = QPushButton('Preview segmentation')
        self.preview_btn.setEnabled(False)
        style_module_primary_button(self.preview_btn)
        self.preview_btn.clicked.connect(self._preview_segmentation)
        action_layout.addWidget(self.preview_btn)
        self.process_btn = QPushButton('Segment File/Folder')
        self.process_btn.setEnabled(False)
        style_module_primary_button(self.process_btn)
        self.process_btn.clicked.connect(self._process_file_or_folder)
        action_layout.addWidget(self.process_btn)
        file_outer.addLayout(action_layout)
        infer_layout.addWidget(file_group, 0)
        self._infer_group_widget.setLayout(infer_layout)
        self._scroll_layout.addWidget(self._infer_group_widget)
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
            w = QGroupBox('')
            w.setFlat(True)
            w.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
            self._train_group_widget = w
        try:
            if getattr(self, '_scroll_layout', None) is not None:
                self._scroll_layout.removeWidget(w)
        except Exception:
            pass
        w.setParent(None)
        return w

    def take_infer_group(self) -> QWidget:
        """Detach and return the inference group widget for embedding elsewhere."""
        w = getattr(self, '_infer_group_widget', None)
        if w is None:
            w = QGroupBox('')
            w.setFlat(True)
            w.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
            self._infer_group_widget = w
        try:
            if getattr(self, '_scroll_layout', None) is not None:
                self._scroll_layout.removeWidget(w)
        except Exception:
            pass
        w.setParent(None)
        return w

    def attach_module_preview_mount(self, mount: Optional[QWidget]) -> None:
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
        """Segmentation tab that owns PreviewSurface (see OmniposeSegTab — QWidget.parent() is not SegmentationTab)."""
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

    def _on_embedded_unet_preview_closed(self) -> None:
        self._embedded_unet_preview = None
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

    def _rerun_open_unet_preview_after_model_change(self) -> None:
        """If a U-Net preview is open, rebuild it on the same TIFF when the model changes."""
        d = getattr(self, '_embedded_unet_preview', None)
        if d is None:
            return
        inp = getattr(d, '_segmentation_input_path', None)
        if inp:
            try:
                self._folder_preview_pick = str(inp)
            except Exception:
                pass
        self._preview_segmentation()

    def _browse_training_dir(self) -> None:
        last = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_UNET_SEGMENT) or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, 'Select training directory', last)
        if path:
            if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_UNET_SEGMENT, path)
            self.train_dir_edit.setText(path)

    def _on_auto_detect_filters(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str or not Path(folder_str).is_dir():
            show_message_dialog(self, 'No folder', 'Enter a valid training directory first.', kind='warning')
            return
        folder = Path(folder_str)
        (img_filter, mask_filter) = infer_training_filters(folder)
        if img_filter is None and mask_filter is None:
            show_message_dialog(self, 'Auto-detect filters', 'Could not detect image/mask pairs. Ensure the folder contains paired files (e.g. name_phase.tif and name_mask.tif) and try again, or set the filters manually.', kind='info')
            return
        self.img_filter_edit.setText(img_filter or '')
        self.mask_filter_edit.setText(mask_filter or '_mask')
        pairs = get_training_pairs(folder, img_filter or '', mask_filter or '_mask')
        show_message_dialog(self, 'Auto-detect filters', f'Detected filters. Found {len(pairs)} image/mask pair(s).', kind='info')

    def _on_make_binary(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str or not Path(folder_str).is_dir():
            show_message_dialog(self, 'No folder', 'Set training directory first.', kind='warning')
            return
        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or '_mask'
        (conv, skip, errs, out_dir) = prepare_unet_masks_binary(Path(folder_str), img_f, mask_f)
        msg = f'Converted {conv}, skipped {skip}. Output: {out_dir}'
        if errs:
            msg += '\nErrors: ' + '; '.join(errs[:5])
        show_message_dialog(self, 'Make masks binary', msg, kind='info')
        if conv + skip > 0:
            self.train_dir_edit.setText(str(out_dir))

    def _on_tile(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str or not Path(folder_str).is_dir():
            show_message_dialog(self, 'No folder', 'Set training directory first.', kind='warning')
            return
        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or '_mask'
        out_path = Path(folder_str).resolve() / UNET_MASKS_TILED_SUBDIR
        (n_tiles, n_imgs, errs) = tile_training_data(Path(folder_str), img_f, mask_f, 512, 1, out_path)
        msg = f'Written {n_tiles} tiles from {n_imgs} image(s) to {out_path}'
        if errs:
            msg += '\nWarnings: ' + '; '.join(errs[:5])
        show_message_dialog(self, 'Tile and prepare', msg, kind='info')
        if n_tiles > 0:
            self.train_dir_edit.setText(str(out_path))

    def _start_training(self) -> None:
        train_dir_str = self.train_dir_edit.text().strip()
        if not train_dir_str or not Path(train_dir_str).is_dir():
            show_message_dialog(self, 'Training', 'Set a valid training directory.', kind='warning')
            return
        name = (self.model_name_edit.text().strip() or 'unet_model').replace('/', '_').replace('\\', '_')
        config_path = get_unet_models_dir() / f'{name}_patch_config.json'
        model_path = get_unet_models_dir() / f'{name}.keras'
        self.training_log.clear()
        self.training_log.append('Running patch config analysis...')
        QApplication.processEvents()
        analyze_cmd_args = ['bactomate.scripts.unet_analyze_patch_config', '--train-path', train_dir_str, '--output', str(config_path), '--img-filter', self.img_filter_edit.text().strip(), '--mask-filter', self.mask_filter_edit.text().strip() or '_mask']
        self._unet_analyze_worker = WorkerThread('-m', analyze_cmd_args, parent=self)
        result_holder = [None, None]
        loop = QEventLoop(self)

        def on_analyze_done(success: bool, message: str) -> None:
            (result_holder[0], result_holder[1]) = (success, message)
            loop.quit()
        self._unet_analyze_worker.finished.connect(on_analyze_done)
        self._unet_analyze_worker.progress.connect(lambda line: self.training_log.append(line.strip()) if line.strip() else None)
        self._unet_analyze_worker.start()
        loop.exec()
        try:
            if self._unet_analyze_worker is not None:
                self._unet_analyze_worker.wait(30000)
                self._unet_analyze_worker.deleteLater()
        finally:
            self._unet_analyze_worker = None
        if not result_holder[0]:
            self.training_log.append(result_holder[1] or 'Analyze failed. Check training data and filters.')
            return
        self.training_log.append('Starting training... (GUI stays responsive)')
        train_cmd_args = ['bactomate.scripts.unet_train_patches', '--train-path', train_dir_str, '--config-path', str(config_path), '--output-model-path', str(model_path), '--epochs', str(self.n_epochs_spin.value()), '--batch-size', str(self.batch_spin.value()), '--use-gpu', '1' if self.use_gpu_cb.isChecked() else '0', '--img-filter', self.img_filter_edit.text().strip(), '--mask-filter', self.mask_filter_edit.text().strip() or '_mask']
        self.start_training_btn.setEnabled(False)
        self.start_training_btn.setText('Training…')
        self._unet_train_model_name = name
        self._unet_train_worker = WorkerThread('-m', train_cmd_args, parent=self)
        self._unet_train_worker.progress.connect(self._on_training_log_line)
        self._unet_train_worker.finished.connect(self._on_training_finished)
        self._unet_train_worker.start()

    def _on_training_log_line(self, line: str) -> None:
        clean = _ANSI_ESCAPE.sub('', line).strip()
        if clean:
            self.training_log.append(clean)
        scroll = self.training_log.verticalScrollBar()
        if scroll:
            scroll.setValue(scroll.maximum())

    def _on_training_finished(self, success: bool, message: str) -> None:
        self.start_training_btn.setEnabled(True)
        self.start_training_btn.setText('Start training')
        name = getattr(self, '_unet_train_model_name', '')
        if success:
            self.training_log.append('Training finished. ' + message)
            self._refresh_model_combo()
            if name:
                idx = self.model_combo.findText(name)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
            show_message_dialog(self, 'Training finished', f"Model '{name}' saved.", kind='info')
        else:
            self.training_log.append('Training failed. ' + message)
            show_message_dialog(self, 'Training failed', message, kind='warning')
        self._unet_train_worker = None

    def _refresh_model_combo(self) -> None:
        self.model_combo.clear()
        self.model_combo.addItem('(No model selected)')
        try:
            for p in sorted(get_unet_models_dir().glob('*.keras')):
                if p.is_file():
                    self.model_combo.addItem(p.stem, str(p))
        except Exception:
            pass

    def _select_file(self) -> None:
        last = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_UNET_SEGMENT) or ''
        (path, _) = QFileDialog.getOpenFileName(self, 'Select TIFF file', last, 'TIFF (*.tif *.tiff);;All (*)')
        if path:
            if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_UNET_SEGMENT, str(Path(path).parent))
            self.file_path.setText(path)
        self._update_button_states()

    def _select_folder(self) -> None:
        last = getattr(self.parent_window, 'get_last_dir_for_tab', lambda _: '')(LAST_DIR_UNET_SEGMENT) or ''
        folder_path = QFileDialog.getExistingDirectory(self, 'Select folder', last)
        if folder_path:
            if getattr(self.parent_window, 'set_last_dir_for_tab', None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_UNET_SEGMENT, folder_path)
            self.file_path.setText(folder_path)
        self._update_button_states()

    def _handle_dropped_path(self, dropped_path: str) -> None:
        self.file_path.setText(dropped_path)
        self._update_button_states()

    def _update_button_states(self) -> None:
        has_path = bool(self.file_path.text().strip())
        self.preview_btn.setEnabled(has_path)
        self.process_btn.setEnabled(has_path)

    def _try_push_metadata_from_path(self) -> None:
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
                if pp.is_file() and pp.suffix.lower() in ('.tif', '.tiff') and ('_masks' not in pp.name.lower()):
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

    def _preview_segmentation(self) -> None:
        """Same flow as Omnipose preview: file picker when folder, then preview dialog with frame slider and overlay."""
        path_str = self.file_path.text().strip()
        if not path_str:
            show_message_dialog(self, 'Preview', 'Please select a TIFF file or folder first.', kind='warning')
            return
        path = Path(path_str)
        if not path.exists():
            show_message_dialog(self, 'Preview', 'Path does not exist.', kind='warning')
            return
        model_path = self._current_model_path()
        if not model_path:
            self._show_missing_unet_model_dialog(context='preview')
            return
        config_path = model_path.parent / f'{model_path.stem}_patch_config.json'
        if not config_path.is_file():
            show_message_dialog(self, 'Preview', f'Config not found: {config_path.name}', kind='warning')
            return
        is_folder = path.is_dir()
        if is_folder:

            def _is_unet_preview_input(p: Path) -> bool:
                return p.is_file() and '_masks' not in p.name.lower()
            tiff_files = sorted([f for f in list_tiff_files(path, max_files=5000) if _is_unet_preview_input(f)], key=lambda x: x.name.lower())
            if not tiff_files:
                show_message_dialog(self, 'Preview', 'No .tif / .tiff files found under the selected folder (recursive scan).', kind='warning')
                return
            pick = (getattr(self, '_folder_preview_pick', None) or '').strip()
            chosen: Optional[Path] = None
            if pick:
                try:
                    pp = Path(pick)
                    if pp.is_file() and _is_unet_preview_input(pp):
                        rp = pp.resolve()
                        fr = path.resolve()
                        if fr in rp.parents or rp.parent == fr:
                            chosen = pp
                except OSError:
                    chosen = None
            if chosen is not None:
                file_path = chosen
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
                    file_list.setCurrentRow(0)

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
                item = file_list.currentItem()
                if not item:
                    return
                selected_file = next((f for f in tiff_files if f.name == item.text()), None)
                if not selected_file:
                    return
                file_path = selected_file
        else:
            file_path = path
        self._open_unet_preview_dialog(file_path, model_path, config_path)

    def _open_unet_preview_dialog(self, file_path: Path, model_path: Path, config_path: Path) -> None:
        """Load file and model, open preview dialog with frame slider and overlay (same UX as Omnipose)."""
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import numpy as np
        import tifffile
        try:
            from bactomate.scripts.unet_predict_impl import load_model_and_config, predict_one_image
        except ImportError as e:
            show_message_dialog(self, 'Preview', f'U-Net preview requires TensorFlow: {e}', kind='warning')
            return
        progress = QProgressDialog('Loading file and model...', None, 0, 0, self)
        progress.setWindowTitle('Loading')
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        try:
            progress.setLabelText('Loading TIFF file...')
            QApplication.processEvents()
            arr = tifffile.imread(str(file_path))
            if arr.ndim == 2:
                images = np.asarray(arr)[np.newaxis, ...]
            elif arr.ndim == 3:
                if arr.shape[-1] in (2, 3, 4):
                    images = np.asarray(arr, dtype=np.float32)[..., 0]
                    images = images[np.newaxis, ...]
                else:
                    images = np.asarray(arr, dtype=np.float32)
            else:
                progress.close()
                show_message_dialog(self, 'Preview', 'Unsupported image dimensions.', kind='warning')
                return
            if images.size == 0 or images.shape[0] == 0:
                progress.close()
                show_message_dialog(self, 'Preview', 'The file has no frames.', kind='warning')
                return
            num_frames = images.shape[0]
            progress.setLabelText('Loading U-Net model...')
            QApplication.processEvents()
            (model, cfg) = load_model_and_config(model_path, config_path)
        except Exception as e:
            progress.close()
            show_message_dialog(self, 'Preview', f'Load failed: {e}', kind='warning')
            return
        progress.close()
        model_name = model_path.stem
        main_win = self.window() if hasattr(self, 'window') and callable(self.window) else None
        mount = getattr(self, '_module_preview_mount', None)
        dlg_parent = mount if mount is not None else main_win
        embed_u = mount is not None
        from bactomate.gui.preview.preview_window import PreviewDialogBase, WorkerGuard
        from bactomate.gui.preview.navigation_controls import PreviewNavSpec, PreviewNavState, PreviewNavigationControls
        from bactomate.gui.preview.render_utils import apply_lut_to_uint8
        from bactomate.gui.preview.preview_window import RenderResult
        d = PreviewDialogBase(dlg_parent, title='Segmentation Preview', primary_action_label='Re-segment', primary_action_callback=None, render_callback=None, worker_guard=None, minimum_size=(900, 700), embedded=embed_u, show_close_button=not embed_u)
        title_label = d.title_label
        d.images = images
        d.model = model
        d.cfg = cfg
        d.model_name = model_name
        d.file_name = file_path.name
        d.segmentation_cache = {}
        d.current_frame = 0
        d._unet_request_id = 0
        d._unet_current_requested_frame = 0
        d._unet_worker = None
        d._unet_progress_dialog = None
        d._unet_worker_slot = None
        d._dialog_closed = False
        d.set_extra('show_cells', True)
        d.set_extra('show_foci', False)

        def _unet_is_running() -> bool:
            w = getattr(d, '_unet_worker', None)
            return bool(w is not None and w.isRunning())

        def _unet_request_stop() -> None:
            return

        def _unet_wait(ms: int) -> bool:
            w = getattr(d, '_unet_worker', None)
            if w is None:
                return True
            try:
                return bool(w.wait(int(ms)))
            except Exception:
                return True

        def _unet_disconnect() -> None:
            w = getattr(d, '_unet_worker', None)
            s = getattr(d, '_unet_worker_slot', None)
            if w is not None and s is not None:
                try:
                    w.finished.disconnect(s)
                except Exception:
                    pass
            try:
                d._unet_worker = None
            except Exception:
                pass
            try:
                d._unet_worker_slot = None
            except Exception:
                pass
        try:
            d.set_worker_guard(WorkerGuard(is_running=_unet_is_running, request_stop=_unet_request_stop, wait=_unet_wait, disconnect=_unet_disconnect, on_close_while_running='block_with_warning', warning_title='Segmentation in progress', warning_text='Please wait for the current segmentation to finish before closing the preview.'))
        except Exception:
            pass

        def get_display_frame(idx: int):
            frame = images[idx]
            if frame.ndim == 3:
                return frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
            return frame

        class UnetPreviewWorker(QThread):
            """Runs predict_one_image in background; emits (request_id, frame_idx, mask, error_str)."""
            finished = pyqtSignal(int, int, object, str)

            def __init__(self, request_id: int, frame_idx: int, model: Any, cfg: Any, image: np.ndarray):
                super().__init__()
                self.setObjectName(type(self).__name__)
                self._request_id = request_id
                self._frame_idx = frame_idx
                self._model = model
                self._cfg = cfg
                self._image = np.asarray(image).copy()

            def run(self) -> None:
                if self.isInterruptionRequested():
                    self.finished.emit(self._request_id, self._frame_idx, None, 'cancelled')
                    return
                (mask, err) = (None, '')
                try:
                    mask = predict_one_image(self._model, self._cfg, self._image)
                    mask = (mask > 0).astype(np.uint8)
                except Exception as e:
                    err = str(e)
                self.finished.emit(self._request_id, self._frame_idx, mask, err)

        def _apply_mask_to_ui(frame_idx: int, mask: np.ndarray) -> None:
            """Apply segmentation result to cache + refresh preview."""
            d.segmentation_cache[frame_idx] = mask
            num_cells = len(np.unique(mask)) - 1
            title_label.setText(f'Frame {frame_idx + 1} of {num_frames} | Cells: {num_cells} | Model: {d.model_name} | {d.file_name}\nParameters: model={d.model_name}')
            if status_label:
                status_label.setText(f'Ready (cached: {len(d.segmentation_cache)})')
            if frame_spin is not None:
                frame_spin.setEnabled(True)
            try:
                d.refresh()
            except Exception:
                pass

        def update_display(frame_idx: int) -> None:
            d.current_frame = frame_idx
            if frame_idx in d.segmentation_cache:
                if status_label:
                    status_label.setText(f'Ready (cached: {len(d.segmentation_cache)})')
                try:
                    d.refresh()
                except Exception:
                    pass
                return
            existing = getattr(d, '_unet_worker', None)
            if existing is not None and existing.isRunning():
                d._unet_request_id += 1
                d._unet_current_requested_frame = frame_idx
                if status_label:
                    status_label.setText('Segmenting... (requested frame %s)' % (frame_idx + 1))
                title_label.setText(f'Frame {frame_idx + 1} of {num_frames} | Segmenting... | Model: {d.model_name} | {d.file_name}\nParameters: model={d.model_name}')
                return
            d._unet_request_id += 1
            d._unet_current_requested_frame = frame_idx
            progress = QProgressDialog('Segmenting frame...', None, 0, 0, d)
            progress.setWindowTitle('Processing')
            progress.setAutoClose(False)
            progress.setCancelButton(None)
            progress.show()
            QApplication.processEvents()
            d._unet_progress_dialog = progress

            def _on_worker_finished(rid: int, fid: int, m: object, err: str) -> None:
                try:
                    if getattr(d, '_dialog_closed', False):
                        return
                    if rid != getattr(d, '_unet_request_id', -1):
                        update_display(getattr(d, '_unet_current_requested_frame', 0))
                        return
                    prog = getattr(d, '_unet_progress_dialog', None)
                    if prog is not None:
                        try:
                            prog.close()
                        except Exception:
                            pass
                        d._unet_progress_dialog = None
                    d._unet_worker = None
                    if err:
                        title_label.setText(f'Frame {fid + 1} of {num_frames} | Error: {err[:60]} | {d.file_name}\nParameters: model={d.model_name}')
                        if status_label:
                            status_label.setText('Segmentation failed')
                        if frame_spin is not None:
                            frame_spin.setEnabled(True)
                        return
                    _apply_mask_to_ui(fid, m)
                except RuntimeError:
                    pass
            if frame_spin is not None:
                frame_spin.setEnabled(False)
            worker = UnetPreviewWorker(d._unet_request_id, frame_idx, d.model, d.cfg, images[frame_idx])
            d._unet_worker_slot = _on_worker_finished
            worker.finished.connect(_on_worker_finished)
            d._unet_worker = worker
            worker.start()
            title_label.setText(f'Frame {frame_idx + 1} of {num_frames} | Segmenting... | Model: {d.model_name} | {d.file_name}\nParameters: model={d.model_name}')
            if status_label:
                status_label.setText('Segmenting...')
        nav_layout = QHBoxLayout()
        frame_spin = None
        scale_label = QLabel('')
        nav_layout.addWidget(scale_label)
        nav_layout.addStretch()
        status_label = QLabel('Ready')
        status_label.setStyleSheet('color: #7F8C8D; font-size: 9pt; font-style: italic;')
        nav_layout.addWidget(status_label)
        bottom_extra = QWidget()
        bottom_extra.setLayout(nav_layout)
        nav = PreviewNavigationControls(spec=PreviewNavSpec(frame_count=max(1, num_frames), show_image=False, show_channel=False, show_z=False, show_cell_overlay_toggle=True, show_foci_overlay_toggle=True, cell_overlay_default=True, foci_overlay_available=False, foci_overlay_default=False, debounce_ms=400))
        nav.set_state(PreviewNavState(frame_idx=0), emit=False)
        nav_host = QWidget()
        nav_host_layout = QHBoxLayout(nav_host)
        nav_host_layout.setContentsMargins(0, 0, 0, 0)
        nav_host_layout.setSpacing(10)
        nav_host_layout.addWidget(nav)
        nav_host_layout.addWidget(bottom_extra)
        d.set_navigation_controls(nav_host)

        def _reset_lut() -> None:
            try:
                d.state().vmin = None
                d.state().vmax = None
                d.lut_widget.set_lut(None, None)
            except Exception:
                pass

        def _render(state) -> RenderResult:
            fi = max(0, min(int(state.frame_idx), max(0, num_frames - 1)))
            disp_frame = np.asarray(get_display_frame(fi), dtype=np.float32)
            mask = d.segmentation_cache.get(fi, None)
            if mask is None:
                mask = np.zeros(disp_frame.shape[:2], dtype=np.uint8)
            else:
                mask = np.asarray(mask)
            u8 = apply_lut_to_uint8(disp_frame, state.vmin, state.vmax)
            (disp_scaled, mask_scaled) = scale_for_preview(u8, mask)
            if bool(state.extra.get('show_cells', True)):
                overlay = build_segmentation_overlay(disp_scaled, mask_scaled, alpha=0.5)
            else:
                overlay = disp_scaled
            pm = to_qpixmap(overlay)
            num_cells = max(0, int(len(np.unique(mask)) - 1)) if mask is not None else 0
            title = f'Frame {fi + 1} of {num_frames} | Cells: {num_cells} | Model: {d.model_name} | {d.file_name}\nParameters: model={d.model_name}'
            info = 'Ready' if fi in d.segmentation_cache else 'Segmenting... (not cached yet)'
            return RenderResult(pixmap=pm, image_size=(int(overlay.shape[1]), int(overlay.shape[0])), lut_frame=np.asarray(disp_frame, dtype=np.float32), title_text=title, info_text=info)
        try:
            d.set_render_callback(_render)
        except Exception:
            pass

        def go_to_frame(idx: int) -> None:
            if 0 <= idx < num_frames:
                d.current_frame = idx
                update_display(idx)

        def _on_nav_changed(st: PreviewNavState, reason: str) -> None:
            d.set_frame_idx(int(st.frame_idx))
            if reason in ('frame', 'init'):
                go_to_frame(int(st.frame_idx))
            d.refresh()
        nav.nav_changed.connect(_on_nav_changed)

        def _on_overlay_changed(key: str, checked: bool) -> None:
            d.set_extra(str(key), bool(checked))
            d.refresh()
        nav.overlay_changed.connect(_on_overlay_changed)

        def on_resegment() -> None:
            try:
                if d.current_frame in d.segmentation_cache:
                    del d.segmentation_cache[d.current_frame]
            except Exception:
                pass
            _reset_lut()
            update_display(d.current_frame)
        try:
            d.set_primary_action(on_resegment, label='Re-segment')
        except Exception:
            pass
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, d)
        close_shortcut.activated.connect(d.close)
        minimize_shortcut = QShortcut(QKeySequence('Ctrl+M'), d)
        minimize_shortcut.activated.connect(d.showMinimized)
        title_label.setText(f'Frame 1 of {num_frames} | Segmenting... | Model: {d.model_name} | {d.file_name}\nParameters: model={d.model_name}')
        prev_u = getattr(self, '_embedded_unet_preview', None)
        if prev_u is not None and prev_u is not d:
            detach_mount_widget(prev_u)
        self._embedded_unet_preview = d
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
            lay.addWidget(d, 1)
            if surf is not None:
                try:
                    surf.set_preview_chrome_visible(False)
                except Exception:
                    pass
        d.finished.connect(lambda _r=0: self._on_embedded_unet_preview_closed())
        d.show()
        if mount is None:
            d.raise_()
            d.activateWindow()

        def run_first_segment() -> None:
            if mount is None:
                d.raise_()
                d.activateWindow()
            try:
                d.set_frame_idx(0)
            except Exception:
                pass
            try:
                d.refresh()
            except Exception:
                pass
            update_display(0)
            status_label.setText(f'Ready (cached: {len(d.segmentation_cache)})')
        QTimer.singleShot(80, run_first_segment)

    def _current_model_path(self) -> Optional[Path]:
        idx = self.model_combo.currentIndex()
        if idx <= 0:
            return None
        path_str = self.model_combo.currentData()
        if path_str and Path(path_str).is_file():
            return Path(path_str)
        name = self.model_combo.currentText()
        p = get_unet_models_dir() / f'{name}.keras'
        return p if p.is_file() else None

    def _open_unet_models_folder(self) -> None:
        models_dir = get_unet_models_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(models_dir)))

    def _show_missing_unet_model_dialog(self, *, context: str) -> None:
        models_dir = get_unet_models_dir()
        action = 'preview segmentation' if context == 'preview' else 'run U-Net segmentation'
        should_open = show_confirm_dialog(self, title='Select a U-Net model', message=f'Choose a trained U-Net model before you {action}.\n\nBactoMate saves trained U-Net models as .keras files in:\n{models_dir}', confirm_label='Open Models Folder', cancel_label='OK', default_confirm=False)
        if should_open:
            self._open_unet_models_folder()

    def _process_file_or_folder(self) -> None:
        path_str = self.file_path.text().strip()
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            show_message_dialog(self, 'Process', 'Path does not exist.', kind='warning')
            return
        model_path = self._current_model_path()
        if not model_path:
            self._show_missing_unet_model_dialog(context='process')
            return
        if path.is_file():
            input_dir = path.parent
            if path.parent.name.startswith('Tif_'):
                output_dir = path.parent / '_segmentation_unet'
            else:
                output_dir = path.parent / f'{path.stem}_unet_segmentation'
        else:
            input_dir = path
            output_dir = path / '_segmentation_unet'
        output_dir.mkdir(parents=True, exist_ok=True)
        predict_cmd_args = ['bactomate.scripts.unet_predict_patches', '-i', str(input_dir), '-o', str(output_dir), '-m', str(model_path)]
        self.parent_window.log_output(f'Running U-Net segmentation: {input_dir} -> {output_dir}\n')
        predict_worker = WorkerThread('-m', predict_cmd_args, parent=self)
        self.worker = predict_worker
        result_holder = [None, None]
        loop = QEventLoop(self)

        def on_predict_done(success: bool, message: str) -> None:
            (result_holder[0], result_holder[1]) = (success, message)
            loop.quit()
        predict_worker.finished.connect(on_predict_done)
        predict_worker.progress.connect(self.parent_window.log_output)
        predict_worker.start()
        loop.exec()
        if result_holder[0]:
            self.parent_window.log_output(f'U-Net segmentation done: {output_dir}\n')
            self._performed_this_session = True
            self._log_unet_result_summaries(path, output_dir)
            try:
                self.workflow_step_finished.emit(True)
            except Exception:
                pass
            from bactomate.gui.widgets import show_task_complete_dialog
            show_task_complete_dialog(self, title='Done', message=f'Output saved to:\n{output_dir}', output_dir=output_dir, on_ok=self._advance_segmentation_shell_to_filtering)
        else:
            self.parent_window.log_output((result_holder[1] or 'Unknown error') + '\n')
            try:
                self.workflow_step_finished.emit(False)
            except Exception:
                pass
            show_message_dialog(self, 'Failed', result_holder[1] or 'Unknown error', kind='warning')

    def process_file_or_folder(self) -> None:
        """Public entry for composed Segmentation tab and workflow replay."""
        self._process_file_or_folder()

    def _summarize_unet_mask_file(self, mask_path: Path) -> Optional[dict[str, Any]]:
        try:
            arr = np.asarray(tifffile.imread(str(mask_path)))
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim < 2:
            return None
        frame_counts: list[int] = []
        for i in range(arr.shape[0]):
            frame = arr[i]
            from skimage.measure import label as sk_label
            labeled = sk_label(frame > 0, connectivity=1)
            n_cells = int(labeled.max()) if labeled.size else 0
            frame_counts.append(n_cells)
        return {'frames': int(len(frame_counts)), 'cells_total': int(sum(frame_counts)), 'cells_min': int(min(frame_counts)), 'cells_max': int(max(frame_counts)), 'cells_mean': float(sum(frame_counts) / len(frame_counts))}

    def _log_unet_result_summaries(self, input_path: Path, output_dir: Path) -> None:
        mask_files = sorted(output_dir.glob('*_unet_masks.tif'))
        self.parent_window.log_output(f'[RESULT][SEGMENT] engine=unet input={input_path.resolve()} output_dir={output_dir.resolve()} mask_files={len(mask_files)}\n')
        for mask_file in mask_files:
            summary = self._summarize_unet_mask_file(mask_file)
            if summary is None:
                self.parent_window.log_output(f'[RESULT][SEGMENT] engine=unet mask={mask_file.resolve()} cells=unavailable\n')
                continue
            self.parent_window.log_output(f"[RESULT][SEGMENT] engine=unet mask={mask_file.resolve()} frames={summary['frames']} total_cells={summary['cells_total']} cells_min={summary['cells_min']} cells_mean={summary['cells_mean']:.2f} cells_max={summary['cells_max']}\n")

    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        d = get_from_widgets([('unet_epochs', self.n_epochs_spin), ('unet_batch_size', self.batch_spin), ('unet_use_gpu', self.use_gpu_cb)])
        d['unet_model'] = self.model_combo.currentText() if self.model_combo.currentIndex() > 0 else ''
        d['unet_img_filter'] = self.img_filter_edit.text().strip()
        d['unet_mask_filter'] = self.mask_filter_edit.text().strip()
        d['unet_model_name'] = self.model_name_edit.text().strip()
        d['performed'] = bool(getattr(self, '_performed_this_session', False))
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [('unet_epochs', self.n_epochs_spin), ('unet_batch_size', self.batch_spin), ('unet_use_gpu', self.use_gpu_cb)])
        if 'unet_model' in data and data['unet_model']:
            idx = self.model_combo.findText(str(data['unet_model']))
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if 'unet_img_filter' in data:
            self.img_filter_edit.setText(str(data['unet_img_filter']))
        if 'unet_mask_filter' in data:
            self.mask_filter_edit.setText(str(data['unet_mask_filter']))
        if 'unet_model_name' in data:
            self.model_name_edit.setText(str(data['unet_model_name']))
        if 'performed' in data:
            self._performed_this_session = bool(data['performed'])
