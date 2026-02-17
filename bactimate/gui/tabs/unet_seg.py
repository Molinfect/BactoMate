"""
U-NET Segment tab: patch-based U-Net segmentation with binary masks.
Training and inference in one tab: configure training data and params, then train;
select a model and input to preview or process.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.log_config import get_logger
from bactimate.gui.paths import get_unet_models_dir
from bactimate.gui.preferences import LAST_DIR_UNET_SEGMENT
from bactimate.gui.mask_utils import (
    UNET_MASKS_TILED_SUBDIR,
    get_training_pairs,
    infer_training_filters,
    prepare_unet_masks_binary,
    tile_training_data,
)
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon
from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.image_utils import build_segmentation_overlay, scale_for_preview, to_qpixmap


class _UnetTrainWorker(QThread):
    """Runs U-Net training subprocess in background so the GUI stays responsive."""
    log_line = pyqtSignal(str)
    finished_with = pyqtSignal(bool, str)  # success, message

    def __init__(self, cmd: list, model_name: str):
        super().__init__()
        self.cmd = cmd
        self.model_name = model_name
        self._proc = None

    def run(self) -> None:
        import os
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            for line in self._proc.stdout:
                if line:
                    self.log_line.emit(line.rstrip())
            self._proc.wait()
            if self._proc.returncode == 0:
                self.finished_with.emit(True, "Model saved.")
            else:
                self.finished_with.emit(False, f"Training exited with code {self._proc.returncode}.")
        except Exception as e:
            self.finished_with.emit(False, str(e))
        finally:
            self._proc = None


class UNetSegmentTab(QWidget):
    """Tab for U-Net: train a model (all controls inline) and run segmentation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()

    def init_ui(self) -> None:
        self._unet_train_worker = None
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # ----- Section 1: Train a new U-Net model (everything in-tab) -----
        train_group = QGroupBox("Train U-Net model")
        train_layout = QVBoxLayout()

        # Training directory (drag-and-drop supported)
        def handle_dropped_training_path(dropped_path: str) -> None:
            p = Path(dropped_path)
            if p.is_dir():
                self.train_dir_edit.setText(dropped_path)
            elif p.is_file():
                self.train_dir_edit.setText(str(p.parent))

        train_dir_group = DragDropGroupBox(
            "Training directory (images + masks)",
            parent=self,
            on_drop_callback=handle_dropped_training_path,
        )
        train_dir_row = QHBoxLayout()
        train_dir_row.addWidget(QLabel("Path:"))
        train_dir_info = create_info_icon(
            "Folder containing your training data: one image file and one mask file per pair.\n\n"
            "U-Net expects binary masks: background = 0, cells = 255. For each image 'NAME.tif', provide a mask file 'NAME<Mask filter>.tif' (e.g. NAME_mask.tif). "
            "Use 'Auto-detect filters' to infer image/mask suffixes from filenames. Optionally run 'Make masks binary' to convert instance or grayscale masks to [0, 255] in a new subfolder.",
            self,
        )
        train_dir_row.addWidget(train_dir_info)
        self.train_dir_edit = QLineEdit()
        self.train_dir_edit.setPlaceholderText("Path to folder with image/mask pairs (or drag and drop)")
        self.train_dir_edit.setMinimumWidth(400)
        self.train_dir_edit.setAcceptDrops(False)
        train_dir_row.addWidget(self.train_dir_edit, 1)
        browse_train_btn = QPushButton("Browse…")
        browse_train_btn.setAcceptDrops(False)
        browse_train_btn.clicked.connect(self._browse_training_dir)
        train_dir_row.addWidget(browse_train_btn)
        train_dir_group.setLayout(train_dir_row)
        train_layout.addWidget(train_dir_group)

        # Filters
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Image filter:"))
        self.img_filter_edit = QLineEdit()
        self.img_filter_edit.setPlaceholderText("empty or _img")
        self.img_filter_edit.setMinimumWidth(100)
        filter_row.addWidget(self.img_filter_edit)
        filter_row.addWidget(QLabel("Mask filter:"))
        self.mask_filter_edit = QLineEdit()
        self.mask_filter_edit.setText("_mask")
        self.mask_filter_edit.setMinimumWidth(100)
        filter_row.addWidget(self.mask_filter_edit)
        auto_detect_btn = QPushButton("Auto-detect filters")
        auto_detect_btn.clicked.connect(self._on_auto_detect_filters)
        filter_row.addWidget(auto_detect_btn)
        filter_info = create_info_icon(
            "Suffix that identifies image vs mask files.\n\n"
            "Image filter: leave empty if filenames are e.g. A.tif; use _img if you have A_img.tif.\n"
            "Mask filter: suffix on mask files. With _mask, mask for 'cell1.tif' must be 'cell1_mask.tif'. With _segmentation use 'cell1_segmentation.tif'. Must match your filenames exactly.\n\n"
            "'Auto-detect filters' infers these suffixes from the folder and fills the fields, then reports how many image/mask pairs were found.",
            self,
        )
        filter_row.addWidget(filter_info)
        train_layout.addLayout(filter_row)

        # Make binary / Tile
        prep_row = QHBoxLayout()
        make_binary_btn = QPushButton("Make masks binary (U-Net)")
        make_binary_btn.clicked.connect(self._on_make_binary)
        prep_row.addWidget(make_binary_btn)
        make_binary_info = create_info_icon(
            "Convert all masks to strict binary [0, 255] and copy images into unet_masks_binary/.\n\n"
            "Logic: 0 stays 0 (background); any value from 1 to whatever (instance IDs 1, 2, 3, … or grayscale) becomes 255 (foreground). No normalization or threshold.\n\n"
            "U-Net training expects: background = 0, cells = 255. Original files are not modified. After running, you can set Training directory to the new subfolder.",
            self,
        )
        prep_row.addWidget(make_binary_info)
        tile_btn = QPushButton("Tile and prepare for training")
        tile_btn.clicked.connect(self._on_tile)
        prep_row.addWidget(tile_btn)
        tile_info = create_info_icon(
            "Split images into smaller tiles and drop mostly-empty tiles. Use when images are large or sparse.\n\n"
            "Output goes to unet_masks_tiled/. Each run clears that folder. After tiling, set Training directory to the tiled folder if you want to train on tiles.",
            self,
        )
        prep_row.addWidget(tile_info)
        train_layout.addLayout(prep_row)

        # Epochs, Batch, Model name, GPU
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Epochs:"))
        epochs_info = create_info_icon(
            "Number of full passes over the training set. Start with 50; increase (e.g. 100–200) if loss is still decreasing. Early stopping may end training earlier.",
            self,
        )
        params_row.addWidget(epochs_info)
        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setRange(1, 5000)
        self.n_epochs_spin.setValue(50)
        params_row.addWidget(self.n_epochs_spin)
        params_row.addWidget(QLabel("Batch size:"))
        batch_info = create_info_icon(
            "Number of patches per training step. Larger (e.g. 16) trains faster with enough GPU memory; smaller (e.g. 4–8) uses less memory.",
            self,
        )
        params_row.addWidget(batch_info)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        params_row.addWidget(self.batch_spin)
        params_row.addWidget(QLabel("Model name:"))
        name_info = create_info_icon(
            "Short name for your model (letters, numbers, underscores). Saved as name.keras in BactiMate's U-Net models folder and will appear in the Model dropdown below.",
            self,
        )
        params_row.addWidget(name_info)
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("e.g. my_unet_v1")
        self.model_name_edit.setMinimumWidth(180)
        params_row.addWidget(self.model_name_edit, 1)
        self.use_gpu_cb = QCheckBox("Use GPU")
        self.use_gpu_cb.setChecked(True)
        params_row.addWidget(self.use_gpu_cb)
        train_layout.addLayout(params_row)

        # Training log + Start training
        log_label_row = QHBoxLayout()
        log_label_row.addWidget(QLabel("Training log:"))
        log_info = create_info_icon(
            "Output from patch config analysis and training. Check here for errors or progress. When training finishes, the new model appears in the Model dropdown below.",
            self,
        )
        log_label_row.addWidget(log_info)
        log_label_row.addStretch()
        train_layout.addLayout(log_label_row)
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(130)
        train_layout.addWidget(self.training_log)
        self.start_training_btn = QPushButton("Start training")
        self.start_training_btn.clicked.connect(self._start_training)
        train_layout.addWidget(self.start_training_btn)

        train_group.setLayout(train_layout)
        scroll_layout.addWidget(train_group)

        # ----- Section 2: Run segmentation (inference) -----
        infer_group = QGroupBox("Run segmentation")
        infer_layout = QVBoxLayout()

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        model_info = create_info_icon(
            "Select a trained U-Net model (.keras) to use for segmentation. Models are listed from BactiMate's U-Net models folder. Click Refresh after training a new model.",
            self,
        )
        model_row.addWidget(model_info)
        self.model_combo = QComboBox()
        self._refresh_model_combo()
        model_row.addWidget(self.model_combo, 1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self._refresh_model_combo)
        model_row.addWidget(refresh_btn)
        infer_layout.addLayout(model_row)

        # Input file/folder for prediction
        file_group = DragDropGroupBox(
            "Input File/Folder (for prediction)",
            parent=self,
            on_drop_callback=self._handle_dropped_path,
        )
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a TIFF file or folder to segment... (or drag and drop)")
        self.file_path.setMinimumWidth(450)
        file_layout.addWidget(self.file_path, 0)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(110)
        file_btn.clicked.connect(self._select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(120)
        folder_btn.clicked.connect(self._select_folder)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        file_group.setLayout(file_layout)
        infer_layout.addWidget(file_group)

        self.preview_btn = QPushButton("Preview U-Net Segmentation")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._preview_segmentation)
        infer_layout.addWidget(self.preview_btn)
        self.process_btn = QPushButton("Process File or Folder")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._process_file_or_folder)
        infer_layout.addWidget(self.process_btn)

        infer_group.setLayout(infer_layout)
        scroll_layout.addWidget(infer_group)

        self.file_path.textChanged.connect(self._update_button_states)
        self._update_button_states()

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _browse_training_dir(self) -> None:
        last = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_UNET_SEGMENT) or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select training directory", last)
        if path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_UNET_SEGMENT, path)
            self.train_dir_edit.setText(path)

    def _on_auto_detect_filters(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str or not Path(folder_str).is_dir():
            QMessageBox.warning(self, "No folder", "Enter a valid training directory first.")
            return
        folder = Path(folder_str)
        img_filter, mask_filter = infer_training_filters(folder)
        if img_filter is None and mask_filter is None:
            QMessageBox.information(
                self,
                "Auto-detect filters",
                "Could not detect image/mask pairs. Ensure the folder contains paired files (e.g. name_phase.tif and name_mask.tif) and try again, or set the filters manually.",
            )
            return
        self.img_filter_edit.setText(img_filter or "")
        self.mask_filter_edit.setText(mask_filter or "_mask")
        pairs = get_training_pairs(folder, img_filter or "", mask_filter or "_mask")
        QMessageBox.information(
            self, "Auto-detect filters", f"Detected filters. Found {len(pairs)} image/mask pair(s)."
        )

    def _on_make_binary(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str or not Path(folder_str).is_dir():
            QMessageBox.warning(self, "No folder", "Set training directory first.")
            return
        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or "_mask"
        conv, skip, errs, out_dir = prepare_unet_masks_binary(Path(folder_str), img_f, mask_f)
        msg = f"Converted {conv}, skipped {skip}. Output: {out_dir}"
        if errs:
            msg += "\nErrors: " + "; ".join(errs[:5])
        QMessageBox.information(self, "Make masks binary", msg)
        if conv + skip > 0:
            self.train_dir_edit.setText(str(out_dir))

    def _on_tile(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str or not Path(folder_str).is_dir():
            QMessageBox.warning(self, "No folder", "Set training directory first.")
            return
        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or "_mask"
        out_path = Path(folder_str).resolve() / UNET_MASKS_TILED_SUBDIR
        n_tiles, n_imgs, errs = tile_training_data(Path(folder_str), img_f, mask_f, 512, 1, out_path)
        msg = f"Written {n_tiles} tiles from {n_imgs} image(s) to {out_path}"
        if errs:
            msg += "\nWarnings: " + "; ".join(errs[:5])
        QMessageBox.information(self, "Tile and prepare", msg)
        if n_tiles > 0:
            self.train_dir_edit.setText(str(out_path))

    def _start_training(self) -> None:
        train_dir_str = self.train_dir_edit.text().strip()
        if not train_dir_str or not Path(train_dir_str).is_dir():
            QMessageBox.warning(self, "Training", "Set a valid training directory.")
            return
        name = (self.model_name_edit.text().strip() or "unet_model").replace("/", "_").replace("\\", "_")
        config_path = get_unet_models_dir() / f"{name}_patch_config.json"
        model_path = get_unet_models_dir() / f"{name}.keras"
        self.training_log.clear()
        self.training_log.append("Running patch config analysis...")
        QApplication.processEvents()
        analyze_cmd = [
            sys.executable, "-m", "bactimate.scripts.unet_analyze_patch_config",
            "--train-path", train_dir_str,
            "--output", str(config_path),
            "--img-filter", self.img_filter_edit.text().strip(),
            "--mask-filter", self.mask_filter_edit.text().strip() or "_mask",
        ]
        try:
            r = subprocess.run(analyze_cmd, capture_output=True, text=True, timeout=300)
            self.training_log.append(r.stdout or "")
            if r.stderr:
                self.training_log.append(r.stderr)
            if r.returncode != 0:
                self.training_log.append("Analyze failed. Check training data and filters.")
                return
        except Exception as e:
            self.training_log.append(f"Analyze error: {e}")
            return
        self.training_log.append("Starting training... (GUI stays responsive)")
        train_cmd = [
            sys.executable, "-m", "bactimate.scripts.unet_train_patches",
            "--train-path", train_dir_str,
            "--config-path", str(config_path),
            "--output-model-path", str(model_path),
            "--epochs", str(self.n_epochs_spin.value()),
            "--batch-size", str(self.batch_spin.value()),
            "--use-gpu", "1" if self.use_gpu_cb.isChecked() else "0",
            "--img-filter", self.img_filter_edit.text().strip(),
            "--mask-filter", self.mask_filter_edit.text().strip() or "_mask",
        ]
        self.start_training_btn.setEnabled(False)
        self.start_training_btn.setText("Training…")
        self._unet_train_worker = _UnetTrainWorker(train_cmd, name)
        self._unet_train_worker.log_line.connect(self._on_training_log_line)
        self._unet_train_worker.finished_with.connect(self._on_training_finished)
        self._unet_train_worker.start()

    def _on_training_log_line(self, line: str) -> None:
        clean = _ANSI_ESCAPE.sub("", line).strip()
        if clean:
            self.training_log.append(clean)
        # Auto-scroll to bottom
        scroll = self.training_log.verticalScrollBar()
        if scroll:
            scroll.setValue(scroll.maximum())

    def _on_training_finished(self, success: bool, message: str) -> None:
        self.start_training_btn.setEnabled(True)
        self.start_training_btn.setText("Start training")
        name = getattr(self._unet_train_worker, "model_name", "")
        if success:
            self.training_log.append("Training finished. " + message)
            self._refresh_model_combo()
            if name:
                idx = self.model_combo.findText(name)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
            QMessageBox.information(self, "Training finished", f"Model '{name}' saved.")
        else:
            self.training_log.append("Training failed. " + message)
            QMessageBox.warning(self, "Training failed", message)
        self._unet_train_worker = None

    def _refresh_model_combo(self) -> None:
        self.model_combo.clear()
        self.model_combo.addItem("(No model selected)")
        try:
            for p in sorted(get_unet_models_dir().glob("*.keras")):
                if p.is_file():
                    self.model_combo.addItem(p.stem, str(p))
        except Exception:
            pass

    def _select_file(self) -> None:
        last = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_UNET_SEGMENT) or ""
        path, _ = QFileDialog.getOpenFileName(self, "Select TIFF file", last, "TIFF (*.tif *.tiff);;All (*)")
        if path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_UNET_SEGMENT, str(Path(path).parent))
            self.file_path.setText(path)
        self._update_button_states()

    def _select_folder(self) -> None:
        last = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_UNET_SEGMENT) or ""
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder", last)
        if folder_path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
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

    def _preview_segmentation(self) -> None:
        """Same flow as Omnipose preview: file picker when folder, then preview dialog with frame slider and overlay."""
        path_str = self.file_path.text().strip()
        if not path_str:
            QMessageBox.warning(self, "Preview", "Please select a TIFF file or folder first.")
            return
        path = Path(path_str)
        if not path.exists():
            QMessageBox.warning(self, "Preview", "Path does not exist.")
            return
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self, "Preview", "Select a trained U-Net model first.")
            return
        config_path = model_path.parent / f"{model_path.stem}_patch_config.json"
        if not config_path.is_file():
            QMessageBox.warning(self, "Preview", f"Config not found: {config_path.name}")
            return
        is_folder = path.is_dir()
        if is_folder:
            tiff_files = sorted(
                [f for f in (set(path.glob("*.tif")) | set(path.glob("*.tiff"))) if f.is_file() and "_masks" not in f.name.lower()],
                key=lambda x: x.name.lower(),
            )
            if not tiff_files:
                QMessageBox.warning(self, "Preview", "No TIFF files found in the selected folder.")
                return
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select File for Preview")
            file_dialog.setMinimumSize(500, 400)
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Select a file to preview segmentation:"))
            file_list = QListWidget()
            for f in tiff_files:
                file_list.addItem(f.name)
            if file_list.count() > 0:
                file_list.setCurrentRow(0)
            file_list.itemDoubleClicked.connect(lambda: file_dialog.accept())
            layout.addWidget(file_list)
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            bb.accepted.connect(file_dialog.accept)
            bb.rejected.connect(file_dialog.reject)
            layout.addWidget(bb)
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
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import numpy as np
        import tifffile
        try:
            from bactimate.scripts.unet_predict_impl import load_model_and_config, predict_one_image
        except ImportError as e:
            QMessageBox.warning(self, "Preview", f"U-Net preview requires TensorFlow: {e}")
            return
        progress = QProgressDialog("Loading file and model...", None, 0, 0, self)
        progress.setWindowTitle("Loading")
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        try:
            progress.setLabelText("Loading TIFF file...")
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
                QMessageBox.warning(self, "Preview", "Unsupported image dimensions.")
                return
            if images.size == 0 or images.shape[0] == 0:
                progress.close()
                QMessageBox.warning(self, "Preview", "The file has no frames.")
                return
            num_frames = images.shape[0]
            progress.setLabelText("Loading U-Net model...")
            QApplication.processEvents()
            model, cfg = load_model_and_config(model_path, config_path)
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Preview", f"Load failed: {e}")
            return
        progress.close()
        model_name = model_path.stem
        d = QDialog(None)
        d.setWindowTitle("U-Net Segmentation Preview")
        d.setMinimumSize(900, 700)
        d.setModal(False)
        layout = QVBoxLayout()
        title_label = QLabel("")
        title_label.setStyleSheet("font-weight: bold; color: #ddd;")
        layout.addWidget(title_label)
        canvas = ImageCanvas()
        canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        canvas.setMinimumSize(400, 300)
        canvas.setStyleSheet("background-color: #1e1e1e;")
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(canvas, 1)
        d.images = images
        d.model = model
        d.cfg = cfg
        d.model_name = model_name
        d.file_name = file_path.name
        d.segmentation_cache = {}
        d.current_frame = 0

        def get_display_frame(idx: int):
            frame = images[idx]
            if frame.ndim == 3:
                return frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
            return frame

        def update_display(frame_idx: int) -> None:
            d.current_frame = frame_idx
            if frame_idx in d.segmentation_cache:
                mask = d.segmentation_cache[frame_idx]
            else:
                try:
                    mask = predict_one_image(d.model, d.cfg, images[frame_idx])
                    mask = (mask > 0).astype(np.uint8)
                    d.segmentation_cache[frame_idx] = mask
                except Exception as e:
                    title_label.setText(f"Frame {frame_idx + 1} of {num_frames} | Error: {e} | {d.file_name}")
                    return
            disp_frame = get_display_frame(frame_idx)
            disp, mask_scaled = scale_for_preview(disp_frame, mask)
            overlay = build_segmentation_overlay(disp, mask_scaled, alpha=0.5)
            pixmap = to_qpixmap(overlay)
            h, w = overlay.shape[:2]
            canvas.set_pixmap(pixmap, (w, h), reset_view=False)
            title_label.setText(f"Frame {frame_idx + 1} of {num_frames} | Model: {d.model_name} | {d.file_name}")

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Frame:"))
        frame_spin = QSpinBox()
        frame_spin.setRange(1, max(1, num_frames))
        frame_spin.setValue(1)
        frame_spin.setPrefix("")
        nav_layout.addWidget(frame_spin)
        nav_layout.addStretch()
        status_label = QLabel("Ready")
        status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        nav_layout.addWidget(status_label)
        layout.addLayout(nav_layout)

        def on_frame_changed(value: int) -> None:
            idx = value - 1
            if 0 <= idx < num_frames:
                update_display(idx)

        frame_spin.valueChanged.connect(on_frame_changed)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(d.close)
        layout.addWidget(close_btn)
        d.setLayout(layout)
        disp0 = get_display_frame(0)
        empty = np.zeros(disp0.shape, dtype=np.uint8)
        disp0_s, _ = scale_for_preview(disp0, empty)
        overlay0 = build_segmentation_overlay(disp0_s, empty, alpha=0.0)
        canvas.set_pixmap(to_qpixmap(overlay0), (overlay0.shape[1], overlay0.shape[0]), reset_view=True)
        title_label.setText(f"Frame 1 of {num_frames} | Segmenting... | Model: {d.model_name} | {d.file_name}")
        d.show()
        d.raise_()
        d.activateWindow()

        def run_first_segment() -> None:
            d.raise_()
            d.activateWindow()
            update_display(0)
            status_label.setText(f"Ready (cached: {len(d.segmentation_cache)})")

        QTimer.singleShot(80, run_first_segment)

    def _current_model_path(self) -> Optional[Path]:
        idx = self.model_combo.currentIndex()
        if idx <= 0:
            return None
        path_str = self.model_combo.currentData()
        if path_str and Path(path_str).is_file():
            return Path(path_str)
        name = self.model_combo.currentText()
        p = get_unet_models_dir() / f"{name}.keras"
        return p if p.is_file() else None

    def _process_file_or_folder(self) -> None:
        path_str = self.file_path.text().strip()
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            QMessageBox.warning(self, "Process", "Path does not exist.")
            return
        model_path = self._current_model_path()
        if not model_path:
            QMessageBox.warning(self, "Process", "Select a trained U-Net model.")
            return
        if path.is_file():
            input_dir = path.parent
            output_dir = path.parent / f"{path.stem}_unet_segmentation"
        else:
            input_dir = path
            output_dir = path / "unet_segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "bactimate.scripts.unet_predict_patches",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-m", str(model_path),
        ]
        self.parent_window.log_output(f"Running U-Net segmentation: {input_dir} -> {output_dir}\n")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode == 0:
                self.parent_window.log_output(f"U-Net segmentation done: {output_dir}\n")
                QMessageBox.information(self, "Done", f"Output saved to:\n{output_dir}")
            else:
                self.parent_window.log_output(f"Failed: {r.stderr or r.stdout}\n")
                QMessageBox.warning(self, "Failed", r.stderr or r.stdout or "Unknown error")
        except subprocess.TimeoutExpired:
            self.parent_window.log_output("Timed out.\n")
            QMessageBox.warning(self, "Timeout", "Segmentation timed out.")
        except Exception as e:
            self.parent_window.log_output(f"Error: {e}\n")
            QMessageBox.warning(self, "Error", str(e))

    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        d = get_from_widgets([
            ("unet_epochs", self.n_epochs_spin),
            ("unet_batch_size", self.batch_spin),
            ("unet_use_gpu", self.use_gpu_cb),
        ])
        d["unet_model"] = self.model_combo.currentText() if self.model_combo.currentIndex() > 0 else ""
        d["unet_img_filter"] = self.img_filter_edit.text().strip()
        d["unet_mask_filter"] = self.mask_filter_edit.text().strip()
        d["unet_model_name"] = self.model_name_edit.text().strip()
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("unet_epochs", self.n_epochs_spin),
            ("unet_batch_size", self.batch_spin),
            ("unet_use_gpu", self.use_gpu_cb),
        ])
        if "unet_model" in data and data["unet_model"]:
            idx = self.model_combo.findText(str(data["unet_model"]))
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if "unet_img_filter" in data:
            self.img_filter_edit.setText(str(data["unet_img_filter"]))
        if "unet_mask_filter" in data:
            self.mask_filter_edit.setText(str(data["unet_mask_filter"]))
        if "unet_model_name" in data:
            self.model_name_edit.setText(str(data["unet_model_name"]))
