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
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, QCoreApplication, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QPixmap, QImage, QPainter, QColor, QPen, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.constants import PROJECT_ROOT
from bactimate.gui.log_config import get_logger, append_omnipose_debug
from bactimate.gui.paths import get_omnipose_user_models_dir
from bactimate.gui.patches import _patch_omnipose_get_masks
from bactimate.gui.preferences import LAST_DIR_SEGMENTATION
from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.mask_utils import (
    OMNIPOSE_MASKS_SUBDIR,
    OMNIPOSE_MASKS_TILED_SUBDIR,
    ensure_omnipose_masks_has_images,
    ensure_training_images_nchan,
    get_training_pairs,
    infer_training_filters,
    prepare_omnipose_masks_from_pairs,
    tile_training_data,
)
from bactimate.gui.omnipose_utils import ensure_omni_in_model_path, inspect_omnipose_checkpoint
from bactimate.gui.utils import safe_imread
from bactimate.scripts.check_omnipose_pipeline import (
    ensure_builtin_model_cached,
    get_checkable_model_path,
    _check_model,
    _check_training_data,
    _check_inference,
    format_model_report,
    format_training_report,
    format_inference_report,
    run_checkpoint_comparison,
)
from bactimate.gui.widgets import (
    ClickableTooltip,
    TooltipClickFilter,
    DragDropGroupBox,
    create_info_icon,
    show_file_selection_dialog,
)
from bactimate.gui.workers import WorkerThread
from bactimate.tiff_metadata import read_calibration_from_path_or_original

# Built-in Omnipose model types (no Cellpose-only). User-trained models are listed separately.
BUILTIN_OMNIPOSE_MODELS = [
    "bact_phase_omni",
    "bact_fluor_omni",
    "bact_phase_affinity",
]
# Extension for user-trained model files saved in get_omnipose_user_models_dir()
USER_MODEL_EXTENSION = ".pth"

# Transfer learning: only offer models that exist in the main tab (BUILTIN_OMNIPOSE_MODELS).
# (nchan, nclasses, dim) per Omnipose models table; used to lock nchan/nclasses when selected.
# nclasses=3 is the Omnipose CLI parameter (flow + distance + boundary). The loader uses only
# top-level nchan/nclasses from the checkpoint (default nclasses=3); it does not infer from state_dict.
PRETRAINED_NCHAN_NCLASSES_DIM = {
    "bact_phase_omni": (2, 3, 2),
    "bact_fluor_omni": (2, 3, 2),
    "bact_phase_affinity": (1, 2, 2),
}

# When user selects "New model (same architecture)", we initialize from this so the result is
# identical to the built-in model (same architecture, same init) but trained on their data.
CANONICAL_BACTERIAL_MODEL = "bact_phase_omni"

# Default (diameter, flow_threshold, mask_threshold) per built-in model; applied when user selects that model.
SEGMENTATION_MODEL_DEFAULTS = {
    "bact_phase_omni": (30.0, 0.4, 0.0),
    "bact_fluor_omni": (30.0, 0.4, 0.0),
    "bact_phase_affinity": (30.0, 0.4, 0.0),
}
# Fallback for other built-in names (e.g. bact_omni) or user models
SEGMENTATION_DEFAULT_FALLBACK = (30.0, 0.4, 0.0)
USER_MODEL_DEFAULTS = (30.0, 0.0, 0.0)  # diameter, flow, mask for user-trained models


class _OmniposeTrainWorker(QThread):
    """Runs Omnipose training subprocess in background so the GUI stays responsive."""
    log_line = pyqtSignal(str)
    finished_with = pyqtSignal(bool, str, str)  # success, model_name, message

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
            ckpt = torch.load(self.dest_file, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(state, dict):
                return
            first_key = next(iter(state.keys()), "")
            if first_key.startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            torch.save(state, self.dest_file)
        except Exception:
            pass

    def run(self):
        import shutil
        try:
            proc = subprocess.Popen(
                self.cmd,
                env=self.env,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                self.log_line.emit(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                self.finished_with.emit(False, self.model_name, f"Omnipose exited with code {proc.returncode}.")
                return
            pth_files = list(Path(self.save_dir).glob("*.pth"))
            alt = Path(self.cwd) / "models"
            if not pth_files and alt.is_dir():
                pth_files = list(alt.glob("*.pth"))
            if not pth_files and alt.is_dir():
                all_files = [p for p in alt.iterdir() if p.is_file()]
                if all_files:
                    newest = max(all_files, key=lambda p: p.stat().st_mtime)
                    pth_files = [newest]
            if not pth_files:
                self.finished_with.emit(False, self.model_name, "No .pth checkpoint found after training.")
                return
            newest = max(pth_files, key=lambda p: p.stat().st_mtime)
            shutil.copy2(newest, self.dest_file)
            self._write_checkpoint_metadata()
            if not self.keep_save_dir:
                shutil.rmtree(self.save_dir, ignore_errors=True)
            msg = f"Saved to {self.dest_file}"
            if self.keep_save_dir:
                msg += f". Intermediate checkpoints are in: {self.save_dir}"
            self.finished_with.emit(True, self.model_name, msg)
        except Exception as e:
            self.finished_with.emit(False, self.model_name, str(e))


def _get_pretrained_training_options():
    """From scratch (None), or a built-in model name to initialize from. Same architecture: nchan=2, nclasses=3."""
    return [
        ("From scratch", "None"),  # --pretrained_model None
        (CANONICAL_BACTERIAL_MODEL, None),  # init from built-in bact_phase_omni (same as other built-in names)
    ] + [(name, name) for name in BUILTIN_OMNIPOSE_MODELS if name in PRETRAINED_NCHAN_NCLASSES_DIM and name != CANONICAL_BACTERIAL_MODEL]


class OmniposeSegTab(QWidget):
    """Unified tab for Omnipose: train models and run single file or folder segmentation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._trained_models_this_session: list[str] = []  # for session log when user trains a model
        self.init_ui()

    def init_ui(self) -> None:
        self._omnipose_train_worker = None
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        train_group = self._build_omnipose_train_group()
        scroll_layout.addWidget(train_group)

        seg_group = self._build_segmentation_params()
        scroll_layout.addWidget(seg_group)

        # Connect file path changes (file_group is inside seg_group)
        self.file_path.textChanged.connect(self._update_overlay_checkbox_state)
        self._update_overlay_checkbox_state()
        self.file_path.textChanged.connect(self._update_button_states)
        self.file_path.textChanged.connect(self._try_push_metadata_from_path)
        self._update_button_states()

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_omnipose_train_group(self) -> QGroupBox:
        """Build 'Train Omnipose model' section (all controls inline, no dialog)."""
        from bactimate.gui.paths import get_omnipose_user_models_dir
        train_group = QGroupBox("Train Omnipose model")
        layout = QVBoxLayout()
        last_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_SEGMENTATION) or str(Path.home())

        def handle_dropped_training_path(dropped_path):
            path_obj = Path(dropped_path)
            if path_obj.is_dir():
                self.omnipose_train_dir_edit.setText(dropped_path)
            elif path_obj.is_file():
                self.omnipose_train_dir_edit.setText(str(path_obj.parent))

        dir_group = DragDropGroupBox(
            "Training directory (images + masks)",
            self,
            on_drop_callback=handle_dropped_training_path,
        )
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Path:"))
        dir_layout.addWidget(create_info_icon(
            "Folder containing your training data: one image file and one mask file per pair.\n\n"
            "Omnipose expects: for each image 'NAME.tif', a mask file 'NAME<Mask filter>.tif' (e.g. NAME_masks.tif). Same base name, mask = base + Mask filter + extension.\n\n"
            "Mask format: must be instance labels (background = 0; each cell = a unique integer). When you click Start training, an 'omnipose_masks' folder is created automatically and used for training.\n\n"
            "Tip: For large or sparse images, tile to ~512×512 and drop mostly-empty tiles before training; set Crop size (tyx) to ~half tile size.",
            self,
        ))
        self.omnipose_train_dir_edit = QLineEdit()
        self.omnipose_train_dir_edit.setPlaceholderText("Absolute path to folder with image/mask pairs (or drag and drop)")
        self.omnipose_train_dir_edit.setAcceptDrops(False)
        dir_layout.addWidget(self.omnipose_train_dir_edit)
        def browse_train_dir():
            path = QFileDialog.getExistingDirectory(self, "Select training directory", last_dir)
            if path:
                self.omnipose_train_dir_edit.setText(path)
                if getattr(self.parent_window, "set_last_dir_for_tab", None):
                    self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, path)
        browse_btn = QPushButton("Browse…")
        browse_btn.setAcceptDrops(False)
        browse_btn.setToolTip("")
        browse_btn.clicked.connect(browse_train_dir)
        dir_layout.addWidget(browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Image filter:"))
        self.omnipose_img_filter_edit = QLineEdit()
        self.omnipose_img_filter_edit.setPlaceholderText("empty or _img")
        self.omnipose_img_filter_edit.setMinimumWidth(100)
        filter_layout.addWidget(self.omnipose_img_filter_edit)
        filter_layout.addWidget(QLabel("Mask filter:"))
        self.omnipose_mask_filter_edit = QLineEdit()
        self.omnipose_mask_filter_edit.setText("_mask")
        self.omnipose_mask_filter_edit.setMinimumWidth(100)
        filter_layout.addWidget(self.omnipose_mask_filter_edit)
        auto_detect_btn = QPushButton("Auto-detect filters")
        auto_detect_btn.setToolTip("")
        auto_detect_btn.clicked.connect(self._on_omnipose_auto_detect_filters)
        filter_layout.addWidget(auto_detect_btn)
        filter_layout.addWidget(create_info_icon(
            "Suffix that identifies image vs mask files.\n\n"
            "Image filter: leave empty if filenames are e.g. A.tif; use _img if you have A_img.tif.\n"
            "Mask filter: suffix on mask files. With _masks, mask for 'cell1.tif' must be 'cell1_masks.tif'. Must match your filenames exactly.",
            self,
        ))
        layout.addLayout(filter_layout)

        prep_row = QHBoxLayout()
        make_masks_btn = QPushButton("Make masks instance labeled")
        make_masks_btn.setToolTip("")
        make_masks_btn.clicked.connect(self._on_omnipose_make_masks_instance_labeled)
        prep_row.addWidget(make_masks_btn)
        prep_row.addWidget(create_info_icon(
            "Omnipose needs instance-label masks (background = 0; each cell = a unique integer 1, 2, 3, …). "
            "When you click Start training, an 'omnipose_masks' folder is created automatically and used for training; no separate conversion step is needed.",
            self,
        ))
        tile_btn = QPushButton("Tile and prepare for training")
        tile_btn.setToolTip("")
        tile_btn.clicked.connect(self._on_omnipose_tile_and_prepare)
        prep_row.addWidget(tile_btn)
        prep_row.addWidget(create_info_icon(
            "Split images into smaller tiles and drop mostly-empty tiles. Use when images are large or sparse "
            "to avoid the model collapsing to 0 cells. Output goes to a separate folder; set training directory to it and Crop size (tyx) to half the tile size. "
            "Each run clears the output folder first.",
            self,
        ))
        layout.addLayout(prep_row)

        self.omnipose_use_tiled_cb = QCheckBox("Use tiled/cropped images for training")
        self.omnipose_use_tiled_cb.setChecked(True)
        self.omnipose_use_tiled_cb.setToolTip("")
        tile_crop_row = QHBoxLayout()
        tile_crop_row.addWidget(self.omnipose_use_tiled_cb)
        tile_crop_row.addWidget(create_info_icon(
            "Uncheck to train on full-size images instead of tiled crops. Use when your images are already small enough or you prefer no tiling.",
            self,
        ))
        tile_crop_row.addSpacing(16)
        self.omnipose_use_gpu_cb = QCheckBox("Use GPU")
        self.omnipose_use_gpu_cb.setChecked(True)
        self.omnipose_use_gpu_cb.setToolTip("")
        tile_crop_row.addWidget(self.omnipose_use_gpu_cb)
        tile_crop_row.addStretch()
        layout.addLayout(tile_crop_row)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Model name:"))
        user_models_path = get_omnipose_user_models_dir()
        name_layout.addWidget(create_info_icon(
            "A short name for your model (letters, numbers, underscores). It will appear in the model dropdown and is used to save the .pth file. "
            "Example: my_bacteria_v1. BactiMate automatically adds '_omni' to the filename if your name does not already contain 'omni'. "
            f"User models folder:\n{user_models_path}",
            self,
        ))
        self.omnipose_model_name_edit = QLineEdit()
        self.omnipose_model_name_edit.setPlaceholderText("e.g. my_bacteria_v1")
        self.omnipose_model_name_edit.setMinimumWidth(220)
        name_layout.addWidget(self.omnipose_model_name_edit, 1)
        layout.addLayout(name_layout)

        start_from_layout = QHBoxLayout()
        start_from_layout.addWidget(QLabel("Start from:"))
        self.omnipose_pretrained_combo = QComboBox()
        self.omnipose_pretrained_combo.setMinimumWidth(260)
        for display_name, _ in _get_pretrained_training_options():
            self.omnipose_pretrained_combo.addItem(display_name)
        if self.omnipose_pretrained_combo.count() >= 2:
            self.omnipose_pretrained_combo.setCurrentIndex(1)
        start_from_layout.addWidget(self.omnipose_pretrained_combo, 1)
        start_from_layout.addWidget(create_info_icon(
            "Choose which checkpoint to start training from. From scratch: no pretrained model. "
            "bact_phase_omni / bact_fluor_omni / bact_phase_affinity: new model file initialized from that built-in; training runs on your data.",
            self,
        ))
        check_compat_btn = QPushButton("Check compatibility")
        check_compat_btn.setFixedWidth(260)
        check_compat_btn.setToolTip("")
        check_compat_btn.clicked.connect(self._on_omnipose_check_compatibility)
        start_from_layout.addWidget(check_compat_btn)
        start_from_layout.addWidget(create_info_icon(
            "Validates image/mask pairs in your training folder and that image dimensions match the selected model (e.g. 2D for bact_phase_omni).",
            self,
        ))
        layout.addLayout(start_from_layout)

        layout.addWidget(QLabel("Training parameters:"))
        params_row1 = QHBoxLayout()
        params_row1.addWidget(QLabel("Epochs:"))
        params_row1.addWidget(create_info_icon(
            "Number of full passes over the training set. From scratch: Omnipose paper uses 4000. Fine-tuning: start with 50 epochs (default).",
            self,
        ))
        self.omnipose_n_epochs_spin = QSpinBox()
        self.omnipose_n_epochs_spin.setRange(1, 50000)
        self.omnipose_n_epochs_spin.setSingleStep(100)
        self.omnipose_n_epochs_spin.setValue(4000)
        self.omnipose_n_epochs_spin.setMinimumWidth(90)
        params_row1.addWidget(self.omnipose_n_epochs_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel("Save N epochs:"))
        params_row1.addWidget(create_info_icon(
            "Save a checkpoint every N epochs (0 = only final model). Use e.g. 50 to compare epochs and pick the best.",
            self,
        ))
        self.omnipose_save_each_spin = QSpinBox()
        self.omnipose_save_each_spin.setRange(0, 1000)
        self.omnipose_save_each_spin.setValue(0)
        self.omnipose_save_each_spin.setSpecialValueText("final only")
        self.omnipose_save_each_spin.setMinimumWidth(90)
        params_row1.addWidget(self.omnipose_save_each_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel("LR:"))
        params_row1.addWidget(create_info_icon(
            "Learning rate: Step size for the optimizer. From scratch: 0.1. Fine-tuning: use 0.01 or 0.005 so the pretrained model does not drift.",
            self,
        ))
        self.omnipose_lr_spin = QDoubleSpinBox()
        self.omnipose_lr_spin.setRange(0.001, 1.0)
        self.omnipose_lr_spin.setValue(0.1)
        self.omnipose_lr_spin.setDecimals(3)
        self.omnipose_lr_spin.setMinimumWidth(90)
        params_row1.addWidget(self.omnipose_lr_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel("Batch:"))
        params_row1.addWidget(create_info_icon(
            "Number of image crops per training step. Larger (e.g. 16) trains faster if you have enough GPU memory; smaller (e.g. 4–8) uses less memory.",
            self,
        ))
        self.omnipose_batch_spin = QSpinBox()
        self.omnipose_batch_spin.setRange(1, 128)
        self.omnipose_batch_spin.setValue(16)
        self.omnipose_batch_spin.setMinimumWidth(90)
        params_row1.addWidget(self.omnipose_batch_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(QLabel("Crop size:"))
        params_row1.addWidget(create_info_icon(
            "Training crop size (Y, X) for random crops each epoch. Default 256×256. Must be divisible by 8. Use ~half your image/tile size.",
            self,
        ))
        params_row1.addWidget(QLabel("Y:"))
        self.omnipose_tyx_y_spin = QSpinBox()
        self.omnipose_tyx_y_spin.setRange(64, 512)
        self.omnipose_tyx_y_spin.setSingleStep(8)
        self.omnipose_tyx_y_spin.setValue(256)
        self.omnipose_tyx_y_spin.setMinimumWidth(72)
        params_row1.addWidget(self.omnipose_tyx_y_spin)
        params_row1.addWidget(QLabel("X:"))
        self.omnipose_tyx_x_spin = QSpinBox()
        self.omnipose_tyx_x_spin.setRange(64, 512)
        self.omnipose_tyx_x_spin.setSingleStep(8)
        self.omnipose_tyx_x_spin.setValue(256)
        self.omnipose_tyx_x_spin.setMinimumWidth(72)
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

        _locked_style = "QSpinBox:disabled { background-color: palette(mid); color: palette(placeholder-text); }"
        _normal_style = ""

        def update_pretrained_lock():
            idx = self.omnipose_pretrained_combo.currentIndex()
            options = _get_pretrained_training_options()
            cli_value = options[idx][1] if 0 <= idx < len(options) else None
            model_for_arch = CANONICAL_BACTERIAL_MODEL if (cli_value is None or cli_value == "None") else cli_value
            if model_for_arch in PRETRAINED_NCHAN_NCLASSES_DIM:
                nchan, nclass, _ = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_arch]
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

        layout.addWidget(QLabel("Training log:"))
        self.omnipose_training_log = QTextEdit()
        self.omnipose_training_log.setReadOnly(True)
        self.omnipose_training_log.setMaximumHeight(180)
        self.omnipose_training_log.setFont(QFont("Menlo", 10) if sys.platform == "darwin" else QFont("Consolas", 9))
        layout.addWidget(self.omnipose_training_log)
        self.omnipose_start_training_btn = QPushButton("Start training")
        self.omnipose_start_training_btn.clicked.connect(self._start_omnipose_training)
        layout.addWidget(self.omnipose_start_training_btn)

        train_group.setLayout(layout)
        return train_group

    def _training_pairs_for_train(self, train_path: Path, img_filter: str, mask_filter: str):
        """Return (n_pairs, first_paired_image_path, image_candidates) for compatibility check and start_training."""
        exts = (".tif", ".tiff", ".png", ".npy", ".npz")
        mask_suffixes = ("_cp_masks", "_cp_output", "_flows", mask_filter)

        def base_name(p, suf):
            return p.stem.replace(suf, "") if suf else p.stem

        candidates = [
            f for f in train_path.iterdir()
            if f.is_file() and f.suffix.lower() in exts
            and not any(f.stem.endswith(s) or (f.stem + s == f.stem) for s in mask_suffixes)
        ]
        n, first = 0, None
        for f in candidates:
            base_val = base_name(f, img_filter)
            if not base_val or base_val == ".":
                continue
            if any((train_path / f"{base_val}{mask_filter}{e}").exists() for e in exts):
                n += 1
                if first is None:
                    first = f
        return n, first, candidates

    def _format_omnipose_training_log_line(self, raw: str) -> str:
        """Strip ANSI codes and reformat Omnipose 'Train epoch' lines for readability."""
        stripped = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw)
        stripped = re.sub(r"\033\[[0-9;]*[a-zA-Z]", "", stripped)
        stripped = re.sub(r"\[0m|\[38;5;\d+m", "", stripped)
        stripped = stripped.strip()
        if not stripped:
            return ""
        if "\t" in stripped:
            parts = stripped.split("\t")
            last = parts[-1].strip() if parts else ""
            if last and ("/cellpose_" in last or "cellpose_residual" in last) and ("epoch_" in last or re.search(r"_\d{4}_\d{2}_\d{2}", last)):
                return ""
        if "Train epoch:" in stripped and "\t" in stripped:
            parts = stripped.split("\t")
            msg = stripped
            for i in range(len(parts) - 1, -1, -1):
                if "Train epoch:" in parts[i]:
                    msg = parts[i].strip()
                    break
        else:
            msg = stripped
        if "Train epoch:" in msg and " | " in msg:
            segments = [s.strip() for s in msg.split(" | ")]
            segments = [s for s in segments if "sec/batch" not in s.lower() and "last batch loss" not in s.lower()]
            if not segments:
                return ""
            def _shorten(s):
                m = re.search(r"(Loss>:?\s*)([\d.]+)", s)
                if m:
                    try:
                        val = float(m.group(2))
                        return s[: m.start(2)] + f"{val:.2f}"
                    except ValueError:
                        pass
                return s
            segments = [_shorten(s) for s in segments]
            return "  " + " | ".join(segments)
        return msg

    def _on_omnipose_auto_detect_filters(self):
        folder_str = self.omnipose_train_dir_edit.text().strip()
        if not folder_str:
            QMessageBox.warning(self, "No folder", "Enter a training directory first.")
            return
        folder = Path(folder_str)
        if not folder.is_dir():
            QMessageBox.warning(self, "Invalid folder", "Training directory does not exist or is not a folder.")
            return
        img_filter, mask_filter = infer_training_filters(folder)
        if img_filter is None and mask_filter is None:
            QMessageBox.information(
                self,
                "Auto-detect filters",
                "Could not detect image/mask pairs. Ensure the folder contains paired files (e.g. 634_phase.tif and 634_masks.tif) and try again, or set the filters manually.",
            )
            return
        self.omnipose_img_filter_edit.setText(img_filter or "")
        self.omnipose_mask_filter_edit.setText(mask_filter or "_mask")
        pairs = get_training_pairs(folder, img_filter or "", mask_filter or "_mask")
        QMessageBox.information(
            self, "Auto-detect filters", f"Detected filters. Found {len(pairs)} image/mask pair(s)."
        )

    def _on_omnipose_make_masks_instance_labeled(self):
        """Create omnipose_masks folder with instance-labeled masks from current training dir and filters."""
        train_dir = self.omnipose_train_dir_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            QMessageBox.warning(self, "No folder", "Set a valid training directory first.")
            return
        img_filter = self.omnipose_img_filter_edit.text().strip()
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or "_mask"
        parent_dir = Path(train_dir).resolve()
        try:
            train_path, n_from_pairs = prepare_omnipose_masks_from_pairs(parent_dir, img_filter, mask_filter)
        except Exception as e:
            QMessageBox.warning(self, "Convert masks", f"Failed to prepare instance-labeled masks:\n{e}")
            return
        if n_from_pairs == 0:
            QMessageBox.warning(
                self,
                "No pairs",
                "No image/mask pairs found. Set Image and Mask filters so each image has a matching mask, or run Auto-detect filters.",
            )
            return
        QMessageBox.information(
            self,
            "Masks prepared",
            f"Created instance-labeled masks for {n_from_pairs} pair(s) in:\n{train_path}\n\n"
            "Start training will use this folder automatically, or you can set Training directory to the path above.",
        )

    def _on_omnipose_tile_and_prepare(self):
        train_dir = self.omnipose_train_dir_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            QMessageBox.warning(self, "No folder", "Set a valid training directory first.")
            return
        img_filter = self.omnipose_img_filter_edit.text().strip()
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or "_mask"
        pairs = get_training_pairs(Path(train_dir), img_filter, mask_filter)
        if not pairs:
            QMessageBox.warning(
                self,
                "No pairs",
                "No image/mask pairs found. Set Image and Mask filters so each image has a matching mask.",
            )
            return
        tile_dlg = QDialog(self)
        tile_dlg.setWindowTitle("Tile and prepare for training")
        tile_layout = QVBoxLayout(tile_dlg)
        tile_layout.addWidget(QLabel("Tile size (Y and X). Must be divisible by 8. Use 512 or 256."))
        tile_size_spin = QSpinBox()
        tile_size_spin.setRange(64, 512)
        tile_size_spin.setSingleStep(8)
        tile_size_spin.setValue(512)
        tile_layout.addWidget(tile_size_spin)
        tile_layout.addWidget(QLabel("Minimum cells per tile (tiles with fewer cells are skipped):"))
        min_cells_spin = QSpinBox()
        min_cells_spin.setRange(0, 50)
        min_cells_spin.setValue(1)
        tile_layout.addWidget(min_cells_spin)
        tile_layout.addWidget(QLabel("Output folder (default: training_dir/omnipose_masks_tiled):"))
        out_edit = QLineEdit()
        out_edit.setText(str(Path(train_dir).resolve() / OMNIPOSE_MASKS_TILED_SUBDIR))
        out_edit.setMinimumWidth(400)
        tile_layout.addWidget(out_edit)
        bb_tile = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb_tile.accepted.connect(tile_dlg.accept)
        bb_tile.rejected.connect(tile_dlg.reject)
        tile_layout.addWidget(bb_tile)
        if tile_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        tile_size = max(8, (tile_size_spin.value() // 8) * 8)
        min_cells = min_cells_spin.value()
        out_path = Path(out_edit.text().strip())
        if not out_path:
            out_path = Path(train_dir).resolve() / OMNIPOSE_MASKS_TILED_SUBDIR
        try:
            n_tiles, n_imgs, errs = tile_training_data(
                Path(train_dir).resolve(),
                img_filter,
                mask_filter,
                tile_size,
                min_cells,
                out_path,
            )
        except Exception as e:
            QMessageBox.warning(self, "Tiling failed", str(e))
            return
        if errs:
            msg = f"Written {n_tiles} tiles from {n_imgs} image(s) to:\n{out_path}\n\nWarnings:\n" + "\n".join(errs[:8])
            if len(errs) > 8:
                msg += f"\n... and {len(errs) - 8} more."
        else:
            msg = f"Written {n_tiles} tiles from {n_imgs} image(s) to:\n{out_path}"
        if n_tiles > 0:
            tyx_half = max(8, (tile_size // 2 // 8) * 8)
            self.omnipose_tyx_y_spin.setValue(tyx_half)
            self.omnipose_tyx_x_spin.setValue(tyx_half)
            msg += f"\n\nCrop size (tyx) has been set to {tyx_half}×{tyx_half} (half the tile size)."
        msg += "\n\nNext: set Training directory to the folder above if not already, and keep the same Image/Mask filters."
        QMessageBox.information(self, "Tiling done", msg)
        if n_tiles > 0 and QMessageBox.question(
            self,
            "Set training directory?",
            "Set Training directory to the tiled folder now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        ) == QMessageBox.StandardButton.Yes:
            self.omnipose_train_dir_edit.setText(str(out_path))

    def _on_omnipose_check_compatibility(self):
        train_dir = self.omnipose_train_dir_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            QMessageBox.warning(self, "Invalid directory", "Please choose an absolute path to a training directory.")
            return
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or "_mask"
        img_filter = self.omnipose_img_filter_edit.text().strip()
        train_path = ensure_omnipose_masks_has_images(Path(train_dir).resolve(), mask_filter, img_filter=img_filter)
        n_pairs, first_img, _ = self._training_pairs_for_train(train_path, img_filter, mask_filter)
        if n_pairs == 0:
            QMessageBox.warning(
                self,
                "Compatibility",
                "No image/mask pairs found. Set Image and Mask filters so each image has a matching mask file.",
            )
            return
        idx = self.omnipose_pretrained_combo.currentIndex()
        options = _get_pretrained_training_options()
        cli_value = options[idx][1] if 0 <= idx < len(options) else None
        model_for_arch = cli_value or CANONICAL_BACTERIAL_MODEL
        detail_parts = []
        if model_for_arch in PRETRAINED_NCHAN_NCLASSES_DIM:
            _, _, dim = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_arch]
            if dim == 2 and first_img:
                try:
                    arr = safe_imread(str(first_img))
                except Exception:
                    arr = None
                if arr is not None and arr.ndim == 3 and arr.shape[0] > 4:
                    QMessageBox.warning(
                        self,
                        "Compatibility",
                        f"Found {n_pairs} pair(s), but the first image has shape {tuple(arr.shape)} (likely a 3D stack). "
                        "The selected model is 2D. Use 2D slices or a 3D model (e.g. plant_omni).",
                    )
                    return
                if arr is not None:
                    detail_parts.append(
                        f"Checked first image dimensions: shape {tuple(arr.shape)} — 2D, compatible with the selected 2D model ({model_for_arch})."
                    )
                else:
                    detail_parts.append("Could not load first image to verify dimensions; pair count and filters are valid.")
            elif dim == 2:
                detail_parts.append("Pair count and filters valid. No image could be sampled for dimension check.")
            else:
                detail_parts.append(f"Pair count and filters valid. Selected model is 3D ({model_for_arch}); dimension check skipped.")
        else:
            detail_parts.append(
                "Filters and pair naming are valid. Select a pretrained model in 'Start from' to also verify image dimensions (2D vs 3D)."
            )
        msg = f"Found {n_pairs} image/mask pair(s).\n\n" + "\n\n".join(detail_parts)
        QMessageBox.information(self, "Compatibility", msg)

    def _start_omnipose_training(self):
        """Run Omnipose training from tab widgets (no dialog)."""
        train_dir = self.omnipose_train_dir_edit.text().strip()
        model_name = self.omnipose_model_name_edit.text().strip()
        if not train_dir or not Path(train_dir).is_dir():
            QMessageBox.warning(self, "Invalid directory", "Please choose an absolute path to a training directory.")
            return
        if not model_name:
            QMessageBox.warning(self, "Invalid name", "Please enter a model name.")
            return
        if "/" in model_name or "\\" in model_name:
            QMessageBox.warning(self, "Invalid name", "Model name must not contain path separators.")
            return
        final_stem = model_name if "omni" in model_name.lower() else f"{model_name}_omni"
        dest_file = get_omnipose_user_models_dir() / f"{final_stem}{USER_MODEL_EXTENSION}"
        if dest_file.exists():
            if QMessageBox.question(
                self, "Model already exists",
                f"A model named '{final_stem}' already exists. Overwrite it? This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
        train_dir_abs = Path(train_dir).resolve()
        img_filter = self.omnipose_img_filter_edit.text().strip()
        mask_filter = self.omnipose_mask_filter_edit.text().strip() or "_mask"
        use_tiled = self.omnipose_use_tiled_cb.isChecked()
        if train_dir_abs.name == OMNIPOSE_MASKS_TILED_SUBDIR and use_tiled:
            train_path = train_dir_abs
        else:
            parent_dir = train_dir_abs.parent if (train_dir_abs.name == OMNIPOSE_MASKS_SUBDIR or train_dir_abs.name == OMNIPOSE_MASKS_TILED_SUBDIR) else train_dir_abs
            train_path, n_from_pairs = prepare_omnipose_masks_from_pairs(parent_dir, img_filter, mask_filter)
            if n_from_pairs == 0:
                train_path = ensure_omnipose_masks_has_images(parent_dir, mask_filter, img_filter=img_filter)
        train_dir_abs = str(train_path)
        pairs_ok, first_img_path, image_candidates = self._training_pairs_for_train(train_path, img_filter, mask_filter)

        def base_name(p, suf):
            return p.stem.replace(suf, "") if suf else p.stem

        bad_base = [f.name for f in image_candidates if not base_name(f, img_filter) or base_name(f, img_filter) == "."]
        if pairs_ok == 0 and not img_filter and image_candidates:
            mask_files = [f for f in train_path.iterdir() if f.is_file() and f.stem.endswith(mask_filter)]
            for img_f in image_candidates:
                stem = img_f.stem
                for mf in mask_files:
                    base = mf.stem[: -len(mask_filter)] if mask_filter else mf.stem
                    if base and stem.startswith(base) and stem != base:
                        inferred = stem[len(base):]
                        if self._training_pairs_for_train(train_path, inferred, mask_filter)[0] > 0:
                            img_filter = inferred
                            self.omnipose_img_filter_edit.setText(inferred)
                            pairs_ok, first_img_path, image_candidates = self._training_pairs_for_train(train_path, img_filter, mask_filter)
                            bad_base = [f.name for f in image_candidates if not base_name(f, img_filter) or base_name(f, img_filter) == "."]
                            break
                if pairs_ok > 0:
                    break

        if bad_base:
            QMessageBox.warning(
                self, "Invalid training files",
                f"Some image filenames give an empty base name: {bad_base[:5]}{'…' if len(bad_base) > 5 else ''}. "
                "Use names like 'image.tif' with a matching 'image_masks.tif' (no leading dot).",
            )
            return
        if pairs_ok == 0:
            folder_used = getattr(train_path, "name", str(train_path))
            if folder_used == OMNIPOSE_MASKS_SUBDIR or str(train_path).endswith(os.path.sep + OMNIPOSE_MASKS_SUBDIR):
                hint = (
                    f"The folder '{train_path}' contains only mask files (no images). "
                    f"Images are copied from the parent folder when they match the mask names. "
                    "Check that the parent folder has image files with matching base names and that Image filter matches your image filenames."
                )
            else:
                hint = (
                    f"For each image we need a mask: base name + '{mask_filter}' + extension. "
                    f"Image filter is '{img_filter or '(empty)'}' — the base is taken from the image by removing this suffix."
                )
            QMessageBox.warning(self, "No image/mask pairs found", f"No matching mask files found. {hint}")
            return

        pretrained_idx = self.omnipose_pretrained_combo.currentIndex()
        options = _get_pretrained_training_options()
        pretrained_cli = options[pretrained_idx][1] if 0 <= pretrained_idx < len(options) else None
        model_for_dim = CANONICAL_BACTERIAL_MODEL if (pretrained_cli is None or pretrained_cli == "None") else pretrained_cli
        if model_for_dim in PRETRAINED_NCHAN_NCLASSES_DIM:
            _, _, dim = PRETRAINED_NCHAN_NCLASSES_DIM[model_for_dim]
            if dim == 2 and first_img_path:
                try:
                    arr = safe_imread(str(first_img_path))
                except Exception:
                    arr = None
                if arr is not None and arr.ndim == 3 and arr.shape[0] > 4:
                    if QMessageBox.question(
                        self, "Compatibility",
                        f"First image has shape {tuple(arr.shape)} (likely a 3D stack). The selected model is 2D. Continue anyway?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    ) != QMessageBox.StandardButton.Yes:
                        return
        n_epochs = self.omnipose_n_epochs_spin.value()
        if pretrained_cli == "None" and n_epochs < 1000:
            if QMessageBox.question(
                self,
                "From-scratch training",
                f"You chose \"From scratch\" with only {n_epochs} epochs. Omnipose from-scratch typically needs around 4000 epochs to detect cells. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
        if pretrained_cli and pretrained_cli != "None" and n_epochs > 500:
            if QMessageBox.question(
                self,
                "Fine-tuning with many epochs",
                f"You are fine-tuning with {n_epochs} epochs. Many epochs often drift the output head and cause 0 cells. Recommended: start with 50 epochs. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
        nchan_val = self.omnipose_nchan_spin.value()
        n_chan_updated, chan_errors = ensure_training_images_nchan(
            Path(train_dir_abs), img_filter, mask_filter, nchan_val
        )
        self.omnipose_start_training_btn.setEnabled(False)
        self.omnipose_training_log.clear()
        self.omnipose_training_log.append(f"Found {pairs_ok} image/mask pair(s). All will be used every epoch for {n_epochs} epochs.")
        if first_img_path:
            try:
                arr = safe_imread(str(first_img_path))
                if arr is not None:
                    if arr.ndim == 3 and arr.shape[0] <= 8:
                        self.omnipose_training_log.append(f"Dataset dimensions (first image): Y={arr.shape[1]}, X={arr.shape[2]} (nchan={arr.shape[0]}).")
                    elif arr.ndim == 2:
                        self.omnipose_training_log.append(f"Dataset dimensions (first image): Y={arr.shape[0]}, X={arr.shape[1]}.")
            except Exception:
                pass
        tyx_y = max(8, (self.omnipose_tyx_y_spin.value() // 8) * 8)
        tyx_x = max(8, (self.omnipose_tyx_x_spin.value() // 8) * 8)
        self.omnipose_training_log.append(f"Crop size (tyx): {tyx_y}×{tyx_x}.")
        if use_tiled:
            self.omnipose_training_log.append(
                "Using tiled images. Crop size is typically half the tile size so each epoch uses random crops (data augmentation)."
            )
        if chan_errors:
            for err in chan_errors[:5]:
                self.omnipose_training_log.append(f"Warning: {err}")
            if len(chan_errors) > 5:
                self.omnipose_training_log.append(f"... and {len(chan_errors) - 5} more.")
        if n_chan_updated > 0:
            self.omnipose_training_log.append(
                f"Ensured {n_chan_updated} image(s) have {nchan_val} channels in CYX order for training."
            )
        pretrained_arg = pretrained_cli if pretrained_cli is not None else CANONICAL_BACTERIAL_MODEL
        if pretrained_cli == "None":
            self.omnipose_training_log.append("From scratch: --pretrained_model None, nchan=2, nclasses=3.")
        elif pretrained_cli:
            self.omnipose_training_log.append(f"Transfer learning: starting from {pretrained_cli} (nchan/nclasses match that model).")
        else:
            self.omnipose_training_log.append(f"Initializing from {CANONICAL_BACTERIAL_MODEL} (same architecture), training on your data.")
        self.omnipose_training_log.append("Starting Omnipose training (this may take a while)…")

        cmd = [
            sys.executable, "-m", "omnipose", "--train",
            "--dir", train_dir_abs,
            "--mask_filter", mask_filter,
            "--pretrained_model", str(pretrained_arg),
            "--diameter", "0",
            "--n_epochs", str(self.omnipose_n_epochs_spin.value()),
            "--learning_rate", str(self.omnipose_lr_spin.value()),
            "--batch_size", str(self.omnipose_batch_spin.value()),
            "--nchan", str(self.omnipose_nchan_spin.value()),
            "--nclasses", str(self.omnipose_nclasses_spin.value()),
            "--residual_on", "1",
            "--style_on", "1",
            "--concatenation", "0",
            "--RAdam",
        ]
        if img_filter:
            cmd.extend(["--img_filter", img_filter])
        if self.omnipose_use_gpu_cb.isChecked():
            cmd.append("--use_gpu")
        if tyx_y != 224 or tyx_x != 224:
            cmd.extend(["--tyx", f"{tyx_y},{tyx_x}"])
        save_every_val = self.omnipose_save_each_spin.value()
        if save_every_val > 0:
            cmd.extend(["--save_every", str(save_every_val), "--save_each"])
        if self.omnipose_nchan_spin.value() >= 2:
            cmd.extend(["--channel_axis", "0", "--all_channels"])
        import tempfile
        if save_every_val > 0:
            save_dir_path = Path(train_dir_abs) / "models"
            save_dir_path.mkdir(parents=True, exist_ok=True)
            save_dir = str(save_dir_path)
            keep_save_dir = True
        else:
            save_dir = tempfile.mkdtemp(prefix="bactimate_train_")
            keep_save_dir = False
        env = os.environ.copy()
        env["CELLPOSE_LOCAL_MODELS_PATH"] = save_dir

        worker = _OmniposeTrainWorker(
            cmd, env, train_dir_abs, save_dir, str(dest_file), final_stem,
            self.omnipose_nchan_spin.value(), self.omnipose_nclasses_spin.value(), keep_save_dir=keep_save_dir,
        )
        self._omnipose_train_worker = worker

        def on_log(line):
            formatted = self._format_omnipose_training_log_line(line)
            if formatted:
                self.omnipose_training_log.append(formatted)

        def on_finished(success, name, msg):
            self.omnipose_start_training_btn.setEnabled(True)
            self.omnipose_training_log.append(f"\n{msg}")
            if success:
                training_record = {
                    "model_name": name,
                    "n_epochs": self.omnipose_n_epochs_spin.value(),
                    "learning_rate": self.omnipose_lr_spin.value(),
                    "batch_size": self.omnipose_batch_spin.value(),
                    "nchan": self.omnipose_nchan_spin.value(),
                    "nclasses": self.omnipose_nclasses_spin.value(),
                    "use_gpu": self.omnipose_use_gpu_cb.isChecked(),
                    "mask_filter": self.omnipose_mask_filter_edit.text().strip() or "_mask",
                    "img_filter": self.omnipose_img_filter_edit.text().strip() or "",
                    "train_dir": self.omnipose_train_dir_edit.text().strip(),
                    "pretrained_model": pretrained_arg,
                }
                self._trained_models_this_session.append(training_record)
                if getattr(self.parent_window, "_save_session_log", None):
                    self.parent_window._save_session_log()
                done_msg = (
                    f"Model '{name}' saved to:\n{dest_file}\n\n"
                    "It will appear in the model dropdown below. "
                    "When using it: if no cells are detected, set Diameter (e.g. 30 px) in Segmentation parameters; custom models do not support auto-diameter."
                )
                if save_every_val > 0:
                    done_msg += f"\n\nIntermediate checkpoints (every {save_every_val} epochs) are in:\n{Path(train_dir_abs) / 'models'}"
                done_msg = f"(Initialized from {pretrained_arg})\n\n" + done_msg
                QMessageBox.information(self, "Done", done_msg)
                current = self.model_combo.currentText()
                self.model_combo.clear()
                self.model_combo.addItems(self._get_available_models())
                if self.model_combo.findText(name) >= 0:
                    self.model_combo.setCurrentText(name)
                else:
                    self.model_combo.setCurrentText(current or "bact_phase_omni")
            else:
                QMessageBox.warning(self, "Training failed", msg)
            self._omnipose_train_worker = None

        worker.log_line.connect(on_log)
        worker.finished_with.connect(on_finished)
        worker.start()

    def _build_file_group(self) -> QWidget:
        """Build the file/folder input group with path line edit and browse buttons."""
        file_group = DragDropGroupBox(
            "Input File/Folder (for prediction)",
            parent=self,
            on_drop_callback=self.handle_dropped_path_segmentation
        )
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a TIFF file or folder... (or drag and drop)")
        self.file_path.setMinimumWidth(500)
        file_font = QFont()
        file_font.setPointSize(12)
        self.file_path.setFont(file_font)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.file_path.setSizePolicy(size_policy)
        file_btn = QPushButton("Browse File...")
        file_btn.setFixedWidth(120)
        file_btn.clicked.connect(self.select_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.setFixedWidth(130)
        folder_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(self.file_path, 0)
        file_layout.addWidget(file_btn, 0)
        file_layout.addWidget(folder_btn, 0)
        file_group.setLayout(file_layout)
        return file_group

    def _build_segmentation_params(self) -> QGroupBox:
        """Build Run segmentation section (U-Net style: model full line, then file, params, options, buttons)."""
        seg_group = QGroupBox("Run segmentation")
        main_layout = QVBoxLayout()

        # Model: full line (like U-Net – combo stretches to fill)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        model_row.addWidget(create_info_icon(
            "Omnipose model type for cell segmentation.\n\n"
            "Default: 'bact_phase_omni'\n"
            "Models are downloaded from: https://github.com/kevinjohncutler/omnipose-models\n"
            "Documentation: https://omnipose.readthedocs.io/models.html\n\n"
            "Available Omnipose models:\n"
            "- bact_phase_omni: Bacterial phase contrast (recommended for bacteria)\n"
            "- bact_fluor_omni: Bacterial fluorescence\n"
            "- bact_phase_affinity: Bacterial phase contrast (affinity-based)\n\n"
            "Tip: Choose based on your cell type and imaging modality. User-trained models appear here after training above.",
            self,
        ))
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        available_models = self._get_available_models()
        self.model_combo.addItems(available_models)
        if available_models:
            if 'bact_phase_omni' in available_models:
                self.model_combo.setCurrentText('bact_phase_omni')
            else:
                self.model_combo.setCurrentText(available_models[0])
        model_row.addWidget(self.model_combo, 1)
        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet("color: #7F8C8D; font-size: 9pt; font-style: italic;")
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        model_row.addWidget(self.model_status_label)
        main_layout.addLayout(model_row)

        # Input File/Folder (like U-Net)
        file_group = self._build_file_group()
        main_layout.addWidget(file_group)

        # Parameters: all in one row (diameter, flow, mask)
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Diameter:"))
        params_row.addWidget(create_info_icon(
            "Expected cell diameter in pixels. 0 = auto-detect. Specify a value (e.g. 20–50 for bacteria) for more consistent results.",
            self,
        ))
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setValue(0)
        self.diameter.setDecimals(1)
        self.diameter.setFixedWidth(80)
        params_row.addWidget(self.diameter)
        params_row.addSpacing(12)
        params_row.addWidget(QLabel("Flow:"))
        params_row.addWidget(create_info_icon(
            "Flow error threshold. Default 0.4. Use 0.0 if you get 0 cells with low mask threshold.",
            self,
        ))
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 1.0)
        self.flow_threshold.setValue(0.4)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setDecimals(2)
        self.flow_threshold.setFixedWidth(80)
        params_row.addWidget(self.flow_threshold)
        params_row.addSpacing(12)
        params_row.addWidget(QLabel("Mask thresh:"))
        params_row.addWidget(create_info_icon(
            "Distance/cell probability threshold. If 'No cell pixels found', try -1, -5, or -10; also set Flow to 0.",
            self,
        ))
        self.mask_threshold = QDoubleSpinBox()
        self.mask_threshold.setRange(-20.0, 1.0)
        self.mask_threshold.setValue(0.0)
        self.mask_threshold.setSingleStep(0.5)
        self.mask_threshold.setDecimals(2)
        self.mask_threshold.setFixedWidth(80)
        params_row.addWidget(self.mask_threshold)
        params_row.addStretch()
        reset_seg_btn = QPushButton("Reset to Default")
        reset_seg_btn.setFixedWidth(120)
        reset_seg_btn.setToolTip("Reset all segmentation parameters to their default values")
        reset_seg_btn.clicked.connect(self.reset_segmentation_defaults)
        params_row.addWidget(reset_seg_btn)
        check_pipeline_btn = QPushButton("Check pipeline")
        check_pipeline_btn.setFixedWidth(130)
        check_pipeline_btn.setToolTip("Run model diagnostic: check and compare models.")
        check_pipeline_btn.clicked.connect(self._open_check_pipeline_dialog)
        params_row.addWidget(check_pipeline_btn)
        params_row.addSpacing(10)
        main_layout.addLayout(params_row)

        # Options: all in one row (Use GPU, DBSCAN, Create overlay)
        options_row = QHBoxLayout()
        self.use_gpu = QCheckBox("Use GPU")
        self.use_gpu.setChecked(True)
        options_row.addWidget(self.use_gpu)
        options_row.addSpacing(37)
        self.cluster = QCheckBox("Use DBSCAN clustering")
        self.cluster.setChecked(False)
        options_row.addWidget(self.cluster)
        options_row.addWidget(create_info_icon(
            "Use DBSCAN clustering to help segment thin/elongated cells (e.g. bacteria, worms). Slightly increases processing time.",
            self,
        ))
        options_row.addSpacing(37)
        self.create_overlay = QCheckBox("Create overlay visualization")
        self.create_overlay.setChecked(True)
        options_row.addWidget(self.create_overlay)
        options_row.addWidget(create_info_icon(
            "Create overlay TIFF (masks over original). Saved in an 'overlays' subfolder so the main folder has only mask files for Analyze Masks. Disable to speed up when you only need masks.",
            self,
        ))
        options_row.addStretch()
        main_layout.addLayout(options_row)

        def check_model_availability():
            try:
                display = self.model_combo.currentText()
                model_type, pretrained_path = self._resolve_model_selection(display)
                if pretrained_path is not None:
                    self.model_status_label.setText("✓ User model")
                    self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                    d, f, m = USER_MODEL_DEFAULTS
                    self.diameter.setValue(d)
                    self.flow_threshold.setValue(f)
                    self.mask_threshold.setValue(m)
                    return
                from cellpose_omni.models import MODEL_DIR
                torch_str = "torch"
                basename = f"{model_type}{torch_str}_0"
                cache_model_file = Path(MODEL_DIR) / basename
                if cache_model_file.exists():
                    self.model_status_label.setText("✓ Cached")
                    self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 9pt;")
                else:
                    self.model_status_label.setText("Will download on first use")
                    self.model_status_label.setStyleSheet("color: #FF9800; font-size: 9pt;")
                # Apply default diameter, flow, mask for this built-in model so user sees them
                d, f, m = SEGMENTATION_MODEL_DEFAULTS.get(model_type or display, SEGMENTATION_DEFAULT_FALLBACK)
                self.diameter.setValue(d)
                self.flow_threshold.setValue(f)
                self.mask_threshold.setValue(m)
            except Exception:
                get_logger().debug("Model availability check failed", exc_info=True)
                self.model_status_label.setText("")
        self.model_combo.currentTextChanged.connect(check_model_availability)
        check_model_availability()

        self.preview_btn = QPushButton("Preview Omnipose Segmentation")
        self.preview_btn.setEnabled(False)
        def safe_preview_segmentation():
            try:
                self.preview_segmentation()
            except BaseException as e:
                get_logger().exception("Error in preview segmentation")
                import traceback
                try:
                    QMessageBox.critical(self, "Preview Error", f"Error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}")
                except Exception:
                    pass
        self.preview_btn.clicked.connect(safe_preview_segmentation)
        main_layout.addWidget(self.preview_btn)

        self.process_btn = QPushButton("Process File or Folder")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_file_or_folder)
        main_layout.addWidget(self.process_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the current segmentation job.")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self._user_cancelled_processing = False
        def on_cancel_processing():
            if hasattr(self, "worker") and self.worker is not None and self.worker.isRunning():
                self._user_cancelled_processing = True
                self.worker.terminate_process()
                self.parent_window.log_output("Cancelled by user.\n")
        self.cancel_btn.clicked.connect(on_cancel_processing)
        main_layout.addWidget(self.cancel_btn)

        seg_group.setLayout(main_layout)
        return seg_group

    def _update_button_states(self):
        """Update button enabled states: path must exist and be a file or directory."""
        raw = self.file_path.text() and self.file_path.text().strip()
        has_path = False
        if raw:
            p = Path(raw)
            has_path = p.exists() and (p.is_file() or p.is_dir())
        if hasattr(self, 'preview_btn'):
            self.preview_btn.setEnabled(has_path)
        if hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(has_path)
    
    def _update_overlay_checkbox_state(self):
        """Update overlay checkbox tooltip based on folder vs file mode"""
        is_folder = self._is_folder_mode()
        # Overlay is now supported in both modes, so always enable it
        self.create_overlay.setEnabled(True)
        if is_folder:
            self.create_overlay.setToolTip(
                "Create overlay visualization showing masks overlaid on original images.\n\n"
                "Default: Enabled (checked)\n"
                "When enabled: Creates a colored overlay TIFF file for each file in the folder.\n"
                "When disabled: Skips overlay creation (faster processing).\n\n"
                "Saved as: filename_overlay.tif (in each file's output directory)"
            )
        else:
            self.create_overlay.setToolTip(
                "Create overlay visualization showing masks overlaid on original images.\n\n"
                "Default: Enabled (checked)\n"
                "When enabled: Creates a colored overlay TIFF file.\n"
                "When disabled: Skips overlay creation (faster processing).\n\n"
                "Saved as: filename_overlay.tif"
            )

    def _try_push_metadata_from_path(self):
        """When path is a TIFF file (or folder with a TIFF), read calibration via tiff_metadata and push to shared_metadata."""
        raw = (self.file_path.text() or "").strip()
        if not raw:
            return
        path = Path(raw)
        tiff_path = None
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            tiff_path = path
        elif path.is_dir():
            for ext in ("*.tif", "*.tiff"):
                first = next(path.glob(ext), None)
                if first is not None and first.is_file():
                    tiff_path = first
                    break
        if tiff_path is None:
            return
        try:
            ps, tp = read_calibration_from_path_or_original(tiff_path)
            if (ps > 0 or tp > 0) and hasattr(self, "parent_window") and getattr(self.parent_window, "shared_metadata", None) is not None:
                if ps > 0:
                    self.parent_window.shared_metadata["pixel_size"] = ps
                if tp > 0:
                    self.parent_window.shared_metadata["time_per_frame_min"] = tp
                self.parent_window.update_all_tabs_with_metadata()
        except Exception:
            pass

    def select_file(self):
        """Open file dialog to select a TIFF file"""
        try:
            # Get the current file path if one exists, to use as starting directory
            start_dir = ""
            if self.file_path.text() and Path(self.file_path.text()).exists():
                start_dir = str(Path(self.file_path.text()).parent)
            elif self.file_path.text():
                start_dir = str(Path(self.file_path.text()).parent)
            else:
                start_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_SEGMENTATION) or str(Path.home())

            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select TIFF File",
                start_dir,
                "TIFF Files (*.tif *.tiff);;All Files (*)"
            )
            if file_path:
                if getattr(self.parent_window, "set_last_dir_for_tab", None):
                    self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, file_path)
                self.file_path.setText(file_path)
                self._update_overlay_checkbox_state()
                if self.parent_window and hasattr(self.parent_window, 'log_output'):
                    self.parent_window.log_output(f"Selected file: {file_path}\n")
        except Exception as e:
            QMessageBox.critical(
                self, 
                "File Selection Error", 
                f"Could not open file dialog:\n{str(e)}\n\n{traceback.format_exc()}"
            )
            if self.parent_window and hasattr(self.parent_window, 'log_output'):
                self.parent_window.log_output(f"Error selecting file: {str(e)}\n{traceback.format_exc()}\n")
    
    def select_folder(self):
        """Open folder dialog to select a folder"""
        start_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_SEGMENTATION) or str(Path.home())
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if folder_path:
            if getattr(self.parent_window, "set_last_dir_for_tab", None):
                self.parent_window.set_last_dir_for_tab(LAST_DIR_SEGMENTATION, folder_path)
            self.file_path.setText(folder_path)
            self._update_overlay_checkbox_state()
    
    def handle_dropped_path_segmentation(self, dropped_path):
        """Handle dropped file or folder path for segmentation tab"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        # Update overlay checkbox state
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
        Return display names for the model dropdown: built-in Omnipose models first,
        then user-trained models (filename stem) from get_omnipose_user_models_dir().
        """
        names = list(BUILTIN_OMNIPOSE_MODELS)
        try:
            user_dir = get_omnipose_user_models_dir()
            for p in sorted(user_dir.glob(f"*{USER_MODEL_EXTENSION}")):
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
        try:
            user_path = get_omnipose_user_models_dir() / f"{display_name}{USER_MODEL_EXTENSION}"
            if user_path.is_file():
                return (None, user_path)
        except Exception:
            pass
        return (None, None)

    def reset_segmentation_defaults(self):
        """Reset all segmentation parameters to their default values (model + diameter/flow/mask for that model)."""
        # Set default model, fallback to first available if bact_phase_omni not available
        if 'bact_phase_omni' in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText('bact_phase_omni')
        elif self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
        # currentTextChanged will run and set diameter/flow/mask from SEGMENTATION_MODEL_DEFAULTS;
        # if it didn't fire (same text), apply defaults for current model here
        display = self.model_combo.currentText()
        model_type, pretrained_path = self._resolve_model_selection(display)
        if pretrained_path is not None:
            d, f, m = USER_MODEL_DEFAULTS
        else:
            d, f, m = SEGMENTATION_MODEL_DEFAULTS.get(model_type or display, SEGMENTATION_DEFAULT_FALLBACK)
        self.diameter.setValue(d)
        self.flow_threshold.setValue(f)
        self.mask_threshold.setValue(m)
        self.use_gpu.setChecked(True)
        self.cluster.setChecked(False)
        self.create_overlay.setChecked(True)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        out = get_from_widgets([
            ("model", self.model_combo),
            ("diameter", self.diameter),
            ("flow_threshold", self.flow_threshold),
            ("mask_threshold", self.mask_threshold),
            ("use_gpu", self.use_gpu),
            ("cluster", self.cluster),
            ("create_overlay", self.create_overlay),
        ])
        out["trained_models_this_session"] = list(self._trained_models_this_session)
        # Omnipose training section (inline on tab)
        if hasattr(self, "omnipose_train_dir_edit"):
            out_omni = get_from_widgets([
                ("omnipose_train_dir", self.omnipose_train_dir_edit),
                ("omnipose_img_filter", self.omnipose_img_filter_edit),
                ("omnipose_mask_filter", self.omnipose_mask_filter_edit),
                ("omnipose_model_name", self.omnipose_model_name_edit),
                ("omnipose_pretrained", self.omnipose_pretrained_combo),
                ("omnipose_n_epochs", self.omnipose_n_epochs_spin),
                ("omnipose_lr", self.omnipose_lr_spin),
                ("omnipose_batch", self.omnipose_batch_spin),
                ("omnipose_tyx_y", self.omnipose_tyx_y_spin),
                ("omnipose_tyx_x", self.omnipose_tyx_x_spin),
                ("omnipose_save_each", self.omnipose_save_each_spin),
            ])
            out_omni["omnipose_use_gpu"] = self.omnipose_use_gpu_cb.isChecked()
            out_omni["omnipose_use_tiled"] = self.omnipose_use_tiled_cb.isChecked()
            out.update(out_omni)
        return out

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("model", self.model_combo),
            ("diameter", self.diameter),
            ("flow_threshold", self.flow_threshold),
            ("mask_threshold", self.mask_threshold),
            ("use_gpu", self.use_gpu),
            ("cluster", self.cluster),
            ("create_overlay", self.create_overlay),
        ])
        if hasattr(self, "omnipose_train_dir_edit"):
            apply_to_widgets(data, [
                ("omnipose_train_dir", self.omnipose_train_dir_edit),
                ("omnipose_img_filter", self.omnipose_img_filter_edit),
                ("omnipose_mask_filter", self.omnipose_mask_filter_edit),
                ("omnipose_model_name", self.omnipose_model_name_edit),
                ("omnipose_pretrained", self.omnipose_pretrained_combo),
                ("omnipose_n_epochs", self.omnipose_n_epochs_spin),
                ("omnipose_lr", self.omnipose_lr_spin),
                ("omnipose_batch", self.omnipose_batch_spin),
                ("omnipose_tyx_y", self.omnipose_tyx_y_spin),
                ("omnipose_tyx_x", self.omnipose_tyx_x_spin),
                ("omnipose_save_each", self.omnipose_save_each_spin),
            ])
            if "omnipose_use_gpu" in data:
                self.omnipose_use_gpu_cb.setChecked(bool(data["omnipose_use_gpu"]))
            if "omnipose_use_tiled" in data:
                self.omnipose_use_tiled_cb.setChecked(bool(data["omnipose_use_tiled"]))
    
    def preview_segmentation(self):
        """Preview segmentation on a single frame or file from folder"""
        # Note: This method is called from safe_preview_segmentation() wrapper
        # which catches all exceptions to prevent PyQt6 from aborting
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a TIFF file or folder first.")
            return
        
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return
        
        is_folder = self._is_folder_mode()
        if is_folder:
            # Folder mode: show file selection dialog first
            folder = Path(self.file_path.text())
            # Look for TIFF files (not mask files - this is for processing, not analyzing)
            # Use set to avoid duplicates from multiple glob patterns
            tiff_files_set = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                            set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out mask files (we want original files for segmentation preview)
            tiff_files = sorted([f for f in tiff_files_set if '_masks' not in f.name.lower()], 
                               key=lambda x: x.name.lower())
            
            if not tiff_files:
                QMessageBox.warning(self, "No TIFF Files Found", 
                                  "No TIFF files found in the selected folder.\n\n"
                                  "Please select a folder containing TIFF files to process.")
                return
            
            # Let user choose which file to preview - use custom dialog
            file_dialog = QDialog(self)
            file_dialog.setWindowTitle("Select File for Preview")
            file_dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Label
            label = QLabel("Select a file to preview segmentation:")
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
            for f in tiff_files:
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
            for f in tiff_files:
                if f.name == selected_name:
                    selected_file = f
                    break
            
            if not selected_file:
                return
            
            # Use selected file for preview
            file_path = str(selected_file)
        else:
            # Single file mode: use the selected file directly
            file_path = self.file_path.text()
        
        # Show progress dialog while loading
        loading_progress = QProgressDialog("Loading file and initializing preview...", None, 0, 0, self)
        loading_progress.setWindowTitle("Loading")
        loading_progress.setAutoClose(False)
        loading_progress.setAutoReset(False)
        loading_progress.setCancelButton(None)
        loading_progress.show()
        QApplication.processEvents()
        
        # Load TIFF file
        try:
            loading_progress.setLabelText("Loading TIFF file...")
            QApplication.processEvents()
            
            import tifffile
            from bactimate.gui.image_utils import (
                to_qpixmap, build_segmentation_overlay, scale_for_preview, draw_scale_bar,
                downsample_for_preview_seg, upsample_mask_to_shape, PREVIEW_SEG_MAX_SIZE,
            )
            from bactimate.gui.image_canvas import ImageCanvas

            # CRITICAL: Patch omnipose.core.OMNI_MODELS before cellpose_omni imports it
            # This fixes "cannot import name 'OMNI_MODELS' from 'omnipose.core'" error in frozen builds
            # Also ensure get_masks patch is applied
            try:
                import omnipose.core
                # Re-apply patch in case it wasn't applied earlier (omnipose imported late)
                is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                if is_frozen:
                    _patch_omnipose_get_masks()
                if not hasattr(omnipose.core, 'OMNI_MODELS'):
                    # Construct OMNI_MODELS from available model constants
                    OMNI_MODELS = []
                    if hasattr(omnipose.core, 'C1_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_MODELS)
                    if hasattr(omnipose.core, 'C2_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_MODELS)
                    if hasattr(omnipose.core, 'C1_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C1_BD_MODELS)
                    if hasattr(omnipose.core, 'C2_BD_MODELS'):
                        OMNI_MODELS.extend(omnipose.core.C2_BD_MODELS)
                    # Remove duplicates
                    seen = set()
                    OMNI_MODELS = [x for x in OMNI_MODELS if not (x in seen or seen.add(x))]
                    # Default fallback if still empty
                    if not OMNI_MODELS:
                        OMNI_MODELS = ['bact_omni', 'bact_phase_omni',
                                      'bact_fluor_omni']
                    omnipose.core.OMNI_MODELS = OMNI_MODELS
            except Exception:
                get_logger().debug("Omnipose get_masks patch failed, using fallback", exc_info=True)
            
            from cellpose_omni import models
            from cellpose_omni.io import imread
            import numpy as np
            
            images = safe_imread(file_path)
            
            loading_progress.setLabelText("Processing image data...")
            QApplication.processEvents()
            
            # Convert to float32 for MPS compatibility (Apple Silicon GPU doesn't support float64)
            if images.dtype != np.float32:
                images = images.astype(np.float32)
            
            # Handle different input formats
            if images.ndim == 2:
                images = images[np.newaxis, ...]
            elif images.ndim == 3:
                if images.shape[-1] < 10:  # Likely channels last
                    images = images.transpose(2, 0, 1)
                    images = images[np.newaxis, ...]
            elif images.ndim == 4:
                if images.shape[-1] < 10:  # Channels last
                    images = images.transpose(0, 3, 1, 2)
                elif images.shape[0] < 10:  # Channels first
                    images = images.transpose(1, 0, 2, 3)
            
            # Check if we have any frames
            if images.size == 0 or images.shape[0] == 0:
                QMessageBox.critical(preview_dialog, "Empty File", 
                                   "The TIFF file appears to be empty or has no frames.")
                return
            
            num_frames = images.shape[0]
            
            # Non-modal so user can access main window; parent=main window so preview closes when app quits (e.g. Cmd+Q).
            main_win = self.window() if hasattr(self, "window") and callable(self.window) else None
            preview_dialog = QDialog(main_win)
            preview_dialog.setWindowTitle("Segmentation Preview")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            
            layout = QVBoxLayout()

            # Title label (frame / cells / filename)
            title_label = QLabel("")
            title_label.setStyleSheet("font-weight: bold; color: #ddd;")
            layout.addWidget(title_label)

            # Qt-native image canvas (zoom: wheel; pan: left-drag or middle-drag)
            image_canvas = ImageCanvas()
            image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
            image_canvas.setMinimumSize(400, 300)
            image_canvas.setStyleSheet("background-color: #1e1e1e;")
            image_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            layout.addWidget(image_canvas, 1)

            # Interactive LUT histogram (same as Convert TIFF: drag min/max lines to adjust display)
            lut_data = {'frame_idx': None, 'vmin': None, 'vmax': None, 'frame_min': None, 'frame_max': None}
            hist_vmin_line = None
            hist_vmax_line = None
            dragging_line = None
            try:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.figure import Figure
                _has_histogram = True
            except ImportError:
                _has_histogram = False
            if _has_histogram:
                hist_group = QGroupBox("LUT (click/drag lines to adjust min/max)")
                hist_group.setStyleSheet("QGroupBox { font-weight: 600; }")
                hist_layout = QVBoxLayout()
                # Match TIFF conversion preview: same figure size and canvas height so histogram fills the section
                hist_fig = Figure(figsize=(10, 3.0))
                hist_canvas = FigureCanvas(hist_fig)
                hist_canvas.setMinimumHeight(180)
                hist_canvas.setMaximumHeight(180)
                hist_ax = hist_fig.add_subplot(111)
                hist_fig.patch.set_facecolor('#2b2b2b')
                hist_ax.set_facecolor('#2b2b2b')
                hist_ax.tick_params(colors='white', labelsize=9)
                for spine in hist_ax.spines.values():
                    spine.set_color('white')
                hist_ax.xaxis.label.set_color('white')
                hist_ax.yaxis.label.set_color('white')
                hist_fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.15)
                hist_canvas.setStyleSheet("background-color: #2b2b2b;")
                hist_layout.addWidget(hist_canvas)
                hist_group.setLayout(hist_layout)
                hist_group.setMaximumHeight(280)
                layout.addWidget(hist_group)

                def refresh_image_display():
                    """Redraw overlay using current display_frame, mask, and LUT (vmin/vmax)."""
                    display_frame = getattr(preview_dialog, 'current_display_frame', None)
                    mask = getattr(preview_dialog, 'masks', None)
                    if display_frame is None or mask is None:
                        return
                    vmin = lut_data.get('vmin')
                    vmax = lut_data.get('vmax')
                    if vmin is None or vmax is None or vmax <= vmin:
                        vmin = lut_data.get('frame_min', 0)
                        vmax = lut_data.get('frame_max', 255)
                        if vmax <= vmin:
                            vmax = vmin + 1.0
                        lut_data['vmin'] = vmin
                        lut_data['vmax'] = vmax
                    vmin, vmax = float(vmin), float(vmax)
                    vrange = max(vmax - vmin, 1e-6)
                    display_uint8 = ((np.asarray(display_frame).astype(np.float64) - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
                    disp_scaled, mask_scaled = scale_for_preview(display_uint8, mask, max_size=PREVIEW_SEG_MAX_SIZE)
                    overlay = build_segmentation_overlay(disp_scaled, mask_scaled, alpha=0.5)
                    pixmap = to_qpixmap(overlay)
                    h, w = overlay.shape[:2]
                    image_canvas.set_pixmap(pixmap, (w, h), reset_view=False)

                def update_histogram(frame_2d, vmin, vmax):
                    """Draw histogram with min/max lines (and optional drag)."""
                    nonlocal hist_vmin_line, hist_vmax_line
                    hist_ax.clear()
                    if frame_2d is None or not isinstance(frame_2d, np.ndarray) or frame_2d.size == 0:
                        hist_ax.set_xlabel("Intensity", color="white", fontsize=9)
                        hist_ax.set_ylabel("Log(Count+1)", color="white", fontsize=9)
                        hist_canvas.draw()
                        return
                    flat = np.asarray(frame_2d).flatten().astype(np.float64)
                    flat = flat[np.isfinite(flat)]
                    if flat.size == 0:
                        hist_canvas.draw()
                        return
                    fmin, fmax = float(flat.min()), float(flat.max())
                    lut_data['frame_min'] = fmin
                    lut_data['frame_max'] = fmax
                    # Exclude zeros for histogram so the signal peak is prominent (like TIFF conversion preview)
                    flat_hist = flat[flat > 0]
                    if flat_hist.size == 0:
                        flat_hist = flat
                    if vmin is None:
                        vmin = fmin
                    if vmax is None or vmax <= vmin:
                        vmax = fmax if fmax > fmin else fmin + 1.0
                    lut_data['vmin'] = vmin
                    lut_data['vmax'] = vmax
                    bins = min(256, max(2, int(fmax - fmin + 1)))
                    hist, bin_edges = np.histogram(flat_hist, bins=bins, range=(fmin, fmax))
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    hist_ax.bar(bin_centers, np.log1p(hist), width=bin_edges[1] - bin_edges[0],
                                color='#4A90E2', alpha=0.8, edgecolor='none')
                    hist_vmin_line = hist_ax.axvline(vmin, color='#FF6B6B', linewidth=2, linestyle='--', label=f'Min: {vmin:.1f}')
                    hist_vmax_line = hist_ax.axvline(vmax, color='#4ECDC4', linewidth=2, linestyle='--', label=f'Max: {vmax:.1f}')
                    hist_ax.axvspan(vmin, vmax, alpha=0.15, color='yellow')
                    hist_ax.set_xlabel("Intensity", color="white", fontsize=9)
                    hist_ax.set_ylabel("Log(Count+1)", color="white", fontsize=9)
                    hist_ax.set_title("Click/drag lines to adjust min/max", color="white", fontsize=9, pad=10)
                    hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                    hist_ax.grid(True, alpha=0.25, color="white", linestyle="-", linewidth=0.5)
                    hist_ax.margins(x=0.02, y=0.08)
                    hist_fig.tight_layout(pad=2.0)
                    hist_canvas.draw()

                def on_histogram_press(event):
                    if event.inaxes != hist_ax or event.xdata is None:
                        return
                    nonlocal dragging_line
                    x = event.xdata
                    vmin, vmax = lut_data.get('vmin'), lut_data.get('vmax')
                    fmin, fmax = lut_data.get('frame_min', 0), lut_data.get('frame_max', 65535)
                    if vmin is None or vmax is None:
                        return
                    threshold = max((fmax - fmin) * 0.05, 1.0)
                    if abs(x - vmin) < threshold and abs(x - vmin) <= abs(x - vmax):
                        dragging_line = 'min'
                    elif abs(x - vmax) < threshold:
                        dragging_line = 'max'
                    else:
                        dragging_line = 'min' if x < (vmin + vmax) / 2 else 'max'

                def on_histogram_motion(event):
                    if event.inaxes != hist_ax or event.xdata is None or dragging_line is None:
                        return
                    x = event.xdata
                    fmin = lut_data.get('frame_min', 0)
                    fmax = lut_data.get('frame_max', 65535)
                    vmin, vmax = lut_data.get('vmin'), lut_data.get('vmax')
                    if dragging_line == 'min':
                        new_min = max(fmin, min(x, (vmax or fmax) - 1e-6))
                        lut_data['vmin'] = new_min
                    else:
                        new_max = min(fmax, max(x, (vmin or fmin) + 1e-6))
                        lut_data['vmax'] = new_max
                    refresh_image_display()
                    vmin, vmax = lut_data['vmin'], lut_data['vmax']
                    if hist_vmin_line is not None:
                        hist_vmin_line.set_xdata([vmin, vmin])
                    if hist_vmax_line is not None:
                        hist_vmax_line.set_xdata([vmax, vmax])
                    for patch in list(hist_ax.patches):
                        if getattr(patch, 'get_alpha', lambda: None)() == 0.15:
                            patch.remove()
                    hist_ax.axvspan(vmin, vmax, alpha=0.15, color='yellow')
                    if hist_vmin_line is not None and hist_vmax_line is not None:
                        hist_vmin_line.set_label(f'Min: {vmin:.1f}')
                        hist_vmax_line.set_label(f'Max: {vmax:.1f}')
                        hist_ax.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
                    hist_canvas.draw()

                def on_histogram_release(event):
                    nonlocal dragging_line
                    dragging_line = None

                hist_canvas.mpl_connect('button_press_event', on_histogram_press)
                hist_canvas.mpl_connect('motion_notify_event', on_histogram_motion)
                hist_canvas.mpl_connect('button_release_event', on_histogram_release)
            else:
                def update_histogram(frame_2d, vmin=None, vmax=None):
                    pass
                def refresh_image_display():
                    pass

            # Store data for navigation
            preview_dialog.images = images
            preview_dialog.current_frame = 0
            preview_dialog.masks = None
            preview_dialog.model = None
            
            # Status label for model loading messages (defined early so load_model can use it)
            status_label = QLabel("Ready")
            status_label.setStyleSheet("color: #7F8C8D; font-size: 9pt; font-style: italic;")
            
            def find_model_file(model_type, model_index=0, use_torch=True):
                """
                Find model file in the default cache location.
                Models are cached in ~/.cellpose/models and persist between sessions.
                Returns the path to the model file if found, None otherwise.
                """
                from pathlib import Path
                from cellpose_omni.models import MODEL_DIR
                
                # Construct model filename
                torch_str = 'torch' if use_torch else ''
                basename = f'{model_type}{torch_str}_{model_index}'
                
                # Check default cache location
                cache_model_file = Path(MODEL_DIR) / basename
                
                if cache_model_file.exists():
                    return str(cache_model_file)
                
                # Not found
                return None
            
            def load_model():
                """Load Omnipose model (only reloads if model type or GPU setting changed)."""
                params = get_current_parameters()
                current_model_type = params.get("model_type")
                pretrained_path = params.get("pretrained_model")
                current_use_gpu = params["use_gpu"]
                current_key = pretrained_path or current_model_type or ""
                needs_reload = (
                    preview_dialog.model is None
                    or (getattr(preview_dialog, "_last_model_key", None) != current_key)
                    or (getattr(preview_dialog, "_last_use_gpu", None) != current_use_gpu)
                )
                if needs_reload:
                    preview_dialog.segmentation_cache.clear()
                def set_loaded_status():
                    """Set status label to show whether GPU is actually in use (CUDA / MPS / CPU)."""
                    import torch
                    if not current_use_gpu:
                        status_label.setText("Model loaded (CPU – Use GPU unchecked)")
                        return
                    if torch.cuda.is_available():
                        try:
                            name = torch.cuda.get_device_name(0) or "CUDA"
                        except Exception:
                            name = "CUDA"
                        status_label.setText(f"Model loaded (GPU: {name})")
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        status_label.setText("Model loaded (GPU: Apple MPS)")
                    else:
                        status_label.setText("Model loaded (CPU – GPU requested but not available)")
                if needs_reload:
                    try:
                        import os
                        from urllib.error import HTTPError  # required for except HTTPError below (must be in scope for all paths)
                        # User-trained model: load by path
                        if pretrained_path and os.path.isfile(pretrained_path):
                            status_label.setText("Loading user model...")
                            QApplication.processEvents()
                            # Debug: inspect checkpoint before load (see Logs/omnipose_debug.log)
                            try:
                                inspect_info = inspect_omnipose_checkpoint(pretrained_path)
                                ckpt_nout = inspect_info.get("output_layer_out_channels")
                                append_omnipose_debug(
                                    f"[BactiMate USER_MODEL pre-load] path={pretrained_path!r} "
                                    f"inspect: nchan={inspect_info.get('nchan')} nclasses={inspect_info.get('nclasses')} "
                                    f"ckpt_output_channels={ckpt_nout} first_layer_shape={inspect_info.get('first_layer_shape')} keys={inspect_info.get('keys', [])[:15]!r}"
                                )
                            except Exception:
                                pass
                            # cellpose_omni only sets omni=True when "omni" is in the filename
                            path_for_loader = ensure_omni_in_model_path(pretrained_path)
                            preview_dialog.model = models.Cellpose(
                                gpu=current_use_gpu,
                                pretrained_model=path_for_loader,
                                omni=True,
                            )
                            # Debug: what did the patched loader set on model.cp?
                            try:
                                cp = getattr(preview_dialog.model, "cp", None)
                                nchan = getattr(cp, "nchan", None)
                                nclasses = getattr(cp, "nclasses", None)
                                append_omnipose_debug(
                                    f"[BactiMate USER_MODEL post-load] model.cp.nchan={nchan} model.cp.nclasses={nclasses}"
                                )
                                if inspect_info.get("output_layer_out_channels") is not None and nclasses != inspect_info.get("output_layer_out_channels"):
                                    append_omnipose_debug(
                                        f"[BactiMate USER_MODEL] MISMATCH: checkpoint output head has {inspect_info['output_layer_out_channels']} channels but model.cp.nclasses={nclasses}. "
                                        "This can cause crash or 0 cells. See docs/CUSTOM_MODEL_TROUBLESHOOTING.md or run: python -m bactimate.scripts.inspect_omnipose_checkpoint <path>"
                                    )
                                get_logger().info("User model loaded: nchan=%s nclasses=%s", nchan, nclasses)
                                # If size model failed to load, diameter=0 won't rescale -> often 0 cells
                                if getattr(preview_dialog.model, "pretrained_size", None) is None:
                                    append_omnipose_debug(
                                        "[BactiMate USER_MODEL] size model not loaded (pretrained_size is None). "
                                        "diameter=0 will not rescale -> often 0 cells. "
                                        "Run from project root: venv/bin/python3 setup/apply_all_patches.py then restart BactiMate."
                                    )
                            except Exception:
                                pass
                            preview_dialog._last_model_key = current_key
                            preview_dialog._last_use_gpu = current_use_gpu
                            set_loaded_status()
                            return True
                        from cellpose_omni.models import model_path, MODEL_DIR
                        from pathlib import Path
                        expected_model_dir = PROJECT_ROOT / "models"
                        if expected_model_dir.exists() and str(MODEL_DIR) != str(expected_model_dir):
                            pass
                        model_file = find_model_file(current_model_type, 0, True)
                        
                        if model_file and os.path.exists(model_file):
                            # Model found in cache
                            status_label.setText("Loading model from cache...")
                            QApplication.processEvents()
                        else:
                            # Model not found - need to download
                            status_label.setText("Downloading model (this may take a moment)...")
                            QApplication.processEvents()
                            # Call model_path - it will download if needed
                            try:
                                model_file = model_path(current_model_type, 0, True)
                                # Verify the model was actually downloaded
                                if not model_file or not os.path.exists(model_file):
                                    # model_path returned a path but file doesn't exist - try to download explicitly
                                    status_label.setText("Model not found, downloading...")
                                    QApplication.processEvents()
                                    # Force download by calling model_path again (it should download if file doesn't exist)
                                    model_file = model_path(current_model_type, 0, True)
                                    # Wait a moment for download to complete
                                    import time
                                    time.sleep(0.5)
                                    # Verify again
                                    if not model_file or not os.path.exists(model_file):
                                        raise FileNotFoundError(f"Model file not found after download attempt: {model_file}")
                                status_label.setText("Model downloaded, loading...")
                                QApplication.processEvents()
                            except HTTPError as http_err:
                                # Handle 404 errors (model not found in repository)
                                if http_err.code == 404:
                                    # Check if model exists in cache
                                    model_file = find_model_file(current_model_type, 0, True)
                                    if model_file and os.path.exists(model_file):
                                        status_label.setText("Using cached model (not in repo)...")
                                        QApplication.processEvents()
                                    else:
                                        # Model not available - show helpful error
                                        QMessageBox.warning(
                                            preview_dialog,
                                            "Model Not Available",
                                            f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                            f"Available Omnipose models:\n"
                                            f"  • bact_phase_omni (recommended for bacteria)\n"
                                            f"  • bact_fluor_omni\n"
                                            f"  • bact_phase_affinity\n\n"
                                            f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                            f"Please select a different model from the dropdown."
                                        )
                                        status_label.setText("Model not available")
                                        return False
                                else:
                                    # Other HTTP errors - check local directory one more time
                                    model_file = find_model_file(current_model_type, 0, True)
                                    if model_file and os.path.exists(model_file):
                                        status_label.setText("Download failed, using local/cached model...")
                                        QApplication.processEvents()
                                    else:
                                        # Re-raise to be handled by outer exception handler
                                        raise
                            except Exception as download_err:
                                # Check one more time if model exists (might have been downloaded in background)
                                model_file = find_model_file(current_model_type, 0, True)
                                if model_file and os.path.exists(model_file):
                                    status_label.setText("Using model from cache...")
                                    QApplication.processEvents()
                                else:
                                    # Re-raise to be handled by outer exception handler
                                    raise
                        
                        # Final verification: ensure model file exists before loading
                        if not model_file or not os.path.exists(model_file):
                            raise FileNotFoundError(
                                f"Model file not found: {model_file}\n\n"
                                f"Model '{current_model_type}' may not be available or download failed.\n"
                                f"Please check your internet connection and try again."
                            )
                        
                        status_label.setText("Loading model...")
                        QApplication.processEvents()
                        
                        preview_dialog.model = models.Cellpose(
                            gpu=current_use_gpu, 
                            model_type=current_model_type, 
                            omni=True
                        )
                        # Debug: built-in model load (see Logs/omnipose_debug.log)
                        try:
                            cp = getattr(preview_dialog.model, "cp", None)
                            nchan = getattr(cp, "nchan", None)
                            nclasses = getattr(cp, "nclasses", None)
                            append_omnipose_debug(
                                f"[BactiMate BUILTIN post-load] model_type={current_model_type!r} "
                                f"model.cp.nchan={nchan} model.cp.nclasses={nclasses}"
                            )
                        except Exception:
                            pass

                        preview_dialog._last_model_key = current_key
                        preview_dialog._last_use_gpu = current_use_gpu
                        set_loaded_status()
                        QApplication.processEvents()
                    except RuntimeError as e:
                        error_msg = str(e)
                        # Check if it's a model architecture mismatch
                        if "size mismatch" in error_msg or "state_dict" in error_msg or "checkpoint" in error_msg.lower() or "Unexpected key" in error_msg:
                            # Check if there's a corrupted or wrong model file (e.g., cp3 symlinked as omni)
                            from cellpose_omni.models import model_path, MODEL_DIR
                            import os
                            from pathlib import Path
                            expected_basename = f'{current_model_type}torch_0'
                            model_file_path = Path(MODEL_DIR) / expected_basename
                            
                            # Check if model file is a symlink pointing to wrong model (e.g., cp3 instead of omni)
                            if model_file_path.exists() and model_file_path.is_symlink():
                                try:
                                    target = model_file_path.readlink()
                                    # If symlink points to cp3 model but we need omni, remove it
                                    if '_cp3' in str(target) and '_omni' in current_model_type:
                                        model_file_path.unlink()
                                        status_label.setText("Removed incorrect model symlink...")
                                        QApplication.processEvents()
                                except Exception:
                                    get_logger().debug("Could not remove incorrect model symlink", exc_info=True)
                            
                            # Try to fix by clearing model cache and retrying
                            try:
                                model_file = model_path(current_model_type, 0, True)
                                if os.path.exists(model_file):
                                    status_label.setText("Fixing model architecture mismatch...")
                                    QApplication.processEvents()
                                    # Backup the corrupted model
                                    backup_file = model_file + ".corrupted"
                                    if os.path.exists(backup_file):
                                        os.remove(backup_file)
                                    os.rename(model_file, backup_file)
                                    # Retry loading model (will download if needed)
                                    status_label.setText("Re-loading model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_key = current_key
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Model reloaded successfully!")
                                    QApplication.processEvents()
                                else:
                                    raise RuntimeError("Model file not found after clearing cache")
                            except Exception as retry_error:
                                error_details = f"Original error: {error_msg}\nRetry error: {retry_error}"
                                # Provide helpful error message
                                help_text = (
                                    f"The model checkpoint has a different architecture than expected.\n\n"
                                    f"{error_details}\n\n"
                                )
                                
                                # Check if user has cp3 models that might be causing confusion
                                script_dir = PROJECT_ROOT
                                local_models_dir = script_dir / 'models'
                                has_cp3 = (local_models_dir / 'bact_phase_cp3').exists() if local_models_dir.exists() else False
                                
                                if has_cp3 and '_omni' in current_model_type:
                                    help_text += (
                                        f"⚠️ IMPORTANT: You have 'bact_phase_cp3' in your models directory, "
                                        f"but this is a Cellpose model, NOT an Omnipose model.\n"
                                        f"They have different architectures and cannot be used interchangeably.\n\n"
                                    )
                                
                                help_text += (
                                    f"Please try:\n"
                                    f"1. Wait for the model server to come back online (currently HTTP 500 error)\n"
                                    f"2. Try a different model type\n"
                                    f"3. If you have cp3 models, use them with Cellpose model types, not Omnipose\n"
                                    f"4. Delete corrupted models: rm -rf {MODEL_DIR}/*.corrupted\n"
                                    f"5. Restart the application"
                                )
                                
                                QMessageBox.critical(
                                    preview_dialog, 
                                    "Model Architecture Mismatch",
                                    help_text
                                )
                                status_label.setText("Model load failed")
                                return False
                        else:
                            QMessageBox.critical(preview_dialog, "Model Error", 
                                               f"Could not load model:\n{error_msg}")
                            status_label.setText("Model load failed")
                        return False
                    except HTTPError as http_err:
                        # Handle HTTP errors (network issues when downloading models)
                        error_msg = str(http_err)
                        
                        # Special handling for 404 errors (model not found)
                        if http_err.code == 404:
                            model_file = find_model_file(current_model_type, 0, True)
                            if model_file and os.path.exists(model_file):
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Using cached model (not in repo)...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_key = current_key
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Model Not Available",
                                        f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                        f"Available Omnipose models:\n"
                                        f"  • bact_phase_omni (recommended for bacteria)\n"
                                        f"  • bact_fluor_omni\n"
                                        f"  • bact_phase_affinity\n\n"
                                        f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                        f"Please select a different model from the dropdown."
                                    )
                                    status_label.setText("Model not available")
                                    return False
                            else:
                                # Model not available and not cached
                                QMessageBox.warning(
                                    preview_dialog, 
                                    "Model Not Available",
                                    f"Model '{current_model_type}' is not available in the repository (404 error).\n\n"
                                    f"Available Omnipose models:\n"
                                    f"  • bact_phase_omni (recommended for bacteria)\n"
                                    f"  • bact_fluor_omni\n"
                                    f"  • bact_phase_affinity\n\n"
                                    f"See: https://omnipose.readthedocs.io/models.html\n\n"
                                    f"Please select a different model from the dropdown."
                                )
                                status_label.setText("Model not available")
                                return False
                        
                        # Check for model in local directory or cache without triggering another download
                        model_file = find_model_file(current_model_type, 0, True)
                        cached_model_exists = (model_file is not None and os.path.exists(model_file))
                        
                        if cached_model_exists:
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Trying cached model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_key = current_key
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Network Error & Cache Issue",
                                        f"Could not download model (server error) and cached model also failed:\n\n"
                                        f"Download error: {error_msg}\n"
                                        f"Cache error: {cache_error}\n\n"
                                        f"Please try:\n"
                                        f"1. Wait a few minutes and try again (server may be temporarily down)\n"
                                        f"2. Try a different model type\n"
                                        f"3. Check your internet connection\n"
                                        f"4. If persistent, clear cache: rm -rf ~/.cellpose/models/"
                                    )
                                    status_label.setText("Network & cache error")
                                    return False
                        else:
                            # No cached model available
                            QMessageBox.warning(
                                preview_dialog, 
                                "Network Error - Model Server Unavailable",
                                f"Could not download model due to network error:\n\n{error_msg}\n\n"
                                f"The model server appears to be temporarily unavailable (HTTP 500).\n\n"
                                f"Please try:\n"
                                f"1. Wait 5-10 minutes and try again (server may be temporarily down)\n"
                                f"2. Check your internet connection\n"
                                f"3. Try a different model type (may already be cached)\n"
                                f"4. Check the Cellpose website status: https://www.cellpose.org\n\n"
                                f"Note: Once downloaded, models are cached locally and won't need to be re-downloaded."
                            )
                            status_label.setText("Server unavailable - try again later")
                            return False
                    except Exception as e:
                        error_msg = str(e)
                        # Handle other HTTP/network errors that might be wrapped
                        if "HTTP" in error_msg or "500" in error_msg or "connection" in error_msg.lower() or "network" in error_msg.lower() or "urllib" in error_msg.lower() or "download" in error_msg.lower():
                            # Check for model in local directory or cache without triggering another download
                            model_file = find_model_file(current_model_type, 0, True)
                            cached_model_exists = (model_file is not None)
                            
                            if cached_model_exists:
                                # Try to use cached model even if download failed
                                try:
                                    status_label.setText("Trying cached model...")
                                    QApplication.processEvents()
                                    preview_dialog.model = models.Cellpose(
                                        gpu=current_use_gpu, 
                                        model_type=current_model_type, 
                                        omni=True
                                    )
                                    preview_dialog._last_model_key = current_key
                                    preview_dialog._last_use_gpu = current_use_gpu
                                    status_label.setText("Using cached model")
                                    QApplication.processEvents()
                                    return True  # Success with cached model
                                except Exception as cache_error:
                                    # Cached model also failed
                                    QMessageBox.warning(
                                        preview_dialog, 
                                        "Network Error & Cache Issue",
                                        f"Could not download model (server error) and cached model also failed:\n\n"
                                        f"Download error: {error_msg}\n"
                                        f"Cache error: {cache_error}\n\n"
                                        f"Please try:\n"
                                        f"1. Wait a few minutes and try again (server may be temporarily down)\n"
                                        f"2. Try a different model type\n"
                                        f"3. Check your internet connection\n"
                                        f"4. If persistent, clear cache: rm -rf ~/.cellpose/models/"
                                    )
                                    status_label.setText("Network & cache error")
                                    return False
                            else:
                                # No cached model available
                                QMessageBox.warning(
                                    preview_dialog, 
                                    "Network Error - Model Server Unavailable",
                                    f"Could not download model due to network error:\n\n{error_msg}\n\n"
                                    f"The model server appears to be temporarily unavailable (HTTP 500).\n\n"
                                    f"Please try:\n"
                                    f"1. Wait 5-10 minutes and try again (server may be temporarily down)\n"
                                    f"2. Check your internet connection\n"
                                    f"3. Try a different model type (may already be cached)\n"
                                    f"4. Check the Cellpose website status: https://www.cellpose.org\n\n"
                                    f"Note: Once downloaded, models are cached locally and won't need to be re-downloaded."
                                )
                                status_label.setText("Server unavailable - try again later")
                                return False
                        else:
                            QMessageBox.critical(preview_dialog, "Model Error", 
                                               f"Could not load model:\n{error_msg}")
                            status_label.setText("Model load failed")
                        return False
                return True
            
            def get_current_parameters():
                """Get current parameters from main window (works because dialog is non-modal)."""
                import sys
                is_frozen = getattr(sys, "frozen", False) or hasattr(sys, "__compiled__")
                diameter_value = self._get_effective_diameter_pixels()
                display = self.model_combo.currentText()
                model_type, pretrained_path = self._resolve_model_selection(display)
                # Built-in: only default to 30 when frozen (frozen build workaround).
                if diameter_value == 0 and is_frozen and not pretrained_path:
                    default_diameters = {
                        "bact_phase_omni": 30.0,
                        "bact_omni": 30.0,
                        "bact_fluor_omni": 30.0,
                    }
                    diameter_value = default_diameters.get(model_type or display, 30.0)
                flow_val = float(self.flow_threshold.value())
                return {
                    "diameter": diameter_value if diameter_value and diameter_value > 0 else 0,
                    "flow_threshold": flow_val,
                    "mask_threshold": self._get_effective_mask_threshold(),
                    "cluster": self.cluster.isChecked(),
                    "model_type": model_type,
                    "pretrained_model": str(pretrained_path) if pretrained_path else None,
                    "use_gpu": self.use_gpu.isChecked(),
                }
            
            # Cache for segmentation results: key = (frame_idx, diameter, flow_threshold, mask_threshold, cluster, model_type, use_gpu)
            preview_dialog.segmentation_cache = {}
            preview_dialog.display_cache_key = None  # None = show result for main-window params; else show this cached run

            # Round floats for display/cache to avoid binary representation noise (e.g. 0.3 -> 0.30000000000004)
            _PARAM_DECIMALS = 4  # cache key; display uses 2 decimals in format strings

            def format_omnipose_params_from_key(key):
                """Short param string from cache key for display/combo."""
                if not key or len(key) < 7:
                    return "?"
                _, diameter, flow_threshold, mask_threshold, cluster, model_str, _ = key
                model_short = Path(model_str).stem if model_str and getattr(Path(model_str), "suffix", "") else (model_str or "?")
                return f"d={diameter:.2f} flow={flow_threshold:.2f} mask={mask_threshold:.2f} cluster={'on' if cluster else 'off'} {model_short}"

            def format_omnipose_params_from_params(params):
                """Short param string from params dict for display."""
                d = params.get("diameter")
                d = 0.0 if d is None or not isinstance(d, (int, float)) else float(d)
                f = params.get("flow_threshold", 0.4)
                f = 0.4 if f is None or not isinstance(f, (int, float)) else float(f)
                m = params.get("mask_threshold", 0.0)
                m = 0.0 if m is None or not isinstance(m, (int, float)) else float(m)
                cluster = bool(params.get("cluster", False))
                model_str = params.get("model_type") or params.get("pretrained_model") or "bact_phase_omni"
                model_short = Path(model_str).stem if model_str and getattr(Path(model_str), "suffix", "") else (model_str or "?")
                return f"d={d:.2f} flow={f:.2f} mask={m:.2f} cluster={'on' if cluster else 'off'} {model_short}"

            def get_cache_key(frame_idx, params):
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
                
                return (frame_idx, diameter, flow_threshold, mask_threshold,
                       bool(params.get("cluster", False)),
                       str(params.get("model_type") or params.get("pretrained_model") or "bact_phase_omni"),
                       bool(params.get("use_gpu", True)))
            
            # Segmentation worker thread class
            class SegmentationWorker(QThread):
                finished = pyqtSignal(object, object)  # mask, error
                
                def __init__(self, model, frame, diameter, flow_threshold, mask_threshold, cluster):
                    super().__init__()
                    self.model = model
                    self.frame = frame
                    self.diameter = diameter
                    self.flow_threshold = flow_threshold
                    self.mask_threshold = mask_threshold
                    self.cluster = cluster
                
                def run(self):
                    try:
                        # CRITICAL: Ensure all parameters are valid numbers, not None
                        # This prevents comparison errors in model.eval() which expects numbers
                        diameter = self.diameter
                        if diameter is None or not isinstance(diameter, (int, float)):
                            diameter = 0
                        diameter = float(diameter)
                        
                        flow_threshold = self.flow_threshold
                        if flow_threshold is None or not isinstance(flow_threshold, (int, float)):
                            flow_threshold = 0.4
                        flow_threshold = float(flow_threshold)
                        mask_threshold = self.mask_threshold
                        if mask_threshold is None or not isinstance(mask_threshold, (int, float)):
                            mask_threshold = 0.0
                        mask_threshold = float(mask_threshold)
                        
                        mask, flows, styles, diams = self.model.eval(
                            self.frame,
                            channels=[0, 0],
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            mask_threshold=mask_threshold,
                            cellprob_threshold=0.0,
                            do_3D=False,
                            omni=True,
                            cluster=self.cluster,
                            net_avg=False,  # Faster, omni models don't support net_avg anyway
                            resample=True,
                            verbose=False
                        )
                        self.finished.emit(mask, None)
                    except Exception as e:
                        self.finished.emit(None, str(e))
            
            def run_segmentation(frame_idx, use_cache=True):
                """Run segmentation on a specific frame with caching and background threading"""
                # Get current parameters
                params = get_current_parameters()
                
                # CRITICAL: Clean parameters IMMEDIATELY - ensure no None values anywhere
                # This must happen before cache key generation or any other operations
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
                
                # Now generate cache key with clean parameters
                cache_key = get_cache_key(frame_idx, params)
                
                # Optional disk cache (session): same key as in-memory, stored under temp dir
                import hashlib
                import tempfile
                _cache_dir = Path(tempfile.gettempdir()) / "bactimate_preview_cache"
                _disk_key = hashlib.sha256(f"{file_path}|{cache_key}".encode()).hexdigest()[:24]
                _disk_path = _cache_dir / f"{_disk_key}.npy"
                
                # Check cache first (safely - handle any old cache entries with None values)
                if use_cache:
                    try:
                        if cache_key in preview_dialog.segmentation_cache:
                            return preview_dialog.segmentation_cache[cache_key]
                        # Check disk cache
                        if _disk_path.exists():
                            try:
                                mask_disk = np.load(_disk_path)
                                preview_dialog.segmentation_cache[cache_key] = mask_disk
                                return mask_disk
                            except Exception:
                                pass
                    except (TypeError, ValueError) as e:
                        # Cache key comparison failed (likely old entries with None values)
                        # Clean cache and continue
                        preview_dialog.segmentation_cache.clear()
                
                # Load model if needed (only when model type or GPU setting changes)
                if not load_model():
                    return None
                
                frame = images[frame_idx]
                # Handle multi-channel: use first channel if needed
                if frame.ndim == 3:
                    frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                
                # Ensure frame is float32 for MPS compatibility (Apple Silicon GPU doesn't support float64)
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32)
                
                # Downsample for fast preview segmentation (model runs on smaller image; mask upscaled after)
                seg_frame, seg_scale, (orig_h, orig_w) = downsample_for_preview_seg(frame, max_size=PREVIEW_SEG_MAX_SIZE)
                if seg_scale < 1.0:
                    get_logger().info(
                        "Preview: downsampled frame for speed: %s -> %s (scale=%.3f)",
                        (orig_h, orig_w), getattr(seg_frame, "shape", None), seg_scale,
                    )
                
                # Extract parameters (already validated and converted above)
                diameter = params['diameter']
                flow_threshold = params['flow_threshold']
                mask_threshold = params['mask_threshold']
                cluster = params['cluster']
                # When diameter is 0 (GUI "auto"), use model default so we don't pass 0 or clamp to 1
                default_diameters_preview = {"bact_phase_omni": 30.0, "bact_omni": 30.0, "bact_fluor_omni": 30.0}
                effective_diameter = (diameter if (diameter is not None and diameter > 0) else
                                     default_diameters_preview.get(params.get("model_type") or "", 30.0))
                # Diameter in downscaled image pixels (so Omnipose uses correct relative cell size)
                diameter_seg = effective_diameter * seg_scale
                if diameter_seg < 1:
                    diameter_seg = max(1.0, diameter_seg)
                
                try:
                    # Final validation before calling model.eval()
                    # Double-check all parameters are valid numbers (defensive programming)
                    assert isinstance(diameter, (int, float)) and diameter is not None, f"Invalid diameter: {diameter}"
                    assert isinstance(flow_threshold, (int, float)) and flow_threshold is not None, f"Invalid flow_threshold: {flow_threshold}"
                    assert isinstance(mask_threshold, (int, float)) and mask_threshold is not None, f"Invalid mask_threshold: {mask_threshold}"
                    
                    # Run segmentation (synchronously on main thread - if this hangs, GUI freezes)
                    # Log so user can run from terminal and see progress (same code path for built-in and user models)
                    kind = "USER_MODEL" if params.get("pretrained_model") else "BUILTIN"
                    get_logger().info(
                        "Preview: running model.eval() (%s) frame_shape=%s diameter=%s flow_thr=%s",
                        kind, getattr(seg_frame, "shape", None), diameter_seg, flow_threshold,
                    )
                    import sys
                    is_frozen = getattr(sys, 'frozen', False) or hasattr(sys, '__compiled__')
                    
                    if is_frozen:
                        # In frozen builds, wrap the call to handle the omnipose bug
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                mask, flows, styles, diams = preview_dialog.model.eval(
                                    seg_frame,
                                    channels=[0, 0],
                                    diameter=diameter_seg if (diameter_seg and diameter_seg > 0) else 30.0,  # Never use 0 in frozen builds
                                    flow_threshold=flow_threshold,
                                    mask_threshold=mask_threshold,
                                    cellprob_threshold=0.0,
                                    do_3D=False,
                                    omni=True,
                                    cluster=cluster,
                                    net_avg=False,
                                    resample=False,  # Faster preview; full run uses resample=True
                                    verbose=False
                                )
                                break  # Success, exit retry loop
                            except TypeError as e:
                                if "'>' not supported between instances of 'NoneType'" in str(e) or "'<' not supported between instances of 'NoneType'" in str(e):
                                    if attempt < max_retries - 1:
                                        # Try with a different diameter
                                        diameter = 30.0 + (attempt * 5.0)  # Try 30, 35, 40
                                        diameter_seg = (diameter * seg_scale) if seg_scale < 1.0 else diameter
                                        continue
                                    else:
                                        # Last attempt failed, raise the error
                                        raise
                                else:
                                    # Different TypeError, raise immediately
                                    raise
                    else:
                        # Not frozen build, use normal call
                        mask, flows, styles, diams = preview_dialog.model.eval(
                            seg_frame,
                            channels=[0, 0],
                            diameter=diameter_seg,
                            flow_threshold=flow_threshold,
                            mask_threshold=mask_threshold,
                            cellprob_threshold=0.0,
                            do_3D=False,
                            omni=True,
                            cluster=cluster,
                            net_avg=False,
                            resample=False,  # Faster preview; full run uses resample=True
                            verbose=False
                        )
                    get_logger().info("Preview: model.eval() finished (mask shape %s)", getattr(mask, "shape", None))
                    # Upscale mask to original frame size if we ran on downsampled image
                    if seg_scale < 1.0 and mask is not None:
                        mask = upsample_mask_to_shape(mask, orig_h, orig_w)
                    # Debug: log eval result for user vs built-in (see Logs/omnipose_debug.log)
                    try:
                        kind = "USER_MODEL" if params.get("pretrained_model") else "BUILTIN"
                        model_id = params.get("pretrained_model") or params.get("model_type") or "?"
                        n_cells = len(np.unique(mask)) - 1
                        append_omnipose_debug(
                            f"[BactiMate eval] kind={kind} model={model_id!r} "
                            f"diameter={diameter} flow_thr={flow_threshold} mask_thr={mask_threshold} "
                            f"frame_shape={getattr(frame, 'shape', None)} channels=[0,0] -> "
                            f"n_cells={n_cells} mask_min={int(mask.min())} mask_max={int(mask.max())}"
                        )
                        # When 0 cells: log distance map range so user knows required mask_threshold.
                        # The *returned* flows from model.eval() may not expose the same channel Omnipose uses
                        # internally (get_masks uses _d). Omnipose logs the real range as
                        # "[omnipose no cell pixels] dist min=... max=...". If we read all zeros, tell the user to use that line.
                        if n_cells == 0 and flows is not None:
                            try:
                                farr = np.asarray(flows) if not isinstance(flows, (list, tuple)) else flows
                                n_ch = len(farr) if isinstance(farr, (list, tuple)) else (int(farr.shape[0]) if hasattr(farr, "shape") and getattr(farr, "ndim", 0) >= 3 else 1)
                                best_min, best_max, best_mean = None, None, None
                                best_idx = -1
                                for idx in range(n_ch):
                                    if isinstance(farr, (list, tuple)):
                                        ch = np.asarray(farr[idx])
                                    elif hasattr(farr, "shape") and farr.ndim >= 3 and idx < farr.shape[0]:
                                        ch = np.asarray(farr[idx])
                                    else:
                                        ch = np.asarray(farr).ravel()
                                    if ch.size > 0 and np.issubdtype(ch.dtype, np.floating):
                                        c_min, c_max, c_mean = float(np.nanmin(ch)), float(np.nanmax(ch)), float(np.nanmean(ch))
                                        # Prefer a channel that is not all zeros (so we can suggest a real threshold)
                                        if best_max is None or (best_min == best_max == 0 and (c_min != 0 or c_max != 0 or c_min != c_max)):
                                            best_min, best_max, best_mean, best_idx = c_min, c_max, c_mean, idx
                                if best_max is not None and (best_min != 0 or best_max != 0):
                                    append_omnipose_debug(
                                        f"[BactiMate 0 cells] distance map (channel {best_idx}): min={best_min:.4f} max={best_max:.4f} mean={best_mean:.4f} "
                                        f"-> need mask_threshold < {best_max:.2f} (and try flow_threshold=0)"
                                    )
                                else:
                                    append_omnipose_debug(
                                        "[BactiMate 0 cells] The returned 'flows' do not expose the distance channel (all zeros). "
                                        "Use the '[omnipose no cell pixels] dist min=... max=...' line in this log for the real distance range — set mask_threshold below that max (e.g. if max=-2.3, try mask_threshold=-3 or -5). "
                                        "If dist is constant (min=max) and you still get 0 cells after lowering the threshold, the flow field may be flat; try flow_threshold=0."
                                    )
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Cache the result (memory and optional disk)
                    preview_dialog.segmentation_cache[cache_key] = mask
                    try:
                        _cache_dir.mkdir(parents=True, exist_ok=True)
                        np.save(_disk_path, mask)
                    except Exception:
                        pass
                    return mask
                except RecursionError as e:
                    # Handle recursion errors (e.g., from broken runtime hooks)
                    import traceback
                    error_details = f"Recursion error detected:\n{str(e)}\n\n"
                    error_details += f"This may be caused by a bug in the frozen build.\n"
                    error_details += f"Try rebuilding the application.\n\n"
                    error_details += f"Parameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\n"
                    error_details += f"Traceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error - Recursion", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
                except TypeError as e:
                    import traceback
                    error_details = f"TypeError in omnipose: {str(e)}\n\n"
                    error_details += f"Traceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
                except AssertionError as e:
                    # Parameter validation error
                    import traceback
                    error_details = f"Parameter validation error:\n{str(e)}\n\nParameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\nTraceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
                except Exception as e:
                    # Other errors - include full traceback for debugging
                    import traceback
                    error_details = f"{str(e)}\n\nParameters:\n  diameter: {diameter} (type: {type(diameter)})\n  flow_threshold: {flow_threshold} (type: {type(flow_threshold)})\n\nTraceback:\n{traceback.format_exc()}"
                    QMessageBox.critical(preview_dialog, "Segmentation Error", 
                                       f"Could not segment frame:\n{error_details}")
                    return None
            
            def update_display(frame_idx, show_progress=True, force_recalc=False):
                """Update the display with segmentation overlay"""
                # Check cache first
                params = get_current_parameters()
                cache_key = get_cache_key(frame_idx, params)
                has_cache = cache_key in preview_dialog.segmentation_cache and not force_recalc
                
                # Show progress dialog if we need to calculate
                progress_dialog = None
                if show_progress and not has_cache:
                    progress_dialog = QProgressDialog("Segmenting frame...", None, 0, 0, preview_dialog)
                    progress_dialog.setWindowTitle("Processing")
                    progress_dialog.setAutoClose(False)
                    progress_dialog.setAutoReset(False)
                    progress_dialog.setCancelButton(None)  # Don't allow cancellation
                    progress_dialog.show()
                    QApplication.processEvents()
                
                # Disable controls during processing
                if progress_dialog:
                    resegment_btn.setEnabled(False)
                    clear_cache_btn.setEnabled(False)
                    frame_spin.setEnabled(False)
                
                try:
                    frame = images[frame_idx]
                    # Handle multi-channel: use first channel if needed
                    if frame.ndim == 3:
                        display_frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                    else:
                        display_frame = frame
                    
                    # Update progress message
                    if progress_dialog:
                        progress_dialog.setLabelText("Running segmentation...")
                        QApplication.processEvents()
                    
                    # Run segmentation (with caching)
                    mask = run_segmentation(frame_idx, use_cache=not force_recalc)
                    if mask is None:
                        if progress_dialog:
                            progress_dialog.close()
                        status_label.setText("Segmentation failed")
                        status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
                        return
                    
                    preview_dialog.masks = mask
                    preview_dialog.current_display_frame = display_frame
                    # Reset LUT range when frame changes so each frame gets its own range initially
                    if lut_data.get('frame_idx') != frame_idx:
                        lut_data['frame_idx'] = frame_idx
                        fmin, fmax = float(display_frame.min()), float(display_frame.max())
                        lut_data['vmin'] = fmin
                        lut_data['vmax'] = fmax if fmax > fmin else fmin + 1.0
                        lut_data['frame_min'] = fmin
                        lut_data['frame_max'] = fmax
                    vmin = lut_data.get('vmin')
                    vmax = lut_data.get('vmax')
                    if vmin is None or vmax is None or vmax <= vmin:
                        vmin = float(display_frame.min())
                        vmax = float(display_frame.max()) if display_frame.max() > vmin else vmin + 1.0
                        lut_data['vmin'], lut_data['vmax'] = vmin, vmax
                    vrange = max(vmax - vmin, 1e-6)
                    display_uint8 = ((np.asarray(display_frame).astype(np.float64) - vmin) / vrange * 255).clip(0, 255).astype(np.uint8)
                    # Scale down for preview so overlay + pixmap + zoom/pan paint stay fast (cap 1024)
                    disp_scaled, mask_scaled = scale_for_preview(
                        display_uint8, mask, max_size=PREVIEW_SEG_MAX_SIZE
                    )
                    overlay = build_segmentation_overlay(disp_scaled, mask_scaled, alpha=0.5)
                    pixmap = to_qpixmap(overlay)
                    height, width = overlay.shape[:2]
                    image_canvas.set_pixmap(pixmap, (width, height), reset_view=False)
                    QApplication.processEvents()

                    preview_dialog.display_cache_key = None  # showing current-params result
                    file_name = Path(file_path).name
                    num_cells = len(np.unique(mask)) - 1
                    params = get_current_parameters()
                    model_name = Path(params["pretrained_model"]).stem if params.get("pretrained_model") else params.get("model_type", "—")
                    params_str = format_omnipose_params_from_params(params)
                    title_label.setText(
                        f"Frame {frame_idx + 1} of {num_frames} | Cells: {num_cells} | Model: {model_name} | {file_name}\n"
                        f"Parameters: {params_str}"
                    )
                    update_histogram(display_frame, lut_data.get('vmin'), lut_data.get('vmax'))
                    
                    # Close progress dialog
                    if progress_dialog:
                        progress_dialog.close()
                    
                    # Re-enable controls
                    resegment_btn.setEnabled(True)
                    clear_cache_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    
                    # Update status
                    cache_size = len(preview_dialog.segmentation_cache)
                    if num_cells == 0:
                        params = get_current_parameters()
                        if params.get("pretrained_model"):
                            status_label.setText("No cells detected. Try Flow 0, Mask -2, Diameter 0 or cell size (px). See docs/CUSTOM_MODEL_TROUBLESHOOTING.md.")
                            status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                        else:
                            if has_cache:
                                status_label.setText("Ready (from cache) — no cells")
                            else:
                                status_label.setText(f"Ready (cached: {cache_size}) — no cells")
                            status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
                    else:
                        if has_cache:
                            status_label.setText("Ready (from cache)")
                        else:
                            status_label.setText(f"Ready (cached: {cache_size})")
                        status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
                    refresh_cached_runs_combo()
                except Exception as e:
                    if progress_dialog:
                        progress_dialog.close()
                    # Re-enable controls on error
                    resegment_btn.setEnabled(True)
                    clear_cache_btn.setEnabled(True)
                    frame_spin.setEnabled(True)
                    status_label.setText(f"Error: {str(e)}")
                    status_label.setStyleSheet("color: #FF0000; font-weight: bold;")
                    QMessageBox.warning(preview_dialog, "Preview Error", f"Could not update display:\n{str(e)}")

            def show_cached_run(cache_key):
                """Display a cached segmentation (mask + params) without running model. Used when user picks a cached run from combo."""
                if cache_key not in preview_dialog.segmentation_cache:
                    return
                frame_idx = cache_key[0]
                mask = preview_dialog.segmentation_cache[cache_key]
                frame = images[frame_idx]
                if frame.ndim == 3:
                    display_frame = frame[0] if frame.shape[0] < 10 else frame[:, :, 0]
                else:
                    display_frame = frame
                disp_scaled, mask_scaled = scale_for_preview(
                    display_frame, mask, max_size=PREVIEW_SEG_MAX_SIZE
                )
                overlay = build_segmentation_overlay(disp_scaled, mask_scaled, alpha=0.5)
                pixmap = to_qpixmap(overlay)
                h, w = overlay.shape[:2]
                image_canvas.set_pixmap(pixmap, (w, h), reset_view=False)
                num_cells = len(np.unique(mask)) - 1
                file_name = Path(file_path).name
                params_str = format_omnipose_params_from_key(cache_key)
                model_short = cache_key[5] if len(cache_key) > 5 else "—"
                if model_short and getattr(Path(model_short), "suffix", None):
                    model_short = Path(model_short).stem
                title_label.setText(
                    f"Frame {frame_idx + 1} of {num_frames} | Cells: {num_cells} | Model: {model_short} | {file_name}\n"
                    f"Parameters: {params_str}"
                )

            # Frame navigation (spinbox only; no prev/next buttons)
            nav_layout = QHBoxLayout()
            frame_spin = QSpinBox()
            frame_spin.setRange(1, num_frames)
            frame_spin.setValue(1)
            nav_layout.addWidget(QLabel("Frame:"))
            nav_layout.addWidget(frame_spin)
            nav_layout.addWidget(QLabel(f"of {num_frames}"))
            scale_label = QLabel("")
            nav_layout.addWidget(scale_label)
            # Add status label to navigation layout
            nav_layout.addStretch()
            nav_layout.addWidget(status_label)
            
            # Track if we're currently updating to prevent concurrent updates
            is_updating_frame = False
            
            def go_to_frame(idx, force_recalc=False):
                nonlocal is_updating_frame
                # Prevent concurrent frame updates
                if is_updating_frame:
                    return
                
                if not (0 <= idx < num_frames):
                    # Invalid frame index - reset to current frame
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
                    return
                
                is_updating_frame = True
                preview_dialog.current_frame = idx
                preview_dialog.display_cache_key = None  # when changing frame, show current params for new frame
                # Block signals to prevent recursive calls
                frame_spin.blockSignals(True)
                frame_spin.setValue(idx + 1)
                frame_spin.blockSignals(False)
                
                try:
                    # Check if we have cached result for this frame with current parameters
                    params = get_current_parameters()
                    cache_key = get_cache_key(idx, params)
                    has_cache = cache_key in preview_dialog.segmentation_cache
                    
                    # Only show progress if we need to recalculate
                    update_display(idx, show_progress=(not has_cache or force_recalc))
                    refresh_cached_runs_combo()
                except Exception as e:
                    import traceback
                    error_msg = f"Could not display frame {idx + 1}:\n{str(e)}"
                    if hasattr(e, '__traceback__'):
                        error_msg += f"\n\n{traceback.format_exc()}"
                    QMessageBox.warning(preview_dialog, "Preview Error", error_msg)
                finally:
                    is_updating_frame = False
            
            # editingFinished: when user presses Enter or leaves field. valueChanged + 300ms debounce: for arrow keys.
            def on_frame_changed():
                if hasattr(frame_spin, '_change_timer'):
                    frame_spin._change_timer.stop()
                frame_num = frame_spin.value() - 1
                if 0 <= frame_num < num_frames:
                    go_to_frame(frame_num)
                else:
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)

            frame_spin.editingFinished.connect(on_frame_changed)
            frame_spin._change_timer = QTimer()
            frame_spin._change_timer.setSingleShot(True)
            frame_spin._change_timer.timeout.connect(on_frame_changed)

            def on_value_changed(v):
                if 1 <= v <= num_frames:
                    frame_spin._change_timer.start(300)
                else:
                    frame_spin._change_timer.stop()
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(preview_dialog.current_frame + 1)
                    frame_spin.blockSignals(False)
            frame_spin.valueChanged.connect(on_value_changed)
            
            nav_layout.addStretch()
            
            # Re-segment button (to update with current parameters from main window)
            resegment_btn = QPushButton("Re-segment")
            resegment_btn.setToolTip(
                "Re-run segmentation on the current frame using the current parameters from the main window.\n"
                "You can change parameters in the main window (it stays accessible) and click this button to see the effect.\n"
                "Results are cached - changing parameters and clicking again will use cached results if parameters match."
            )
            def on_resegment():
                # Clear cache for current frame to force recalculation
                params = get_current_parameters()
                cache_key = get_cache_key(preview_dialog.current_frame, params)
                if cache_key in preview_dialog.segmentation_cache:
                    del preview_dialog.segmentation_cache[cache_key]
                # Force recalculation with progress
                update_display(preview_dialog.current_frame, show_progress=True, force_recalc=True)
            resegment_btn.clicked.connect(on_resegment)
            nav_layout.addWidget(resegment_btn)
            
            # Clear cache button
            clear_cache_btn = QPushButton("Clear Cache")
            clear_cache_btn.setToolTip(
                "Clear all cached segmentation results.\n"
                "Use this if you want to force recalculation of all frames."
            )
            def on_clear_cache():
                preview_dialog.segmentation_cache.clear()
                preview_dialog.display_cache_key = None
                status_label.setText("Cache cleared")
                status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                QTimer.singleShot(2000, lambda: status_label.setText("Ready") or status_label.setStyleSheet("color: #00FF00; font-weight: bold;"))
                refresh_cached_runs_combo()
            clear_cache_btn.clicked.connect(on_clear_cache)
            nav_layout.addWidget(clear_cache_btn)

            # Cached run switcher: show params for current view, switch between cached segmentations for this frame
            nav_layout.addWidget(QLabel("Cached run:"))
            cached_runs_combo = QComboBox()
            cached_runs_combo.setToolTip(
                "Current = main window parameters. Other entries = previously computed segmentations for this frame; select to view and compare parameters."
            )
            cached_runs_combo.setMinimumWidth(280)

            def refresh_cached_runs_combo():
                cur_frame = preview_dialog.current_frame
                keys_for_frame = [k for k in preview_dialog.segmentation_cache if k[0] == cur_frame]
                keys_for_frame.sort(key=lambda k: (k[1], k[2], k[3]))  # diameter, flow, mask
                cached_runs_combo.blockSignals(True)
                cached_runs_combo.clear()
                cached_runs_combo.addItem("Current (main window)", None)
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
                    update_display(preview_dialog.current_frame, show_progress=False)
                else:
                    preview_dialog.display_cache_key = key
                    show_cached_run(key)

            cached_runs_combo.currentIndexChanged.connect(on_cached_run_selected)
            nav_layout.addWidget(cached_runs_combo)
            
            layout.addLayout(nav_layout)

            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)

            close_shortcut = QShortcut(QKeySequence.StandardKey.Close, preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), preview_dialog)  # Cmd+M on macOS
            minimize_shortcut.activated.connect(preview_dialog.showMinimized)

            preview_dialog.setLayout(layout)

            # Show first frame immediately (image only), scaled for fast display and zoom/pan
            frame0 = images[0]
            if frame0.ndim == 3:
                display_frame0 = frame0[0] if frame0.shape[0] < 10 else frame0[:, :, 0]
            else:
                display_frame0 = frame0
            empty_mask = np.zeros(display_frame0.shape, dtype=np.int32)
            disp0, _ = scale_for_preview(display_frame0, empty_mask, max_size=PREVIEW_SEG_MAX_SIZE)
            empty_scaled = np.zeros(disp0.shape, dtype=np.int32)
            initial_overlay = build_segmentation_overlay(disp0, empty_scaled, alpha=0.0)
            initial_pixmap = to_qpixmap(initial_overlay)
            h0, w0 = initial_overlay.shape[:2]
            image_canvas.set_pixmap(initial_pixmap, (w0, h0), reset_view=True)
            file_name = Path(file_path).name
            params0 = get_current_parameters()
            model_name0 = Path(params0["pretrained_model"]).stem if params0.get("pretrained_model") else params0.get("model_type", "—")
            params_str0 = format_omnipose_params_from_params(params0)
            title_label.setText(
                f"Frame 1 of {num_frames} | Segmenting... | Model: {model_name0} | {file_name}\n"
                f"Parameters: {params_str0}"
            )
            status_label.setText("Segmenting first frame...")
            loading_progress.close()
            preview_dialog.show()
            preview_dialog.raise_()
            preview_dialog.activateWindow()

            def start_preview():
                preview_dialog.raise_()
                preview_dialog.activateWindow()
                # Use show_progress=False for the initial frame to avoid a progress dialog on top
                update_display(0, show_progress=False)
            # Short delay so the preview window has time to paint before we run segmentation
            QTimer.singleShot(80, start_preview)

            # Show cache info in status
            def update_cache_status():
                cache_size = len(preview_dialog.segmentation_cache)
                if cache_size > 0:
                    status_label.setText(f"Ready (cached: {cache_size})")
                else:
                    status_label.setText("Ready")

            QTimer.singleShot(100, update_cache_status)
            
        except ImportError as e:
            loading_progress.close()
            error_msg = str(e)
            # Handle specific import errors
            if 'torch.testing' in error_msg:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  "Preview requires torch.testing module.\n\n"
                                  "This is a known issue with frozen builds.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try running from source: python main.py")
            else:
                QMessageBox.warning(self, "Preview Unavailable", 
                                  f"Preview requires additional libraries:\n\n{error_msg}\n\n"
                                  f"Install with: pip install cellpose-omni")
        except Exception as e:
            loading_progress.close()
            error_msg = str(e)
            # Handle PyTorch RpcBackendOptions duplicate initialization error
            if 'RpcBackendOptions' in error_msg or 'generic_type' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "PyTorch initialization error detected.\n\n"
                                  "This is a known issue with frozen builds and PyTorch.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try:\n"
                                  "1. Restart the application\n"
                                  "2. Run from source: python main.py\n"
                                  "3. Use the processing features instead of preview")
            # Handle pbr versioning error (mgen package version detection issue)
            elif 'Versioning for this project requires' in error_msg or 'pbr.version.VersionInfo' in error_msg or 'mgen' in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "Package versioning error detected.\n\n"
                                  "This is a known issue with frozen builds and the mgen package.\n"
                                  "The preview feature may not work in the packaged application.\n\n"
                                  "Try:\n"
                                  "1. Restart the application\n"
                                  "2. Run from source: python main.py\n"
                                  "3. Use the processing features instead of preview\n\n"
                                  "This error is harmless and doesn't affect actual processing.")
            # Handle TorchScript source access error
            elif 'TorchScript requires source access' in error_msg or "Can't get source for" in error_msg:
                QMessageBox.warning(self, "Preview Error", 
                                  "TorchScript compilation error detected.\n\n"
                                  "This is a known issue with frozen builds and PyTorch's TorchScript.\n"
                                  "TorchScript requires access to original source files which aren't\n"
                                  "available in packaged applications.\n\n"
                                  "Try:\n"
                                  "1. Restart the application (the runtime hook should fix this)\n"
                                  "2. Run from source: python main.py\n"
                                  "3. Use the processing features instead of preview\n\n"
                                  "Note: This has been fixed in the latest build. Please rebuild the app.")
            # Handle Windows PyTorch DLL load failure (c10.dll / WinError 1114)
            elif ('WinError 1114' in error_msg or 'c10.dll' in error_msg or
                  'DLL initialization routine failed' in error_msg or 'dynamic link library' in error_msg.lower()):
                QMessageBox.warning(self, "Preview Error",
                                  "PyTorch failed to load on Windows (DLL error).\n\n"
                                  "To keep using the GPU:\n\n"
                                  "1. Install Microsoft Visual C++ Redistributable (64-bit, 2015–2022) from Microsoft.\n\n"
                                  "2. Reinstall PyTorch with the correct CUDA index for your driver.\n"
                                  "   Check version: nvidia-smi\n"
                                  "   Example for CUDA 12.8:\n"
                                  "   pip uninstall torch torchvision torchaudio -y\n"
                                  "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n\n"
                                  "   For 12.4 use cu124, for 11.8 use cu118. See INSTALL_AND_LAUNCH_windows.md.")
            else:
                QMessageBox.critical(self, "Preview Error", f"Could not preview segmentation:\n{error_msg}")
            import traceback
            traceback.print_exc()
        except BaseException as e:
            # PyQt6 is strict about exceptions in slots - catch ALL exceptions (including SystemExit, KeyboardInterrupt)
            # This outer try-except catches any exceptions not handled by inner handlers
            import traceback
            error_msg = f"Unexpected error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Preview Error", error_msg)
            # Don't re-raise - PyQt6 will abort if we do
        except BaseException as e:
            # Final catch-all for any exceptions that slip through
            get_logger().exception("Critical error in preview segmentation")
            import traceback
            error_msg = f"Critical error in preview segmentation:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            try:
                QMessageBox.critical(self, "Preview Error", error_msg)
            except Exception:
                get_logger().exception("Could not show preview error dialog")
                print(f"CRITICAL ERROR in preview_segmentation: {error_msg}", file=sys.stderr)
            # Don't re-raise - PyQt6 will abort if we do

    def _open_check_pipeline_dialog(self):
        """Open diagnostic dialog: check current model (or compare two), optional training dir, optional image. Can download built-in models if needed."""
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDialog,
            QFileDialog,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
        )
        from PyQt6.QtGui import QFont

        d = QDialog(self)
        d.setWindowTitle("Check Omnipose pipeline")
        d.setMinimumSize(620, 520)
        layout = QVBoxLayout()
        last_dir = getattr(self.parent_window, "get_last_dir_for_tab", lambda _: "")(LAST_DIR_SEGMENTATION) or str(Path.home())
        model_names = self._get_available_models()

        # Mode: single check vs compare two models
        download_cb = QCheckBox("Download built-in models if needed")
        download_cb.setChecked(True)
        download_cb.setToolTip("When a built-in model (e.g. bact_phase_omni) is selected, download it into the cache first so it can be checked or compared.")
        layout.addWidget(download_cb)

        compare_cb = QCheckBox("Compare two models")
        compare_cb.setToolTip("Run model check on two models (Model A and Model B) and show both reports. Use to compare user vs user or user vs built-in.")
        layout.addWidget(compare_cb)

        # Single: current model (read-only label; snapshot when dialog opens)
        single_group = QGroupBox("Single check (current tab model)")
        single_layout = QVBoxLayout()
        display = self.model_combo.currentText()
        model_type, user_path = self._resolve_model_selection(display)
        path = get_checkable_model_path(model_type, user_path, ensure_builtin_cached=False)
        model_label = QLabel(f"Current model: {display}")
        if path is None and (model_type or user_path):
            if model_type:
                model_label.setText(f"Current model: {display} (not cached — enable 'Download built-in…' and run, or run segmentation once)")
            else:
                model_label.setText(f"Current model: {display} (file not found)")
        model_label.setWordWrap(True)
        single_layout.addWidget(model_label)
        single_group.setLayout(single_layout)
        layout.addWidget(single_group)

        # Compare: Model A and Model B combos
        compare_group = QGroupBox("Compare two models")
        compare_layout = QHBoxLayout()
        compare_layout.addWidget(QLabel("Model A:"))
        combo_a = QComboBox()
        combo_a.addItems(model_names)
        compare_layout.addWidget(combo_a)
        compare_layout.addWidget(QLabel("Model B:"))
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

        # Optional: training directory
        train_group = QGroupBox("Training data (optional)")
        train_layout = QVBoxLayout()
        train_dir_edit = QLineEdit()
        train_dir_edit.setPlaceholderText("Path to folder with image/mask pairs")
        train_row = QHBoxLayout()
        train_row.addWidget(train_dir_edit)
        browse_train_btn = QPushButton("Browse…")
        browse_train_btn.clicked.connect(lambda: _browse_train())
        train_row.addWidget(browse_train_btn)
        train_layout.addLayout(train_row)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Image filter:"))
        img_filter_edit = QLineEdit()
        img_filter_edit.setPlaceholderText("e.g. _phase")
        img_filter_edit.setFixedWidth(100)
        filter_row.addWidget(img_filter_edit)
        filter_row.addWidget(QLabel("Mask filter:"))
        mask_filter_edit = QLineEdit()
        mask_filter_edit.setText("_mask")
        mask_filter_edit.setFixedWidth(100)
        filter_row.addWidget(mask_filter_edit)
        train_layout.addLayout(filter_row)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        def _browse_train():
            path = QFileDialog.getExistingDirectory(d, "Select training directory", last_dir)
            if path:
                train_dir_edit.setText(path)

        # Optional: image for inference check
        image_group = QGroupBox("Inference test image (optional)")
        image_layout = QHBoxLayout()
        image_edit = QLineEdit()
        image_edit.setPlaceholderText("Path to an image to test with current / Model A")
        image_layout.addWidget(image_edit)

        def browse_image():
            path, _ = QFileDialog.getOpenFileName(d, "Select image", last_dir, "Images (*.tif *.tiff *.png);;All (*)")
            if path:
                image_edit.setText(path)

        browse_image_btn = QPushButton("Browse…")
        browse_image_btn.clicked.connect(browse_image)
        image_layout.addWidget(browse_image_btn)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Report area
        report = QTextEdit()
        report.setReadOnly(True)
        report.setFont(QFont("Monospace", 9))
        report.setPlaceholderText("Click 'Run diagnostic'. Single: checks current tab model. Compare: checks Model A and Model B. Enable 'Download built-in…' to cache Omnipose models first.")
        layout.addWidget(report)

        # Run diagnostic in a worker thread so model loading cannot block or crash the GUI
        class DiagnosticWorker(QThread):
            result_ready = pyqtSignal(str)

            def __init__(self, tab, is_compare, name_a, name_b, do_download,
                         train_dir_str, img_filter, mask_filter, image_str, single_display):
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
                        type_a, path_a = self.tab._resolve_model_selection(self.name_a)
                        type_b, path_b = self.tab._resolve_model_selection(self.name_b)
                        try:
                            model_path_a = get_checkable_model_path(type_a, path_a, ensure_builtin_cached=self.do_download and bool(type_a))
                        except Exception as e:
                            model_path_a = None
                            lines.append(f"Model A ({self.name_a}): failed to resolve — {e}")
                            lines.append("")
                        try:
                            model_path_b = get_checkable_model_path(type_b, path_b, ensure_builtin_cached=self.do_download and bool(type_b))
                        except Exception as e:
                            model_path_b = None
                            lines.append(f"Model B ({self.name_b}): failed to resolve — {e}")
                            lines.append("")
                        if model_path_a is None and not any("Model A" in ln for ln in lines):
                            lines.append(f"Model A ({self.name_a}): no file (built-in not cached or user file missing).")
                            lines.append("")
                        elif model_path_a is not None:
                            try:
                                r = _check_model(Path(model_path_a))
                                lines.append(f"=== Model A: {self.name_a} ===")
                                lines.append(format_model_report(r))
                                lines.append("")
                            except Exception as e:
                                lines.append(f"Model A check error: {e}")
                                lines.append("")
                            import gc
                            gc.collect()
                        if model_path_b is None and not any("Model B" in ln for ln in lines):
                            lines.append(f"Model B ({self.name_b}): no file (built-in not cached or user file missing).")
                            lines.append("")
                        elif model_path_b is not None:
                            try:
                                r = _check_model(Path(model_path_b))
                                lines.append(f"=== Model B: {self.name_b} ===")
                                lines.append(format_model_report(r))
                                lines.append("")
                            except Exception as e:
                                lines.append(f"Model B check error: {e}")
                                lines.append("")
                        primary_model_path = model_path_a if model_path_a is not None else model_path_b
                    else:
                        model_type, user_path = self.tab._resolve_model_selection(self.single_display)
                        try:
                            primary_model_path = get_checkable_model_path(model_type, user_path, ensure_builtin_cached=self.do_download and bool(model_type))
                        except Exception as e:
                            primary_model_path = None
                            lines.append(f"Model resolve error: {e}")
                            lines.append("")
                        if primary_model_path is not None:
                            try:
                                r = _check_model(Path(primary_model_path))
                                lines.append(format_model_report(r))
                                lines.append("")
                            except Exception as e:
                                lines.append(f"Model check error: {e}")
                                lines.append("")
                        elif not any("Model resolve error" in ln for ln in lines):
                            lines.append("Model check skipped: no model file (user model missing or built-in not cached).")
                            lines.append("")
                    # Checkpoint comparison (weight summary + output-head verdict) for resolved model(s)
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
                            lines.append("")
                        except Exception as e:
                            lines.append(f"Checkpoint comparison error: {e}")
                            lines.append("")
                    if self.train_dir_str and Path(self.train_dir_str).is_dir():
                        try:
                            r = _check_training_data(
                                Path(self.train_dir_str),
                                img_filter=self.img_filter or "",
                                mask_filter=self.mask_filter or "_mask",
                                dry_run=True,
                            )
                            lines.append(format_training_report(r))
                            lines.append("")
                        except Exception as e:
                            lines.append(f"Training data check error: {e}")
                            lines.append("")
                    if self.image_str and Path(self.image_str).is_file():
                        if primary_model_path is None:
                            lines.append("Inference check skipped: no model file available.")
                        else:
                            try:
                                r = _check_inference(Path(primary_model_path), Path(self.image_str))
                                lines.append(format_inference_report(r))
                            except Exception as e:
                                lines.append(f"Inference check error: {e}")
                    elif self.image_str:
                        lines.append("Inference check skipped: image path is not a file.")
                        lines.append("")
                except Exception as e:
                    import traceback
                    lines.append(f"Diagnostic error: {e}")
                    lines.append("")
                    lines.append(traceback.format_exc())
                self.result_ready.emit("\n".join(lines))

        worker = [None]  # use list so closure can rebind

        def run_diagnostic():
            report.clear()
            report.append("Running diagnostic in background…")
            run_btn.setEnabled(False)
            w = DiagnosticWorker(
                self,
                compare_cb.isChecked(),
                combo_a.currentText(),
                combo_b.currentText(),
                download_cb.isChecked(),
                train_dir_edit.text().strip(),
                img_filter_edit.text().strip() or "",
                mask_filter_edit.text().strip() or "_mask",
                image_edit.text().strip(),
                self.model_combo.currentText(),
            )
            worker[0] = w

            def on_done(text):
                report.clear()
                report.setText(text)
                run_btn.setEnabled(True)

            w.result_ready.connect(on_done)
            w.finished.connect(lambda: run_btn.setEnabled(True))
            w.start()

        run_btn = QPushButton("Run diagnostic")
        run_btn.clicked.connect(run_diagnostic)
        layout.addWidget(run_btn)
        d.setLayout(layout)
        d.exec()

    def _get_effective_diameter_pixels(self) -> float:
        """Return diameter in pixels (the single Diameter field)."""
        return self.diameter.value()

    def _get_run_diameter_and_flow(self) -> tuple[float, float]:
        """
        Return (diameter, flow_threshold) to pass to the batch script when running segmentation.
        Uses GUI values as-is; no overrides for custom models (Omnipose defaults/behavior unchanged).
        """
        display = self.model_combo.currentText()
        model_type, pretrained_path = self._resolve_model_selection(display)
        diameter = self._get_effective_diameter_pixels()
        flow = float(self.flow_threshold.value())
        return (float(diameter), flow)

    def _get_effective_mask_threshold(self) -> float:
        """Return mask_threshold to pass to model.eval(). Use UI value as-is (no special case for user models)."""
        return float(self.mask_threshold.value())

    def _open_in_file_manager(self, path: Path) -> None:
        """Open the given path (file or directory) in the OS default file manager."""
        path = Path(path)
        if not path.exists():
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif sys.platform == "win32":
                subprocess.run(["explorer", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            get_logger().debug("Could not open path in file manager", exc_info=True)

    def _validate_for_run(self) -> list[str]:
        """Return a list of error messages; if non-empty, do not start the job."""
        errors: list[str] = []
        if not self.file_path.text():
            errors.append("Please select a TIFF file or folder first.")
            return errors
        path = Path(self.file_path.text())
        if not path.exists():
            errors.append(f"The selected path does not exist: {path}")
            return errors
        script_path = PROJECT_ROOT / "scripts" / "batch_process_tiff_sequence.py"
        if not script_path.exists():
            errors.append(f"Segmentation script not found: {script_path}")
        if self._get_effective_diameter_pixels() < 0:
            errors.append("Diameter must be ≥ 0 (use 0 for auto-detect).")
        return errors

    def process_file_or_folder(self):
        errors = self._validate_for_run()
        if errors:
            QMessageBox.warning(
                self, "Cannot Start",
                "Please fix the following:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return

        script_dir = PROJECT_ROOT
        is_folder = self._is_folder_mode()
        
        if not is_folder:
            # Single file mode
            script_path = str(script_dir / "scripts" / "batch_process_tiff_sequence.py")
            input_file = Path(self.file_path.text())
            
            # When input is in any Tif_* folder (Tif_files, Tif_files2, Tif_1-3, ...), save masks in that folder's _segmentation_omni subfolder
            base_dir = input_file.parent
            if base_dir.name.startswith('Tif_'):
                output_dir = base_dir / "_segmentation_omni"
            else:
                output_dir = base_dir / f"{input_file.stem}_omnipose_segmentation"
            
            _mt, _path = self._resolve_model_selection(self.model_combo.currentText())
            if _path is not None:
                model_arg = ensure_omni_in_model_path(_path)
            else:
                model_arg = _mt or self.model_combo.currentText()
            run_diameter, run_flow = self._get_run_diameter_and_flow()
            cmd_args = [
                self.file_path.text(),
                '--output', str(output_dir),
                '--model', model_arg,
                '--diameter', str(run_diameter),
                '--flow-threshold', str(run_flow),
                '--mask-threshold', str(self._get_effective_mask_threshold()),
                '--batch-size', '32',  # Increased batch size for faster GPU processing (adjust if OOM errors occur)
            ]
            
            if not self.use_gpu.isChecked():
                cmd_args.append('--no-gpu')
            if self.cluster.isChecked():
                cmd_args.append('--cluster')
            if not self.create_overlay.isChecked():
                cmd_args.append('--no-overlay')
            
            self._last_output_dir = output_dir
            self.process_btn.setEnabled(False)
            self.process_btn.setText("Processing...")
            self.cancel_btn.setVisible(True)
            self.cancel_btn.setEnabled(True)
            self.parent_window.log_output(f"Starting single file processing: {self.file_path.text()}\n")
            self.parent_window.log_output(f"Output directory: {output_dir}\n")
        else:
            # Folder mode - show file selection dialog first
            folder = Path(self.file_path.text())
            # Find TIFF files (not mask files)
            tiff_files_set = set(folder.glob("*.tif")) | set(folder.glob("*.tiff")) | \
                            set(folder.glob("*.TIF")) | set(folder.glob("*.TIFF"))
            
            # Filter out mask files and output files
            tiff_files = sorted([f for f in tiff_files_set 
                                if '_masks' not in f.name.lower() 
                                and '_overlay' not in f.name.lower()
                                and '_filter_visualization' not in f.name.lower()], 
                               key=lambda x: x.name.lower())
            
            if not tiff_files:
                QMessageBox.warning(self, "No TIFF Files Found", 
                                  "No TIFF files found in the selected folder.\n\n"
                                  "Please select a folder containing TIFF files to process.")
                return
            
            # Show file selection dialog
            selected_files = show_file_selection_dialog(
                self,
                tiff_files,
                title="Select Files for Segmentation",
                message=f"Select files to process (all {len(tiff_files)} files selected by default):"
            )
            
            if selected_files is None:
                # User cancelled or no files selected
                return
            
            if len(selected_files) == 0:
                QMessageBox.warning(self, "No Files Selected", 
                                  "Please select at least one file to process.")
                return
            
            # Process selected files - use batch_process_folder.py but we'll need to filter files
            # For now, let's process each file individually (simpler approach)
            script_path = str(script_dir / "scripts" / "batch_process_tiff_sequence.py")
            
            # Store files and current index for sequential processing
            self.folder_files = selected_files
            self.current_file_index = 0
            # Track progress within current file (for batch/frame progress)
            self.current_file_progress = 0.0  # 0.0 to 1.0
            
            # Show progress bar
            self.parent_window.progress_bar.setVisible(True)
            self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range for percentage display
            self.parent_window.progress_bar.setValue(0)
            
            self.process_btn.setEnabled(False)
            self.process_btn.setText(f"Processing... (0/{len(selected_files)})")
            self.cancel_btn.setVisible(True)
            self.cancel_btn.setEnabled(True)
            self.parent_window.log_output(f"Starting batch folder processing: {len(selected_files)} file(s) selected from {self.file_path.text()}\n")
            
            # When folder is any Tif_* (Tif_files, Tif_1-3, ...), masks go into _segmentation_omni; else one subfolder per file
            self._last_output_dir = (folder / "_segmentation_omni") if folder.name.startswith('Tif_') else folder
            # Process first file
            self._process_next_segmentation_file(script_path)
            return  # Early return, processing continues in _process_next_segmentation_file
        
        # Show progress bar
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range for percentage display
        self.parent_window.progress_bar.setValue(0)
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()
    
    def _process_next_segmentation_file(self, script_path):
        """Process next file in folder mode (segmentation)"""
        if self.current_file_index >= len(self.folder_files):
            # All files processed
            self.cancel_btn.setVisible(False)
            self.cancel_btn.setEnabled(False)
            self.process_btn.setEnabled(True)
            is_folder = self._is_folder_mode()
            if is_folder:
                self.process_btn.setText("Process All Files in Folder")
            else:
                self.process_btn.setText("Process File")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output(f"\n✓ Segmentation complete for all {len(self.folder_files)} file(s)!\n")
            msg = QMessageBox(self)
            msg.setWindowTitle("Processing Complete")
            out_desc = "Output: _segmentation_omni folder (one mask file per image)." if (self.folder_files and Path(self.folder_files[0]).parent.name.startswith('Tif_')) else "Output: one _omnipose_segmentation subfolder per file (masks and overlay)."
            msg.setText(
                f"Successfully processed {len(self.folder_files)} file(s).\n\n"
                f"{out_desc}"
            )
            msg.setIcon(QMessageBox.Icon.Information)
            open_btn = msg.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
            msg.addButton(QMessageBox.StandardButton.Ok)
            msg.exec()
            if msg.clickedButton() == open_btn and getattr(self, "_last_output_dir", None):
                self._open_in_file_manager(Path(self._last_output_dir))
            return
        
        tiff_file = self.folder_files[self.current_file_index]
        self.parent_window.log_output(f"\nProcessing file {self.current_file_index + 1}/{len(self.folder_files)}: {tiff_file.name}\n")
        
        # When folder is any Tif_* (Tif_files, Tif_1-3, ...), all masks go into that folder's _segmentation_omni subfolder
        base_dir = tiff_file.parent
        if base_dir.name.startswith('Tif_'):
            output_dir = base_dir / "_segmentation_omni"
        else:
            output_dir = base_dir / f"{tiff_file.stem}_omnipose_segmentation"
        
        _mt, _path = self._resolve_model_selection(self.model_combo.currentText())
        if _path is not None:
            model_arg = ensure_omni_in_model_path(_path)
        else:
            model_arg = _mt or self.model_combo.currentText()
        run_diameter, run_flow = self._get_run_diameter_and_flow()
        cmd_args = [
            str(tiff_file),
            '--output', str(output_dir),
            '--model', model_arg,
            '--diameter', str(run_diameter),
            '--flow-threshold', str(run_flow),
            '--mask-threshold', str(self._get_effective_mask_threshold()),
            '--batch-size', '32',  # Increased batch size for faster GPU processing (adjust if OOM errors occur)
        ]
        
        if not self.use_gpu.isChecked():
            cmd_args.append('--no-gpu')
        if self.cluster.isChecked():
            cmd_args.append('--cluster')
        if not self.create_overlay.isChecked():
            cmd_args.append('--no-overlay')
        
        self.parent_window.log_output(f"Output directory: {output_dir}\n")
        
        # Reset current file progress when starting a new file
        self.current_file_progress = 0.0
        
        # Update progress bar to show file start
        file_progress = int((self.current_file_index / len(self.folder_files)) * 100)
        self.parent_window.progress_bar.setValue(file_progress)
        self.process_btn.setText(f"Processing... ({self.current_file_index + 1}/{len(self.folder_files)})")
        
        # Store reference to prevent garbage collection
        self.worker = WorkerThread(script_path, cmd_args, parent=self)
        self.worker.finished.connect(lambda success, msg: self._on_segmentation_file_finished(success, msg, script_path))
        self.worker.progress.connect(self._parse_segmentation_progress)
        self.worker.start()
    
    def _on_segmentation_file_finished(self, success, message, script_path):
        """Handle completion of a single file in folder mode"""
        self.parent_window.log_output(message)
        if success:
            self.parent_window.log_output(f"✓ File {self.current_file_index + 1}/{len(self.folder_files)} processed successfully!\n")
            # Mark file as 100% complete
            self.current_file_progress = 1.0
        else:
            self.parent_window.log_output(f"✗ File {self.current_file_index + 1}/{len(self.folder_files)} failed!\n")
            # Still mark as complete (even if failed) for progress tracking
            self.current_file_progress = 1.0
        
        # Update progress bar to show file completion
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            file_portion = 100.0 / len(self.folder_files)
            file_progress_base = ((self.current_file_index + 1) / len(self.folder_files)) * 100
            self.parent_window.progress_bar.setValue(int(min(100, file_progress_base)))
        
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()
            self.worker = None
        
        # If user cancelled, stop the queue and reset UI
        if getattr(self, "_user_cancelled_processing", False):
            self._user_cancelled_processing = False
            self.cancel_btn.setVisible(False)
            self.cancel_btn.setEnabled(False)
            self.process_btn.setEnabled(True)
            self.process_btn.setText("Process All Files in Folder" if self._is_folder_mode() else "Process File")
            self.parent_window.progress_bar.setVisible(False)
            self.parent_window.log_output("Processing cancelled.\n")
            return
        
        # Move to next file
        self.current_file_index += 1
        self._process_next_segmentation_file(script_path)
    
    def _parse_segmentation_progress(self, message):
        """Parse progress messages from segmentation script and update progress bar"""
        import re
        # Log the message
        self.parent_window.log_output(message)
        
        # Update overall progress based on file, batch, and frame progress
        if hasattr(self, 'folder_files') and len(self.folder_files) > 0:
            # Initialize current_file_progress if not set
            if not hasattr(self, 'current_file_progress'):
                self.current_file_progress = 0.0
            
            # Handle model download progress (small portion of first file, ~5%)
            if 'Downloading:' in message or 'to /' in message and '.cellpose/models/' in message:
                download_match = re.search(r'(\d+)%', message)
                if download_match:
                    try:
                        download_pct = int(download_match.group(1))
                        # Model download is ~5% of first file's progress
                        if self.current_file_index == 0:
                            self.current_file_progress = (download_pct / 100.0) * 0.05
                    except (ValueError, AttributeError):
                        pass  # ignore progress parse errors
            
            # Try to parse batch completion: "✓ Batch 2/6 complete: 64/181 frames (35%)"
            batch_complete_match = re.search(r'Batch (\d+)/(\d+) complete', message)
            if batch_complete_match:
                try:
                    batch_num = int(batch_complete_match.group(1))
                    total_batches = int(batch_complete_match.group(2))
                    # Update progress: batch_num/total_batches (e.g., 2/6 = 0.33)
                    self.current_file_progress = batch_num / total_batches
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
            
            # Try to parse batch start: "Processing batch 3/6: frames 65-96 of 181"
            batch_start_match = re.search(r'Processing batch (\d+)/(\d+):', message)
            if batch_start_match and not batch_complete_match:
                try:
                    batch_num = int(batch_start_match.group(1))
                    total_batches = int(batch_start_match.group(2))
                    # Progress at start of batch (e.g., batch 3/6 means we're at 2/6 = 0.33)
                    # But if this is the first batch and we haven't started processing yet, set to 0.05 (after model load)
                    if batch_num == 1 and self.current_file_progress < 0.05:
                        self.current_file_progress = 0.05
                    else:
                        self.current_file_progress = (batch_num - 1) / total_batches
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
            
            # Try to parse frame progress within batch: "12%|#2 | 4/32 [00:19<02:18, 4.95s/it]"
            frame_progress_match = re.search(r'(\d+)%\s*\|\s*\d+/(\d+)', message)
            if frame_progress_match:
                try:
                    frame_pct = int(frame_progress_match.group(1))
                    # Find which batch we're in
                    batch_info = re.search(r'batch (\d+)/(\d+)', message)
                    if batch_info:
                        batch_num = int(batch_info.group(1))
                        total_batches = int(batch_info.group(2))
                        # Progress = (batch-1)/total + (frame_pct/100)/total
                        batch_progress = (batch_num - 1) / total_batches
                        frame_progress = (frame_pct / 100.0) / total_batches
                        self.current_file_progress = batch_progress + frame_progress
                    else:
                        # Fallback: use frame percentage directly
                        self.current_file_progress = frame_pct / 100.0
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
            
            # Calculate overall progress across all files
            file_portion = 100.0 / len(self.folder_files)
            # Base progress from completed files
            file_progress_base = (self.current_file_index / len(self.folder_files)) * 100
            # Current file's progress scaled to its portion
            file_progress_current = self.current_file_progress * file_portion
            # Total progress
            total_progress = int(min(100, file_progress_base + file_progress_current))
            self.parent_window.progress_bar.setValue(total_progress)
        else:
            # Single file mode - just use percentage from message
            progress_match = re.search(r'(\d+)%', message)
            if progress_match:
                try:
                    progress_pct = int(progress_match.group(1))
                    self.parent_window.progress_bar.setValue(progress_pct)
                except (ValueError, AttributeError):
                    pass  # ignore progress parse errors
    
    def on_finished(self, success, message):
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        is_folder = self._is_folder_mode()
        if is_folder:
            self.process_btn.setText("Process All Files in Folder")
        else:
            self.process_btn.setText("Process File")
        self.parent_window.log_output(message)
        # Set progress to 100% on completion
        if success:
            self.parent_window.progress_bar.setValue(100)
        # Clean up worker thread
        if hasattr(self, 'worker'):
            self.worker.wait()  # Wait for thread to finish
            self.worker = None
        # Hide progress bar
        self.parent_window.progress_bar.setVisible(False)
        if success:
            if is_folder:
                self.parent_window.log_output("✓ Folder processed successfully!\n")
            else:
                self.parent_window.log_output("✓ File processed successfully!\n")
            # Offer to open output folder (single-file mode only; folder mode has its own dialog)
            if not is_folder and getattr(self, "_last_output_dir", None):
                out_path = Path(self._last_output_dir)
                if out_path.exists():
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Processing Complete")
                    msg.setText("File processed successfully.")
                    msg.setIcon(QMessageBox.Icon.Information)
                    open_btn = msg.addButton("Open output folder", QMessageBox.ButtonRole.ActionRole)
                    msg.addButton(QMessageBox.StandardButton.Ok)
                    msg.exec()
                    if msg.clickedButton() == open_btn:
                        self._open_in_file_manager(out_path)
        else:
            if getattr(self, "_user_cancelled_processing", False):
                self._user_cancelled_processing = False
                self.parent_window.log_output("Processing was cancelled.\n")
            else:
                QMessageBox.critical(self, "Error", f"Processing failed:\n{message}")


