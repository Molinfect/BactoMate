"""
Composed Training tab: choose backend (Omnipose / Cellpose / U-Net) and show the
backend-specific training UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.mask_utils import get_training_pairs
from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.styles import MODULE_PANEL_COMBO_STYLESHEET
from bactomate.gui.preview.preview_window import (
    CollapsibleSection,
    LUT_HIST_CANVAS_HEIGHT_PX,
    LUT_HIST_FIGSIZE_INCHES,
    apply_lut_histogram_figure_margins,
)
from bactomate.gui.image_utils import build_segmentation_overlay, to_qpixmap
from bactomate.gui.layout_helpers import style_compact_action_button, style_compact_browse_button
from bactomate.gui.widgets import DragDropGroupBox, create_info_icon
from bactomate.gui.tabs.omnipose_seg import OmniposeSegTab, OmniposeTrainWidget
from bactomate.gui.tabs.unet_seg import UNetSegmentTab, UNetTrainWidget
from bactomate.modules.training_augmentation import (
    AugmentOptions,
    augment_one_pair_in_memory,
    augment_training_dataset,
)


class TrainingTab(QWidget):
    """Top-to-bottom training flow with backend selection."""

    BACKENDS = ("Omnipose", "Cellpose-SAM", "U-Net")

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        omnipose_tab: OmniposeSegTab,
        unet_tab: UNetSegmentTab,
    ) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.omnipose_tab = omnipose_tab
        self.unet_tab = unet_tab
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Dataset modification / augmentation tools live below the main training controls.
        dataset_group = QGroupBox("")
        dataset_group.setFlat(True)
        dataset_layout = QVBoxLayout()
        dataset_layout.setSpacing(6)

        # Short, non-expert explanation at the top
        dataset_hint_row = QHBoxLayout()
        dataset_hint = QLabel("Pick, validate, or augment paired training images and mask images (ground-truth).")
        dataset_hint.setWordWrap(True)
        dataset_hint.setStyleSheet("color: #9e9e9e;")
        dataset_hint_row.addWidget(dataset_hint, 1)
        dataset_hint_row.addWidget(
            create_info_icon(
                "Training dataset = example images + their correct segmentation masks.\n\n"
                "You need pairs:\n"
                "- Image: the raw microscopy image\n"
                "- Mask: a label image where pixels belong to cell(s) (often >0 = cell)\n\n"
                "BactoMate uses these pairs to:\n"
                "- (optional) create augmented copies for more variety\n"
                "- train / fine-tune a segmentation model (backend) so it works on your data",
                parent_widget=self,
            )
        )
        dataset_layout.addLayout(dataset_hint_row)

        def _on_drop_training_dir(dropped_path: str) -> None:
            p = Path(dropped_path)
            if p.is_dir():
                self.train_dir_edit.setText(str(p))
            elif p.is_file():
                self.train_dir_edit.setText(str(p.parent))

        train_dir_group = DragDropGroupBox(
            "Training directory (images + masks)",
            parent=self,
            on_drop_callback=_on_drop_training_dir,
        )
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Training directory:"))
        path_row.addWidget(
            create_info_icon(
                "Choose a folder containing BOTH images and masks.\n\n"
                "Typical layout:\n"
                "  mydata/\n"
                "    sample_001.tif\n"
                "    sample_001_mask.tif\n"
                "    sample_002.tif\n"
                "    sample_002_mask.tif\n\n"
                "Tip: You can drag & drop a folder (or any file inside the folder) into this box.",
                parent_widget=self,
                compact=True,
            )
        )
        self.train_dir_edit = QLineEdit()
        self.train_dir_edit.setPlaceholderText("Folder with images + masks (or drag and drop)")
        path_row.addWidget(self.train_dir_edit, 1)
        browse_btn = QPushButton("Folder…")
        style_compact_browse_button(browse_btn)
        browse_btn.clicked.connect(self._browse_training_dir)
        path_row.addWidget(browse_btn)
        train_dir_group.setLayout(path_row)
        dataset_layout.addWidget(train_dir_group)

        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Image filter:"))
        filt_row.addWidget(
            create_info_icon(
                "Optional filename hint used to find training images.\n\n"
                "If empty, BactoMate tries to use all images in the folder.\n"
                "If set (e.g. `_img`), only files containing that text are treated as images.\n\n"
                "Example:\n"
                "- Image filter `_img` matches: `sample_01_img.tif`",
                parent_widget=self,
                compact=True,
            )
        )
        self.img_filter_edit = QLineEdit()
        self.img_filter_edit.setPlaceholderText("empty or _img")
        self.img_filter_edit.setMinimumWidth(110)
        filt_row.addWidget(self.img_filter_edit)
        filt_row.addSpacing(8)
        filt_row.addWidget(QLabel("Mask filter:"))
        filt_row.addWidget(
            create_info_icon(
                "Filename hint used to find the corresponding mask for each image.\n\n"
                "Example (common pattern):\n"
                "- Image: `sample_01.tif`\n"
                "- Mask:  `sample_01_mask.tif`  → mask filter `_mask`\n\n"
                "Use “Validate pairs” to check whether images and masks match.",
                parent_widget=self,
                compact=True,
            )
        )
        self.mask_filter_edit = QLineEdit()
        self.mask_filter_edit.setText("_mask")
        self.mask_filter_edit.setMinimumWidth(110)
        filt_row.addWidget(self.mask_filter_edit)
        filt_row.addStretch(1)
        validate_btn = QPushButton("Validate pairs")
        style_compact_action_button(validate_btn)
        validate_btn.clicked.connect(self._validate_pairs)
        filt_row.addWidget(validate_btn)
        dataset_layout.addLayout(filt_row)

        self._pair_status = QLabel("")
        self._pair_status.setWordWrap(True)
        self._pair_status.setStyleSheet("color: #9e9e9e;")
        dataset_layout.addWidget(self._pair_status)

        aug_group = QGroupBox("Data augmentation")
        aug_layout = QVBoxLayout()
        aug_layout.setSpacing(6)

        aug_hint_row = QHBoxLayout()
        aug_hint = QLabel("Optional: create slightly modified copies of your training data to improve robustness.")
        aug_hint.setWordWrap(True)
        aug_hint.setStyleSheet("color: #9e9e9e;")
        aug_hint_row.addWidget(aug_hint, 1)
        aug_hint_row.addWidget(
            create_info_icon(
                "Data augmentation makes extra training examples by transforming your images (and masks) in simple ways.\n\n"
                "Why it helps:\n"
                "- Makes the model less sensitive to small changes (orientation, noise, illumination)\n"
                "- Useful when you have only a small dataset\n\n"
                "Important:\n"
                "- Geometric transforms (rotate/flip) are applied to BOTH image and mask\n"
                "- Intensity transforms (noise/brightness) change the image only (mask stays the same)\n\n"
                "Preview first if you're unsure.",
                parent_widget=self,
            )
        )
        aug_layout.addLayout(aug_hint_row)

        aug_controls_row = QHBoxLayout()
        aug_controls_row.setSpacing(10)
        self.aug_enable = QCheckBox("Enable augmentation")
        self.aug_enable.setChecked(False)
        aug_controls_row.addWidget(self.aug_enable)
        aug_controls_row.addSpacing(12)
        aug_controls_row.addWidget(QLabel("Copies per image:"))
        aug_controls_row.addWidget(
            create_info_icon(
                "How many augmented versions to create for each original image.\n\n"
                "Example: 50 originals × 4 copies → up to 200 extra images (and 200 masks).\n\n"
                "More copies = more variety, but also more disk usage and longer training.",
                parent_widget=self,
                compact=True,
            )
        )
        self.aug_num_spin = QSpinBox()
        self.aug_num_spin.setRange(1, 500)
        self.aug_num_spin.setValue(4)
        self.aug_num_spin.setEnabled(False)
        aug_controls_row.addWidget(self.aug_num_spin)
        aug_controls_row.addSpacing(14)

        self.aug_rotate = QCheckBox("Rotate (90° steps)")
        self.aug_rotate.setChecked(True)
        self.aug_rotate.setEnabled(False)
        aug_controls_row.addWidget(self.aug_rotate)
        aug_controls_row.addWidget(
            create_info_icon(
                "Rotate images by 0°, 90°, 180°, or 270°.\n\n"
                "Good default for microscopy because cells don't have a fixed orientation.",
                parent_widget=self,
                compact=True,
            )
        )
        self.aug_flip = QCheckBox("Flip")
        self.aug_flip.setChecked(True)
        self.aug_flip.setEnabled(False)
        aug_controls_row.addWidget(self.aug_flip)
        aug_controls_row.addWidget(
            create_info_icon(
                "Randomly flip left↔right and/or up↔down.\n\n"
                "Usually safe for microscopy. Disable if your images have a meaningful direction.",
                parent_widget=self,
                compact=True,
            )
        )
        aug_controls_row.addSpacing(14)

        self.aug_noise = QCheckBox("Add noise")
        self.aug_noise.setChecked(False)
        self.aug_noise.setEnabled(False)
        aug_controls_row.addWidget(self.aug_noise)
        aug_controls_row.addWidget(QLabel("σ:"))
        aug_controls_row.addWidget(create_info_icon(
            "Add Gaussian noise to the image after geometric transforms.\n\n"
            "σ (sigma) is the standard deviation of the noise in raw intensity units.\n"
            "- Larger σ → grainier image (lower signal-to-noise)\n"
            "- σ = 0 → no effect\n\n"
            "Tip: Choose σ based on typical background fluctuations in your raw data.",
            parent_widget=self,
            compact=True,
        ))
        self.aug_noise_sigma = QDoubleSpinBox()
        self.aug_noise_sigma.setRange(0.0, 10000.0)
        self.aug_noise_sigma.setValue(5.0)
        self.aug_noise_sigma.setDecimals(2)
        self.aug_noise_sigma.setEnabled(False)
        self.aug_noise_sigma.setFixedWidth(90)
        aug_controls_row.addWidget(self.aug_noise_sigma)
        aug_controls_row.addSpacing(14)
        self.aug_brightness = QCheckBox("Adjust brightness")
        self.aug_brightness.setChecked(False)
        self.aug_brightness.setEnabled(False)
        aug_controls_row.addWidget(self.aug_brightness)
        aug_controls_row.addWidget(QLabel("× range:"))
        aug_controls_row.addWidget(create_info_icon(
            "Adjust brightness by multiplying the whole image by a random factor sampled from the range.\n\n"
            "Example: range 0.9 to 1.1 randomly scales the image intensity by -10% to +10%.\n"
            "- Wider range → stronger intensity variation\n"
            "- Range 1.0 to 1.0 → no effect\n\n"
            "Note: This affects pixel intensities only (masks are unchanged).",
            parent_widget=self,
            compact=True,
        ))
        self.aug_bright_min = QDoubleSpinBox()
        self.aug_bright_min.setRange(0.0, 10.0)
        self.aug_bright_min.setValue(0.9)
        self.aug_bright_min.setDecimals(2)
        self.aug_bright_min.setEnabled(False)
        self.aug_bright_min.setFixedWidth(75)
        aug_controls_row.addWidget(self.aug_bright_min)
        aug_controls_row.addWidget(QLabel("to"))
        self.aug_bright_max = QDoubleSpinBox()
        self.aug_bright_max.setRange(0.0, 10.0)
        self.aug_bright_max.setValue(1.1)
        self.aug_bright_max.setDecimals(2)
        self.aug_bright_max.setEnabled(False)
        self.aug_bright_max.setFixedWidth(75)
        aug_controls_row.addWidget(self.aug_bright_max)
        aug_controls_row.addStretch(1)
        aug_layout.addLayout(aug_controls_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output directory (optional):"))
        out_row.addWidget(
            create_info_icon(
                "Where to write the augmented images/masks.\n\n"
                "If left empty, files are written INTO the training directory.\n"
                "If you want to keep your original dataset clean, choose a separate output folder.",
                parent_widget=self,
                compact=True,
            )
        )
        self.aug_out_dir = QLineEdit()
        self.aug_out_dir.setPlaceholderText("If empty, augmented data is written into the training directory")
        self.aug_out_dir.setEnabled(False)
        out_row.addWidget(self.aug_out_dir, 1)
        out_browse = QPushButton("Folder…")
        style_compact_browse_button(out_browse)
        out_browse.setEnabled(False)
        out_browse.clicked.connect(self._browse_aug_output_dir)
        out_row.addWidget(out_browse)
        aug_layout.addLayout(out_row)

        run_row = QHBoxLayout()
        self.aug_preview_btn = QPushButton("Preview augmentation")
        style_compact_action_button(self.aug_preview_btn)
        self.aug_preview_btn.setEnabled(False)
        self.aug_preview_btn.clicked.connect(self._preview_augmentation)
        run_row.addWidget(self.aug_preview_btn)
        run_row.addWidget(
            create_info_icon(
                "Show a before/after example using one random image + mask pair.\n\n"
                "Use this to confirm the transformations look reasonable before generating many files.",
                parent_widget=self,
                compact=True,
            )
        )
        self.aug_run_btn = QPushButton("Run augmentation")
        style_compact_action_button(self.aug_run_btn)
        self.aug_run_btn.setEnabled(False)
        self.aug_run_btn.clicked.connect(self._run_augmentation)
        run_row.addWidget(self.aug_run_btn)
        run_row.addWidget(
            create_info_icon(
                "Generate augmented files on disk.\n\n"
                "This creates new image/mask pairs in the output directory (or training directory if output is empty).",
                parent_widget=self,
                compact=True,
            )
        )
        run_row.addStretch(1)
        aug_layout.addLayout(run_row)

        self._aug_status = QLabel("")
        self._aug_status.setWordWrap(True)
        self._aug_status.setStyleSheet("color: #9e9e9e;")
        aug_layout.addWidget(self._aug_status)

        aug_group.setLayout(aug_layout)
        dataset_layout.addWidget(aug_group)

        dataset_group.setLayout(dataset_layout)

        training_content = QWidget()
        training_layout = QVBoxLayout(training_content)
        training_layout.setContentsMargins(0, 0, 0, 0)
        training_layout.setSpacing(12)

        method_group = QGroupBox("")
        method_group.setFlat(True)
        method_layout = QVBoxLayout()
        method_layout.setSpacing(6)

        backend_hint_row = QHBoxLayout()
        backend_hint = QLabel("Choose which training method to use (different model families).")
        backend_hint.setWordWrap(True)
        backend_hint.setStyleSheet("color: #9e9e9e;")
        backend_hint_row.addWidget(backend_hint, 1)
        backend_hint_row.addWidget(
            create_info_icon(
                "Backend = the segmentation model family and training code.\n\n"
                "Which one should I pick?\n"
                "- Omnipose: strong general bacterial segmentation; good starting point.\n"
                "- Cellpose-SAM: fine-tune Cellpose-SAM (good if you already use Cellpose).\n"
                "- U-Net: classic deep learning model; more configuration but very flexible.\n\n"
                "You can train multiple backends and compare what works best for your images.",
                parent_widget=self,
            )
        )
        method_layout.addLayout(backend_hint_row)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(list(self.BACKENDS))
        self.backend_combo.setCurrentIndex(0)
        self.backend_combo.setStyleSheet(MODULE_PANEL_COMBO_STYLESHEET)
        self.backend_combo.setMaxVisibleItems(len(self.BACKENDS))
        method_layout.addWidget(QLabel("Backend:"))
        method_layout.addWidget(self.backend_combo)
        method_group.setLayout(method_layout)
        training_layout.addWidget(method_group)

        backend_dataset_note = QLabel(
            "Dataset path and filename filters are managed in the Modify training dataset section below and reused by the selected backend."
        )
        backend_dataset_note.setWordWrap(True)
        backend_dataset_note.setStyleSheet("color: #9e9e9e;")
        training_layout.addWidget(backend_dataset_note)

        self.stack = QStackedWidget()
        self._omni_panel = OmniposeTrainWidget(self.omnipose_tab, parent=self)
        from bactomate.gui.tabs.cellpose_sam_train import CellposeSamTrainWidget
        self._cellpose_panel = CellposeSamTrainWidget(
            self,
            get_dataset_dir=lambda: (self.train_dir_edit.text() or "").strip(),
            get_img_filter=lambda: (self.img_filter_edit.text() or "").strip(),
            get_mask_filter=lambda: (self.mask_filter_edit.text() or "").strip(),
            log_output=(getattr(self.parent_window, "log_output", None) or (lambda _s: None)),
        )
        self._unet_panel = UNetTrainWidget(self.unet_tab, parent=self)
        self.stack.addWidget(self._omni_panel)
        self.stack.addWidget(self._cellpose_panel)
        self.stack.addWidget(self._unet_panel)
        self._hide_backend_dataset_controls()
        training_layout.addWidget(self.stack, 1)

        self.training_section = CollapsibleSection(
            "Training backend and actual training",
            training_content,
            parent=self,
            expanded=True,
        )
        layout.addWidget(self.training_section, 0)

        self.modify_dataset_section = CollapsibleSection(
            "Modify training dataset",
            dataset_group,
            parent=self,
            expanded=False,
        )
        layout.addWidget(self.modify_dataset_section, 0)

        self.backend_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.backend_combo.currentIndexChanged.connect(self._sync_dataset_to_backend)
        self.train_dir_edit.textChanged.connect(self._sync_dataset_to_backend)
        self.img_filter_edit.textChanged.connect(self._sync_dataset_to_backend)
        self.mask_filter_edit.textChanged.connect(self._sync_dataset_to_backend)
        self.aug_enable.toggled.connect(lambda _: self._update_aug_ui_state(out_browse))
        self._sync_dataset_to_backend()
        self._update_aug_ui_state(out_browse)

        # Finalize scroll area
        layout.addStretch(1)
        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)
        outer_layout.addWidget(scroll)
        self.setLayout(outer_layout)

    def _hide_backend_dataset_controls(self) -> None:
        """TrainingTab owns dataset path/filter inputs; backend panels reuse synced values."""
        for owner, attrs in (
            (
                self.omnipose_tab,
                (
                    "_omnipose_train_dir_group_widget",
                    "_omnipose_train_filter_widget",
                ),
            ),
            (
                self.unet_tab,
                (
                    "_unet_train_dir_group_widget",
                    "_unet_train_filter_widget",
                ),
            ),
        ):
            for attr in attrs:
                w = getattr(owner, attr, None)
                if w is not None:
                    try:
                        w.setVisible(False)
                    except RuntimeError:
                        pass

    def _browse_training_dir(self) -> None:
        start = self.train_dir_edit.text().strip()
        d = QFileDialog.getExistingDirectory(self, "Select training directory", start or str(Path.home()))
        if d:
            self.train_dir_edit.setText(d)

    def _browse_aug_output_dir(self) -> None:
        start = self.aug_out_dir.text().strip() or self.train_dir_edit.text().strip()
        d = QFileDialog.getExistingDirectory(self, "Select output directory", start or str(Path.home()))
        if d:
            self.aug_out_dir.setText(d)

    def _update_aug_ui_state(self, out_browse_btn: QPushButton) -> None:
        enabled = bool(self.aug_enable.isChecked())
        for w in [
            self.aug_num_spin,
            self.aug_rotate,
            self.aug_flip,
            self.aug_noise,
            self.aug_noise_sigma,
            self.aug_brightness,
            self.aug_bright_min,
            self.aug_bright_max,
            self.aug_out_dir,
        ]:
            w.setEnabled(enabled)
        out_browse_btn.setEnabled(enabled)
        self.aug_run_btn.setEnabled(enabled)
        self.aug_preview_btn.setEnabled(enabled)

    def _validate_pairs(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str:
            self._pair_status.setText("Set a training directory.")
            return
        folder = Path(folder_str)
        if not folder.is_dir():
            self._pair_status.setText("Training directory does not exist.")
            return
        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or "_mask"
        pairs = get_training_pairs(folder, img_f, mask_f)
        # Estimate missing by scanning candidates (best effort)
        from bactomate.modules.training_augmentation import find_training_pairs
        _pairs2, missing = find_training_pairs(folder, img_f, mask_f)
        if missing:
            self._pair_status.setText(
                f"Found {len(pairs)} paired image/mask file(s). Missing masks for {len(missing)} image(s): "
                + ", ".join(missing[:8])
                + (" …" if len(missing) > 8 else "")
            )
            self._pair_status.setStyleSheet("color: #FF9800;")
        else:
            self._pair_status.setText(f"✓ Found {len(pairs)} paired image/mask file(s).")
            self._pair_status.setStyleSheet("color: #4CAF50;")

    def _sync_dataset_to_backend(self) -> None:
        """Push shared dataset path/filters into both backend widgets and disable duplicates."""
        folder = self.train_dir_edit.text().strip()
        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or "_mask"

        # Omnipose training widgets
        for attr, val in [
            ("omnipose_train_dir_edit", folder),
            ("omnipose_img_filter_edit", img_f),
            ("omnipose_mask_filter_edit", mask_f),
        ]:
            w = getattr(self.omnipose_tab, attr, None)
            if w is not None:
                try:
                    w.blockSignals(True)
                    w.setText(val)
                except Exception:
                    pass
                finally:
                    try:
                        w.blockSignals(False)
                    except Exception:
                        pass
                try:
                    w.setEnabled(False)
                except Exception:
                    pass

        # U-Net training widgets
        for attr, val in [
            ("train_dir_edit", folder),
            ("img_filter_edit", img_f),
            ("mask_filter_edit", mask_f),
        ]:
            w = getattr(self.unet_tab, attr, None)
            if w is not None:
                try:
                    w.blockSignals(True)
                    w.setText(val)
                except Exception:
                    pass
                finally:
                    try:
                        w.blockSignals(False)
                    except Exception:
                        pass
                try:
                    w.setEnabled(False)
                except Exception:
                    pass

    def _run_augmentation(self) -> None:
        folder_str = self.train_dir_edit.text().strip()
        if not folder_str:
            self._aug_status.setText("Set a training directory first.")
            return
        folder = Path(folder_str)
        if not folder.is_dir():
            self._aug_status.setText("Training directory does not exist.")
            return
        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or "_mask"
        out_dir = self.aug_out_dir.text().strip()
        opts = AugmentOptions(
            num_aug=int(self.aug_num_spin.value()),
            rotate_90=bool(self.aug_rotate.isChecked()),
            flip=bool(self.aug_flip.isChecked()),
            add_noise=bool(self.aug_noise.isChecked()),
            noise_sigma=float(self.aug_noise_sigma.value()),
            adjust_brightness=bool(self.aug_brightness.isChecked()),
            brightness_min=float(self.aug_bright_min.value()),
            brightness_max=float(self.aug_bright_max.value()),
        )
        out_path, n_img, n_mask, missing = augment_training_dataset(
            input_dir=folder,
            img_filter=img_f,
            mask_filter=mask_f,
            output_dir=Path(out_dir) if out_dir else None,
            options=opts,
            seed=0,
        )
        msg = f"Augmentation written to: {out_path}\nWrote images: {n_img}, masks: {n_mask}."
        if missing:
            msg += f"\nMissing masks for {len(missing)} image(s)."
        self._aug_status.setText(msg)
        self._aug_status.setStyleSheet("color: #4CAF50;" if n_img > 0 else "color: #FF9800;")
        # If user chose a separate output, keep training dir as-is; otherwise training dir now contains new files.
        self._validate_pairs()

    def _preview_augmentation(self) -> None:
        import numpy as np
        import tifffile

        folder_str = self.train_dir_edit.text().strip()
        if not folder_str:
            self._aug_status.setText("Set a training directory first.")
            return
        folder = Path(folder_str)
        if not folder.is_dir():
            self._aug_status.setText("Training directory does not exist.")
            return

        img_f = self.img_filter_edit.text().strip()
        mask_f = self.mask_filter_edit.text().strip() or "_mask"
        opts = AugmentOptions(
            num_aug=1,
            rotate_90=bool(self.aug_rotate.isChecked()),
            flip=bool(self.aug_flip.isChecked()),
            add_noise=bool(self.aug_noise.isChecked()),
            noise_sigma=float(self.aug_noise_sigma.value()),
            adjust_brightness=bool(self.aug_brightness.isChecked()),
            brightness_min=float(self.aug_bright_min.value()),
            brightness_max=float(self.aug_bright_max.value()),
        )

        # Pick a random pair from the folder
        from bactomate.modules.training_augmentation import find_training_pairs
        pairs, missing = find_training_pairs(folder, img_f, mask_f)
        if not pairs:
            self._aug_status.setText("No image/mask pairs found for preview. Check filters.")
            return
        if missing:
            # Still allow preview, but hint
            self._aug_status.setText(f"Warning: missing masks for {len(missing)} image(s). Preview uses a valid pair.")

        # Dialog with before/after canvases
        class PreviewDialog(QDialog):
            pass

        d = PreviewDialog(self.parent_window if hasattr(self, "parent_window") else self)
        d.setWindowTitle("Augmentation preview")
        d.setMinimumSize(1100, 700)
        layout = QVBoxLayout()

        title = QLabel("")
        title.setStyleSheet("font-weight: bold; color: #ddd;")
        layout.addWidget(title)

        canv_row = QHBoxLayout()
        before_canvas = ImageCanvas()
        before_canvas.setToolTip("Before (image + mask overlay). Zoom: wheel. Pan: drag.")
        before_canvas.setStyleSheet("background-color: #1e1e1e;")
        after_canvas = ImageCanvas()
        after_canvas.setToolTip("After (augmented image + mask overlay). Zoom: wheel. Pan: drag.")
        after_canvas.setStyleSheet("background-color: #1e1e1e;")
        canv_row.addWidget(before_canvas, 1)
        canv_row.addWidget(after_canvas, 1)
        layout.addLayout(canv_row, 1)

        # Interactive LUT histogram (same UX as File Conversion preview), shared for both views
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
            from matplotlib.figure import Figure  # type: ignore

            _has_lut = True
        except Exception:
            _has_lut = False

        def _make_lut_widget(title_text: str):
            group = QGroupBox(title_text)
            v = QVBoxLayout()
            v.setSpacing(8)
            if not _has_lut:
                note = QLabel("LUTs histogram requires matplotlib.")
                note.setWordWrap(True)
                note.setStyleSheet("color: #9e9e9e;")
                v.addWidget(note)
                group.setLayout(v)
                return (group, None)

            fig = Figure(figsize=LUT_HIST_FIGSIZE_INCHES)
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(LUT_HIST_CANVAS_HEIGHT_PX)
            canvas.setMaximumHeight(LUT_HIST_CANVAS_HEIGHT_PX)
            ax = fig.add_subplot(111)
            fig.patch.set_facecolor("#2b2b2b")
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            apply_lut_histogram_figure_margins(fig)
            canvas.setStyleSheet("background-color: #2b2b2b;")
            _tr_lut_tip = "Drag the red (min) and teal (max) dashed lines to adjust display range."
            canvas.setToolTip(_tr_lut_tip)
            v.addWidget(canvas)

            # Internal widgets (not shown) to reuse the same min/max logic as File Conversion
            lut_min = QDoubleSpinBox()
            lut_min.setRange(-1e12, 1e12)
            lut_min.setDecimals(3)
            lut_max = QDoubleSpinBox()
            lut_max.setRange(-1e12, 1e12)
            lut_max.setDecimals(3)

            group.setLayout(v)
            group.setMaximumHeight(200)
            state = {
                "fig": fig,
                "canvas": canvas,
                "ax": ax,
                "lut_min": lut_min,
                "lut_max": lut_max,
                "dragging": None,
                "vmin": None,
                "vmax": None,
                "frame_min": None,
                "frame_max": None,
                "line_min": None,
                "line_max": None,
                "span": None,
            }
            return (group, state)

        lut_group, lut_state = _make_lut_widget("LUTs — original (blue) / augmented (red)")
        layout.addWidget(lut_group)

        buttons = QDialogButtonBox()
        resample_btn = buttons.addButton("Resample", QDialogButtonBox.ButtonRole.ActionRole)
        close_btn = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close_btn.clicked.connect(d.close)
        layout.addWidget(buttons)

        d.setLayout(layout)

        def _to_2d(arr):
            a = arr
            if a is None:
                return None
            a = np.asarray(a)
            if a.ndim == 2:
                return a
            if a.ndim == 3:
                # Prefer first plane if stack-like; otherwise squeeze singleton channel
                if a.shape[-1] == 1:
                    return a[..., 0]
                return a[0, ...]
            return a

        def render_once() -> None:
            img_path, mask_path, base = pairs[int(np.random.randint(0, len(pairs)))]
            try:
                img = tifffile.imread(str(img_path))
                mask = tifffile.imread(str(mask_path))
            except Exception as e:
                title.setText(f"Failed to read: {img_path.name} / {mask_path.name}: {e}")
                return
            img2 = _to_2d(img)
            mask2 = _to_2d(mask)
            if img2 is None or mask2 is None:
                title.setText(f"Unsupported shapes for preview: {img_path.name} ({np.asarray(img).shape}) / {mask_path.name} ({np.asarray(mask).shape})")
                return

            # Apply one augmentation in-memory
            img_aug, mask_aug = augment_one_pair_in_memory(
                image=np.asarray(img2),
                mask=np.asarray(mask2),
                options=opts,
                seed=None,
            )

            # Overlay: treat mask>0 as foreground
            mask_bin_before = (np.asarray(mask2) > 0).astype(np.uint8)
            mask_bin_after = (np.asarray(mask_aug) > 0).astype(np.uint8)

            def _auto_lut(frame_2d: np.ndarray) -> tuple[float, float]:
                x = np.asarray(frame_2d, dtype=np.float64)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    return (0.0, 1.0)
                lo = float(np.percentile(x, 0.01))
                hi = float(np.percentile(x, 99.99))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo = float(np.nanmin(x))
                    hi = float(np.nanmax(x))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        hi = lo + 1.0
                return (lo, hi)

            def _apply_lut(frame_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
                vmin = float(vmin)
                vmax = float(vmax)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin + 1.0
                vr = max(vmax - vmin, 1e-9)
                return ((frame_2d.astype(np.float64) - vmin) / vr * 255.0).clip(0, 255).astype(np.uint8)

            def _refresh_canvas(target_canvas: ImageCanvas, frame_2d: np.ndarray, mask_bin: np.ndarray, vmin: float, vmax: float) -> None:
                disp = _apply_lut(frame_2d, vmin, vmax)
                overlay = build_segmentation_overlay(disp, mask_bin, alpha=0.35)
                pix = to_qpixmap(overlay)
                target_canvas.set_pixmap(pix, (overlay.shape[1], overlay.shape[0]), reset_view=True)

            # Shared LUT based on combined ranges
            vmin_b, vmax_b = _auto_lut(np.asarray(img2))
            vmin_a, vmax_a = _auto_lut(np.asarray(img_aug))
            vmin = float(min(vmin_b, vmin_a))
            vmax = float(max(vmax_b, vmax_a))
            _refresh_canvas(before_canvas, np.asarray(img2), mask_bin_before, vmin, vmax)
            _refresh_canvas(after_canvas, np.asarray(img_aug), mask_bin_after, vmin, vmax)

            # Update LUT widgets + interactions
            def _update_hist_overlay(state, orig_2d: np.ndarray, aug_2d: np.ndarray, vmin_in: float, vmax_in: float) -> None:
                if state is None:
                    return
                ax = state["ax"]
                fig = state["fig"]
                canvas = state["canvas"]
                ax.clear()

                o = np.asarray(orig_2d).ravel()
                o = o[np.isfinite(o)]
                a = np.asarray(aug_2d).ravel()
                a = a[np.isfinite(a)]
                combined = np.concatenate([o, a]) if (o.size and a.size) else (o if o.size else a)
                if combined.size == 0:
                    combined = np.asarray(orig_2d).ravel()

                frame_min = float(np.nanmin(combined)) if combined.size else 0.0
                frame_max = float(np.nanmax(combined)) if combined.size else 1.0
                if not np.isfinite(frame_min) or not np.isfinite(frame_max) or frame_max <= frame_min:
                    frame_min, frame_max = 0.0, 1.0

                bins = 256
                hist_o, edges = np.histogram(o if o.size else combined, bins=bins, range=(frame_min, frame_max))
                hist_a, _ = np.histogram(a if a.size else combined, bins=bins, range=(frame_min, frame_max))
                centers = (edges[:-1] + edges[1:]) / 2
                w = edges[1] - edges[0]

                ax.bar(centers, np.log1p(hist_o), width=w, color="#4A90E2", alpha=0.55, edgecolor="none", label="Original")
                ax.bar(centers, np.log1p(hist_a), width=w, color="#FF6B6B", alpha=0.45, edgecolor="none", label="Augmented")

                vmin_plot = float(vmin_in)
                vmax_plot = float(vmax_in)
                state["line_min"] = ax.axvline(
                    vmin_plot, color="#FF6B6B", linewidth=1.5, linestyle="--", label=f"Min: {vmin_plot:.1f}"
                )
                state["line_max"] = ax.axvline(
                    vmax_plot, color="#4ECDC4", linewidth=1.5, linestyle="--", label=f"Max: {vmax_plot:.1f}"
                )
                state["span"] = ax.axvspan(vmin_plot, vmax_plot, alpha=0.10, color="yellow")

                ax.set_facecolor("#2b2b2b")
                ax.tick_params(colors="white", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("white")
                ax.set_xlabel("Intensity", color="white", fontsize=8)
                ax.set_ylabel("Log(Count+1)", color="white", fontsize=8)
                ax.legend(
                    loc="upper right",
                    facecolor="#2b2b2b",
                    edgecolor="#5a5a5a",
                    labelcolor="white",
                    fontsize=7,
                    framealpha=0.85,
                    bbox_to_anchor=(0.99, 0.99),
                )
                ax.grid(True, alpha=0.22, color="white", linestyle="-", linewidth=0.5)
                ax.margins(x=0.02, y=0.06)
                apply_lut_histogram_figure_margins(fig)
                canvas.draw()
                state["frame_min"] = frame_min
                state["frame_max"] = frame_max
                state["vmin"] = vmin_plot
                state["vmax"] = vmax_plot

            def _refresh_both(vmin_new: float, vmax_new: float) -> None:
                _refresh_canvas(before_canvas, np.asarray(img2), mask_bin_before, vmin_new, vmax_new)
                _refresh_canvas(after_canvas, np.asarray(img_aug), mask_bin_after, vmin_new, vmax_new)

            if lut_state is not None:
                _update_hist_overlay(lut_state, np.asarray(img2), np.asarray(img_aug), vmin, vmax)

                def on_press(event):
                    if event.inaxes != lut_state["ax"]:
                        return
                    x = event.xdata
                    if x is None:
                        return
                    vmin0 = float(lut_state["vmin"])
                    vmax0 = float(lut_state["vmax"])
                    rng = float(lut_state.get("frame_max", vmax0)) - float(lut_state.get("frame_min", vmin0))
                    thr = max(rng * 0.05, rng * 0.01) if rng > 0 else 1.0
                    if abs(x - vmin0) < thr and abs(x - vmin0) < abs(x - vmax0):
                        lut_state["dragging"] = "min"
                    elif abs(x - vmax0) < thr:
                        lut_state["dragging"] = "max"
                    else:
                        mid = (vmin0 + vmax0) / 2
                        lut_state["dragging"] = "min" if x < mid else "max"

                def on_motion(event):
                    if event.inaxes != lut_state["ax"]:
                        return
                    x = event.xdata
                    if x is None or lut_state["dragging"] is None:
                        return
                    frame_min = float(lut_state.get("frame_min", x))
                    frame_max = float(lut_state.get("frame_max", x))
                    vmin1 = float(lut_state["vmin"])
                    vmax1 = float(lut_state["vmax"])
                    if lut_state["dragging"] == "min":
                        vmin1 = max(frame_min, min(float(x), vmax1 - 1e-6))
                        lut_state["vmin"] = vmin1
                    else:
                        vmax1 = min(frame_max, max(float(x), vmin1 + 1e-6))
                        lut_state["vmax"] = vmax1

                    _refresh_both(float(lut_state["vmin"]), float(lut_state["vmax"]))

                    # Update lines and yellow span to match drag (bugfix)
                    if lut_state.get("line_min") is not None:
                        lut_state["line_min"].set_xdata([lut_state["vmin"], lut_state["vmin"]])
                    if lut_state.get("line_max") is not None:
                        lut_state["line_max"].set_xdata([lut_state["vmax"], lut_state["vmax"]])
                    sp = lut_state.get("span")
                    if sp is not None:
                        try:
                            sp.remove()
                        except Exception:
                            pass
                    lut_state["span"] = lut_state["ax"].axvspan(float(lut_state["vmin"]), float(lut_state["vmax"]), alpha=0.15, color="yellow")
                    lut_state["canvas"].draw()

                def on_release(_event):
                    lut_state["dragging"] = None

                lut_state["canvas"].mpl_connect("button_press_event", on_press)
                lut_state["canvas"].mpl_connect("motion_notify_event", on_motion)
                lut_state["canvas"].mpl_connect("button_release_event", on_release)

            title.setText(f"{base}   (image: {img_path.name}, mask: {mask_path.name})")

        resample_btn.clicked.connect(render_once)
        render_once()
        d.show()
        d.raise_()
        d.activateWindow()

