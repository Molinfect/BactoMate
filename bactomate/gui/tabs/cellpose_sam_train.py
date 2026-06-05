"""
Cellpose-SAM training widget (fine-tune cpsam) using the official Cellpose CLI.

Docs: https://cellpose.readthedocs.io/en/latest/train.html
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.layout_helpers import style_compact_action_button, style_module_primary_button
from bactomate.gui.workers import WorkerThread
from bactomate.gui.widgets import create_info_icon


class CellposeSamTrainWidget(QWidget):
    """Train (fine-tune) Cellpose-SAM using `python -m cellpose --train ...`."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        get_dataset_dir: Callable[[], str],
        get_img_filter: Callable[[], str],
        get_mask_filter: Callable[[], str],
        log_output: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self._get_dataset_dir = get_dataset_dir
        self._get_img_filter = get_img_filter
        self._get_mask_filter = get_mask_filter
        self._log_output = log_output
        self._worker: Optional[WorkerThread] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        group = QGroupBox("Train Cellpose-SAM (cpsam)")
        g = QVBoxLayout()
        g.setSpacing(8)

        # Model name out
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Model name:"))
        name_row.addWidget(create_info_icon(
            "Name of the trained model.\n\n"
            "Cellpose saves models into a `models/` subfolder inside your training directory.\n"
            "This name becomes the output checkpoint name (see `--model_name_out`).",
            parent_widget=self,
        ))
        self.model_name_out = QLineEdit()
        self.model_name_out.setPlaceholderText("e.g. my_cpsam_v1")
        name_row.addWidget(self.model_name_out, 1)
        g.addLayout(name_row)

        # Params
        row = QHBoxLayout()
        row.addWidget(QLabel("Epochs:"))
        row.addWidget(create_info_icon(
            "Number of training epochs.\n\n"
            "Cellpose docs recommend starting around 100 epochs for fine-tuning cpsam.",
            parent_widget=self,
            compact=True,
        ))
        self.n_epochs = QSpinBox()
        self.n_epochs.setRange(1, 100000)
        self.n_epochs.setValue(100)
        row.addWidget(self.n_epochs)

        row.addSpacing(10)
        row.addWidget(QLabel("LR:"))
        row.addWidget(create_info_icon(
            "Learning rate.\n\n"
            "Recommended cpsam fine-tune starting point: 1e-5.\n"
            "Lower LR trains more conservatively; higher LR adapts faster but can overfit/unstable.",
            parent_widget=self,
            compact=True,
        ))
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(7)
        self.learning_rate.setRange(0.0, 1.0)
        self.learning_rate.setValue(1e-5)
        self.learning_rate.setSingleStep(1e-5)
        row.addWidget(self.learning_rate)

        row.addSpacing(10)
        row.addWidget(QLabel("Weight decay:"))
        row.addWidget(create_info_icon(
            "Weight decay (L2 regularization).\n\n"
            "Recommended cpsam fine-tune starting point: 0.1.",
            parent_widget=self,
            compact=True,
        ))
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setDecimals(4)
        self.weight_decay.setRange(0.0, 10.0)
        self.weight_decay.setValue(0.1)
        self.weight_decay.setSingleStep(0.01)
        row.addWidget(self.weight_decay)

        row.addSpacing(10)
        row.addWidget(QLabel("Batch:"))
        row.addWidget(create_info_icon(
            "Training batch size.\n\n"
            "Cellpose docs recommend batch=1 for cpsam fine-tuning.\n"
            "Increase only if you have enough GPU memory.",
            parent_widget=self,
            compact=True,
        ))
        self.train_batch_size = QSpinBox()
        self.train_batch_size.setRange(1, 128)
        self.train_batch_size.setValue(1)
        row.addWidget(self.train_batch_size)

        row.addStretch(1)
        g.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Min masks/image:"))
        row2.addWidget(create_info_icon(
            "Minimum number of masks a training image must contain to be included.\n\n"
            "Use to skip nearly-empty images that can hurt training.",
            parent_widget=self,
            compact=True,
        ))
        self.min_train_masks = QSpinBox()
        self.min_train_masks.setRange(0, 10000)
        self.min_train_masks.setValue(5)
        row2.addWidget(self.min_train_masks)

        row2.addSpacing(10)
        row2.addWidget(QLabel("Save every:"))
        row2.addWidget(create_info_icon(
            "Checkpoint save interval (in epochs).\n\n"
            "Example: 100 saves once at the end for a 100-epoch run; smaller values save intermediate checkpoints.",
            parent_widget=self,
            compact=True,
        ))
        self.save_every = QSpinBox()
        self.save_every.setRange(1, 100000)
        self.save_every.setValue(100)
        row2.addWidget(self.save_every)

        row2.addSpacing(10)
        self.use_gpu = QCheckBox("Use GPU")
        self.use_gpu.setChecked(True)
        row2.addWidget(create_info_icon(
            "Use GPU for training (recommended if available).",
            parent_widget=self,
            compact=True,
        ))
        row2.addWidget(self.use_gpu)
        row2.addStretch(1)
        g.addLayout(row2)

        # Actions
        self.start_btn = QPushButton("Start Cellpose-SAM training")
        style_module_primary_button(self.start_btn)
        self.start_btn.clicked.connect(self._start_training)
        g.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        style_compact_action_button(self.cancel_btn)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_training)
        g.addWidget(self.cancel_btn)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        g.addWidget(self.log)

        hint = QLabel(
            "Cellpose will save the trained model into a `models/` subfolder inside the training directory.\n"
            "Recommended starting point from the docs: cpsam fine-tune with LR=1e-5, weight_decay=0.1, batch=1."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9e9e9e;")
        g.addWidget(hint)

        group.setLayout(g)
        layout.addWidget(group)
        layout.addStretch(1)
        self.setLayout(layout)

    def _append_log(self, s: str) -> None:
        try:
            self.log.append((s or "").rstrip())
        except Exception:
            pass

    def _cancel_training(self) -> None:
        w = self._worker
        if w is not None and w.isRunning():
            w.terminate_process()
            self._append_log("Cancelled by user.")
            try:
                self._log_output("Cellpose-SAM training cancelled by user.\n")
            except Exception:
                pass

    def _start_training(self) -> None:
        train_dir = (self._get_dataset_dir() or "").strip()
        if not train_dir:
            QMessageBox.warning(self, "Cellpose-SAM training", "Set a training directory first.")
            return
        d = Path(train_dir).expanduser().resolve()
        if not d.is_dir():
            QMessageBox.warning(self, "Cellpose-SAM training", "Training directory does not exist.")
            return
        img_filter = (self._get_img_filter() or "").strip()
        mask_filter = (self._get_mask_filter() or "").strip() or "_masks"
        model_name = (self.model_name_out.text() or "").strip()
        if not model_name:
            QMessageBox.warning(self, "Cellpose-SAM training", "Set a model name.")
            return

        args = [
            "cellpose",
            "--train",
            "--dir",
            str(d),
            "--img_filter",
            img_filter,
            "--mask_filter",
            mask_filter,
            "--learning_rate",
            f"{float(self.learning_rate.value()):.8g}",
            "--weight_decay",
            f"{float(self.weight_decay.value()):.8g}",
            "--n_epochs",
            str(int(self.n_epochs.value())),
            "--train_batch_size",
            str(int(self.train_batch_size.value())),
            "--min_train_masks",
            str(int(self.min_train_masks.value())),
            "--save_every",
            str(int(self.save_every.value())),
            "--model_name_out",
            model_name,
        ]
        if self.use_gpu.isChecked():
            args.append("--gpu")

        self._append_log("Running:\npython -m " + " ".join(args))
        try:
            self._log_output(f"Starting Cellpose-SAM training in {d}\n")
        except Exception:
            pass

        self.start_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.cancel_btn.setEnabled(True)

        w = WorkerThread("-m", args, parent=self)
        self._worker = w
        w.progress.connect(self._append_log)
        w.progress.connect(self._log_output)

        def on_done(success: bool, message: str) -> None:
            self.start_btn.setEnabled(True)
            self.cancel_btn.setVisible(False)
            self.cancel_btn.setEnabled(False)
            if success:
                self._append_log("Done. Model saved under: " + str((d / "models").resolve()))
                QMessageBox.information(
                    self,
                    "Cellpose-SAM training",
                    f"Training completed.\n\nModel folder:\n{(d / 'models').resolve()}",
                )
            else:
                QMessageBox.warning(self, "Cellpose-SAM training", "Training failed. See the log for details.")

        w.finished.connect(on_done)
        w.start()

