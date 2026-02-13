"""
SAM2 (Segment Anything Model 2) tab: interactive segmentation on TIFF files/folders.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.session_params import apply_to_widgets, get_from_widgets
from bactimate.gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog

from bactimate.gui.workers import SAM2SegmentWorker

class SAM2SegmentTab(QWidget):
    """Tab for SAM2 (Segment Anything Model 2) interactive segmentation"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        file_group = self._build_file_group()
        scroll_layout.addWidget(file_group)

        # How SAM2 works (short info)
        sam2_info_layout = QHBoxLayout()
        sam2_info_icon = create_info_icon(
            "How SAM2 works:\n\n"
            "SAM2 (Segment Anything Model 2) is a general-purpose segmentation model. It places a regular grid of "
            "points on the image and segments whatever it finds at each point (automatic mode). It was trained on "
            "diverse natural images and works best when objects have clear boundaries and good contrast.\n\n"
            "Why results can be poor on bacteria:\n"
            "• Bacteria are small, rod-shaped, and often low contrast; SAM2 was not trained specifically for them.\n"
            "• For bacteria and time-lapse data, the Omnipose model (Segmentation tab) is usually better suited.\n\n"
            "When to use SAM2: diverse objects, high contrast, or when you want to try a different approach. "
            "Use the Preview button to check segmentation on one frame before processing.",
            parent_widget=self
        )
        sam2_info_label = QLabel("SAM2 is a general-purpose segment-anything model. For bacteria, prefer the Segmentation (Omnipose) tab.")
        sam2_info_label.setWordWrap(True)
        sam2_info_label.setStyleSheet("color: #9aa; font-style: italic; padding: 4px;")
        sam2_info_layout.addWidget(sam2_info_icon)
        sam2_info_layout.addWidget(sam2_info_label, 1)
        scroll_layout.addLayout(sam2_info_layout)

        sam2_group = self._build_sam2_params_group()
        scroll_layout.addWidget(sam2_group)

        # Preview SAM2 segmentation (runs SAM2 on first frame and shows overlay)
        self.preview_btn = QPushButton("Preview SAM2 Segmentation")
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip(
            "Run SAM2 on the selected TIFF (first frame) and show the segmentation overlay.\n"
            "Uses current model and parameters. May take a minute to load the model."
        )
        self.preview_btn.clicked.connect(self.preview_sam2_segmentation)
        scroll_layout.addWidget(self.preview_btn)
        
        # Process button
        self.process_btn = QPushButton("Run SAM2 Segmentation")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        scroll_layout.addWidget(self.process_btn)
        
        # Connect path changes to update button state
        self.file_path.textChanged.connect(self._update_button_state)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)
        
        # Initial button state update
        self._update_button_state()

    def _build_file_group(self) -> QWidget:
        """Build the input file/folder group with path line edit and browse buttons."""
        file_group = DragDropGroupBox(
            "Input File/Folder",
            parent=self,
            on_drop_callback=self.handle_dropped_path_sam2
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

    def _build_sam2_params_group(self) -> QGroupBox:
        """Build the SAM2 Parameters group (model settings + segmentation options)."""
        sam2_group = QGroupBox("SAM2 Parameters")
        sam2_container_layout = QHBoxLayout()

        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        model_select_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_info_icon = create_info_icon(
            "SAM2.1 model variant for segmentation.\n\n"
            "Available models:\n"
            "- sam2.1_hiera_tiny: Fastest, smallest (38.9M params)\n"
            "- sam2.1_hiera_small: Balanced speed/accuracy (46M params)\n"
            "- sam2.1_hiera_base_plus: Better accuracy (80.8M params)\n"
            "- sam2.1_hiera_large: Best accuracy, slowest (224.4M params)\n\n"
            "Tip: Start with 'small' or 'base_plus' for good balance.",
            parent_widget=self
        )
        model_select_layout.addWidget(model_label)
        model_select_layout.addWidget(model_info_icon)
        model_select_layout.addStretch()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "sam2.1_hiera_tiny",
            "sam2.1_hiera_small",
            "sam2.1_hiera_base_plus",
            "sam2.1_hiera_large"
        ])
        self.model_combo.setCurrentText("sam2.1_hiera_small")
        model_select_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_select_layout)
        checkpoint_layout = QHBoxLayout()
        checkpoint_label = QLabel("Checkpoint Path:")
        checkpoint_info_icon = create_info_icon(
            "Path to SAM2 model checkpoint file (.pt).\n\n"
            "If empty, will attempt to download from Hugging Face or use default location.\n\n"
            "Checkpoints can be downloaded from:\n"
            "https://github.com/facebookresearch/sam2#download-checkpoints\n\n"
            "Or use Hugging Face models (auto-downloaded if checkpoint not found).",
            parent_widget=self
        )
        checkpoint_layout.addWidget(checkpoint_label)
        checkpoint_layout.addWidget(checkpoint_info_icon)
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Leave empty for auto-download or default location...")
        self.checkpoint_path.setMinimumWidth(300)
        checkpoint_btn = QPushButton("Browse...")
        checkpoint_btn.setFixedWidth(100)
        checkpoint_btn.clicked.connect(self.select_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_path, 1)
        checkpoint_layout.addWidget(checkpoint_btn, 0)
        model_layout.addLayout(checkpoint_layout)
        model_group.setLayout(model_layout)
        model_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sam2_container_layout.addWidget(model_group)

        options_group = QGroupBox("Segmentation Options")
        options_layout = QVBoxLayout()
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_info_icon = create_info_icon(
            "Segmentation mode for SAM2.\n\n"
            "- Automatic: Generate masks automatically (like SAM)\n"
            "- Interactive: Use prompts (points/boxes) for guided segmentation\n\n"
            "Note: Interactive mode requires additional UI components (future feature).",
            parent_widget=self
        )
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(mode_info_icon)
        mode_layout.addStretch()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Automatic", "Interactive (Coming Soon)"])
        self.mode_combo.setCurrentText("Automatic")
        self.mode_combo.setItemData(1, None, Qt.ItemDataRole.UserRole - 1)
        mode_layout.addWidget(self.mode_combo)
        options_layout.addLayout(mode_layout)
        points_layout = QHBoxLayout()
        points_label = QLabel("Points per side:")
        points_info_icon = create_info_icon(
            "Number of points per side for automatic mask generation.\n\n"
            "Higher values = more masks generated (slower)\n"
            "Lower values = fewer masks (faster)\n\n"
            "Default: 32 (good balance)",
            parent_widget=self
        )
        points_layout.addWidget(points_label)
        points_layout.addWidget(points_info_icon)
        points_layout.addStretch()
        self.points_per_side = QSpinBox()
        self.points_per_side.setRange(8, 128)
        self.points_per_side.setValue(32)
        self.points_per_side.setFixedWidth(100)
        points_layout.addWidget(self.points_per_side)
        options_layout.addLayout(points_layout)
        stability_layout = QHBoxLayout()
        stability_label = QLabel("Stability Score:")
        stability_info_icon = create_info_icon(
            "Minimum stability score for mask filtering.\n\n"
            "Range: 0.0 to 1.0\n"
            "Higher values = more stable masks only\n"
            "Lower values = more masks (including less stable)\n\n"
            "Default: 0.95",
            parent_widget=self
        )
        stability_layout.addWidget(stability_label)
        stability_layout.addWidget(stability_info_icon)
        stability_layout.addStretch()
        self.stability_score = QDoubleSpinBox()
        self.stability_score.setRange(0.0, 1.0)
        self.stability_score.setValue(0.95)
        self.stability_score.setSingleStep(0.05)
        self.stability_score.setDecimals(2)
        self.stability_score.setFixedWidth(100)
        stability_layout.addWidget(self.stability_score)
        options_layout.addLayout(stability_layout)
        gpu_layout = QHBoxLayout()
        self.use_gpu = QCheckBox("Use GPU (if available)")
        self.use_gpu.setChecked(True)
        gpu_info_icon = create_info_icon(
            "Use GPU acceleration for faster processing.\n\n"
            "Default: Enabled\n"
            "Automatically falls back to CPU if GPU not available.",
            parent_widget=self
        )
        gpu_layout.addWidget(self.use_gpu)
        gpu_layout.addWidget(gpu_info_icon)
        gpu_layout.addStretch()
        options_layout.addLayout(gpu_layout)
        overlay_layout = QHBoxLayout()
        self.create_overlay = QCheckBox("Create overlay visualization")
        self.create_overlay.setChecked(True)
        overlay_info_icon = create_info_icon(
            "Create TIFF overlay showing masks on original images.\n\n"
            "Overlay files will be saved with '_sam2_overlay.tif' suffix.",
            parent_widget=self
        )
        overlay_layout.addWidget(self.create_overlay)
        overlay_layout.addWidget(overlay_info_icon)
        overlay_layout.addStretch()
        options_layout.addLayout(overlay_layout)
        options_group.setLayout(options_layout)
        options_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sam2_container_layout.addWidget(options_group)
        sam2_group.setLayout(sam2_container_layout)
        return sam2_group

    def _update_button_state(self):
        """Update process and preview button states based on input fields"""
        has_file = bool(self.file_path.text())
        self.process_btn.setEnabled(has_file)
        self.preview_btn.setEnabled(has_file)
    
    def select_file(self):
        """Select a TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF File",
            "",
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
    
    def select_folder(self):
        """Select a folder containing TIFF files"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with TIFF Files",
            ""
        )
        if folder_path:
            self.file_path.setText(folder_path)
    
    def handle_dropped_path_sam2(self, dropped_path):
        """Handle dropped file or folder path for SAM2 segmentation tab"""
        path_obj = Path(dropped_path)
        # Update file path
        self.file_path.setText(dropped_path)
        # Update button state
        self._update_button_state()
    
    def select_checkpoint(self):
        """Select SAM2 checkpoint file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM2 Checkpoint File",
            "",
            "PyTorch Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.checkpoint_path.setText(file_path)
    
    def preview_sam2_segmentation(self):
        """Run SAM2 on the first frame of the selected TIFF and show segmentation overlay."""
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a TIFF file first.")
            return
        
        file_path = Path(self.file_path.text())
        if not file_path.exists():
            QMessageBox.warning(self, "Not Found", "The selected file does not exist.")
            return
        
        if file_path.is_dir():
            QMessageBox.warning(self, "Folder Selected", "Please select a specific TIFF file for preview, not a folder.")
            return
        
        progress = QProgressDialog("Loading SAM2 model and running segmentation...", None, 0, 100, self)
        progress.setWindowTitle("SAM2 Preview")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            import numpy as np
            import torch
            progress.setLabelText("Loading image...")
            progress.setValue(10)
            QApplication.processEvents()
            
            image_data = safe_imread(str(file_path))
            if image_data.ndim == 2:
                first_frame = image_data
            elif image_data.ndim == 3:
                first_frame = image_data[0] if image_data.shape[0] < 1000 else image_data[:, :, 0]
            elif image_data.ndim == 4:
                first_frame = image_data[0, 0] if image_data.shape[1] < 10 else image_data[0, :, :, 0]
            else:
                progress.close()
                QMessageBox.warning(self, "Unsupported", "Unsupported image dimensions for preview.")
                return
            
            img_min, img_max = first_frame.min(), first_frame.max()
            if img_max > img_min:
                image_norm = ((first_frame - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image_norm = first_frame.astype(np.uint8)
            if len(image_norm.shape) == 2:
                image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
            else:
                image_rgb = np.repeat(image_norm[:, :, np.newaxis], 3, axis=-1)
            
            progress.setLabelText("Loading SAM2 model...")
            progress.setValue(20)
            QApplication.processEvents()
            
            device = "cuda" if (self.use_gpu.isChecked() and torch.cuda.is_available()) else "cpu"
            if self.use_gpu.isChecked() and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            
            try:
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                progress.close()
                QMessageBox.critical(self, "SAM2 Not Found", "SAM2 not installed. pip install sam2")
                return
            
            model_name = self.model_combo.currentText()
            hf_map = {
                'sam2.1_hiera_tiny': 'facebook/sam2-hiera-tiny',
                'sam2.1_hiera_small': 'facebook/sam2-hiera-small',
                'sam2.1_hiera_base_plus': 'facebook/sam2-hiera-base-plus',
                'sam2.1_hiera_large': 'facebook/sam2-hiera-large',
            }
            hf_name = hf_map.get(model_name, 'facebook/sam2-hiera-small')
            
            try:
                predictor = SAM2ImagePredictor.from_pretrained(hf_name, device=device)
                model = predictor.model
            except Exception as e:
                progress.close()
                QMessageBox.critical(self, "Model Load Failed", f"Could not load SAM2 model:\n{str(e)}")
                return
            
            progress.setLabelText("Running SAM2 segmentation on first frame...")
            progress.setValue(50)
            QApplication.processEvents()
            
            mask_generator = SAM2AutomaticMaskGenerator(
                model=model,
                points_per_side=self.points_per_side.value(),
                pred_iou_thresh=0.8,
                stability_score_thresh=self.stability_score.value(),
                box_nms_thresh=0.7,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            with torch.inference_mode():
                if device == "mps":
                    with torch.autocast(device_type="cpu", dtype=torch.float32):
                        sam2_masks = mask_generator.generate(image_rgb)
                else:
                    sam2_masks = mask_generator.generate(image_rgb)
            
            progress.setValue(80)
            QApplication.processEvents()
            
            if not sam2_masks:
                label_mask = np.zeros(image_norm.shape[:2], dtype=np.uint16)
            else:
                label_mask = np.zeros(image_norm.shape[:2], dtype=np.uint16)
                for mask_idx, mask_dict in enumerate(sorted(sam2_masks, key=lambda x: x['area'], reverse=True), start=1):
                    label_mask[mask_dict['segmentation']] = mask_idx
            
            progress.setLabelText("Building overlay...")
            progress.setValue(90)
            QApplication.processEvents()
            
            from matplotlib import pyplot as plt
            if len(image_norm.shape) == 2:
                img_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
            else:
                img_rgb = np.repeat(image_norm[:, :, np.newaxis], 3, axis=-1)
            cmap = plt.get_cmap('gist_ncar')
            num_labels = int(label_mask.max())
            if num_labels > 0:
                colors = cmap(np.linspace(0, 1, max(num_labels + 1, 100)))[:num_labels + 1]
                colors[0] = [0, 0, 0, 0]
                mask_rgb = (colors[label_mask][:, :, :3] * 255).astype(np.uint8)
                overlay = (img_rgb.astype(np.float32) * 0.5 + mask_rgb.astype(np.float32) * 0.5).astype(np.uint8)
            else:
                overlay = img_rgb
            
            progress.setValue(100)
            progress.close()
            
            # Show overlay in dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle(f"SAM2 Preview: {file_path.name} (frame 1)")
            preview_dialog.setMinimumSize(900, 700)
            preview_dialog.setModal(False)
            layout = QVBoxLayout()
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(overlay)
            ax.set_title(f"SAM2 segmentation (frame 1) — {num_labels} regions. Model: {model_name}")
            ax.axis('off')
            fig.tight_layout()
            layout.addWidget(canvas)
            close_shortcut = QShortcut(QKeySequence.StandardKey.Close, preview_dialog)
            close_shortcut.activated.connect(preview_dialog.close)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)
            preview_dialog.setLayout(layout)
            preview_dialog.show()
            
        except Exception as e:
            progress.close()
            import traceback
            QMessageBox.critical(self, "Preview Error",
                f"Could not run SAM2 preview:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def start_processing(self):
        """Start SAM2 segmentation processing"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Processing", "Processing is already in progress.")
            return
        
        input_path = self.file_path.text()
        if not input_path:
            QMessageBox.warning(self, "No Input", "Please select an input file or folder.")
            return
        
        # Get parameters
        model_name = self.model_combo.currentText()
        checkpoint_path = self.checkpoint_path.text() if self.checkpoint_path.text() else None
        mode = self.mode_combo.currentText()
        points_per_side = self.points_per_side.value()
        stability_score = self.stability_score.value()
        use_gpu = self.use_gpu.isChecked()
        create_overlay = self.create_overlay.isChecked()
        # Output saved next to input (same dir or input folder); no separate output field
        output_dir = None
        
        # Disable button during processing
        self.process_btn.setEnabled(False)
        
        # Create and start worker
        self.worker = SAM2SegmentWorker(
            input_path,
            model_name,
            checkpoint_path,
            mode,
            points_per_side,
            stability_score,
            use_gpu,
            create_overlay,
            output_dir
        )
        self.worker.progress.connect(self._parse_sam2_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()
        
        # Show progress bar
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
    
    def _parse_sam2_progress(self, message):
        """Parse progress messages from SAM2 worker and update progress bar"""
        import re
        # Log the message
        self.parent_window.log_output(message)
        
        # Try to parse percentage from message
        progress_match = re.search(r'(\d+)%', message)
        if progress_match:
            try:
                progress_pct = int(progress_match.group(1))
                progress_pct = max(0, min(100, progress_pct))
                self.parent_window.progress_bar.setValue(progress_pct)
            except:
                pass
        # Check for file processing progress (e.g., "Processing file 2/5")
        file_match = re.search(r'file\s+(\d+)/(\d+)', message, re.IGNORECASE)
        if file_match:
            try:
                current_file = int(file_match.group(1))
                total_files = int(file_match.group(2))
                if total_files > 0:
                    progress_pct = int((current_file / total_files) * 100)
                    self.parent_window.progress_bar.setValue(progress_pct)
            except:
                pass
        # Increment slightly for other progress messages
        elif any(keyword in message.lower() for keyword in ['processing', 'loading', 'creating', 'saving']):
            current = self.parent_window.progress_bar.value()
            if current < 95:
                self.parent_window.progress_bar.setValue(min(95, current + 1))
    
    def on_processing_finished(self, success, message):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.parent_window.progress_bar.setValue(100)
        self.parent_window.progress_bar.setVisible(False)
        
        if success:
            self.parent_window.log_output(f"✓ SAM2 segmentation completed!\n{message}\n")
            QMessageBox.information(self, "Success", f"SAM2 segmentation completed!\n\n{message}")
        else:
            self.parent_window.log_output(f"✗ SAM2 segmentation failed:\n{message}\n")
            QMessageBox.critical(self, "Error", f"SAM2 segmentation failed:\n\n{message}")
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        return get_from_widgets([
            ("model", self.model_combo),
            ("mode", self.mode_combo),
            ("points_per_side", self.points_per_side),
            ("stability_score", self.stability_score),
            ("use_gpu", self.use_gpu),
            ("create_overlay", self.create_overlay),
        ])

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("model", self.model_combo),
            ("mode", self.mode_combo),
            ("points_per_side", self.points_per_side),
            ("stability_score", self.stability_score),
            ("use_gpu", self.use_gpu),
            ("create_overlay", self.create_overlay),
        ])


