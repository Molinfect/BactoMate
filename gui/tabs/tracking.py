"""
Tracking tab: run trackastra on mask TIFFs.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QVBoxLayout,
    QWidget,
)

from gui.session_params import apply_to_widgets, get_from_widgets
from gui.widgets import DragDropGroupBox, create_info_icon, show_file_selection_dialog

from gui.workers import TrackingWorker


class TrackingTab(QWidget):
    """Tab for cell tracking using trackastra"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.device_mapping = {}  # Initialize device mapping
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        self._build_input_groups(scroll_layout)

        tracking_group = self._build_tracking_params()
        scroll_layout.addWidget(tracking_group)

        # Process button - modern styling applied via global stylesheet
        self.process_btn = QPushButton("Run Tracking")
        self.process_btn.setEnabled(False)  # Disabled until mask file is selected
        self.process_btn.clicked.connect(self.run_tracking)
        scroll_layout.addWidget(self.process_btn)
        
        # Connect mask path changes to update button state
        self.mask_path.textChanged.connect(self._update_tracking_button_state)
        # Initialize state
        self._update_tracking_button_state()
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _build_input_groups(self, scroll_layout: QVBoxLayout) -> None:
        """Build mask, original, and Excel file input groups and add them to scroll_layout."""
        file_font = QFont()
        file_font.setPointSize(11)
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        mask_group = DragDropGroupBox(
            "Mask File",
            parent=self,
            on_drop_callback=self.handle_dropped_mask_file
        )
        mask_layout = QHBoxLayout()
        self.mask_path = QLineEdit()
        self.mask_path.setPlaceholderText("Select mask TIFF file (time series)... (or drag and drop)")
        self.mask_path.setMinimumWidth(500)
        self.mask_path.setFont(file_font)
        self.mask_path.setSizePolicy(size_policy)
        mask_btn = QPushButton("Browse File...")
        mask_btn.setFixedWidth(120)
        mask_btn.clicked.connect(self.select_mask_file)
        mask_layout.addWidget(self.mask_path, 0)
        mask_layout.addWidget(mask_btn, 0)
        mask_group.setLayout(mask_layout)
        scroll_layout.addWidget(mask_group)

        original_group = DragDropGroupBox(
            "Original Data File (Optional)",
            parent=self,
            on_drop_callback=self.handle_dropped_original_file
        )
        original_layout = QHBoxLayout()
        self.original_path = QLineEdit()
        self.original_path.setPlaceholderText("Select original TIFF file for visualization (optional)... (or drag and drop)")
        self.original_path.setMinimumWidth(500)
        self.original_path.setFont(file_font)
        self.original_path.setSizePolicy(size_policy)
        original_btn = QPushButton("Browse File...")
        original_btn.setFixedWidth(120)
        original_btn.clicked.connect(self.select_original_file)
        original_layout.addWidget(self.original_path, 0)
        original_layout.addWidget(original_btn, 0)
        original_group.setLayout(original_layout)
        scroll_layout.addWidget(original_group)

        excel_group = DragDropGroupBox(
            "Excel Analysis File (Optional)",
            parent=self,
            on_drop_callback=self.handle_dropped_excel_file
        )
        excel_layout = QHBoxLayout()
        self.excel_path = QLineEdit()
        self.excel_path.setPlaceholderText("Select Excel file with cell analysis data (e.g., *_masks_analysis.xlsx)... (or drag and drop)")
        self.excel_path.setMinimumWidth(500)
        self.excel_path.setFont(file_font)
        self.excel_path.setSizePolicy(size_policy)
        excel_info_icon = create_info_icon(
            "Link to existing Excel analysis file containing cell measurements.\n\n"
            "This file should contain per-cell data from the Analyze Masks tab.\n"
            "The tracking will link cell IDs across frames and preserve\n"
            "cell properties from the analysis file.",
            parent_widget=self
        )
        excel_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        excel_btn = QPushButton("Browse File...")
        excel_btn.setFixedWidth(120)
        excel_btn.clicked.connect(self.select_excel_file)
        excel_layout.addWidget(self.excel_path, 0)
        excel_layout.addWidget(excel_info_icon, 0)
        excel_layout.addWidget(excel_btn, 0)
        excel_group.setLayout(excel_layout)
        scroll_layout.addWidget(excel_group)

    def _build_tracking_params(self) -> QGroupBox:
        """Build the Tracking Parameters group (model, mode, device)."""
        tracking_group = QGroupBox("Tracking Parameters")
        tracking_container_layout = QHBoxLayout()

        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['general_2d', 'general_2d_w_SAM2_features'])
        self.model_combo.setCurrentText('general_2d')
        model_info_icon = create_info_icon(
            "Pretrained Trackastra model for cell tracking.\n\n"
            "Default: 'general_2d'\n"
            "Source: Trackastra pretrained models (https://github.com/weigertlab/trackastra)\n\n"
            "Available models:\n"
            "- general_2d: General-purpose 2D tracking model (default)\n"
            "  • Fast and works well for most cell types\n"
            "  • Good balance between speed and accuracy\n"
            "  • Recommended for general use cases\n\n"
            "- general_2d_w_SAM2_features: Enhanced model with SAM2 features\n"
            "  • Uses Segment Anything Model 2 (SAM2) features for improved tracking\n"
            "  • Better performance on certain data types, especially bacteria\n"
            "  • Slower than general_2d (requires GPU for reasonable speed)\n"
            "  • Best for challenging tracking scenarios\n\n"
            "Note: Trackastra won the 7th Cell Tracking Challenge (ISBI 2024).\n"
            "The models use transformer-based architecture for cell linking.",
            parent_widget=self
        )
        model_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(model_info_icon)
        model_layout.addStretch()
        tracking_container_layout.addLayout(model_layout)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['greedy', 'greedy_nodiv', 'ilp'])
        self.mode_combo.setCurrentText('greedy')
        mode_info_icon = create_info_icon(
            "Linking mode for connecting cells across time frames.\n\n"
            "Default: 'greedy'\n"
            "Source: Trackastra linking algorithms (https://github.com/weigertlab/trackastra)\n\n"
            "Available modes:\n"
            "- greedy: Greedy linking with division support (recommended)\n"
            "  • Fast algorithm, no additional dependencies\n"
            "  • Handles cell divisions (splitting)\n"
            "  • Good for most tracking scenarios\n"
            "  • Best balance of speed and accuracy\n\n"
            "- greedy_nodiv: Greedy linking without division support\n"
            "  • Fast algorithm, no additional dependencies\n"
            "  • Cells cannot split (no division tracking)\n"
            "  • Use when cells don't divide or you want faster processing\n"
            "  • Simpler tracking logic\n\n"
            "- ilp: Integer Linear Programming (optimal linking)\n"
            "  • More accurate than greedy methods\n"
            "  • Solves global optimization problem\n"
            "  • Slower than greedy methods\n"
            "  • Requires additional dependencies (motile, Gurobi or SCIP solver)\n"
            "  • Best for challenging cases where accuracy is critical\n"
            "  • Note: Academic Gurobi licenses are free\n\n"
            "Tip: Start with 'greedy' for most cases. Use 'ilp' if you need maximum accuracy.",
            parent_widget=self
        )
        mode_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(mode_info_icon)
        mode_layout.addStretch()
        tracking_container_layout.addLayout(mode_layout)

        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        available_devices_display = ['CPU']
        self.device_mapping = {'CPU': 'cpu'}
        device_info_text = "Compute device for tracking.\n\nDefault: CPU\nSource: PyTorch device selection\n\nOptions:\n- CPU: CPU processing (works everywhere, slower)\n"
        try:
            import torch
            if torch.cuda.is_available():
                available_devices_display.append('CUDA')
                self.device_mapping['CUDA'] = 'cuda'
                device_info_text += "- CUDA: NVIDIA GPU processing (faster, requires CUDA-capable GPU)\n"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available_devices_display.append('GPU')
                self.device_mapping['GPU'] = 'mps'
                device_info_text += "- GPU: Apple Silicon GPU (M1/M2/M3) or NVIDIA GPU (faster)\n"
        except ImportError:
            pass
        device_info_text += "\nTip: Use GPU if available for faster processing."
        device_info_icon = create_info_icon(device_info_text, parent_widget=self)
        device_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices_display)
        self.device_combo.setCurrentText('CPU')
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(device_info_icon)
        device_layout.addStretch()
        tracking_container_layout.addLayout(device_layout)

        tracking_group.setLayout(tracking_container_layout)
        return tracking_group

    def _update_tracking_button_state(self):
        """Update tracking button enabled state based on whether mask file is selected"""
        has_mask = bool(self.mask_path.text() and self.mask_path.text().strip())
        if hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(has_mask)
    
    def select_mask_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask File", "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.mask_path.setText(file_path)
            self._update_tracking_button_state()
    
    def select_original_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Original Data File", "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.original_path.setText(file_path)
    
    def select_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel Analysis File", "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.excel_path.setText(file_path)
    
    def handle_dropped_mask_file(self, dropped_path):
        """Handle dropped mask file for tracking tab"""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ['.tif', '.tiff']:
            self.mask_path.setText(dropped_path)
            self._update_tracking_button_state()
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid TIFF mask file.")
    
    def handle_dropped_original_file(self, dropped_path):
        """Handle dropped original data file for tracking tab"""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ['.tif', '.tiff']:
            self.original_path.setText(dropped_path)
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid TIFF file.")
    
    def handle_dropped_excel_file(self, dropped_path):
        """Handle dropped Excel file for tracking tab"""
        path_obj = Path(dropped_path)
        if path_obj.is_file() and path_obj.suffix.lower() in ['.xlsx', '.xls']:
            self.excel_path.setText(dropped_path)
        else:
            QMessageBox.warning(self, "Invalid File", 
                              "Please drop a valid Excel file (.xlsx or .xls).")
    
    def run_tracking(self):
        if not self.mask_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select a mask file.")
            return
        
        mask_file = Path(self.mask_path.text())
        if not mask_file.exists():
            QMessageBox.warning(self, "File Not Found", f"Mask file not found:\n{mask_file}")
            return
        
        # Get optional files
        original_file = self.original_path.text() if self.original_path.text() else None
        excel_file = self.excel_path.text() if self.excel_path.text() else None
        
        if original_file and not Path(original_file).exists():
            QMessageBox.warning(self, "File Not Found", f"Original file not found:\n{original_file}")
            return
        
        if excel_file and not Path(excel_file).exists():
            QMessageBox.warning(self, "File Not Found", f"Excel file not found:\n{excel_file}")
            return
        
        # Get parameters
        model_name = self.model_combo.currentText()
        mode = self.mode_combo.currentText()
        # Map display name to actual device value
        device_display = self.device_combo.currentText()
        device = self.device_mapping.get(device_display, 'cpu')
        
        # Check device availability with detailed debugging
        if device == 'cuda':
            debug_info = []
            debug_info.append("=== CUDA Debugging Information ===")
            
            try:
                import torch
                debug_info.append(f"✓ PyTorch version: {torch.__version__}")
                
                # Check CUDA availability
                cuda_available = torch.cuda.is_available()
                debug_info.append(f"✓ torch.cuda.is_available(): {cuda_available}")
                
                if not cuda_available:
                    debug_info.append("✗ CUDA is not available")
                    debug_info.append(f"  - CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
                    debug_info.append(f"  - cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
                    
                    # Check if CUDA libraries are found
                    try:
                        import subprocess
                        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
                        if result.returncode == 0:
                            debug_info.append(f"  - nvcc found: {result.stdout.strip()}")
                        else:
                            debug_info.append("  - nvcc not found in PATH")
                    except:
                        pass
                    
                    error_msg = "\n".join(debug_info)
                    error_msg += "\n\nCUDA GPU is not available on this system."
                    error_msg += "\n\nPossible reasons:"
                    error_msg += "\n- No NVIDIA GPU installed"
                    error_msg += "\n- CUDA drivers not installed"
                    error_msg += "\n- PyTorch was installed without CUDA support"
                    error_msg += "\n- GPU drivers are outdated"
                    
                    reply = QMessageBox.question(
                        self, "CUDA Not Available",
                        error_msg + "\n\nWould you like to continue with CPU mode?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = 'cpu'
                    self.device_combo.setCurrentText('CPU')
                    self.parent_window.log_output("\n".join(debug_info) + "\n")
                else:
                    # CUDA is available, get detailed info
                    debug_info.append("✓ CUDA is available!")
                    debug_info.append(f"  - CUDA version (compiled): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
                    debug_info.append(f"  - Number of GPUs: {torch.cuda.device_count()}")
                    
                    for i in range(torch.cuda.device_count()):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_props = torch.cuda.get_device_properties(i)
                        debug_info.append(f"  - GPU {i}: {gpu_name}")
                        debug_info.append(f"    - Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
                        debug_info.append(f"    - Compute Capability: {gpu_props.major}.{gpu_props.minor}")
                    
                    # Test CUDA with a simple operation
                    debug_info.append("\nTesting CUDA functionality...")
                    try:
                        # Test basic tensor creation
                        test_tensor = torch.tensor([1.0], device='cuda')
                        debug_info.append("  ✓ Tensor creation: SUCCESS")
                        
                        # Test simple operation
                        result = test_tensor * 2.0
                        debug_info.append("  ✓ Tensor operation: SUCCESS")
                        
                        # Test memory allocation
                        test_large = torch.zeros(1000, 1000, device='cuda')
                        debug_info.append("  ✓ Memory allocation: SUCCESS")
                        
                        del test_tensor, result, test_large
                        torch.cuda.empty_cache()
                        debug_info.append("  ✓ Memory cleanup: SUCCESS")
                        
                        debug_info.append("\n✓ CUDA is working correctly!")
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                        
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        debug_info.append(f"  ✗ CUDA test failed: {str(e)}")
                        debug_info.append(f"\nError details:\n{error_details}")
                        
                        error_msg = "\n".join(debug_info)
                        error_msg += "\n\nCUDA is available but encountered an error during testing."
                        error_msg += "\nThis might indicate:"
                        error_msg += "\n- GPU driver issues"
                        error_msg += "\n- CUDA runtime problems"
                        error_msg += "\n- Memory allocation issues"
                        
                        reply = QMessageBox.question(
                            self, "CUDA Error",
                            error_msg + "\n\nWould you like to continue with CPU mode?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.No:
                            return
                        device = 'cpu'
                        self.device_combo.setCurrentText('cpu')
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                        
            except ImportError:
                debug_info.append("✗ PyTorch is not installed")
                error_msg = "\n".join(debug_info)
                error_msg += "\n\nPyTorch is required for CUDA support."
                error_msg += "\nInstall with: pip install torch torchvision"
                error_msg += "\nFor CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                
                QMessageBox.warning(
                    self, "PyTorch Not Found",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
            except Exception as e:
                import traceback
                debug_info.append(f"✗ Unexpected error: {str(e)}")
                debug_info.append(f"\n{traceback.format_exc()}")
                error_msg = "\n".join(debug_info)
                
                QMessageBox.warning(
                    self, "CUDA Check Error",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
        elif device == 'mps':
            # Check MPS (Apple Silicon) availability
            debug_info = []
            debug_info.append("=== GPU (Apple Silicon) Debugging Information ===")
            
            try:
                import torch
                debug_info.append(f"✓ PyTorch version: {torch.__version__}")
                
                # Check MPS availability
                if hasattr(torch.backends, 'mps'):
                    mps_available = torch.backends.mps.is_available()
                    debug_info.append(f"✓ torch.backends.mps.is_available(): {mps_available}")
                    
                    if not mps_available:
                        debug_info.append("✗ GPU is not available")
                        debug_info.append("  - This usually means you're not on Apple Silicon (M1/M2/M3)")
                        debug_info.append("  - Or PyTorch was built without GPU support")
                        
                        error_msg = "\n".join(debug_info)
                        error_msg += "\n\nMPS (Apple Silicon GPU) is not available."
                        error_msg += "\n\nPossible reasons:"
                        error_msg += "\n- Not running on Apple Silicon (M1/M2/M3 chip)"
                        error_msg += "\n- PyTorch was installed without MPS support"
                        error_msg += "\n- macOS version too old (needs macOS 12.3+)"
                        
                        reply = QMessageBox.question(
                            self, "MPS Not Available",
                            error_msg + "\n\nWould you like to continue with CPU mode?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.No:
                            return
                        device = 'cpu'
                        self.device_combo.setCurrentText('cpu')
                        self.parent_window.log_output("\n".join(debug_info) + "\n")
                    else:
                        # MPS is available, test it
                        debug_info.append("✓ GPU is available!")
                        debug_info.append("  - Device: Apple Silicon GPU")
                        
                        # Test MPS with a simple operation
                        debug_info.append("\nTesting GPU functionality...")
                        try:
                            # Test basic tensor creation
                            test_tensor = torch.tensor([1.0], device='mps')
                            debug_info.append("  ✓ Tensor creation: SUCCESS")
                            
                            # Test simple operation
                            result = test_tensor * 2.0
                            debug_info.append("  ✓ Tensor operation: SUCCESS")
                            
                            # Test memory allocation
                            test_large = torch.zeros(1000, 1000, device='mps')
                            debug_info.append("  ✓ Memory allocation: SUCCESS")
                            
                            del test_tensor, result, test_large
                            torch.mps.empty_cache()
                            debug_info.append("  ✓ Memory cleanup: SUCCESS")
                            
                            debug_info.append("\n✓ GPU is working correctly!")
                            self.parent_window.log_output("\n".join(debug_info) + "\n")
                            
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            debug_info.append(f"  ✗ MPS test failed: {str(e)}")
                            debug_info.append(f"\nError details:\n{error_details}")
                            
                            error_msg = "\n".join(debug_info)
                            error_msg += "\n\GPU is available but encountered an error during testing."
                            error_msg += "\nThis might indicate:"
                            error_msg += "\n- macOS version compatibility issues"
                            error_msg += "\n- PyTorch MPS backend problems"
                            error_msg += "\n- Memory allocation issues"
                            
                            reply = QMessageBox.question(
                                self, "MPS Error",
                                error_msg + "\n\nWould you like to continue with CPU mode?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.Yes
                            )
                            if reply == QMessageBox.StandardButton.No:
                                return
                            device = 'cpu'
                            self.device_combo.setCurrentText('cpu')
                            self.parent_window.log_output("\n".join(debug_info) + "\n")
                else:
                    debug_info.append("✗ MPS backend not available in this PyTorch version")
                    error_msg = "\n".join(debug_info)
                    error_msg += "\n\nMPS backend is not available."
                    error_msg += "\nThis PyTorch version may not support MPS."
                    
                    reply = QMessageBox.question(
                        self, "MPS Not Supported",
                        error_msg + "\n\nWould you like to continue with CPU mode?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    device = 'cpu'
                    self.device_combo.setCurrentText('CPU')
                    self.parent_window.log_output("\n".join(debug_info) + "\n")
                    
            except ImportError:
                debug_info.append("✗ PyTorch is not installed")
                error_msg = "\n".join(debug_info)
                error_msg += "\n\nPyTorch is required for MPS support."
                
                QMessageBox.warning(
                    self, "PyTorch Not Found",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
            except Exception as e:
                import traceback
                debug_info.append(f"✗ Unexpected error: {str(e)}")
                debug_info.append(f"\n{traceback.format_exc()}")
                error_msg = "\n".join(debug_info)
                
                QMessageBox.warning(
                    self, "MPS Check Error",
                    error_msg + "\n\nSwitching to CPU mode."
                )
                device = 'cpu'
                self.device_combo.setCurrentText('CPU')
                self.parent_window.log_output("\n".join(debug_info) + "\n")
        
        # Create worker thread
        self.worker = TrackingWorker(
            mask_file, original_file, excel_file,
            model_name, mode, device, self
        )
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.start()
        
        # Disable button during processing
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)  # 0-100 range
        self.parent_window.progress_bar.setValue(0)
        self.parent_window.log_output(f"Starting tracking for: {mask_file.name}\n")
        
        # Connect progress signal
        self.worker.progress.connect(self.on_tracking_progress)
    
    def on_tracking_progress(self, value, message):
        """Update progress bar and log message"""
        self.parent_window.progress_bar.setValue(value)
        if message:
            self.parent_window.log_output(f"[{value}%] {message}\n")
    
    def on_tracking_finished(self, success, message):
        self.parent_window.progress_bar.setValue(100)
        if success:
            self.parent_window.log_output("✓ Tracking completed successfully!\n")
        else:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{message}")
        # Hide progress bar after a short delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.parent_window.progress_bar.setVisible(False))
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        return get_from_widgets([
            ("model", self.model_combo),
            ("mode", self.mode_combo),
            ("device", self.device_combo),
        ])

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("model", self.model_combo),
            ("mode", self.mode_combo),
            ("device", self.device_combo),
        ])


