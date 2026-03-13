"""
Background workers for long-running tasks (subprocess runner, tracking, growth curve video).
"""
import io
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from bactimate.gui.log_config import get_logger, log_exception, write_crash_log
from bactimate.gui.paths import get_app_root, get_python_executable_for_subprocess


class _TrackingLogHandler(logging.Handler):
    """Forwards log records to a TrackingWorker's progress signal so the GUI log shows trackastra status."""

    def __init__(self, worker, progress_pct: int):
        super().__init__(level=logging.INFO)
        self._worker = worker
        self._progress_pct = progress_pct

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage().strip()
            if msg:
                self._worker.progress.emit(self._progress_pct, msg)
        except Exception:
            pass


class WorkerThread(QThread):
    """Thread for running long-running tasks without freezing the GUI."""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # status message

    def __init__(self, script_path, cmd_args, parent=None):
        super().__init__(parent)
        self.script_path = script_path
        self.cmd_args = cmd_args
        self._process = None

    def terminate_process(self) -> None:
        """Request termination of the running subprocess (e.g. for Cancel). Safe to call from main thread."""
        if getattr(self, "_process", None) is not None:
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None

    @staticmethod
    def strip_ansi_codes(text: str) -> str:
        """Remove ANSI escape codes from text (e.g. [A for cursor movement)."""
        import re
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    @staticmethod
    def strip_log_prefix(text: str) -> str:
        """Strip timestamp and [INFO]/[DEBUG] prefix so tqdm/progress detection and display fit in one line."""
        import re
        s = text.strip()
        s = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+\[(?:INFO|DEBUG|WARNING|ERROR)\]\s*", "", s)
        s = re.sub(r"^\[(?:INFO|DEBUG|WARNING|ERROR)\]\s*", "", s)
        s = re.sub(r"^[\w.]+:(?:INFO|DEBUG|WARNING|ERROR):\s*", "", s)
        s = re.sub(r"^line\s+\d+\s*[\t ]*", "", s)
        return s.strip()

    @staticmethod
    def is_tqdm_progress_line(text: str) -> bool:
        """Check if a line is a tqdm progress bar (e.g. '12%|#2        | 4/32' or after stripping log prefix)."""
        import re
        stripped = WorkerThread.strip_log_prefix(text)
        tqdm_pattern = re.compile(r"^\d+%\s*\|[#\s]+\|\s*\d+/\d+\s*\[.*\]\s*$")
        return bool(tqdm_pattern.match(stripped))

    def _run_script_in_process(self, module_path: str, main_name: str = "main", argv_first: str = None) -> bool:
        """Run a script's main() in this thread (same interpreter = has bundled deps). argv_first = first argv element (e.g. script name)."""
        old_argv = list(sys.argv)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        buf = io.StringIO()
        try:
            if argv_first is not None:
                sys.argv = [argv_first] + [str(a) for a in self.cmd_args]
            else:
                sys.argv = [str(self.script_path)] + [str(a) for a in self.cmd_args]
            sys.stdout = buf
            sys.stderr = buf
            try:
                import importlib
                mod = importlib.import_module(module_path)
                main_fn = getattr(mod, main_name)
                code = main_fn()
                if code is None:
                    code = 0
            except SystemExit as e:
                c = getattr(e, "code", None)
                code = c if isinstance(c, int) else (0 if c is None else 1)
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            output = buf.getvalue()
            for line in output.splitlines():
                line = line.strip()
                if line:
                    self.progress.emit(self.strip_ansi_codes(line) + "\n")
            self.finished.emit(code == 0, output)
            return True
        except Exception as e:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            err_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            write_crash_log("Worker (in-process script) failed:\n" + err_msg)
            self.finished.emit(False, err_msg)
            return True

    def _run_convert_nd2_in_process(self) -> bool:
        """Run convert_nd2_to_tif in this thread (same interpreter = has numpy)."""
        return self._run_script_in_process("bactimate.scripts.convert_nd2_to_tif", argv_first=str(self.script_path))

    def _run_batch_tiff_in_process(self) -> bool:
        """Run batch_process_tiff_sequence in this thread (same interpreter = has numpy, cellpose_omni, etc.)."""
        return self._run_script_in_process("bactimate.scripts.batch_process_tiff_sequence", argv_first=str(self.script_path))

    def _run_analyze_cell_areas_in_process(self) -> bool:
        """Run analyze_cell_areas via -m in this thread. cmd_args = [module_name, ...cli_args]."""
        return self._run_module_main_in_process("bactimate.scripts.analyze_cell_areas", "analyze_cell_areas")

    def _run_module_main_in_process(self, module_path: str, argv0: str) -> bool:
        """Run a module's main() in this thread. cmd_args = [module_name, ...cli_args]. argv0 = first argv for the script."""
        old_argv = list(sys.argv)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        buf = io.StringIO()
        try:
            sys.argv = [argv0] + [str(a) for a in (self.cmd_args[1:] if len(self.cmd_args) > 1 else [])]
            sys.stdout = buf
            sys.stderr = buf
            try:
                import importlib
                mod = importlib.import_module(module_path)
                main_fn = getattr(mod, "main")
                code = main_fn()
                if code is None:
                    code = 0
            except SystemExit as e:
                c = getattr(e, "code", None)
                code = c if isinstance(c, int) else (0 if c is None else 1)
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            output = buf.getvalue()
            for line in output.splitlines():
                line = line.strip()
                if line:
                    self.progress.emit(self.strip_ansi_codes(line) + "\n")
            self.finished.emit(code == 0, output)
            return True
        except Exception as e:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            err_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            write_crash_log("Worker (%s in-process) failed:\n%s" % (argv0, err_msg))
            self.finished.emit(False, err_msg)
            return True

    def _is_running_from_app_bundle(self) -> bool:
        """True when app root is inside a .app bundle (e.g. Contents/Resources)."""
        try:
            root = get_app_root()
            return "Contents/Resources" in str(root) or (root.parent / "MacOS").is_dir()
        except Exception:
            return False

    def run(self) -> None:
        try:
            # When running from .app (or frozen): subprocess often has no numpy. Run scripts in-process.
            # Use sys.frozen, app-bundle path, or script path inside BactiMate.app so in-process works.
            from_app = self._is_running_from_app_bundle()
            frozen = getattr(sys, "frozen", False) or getattr(sys, "__compiled__", False)
            sp = Path(self.script_path) if self.script_path else Path("")
            sp_str = str(self.script_path)
            script_in_app = "BactiMate.app" in sp_str and "Contents" in sp_str
            if from_app or frozen or script_in_app:
                if "convert_nd2_to_tif" in sp.name or "convert_nd2_to_tif" in sp_str:
                    self._run_convert_nd2_in_process()
                    return
                if "batch_process_tiff_sequence" in sp.name or "batch_process_tiff_sequence" in sp_str:
                    self._run_batch_tiff_in_process()
                    return
                if self.script_path == "-m" and self.cmd_args:
                    first = str(self.cmd_args[0])
                    if "analyze_cell_areas" in first:
                        self._run_analyze_cell_areas_in_process()
                        return
                    if "unet_analyze_patch_config" in first:
                        self._run_module_main_in_process("bactimate.scripts.unet_analyze_patch_config", "unet_analyze_patch_config")
                        return
                    if "unet_train_patches" in first:
                        self._run_module_main_in_process("bactimate.scripts.unet_train_patches", "unet_train_patches")
                        return
                    if "unet_predict_patches" in first:
                        self._run_module_main_in_process("bactimate.scripts.unet_predict_patches", "unet_predict_patches")
                        return
            python_exe = get_python_executable_for_subprocess()
            # Never use the .app launcher as subprocess Python: it would start a second GUI. Force in-process for our scripts.
            exe_str = str(python_exe).lower()
            if "bactimate" in exe_str and ("macos" in exe_str or exe_str.endswith("bactimate")):
                if "convert_nd2_to_tif" in sp.name or "convert_nd2_to_tif" in sp_str:
                    self._run_convert_nd2_in_process()
                    return
                if "batch_process_tiff_sequence" in sp.name or "batch_process_tiff_sequence" in sp_str:
                    self._run_batch_tiff_in_process()
                    return
            cmd = [python_exe, self.script_path] + self.cmd_args
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["TQDM_DISABLE"] = "1"
            # On Windows, subprocess uses cp1252 for stdout; script print() with ✓/✗ then raises
            # charmap encode error. Force child to use UTF-8 so conversion (and other scripts) can print safely.
            popen_kw: dict = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                env=env,
            )
            if sys.platform == "win32":
                env["PYTHONIOENCODING"] = "utf-8"
                popen_kw["encoding"] = "utf-8"
                popen_kw["errors"] = "replace"

            process = subprocess.Popen(cmd, **popen_kw)
            self._process = process

            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                line = line.rstrip()
                if line:
                    cleaned_line = self.strip_ansi_codes(line)
                    if self.is_tqdm_progress_line(cleaned_line):
                        # Emit one short progress line so the status log fits in one line
                        import re
                        stripped = self.strip_log_prefix(cleaned_line)
                        m = re.match(r"^(\d+)%\s*\|[#\s]+\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]", stripped)
                        if m:
                            pct, cur, tot, rest = m.groups()
                            eta = re.search(r"<(\d+):(\d+)", rest)
                            short = f"  {pct}% ({cur}/{tot})  ETA {eta.group(1)}:{eta.group(2)}" if eta else f"  {pct}% ({cur}/{tot})"
                            self.progress.emit(short + "\n")
                        continue
                    content = self.strip_log_prefix(cleaned_line) or cleaned_line.strip()
                    if content:
                        output_lines.append(cleaned_line)
                        self.progress.emit(content + "\n")

            process.wait()
            self._process = None
            output = "\n".join(output_lines)
            if process.returncode == 0:
                self.finished.emit(True, output)
            else:
                write_crash_log("Worker script failed (exit %s):\n%s" % (process.returncode, output))
                self.finished.emit(False, output)
        except Exception as e:
            self._process = None
            log_exception(get_logger(), "WorkerThread.run failed")
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            write_crash_log("WorkerThread.run exception:\n" + error_msg)
            self.finished.emit(False, error_msg)


class TrackingWorker(QThread):
    """Worker thread for running tracking"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)  # progress percentage, status message
    
    def __init__(self, mask_file, original_file, excel_file, model_name, mode, device, parent=None, output_dir=None):
        super().__init__(parent)
        self.mask_file = mask_file
        self.original_file = original_file
        self.excel_file = excel_file
        self.model_name = model_name
        self.mode = mode
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None

    def run(self) -> None:
        # Initialize debug output list
        debug_output = []
        debug_output.append("=== Tracking Debug Information ===")
        
        try:
            # NumPy 2.0+ removed np.bool; geff (trackastra dependency) still uses it in typing
            import numpy as np
            if not hasattr(np, "bool"):
                np.bool = np.bool_
            from trackastra.model import Trackastra
            from trackastra.tracking import graph_to_edge_table, graph_to_ctc, graph_to_napari_tracks
            import tifffile
            import pandas as pd
            from pathlib import Path
            import networkx as nx
            
            debug_output.append("✓ All imports successful")
            self.progress.emit(5, "Loading masks...")
            
            # Load masks
            debug_output.append(f"\nLoading masks from: {self.mask_file}")
            masks = tifffile.imread(str(self.mask_file))
            debug_output.append(f"✓ Masks loaded - Shape: {masks.shape}")
            self.progress.emit(10, "Masks loaded")
            
            if len(masks.shape) != 3:
                error_msg = f"Expected 3D mask array (T, Y, X), got shape: {masks.shape}\n\n" + "\n".join(debug_output)
                self.finished.emit(False, error_msg)
                return
            
            # Load original images if provided (required for trackastra)
            if self.original_file:
                self.progress.emit(15, "Loading original images...")
                debug_output.append(f"\nLoading original images from: {self.original_file}")
                imgs = tifffile.imread(str(self.original_file))
                debug_output.append(f"✓ Images loaded - Shape: {imgs.shape}")
                self.progress.emit(20, "Images loaded")
                
                if len(imgs.shape) != 3:
                    error_msg = f"Expected 3D image array (T, Y, X), got shape: {imgs.shape}\n\n" + "\n".join(debug_output)
                    self.finished.emit(False, error_msg)
                    return
                # Ensure images and masks have same number of frames
                if imgs.shape[0] != masks.shape[0]:
                    error_msg = f"Image frames ({imgs.shape[0]}) don't match mask frames ({masks.shape[0]})\n\n" + "\n".join(debug_output)
                    self.finished.emit(False, error_msg)
                    return
            else:
                # If no original images provided, create dummy images (trackastra needs them)
                # Use masks as images (will work but not ideal)
                self.progress.emit(15, "Creating images from masks...")
                debug_output.append("\nNo original images provided - using masks as images")
                imgs = masks.astype(np.float32)
                debug_output.append("✓ Created dummy images from masks")
                self.progress.emit(20, "Images ready")
            
            # Load Excel data if provided
            cell_data = None
            if self.excel_file:
                self.progress.emit(25, "Loading Excel analysis data...")
                try:
                    excel_path = Path(self.excel_file)
                    debug_output.append(f"\nLoading Excel file: {excel_path}")
                    # Try to read the per-cell data sheet
                    excel_data = pd.read_excel(excel_path, sheet_name=None)
                    debug_output.append(f"✓ Excel file loaded with sheets: {list(excel_data.keys())}")
                    
                    # Look for sheet with per-cell data (usually named "Per-Cell Data" or similar)
                    for sheet_name in excel_data.keys():
                        if 'cell' in sheet_name.lower() or 'per' in sheet_name.lower():
                            cell_data = excel_data[sheet_name]
                            debug_output.append(f"✓ Using sheet: {sheet_name} (per-cell data)")
                            break
                    if cell_data is None:
                        # Use first sheet if no per-cell sheet found
                        cell_data = list(excel_data.values())[0]
                        debug_output.append(f"✓ Using first sheet: {list(excel_data.keys())[0]}")
                    
                    debug_output.append(f"  - Cell data shape: {cell_data.shape}")
                    debug_output.append(f"  - Cell data columns: {list(cell_data.columns)}")
                    if len(cell_data) > 0:
                        debug_output.append(f"  - Sample row: {cell_data.iloc[0].to_dict()}")
                    self.progress.emit(30, "Excel data loaded")
                except Exception as e:
                    import traceback
                    error_msg = f"Error reading Excel file: {str(e)}\n\n{traceback.format_exc()}"
                    self.finished.emit(False, error_msg)
                    return
            
            # Load pretrained model with detailed debugging
            self.progress.emit(35, f"Loading Trackastra model ({self.model_name})...")
            debug_output.append(f"\nLoading Trackastra model: {self.model_name}")
            debug_output.append(f"Device: {self.device}")
            
            try:
                import torch
                if self.device == 'cuda':
                    debug_output.append(f"CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        debug_output.append(f"GPU: {torch.cuda.get_device_name(0)}")
                        debug_output.append(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                elif self.device == 'mps':
                    debug_output.append(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        debug_output.append("GPU: Apple Silicon (MPS)")
                else:
                    debug_output.append("Using CPU")
            except:
                pass
            
            try:
                debug_output.append("Calling Trackastra.from_pretrained()...")
                model = Trackastra.from_pretrained(self.model_name, device=self.device)
                debug_output.append("✓ Model loaded successfully")
                self.progress.emit(50, "Model loaded, starting tracking...")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                debug_output.append(f"✗ Error loading model: {str(e)}")
                debug_output.append(f"\nFull traceback:\n{error_details}")
                
                error_msg = f"Error loading Trackastra model: {str(e)}\n\n"
                error_msg += "Debug information:\n" + "\n".join(debug_output)
                error_msg += "\n\nPossible issues:"
                error_msg += "\n- Model name incorrect"
                error_msg += "\n- Trackastra not properly installed"
                error_msg += "\n- CUDA/GPU issues (if using cuda device)"
                error_msg += "\n- Network issues (downloading model)"
                
                self.finished.emit(False, error_msg)
                return
            
            # Run tracking with detailed debugging
            debug_output.append(f"\nStarting tracking with mode: {self.mode}")
            debug_output.append(f"Input shapes - Images: {imgs.shape}, Masks: {masks.shape}")
            
            try:
                import torch
                if self.device == 'cuda' and torch.cuda.is_available():
                    debug_output.append(f"GPU memory before tracking: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                    debug_output.append(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    debug_output.append("MPS device ready for tracking")
            except:
                pass
            
            # Initialize trackid_to_cellid mapping (will be set if Excel file is provided)
            trackid_to_cellid = None
            
            # Forward trackastra logging (Extracting features, Building windows, etc.) to GUI status log
            trackastra_logger = logging.getLogger("trackastra")
            log_handler = _TrackingLogHandler(self, 55)
            trackastra_logger.addHandler(log_handler)
            try:
                self.progress.emit(
                    55,
                    "Running tracking algorithm (feature extraction & linking — this may take several minutes, please wait)..."
                )
                debug_output.append("Calling model.track()...")
                # Run model.track() in a background thread so we can emit "still running" updates
                result_holder = [None, None, None]  # track_graph, masks_tracked, exception
                def run_track():
                    try:
                        g, m = model.track(imgs, masks, mode=self.mode)
                        result_holder[0], result_holder[1] = g, m
                    except Exception as e:
                        result_holder[2] = e
                tracking_thread = threading.Thread(target=run_track, daemon=False)
                tracking_thread.start()
                elapsed_sec = 0
                update_interval_sec = 15
                while tracking_thread.is_alive():
                    time.sleep(update_interval_sec)
                    elapsed_sec += update_interval_sec
                    if elapsed_sec < 60:
                        time_str = f"{elapsed_sec} s"
                    else:
                        time_str = f"{elapsed_sec // 60} min"
                    self.progress.emit(
                        55,
                        f"Still running… ({time_str}) — feature extraction and linking can take several minutes"
                    )
                tracking_thread.join()
                if result_holder[2] is not None:
                    raise result_holder[2]
                track_graph, masks_tracked = result_holder[0], result_holder[1]
                debug_output.append("✓ Tracking completed successfully")
                self.progress.emit(70, "Tracking completed, processing results...")
                
                # Map track IDs back to analysis CellIDs so one CellID stays with the same cell
                if self.excel_file and Path(self.excel_file).exists():
                    self.progress.emit(72, "Mapping track IDs to analysis CellIDs (from Excel)...")
                    debug_output.append("\nMapping track IDs to analysis CellIDs (from Excel)...")
                    try:
                        masks_tracked, trackid_to_cellid = self._map_track_ids_to_analysis_cellids(
                            masks, masks_tracked, self.excel_file
                        )
                        debug_output.append(f"✓ Mapped {len(trackid_to_cellid)} track IDs to analysis CellIDs")
                    except Exception as e:
                        import traceback
                        debug_output.append(f"⚠ Could not map track IDs to analysis CellIDs: {str(e)}")
                        debug_output.append(f"  {traceback.format_exc()}")
                        trackid_to_cellid = None
                else:
                    # No Excel: preserve CellIDs from input masks when they already have them (e.g. from Analyze Masks)
                    total_unique = len(np.unique(masks))
                    per_frame_uniques = [len(np.unique(masks[i])) for i in range(masks.shape[0])]
                    avg_per_frame = sum(per_frame_uniques) / len(per_frame_uniques) if per_frame_uniques else 0
                    # Persistent IDs reuse the same labels across frames; per-frame labels do not
                    input_looks_persistent = avg_per_frame > 0 and total_unique <= 2.0 * avg_per_frame
                    if input_looks_persistent:
                        self.progress.emit(72, "Preserving CellIDs from input masks...")
                        try:
                            masks_tracked, trackid_to_cellid = self._map_track_ids_to_cellids_from_input_masks(
                                masks, masks_tracked
                            )
                            if trackid_to_cellid:
                                debug_output.append(f"✓ Preserved {len(trackid_to_cellid)} CellIDs from input masks")
                            else:
                                trackid_to_cellid = None
                        except Exception as e:
                            debug_output.append(f"  (Could not preserve CellIDs from input: {e})")
                            trackid_to_cellid = None
                    else:
                        trackid_to_cellid = None
                debug_output.append(f"  - Track graph type: {type(track_graph)}")
                debug_output.append(f"  - Track graph nodes: {track_graph.number_of_nodes() if hasattr(track_graph, 'number_of_nodes') else 'N/A'}")
                debug_output.append(f"  - Track graph edges: {track_graph.number_of_edges() if hasattr(track_graph, 'number_of_edges') else 'N/A'}")
                debug_output.append(f"  - Tracked masks shape: {masks_tracked.shape}")
                debug_output.append(f"  - Tracked masks dtype: {masks_tracked.dtype}")
                debug_output.append(f"  - Original masks max: {masks.max()}")
                debug_output.append(f"  - Tracked masks max: {masks_tracked.max()}")
                debug_output.append(f"  - Unique labels in first frame (original): {len(np.unique(masks[0]))}")
                debug_output.append(f"  - Unique labels in first frame (tracked): {len(np.unique(masks_tracked[0]))}")
                debug_output.append(f"  - Unique labels in last frame (original): {len(np.unique(masks[-1]))}")
                debug_output.append(f"  - Unique labels in last frame (tracked): {len(np.unique(masks_tracked[-1]))}")
                # Check if track IDs are consistent across frames
                if masks_tracked.shape[0] > 1:
                    unique_tracks = set()
                    for i in range(masks_tracked.shape[0]):
                        unique_tracks.update(np.unique(masks_tracked[i]))
                    debug_output.append(f"  - Total unique track IDs across all frames: {len(unique_tracks)}")
                    # Check if masks changed (should have consistent track IDs)
                    if np.array_equal(masks, masks_tracked):
                        debug_output.append("  ⚠ WARNING: Tracked masks are identical to input masks!")
                        debug_output.append("    This suggests track IDs were not properly assigned.")
                    else:
                        debug_output.append("  ✓ Tracked masks differ from input (track IDs assigned)")
                
                try:
                    import torch
                    if self.device == 'cuda' and torch.cuda.is_available():
                        debug_output.append(f"GPU memory after tracking: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                        torch.cuda.empty_cache()
                        debug_output.append("✓ GPU cache cleared")
                    elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        debug_output.append("✓ MPS cache cleared")
                except:
                    pass
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                debug_output.append(f"✗ Error during tracking: {str(e)}")
                debug_output.append(f"\nFull traceback:\n{error_details}")
                
                try:
                    import torch
                    if self.device == 'cuda' and torch.cuda.is_available():
                        debug_output.append(f"\nGPU memory at error: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                        debug_output.append(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                    elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        debug_output.append("\nMPS device error occurred")
                except:
                    pass
                
                error_msg = f"Error during tracking: {str(e)}\n\n"
                error_msg += "Debug information:\n" + "\n".join(debug_output)
                error_msg += "\n\nPossible issues:"
                error_msg += "\n- GPU out of memory (try CPU mode)"
                error_msg += "\n- CUDA runtime error"
                error_msg += "\n- Input data format issues"
                error_msg += "\n- Model compatibility issues"
                
                self.finished.emit(False, error_msg)
                return
            finally:
                trackastra_logger.removeHandler(log_handler)
            
            # Save tracking results
            mask_path = Path(self.mask_file)
            if self.output_dir is not None:
                output_dir = self.output_dir
            else:
                # Fallback: determine base directory (go up one level if in Tif_*, _segmentation, or _analysis)
                base_dir = mask_path.parent
                if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name or base_dir.name == '_analysis':
                    base_dir = base_dir.parent
                output_dir = base_dir / "_tracking" / f"{mask_path.stem}_tracking"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Read calibration from original or mask TIFF so we can preserve it in written TIFFs
            from bactimate.modules.tiff_metadata import read_calibration_from_path, build_metadata_dict
            _cal_path = Path(self.original_file) if self.original_file else (Path(self.mask_file) if self.mask_file else None)
            self._cal_ps, self._cal_tp = read_calibration_from_path(_cal_path)
            
            # Extract track information from the graph
            self.progress.emit(75, "Extracting track information...")
            debug_output.append("\nExtracting track information from graph...")
            
            # Method 1: Use graph_to_napari_tracks to get proper track data
            try:
                napari_tracks, napari_graph, _ = graph_to_napari_tracks(track_graph)
                debug_output.append(f"✓ Extracted {len(napari_tracks)} track points from graph")
                debug_output.append(f"  - Napari tracks shape: {napari_tracks.shape if hasattr(napari_tracks, 'shape') else 'N/A'}")
                
                # Convert napari tracks to DataFrame
                # napari_tracks is numpy array with columns: [track_id, frame, y, x, ...]
                if len(napari_tracks) > 0 and hasattr(napari_tracks, 'shape'):
                    # napari_tracks is typically (N, 4+) array: [track_id, frame, y, x, ...]
                    num_cols = napari_tracks.shape[1] if len(napari_tracks.shape) > 1 else 4
                    col_names = ['track_id', 'frame', 'y', 'x']
                    if num_cols > 4:
                        col_names.extend([f'property_{i}' for i in range(num_cols - 4)])
                    
                    tracks_df = pd.DataFrame(napari_tracks, columns=col_names[:num_cols])
                    # Ensure track_id and frame are integers
                    if 'track_id' in tracks_df.columns:
                        tracks_df['track_id'] = tracks_df['track_id'].astype(int)
                    if 'frame' in tracks_df.columns:
                        tracks_df['frame'] = tracks_df['frame'].astype(int)
                    debug_output.append(f"✓ Created tracks DataFrame with {len(tracks_df)} rows")
                    debug_output.append(f"  - Columns: {list(tracks_df.columns)}")
                else:
                    debug_output.append("⚠ Warning: No tracks extracted from graph or invalid format")
                    tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
            except Exception as e:
                debug_output.append(f"⚠ graph_to_napari_tracks failed: {str(e)}")
                # Fallback: Extract tracks manually from graph
                debug_output.append("Trying manual track extraction from graph...")
                tracks_list = []
                
                # Extract all nodes and their properties
                for node_id, node_data in track_graph.nodes(data=True):
                    # Node data typically has 'time' or 'frame' and position info
                    frame = node_data.get('time', node_data.get('frame', node_data.get('t', None)))
                    if frame is not None:
                        track_id = node_data.get('track_id', node_id)
                        y = node_data.get('y', node_data.get('centroid_y', 0))
                        x = node_data.get('x', node_data.get('centroid_x', 0))
                        tracks_list.append({
                            'track_id': track_id,
                            'frame': int(frame),
                            'y': y,
                            'x': x,
                            'node_id': node_id
                        })
                
                if tracks_list:
                    tracks_df = pd.DataFrame(tracks_list)
                    debug_output.append(f"✓ Manually extracted {len(tracks_df)} track points")
                else:
                    # Last resort: use edge table
                    debug_output.append("Trying graph_to_edge_table as fallback...")
                    try:
                        edge_df = graph_to_edge_table(track_graph)
                        debug_output.append(f"✓ Edge table has {len(edge_df)} edges")
                        # Try to extract track info from edges
                        if len(edge_df) > 0:
                            tracks_df = edge_df
                        else:
                            tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
                    except Exception as e2:
                        debug_output.append(f"✗ All extraction methods failed: {str(e2)}")
                        tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
            
            # Add analysis Cell_ID when we have an Excel from Analyze Masks (Omnipose/U-Net → Analyze Masks → Excel in _analysis)
            if trackid_to_cellid is not None and len(trackid_to_cellid) > 0:
                if 'track_id' in tracks_df.columns:
                    tracks_df['Cell_ID'] = tracks_df['track_id'].map(trackid_to_cellid)
                    # Fill NaN with original track_id for unmapped tracks
                    tracks_df['Cell_ID'] = tracks_df['Cell_ID'].fillna(tracks_df['track_id']).astype(int)
                    debug_output.append(f"✓ Updated tracks DataFrame with analysis CellIDs ({len(trackid_to_cellid)} mappings)")
            
            # Add debug info about tracks
            debug_output.append(f"\nTrack DataFrame info:")
            debug_output.append(f"  - Shape: {tracks_df.shape}")
            debug_output.append(f"  - Columns: {list(tracks_df.columns)}")
            if len(tracks_df) > 0:
                debug_output.append(f"  - Unique track IDs: {tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 'N/A'}")
                if 'Cell_ID' in tracks_df.columns:
                    debug_output.append(f"  - Unique Cell IDs: {tracks_df['Cell_ID'].nunique()}")
                debug_output.append(f"  - Frame range: {tracks_df['frame'].min() if 'frame' in tracks_df.columns else 'N/A'} to {tracks_df['frame'].max() if 'frame' in tracks_df.columns else 'N/A'}")
            
            # Save tracks as CSV: track_id, frame, y, x (and Cell_ID if analysis Excel was used).
            # y,x come from Trackastra's track graph (centroid-like node positions; napari format or node centroid_y/x).
            self.progress.emit(80, "Saving tracks...")
            tracks_file = output_dir / f"{Path(self.mask_file).stem}_tracks.csv"
            tracks_df.to_csv(tracks_file, index=False)
            debug_output.append(f"✓ Saved tracks to: {tracks_file}")
            
            # Save tracked masks
            self.progress.emit(85, "Saving tracked masks...")
            masks_tracked_file = output_dir / f"{Path(self.mask_file).stem}_masks_tracked.tif"
            debug_output.append(f"\nSaving tracked masks...")
            debug_output.append(f"  - Shape: {masks_tracked.shape}")
            debug_output.append(f"  - Unique labels per frame: {[len(np.unique(masks_tracked[i])) for i in range(min(5, masks_tracked.shape[0]))]}")
            T = masks_tracked.shape[0]
            _meta = build_metadata_dict(
                pixel_size_um=self._cal_ps if self._cal_ps > 0 else None,
                time_per_frame_min=self._cal_tp if self._cal_tp > 0 else None,
                images=T, frames=T,
            )
            tifffile.imwrite(str(masks_tracked_file), masks_tracked, metadata=_meta, imagej=True)
            debug_output.append(f"✓ Saved tracked masks to: {masks_tracked_file}")
            
            # Create overlay with track IDs (consistent CellIDs across all outputs)
            overlay_array = None
            if self.original_file:
                self.progress.emit(87, "Creating overlay with track IDs...")
                debug_output.append("\nCreating overlay with track IDs...")
                try:
                    overlay_file = output_dir / f"{Path(self.mask_file).stem}_overlay_tracked.tif"
                    def overlay_progress(frame_idx, total, _msg=None):
                        pct = 87 + int((frame_idx + 1) / max(total, 1) * 2)  # 87–89%
                        msg = f"Creating overlay: {frame_idx + 1}/{total}"
                        self.progress.emit(pct, msg)
                    overlay_array = self._create_overlay_with_track_ids(
                        imgs, masks_tracked, overlay_file, progress_callback=overlay_progress
                    )
                    debug_output.append(f"✓ Saved tracked overlay to: {overlay_file}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not create tracked overlay: {str(e)}")
                    debug_output.append(f"  {traceback.format_exc()}")
            
            # Create visualization with trajectories overlaid (use overlay as base when available — much faster)
            if self.original_file:
                self.progress.emit(90, "Creating trajectory visualization...")
                debug_output.append("\nCreating trajectory visualization...")
                if overlay_array is not None:
                    debug_output.append("  (Using overlay as base — drawing trajectories only.)")
                else:
                    debug_output.append("  (This may take a while for large datasets - optimizing...)")
                try:
                    vis_file = output_dir / f"{Path(self.mask_file).stem}_trajectories.tif"
                    def progress_callback(pct, msg):
                        self.progress.emit(pct, msg)
                    self._create_trajectory_visualization(
                        imgs, masks_tracked, tracks_df, vis_file, progress_callback,
                        overlay_frames=overlay_array,
                    )
                    debug_output.append(f"✓ Saved trajectory visualization to: {vis_file}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not create trajectory visualization: {str(e)}")
                    debug_output.append(f"  {traceback.format_exc()}")
                    self.progress.emit(96, f"WARNING: Trajectory file was not created: {e}")
            
            # Save in CTC format (Cell Tracking Challenge format)
            try:
                ctc_dir = output_dir / "ctc_format"
                ctc_dir.mkdir(exist_ok=True)
                graph_to_ctc(track_graph, masks_tracked, outdir=str(ctc_dir))
            except Exception as e:
                # CTC format is optional, don't fail if it doesn't work
                pass
            
            # If Excel data was provided, merge with tracks
            if cell_data is not None:
                self.progress.emit(95, "Merging with analysis data...")
                # Merge tracking data with cell properties
                merged_data = self._merge_tracking_with_analysis(tracks_df, cell_data, masks_tracked)
                merged_file = output_dir / f"{Path(self.mask_file).stem}_tracks_with_analysis.xlsx"
                with pd.ExcelWriter(merged_file, engine='openpyxl') as writer:
                    merged_data.to_excel(writer, sheet_name='Tracks with Analysis', index=False)
                    tracks_df.to_excel(writer, sheet_name='Tracks', index=False)
                    if isinstance(cell_data, pd.DataFrame):
                        cell_data.to_excel(writer, sheet_name='Original Analysis', index=False)
                self.progress.emit(100, "Tracking completed!")
                # Add debug output to success message
                success_msg = "Tracking completed!\n\n"
                success_msg += "\n".join(debug_output)
                success_msg += f"\n\nTracks saved to: {tracks_file}\n"
                success_msg += f"Tracked masks saved to: {masks_tracked_file}\n"
                success_msg += f"Merged data saved to: {merged_file}"
                self.finished.emit(True, success_msg)
            else:
                self.progress.emit(100, "Tracking completed!")
                # Add debug output to success message
                success_msg = "Tracking completed!\n\n"
                success_msg += "\n".join(debug_output)
                success_msg += f"\n\nTracks saved to: {tracks_file}\n"
                success_msg += f"Tracked masks saved to: {masks_tracked_file}"
                self.finished.emit(True, success_msg)
                
        except ImportError as e:
            debug_output.append(f"\n✗ Import error: {str(e)}")
            error_msg = f"trackastra is not installed or import failed: {str(e)}\n\n"
            error_msg += "Debug information:\n" + "\n".join(debug_output)
            err_str = str(e).lower()
            if "cbuffer_sizes" in err_str and "numcodecs" in err_str:
                error_msg += (
                    "\n\nThis is a known numcodecs/zarr compatibility issue. Try one of:\n"
                    "  1. Force-reinstall numcodecs:  pip install --force-reinstall numcodecs==0.15.1\n"
                    "  2. Or use zarr 3 (no cbuffer_sizes):  pip install 'zarr>=3.0'\n"
                )
            else:
                error_msg += "\n\nPlease install trackastra: pip install trackastra"
            self.finished.emit(False, error_msg)
        except Exception as e:
            log_exception(get_logger(), "TrackingWorker.run failed")
            import traceback
            debug_output.append(f"\n✗ Unexpected error: {str(e)}")
            debug_output.append(f"\nFull traceback:\n{traceback.format_exc()}")
            error_msg = f"Tracking error: {str(e)}\n\n"
            error_msg += "Debug information:\n" + "\n".join(debug_output)
            self.finished.emit(False, error_msg)
    
    def _map_track_ids_to_analysis_cellids(self, masks_original, masks_tracked, excel_file):
        """
        Map track IDs back to analysis CellIDs by matching cells between original and tracked masks.
        
        Parameters:
        -----------
        masks_original : array
            Original masks (may have analysis CellIDs if from analyzed masks)
        masks_tracked : array
            Tracked masks with track IDs
        excel_file : str
            Path to Excel file with analysis data containing Cell_ID column
            
        Returns:
        --------
        masks_tracked_remapped : array
            Tracked masks remapped to use analysis CellIDs
        trackid_to_cellid : dict
            Mapping from track_id to analysis CellID
        """
        import pandas as pd
        import numpy as np
        from skimage.measure import regionprops
        from scipy.spatial.distance import cdist
        
        # Load Excel data to get Cell_IDs
        try:
            excel_data = pd.read_excel(excel_file, sheet_name='Per-Cell Data')
        except:
            # Try other sheet names
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=0)
            except:
                return masks_tracked, {}
        
        # Find Cell_ID column
        cell_id_col = None
        for col in ['Cell_ID', 'cell_id', 'Cell ID', 'CellID']:
            if col in excel_data.columns:
                cell_id_col = col
                break
        
        if cell_id_col is None:
            # No Cell_ID column found, return original tracked masks
            return masks_tracked, {}
        
        # Find frame column
        frame_col = None
        for col in ['Frame', 'frame', 't', 'time', 'Time']:
            if col in excel_data.columns:
                frame_col = col
                break
        
        if frame_col is None:
            return masks_tracked, {}
        
        # Build mapping: (frame, original_label) -> Cell_ID from Excel
        excel_mapping = {}
        for _, row in excel_data.iterrows():
            frame = int(row[frame_col]) - 1  # Convert to 0-based
            cell_id = row[cell_id_col]
            # Try to find original label (might be in 'Cell Label' column)
            if 'Cell Label' in excel_data.columns:
                original_label = row['Cell Label']
            else:
                # Can't map without original label
                continue
            if pd.notna(cell_id) and pd.notna(original_label):
                excel_mapping[(frame, int(original_label))] = int(cell_id)
        
        if not excel_mapping:
            return masks_tracked, {}
        
        # Now map track IDs to analysis CellIDs by matching cells
        # Strategy: Match cells in each frame between original and tracked masks
        trackid_to_cellid = {}
        num_frames = min(masks_original.shape[0], masks_tracked.shape[0])
        max_distance = 10.0  # pixels
        
        for frame_idx in range(num_frames):
            frame_original = masks_original[frame_idx]
            frame_tracked = masks_tracked[frame_idx]
            
            # Get centroids for both
            regions_original = regionprops(frame_original)
            regions_tracked = regionprops(frame_tracked)
            
            # Build mapping for this frame: original_label -> CellID
            orig_label_to_cellid = {}
            for region in regions_original:
                if region.label > 0:
                    if (frame_idx, region.label) in excel_mapping:
                        orig_label_to_cellid[region.label] = excel_mapping[(frame_idx, region.label)]
            
            if not orig_label_to_cellid:
                continue
            
            # Match tracked cells to original cells by position
            orig_centroids = {}
            for region in regions_original:
                if region.label > 0 and region.label in orig_label_to_cellid:
                    orig_centroids[region.label] = np.array(region.centroid)
            
            tracked_centroids = {}
            for region in regions_tracked:
                if region.label > 0:
                    tracked_centroids[region.label] = np.array(region.centroid)
            
            if len(orig_centroids) > 0 and len(tracked_centroids) > 0:
                orig_points = np.array(list(orig_centroids.values()))
                orig_labels = list(orig_centroids.keys())
                tracked_points = np.array(list(tracked_centroids.values()))
                tracked_labels = list(tracked_centroids.keys())
                
                distances = cdist(orig_points, tracked_points)
                
                # Greedy assignment
                used_orig = set()
                used_tracked = set()
                assignments = []
                for i, orig_label in enumerate(orig_labels):
                    for j, track_id in enumerate(tracked_labels):
                        if distances[i, j] <= max_distance:
                            assignments.append((distances[i, j], orig_label, track_id))
                
                assignments.sort()
                
                for dist, orig_label, track_id in assignments:
                    if orig_label not in used_orig and track_id not in used_tracked:
                        # Map track_id to analysis CellID
                        cell_id = orig_label_to_cellid[orig_label]
                        trackid_to_cellid[track_id] = cell_id
                        used_orig.add(orig_label)
                        used_tracked.add(track_id)
        
        # Remap tracked masks to use analysis CellIDs
        masks_tracked_remapped = np.zeros_like(masks_tracked)
        for frame_idx in range(num_frames):
            frame_tracked = masks_tracked[frame_idx]
            frame_remapped = masks_tracked_remapped[frame_idx]
            
            for track_id, cell_id in trackid_to_cellid.items():
                frame_remapped[frame_tracked == track_id] = cell_id
        
        return masks_tracked_remapped, trackid_to_cellid
    
    def _map_track_ids_to_cellids_from_input_masks(self, masks_original, masks_tracked):
        """
        Preserve CellIDs from input masks when they already contain persistent IDs
        (e.g. from Analyze Masks). Build track_id -> cell_id by sampling the original
        mask at each track's centroid per frame and using the most common value.
        No Excel required.
        """
        import numpy as np
        from skimage.measure import regionprops
        from collections import Counter
        
        num_frames = min(masks_original.shape[0], masks_tracked.shape[0])
        track_id_to_values = {}  # track_id -> list of (frame_idx, original_label at centroid)
        
        for frame_idx in range(num_frames):
            frame_orig = masks_original[frame_idx]
            frame_tracked = masks_tracked[frame_idx]
            h, w = frame_orig.shape[:2]
            for region in regionprops(frame_tracked):
                if region.label == 0:
                    continue
                track_id = region.label
                r, c = int(round(region.centroid[0])), int(round(region.centroid[1]))
                if 0 <= r < h and 0 <= c < w:
                    orig_val = int(frame_orig[r, c])
                    if track_id not in track_id_to_values:
                        track_id_to_values[track_id] = []
                    track_id_to_values[track_id].append(orig_val)
        
        trackid_to_cellid = {}
        for track_id, values in track_id_to_values.items():
            # Exclude background (0); take most common label as CellID
            non_zero = [v for v in values if v != 0]
            if not non_zero:
                continue
            mode_val, count = Counter(non_zero).most_common(1)[0]
            # Only assign if the same ID appears in at least half of the samples (persistent)
            if count >= max(1, len(values) // 2):
                trackid_to_cellid[track_id] = mode_val
        
        if not trackid_to_cellid:
            return masks_tracked, {}
        
        # Remap tracked masks to use those CellIDs (same as Excel path)
        masks_tracked_remapped = np.zeros_like(masks_tracked)
        for frame_idx in range(num_frames):
            frame_tracked = masks_tracked[frame_idx]
            frame_remapped = masks_tracked_remapped[frame_idx]
            for track_id, cell_id in trackid_to_cellid.items():
                frame_remapped[frame_tracked == track_id] = cell_id
            # Preserve track_ids that were not mapped (keep original track_id)
            unmapped = (frame_tracked > 0) & (frame_remapped == 0)
            frame_remapped[unmapped] = frame_tracked[unmapped]
        
        return masks_tracked_remapped, trackid_to_cellid
    
    def _merge_tracking_with_analysis(self, tracks_df, cell_data, masks_tracked):
        """Merge tracking results with Excel analysis data"""
        import pandas as pd
        import numpy as np
        from skimage.measure import regionprops
        from scipy.spatial.distance import cdist
        
        # Ensure cell_data is a DataFrame
        if not isinstance(cell_data, pd.DataFrame):
            cell_data = pd.DataFrame(cell_data)
        
        # Debug: Check what columns we have
        debug_info = []
        debug_info.append(f"Tracks DataFrame columns: {list(tracks_df.columns)}")
        debug_info.append(f"Cell data DataFrame columns: {list(cell_data.columns)}")
        debug_info.append(f"Tracks shape: {tracks_df.shape}, Cell data shape: {cell_data.shape}")
        
        # Find frame column in cell_data (could be 'frame', 'Frame', 't', 'time', etc.)
        frame_col = None
        for col in ['frame', 'Frame', 't', 'time', 'Time', 'T']:
            if col in cell_data.columns:
                frame_col = col
                break
        
        if frame_col is None:
            debug_info.append("⚠ Warning: Could not find frame column in cell_data")
            # Return tracks_df with a note
            return tracks_df
        
        debug_info.append(f"Using frame column: {frame_col}")
        
        # Find cell_id column in cell_data
        cell_id_col = None
        for col in ['cell_id', 'Cell_ID', 'cell', 'Cell', 'label', 'Label']:
            if col in cell_data.columns:
                cell_id_col = col
                break
        
        # Extract track information from tracked masks
        # In tracked masks, each label is a track_id that persists across frames
        # We need to map (frame, track_id) to original cell properties from Excel
        
        merged_list = []
        
        # For each track point, find matching cell data
        matches_found = 0
        matches_by_position = 0
        matches_by_first = 0
        
        # Track which frames we've already warned about
        frames_with_no_cells = set()
        
        # Detect if Excel uses 1-based indexing by checking if frame 0 has no data but frame 1 does
        frame_0_count = len(cell_data[cell_data[frame_col].astype(int) == 0]) if frame_col else 0
        frame_1_count = len(cell_data[cell_data[frame_col].astype(int) == 1]) if frame_col else 0
        excel_is_1_indexed = (frame_0_count == 0 and frame_1_count > 0)
        if excel_is_1_indexed:
            debug_info.append(f"Detected 1-based frame indexing in Excel (Frame 0 has {frame_0_count} cells, Frame 1 has {frame_1_count} cells)")
        
        for idx, track_row in tracks_df.iterrows():
            # Get frame - try multiple column names
            frame = None
            for col in ['frame', 't', 'time', 'Frame', 'T', 'Time']:
                if col in track_row.index:
                    frame_val = track_row[col]
                    if pd.notna(frame_val):
                        frame = int(frame_val)
                        break
            
            # Get track_id
            track_id = None
            for col in ['track_id', 'track', 'Track', 'track_ID']:
                if col in track_row.index:
                    track_id_val = track_row[col]
                    if pd.notna(track_id_val):
                        track_id = int(track_id_val)
                        break
            
            if frame is None:
                debug_info.append(f"  Skipping row {idx}: frame is None")
                continue
            if track_id is None:
                debug_info.append(f"  Skipping row {idx}: track_id is None")
                continue
            
            # Get track position from tracks_df
            track_y = None
            track_x = None
            for col in ['y', 'Y', 'centroid_y', 'centroid-0']:
                if col in track_row.index and pd.notna(track_row[col]):
                    track_y = float(track_row[col])
                    break
            for col in ['x', 'X', 'centroid_x', 'centroid-1']:
                if col in track_row.index and pd.notna(track_row[col]):
                    track_x = float(track_row[col])
                    break
            
            # Start with track information
            merged_row = track_row.to_dict()
            
            # Find cells in this frame from cell_data
            # Handle 0-based vs 1-based frame indexing
            # Tracking uses 0-based (0, 1, 2...), Excel might use 1-based (1, 2, 3...)
            frame_cells = pd.DataFrame()
            
            # Use detected indexing preference
            excel_frame = frame + 1 if excel_is_1_indexed else frame
            
            # Try to find cells for this frame
            try:
                frame_cells = cell_data[cell_data[frame_col].astype(int) == excel_frame].copy()
            except:
                try:
                    frame_cells = cell_data[cell_data[frame_col] == excel_frame].copy()
                except:
                    pass
            
            # If still no match and we haven't tried the other indexing, try it
            if len(frame_cells) == 0 and not excel_is_1_indexed:
                try:
                    frame_cells = cell_data[cell_data[frame_col].astype(int) == (frame + 1)].copy()
                    if len(frame_cells) > 0:
                        excel_is_1_indexed = True  # Update detection
                        debug_info.append(f"Detected 1-based frame indexing: Frame {frame} matched to Excel Frame {frame + 1}")
                except:
                    pass
            
            if len(frame_cells) > 0:
                matched = False
                # Try to match by position if we have coordinates
                if track_y is not None and track_x is not None:
                    # Find centroid columns in cell_data
                    y_col = None
                    x_col = None
                    for col in ['centroid_y', 'Centroid_Y', 'y', 'Y', 'centroid-0', 'centroid_y-0']:
                        if col in frame_cells.columns:
                            y_col = col
                            break
                    for col in ['centroid_x', 'Centroid_X', 'x', 'X', 'centroid-1', 'centroid_x-1']:
                        if col in frame_cells.columns:
                            x_col = col
                            break
                    
                    if y_col and x_col:
                        # Calculate distances and find closest match
                        try:
                            distances = np.sqrt(
                                (frame_cells[y_col].astype(float) - track_y)**2 + 
                                (frame_cells[x_col].astype(float) - track_x)**2
                            )
                            closest_idx = distances.idxmin()
                            closest_dist = distances.loc[closest_idx]
                            
                            # If close enough (within 20 pixels), use this match
                            if closest_dist < 20:
                                matched_cell = frame_cells.loc[closest_idx]
                                # Add all cell properties to merged row
                                for col in matched_cell.index:
                                    if col not in merged_row or pd.isna(merged_row.get(col)):
                                        merged_row[col] = matched_cell[col]
                                matches_by_position += 1
                                matched = True
                        except Exception as e:
                            debug_info.append(f"  Error matching by position: {str(e)}")
                
                if not matched:
                    # No position match, use first cell in frame (or match by track_id if available)
                    if cell_id_col and 'track_id' in frame_cells.columns:
                        # Try to match by track_id
                        matching = frame_cells[frame_cells[cell_id_col] == track_id]
                        if len(matching) > 0:
                            matched_cell = matching.iloc[0]
                        else:
                            matched_cell = frame_cells.iloc[0]
                    else:
                        matched_cell = frame_cells.iloc[0]
                    
                    # Add all cell properties to merged row
                    for col in matched_cell.index:
                        if col not in merged_row or pd.isna(merged_row.get(col)):
                            merged_row[col] = matched_cell[col]
                    matches_by_first += 1
                    matched = True
                
                if matched:
                    matches_found += 1
            else:
                # Only log once per frame to reduce noise
                if frame not in frames_with_no_cells:
                    frames_with_no_cells.add(frame)
                    debug_info.append(f"  Frame {frame} (Excel Frame {excel_frame}): No cells found in cell_data")
            
            merged_list.append(merged_row)
        
        debug_info.append(f"\nMerge statistics:")
        debug_info.append(f"  - Total track points: {len(tracks_df)}")
        debug_info.append(f"  - Matches found: {matches_found}")
        debug_info.append(f"  - Matched by position: {matches_by_position}")
        debug_info.append(f"  - Matched by first cell: {matches_by_first}")
        
        if merged_list:
            merged = pd.DataFrame(merged_list)
            debug_info.append(f"✓ Merged DataFrame created with {len(merged)} rows")
        else:
            debug_info.append("⚠ Warning: No merged rows created")
            merged = tracks_df.copy()
        
        # Log debug info
        import sys
        print("\n".join(debug_info), file=sys.stderr)
        
        return merged
    
    def _create_trajectory_visualization(self, images, masks_tracked, tracks_df, output_path, progress_callback=None, overlay_frames=None):
        """Create visualization showing original images with tracked cell IDs and trajectories.
        If overlay_frames is provided (T, H, W, 3), use it as base and only draw trajectories — much faster."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib import cm
        from scipy.ndimage import label, binary_dilation
        import tifffile
        
        if progress_callback is None and hasattr(self, '_progress_callback'):
            progress_callback = self._progress_callback
        
        debug_output = []
        debug_output.append("Creating trajectory visualization...")
        use_overlay_base = overlay_frames is not None and overlay_frames.ndim == 4
        
        if use_overlay_base:
            num_frames, height, width = overlay_frames.shape[0], overlay_frames.shape[1], overlay_frames.shape[2]
            debug_output.append("  - Using overlay as base (trajectories only)")
        else:
            num_frames = images.shape[0]
            height, width = images.shape[1], images.shape[2]
            if progress_callback:
                progress_callback(91, "Normalizing images...")
            img_min = images.min(axis=(1, 2), keepdims=True)
            img_max = images.max(axis=(1, 2), keepdims=True)
            img_range = img_max - img_min
            img_range[img_range == 0] = 1
            images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)
        
        # Get unique track IDs and colormap
        if 'track_id' in tracks_df.columns:
            unique_tracks = tracks_df['track_id'].unique()
            num_tracks = len(unique_tracks)
            debug_output.append(f"  - Found {num_tracks} unique tracks")
        else:
            unique_tracks = np.unique(masks_tracked)
            unique_tracks = unique_tracks[unique_tracks > 0]
            num_tracks = len(unique_tracks)
            debug_output.append(f"  - Found {num_tracks} unique track IDs in masks")
        
        cmap = plt.get_cmap('gist_ncar')
        track_colors = {}
        for i, track_id in enumerate(unique_tracks):
            color = cmap(i / max(num_tracks, 1))
            track_colors[int(track_id)] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        
        visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        
        if 'frame' in tracks_df.columns and 'track_id' in tracks_df.columns:
            tracks_by_frame = {}
            for _, row in tracks_df.iterrows():
                frame = int(row['frame'])
                track_id = int(row['track_id'])
                if frame not in tracks_by_frame:
                    tracks_by_frame[frame] = {}
                if track_id not in tracks_by_frame[frame]:
                    tracks_by_frame[frame][track_id] = []
                y = row.get('y', 0)
                x = row.get('x', 0)
                tracks_by_frame[frame][track_id].append((y, x))
        
        for frame_idx in range(num_frames):
            try:
                if progress_callback:
                    progress_pct = int(90 + (frame_idx / num_frames) * 10)
                    progress_callback(progress_pct, f"Creating visualization: {frame_idx + 1}/{num_frames} ({progress_pct}%)")
                
                if use_overlay_base:
                    vis_frame = np.ascontiguousarray(overlay_frames[frame_idx].copy())
                else:
                    img = images_norm[frame_idx]
                    vis_frame = np.stack([img, img, img], axis=-1)
                    mask_frame = masks_tracked[frame_idx]
                    for track_id in unique_tracks:
                        track_id_int = int(track_id)
                        if track_id_int in track_colors:
                            color = track_colors[track_id_int]
                            mask_region = (mask_frame == track_id_int)
                            vis_frame[mask_region, 0] = np.clip(vis_frame[mask_region, 0] * 0.5 + color[0] * 0.5, 0, 255)
                            vis_frame[mask_region, 1] = np.clip(vis_frame[mask_region, 1] * 0.5 + color[1] * 0.5, 0, 255)
                            vis_frame[mask_region, 2] = np.clip(vis_frame[mask_region, 2] * 0.5 + color[2] * 0.5, 0, 255)
                
                # Draw trajectories up to current frame (only for tracks visible in this frame)
                # Pre-filter tracks to only those present in current frame for speed
                if 'frame' in tracks_df.columns and 'track_id' in tracks_df.columns and 'y' in tracks_df.columns and 'x' in tracks_df.columns:
                    # Get tracks that have points up to current frame
                    tracks_in_frame = tracks_df[
                        (tracks_df['frame'] <= frame_idx) & 
                        (tracks_df['track_id'].isin(unique_tracks))
                    ]['track_id'].unique()
                    
                    for track_id_int in tracks_in_frame:
                        track_id_int = int(track_id_int)
                        if track_id_int not in track_colors:
                            continue
                        
                        # Get all points for this track up to current frame
                        track_points = tracks_df[
                            (tracks_df['track_id'] == track_id_int) & 
                            (tracks_df['frame'] <= frame_idx)
                        ]
                        
                        if len(track_points) > 1:
                            color = track_colors[track_id_int]
                            # Draw lines connecting consecutive points using vectorized operations
                            points = track_points[['y', 'x']].values.astype(int)
                            # Clip points to valid bounds
                            points[:, 0] = np.clip(points[:, 0], 0, height - 1)
                            points[:, 1] = np.clip(points[:, 1], 0, width - 1)
                            
                            # Draw lines between consecutive points using optimized approach
                            # For speed, we'll draw lines more efficiently
                            for i in range(len(points) - 1):
                                y1, x1 = points[i]
                                y2, x2 = points[i + 1]
                                
                                # Calculate line length
                                dy = abs(y2 - y1)
                                dx = abs(x2 - x1)
                                
                                if dx == 0 and dy == 0:
                                    continue
                                
                                # Optimize: Use fewer points for long lines (subsample)
                                max_steps = max(dx, dy)
                                # For very long lines, subsample more aggressively
                                if max_steps > 200:
                                    step_size = max(1, max_steps // 200)  # Very aggressive subsampling
                                elif max_steps > 50:
                                    step_size = max(1, max_steps // 50)  # Moderate subsampling
                                else:
                                    step_size = 1  # Full resolution for short lines
                                
                                num_steps = max_steps // step_size + 1
                                
                                # Generate line points using numpy (much faster than loops)
                                if dx > dy:
                                    # Horizontal line
                                    x_coords = np.linspace(x1, x2, num_steps, dtype=int)
                                    y_coords = np.round(np.linspace(y1, y2, num_steps)).astype(int)
                                else:
                                    # Vertical line
                                    y_coords = np.linspace(y1, y2, num_steps, dtype=int)
                                    x_coords = np.round(np.linspace(x1, x2, num_steps)).astype(int)
                                
                                # Clip coordinates
                                y_coords = np.clip(y_coords, 1, height - 2)  # Keep 1 pixel margin for 3x3 drawing
                                x_coords = np.clip(x_coords, 1, width - 2)
                                
                                # Draw thick line using vectorized operations (batch process all points)
                                # Use advanced indexing for speed
                                for y, x in zip(y_coords, x_coords):
                                    # Direct assignment is faster than clipping each time
                                    y_start, y_end = y - 1, y + 2
                                    x_start, x_end = x - 1, x + 2
                                    vis_frame[y_start:y_end, x_start:x_end, 0] = (
                                        vis_frame[y_start:y_end, x_start:x_end, 0] * 0.3 + color[0] * 0.7
                                    ).astype(np.uint8)
                                    vis_frame[y_start:y_end, x_start:x_end, 1] = (
                                        vis_frame[y_start:y_end, x_start:x_end, 1] * 0.3 + color[1] * 0.7
                                    ).astype(np.uint8)
                                    vis_frame[y_start:y_end, x_start:x_end, 2] = (
                                        vis_frame[y_start:y_end, x_start:x_end, 2] * 0.3 + color[2] * 0.7
                                    ).astype(np.uint8)
                        
                        # Draw track ID label and marker at current position
                        current_track = track_points[track_points['frame'] == frame_idx]
                        if len(current_track) > 0:
                            y = int(current_track.iloc[0]['y'])
                            x = int(current_track.iloc[0]['x'])
                            if 0 <= y < height and 0 <= x < width:
                                # Draw a more visible colored dot with white outline for contrast
                                dot_size = 5  # Larger dot for better visibility
                                outline_size = 2  # White outline thickness
                                
                                # Draw white outline first (for contrast against dark backgrounds)
                                y_outline_start = max(0, y - dot_size - outline_size)
                                y_outline_end = min(height, y + dot_size + outline_size + 1)
                                x_outline_start = max(0, x - dot_size - outline_size)
                                x_outline_end = min(width, x + dot_size + outline_size + 1)
                                
                                # Create outline mask (ring shape)
                                yy, xx = np.ogrid[y_outline_start:y_outline_end, x_outline_start:x_outline_end]
                                center_y, center_x = y - y_outline_start, x - x_outline_start
                                dist_sq = (yy - center_y)**2 + (xx - center_x)**2
                                outline_mask = (dist_sq <= (dot_size + outline_size)**2) & (dist_sq > dot_size**2)
                                
                                # Draw white outline
                                vis_frame[y_outline_start:y_outline_end, x_outline_start:x_outline_end, :][outline_mask] = [255, 255, 255]
                                
                                # Draw colored center dot
                                dot_mask = dist_sq <= dot_size**2
                                vis_frame[y_outline_start:y_outline_end, x_outline_start:x_outline_end, :][dot_mask] = color
                                
                                # Try to add text label if PIL is available
                                try:
                                    from PIL import Image, ImageDraw, ImageFont
                                    # Convert numpy array to PIL Image
                                    pil_img = Image.fromarray(vis_frame)
                                    draw = ImageDraw.Draw(pil_img)
                                    
                                    # Try to use a nice font, fallback to default if not available
                                    try:
                                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                                    except:
                                        try:
                                            font = ImageFont.truetype("arial.ttf", 10)
                                        except:
                                            font = ImageFont.load_default()
                                    
                                    # Draw track ID text with shadow for readability
                                    track_id_str = str(track_id_int)
                                    text_x, text_y = x + dot_size + outline_size + 2, y - 5
                                    
                                    # Draw text shadow (black)
                                    draw.text((text_x + 1, text_y + 1), track_id_str, fill=(0, 0, 0), font=font)
                                    # Draw text (white)
                                    draw.text((text_x, text_y), track_id_str, fill=(255, 255, 255), font=font)
                                    
                                    # Convert back to numpy array
                                    vis_frame = np.array(pil_img)
                                except ImportError:
                                    # PIL not available, skip text rendering
                                    pass
                                except Exception:
                                    # If text rendering fails for any reason, just continue without it
                                    pass
                
                visualization[frame_idx] = vis_frame
            except Exception as e:
                debug_output.append(f"  Frame {frame_idx}: trajectory drawing failed: {e}")
                if use_overlay_base:
                    vis_frame = np.ascontiguousarray(overlay_frames[frame_idx].copy())
                else:
                    img = images_norm[frame_idx]
                    vis_frame = np.stack([img, img, img], axis=-1)
                visualization[frame_idx] = vis_frame
        
        # Save visualization with compression for faster I/O
        # Use compression to reduce file size and write time
        if progress_callback:
            progress_callback(99, "Saving visualization file...")
        
        T = visualization.shape[0]
        meta = None
        try:
            from bactimate.modules.tiff_metadata import build_metadata_dict
            ps = getattr(self, '_cal_ps', 0) or 0
            tp = getattr(self, '_cal_tp', 0) or 0
            # axes TYXC so metadata matches 4D shape (T, H, W, 3); avoids tifffile "axes do not match shape" warning
            meta = build_metadata_dict(
                pixel_size_um=ps if ps > 0 else None,
                time_per_frame_min=tp if tp > 0 else None,
                axes="TYXC",
                images=T,
                frames=T,
            )
        except Exception:
            pass
        try:
            if meta:
                tifffile.imwrite(str(output_path), visualization, metadata=meta, compression='zlib')
            else:
                tifffile.imwrite(str(output_path), visualization, compression='zlib')
            debug_output.append(f"✓ Visualization saved with {num_tracks} tracks (compressed)")
        except Exception as e:
            debug_output.append(f"⚠ Compression failed, saving uncompressed: {str(e)}")
            tifffile.imwrite(str(output_path), visualization, metadata=meta if meta else {})
            debug_output.append(f"✓ Visualization saved with {num_tracks} tracks (uncompressed)")
    
    def _create_overlay_with_track_ids(self, images, masks_tracked, output_path, progress_callback=None):
        """Create overlay visualization with track IDs as CellIDs (consistent with tracking visualization).
        If progress_callback(frame_idx, total, msg) is given, it is called each frame and tqdm is disabled."""
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        import tifffile
        
        num_frames = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Normalize images
        img_min = images.min(axis=(1, 2), keepdims=True)
        img_max = images.max(axis=(1, 2), keepdims=True)
        img_range = img_max - img_min
        img_range[img_range == 0] = 1
        images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)
        
        # Get unique track IDs
        unique_tracks = np.unique(masks_tracked)
        unique_tracks = unique_tracks[unique_tracks > 0]  # Remove background
        num_tracks = len(unique_tracks)
        
        # Generate colormap for tracks
        cmap = plt.get_cmap('gist_ncar')
        colors = cmap(np.linspace(0, 1, max(num_tracks + 1, 1000)))[:num_tracks + 1]
        colors[0] = [0, 0, 0, 0]  # Background transparent
        
        # Create color mapping
        track_colors = {}
        for i, track_id in enumerate(unique_tracks):
            track_colors[int(track_id)] = colors[i + 1]  # +1 because 0 is background
        
        # Try to import PIL for text rendering
        try:
            from PIL import Image, ImageDraw, ImageFont
            from skimage.measure import regionprops
            PIL_AVAILABLE = True
        except ImportError:
            PIL_AVAILABLE = False
        
        use_callback = callable(progress_callback)
        overlays = []
        for frame_idx in tqdm(range(num_frames), desc="Creating tracked overlay", disable=use_callback):
            if use_callback:
                progress_callback(frame_idx, num_frames, "Creating overlay")
            img = images_norm[frame_idx]
            mask = masks_tracked[frame_idx]
            
            # Convert grayscale to RGB
            img_rgb = np.stack([img, img, img], axis=-1)
            
            # Colorize mask using track IDs
            mask_rgb = np.zeros((height, width, 3), dtype=np.float32)
            for track_id, color in track_colors.items():
                mask_rgb[mask == track_id] = color[:3]
            mask_rgb = (mask_rgb * 255).astype(np.uint8)
            
            # Blend
            overlay = (img_rgb * 0.5 + mask_rgb * 0.5).astype(np.uint8)
            
            # Add track ID labels if PIL is available
            if PIL_AVAILABLE and mask.max() > 0:
                try:
                    # Convert to PIL Image
                    pil_img = Image.fromarray(overlay)
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Try to use a nice font
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                    except:
                        try:
                            font = ImageFont.truetype("arial.ttf", 10)
                        except:
                            font = ImageFont.load_default()
                    
                    # Get cell centroids and track IDs
                    regions = regionprops(mask)
                    for region in regions:
                        if region.label == 0:
                            continue
                        
                        # Get centroid
                        centroid_y, centroid_x = region.centroid
                        x, y = int(centroid_x), int(centroid_y)
                        
                        # Draw track ID text with shadow for readability
                        track_id_str = str(region.label)
                        text_x, text_y = x + 2, y - 5
                        
                        # Draw text shadow (black)
                        draw.text((text_x + 1, text_y + 1), track_id_str, fill=(0, 0, 0), font=font)
                        # Draw text (white)
                        draw.text((text_x, text_y), track_id_str, fill=(255, 255, 255), font=font)
                    
                    # Convert back to numpy array
                    overlay = np.array(pil_img)
                except Exception:
                    # If text rendering fails, continue without labels
                    pass
            
            overlays.append(overlay)
        
        overlay_array = np.stack(overlays, axis=0)
        T = overlay_array.shape[0]
        try:
            from bactimate.modules.tiff_metadata import build_metadata_dict
            ps = getattr(self, '_cal_ps', 0) or 0
            tp = getattr(self, '_cal_tp', 0) or 0
            # axes TYXC so metadata matches 4D RGB stack (T, H, W, 3); avoids tifffile "axes do not match shape" warning
            meta = build_metadata_dict(
                pixel_size_um=ps if ps > 0 else None,
                time_per_frame_min=tp if tp > 0 else None,
                axes="TYXC",
                images=T,
                frames=T,
            )
            tifffile.imwrite(str(output_path), overlay_array, metadata=meta, compression='zlib')
        except Exception:
            tifffile.imwrite(str(output_path), overlay_array, compression='zlib')
        return overlay_array


class GrowthCurveVideoWorker(QThread):
    """Worker thread for creating growth curve videos"""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, str)  # value, message
    
    def __init__(self, video_path, excel_path, output_dir, time_interval, yaxis_min, yaxis_max, fps, formats, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.excel_path = excel_path
        self.output_dir = output_dir
        self.time_interval = time_interval
        self.yaxis_min = yaxis_min
        self.yaxis_max = yaxis_max
        self.fps = fps  # Frames per second for video playback
        self.formats = formats  # List of formats to save: ['tif', 'avi', 'gif', 'mp4']

    def run(self) -> None:
        try:
            import pandas as pd
            import numpy as np
            import tifffile
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import cv2
            
            self.progress.emit(5, "Loading Excel data...")
            
            # Load Excel data
            df = pd.read_excel(self.excel_path, sheet_name='Summary')
            
            # Extract frame and relative area columns
            if 'Frame' not in df.columns or 'Total Cell Area (Relative)' not in df.columns:
                self.finished.emit(False, f"Excel file must contain 'Frame' and 'Total Cell Area (Relative)' columns.\n"
                                        f"Found columns: {', '.join(df.columns)}")
                return
            
            frames = df['Frame'].values
            relative_areas = df['Total Cell Area (Relative)'].values
            
            # Convert frames to time (minutes)
            time_minutes = (frames - 1) * self.time_interval
            
            self.progress.emit(10, "Loading video frames...")
            
            # Load video frames
            if self.video_path.is_file():
                # Single multi-page TIFF file
                video_frames = tifffile.imread(str(self.video_path))
                if len(video_frames.shape) == 2:
                    # Single frame, add time dimension
                    video_frames = video_frames[np.newaxis, ...]
            else:
                # Folder with TIFF sequence
                tiff_files = sorted(self.video_path.glob("*.tif")) + sorted(self.video_path.glob("*.tiff")) + \
                            sorted(self.video_path.glob("*.TIF")) + sorted(self.video_path.glob("*.TIFF"))
                if not tiff_files:
                    self.finished.emit(False, f"No TIFF files found in folder: {self.video_path}")
                    return
                
                video_frames = []
                for tiff_file in tiff_files:
                    frame = tifffile.imread(str(tiff_file))
                    if len(frame.shape) == 2:
                        video_frames.append(frame)
                    else:
                        # Multi-page, take first page
                        video_frames.append(frame[0] if len(frame.shape) == 3 else frame)
                video_frames = np.array(video_frames)
            
            num_frames = min(len(video_frames), len(frames))
            video_frames = video_frames[:num_frames]
            frames = frames[:num_frames]
            relative_areas = relative_areas[:num_frames]
            time_minutes = time_minutes[:num_frames]
            
            self.progress.emit(20, f"Processing {num_frames} frames...")
            
            # Get video dimensions
            if len(video_frames.shape) == 3:
                frame_height, frame_width = video_frames.shape[1], video_frames.shape[2]
            else:
                frame_height, frame_width = video_frames.shape[0], video_frames.shape[1]
            
            # Graph and video share 50/50 of the space
            graph_width = frame_width  # Graph gets same width as video (50/50 split)
            video_width = frame_width
            total_width = graph_width + video_width
            
            # Create output frames
            output_frames = []
            
            # Determine y-axis limits
            y_min = self.yaxis_min if self.yaxis_min is not None else max(0.0, relative_areas.min() * 0.9)
            y_max = self.yaxis_max if self.yaxis_max is not None else relative_areas.max() * 1.1
            
            for frame_idx in range(num_frames):
                if frame_idx % 10 == 0:
                    progress = 20 + int((frame_idx / num_frames) * 70)
                    self.progress.emit(progress, f"Processing frame {frame_idx + 1}/{num_frames}...")
                
                # Get current video frame
                if len(video_frames.shape) == 3:
                    video_frame = video_frames[frame_idx]
                else:
                    video_frame = video_frames
                
                # Normalize to 0-255 if needed
                if video_frame.dtype != np.uint8:
                    video_frame = ((video_frame - video_frame.min()) / (video_frame.max() - video_frame.min() + 1e-10) * 255).astype(np.uint8)
                
                # Convert to RGB if grayscale
                if len(video_frame.shape) == 2:
                    video_frame = np.stack([video_frame] * 3, axis=-1)
                elif video_frame.shape[2] == 1:
                    video_frame = np.repeat(video_frame, 3, axis=2)
                
                # Create graph
                fig = Figure(figsize=(graph_width / 100, frame_height / 100), dpi=100)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                
                # Plot growth curve up to current frame
                current_frames = frames[:frame_idx + 1]
                current_areas = relative_areas[:frame_idx + 1]
                current_times = time_minutes[:frame_idx + 1]
                
                ax.plot(current_times, current_areas, 'k-', linewidth=8)
                ax.scatter(current_times[-1], current_areas[-1], color='black', s=200, zorder=5, edgecolors='black', linewidths=2)
                ax.set_xlabel('Time [min]', fontsize=36, fontweight='bold')
                ax.set_ylabel('rel. cell area', fontsize=36, fontweight='bold')
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(0, time_minutes.max() * 1.05)
                ax.grid(True, alpha=0.4, linewidth=1.5)
                
                # Set white background style (matching desired output)
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                # Make tick labels bigger and bolder
                ax.tick_params(axis='both', which='major', labelsize=24, width=2, colors='black')
                ax.tick_params(axis='both', which='minor', labelsize=24, colors='black')
                # Make tick labels bold by setting font weight on the label objects
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                ax.xaxis.label.set_color('black')
                ax.yaxis.label.set_color('black')
                # Make axis lines (spines) thicker and bolder
                for spine in ax.spines.values():
                    spine.set_color('black')
                    spine.set_linewidth(3.5)
                
                # Render graph to numpy array
                canvas.draw()
                # Get buffer from canvas (works with both old and new matplotlib)
                try:
                    # Try new matplotlib API first (buffer_rgba)
                    buf = canvas.buffer_rgba()
                    graph_image = np.asarray(buf)
                    # Convert RGBA to RGB
                    if graph_image.shape[2] == 4:
                        graph_image = graph_image[:, :, :3]
                except AttributeError:
                    # Fallback to old API (tostring_rgb)
                    try:
                        graph_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                        graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (3,))
                    except AttributeError:
                        # Alternative: use renderer buffer
                        buf = np.array(canvas.renderer.buffer_rgba())
                        graph_image = buf[:, :, :3]  # Convert RGBA to RGB
                
                # Resize graph to match video frame height
                if graph_image.shape[0] != frame_height or graph_image.shape[1] != graph_width:
                    graph_image = cv2.resize(graph_image, (graph_width, frame_height), interpolation=cv2.INTER_LINEAR)
                
                # Combine graph (left) and video (right)
                combined_frame = np.hstack([graph_image, video_frame])
                output_frames.append(combined_frame)
            
            self.progress.emit(95, "Saving output files...")
            
            output_files = []
            output_frames_array = np.array(output_frames)
            
            # Save as multi-page TIFF
            if 'tif' in self.formats:
                try:
                    output_tiff = self.output_dir / f"{self.output_dir.name}.tif"
                    # Ensure uint8 for smaller file size
                    if output_frames_array.dtype != np.uint8:
                        output_frames_array = ((output_frames_array - output_frames_array.min()) / 
                                             (output_frames_array.max() - output_frames_array.min() + 1e-10) * 255).astype(np.uint8)
                    # Use LZW compression for better compression ratio
                    tifffile.imwrite(str(output_tiff), output_frames_array, compression='lzw', photometric='rgb')
                    output_files.append(f"TIFF: {output_tiff.name}")
                except Exception as e:
                    output_files.append(f"TIFF: Error - {str(e)}")
            
            # Save as AVI
            if 'avi' in self.formats:
                try:
                    output_avi = self.output_dir / f"{self.output_dir.name}.avi"
                    # Try multiple codecs in order of preference
                    codecs = ['MJPG', 'XVID', 'H264', 'mp4v']
                    success = False
                    for codec_name in codecs:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec_name)
                            out = cv2.VideoWriter(str(output_avi), fourcc, self.fps, (total_width, frame_height))
                            if out.isOpened():
                                for frame in output_frames:
                                    out.write(frame)
                                out.release()
                                output_files.append(f"AVI: {output_avi.name} (codec: {codec_name})")
                                success = True
                                break
                        except Exception:
                            continue
                    if not success:
                        output_files.append(f"AVI: Error - No compatible codec found")
                except Exception as e:
                    output_files.append(f"AVI: Error - {str(e)}")
            
            # Save as GIF
            if 'gif' in self.formats:
                try:
                    output_gif = self.output_dir / f"{self.output_dir.name}.gif"
                    import imageio
                    from PIL import Image
                    # Convert frames to uint8 if needed and optimize
                    gif_frames = []
                    for frame in output_frames:
                        if frame.dtype != np.uint8:
                            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                        # Convert to PIL Image for better GIF optimization
                        if len(frame.shape) == 3:
                            pil_frame = Image.fromarray(frame)
                            # Quantize to reduce colors (256 colors for better compression)
                            pil_frame = pil_frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                            gif_frames.append(pil_frame)
                        else:
                            gif_frames.append(frame)
                    # Save with optimization
                    imageio.mimsave(str(output_gif), gif_frames, fps=self.fps, loop=0, subrectangles=True)
                    output_files.append(f"GIF: {output_gif.name}")
                except Exception as e:
                    # Fallback to simple save if PIL optimization fails
                    try:
                        gif_frames = []
                        for frame in output_frames:
                            if frame.dtype != np.uint8:
                                frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                            gif_frames.append(frame)
                        imageio.mimsave(str(output_gif), gif_frames, fps=self.fps, loop=0)
                        output_files.append(f"GIF: {output_gif.name}")
                    except Exception as e2:
                        output_files.append(f"GIF: Error - {str(e2)}")
            
            # Save as MP4
            if 'mp4' in self.formats:
                try:
                    output_mp4 = self.output_dir / f"{self.output_dir.name}.mp4"
                    import imageio
                    # Convert frames to uint8 if needed
                    mp4_frames = []
                    for frame in output_frames:
                        if frame.dtype != np.uint8:
                            frame = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-10) * 255).astype(np.uint8)
                        mp4_frames.append(frame)
                    # Try imageio first (uses ffmpeg if available)
                    try:
                        imageio.mimsave(str(output_mp4), mp4_frames, fps=self.fps, codec='libx264', quality=8)
                        output_files.append(f"MP4: {output_mp4.name}")
                    except Exception:
                        # Fallback to opencv
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(output_mp4), fourcc, self.fps, (total_width, frame_height))
                        if out.isOpened():
                            for frame in mp4_frames:
                                out.write(frame)
                            out.release()
                            output_files.append(f"MP4: {output_mp4.name}")
                        else:
                            output_files.append(f"MP4: Error - Could not create MP4 file")
                except Exception as e:
                    output_files.append(f"MP4: Error - {str(e)}")
            
            self.progress.emit(100, "Complete!")
            output_summary = "\n".join(output_files)
            self.finished.emit(True, f"Growth curve video created successfully!\n\n"
                                   f"Output directory: {self.output_dir}\n\n"
                                   f"Files created:\n{output_summary}")
        
        except Exception as e:
            log_exception(get_logger(), "GrowthCurveVideoWorker.run failed")
            import traceback
            error_msg = f"Error creating growth curve video:\n{str(e)}\n\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)



if __name__ == "__main__":
    from bactimate.gui.main_window import main
    main()
