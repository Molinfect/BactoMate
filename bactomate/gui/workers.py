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
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from bactomate.gui.log_config import get_logger, log_exception, write_crash_log
from bactomate.gui.paths import (
    get_app_root,
    get_python_executable_for_subprocess,
    get_subprocess_pythonpath_root,
)
from bactomate.modules.label_render import draw_hq_labels


_TRACKASTRA_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_TRACKASTRA_MODEL_CACHE_LOCK = threading.Lock()
_TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE = 384

DEFAULT_GROWTH_GRAPH_STYLE: dict[str, Any] = {
    "x_label": "Time [min]",
    "y_label": "rel. cell area",
    "axis_label_font_size": 36,
    "tick_font_size": 24,
    "font_weight": "bold",
    "line_width": 8.0,
    "marker_size": 200.0,
    "marker_edge_width": 2.0,
    "axis_stroke_width": 3.5,
    "tick_stroke_width": 2.0,
    "show_grid": True,
    "grid_line_width": 1.5,
    "grid_alpha": 0.4,
}

DEFAULT_GROWTH_Y_COLUMN = "Total Cell Area (Relative)"
VIDEO_FI_LAYOUT_MERGED = "merged"
VIDEO_FI_LAYOUT_SEPARATE = "separate_tiles"
DEFAULT_VIDEO_FI_LAYOUT = VIDEO_FI_LAYOUT_MERGED
VIDEO_FI_OVERLAY_TILE_LABELS = frozenset({"Phase", "Merged"})
PREFERRED_GROWTH_Y_COLUMNS: tuple[str, ...] = (
    "Total Cell Area (Relative)",
    "Total Cell Area (µm²)",
    "Number of Cells",
    "Total Foci",
    "Mean Foci per Cell",
)

BUILTIN_GROWTH_GRAPH_PRESETS: dict[str, dict[str, Any]] = {
    "Paper": {
        "axis_label_font_size": 28,
        "tick_font_size": 18,
        "line_width": 4.0,
        "marker_size": 120.0,
        "marker_edge_width": 1.5,
        "axis_stroke_width": 2.5,
        "tick_stroke_width": 1.5,
        "show_grid": True,
        "grid_line_width": 1.0,
        "grid_alpha": 0.25,
    },
    "Presentation": {
        "axis_label_font_size": 44,
        "tick_font_size": 30,
        "line_width": 10.0,
        "marker_size": 260.0,
        "marker_edge_width": 2.5,
        "axis_stroke_width": 4.0,
        "tick_stroke_width": 2.5,
        "show_grid": True,
        "grid_line_width": 2.0,
        "grid_alpha": 0.35,
    },
    "Minimal": {
        "show_grid": False,
        "line_width": 5.0,
        "marker_size": 140.0,
        "axis_stroke_width": 2.0,
        "tick_stroke_width": 1.5,
    },
}


class _InProcessStdoutMirror:
    """Duplicate stdout/stderr to a buffer and emit each completed line to ``WorkerThread.progress``."""

    __slots__ = ("_worker", "_sink", "_pending")

    def __init__(self, worker: "WorkerThread", sink: io.StringIO) -> None:
        self._worker = worker
        self._sink = sink
        self._pending = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        n = self._sink.write(s)
        self._pending += s
        while True:
            idx = self._pending.find("\n")
            if idx < 0:
                break
            line = self._pending[:idx]
            self._pending = self._pending[idx + 1 :]
            st = line.strip()
            if st:
                self._worker.progress.emit(self._worker.strip_ansi_codes(st) + "\n")
        return n

    def flush(self) -> None:
        self._sink.flush()

    def emit_tail(self) -> None:
        st = self._pending.strip()
        self._pending = ""
        if st:
            self._worker.progress.emit(self._worker.strip_ansi_codes(st) + "\n")


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


def _cached_trackastra_model(Trackastra: Any, model_name: str, device: str, debug_output: list[str]) -> Any:
    """Reuse Trackastra models across preview/run attempts in the current GUI process."""
    key = (str(model_name), str(device))
    with _TRACKASTRA_MODEL_CACHE_LOCK:
        model = _TRACKASTRA_MODEL_CACHE.get(key)
    if model is not None:
        debug_output.append(f"✓ Reusing cached Trackastra model ({model_name}, {device})")
        return model

    model = Trackastra.from_pretrained(model_name, device=device)
    with _TRACKASTRA_MODEL_CACHE_LOCK:
        _TRACKASTRA_MODEL_CACHE[key] = model
    return model


class WorkerThread(QThread):
    """Thread for running long-running tasks without freezing the GUI."""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # status message

    def __init__(self, script_path, cmd_args, parent=None):
        super().__init__(parent)
        self.script_path = script_path
        self.cmd_args = cmd_args
        self._process = None
        # Name thread for clearer Qt fatal logs (e.g. "QThread: Destroyed while thread 'analyze_cell_areas' is still running").
        try:
            if str(script_path) == "-m" and cmd_args:
                first = str(cmd_args[0]).strip()
                label = first.rsplit(".", 1)[-1] if first else "module"
            else:
                label = Path(str(script_path)).stem if script_path else "worker"
            self.setObjectName(f"WorkerThread:{label}")
        except Exception:
            pass

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
        mirror = _InProcessStdoutMirror(self, buf)
        try:
            if argv_first is not None:
                sys.argv = [argv_first] + [str(a) for a in self.cmd_args]
            else:
                sys.argv = [str(self.script_path)] + [str(a) for a in self.cmd_args]
            sys.stdout = mirror
            sys.stderr = mirror
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
            mirror.emit_tail()
            output = buf.getvalue()
            self.finished.emit(code == 0, output)
            return True
        except Exception as e:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                mirror.emit_tail()
            except Exception:
                pass
            err_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            write_crash_log("Worker (in-process script) failed:\n" + err_msg)
            self.finished.emit(False, err_msg)
            return True

    def _run_convert_nd2_in_process(self) -> bool:
        """Run convert_nd2_to_tif in this thread (same interpreter = has numpy)."""
        return self._run_script_in_process("bactomate.scripts.convert_nd2_to_tif", argv_first=str(self.script_path))

    def _run_batch_tiff_in_process(self) -> bool:
        """Run batch_process_tiff_sequence in this thread (same interpreter = has numpy, cellpose_omni, etc.)."""
        return self._run_script_in_process("bactomate.scripts.batch_process_tiff_sequence", argv_first=str(self.script_path))

    def _run_analyze_cell_areas_in_process(self) -> bool:
        """Run analyze_cell_areas via -m in this thread. cmd_args = [module_name, ...cli_args]."""
        return self._run_module_main_in_process("bactomate.scripts.analyze_cell_areas", "analyze_cell_areas")

    def _run_module_main_in_process(self, module_path: str, argv0: str) -> bool:
        """Run a module's main() in this thread. cmd_args = [module_name, ...cli_args]. argv0 = first argv for the script."""
        old_argv = list(sys.argv)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        buf = io.StringIO()
        mirror = _InProcessStdoutMirror(self, buf)
        try:
            sys.argv = [argv0] + [str(a) for a in (self.cmd_args[1:] if len(self.cmd_args) > 1 else [])]
            sys.stdout = mirror
            sys.stderr = mirror
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
            mirror.emit_tail()
            output = buf.getvalue()
            self.finished.emit(code == 0, output)
            return True
        except Exception as e:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                mirror.emit_tail()
            except Exception:
                pass
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
        # Ensure fallback logic in the exception handler is always safe to reference.
        packaged_like = False
        force_inproc = False
        python_is_app_launcher = False
        frozen_same_exe = False
        def _run_known_in_process() -> bool:
            return False
        try:
            # When running from .app (or frozen): subprocess often has no numpy. Run scripts in-process.
            # Use sys.frozen, app-bundle path, or script path inside BactoMate.app so in-process works.
            from_app = self._is_running_from_app_bundle()
            frozen = getattr(sys, "frozen", False) or getattr(sys, "__compiled__", False)
            sp = Path(self.script_path) if self.script_path else Path("")
            sp_str = str(self.script_path)
            script_in_app = "BactoMate.app" in sp_str and "Contents" in sp_str
            packaged_like = bool(from_app or frozen or script_in_app)

            def _run_known_in_process() -> bool:
                """Run known heavy scripts in-process (fallback path for packaged builds)."""
                if "convert_nd2_to_tif" in sp.name or "convert_nd2_to_tif" in sp_str:
                    self._run_convert_nd2_in_process()
                    return True
                if "batch_process_tiff_sequence" in sp.name or "batch_process_tiff_sequence" in sp_str:
                    self._run_batch_tiff_in_process()
                    return True
                if self.script_path == "-m" and self.cmd_args:
                    first = str(self.cmd_args[0])
                    if "analyze_cell_areas" in first:
                        self._run_analyze_cell_areas_in_process()
                        return True
                    if "unet_analyze_patch_config" in first:
                        self._run_module_main_in_process("bactomate.scripts.unet_analyze_patch_config", "unet_analyze_patch_config")
                        return True
                    if "unet_train_patches" in first:
                        self._run_module_main_in_process("bactomate.scripts.unet_train_patches", "unet_train_patches")
                        return True
                    if "unet_predict_patches" in first:
                        self._run_module_main_in_process("bactomate.scripts.unet_predict_patches", "unet_predict_patches")
                        return True
                return False

            # Operator override: force legacy in-process execution when needed.
            force_inproc = str(os.environ.get("BACTOMATE_FORCE_INPROCESS_WORKERS", "")).strip().lower() in {"1", "true", "yes", "on"}

            python_exe = get_python_executable_for_subprocess()
            # Never use the .app launcher as subprocess Python: it would start a second GUI.
            exe_str = str(python_exe).lower()
            python_is_app_launcher = ("bactomate" in exe_str and ("macos" in exe_str or exe_str.endswith("bactomate")))
            # PyInstaller (and similar): sys.executable is the GUI binary (e.g. BactoMate.exe), not python.exe.
            # Spawning it with a script path starts another GUI instead of running the script.
            try:
                frozen_same_exe = frozen and os.path.normcase(
                    os.path.abspath(str(python_exe))
                ) == os.path.normcase(os.path.abspath(str(sys.executable)))
            except (OSError, ValueError):
                frozen_same_exe = frozen and str(python_exe) == str(sys.executable)

            # In packaged builds, prefer subprocess for isolation when safe; use in-process when the
            # interpreter path is the GUI entrypoint (macOS .app) or the frozen exe (Windows one-folder).
            if packaged_like and (force_inproc or python_is_app_launcher):
                if _run_known_in_process():
                    return
            if packaged_like and frozen_same_exe:
                if _run_known_in_process():
                    return
                msg = (
                    "This job cannot run in the packaged app (no standalone Python interpreter). "
                    "Run BactoMate from source (see setup\\install_windows.bat) or use a feature supported in-process."
                )
                self.finished.emit(False, msg)
                return
            cmd = [python_exe, self.script_path] + self.cmd_args
            env = os.environ.copy()
            # Scripts import ``bactomate.*``; subprocess does not inherit sys.path from the GUI.
            pp_root = get_subprocess_pythonpath_root()
            if env.get("PYTHONPATH"):
                env["PYTHONPATH"] = pp_root + os.pathsep + env["PYTHONPATH"]
            else:
                env["PYTHONPATH"] = pp_root
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
                # Packaged builds sometimes have an incomplete subprocess env (missing numpy/torch/etc).
                # For known heavy scripts, fall back to the in-process runner once when the output
                # looks like an import/dependency failure.
                if packaged_like and (not force_inproc) and (not python_is_app_launcher):
                    out_l = (output or "").lower()
                    looks_like_import_failure = (
                        "modulenotfounderror" in out_l
                        or "importerror" in out_l
                        or "no module named" in out_l
                        or "cellpose_omni not found" in out_l
                        or "cellpose-omni not found" in out_l
                        or "cannot import name" in out_l
                    )
                    if looks_like_import_failure and _run_known_in_process():
                        return
                write_crash_log("Worker script failed (exit %s):\n%s" % (process.returncode, output))
                self.finished.emit(False, output)
        except Exception as e:
            self._process = None
            log_exception(get_logger(), "WorkerThread.run failed")
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            # If subprocess failed to start in a packaged context, try legacy in-process fallback
            # for known scripts before surfacing the error.
            try:
                if packaged_like and (not force_inproc) and _run_known_in_process():
                    return
            except Exception:
                pass
            write_crash_log("WorkerThread.run exception:\n" + error_msg)
            self.finished.emit(False, error_msg)


@dataclass
class TrackingPreviewResult:
    """In-memory tracking preview (no files written)."""

    mask_path: Path
    frames_rgb: np.ndarray
    tracks_df: Any
    original_path: Optional[Path] = None
    n_tracks: int = 0
    # (T, Y, X) intensity stack used for Trackastra (may be downsampled in preview mode).
    stack_images: Any = None
    masks_tracked: Any = None
    edges_df: Any = None
    founder_by_track: Optional[dict[int, int]] = None
    lineage_tables: Optional[dict[str, Any]] = None
    is_lineage_preview: bool = False


class TrackingWorker(QThread):
    """Worker thread for running tracking"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)  # progress percentage, status message
    preview_ready = pyqtSignal(object)  # TrackingPreviewResult when write_outputs is False
    
    def __init__(
        self,
        mask_file,
        original_file,
        excel_file,
        model_name,
        mode,
        device,
        parent=None,
        output_dir=None,
        track_action: str = "movement",
        frame_limit: Optional[int] = None,
        *,
        write_outputs: bool = True,
        run_metadata: Optional[dict] = None,
        filter_params: Optional[dict] = None,
        write_visualization: bool = True,
        preview_lineage: bool = False,
    ):
        super().__init__(parent)
        self.setObjectName(type(self).__name__)
        self.mask_file = mask_file
        self.original_file = original_file
        self.excel_file = excel_file if write_outputs else None
        self.model_name = model_name
        self.mode = mode
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.track_action = track_action  # "preview" | "movement" | "lineage"
        self.preview_lineage = bool(preview_lineage)
        self.frame_limit = frame_limit
        self.write_outputs = bool(write_outputs)
        self.run_metadata = run_metadata or {}
        self.filter_params = filter_params or {}
        self.write_visualization = bool(write_visualization)
        profile = os.getenv("BACTOMATE_TRACKING_OUTPUT_PROFILE", "minimal")
        profile = str(profile).strip().lower()
        if profile not in {"minimal", "legacy", "both"}:
            profile = "minimal"
        self.output_profile = profile

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

            def _read_tiff_stack(path: str | Path, label: str) -> tuple[np.ndarray, int | None, bool]:
                """Preview optimization: read only the requested leading frames when possible."""
                p = str(path)
                limit = int(self.frame_limit or 0)
                if not self.write_outputs and limit > 0:
                    try:
                        with tifffile.TiffFile(p) as tif:
                            n_pages = len(tif.pages)
                        if n_pages > 0:
                            n_use = min(limit, n_pages)
                            arr = tifffile.imread(p, key=range(n_use))
                            if getattr(arr, "ndim", 0) == 2:
                                arr = arr[np.newaxis, ...]
                            debug_output.append(
                                f"✓ {label}: preview-read first {arr.shape[0]} of {n_pages} frame(s)"
                            )
                            return arr, n_pages, True
                    except Exception as exc:
                        debug_output.append(
                            f"  {label}: partial preview read failed ({exc}); falling back to full read"
                        )
                arr = tifffile.imread(p)
                n_total = int(arr.shape[0]) if getattr(arr, "ndim", 0) >= 3 else None
                return arr, n_total, False
            
            debug_output.append("✓ All imports successful")
            self.progress.emit(5, "Loading masks...")
            
            # Load masks
            debug_output.append(f"\nLoading masks from: {self.mask_file}")
            masks, mask_total_frames, mask_was_limited_read = _read_tiff_stack(self.mask_file, "Masks")
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
                imgs, image_total_frames, image_was_limited_read = _read_tiff_stack(
                    self.original_file, "Images"
                )
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
                image_total_frames = mask_total_frames
                image_was_limited_read = mask_was_limited_read

            full_T = int(mask_total_frames or masks.shape[0])
            if self.frame_limit is not None and int(self.frame_limit) > 0:
                n_use = min(int(self.frame_limit), full_T)
                if masks.shape[0] > n_use:
                    masks = masks[:n_use]
                if imgs.shape[0] > n_use:
                    imgs = imgs[:n_use]
                if n_use < full_T:
                    debug_output.append(
                        f"Frame limit ({self.track_action}): using first {n_use} of {full_T} frames"
                    )
                elif mask_was_limited_read or image_was_limited_read:
                    debug_output.append(
                        f"Frame limit ({self.track_action}): using all {n_use} available frame(s)"
                    )

            if not self.write_outputs:
                masks, imgs = self._downsample_stacks_for_preview(
                    masks,
                    imgs,
                    max_size=_TRACKING_PREVIEW_DOWNSAMPLE_MAX_SIZE,
                )
                debug_output.append(
                    f"Preview mode: spatial downsample for speed (shape {masks.shape})"
                )
            
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
                model = _cached_trackastra_model(
                    Trackastra,
                    self.model_name,
                    self.device,
                    debug_output,
                )
                debug_output.append("✓ Model loaded successfully")
                self.progress.emit(50, "Model loaded, starting tracking...")
            except Exception as e:
                import traceback
                from bactomate.gui.trackastra_utils import format_trackastra_model_error

                error_details = traceback.format_exc()
                debug_output.append(f"✗ Error loading model: {str(e)}")
                debug_output.append(f"\nFull traceback:\n{error_details}")

                short = format_trackastra_model_error(e)
                if short != str(e):
                    error_msg = f"Error loading Trackastra model:\n\n{short}\n\n"
                    error_msg += "Debug information:\n" + "\n".join(debug_output[-8:])
                else:
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
            link_mode = self.mode
            if (
                self.track_action == "lineage" or getattr(self, "preview_lineage", False)
            ) and link_mode == "greedy_nodiv":
                link_mode = "greedy"
                debug_output.append(
                    "Lineage run: greedy_nodiv cannot represent divisions — using greedy linking for model.track()."
                )
            debug_output.append(f"\nStarting tracking with mode: {link_mode}")
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
                        g, m = model.track(imgs, masks, mode=link_mode)
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

            # Extract track information from the graph (preview and full runs)
            self.progress.emit(75, "Extracting track information...")
            debug_output.append("\nExtracting track information from graph...")

            # Method 1: Use graph_to_napari_tracks to get proper track data
            napari_graph = None
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

            if not self.write_outputs:
                self._emit_preview_result(
                    debug_output,
                    imgs,
                    masks_tracked,
                    tracks_df,
                    napari_graph=napari_graph,
                )
                return

            # Save tracking results (full runs only — preview never reaches here)
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
            stem = Path(self.mask_file).stem
            write_legacy_outputs = self.output_profile in {"legacy", "both"}
            write_minimal_outputs = self.output_profile in {"minimal", "both"}
            debug_output.append(
                f"Output profile: {self.output_profile} (minimal={write_minimal_outputs}, legacy={write_legacy_outputs})"
            )

            # Read calibration from original or mask TIFF so we can preserve it in written TIFFs
            from bactomate.modules.tiff_metadata import build_metadata_dict, resolve_calibration_for_mask
            _cal_path = Path(self.mask_file) if self.mask_file else None
            if _cal_path is not None:
                self._cal_ps, self._cal_tp, _, _ = resolve_calibration_for_mask(_cal_path)
            else:
                self._cal_ps, self._cal_tp = 0.0, 0.0

            # Save tracks as CSV: track_id, frame, y, x (and Cell_ID if analysis Excel was used).
            # y,x come from Trackastra's track graph (centroid-like node positions; napari format or node centroid_y/x).
            self.progress.emit(80, "Saving tracks...")
            tracks_file = output_dir / f"{stem}_tracks.csv"
            tracks_df.to_csv(tracks_file, index=False)
            debug_output.append(f"✓ Saved tracks to: {tracks_file}")
            
            # Save tracked masks
            self.progress.emit(85, "Saving tracked masks...")
            masks_tracked_file = output_dir / f"{stem}_masks_tracked.tif"
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
            
            # Create overlay frames with track IDs in memory only (used as base for trajectories)
            overlay_array = None
            if self.write_visualization and self.original_file:
                self.progress.emit(87, "Creating overlay with track IDs...")
                debug_output.append("\nCreating in-memory overlay with track IDs...")
                try:
                    def overlay_progress(frame_idx, total, _msg=None):
                        pct = 87 + int((frame_idx + 1) / max(total, 1) * 2)  # 87–89%
                        msg = f"Creating overlay: {frame_idx + 1}/{total}"
                        self.progress.emit(pct, msg)
                    overlay_array = self._create_overlay_with_track_ids(
                        imgs, masks_tracked, output_path=None, progress_callback=overlay_progress
                    )
                    debug_output.append("✓ Prepared tracked overlay frames (not saved separately)")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not create tracked overlay: {str(e)}")
                    debug_output.append(f"  {traceback.format_exc()}")
            
            # Create visualization with trajectories overlaid (use overlay as base when available — much faster)
            if self.write_visualization and self.original_file:
                self.progress.emit(90, "Creating trajectory visualization...")
                debug_output.append("\nCreating trajectory visualization...")
                if overlay_array is not None:
                    debug_output.append("  (Using overlay as base — drawing trajectories only.)")
                else:
                    debug_output.append("  (This may take a while for large datasets - optimizing...)")
                try:
                    vis_file = output_dir / f"{stem}_trajectories.tif"
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
            
            edges_df = pd.DataFrame(
                columns=["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
            )
            edges_path = None
            lineage_summary_legacy = None
            if self.track_action == "lineage":
                self.progress.emit(92, "Building lineage exports...")
                try:
                    from bactomate.modules.lineage_integration import lineage_edges_from_napari_graph

                    edges_df = lineage_edges_from_napari_graph(napari_graph, tracks_df)
                    if write_legacy_outputs:
                        edges_path = output_dir / f"{stem}_lineage_edges.csv"
                        edges_df.to_csv(edges_path, index=False)
                        debug_output.append(f"✓ Saved lineage edges: {edges_path.name}")
                        summary_df = self._lineage_per_track_summary(edges_df, tracks_df)
                        lineage_summary_legacy = output_dir / f"{stem}_lineage_summary.xlsx"
                        with pd.ExcelWriter(lineage_summary_legacy, engine="openpyxl") as writer:
                            edges_df.to_excel(writer, sheet_name="Lineage edges", index=False)
                            summary_df.to_excel(writer, sheet_name="Per-track summary", index=False)
                        debug_output.append(f"✓ Saved lineage summary: {lineage_summary_legacy.name}")
                    if self.write_visualization and self.original_file:
                        try:
                            founder_by_track = self._lineage_founder_map(edges_df, tracks_df)
                            vis_lineage = output_dir / f"{stem}_lineage.tif"
                            self._create_lineage_visualization(
                                imgs,
                                masks_tracked,
                                tracks_df,
                                edges_df,
                                founder_by_track,
                                vis_lineage,
                                progress_callback=lambda pct, msg: self.progress.emit(
                                    min(99, max(92, int(pct))), msg
                                ),
                                overlay_frames=overlay_array,
                            )
                            debug_output.append(f"✓ Saved lineage visualization: {vis_lineage.name}")
                        except Exception as e:
                            import traceback
                            debug_output.append(f"⚠ Lineage visualization failed: {e}")
                            debug_output.append(traceback.format_exc())
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Lineage export failed: {e}")
                    debug_output.append(traceback.format_exc())
            
            # Save in CTC format (Cell Tracking Challenge format)
            try:
                ctc_dir = output_dir / "ctc_format"
                ctc_dir.mkdir(exist_ok=True)
                graph_to_ctc(track_graph, masks_tracked, outdir=str(ctc_dir))
            except Exception as e:
                # CTC format is optional, don't fail if it doesn't work
                pass
            
            # If Excel data was provided, merge with tracks
            merged_data = None
            merged_file = None
            if cell_data is not None:
                self.progress.emit(95, "Merging with analysis data...")
                # Merge tracking data with cell properties
                merged_data = self._merge_tracking_with_analysis(tracks_df, cell_data, masks_tracked)
                if write_legacy_outputs:
                    merged_file = output_dir / f"{stem}_tracks_with_analysis.xlsx"
                    with pd.ExcelWriter(merged_file, engine='openpyxl') as writer:
                        merged_data.to_excel(writer, sheet_name='Tracks with Analysis', index=False)
                        tracks_df.to_excel(writer, sheet_name='Tracks', index=False)
                        if isinstance(cell_data, pd.DataFrame):
                            cell_data.to_excel(writer, sheet_name='Original Analysis', index=False)
                    debug_output.append(f"✓ Saved merged tracks workbook: {merged_file.name}")

            report_file = None
            tables_dir = None
            written_tables: dict[str, str] = {}
            if write_minimal_outputs:
                try:
                    self.progress.emit(97, "Building simplified tabular outputs...")
                    trajectory_tables = self._build_trajectory_tables(
                        tracks_df=tracks_df,
                        masks_tracked=masks_tracked,
                        merged_data=merged_data,
                    )
                    lineage_tables = None
                    lineage_tables = None
                    if self.track_action == "lineage":
                        lineage_tables = self._build_integrated_lineage_tables(
                            tracks_df=tracks_df,
                            edges_df=edges_df,
                            masks_tracked=masks_tracked,
                            merged_data=merged_data,
                            output_dir=output_dir,
                            stem=stem,
                        )
                    report_file = output_dir / f"{stem}_report.xlsx"
                    self._write_report_workbook(
                        output_path=report_file,
                        trajectory_tables=trajectory_tables,
                        lineage_tables=lineage_tables,
                        run_metadata=self.run_metadata,
                        filter_params=self.filter_params,
                    )
                    debug_output.append(f"✓ Saved report workbook: {report_file.name}")
                    tables_dir, written_tables = self._write_canonical_tables(
                        output_dir=output_dir,
                        trajectory_tables=trajectory_tables,
                        lineage_tables=lineage_tables,
                    )
                    debug_output.append(f"✓ Saved canonical tables in: {tables_dir}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not write simplified tabular outputs: {e}")
                    debug_output.append(traceback.format_exc())

            track_count = int(tracks_df["track_id"].nunique()) if ("track_id" in tracks_df.columns and len(tracks_df) > 0) else 0
            track_points = int(len(tracks_df))
            frame_count = int(masks_tracked.shape[0]) if hasattr(masks_tracked, "shape") and len(masks_tracked.shape) > 0 else 0
            division_count = int(len(edges_df)) if edges_df is not None and len(edges_df) > 0 else 0
            self.progress.emit(
                99,
                "[RESULT][TRACK] "
                f"mask={Path(self.mask_file).resolve()} "
                f"original={Path(self.original_file).resolve() if self.original_file else 'none'} "
                f"analysis_excel={Path(self.excel_file).resolve() if self.excel_file else 'none'} "
                f"output_dir={Path(output_dir).resolve()}",
            )
            self.progress.emit(
                99,
                "[RESULT][TRACK] "
                f"tracks={track_count} track_points={track_points} frames={frame_count} "
                f"divisions={division_count} action={self.track_action}",
            )
            self.progress.emit(
                99,
                "[RESULT][TRACK] "
                f"tracks_csv={Path(tracks_file).resolve()} "
                f"tracked_masks={Path(masks_tracked_file).resolve()} "
                f"report={Path(report_file).resolve() if report_file is not None else 'none'}",
            )

            self.progress.emit(100, "Tracking completed!")
            success_msg = "Tracking completed!\n\n"
            success_msg += "\n".join(debug_output)
            success_msg += f"\n\nTracks saved to: {tracks_file}\n"
            success_msg += f"Tracked masks saved to: {masks_tracked_file}\n"
            if report_file is not None:
                success_msg += f"Report workbook saved to: {report_file}\n"
            if tables_dir is not None:
                success_msg += f"Canonical tables saved to: {tables_dir}\n"
                if written_tables:
                    success_msg += "Canonical table files:\n"
                    for k in sorted(written_tables.keys()):
                        success_msg += f"  - {k}: {written_tables[k]}\n"
            if edges_path is not None:
                success_msg += f"Legacy lineage edges saved to: {edges_path}\n"
            if lineage_summary_legacy is not None:
                success_msg += f"Legacy lineage summary saved to: {lineage_summary_legacy}\n"
            if merged_file is not None:
                success_msg += f"Merged data saved to: {merged_file}\n"
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

    @staticmethod
    def _json_list(values: list[int]) -> str:
        import json

        return json.dumps([int(v) for v in values], separators=(",", ":"))

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            f = float(value)
            if np.isnan(f):
                return None
            return f
        except Exception:
            return None

    def _build_track_cell_table(self, tracks_df) -> tuple[dict[int, list[int]], Any]:
        import pandas as pd

        empty_df = pd.DataFrame(columns=["track_id", "cell_id"])
        if tracks_df is None or len(tracks_df) == 0 or "track_id" not in tracks_df.columns:
            return {}, empty_df
        cell_col = "Cell_ID" if "Cell_ID" in tracks_df.columns else "track_id"
        mapping: dict[int, list[int]] = {}
        rows: list[dict[str, int]] = []
        for tid_raw, g in tracks_df.groupby("track_id"):
            tid = self._safe_int(tid_raw)
            if tid is None:
                continue
            vals: list[int] = []
            for v in g[cell_col].tolist():
                iv = self._safe_int(v)
                if iv is None:
                    continue
                vals.append(iv)
            unique_vals = sorted(set(vals)) if vals else [tid]
            mapping[tid] = unique_vals
            for cid in unique_vals:
                rows.append({"track_id": tid, "cell_id": cid})
        return mapping, (pd.DataFrame(rows) if rows else empty_df)

    @staticmethod
    def _guess_area_column(df) -> Optional[str]:
        if df is None or len(df) == 0:
            return None
        preferred = [
            "area_um2",
            "area_µm2",
            "area_um^2",
            "area",
            "Area",
            "cell_area",
            "Cell_Area",
            "Area (um^2)",
            "Area_um2",
            "size",
            "Size",
        ]
        cols = list(df.columns)
        for c in preferred:
            if c in cols:
                return c
        for c in cols:
            s = str(c).lower()
            if "area" in s:
                return c
        return None

    def _compute_track_metrics_table(
        self,
        tracks_df,
        pixel_size_um: float,
        time_per_frame_min: float,
    ):
        import pandas as pd

        cols = [
            "track_id",
            "trajectory_length",
            "duration_frames",
            "gap_frames",
            "duration_min",
            "first_frame",
            "last_frame",
            "displacement_px",
            "displacement_um",
            "path_length_px",
            "path_length_um",
            "mean_speed_px_frame",
            "mean_speed_um_s",
            "straight_line_speed_px_frame",
            "straight_line_speed_um_s",
            "straightness",
        ]
        if tracks_df is None or len(tracks_df) == 0 or "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return pd.DataFrame(columns=cols)
        need_xy = "y" in tracks_df.columns and "x" in tracks_df.columns
        rows = []
        for tid, g in tracks_df.groupby("track_id"):
            g = g.sort_values("frame")
            n = len(g)
            first_frame = int(g["frame"].iloc[0])
            last_frame = int(g["frame"].iloc[-1])
            duration = max(1, int(last_frame - first_frame + 1))
            displacement_px = 0.0
            path_length_px = 0.0
            if need_xy and n >= 2:
                ys = g["y"].values
                xs = g["x"].values
                displacement_px = float(np.hypot(ys[-1] - ys[0], xs[-1] - xs[0]))
                path_length_px = float(np.sum(np.hypot(np.diff(ys), np.diff(xs))))
            displacement_um = displacement_px * pixel_size_um if pixel_size_um > 0 else displacement_px
            path_length_um = path_length_px * pixel_size_um if pixel_size_um > 0 else path_length_px
            straightness = float(displacement_px / path_length_px) if path_length_px > 0 else 0.0
            mean_speed_px_frame = path_length_px / duration if duration > 0 else 0.0
            duration_sec = duration * time_per_frame_min * 60.0 if time_per_frame_min > 0 else 0.0
            if pixel_size_um > 0 and duration_sec > 0:
                mean_speed_um_s = path_length_um / duration_sec
            else:
                mean_speed_um_s = mean_speed_px_frame
            duration_min = duration * time_per_frame_min if time_per_frame_min > 0 else 0.0
            straight_line_speed_px_frame = displacement_px / duration if duration > 0 else 0.0
            if duration_sec > 0 and pixel_size_um > 0:
                straight_line_speed_um_s = displacement_um / duration_sec
            else:
                straight_line_speed_um_s = straight_line_speed_px_frame
            gap_frames = max(0, int(duration) - int(n))
            rows.append(
                {
                    "track_id": int(tid),
                    "trajectory_length": int(n),
                    "duration_frames": int(duration),
                    "gap_frames": int(gap_frames),
                    "duration_min": float(duration_min),
                    "first_frame": int(first_frame),
                    "last_frame": int(last_frame),
                    "displacement_px": float(displacement_px),
                    "displacement_um": float(displacement_um),
                    "path_length_px": float(path_length_px),
                    "path_length_um": float(path_length_um),
                    "mean_speed_px_frame": float(mean_speed_px_frame),
                    "mean_speed_um_s": float(mean_speed_um_s),
                    "straight_line_speed_px_frame": float(straight_line_speed_px_frame),
                    "straight_line_speed_um_s": float(straight_line_speed_um_s),
                    "straightness": float(straightness),
                }
            )
        return pd.DataFrame(rows, columns=cols)

    def _build_track_information_table(self, tracks_df, metrics_df):
        import pandas as pd

        cols = [
            "Cell_ID",
            "TrackID",
            "trajectory_length",
            "duration_frames",
            "gap_frames",
            "duration_min",
            "first_frame",
            "last_frame",
            "displacement_px",
            "displacement_um",
            "path_length_px",
            "path_length_um",
            "mean_speed_px_frame",
            "mean_speed_um_min",
            "mean_speed_um_s",
            "straight_line_speed_px_frame",
            "straight_line_speed_um_min",
            "straight_line_speed_um_s",
            "straightness",
        ]
        if metrics_df is None or len(metrics_df) == 0:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame()
        if "Cell_ID" in tracks_df.columns:
            first_cell = tracks_df.groupby("track_id")["Cell_ID"].first()
            out["Cell_ID"] = metrics_df["track_id"].map(first_cell).fillna(metrics_df["track_id"]).astype(int)
        else:
            out["Cell_ID"] = metrics_df["track_id"].astype(int)
        out["TrackID"] = metrics_df["track_id"].astype(int)
        out["trajectory_length"] = metrics_df["trajectory_length"].astype(int)
        out["duration_frames"] = metrics_df["duration_frames"].astype(int)
        out["gap_frames"] = metrics_df["gap_frames"].astype(int)
        out["duration_min"] = np.round(metrics_df["duration_min"], 4)
        out["first_frame"] = metrics_df["first_frame"].astype(int)
        out["last_frame"] = metrics_df["last_frame"].astype(int)
        out["displacement_px"] = np.round(metrics_df["displacement_px"], 4)
        out["displacement_um"] = np.round(metrics_df["displacement_um"], 4)
        out["path_length_px"] = np.round(metrics_df["path_length_px"], 4)
        out["path_length_um"] = np.round(metrics_df["path_length_um"], 4)
        out["mean_speed_px_frame"] = np.round(metrics_df["mean_speed_px_frame"], 6)
        out["mean_speed_um_s"] = np.round(metrics_df["mean_speed_um_s"], 6)
        out["mean_speed_um_min"] = np.round(metrics_df["mean_speed_um_s"] * 60.0, 6)
        out["straight_line_speed_px_frame"] = np.round(metrics_df["straight_line_speed_px_frame"], 6)
        out["straight_line_speed_um_s"] = np.round(metrics_df["straight_line_speed_um_s"], 6)
        out["straight_line_speed_um_min"] = np.round(metrics_df["straight_line_speed_um_s"] * 60.0, 6)
        out["straightness"] = np.round(metrics_df["straightness"], 6)
        return out[cols]

    def _track_area_table(self, tracks_df, masks_tracked, merged_data=None):
        import pandas as pd

        cols = ["track_id", "frame", "area", "area_source"]
        if tracks_df is None or len(tracks_df) == 0 or "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return pd.DataFrame(columns=cols)
        points = tracks_df[["track_id", "frame"]].copy()
        points["track_id"] = pd.to_numeric(points["track_id"], errors="coerce")
        points["frame"] = pd.to_numeric(points["frame"], errors="coerce")
        points = points.dropna().astype({"track_id": int, "frame": int}).drop_duplicates()
        if len(points) == 0:
            return pd.DataFrame(columns=cols)

        area_map: dict[tuple[int, int], tuple[float, str]] = {}
        if (
            merged_data is not None
            and len(merged_data) > 0
            and "track_id" in merged_data.columns
            and "frame" in merged_data.columns
        ):
            area_col = self._guess_area_column(merged_data)
            if area_col is not None:
                m = merged_data[["track_id", "frame", area_col]].copy()
                m["track_id"] = pd.to_numeric(m["track_id"], errors="coerce")
                m["frame"] = pd.to_numeric(m["frame"], errors="coerce")
                m[area_col] = pd.to_numeric(m[area_col], errors="coerce")
                m = m.dropna()
                if len(m) > 0:
                    m = (
                        m.groupby(["track_id", "frame"], as_index=False)[area_col]
                        .median()
                        .astype({"track_id": int, "frame": int})
                    )
                    for _, r in m.iterrows():
                        v = float(r[area_col])
                        if v > 0:
                            area_map[(int(r["track_id"]), int(r["frame"]))] = (v, f"analysis:{area_col}")

        frame_counts: dict[int, dict[int, int]] = {}
        for fr in sorted(points["frame"].unique().tolist()):
            if fr < 0 or fr >= int(masks_tracked.shape[0]):
                frame_counts[fr] = {}
                continue
            vals, cnts = np.unique(masks_tracked[fr], return_counts=True)
            counts = {}
            for v, c in zip(vals, cnts):
                iv = int(v)
                if iv > 0:
                    counts[iv] = int(c)
            frame_counts[fr] = counts

        rows = []
        for r in points.itertuples(index=False):
            tid = int(r.track_id)
            fr = int(r.frame)
            area, source = area_map.get((tid, fr), (None, None))
            if area is None:
                area = float(frame_counts.get(fr, {}).get(tid, 0))
                source = "mask_px"
            rows.append({"track_id": tid, "frame": fr, "area": float(area), "area_source": source})
        return pd.DataFrame(rows, columns=cols)

    def _build_trajectory_tables(self, tracks_df, masks_tracked, merged_data=None) -> dict[str, Any]:
        import pandas as pd

        pixel_size = float(getattr(self, "_cal_ps", 0.0) or 0.0)
        time_per_frame = float(getattr(self, "_cal_tp", 0.0) or 0.0)
        metrics_df = self._compute_track_metrics_table(tracks_df, pixel_size, time_per_frame)
        track_info_df = self._build_track_information_table(tracks_df, metrics_df)
        track_cell_map, track_cells_df = self._build_track_cell_table(tracks_df)
        area_df = self._track_area_table(tracks_df, masks_tracked, merged_data=merged_data)
        summary_cols = [
            "track_id",
            "containing_cell_ids",
            "start_frame",
            "start_coord",
            "start_y",
            "start_x",
            "end_frame",
            "end_coord",
            "end_y",
            "end_x",
            "total_traveled",
            "total_traveled_unit",
            "total_traveled_px",
            "total_traveled_um",
            "displacement_px",
            "displacement_um",
            "duration_frames",
            "gap_frames",
            "duration_min",
            "mean_speed_px_frame",
            "mean_speed_um_s",
            "straightness",
        ]

        summary_rows = []
        tracks_sorted = tracks_df.sort_values(["track_id", "frame"]) if len(tracks_df) > 0 and "frame" in tracks_df.columns else tracks_df
        metrics_lookup = (
            metrics_df.set_index("track_id").to_dict(orient="index")
            if len(metrics_df) > 0 and "track_id" in metrics_df.columns
            else {}
        )
        for tid_raw, g in tracks_sorted.groupby("track_id") if len(tracks_sorted) > 0 and "track_id" in tracks_sorted.columns else []:
            tid = int(tid_raw)
            m = metrics_lookup.get(tid, {})
            start_row = g.iloc[0]
            end_row = g.iloc[-1]
            start_frame = self._safe_int(start_row.get("frame"))
            end_frame = self._safe_int(end_row.get("frame"))
            start_y = self._safe_float(start_row.get("y")) if "y" in g.columns else None
            start_x = self._safe_float(start_row.get("x")) if "x" in g.columns else None
            end_y = self._safe_float(end_row.get("y")) if "y" in g.columns else None
            end_x = self._safe_float(end_row.get("x")) if "x" in g.columns else None
            start_coord = self._json_list([int(round(start_y)), int(round(start_x))]) if start_y is not None and start_x is not None else ""
            end_coord = self._json_list([int(round(end_y)), int(round(end_x))]) if end_y is not None and end_x is not None else ""
            containing_cells = track_cell_map.get(tid, [tid])
            total_traveled_px = float(m.get("path_length_px", 0.0))
            total_traveled_um = float(m.get("path_length_um", total_traveled_px))
            if pixel_size > 0:
                total_traveled = total_traveled_um
                total_traveled_unit = "um"
            else:
                total_traveled = total_traveled_px
                total_traveled_unit = "px"
            summary_rows.append(
                {
                    "track_id": tid,
                    "containing_cell_ids": self._json_list(containing_cells),
                    "start_frame": start_frame,
                    "start_coord": start_coord,
                    "start_y": start_y,
                    "start_x": start_x,
                    "end_frame": end_frame,
                    "end_coord": end_coord,
                    "end_y": end_y,
                    "end_x": end_x,
                    "total_traveled": total_traveled,
                    "total_traveled_unit": total_traveled_unit,
                    "total_traveled_px": total_traveled_px,
                    "total_traveled_um": total_traveled_um,
                    "displacement_px": float(m.get("displacement_px", 0.0)),
                    "displacement_um": float(m.get("displacement_um", 0.0)),
                    "duration_frames": int(m.get("duration_frames", 0) or 0),
                    "gap_frames": int(m.get("gap_frames", 0) or 0),
                    "duration_min": float(m.get("duration_min", 0.0)),
                    "mean_speed_px_frame": float(m.get("mean_speed_px_frame", 0.0)),
                    "mean_speed_um_s": float(m.get("mean_speed_um_s", 0.0)),
                    "straightness": float(m.get("straightness", 0.0)),
                }
            )
        track_summary_df = pd.DataFrame(summary_rows, columns=summary_cols)

        track_points_df = tracks_df.copy()
        if "Cell_ID" in track_points_df.columns:
            track_points_df["cell_id"] = track_points_df["Cell_ID"]
        elif "track_id" in track_points_df.columns:
            track_points_df["cell_id"] = track_points_df["track_id"]
        if "track_id" in track_points_df.columns and "frame" in track_points_df.columns:
            track_points_df = track_points_df.sort_values(["track_id", "frame"])

        return {
            "track_summary": track_summary_df,
            "track_information": track_info_df,
            "track_points": track_points_df,
            "track_cells": track_cells_df,
            "track_area": area_df,
        }

    def _build_integrated_lineage_tables(
        self,
        tracks_df,
        edges_df,
        masks_tracked,
        merged_data=None,
        *,
        output_dir: Path,
        stem: str,
    ) -> dict[str, Any]:
        """Core lineage tables plus analysis joins, QC, events, and lineage report workbook."""
        from bactomate.modules.lineage_integration import (
            canonical_lineage_tables,
            enrich_lineage_tables,
            lineage_run_info_rows,
            load_per_cell_table,
            preflight_lineage_run,
            write_lineage_report_workbook,
        )

        base = self._build_lineage_tables(
            tracks_df=tracks_df,
            edges_df=edges_df,
            masks_tracked=masks_tracked,
            merged_data=merged_data,
        )
        area_df = self._track_area_table(tracks_df, masks_tracked, merged_data=merged_data)
        per_cell_df = None
        excel_path = Path(self.excel_file) if self.excel_file else None
        if excel_path is not None and excel_path.is_file():
            try:
                per_cell_df = load_per_cell_table(excel_path)
            except Exception:
                get_logger().debug("Could not load Per-Cell Data for lineage integration", exc_info=True)
        full = enrich_lineage_tables(
            base,
            tracks_df=tracks_df,
            edges_df=edges_df,
            area_df=area_df,
            per_cell_df=per_cell_df,
            time_per_frame_min=float(getattr(self, "_cal_tp", 0.0) or 0.0),
        )
        try:
            pf = preflight_lineage_run(
                Path(self.mask_file),
                linker_mode=str(self.mode or ""),
                model_name=str(self.model_name or ""),
            )
            run_info = lineage_run_info_rows(
                mask_path=Path(self.mask_file),
                analysis_workbook=excel_path,
                linker_mode=str(self.mode or ""),
                model_name=str(self.model_name or ""),
                preflight=pf,
            )
            report_path = Path(output_dir) / f"{stem}_lineage_report.xlsx"
            write_lineage_report_workbook(report_path, full, run_info_rows=run_info)
            get_logger().info("Wrote lineage report: %s", report_path)
        except Exception:
            get_logger().debug("Lineage report workbook write failed", exc_info=True)
        return canonical_lineage_tables(full)

    def _build_lineage_tables(
        self,
        tracks_df,
        edges_df,
        masks_tracked,
        merged_data=None,
    ) -> dict[str, Any]:
        import pandas as pd

        if tracks_df is None:
            tracks_df = pd.DataFrame(columns=["track_id", "frame", "y", "x"])
        if edges_df is None:
            edges_df = pd.DataFrame(
                columns=["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
            )

        founder_by_track = self._lineage_founder_map(edges_df, tracks_df)
        if not founder_by_track and "track_id" in tracks_df.columns:
            founder_by_track = {int(t): int(t) for t in tracks_df["track_id"].dropna().unique().tolist()}

        track_cell_map, _ = self._build_track_cell_table(tracks_df)
        lineage_tracks_rows = [
            {"lineage_id": int(fid), "track_id": int(tid)}
            for tid, fid in sorted(founder_by_track.items(), key=lambda x: (int(x[1]), int(x[0])))
        ]
        lineage_tracks_df = pd.DataFrame(lineage_tracks_rows, columns=["lineage_id", "track_id"])

        lineage_cells_rows = []
        for lid in sorted(set(founder_by_track.values())):
            tracks_in_lineage = [int(tid) for tid, lf in founder_by_track.items() if int(lf) == int(lid)]
            cells: set[int] = set()
            for tid in tracks_in_lineage:
                cells.update(track_cell_map.get(tid, [tid]))
            for cid in sorted(cells):
                lineage_cells_rows.append({"lineage_id": int(lid), "cell_id": int(cid)})
        lineage_cells_df = pd.DataFrame(lineage_cells_rows, columns=["lineage_id", "cell_id"])

        edge_lineage_count: dict[int, int] = defaultdict(int)
        if len(edges_df) > 0 and "parent_track_id" in edges_df.columns and "daughter_track_id" in edges_df.columns:
            for er in edges_df.itertuples(index=False):
                parent = self._safe_int(getattr(er, "parent_track_id", None))
                daughter = self._safe_int(getattr(er, "daughter_track_id", None))
                lid = None
                if parent is not None:
                    lid = founder_by_track.get(parent)
                if lid is None and daughter is not None:
                    lid = founder_by_track.get(daughter)
                if lid is not None:
                    edge_lineage_count[int(lid)] += 1

        lineage_summary_cols = [
            "lineage_id",
            "containing_cells",
            "containing_tracks",
            "start_frame",
            "start_coord",
            "start_y",
            "start_x",
            "end_frame",
            "end_coord",
            "end_y",
            "end_x",
            "total_divisions",
            "n_tracks",
            "n_cells",
        ]
        lineage_summary_rows = []
        if len(tracks_df) > 0 and "track_id" in tracks_df.columns and "frame" in tracks_df.columns:
            for lid in sorted(set(founder_by_track.values())):
                tracks_in_lineage = [int(tid) for tid, lf in founder_by_track.items() if int(lf) == int(lid)]
                sub = tracks_df[tracks_df["track_id"].isin(tracks_in_lineage)].copy()
                if len(sub) == 0:
                    continue
                sub = sub.sort_values(["frame", "track_id"])
                first = sub.iloc[0]
                last = sub.iloc[-1]
                start_frame = self._safe_int(first.get("frame"))
                end_frame = self._safe_int(last.get("frame"))
                start_y = self._safe_float(first.get("y")) if "y" in sub.columns else None
                start_x = self._safe_float(first.get("x")) if "x" in sub.columns else None
                end_y = self._safe_float(last.get("y")) if "y" in sub.columns else None
                end_x = self._safe_float(last.get("x")) if "x" in sub.columns else None
                start_coord = self._json_list([int(round(start_y)), int(round(start_x))]) if start_y is not None and start_x is not None else ""
                end_coord = self._json_list([int(round(end_y)), int(round(end_x))]) if end_y is not None and end_x is not None else ""
                cells = sorted(
                    set(
                        lineage_cells_df.loc[lineage_cells_df["lineage_id"] == int(lid), "cell_id"].astype(int).tolist()
                    )
                )
                lineage_summary_rows.append(
                    {
                        "lineage_id": int(lid),
                        "containing_cells": self._json_list(cells),
                        "containing_tracks": self._json_list(sorted(tracks_in_lineage)),
                        "start_frame": start_frame,
                        "start_coord": start_coord,
                        "start_y": start_y,
                        "start_x": start_x,
                        "end_frame": end_frame,
                        "end_coord": end_coord,
                        "end_y": end_y,
                        "end_x": end_x,
                        "total_divisions": int(edge_lineage_count.get(int(lid), 0)),
                        "n_tracks": int(len(tracks_in_lineage)),
                        "n_cells": int(len(cells)),
                    }
                )
        lineage_summary_df = pd.DataFrame(lineage_summary_rows, columns=lineage_summary_cols)

        area_df = self._track_area_table(tracks_df, masks_tracked, merged_data=merged_data)
        lineage_growth_cols = [
            "lineage_id",
            "growth",
            "doubling_time",
            "growth_unit",
            "metric_used",
            "fit_r2",
            "n_points",
        ]
        lineage_growth_rows = []
        if len(area_df) > 0 and "track_id" in area_df.columns and "frame" in area_df.columns:
            area_df = area_df.copy()
            area_df["lineage_id"] = area_df["track_id"].map(
                lambda t: founder_by_track.get(int(t), int(t)) if self._safe_int(t) is not None else None
            )
            time_per_frame = float(getattr(self, "_cal_tp", 0.0) or 0.0)
            for lid, g in area_df.groupby("lineage_id"):
                if self._safe_int(lid) is None:
                    continue
                gsum = g.groupby("frame", as_index=False)["area"].sum().sort_values("frame")
                positive = gsum[gsum["area"] > 0].copy()
                sources = sorted(set(g["area_source"].astype(str).tolist()))
                source_label = "mask_px"
                analysis_sources = [s for s in sources if s.startswith("analysis:")]
                if analysis_sources and len(sources) > 1:
                    source_label = f"mixed({','.join(analysis_sources + ['mask_px'])})"
                elif analysis_sources:
                    source_label = analysis_sources[0]
                n_points = int(len(positive))
                growth = np.nan
                doubling_time = np.nan
                r2 = np.nan
                if n_points >= 2:
                    x = positive["frame"].to_numpy(dtype=float)
                    if time_per_frame > 0:
                        x = x * time_per_frame
                    y = np.log(positive["area"].to_numpy(dtype=float))
                    try:
                        slope, intercept = np.polyfit(x, y, 1)
                        y_pred = slope * x + intercept
                        ss_res = float(np.sum((y - y_pred) ** 2))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                        growth = float(slope)
                        if growth > 0:
                            doubling_time = float(np.log(2.0) / growth)
                    except Exception:
                        pass
                lineage_growth_rows.append(
                    {
                        "lineage_id": int(lid),
                        "growth": growth,
                        "doubling_time": doubling_time,
                        "growth_unit": "1/min" if time_per_frame > 0 else "1/frame",
                        "metric_used": source_label,
                        "fit_r2": r2,
                        "n_points": n_points,
                    }
                )
        lineage_growth_df = pd.DataFrame(lineage_growth_rows, columns=lineage_growth_cols)

        lineage_points_df = tracks_df.copy()
        if "track_id" in lineage_points_df.columns:
            lineage_points_df["lineage_id"] = lineage_points_df["track_id"].map(
                lambda t: founder_by_track.get(int(t), int(t))
                if self._safe_int(t) is not None
                else None
            )
            if "frame" in lineage_points_df.columns:
                lineage_points_df = lineage_points_df.sort_values(["lineage_id", "track_id", "frame"])

        return {
            "lineage_summary": lineage_summary_df,
            "lineage_growth": lineage_growth_df,
            "lineage_edges": edges_df.copy(),
            "lineage_tracks": lineage_tracks_df,
            "lineage_cells": lineage_cells_df,
            "lineage_points": lineage_points_df,
        }

    def _write_trajectory_workbook(
        self,
        tracks_df,
        masks_tracked,
        output_path: Path,
        merged_data=None,
    ) -> None:
        import pandas as pd

        tables = self._build_trajectory_tables(
            tracks_df=tracks_df,
            masks_tracked=masks_tracked,
            merged_data=merged_data,
        )
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name in ("track_summary", "track_information", "track_points", "track_cells", "track_area"):
                tables[name].to_excel(writer, sheet_name=name, index=False)

    def _write_lineage_workbook(
        self,
        tracks_df,
        edges_df,
        masks_tracked,
        output_path: Path,
        merged_data=None,
    ) -> None:
        import pandas as pd

        tables = self._build_lineage_tables(
            tracks_df=tracks_df,
            edges_df=edges_df,
            masks_tracked=masks_tracked,
            merged_data=merged_data,
        )
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name in (
                "lineage_summary",
                "lineage_growth",
                "lineage_edges",
                "lineage_tracks",
                "lineage_cells",
                "lineage_points",
            ):
                tables[name].to_excel(writer, sheet_name=name, index=False)

    @staticmethod
    def _autofit_excel_sheet(writer, sheet_name: str, df, *, max_width: float = 72.0) -> None:
        """Set column widths from header + cell content (openpyxl has no true Excel autofit)."""
        try:
            ws = writer.sheets[sheet_name]
        except KeyError:
            return
        from openpyxl.utils import get_column_letter

        if df is None or not hasattr(df, "columns"):
            return
        for col_idx, column in enumerate(df.columns, start=1):
            letter = get_column_letter(col_idx)
            best = len(str(column))
            try:
                series = df[column]
                if len(series) > 0:
                    sample = series.astype(str).head(2500)
                    best = max(best, int(sample.str.len().max()))
            except Exception:
                pass
            ws.column_dimensions[letter].width = min(max(best + 2, 8), max_width)

    def _build_run_info_sheet(self, run_metadata: dict) -> Any:
        import pandas as pd

        from bactomate.gui.tracking_filters import filter_params_run_info_rows

        rows: list[tuple[str, str]] = []
        for key in (
            "generated_utc",
            "mask_file",
            "original_file",
            "analysis_excel",
            "trackastra_model",
            "linking_mode",
            "device",
            "track_action",
            "frame_limit",
        ):
            val = run_metadata.get(key)
            if val is not None and str(val) != "":
                rows.append((key, str(val)))
        rows.append(("pixel_size_um_worker", str(float(getattr(self, "_cal_ps", 0.0) or 0.0))))
        rows.append(("time_per_frame_min_worker", str(float(getattr(self, "_cal_tp", 0.0) or 0.0))))
        fp = run_metadata.get("filter_params") or {}
        if fp:
            rows.append(("", ""))
            rows.append(("filter_settings", ""))
            rows.extend(filter_params_run_info_rows(fp))
        return pd.DataFrame(rows, columns=["parameter", "value"])

    def _tracks_with_pass_column(
        self,
        track_info_df: Any,
        filter_params: dict,
    ) -> tuple[Any, Any]:
        import pandas as pd

        from bactomate.gui.tracking_filters import apply_track_filters

        raw_df = track_info_df.copy()
        if raw_df.empty or "TrackID" not in raw_df.columns:
            raw_df["Pass"] = []
            return raw_df, raw_df.iloc[0:0].copy()
        pixel_size = float(getattr(self, "_cal_ps", 0.0) or 0.0)
        time_per_frame = float(getattr(self, "_cal_tp", 0.0) or 0.0)
        if not filter_params:
            filter_params = {
                "pixel_size": pixel_size,
                "time_per_frame": time_per_frame,
                "use_physical_units": pixel_size > 0,
                "use_speed_physical": pixel_size > 0 and time_per_frame > 0,
            }
        metrics_cols = [
            "track_id",
            "trajectory_length",
            "duration_frames",
            "gap_frames",
            "displacement_px",
            "displacement_um",
            "path_length_px",
            "path_length_um",
            "mean_speed_px_frame",
            "mean_speed_um_s",
            "straightness",
        ]
        metrics_df = raw_df.rename(columns={"TrackID": "track_id"})
        for col in metrics_cols:
            if col not in metrics_df.columns and col in raw_df.columns:
                metrics_df[col] = raw_df[col]
        passing = apply_track_filters(metrics_df, filter_params)
        raw_df["Pass"] = raw_df["TrackID"].map(lambda t: "Yes" if int(t) in passing else "No")
        filtered_df = raw_df[raw_df["Pass"] == "Yes"].copy()
        return raw_df, filtered_df

    def _write_report_workbook(
        self,
        output_path: Path,
        trajectory_tables: dict[str, Any],
        lineage_tables: Optional[dict[str, Any]] = None,
        *,
        run_metadata: Optional[dict] = None,
        filter_params: Optional[dict] = None,
    ) -> None:
        import pandas as pd

        meta = run_metadata or {}
        fp = filter_params if filter_params is not None else meta.get("filter_params") or {}
        run_info_df = self._build_run_info_sheet(meta)
        track_info = trajectory_tables["track_information"]
        raw_df, filtered_df = self._tracks_with_pass_column(track_info, fp)

        sheets: list[tuple[str, Any]] = [
            ("Run info", run_info_df),
            ("raw", raw_df),
            ("filtered", filtered_df),
            ("track_summary", trajectory_tables["track_summary"]),
        ]
        if lineage_tables is not None:
            sheets.append(("lineage_summary", lineage_tables["lineage_summary"]))
            sheets.append(("lineage_growth", lineage_tables["lineage_growth"]))

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                self._autofit_excel_sheet(writer, sheet_name, df)

    def _write_canonical_tables(
        self,
        output_dir: Path,
        trajectory_tables: dict[str, Any],
        lineage_tables: Optional[dict[str, Any]] = None,
    ) -> tuple[Path, dict[str, str]]:
        import json

        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}

        all_tables: list[tuple[str, Any]] = list(trajectory_tables.items())
        if lineage_tables is not None:
            all_tables.extend(list(lineage_tables.items()))

        for key, df in all_tables:
            if df is None:
                continue
            if hasattr(df, "empty") and df.empty:
                # Write empty datasets too so schema is explicit downstream.
                pass
            base = tables_dir / key
            try:
                out_path = base.with_suffix(".parquet")
                df.to_parquet(out_path, index=False)
            except Exception:
                out_path = base.with_suffix(".csv.gz")
                df.to_csv(out_path, index=False, compression="gzip")
            written[key] = out_path.name

        meta = {
            "schema_version": 1,
            "output_profile": self.output_profile,
            "pixel_size_um": float(getattr(self, "_cal_ps", 0.0) or 0.0),
            "time_per_frame_min": float(getattr(self, "_cal_tp", 0.0) or 0.0),
            "tables": written,
        }
        meta_path = tables_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)
        written["metadata"] = meta_path.name
        return tables_dir, written
    
    def _create_trajectory_visualization(
        self,
        images,
        masks_tracked,
        tracks_df,
        output_path,
        progress_callback=None,
        overlay_frames=None,
        *,
        preview_fast: bool = False,
    ):
        """Create visualization showing original images with tracked cell IDs and trajectories.
        If overlay_frames is provided (T, H, W, 3), use it as base and only draw trajectories — much faster.
        ``preview_fast`` skips expensive text labels and uses coarser trajectory lines."""
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
                            vis_frame[mask_region, 0] = np.clip(vis_frame[mask_region, 0] * 0.45 + color[0] * 0.55, 0, 255)
                            vis_frame[mask_region, 1] = np.clip(vis_frame[mask_region, 1] * 0.45 + color[1] * 0.55, 0, 255)
                            vis_frame[mask_region, 2] = np.clip(vis_frame[mask_region, 2] * 0.45 + color[2] * 0.55, 0, 255)
                
                labels_to_draw = []
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
                                if preview_fast:
                                    step_size = max(1, max_steps // 40)
                                elif max_steps > 200:
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
                                    if preview_fast:
                                        vis_frame[y, x, 0] = np.clip(
                                            vis_frame[y, x, 0] * 0.55 + color[0] * 0.45, 0, 255
                                        ).astype(np.uint8)
                                        vis_frame[y, x, 1] = np.clip(
                                            vis_frame[y, x, 1] * 0.55 + color[1] * 0.45, 0, 255
                                        ).astype(np.uint8)
                                        vis_frame[y, x, 2] = np.clip(
                                            vis_frame[y, x, 2] * 0.55 + color[2] * 0.45, 0, 255
                                        ).astype(np.uint8)
                                    else:
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
                        
                        # Centroid markers only for full-quality export (preview uses mask tint above).
                        if not preview_fast:
                            current_track = track_points[track_points['frame'] == frame_idx]
                            if len(current_track) > 0:
                                y = int(current_track.iloc[0]['y'])
                                x = int(current_track.iloc[0]['x'])
                                if 0 <= y < height and 0 <= x < width:
                                    dot_size = 5
                                    outline_size = 2
                                    y_outline_start = max(0, y - dot_size - outline_size)
                                    y_outline_end = min(height, y + dot_size + outline_size + 1)
                                    x_outline_start = max(0, x - dot_size - outline_size)
                                    x_outline_end = min(width, x + dot_size + outline_size + 1)
                                    yy, xx = np.ogrid[
                                        y_outline_start:y_outline_end,
                                        x_outline_start:x_outline_end,
                                    ]
                                    center_y, center_x = y - y_outline_start, x - x_outline_start
                                    dist_sq = (yy - center_y) ** 2 + (xx - center_x) ** 2
                                    outline_mask = (dist_sq <= (dot_size + outline_size) ** 2) & (
                                        dist_sq > dot_size ** 2
                                    )
                                    vis_frame[
                                        y_outline_start:y_outline_end,
                                        x_outline_start:x_outline_end,
                                        :,
                                    ][outline_mask] = [255, 255, 255]
                                    dot_mask = dist_sq <= dot_size ** 2
                                    vis_frame[
                                        y_outline_start:y_outline_end,
                                        x_outline_start:x_outline_end,
                                        :,
                                    ][dot_mask] = color
                                    labels_to_draw.append(
                                        (str(track_id_int), x + dot_size + outline_size + 2, y - 6)
                                    )

                if labels_to_draw and not preview_fast:
                    try:
                        from PIL import Image

                        pil_img = Image.fromarray(vis_frame)
                        pil_img = draw_hq_labels(
                            pil_img,
                            labels_to_draw,
                            font_size=12,
                            scale=2,
                            stroke_width=2,
                        )
                        vis_frame = np.array(pil_img)
                    except Exception:
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
        
        if output_path is not None:
            if progress_callback:
                progress_callback(99, "Saving visualization file...")
            T = visualization.shape[0]
            meta = None
            try:
                from bactomate.modules.tiff_metadata import build_metadata_dict
                ps = getattr(self, '_cal_ps', 0) or 0
                tp = getattr(self, '_cal_tp', 0) or 0
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
        return visualization

    @staticmethod
    def _downsample_stacks_for_preview(
        masks: np.ndarray,
        imgs: np.ndarray,
        *,
        max_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Downsample (T,Y,X) stacks for fast in-memory preview tracking."""
        if masks.ndim != 3 or imgs.ndim != 3:
            return masks, imgs
        _t, h, w = masks.shape
        if max(h, w) <= max_size:
            return masks, imgs
        scale = max_size / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        sy = max(1, h // new_h)
        sx = max(1, w // new_w)
        return masks[:, ::sy, ::sx], imgs[:, ::sy, ::sx]

    def _emit_preview_result(
        self,
        debug_output: list[str],
        imgs: np.ndarray,
        masks_tracked: np.ndarray,
        tracks_df: Any,
        *,
        napari_graph: Any = None,
    ) -> None:
        """Build trajectory or lineage overlay in memory and emit preview_ready (no disk writes)."""
        import pandas as pd
        from pathlib import Path as _Path

        mask_path = _Path(self.mask_file)
        original_path = _Path(self.original_file) if self.original_file else None
        from bactomate.modules.tiff_metadata import resolve_calibration_for_mask

        self._cal_ps, self._cal_tp, _, _ = resolve_calibration_for_mask(mask_path)
        is_lineage = bool(getattr(self, "preview_lineage", False))
        edges_df = None
        founder_by_track: dict[int, int] = {}
        lineage_tables = None
        if is_lineage and napari_graph is not None:
            from bactomate.modules.lineage_integration import (
                enrich_lineage_tables,
                lineage_edges_from_napari_graph,
                lineage_founder_map,
                load_per_cell_table,
            )

            edges_df = lineage_edges_from_napari_graph(napari_graph, tracks_df)
            founder_by_track = lineage_founder_map(edges_df, tracks_df)
            base = self._build_lineage_tables(
                tracks_df=tracks_df,
                edges_df=edges_df,
                masks_tracked=masks_tracked,
                merged_data=None,
            )
            per_cell_df = None
            if self.excel_file and Path(self.excel_file).is_file():
                try:
                    per_cell_df = load_per_cell_table(Path(self.excel_file))
                except Exception:
                    pass
            lineage_tables = enrich_lineage_tables(
                base,
                tracks_df=tracks_df,
                edges_df=edges_df,
                area_df=self._track_area_table(tracks_df, masks_tracked, merged_data=None),
                per_cell_df=per_cell_df,
                time_per_frame_min=float(getattr(self, "_cal_tp", 0.0) or 0.0),
            )

        self.progress.emit(82, "Building preview overlay…")
        if is_lineage:
            debug_output.append(
                "\nLineage preview: founder colors and division markers (candidate links)."
            )
        else:
            debug_output.append(
                "\nPreview mode: in-memory trajectories only (no disk writes; labels show Trackastra track ID)."
            )
        frames_rgb = None
        try:
            def vis_progress(pct: int, msg: str) -> None:
                self.progress.emit(min(99, int(pct)), str(msg))

            if is_lineage and founder_by_track:
                frames_rgb = self._create_lineage_visualization(
                    imgs,
                    masks_tracked,
                    tracks_df,
                    edges_df if edges_df is not None else pd.DataFrame(),
                    founder_by_track,
                    output_path=None,
                    progress_callback=vis_progress,
                    overlay_frames=None,
                )
            else:
                frames_rgb = self._create_trajectory_visualization(
                    imgs,
                    masks_tracked,
                    tracks_df,
                    output_path=None,
                    progress_callback=vis_progress,
                    overlay_frames=None,
                    preview_fast=True,
                )
        except Exception as e:
            debug_output.append(f"⚠ Preview visualization failed: {e}")
        if frames_rgb is None:
            self.finished.emit(
                False,
                "Preview tracking finished but visualization could not be built.\n\n"
                + "\n".join(debug_output),
            )
            return

        n_tracks = int(tracks_df["track_id"].nunique()) if (
            isinstance(tracks_df, pd.DataFrame)
            and not tracks_df.empty
            and "track_id" in tracks_df.columns
        ) else 0
        payload = TrackingPreviewResult(
            mask_path=mask_path,
            frames_rgb=np.asarray(frames_rgb, dtype=np.uint8),
            tracks_df=tracks_df,
            original_path=original_path if original_path and original_path.is_file() else None,
            n_tracks=n_tracks,
            stack_images=np.asarray(imgs),
            masks_tracked=np.asarray(masks_tracked),
            edges_df=edges_df,
            founder_by_track=founder_by_track or None,
            lineage_tables=lineage_tables,
            is_lineage_preview=is_lineage,
        )
        self.preview_ready.emit(payload)
        self.progress.emit(100, "Preview ready")
        label = "Lineage preview" if is_lineage else "Preview tracking"
        success_msg = (
            f"{label} complete ({n_tracks} tracks, {frames_rgb.shape[0]} frames). "
            "Nothing was written to disk.\n\n"
        )
        success_msg += "\n".join(debug_output[-12:])
        self.finished.emit(True, success_msg)
    
    def _create_overlay_with_track_ids(self, images, masks_tracked, output_path=None, progress_callback=None):
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
            from PIL import Image
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

                    # Get cell centroids and track IDs
                    regions = regionprops(mask)
                    labels_to_draw = []
                    for region in regions:
                        if region.label == 0:
                            continue
                        
                        # Get centroid
                        centroid_y, centroid_x = region.centroid
                        x, y = int(centroid_x), int(centroid_y)
                        
                        track_id_str = str(region.label)
                        text_x, text_y = x + 2, y - 5
                        labels_to_draw.append((track_id_str, text_x, text_y))

                    pil_img = draw_hq_labels(
                        pil_img,
                        labels_to_draw,
                        font_size=12,
                        scale=2,
                        stroke_width=2,
                    )
                    overlay = np.array(pil_img)
                except Exception:
                    # If text rendering fails, continue without labels
                    pass
            
            overlays.append(overlay)
        
        overlay_array = np.stack(overlays, axis=0)
        if output_path is not None:
            T = overlay_array.shape[0]
            try:
                from bactomate.modules.tiff_metadata import build_metadata_dict
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

    @staticmethod
    def _lineage_edges_dataframe(napari_graph: Any, tracks_df) -> Any:
        """Napari convention: graph[child_track_id] = [parent_track_id, ...]."""
        import numpy as np
        import pandas as pd

        cols = ["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
        empty = pd.DataFrame(columns=cols)
        if napari_graph is None or tracks_df is None or len(tracks_df) == 0:
            return empty
        if "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return empty
        first_frame = tracks_df.groupby("track_id")["frame"].min()
        rows = []
        seen = set()
        try:
            items = list(napari_graph.items())
        except Exception:
            items = []
        for child_raw, parents in items:
            try:
                child = int(child_raw)
            except (TypeError, ValueError):
                continue
            if parents is None:
                continue
            if isinstance(parents, (int, np.integer)):
                plist = [int(parents)]
            else:
                try:
                    plist = [int(p) for p in parents if p is not None]
                except (TypeError, ValueError):
                    continue
            for p in plist:
                key = (p, child)
                if key in seen:
                    continue
                seen.add(key)
                if child not in first_frame.index:
                    continue
                div_f = int(first_frame.loc[child])
                pf = tracks_df.loc[tracks_df["track_id"] == p, "frame"]
                before = pf[pf < div_f]
                parent_last = int(before.max()) if len(before) > 0 else (int(pf.max()) if len(pf) > 0 else "")
                rows.append(
                    {
                        "parent_track_id": p,
                        "daughter_track_id": child,
                        "division_frame": div_f,
                        "parent_last_frame": parent_last,
                    }
                )
        return pd.DataFrame(rows, columns=cols) if rows else empty

    @staticmethod
    def _lineage_founder_map(edges_df, tracks_df) -> dict[int, int]:
        parent_of: dict[int, int] = {}
        if edges_df is not None and len(edges_df) > 0:
            for _, row in edges_df.iterrows():
                parent_of[int(row["daughter_track_id"])] = int(row["parent_track_id"])
        cache: dict[int, int] = {}

        def founder(tid: int) -> int:
            if tid in cache:
                return cache[tid]
            p = parent_of.get(tid)
            if p is None:
                cache[tid] = tid
                return tid
            f = founder(p)
            cache[tid] = f
            return f

        out: dict[int, int] = {}
        if tracks_df is not None and "track_id" in tracks_df.columns:
            for t in tracks_df["track_id"].unique():
                out[int(t)] = founder(int(t))
        return out

    def _lineage_per_track_summary(self, edges_df, tracks_df) -> Any:
        import pandas as pd

        parent_of: dict[int, int] = {}
        children_of: dict[int, list[int]] = defaultdict(list)
        if edges_df is not None and len(edges_df) > 0:
            for _, row in edges_df.iterrows():
                p, d = int(row["parent_track_id"]), int(row["daughter_track_id"])
                parent_of[d] = p
                children_of[p].append(d)

        def subtree_depth(tid: int) -> int:
            ch = children_of.get(tid, [])
            if not ch:
                return 0
            return 1 + max(subtree_depth(c) for c in ch)

        def generation(tid: int) -> int:
            g = 0
            t = tid
            while t in parent_of:
                t = parent_of[t]
                g += 1
            return g

        founder_map = self._lineage_founder_map(edges_df, tracks_df)
        rows = []
        if tracks_df is not None and "track_id" in tracks_df.columns:
            for tid in sorted(tracks_df["track_id"].unique()):
                tid = int(tid)
                cell_id = None
                if "Cell_ID" in tracks_df.columns:
                    sub = tracks_df.loc[tracks_df["track_id"] == tid, "Cell_ID"]
                    if len(sub) > 0:
                        v = sub.iloc[0]
                        cell_id = None if pd.isna(v) else v
                rows.append(
                    {
                        "track_id": tid,
                        "founder_track_id": founder_map.get(tid, tid),
                        "generation": generation(tid),
                        "branch_depth": subtree_depth(tid),
                        "Cell_ID": cell_id,
                    }
                )
        return pd.DataFrame(rows)

    def _create_lineage_visualization(
        self,
        images,
        masks_tracked,
        tracks_df,
        edges_df,
        founder_by_track: dict[int, int],
        output_path: Optional[Path],
        progress_callback=None,
        overlay_frames=None,
    ) -> Optional[np.ndarray]:
        """RGB stack: founder-family colors, trajectories, division markers, track IDs (same style as trajectories)."""
        import numpy as np
        import matplotlib.pyplot as plt
        import tifffile

        use_overlay_base = overlay_frames is not None and overlay_frames.ndim == 4
        if use_overlay_base:
            num_frames, height, width = overlay_frames.shape[0], overlay_frames.shape[1], overlay_frames.shape[2]
            images_norm = None
        else:
            num_frames = images.shape[0]
            height, width = images.shape[1], images.shape[2]
            if progress_callback:
                progress_callback(91, "Normalizing images (lineage)...")
            img_min = images.min(axis=(1, 2), keepdims=True)
            img_max = images.max(axis=(1, 2), keepdims=True)
            img_range = img_max - img_min
            img_range[img_range == 0] = 1
            images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)

        unique_founders = sorted(set(founder_by_track.values()))
        cmap = plt.get_cmap("tab20")
        founder_rgb = {}
        for i, fid in enumerate(unique_founders):
            c = cmap((i % 20) / 20.0)
            founder_rgb[int(fid)] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

        track_colors = {}
        for tid, f in founder_by_track.items():
            track_colors[int(tid)] = founder_rgb.get(int(f), (180, 180, 180))

        div_marks: list[tuple[int, int, int]] = []
        if edges_df is not None and len(edges_df) > 0 and "y" in tracks_df.columns and "x" in tracks_df.columns:
            for _, er in edges_df.iterrows():
                df = int(er["division_frame"])
                did = int(er["daughter_track_id"])
                sub = tracks_df[(tracks_df["track_id"] == did) & (tracks_df["frame"] == df)]
                if len(sub) == 0:
                    sub = tracks_df[tracks_df["track_id"] == did].sort_values("frame").head(1)
                    if len(sub) == 0:
                        continue
                    df = int(sub["frame"].iloc[0])
                y = int(sub.iloc[0]["y"])
                x = int(sub.iloc[0]["x"])
                div_marks.append((df, y, x))

        if "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return None
        unique_tracks = tracks_df["track_id"].unique()
        visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

        for frame_idx in range(num_frames):
            if progress_callback:
                progress_callback(
                    90 + int((frame_idx + 1) / max(num_frames, 1) * 9),
                    f"Lineage visualization: {frame_idx + 1}/{num_frames}",
                )

            if use_overlay_base:
                vis_frame = np.ascontiguousarray(overlay_frames[frame_idx].copy())
            else:
                img = images_norm[frame_idx]
                vis_frame = np.stack([img, img, img], axis=-1)
                mask_frame = masks_tracked[frame_idx]
                for track_id in unique_tracks:
                    tid = int(track_id)
                    if tid not in track_colors:
                        continue
                    color = track_colors[tid]
                    region = mask_frame == tid
                    vis_frame[region, 0] = np.clip(vis_frame[region, 0] * 0.5 + color[0] * 0.5, 0, 255).astype(np.uint8)
                    vis_frame[region, 1] = np.clip(vis_frame[region, 1] * 0.5 + color[1] * 0.5, 0, 255).astype(np.uint8)
                    vis_frame[region, 2] = np.clip(vis_frame[region, 2] * 0.5 + color[2] * 0.5, 0, 255).astype(np.uint8)

            if "y" in tracks_df.columns and "x" in tracks_df.columns:
                tracks_in_frame = tracks_df[
                    (tracks_df["frame"] <= frame_idx) & (tracks_df["track_id"].isin(unique_tracks))
                ]["track_id"].unique()
                labels_to_draw = []
                for track_id_int in tracks_in_frame:
                    track_id_int = int(track_id_int)
                    if track_id_int not in track_colors:
                        continue
                    track_points = tracks_df[
                        (tracks_df["track_id"] == track_id_int) & (tracks_df["frame"] <= frame_idx)
                    ]
                    color = track_colors[track_id_int]
                    if len(track_points) > 1:
                        points = track_points[["y", "x"]].values.astype(int)
                        points[:, 0] = np.clip(points[:, 0], 0, height - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, width - 1)
                        for i in range(len(points) - 1):
                            y1, x1 = points[i]
                            y2, x2 = points[i + 1]
                            dy, dx = abs(y2 - y1), abs(x2 - x1)
                            if dx == 0 and dy == 0:
                                continue
                            max_steps = max(dx, dy)
                            step_size = max(1, max_steps // 50) if max_steps > 50 else 1
                            num_steps = max_steps // step_size + 1
                            if dx > dy:
                                x_coords = np.linspace(x1, x2, num_steps, dtype=int)
                                y_coords = np.round(np.linspace(y1, y2, num_steps)).astype(int)
                            else:
                                y_coords = np.linspace(y1, y2, num_steps, dtype=int)
                                x_coords = np.round(np.linspace(x1, x2, num_steps)).astype(int)
                            y_coords = np.clip(y_coords, 1, height - 2)
                            x_coords = np.clip(x_coords, 1, width - 2)
                            for y, x in zip(y_coords, x_coords):
                                ys, ye = y - 1, y + 2
                                xs_, xe = x - 1, x + 2
                                vis_frame[ys:ye, xs_:xe, 0] = (
                                    vis_frame[ys:ye, xs_:xe, 0] * 0.3 + color[0] * 0.7
                                ).astype(np.uint8)
                                vis_frame[ys:ye, xs_:xe, 1] = (
                                    vis_frame[ys:ye, xs_:xe, 1] * 0.3 + color[1] * 0.7
                                ).astype(np.uint8)
                                vis_frame[ys:ye, xs_:xe, 2] = (
                                    vis_frame[ys:ye, xs_:xe, 2] * 0.3 + color[2] * 0.7
                                ).astype(np.uint8)
                    current_track = track_points[track_points["frame"] == frame_idx]
                    if len(current_track) > 0:
                        y = int(current_track.iloc[0]["y"])
                        x = int(current_track.iloc[0]["x"])
                        if 0 <= y < height and 0 <= x < width:
                            dot_size = 5
                            color = track_colors[track_id_int]
                            y0 = max(0, y - dot_size)
                            y1_ = min(height, y + dot_size + 1)
                            x0 = max(0, x - dot_size)
                            x1_ = min(width, x + dot_size + 1)
                            vis_frame[y0:y1_, x0:x1_, :] = color
                            labels_to_draw.append((str(track_id_int), x + dot_size + 2, y - 6))

                if labels_to_draw:
                    try:
                        from PIL import Image

                        pil_img = Image.fromarray(vis_frame)
                        pil_img = draw_hq_labels(
                            pil_img,
                            labels_to_draw,
                            font_size=12,
                            scale=2,
                            stroke_width=2,
                        )
                        vis_frame = np.array(pil_img)
                    except Exception:
                        pass

            for df, y, x in div_marks:
                if df != frame_idx or not (0 <= y < height and 0 <= x < width):
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        vis_frame[yy, xx] = (0, 255, 100)

            visualization[frame_idx] = vis_frame

        if output_path is None:
            return visualization
        if progress_callback:
            progress_callback(99, "Saving lineage visualization...")
        T = visualization.shape[0]
        meta = None
        try:
            from bactomate.modules.tiff_metadata import build_metadata_dict

            ps = getattr(self, "_cal_ps", 0) or 0
            tp = getattr(self, "_cal_tp", 0) or 0
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
                tifffile.imwrite(str(output_path), visualization, metadata=meta, compression="zlib")
            else:
                tifffile.imwrite(str(output_path), visualization, compression="zlib")
        except Exception:
            tifffile.imwrite(str(output_path), visualization, metadata=meta if meta else {})
        return visualization


def _read_tiff_series_array_and_axes(path: Path) -> tuple[np.ndarray, str]:
    """Load first series from a TIFF; return (array, axes string)."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        s0 = tif.series[0]
        arr = s0.asarray()
        axes = getattr(s0, "axes", None) or ""
    return arr, axes


def _transpose_to_tcyx(arr: np.ndarray, axes: str) -> Optional[np.ndarray]:
    """If axes is exactly T,C,Y,X (no Z) return array shaped (T, C, Y, X)."""
    if arr.ndim != 4 or not axes or len(axes) != len(arr.shape):
        return None
    if "Z" in axes or set(axes) != {"T", "C", "Y", "X"}:
        return None
    d = {a: i for i, a in enumerate(axes)}
    return np.transpose(arr, (d["T"], d["C"], d["Y"], d["X"]))


def _heuristic_tchw(arr: np.ndarray) -> Optional[np.ndarray]:
    """Guess (T, C, H, W) multi-channel stack from shape when axes are missing."""
    if arr.ndim != 4:
        return None
    t, a, b, c = arr.shape
    if 2 <= a <= 8 and a < b and a < c:
        return arr
    if 2 <= c <= 8 and c < b and c < t and c != 3:
        return np.transpose(arr, (0, 3, 1, 2))
    return None


def _merge_tcyx_to_rgb_uint8(stack_tcyx: np.ndarray) -> np.ndarray:
    """
    Composite C fluorescence channels into RGB using black-to-color LUTs (additive blend).
    stack_tcyx: (T, C, Y, X) numeric array.
    """
    from matplotlib.colors import LinearSegmentedColormap

    luts = [
        LinearSegmentedColormap.from_list("fluo_g", [(0, 0, 0), (0.0, 1.0, 0.0)]),
        LinearSegmentedColormap.from_list("fluo_m", [(0, 0, 0), (1.0, 0.0, 1.0)]),
        LinearSegmentedColormap.from_list("fluo_c", [(0, 0, 0), (0.0, 1.0, 1.0)]),
        LinearSegmentedColormap.from_list("fluo_y", [(0, 0, 0), (1.0, 1.0, 0.0)]),
        LinearSegmentedColormap.from_list("fluo_r", [(0, 0, 0), (1.0, 0.2, 0.2)]),
        LinearSegmentedColormap.from_list("fluo_o", [(0, 0, 0), (1.0, 0.55, 0.0)]),
    ]
    t, nc, h, w = stack_tcyx.shape
    out = np.zeros((t, h, w, 3), dtype=np.float32)
    scale = 1.0 / float(max(nc, 1))
    for ti in range(t):
        acc = np.zeros((h, w, 3), dtype=np.float32)
        for ci in range(nc):
            ch = stack_tcyx[ti, ci].astype(np.float32, copy=False)
            p_lo, p_hi = np.percentile(ch, (1.5, 99.5))
            if p_hi <= p_lo:
                p_hi = p_lo + 1e-6
            norm = np.clip((ch - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            cmap = luts[ci % len(luts)]
            rgb = cmap(norm)[..., :3]
            acc += rgb * scale
        acc = np.clip(acc, 0.0, 1.0)
        out[ti] = (acc * 255.0).astype(np.uint8)
    return out


def _grayscale_stack_to_rgb_uint8(frames_thw: np.ndarray) -> np.ndarray:
    """(T, H, W) -> (T, H, W, 3) uint8."""
    if frames_thw.ndim != 3:
        raise ValueError(f"Expected grayscale stack (T,H,W), got shape {frames_thw.shape}")
    t, h, w = frames_thw.shape
    out = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        f = frames_thw[i]
        if f.dtype != np.uint8:
            f = ((f - f.min()) / (f.max() - f.min() + 1e-10) * 255.0).astype(np.uint8)
        out[i] = np.stack([f, f, f], axis=-1)
    return out


def _normalize_frame_to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    f = frame.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros(frame.shape[:2] + (3,), dtype=np.uint8)
    g = ((f - lo) / (hi - lo) * 255.0).astype(np.uint8)
    if g.ndim == 2:
        return np.stack([g, g, g], axis=-1)
    return g


def _load_growth_curve_folder_rgb(folder: Path) -> np.ndarray:
    """Sequence of TIFF files in a folder -> (T, H, W, 3) uint8."""
    import tifffile

    tiff_files = (
        sorted(folder.glob("*.tif"))
        + sorted(folder.glob("*.tiff"))
        + sorted(folder.glob("*.TIF"))
        + sorted(folder.glob("*.TIFF"))
    )
    if not tiff_files:
        raise ValueError(f"No TIFF files found in folder: {folder}")
    video_frames = []
    for tiff_file in tiff_files:
        frame = tifffile.imread(str(tiff_file))
        if frame.ndim == 2:
            video_frames.append(frame)
        elif frame.ndim == 3:
            video_frames.append(frame[0])
        else:
            video_frames.append(frame.reshape(-1, frame.shape[-2], frame.shape[-1])[0])
    stack = np.array(video_frames)
    return _grayscale_stack_to_rgb_uint8(stack)


def _to_thw_single_channel_stack(arr: np.ndarray) -> np.ndarray:
    """Normalize a single-channel TIFF stack to (T, H, W)."""
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        t, a, b, c = arr.shape
        if a <= 8 and a < b and a == 1:
            return arr[:, 0, :, :]
        if c <= 4 and c < b and c == 1:
            return arr[:, :, :, 0]
        raise ValueError(
            "Each channel file must be a single-channel stack (T,H,W). "
            "Use one multi-channel TIFF or split channels into separate files."
        )
    raise ValueError(f"Unsupported TIFF shape for channel file: {arr.shape}")


def _load_growth_curve_multi_file_rgb(paths: Sequence[Path]) -> np.ndarray:
    """Several TIFF stacks (one per channel), same T/H/W -> LUT merge -> (T, H, W, 3) uint8."""
    stacks = _load_growth_curve_multi_file_thw_stacks(paths)
    tcyx = np.stack(stacks, axis=1)
    return _merge_tcyx_to_rgb_uint8(tcyx)


def _load_growth_curve_multi_file_thw_stacks(paths: Sequence[Path]) -> list[np.ndarray]:
    """Load several single-channel TIFF stacks as a list of (T, H, W) arrays."""
    stacks: list[np.ndarray] = []
    for p in paths:
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")
        arr, _ = _read_tiff_series_array_and_axes(p)
        stacks.append(_to_thw_single_channel_stack(arr))
    if not stacks:
        raise ValueError("No channel TIFF stacks loaded.")
    t0 = stacks[0].shape[0]
    h0, w0 = stacks[0].shape[1], stacks[0].shape[2]
    for i, s in enumerate(stacks):
        if s.shape[0] != t0 or s.shape[1] != h0 or s.shape[2] != w0:
            raise ValueError(
                "All channel TIFFs must have the same number of frames and image size.\n"
                f"Expected ({t0}, {h0}, {w0}), got ({s.shape[0]}, {s.shape[1]}, {s.shape[2]}) for {paths[i]}."
            )
    return stacks


_GROWTH_VIDEO_CHANNEL_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_phase60x", "Phase"),
    ("_phase40x", "Phase"),
    ("_phase100x", "Phase"),
    ("_phase", "Phase"),
    ("_Phase60X", "Phase"),
    ("_Phase", "Phase"),
    ("_brightfield", "Phase"),
    ("_bf", "Phase"),
    ("_dic", "Phase"),
    ("_gfp", "GFP"),
    ("_GFP", "GFP"),
    ("_egfp", "GFP"),
    ("_mscarlet", "mScarlet"),
    ("_mScarlet", "mScarlet"),
    ("_dsred", "dsRED"),
    ("_dsRed", "dsRED"),
    ("_mcherry", "mCherry"),
    ("_rfp", "RFP"),
    ("_561", "561nm"),
    ("_orange", "Orange"),
    ("_yfp", "YFP"),
    ("_YFP", "YFP"),
    ("_venus", "Venus"),
    ("_cfp", "CFP"),
    ("_cyan", "Cyan"),
    ("_fi", "FI"),
    ("_FI", "FI"),
    ("_foci", "FI"),
)


def _infer_growth_video_channel_label(path: Path) -> str:
    """Human-readable channel label from a TIFF filename."""
    stem = path.stem
    if "_fi_" in stem:
        return stem.partition("_fi_")[2].strip("_") or "FI"
    if "_foci_" in stem:
        return stem.partition("_foci_")[2].strip("_") or "FI"
    stem_lower = stem.lower()
    for suffix, label in _GROWTH_VIDEO_CHANNEL_SUFFIXES:
        if stem_lower.endswith(suffix.lower()):
            return label
    if "phase" in stem_lower or stem_lower.endswith("_bf"):
        return "Phase"
    for token in ("gfp", "rfp", "mcherry", "mscarlet", "cy5", "yfp", "cfp"):
        if token in stem_lower:
            return token.upper() if len(token) <= 3 else token.replace("mscarlet", "mScarlet").replace("mcherry", "mCherry")
    return stem


def _infer_growth_video_channel_role(path: Path) -> str:
    """Return ``phase``, ``fi``, or ``unknown`` based on filename tokens."""
    label = _infer_growth_video_channel_label(path)
    label_lower = label.lower()
    if label_lower in {"phase", "bf", "brightfield", "dic", "transmission"}:
        return "phase"
    if "phase" in path.stem.lower() or path.stem.lower().endswith("_bf"):
        return "phase"
    if label_lower in {"fi", "foci"}:
        return "fi"
    for token in (
        "gfp", "egfp", "rfp", "mcherry", "mscarlet", "dsred", "cy5", "yfp", "cfp",
        "561", "orange", "venus", "cyan", "foci", "_fi",
    ):
        if token in path.stem.lower():
            return "fi"
    return "unknown"


def _assign_growth_video_channel_roles(paths: Sequence[Path]) -> list[tuple[Path, str, str]]:
    """Assign phase/FI roles; fall back to first file as phase when inference is unclear."""
    items: list[tuple[Path, str, str]] = []
    for path in paths:
        role = _infer_growth_video_channel_role(path)
        label = _infer_growth_video_channel_label(path)
        items.append((path, role, label))

    if not items:
        return items

    has_phase = any(role == "phase" for _, role, _ in items)
    if not has_phase:
        path, _, label = items[0]
        phase_label = "Phase" if label.lower() in {"fi", "foci", "unknown"} or label == path.stem else label
        items[0] = (path, "phase", phase_label if phase_label != path.stem else "Phase")

    for idx, (path, role, label) in enumerate(items):
        if role == "unknown":
            if idx == 0 and not has_phase:
                continue
            items[idx] = (path, "fi", label if label != path.stem else f"Ch{idx + 1}")
    return items


def _thw_stack_to_lut_rgb_uint8(stack_thw: np.ndarray, *, lut_index: int = 0) -> np.ndarray:
    """Colorize one (T, H, W) stack using the growth-video LUT palette."""
    t, h, w = stack_thw.shape
    lut_index = max(0, int(lut_index))
    if lut_index == 0:
        tcyx = stack_thw[:, np.newaxis, :, :]
    else:
        tcyx = np.zeros((t, lut_index + 1, h, w), dtype=stack_thw.dtype)
        tcyx[:, lut_index, :, :] = stack_thw
    return _merge_tcyx_to_rgb_uint8(tcyx)


def _order_separate_video_tiles(tiles: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Order tiles as Phase, FI channels, then Merged."""
    phase: list[tuple[str, np.ndarray]] = []
    fi: list[tuple[str, np.ndarray]] = []
    merged: list[tuple[str, np.ndarray]] = []
    for label, frames in tiles:
        label_lower = label.lower()
        if label_lower == "merged":
            merged.append((label, frames))
        elif "phase" in label_lower or label_lower in {"bf", "brightfield"}:
            phase.append((label, frames))
        else:
            fi.append((label, frames))
    if not phase and fi:
        phase.append(fi.pop(0))
        if phase[0][0].lower() not in {"phase", "bf"}:
            phase = [("Phase", phase[0][1])]
    return phase + fi + merged


def _load_growth_curve_separate_tiles(paths: Sequence[Path]) -> list[tuple[str, np.ndarray]]:
    """Build named RGB tile stacks for separate-FI video layout."""
    paths = [Path(p) for p in paths if str(p).strip()]
    if not paths:
        raise ValueError("No video path(s) provided.")

    if len(paths) == 1:
        path = paths[0]
        if path.is_dir():
            arr = _load_growth_curve_folder_rgb(path)
            return [("Phase", arr), ("Merged", arr)]
        if not path.is_file():
            raise ValueError(f"Path not found: {path}")
        raw, axes = _read_tiff_series_array_and_axes(path)
        tcyx = _transpose_to_tcyx(raw, axes)
        if tcyx is None:
            tcyx = _heuristic_tchw(raw)
        if tcyx is not None and tcyx.shape[1] >= 2:
            tiles: list[tuple[str, np.ndarray]] = []
            merge_stack = tcyx
            for ci in range(tcyx.shape[1]):
                label = "Phase" if ci == 0 else f"Ch{ci + 1}"
                if ci == 0:
                    rgb = _grayscale_stack_to_rgb_uint8(tcyx[:, ci, :, :])
                else:
                    rgb = _thw_stack_to_lut_rgb_uint8(tcyx[:, ci, :, :], lut_index=ci - 1)
                tiles.append((label, rgb))
            tiles.append(("Merged", _merge_tcyx_to_rgb_uint8(merge_stack)))
            return _order_separate_video_tiles(tiles)
        rgb = _load_growth_curve_single_file_rgb(path)
        return [("Phase", rgb), ("Merged", rgb)]

    roles = _assign_growth_video_channel_roles(paths)
    stacks_thw = _load_growth_curve_multi_file_thw_stacks([p for p, _, _ in roles])
    tiles = []
    merge_stacks: list[np.ndarray] = []
    fi_index = 0
    for (path, role, label), thw in zip(roles, stacks_thw):
        merge_stacks.append(thw)
        if role == "phase":
            display = "Phase" if label.lower() in {"fi", "foci"} else label
            tiles.append((display, _grayscale_stack_to_rgb_uint8(thw)))
        else:
            display = label if label.lower() not in {"fi", "foci"} else f"Ch{fi_index + 1}"
            tiles.append((display, _thw_stack_to_lut_rgb_uint8(thw, lut_index=fi_index)))
            fi_index += 1
    tcyx = np.stack(merge_stacks, axis=1)
    tiles.append(("Merged", _merge_tcyx_to_rgb_uint8(tcyx)))
    return _order_separate_video_tiles(tiles)


@dataclass
class GrowthCurveVideoSource:
    """Loaded microscopy input for growth-curve preview/export."""

    fi_layout: str
    num_frames: int
    reference_width: int
    reference_height: int
    merged_frames: Optional[np.ndarray] = None
    tiles: Optional[list[tuple[str, np.ndarray]]] = None


def load_growth_curve_video_source(
    video_paths: Sequence[Path],
    fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
) -> GrowthCurveVideoSource:
    """Load merged video or separate named channel tiles depending on layout mode."""
    layout = fi_layout if fi_layout in (VIDEO_FI_LAYOUT_MERGED, VIDEO_FI_LAYOUT_SEPARATE) else DEFAULT_VIDEO_FI_LAYOUT
    paths = [Path(p) for p in video_paths if str(p).strip()]
    if not paths:
        raise ValueError("No video path(s) provided.")

    if layout == VIDEO_FI_LAYOUT_SEPARATE:
        tiles = _load_growth_curve_separate_tiles(paths)
        num_frames = int(tiles[0][1].shape[0])
        ref_h, ref_w = int(tiles[0][1].shape[1]), int(tiles[0][1].shape[2])
        return GrowthCurveVideoSource(
            fi_layout=layout,
            num_frames=num_frames,
            reference_width=ref_w,
            reference_height=ref_h,
            merged_frames=None,
            tiles=tiles,
        )

    merged = load_growth_curve_video_frames_rgb(paths)
    _, ref_h, ref_w, _ = merged.shape
    return GrowthCurveVideoSource(
        fi_layout=layout,
        num_frames=int(merged.shape[0]),
        reference_width=int(ref_w),
        reference_height=int(ref_h),
        merged_frames=merged,
        tiles=None,
    )


def _load_growth_curve_single_file_rgb(path: Path) -> np.ndarray:
    """One TIFF: grayscale time series, RGB (T,H,W,3), or multi-channel (T,C,H,W) with merge."""
    arr, axes = _read_tiff_series_array_and_axes(path)
    tcyx = _transpose_to_tcyx(arr, axes)
    if tcyx is None:
        tcyx = _heuristic_tchw(arr)
    if tcyx is not None and tcyx.shape[1] >= 2:
        return _merge_tcyx_to_rgb_uint8(tcyx)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        t, h, w, _ = arr.shape
        out = np.zeros((t, h, w, 3), dtype=np.uint8)
        for i in range(t):
            out[i] = _normalize_frame_to_uint8_rgb(arr[i])
        return out
    if arr.ndim == 2:
        g = arr
        if g.dtype != np.uint8:
            g = ((g - g.min()) / (g.max() - g.min() + 1e-10) * 255.0).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
        return rgb[np.newaxis, ...]
    if arr.ndim == 3:
        return _grayscale_stack_to_rgb_uint8(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return _grayscale_stack_to_rgb_uint8(arr[:, :, :, 0])
    raise ValueError(f"Unsupported TIFF shape for growth curve video: {arr.shape} (axes={axes!r})")


def load_growth_curve_video_frames_rgb(video_paths: Sequence[Path]) -> np.ndarray:
    """
    Load TIFF input(s) as (T, H, W, 3) uint8 for the growth-curve composer.

    - One folder: sorted *.tif sequence (grayscale -> RGB).
    - One file: grayscale stack, RGB (T,H,W,3), or multi-channel (T,C,H,W) merged with LUTs.
    - Multiple files: one single-channel stack per file, same dimensions; merged with LUTs.
    """
    paths = [Path(p) for p in video_paths if str(p).strip()]
    if not paths:
        raise ValueError("No video path(s) provided.")
    if len(paths) == 1:
        p = paths[0]
        if p.is_dir():
            return _load_growth_curve_folder_rgb(p)
        if p.is_file():
            return _load_growth_curve_single_file_rgb(p)
        raise ValueError(f"Path not found: {p}")
    return _load_growth_curve_multi_file_rgb(paths)


def list_growth_curve_excel_columns(excel_path: Path) -> list[str]:
    """Numeric Summary-sheet columns usable as the growth-curve Y axis (excluding Frame)."""
    import pandas as pd

    df = pd.read_excel(excel_path, sheet_name="Summary")
    if "Frame" not in df.columns:
        return []
    numeric_cols: list[str] = []
    for col in df.columns:
        if str(col) == "Frame":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(str(col))
    ordered: list[str] = []
    for pref in PREFERRED_GROWTH_Y_COLUMNS:
        if pref in numeric_cols:
            ordered.append(pref)
    for col in numeric_cols:
        if col not in ordered:
            ordered.append(col)
    return ordered


def load_growth_curve_summary_data(
    excel_path: Path,
    time_interval: float,
    y_column: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Summary sheet columns for growth-curve graph (frames, Y values, time in minutes)."""
    import pandas as pd

    df = pd.read_excel(excel_path, sheet_name="Summary")
    if "Frame" not in df.columns:
        raise ValueError(
            "Excel Summary sheet must contain a 'Frame' column.\n"
            f"Found columns: {', '.join(map(str, df.columns))}"
        )
    available = list_growth_curve_excel_columns(excel_path)
    if not available:
        raise ValueError(
            "Excel Summary sheet has no numeric columns for the growth graph.\n"
            f"Found columns: {', '.join(map(str, df.columns))}"
        )
    y_col = str(y_column or DEFAULT_GROWTH_Y_COLUMN).strip()
    if y_col not in available:
        y_col = DEFAULT_GROWTH_Y_COLUMN if DEFAULT_GROWTH_Y_COLUMN in available else available[0]
    frames = pd.to_numeric(df["Frame"], errors="coerce").values
    y_values = pd.to_numeric(df[y_col], errors="coerce").values
    if np.isnan(frames).all() or np.isnan(y_values).all():
        raise ValueError(f"Column '{y_col}' has no usable numeric values.")
    time_minutes = (frames - 1) * float(time_interval)
    return frames, y_values, time_minutes


def _ensure_growth_curve_video_frame_rgb(video_frame: np.ndarray) -> np.ndarray:
    """Normalize one video frame to (H, W, 3) uint8."""
    import numpy as np

    if video_frame.dtype != np.uint8:
        video_frame = (
            (video_frame - video_frame.min())
            / (video_frame.max() - video_frame.min() + 1e-10)
            * 255
        ).astype(np.uint8)
    if len(video_frame.shape) == 2:
        return np.stack([video_frame] * 3, axis=-1)
    if video_frame.shape[2] == 1:
        return np.repeat(video_frame, 3, axis=2)
    return video_frame


def _load_video_mask_frames(mask_path: Optional[Path]) -> Optional[np.ndarray]:
    """Load a label-mask TIFF as (T, H, W), preserving integer labels."""
    if mask_path is None or not Path(mask_path).exists():
        return None
    import tifffile

    arr = np.asarray(tifffile.imread(str(mask_path)))
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim >= 4:
        return arr.reshape((-1,) + arr.shape[-2:])
    return None


def _mask_contour(mask_2d: np.ndarray) -> np.ndarray:
    """Return a boolean contour mask for labeled cells."""
    labels = np.asarray(mask_2d)
    if labels.size == 0:
        return np.zeros(labels.shape, dtype=bool)
    try:
        from skimage.segmentation import find_boundaries

        return find_boundaries(labels, mode="inner") & (labels > 0)
    except Exception:
        try:
            from scipy.ndimage import binary_erosion

            occupied = labels > 0
            return occupied & ~binary_erosion(occupied)
        except Exception:
            occupied = labels > 0
            edge = np.zeros_like(occupied, dtype=bool)
            edge[1:, :] |= occupied[1:, :] != occupied[:-1, :]
            edge[:, 1:] |= occupied[:, 1:] != occupied[:, :-1]
            return edge & occupied


def _load_foci_points_by_frame(foci_path: Optional[Path]) -> dict[int, list[tuple[int, int]]]:
    """Load simple foci coordinates grouped by zero-based frame index from CSV/XLSX."""
    if foci_path is None or not Path(foci_path).exists():
        return {}
    import pandas as pd

    path = Path(foci_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        return {}

    lower = {str(c).strip().lower(): c for c in df.columns}

    def _first(names: Sequence[str]) -> Optional[Any]:
        for name in names:
            if name in lower:
                return lower[name]
        return None

    frame_col = _first(("frame", "t", "timepoint", "time point", "frame_idx", "frame_index"))
    x_col = _first(("x", "x_px", "x [px]", "centroid_x", "center_x", "peak_x"))
    y_col = _first(("y", "y_px", "y [px]", "centroid_y", "center_y", "peak_y"))
    if x_col is None or y_col is None:
        return {}

    frame_values = df[frame_col].to_numpy() if frame_col is not None else np.ones(len(df), dtype=int)
    try:
        one_based = bool(len(frame_values) and np.nanmin(frame_values.astype(float)) >= 1)
    except Exception:
        one_based = True
    grouped: dict[int, list[tuple[int, int]]] = {}
    for idx, row in df.iterrows():
        try:
            frame_value = int(row[frame_col]) if frame_col is not None else 1
            frame_idx = frame_value - 1 if one_based else frame_value
            x = int(round(float(row[x_col])))
            y = int(round(float(row[y_col])))
        except Exception:
            continue
        grouped.setdefault(frame_idx, []).append((x, y))
    return grouped


def _draw_video_overlays(
    video_frame: np.ndarray,
    frame_idx: int,
    *,
    mask_frames: Optional[np.ndarray] = None,
    foci_points: Optional[dict[int, list[tuple[int, int]]]] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
) -> np.ndarray:
    """Draw contour-only masks and simple foci rings over real microscopy data."""
    rgb = _ensure_growth_curve_video_frame_rgb(video_frame).copy()
    h, w = rgb.shape[:2]
    if show_mask_contours and mask_frames is not None and len(mask_frames):
        mask = np.asarray(mask_frames[min(frame_idx, len(mask_frames) - 1)])
        if mask.shape[:2] != (h, w):
            try:
                import cv2

                mask = cv2.resize(mask.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                mask = None
        if mask is not None:
            contour = _mask_contour(mask)
            rgb[contour] = np.array([255, 215, 0], dtype=np.uint8)

    if show_foci and foci_points:
        points = foci_points.get(frame_idx, [])
        for x, y in points:
            if not (0 <= x < w and 0 <= y < h):
                continue
            yy, xx = np.ogrid[max(0, y - 5) : min(h, y + 6), max(0, x - 5) : min(w, x + 6)]
            dist = (yy - y) ** 2 + (xx - x) ** 2
            ring = (dist >= 9) & (dist <= 25)
            sub = rgb[max(0, y - 5) : min(h, y + 6), max(0, x - 5) : min(w, x + 6)]
            sub[ring] = np.array([255, 80, 255], dtype=np.uint8)
    return rgb


def apply_video_montage_overlays(
    video_frames: np.ndarray,
    *,
    mask_path: Optional[Path] = None,
    foci_path: Optional[Path] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
) -> np.ndarray:
    """Return RGB video frames with optional contour/foci overlays drawn on real data."""
    if not show_mask_contours and not show_foci:
        return video_frames
    mask_frames = _load_video_mask_frames(mask_path) if show_mask_contours else None
    foci_points = _load_foci_points_by_frame(foci_path) if show_foci else {}
    return np.stack(
        [
            _draw_video_overlays(
                frame,
                idx,
                mask_frames=mask_frames,
                foci_points=foci_points,
                show_mask_contours=show_mask_contours,
                show_foci=show_foci,
            )
            for idx, frame in enumerate(video_frames)
        ],
        axis=0,
    )


def apply_video_montage_overlays_to_tiles(
    tiles: Sequence[tuple[str, np.ndarray]],
    *,
    overlay_labels: Sequence[str] = VIDEO_FI_OVERLAY_TILE_LABELS,
    mask_path: Optional[Path] = None,
    foci_path: Optional[Path] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Apply mask/foci overlays only to selected named tiles (Phase, Merged by default)."""
    labels = set(overlay_labels)
    if not show_mask_contours and not show_foci:
        return [(label, frames) for label, frames in tiles]
    result: list[tuple[str, np.ndarray]] = []
    for label, frames in tiles:
        if label in labels:
            frames = apply_video_montage_overlays(
                frames,
                mask_path=mask_path,
                foci_path=foci_path,
                show_mask_contours=show_mask_contours,
                show_foci=show_foci,
            )
        result.append((label, frames))
    return result


def _compose_separate_tiles_frame(
    frame_idx: int,
    tiles: Sequence[tuple[str, np.ndarray]],
    *,
    reference_width: int,
) -> np.ndarray:
    """Horizontally stack named tiles for one frame, scaling to keep total width near reference_width."""
    import cv2

    frame_list = [_ensure_growth_curve_video_frame_rgb(frames[frame_idx]) for _, frames in tiles]
    if not frame_list:
        raise ValueError("No video tiles available for separate-FI layout.")
    h = int(frame_list[0].shape[0])
    n = max(1, len(frame_list))
    tile_w = max(64, int(reference_width) // n)
    resized = [cv2.resize(frame, (tile_w, h), interpolation=cv2.INTER_LINEAR) for frame in frame_list]
    return np.hstack(resized)


def _render_growth_curve_graph_image(
    frame_idx: int,
    frames: np.ndarray,
    relative_areas: np.ndarray,
    time_minutes: np.ndarray,
    *,
    graph_width: int,
    frame_height: int,
    yaxis_min: Optional[float],
    yaxis_max: Optional[float],
    graph_style: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Render the growth-curve plot for one frame index to (H, W, 3) uint8."""
    import numpy as np
    import cv2
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    y_min = yaxis_min if yaxis_min is not None else max(0.0, float(relative_areas.min()) * 0.9)
    y_max = yaxis_max if yaxis_max is not None else float(relative_areas.max()) * 1.1
    style = dict(DEFAULT_GROWTH_GRAPH_STYLE)
    if graph_style:
        style.update({k: v for k, v in graph_style.items() if v is not None})
    font_weight = str(style.get("font_weight") or "bold")

    current_times = time_minutes[: frame_idx + 1]
    current_areas = relative_areas[: frame_idx + 1]

    fig = Figure(figsize=(graph_width / 100, frame_height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(current_times, current_areas, "k-", linewidth=float(style.get("line_width", 8.0)))
    ax.scatter(
        current_times[-1],
        current_areas[-1],
        color="black",
        s=float(style.get("marker_size", 200.0)),
        zorder=5,
        edgecolors="black",
        linewidths=float(style.get("marker_edge_width", 2.0)),
    )
    ax.set_xlabel(
        str(style.get("x_label") or "Time [min]"),
        fontsize=int(style.get("axis_label_font_size", 36)),
        fontweight=font_weight,
    )
    ax.set_ylabel(
        str(style.get("y_label") or "rel. cell area"),
        fontsize=int(style.get("axis_label_font_size", 36)),
        fontweight=font_weight,
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, float(time_minutes.max()) * 1.05)
    ax.grid(
        bool(style.get("show_grid", True)),
        alpha=float(style.get("grid_alpha", 0.4)),
        linewidth=float(style.get("grid_line_width", 1.5)),
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=int(style.get("tick_font_size", 24)),
        width=float(style.get("tick_stroke_width", 2.0)),
        colors="black",
    )
    ax.tick_params(axis="both", which="minor", labelsize=int(style.get("tick_font_size", 24)), colors="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(font_weight)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(float(style.get("axis_stroke_width", 3.5)))

    canvas.draw()
    try:
        buf = canvas.buffer_rgba()
        graph_image = np.asarray(buf)
        if graph_image.shape[2] == 4:
            graph_image = graph_image[:, :, :3]
    except AttributeError:
        try:
            graph_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            graph_image = graph_image.reshape(canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            buf = np.array(canvas.renderer.buffer_rgba())
            graph_image = buf[:, :, :3]

    if graph_image.shape[0] != frame_height or graph_image.shape[1] != graph_width:
        graph_image = cv2.resize(graph_image, (graph_width, frame_height), interpolation=cv2.INTER_LINEAR)
    return graph_image


def compose_growth_curve_combined_frame(
    frame_idx: int,
    video_frames: Optional[np.ndarray] = None,
    frames: Optional[np.ndarray] = None,
    relative_areas: Optional[np.ndarray] = None,
    time_minutes: Optional[np.ndarray] = None,
    *,
    tiles: Optional[Sequence[tuple[str, np.ndarray]]] = None,
    fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
    reference_width: Optional[int] = None,
    yaxis_min: Optional[float] = None,
    yaxis_max: Optional[float] = None,
    graph_style: Optional[dict[str, Any]] = None,
    graph_width_ratio: float = 0.5,
) -> np.ndarray:
    """Compose one frame: graph + video when growth data exists, otherwise video only."""
    import numpy as np

    if fi_layout == VIDEO_FI_LAYOUT_SEPARATE and tiles:
        video_frame = _compose_separate_tiles_frame(
            frame_idx,
            tiles,
            reference_width=int(reference_width or tiles[0][1].shape[2]),
        )
    else:
        if video_frames is None:
            raise ValueError("Merged video frames are required for merged FI layout.")
        video_frame = _ensure_growth_curve_video_frame_rgb(video_frames[frame_idx])
    if frames is None or relative_areas is None or time_minutes is None:
        return video_frame
    frame_height, frame_width = int(video_frame.shape[0]), int(video_frame.shape[1])
    ratio = max(0.2, min(float(graph_width_ratio), 0.8))
    graph_width = max(64, int(round(frame_width * ratio / max(1e-6, 1.0 - ratio))))
    graph_image = _render_growth_curve_graph_image(
        frame_idx,
        frames,
        relative_areas,
        time_minutes,
        graph_width=graph_width,
        frame_height=frame_height,
        yaxis_min=yaxis_min,
        yaxis_max=yaxis_max,
        graph_style=graph_style,
    )
    return np.hstack([graph_image, video_frame])


def build_growth_curve_output_frames(
    video_paths: Sequence[Path],
    excel_path: Optional[Path],
    time_interval: float,
    yaxis_min: Optional[float],
    yaxis_max: Optional[float],
    *,
    mask_path: Optional[Path] = None,
    foci_path: Optional[Path] = None,
    show_mask_contours: bool = False,
    show_foci: bool = False,
    graph_style: Optional[dict[str, Any]] = None,
    graph_width_ratio: float = 0.5,
    y_column: Optional[str] = None,
    fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Load inputs and build all combined output frames."""
    source = load_growth_curve_video_source(video_paths, fi_layout=fi_layout)
    video_frames: Optional[np.ndarray] = None
    tiles: Optional[list[tuple[str, np.ndarray]]] = None

    if source.fi_layout == VIDEO_FI_LAYOUT_SEPARATE and source.tiles:
        tiles = apply_video_montage_overlays_to_tiles(
            source.tiles,
            mask_path=mask_path,
            foci_path=foci_path,
            show_mask_contours=show_mask_contours,
            show_foci=show_foci,
        )
    else:
        video_frames = apply_video_montage_overlays(
            source.merged_frames,
            mask_path=mask_path,
            foci_path=foci_path,
            show_mask_contours=show_mask_contours,
            show_foci=show_foci,
        )
    frames: Optional[np.ndarray] = None
    relative_areas: Optional[np.ndarray] = None
    time_minutes: Optional[np.ndarray] = None
    if excel_path is not None and Path(excel_path).exists():
        frames, relative_areas, time_minutes = load_growth_curve_summary_data(
            excel_path,
            time_interval,
            y_column=y_column,
        )
        num_frames = min(source.num_frames, len(frames))
        frames = frames[:num_frames]
        relative_areas = relative_areas[:num_frames]
        time_minutes = time_minutes[:num_frames]
    else:
        num_frames = source.num_frames
    if video_frames is not None:
        video_frames = video_frames[:num_frames]
    if tiles is not None:
        tiles = [(label, stack[:num_frames]) for label, stack in tiles]

    frame_height = int(source.reference_height)
    reference_width = int(source.reference_width)
    if tiles is not None and tiles:
        sample = _compose_separate_tiles_frame(0, tiles, reference_width=reference_width)
        frame_height = int(sample.shape[0])
        frame_width = int(sample.shape[1])
    elif video_frames is not None:
        frame_height = int(video_frames.shape[1])
        frame_width = int(video_frames.shape[2])
    else:
        frame_width = reference_width
    if frames is not None:
        ratio = max(0.2, min(float(graph_width_ratio), 0.8))
        graph_width = max(64, int(round(frame_width * ratio / max(1e-6, 1.0 - ratio))))
        total_width = graph_width + frame_width
    else:
        total_width = frame_width
    output_frames: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        if progress_callback is not None and frame_idx % 10 == 0:
            progress = 20 + int((frame_idx / max(num_frames, 1)) * 70)
            progress_callback(progress, f"Processing frame {frame_idx + 1}/{num_frames}...")

        output_frames.append(
            compose_growth_curve_combined_frame(
                frame_idx,
                video_frames=video_frames,
                frames=frames,
                relative_areas=relative_areas,
                time_minutes=time_minutes,
                tiles=tiles,
                fi_layout=source.fi_layout,
                reference_width=reference_width,
                yaxis_min=yaxis_min,
                yaxis_max=yaxis_max,
                graph_style=graph_style,
                graph_width_ratio=graph_width_ratio,
            )
        )

    meta = {
        "num_frames": num_frames,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "total_width": total_width,
        "has_graph": int(frames is not None),
        "fi_layout": source.fi_layout,
    }
    return output_frames, meta


class GrowthCurveVideoWorker(QThread):
    """Worker thread for creating growth curve videos"""
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, str)  # value, message
    
    def __init__(
        self,
        video_paths,
        excel_path,
        output_dir,
        time_interval,
        yaxis_min,
        yaxis_max,
        fps,
        formats,
        parent=None,
        mask_path=None,
        foci_path=None,
        show_mask_contours: bool = False,
        show_foci: bool = False,
        graph_style: Optional[dict[str, Any]] = None,
        graph_width_ratio: float = 0.5,
        y_column: Optional[str] = None,
        fi_layout: str = DEFAULT_VIDEO_FI_LAYOUT,
    ):
        super().__init__(parent)
        self.setObjectName(type(self).__name__)
        self.video_paths = list(video_paths)
        self.excel_path = excel_path
        self.output_dir = output_dir
        self.time_interval = time_interval
        self.yaxis_min = yaxis_min
        self.yaxis_max = yaxis_max
        self.fps = fps  # Frames per second for video playback
        self.formats = formats  # List of formats to save: ['tif', 'avi', 'gif', 'mp4']
        self.mask_path = mask_path
        self.foci_path = foci_path
        self.show_mask_contours = bool(show_mask_contours)
        self.show_foci = bool(show_foci)
        self.graph_style = dict(graph_style or {})
        self.graph_width_ratio = float(graph_width_ratio)
        self.y_column = y_column
        self.fi_layout = fi_layout

    def run(self) -> None:
        try:
            import numpy as np
            import tifffile
            import cv2

            self.progress.emit(5, "Loading Excel data...")
            self.progress.emit(10, "Loading video frames...")

            def _progress(value: int, message: str) -> None:
                self.progress.emit(value, message)

            try:
                output_frames, meta = build_growth_curve_output_frames(
                    self.video_paths,
                    Path(self.excel_path) if self.excel_path else None,
                    self.time_interval,
                    self.yaxis_min,
                    self.yaxis_max,
                    mask_path=Path(self.mask_path) if self.mask_path else None,
                    foci_path=Path(self.foci_path) if self.foci_path else None,
                    show_mask_contours=self.show_mask_contours,
                    show_foci=self.show_foci,
                    graph_style=self.graph_style,
                    graph_width_ratio=self.graph_width_ratio,
                    y_column=self.y_column,
                    fi_layout=self.fi_layout,
                    progress_callback=_progress,
                )
            except ValueError as e:
                self.finished.emit(False, str(e))
                return

            num_frames = int(meta["num_frames"])
            frame_height = int(meta["frame_height"])
            total_width = int(meta["total_width"])
            self.progress.emit(20, f"Processing {num_frames} frames...")
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
    from bactomate.gui.main_window import main
    main()
