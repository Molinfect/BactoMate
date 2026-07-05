"""
Subprocess / in-process script worker thread.
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
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from bactomate.gui.log_config import get_logger, log_exception, write_crash_log
from bactomate.gui.paths import (
    configure_macos_bundle_subprocess_env,
    configure_windows_subprocess_env,
    get_app_root,
    get_python_executable_for_subprocess,
    get_subprocess_pythonpath_root,
)

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
                    "Run BactoMate from source (see setup\\windows\\install.bat) or use a feature supported in-process."
                )
                self.finished.emit(False, msg)
                return
            cmd = [python_exe, self.script_path] + self.cmd_args
            env = os.environ.copy()
            configure_macos_bundle_subprocess_env(env)
            configure_windows_subprocess_env(env)
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

