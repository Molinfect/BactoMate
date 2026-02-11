"""
BactiMate runtime patches: pbr versioning, omnipose get_masks, Qt plugin paths.
Must be imported before PyQt6 so Qt env vars are set.
"""
import sys
import os


def _apply_pbr_patch() -> None:
    """Patch pbr.version.VersionInfo for frozen builds to avoid versioning errors."""
    if not (getattr(sys, "frozen", False) or hasattr(sys, "__compiled__")):
        return
    try:
        import pbr.version
        try:
            test_result = pbr.version.VersionInfo("test")
            if not hasattr(test_result, "release_string"):
                raise AttributeError("Not properly patched")
        except (Exception, SystemExit):
            class FakeVersionInfo:
                def __init__(self, project_name):
                    self.project_name = project_name
                    default_versions = {"mgen": "1.2.1"}
                    self.version = default_versions.get(project_name, "0.0.0")

                def release_string(self):
                    return self.version

                def version_string(self):
                    return self.version

                def semantic_version(self):
                    class FakeSemVer:
                        def __init__(self, v):
                            self.version_string = v
                            parts = v.split(".")
                            self.major = int(parts[0]) if len(parts) > 0 else 0
                            self.minor = int(parts[1]) if len(parts) > 1 else 0
                            self.patch = int(parts[2]) if len(parts) > 2 else 0

                        def release_string(self):
                            return self.version_string

                        def version_tuple(self):
                            return (self.major, self.minor, self.patch)

                    return FakeSemVer(self.version)

            pbr.version.VersionInfo = lambda name: FakeVersionInfo(name)
    except (ImportError, Exception):
        pass


def _patch_omnipose_get_masks() -> None:
    """Patch omnipose.core.get_masks to handle bd=None in frozen builds."""
    try:
        import omnipose.core as oc
        import numpy as np

        if not hasattr(oc, "get_masks") or getattr(oc.get_masks, "_bactimate_patched", False):
            return
        _orig_get_masks = oc.get_masks

        def get_masks_safe(*args, **kwargs):
            if len(args) >= 2:
                bd = args[1]
                if bd is None:
                    shape = None
                    if len(args) >= 3 and args[2] is not None and hasattr(args[2], "shape"):
                        shape = args[2].shape
                    elif len(args) >= 4 and args[3] is not None and hasattr(args[3], "shape"):
                        shape = args[3].shape
                    elif len(args) >= 1 and args[0] is not None and hasattr(args[0], "shape"):
                        shape = args[0].shape

                    if shape is not None:
                        bd = np.full(shape, -2, dtype=np.float32)
                    else:
                        for arg in args:
                            if arg is not None and hasattr(arg, "shape") and len(arg.shape) >= 2:
                                shape = arg.shape
                                bd = np.full(shape, -2, dtype=np.float32)
                                break
                        if bd is None:
                            bd = np.array([[-2.0]], dtype=np.float32)

                    args = list(args)
                    args[1] = bd
                    args = tuple(args)

            if len(args) >= 2 and args[1] is None:
                args = list(args)
                args[1] = np.array([[-2.0]], dtype=np.float32)
                args = tuple(args)

            return _orig_get_masks(*args, **kwargs)

        get_masks_safe._bactimate_patched = True
        get_masks_safe.__wrapped__ = _orig_get_masks
        oc.get_masks = get_masks_safe
    except (ImportError, AttributeError, Exception):
        pass


def _apply_qt_paths() -> None:
    """Set Qt plugin/lib paths before any Qt import. Exposes base_path on this module."""
    global base_path
    if getattr(sys, "frozen", False) or hasattr(sys, "__compiled__"):
        base_path = os.path.dirname(sys.executable)
    else:
        # Repo root (bactimate/gui/patches.py -> ../.. = bactimate, ../../.. = repo) for venv and img/
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    abs_plugin_path = None
    abs_lib_dir = None

    if sys.platform not in ("darwin", "win32"):
        return

    if getattr(sys, "frozen", False) or hasattr(sys, "__compiled__"):
        plugin_candidate = os.path.join(base_path, "PyQt6", "Qt6", "plugins")
        if os.path.exists(plugin_candidate):
            abs_plugin_path = os.path.abspath(plugin_candidate)
            if sys.platform == "darwin":
                lib_candidate = os.path.join(base_path, "PyQt6", "Qt6", "lib")
                abs_lib_dir = os.path.abspath(lib_candidate) if os.path.exists(lib_candidate) else None
    else:
        script_dir = base_path
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_candidates = [
            os.path.join(script_dir, "venv_py310"),
            os.path.join(script_dir, "venv"),
        ]
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            venv_candidates.insert(0, sys.prefix)

        for venv_dir in venv_candidates:
            if not os.path.exists(venv_dir):
                continue
            for py_ver in [python_version, "python3.11", "python3.10"]:
                candidate = os.path.join(
                    venv_dir, "lib", py_ver, "site-packages", "PyQt6", "Qt6", "plugins"
                )
                if os.path.exists(candidate):
                    abs_plugin_path = os.path.abspath(candidate)
                    qt_lib_candidate = os.path.join(
                        venv_dir, "lib", py_ver, "site-packages", "PyQt6", "Qt6", "lib"
                    )
                    if os.path.exists(qt_lib_candidate):
                        abs_lib_dir = os.path.abspath(qt_lib_candidate)
                    break
            if abs_plugin_path:
                break

        if not abs_plugin_path:
            try:
                import PyQt5
                pyqt5_path = os.path.dirname(PyQt5.__file__)
                candidate = os.path.join(pyqt5_path, "Qt5", "plugins")
                if os.path.exists(candidate):
                    abs_plugin_path = os.path.abspath(candidate)
                    qt_lib_candidate = os.path.join(pyqt5_path, "Qt5", "lib")
                    if os.path.exists(qt_lib_candidate):
                        abs_lib_dir = os.path.abspath(qt_lib_candidate)
            except ImportError:
                pass

    if not abs_plugin_path:
        return

    if sys.platform == "darwin":
        conflicting_paths = ["/opt/homebrew", "/usr/local", "/anaconda", "/miniconda"]
        if "QT_PLUGIN_PATH" in os.environ:
            current_val = os.environ["QT_PLUGIN_PATH"]
            if any(c in current_val for c in conflicting_paths):
                os.environ["QT_PLUGIN_PATH"] = abs_plugin_path
            elif abs_plugin_path not in current_val:
                os.environ["QT_PLUGIN_PATH"] = abs_plugin_path + ":" + current_val
        else:
            os.environ["QT_PLUGIN_PATH"] = abs_plugin_path

        if abs_lib_dir and os.path.exists(abs_lib_dir):
            current_dyld = os.environ.get("DYLD_FRAMEWORK_PATH", "")
            if current_dyld:
                paths = current_dyld.split(":")
                filtered = [p for p in paths if not any(c in p for c in conflicting_paths)]
                os.environ["DYLD_FRAMEWORK_PATH"] = (
                    abs_lib_dir + ":" + ":".join(filtered) if filtered else abs_lib_dir
                )
            else:
                os.environ["DYLD_FRAMEWORK_PATH"] = abs_lib_dir

        if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
            qpa_val = os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]
            if any(c in qpa_val for c in conflicting_paths):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = abs_plugin_path
    elif sys.platform == "win32":
        os.environ["QT_PLUGIN_PATH"] = abs_plugin_path


# Module-level base_path for logo/bundle resolution (set by _apply_qt_paths)
base_path = None

# Run all patches on import
_apply_pbr_patch()
is_frozen = getattr(sys, "frozen", False) or hasattr(sys, "__compiled__")
if is_frozen:
    _patch_omnipose_get_masks()
_apply_qt_paths()
