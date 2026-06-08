"""
BactoMate runtime patches: pbr versioning, omnipose get_masks, Qt plugin paths.
Must be imported before PyQt6 so Qt env vars are set.
"""
from __future__ import annotations

import sys
import os


def _apply_pbr_patch() -> None:
    """Patch pbr.version.VersionInfo for packaged builds to avoid versioning errors."""
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
    """
    Compatibility fix: ensure omnipose.core.get_masks never receives bd=None.
    In some environments (packaged builds, certain user models) bd can be None,
    causing TypeError (e.g. '>' not supported between 'NoneType' and 'int').
    This only substitutes a fallback array when bd is None; it does not change
    flow_threshold, mask_threshold, diameter, or any other Omnipose behavior.
    """
    try:
        import omnipose.core as oc
        import numpy as np

        if not hasattr(oc, "get_masks") or getattr(oc.get_masks, "_bactomate_patched", False):
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

        get_masks_safe._bactomate_patched = True
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
        # Repo root (bactomate/gui/patches.py -> .. = bactomate, ../.. = repo) for venv and img/
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    abs_plugin_path = None
    abs_lib_dir = None

    if sys.platform not in ("darwin", "win32"):
        return

    if getattr(sys, "frozen", False) or hasattr(sys, "__compiled__"):
        frozen_bases = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            frozen_bases.append(meipass)
        frozen_bases.extend(
            [
                base_path,
                os.path.join(base_path, "_internal"),
            ]
        )
        for frozen_base in frozen_bases:
            plugin_candidate = os.path.join(frozen_base, "PyQt6", "Qt6", "plugins")
            if os.path.exists(plugin_candidate):
                abs_plugin_path = os.path.abspath(plugin_candidate)
                lib_candidate = os.path.join(frozen_base, "PyQt6", "Qt6", "lib")
                if os.path.exists(lib_candidate):
                    abs_lib_dir = os.path.abspath(lib_candidate)
                break
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
                import PyQt6
                pyqt6_path = os.path.dirname(PyQt6.__file__)
                candidate = os.path.join(pyqt6_path, "Qt6", "plugins")
                if os.path.exists(candidate):
                    abs_plugin_path = os.path.abspath(candidate)
                    qt_lib_candidate = os.path.join(pyqt6_path, "Qt6", "lib")
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

def _ensure_omnipose_omni_models_constant() -> None:
    """cellpose_omni imports OMNI_MODELS at import time; define it when missing (common on macOS installs)."""
    try:
        import omnipose.core as oc
    except ImportError:
        return
    if hasattr(oc, "OMNI_MODELS"):
        return
    names: list = []
    for attr in ("C1_MODELS", "C2_MODELS", "C1_BD_MODELS", "C2_BD_MODELS"):
        names.extend(getattr(oc, attr, []) or [])
    seen: set = set()
    oc.OMNI_MODELS = [x for x in names if not (x in seen or seen.add(x))]
    if not oc.OMNI_MODELS:
        oc.OMNI_MODELS = ["bact_omni", "bact_phase_omni", "bact_fluor_omni", "bact_phase_affinity"]


def _patch_cellpose_omni_affinity_architecture() -> None:
    """
    cellpose_omni CellposeModel ignores constructor omni= when the checkpoint path contains
    'omni' (e.g. stale *_omni.pth symlinks) and then builds a 4-class head for 3-class affinity weights.
    """
    _ensure_omnipose_omni_models_constant()
    try:
        from cellpose_omni.models import CellposeModel
    except ImportError:
        return
    if getattr(CellposeModel.__init__, "_bactomate_affinity_patch", False):
        return

    from bactomate.gui.omnipose_utils import (
        affinity_loader_path_without_omni_in_stem,
        uses_affinity_architecture,
    )

    _orig_init = CellposeModel.__init__

    def _checkpoint_path(pretrained_model) -> str | None:
        if not pretrained_model:
            return None
        if isinstance(pretrained_model, (list, tuple)):
            return str(pretrained_model[0]) if pretrained_model else None
        return str(pretrained_model)

    def _patched_init(self, *args, **kwargs):
        model_type = kwargs.get("model_type")
        pretrained = kwargs.get("pretrained_model", False)
        ckpt_path = _checkpoint_path(pretrained)
        if uses_affinity_architecture(model_type=model_type, loader_path=ckpt_path):
            kwargs["omni"] = False
            kwargs["nchan"] = 1
            # cellpose_omni adds dim-1 to nclasses internally; 2 -> 3 output channels in 2D.
            kwargs["nclasses"] = 2
            if kwargs.get("diam_mean") in (None, 27.0):
                kwargs["diam_mean"] = 30.0
            if ckpt_path:
                safe = affinity_loader_path_without_omni_in_stem(ckpt_path)
                if isinstance(pretrained, list):
                    kwargs["pretrained_model"] = [safe]
                elif isinstance(pretrained, tuple):
                    kwargs["pretrained_model"] = (safe,)
                else:
                    kwargs["pretrained_model"] = safe
        return _orig_init(self, *args, **kwargs)

    _patched_init._bactomate_affinity_patch = True
    CellposeModel.__init__ = _patched_init


# Run all patches on import
_apply_pbr_patch()
is_frozen = getattr(sys, "frozen", False) or hasattr(sys, "__compiled__")
if is_frozen:
    _patch_omnipose_get_masks()
_patch_cellpose_omni_affinity_architecture()
_apply_qt_paths()
