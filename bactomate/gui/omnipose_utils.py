"""
Utilities for Omnipose model loading and checkpoint inspection.
Used to ensure user-trained models are loaded with the same nchan/nclasses as at training.
See: https://omnipose.readthedocs.io (Training, Settings).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_omnipose_checkpoint(pth_path: str | Path) -> dict[str, Any]:
    """
    Load a .pth checkpoint and return top-level keys and any stored nchan/nclasses/diam_mean.
    Omnipose/Cellpose may store these in the checkpoint or in the filename; this inspects the file.

    Returns:
        dict with:
          - "keys": list of top-level keys in the checkpoint
          - "nchan": int or None if present in checkpoint
          - "nclasses": int or None if present in checkpoint
          - "first_layer_shape": shape of first conv weight if state_dict present (to infer input channels)
          - "error": str if loading failed
    """
    pth_path = Path(pth_path)
    out: dict[str, Any] = {"keys": [], "nchan": None, "nclasses": None, "first_layer_shape": None, "output_layer_out_channels": None}
    try:
        import torch
        try:
            ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(pth_path), map_location="cpu")
    except Exception as e:
        out["error"] = str(e)
        return out

    if hasattr(ckpt, "keys"):
        keys = list(ckpt.keys())
    elif isinstance(ckpt, dict):
        keys = list(ckpt.keys())
    else:
        out["keys"] = [type(ckpt).__name__]
        return out

    out["keys"] = keys

    # Top-level nchan/nclasses (may be wrong if written by an older fix; we prefer inferred from weights)
    if "nchan" in keys:
        try:
            out["nchan"] = int(ckpt["nchan"])
        except (TypeError, ValueError):
            pass
    if "nclasses" in keys:
        try:
            out["nclasses"] = int(ckpt["nclasses"])
        except (TypeError, ValueError):
            pass
    if "diam_mean" in keys:
        try:
            out["diam_mean"] = float(ckpt["diam_mean"])
        except (TypeError, ValueError):
            pass

    # Infer nchan from first conv weight (match patched cellpose_omni loader: any 4D weight with conv/encoder/.0.)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state is None and isinstance(ckpt, dict):
        if any(str(k).startswith("conv") or str(k).startswith("0.") for k in keys):
            state = ckpt
        else:
            for k in keys:
                if "conv" in str(k).lower() and "weight" in str(k).lower():
                    state = ckpt
                    break
    if isinstance(state, dict):
        for k, v in state.items():
            if (
                hasattr(v, "shape")
                and "weight" in str(k).lower()
                and len(getattr(v, "shape", [])) == 4
            ):
                key_str = str(k)
                key_lower = key_str.lower()
                # Match conv/encoder/.0./ or keys like "0.weight", "model.0.weight" (Omnipose/Cellpose)
                if (
                    "conv" in key_lower
                    or "encoder" in key_lower
                    or ".0." in key_lower
                    or key_str.startswith("0.")
                ):
                    try:
                        out["first_layer_shape"] = list(v.shape)
                        if out["nchan"] is None:
                            out["nchan"] = int(v.shape[1])
                    except Exception:
                        pass
                    break
        if out["nchan"] is None:
            for k, v in state.items():
                if hasattr(v, "shape") and len(getattr(v, "shape", [])) == 4:
                    try:
                        out["first_layer_shape"] = list(v.shape)
                        out["nchan"] = int(v.shape[1])
                    except Exception:
                        pass
                    break

        # Infer nclasses only when the file doesn't have it. Omnipose uses nclasses=3 (flow+distance+boundary).
        # Prefer 3, then 2; never infer 4 (BactoMate/Omnipose user models use 3).
        if out["nclasses"] is None:
            candidates: list[tuple[str, Any, int]] = []
            for k, v in state.items():
                if not hasattr(v, "shape") or len(getattr(v, "shape", [])) != 4:
                    continue
                try:
                    s = v.shape
                    if s[0] in (2, 3):
                        spatial = int(s[2]) * int(s[3])
                        candidates.append((str(k), v, spatial))
                except Exception:
                    continue
            if candidates:
                candidates.sort(key=lambda x: ("out" not in x[0].lower(), x[2]))
                try:
                    out["nclasses"] = int(candidates[0][1].shape[0])
                except Exception:
                    pass
            if out["nclasses"] is None:
                out["nclasses"] = 3

        # Output head: CPnet has self.output = batchconv(..., nout, ...); state_dict key is "output.conv.*" or "output.proj.*"
        # First dimension of output weight = nout (3 or 4). Use this to confirm checkpoint matches built model (model.cp.nclasses).
        for k, v in state.items():
            key_str = str(k)
            if "output" in key_str.lower() and "weight" in key_str and hasattr(v, "shape"):
                try:
                    s = getattr(v, "shape", ())
                    if len(s) >= 2:
                        out["output_layer_out_channels"] = int(s[0])
                        break
                except Exception:
                    pass

    return out


def fix_user_model_metadata(
    pth_path: str | Path,
    nchan_override: int | None = None,
    nclasses_override: int | None = None,
) -> tuple[bool, str]:
    """
    Add missing nchan/nclasses to a .pth so the loader can load it. No architecture change:
    only for checkpoints that lack these (e.g. saved by Omnipose CLI and copied in). BactoMate-
    trained models already get them from training. When the file has them we keep them unless
    overrides are given (e.g. --nclasses 3 to correct a wrong value).

    Returns:
        (success, message)
    """
    pth_path = Path(pth_path).resolve()
    if not pth_path.is_file():
        return (False, f"File not found: {pth_path}")
    info = inspect_omnipose_checkpoint(pth_path)
    if "error" in info:
        return (False, f"Could not load checkpoint: {info['error']}")
    default_nchan, default_nclasses = get_default_user_model_architecture()
    # Values we will write: override if given, else from file/infer, else default
    write_nchan = (
        nchan_override
        if nchan_override is not None
        else (info.get("nchan") if info.get("nchan") is not None else default_nchan)
    )
    write_nclasses = (
        nclasses_override
        if nclasses_override is not None
        else (info.get("nclasses") if info.get("nclasses") is not None else default_nclasses)
    )
    try:
        import torch
        ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    except Exception as e:
        return (False, str(e))
    if not isinstance(ckpt, dict):
        ckpt = {"state_dict": ckpt}
    # Skip only when file already has the same values we would write (no need to overwrite)
    file_nchan = ckpt.get("nchan")
    file_nclasses = ckpt.get("nclasses")
    if (
        file_nchan is not None
        and file_nclasses is not None
        and int(file_nchan) == write_nchan
        and int(file_nclasses) == write_nclasses
    ):
        return (True, f"Checkpoint already has nchan={write_nchan}, nclasses={write_nclasses}.")
    nchan = write_nchan
    nclasses = write_nclasses
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    ckpt = dict(ckpt)
    ckpt["state_dict"] = state
    ckpt["nchan"] = int(nchan)
    ckpt["nclasses"] = int(nclasses)
    try:
        torch.save(ckpt, pth_path)
    except Exception as e:
        return (False, str(e))
    return (True, f"Written nchan={nchan}, nclasses={nclasses} to {pth_path.name}")


def get_default_user_model_architecture() -> tuple[int, int]:
    """
    Default (nchan, nclasses) for BactoMate-trained Omnipose models.
    Must match the training dialog defaults (nchan=2, nclasses=3).
    """
    return (2, 3)


def convert_checkpoint_to_plain_state_dict(
    pth_path: str | Path,
    out_path: str | Path | None = None,
) -> tuple[bool, str]:
    """
    Save a checkpoint as plain state_dict only (no nchan/nclasses wrapper).
    Built-in Omnipose models are plain state_dict; BactoMate-trained checkpoints
    are dicts with state_dict + nchan + nclasses. Some loaders expect the plain
    format. BactoMate's patch (apply_all_patches) makes cellpose_omni unwrap
    automatically; if you use an unpatched env or another tool, convert first.

    Returns:
        (success, message). On success, out_path is the path written (default: <stem>_plain.pth).
    """
    pth_path = Path(pth_path).resolve()
    if not pth_path.is_file():
        return (False, f"File not found: {pth_path}")
    if out_path is None:
        out_path = pth_path.parent / f"{pth_path.stem}_plain.pth"
    else:
        out_path = Path(out_path).resolve()
    try:
        import torch
        ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    except Exception as e:
        return (False, str(e))
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        return (False, "Checkpoint has no state_dict and is not a plain state_dict")
    # Strip 'module.' prefix if present (same as patched loader)
    first_key = next(iter(state_dict.keys()), "")
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    try:
        torch.save(state_dict, out_path)
    except Exception as e:
        return (False, str(e))
    return (True, f"Saved plain state_dict to {out_path.name}")


# Built-in Omnipose affinity checkpoint (3 output channels); must not use _omni symlink or omni=True at load.
AFFINITY_BUILTIN_MODEL_TYPES = frozenset({"bact_phase_affinity"})


def is_affinity_builtin_model(model_type: str | None) -> bool:
    """True for built-in affinity models (e.g. bact_phase_affinity), not flow-based omni models."""
    return bool(model_type and str(model_type).strip() in AFFINITY_BUILTIN_MODEL_TYPES)


def is_affinity_checkpoint_path(pth_path: str | Path | None) -> bool:
    """True when a checkpoint path/name indicates an affinity model (including stale *_omni aliases)."""
    if not pth_path:
        return False
    stem = Path(pth_path).stem.lower()
    return "affinity" in stem


def uses_affinity_architecture(
    *,
    model_type: str | None = None,
    loader_path: str | Path | None = None,
) -> bool:
    """True if this built-in type or checkpoint should load as 1-channel / 3-class affinity."""
    return is_affinity_builtin_model(model_type) or is_affinity_checkpoint_path(loader_path)


_AFFINITY_LOADER_ALIAS: dict[str, str] = {}


def affinity_loader_path_without_omni_in_stem(checkpoint_path: str | Path) -> str:
    """
    Return a path safe for cellpose_omni CellposeModel.

    The loader sets ``self.omni = ('omni' in filename)`` and ignores the constructor's
    ``omni=`` argument, then forces ``nclasses=4`` when omni is true. Affinity weights are
 3-class; any ``_omni`` symlink/name causes the size-mismatch error.
    """
    resolved = Path(checkpoint_path).resolve()
    key = str(resolved)
    if "omni" not in resolved.stem.lower():
        return key
    cached = _AFFINITY_LOADER_ALIAS.get(key)
    if cached and Path(cached).is_file():
        return cached
    import hashlib
    import os
    import shutil
    import tempfile

    alias_dir = Path(tempfile.gettempdir()) / "bactomate_affinity_loaders"
    alias_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    alias = alias_dir / f"affinity_{digest}.pth"
    if not alias.is_file():
        try:
            os.symlink(resolved, alias)
        except OSError:
            shutil.copy2(resolved, alias)
    # Do not .resolve(): that follows the symlink and restores the _omni filename.
    out = str(alias.absolute())
    _AFFINITY_LOADER_ALIAS[key] = out
    return out


def cellpose_omni_load_flag(
    *,
    model_type: str | None = None,
    pretrained_path: str | Path | None = None,
) -> bool:
    """
    omni= argument for Cellpose(...) at load time.

    Affinity checkpoints are 3-class; forcing omni=True (via _omni symlink or flag) builds a
    4-channel head and causes state_dict size mismatch.
    """
    if uses_affinity_architecture(model_type=model_type, loader_path=pretrained_path):
        return False
    return True


def eval_uses_affinity_segmentation(
    *,
    model_type: str | None = None,
    pretrained_path: str | Path | None = None,
) -> bool:
    """Whether model.eval should pass affinity_seg=True (Omnipose affinity graph)."""
    return uses_affinity_architecture(model_type=model_type, loader_path=pretrained_path)


def prepare_builtin_pretrained_loader_path(
    checkpoint_path: str | Path,
    model_type: str | None,
) -> str:
    """Path for Cellpose(pretrained_model=...). Skips _omni symlink for affinity built-ins."""
    path = Path(checkpoint_path).resolve()
    if uses_affinity_architecture(model_type=model_type, loader_path=path):
        return affinity_loader_path_without_omni_in_stem(path)
    return ensure_omni_in_model_path(path)


def prepare_user_pretrained_loader_path(pretrained_path: str | Path) -> str:
    """Path for user .pth loads. Affinity checkpoints keep their original filename."""
    path = Path(pretrained_path).resolve()
    if is_affinity_checkpoint_path(path):
        return affinity_loader_path_without_omni_in_stem(path)
    return ensure_omni_in_model_path(path)


def bootstrap_cellpose_omni_for_bactomate() -> None:
    """OMNI_MODELS + affinity CellposeModel patch; safe to call repeatedly."""
    try:
        from bactomate.gui.patches import (
            _ensure_omnipose_omni_models_constant,
            _patch_cellpose_omni_affinity_architecture,
        )

        _ensure_omnipose_omni_models_constant()
        _patch_cellpose_omni_affinity_architecture()
    except Exception:
        pass
    try:
        import omnipose.gpu as omni_gpu

        gpu_logger = getattr(omni_gpu, "gpu_logger", None)
        if gpu_logger is not None and not getattr(gpu_logger, "_bactomate_info_safe", False):
            original_info = gpu_logger.info

            def _safe_info(msg, *args, **kwargs):
                if args:
                    try:
                        msg = str(msg) % args
                    except Exception:
                        msg = " ".join([str(msg), *(str(a) for a in args)])
                    args = ()
                return original_info(msg, *args, **kwargs)

            gpu_logger.info = _safe_info
            gpu_logger._bactomate_info_safe = True
    except Exception:
        pass


# Bumped when affinity load logic changes (shown in debug logs).
BACTOMATE_AFFINITY_LOADER_ID = "seg3-materialized-v2"


def materialize_bact_phase_affinity_checkpoint(source: str | Path) -> Path:
    """
    Copy built-in affinity weights to a stable filename without 'omni' in the stem.

    cellpose_omni sets ``self.omni = ('omni' in filename)`` and upgrades to a 4-class head;
    a materialized copy avoids stale *_omni.pth symlinks in cache or temp dirs.
    """
    src = Path(source).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Affinity checkpoint not found: {src}")
    # cellpose_omni checks for the substring "omni" in the *full* path, not just
    # the file name. Keep this materialized copy outside omnipose_pretrained.
    import tempfile

    dest = Path(tempfile.gettempdir()) / "bactomate_affinity_models" / "bact_phase_affinity_seg3.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or dest.stat().st_mtime < src.stat().st_mtime:
        import shutil

        shutil.copy2(src, dest)
    return dest


def load_bact_phase_affinity_model(*, gpu: bool, checkpoint_path: str | Path):
    """
    Load bact_phase_affinity with a 3-class head (never 4-class Omnipose flow head).

    Uses a materialized checkpoint name, then falls back to building the network with
    ``pretrained_model=False`` and loading weights explicitly if needed.
    """
    bootstrap_cellpose_omni_for_bactomate()
    from cellpose_omni.models import CellposeModel

    safe = materialize_bact_phase_affinity_checkpoint(checkpoint_path)
    safe_str = str(safe)

    def _check(model) -> None:
        nc = getattr(model, "nclasses", None)
        if nc not in (3, None):
            raise RuntimeError(
                f"bact_phase_affinity: expected nclasses=3 after load, got {nc} "
                f"(loader={BACTOMATE_AFFINITY_LOADER_ID}, path={safe_str!r})"
            )

    try:
        model = CellposeModel(
            gpu=gpu,
            pretrained_model=safe_str,
            omni=False,
            nchan=1,
            # cellpose_omni adds dim-1 internally; 2 yields the 3-channel affinity head.
            nclasses=2,
            diam_mean=30.0,
        )
        _check(model)
        return model
    except Exception:
        pass

    # Fallback: do not pass a path into __init__ (avoids filename-based omni=True).
    model = CellposeModel(
        gpu=gpu,
        pretrained_model=False,
        omni=False,
        nchan=1,
        nclasses=2,
        diam_mean=30.0,
    )
    model.pretrained_model = [safe_str]
    model.net.load_model(safe_str, cpu=not gpu)
    _check(model)
    return model


def load_cellpose_segmentation_model(
    *,
    gpu: bool,
    loader_path: str | Path,
    model_type: str | None = None,
    pretrained_path: str | Path | None = None,
):
    """
    Load Omnipose/cellpose_omni for segmentation.

    Affinity models use CellposeModel directly with nchan=1 and an abstract
    nclasses=2 (cellpose_omni adds dim-1 internally, giving a 3-channel head).
    This avoids ``_omni`` filename/path aliases forcing a 4-class head.
    """
    bootstrap_cellpose_omni_for_bactomate()
    from cellpose_omni import models

    path = Path(loader_path)
    if uses_affinity_architecture(model_type=model_type, loader_path=path):
        return load_bact_phase_affinity_model(gpu=gpu, checkpoint_path=path)

    load_omni = cellpose_omni_load_flag(model_type=model_type, pretrained_path=pretrained_path)
    if pretrained_path:
        prepared = prepare_user_pretrained_loader_path(pretrained_path)
    elif model_type:
        prepared = prepare_builtin_pretrained_loader_path(path, model_type)
    else:
        prepared = ensure_omni_in_model_path(path)

    try:
        return models.Cellpose(gpu=gpu, pretrained_model=prepared, omni=load_omni)
    except TypeError:
        # Unpatched cellpose_omni Cellpose wrapper has no pretrained_model= kwarg.
        if uses_affinity_architecture(model_type=model_type, loader_path=prepared):
            safe = affinity_loader_path_without_omni_in_stem(prepared)
            return models.CellposeModel(
                gpu=gpu,
                pretrained_model=safe,
                omni=False,
                nchan=1,
                nclasses=2,
                diam_mean=30.0,
            )
        return models.CellposeModel(
            gpu=gpu,
            pretrained_model=prepared,
            omni=load_omni,
            nchan=2,
            nclasses=4 if load_omni else 3,
            diam_mean=30.0,
        )


def ensure_omni_in_model_path(pth_path: str | Path) -> str:
    """
    Return a path to the same checkpoint whose filename stem contains "omni".
    cellpose_omni only sets omni=True when "omni" is in the filename; otherwise
    it loads as standard Cellpose and produces wrong/random detections.
    If the stem already contains "omni", returns the path as-is. Otherwise
    creates a temporary symlink (or copy on Windows) with stem like "{name}_omni".
    Caller should use the returned path for models.Cellpose(pretrained_model=...).
    """
    pth_path = Path(pth_path).resolve()
    if not pth_path.is_file():
        return str(pth_path)
    if is_affinity_checkpoint_path(pth_path):
        return affinity_loader_path_without_omni_in_stem(pth_path)
    stem = pth_path.stem
    if "omni" in stem.lower():
        return str(pth_path)
    import os
    import tempfile
    _dir = Path(tempfile.mkdtemp(prefix="bactomate_omni_"))
    link_path = _dir / f"{stem}_omni.pth"
    try:
        os.symlink(pth_path, link_path)
    except OSError:
        import shutil
        shutil.copy2(pth_path, link_path)
    return str(link_path)
