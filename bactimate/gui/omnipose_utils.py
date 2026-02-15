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
    out: dict[str, Any] = {"keys": [], "nchan": None, "nclasses": None, "first_layer_shape": None}
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

        # Infer nclasses from output conv only when the file doesn't have it (avoid overwriting correct 3 with wrong 4).
        # Output layer: 4D weight with shape[0] in (2,3,4); prefer key with "out" or smallest spatial.
        if out["nclasses"] is None:
            candidates: list[tuple[str, Any, int]] = []
            for k, v in state.items():
                if not hasattr(v, "shape") or len(getattr(v, "shape", [])) != 4:
                    continue
                try:
                    s = v.shape
                    if s[0] in (2, 3, 4):
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

    return out


def fix_user_model_metadata(
    pth_path: str | Path,
    nchan_override: int | None = None,
    nclasses_override: int | None = None,
) -> tuple[bool, str]:
    """
    Add missing nchan/nclasses to a .pth so the loader can load it. No architecture change:
    only for checkpoints that lack these (e.g. saved by Omnipose CLI and copied in). BactiMate-
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
    Default (nchan, nclasses) for BactiMate-trained Omnipose models.
    Must match the training dialog defaults (nchan=2, nclasses=3).
    """
    return (2, 3)


def convert_checkpoint_to_plain_state_dict(
    pth_path: str | Path,
    out_path: str | Path | None = None,
) -> tuple[bool, str]:
    """
    Save a checkpoint as plain state_dict only (no nchan/nclasses wrapper).
    Built-in Omnipose models are plain state_dict; BactiMate-trained checkpoints
    are dicts with state_dict + nchan + nclasses. Some loaders expect the plain
    format. BactiMate's patch (apply_all_patches) makes cellpose_omni unwrap
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
    stem = pth_path.stem
    if "omni" in stem.lower():
        return str(pth_path)
    import os
    import tempfile
    _dir = Path(tempfile.mkdtemp(prefix="bactimate_omni_"))
    link_path = _dir / f"{stem}_omni.pth"
    try:
        os.symlink(pth_path, link_path)
    except OSError:
        import shutil
        shutil.copy2(pth_path, link_path)
    return str(link_path)
