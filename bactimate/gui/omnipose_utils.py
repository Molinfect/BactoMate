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

    # Common places Omnipose/Cellpose might store architecture
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

    # Infer nchan from first conv weight: e.g. conv0.weight has shape (out, in, k, k) -> in = nchan
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state is None and isinstance(ckpt, dict):
        if any(str(k).startswith("conv") or str(k).startswith("0.") for k in keys):
            state = ckpt
        else:
            for k in keys:
                if "conv" in str(k).lower() and "weight" in str(k).lower():
                    state = ckpt
                    break
    # Only use 4D tensors (Conv2d weight: out, in, kH, kW); skip 1D/2D (e.g. BatchNorm)
    if isinstance(state, dict):
        for k, v in state.items():
            if (
                hasattr(v, "shape")
                and "conv" in str(k).lower()
                and "weight" in str(k).lower()
                and len(getattr(v, "shape", [])) == 4
            ):
                try:
                    out["first_layer_shape"] = list(v.shape)
                    if out["nchan"] is None:
                        out["nchan"] = int(v.shape[1])
                except Exception:
                    pass
                break

    return out


def get_default_user_model_architecture() -> tuple[int, int]:
    """
    Default (nchan, nclasses) for BactiMate-trained Omnipose models.
    Must match the training dialog defaults (nchan=2, nclasses=3).
    """
    return (2, 3)


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
