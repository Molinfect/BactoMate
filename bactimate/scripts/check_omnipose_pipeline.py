#!/usr/bin/env python3
"""
Diagnostic script: check how BactiMate handles Omnipose model creation/saving,
training data (masks + images), inference compatibility, and checkpoint comparison.

Use this when models built from scratch or on top of Omnipose models fail:
  - Verify .pth has correct metadata and will load with omni=True
  - Verify training directory has valid image/mask pairs and Omnipose-ready masks
  - Verify an image is compatible with a given model (channels, shape)
  - Compare checkpoints (weight summary, output-head verdict: likely segments vs 0 cells)

Usage:
  # Check a saved model (.pth)
  python -m bactimate.scripts.check_omnipose_pipeline model path/to/model_omni.pth

  # Check training data (directory + optional filters)
  python -m bactimate.scripts.check_omnipose_pipeline training /path/to/train_dir
  python -m bactimate.scripts.check_omnipose_pipeline training /path/to/train_dir --img-filter _phase --mask-filter _masks

  # Check inference: model + image compatibility
  python -m bactimate.scripts.check_omnipose_pipeline inference path/to/model_omni.pth path/to/image.tif

  # Run all checks (model + training dir + optional image)
  python -m bactimate.scripts.check_omnipose_pipeline all path/to/model_omni.pth /path/to/train_dir --image path/to/test.tif

  # Compare two or more checkpoints (weight stats + output-head verdict)
  python -m bactimate.scripts.check_omnipose_pipeline compare a.pth b.pth [c.pth ...] [--builtin] [--forward image.tif]

  # Full diagnostic for flat/0-cell models (checkpoint, forward pass, training data, BactiMate vs Omnipose)
  python -m bactimate.scripts.diagnose_flat_model omnipose_user_models/1000.pth [--image path/to/image.tif] [--training-dir path] [--compare-builtin bact_phase_omni]

Runs inside BactiMate GUI ("Check pipeline" button) and from CLI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def ensure_builtin_model_cached(model_type: str) -> Optional[Path]:
    """
    Ensure a built-in Omnipose model is downloaded and cached. Calls cellpose_omni's
    model_path(model_type, 0, True), which downloads the model if not present.
    Returns the path to the cached file, or None on failure.
    """
    if not model_type:
        return None
    try:
        from cellpose_omni.models import model_path
        p = model_path(model_type, 0, True)
        path = Path(p).resolve() if p else None
        return path if path and path.is_file() else None
    except Exception:
        return None


def is_resnet_loader_patched() -> bool:
    """
    Return True if cellpose_omni/resnet_torch load_model unwraps checkpoint dicts
    (state_dict = ckpt.get('state_dict', ckpt)). BactiMate's apply_all_patches adds this
    so BactiMate-style checkpoints (with nchan/nclasses) load correctly.
    """
    import sys
    for p in getattr(sys, "path", []):
        resnet = Path(p) / "cellpose_omni" / "resnet_torch.py"
        if resnet.is_file():
            try:
                text = resnet.read_text()
                return "ckpt.get('state_dict'" in text or 'ckpt.get("state_dict"' in text
            except Exception:
                pass
    return False


def get_checkable_model_path(
    model_type: Optional[str],
    user_path: Optional[Path],
    ensure_builtin_cached: bool = False,
) -> Optional[Path]:
    """
    Resolve the current model selection to a path suitable for _check_model.
    - If user_path is set and is a file, return it (user .pth).
    - If model_type is set (built-in), return cache path if it exists
      (e.g. ~/.cellpose/models/bact_phase_omnitorch_0). Returns None if not yet cached.
    - If ensure_builtin_cached is True and model_type is set, call ensure_builtin_model_cached
      first so the model is downloaded if needed.
    """
    if user_path is not None and user_path.resolve().is_file():
        return user_path.resolve()
    if model_type:
        if ensure_builtin_cached:
            path = ensure_builtin_model_cached(model_type)
            if path is not None:
                return path
        try:
            from cellpose_omni.models import MODEL_DIR
            cache_name = f"{model_type}torch_0"
            path = Path(MODEL_DIR) / cache_name
            if path.is_file():
                return path.resolve()
        except Exception:
            pass
    return None


def _check_model(pth_path: Path) -> dict:
    """Check a .pth: inspect checkpoint, filename 'omni', nchan/nclasses. Optionally try loading with cellpose_omni."""
    from bactimate.gui.omnipose_utils import (
        inspect_omnipose_checkpoint,
        get_default_user_model_architecture,
    )

    pth_path = pth_path.resolve()
    if not pth_path.is_file():
        return {"ok": False, "error": f"File not found: {pth_path}"}

    info = inspect_omnipose_checkpoint(pth_path)
    if "error" in info:
        return {"ok": False, "error": info["error"], "inspect": info}

    stem = pth_path.stem
    has_omni_in_name = "omni" in stem.lower()
    default_nchan, default_nclasses = get_default_user_model_architecture()
    nchan = info.get("nchan")
    nclasses = info.get("nclasses")

    # Optional: try loading with cellpose_omni to see what actually gets used
    load_ok, load_msg = None, None
    try:
        from bactimate.gui.omnipose_utils import ensure_omni_in_model_path
        from cellpose_omni import models
        path_for_load = ensure_omni_in_model_path(pth_path)
        model = models.Cellpose(gpu=False, pretrained_model=path_for_load, omni=True)
        if hasattr(model, "cp"):
            load_nchan = getattr(model.cp, "nchan", None)
            load_nclasses = getattr(model.cp, "nclasses", None)
            load_ok = True
            load_msg = f"Loaded: nchan={load_nchan}, nclasses={load_nclasses}"
        else:
            load_ok = True
            load_msg = "Loaded (cp not inspected)"
    except Exception as e:
        load_ok = False
        load_msg = str(e)

    return {
        "ok": True,
        "path": str(pth_path),
        "stem": stem,
        "has_omni_in_name": has_omni_in_name,
        "inspect": info,
        "default_nchan": default_nchan,
        "default_nclasses": default_nclasses,
        "load_ok": load_ok,
        "load_msg": load_msg,
    }


def _validate_mask_file(mask_path: Path) -> tuple[bool, str]:
    """Read mask and check if it's Omnipose-ready: 2D, instance labels (0,1,2,...) or binary (convertible)."""
    try:
        import numpy as np
        import tifffile
        data = tifffile.imread(str(mask_path))
    except Exception as e:
        return (False, f"Read error: {e}")

    if data.ndim == 3:
        if data.shape[-1] in (3, 4):
            gray = data.max(axis=-1)
        else:
            gray = data.max(axis=0)
    elif data.ndim == 2:
        gray = data
    else:
        return (False, f"Unsupported shape {data.shape}")

    unique = np.unique(gray)
    if unique.size == 0:
        return (False, "Empty mask")
    if 0 not in unique and unique.size == 1:
        return (False, "No background (0); single value only")
    # Instance labels: 0 and consecutive positives 1,2,...,N
    if 0 in unique and unique.size > 2:
        expected = set(range(int(unique.max()) + 1))
        if set(unique.astype(int)) != expected:
            return (True, "Instance labels (non-consecutive IDs; Omnipose may still accept)")
        return (True, "Instance labels (0,1,2,...) — OK")
    # Binary
    if 0 in unique and unique.size == 2:
        return (True, "Binary (0 + one other value) — will be converted to instance labels by BactiMate")
    return (True, "Other (will be converted if binary-like)")


def _check_training_data(
    train_dir: Path,
    img_filter: str = "",
    mask_filter: str = "_mask",
    dry_run: bool = True,
) -> dict:
    """Check training directory: discover pairs, validate masks and images, optionally prepare omnipose_masks."""
    from bactimate.gui.mask_utils import (
        OMNIPOSE_MASKS_SUBDIR,
        get_training_pairs,
        prepare_omnipose_masks_from_pairs,
        ensure_omnipose_masks_has_images,
    )
    from bactimate.gui.utils import safe_imread

    train_dir = train_dir.resolve()
    if not train_dir.is_dir():
        return {"ok": False, "error": f"Not a directory: {train_dir}"}

    pairs = get_training_pairs(train_dir, img_filter, mask_filter)
    if not pairs:
        return {
            "ok": False,
            "error": "No image/mask pairs found.",
            "hint": f"Image filter={img_filter!r}, mask filter={mask_filter!r}. "
            "For each image NAME{img_filter}.tif there must be NAME{mask_filter}.tif",
            "train_dir": str(train_dir),
        }
    # Validate each pair
    mask_results = []
    image_results = []
    for img_path, mask_path in pairs:
        status, msg = _validate_mask_file(mask_path)
        mask_results.append((mask_path.name, status, msg))
        try:
            arr = safe_imread(str(img_path))
            if arr is None:
                image_results.append((img_path.name, False, "Could not read"))
            else:
                shp = getattr(arr, "shape", ())
                if arr.ndim == 3 and len(shp) >= 2 and shp[0] < 10:
                    image_results.append((img_path.name, True, f"Shape {shp} (channels first?)"))
                else:
                    image_results.append((img_path.name, True, f"Shape {shp}"))
        except Exception as e:
            image_results.append((img_path.name, False, str(e)))

    # Optionally run prepare (populate omnipose_masks)
    omnipose_masks_dir = train_dir / OMNIPOSE_MASKS_SUBDIR
    prepared_path = None
    n_prepared = 0
    if not dry_run:
        parent = train_dir.parent if train_dir.name == OMNIPOSE_MASKS_SUBDIR else train_dir
        prepared_path, n_prepared = prepare_omnipose_masks_from_pairs(parent, img_filter, mask_filter)
        prepared_path = str(prepared_path)

    return {
        "ok": True,
        "train_dir": str(train_dir),
        "img_filter": img_filter,
        "mask_filter": mask_filter,
        "n_pairs": len(pairs),
        "mask_checks": mask_results,
        "image_checks": image_results,
        "all_masks_ok": all(r[1] for r in mask_results),
        "all_images_ok": all(r[1] for r in image_results),
        "dry_run": dry_run,
        "omnipose_masks_path": prepared_path,
        "n_prepared": n_prepared,
    }


def _check_inference(model_path: Path, image_path: Path) -> dict:
    """Check that an image is compatible with a model (shape, channels) and optionally run one eval."""
    from bactimate.gui.omnipose_utils import ensure_omni_in_model_path, inspect_omnipose_checkpoint
    from bactimate.gui.utils import safe_imread
    import numpy as np

    model_path = model_path.resolve()
    image_path = image_path.resolve()
    if not model_path.is_file():
        return {"ok": False, "error": f"Model not found: {model_path}"}
    if not image_path.is_file():
        return {"ok": False, "error": f"Image not found: {image_path}"}

    info = inspect_omnipose_checkpoint(model_path)
    if "error" in info:
        return {"ok": False, "error": f"Model inspect failed: {info['error']}"}
    nchan = info.get("nchan")
    if nchan is None:
        nchan = 2  # BactiMate default

    img = safe_imread(str(image_path))
    if img is None:
        return {"ok": False, "error": "Could not read image"}
    if hasattr(img, "shape"):
        shape = img.shape
    else:
        shape = getattr(img, "shape", None)

    # BactiMate/Omnipose expect (C, Y, X) or (N, C, Y, X); often 2D (Y, X) -> (1, 1, Y, X) or (1, Y, X)
    if img.ndim == 2:
        img_channels = 1
        compatible = nchan in (1, 2)  # grayscale often duplicated to 2 channels
    elif img.ndim == 3:
        if shape[0] < 10:
            img_channels = shape[0]
        else:
            img_channels = shape[-1] if shape[-1] < 10 else 1
        compatible = img_channels == nchan or (nchan == 2 and img_channels == 1)
    else:
        img_channels = None
        compatible = False

    # Optional: run one eval (same format as BactiMate processing tab)
    eval_ok, eval_msg = None, None
    try:
        from cellpose_omni import models
        path_for_load = ensure_omni_in_model_path(model_path)
        model = models.Cellpose(gpu=False, pretrained_model=path_for_load, omni=True)
        # Prepare image like processing tab: (C, Y, X) for single frame
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.ndim == 2:
            frame = img[np.newaxis, ...]  # (1, Y, X)
        elif img.ndim == 3:
            if img.shape[-1] < 10:
                frame = img.transpose(2, 0, 1)  # (C, Y, X)
            else:
                frame = img
        else:
            frame = img
        result = model.eval(
            frame,
            channels=[0, 0],
            diameter=0,
            flow_threshold=0.0,
            mask_threshold=0.0,
            cellprob_threshold=0.0,
            do_3D=False,
            omni=True,
            cluster=False,
            net_avg=False,
            resample=True,
            verbose=False,
        )
        if isinstance(result, tuple):
            masks = result[0]
            flows = result[1] if len(result) > 1 else None
        else:
            masks = result
            flows = None
        n_cells = 0
        if masks is not None and hasattr(masks, "ndim"):
            n_cells = len(np.unique(masks)) - (1 if 0 in np.unique(masks) else 0)
        eval_ok = True
        eval_msg = f"Eval OK; mask shape {getattr(masks, 'shape', 'N/A')}, {max(0, n_cells)} objects"
    except Exception as e:
        eval_ok = False
        eval_msg = str(e)

    return {
        "ok": True,
        "model_path": str(model_path),
        "image_path": str(image_path),
        "model_nchan": nchan,
        "image_shape": shape,
        "image_channels_inferred": img_channels,
        "channel_compatible": compatible,
        "eval_ok": eval_ok,
        "eval_msg": eval_msg,
    }


def _print_model_report(r: dict) -> None:
    print(format_model_report(r))


def format_model_report(r: dict) -> str:
    """Return model check report as a string (for GUI or CLI)."""
    lines = []
    if not r.get("ok"):
        lines.append(f"Model check FAILED: {r.get('error', 'unknown')}")
        return "\n".join(lines)
    lines.append("=== Model check ===")
    lines.append(f"Path: {r['path']}")
    lines.append(f"Stem: {r['stem']}")
    if not r["has_omni_in_name"]:
        lines.append("*** WARNING: Filename does not contain 'omni'. BactiMate/cellpose_omni need 'omni' in the name to use Omnipose dynamics.")
    else:
        lines.append("Filename contains 'omni' — OK")
    i = r["inspect"]
    lines.append(f"nchan: {i.get('nchan')} (BactiMate default: {r['default_nchan']})")
    file_nclasses = i.get("nclasses")
    lines.append(f"nclasses: {file_nclasses} in file (BactiMate default: {r['default_nclasses']}); after load see 'Load test' for runtime nclasses.")
    lines.append("  (3 = Omnipose parameter flow+dist+boundary; output layer has 4 channels — file 3 vs runtime 4 is same architecture.)")
    lines.append(f"First layer shape: {i.get('first_layer_shape')}")
    lines.append(f"Load test: {'OK' if r.get('load_ok') else 'FAILED'} — {r.get('load_msg', '')}")
    # Checkpoint format vs loader: BactiMate checkpoints have state_dict + nchan + nclasses; built-in are plain state_dict.
    loader_patched = is_resnet_loader_patched()
    lines.append(f"Loader unwraps checkpoint dict: {'Yes (patched)' if loader_patched else 'No (unpatched)'}")
    if not loader_patched:
        lines.append("  → If this is a BactiMate/user checkpoint and segmentation fails, re-run setup/apply_all_patches.py or convert to plain state_dict (see docs).")
    return "\n".join(lines)


def _print_training_report(r: dict) -> None:
    print(format_training_report(r))


def format_training_report(r: dict) -> str:
    """Return training data check report as a string (for GUI or CLI)."""
    lines = []
    if not r.get("ok"):
        lines.append(f"Training data check FAILED: {r.get('error', 'unknown')}")
        if r.get("hint"):
            lines.append(f"Hint: {r['hint']}")
        return "\n".join(lines)
    lines.append("=== Training data check ===")
    lines.append(f"Train dir: {r['train_dir']}")
    lines.append(f"Image filter: {r['img_filter']!r}  Mask filter: {r['mask_filter']!r}")
    lines.append(f"Pairs found: {r['n_pairs']}")
    lines.append("Mask validation:")
    for name, ok, msg in r["mask_checks"]:
        lines.append(f"  {name}: {'OK' if ok else 'FAIL'} — {msg}")
    lines.append("Image validation:")
    for name, ok, msg in r["image_checks"]:
        lines.append(f"  {name}: {'OK' if ok else 'FAIL'} — {msg}")
    lines.append(f"All masks OK: {r['all_masks_ok']}, All images OK: {r['all_images_ok']}")
    if r.get("omnipose_masks_path"):
        lines.append(f"Prepared omnipose_masks: {r['omnipose_masks_path']} ({r['n_prepared']} pairs)")
    elif r.get("dry_run"):
        lines.append("(Dry run; use --prepare to populate omnipose_masks/)")
    return "\n".join(lines)


def _print_inference_report(r: dict) -> None:
    print(format_inference_report(r))


def format_inference_report(r: dict) -> str:
    """Return inference check report as a string (for GUI or CLI)."""
    lines = []
    if not r.get("ok"):
        lines.append(f"Inference check FAILED: {r.get('error', 'unknown')}")
        return "\n".join(lines)
    lines.append("=== Inference check ===")
    lines.append(f"Model: {r['model_path']}")
    lines.append(f"Image: {r['image_path']}")
    lines.append(f"Model nchan: {r['model_nchan']}, Image shape: {r['image_shape']}, Inferred channels: {r.get('image_channels_inferred')}")
    lines.append(f"Channel compatible: {r['channel_compatible']}")
    lines.append(f"Eval test: {'OK' if r.get('eval_ok') else 'FAILED'} — {r.get('eval_msg', '')}")
    return "\n".join(lines)


# --- Checkpoint comparison (weight stats + output-head verdict); used by GUI "Check pipeline" and by CLI mode "compare" ---


def _get_state_dict(path: Path):
    """Load .pth and return state_dict (unwrap if checkpoint dict)."""
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return state if isinstance(state, dict) else None


def _tensor_stats(t) -> dict:
    """Return min, max, mean, std, n_negative, n_total for a tensor."""
    import numpy as np
    a = t.detach().cpu().numpy()
    n_neg = int((a < 0).sum())
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()) if a.size > 0 else 0.0,
        "n_negative": n_neg,
        "n_total": int(a.size),
    }


def run_checkpoint_comparison(
    paths: list[Path],
    labels: list[str],
    *,
    brief: bool = True,
    image_path: Optional[Path] = None,
) -> str:
    """
    Compare one or more Omnipose checkpoints: weight summary, output-head stats, and verdict
    (likely segments vs likely 0 cells from output.0.bias mean).
    Used by the GUI "Check pipeline" (brief=True) and by CLI mode "compare" (brief=False).
    """
    lines: list[str] = []
    if not paths or not labels or len(paths) != len(labels):
        return "Checkpoint comparison: need same number of paths and labels."
    states = []
    for p in paths:
        p = Path(p).resolve()
        if not p.is_file():
            lines.append(f"  File not found: {p}")
            return "\n".join(lines)
        sd = _get_state_dict(p)
        if sd is None:
            lines.append(f"  Failed to load state_dict from {p.name}")
            return "\n".join(lines)
        states.append(sd)
    keys = sorted(states[0].keys())
    for i, s in enumerate(states[1:], 1):
        if set(s.keys()) != set(keys):
            lines.append(f"  Key mismatch: {labels[0]} vs {labels[i]}")
            return "\n".join(lines)

    # Global summary
    lines.append("=== Checkpoint comparison (weight summary) ===")
    lines.append("")
    for s, label in zip(states, labels):
        total_params = 0
        total_negative = 0
        global_min, global_max = float("inf"), float("-inf")
        for k in keys:
            t = s.get(k)
            if t is None or not hasattr(t, "numel"):
                continue
            a = t.detach().cpu().numpy()
            total_params += a.size
            total_negative += (a < 0).sum()
            global_min = min(global_min, float(a.min()))
            global_max = max(global_max, float(a.max()))
        pct_neg = 100.0 * total_negative / total_params if total_params else 0
        lines.append(f"  {label}: params={total_params}, negative={total_negative} ({pct_neg:.1f}%), weight range [{global_min:.4f}, {global_max:.4f}]")
    lines.append("")

    # Output head + verdict
    out_keys = [k for k in keys if "output." in k and ("weight" in k or "bias" in k)]
    if out_keys:
        lines.append("=== Output head (distance/cellprob map; drift here → 0 cells) ===")
        for k in out_keys:
            row = f"  {k:<45}"
            for s, label in zip(states, labels):
                st = _tensor_stats(s[k])
                row += f" | {label[:8]} min={st['min']:.3f} max={st['max']:.3f} mean={st['mean']:.3f}"
            lines.append(row)
        # Resolve key: exact "output.0.bias" or any key ending with it (e.g. module.output.0.bias)
        out0_bias_key = next((k for k in keys if k == "output.0.bias" or k.endswith(".output.0.bias")), None)
        out0_weight_key = next((k for k in keys if k == "output.0.weight" or k.endswith(".output.0.weight")), None)
        if out0_bias_key is not None and out0_bias_key in states[0]:
            lines.append("")
            lines.append("=== Output head verdict (inference uses distance > mask_threshold) ===")
            lines.append("  (Bias mean is from the checkpoint only. At inference, distance map = f(image); it can be much more negative than the bias. If 0 cells with low mask_threshold, check Logs/debug_*.log for the actual distance range.)")
            weight_max_ok = 0.2
            for s, label in zip(states, labels):
                t_b = s[out0_bias_key]
                st_b = _tensor_stats(t_b)
                mean_b = st_b["mean"]
                arr = t_b.detach().cpu().numpy().ravel()
                vals_str = ", ".join(f"{x:.3f}" for x in arr[:8]) + ("..." if arr.size > 8 else "")
                w_max = _tensor_stats(s[out0_weight_key])["max"] if out0_weight_key and out0_weight_key in s else 0.0
                lines.append(f"  [{label}] {out0_bias_key} shape={tuple(t_b.shape)} values=[{vals_str}] mean={mean_b:.3f}")
                if mean_b >= 0:
                    verdict = "likely segments (bias non-negative); if 0 cells try mask threshold -1 or -2"
                elif w_max < weight_max_ok:
                    verdict = f"likely 0 cells (weak head: weight max = {w_max:.3f} < {weight_max_ok})"
                else:
                    verdict = f"may segment (weight max = {w_max:.3f}); if 0 cells try mask threshold -1 or -2"
                lines.append(f"  {label}: output.0.bias mean = {mean_b:.3f}, output.0.weight max = {w_max:.3f}  →  {verdict}")
            # Also report final layer (output.2 = last conv that produces distance/cellprob channels)
            out2_bias_key = next((k for k in keys if k == "output.2.bias" or k.endswith(".output.2.bias")), None)
            if out2_bias_key and out2_bias_key in states[0]:
                lines.append("")
                lines.append("  Final layer (output.2) produces the 4-channel map (flow_y, flow_x, cellprob, distance); distance is compared to mask_threshold:")
                for s, label in zip(states, labels):
                    t2 = s[out2_bias_key]
                    arr2 = t2.detach().cpu().numpy().ravel()
                    vals2 = ", ".join(f"{x:.3f}" for x in arr2)
                    lines.append(f"  [{label}] {out2_bias_key} shape={tuple(t2.shape)} values=[{vals2}] mean={float(arr2.mean()):.3f}")
            lines.append("")
            lines.append("  Tip: Start with 50 epochs at LR 0.01; run Check pipeline after training. If verdict is 'likely segments', the model is good. More epochs (200, 500) can drift the head (negative bias, smaller weight max → 0 cells); re-check the verdict if you increase epochs.")
    lines.append("")

    if not brief:
        # Full per-parameter table (CLI only)
        col_width = 14
        header = f"{'Parameter':<52}"
        for lab in labels:
            header += f" | {lab[:12]:>{col_width}}"
        lines.append("=== Weight statistics per parameter (all checkpoints) ===")
        lines.append("")
        lines.append(header)
        lines.append("  " + "-" * (52 + (col_width + 3) * len(labels)))
        for k in keys:
            v0 = states[0].get(k)
            if v0 is None or not hasattr(v0, "shape"):
                continue
            row = f"{k[:52]:<52}"
            for s in states:
                st = _tensor_stats(s[k])
                cell = f"{st['min']:.3f}/{st['max']:.3f}/{st['mean']:.3f}/{st['std']:.3f}"
                row += f" | {cell:>{col_width}}"
            lines.append(row)
        lines.append("")
        # Small std (collapse)
        lines.append("=== Parameters with very small std (possible collapse) ===")
        for k in keys:
            if "num_batches" in k:
                continue
            for s, label in zip(states, labels):
                t = s.get(k)
                if t is not None and hasattr(t, "shape") and t.numel() > 0:
                    st = _tensor_stats(t)
                    if st["std"] < 1e-4:
                        lines.append(f"  {label}: {k}  std={st['std']:.2e} mean={st['mean']:.4f}")
                        break
        lines.append("")

    if image_path and image_path.is_file():
        _append_forward_comparison(lines, paths, labels, image_path)

    return "\n".join(lines)


def _append_forward_comparison(
    lines: list[str], paths: list[Path], labels: list[str], image_path: Path
) -> None:
    """Append one eval per model and output (mask + flows) stats to lines."""
    import numpy as np
    from bactimate.gui.utils import safe_imread
    from bactimate.gui.omnipose_utils import ensure_omni_in_model_path
    img = safe_imread(str(image_path))
    if img is None:
        lines.append("Forward comparison: could not load image.")
        return
    if img.ndim == 2:
        frame = img[np.newaxis, ...].astype(np.float32)
    elif img.ndim == 3 and img.shape[-1] < 10:
        frame = img.transpose(2, 0, 1).astype(np.float32)
    else:
        frame = img.astype(np.float32)
    if frame.ndim == 2:
        frame = frame[np.newaxis, ...]
    try:
        from cellpose_omni import models
    except Exception as e:
        lines.append(f"Forward comparison: {e}")
        return
    lines.append("=== Forward pass (output stats per model) ===")
    for path, label in zip(paths, labels):
        if not Path(path).is_file():
            continue
        try:
            p = ensure_omni_in_model_path(path)
            model = models.Cellpose(gpu=False, pretrained_model=p, omni=True)
            result = model.eval(
                frame,
                channels=[0, 0],
                diameter=0,
                flow_threshold=0.0,
                mask_threshold=0.0,
                cellprob_threshold=0.0,
                do_3D=False,
                omni=True,
                cluster=False,
                net_avg=False,
                resample=True,
                verbose=False,
            )
            mask = result[0] if isinstance(result, tuple) else result
            flows = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            n_cells = 0
            if mask is not None and hasattr(mask, "ndim"):
                n_cells = len(np.unique(mask)) - (1 if 0 in np.unique(mask) else 0)
            lines.append(f"  {label}: {n_cells} objects")
            if flows is not None:
                try:
                    f = np.asarray(flows)
                    if isinstance(f, np.ndarray):
                        lines.append(f"    Flows: min={f.min():.4f} max={f.max():.4f} mean={f.mean():.4f}")
                    elif hasattr(f, "__iter__"):
                        for i, fi in enumerate(f):
                            arr = np.asarray(fi)
                            lines.append(f"    Flows[{i}]: min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
                except Exception:
                    pass
        except Exception as e:
            lines.append(f"  {label}: ERROR — {e}")
    lines.append("")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check BactiMate Omnipose pipeline: model, training, inference, or compare checkpoints.",
    )
    parser.add_argument(
        "mode",
        choices=["model", "training", "inference", "all", "compare"],
        help="model (.pth), training (dir), inference (model+image), all, or compare (2+ .pth)",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="*",
        help="For model/training: one path. For inference: model then image. For all: model [train_dir]. For compare: two or more .pth paths.",
    )
    parser.add_argument(
        "--img-filter",
        default="",
        help="Image filename suffix to strip for base name (e.g. _phase)",
    )
    parser.add_argument(
        "--mask-filter",
        default="_mask",
        help="Mask filename suffix (e.g. _masks). Default: _mask",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="For training: actually populate omnipose_masks/ (default is dry run)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="For 'all': path to a test image for inference check",
    )
    parser.add_argument(
        "--builtin",
        action="store_true",
        help="For compare: include cached bact_phase_omni in comparison",
    )
    parser.add_argument(
        "--forward",
        type=Path,
        metavar="IMAGE",
        help="For compare: run one eval per model on this image and compare output",
    )
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in (args.path or [])]

    if args.mode == "model":
        if len(paths) != 1:
            print("Error: model mode requires one path to a .pth file", file=sys.stderr)
            return 1
        r = _check_model(paths[0])
        _print_model_report(r)
        return 0 if r.get("ok") else 1

    if args.mode == "training":
        if len(paths) != 1:
            print("Error: training mode requires one path (training directory)", file=sys.stderr)
            return 1
        r = _check_training_data(
            paths[0],
            img_filter=args.img_filter or "",
            mask_filter=args.mask_filter or "_mask",
            dry_run=not args.prepare,
        )
        _print_training_report(r)
        return 0 if r.get("ok") else 1

    if args.mode == "inference":
        if len(paths) != 2:
            print("Error: inference mode requires two paths (model .pth and image)", file=sys.stderr)
            return 1
        r = _check_inference(paths[0], paths[1])
        _print_inference_report(r)
        return 0 if r.get("ok") else 1

    if args.mode == "all":
        if len(paths) < 1:
            print("Error: 'all' requires at least model path (.pth)", file=sys.stderr)
            return 1
        exit_code = 0
        r_model = _check_model(paths[0])
        _print_model_report(r_model)
        if not r_model.get("ok"):
            exit_code = 1
        if len(paths) >= 2 and paths[1].is_dir():
            r_train = _check_training_data(
                paths[1],
                img_filter=args.img_filter or "",
                mask_filter=args.mask_filter or "_mask",
                dry_run=not args.prepare,
            )
            _print_training_report(r_train)
            if not r_train.get("ok"):
                exit_code = 1
        if args.image and args.image.is_file():
            r_inf = _check_inference(paths[0], args.image)
            _print_inference_report(r_inf)
            if not r_inf.get("ok"):
                exit_code = 1
        return exit_code

    if args.mode == "compare":
        if len(paths) < 2:
            print("Error: compare mode requires at least two .pth paths", file=sys.stderr)
            return 1
        # Filter to existing files only
        valid = [(p, p.stem) for p in paths if p.is_file()]
        if not valid:
            print("Error: no valid .pth files found", file=sys.stderr)
            return 1
        for p in paths:
            if not p.is_file():
                print(f"Warning: skipping (not a file): {p}", file=sys.stderr)
        compare_paths = [p for p, _ in valid]
        labels = [lab for _, lab in valid]
        if args.builtin:
            builtin_path = get_checkable_model_path("bact_phase_omni", None, ensure_builtin_cached=True)
            if builtin_path:
                compare_paths.append(Path(builtin_path))
                labels.append("bact_phase_omni")
        if len(compare_paths) < 2:
            print("Error: need at least two checkpoints to compare", file=sys.stderr)
            return 1
        result = run_checkpoint_comparison(
            compare_paths,
            labels,
            brief=False,
            image_path=args.forward,
        )
        print(result)
        print("\n--- Note: 'Mask threshold' in BactiMate is applied to the model output (distance map at inference), not to the saved weights.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
