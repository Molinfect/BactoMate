#!/usr/bin/env python3
"""
Comprehensive diagnostic for user-trained Omnipose models that produce flat distance/flow
(0 cells even with low mask_threshold). Checks in all directions:

  - Checkpoint: format, keys vs built-in, output-head weights, module. prefix
  - Forward pass: raw 4-channel output stats (distance + flow); compare user vs built-in on same image
  - Training data: format, value ranges, mask instance labels, image (C,Y,X)
  - BactoMate vs Omnipose: what we change (nothing architectural); how we save

Use when a model trained on top of a pretrained Omnipose model gives 0 cells and
Logs/debug_*.log shows constant distance (e.g. min=max=-2.3) and/or flat flow.

Usage:
  python -m bactomate.scripts.diagnose_flat_model omnipose_user_models/1000.pth
  python -m bactomate.scripts.diagnose_flat_model omnipose_user_models/1000.pth --image path/to/image.tif
  python -m bactomate.scripts.diagnose_flat_model omnipose_user_models/1000.pth --training-dir /path/to/train_dir --image path/to/test.tif
  python -m bactomate.scripts.diagnose_flat_model omnipose_user_models/1000.pth --compare-builtin bact_phase_omni --image path/to/image.tif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _subsection(title: str) -> None:
    print()
    print(f"--- {title} ---")


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  {msg}")


def run_checkpoint_checks(user_path: Path, builtin_type: Optional[str]) -> None:
    """Checkpoint format, keys, output head, and optional comparison with built-in."""
    _section("1. CHECKPOINT FORMAT AND OUTPUT HEAD")

    import torch

    user_path = user_path.resolve()
    if not user_path.is_file():
        _fail(f"User model not found: {user_path}")
        return

    try:
        ckpt = torch.load(str(user_path), map_location="cpu", weights_only=False)
    except Exception as e:
        _fail(f"Could not load checkpoint: {e}")
        return

    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        _fail("Checkpoint has no state_dict and is not a plain state_dict.")
        return

    keys = list(state.keys())
    has_module = any(k.startswith("module.") for k in keys)
    if has_module:
        _warn("State dict has 'module.' prefix (DataParallel). BactoMate's post-training step should strip this; if you saved manually, run convert_checkpoint_to_plain or load with patched cellpose_omni.")
    else:
        _ok("No 'module.' prefix in state dict.")

    top_level = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    nchan = ckpt.get("nchan") if isinstance(ckpt, dict) else None
    nclasses = ckpt.get("nclasses") if isinstance(ckpt, dict) else None
    _info(f"Top-level keys: {top_level[:20]}{'...' if len(top_level) > 20 else ''}")
    _info(f"nchan (top-level): {nchan}; nclasses (top-level): {nclasses}")
    if nchan is None or nclasses is None:
        _info("Loader will infer nchan from first conv weight; nclasses often defaults to 3.")

    # Output head
    out_bias = next((k for k in keys if k == "output.0.bias" or k.endswith(".output.0.bias")), None)
    out2_bias = next((k for k in keys if k == "output.2.bias" or k.endswith(".output.2.bias")), None)
    if out_bias and out_bias in state:
        t = state[out_bias]
        arr = t.detach().cpu().numpy().ravel()
        _info(f"output.0.bias shape={t.shape} mean={float(arr.mean()):.4f} (intermediate layer)")
    if out2_bias and out2_bias in state:
        t = state[out2_bias]
        arr = t.detach().cpu().numpy().ravel()
        _info(f"output.2.bias shape={t.shape} values=[{', '.join(f'{x:.3f}' for x in arr)}] (final 4-channel layer)")
        if arr.size == 4 and (arr <= -2).any():
            _warn("One output.2.bias channel is very negative (~distance); at inference the distance map can be constant → 0 cells.")

    # Compare with built-in
    if builtin_type:
        _subsection("Comparison with built-in: " + builtin_type)
        from bactomate.scripts.check_omnipose_pipeline import ensure_builtin_model_cached, run_checkpoint_comparison

        builtin_path = ensure_builtin_model_cached(builtin_type)
        if builtin_path and builtin_path.is_file():
            report = run_checkpoint_comparison(
                [user_path, Path(builtin_path)],
                [user_path.stem, builtin_type],
                brief=True,
            )
            for line in report.splitlines():
                print(f"  {line}")
        else:
            _warn(f"Could not resolve built-in model: {builtin_type}")


def run_forward_pass_checks(
    user_path: Path,
    image_path: Optional[Path],
    builtin_type: Optional[str],
    is_first_training_image: bool = False,
) -> tuple[bool, Optional[int], bool]:
    """Run eval, capture flows, report per-channel stats and n_cells. Returns (had_flat, n_cells_same_image or None, forward_pass_ran)."""
    _section("2. FORWARD PASS (EVAL OUTPUT)")

    import numpy as np

    user_path = user_path.resolve()
    had_flat = True
    n_cells_same_image: Optional[int] = None
    forward_pass_ran = False
    if not user_path.is_file():
        _fail("User model not found.")
        return (had_flat, n_cells_same_image, forward_pass_ran)

    # Use provided image, or first training image when --training-dir was set, or synthetic
    if image_path and image_path.is_file():
        from bactomate.gui.utils import safe_imread
        img = safe_imread(str(image_path))
        if img is None:
            _fail(f"Could not read image: {image_path}")
            return (had_flat, n_cells_same_image, forward_pass_ran)
        _info(f"Using image: {image_path.name} shape={getattr(img, 'shape', None)}")
        if is_first_training_image:
            _info("This is the first training image (model was trained on this). Result = same-image test.")
    else:
        # Small synthetic (1, 128, 128) so it runs quickly
        np.random.seed(42)
        img = np.random.rand(128, 128).astype(np.float32) * 0.5 + 0.25
        _info("Using synthetic 128x128 image (no --image and no --training-dir first image).")

    if img.ndim == 2:
        frame = img[np.newaxis, ...]
    elif img.ndim == 3 and img.shape[-1] < 10:
        frame = np.transpose(img, (2, 0, 1)).astype(np.float32)
    else:
        frame = img.astype(np.float32) if img.dtype != np.float32 else img

    def _ensure_omnipose_omni_models() -> None:
        """Patch omnipose.core so OMNI_MODELS exists (fixes 'cannot import name OMNI_MODELS' in some installs)."""
        try:
            import omnipose.core
            if hasattr(omnipose.core, "OMNI_MODELS"):
                return
            omni = []
            for attr in ("C1_MODELS", "C2_MODELS", "C1_BD_MODELS", "C2_BD_MODELS"):
                if hasattr(omnipose.core, attr):
                    omni.extend(getattr(omnipose.core, attr))
            seen: set = set()
            omni = [x for x in omni if not (x in seen or seen.add(x))]
            if not omni:
                omni = ["bact_omni", "bact_phase_omni", "bact_fluor_omni"]
            omnipose.core.OMNI_MODELS = omni
        except Exception:
            pass

    def run_eval_and_report_flows(model_path: Path) -> dict:
        from bactomate.gui.omnipose_utils import ensure_omni_in_model_path

        _ensure_omnipose_omni_models()
        path_for_load = ensure_omni_in_model_path(model_path)
        try:
            from cellpose_omni import models
            model = models.Cellpose(gpu=False, pretrained_model=str(path_for_load), omni=True)
        except TypeError as e:
            if "pretrained_model" in str(e):
                return {
                    "error": "Cellpose() does not accept pretrained_model= (unpatched cellpose_omni). "
                    "Apply BactoMate patches: from the BactoMate repo run "
                    "python3 setup/apply_all_patches.py "
                    "(this patches the cellpose_omni in your current Python env). Or run from the BactoMate app.",
                    "n_cells": None,
                    "flows_per_channel": None,
                }
            return {"error": str(e), "n_cells": None, "flows_per_channel": None}
        except Exception as e:
            return {"error": str(e), "n_cells": None, "flows_per_channel": None}

        try:
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
        except Exception as e:
            return {"error": str(e), "n_cells": None, "flows_per_channel": None}

        masks = result[0] if isinstance(result, tuple) else result
        flows = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        n_cells = 0
        if masks is not None and hasattr(masks, "ndim"):
            uniq = np.unique(masks)
            n_cells = len(uniq) - (1 if 0 in uniq else 0)

        flows_per_channel = []
        if flows is not None:
            if isinstance(flows, (list, tuple)):
                farr = list(flows)
            elif hasattr(flows, "shape") and getattr(flows, "ndim", 0) >= 3:
                farr = [np.asarray(flows[i]) for i in range(flows.shape[0])]
            else:
                farr = [np.asarray(flows)]
            for idx in range(len(farr)):
                ch = np.asarray(farr[idx])
                if ch.size > 0 and np.issubdtype(ch.dtype, np.floating):
                    c_min, c_max, c_mean = float(np.nanmin(ch)), float(np.nanmax(ch)), float(np.nanmean(ch))
                    flows_per_channel.append((idx, c_min, c_max, c_mean))

        return {"error": None, "n_cells": n_cells, "flows_per_channel": flows_per_channel}

    # User model
    _subsection("User model: " + user_path.stem)
    out = run_eval_and_report_flows(user_path)
    if out.get("error"):
        _fail(out["error"])
    else:
        forward_pass_ran = True
        had_flat = out.get("n_cells") == 0
        if is_first_training_image:
            n_cells_same_image = out.get("n_cells")
        _info(f"n_cells = {out['n_cells']}")
        if out.get("flows_per_channel"):
            for idx, c_min, c_max, c_mean in out["flows_per_channel"]:
                const = " (CONSTANT)" if c_min == c_max else ""
                _info(f"  Channel {idx}: min={c_min:.4f} max={c_max:.4f} mean={c_mean:.4f}{const}")
            # Heuristic: if one channel is constant and negative and n_cells==0, that's likely the distance channel gone flat
            if out["n_cells"] == 0 and out["flows_per_channel"]:
                constants = [(i, mn, mx) for i, mn, mx, _ in out["flows_per_channel"] if mn == mx]
                if constants:
                    _warn("At least one channel is constant. If that is the distance channel, set mask_threshold below that value (see Logs/debug_*.log '[omnipose no cell pixels] dist max=...').")
        else:
            _warn("Could not read flow channels from eval return value.")

    # Built-in comparison
    if builtin_type:
        from bactomate.scripts.check_omnipose_pipeline import ensure_builtin_model_cached
        builtin_path = ensure_builtin_model_cached(builtin_type)
        if builtin_path and builtin_path.is_file():
            _subsection("Built-in: " + builtin_type)
            out_b = run_eval_and_report_flows(Path(builtin_path))
            if out_b.get("error"):
                _fail(out_b["error"])
            else:
                _info(f"n_cells = {out_b['n_cells']}")
                if out_b.get("flows_per_channel"):
                    for idx, c_min, c_max, c_mean in out_b["flows_per_channel"]:
                        const = " (CONSTANT)" if c_min == c_max else ""
                        _info(f"  Channel {idx}: min={c_min:.4f} max={c_max:.4f} mean={c_mean:.4f}{const}")
                if out.get("n_cells") == 0 and out_b.get("n_cells", 0) > 0:
                    _warn("User model gives 0 cells on this image while built-in gives cells → flat/headed output likely from training (epochs, LR, or data).")
    return (had_flat, n_cells_same_image, forward_pass_ran)


def run_training_data_checks(train_dir: Optional[Path], img_filter: str, mask_filter: str) -> None:
    """Training data format, value ranges, mask labels."""
    _section("3. TRAINING DATA (if --training-dir provided)")

    if not train_dir or not train_dir.is_dir():
        _info("No --training-dir provided; skipping training data checks.")
        return

    from bactomate.scripts.check_omnipose_pipeline import _check_training_data
    from bactomate.gui.mask_utils import get_training_pairs
    from bactomate.gui.utils import safe_imread
    import numpy as np

    r = _check_training_data(train_dir, img_filter=img_filter, mask_filter=mask_filter, dry_run=True)
    if not r.get("ok"):
        _fail(r.get("error", "Unknown error"))
        _info(r.get("hint", ""))
        return

    _ok(f"Found {r['n_pairs']} image/mask pair(s).")
    if not r.get("all_masks_ok"):
        _warn("Some mask files failed validation; check mask_checks below.")
    if not r.get("all_images_ok"):
        _warn("Some image files failed validation.")

    pairs = get_training_pairs(train_dir, img_filter, mask_filter)
    if not pairs:
        return
    img_path, mask_path = pairs[0]
    _subsection("First pair: " + img_path.name + " / " + mask_path.name)
    try:
        img = safe_imread(str(img_path))
        mask = safe_imread(str(mask_path))
    except Exception as e:
        _fail(f"Could not read first pair: {e}")
        return
    if img is not None:
        img = np.asarray(img)
        _info(f"Image shape={img.shape} dtype={img.dtype}")
        if img.size > 0 and np.issubdtype(img.dtype, np.floating):
            _info(f"  Value range: min={float(np.nanmin(img)):.4f} max={float(np.nanmax(img)):.4f} mean={float(np.nanmean(img)):.4f}")
        elif img.size > 0:
            _info(f"  Value range: min={int(np.nanmin(img))} max={int(np.nanmax(img))}")
        # Omnipose expects (C, Y, X); BactoMate uses channel_axis 0
        if img.ndim == 3:
            if img.shape[0] < 10:
                _ok("Image is channels-first (C, Y, X) — correct for --channel_axis 0.")
            else:
                _warn("Image is likely (Y, X, C). Omnipose with channel_axis=0 expects (C, Y, X); BactoMate may convert when preparing.")
    if mask is not None:
        mask = np.asarray(mask)
        uniq = np.unique(mask)
        _info(f"Mask shape={mask.shape} dtype={mask.dtype} unique values: {len(uniq)}")
        _info(f"  Value range: min={int(mask.min())} max={int(mask.max())}")
        if len(uniq) <= 2 and 0 in uniq:
            _warn("Mask looks binary (0 and one other value). Omnipose needs instance labels (0, 1, 2, ...). BactoMate's 'Make masks Omnipose fittable' converts.")
        elif 0 in uniq and len(uniq) > 2:
            _ok("Mask has background 0 and multiple object IDs — instance label format.")
    _info("")
    _info("Note: At inference the library normalizes (1st–99th percentile → 0–1) per channel, so the network sees 0–1.")
    _info("Raw value range (e.g. uint16 2452–36453) is not the cause of 0 cells; per-tile vs per-image normalization can be (section 4).")


def run_train_vs_inference_section(
    n_cells_same_image: Optional[int],
    had_flat: bool,
    train_dir: Optional[Path],
) -> None:
    """Explain train vs inference normalization and interpret same-image test."""
    _section("4. TRAIN VS INFERENCE (why same image can give 0 cells or background-as-cell)")

    _info("Training: Omnipose loads images, normalizes (e.g. 0.01–99.99 percentile → 0–1 per channel), then")
    _info("random_crop_warp (random patches). The model sees normalized patches.")
    _info("Inference: We pass the full image; model.eval() normalizes by default (1st–99th percentile → 0–1).")
    _info("So the network sees 0–1 values in both cases, but:")
    _info("  • If the library uses per-tile normalization at inference (tile_norm_blocksize > 0), each tile gets")
    _info("    its own percentile. In a tile that is mostly background, dim pixels can map to high normalized")
    _info("    values → model predicts 'cell' (background predicted as cell).")
    _info("  • Training always had a mix of cell/background in each crop; inference tiles can be 100% background")
    _info("    → different normalized distribution → wrong or 0 cells.")
    _subsection("Same-image test (section 2 used first training image)")
    if n_cells_same_image is not None:
        if n_cells_same_image == 0:
            _fail(f"On the first training image we got 0 cells. The model was trained on this image; it should segment it.")
            _info("This points to: (1) output head drift (bias/weights → constant or wrong-scale distance map), or")
            _info("(2) train/inference input mismatch (e.g. normalization or tiling differs between train and eval).")
        else:
            _ok(f"On the first training image we got {n_cells_same_image} cells.")
            _info("If you still see 0 cells on other images or background predicted as cell, the cause is likely")
            _info("per-tile normalization or different image content, not head drift on this checkpoint.")
    else:
        _info("No same-image test. Run with --training-dir (and no --image) to use the first training image for the forward pass.")
        _info("If section 2 failed (e.g. import error), fix the environment so model.eval() runs, then re-run with --training-dir.")
    _subsection("Why background (mask=0) can be predicted as cell")
    _info("Thin vertical lines or background regions that the model predicts as cell: often the tile containing")
    _info("that region is mostly background, so its 1st–99th percentile are from that tile; slightly brighter")
    _info("pixels (e.g. lines) then map to high normalized values; the model learned 'high normalized = cell'.")
    _info("Fix: use whole-image normalization at inference (e.g. normalize dict with tile_norm_blocksize=0) or")
    _info("normalize the image yourself (1st–99th to 0–1) and call eval(..., normalize=False).")


def run_bactomate_vs_omnipose() -> None:
    """What BactoMate does and does not change."""
    _section("5. BACTOMATE VS OMNIPOSE (what we do)")

    _info("BactoMate does NOT change the Omnipose architecture or training loop.")
    _info("Training: we run 'python -m omnipose --train --dir ... --nchan 2 --nclasses 3 --pretrained_model ...'.")
    _info("After training: we copy the newest .pth, strip 'module.' if present, save plain state_dict only.")
    _info("Inference: we pass the image and thresholds to model.eval(); normalization is done inside the library.")


def run_what_we_learned(
    model_path: Path,
    had_flat: bool,
    n_cells_same_image: Optional[int],
    forward_pass_ran: bool,
    builtin_resolved: bool,
) -> None:
    """Summarize what the diagnostic actually tells us."""
    _section("6. WHAT WE LEARNED")

    import torch

    model_path = model_path.resolve()
    out0_mean: Optional[float] = None
    out2_vals: Optional[list] = None
    try:
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            keys = list(state.keys())
            out0_key = next((k for k in keys if k == "output.0.bias" or k.endswith(".output.0.bias")), None)
            out2_key = next((k for k in keys if k == "output.2.bias" or k.endswith(".output.2.bias")), None)
            if out0_key:
                out0_mean = float(state[out0_key].detach().cpu().numpy().mean())
            if out2_key:
                out2_vals = state[out2_key].detach().cpu().numpy().ravel().tolist()
    except Exception:
        pass

    _subsection("From the checkpoint (section 1)")
    if out0_mean is not None and out2_vals is not None:
        if out0_mean >= 0 and len(out2_vals) == 4 and not all(x <= -1.5 for x in out2_vals):
            _ok("Output head looks reasonable: output.0.bias mean ≥ 0 or output.2.bias not all very negative.")
            _info("This checkpoint is more likely to segment than one with e.g. output.2.bias one channel ≈ -2.3 (flat distance).")
        elif len(out2_vals) == 4 and any(x <= -2.0 for x in out2_vals):
            _warn("One output.2.bias channel is ≤ -2 (likely distance). At inference the distance map can be constant → 0 cells.")
        else:
            _info("Output head: output.0.bias mean = {:.3f}; output.2.bias = {}.".format(out0_mean, [round(x, 3) for x in out2_vals]))
    else:
        _info("Could not read output head from checkpoint.")

    _subsection("Forward pass (section 2)")
    if forward_pass_ran:
        if had_flat:
            _warn("Eval ran but returned 0 cells on this image.")
        else:
            _ok("Eval ran and found cells on this image.")
    else:
        _fail("Eval did not run (import or runtime error). Without it we cannot see n_cells or flow stats.")
        _info("This script now patches omnipose.core.OMNI_MODELS before importing cellpose_omni; if you still see 'OMNI_MODELS', your omnipose/core.py may differ. Run from the BactoMate app (which applies the same patch) to get eval results.")

    if n_cells_same_image is not None:
        _subsection("Same-image test")
        if n_cells_same_image == 0:
            _fail("On the first training image we got 0 cells → model does not segment an image it was trained on.")
        else:
            _ok("On the first training image we got {} cells.".format(n_cells_same_image))

    if not builtin_resolved:
        _info("Built-in model was not compared (could not resolve bact_phase_omni or similar).")


def run_recommendations(
    user_path: Path,
    had_flat: bool,
    n_cells_same_image: Optional[int] = None,
) -> None:
    """Technical, conditional recommendations."""
    _section("7. RECOMMENDATIONS")

    if n_cells_same_image is not None and n_cells_same_image == 0:
        _info("Same training image → 0 cells (section 4). Technical next steps:")
        _info("  1. Try an earlier epoch checkpoint (e.g. 50 or 100); head drift often appears with more epochs.")
        _info("  2. If you have control over eval: try normalize=False and normalize the image yourself (1st–99th")
        _info("     percentile to 0–1) so train and inference use the same scheme.")
        _info("  3. Check that training and inference use the same channel order and value range (section 3).")
    elif had_flat:
        _info("Forward pass gave 0 cells (or failed). Try:")
        _info("  1. Earlier checkpoint; 2. mask_threshold below distance max (Logs/debug_*.log); 3. flow_threshold=0.")
    else:
        _info("Forward pass found cells. If you see 0 cells elsewhere or background-as-cell, see section 4 (per-tile norm).")
    _info("")
    _info("Check pipeline: python -m bactomate.scripts.check_omnipose_pipeline compare USER.pth BUILTIN_PATH")
    _info("Convert to plain state_dict: python -m bactomate.scripts.convert_checkpoint_to_plain USER.pth")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostic for user Omnipose models that produce flat distance/flow (0 cells).",
    )
    parser.add_argument("model", type=Path, help="Path to user .pth model")
    parser.add_argument("--image", type=Path, default=None, help="Image to run forward pass (optional)")
    parser.add_argument("--training-dir", type=Path, default=None, help="Training directory to validate (optional)")
    parser.add_argument("--img-filter", type=str, default="", help="Image filter for training dir (default: '')")
    parser.add_argument("--mask-filter", type=str, default="_mask", help="Mask filter for training dir (default: _mask)")
    parser.add_argument("--compare-builtin", type=str, default=None, help="Built-in model to compare (e.g. bact_phase_omni)")
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1

    # When --training-dir is set and no --image, use first training image for forward pass (same-image test)
    image_path = args.image
    is_first_training_image = False
    if image_path is None and args.training_dir and args.training_dir.is_dir():
        try:
            from bactomate.gui.mask_utils import get_training_pairs
            pairs = get_training_pairs(args.training_dir, args.img_filter, args.mask_filter)
            if pairs:
                image_path = pairs[0][0]
                is_first_training_image = True
        except Exception:
            pass

    run_checkpoint_checks(model_path, args.compare_builtin)
    had_flat, n_cells_same_image, forward_pass_ran = run_forward_pass_checks(
        model_path, image_path, args.compare_builtin, is_first_training_image=is_first_training_image
    )
    run_training_data_checks(args.training_dir, args.img_filter, args.mask_filter)
    run_train_vs_inference_section(n_cells_same_image, had_flat, args.training_dir)
    run_bactomate_vs_omnipose()
    builtin_ok = False
    if args.compare_builtin:
        try:
            from bactomate.scripts.check_omnipose_pipeline import ensure_builtin_model_cached
            builtin_ok = bool(ensure_builtin_model_cached(args.compare_builtin))
        except Exception:
            pass
    run_what_we_learned(model_path, had_flat, n_cells_same_image, forward_pass_ran, builtin_ok)
    run_recommendations(model_path, had_flat, n_cells_same_image=n_cells_same_image)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
