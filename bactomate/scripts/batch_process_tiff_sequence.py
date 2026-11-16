#!/usr/bin/env python3
"""
Batch process multipage TIFF sequences with Omnipose.

This script:
1. Splits multipage TIFF into individual frames
2. Runs Omnipose segmentation on each frame
3. Concatenates masks back into a sequence
4. Optionally overlays masks with original images
5. Saves results

Usage:
    python batch_process_tiff_sequence.py input.tif [options]
"""

import argparse
import os
import sys
import traceback
import warnings
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil
from datetime import datetime

from bactomate.modules.label_render import draw_hq_labels


def _append_omni_debug(msg: str) -> None:
    """Append a timestamped line to BactoMate debug log if env BACTOMATE_OMNI_DEBUG_LOG is set (for Run button / batch path)."""
    path = os.environ.get("BACTOMATE_OMNI_DEBUG_LOG")
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [batch] {msg}\n")
    except Exception:
        pass

# Compatibility wrapper for NumPy 2.0
def safe_imread(file_path):
    """
    Safely read TIFF file with NumPy 2.0 compatibility.
    Handles the newbyteorder() issue with tifffile.
    """
    try:
        # Try standard reading first
        return tifffile.imread(file_path)
    except (AttributeError, TypeError) as e:
        if 'newbyteorder' in str(e):
            # NumPy 2.0 compatibility issue - use PIL as fallback
            try:
                from PIL import Image
                # Use PIL to read TIFF (handles NumPy 2.0 fine)
                img = Image.open(file_path)
                frames = []
                try:
                    while True:
                        frames.append(np.array(img))
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass
                
                if len(frames) == 1:
                    return frames[0]
                elif len(frames) > 1:
                    return np.stack(frames, axis=0)
                else:
                    # Single frame case
                    img.seek(0)
                    return np.array(img)
            except ImportError:
                raise RuntimeError(
                    "Required dependency Pillow is missing (needed for TIFF reading). "
                    "Reinstall BactoMate dependencies: pip install -r requirements.txt"
                ) from e
            except Exception as pil_error:
                # If PIL also fails, raise original error
                raise e from pil_error
        else:
            raise

# Configure tqdm to work better when output is redirected (not a terminal)
# Disable tqdm progress bars when output is redirected to avoid cluttering logs
import sys
import os

# Disable tqdm when output is redirected (e.g., when called from GUI)
if not sys.stdout.isatty():
    # When output is redirected, disable tqdm entirely for cleaner logs
    # Set environment variable to disable tqdm in omnipose/cellpose
    os.environ['TQDM_DISABLE'] = '1'
    tqdm_kwargs = {'disable': True}  # Disable our own tqdm too
else:
    tqdm_kwargs = {}
    # Use a simple one-line progress format (no ### bar, no rate/s per it) for all tqdm in this process
    # This affects our tqdm and Omnipose/cellpose tqdm; patch before importing cellpose_omni
    _tqdm_orig_init = tqdm.__init__

    def _tqdm_init_with_simple_format(self, *args, **kwargs):
        kwargs.setdefault(
            'bar_format',
            '{percentage:3.0f}% ({n_fmt}/{total_fmt})  ETA {remaining}',
        )
        _tqdm_orig_init(self, *args, **kwargs)

    tqdm.__init__ = _tqdm_init_with_simple_format

# Suppress FutureWarning from omnipose about deprecated area_threshold parameter
warnings.filterwarnings('ignore', category=FutureWarning, message='.*area_threshold.*')

try:
    from cellpose_omni import models
    from cellpose_omni.io import imread
except ImportError:
    print("ERROR: cellpose_omni not found. Make sure you're in the Omnipose environment.")
    print("Activate with: source venv/bin/activate")
    sys.exit(1)


def _configure_torch_sparse_invariants() -> None:
    """
    Explicitly configure torch sparse invariant checks.

    PyTorch warns that checks are implicitly disabled and invalid sparse tensors may
    crash natively (e.g. Bus error / SIGBUS). We default to ENABLED for safety.

    Override with env var BACTOMATE_SPARSE_INVARIANTS:
      - "1"/"true"/"on" (default): enable checks
      - "0"/"false"/"off": disable checks (faster, less safe)
    """
    raw = str(os.environ.get("BACTOMATE_SPARSE_INVARIANTS", "1")).strip().lower()
    enable = raw not in {"0", "false", "off", "no"}
    try:
        import torch
        checker = getattr(getattr(torch, "sparse", None), "check_sparse_tensor_invariants", None)
        if checker is None:
            return
        if enable and hasattr(checker, "enable"):
            checker.enable()
            print("PyTorch sparse invariant checks: enabled")
        elif (not enable) and hasattr(checker, "disable"):
            checker.disable()
            print("PyTorch sparse invariant checks: disabled")
    except Exception as e:
        print(f"Warning: could not configure PyTorch sparse invariant checks: {e}")


_configure_torch_sparse_invariants()


def split_multipage_tiff(input_path, output_dir=None, prefix="frame"):
    """
    Split a multipage TIFF into individual frames.
    
    Parameters:
    -----------
    input_path : str
        Path to input multipage TIFF
    output_dir : str, optional
        Directory to save individual frames. If None, uses temp directory.
    prefix : str
        Prefix for output frame filenames
        
    Returns:
    --------
    frame_paths : list
        List of paths to individual frame files
    temp_dir : str or None
        Temporary directory created (if any), for cleanup
    """
    print(f"Loading multipage TIFF: {input_path}")
    images = safe_imread(input_path)
    
    # Handle different input formats
    if images.ndim == 2:
        # Single 2D image
        images = images[np.newaxis, ...]
    elif images.ndim == 3:
        # Could be (T, Y, X) or (Y, X, C)
        if images.shape[-1] < 10:  # Likely channels last
            images = images.transpose(2, 0, 1)  # (C, Y, X)
            images = images[np.newaxis, ...]  # (1, C, Y, X)
        # Otherwise assume (T, Y, X) or (T, C, Y, X)
    elif images.ndim == 4:
        # Could be (T, C, Y, X) or (C, T, Y, X) or (T, Y, X, C)
        if images.shape[-1] < 10:  # Channels last
            images = images.transpose(0, 3, 1, 2)  # (T, C, Y, X)
        elif images.shape[0] < 10:  # Channels first
            images = images.transpose(1, 0, 2, 3)  # (T, C, Y, X)
    
    num_frames = images.shape[0]
    print(f"Found {num_frames} frames")
    
    # Create output directory
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="omnipose_frames_")
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    print("Saving individual frames...")
    for i in tqdm(range(num_frames), desc="Splitting frames", **tqdm_kwargs):
        frame = images[i]
        frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.tif")
        tifffile.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    return frame_paths, temp_dir, images.shape


def process_frames_with_omnipose(frame_paths, model_type='cyto2_omni', pretrained_model=None,
                                  diameter=0, channels=[0, 0],
                                  use_gpu=True, batch_size=8,
                                  flow_threshold=0.4, mask_threshold=0.0,
                                  omni=True, cluster=False, verbose=False):
    """
    Process individual frames with Omnipose.

    Parameters:
    -----------
    frame_paths : list
        List of paths to frame images
    model_type : str
        Omnipose model type (e.g., 'bact_phase_omni'). Ignored if pretrained_model is set.
    pretrained_model : str, optional
        Path to a user-trained .pth model. If set, model_type is ignored.
    diameter : float
        Cell diameter (0 for auto-detection)
    channels : list
        [channel1, channel2] for segmentation
    use_gpu : bool
        Use GPU if available
    batch_size : int
        Batch size for processing
    flow_threshold : float
        Flow error threshold
    mask_threshold : float
        Mask threshold
    omni : bool
        Use Omnipose mask reconstruction
    cluster : bool
        Use DBSCAN clustering
    verbose : bool
        Verbose output
        
    Returns:
    --------
    masks : list
        List of mask arrays
    """
    _original_pretrained_arg = pretrained_model

    if pretrained_model:
        print(f"\nInitializing Omnipose model from user checkpoint: {pretrained_model}")
    else:
        print(f"\nInitializing Omnipose model: {model_type}")

    from urllib.error import HTTPError

    if pretrained_model:
        if not os.path.isfile(pretrained_model):
            raise FileNotFoundError(f"User model not found: {pretrained_model}")
        from bactomate.gui.omnipose_utils import load_cellpose_segmentation_model

        pretrained_model = str(Path(pretrained_model).resolve())
        _ckpt_path_for_inspect = pretrained_model
        # Inspect checkpoint so we can log architecture (user vs built-in must match training)
        try:
            from bactomate.gui.omnipose_utils import inspect_omnipose_checkpoint, get_default_user_model_architecture
            ckpt_info = inspect_omnipose_checkpoint(_ckpt_path_for_inspect)
            if "error" not in ckpt_info:
                print(f"Checkpoint keys (first 15): {ckpt_info.get('keys', [])[:15]}")
                if ckpt_info.get("nchan") is not None or ckpt_info.get("first_layer_shape"):
                    print(f"Checkpoint nchan: {ckpt_info.get('nchan')} (from file or inferred from first conv shape {ckpt_info.get('first_layer_shape')})")
                if ckpt_info.get("nclasses") is not None:
                    print(f"Checkpoint nclasses: {ckpt_info.get('nclasses')}")
            def_nchan, def_nclasses = get_default_user_model_architecture()
            print(f"BactoMate inference uses nchan={def_nchan}, nclasses={def_nclasses} (must match training).")
        except Exception as _e:
            pass  # non-fatal
        model = load_cellpose_segmentation_model(
            gpu=use_gpu,
            loader_path=pretrained_model,
            pretrained_path=pretrained_model,
        )
        # Log what the loaded model actually has (cellpose_omni may read from filename or use defaults)
        try:
            if hasattr(model, 'cp') and hasattr(model.cp, 'nchan') and hasattr(model.cp, 'nclasses'):
                print(f"Loaded model architecture: nchan={model.cp.nchan}, nclasses={model.cp.nclasses}")
                if (getattr(model.cp, 'nchan', None) != 2 or getattr(model.cp, 'nclasses', None) not in (3, 4)):
                    print("WARNING: Model nchan/nclasses differ from BactoMate training defaults (2, 3). If you trained with BactoMate, this may cause wrong or no detections.")
                _append_omni_debug(
                    f"USER_MODEL post-load path={pretrained_model!r} model.cp.nchan={model.cp.nchan} model.cp.nclasses={model.cp.nclasses}"
                )
        except Exception:
            pass
    else:
        # Built-in Omnipose: mirror weights into BactoMate omnipose_pretrained cache, then load by path
        # (avoids relying only on ~/.cellpose/models; same file naming as cellpose_omni).
        from bactomate.gui.omnipose_pretrained_cache import ensure_builtin_in_bactomate_cache
        from bactomate.gui.omnipose_utils import (
            eval_uses_affinity_segmentation,
            is_affinity_builtin_model,
            load_bact_phase_affinity_model,
            load_cellpose_segmentation_model,
        )

        bact_checkpoint = ensure_builtin_in_bactomate_cache(model_type)
        print(f"\nBuilt-in Omnipose checkpoint (BactoMate cache): {bact_checkpoint}")
        if is_affinity_builtin_model(model_type):
            model = load_bact_phase_affinity_model(gpu=use_gpu, checkpoint_path=bact_checkpoint)
        else:
            model = load_cellpose_segmentation_model(
                gpu=use_gpu,
                loader_path=bact_checkpoint,
                model_type=model_type,
            )
        try:
            if hasattr(model, 'cp'):
                nchan = getattr(model.cp, 'nchan', None)
                nclasses = getattr(model.cp, 'nclasses', None)
                _append_omni_debug(
                    f"BUILTIN post-load model_type={model_type!r} model.cp.nchan={nchan} model.cp.nclasses={nclasses}"
                )
        except Exception:
            pass

    from bactomate.gui.omnipose_utils import eval_uses_affinity_segmentation

    affinity_seg = eval_uses_affinity_segmentation(
        model_type=model_type if not _original_pretrained_arg else None,
        pretrained_path=_original_pretrained_arg,
    )

    try:
        # Additional optimizations if available
        try:
            import torch
            if use_gpu and torch.cuda.is_available():
                # Enable cuDNN benchmarking for faster inference (finds optimal algorithms)
                torch.backends.cudnn.benchmark = True
                # Use deterministic algorithms for reproducibility (slower) - disabled for speed
                torch.backends.cudnn.deterministic = False
                
                # Memory pooling optimizations
                # Enable memory pool allocator for better memory reuse
                # This reduces fragmentation and allows larger batch sizes
                torch.cuda.empty_cache()  # Clear any existing cache
                # PyTorch uses memory pools by default, but we can optimize further
                # Set memory fraction to allow better memory management (optional)
                # torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
                
                print("GPU optimizations enabled (cuDNN benchmarking, memory pooling)")
        except:
            pass  # If torch not available or other error, continue without optimizations
        
        # Verify architecture is correct (for C2_BD models: nchan=2, nclasses=4); skip for user models
        segmentation_model = getattr(model, 'cp', model)
        if not pretrained_model and hasattr(segmentation_model, 'nchan') and hasattr(segmentation_model, 'nclasses'):
            expected_nchan = 2 if model_type in ['bact_phase_omni', 'cyto2_omni', 'bact_fluor_omni'] else 1
            expected_nclasses = 4 if model_type in ['bact_phase_omni', 'cyto2_omni', 'bact_fluor_omni'] else 3
            if segmentation_model.nchan != expected_nchan or segmentation_model.nclasses != expected_nclasses:
                print(f"WARNING: Architecture mismatch detected!")
                print(f"  Expected: nchan={expected_nchan}, nclasses={expected_nclasses}")
                print(f"  Got: nchan={segmentation_model.nchan}, nclasses={segmentation_model.nclasses}")
                print(f"  This may cause errors. Trying to fix...")
                # Try to fix by recreating with correct parameters
                raise RuntimeError("Architecture mismatch - need to recreate model")
    except RuntimeError as e:
        if "Architecture mismatch" in str(e) and not pretrained_model:
            # Recreate built-in: clear BactoMate mirror + upstream file, re-download and reload by path
            from bactomate.gui.omnipose_pretrained_cache import (
                clear_bactomate_builtin_cache_entry,
                ensure_builtin_in_bactomate_cache,
            )
            from bactomate.gui.omnipose_utils import load_cellpose_segmentation_model
            from cellpose_omni.models import MODEL_DIR

            clear_bactomate_builtin_cache_entry(model_type)
            upstream_file = Path(MODEL_DIR) / f"{model_type}torch_0"
            if upstream_file.is_file():
                try:
                    upstream_file.unlink()
                except OSError:
                    pass
            bact_checkpoint = ensure_builtin_in_bactomate_cache(model_type)
            model = load_cellpose_segmentation_model(
                gpu=use_gpu,
                loader_path=bact_checkpoint,
                model_type=model_type,
            )
    except RuntimeError as e:
        if "size mismatch" in str(e) and ("downsample" in str(e) or "output" in str(e)):
            print("\n" + "="*60)
            print("ERROR: Model architecture mismatch detected!")
            print("="*60)
            print("\nThe model checkpoint expects a different architecture than initialized.")
            print("This usually happens when the model cache is corrupted or incompatible.")
            print("\nSOLUTION: Delete the model cache and let it re-download:")
            # Generate correct model filename (keep full model name, add torch_0)
            model_filename = model_type + 'torch_0'
            print(f"  rm ~/.cellpose/models/{model_filename}")
            print("  and delete the copy under BactoMate's omnipose_pretrained folder (see Log for path).")
            print("\nOr run the fix script:")
            print("  ./clear_models.sh")
            print("\nThen try again. The model will be re-downloaded with correct architecture.")
            print("\nAlternatively, try a different model:")
            print("  --model cyto_omni  (instead of cyto2_omni)")
            print("="*60)
            raise RuntimeError(f"Model initialization failed: {e}") from e
        else:
            raise
    
    # NOTE: This function is called by the GUI through WorkerThread. All stdout is captured and
    # displayed in the GUI log, so we use print() for structured, greppable debugging output.
    #
    # The previous implementation pre-loaded *all* frames into memory before batching, which
    # caused large peak RAM usage and increased the likelihood of GPU OOM (CUDA/MPS). We now
    # stream frames per batch and add adaptive batch-size backoff for OOM-like failures.

    def _device_label() -> str:
        if not use_gpu:
            return "cpu(--no-gpu)"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _is_oom_error(exc: BaseException) -> bool:
        s = str(exc).lower()
        # CUDA: typically "CUDA out of memory"; MPS: varies, but commonly includes "mps" + allocation wording.
        return ("out of memory" in s) or ("cuda" in s and "memory" in s) or ("mps" in s and ("memory" in s or "allocated" in s))

    def _clear_accelerator_cache() -> None:
        try:
            import torch
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # MPS has fewer stable cache management hooks; best-effort is simply to retry with smaller batches.

    def _log_cuda_mem(prefix: str) -> None:
        try:
            import torch
            if use_gpu and torch.cuda.is_available():
                alloc = float(torch.cuda.memory_allocated(0)) / (1024.0**2)
                rsvd = float(torch.cuda.memory_reserved(0)) / (1024.0**2)
                print(f"[mem] {prefix} cuda_alloc_mb={alloc:.1f} cuda_reserved_mb={rsvd:.1f}")
        except Exception:
            pass

    kind = "USER_MODEL" if pretrained_model else "BUILTIN"
    model_id = str(pretrained_model) if pretrained_model else str(model_type)
    base_bs = max(1, int(batch_size) if batch_size else 1)
    num_frames_total = len(frame_paths)
    print(f"[run] device={_device_label()} use_gpu={bool(use_gpu)} kind={kind} model={model_id!r}")
    print(f"[run] frames_total={num_frames_total} base_batch_size={base_bs} diameter={diameter} flow_thr={flow_threshold} mask_thr={mask_threshold} cluster={bool(cluster)}")

    masks: list = []
    t0_all = time.time() if "time" in globals() else None
    try:
        import time as _time_mod
        t0_all = _time_mod.time()
    except Exception:
        t0_all = None

    batch_idx = 0
    frame_idx = 0
    while frame_idx < num_frames_total:
        batch_idx += 1
        attempt_bs = min(base_bs, num_frames_total - frame_idx)
        retries = 0

        while True:
            batch_paths = frame_paths[frame_idx: frame_idx + attempt_bs]
            batch_start_1 = frame_idx + 1
            batch_end_1 = frame_idx + len(batch_paths)

            # Load only the current batch into memory.
            t_load0 = None
            try:
                import time as _time_mod
                t_load0 = _time_mod.time()
            except Exception:
                t_load0 = None

            batch_images = []
            first_shape = None
            for p in batch_paths:
                try:
                    img = imread(p)
                    if img.ndim == 3 and img.shape[0] < 10:
                        img = img.transpose(1, 2, 0)  # (Y, X, C) when channels-first
                    if first_shape is None:
                        first_shape = getattr(img, "shape", None)
                    batch_images.append(img)
                except Exception as e:
                    print(f"[load] ERROR path={p!r} err={type(e).__name__}: {e}")
                    if first_shape is None:
                        try:
                            dummy = imread(batch_paths[0])
                            first_shape = dummy.shape
                        except Exception:
                            first_shape = (512, 512)
                    empty = np.zeros(first_shape[:2], dtype=np.uint8)
                    batch_images.append(empty)

            load_s = None
            if t_load0 is not None:
                try:
                    import time as _time_mod
                    load_s = _time_mod.time() - t_load0
                except Exception:
                    load_s = None

            print(f"[batch] idx={batch_idx} frames={batch_start_1}-{batch_end_1}/{num_frames_total} n={len(batch_images)} bs={attempt_bs} retry={retries}" + (f" load_s={load_s:.2f}" if load_s is not None else ""))
            _log_cuda_mem(prefix=f"pre_eval batch={batch_idx}")

            # Run segmentation; on OOM, back off batch size and retry.
            try:
                t_eval0 = None
                try:
                    import time as _time_mod
                    t_eval0 = _time_mod.time()
                except Exception:
                    t_eval0 = None

                result = model.eval(
                    batch_images,
                    channels=channels,
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    mask_threshold=mask_threshold,
                    omni=omni,
                    affinity_seg=affinity_seg,
                    cluster=cluster,
                    net_avg=False,
                    resample=False,
                    tile=False,
                    verbose=verbose,
                )

                eval_s = None
                if t_eval0 is not None:
                    try:
                        import time as _time_mod
                        eval_s = _time_mod.time() - t_eval0
                    except Exception:
                        eval_s = None

                batch_masks = result[0] if isinstance(result, tuple) else result

                out_masks = []
                if isinstance(batch_masks, list):
                    out_masks = batch_masks
                elif isinstance(batch_masks, np.ndarray):
                    if batch_masks.ndim == 2:
                        out_masks = [batch_masks]
                    elif batch_masks.ndim == 3:
                        out_masks = [batch_masks[i] for i in range(min(batch_masks.shape[0], len(batch_images)))]
                    else:
                        out_masks = [batch_masks]
                else:
                    try:
                        out_masks = list(batch_masks)
                    except Exception:
                        out_masks = [batch_masks]

                # Ensure we produce exactly one mask per input frame in the batch.
                if len(out_masks) != len(batch_images):
                    print(f"[warn] mask_count_mismatch batch={batch_idx} got={len(out_masks)} expected={len(batch_images)}; padding/truncating")
                    if len(out_masks) > len(batch_images):
                        out_masks = out_masks[: len(batch_images)]
                    else:
                        # Pad with empty masks.
                        ref = out_masks[0] if out_masks else np.zeros(batch_images[0].shape[:2], dtype=np.uint16)
                        while len(out_masks) < len(batch_images):
                            out_masks.append(np.zeros_like(ref))

                masks.extend(out_masks)

                # Debug log after first batch (same format as Preview dialog for comparison)
                if batch_idx == 1:
                    try:
                        n_cells_first = sum(len(np.unique(m)) - 1 for m in out_masks) if out_masks else 0
                        _append_omni_debug(
                            f"eval kind={kind} model={model_id!r} diameter={diameter} flow_thr={flow_threshold} mask_thr={mask_threshold} "
                            f"batch_frames={len(batch_images)} -> n_cells_first_batch={n_cells_first}"
                        )
                    except Exception:
                        pass

                _log_cuda_mem(prefix=f"post_eval batch={batch_idx}")
                frames_completed = len(masks)
                progress_pct = int((frames_completed / max(1, num_frames_total)) * 100)
                msg = f"[batch] ok idx={batch_idx} done={frames_completed}/{num_frames_total} pct={progress_pct}%"
                if eval_s is not None:
                    msg += f" eval_s={eval_s:.2f}"
                print(msg)

                # Release batch arrays as soon as possible.
                del batch_images
                del out_masks
                _clear_accelerator_cache()

                frame_idx += len(batch_paths)
                break

            except Exception as e:
                # Always log the raw exception for cross-platform debugging (CUDA/MPS wording varies).
                print(f"[batch] ERROR idx={batch_idx} frames={batch_start_1}-{batch_end_1} err={type(e).__name__}: {e}")
                # Full traceback (essential for diagnosing things like `int(None)` deep inside omnipose).
                print("[batch] traceback:")
                traceback.print_exc()
                if _is_oom_error(e) and attempt_bs > 1:
                    new_bs = max(1, attempt_bs // 2)
                    retries += 1
                    print(f"[oom] backoff batch={batch_idx} bs={attempt_bs}->{new_bs} retry={retries}")
                    _clear_accelerator_cache()
                    attempt_bs = new_bs
                    continue

                # Non-OOM (or OOM at bs=1): fall back to per-frame processing for this batch.
                print(f"[batch] falling_back mode=per_frame idx={batch_idx} n={len(batch_images)}")
                for i, image in enumerate(batch_images):
                    try:
                        result = model.eval(
                            image,
                            channels=channels,
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            mask_threshold=mask_threshold,
                            omni=omni,
                            affinity_seg=affinity_seg,
                            cluster=cluster,
                            net_avg=False,
                            resample=False,
                            verbose=verbose,
                        )
                        mask = result[0] if isinstance(result, tuple) else result
                        masks.append(mask)
                    except Exception as e2:
                        print(f"[frame] ERROR idx={frame_idx + i + 1}/{num_frames_total} err={type(e2).__name__}: {e2}")
                        # Full traceback so we can see where (e.g., omnipose.compute_masks) the error originated.
                        print("[frame] traceback:")
                        traceback.print_exc()
                        ref_shape = (masks[0].shape if masks else image.shape[:2])
                        masks.append(np.zeros(ref_shape, dtype=np.uint16))

                del batch_images
                _clear_accelerator_cache()
                frame_idx += len(batch_paths)
                break
    
    # Final memory cleanup
    try:
        import torch
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: show memory stats if verbose
            if verbose:
                allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
                print(f"GPU memory after processing: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
    except:
        pass
    
    return masks


def concatenate_masks(masks, output_path, source_image_path=None):
    """
    Concatenate individual masks into a single multipage TIFF.
    If source_image_path is provided, copies pixel size and time-per-frame metadata from the source image.

    Parameters:
    -----------
    masks : list
        List of mask arrays
    output_path : str
        Path to save concatenated masks
    source_image_path : str or pathlib.Path, optional
        Path to the source image TIFF for metadata transfer
    """
    print(f"\nConcatenating {len(masks)} masks...")
    masks_array = np.stack(masks, axis=0)
    print(f"Mask sequence shape: {masks_array.shape}")

    print(f"Saving to: {output_path}")
    meta = None
    try:
        from bactomate.modules.tiff_metadata import build_metadata_dict_for_mask_from_source
        from pathlib import Path
        src = Path(source_image_path) if source_image_path else None
        meta = build_metadata_dict_for_mask_from_source(src, num_frames=len(masks))
    except Exception:
        pass
    if meta:
        tifffile.imwrite(output_path, masks_array.astype(np.uint16), metadata=meta, imagej=True)
    else:
        tifffile.imwrite(output_path, masks_array.astype(np.uint16))
    print("✓ Masks saved")


def overlay_masks_on_images(images, masks, output_path, alpha=0.5, colormap='gist_ncar'):
    """
    Create overlay of masks on original images.
    
    Parameters:
    -----------
    images : array
        Original image sequence (T, C, Y, X) or (T, Y, X)
    masks : list
        List of mask arrays
    output_path : str
        Path to save overlay
    alpha : float
        Transparency of overlay (0-1)
    colormap : str
        Colormap name for masks
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("WARNING: matplotlib not available, skipping overlay creation")
        return
    
    print(f"\nCreating overlay visualization...")
    
    # Normalize images to 0-255
    # Handle different input dimensions
    if images.ndim == 2:
        # Single 2D image (Y, X) - add time dimension
        images = images[np.newaxis, ...]  # (1, Y, X)
    elif images.ndim == 3:
        # Could be (T, Y, X) or (Y, X, C) or (C, Y, X)
        if images.shape[-1] < 10:  # Likely channels last (Y, X, C)
            images = images.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, Y, X)
            if images.shape[1] == 1:
                images = images[:, 0, :, :]  # (1, Y, X)
            else:
                images = images[:, 0, :, :]  # Take first channel
        elif images.shape[0] < 10:  # Likely channels first (C, Y, X)
            images = images.transpose(1, 2, 0)[np.newaxis, ...]  # (1, Y, X, C)
            if images.shape[-1] == 1:
                images = images[:, :, :, 0]  # (1, Y, X)
            else:
                images = images[:, :, :, 0]  # Take first channel
        # Otherwise assume (T, Y, X) - keep as is
    elif images.ndim == 4:
        # (T, C, Y, X) - take first channel or convert to grayscale
        if images.shape[1] == 1:
            images = images[:, 0, :, :]  # (T, Y, X)
        else:
            # Convert to grayscale
            images = images[:, 0, :, :]  # Take first channel
    else:
        raise ValueError(f"Unsupported image dimension: {images.ndim}")
    
    # Ensure images are contiguous and have correct shape (T, Y, X)
    images = np.ascontiguousarray(images)
    
    # Final check: ensure we have (T, Y, X) shape
    if images.ndim != 3:
        raise ValueError(f"After processing, images should be 3D (T, Y, X), got shape: {images.shape}")
    
    # Normalize images
    images_norm = []
    for img in images:
        # Ensure each image is contiguous
        img = np.ascontiguousarray(img)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = img.astype(np.uint8)
        # Ensure normalized image is contiguous
        img_norm = np.ascontiguousarray(img_norm)
        images_norm.append(img_norm)
    images_norm = np.stack(images_norm)
    
    # Create colored masks
    masks_array = np.stack(masks, axis=0)
    # Ensure masks are contiguous
    masks_array = np.ascontiguousarray(masks_array)
    num_cells = int(masks_array.max())
    
    # Generate colormap
    cmap = plt.get_cmap(colormap)
    # Ensure we have enough colors (add buffer to avoid index errors)
    num_colors = max(num_cells + 1, 1000)
    colors = cmap(np.linspace(0, 1, num_colors))[:num_cells + 1]
    colors[0] = [0, 0, 0, 0]  # Background transparent
    # Ensure colors array is contiguous and properly formatted
    colors = np.ascontiguousarray(colors)
    
    # Create overlay with CellID labels
    overlays = []
    # Ensure we don't go out of bounds
    num_frames = min(len(images_norm), len(masks_array))
    
    # Try to import PIL for text rendering
    try:
        from PIL import Image
        from skimage.measure import regionprops
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        print("WARNING: Pillow missing (required dependency); overlay labels disabled. Run: pip install -r requirements.txt")
    
    for i in tqdm(range(num_frames), desc="Creating overlays", **tqdm_kwargs):
        img = images_norm[i]
        mask = masks_array[i]
        
        # Ensure img is 2D (Y, X)
        if img.ndim != 2:
            print(f"Warning: Image at frame {i} has unexpected shape {img.shape}, expected 2D (Y, X)")
            if img.ndim == 1:
                # Try to reshape if it's 1D (shouldn't happen, but handle it)
                print(f"Error: Image is 1D with shape {img.shape}, cannot create overlay")
                continue
            elif img.ndim == 3:
                # Take first channel if it's (Y, X, C)
                img = img[:, :, 0]
            else:
                print(f"Error: Cannot handle image with {img.ndim} dimensions")
                continue
        
        # Ensure mask is 2D (Y, X)
        if mask.ndim != 2:
            print(f"Warning: Mask at frame {i} has unexpected shape {mask.shape}, expected 2D (Y, X)")
            if mask.ndim == 1:
                print(f"Error: Mask is 1D with shape {mask.shape}, cannot create overlay")
                continue
            elif mask.ndim == 3:
                # Take first channel if it's (Y, X, C)
                mask = mask[:, :, 0]
            else:
                print(f"Error: Cannot handle mask with {mask.ndim} dimensions")
                continue
        
        # Verify shapes match
        if img.shape != mask.shape:
            print(f"Warning: Image shape {img.shape} doesn't match mask shape {mask.shape} at frame {i}")
            # Try to fix by resizing mask to match image
            from skimage.transform import resize
            mask = resize(mask, img.shape, order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)
            print(f"Resized mask to {mask.shape}")
        
        # Ensure mask values are within valid range for colormap indexing
        mask_clipped = np.clip(mask, 0, num_cells).astype(np.int32)
        # Ensure mask is contiguous
        mask_clipped = np.ascontiguousarray(mask_clipped)
        
        # Convert grayscale to RGB - ensure img is 2D first
        if img.ndim != 2:
            print(f"Error: img must be 2D for RGB conversion, got shape {img.shape}")
            continue
        img_rgb = np.stack([img, img, img], axis=-1)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        
        # Colorize mask - use a more robust indexing method
        # Flatten mask, index colors, then reshape
        mask_flat = mask_clipped.flatten()
        colors_flat = colors[mask_flat][:, :3] * 255
        mask_rgb = colors_flat.reshape(mask_clipped.shape[0], mask_clipped.shape[1], 3)
        mask_rgb = np.ascontiguousarray(mask_rgb.astype(np.uint8))
        
        # Verify img_rgb and mask_rgb have matching shapes
        if img_rgb.shape != mask_rgb.shape:
            print(f"Error: img_rgb shape {img_rgb.shape} doesn't match mask_rgb shape {mask_rgb.shape} at frame {i}")
            # Force matching shapes
            min_h = min(img_rgb.shape[0], mask_rgb.shape[0])
            min_w = min(img_rgb.shape[1], mask_rgb.shape[1])
            img_rgb = img_rgb[:min_h, :min_w, :]
            mask_rgb = mask_rgb[:min_h, :min_w, :]
        
        # Blend
        overlay = (img_rgb * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
        
        # Add CellID labels if PIL is available
        if PIL_AVAILABLE and mask.max() > 0:
            try:
                # Ensure overlay is contiguous and in correct format for PIL
                # PIL expects uint8 RGB images
                overlay_contiguous = np.ascontiguousarray(overlay, dtype=np.uint8)
                
                # Convert to PIL Image - explicitly specify mode to avoid issues
                pil_img = Image.fromarray(overlay_contiguous, mode='RGB')
                
                # Get cell centroids and labels
                regions = regionprops(mask)
                labels_to_draw = []
                for region in regions:
                    if region.label == 0:
                        continue
                    
                    # Get centroid
                    centroid_y, centroid_x = region.centroid
                    x, y = int(centroid_x), int(centroid_y)
                    
                    cell_id_str = str(region.label)
                    text_x, text_y = x + 2, y - 5
                    labels_to_draw.append((cell_id_str, text_x, text_y))

                pil_img = draw_hq_labels(
                    pil_img,
                    labels_to_draw,
                    font_size=12,
                    scale=2,
                    stroke_width=2,
                )
                
                # Convert back to numpy array - ensure it's contiguous and correct shape
                overlay = np.ascontiguousarray(np.array(pil_img), dtype=np.uint8)
                
                # Verify shape matches original
                if overlay.shape != img_rgb.shape:
                    print(f"Warning: Overlay shape mismatch after PIL conversion: {overlay.shape} vs {img_rgb.shape}")
                    # Try to fix by ensuring correct shape
                    if overlay.shape[0] == img_rgb.shape[1] and overlay.shape[1] == img_rgb.shape[0]:
                        # PIL might have transposed - transpose back
                        overlay = overlay.transpose(1, 0, 2)
            except Exception as e:
                # If text rendering fails, continue without labels
                import traceback
                print(f"Warning: Could not add labels to overlay: {e}")
                traceback.print_exc()
        
        # Ensure overlay is contiguous before appending
        overlay = np.ascontiguousarray(overlay, dtype=np.uint8)
        overlays.append(overlay)
    
    overlay_array = np.stack(overlays, axis=0)
    # Ensure final array is contiguous
    overlay_array = np.ascontiguousarray(overlay_array, dtype=np.uint8)
    
    print(f"Saving overlay to: {output_path}")
    # Save with compression to avoid any strip-related issues
    tifffile.imwrite(output_path, overlay_array, compression='lzw', photometric='rgb')
    print("✓ Overlay saved")


# Cellpose-SAM (official cellpose + cpsam): inference-only path when --model cellpose_sam
CELLPOSE_SAM_MODEL = "cellpose_sam"
# cellprob_threshold (logit scale, ~ -6..+6 per Cellpose docs); match GUI clamp in omnipose_seg.py
_CELLPOSE_SAM_CELLPROB_MIN = -6.0
_CELLPOSE_SAM_CELLPROB_MAX = 6.0


def _to_three_channel(image):
    """Convert image to 3-channel (H, W, 3) for Cellpose-SAM. Grayscale is duplicated."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return np.concatenate([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 2:
            return np.concatenate([arr, arr[..., 0:1]], axis=-1)
        if arr.shape[-1] == 3:
            return arr
    raise ValueError(f"Cellpose-SAM expects 2D or 3-channel image; got shape {getattr(image, 'shape', '?')}")


def process_frames_with_cellpose_sam(
    frame_paths,
    diameter=30.0,
    flow_threshold=0.4,
    mask_threshold=0.0,
    use_gpu=True,
    batch_size=8,
):
    """
    Run Cellpose-SAM (official cellpose, pretrained_model='cpsam') on a list of frame paths.
    Returns list of mask arrays (same length as frame_paths).
    """
    cp = float(mask_threshold)
    cellprob_t = max(_CELLPOSE_SAM_CELLPROB_MIN, min(_CELLPOSE_SAM_CELLPROB_MAX, cp))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        from cellpose import models
    except ImportError as e:
        raise ImportError(
            "Cellpose-SAM requires the official 'cellpose' package. "
            "Install with: pip install cellpose"
        ) from e

    print("\nInitializing Cellpose-SAM model (cpsam)...")
    model = models.CellposeModel(pretrained_model="cpsam", gpu=use_gpu)

    masks = []
    num_frames = len(frame_paths)
    batch_size = max(1, min(batch_size, num_frames))
    num_batches = (num_frames + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_frames)
        batch_paths = frame_paths[start:end]
        batch_images = []
        for p in batch_paths:
            img = safe_imread(p)
            if img.ndim == 3 and img.shape[0] < 10:
                img = img[0]
            batch_images.append(_to_three_channel(img))

        batch_num = batch_idx + 1
        print(f"Processing batch {batch_num}/{num_batches}: frames {start + 1}-{end} of {num_frames}")
        result = model.eval(
            batch_images,
            diameter=diameter if diameter > 0 else None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_t,
            batch_size=len(batch_images),
        )
        if isinstance(result, tuple):
            batch_masks = result[0]
        else:
            batch_masks = result
        if isinstance(batch_masks, np.ndarray) and batch_masks.ndim == 3:
            for i in range(batch_masks.shape[0]):
                masks.append(batch_masks[i].astype(np.uint16))
        else:
            for m in batch_masks:
                masks.append(np.asarray(m, dtype=np.uint16))

        progress_pct = int((len(masks) / num_frames) * 100)
        print(f"✓ Batch {batch_num}/{num_batches} complete: {len(masks)}/{num_frames} frames ({progress_pct}%)")

    return masks


def _run_cellpose_sam_pipeline(args, output_dir):
    """Run full pipeline for Cellpose-SAM: split TIFF, segment, concatenate masks, optional overlay."""
    frame_paths, temp_dir, _ = split_multipage_tiff(
        args.input,
        output_dir=args.save_frames if getattr(args, "save_frames", None) else None,
        prefix="frame",
    )
    os.makedirs(output_dir, exist_ok=True)
    masks_output = os.path.join(output_dir, f"{Path(args.input).stem}_masks.tif")

    try:
        diameter = float(args.diameter) if getattr(args, "diameter", 0) else 30.0
        if diameter <= 0:
            diameter = 30.0
        masks = process_frames_with_cellpose_sam(
            frame_paths,
            diameter=diameter,
            flow_threshold=float(getattr(args, "flow_threshold", 0.4)),
            mask_threshold=float(getattr(args, "mask_threshold", 0.0)),
            use_gpu=not getattr(args, "no_gpu", False),
            batch_size=int(getattr(args, "batch_size", 32)),
        )
        concatenate_masks(masks, masks_output, source_image_path=args.input)

        if not getattr(args, "no_overlay", False):
            overlay_dir = os.path.join(output_dir, "overlays")
            os.makedirs(overlay_dir, exist_ok=True)
            images = safe_imread(args.input)
            if images.ndim == 3:
                if images.shape[-1] < 10:
                    images = images.transpose(2, 0, 1)[np.newaxis, ...]
                else:
                    images = images[:, np.newaxis, :, :]
            elif images.ndim == 4:
                if images.shape[-1] < 10:
                    images = images.transpose(0, 3, 1, 2)
                elif images.shape[0] < 10:
                    images = images.transpose(1, 0, 2, 3)
            overlay_output = os.path.join(overlay_dir, f"{Path(args.input).stem}_overlay.tif")
            overlay_masks_on_images(images, masks, overlay_output)
            print(f"  Overlay saved to: {overlay_output}")

        print(f"\n{'='*60}")
        print("✓ Cellpose-SAM processing complete!")
        print(f"  Masks saved to: {masks_output}")
        print(f"{'='*60}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            print("\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multipage TIFF sequences with Omnipose',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-diameter
  python batch_process_tiff_sequence.py input.tif
  
  # Specify diameter and model
  python batch_process_tiff_sequence.py input.tif --diameter 30 --model cyto2_omni
  
  # Process with specific channels
  python batch_process_tiff_sequence.py input.tif --channels 0 0 --diameter 25
  
  # Save intermediate frames (for debugging)
  python batch_process_tiff_sequence.py input.tif --save-frames frames_dir/
        """
    )
    
    parser.add_argument('input', type=str, help='Input multipage TIFF file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory (default: same as input)')
    parser.add_argument('--model', type=str, default='bact_phase_omni',
                       help='Omnipose model (e.g. bact_phase_omni), path to .pth, or cellpose_sam for Cellpose-SAM')
    parser.add_argument('--diameter', type=float, default=0,
                       help='Cell diameter in pixels (0 for auto-detection, default: 0)')
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 0],
                       metavar=('CHAN1', 'CHAN2'),
                       help='Channels for segmentation: [main_channel, nuclear_channel] (default: 0 0)')
    parser.add_argument('--flow-threshold', type=float, default=0.4,
                       help='Flow error threshold (default: 0.4)')
    parser.add_argument('--mask-threshold', type=float, default=0.0,
                       help='Mask threshold (default: 0.0)')
    parser.add_argument('--no-omni', action='store_true',
                       help='Disable Omnipose mask reconstruction')
    parser.add_argument('--cluster', action='store_true',
                       help='Use DBSCAN clustering (helps with thin cells)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU (use CPU)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing frames (default: 32, increase if GPU memory allows)')
    parser.add_argument('--save-frames', type=str, default=None,
                       help='Save individual frames to this directory (for debugging)')
    parser.add_argument('--no-overlay', action='store_true',
                       help='Skip creating overlay visualization')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Setup output directory
    if args.output is None:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        input_basename = Path(args.input).stem
        output_dir = os.path.join(input_dir, f"{input_basename}_omnipose_segmentation")
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Cellpose-SAM path: use official cellpose + cpsam (separate from Omnipose)
    if args.model == CELLPOSE_SAM_MODEL:
        _run_cellpose_sam_pipeline(args, output_dir)
        return

    # Step 1: Split multipage TIFF
    frame_dir = args.save_frames if args.save_frames else None
    frame_paths, temp_dir, image_shape = split_multipage_tiff(
        args.input, 
        output_dir=frame_dir,
        prefix="frame"
    )
    
    try:
        # Step 2: Process frames with Omnipose (--model can be type name or path to user .pth)
        pretrained = args.model if os.path.isfile(args.model) else None
        model_type = None if pretrained else args.model
        diameter = float(args.diameter)
        flow_threshold = float(args.flow_threshold)
        masks = process_frames_with_omnipose(
            frame_paths,
            model_type=model_type,
            pretrained_model=pretrained,
            diameter=diameter,
            channels=args.channels,
            use_gpu=not args.no_gpu,
            batch_size=args.batch_size,
            flow_threshold=flow_threshold,
            mask_threshold=args.mask_threshold,
            omni=not args.no_omni,
            cluster=args.cluster,
            verbose=args.verbose
        )
        
        # Step 3: Concatenate masks
        masks_output = os.path.join(output_dir, f"{Path(args.input).stem}_masks.tif")
        concatenate_masks(masks, masks_output, source_image_path=args.input)
        
        # Step 4: Create overlay (optional) in subfolder so segmentation folder has only masks for Analyze Masks
        if not args.no_overlay:
            overlay_dir = os.path.join(output_dir, "overlays")
            os.makedirs(overlay_dir, exist_ok=True)
            # Reload original images for overlay
            images = safe_imread(args.input)
            # Handle different formats
            if images.ndim == 3:
                if images.shape[-1] < 10:
                    images = images.transpose(2, 0, 1)[np.newaxis, ...]
                else:
                    images = images[:, np.newaxis, :, :]
            elif images.ndim == 4:
                if images.shape[-1] < 10:
                    images = images.transpose(0, 3, 1, 2)
                elif images.shape[0] < 10:
                    images = images.transpose(1, 0, 2, 3)
            
            overlay_output = os.path.join(overlay_dir, f"{Path(args.input).stem}_overlay.tif")
            overlay_masks_on_images(images, masks, overlay_output)
        
        print(f"\n{'='*60}")
        print("✓ Processing complete!")
        print(f"  Masks saved to: {masks_output}")
        if not args.no_overlay:
            print(f"  Overlay saved to: {overlay_output}")
        print(f"{'='*60}")
        
    finally:
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
