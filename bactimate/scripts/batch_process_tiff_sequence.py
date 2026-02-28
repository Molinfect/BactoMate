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
import warnings
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil
from datetime import datetime


def _append_omni_debug(msg: str) -> None:
    """Append a timestamped line to BactiMate omnipose_debug.log if env is set (for Run button / batch path)."""
    path = os.environ.get("BACTIMATE_OMNI_DEBUG_LOG")
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
                    "Reinstall BactiMate dependencies: pip install -r requirements.txt"
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
    if pretrained_model:
        print(f"\nInitializing Omnipose model from user checkpoint: {pretrained_model}")
    else:
        print(f"\nInitializing Omnipose model: {model_type}")

    from urllib.error import HTTPError

    if pretrained_model:
        if not os.path.isfile(pretrained_model):
            raise FileNotFoundError(f"User model not found: {pretrained_model}")
        pretrained_model = str(Path(pretrained_model).resolve())
        _ckpt_path_for_inspect = pretrained_model
        # cellpose_omni sets omni=True only when "omni" is in the checkpoint filename (stem).
        # Our saved models are e.g. mf_test.pth -> stem "mf_test" -> omni=False -> wrong dynamics.
        # Use a path whose filename contains "omni" so the loader uses Omnipose dynamics.
        stem = Path(pretrained_model).stem
        if "omni" not in stem.lower():
            import tempfile
            _omni_link_dir = tempfile.mkdtemp(prefix="bactimate_omni_")
            _omni_link_path = os.path.join(_omni_link_dir, f"{stem}_omni.pth")
            try:
                os.symlink(pretrained_model, _omni_link_path)
            except OSError:
                # e.g. Windows without symlink privilege; copy instead (slower but works)
                import shutil
                shutil.copy2(pretrained_model, _omni_link_path)
            pretrained_model = _omni_link_path
            print("Note: Using Omnipose loader (filename contains 'omni') so custom model runs with correct dynamics.")
        # Inspect checkpoint so we can log architecture (user vs built-in must match training)
        try:
            from bactimate.gui.omnipose_utils import inspect_omnipose_checkpoint, get_default_user_model_architecture
            ckpt_info = inspect_omnipose_checkpoint(_ckpt_path_for_inspect)
            if "error" not in ckpt_info:
                print(f"Checkpoint keys (first 15): {ckpt_info.get('keys', [])[:15]}")
                if ckpt_info.get("nchan") is not None or ckpt_info.get("first_layer_shape"):
                    print(f"Checkpoint nchan: {ckpt_info.get('nchan')} (from file or inferred from first conv shape {ckpt_info.get('first_layer_shape')})")
                if ckpt_info.get("nclasses") is not None:
                    print(f"Checkpoint nclasses: {ckpt_info.get('nclasses')}")
            def_nchan, def_nclasses = get_default_user_model_architecture()
            print(f"BactiMate inference uses nchan={def_nchan}, nclasses={def_nclasses} (must match training).")
        except Exception as _e:
            pass  # non-fatal
        model = models.Cellpose(gpu=use_gpu, pretrained_model=pretrained_model, omni=omni)
        # Log what the loaded model actually has (cellpose_omni may read from filename or use defaults)
        try:
            if hasattr(model, 'cp') and hasattr(model.cp, 'nchan') and hasattr(model.cp, 'nclasses'):
                print(f"Loaded model architecture: nchan={model.cp.nchan}, nclasses={model.cp.nclasses}")
                if (getattr(model.cp, 'nchan', None) != 2 or getattr(model.cp, 'nclasses', None) not in (3, 4)):
                    print("WARNING: Model nchan/nclasses differ from BactiMate training defaults (2, 3). If you trained with BactiMate, this may cause wrong or no detections.")
                _append_omni_debug(
                    f"USER_MODEL post-load path={pretrained_model!r} model.cp.nchan={model.cp.nchan} model.cp.nclasses={model.cp.nclasses}"
                )
        except Exception:
            pass
    else:
        # Workaround for cellpose_omni bug: Cellpose doesn't pass model_type to CellposeModel
        from cellpose_omni.models import model_path
        try:
            model_file = model_path(model_type, 0, True)
            if os.path.exists(model_file):
                print(f"Removing existing model file to ensure correct architecture...")
                os.remove(model_file)
        except HTTPError as e:
            if e.code == 404:
                print(f"\n⚠️  WARNING: Model '{model_type}' not found in GitHub repository (404 error).")
                print(f"   This model may not be available in the Omnipose GitHub repository.")
                print(f"   Trying to use cached model if available, or falling back to Cellpose server...")
                # Check if model exists in cache (Path is imported at top of module)
                from cellpose_omni.models import MODEL_DIR
                torch_str = 'torch'
                basename = f'{model_type}{torch_str}_0'
                cache_model_file = Path(MODEL_DIR) / basename
                if cache_model_file.exists():
                    print(f"   Using cached model: {cache_model_file}")
                    model_file = str(cache_model_file)
                else:
                    # Try to get model from original Cellpose server as fallback
                    print(f"   Attempting to download from original Cellpose server...")
                    # Temporarily restore original URL
                    import cellpose_omni.models as models_module
                    original_url = getattr(models_module, '_MODEL_URL', None)
                    if original_url and 'github' in original_url:
                        # Try original Cellpose server
                        models_module._MODEL_URL = 'https://www.cellpose.org/models'
                        try:
                            model_file = model_path(model_type, 0, True)
                            print(f"   ✓ Downloaded from Cellpose server")
                        except Exception:
                            raise RuntimeError(
                                f"Model '{model_type}' is not available.\n\n"
                                f"Available Omnipose models (see https://omnipose.readthedocs.io/models.html):\n"
                                f"  - bact_phase_omni (recommended for bacteria)\n"
                                f"  - bact_fluor_omni\n"
                                f"  - cyto2_omni\n"
                                f"  - worm_omni\n"
                                f"  - plant_omni\n"
                                f"  - bact_phase_affinity\n\n"
                                f"Please select a different model or check if '{model_type}' exists in the repository."
                            )
                        finally:
                            # Restore GitHub URL
                            if original_url:
                                models_module._MODEL_URL = original_url
                    else:
                        raise RuntimeError(
                            f"Model '{model_type}' is not available in the repository.\n\n"
                            f"Available Omnipose models (see https://omnipose.readthedocs.io/models.html):\n"
                            f"  - bact_phase_omni (recommended for bacteria)\n"
                            f"  - bact_fluor_omni\n"
                            f"  - cyto2_omni\n"
                            f"  - worm_omni\n"
                            f"  - plant_omni\n"
                            f"  - bact_phase_affinity"
                        )
            else:
                raise

        model = models.Cellpose(gpu=use_gpu, model_type=model_type, omni=omni)
        try:
            if hasattr(model, 'cp'):
                nchan = getattr(model.cp, 'nchan', None)
                nclasses = getattr(model.cp, 'nclasses', None)
                _append_omni_debug(
                    f"BUILTIN post-load model_type={model_type!r} model.cp.nchan={nchan} model.cp.nclasses={nclasses}"
                )
        except Exception:
            pass

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
        if not pretrained_model and hasattr(model.cp, 'nchan') and hasattr(model.cp, 'nclasses'):
            expected_nchan = 2 if model_type in ['bact_phase_omni', 'cyto2_omni', 'bact_fluor_omni'] else 1
            expected_nclasses = 4 if model_type in ['bact_phase_omni', 'cyto2_omni', 'bact_fluor_omni'] else 3
            if model.cp.nchan != expected_nchan or model.cp.nclasses != expected_nclasses:
                print(f"WARNING: Architecture mismatch detected!")
                print(f"  Expected: nchan={expected_nchan}, nclasses={expected_nclasses}")
                print(f"  Got: nchan={model.cp.nchan}, nclasses={model.cp.nclasses}")
                print(f"  This may cause errors. Trying to fix...")
                # Try to fix by recreating with correct parameters
                raise RuntimeError("Architecture mismatch - need to recreate model")
    except RuntimeError as e:
        if "Architecture mismatch" in str(e) and not pretrained_model:
            # Recreate model by deleting and trying again (built-in only)
            from cellpose_omni.models import model_path
            model_file = model_path(model_type, 0, True)
            if os.path.exists(model_file):
                os.remove(model_file)
            model = models.Cellpose(gpu=use_gpu, model_type=model_type, omni=omni)
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
            print("\nOr run the fix script:")
            print("  ./clear_models.sh")
            print("\nThen try again. The model will be re-downloaded with correct architecture.")
            print("\nAlternatively, try a different model:")
            print("  --model cyto_omni  (instead of cyto2_omni)")
            print("="*60)
            raise RuntimeError(f"Model initialization failed: {e}") from e
        else:
            raise
    
    masks = []
    num_frames_total = len(frame_paths)
    num_batches = (num_frames_total + batch_size - 1) // batch_size  # Ceiling division
    print(f"Processing {num_frames_total} frames in {num_batches} batch(es) of {batch_size}...")
    
    # Pre-load all frames into memory for faster batch processing
    print(f"Loading {num_frames_total} frames into memory...")
    images_list = []
    for i, frame_path in enumerate(tqdm(frame_paths, desc="Loading frames", **tqdm_kwargs), 1):
        try:
            image = imread(frame_path)
            # Ensure image is properly formatted
            if image.ndim == 2:
                # 2D grayscale - keep as is, Omnipose will handle it
                pass
            elif image.ndim == 3:
                # Could be (Y, X, C) or (C, Y, X)
                if image.shape[0] < 10:  # Channels first
                    image = image.transpose(1, 2, 0)  # (Y, X, C)
            images_list.append(image)
        except Exception as e:
            print(f"\nERROR loading {frame_path}: {e}")
            # Use first image shape if available, otherwise create dummy
            if len(images_list) > 0:
                empty_image = np.zeros_like(images_list[0])
            else:
                # Try to read to get shape
                try:
                    dummy = imread(frame_paths[0])
                    empty_image = np.zeros(dummy.shape[:2], dtype=np.uint8)
                except:
                    empty_image = np.zeros((512, 512), dtype=np.uint8)
            images_list.append(empty_image)
    
    # Process frames in batches for better GPU utilization
    num_frames = len(images_list)
    batch_num = 0
    for batch_start in range(0, num_frames, batch_size):
        batch_num += 1
        batch_end = min(batch_start + batch_size, num_frames)
        batch_images = images_list[batch_start:batch_end]
        frames_in_batch = len(batch_images)
        
        # Clear progress message showing actual frame numbers
        print(f"Processing batch {batch_num}/{num_batches}: frames {batch_start+1}-{batch_end} of {num_frames_total} ({frames_in_batch} frames)")
        
        try:
            # Process batch - model.eval can handle lists of images
            # This is much faster than processing one at a time
            result = model.eval(
                batch_images,  # Pass list of images for batch processing
                channels=channels,
                diameter=diameter,
                flow_threshold=flow_threshold,
                mask_threshold=mask_threshold,
                omni=omni,
                cluster=cluster,
                net_avg=False,  # Faster, omni models don't support net_avg anyway
                resample=False,  # Disable resampling for speed (was True)
                tile=False,  # Disable tiling for speed (if GPU memory allows)
                verbose=verbose
            )
            
            # Handle different return formats
            if isinstance(result, tuple):
                batch_masks = result[0]  # First element is always the masks
            else:
                batch_masks = result
            
            # batch_masks might be a list or array
            if isinstance(batch_masks, list):
                masks.extend(batch_masks)
            elif isinstance(batch_masks, np.ndarray):
                # If single array, might be stacked - check dimensions
                if batch_masks.ndim == 2:
                    # Single mask
                    masks.append(batch_masks)
                else:
                    # Multiple masks - split them
                    for i in range(len(batch_images)):
                        if batch_masks.ndim == 3:
                            masks.append(batch_masks[i])
                        else:
                            masks.append(batch_masks)
            else:
                # Fallback: try to iterate
                try:
                    masks.extend(batch_masks)
                except:
                    masks.append(batch_masks)
            
            # Debug log after first batch (same format as Preview dialog for comparison)
            if batch_num == 1:
                try:
                    first_masks = batch_masks if isinstance(batch_masks, list) else [batch_masks[i] for i in range(len(batch_images))] if batch_masks.ndim >= 3 else [batch_masks]
                    n_cells_first = sum(len(np.unique(m)) - 1 for m in first_masks) if first_masks else 0
                    kind = "USER_MODEL" if pretrained_model else "BUILTIN"
                    model_id = str(pretrained_model) if pretrained_model else model_type
                    _append_omni_debug(
                        f"eval kind={kind} model={model_id!r} diameter={diameter} flow_thr={flow_threshold} mask_thr={mask_threshold} "
                        f"batch_frames={len(batch_images)} -> n_cells_first_batch={n_cells_first}"
                    )
                except Exception:
                    pass
            
            # Progress update after batch completion
            frames_completed = len(masks)
            progress_pct = int((frames_completed / num_frames_total) * 100)
            print(f"✓ Batch {batch_num}/{num_batches} complete: {frames_completed}/{num_frames_total} frames ({progress_pct}%)")
            
            # Memory management: clear GPU cache between batches to prevent fragmentation
            # This allows memory pooling to work better and can enable larger batch sizes
            try:
                import torch
                if use_gpu and torch.cuda.is_available():
                    # Clear unused memory but keep allocated memory pools
                    # This helps with memory fragmentation without losing the benefits of pooling
                    torch.cuda.empty_cache()
            except:
                pass
            
        except Exception as e:
            print(f"\nERROR processing batch {batch_start}-{batch_end}: {e}")
            # Fallback to individual processing for this batch
            print(f"Falling back to individual frame processing for this batch...")
            for i, image in enumerate(batch_images):
                try:
                    result = model.eval(
                        image,
                        channels=channels,
                        diameter=diameter,
                        flow_threshold=flow_threshold,
                        mask_threshold=mask_threshold,
                        omni=omni,
                        cluster=cluster,
                        net_avg=False,
                        resample=False,
                        verbose=verbose
                    )
                    if isinstance(result, tuple):
                        mask = result[0]
                    else:
                        mask = result
                    masks.append(mask)
                except Exception as e2:
                    print(f"ERROR processing frame {batch_start + i}: {e2}")
                    # Create empty mask
                    if len(masks) > 0:
                        empty_mask = np.zeros_like(masks[0])
                    else:
                        empty_mask = np.zeros(image.shape[:2], dtype=np.uint16)
                    masks.append(empty_mask)
    
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
        from bactimate.tiff_metadata import build_metadata_dict_for_mask_from_source
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
        from PIL import Image, ImageDraw, ImageFont
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
                draw = ImageDraw.Draw(pil_img)
                
                # Try to use a nice font, fallback to default if not available
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 10)
                    except:
                        font = ImageFont.load_default()
                
                # Get cell centroids and labels
                regions = regionprops(mask)
                for region in regions:
                    if region.label == 0:
                        continue
                    
                    # Get centroid
                    centroid_y, centroid_x = region.centroid
                    x, y = int(centroid_x), int(centroid_y)
                    
                    # Draw CellID text with shadow for readability
                    cell_id_str = str(region.label)
                    text_x, text_y = x + 2, y - 5
                    
                    # Draw text shadow (black)
                    draw.text((text_x + 1, text_y + 1), cell_id_str, fill=(0, 0, 0), font=font)
                    # Draw text (white)
                    draw.text((text_x, text_y), cell_id_str, fill=(255, 255, 255), font=font)
                
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
                       help='Omnipose model type (e.g. bact_phase_omni) or path to a user-trained .pth model')
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
