"""
Utilities to convert binary/semantic masks to Omnipose-compatible instance label matrices.
Used by the Train Omnipose dialog ("Make masks Omnipose fittable").
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Tuple

from bactimate.gui.utils import safe_imread

# Match dialog pair discovery (processing._training_pairs): exts and suffixes
_PAIR_EXTS = (".tif", ".tiff", ".png", ".npy", ".npz")
_MASK_STEM_SUFFIXES = ("_cp_masks", "_cp_output", "_flows")

import numpy as np
import tifffile
from scipy.ndimage import label as scipy_label

OMNIPOSE_MASKS_SUBDIR = "omnipose_masks"


def convert_single_mask_to_omnipose(
    mask_path: Path,
    out_path: Path | None = None,
) -> Tuple[Literal["converted", "skipped"], str] | Tuple[Literal["error"], str]:
    """
    Convert a single mask file to Omnipose format if it is binary; otherwise leave as-is.
    Writes to out_path if given; otherwise overwrites mask_path (caller should avoid that).

    Omnipose needs instance labels: background=0, each cell=unique integer 1,2,3,...
    Binary masks (e.g. 0 and 255 from Ilastik) are converted via connected-components labeling.

    Returns:
        ("converted", "") if the file was converted and written.
        ("skipped", "") if the mask was already instance labels (no change).
        ("error", message) on failure.
    """
    try:
        data = tifffile.imread(str(mask_path))
    except Exception as e:
        return ("error", f"Could not read {mask_path.name}: {e}")

    # Reduce to 2D (RGB/multi-channel: any pixel > 0 -> foreground)
    if data.ndim == 3:
        if data.shape[-1] in (3, 4):
            gray = np.max(data, axis=-1)
        else:
            gray = np.max(data, axis=0)
    elif data.ndim == 2:
        gray = data
    else:
        return ("error", f"Unsupported mask shape {data.shape}")

    binary = (gray > 0).astype(np.uint8)
    unique = np.unique(gray)

    # Already instance labels (0, 1, 2, ...)? Skip (still write a copy to out_path if given).
    if unique.size > 2 and 0 in unique:
        if out_path is not None:
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path, out_path)
            except Exception as e:
                return ("error", f"Could not write {out_path.name}: {e}")
        return ("skipped", "")

    # Binary (0 and one other value) or single value: run connected components and write
    labels, _ = scipy_label(binary)
    out = labels.astype(np.uint32)

    dest = out_path if out_path is not None else mask_path
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(dest), out)
    except Exception as e:
        return ("error", f"Could not write {dest.name}: {e}")

    return ("converted", "")


def convert_masks_in_folder(
    folder: Path,
    mask_filter: str,
    img_filter: str = "",
) -> Tuple[int, int, list[str], Path]:
    """
    Find all mask files in folder matching *<mask_filter>.tif / .tiff.
    Write converted (or copied) masks to folder/omnipose_masks/. Original files
    in the training folder are never overwritten.
    Also copies corresponding image files into omnipose_masks so that folder can be
    used as the Training directory.

    Returns:
        (converted_count, skipped_count, errors, out_dir).
    """
    if not folder.is_dir():
        return (0, 0, [f"Not a directory: {folder}"], folder / OMNIPOSE_MASKS_SUBDIR)

    mask_files: list[Path] = []
    for ext in (".tif", ".tiff"):
        mask_files.extend(folder.glob(f"*{mask_filter}{ext}"))

    mask_files = sorted(set(mask_files))
    out_dir = folder / OMNIPOSE_MASKS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    errors: list[str] = []

    for path in mask_files:
        out_path = out_dir / path.name
        status, msg = convert_single_mask_to_omnipose(path, out_path=out_path)
        if status == "converted":
            converted += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors.append(msg)

    # Copy image files into out_dir so user can set Training directory to out_dir
    for path in mask_files:
        base = path.stem
        if not base.endswith(mask_filter):
            continue
        base = base[: -len(mask_filter)] if mask_filter else base
        for ext in (".tif", ".tiff"):
            img_name = f"{base}{img_filter}{ext}"
            src = folder / img_name
            if src.is_file() and src != path:
                try:
                    shutil.copy2(src, out_dir / img_name)
                except Exception:
                    pass
                break

    return (converted, skipped, errors, out_dir)


def get_training_pairs(
    folder: Path,
    img_filter: str,
    mask_filter: str,
) -> list[tuple[Path, Path]]:
    """
    Discover (image, mask) pairs in folder using the same logic as the dialog's
    _training_pairs: image candidates (non-mask files), base = stem minus img_filter,
    mask = base + mask_filter + ext. Used so Start training can populate omnipose_masks
    from the same set of pairs we count for validation (avoids glob mismatch).
    """
    if not folder.is_dir():
        return []
    suffixes_to_skip = (*_MASK_STEM_SUFFIXES, mask_filter)
    candidates = [
        f
        for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in (e.lower() for e in _PAIR_EXTS)
        and not any(
            f.stem.endswith(s) or (f.stem + s == f.stem) for s in suffixes_to_skip
        )
    ]
    pairs: list[tuple[Path, Path]] = []
    for f in candidates:
        base = f.stem.replace(img_filter, "") if img_filter else f.stem
        if not base or base == ".":
            continue
        for ext in _PAIR_EXTS:
            mask_path = folder / f"{base}{mask_filter}{ext}"
            if mask_path.exists():
                pairs.append((f, mask_path))
                break
    return pairs


# Common (image_suffix, mask_suffix) combinations to try when auto-detecting filters
_INFER_CANDIDATES: list[tuple[str, str]] = [
    ("_phase", "_masks"),
    ("", "_masks"),
    ("_img", "_masks"),
    ("_image", "_masks"),
    ("_phase", "_mask"),
    ("", "_mask"),
    ("_img", "_mask"),
    ("", "_segmentation"),
    ("_img", "_segmentation"),
    ("", "_seg"),
    ("_labels", "_masks"),
]


def infer_training_filters(folder: Path) -> tuple[str | None, str | None]:
    """
    Infer (img_filter, mask_filter) from filenames in folder by trying common
    suffix combinations and returning the one that yields the most image/mask pairs.
    Returns (None, None) if folder is empty or no pairs found for any combination.
    """
    if not folder.is_dir():
        return (None, None)
    best_count = 0
    best: tuple[str | None, str | None] = (None, None)
    for img_filter, mask_filter in _INFER_CANDIDATES:
        pairs = get_training_pairs(folder, img_filter, mask_filter)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best = (img_filter, mask_filter)
    return best if best_count > 0 else (None, None)


def prepare_omnipose_masks_from_pairs(
    parent_dir: Path,
    img_filter: str,
    mask_filter: str,
) -> tuple[Path, int]:
    """
    Populate parent_dir/omnipose_masks from discovered (image, mask) pairs in parent_dir.
    Uses the same pair logic as get_training_pairs (and the dialog's _training_pairs),
    so training always uses converted masks when pairs exist, even if the mask_filter
    glob would find no files (e.g. default _mask vs files named *_masks.tif).
    Returns (train_dir, n_pairs). train_dir is omnipose_masks if n_pairs > 0, else parent_dir.
    """
    pairs = get_training_pairs(parent_dir, img_filter, mask_filter)
    if not pairs:
        return (parent_dir, 0)
    out_dir = parent_dir / OMNIPOSE_MASKS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path, mask_path in pairs:
        out_mask = out_dir / mask_path.name
        convert_single_mask_to_omnipose(mask_path, out_path=out_mask)
        # Copy image with same name as in source so Omnipose finds it
        dst_img = out_dir / img_path.name
        if not dst_img.is_file() or dst_img.stat().st_mtime < img_path.stat().st_mtime:
            try:
                shutil.copy2(img_path, dst_img)
            except Exception:
                pass
    return (out_dir, len(pairs))


# Extensions we treat as images when syncing into omnipose_masks (try multiple for robustness)
_IMAGE_EXTS = (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG")


def ensure_omnipose_masks_has_images(
    parent_dir: Path,
    mask_filter: str,
    img_filter: str = "",
) -> Path:
    """
    If parent_dir/omnipose_masks exists and contains mask files, copy missing
    image files from parent_dir so the subfolder has both images and masks.
    Returns the directory to use for training (omnipose_masks if it had masks, else parent_dir).
    Tries multiple extensions and case variants so parent images are found.
    """
    out_dir = parent_dir / OMNIPOSE_MASKS_SUBDIR
    if not out_dir.is_dir():
        return parent_dir
    mask_files: list[Path] = []
    for ext in (".tif", ".tiff"):
        mask_files.extend(out_dir.glob(f"*{mask_filter}{ext}"))
    mask_files = sorted(set(mask_files))
    if not mask_files:
        return parent_dir
    for path in mask_files:
        base = path.stem
        if not base.endswith(mask_filter):
            continue
        base = base[: -len(mask_filter)] if mask_filter else base
        # Target image base name (what we'll name the copy)
        image_base = f"{base}{img_filter}"
        dst_any = out_dir / f"{image_base}.tif"
        if dst_any.is_file():
            continue
        # Try exact name first, then multiple extensions and case variants
        copied = False
        for ext in _IMAGE_EXTS:
            img_name = f"{image_base}{ext}"
            dst = out_dir / img_name
            if dst.is_file():
                copied = True
                break
            src = parent_dir / img_name
            if src.is_file() and src != path:
                try:
                    shutil.copy2(src, dst)
                    copied = True
                    break
                except Exception:
                    pass
        if copied:
            continue
        # Fallback: find any file in parent whose stem matches (case-insensitive) image_base
        try:
            low = image_base.lower()
            for f in parent_dir.iterdir():
                if not f.is_file() or f == path:
                    continue
                if f.stem.lower() == low and f.suffix.lower() in (e.lower() for e in _IMAGE_EXTS):
                    dest = out_dir / f"{image_base}{f.suffix}"
                    if not dest.is_file():
                        shutil.copy2(f, dest)
                    break
        except Exception:
            pass
    return out_dir


def ensure_training_images_nchan(
    train_path: Path,
    img_filter: str,
    mask_filter: str,
    nchan: int,
) -> tuple[int, list[str]]:
    """
    When nchan >= 2, ensure every training image has nchan channels (CYX, channel_axis=0).
    Single-channel images (shape (Y,X) or (1,Y,X)) are converted in-place to (nchan,Y,X) by
    duplicating the channel, so Omnipose does not raise IndexError in random_crop_warp.
    Returns (n_updated, errors).
    """
    if nchan < 2:
        return (0, [])
    pairs = get_training_pairs(train_path, img_filter, mask_filter)
    n_updated = 0
    errors: list[str] = []
    for img_path, _ in pairs:
        try:
            arr = safe_imread(str(img_path))
        except Exception as e:
            errors.append(f"{img_path.name}: {e}")
            continue
        # Need (nchan, Y, X) for Omnipose channel_axis=0. Single-channel: (Y, X) or (1, Y, X) or (Y, X, 1).
        if arr.ndim == 2:
            # (Y, X) -> (nchan, Y, X)
            arr = np.stack([arr] * nchan, axis=0)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            # (1, Y, X) -> (nchan, Y, X)
            arr = np.concatenate([arr] * nchan, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            # (Y, X, 1) -> (nchan, Y, X)
            arr = np.stack([arr[:, :, 0]] * nchan, axis=0)
        elif arr.ndim == 3 and arr.shape[0] == nchan:
            # Already (C, Y, X) with C == nchan
            continue
        elif arr.ndim == 3 and arr.shape[-1] == nchan and arr.shape[-1] in (1, 2, 3, 4):
            # Channels last (Y, X, C) -> (C, Y, X)
            arr = np.transpose(arr, (2, 0, 1))
            if arr.shape[0] < nchan:
                arr = np.concatenate([arr] * nchan, axis=0)
        elif arr.ndim == 3 and 2 <= arr.shape[0] <= 8 and arr.shape[0] >= nchan:
            # Channels first (C, Y, X) with C >= nchan; use first nchan
            arr = arr[:nchan].copy()
        else:
            errors.append(f"{img_path.name}: unexpected shape {arr.shape} for nchan={nchan}")
            continue
        try:
            tifffile.imwrite(str(img_path), arr)
        except Exception as e:
            errors.append(f"{img_path.name}: could not write: {e}")
            continue
        n_updated += 1
    return (n_updated, errors)


# Default subfolder for tiled training data (so we don't overwrite omnipose_masks)
OMNIPOSE_MASKS_TILED_SUBDIR = "omnipose_masks_tiled"


def tile_training_data(
    parent_dir: Path,
    img_filter: str,
    mask_filter: str,
    tile_size: int,
    min_cells_per_tile: int,
    out_dir: Path,
) -> tuple[int, int, list[str]]:
    """
    Tile image/mask pairs into smaller crops and write to out_dir. Dense tiles only.
    Reduces "many background crops" during Omnipose training and helps avoid 0-cell collapse.

    - Non-overlapping tiles of size (tile_size, tile_size). Last tile on each edge may be smaller.
    - Mask is relabeled per-tile to instance IDs 0, 1, 2, ... (Omnipose expects instance labels).
    - Tiles with fewer than min_cells_per_tile cells are skipped.
    - Output naming: base_tile000.tif, base_tile000_masks.tif (same img_filter/mask_filter so
      training dialog can use the same filters). Base = original stem minus img_filter.

    Returns (n_tiles_written, n_images_processed, errors).
    Each run clears the output folder first so the folder always contains exactly the tiles from this run (no stale files from a previous tiling).
    """
    pairs = get_training_pairs(parent_dir, img_filter, mask_filter)
    if not pairs:
        return (0, 0, ["No image/mask pairs found."])
    out_dir = Path(out_dir).resolve()
    # Clear existing content so this run fully overwrites (no mix of old and new tiles)
    if out_dir.exists():
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_size = max(8, (tile_size // 8) * 8)
    n_tiles_written = 0
    n_images_processed = 0
    errors: list[str] = []

    for img_path, mask_path in pairs:
        try:
            img = safe_imread(str(img_path))
            mask = safe_imread(str(mask_path))
        except Exception as e:
            errors.append(f"{img_path.name}: {e}")
            continue
        # Spatial dimensions: last two axes (Y, X)
        if img.ndim == 2:
            h, w = img.shape
            img_2d = img
        elif img.ndim == 3:
            h, w = img.shape[1], img.shape[2]
            img_2d = None
        else:
            errors.append(f"{img_path.name}: unsupported image ndim {img.ndim}")
            continue
        if mask.ndim != 2:
            if mask.ndim == 3 and mask.shape[-1] in (1, 3, 4):
                mask = np.max(mask, axis=-1) if mask.shape[-1] > 1 else mask[:, :, 0]
            else:
                mask = np.max(mask, axis=0) if mask.ndim == 3 else mask
        if mask.shape[0] != h or mask.shape[1] != w:
            errors.append(f"{img_path.name}: image shape {img.shape} vs mask shape {mask.shape}")
            continue

        base_original = img_path.stem.replace(img_filter, "") if img_filter else img_path.stem
        if not base_original or base_original == ".":
            continue
        n_images_processed += 1
        tile_idx = 0
        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                if img.ndim == 2:
                    img_crop = img[y0:y1, x0:x1]
                else:
                    img_crop = img[:, y0:y1, x0:x1].copy()
                mask_crop = mask[y0:y1, x0:x1].copy()
                # Relabel to instance 0, 1, 2, ...
                unique_vals = np.unique(mask_crop)
                unique_vals = unique_vals[unique_vals > 0]
                n_cells = len(unique_vals)
                if n_cells < min_cells_per_tile:
                    continue
                mapping = {int(v): i + 1 for i, v in enumerate(unique_vals)}
                mapping[0] = 0
                relabeled = np.zeros_like(mask_crop, dtype=np.uint32)
                for old_id, new_id in mapping.items():
                    relabeled[mask_crop == old_id] = new_id
                crop_h, crop_w = relabeled.shape[0], relabeled.shape[1]
                # If crop is smaller than tile_size, pad to tile_size with black only when padding is minimal (both dims >= half tile). Otherwise skip (would be mostly black).
                min_keep = tile_size // 2  # keep tile if both dims at least this; else skip
                if crop_h < tile_size or crop_w < tile_size:
                    is_full_image = y0 == 0 and x0 == 0 and y1 == h and x1 == w
                    if is_full_image and (crop_h <= tile_size and crop_w <= tile_size):
                        # Whole image smaller than tile: pad to tile_size so we have one consistent-size tile
                        pad_h = tile_size - crop_h
                        pad_w = tile_size - crop_w
                    elif crop_h >= min_keep and crop_w >= min_keep:
                        # Substantial crop (e.g. 415×512): pad to tile_size with least black
                        pad_h = tile_size - crop_h
                        pad_w = tile_size - crop_w
                    else:
                        continue  # would be mostly black, skip
                    if img.ndim == 2:
                        img_crop = np.pad(img_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
                        relabeled = np.pad(relabeled, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
                    else:
                        img_crop = np.pad(img_crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
                        relabeled = np.pad(relabeled, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
                stem = f"{base_original}_tile{tile_idx:03d}"
                img_out = out_dir / f"{stem}{img_filter}{img_path.suffix}"
                mask_out = out_dir / f"{stem}{mask_filter}{mask_path.suffix}"
                try:
                    tifffile.imwrite(str(img_out), img_crop)
                    tifffile.imwrite(str(mask_out), relabeled)
                except Exception as e:
                    errors.append(f"{stem}: {e}")
                    continue
                n_tiles_written += 1
                tile_idx += 1

    return (n_tiles_written, n_images_processed, errors)
