"""
Utilities to convert binary/semantic masks to Omnipose-compatible instance label matrices.
Used by the Train Omnipose dialog ("Make masks Omnipose fittable").
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Tuple

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
