"""
Minimal, dependency-light training data augmentation utilities for microscopy TIFF datasets.

This module is intentionally lightweight (numpy + tifffile) so it can be used from the GUI
without pulling in heavy augmentation dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


@dataclass(frozen=True)
class AugmentOptions:
    num_aug: int = 4
    rotate_90: bool = True
    flip: bool = True
    add_noise: bool = False
    noise_sigma: float = 5.0  # in raw intensity units for integer images
    adjust_brightness: bool = False
    brightness_min: float = 0.9
    brightness_max: float = 1.1


def _iter_image_candidates(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        yield p


def find_training_pairs(folder: Path, img_filter: str, mask_filter: str) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    """
    Return (pairs, missing_mask_image_names).

    Pairing logic matches the GUI's `get_training_pairs` approach: base stem is image stem with
    `img_filter` removed, and mask is `base + mask_filter + ext`.
    """
    if not folder.is_dir():
        return ([], [])
    pairs: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    for img_path in _iter_image_candidates(folder):
        stem = img_path.stem
        # Skip masks
        if mask_filter and stem.endswith(mask_filter):
            continue
        if img_filter and img_filter not in stem:
            continue
        base = stem.replace(img_filter, "") if img_filter else stem
        if not base or base == ".":
            continue
        mask_path = folder / f"{base}{mask_filter}{img_path.suffix}"
        if mask_path.is_file():
            pairs.append((img_path, mask_path, base))
        else:
            missing.append(img_path.name)
    return (pairs, missing)


def _preserve_dtype(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(x), info.min, info.max).astype(dtype, copy=False)
    return x.astype(dtype, copy=False)


def _augment_one(
    img: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    opts: AugmentOptions,
) -> tuple[np.ndarray, np.ndarray]:
    img_aug = np.asarray(img)
    mask_aug = np.asarray(mask)

    # Ensure we apply the same geometric transforms to image and mask.
    if opts.rotate_90:
        k = int(rng.integers(0, 4))
        if k:
            img_aug = np.rot90(img_aug, k, axes=(0, 1))
            mask_aug = np.rot90(mask_aug, k, axes=(0, 1))

    if opts.flip:
        # horizontal
        if bool(rng.integers(0, 2)):
            img_aug = np.flip(img_aug, axis=1)
            mask_aug = np.flip(mask_aug, axis=1)
        # vertical
        if bool(rng.integers(0, 2)):
            img_aug = np.flip(img_aug, axis=0)
            mask_aug = np.flip(mask_aug, axis=0)

    # Intensity augmentations only for image.
    if opts.adjust_brightness:
        factor = float(rng.uniform(opts.brightness_min, opts.brightness_max))
        img_dtype = img_aug.dtype
        img_aug = _preserve_dtype(img_aug.astype(np.float32) * factor, img_dtype)

    if opts.add_noise:
        img_dtype = img_aug.dtype
        x = img_aug.astype(np.float32, copy=False)
        sigma = float(opts.noise_sigma)
        x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        img_aug = _preserve_dtype(x, img_dtype)

    return (np.ascontiguousarray(img_aug), np.ascontiguousarray(mask_aug))


def augment_one_pair_in_memory(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    options: AugmentOptions,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one randomized augmentation to an (image, mask) pair and return augmented arrays.

    This is used for GUI preview (no disk writes).
    """
    rng = np.random.default_rng(seed)
    return _augment_one(image, mask, rng, options)


def augment_training_dataset(
    *,
    input_dir: Path,
    img_filter: str,
    mask_filter: str,
    output_dir: Path | None,
    options: AugmentOptions,
    seed: int = 0,
) -> tuple[Path, int, int, list[str]]:
    """
    Augment dataset and write augmented image/mask TIFF pairs.

    If `output_dir` is None, writes into `input_dir` (requested behavior). Files are never overwritten;
    new files are written as `<base>_augNNN<img_filter>.tif` and `<base>_augNNN<mask_filter>.tif`.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir) if output_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, missing = find_training_pairs(in_dir, img_filter, mask_filter)
    if not pairs:
        return (out_dir, 0, 0, missing)

    rng = np.random.default_rng(int(seed))
    written_images = 0
    written_masks = 0

    for img_path, mask_path, base in pairs:
        try:
            img = tifffile.imread(str(img_path))
            mask = tifffile.imread(str(mask_path))
        except Exception:
            continue

        for i in range(1, int(options.num_aug) + 1):
            aug_base = f"{base}_aug{i:03d}"
            out_img = out_dir / f"{aug_base}{img_filter}{img_path.suffix}"
            out_mask = out_dir / f"{aug_base}{mask_filter}{mask_path.suffix}"
            if out_img.exists() or out_mask.exists():
                # Don't overwrite; skip if either already exists.
                continue
            img_aug, mask_aug = _augment_one(img, mask, rng, options)
            try:
                tifffile.imwrite(str(out_img), img_aug)
                written_images += 1
            except Exception:
                pass
            try:
                tifffile.imwrite(str(out_mask), mask_aug)
                written_masks += 1
            except Exception:
                pass

    return (out_dir, written_images, written_masks, missing)

