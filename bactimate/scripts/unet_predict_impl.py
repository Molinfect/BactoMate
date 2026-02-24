"""
In-repo U-Net patch-based prediction. Requires TensorFlow.
Loads a trained .keras model and patch_config, runs inference on images in a folder,
saves binary mask TIFFs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

# TensorFlow is an optional dependency for the U-Net tab
import tensorflow as tf
from tensorflow import keras


def _load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        return {"patch_height": 256, "patch_width": 256}
    with open(config_path) as f:
        return json.load(f)


def _read_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[-1] in (2, 3, 4):
        arr = arr[..., 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


def load_model_and_config(
    model_path: Path,
    config_path: Path,
):
    """Load U-Net model and config for reuse (e.g. preview dialog). Returns (model, cfg_dict)."""
    cfg = _load_config(config_path)
    model = keras.models.load_model(model_path)
    return model, cfg


def predict_one_image(
    model: "keras.Model",
    cfg: dict,
    image: np.ndarray,
) -> np.ndarray:
    """
    Run U-Net on a single image array. Returns mask (H, W) uint8, 0 or 255.
    image: (H, W) or (H, W, C) float or uint.
    """
    patch_h = int(cfg.get("patch_height", 256))
    patch_w = int(cfg.get("patch_width", 256))
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[-1] in (2, 3, 4) else img[0]
    h, w = img.shape[:2]
    img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-8)
    img_4d = np.expand_dims(np.expand_dims(img_norm, -1), 0)
    img_patch = tf.image.resize(img_4d, (patch_h, patch_w), method="bilinear")
    pred = model.predict(img_patch, verbose=0)
    pred = np.asarray(pred[0, ..., 0])
    mask = (pred > 0.5).astype(np.uint8) * 255
    if mask.shape[0] != h or mask.shape[1] != w:
        mask_4d = np.expand_dims(np.expand_dims(mask, -1), 0)
        mask_resized = tf.image.resize(
            mask_4d, (h, w), method="nearest"
        ).numpy()[0, ..., 0]
    else:
        mask_resized = mask
    return np.asarray(mask_resized, dtype=np.uint8)


def run(
    input_folder: Path,
    output_folder: Path,
    model_path: Path,
    config_path: Path,
) -> int:
    cfg = _load_config(config_path)
    patch_h = int(cfg.get("patch_height", 256))
    patch_w = int(cfg.get("patch_width", 256))

    model = keras.models.load_model(model_path)

    exts = (".tif", ".tiff")
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for f in sorted(input_path.iterdir()):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        img = _read_image(f)
        if img.size == 0:
            continue
        h, w = img.shape[:2]
        # Resize to patch size, predict, resize mask back to original size
        img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-8)
        img_4d = np.expand_dims(np.expand_dims(img_norm, -1), 0)
        img_patch = tf.image.resize(img_4d, (patch_h, patch_w), method="bilinear")
        pred = model.predict(img_patch, verbose=0)
        pred = np.asarray(pred[0, ..., 0])
        mask = (pred > 0.5).astype(np.uint8) * 255
        if mask.shape[0] != h or mask.shape[1] != w:
            mask_4d = np.expand_dims(np.expand_dims(mask, -1), 0)
            mask_resized = tf.image.resize(
                mask_4d, (h, w), method="nearest"
            ).numpy()[0, ..., 0]
        else:
            mask_resized = mask
        out_file = output_path / f"{f.stem}_unet_masks.tif"
        from bactimate.tiff_metadata import build_metadata_dict_for_mask_from_source
        meta = build_metadata_dict_for_mask_from_source(Path(f), num_frames=1)
        tifffile.imwrite(
            out_file,
            np.asarray(mask_resized, dtype=np.uint8),
            metadata=meta,
            imagej=True,
        )

    return 0
