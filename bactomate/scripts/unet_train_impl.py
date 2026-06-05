"""
In-repo U-Net patch-based training. Requires TensorFlow.
Trains a small 2D U-Net for binary segmentation from image/mask pairs and patch_config.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from bactomate.gui.mask_utils import get_training_pairs

# TensorFlow is an optional dependency for the U-Net tab
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _log(msg: str) -> None:
    """Print to stdout so the GUI training log shows it."""
    print(msg, flush=True)


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _read_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[-1] in (2, 3, 4):
        arr = arr[..., 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


def _read_mask_binary(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.asarray(arr, dtype=np.float32)
    arr = (arr > 0).astype(np.float32)
    return arr


def _build_unet(patch_h: int, patch_w: int) -> keras.Model:
    """Small 2D U-Net for binary segmentation."""
    inputs = keras.Input(shape=(patch_h, patch_w, 1))
    # Encoder
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    # Bottleneck
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(b)
    # Decoder
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Concatenate()([u1, c3])
    u1 = layers.Conv2D(128, 3, activation="relu", padding="same")(u1)
    u1 = layers.Conv2D(128, 3, activation="relu", padding="same")(u1)
    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    u2 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    u3 = layers.UpSampling2D(2)(u2)
    u3 = layers.Concatenate()([u3, c1])
    u3 = layers.Conv2D(32, 3, activation="relu", padding="same")(u3)
    u3 = layers.Conv2D(32, 3, activation="relu", padding="same")(u3)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(u3)
    return keras.Model(inputs, outputs)


def _make_dataset(
    train_path: Path,
    img_filter: str,
    mask_filter: str,
    patch_h: int,
    patch_w: int,
    batch_size: int,
    num_samples: int,
) -> tf.data.Dataset:
    pairs = get_training_pairs(train_path, img_filter, mask_filter)
    if not pairs:
        raise ValueError("No image/mask pairs found in training directory.")

    def gen():
        rng = np.random.default_rng()
        for _ in range(num_samples):
            img_path, mask_path = pairs[rng.integers(0, len(pairs))]
            img = _read_image(img_path)
            mask = _read_mask_binary(mask_path)
            if img.shape != mask.shape:
                if img.shape[:2] == mask.shape[:2]:
                    mask = np.expand_dims(mask, -1) if mask.ndim == 2 else mask
                else:
                    continue
            h, w = img.shape[:2]
            if h < patch_h or w < patch_w:
                continue
            y = rng.integers(0, h - patch_h + 1)
            x = rng.integers(0, w - patch_w + 1)
            img_patch = img[y : y + patch_h, x : x + patch_w]
            mask_patch = mask[y : y + patch_h, x : x + patch_w]
            if img_patch.ndim == 2:
                img_patch = np.expand_dims(img_patch, -1)
            if mask_patch.ndim == 2:
                mask_patch = np.expand_dims(mask_patch, -1)
            img_patch = (img_patch - np.min(img_patch)) / (np.ptp(img_patch) + 1e-8)
            yield img_patch.astype(np.float32), mask_patch.astype(np.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(patch_h, patch_w, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(patch_h, patch_w, 1), dtype=tf.float32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def run(
    train_path: Path,
    config_path: Path,
    output_model_path: Path,
    epochs: int = 50,
    batch_size: int = 16,
    use_gpu: bool = True,
    img_filter: str = "",
    mask_filter: str = "_mask",
) -> int:
    # --- Device info (so user sees in log whether GPU is used) ---
    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")
        _log("[U-Net] Training on CPU (Use GPU unchecked).")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            _log(f"[U-Net] Using GPU: {gpus[0].name}")
        else:
            _log("[U-Net] No GPU found — training on CPU (may be slow).")
            if sys.platform == "darwin":
                _log("[U-Net] On Apple Silicon, install tensorflow-metal for GPU: pip install tensorflow-metal")
    cfg = _load_config(config_path)
    patch_h = int(cfg.get("patch_height", 256))
    patch_w = int(cfg.get("patch_width", 256))
    n_pairs = len(get_training_pairs(Path(train_path), img_filter, mask_filter))
    num_samples = int(cfg.get("patches_per_image", 500)) * max(1, n_pairs)
    num_samples = min(num_samples, 50000)
    steps_per_epoch = max(1, num_samples // batch_size)

    _log(f"[U-Net] Patch size: {patch_h}x{patch_w}  |  Pairs: {n_pairs}  |  Samples/epoch: {num_samples}  |  Steps/epoch: {steps_per_epoch}")

    model = _build_unet(patch_h, patch_w)
    # Use legacy Adam on Mac (faster on M1/M2/M4 and avoids "runs slowly" warning)
    try:
        optimizer = keras.optimizers.legacy.Adam(1e-3)
    except AttributeError:
        optimizer = keras.optimizers.Adam(1e-3)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    ds = _make_dataset(
        Path(train_path), img_filter, mask_filter, patch_h, patch_w, batch_size, num_samples
    )
    # verbose=2: one line per epoch (cleaner log than verbose=1 progress bar)
    model.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=2)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)
    # Config is already name_patch_config.json from analyze step; prediction uses same stem.
    return 0
