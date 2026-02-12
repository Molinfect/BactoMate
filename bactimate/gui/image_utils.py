"""
Qt-native image display utilities: convert numpy arrays to QPixmap for preview without matplotlib.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PyQt6.QtGui import QImage, QPixmap

# Max dimension for preview (avoids huge overlays and slow QPixmap/paint)
PREVIEW_MAX_SIZE = 2048


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 in [0, 255]. Handles 2D (grayscale) and 3D (H, W, C)."""
    if image.dtype == np.uint8 and (image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in (1, 3, 4))):
        out = np.ascontiguousarray(image)
        if out.ndim == 3 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val <= min_val:
        out = np.zeros_like(image, dtype=np.uint8)
    else:
        out = ((image.astype(np.float64) - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def to_qpixmap(image: np.ndarray) -> QPixmap:
    """
    Convert a numpy image to QPixmap for display.
    - 2D: grayscale (Format_Grayscale8).
    - 3D with last dim 3: RGB (Format_RGB888).
    - 3D with last dim 4: RGBA (Format_RGBA8888).
    Uses a copy of the buffer so the QImage is safe after the array may be reused.
    """
    image_u8 = _to_uint8(image)
    if image_u8.ndim == 2:
        height, width = image_u8.shape
        qimage = QImage(
            image_u8.data,
            width,
            height,
            image_u8.strides[0],
            QImage.Format.Format_Grayscale8,
        )
    else:
        height, width, channels = image_u8.shape
        if channels == 3:
            # bytesPerLine for RGB888
            bpl = image_u8.strides[0]
            qimage = QImage(
                image_u8.data,
                width,
                height,
                bpl,
                QImage.Format.Format_RGB888,
            )
        else:
            if channels != 4:
                # pad or take first 4 channels
                if channels < 4:
                    pad = np.zeros((height, width, 4 - channels), dtype=np.uint8)
                    image_u8 = np.concatenate([image_u8, pad], axis=-1)
                else:
                    image_u8 = image_u8[..., :4].copy()
            bpl = image_u8.strides[0]
            qimage = QImage(
                image_u8.data,
                width,
                height,
                bpl,
                QImage.Format.Format_RGBA8888,
            )
    return QPixmap.fromImage(qimage.copy())


def _label_colors_palette(n: int, seed: int = 42) -> np.ndarray:
    """Return (n, 3) RGB uint8 palette for label overlays (deterministic, no matplotlib)."""
    rng = np.random.default_rng(seed)
    colors = rng.integers(80, 256, size=(max(n, 1), 3), dtype=np.uint8)
    return colors


def _scale_with_scipy(
    image: np.ndarray, mask: np.ndarray, new_h: int, new_w: int, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import zoom

    if image.ndim == 2:
        scaled_image = zoom(
            np.asarray(image, dtype=np.float64), (new_h / h, new_w / w), order=1
        )
    else:
        scaled_image = zoom(
            image, (new_h / h, new_w / w, 1), order=1
        )
    scaled_mask = zoom(
        mask.astype(np.float64), (new_h / h, new_w / w), order=0
    ).astype(mask.dtype)
    return scaled_image, scaled_mask


def _scale_with_sampling(
    image: np.ndarray, mask: np.ndarray, new_h: int, new_w: int, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: sample every nth pixel (no scipy)."""
    sy, sx = max(1, h // new_h), max(1, w // new_w)
    if image.ndim == 2:
        scaled_image = image[::sy, ::sx].astype(np.float64)
    else:
        scaled_image = image[::sy, ::sx, :].astype(np.float64)
    scaled_mask = mask[::sy, ::sx]
    return scaled_image, scaled_mask


def scale_for_preview(
    image: np.ndarray,
    mask: np.ndarray,
    max_size: int = PREVIEW_MAX_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale image and mask so the longest side is at most max_size.
    Returns (scaled_image, scaled_mask). Speeds up overlay and painting for large images.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image, mask
    scale = max_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    try:
        return _scale_with_scipy(image, mask, new_h, new_w, h, w)
    except ImportError:
        return _scale_with_sampling(image, mask, new_h, new_w, h, w)


def draw_scale_bar(
    rgb: np.ndarray,
    pixel_size_um: float,
    scale_factor: float = 1.0,
    bar_um: float = 10.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    margin: int = 12,
) -> np.ndarray:
    """
    Draw a horizontal scale bar on an (H, W, 3) RGB uint8 array (in-place).
    pixel_size_um: µm per pixel in the *original* image.
    scale_factor: (display size / original size) so 1 display pixel = pixel_size_um/scale_factor µm.
    bar_um: length of the bar in µm.
    Bar is drawn at bottom-right with margin.
    """
    if pixel_size_um <= 0 or scale_factor <= 0:
        return rgb
    h, w = rgb.shape[:2]
    bar_px = max(2, int(round(bar_um * scale_factor / pixel_size_um)))
    # Clamp so bar fits
    bar_px = min(bar_px, w - 2 * margin)
    x0 = w - margin - bar_px
    x1 = w - margin
    y0 = h - margin - thickness
    y1 = h - margin
    y0 = max(0, y0)
    y1 = min(h, y1)
    x0 = max(0, x0)
    x1 = min(w, x1)
    rgb[y0:y1, x0:x1] = color
    return rgb


def build_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Build RGB overlay: normalized grayscale image with colored mask on top.
    image: (H, W) float or uint; mask: (H, W) int, 0 = background.
    Returns (H, W, 3) uint8. Vectorized (no Python loop over labels).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        gray = ((img - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)
    else:
        gray = np.zeros_like(img, dtype=np.uint8)
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float64)
    max_label = int(np.max(mask))
    if max_label < 1:
        return rgb.clip(0, 255).astype(np.uint8)
    palette = _label_colors_palette(max_label + 1)
    valid = mask > 0
    rgb[valid] = (1 - alpha) * rgb[valid] + alpha * palette[mask[valid]]
    return rgb.clip(0, 255).astype(np.uint8)
