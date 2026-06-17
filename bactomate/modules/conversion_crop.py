"""Helpers for optional Y/X crop during microscopy file conversion."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

MIN_CROP_SIZE = 16

RoiTuple = Tuple[int, int, int, int]


def normalize_roi(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    *,
    min_size: int = MIN_CROP_SIZE,
) -> Optional[RoiTuple]:
    """Clamp ROI to image bounds; return None if too small."""
    w_img = max(1, int(width))
    h_img = max(1, int(height))
    left = max(0, min(int(x0), int(x1), w_img - 1))
    top = max(0, min(int(y0), int(y1), h_img - 1))
    right = max(0, min(max(int(x0), int(x1)), w_img))
    bottom = max(0, min(max(int(y0), int(y1)), h_img))
    crop_w = right - left
    crop_h = bottom - top
    if crop_w < int(min_size) or crop_h < int(min_size):
        return None
    return (left, top, right, bottom)


def roi_width_height(roi: RoiTuple) -> Tuple[int, int]:
    x0, y0, x1, y1 = roi
    return max(0, x1 - x0), max(0, y1 - y0)


def format_roi_status(roi: Optional[RoiTuple]) -> str:
    if roi is None:
        return "Crop: full image"
    x0, y0, x1, y1 = roi
    w, h = roi_width_height(roi)
    return f"Crop: x={x0} y={y0} w={w} h={h}"


def roi_display_to_native(
    roi: RoiTuple,
    native_hw: Tuple[int, int],
    display_hw: Tuple[int, int],
) -> Optional[RoiTuple]:
    """Map ROI from scaled preview coordinates to native image pixels."""
    nh, nw = int(native_hw[0]), int(native_hw[1])
    dh, dw = int(display_hw[0]), int(display_hw[1])
    if nh <= 0 or nw <= 0 or dh <= 0 or dw <= 0:
        return None
    x0, y0, x1, y1 = roi
    sx = nw / float(dw)
    sy = nh / float(dh)
    return normalize_roi(
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        nw,
        nh,
    )


def roi_native_to_display(
    roi: RoiTuple,
    native_hw: Tuple[int, int],
    display_hw: Tuple[int, int],
) -> Optional[RoiTuple]:
    """Map native ROI to scaled preview coordinates for overlay drawing."""
    nh, nw = int(native_hw[0]), int(native_hw[1])
    dh, dw = int(display_hw[0]), int(display_hw[1])
    if nh <= 0 or nw <= 0 or dh <= 0 or dw <= 0:
        return None
    x0, y0, x1, y1 = roi
    sx = dw / float(nw)
    sy = dh / float(nh)
    return normalize_roi(
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        dw,
        dh,
        min_size=8,
    )


def apply_yx_crop(arr: np.ndarray, roi: RoiTuple) -> np.ndarray:
    """Crop trailing Y/X axes of an array (TYX, TCYX, TCZYX, YX, etc.)."""
    x0, y0, x1, y1 = roi
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)
    if arr.ndim < 2:
        raise ValueError(f"Cannot crop array with ndim={arr.ndim}")
    native_h, native_w = int(arr.shape[-2]), int(arr.shape[-1])
    clamped = normalize_roi(left, top, right, bottom, native_w, native_h)
    if clamped is None:
        raise ValueError(
            f"Crop ROI {roi!r} is invalid for image size ({native_h}, {native_w})"
        )
    left, top, right, bottom = clamped
    return np.ascontiguousarray(arr[..., top:bottom, left:right])


def parse_crop_cli_args(values: list[int] | tuple[int, ...] | None) -> Optional[RoiTuple]:
    if not values:
        return None
    if len(values) != 4:
        raise ValueError("--crop requires exactly four integers: x0 y0 x1 y1")
    return tuple(int(v) for v in values)  # type: ignore[return-value]
