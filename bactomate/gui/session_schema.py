"""Session log tab payload normalization (legacy keys → composed module UI)."""
from __future__ import annotations

from typing import Any

LEGACY_UNET_TAB_KEY = "unet_segment"
SEGMENTATION_TAB_KEY = "segmentation"
SEGMENTATION_BACKENDS = frozenset({"omnipose", "cellpose", "unet"})


def normalize_session_tabs(tabs: dict[str, Any]) -> dict[str, Any]:
    """
    Merge legacy ``unet_segment`` session data into the Segmentation module block.

    Older session logs stored U-Net inference params under a separate tab key while
    the GUI now uses ``segmentation`` with ``segmentation_backend``.
    """
    if not isinstance(tabs, dict):
        return {}
    out = dict(tabs)
    legacy_unet = out.pop(LEGACY_UNET_TAB_KEY, None)
    if not isinstance(legacy_unet, dict) or not legacy_unet:
        return out

    seg = out.get(SEGMENTATION_TAB_KEY)
    if not isinstance(seg, dict):
        seg = {}

    backend = str(seg.get("segmentation_backend") or "").strip().lower()
    if backend not in SEGMENTATION_BACKENDS:
        if legacy_unet.get("performed"):
            seg = {"segmentation_backend": "unet", **legacy_unet, **seg}
        else:
            seg = {**legacy_unet, **seg}

    out[SEGMENTATION_TAB_KEY] = seg
    return out
