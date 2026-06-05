"""
Shared utilities for BactoMate GUI (e.g. TIFF reading with NumPy 2 compatibility).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import tifffile


def safe_imread(file_path: Union[str, Path]) -> np.ndarray:
    """
    Read image file (TIFF, PNG, JPEG, etc.). TIFF via tifffile; PNG/JPEG/etc. via PIL.
    NumPy 2.0 compatible (handles tifffile newbyteorder issue).
    """
    path = Path(file_path)
    suf = path.suffix.lower()
    if suf in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        from PIL import Image
        img = Image.open(file_path)
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr
    try:
        return tifffile.imread(file_path)
    except (AttributeError, TypeError) as e:
        if "newbyteorder" not in str(e):
            raise
        try:
            from PIL import Image
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
            if len(frames) > 1:
                return np.stack(frames, axis=0)
            img.seek(0)
            return np.array(img)
        except ImportError:
            raise RuntimeError(
                "Required dependency Pillow is missing (needed for TIFF reading). "
                "Reinstall BactoMate dependencies: pip install -r requirements.txt"
            ) from e
        except Exception as pil_error:
            raise e from pil_error
