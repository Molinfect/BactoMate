"""
Shared utilities for BactiMate GUI (e.g. TIFF reading with NumPy 2 compatibility).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import tifffile


def safe_imread(file_path: Union[str, Path]) -> np.ndarray:
    """
    Safely read TIFF file with NumPy 2.0 compatibility.
    Handles the newbyteorder() issue with tifffile.
    """
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
                "NumPy 2.0 compatibility issue reading TIFF file. "
                "Please install Pillow: pip install Pillow"
            ) from e
        except Exception as pil_error:
            raise e from pil_error
