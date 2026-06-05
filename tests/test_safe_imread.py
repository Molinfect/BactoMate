"""
Tests for gui.utils.safe_imread (TIFF reading with NumPy 2 compatibility).
Requires: numpy, tifffile (in requirements.txt). Run with: pytest tests/ or python -m tests.test_safe_imread
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bactomate.gui.utils import safe_imread


def test_safe_imread_single_frame():
    """Read a single-frame TIFF written with tifffile."""
    import tifffile
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "single.tif"
        tifffile.imwrite(path, arr)
        out = safe_imread(path)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out, arr)


def test_safe_imread_stack():
    """Read a multi-frame stack."""
    import tifffile
    arr = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "stack.tif"
        tifffile.imwrite(path, arr)
        out = safe_imread(path)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out, arr)


def test_safe_imread_accepts_path_object():
    """safe_imread accepts pathlib.Path."""
    import tifffile
    arr = np.zeros((2, 2), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.tif"
        tifffile.imwrite(path, arr)
        out = safe_imread(path)
    assert out.shape == (2, 2)


def test_safe_imread_nonexistent_raises():
    try:
        safe_imread("/nonexistent/path/12345.tif")
        assert False, "Expected an exception"
    except Exception:
        pass


if __name__ == "__main__":
    test_safe_imread_single_frame()
    test_safe_imread_stack()
    test_safe_imread_accepts_path_object()
    test_safe_imread_nonexistent_raises()
    print("All safe_imread tests passed.")
