"""
Test that _patch_omnipose_get_masks can be imported and called at startup.
Run with: pytest tests/test_patch_import.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_patch_import_and_callable():
    """Ensure the Omnipose patch hook exists and is idempotent."""
    from bactomate.gui.patches import _patch_omnipose_get_masks

    assert callable(_patch_omnipose_get_masks)
    # Call is idempotent and should not raise
    _patch_omnipose_get_masks()
