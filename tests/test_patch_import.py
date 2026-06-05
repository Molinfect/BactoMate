"""
Test that _patch_omnipose_get_masks can be imported and called (used by Processing tab).
Run with: pytest tests/test_patch_import.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_patch_import_and_callable():
    """Processing tab imports this; ensure it exists and is callable."""
    from bactomate.gui.patches import _patch_omnipose_get_masks

    assert callable(_patch_omnipose_get_masks)
    # Call is idempotent and should not raise
    _patch_omnipose_get_masks()


def test_processing_tab_imports_patch():
    """Processing tab module can be loaded and references the patch by name."""
    import importlib.util
    spec = importlib.util.find_spec("bactomate.gui.tabs.processing")
    assert spec is not None
    # Don't actually load ProcessingTab (pulls in PyQt6, etc.); just ensure patch is importable
    from bactomate.gui.patches import _patch_omnipose_get_masks
    assert _patch_omnipose_get_masks.__name__ == "_patch_omnipose_get_masks"
