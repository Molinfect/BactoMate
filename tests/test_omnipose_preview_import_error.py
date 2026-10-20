"""Preview ImportError text distinguishes a missing package from a bad version."""
from __future__ import annotations

from bactomate.gui.omnipose_utils import format_omnipose_preview_import_error


def test_ncolor_unique_nonzero_is_an_incompatible_version() -> None:
    msg = format_omnipose_preview_import_error(
        "cannot import name 'unique_nonzero' from 'ncolor' "
        "(/venv/lib/python3.11/site-packages/ncolor/__init__.py)"
    )
    assert msg.startswith("Incompatible dependency version:")
    assert "unique_nonzero" in msg
    assert 'pip install "ncolor>=1.4.3,<2"' in msg
    assert "cellpose-omni" not in msg


def test_other_missing_name_does_not_say_install_cellpose_omni() -> None:
    msg = format_omnipose_preview_import_error(
        "cannot import name 'OMNI_MODELS' from 'omnipose.core'"
    )
    assert msg.startswith("Incompatible dependency version:")
    assert "pip install -r requirements.txt" in msg
    assert "cellpose-omni" not in msg


def test_missing_module_still_points_at_cellpose_omni() -> None:
    msg = format_omnipose_preview_import_error("No module named 'cellpose_omni'")
    assert "Preview requires additional libraries:" in msg
    assert "pip install cellpose-omni" in msg


def test_torch_testing_and_imageio_messages_stay_specific() -> None:
    torch_msg = format_omnipose_preview_import_error(
        "cannot import name 'make_tensor' from 'torch.testing'"
    )
    assert "Preview requires torch.testing module." in torch_msg
    assert "python main.py" in torch_msg

    imageio_msg = format_omnipose_preview_import_error(
        "No package metadata was found for imageio"
    )
    assert "missing imageio package metadata" in imageio_msg
    assert "pip install imageio" in imageio_msg
