"""Conversion smoke tests: TIFF metadata round-trip and BioIO reader mocks."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bactomate.modules.bioio_reader import get_info, load_data
from bactomate.modules.tiff_metadata import get_tiff_file_info, normalize_tiff_array_to_tcyx


def test_multichannel_tiff_split_roundtrip_metadata(tmp_path: Path) -> None:
    """Write a minimal TCYX stack, split naming convention, read back dimensions."""
    tifffile = pytest.importorskip("tifffile")
    src = tmp_path / "sample.tif"
    stack = np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8)
    tifffile.imwrite(
        src,
        stack,
        imagej=True,
        metadata={"axes": "TCYX", "frames": 2, "channels": 3},
    )
    info = get_tiff_file_info(src)
    assert info is not None
    assert info["num_time"] == 2
    assert info["num_channels"] == 3
    raw = tifffile.imread(src)
    tcyx = normalize_tiff_array_to_tcyx(raw, num_time=2, num_channels=3)
    assert tcyx.shape == (2, 3, 8, 8)
    out_dir = tmp_path / "Tif_files"
    out_dir.mkdir()
    for ch, tag in enumerate(("phase", "fi_GFP", "foci_GFP")):
        tifffile.imwrite(
            out_dir / f"sample_{tag}.tif",
            tcyx[:, ch],
            imagej=True,
            metadata={"axes": "TYX", "frames": 2},
        )
    assert (out_dir / "sample_phase.tif").is_file()
    phase_info = get_tiff_file_info(out_dir / "sample_phase.tif")
    assert phase_info is not None
    assert phase_info["num_time"] == 2


def test_bioio_get_info_mock(tmp_path: Path) -> None:
    """BioIO path returns normalized metadata when BioImage is available."""
    fake_path = tmp_path / "sample.nd2"
    fake_path.write_bytes(b"\x00")

    fake_img = MagicMock()
    fake_img.scenes = ["0"]
    fake_img.dims.order = "TCZYX"
    fake_img.shape = (4, 2, 1, 64, 64)
    fake_img.channel_names = ["Red", "Green"]
    fake_img.physical_pixel_sizes.X = 0.108
    fake_img.physical_pixel_sizes.Y = 0.108
    fake_img.close = MagicMock()

    import bactomate.modules.bioio_reader as bioio_reader

    bioio_reader._BIOIO_AVAILABLE = None
    with patch.object(bioio_reader, "is_available", return_value=True):
        with patch("bioio.BioImage", return_value=fake_img):
            info = get_info(fake_path)
    assert info is not None
    assert info["file_type"] == "nd2"
    assert info["num_time"] == 4
    assert info["num_channels"] == 2
    assert info["channel_names"] == ["Red", "Green"]
    assert info["pixel_size"] == pytest.approx(0.108)


def test_bioio_load_data_mock(tmp_path: Path) -> None:
    """BioIO load_data returns TCYX array when plugin succeeds."""
    fake_path = tmp_path / "sample.lif"
    fake_path.write_bytes(b"\x00")

    arr = np.zeros((3, 2, 1, 32, 32), dtype=np.uint16)
    fake_img = MagicMock()
    fake_img.scenes = ["0"]
    fake_img.dims.order = "TCZYX"
    fake_img.data = arr
    fake_img.close = MagicMock()

    import bactomate.modules.bioio_reader as bioio_reader

    bioio_reader._BIOIO_AVAILABLE = None
    with patch.object(bioio_reader, "is_available", return_value=True):
        with patch("bioio.BioImage", return_value=fake_img):
            data = load_data(fake_path, max_frames=2, use_first_z_only=True)
    assert data is not None
    assert data.shape == (2, 2, 1, 32, 32)
