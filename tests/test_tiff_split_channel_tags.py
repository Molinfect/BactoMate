"""Channel tag naming for multi-channel TIFF split (FI / Foci flags)."""
from __future__ import annotations

from bactomate.modules.conversion_channel_names import split_channel_output_name


def test_foci_only_channel_gets_foci_suffix():
    name, _ = split_channel_output_name(
        "Hook-mal_001", "Channel_1", in_fi=False, in_foci=True, phase_used=False
    )
    assert name == "Hook-mal_001_foci_Channel_1.tif"


def test_fi_and_foci_combined_tag():
    name, _ = split_channel_output_name("sample", "GFP", in_fi=True, in_foci=True, phase_used=False)
    assert name == "sample_fi_foci_GFP.tif"
