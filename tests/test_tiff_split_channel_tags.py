"""Channel tag naming for multi-channel TIFF split (FI / Foci flags)."""
from __future__ import annotations


def _split_out_name(stem: str, safe: str, *, in_fi: bool, in_foci: bool, phase_used: bool) -> tuple[str, bool]:
    """Mirror file_conversion._convert_tiff naming."""
    if in_fi and in_foci:
        return f"{stem}_fi_foci_{safe}.tif", phase_used
    if in_fi:
        return f"{stem}_fi_{safe}.tif", phase_used
    if in_foci:
        return f"{stem}_foci_{safe}.tif", phase_used
    if not phase_used:
        return f"{stem}_phase.tif", True
    return f"{stem}_{safe}.tif", phase_used


def test_foci_only_channel_gets_foci_suffix():
    name, _ = _split_out_name("Hook-mal_001", "Channel_1", in_fi=False, in_foci=True, phase_used=False)
    assert name == "Hook-mal_001_foci_Channel_1.tif"


def test_fi_and_foci_combined_tag():
    name, _ = _split_out_name("sample", "GFP", in_fi=True, in_foci=True, phase_used=False)
    assert name == "sample_fi_foci_GFP.tif"
