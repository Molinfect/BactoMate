"""BactoMate TIFF channel output naming (phase / FI / foci tags)."""
from __future__ import annotations


def sanitize_channel_name_for_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in (name or "channel"))
    safe = safe.strip("_") or "channel"
    return safe


def split_channel_output_name(
    stem: str,
    channel_name: str,
    *,
    in_fi: bool,
    in_foci: bool,
    phase_used: bool,
) -> tuple[str, bool]:
    """
    Return ``(filename.tif, new_phase_used)`` for one exported channel.

    Matches ND2 conversion and multi-channel TIFF split conventions.
    """
    safe = sanitize_channel_name_for_filename(channel_name)
    if in_fi and in_foci:
        return f"{stem}_fi_foci_{safe}.tif", phase_used
    if in_fi:
        return f"{stem}_fi_{safe}.tif", phase_used
    if in_foci:
        return f"{stem}_foci_{safe}.tif", phase_used
    if not phase_used:
        return f"{stem}_phase.tif", True
    return f"{stem}_{safe}.tif", phase_used
