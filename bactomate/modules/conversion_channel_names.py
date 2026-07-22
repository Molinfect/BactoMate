"""BactoMate TIFF channel output naming (phase / FI / foci tags)."""
from __future__ import annotations

import re
from typing import Optional, Sequence

# Free-form microscope names are unreliable; these are soft hints for defaults only.
_PHASE_NAME_HINTS = frozenset(
    {
        "phase",
        "dia",
        "dic",
        "bf",
        "brightfield",
        "bright",
        "transmission",
        "trans",
        "phc",
        "pc",
        "tl",
        "ttrans",
        "ph",
    }
)


def sanitize_channel_name_for_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in (name or "channel"))
    safe = safe.strip("_") or "channel"
    return safe


def channel_name_looks_like_phase(name: str) -> bool:
    """Best-effort: true when a free-form channel label likely denotes BF/phase/DIA."""
    raw = (name or "").strip().lower()
    if not raw:
        return False
    tokens = set(re.sub(r"[^a-z0-9]+", " ", raw).split())
    if tokens & _PHASE_NAME_HINTS:
        return True
    return any(t.startswith("phase") for t in tokens)


def guess_phase_channel_index(
    channel_names: Sequence[str],
    *,
    channel_wavelengths: Optional[Sequence[dict]] = None,
) -> int:
    """
    Guess which channel should be Phase for conversion defaults.

    Prefer metadata ``phase_contrast``, then name hints (DIA/phase/…). Falls back to
    index 0 only when nothing else is known — never a hard pipeline assumption.
    """
    names = [str(n) for n in (channel_names or ())]
    n = len(names)
    if n <= 0:
        return 0
    if channel_wavelengths:
        for i, wl in enumerate(channel_wavelengths):
            if i >= n or not isinstance(wl, dict):
                continue
            if wl.get("phase_contrast") is True:
                return int(i)
    for i, name in enumerate(names):
        if channel_name_looks_like_phase(name):
            return int(i)
    return 0


def default_fi_indices_for_channels(
    num_channels: int,
    channel_names: Sequence[str],
    *,
    channel_wavelengths: Optional[Sequence[dict]] = None,
    phase_index: Optional[int] = None,
) -> set[int]:
    """Default FI ticks: every channel except the guessed (or given) phase channel."""
    n = max(0, int(num_channels))
    if n <= 0:
        return set()
    if phase_index is None:
        phase_index = guess_phase_channel_index(
            list(channel_names)[:n] if channel_names else [f"Channel_{i + 1}" for i in range(n)],
            channel_wavelengths=channel_wavelengths,
        )
    phase_index = max(0, min(int(phase_index), n - 1))
    return {i for i in range(n) if i != phase_index}


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

    Role tags are mandatory — never emit bare ``{stem}_{ChannelName}.tif``.
    Unmarked channels: first → ``_phase``; further unmarked → ``_fi_{name}``.
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
    # Remaining unmarked channel: treat as FI so free-form names never become bare files.
    return f"{stem}_fi_{safe}.tif", phase_used
