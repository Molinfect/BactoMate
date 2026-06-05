"""Progress text/percent helpers for Filter File/Folder (no Qt dependency)."""
from __future__ import annotations

from pathlib import Path


def format_filter_batch_detail(
    *,
    file_index: int,
    file_count: int,
    mask_name: str,
    channel_name: str | None = None,
    channel_index: int | None = None,
    channel_count: int | None = None,
    extra: str | None = None,
) -> str:
    """Multi-line detail for batch Filter File/Folder (avoids one clipped mega-line)."""
    lines = [f"File {int(file_index)}/{int(file_count)}: {str(mask_name)}"]
    if channel_name:
        ch_line = f"Foci stack: {channel_name}"
        if channel_index is not None and channel_count is not None:
            ch_line += f" ({int(channel_index)}/{int(channel_count)})"
        lines.append(ch_line)
    if extra:
        lines.append(str(extra))
    return "\n".join(lines)


def batch_filter_percent(
    *,
    file_index: int,
    file_count: int,
    cell_phase: float | None = None,
    channel_index: int | None = None,
    channel_count: int | None = None,
    channel_phase: float = 0.0,
    finishing_phase: float | None = None,
) -> int:
    """
    Map batch Filter File/Folder work to 0–100%.

    Per file: ~72% cells, ~3% write mask, ~25% foci (split across channels).
    Optional finishing_phase (0–1) uses the last 8% after all files.
    """
    n_files = max(1, int(file_count))
    fi = max(0, min(int(file_index), n_files - 1))
    file_base = float(fi) / float(n_files)
    file_span = 1.0 / float(n_files)

    if finishing_phase is not None:
        return int(round(100.0 * (0.92 + 0.08 * max(0.0, min(1.0, float(finishing_phase))))))

    pct = 0
    if cell_phase is not None:
        frac = max(0.0, min(1.0, float(cell_phase)))
        pct = int(round(100.0 * (file_base + file_span * 0.72 * frac)))
    elif channel_index is not None:
        n_ch = max(1, int(channel_count or 1))
        ci = max(0, min(int(channel_index), n_ch - 1))
        ch_frac = max(0.0, min(1.0, float(channel_phase)))
        inner = 0.75 + (0.25 * (float(ci) + ch_frac) / float(n_ch))
        pct = int(round(100.0 * (file_base + file_span * inner)))
    else:
        pct = int(round(100.0 * file_base))
    # Reserve 92–100% for post-batch workbook refresh; never show 100% until finishing_phase.
    return min(91, max(0, pct))


def batch_foci_step_title(
    foci_source: Path | None,
    channel_index: int,
    channel_count: int,
    phase: str,
) -> str:
    """Step line that always names the channel (dsRed, GFP, …)."""
    from bactomate.gui.filter_preview_utils import short_foci_stack_label

    n_ch = max(1, int(channel_count))
    ci = max(0, min(int(channel_index), n_ch - 1)) + 1
    if foci_source is not None:
        ch = short_foci_stack_label(Path(foci_source))
    else:
        ch = "foci"
    return f"Foci · {ch} ({ci}/{n_ch}) · {phase}"
