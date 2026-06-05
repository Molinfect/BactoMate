"""Per-cell symmetric colocalization score (no heavy dependencies)."""
from __future__ import annotations


def coloc_match_index_value(
    coloc_pairs: int,
    foci_count_a: int,
    foci_count_b: int,
) -> str:
    """
    Per-cell symmetric colocalization score for one channel pair.

    - Both channels have zero foci → ``no foci``
    - Foci in only one channel → ``no correlation``
    - Otherwise → ``coloc_pairs / max(foci_a, foci_b)`` as a percent (e.g. 1 pair with
      1 vs 6 foci → 17%, not 100%).
    """
    a = max(0, int(foci_count_a))
    b = max(0, int(foci_count_b))
    pairs = max(0, int(coloc_pairs))
    if a == 0 and b == 0:
        return "no foci"
    if a == 0 or b == 0:
        return "no correlation"
    denom = max(a, b)
    pct = int(round(100.0 * min(pairs, denom) / float(denom)))
    return f"{max(0, min(100, pct))}%"
