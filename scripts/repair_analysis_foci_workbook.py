#!/usr/bin/env python3
"""
Re-apply foci CSV integration + Run Info to an existing *_analysis.xlsx.

Usage:
  python scripts/repair_analysis_foci_workbook.py \\
    --mask /path/to/filtered_masks/sample_mask_filtered.tif \\
    [--prefer-filtered]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_analyze_masks_base():
    """Load AnalyzeMasksBase without importing bactomate.gui.tabs (avoids heavy tab deps)."""
    import importlib.util

    path = PROJECT_ROOT / "bactomate" / "gui" / "tabs" / "_analyze_masks_base.py"
    spec = importlib.util.spec_from_file_location("bactomate_analyze_masks_base", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.AnalyzeMasksBase


class _FociResolverStub:
    """Minimal self for AnalyzeMasksBase foci path helpers (no GUI)."""

    def __init__(self, analyze_base_cls) -> None:
        cls = analyze_base_cls
        self._candidate_match_stems = cls._candidate_match_stems.__get__(self, cls)
        self._matches_foci_name = cls._matches_foci_name.__get__(self, cls)
        self._extract_channel_tag_for_mask_stem = cls._extract_channel_tag_for_mask_stem.__get__(self, cls)
        self._looks_non_channel_token = cls._looks_non_channel_token.__get__(self, cls)
        self._resolve_foci_paths_for_mask = cls._resolve_foci_paths_for_mask.__get__(self, cls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="Mask TIFF used for analysis")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=None,
        help="Analysis xlsx (default: _analysis/{stem}_analysis.xlsx)",
    )
    parser.add_argument(
        "--prefer-filtered",
        action="store_true",
        default=True,
        help="Prefer foci_filtered CSVs when present (default: on)",
    )
    parser.add_argument(
        "--no-prefer-filtered",
        action="store_false",
        dest="prefer_filtered",
        help="Use foci_detection CSVs only",
    )
    args = parser.parse_args()

    mask_path = Path(args.mask).expanduser().resolve()
    if not mask_path.is_file():
        print(f"Mask not found: {mask_path}", file=sys.stderr)
        return 1

    from bactomate.gui.analysis_paths import analysis_dir_from_base, base_dir_for_mask, stem_from_mask_path
    from bactomate.gui.analysis_workbook import integrate_foci_csv_into_workbook, write_run_info_sheet

    if args.workbook is not None:
        out_path = Path(args.workbook).expanduser().resolve()
    else:
        stem = stem_from_mask_path(mask_path)
        out_path = analysis_dir_from_base(base_dir_for_mask(mask_path)) / f"{stem}_analysis.xlsx"

    if not out_path.is_file():
        print(f"Workbook not found: {out_path}", file=sys.stderr)
        return 1

    AnalyzeMasksBase = _load_analyze_masks_base()
    stub = _FociResolverStub(AnalyzeMasksBase)

    outcome = integrate_foci_csv_into_workbook(
        stub,
        mask_path,
        out_path,
        prefer_filtered=bool(args.prefer_filtered),
    )
    for line in outcome.messages:
        print(line, end="" if line.endswith("\n") else "\n")

    manifest = {
        "Repair": "foci CSV integration",
        "Filtered mask file used": str(mask_path),
        "Output workbook": str(out_path),
    }
    from bactomate.gui.analysis_workbook import outcome_to_manifest_entries

    manifest.update(outcome_to_manifest_entries(outcome))
    write_run_info_sheet(out_path, manifest)

    print(f"Done. performed={outcome.performed} in_cell_foci={outcome.in_cell_foci}")
    return 0 if outcome.performed or outcome.channels else 1


if __name__ == "__main__":
    raise SystemExit(main())
