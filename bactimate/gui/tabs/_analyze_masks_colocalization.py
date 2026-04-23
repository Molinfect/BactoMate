from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bactimate.gui.analysis_paths import analysis_dir_from_base, base_dir_for_mask, stem_from_mask_path
from bactimate.gui.artifact_paths import (
    foci_csv_path,
    filtered_foci_csv_path,
    legacy_foci_csv_candidates,
    raw_foci_csv_path,
)
from bactimate.gui.utils import safe_imread


def _safe_sheet_name(base: str, used: set[str]) -> str:
    bad = "\\/:*?[]"
    safe = "".join("_" if ch in bad else ch for ch in str(base)).strip() or "Sheet"
    safe = safe[:31]
    if safe not in used:
        used.add(safe)
        return safe
    stem_name = safe[:28] if len(safe) > 28 else safe
    n = 2
    cand = f"{stem_name}_{n}"
    while cand in used:
        n += 1
        cand = f"{stem_name}_{n}"
    used.add(cand)
    return cand


def _safe_col_key(raw: str) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return "foci"
    out = []
    for ch in txt:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    key = "".join(out).strip("_")
    while "__" in key:
        key = key.replace("__", "_")
    return key or "foci"


def _load_mask_frame0(mask_path: Path) -> np.ndarray | None:
    try:
        arr = np.asarray(safe_imread(str(mask_path)))
        if arr.ndim == 2:
            return arr.astype(np.int32)
        if arr.ndim >= 3:
            return np.asarray(arr[0], dtype=np.int32)
    except Exception:
        return None
    return None


def _update_per_cell_data_counts(
    *,
    analysis_path: Path,
    channel_rows: list[tuple[str, list[dict[str, Any]]]],
    pair_tables: list[tuple[str, str, "pd.DataFrame", int]],
) -> None:
    """
    Add per-cell counts to 'Per-Cell Data' in the analysis workbook:
    - {channel}_foci_count (counts of assigned foci in that cell; frame 0)
    - coloc_pairs_count (valid colocalized pairs in that cell; summed across channel pairs; frame 0)
    """
    import pandas as pd

    analysis_path = Path(analysis_path)
    if not analysis_path.exists():
        return

    try:
        per_cell = pd.read_excel(analysis_path, sheet_name="Per-Cell Data")
    except Exception:
        return
    if per_cell is None or per_cell.empty:
        return

    cell_col = "Cell_ID" if "Cell_ID" in per_cell.columns else ("Cell Label" if "Cell Label" in per_cell.columns else None)
    if cell_col is None:
        return
    has_frame = "Frame" in per_cell.columns

    # Foci per cell per channel (frame 0)
    for label, rows in (channel_rows or []):
        col = f"{_safe_col_key(label)}_foci_count"
        counts: dict[int, int] = {}
        for r in rows or []:
            try:
                if not bool(r.get("in_cell", False)):
                    continue
                cid = r.get("cell_id", None)
                if cid in (None, "", "None"):
                    continue
                cid_i = int(cid)
                if cid_i <= 0:
                    continue
                counts[cid_i] = counts.get(cid_i, 0) + 1
            except Exception:
                continue
        if col not in per_cell.columns:
            per_cell[col] = 0
        if counts:
            if has_frame:
                per_cell.loc[per_cell["Frame"].astype(int) == 0, col] = (
                    per_cell.loc[per_cell["Frame"].astype(int) == 0, cell_col]
                    .astype("Int64")
                    .map(lambda v: int(counts.get(int(v), 0)) if v is not None and pd.notna(v) else 0)
                    .astype(int)
                )
            else:
                per_cell[col] = (
                    per_cell[cell_col]
                    .astype("Int64")
                    .map(lambda v: int(counts.get(int(v), 0)) if v is not None and pd.notna(v) else 0)
                    .astype(int)
                )

    # Coloc pairs per cell (frame 0): count only valid pairs in the same cell.
    coloc_counts: dict[int, int] = {}
    for _la, _lb, df, _count in (pair_tables or []):
        try:
            if df is None or df.empty or "message" in df.columns:
                continue
            if "cell_id_primary" not in df.columns:
                continue
            for _, row in df.iterrows():
                try:
                    if "is_valid" in df.columns and not bool(row.get("is_valid", True)):
                        continue
                    if "same_cell" in df.columns and not bool(row.get("same_cell", False)):
                        continue
                    cid = row.get("cell_id_primary", None)
                    if cid in (None, "", "None"):
                        continue
                    cid_i = int(cid)
                    if cid_i <= 0:
                        continue
                    coloc_counts[cid_i] = coloc_counts.get(cid_i, 0) + 1
                except Exception:
                    continue
        except Exception:
            continue

    coloc_col = "coloc_pairs_count"
    if coloc_col not in per_cell.columns:
        per_cell[coloc_col] = 0
    if coloc_counts:
        if has_frame:
            per_cell.loc[per_cell["Frame"].astype(int) == 0, coloc_col] = (
                per_cell.loc[per_cell["Frame"].astype(int) == 0, cell_col]
                .astype("Int64")
                .map(lambda v: int(coloc_counts.get(int(v), 0)) if v is not None and pd.notna(v) else 0)
                .astype(int)
            )
        else:
            per_cell[coloc_col] = (
                per_cell[cell_col]
                .astype("Int64")
                .map(lambda v: int(coloc_counts.get(int(v), 0)) if v is not None and pd.notna(v) else 0)
                .astype(int)
            )

    # Write back updated Per-Cell Data (preserve other sheets)
    with pd.ExcelWriter(analysis_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        per_cell.to_excel(writer, sheet_name="Per-Cell Data", index=False)


def _read_foci_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    import csv

    if csv_path is None or not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                y = int(float(r.get("y", 0) or 0))
                x = int(float(r.get("x", 0) or 0))
            except Exception:
                continue
            try:
                intensity = float(r.get("intensity", 0) or 0.0)
            except Exception:
                intensity = 0.0
            prob_raw = r.get("probability", None)
            prob = None
            if prob_raw not in (None, "", "None"):
                try:
                    prob = float(prob_raw)
                except Exception:
                    prob = None
            frame = 0
            if "frame" in (reader.fieldnames or []):
                try:
                    frame = int(float(r.get("frame", 0) or 0))
                except Exception:
                    frame = 0
            rows.append({"frame": frame, "y": y, "x": x, "intensity": intensity, "probability": prob})
    return rows


def _csv_path_for_source(
    pipeline_dir: Path,
    *,
    mask_path: Path,
    foci_source_path: Path,
    prefer_filtered: bool,
) -> Optional[Path]:
    filt = filtered_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path)
    if prefer_filtered and filt.exists():
        return filt
    canon = foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path)
    if canon.exists():
        return canon
    legacy = raw_foci_csv_path(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path)
    if legacy.exists():
        return legacy
    for cand in legacy_foci_csv_candidates(pipeline_dir, mask_path=mask_path, foci_source_path=foci_source_path):
        if cand.exists():
            return cand
    return None


def _channel_label_for_source(tab, mask_path: Path, foci_source_path: Path, used: set[str]) -> str:
    mask_stem = mask_path.stem.replace("_masks", "")
    tag = None
    try:
        tag = tab._extract_channel_tag_for_mask_stem(Path(foci_source_path).stem, mask_stem)
    except Exception:
        tag = None
    base = (tag or "").strip() or "foci"
    lbl = base
    n = 2
    while lbl in used:
        lbl = f"{base}_{n}"
        n += 1
    used.add(lbl)
    return lbl


def _assigned_foci_rows_for_channel(
    mask_labels: np.ndarray,
    channel_label: str,
    csv_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from bactimate.modules.foci_utils import assign_peaks_to_cells

    # Frame 0 only for these post-analysis sheets (matches prior behavior).
    peaks = [r for r in csv_rows if int(r.get("frame", 0) or 0) == 0]
    assigned, _ = assign_peaks_to_cells(peaks, mask_labels, dilation_px=1, frame_index=0)
    out: list[dict[str, Any]] = []
    for r in assigned:
        q = dict(r)
        q["channel_name"] = channel_label
        out.append(q)
    return out


def should_run_post_analysis_colocalization(tab) -> bool:
    cb = getattr(tab, "coloc_calc_all_pairs", None)
    return bool(cb is not None and cb.isChecked())


def run_post_analysis_colocalization_for_masks(tab, mask_paths: list[Path] | None) -> None:
    if not mask_paths:
        return
    if not should_run_post_analysis_colocalization(tab):
        return
    for mp in mask_paths:
        _run_all_pairs_for_mask(tab, mp)


def _run_all_pairs_for_mask(tab, mask_path: Path) -> tuple[int, list[str]]:
    import pandas as pd
    from openpyxl import load_workbook
    from bactimate.modules.analysis_coloc import ColocSettings, analyze_colocalization_foci_table

    log_lines: list[str] = []
    mask_path = Path(mask_path)
    mask_labels = _load_mask_frame0(mask_path)
    if mask_labels is None:
        return 0, [f"  Colocalization skipped for {mask_path.name}: could not read mask.\n"]

    prefer_filtered = bool(getattr(tab, "use_filtered_masks_cb", None) and tab.use_filtered_masks_cb.isChecked())
    pipeline_dir = mask_path.parent
    try:
        sources = tab._resolve_foci_paths_for_mask(mask_path)
    except Exception:
        sources = []
    if not sources:
        return 0, [f"  Colocalization skipped for {mask_path.name}: no matching *_foci sources.\n"]

    used_labels: set[str] = set()
    channel_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for src in sources:
        lbl = _channel_label_for_source(tab, mask_path, src, used_labels)
        csv_path = _csv_path_for_source(pipeline_dir, mask_path=mask_path, foci_source_path=src, prefer_filtered=prefer_filtered)
        if csv_path is None:
            continue
        rows = _read_foci_csv_rows(csv_path)
        if not rows:
            continue
        assigned = _assigned_foci_rows_for_channel(mask_labels, lbl, rows)
        channel_rows.append((lbl, assigned))

    stem = stem_from_mask_path(mask_path)
    analysis_path = analysis_dir_from_base(base_dir_for_mask(mask_path)) / f"{stem}_analysis.xlsx"
    if not analysis_path.exists():
        return 0, [f"  Colocalization skipped for {mask_path.name}: analysis workbook not found ({analysis_path.name}).\n"]

    # Build per pair tables
    settings = ColocSettings(
        coloc_max_distance=float(getattr(tab, "coloc_merge_max_distance").value()),
        coloc_threshold=float(getattr(tab, "coloc_merge_threshold").value()),
        coloc_filter_boundaries=bool(getattr(tab, "coloc_merge_same_cell_only").isChecked()),
    )
    pair_tables: list[tuple[str, str, pd.DataFrame, int]] = []
    if len(channel_rows) >= 2:
        for (la, rows_a), (lb, rows_b) in itertools.combinations(channel_rows, 2):
            foci_rows = []
            foci_rows.extend(rows_a)
            foci_rows.extend(rows_b)
            out = analyze_colocalization_foci_table(foci_rows, settings, foci_channels=(la, lb), frame=0)
            df = (
                pd.DataFrame(out.pair_records)
                if out.pair_records
                else pd.DataFrame({"message": ["No colocalized focus pairs."]})
            )
            count = 0 if ("message" in df.columns and len(df.columns) == 1) else int(len(df))
            pair_tables.append((la, lb, df, count))

    # Always update Per-Cell Data with per-channel foci counts; coloc count is 0 if no pair tables.
    try:
        _update_per_cell_data_counts(analysis_path=analysis_path, channel_rows=channel_rows, pair_tables=pair_tables)
    except Exception:
        pass

    if not pair_tables:
        # No coloc sheets to write, but Per-Cell counts may still have been updated.
        return 0, log_lines

    # Remove existing Colocalization sheets then append
    try:
        wb = load_workbook(analysis_path)
        for name in list(wb.sheetnames):
            if isinstance(name, str) and name.startswith("Colocalization "):
                del wb[name]
        wb.save(analysis_path)
        wb.close()
    except Exception as exc:
        return 0, [f"  Colocalization failed for {mask_path.name}: could not prepare workbook ({exc}).\n"]

    try:
        wb = load_workbook(analysis_path)
        used = set(wb.sheetnames)
        wb.close()
        with pd.ExcelWriter(analysis_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            for la, lb, df, _count in pair_tables:
                sheet_name = _safe_sheet_name(f"Colocalization {la}×{lb}", used)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        for la, lb, _df, count in pair_tables:
            log_lines.append(f"  Colocalization: {mask_path.name} {la} × {lb} -> {count} pair(s) in {analysis_path.name}\n")
    except Exception as exc:
        return 0, [f"  Colocalization failed for {mask_path.name}: could not write sheets ({exc}).\n"]

    return len(pair_tables), log_lines

