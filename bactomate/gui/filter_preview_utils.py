"""Shared helpers for the Analyze Masks filtering preview."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

# SSB-style replicate ids: cl1__ vs cl1_001_ must not be mixed for one mask.
_REPLICATE_TOKEN = re.compile(r"(cl\d+)((?:__)|(?:_\d+_))", re.IGNORECASE)


def preview_view_label(path: Path) -> str:
    """Short label for the filter-preview View dropdown."""
    p = Path(path)
    low = p.name.lower()
    if "_foci" in low:
        return f"Foci {short_foci_stack_label(p)}"
    if "_fi" in low:
        tag = short_foci_stack_label(p)
        if tag and tag != p.stem:
            return f"FI {tag}"
    if "phase" in low or "dic" in low:
        return "Input image"
    return p.name


def discover_channel_stacks_near_mask(mask_path: Path) -> list[Path]:
    """FI / foci TIFFs in sample or segmentation folders (before replicate filtering)."""
    out: list[Path] = []
    seen: set[str] = set()
    for folder in _search_dirs_for_mask(mask_path):
        if not folder.is_dir():
            continue
        for pat in (
            "*fi_foci*.tif",
            "*fi_foci*.tiff",
            "*_foci_*.tif",
            "*_foci_*.tiff",
            "*_FI_*.tif",
            "*_FI_*.tiff",
        ):
            for p in folder.glob(pat):
                key = file_identity_key(p)
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
    return out


def enrich_background_paths_for_mask(
    tab: Any, mask_path: Path, paths: list[Path]
) -> list[Path]:
    """Merge phase, FI, and foci stacks so the View dropdown lists every previewable channel."""
    merged: list[Path] = [Path(p) for p in paths]
    seen: set[str] = {file_identity_key(p) for p in merged}

    def _stage(p: Path) -> None:
        p = Path(p)
        if not p.is_file():
            return
        key = file_identity_key(p)
        if key in seen:
            return
        seen.add(key)
        merged.append(p)

    for p in phase_stack_candidates_for_mask(mask_path):
        _stage(p)

    for p in discover_channel_stacks_near_mask(mask_path):
        _stage(p)

    for resolver_name in (
        "_resolve_foci_paths_for_mask",
        "_resolve_fi_paths_for_mask",
        "_preview_fi_path_candidates",
    ):
        resolver = getattr(tab, resolver_name, None)
        if not callable(resolver):
            continue
        try:
            cands = resolver(mask_path)
        except Exception:
            cands = []
        for p in cands or []:
            _stage(Path(p))

    return filter_paths_for_mask(tab, mask_path, merged)


CELL_FILTER_SETTING_KEYS: tuple[str, ...] = (
    "pixel_size",
    "exclude_border",
    "min_aspect_ratio",
    "max_aspect_ratio",
    "min_eccentricity",
    "min_solidity",
    "min_area_um2",
    "max_area_um2",
    "min_width_um",
    "max_width_um",
    "min_length_um",
    "max_length_um",
)


FOCI_CHANNEL_SETTING_KEYS: tuple[str, ...] = (
    "foci_prob_enabled",
    "foci_prob_thresh",
    "foci_min_int_enabled",
    "foci_min_int",
    "foci_per_cell_ratio_enabled",
    "foci_per_cell_ratio",
    "foci_exclude_outside_cells",
    "foci_exclude_outside_clear",
    "foci_exclude_outside_touching",
    "foci_blob_threshold_frac",
    "foci_blob_max_radius",
    "foci_min_area_px_enabled",
    "foci_min_area_px",
)


def settings_signature_for_keys(settings: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Stable JSON signature for a subset of settings (cache / incremental Apply)."""
    import json

    norm: dict[str, object] = {}
    for k in keys:
        v = (settings or {}).get(k)
        if v is None:
            norm[k] = None
            continue
        if isinstance(v, bool):
            norm[k] = bool(v)
            continue
        try:
            norm[k] = round(float(v), 6)
            continue
        except (TypeError, ValueError):
            pass
        try:
            norm[k] = str(v)
        except Exception:
            norm[k] = None
    try:
        return json.dumps(norm, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(norm)


def cell_filter_settings_signature(settings: dict[str, Any]) -> str:
    return settings_signature_for_keys(settings, CELL_FILTER_SETTING_KEYS)


def foci_channel_settings_signature(channel_settings: dict[str, Any]) -> str:
    return settings_signature_for_keys(channel_settings, FOCI_CHANNEL_SETTING_KEYS)


def format_incremental_apply_plan(plan: dict[str, Any]) -> str:
    """Human-readable summary of what preview Apply will run (cells + foci channels)."""
    lines: list[str] = []
    if plan.get("reuse_cells"):
        lines.append("Cells: reuse previous accept/reject (settings unchanged)")
    else:
        lines.append("Cells: recompute accept/reject from current settings")
    ch_score = [str(c) for c in (plan.get("channels_to_score") or [])]
    prefilled = [str(k) for k in (plan.get("prefilled") or {}).keys()]
    if ch_score:
        lines.append(
            "Foci: rescore "
            + ", ".join(ch_score)
            + " — peaks in rejected cells automatically fail"
        )
    elif prefilled:
        lines.append("Foci: keep cached pass/fail for " + ", ".join(prefilled))
    else:
        lines.append("Foci: nothing to score on this frame")
    return "\n".join(lines)


def default_foci_channel_settings() -> dict[str, Any]:
    return {
        "foci_prob_enabled": False,
        "foci_prob_thresh": 0.5,
        "foci_min_int_enabled": False,
        "foci_min_int": 0.0,
        "foci_per_cell_ratio_enabled": False,
        "foci_per_cell_ratio": 2.0,
        "foci_exclude_outside_cells": False,
        "foci_exclude_outside_clear": False,
        "foci_exclude_outside_touching": False,
        "foci_blob_threshold_frac": 0.9,
        "foci_blob_max_radius": 5,
        "foci_min_area_px_enabled": False,
        "foci_min_area_px": 1,
    }


def foci_channel_key_for_path(foci_source: Path) -> str:
    return short_foci_stack_label(Path(foci_source))


def legacy_foci_settings_slice(settings: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in FOCI_CHANNEL_SETTING_KEYS:
        if k in settings:
            out[k] = settings.get(k)
    return out


def ensure_foci_channels_map(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Ensure ``settings['foci_channels']`` exists; migrate flat legacy foci_* keys once."""
    raw = settings.get("foci_channels")
    channels: dict[str, dict[str, Any]] = dict(raw) if isinstance(raw, dict) else {}
    legacy = legacy_foci_settings_slice(settings)
    if not channels:
        seed = {**default_foci_channel_settings(), **legacy}
        channels["_default"] = seed
    else:
        for ch_data in channels.values():
            if isinstance(ch_data, dict):
                for k, v in default_foci_channel_settings().items():
                    ch_data.setdefault(k, v)
    settings["foci_channels"] = channels
    return channels


def seed_foci_channel_settings(settings: dict[str, Any], foci_sources: list[Path]) -> None:
    """Create per-channel entries for each foci stack (copy legacy defaults for new tags)."""
    channels = ensure_foci_channels_map(settings)
    legacy = legacy_foci_settings_slice(settings)
    template = {**default_foci_channel_settings(), **legacy}
    for fp in foci_sources:
        key = foci_channel_key_for_path(fp)
        if key not in channels:
            channels[key] = dict(template)


def resolve_foci_settings_for_source(
    settings: dict[str, Any], foci_source: Optional[Path]
) -> dict[str, Any]:
    """Merge global (cell) settings with the foci-channel-specific filter block."""
    merged = dict(settings or {})
    channels = ensure_foci_channels_map(merged)
    if foci_source is None:
        ch = channels.get("_default") or default_foci_channel_settings()
    else:
        key = foci_channel_key_for_path(foci_source)
        ch = channels.get(key)
        if not isinstance(ch, dict):
            template = channels.get("_default") or default_foci_channel_settings()
            ch = dict(template)
            channels[key] = ch
    for k in FOCI_CHANNEL_SETTING_KEYS:
        if k in ch:
            merged[k] = ch[k]
    return merged


def sync_filter_settings_from_preview_dialog(tab: Any) -> bool:
    """Copy cell + active foci filter widgets from the open preview into ``tab`` settings."""
    dlg = getattr(tab, "_filter_preview_dialog", None)
    if dlg is None:
        return False
    sync = getattr(dlg, "_pf_ui_to_settings", None)
    if not callable(sync):
        return False
    try:
        sync()
        return True
    except Exception:
        return False


def wait_for_filter_preview_apply_idle(dlg: Any, *, timeout_ms: int = 3_600_000) -> bool:
    """Block until preview Apply worker (and any queued re-apply) finishes."""
    import time

    from PyQt6.QtWidgets import QApplication

    deadline = time.monotonic() + max(0.0, int(timeout_ms)) / 1000.0
    while time.monotonic() < deadline:
        thread = getattr(dlg, "_preview_thread", None)
        running = thread is not None and thread.isRunning()
        pending = bool(getattr(dlg, "_pending_apply", False))
        busy = False
        try:
            busy = bool(dlg.state().extra.get("filter_apply_busy"))
        except Exception:
            pass
        if not running and not pending and not busy:
            return True
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    return False


def request_filter_preview_apply(
    tab: Any,
    *,
    defer_ms: int = 0,
    blocking: bool = False,
) -> bool:
    """
    Run **Apply current filters** on the embedded/open filter preview.

    Used after **Filter File/Folder** so the canvas matches the settings that were
    just written to disk (same worker path as the preview Apply button).

    When ``blocking`` is True, wait until the preview worker finishes (batch finalize).
    """
    dlg = getattr(tab, "_filter_preview_dialog", None)
    if dlg is None:
        return False
    try:
        if not dlg.isVisible():
            return False
    except Exception:
        return False
    apply_fn = getattr(dlg, "_pf_apply_current", None)
    if not callable(apply_fn):
        return False
    sync_filter_settings_from_preview_dialog(tab)

    def _run() -> None:
        try:
            apply_fn()
        except Exception:
            pass

    if bool(blocking):
        _run()
        return wait_for_filter_preview_apply_idle(dlg)

    if int(defer_ms) > 0:
        try:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(int(defer_ms), _run)
        except Exception:
            _run()
    else:
        _run()
    return True


def short_foci_stack_label(path: Path) -> str:
    """Short channel label for UI (e.g. ``GFP`` from ``sample_fi_foci_GFP.tif``)."""
    stem = Path(path).stem
    for pat in (r"_fi_foci_(?P<ch>.+)$", r"_foci_(?P<ch>.+)$", r"_fi_(?P<ch>.+)$"):
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            ch = (m.group("ch") or "").strip("_")
            if ch:
                return ch
    if len(stem) > 28:
        return stem[:25] + "…"
    return stem


def mask_stem_from_path(mask_path: Path) -> str:
    stem = Path(mask_path).stem
    low = stem.lower()
    for suffix in ("_masks", "_mask_filtered"):
        if low.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def replicate_token(stem: str) -> str:
    """Return normalized replicate delimiter after clN (``__`` or ``_001_``)."""
    s = (stem or "").lower()
    m = _REPLICATE_TOKEN.search(s)
    if m:
        return m.group(2)
    if re.search(r"cl\d+__", s):
        return "__"
    m3 = re.search(r"cl\d+(_\d+_)", s)
    if m3:
        return m3.group(1)
    return ""


def same_replicate(mask_path: Path, file_path: Path) -> bool:
    """True when mask and file belong to the same sample replicate, not a sibling clone."""
    mt = replicate_token(mask_stem_from_path(mask_path))
    ft = replicate_token(file_path.stem)
    if mt and ft:
        return mt == ft
    try:
        from bactomate.gui.artifact_paths import sample_identity_stem

        return sample_identity_stem(mask_stem_from_path(mask_path)).lower() == sample_identity_stem(
            file_path.stem
        ).lower()
    except Exception:
        pass
    # No replicate token — require shared prefix up to channel suffix.
    ms = mask_stem_from_path(mask_path).lower()
    fs = file_path.stem.lower()
    if fs == ms:
        return True
    if fs.startswith(ms + "_") or fs.startswith(ms + "__"):
        return True
    if ms.startswith(fs + "_") or ms.startswith(fs + "__"):
        return True
    return False


def file_identity_key(path: Path) -> str:
    """Dedupe key for the same on-disk file (handles .tif vs .TIF on case-insensitive volumes)."""
    p = Path(path)
    try:
        st = p.resolve().stat()
        return f"{int(st.st_dev)}:{int(st.st_ino)}"
    except OSError:
        pass
    try:
        return str(p.resolve()).casefold()
    except OSError:
        return str(p).casefold()


def _search_dirs_for_mask(mask_path: Path) -> list[Path]:
    """Sample / segmentation folders to look for paired phase and channel stacks."""
    parent = Path(mask_path).parent
    dirs: list[Path] = []
    try:
        from bactomate.gui.analysis_paths import base_dir_for_mask

        sample_dir = base_dir_for_mask(mask_path)
        if sample_dir.is_dir():
            dirs.append(sample_dir)
    except Exception:
        pass
    if parent.name.startswith("_segmentation"):
        dirs.extend([parent.parent, parent])
    else:
        dirs.append(parent)
        if parent.parent != parent:
            dirs.append(parent.parent)
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def phase_stack_candidates_for_mask(mask_path: Path) -> list[Path]:
    """
    Phase (or DIC) stack paired with this mask: ``{mask_stem}.tif`` next to the mask.

    Masks named ``sample_Phase60X_masks.tif`` share a stem with ``sample_Phase60X.tif``;
    those are not returned by FI/foci discovery and must be resolved explicitly.
    """
    ms = mask_stem_from_path(mask_path)
    out: list[Path] = []
    seen: set[str] = set()
    for folder in _search_dirs_for_mask(mask_path):
        if not folder.is_dir():
            continue
        for ext in (".tif", ".tiff"):
            p = folder / f"{ms}{ext}"
            if not p.is_file():
                continue
            key = file_identity_key(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def collect_background_paths_for_mask(tab: Any, mask_path: Path) -> list[Path]:
    """Phase + FI/foci stacks for this mask only (same replicate, not sibling clones)."""
    seen: set[str] = set()
    raw: list[Path] = []

    def _add(p: Path) -> None:
        if not p.is_file():
            return
        key = file_identity_key(p)
        if key in seen:
            return
        seen.add(key)
        raw.append(p)

    for p in phase_stack_candidates_for_mask(mask_path):
        _add(p)
    try:
        for p in tab._preview_fi_path_candidates(mask_path):
            _add(Path(p))
    except Exception:
        pass
    return filter_paths_for_mask(tab, mask_path, raw)


def _same_acquisition_as_mask(tab: Any, mask_path: Path, file_path: Path) -> bool:
    """True when ``file_path`` belongs to the same sample/replicate as ``mask_path``."""
    _ = tab  # kept for API compatibility with older call sites
    return same_replicate(mask_path, file_path)


def path_matches_mask_sample(tab: Any, mask_path: Path, file_path: Path) -> bool:
    """Whether a TIFF belongs to this mask's sample (phase, FI, or foci channel)."""
    if not file_path.is_file():
        return False
    if not _same_acquisition_as_mask(tab, mask_path, file_path):
        return False
    fs = file_path.stem
    ms = mask_stem_from_path(mask_path)
    stems: list[str] = [ms]
    try:
        for s in tab._candidate_match_stems(ms):
            if s and s not in stems:
                stems.append(s)
    except Exception:
        pass
    for stem in stems:
        if fs.lower() == stem.lower():
            return True
        try:
            if tab._matches_foci_name(fs, stem) or tab._matches_fi_name(fs, stem):
                return True
            tag = tab._extract_channel_tag_for_mask_stem(fs, stem)
            if tag:
                return True
        except Exception:
            continue
    return False


def filter_paths_for_mask(tab: Any, mask_path: Path, paths: list[Path]) -> list[Path]:
    """Keep only stacks that match the selected mask file (not sibling replicates in folder)."""
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = file_identity_key(Path(p))
        if key in seen:
            continue
        if not path_matches_mask_sample(tab, mask_path, Path(p)):
            continue
        seen.add(key)
        out.append(Path(p))
    out.sort(key=lambda x: x.name.lower())
    return out


def default_phase_path_for_mask(
    tab: Any, mask_path: Path, bg_paths: list[Path]
) -> Optional[Path]:
    """Best default background: phase stack with the same stem as the mask file."""
    ms = mask_stem_from_path(mask_path)
    for p in bg_paths:
        if p.stem.lower() == ms.lower():
            return p
    for p in bg_paths:
        if "_foci" in p.name.lower():
            continue
        if path_matches_mask_sample(tab, mask_path, p):
            return p
    return None


def default_background_index(tab: Any, mask_path: Path, bg_paths: list[Path]) -> int:
    """Index into ``bg_paths`` for the matching phase image, else first non-foci row."""
    phase = default_phase_path_for_mask(tab, mask_path, bg_paths)
    if phase is not None:
        target = file_identity_key(phase)
        for i, p in enumerate(bg_paths):
            if file_identity_key(p) == target:
                return int(i)
    for i, p in enumerate(bg_paths):
        if "_foci" not in p.name.lower():
            return int(i)
    return 0


def pick_mask_path_for_preview(
    tab: Any,
    path_text: str,
    *,
    force_dialog: bool = False,
    default_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolve which mask TIFF to open in the filter preview.

    When the user selected a folder with multiple mask files, prompt for a choice
    instead of always using the first file in the list. ``force_dialog`` keeps the
    choice explicit even when a previous/default mask is known.
    """
    from bactomate.gui.artifact_paths import is_cell_mask_tiff

    raw = (path_text or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p if is_cell_mask_tiff(p) else None
    masks: list[Path] = []
    try:
        from bactomate.gui.artifact_paths import dedupe_mask_paths_prefer_filtered

        masks = dedupe_mask_paths_prefer_filtered(
            [
                Path(m)
                for m in (tab._get_mask_files_for_feedback() or [])
                if is_cell_mask_tiff(Path(m))
            ]
        )
    except Exception:
        masks = []
    if not masks:
        return None
    if len(masks) == 1 and not force_dialog:
        return masks[0]
    from bactomate.gui.widgets import show_list_pick_dialog

    return show_list_pick_dialog(
        tab,
        masks,
        title="Select mask",
        message="Multiple mask files were found. Choose which mask to preview:",
        default_path=default_path,
    )


def _active_threshold(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _morphology_fail_reasons(prop: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    try:
        aspect = float(prop.get("aspect_ratio", 0.0) or 0.0)
        ecc = float(prop.get("eccentricity", 0.0) or 0.0)
        sol = float(prop.get("solidity", 0.0) or 0.0)
    except (TypeError, ValueError):
        return ["Morphology filter failed"]
    min_ar = float(settings.get("min_aspect_ratio", 1.5) or 1.5)
    max_ar = float(settings.get("max_aspect_ratio", 10.0) or 10.0)
    min_ecc = float(settings.get("min_eccentricity", 0.5) or 0.5)
    min_sol = float(settings.get("min_solidity", 0.7) or 0.7)
    if aspect < min_ar:
        reasons.append(f"Aspect ratio {aspect:.2f} < min {min_ar:.2f}")
    elif aspect > max_ar:
        reasons.append(f"Aspect ratio {aspect:.2f} > max {max_ar:.2f}")
    if ecc < min_ecc:
        reasons.append(f"Eccentricity {ecc:.2f} < min {min_ecc:.2f}")
    if sol < min_sol:
        reasons.append(f"Solidity {sol:.2f} < min {min_sol:.2f}")
    return reasons or ["Morphology filter failed"]


def _size_fail_reasons(prop: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    checks = (
        ("area_um2", "min_area_um2", "max_area_um2", "Area (µm²)"),
        ("width_um", "min_width_um", "max_width_um", "Width (µm)"),
        ("length_um", "min_length_um", "max_length_um", "Length (µm)"),
    )
    for (metric_key, min_key, max_key, label) in checks:
        try:
            value = float(prop.get(metric_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        min_v = _active_threshold(settings.get(min_key))
        max_v = _active_threshold(settings.get(max_key))
        if min_v is not None and value < min_v:
            reasons.append(f"{label} {value:.2f} < min {min_v:.2f}")
        if max_v is not None and value > max_v:
            reasons.append(f"{label} {value:.2f} > max {max_v:.2f}")
    return reasons or ["Size filter failed"]


def evaluate_cell_filter_details(
    mask_2d: Any,
    settings: dict[str, Any],
    *,
    accepted: set[int] | None = None,
    rejected: set[int] | None = None,
    phase_plane: Any | None = None,
) -> list[dict[str, Any]]:
    """Per-cell metrics and fail reasons for filter-preview hover tooltips."""
    import numpy as np
    from skimage.measure import regionprops

    from bactomate.scripts.analyze_cell_areas import (
        calculate_cell_properties,
        filter_by_size,
        filter_rod_shaped_cells,
    )

    mask = np.asarray(mask_2d, dtype=np.int32)
    if mask.ndim != 2:
        return []
    h, w = mask.shape[:2]
    pixel_size = float(settings.get("pixel_size", 0.108) or 0.108)
    exclude_border = bool(settings.get("exclude_border", False))
    accepted_set = {int(x) for x in (accepted or set()) if int(x) > 0}
    rejected_set = {int(x) for x in (rejected or set()) if int(x) > 0}

    centroids: dict[int, tuple[float, float]] = {}
    border_labels: set[int] = set()
    for region in regionprops(mask):
        label = int(region.label)
        if label <= 0:
            continue
        cy, cx = region.centroid
        centroids[label] = (float(cy), float(cx))
        min_row, min_col, max_row, max_col = region.bbox
        if min_row <= 0 or min_col <= 0 or max_row >= h or max_col >= w:
            border_labels.add(label)

    props_list = calculate_cell_properties(
        mask,
        pixel_size=pixel_size,
        exclude_border=exclude_border,
    )
    props_by_label = {int(p["label"]): dict(p) for p in props_list}
    rod_ok = {
        int(p["label"])
        for p in filter_rod_shaped_cells(
            props_list,
            min_aspect_ratio=float(settings.get("min_aspect_ratio", 1.5) or 1.5),
            max_aspect_ratio=float(settings.get("max_aspect_ratio", 10.0) or 10.0),
            min_eccentricity=float(settings.get("min_eccentricity", 0.5) or 0.5),
            min_solidity=float(settings.get("min_solidity", 0.7) or 0.7),
        )
    }
    size_ok = {
        int(p["label"])
        for p in filter_by_size(
            [p for p in props_list if int(p["label"]) in rod_ok],
            min_area_um2=settings.get("min_area_um2"),
            max_area_um2=settings.get("max_area_um2"),
            min_width_um=settings.get("min_width_um"),
            max_width_um=settings.get("max_width_um"),
            min_length_um=settings.get("min_length_um"),
            max_length_um=settings.get("max_length_um"),
        )
    }
    focus_scores: dict[int, float] = {}
    size_accepted = set(size_ok)
    if bool(settings.get("focus_filter_enabled")) and phase_plane is not None and size_accepted:
        try:
            from bactomate.modules.cell_focus_score import (
                filter_labels_by_focus,
                score_cells_in_frame,
            )

            scores = score_cells_in_frame(
                np.asarray(phase_plane, dtype=np.float64),
                mask,
                size_accepted,
            )
            focus_scores = {int(k): float(v) for k, v in dict(scores or {}).items()}
            frac = float(settings.get("focus_min_fraction_of_median", 0.5) or 0.5)
            size_accepted = set(
                filter_labels_by_focus(
                    scores,
                    min_fraction_of_median=frac if frac > 0 else None,
                )
            )
        except Exception:
            focus_scores = {}

    details: list[dict[str, Any]] = []
    for label in sorted(centroids):
        cy, cx = centroids[label]
        prop = props_by_label.get(label)
        if accepted_set or rejected_set:
            status = "pass" if label in accepted_set else "fail"
        else:
            status = "pass" if label in size_accepted else "fail"
        reasons: list[str] = []
        if status == "fail":
            if exclude_border and label in border_labels:
                reasons.append("Touches image border")
            elif prop is None:
                reasons.append("Excluded from analysis")
            elif label not in rod_ok:
                reasons.extend(_morphology_fail_reasons(prop, settings))
            elif label not in size_ok:
                reasons.extend(_size_fail_reasons(prop, settings))
            elif bool(settings.get("focus_filter_enabled")) and label not in size_accepted:
                score = focus_scores.get(label)
                frac = float(settings.get("focus_min_fraction_of_median", 0.5) or 0.5)
                if score is not None:
                    reasons.append(
                        f"Focus score {score:.3f} below {frac:.2f}× frame median"
                    )
                else:
                    reasons.append("Out of focus")
        area_px = int(prop.get("area_pixels", 0) or 0) if prop else 0
        details.append(
            {
                "label": int(label),
                "status": status,
                "y": cy,
                "x": cx,
                "area_px": area_px,
                "fail_reasons": reasons,
                "metrics": prop or {},
                "focus_score": focus_scores.get(label),
            }
        )
    return details


def format_cell_filter_tooltip(cell: dict[str, Any]) -> str:
    """Multi-line tooltip for hover on a segmented cell in the filter preview."""
    label = int(cell.get("label", 0) or 0)
    status = str(cell.get("status", "pass")).lower()
    is_fail = status == "fail"
    title = f"Cell {label} — FAIL" if is_fail else f"Cell {label} — pass"
    metrics = dict(cell.get("metrics") or {})
    lines = [title, f"Position: y={float(cell.get('y', 0.0)):.1f}, x={float(cell.get('x', 0.0)):.1f}"]
    metric_specs = (
        ("Area (µm²)", "area_um2"),
        ("Width (µm)", "width_um"),
        ("Length (µm)", "length_um"),
        ("Aspect", "aspect_ratio"),
        ("Eccentricity", "eccentricity"),
        ("Solidity", "solidity"),
    )
    for label_txt, key in metric_specs:
        if key not in metrics:
            continue
        try:
            lines.append(f"{label_txt}: {float(metrics[key]):.2f}")
        except (TypeError, ValueError):
            continue
    focus_score = cell.get("focus_score")
    if focus_score is not None:
        try:
            lines.append(f"Focus score: {float(focus_score):.3f}")
        except (TypeError, ValueError):
            pass
    reasons = cell.get("fail_reasons")
    if is_fail and isinstance(reasons, list) and reasons:
        parts = [f"<b>{title}</b>"] + [
            line.replace("&", "&amp;").replace("<", "&lt;") for line in lines[1:]
        ]
        parts.append("<br/><b>Failed filters:</b>")
        for reason in reasons:
            text = str(reason).replace("&", "&amp;").replace("<", "&lt;")
            parts.append(f'<span style="color:#ff6666">• {text}</span>')
        return "<br/>".join(parts)
    return "\n".join(lines)


def build_cell_hover_data(
    cell_details: list[dict[str, Any]],
    mask_shape: tuple[int, int],
    display_shape: tuple[int, int],
) -> list[tuple[float, float, str, float]]:
    """Build ImageCanvas hover entries for segmented cells in scaled preview coordinates."""
    import numpy as np

    h0, w0 = int(mask_shape[0]), int(mask_shape[1])
    h1, w1 = int(display_shape[0]), int(display_shape[1])
    sx = w1 / w0 if w0 else 1.0
    sy = h1 / h0 if h0 else 1.0
    scale = max(sx, sy, 1e-06)
    out: list[tuple[float, float, str, float]] = []
    for cell in cell_details or []:
        try:
            cy = float(cell.get("y", 0.0) or 0.0)
            cx = float(cell.get("x", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        area_px = max(1.0, float(cell.get("area_px", 4) or 4))
        radius = max(8.0, min(42.0, (area_px / 3.14159265) ** 0.5 * scale * 0.45))
        out.append((cx * sx, cy * sy, format_cell_filter_tooltip(cell), float(radius)))
    return out
