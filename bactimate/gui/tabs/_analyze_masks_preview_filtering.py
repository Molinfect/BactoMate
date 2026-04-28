from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.analyze_masks_preview_worker import AnalyzeMasksPreviewWorker
from bactimate.gui.image_utils import scale_for_preview, to_qpixmap
from bactimate.gui.log_config import write_gui_log_line
from bactimate.gui.utils import safe_imread
from bactimate.gui.preview.preview_window import PreviewDialogBase, WorkerGuard, RenderResult


def is_preview_cache_hit(last: Optional[dict], frame_idx: int, source_key) -> bool:
    """Decide whether the cached filtering preview matches the displayed frame/source.

    The cached entry (``dlg._last_preview``) reflects the *last applied* filter
    result. We deliberately do NOT compare the cached settings signature to the
    user's current UI / tab settings: per the spec, applying filters is a
    user-driven event (Re-apply), and unapplied UI tweaks must
    not invalidate the cached overlay. The cache is invalidated by
    ``apply_current()`` clearing ``_last_preview`` before starting a new worker;
    the worker then refills the cache once it completes.

    A cached entry is considered valid only when it carries actual filtering
    results (``mask_2d`` present) and matches the currently displayed frame
    and source.
    """
    if not last:
        return False
    if "mask_2d" not in last:
        return False
    try:
        if int(last.get("frame_idx", -1)) != int(frame_idx):
            return False
    except Exception:
        return False
    try:
        cached_src = tuple(last.get("source_key", ("none", -1)))
        if cached_src != tuple(source_key):
            return False
    except Exception:
        return False
    return True


def _safe_release_preview_worker(owner, attr: str = "_preview_worker", wait_ms: int = 10_000) -> None:
    w = getattr(owner, attr, None)
    thread = getattr(owner, "_preview_thread", None)
    try:
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(wait_ms):
                # Never drop/delete a QThread wrapper while Qt still reports it running.
                return
    except Exception:
        return
    try:
        if w is not None:
            w.deleteLater()
    except Exception:
        pass
    try:
        if thread is not None:
            thread.deleteLater()
    except Exception:
        pass
    try:
        setattr(owner, attr, None)
    except Exception:
        pass
    try:
        owner._preview_thread = None
    except Exception:
        pass


def show_filter_preview(tab, path_text: str) -> None:
    """
    Section 2 preview (Select Filter): show accepted vs rejected cells for one mask file.

    This preview is intentionally filtering-only (no re-detection):
    - foci markers come from existing foci CSVs (if present)
    """
    raw = (path_text or "").strip()
    if not raw:
        return
    p = Path(raw)
    if not p.exists():
        QMessageBox.warning(tab, "Not Found", "The selected file or folder does not exist.")
        return

    mask_path: Optional[Path] = None
    if p.is_file():
        mask_path = p
    else:
        masks = []
        try:
            masks = tab._get_mask_files_for_feedback()
        except Exception:
            masks = []
        if masks:
            mask_path = masks[0]
    if mask_path is None or not mask_path.exists():
        QMessageBox.warning(tab, "No Masks Found", "No mask files were found in the selected path.")
        return

    try:
        masks = np.asarray(safe_imread(str(mask_path)))
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
    except Exception as e:
        QMessageBox.warning(tab, "Load Error", f"Could not load mask:\n{e}")
        return

    seg_dir = mask_path.parent
    if not str(seg_dir.name).startswith("_segmentation"):
        cand = mask_path.parent / "_segmentation"
        if cand.exists() and cand.is_dir():
            seg_dir = cand

    num_frames = int(masks.shape[0])

    # Preview dialog top controls: parameter controls live here (the main tab is minimal).
    settings_host = QWidget()
    settings_layout = QHBoxLayout(settings_host)
    settings_layout.setContentsMargins(0, 0, 0, 0)
    settings_layout.setSpacing(10)

    fs0 = tab.get_filter_settings() if hasattr(tab, "get_filter_settings") else {}

    def _mk_spin(*, decimals: int, mn: float, mx: float, step: float, value: float, width: int = 90) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setDecimals(int(decimals))
        w.setRange(float(mn), float(mx))
        w.setSingleStep(float(step))
        w.setValue(float(value))
        w.setFixedWidth(int(width))
        w.setKeyboardTracking(False)
        return w

    def _mk_optional_um(*, value: float | None, max_v: float, label_suffix: str) -> tuple[QDoubleSpinBox, QLabel]:
        # 0 means disabled (stored as None)
        w = _mk_spin(decimals=2, mn=0.0, mx=float(max_v), step=0.1, value=float(value or 0.0), width=90)
        note = QLabel(label_suffix)
        note.setStyleSheet("color: #9e9e9e;")
        return w, note

    def _row_into(target_layout: QVBoxLayout, lbl: str, w: QWidget, extra: Optional[QWidget] = None) -> None:
        r = QHBoxLayout()
        r.addWidget(QLabel(lbl))
        r.addStretch()
        r.addWidget(w)
        if extra is not None:
            r.addWidget(extra)
        target_layout.addLayout(r)

    # ---- Cell filter parameters: morphology --------------------------
    cell_morph_group = QGroupBox("Cell filter — morphology")
    cell_morph_layout = QVBoxLayout(cell_morph_group)
    cell_morph_layout.setSpacing(6)

    w_pixel = _mk_spin(decimals=3, mn=0.001, mx=10.0, step=0.001, value=float(fs0.get("pixel_size", 0.108) or 0.108))
    w_min_ar = _mk_spin(decimals=2, mn=1.0, mx=20.0, step=0.1, value=float(fs0.get("min_aspect_ratio", 1.5) or 1.5))
    w_max_ar = _mk_spin(decimals=2, mn=1.0, mx=50.0, step=0.5, value=float(fs0.get("max_aspect_ratio", 10.0) or 10.0))
    w_min_ecc = _mk_spin(decimals=2, mn=0.0, mx=1.0, step=0.05, value=float(fs0.get("min_eccentricity", 0.5) or 0.5))
    w_min_sol = _mk_spin(decimals=2, mn=0.0, mx=1.0, step=0.05, value=float(fs0.get("min_solidity", 0.7) or 0.7))
    _row_into(cell_morph_layout, "Pixel size (µm):", w_pixel)
    _row_into(cell_morph_layout, "Min aspect ratio:", w_min_ar)
    _row_into(cell_morph_layout, "Max aspect ratio:", w_max_ar)
    _row_into(cell_morph_layout, "Min eccentricity:", w_min_ecc)
    _row_into(cell_morph_layout, "Min solidity:", w_min_sol)
    cell_morph_layout.addStretch()

    # ---- Cell filter parameters: geometry ----------------------------
    cell_geom_group = QGroupBox("Cell filter — geometry")
    cell_geom_layout = QVBoxLayout(cell_geom_group)
    cell_geom_layout.setSpacing(6)

    w_min_area, n_min_area = _mk_optional_um(value=fs0.get("min_area_um2"), max_v=10000.0, label_suffix="(0 = off)")
    w_max_area, n_max_area = _mk_optional_um(value=fs0.get("max_area_um2"), max_v=10000.0, label_suffix="(0 = off)")
    w_min_w, n_min_w = _mk_optional_um(value=fs0.get("min_width_um"), max_v=100.0, label_suffix="(0 = off)")
    w_max_w, n_max_w = _mk_optional_um(value=fs0.get("max_width_um"), max_v=100.0, label_suffix="(0 = off)")
    w_min_len, n_min_len = _mk_optional_um(value=fs0.get("min_length_um"), max_v=500.0, label_suffix="(0 = off)")
    _row_into(cell_geom_layout, "Min area (µm²):", w_min_area, n_min_area)
    _row_into(cell_geom_layout, "Max area (µm²):", w_max_area, n_max_area)
    _row_into(cell_geom_layout, "Min width (µm):", w_min_w, n_min_w)
    _row_into(cell_geom_layout, "Max width (µm):", w_max_w, n_max_w)
    _row_into(cell_geom_layout, "Min length (µm):", w_min_len, n_min_len)

    w_excl_border = QCheckBox("Exclude masks at borders")
    w_excl_border.setChecked(bool(fs0.get("exclude_border", False)))
    cell_geom_layout.addWidget(w_excl_border)
    cell_geom_layout.addStretch()

    # ---- Foci filter parameters --------------------------------------
    foci_group = QGroupBox("Foci filter parameters")
    foci_layout = QVBoxLayout(foci_group)
    foci_layout.setSpacing(6)

    r1 = QHBoxLayout()
    w_prob_en = QCheckBox("Filter by probability ≥")
    w_prob_en.setChecked(bool(fs0.get("foci_prob_enabled", False)))
    w_prob_thr = _mk_spin(decimals=3, mn=0.0, mx=1.0, step=0.05, value=float(fs0.get("foci_prob_thresh", 0.5) or 0.5))
    r1.addWidget(w_prob_en)
    r1.addWidget(w_prob_thr)
    r1.addStretch()
    foci_layout.addLayout(r1)

    r2 = QHBoxLayout()
    w_minint_en = QCheckBox("Min raw intensity ≥")
    w_minint_en.setChecked(bool(fs0.get("foci_min_int_enabled", False)))
    w_minint = _mk_spin(decimals=0, mn=0.0, mx=100000.0, step=100.0, value=float(fs0.get("foci_min_int", 0.0) or 0.0), width=110)
    r2.addWidget(w_minint_en)
    r2.addWidget(w_minint)
    r2.addStretch()
    foci_layout.addLayout(r2)

    r3 = QHBoxLayout()
    w_ratio_en = QCheckBox("Per-cell ratio ≥")
    w_ratio_en.setChecked(bool(fs0.get("foci_per_cell_ratio_enabled", False)))
    w_ratio = _mk_spin(decimals=2, mn=0.5, mx=100.0, step=0.1, value=float(fs0.get("foci_per_cell_ratio", 2.0) or 2.0))
    r3.addWidget(w_ratio_en)
    r3.addWidget(w_ratio)
    r3.addWidget(QLabel("(focus mean / rest-of-cell mean)"))
    r3.addStretch()
    foci_layout.addLayout(r3)
    foci_layout.addStretch()

    settings_layout.addWidget(cell_morph_group, 1)
    settings_layout.addWidget(cell_geom_group, 1)
    settings_layout.addWidget(foci_group, 1)

    bottom = QWidget()
    nav = QHBoxLayout(bottom)
    nav.setContentsMargins(0, 0, 0, 0)

    # Legacy widgets retained for minimal-risk integration: existing helper closures read from these.
    image_combo = QComboBox()
    frame_spin = QSpinBox()
    frame_spin.setRange(1, max(1, num_frames))
    frame_spin.setValue(1)
    try:
        image_combo.setVisible(False)
        frame_spin.setVisible(False)
    except Exception:
        pass
    nav.addWidget(image_combo)
    nav.addWidget(frame_spin)

    from bactimate.gui.preview.navigation_controls import (
        PreviewNavSpec,
        PreviewNavState,
        PreviewSource,
        PreviewNavigationControls,
    )

    nav_ctrl = PreviewNavigationControls(
        spec=PreviewNavSpec(
            frame_count=int(max(1, num_frames)),
            image_sources=[],
            show_image=True,
            show_channel=False,
            show_z=False,
            debounce_ms=400,
        )
    )
    nav.addWidget(nav_ctrl)
    nav.addStretch()

    dlg = None  # forward reference for closures

    # ---- background image discovery + cache --------------------------
    from bactimate.modules.image_stack_utils import coerce_stack_to_tyx

    def _list_background_paths() -> list[Path]:
        """
        Prefer original FI/foci stacks that correspond to this mask (often produced by File Conversion),
        falling back to TIFFs in the segmentation folder if needed.
        """
        out: list[Path] = []
        seen: set[str] = set()

        # 1) Preferred: candidates mapped from mask -> FI/foci (can live in Tif_* or parent directories).
        try:
            cand = tab._preview_fi_path_candidates(mask_path)
        except Exception:
            cand = []
        for fp in cand or []:
            try:
                if not fp.exists() or not fp.is_file():
                    continue
                if fp.suffix.lower() not in (".tif", ".tiff"):
                    continue
                rp = str(fp.resolve())
            except Exception:
                rp = str(fp)
            if rp in seen:
                continue
            seen.add(rp)
            out.append(Path(fp))

        # 1b) Also include other converted channel TIFFs (e.g. phase) next to the FI candidates.
        # `_preview_fi_path_candidates` only returns *_fi* and *_foci* by design; filtering preview
        # should allow non-FI backgrounds too.
        if out:
            try:
                stems = tab._candidate_match_stems(mask_path.stem.replace("_masks", ""))
            except Exception:
                stems = []
            try:
                search_dirs = {Path(p).parent for p in out}
            except Exception:
                search_dirs = set()
            for d0 in list(search_dirs):
                if not d0.exists() or not d0.is_dir():
                    continue
                for f in d0.iterdir():
                    if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                        continue
                    name = f.stem.lower()
                    if "_analysis" in name or "_filter_visualization" in name or "_masks" in name or "_mask_filtered" in name:
                        continue
                    # Keep anything that matches the base stem, excluding explicit mask artifacts.
                    if stems and not any(name == s.lower() or name.startswith(s.lower() + "_") for s in stems):
                        continue
                    try:
                        rp = str(f.resolve())
                    except Exception:
                        rp = str(f)
                    if rp in seen:
                        continue
                    seen.add(rp)
                    out.append(f)

        # 2) Fallback: scan segmentation folder for any non-mask TIFFs.
        if not out:
            folder = seg_dir
            for pat in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
                for fp in folder.glob(pat):
                    name = fp.name.lower()
                    if "_analysis" in name or "_filter_visualization" in name:
                        continue
                    if name.endswith(("_masks.tif", "_masks.tiff", "_mask_filtered.tif", "_mask_filtered.tiff")):
                        continue
                    if "_masks" in name or "_mask_filtered" in name:
                        continue
                    try:
                        rp = str(fp.resolve())
                    except Exception:
                        rp = str(fp)
                    if rp in seen:
                        continue
                    seen.add(rp)
                    out.append(fp)

        out.sort(key=lambda x: x.name.lower())
        return out

    bg_paths = _list_background_paths()
    foci_sources = [Path(fp) for fp in bg_paths if "_foci" in fp.name.lower()]
    sources: list[PreviewSource] = []
    if not bg_paths:
        image_combo.addItem("Mask only", ("none", -1))
        sources.append(PreviewSource(key=("none", -1), label="Mask only", kind="image", resets_lut=True))
    for i, fp in enumerate(bg_paths):
        image_combo.addItem(fp.name, ("file", i))
        sources.append(PreviewSource(key=("file", i), label=str(fp.name), kind="image", resets_lut=True))
    default_idx = 0
    if bg_paths:
        for i, fp in enumerate(bg_paths):
            if "_foci" not in fp.name.lower():
                default_idx = i
                break
        image_combo.setCurrentIndex(default_idx)

    # Apply sources to shared navigation and sync initial selection.
    nav_ctrl.set_spec(
        PreviewNavSpec(
            frame_count=int(max(1, num_frames)),
            image_sources=sources,
            show_image=True,
            show_channel=False,
            show_z=False,
            show_cell_overlay_toggle=True,
            show_foci_overlay_toggle=True,
            cell_overlay_default=True,
            foci_overlay_available=bool(foci_sources),
            foci_overlay_default=bool(foci_sources),
            debounce_ms=400,
        ),
        emit_init=False,
    )
    try:
        nav_ctrl.set_state(
            PreviewNavState(source_key=image_combo.currentData(), frame_idx=int(frame_spin.value()) - 1),
            emit=False,
        )
    except Exception:
        pass

    def _selected_source() -> tuple[str, int]:
        sel = image_combo.currentData()
        if not isinstance(sel, tuple) or len(sel) < 2:
            return ("none", -1)
        try:
            return str(sel[0]), int(sel[1])
        except Exception:
            return ("none", -1)

    def _image_path_for_source_key(source_key) -> Optional[Path]:
        try:
            kind, idx = str(source_key[0]), int(source_key[1])
        except Exception:
            return None
        if kind != "file" or idx < 0:
            return None
        if idx >= len(bg_paths):
            return None
        return Path(bg_paths[idx])

    def _selected_image_path() -> Optional[Path]:
        return _image_path_for_source_key(_selected_source())

    def _foci_source_for_source_key(source_key) -> Optional[Path]:
        img_path = _image_path_for_source_key(source_key)
        if img_path is None:
            if len(foci_sources) == 1:
                return Path(foci_sources[0])
            return None
        img_name = img_path.name.lower()
        if "_foci" in img_name:
            for fp in foci_sources:
                try:
                    if Path(fp).resolve() == img_path.resolve():
                        return Path(fp)
                except Exception:
                    if Path(fp) == img_path:
                        return Path(fp)
        try:
            mapped = tab._foci_path_for_fi_path(mask_path, img_path, [Path(x) for x in foci_sources])
            if mapped is not None:
                return Path(mapped)
        except Exception:
            pass
        if len(foci_sources) == 1:
            return Path(foci_sources[0])
        return None

    def _foci_source_for_selected_image() -> Optional[Path]:
        return _foci_source_for_source_key(_selected_source())

    from bactimate.gui.artifact_paths import foci_csv_path, raw_foci_csv_path

    def _csv_path_for_foci_source(src: Optional[Path]) -> Optional[Path]:
        if src is None:
            return None
        try:
            csv_new = foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
            csv_legacy = raw_foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
            if csv_new.exists():
                return csv_new
            if csv_legacy.exists():
                return csv_legacy
            return None
        except Exception:
            return None

    def _rows_cache_key(p: Optional[Path]) -> str:
        if p is None:
            return "__none__"
        try:
            return str(p.resolve())
        except Exception:
            return str(p)

    def _load_rows_for_source(src: Optional[Path]) -> list[dict]:
        key = _rows_cache_key(src)
        cache = getattr(dlg, "_foci_rows_cache", None) if dlg is not None else None
        if isinstance(cache, dict) and key in cache:
            return list(cache.get(key) or [])
        csv_path = _csv_path_for_foci_source(src)
        rows: list[dict] = []
        if csv_path is not None and csv_path.exists():
            try:
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append(r)
            except Exception:
                rows = []
        if isinstance(cache, dict):
            cache[key] = list(rows)
        return rows

    foci_overlay_available = any(_csv_path_for_foci_source(src) is not None for src in foci_sources)
    nav_ctrl.set_spec(
        PreviewNavSpec(
            frame_count=int(max(1, num_frames)),
            image_sources=sources,
            show_image=True,
            show_channel=False,
            show_z=False,
            show_cell_overlay_toggle=True,
            show_foci_overlay_toggle=True,
            cell_overlay_default=True,
            foci_overlay_available=bool(foci_overlay_available),
            foci_overlay_default=bool(foci_overlay_available),
            debounce_ms=400,
        ),
        emit_init=False,
    )
    try:
        nav_ctrl.set_state(
            PreviewNavState(source_key=image_combo.currentData(), frame_idx=int(frame_spin.value()) - 1),
            emit=False,
        )
    except Exception:
        pass

    def _rows_for_frame(rows: list[dict], frame_idx: int) -> list[dict]:
        out: list[dict] = []
        for r in rows or []:
            try:
                fr = int(float(r.get("frame", 0) or 0)) if "frame" in r else 0
            except Exception:
                fr = 0
            if fr == int(frame_idx):
                out.append(r)
        return out

    def _apply_lut_to_bg(bg_2d: Optional[np.ndarray], vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
        if bg_2d is None:
            return np.zeros(masks.shape[1:3], dtype=np.uint8)
        bg_f = np.asarray(bg_2d, dtype=np.float64)
        if bg_f.ndim != 2:
            bg_f = np.asarray(bg_f.squeeze())
            if bg_f.ndim != 2:
                return np.zeros(masks.shape[1:3], dtype=np.uint8)
        mn = float(np.nanmin(bg_f)) if bg_f.size else 0.0
        mx = float(np.nanmax(bg_f)) if bg_f.size else mn + 1.0
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = mn
            vmax = mx if mx > mn else mn + 1.0
        vrange = max(float(vmax) - float(vmin), 1e-6)
        return ((bg_f - float(vmin)) / vrange * 255.0).clip(0, 255).astype(np.uint8)

    def _get_bg_frame(frame_idx: int, source_key=None) -> Optional[np.ndarray]:
        sel = source_key if source_key is not None else image_combo.currentData()
        if not isinstance(sel, tuple) or len(sel) < 2:
            return None
        kind, idx = sel[0], int(sel[1])
        if kind != "file" or idx < 0:
            return None
        paths = getattr(dlg, "_bg_paths", []) or []
        if idx >= len(paths):
            return None
        cache = getattr(dlg, "_bg_cache", None)
        if not isinstance(cache, dict):
            return None
        arr = cache.get(idx)
        if arr is None:
            try:
                loaded = safe_imread(str(paths[idx]))
                arr = coerce_stack_to_tyx(np.asarray(loaded))
            except Exception:
                arr = None
            if arr is not None and getattr(arr, "size", 0) > 0:
                cache[idx] = arr
        if arr is None or getattr(arr, "size", 0) == 0:
            return None
        fi = max(0, min(int(frame_idx), int(arr.shape[0]) - 1))
        frame = np.asarray(arr[fi])
        if frame.ndim != 2:
            frame = np.asarray(frame[:, :, 0] if frame.ndim == 3 and frame.shape[-1] < 10 else frame[0])
        return frame

    def _render_overlay(
        mask_2d: np.ndarray,
        accepted: set[int],
        rejected: set[int],
        bg_u8: np.ndarray,
        pass_peaks: list[dict],
        fail_peaks: list[dict],
        show_cells: bool = True,
        show_foci: bool = True,
    ) -> np.ndarray:
        from skimage.segmentation import find_boundaries

        scaled_img, scaled_mask = scale_for_preview(bg_u8, mask_2d)
        rgb = np.stack([scaled_img, scaled_img, scaled_img], axis=-1).astype(np.uint8)

        if show_cells and accepted:
            am = np.isin(scaled_mask, np.array(list(accepted), dtype=np.int32))
            if np.any(am):
                alpha = 0.55
                rgb[am, 0] = (rgb[am, 0].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                rgb[am, 1] = (rgb[am, 1].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                rgb[am, 2] = (rgb[am, 2].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                mask_acc = np.where(am, scaled_mask, 0).astype(np.int32)
                b = find_boundaries(mask_acc, mode="outer")
                rgb[b] = 0
        if show_cells and rejected:
            rm = np.isin(scaled_mask, np.array(list(rejected), dtype=np.int32))
            if np.any(rm):
                alpha = 0.55
                rgb[rm, 0] = (rgb[rm, 0].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                rgb[rm, 1] = (rgb[rm, 1].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                rgb[rm, 2] = (rgb[rm, 2].astype(np.float32) * (1 - alpha)).astype(np.uint8)

        def _draw_peaks(peaks: list[dict], color: tuple[int, int, int]) -> None:
            h0, w0 = mask_2d.shape[:2]
            hs, ws = rgb.shape[:2]
            sx = (ws / w0) if w0 else 1.0
            sy = (hs / h0) if h0 else 1.0
            for p in peaks or []:
                yy = int(round(float(p.get("y", 0)) * sy))
                xx = int(round(float(p.get("x", 0)) * sx))
                if 0 <= yy < hs and 0 <= xx < ws:
                    r = 4
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            d2 = dx * dx + dy * dy
                            if d2 < (r - 1) * (r - 1) or d2 > r * r:
                                continue
                            y2 = yy + dy
                            x2 = xx + dx
                            if 0 <= y2 < hs and 0 <= x2 < ws:
                                rgb[y2, x2, 0] = int(color[0])
                                rgb[y2, x2, 1] = int(color[1])
                                rgb[y2, x2, 2] = int(color[2])

        # Foci markers: pass (orange) and filtered-out (red)
        if show_foci:
            try:
                _draw_peaks(pass_peaks, (255, 165, 0))
                _draw_peaks(fail_peaks, (255, 0, 0))
            except Exception:
                pass

        return rgb

    def get_current_parameters(fs: Optional[dict] = None) -> dict:
        if fs is None:
            fs = tab.get_filter_settings() if hasattr(tab, "get_filter_settings") else {}
        return {
            "pixel_size": float(fs.get("pixel_size", 0.108) or 0.108),
            "exclude_border": bool(fs.get("exclude_border", False)),
            "min_aspect_ratio": float(fs.get("min_aspect_ratio", 1.5) or 1.5),
            "max_aspect_ratio": float(fs.get("max_aspect_ratio", 10.0) or 10.0),
            "min_eccentricity": float(fs.get("min_eccentricity", 0.5) or 0.5),
            "min_solidity": float(fs.get("min_solidity", 0.7) or 0.7),
            "min_area": fs.get("min_area_um2", None),
            "max_area": fs.get("max_area_um2", None),
            "min_width": fs.get("min_width_um", None),
            "max_width": fs.get("max_width_um", None),
            "min_length": fs.get("min_length_um", None),
        }

    def _settings_signature_for(fs: Optional[dict]) -> str:
        """Best-effort stable signature for a filter settings snapshot."""
        fs = fs or {}
        # Normalize to JSON-friendly primitives
        norm: dict[str, object] = {}
        for k in sorted((fs or {}).keys()):
            v = (fs or {}).get(k)
            if v is None:
                norm[k] = None
                continue
            if isinstance(v, bool):
                norm[k] = bool(v)
                continue
            try:
                # Avoid float noise; 6 decimals is plenty for UI values.
                norm[k] = round(float(v), 6)
                continue
            except Exception:
                pass
            try:
                norm[k] = str(v)
            except Exception:
                norm[k] = None
        try:
            import json as _json

            return _json.dumps(norm, sort_keys=True, separators=(",", ":"))
        except Exception:
            return str(norm)

    def _settings_signature() -> str:
        """Best-effort stable signature for current filter settings (for cache invalidation)."""
        try:
            fs = tab.get_filter_settings() if hasattr(tab, "get_filter_settings") else {}
        except Exception:
            fs = {}
        return _settings_signature_for(fs)

    def _start_worker(
        frame_idx: int,
        *,
        settings_snapshot: Optional[dict] = None,
        settings_sig: Optional[str] = None,
        source_key=None,
    ) -> None:
        if frame_idx < 0 or frame_idx >= num_frames:
            return
        if settings_snapshot is None:
            try:
                settings_snapshot = tab.get_filter_settings() if hasattr(tab, "get_filter_settings") else {}
            except Exception:
                settings_snapshot = {}
        settings_snapshot = dict(settings_snapshot or {})
        if settings_sig is None:
            settings_sig = _settings_signature_for(settings_snapshot)
        source_key = tuple(source_key if source_key is not None else _selected_source())

        # If a previous progress dialog is still around (e.g. rapid re-apply),
        # close it to avoid stacking multiple "Processing" windows.
        try:
            prev_prog = getattr(dlg, "_progress_dialog", None)
            if prev_prog is not None:
                # Disconnect any cancel handlers before closing programmatically:
                # QProgressDialog.close() implicitly emits ``canceled``, which
                # could otherwise fire a stale handler that bumps
                # _preview_request_id (see _safely_dismiss_progress in
                # _start_worker for the same hazard).
                try:
                    prev_prog.canceled.disconnect()
                except Exception:
                    pass
                prev_prog.close()
                dlg._progress_dialog = None
        except Exception:
            pass

        existing_thread = getattr(dlg, "_preview_thread", None)
        if existing_thread is not None and existing_thread.isRunning():
            # Queue the latest request; it will run immediately after the current worker finishes.
            try:
                # Supersede the in-flight request so its result cannot repaint stale settings.
                dlg._preview_request_id += 1
                dlg._pending_apply = True
                dlg._pending_apply_frame_idx = int(frame_idx)
                dlg._pending_apply_settings = dict(settings_snapshot)
                dlg._pending_apply_settings_sig = str(settings_sig)
                dlg._pending_apply_source_key = tuple(source_key)
            except Exception:
                pass
            try:
                dlg.set_primary_action(None, label="Queued…", enabled=False, visible=True)
            except Exception:
                pass
            return

        dlg._preview_request_id += 1
        rid = int(dlg._preview_request_id)
        mask_2d = np.asarray(masks[frame_idx], dtype=np.int32)
        request_settings = dict(settings_snapshot)
        request_settings_sig = str(settings_sig)
        request_source_key = tuple(source_key)

        # Progress dialog must NOT block the preview window; user needs access to Cancel/Close.
        prog = QProgressDialog("Applying filters...", "Cancel", 0, 100, dlg)
        prog.setWindowTitle("Processing")
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.setValue(0)
        prog.setWindowModality(Qt.WindowModality.NonModal)
        prog.show()
        QApplication.processEvents()
        # Keep a reference so close/stop paths can always dismiss it.
        try:
            dlg._progress_dialog = prog
        except Exception:
            pass

        params = get_current_parameters(request_settings)
        display_params = {"do_fi_contour": False, "fi_contour_thickness_val": 1, "run_foci": False, "foci_snapshot": None}
        foci_src = _foci_source_for_source_key(request_source_key)
        display_params["raw_foci_rows"] = _load_rows_for_source(foci_src)
        # Filtering preview filters (CSV-only summary; per-cell ratio is handled in batch run)
        try:
            fs = request_settings
            display_params["filter_peaks"] = {
                "prob_thresh": (float(fs.get("foci_prob_thresh", 0.5)) if bool(fs.get("foci_prob_enabled", False)) else None),
                "min_peak_raw": (float(fs.get("foci_min_int", 0.0)) if bool(fs.get("foci_min_int_enabled", False)) else None),
                "per_cell_ratio": (
                    float(fs.get("foci_per_cell_ratio", 2.0)) if bool(fs.get("foci_per_cell_ratio_enabled", False)) else None
                ),
            }
        except Exception:
            display_params["filter_peaks"] = {}

        worker = AnalyzeMasksPreviewWorker(
            rid,
            frame_idx,
            mask_2d,
            params,
            fi_image=None,
            display_params=display_params,
            skip_vis=False,
            latest_request_id=lambda: int(getattr(dlg, "_preview_request_id", rid)),
        )
        thread = QThread(dlg)
        thread.setObjectName("AnalyzeMasksPreviewWorkerThread")
        worker.moveToThread(thread)
        dlg._preview_worker = worker
        dlg._preview_thread = thread

        def on_progress(pct: int, msg: str) -> None:
            try:
                prog.setLabelText(str(msg))
                prog.setValue(int(pct))
            except Exception:
                pass

        worker.progress.connect(on_progress)

        def _set_primary_idle() -> None:
            try:
                dlg.set_primary_action(apply_current, label="Re-apply", enabled=True, visible=True)
            except Exception:
                pass

        def _cancel_current() -> None:
            # Cooperative cancel: supersede this request id so the worker exits early.
            try:
                dlg._preview_request_id += 1
            except Exception:
                pass
            # Best-effort: request thread interruption to speed up exit.
            try:
                t0 = getattr(dlg, "_preview_thread", None)
                if t0 is not None:
                    t0.requestInterruption()
                    t0.quit()
            except Exception:
                pass
            try:
                dlg.set_primary_action(None, label="Cancelling…", enabled=False, visible=True)
            except Exception:
                pass
            try:
                prog.setLabelText("Cancelling…")
                prog.setValue(0)
            except Exception:
                pass

        # While running, primary action becomes Cancel.
        try:
            dlg.set_primary_action(_cancel_current, label="Cancel", enabled=True, visible=True)
        except Exception:
            pass
        try:
            prog.canceled.connect(_cancel_current)
        except Exception:
            pass

        def _safely_dismiss_progress() -> None:
            """Dismiss the progress dialog without spuriously firing ``canceled``.

            QProgressDialog emits ``canceled`` for *both* user-initiated cancels
            and any programmatic close/reject. If we leave the
            ``prog.canceled -> _cancel_current`` connection in place when this
            handler dismisses the dialog after the worker finishes, that signal
            fires, ``_cancel_current`` bumps ``_preview_request_id``, and the
            just-received worker result is then treated as superseded --
            ``_last_preview`` never gets populated and the overlay never
            updates. Disconnecting first keeps the cancel handler available
            for genuine user clicks while making programmatic dismissal silent.
            """
            try:
                prog.canceled.disconnect(_cancel_current)
            except Exception:
                pass
            try:
                prog.close()
            except Exception:
                pass

        def on_result_ready(
            rid_out,
            fid,
            acc_list,
            rej_list,
            _all_list,
            n_obj,
            err,
            vis,
            _peaks,
            _n_foci_total,
            _n_foci_drawn,
            _raw_median_fi,
            _min_rel,
            _mode_txt,
        ):
            _safely_dismiss_progress()
            try:
                if getattr(dlg, "_progress_dialog", None) is prog:
                    dlg._progress_dialog = None
            except Exception:
                pass
            try:
                if int(rid_out) != int(getattr(dlg, "_preview_request_id", rid_out)):
                    return
            except Exception:
                pass
            if getattr(dlg, "_close_pending", False):
                return
            if err:
                # Don't treat errored results as a valid cache entry: leave _last_preview cleared
                # so _render falls back to the unfiltered baseline. Surface the error via log.
                try:
                    write_gui_log_line("Preview", f"Filter preview worker error: {str(err)[:200]}")
                except Exception:
                    pass
                dlg._last_preview = {}
                dlg.refresh()
                return
            accepted = set(acc_list or [])
            rejected = set(rej_list or [])
            foci_txt = ""
            try:
                if int(_n_foci_total) > 0:
                    foci_txt = f"  |  Foci CSV: {int(_n_foci_total)} raw → {int(_n_foci_drawn)} pass"
            except Exception:
                foci_txt = ""
            bg = _get_bg_frame(int(fid), request_source_key)
            source_key = tuple(request_source_key)
            foci_src = _foci_source_for_source_key(source_key)
            raw_rows_all = _load_rows_for_source(foci_src)
            frame_rows = _rows_for_frame(raw_rows_all, int(fid))

            # Build raw + filtered peak sets for this frame.
            pass_peaks: list[dict] = []
            fail_peaks: list[dict] = []
            try:
                fs = request_settings
                prob_thr = float(fs.get("foci_prob_thresh", 0.5)) if bool(fs.get("foci_prob_enabled", False)) else None
                min_raw = float(fs.get("foci_min_int", 0.0)) if bool(fs.get("foci_min_int_enabled", False)) else None
                filtered = []
                for r in frame_rows:
                    try:
                        y = int(float(r.get("y", 0) or 0))
                        x = int(float(r.get("x", 0) or 0))
                    except Exception:
                        continue
                    try:
                        intensity = float(r.get("intensity", 0) or 0.0)
                    except Exception:
                        intensity = 0.0
                    pr = r.get("probability", None)
                    prob = None
                    if pr not in (None, "", "None"):
                        try:
                            prob = float(pr)
                        except Exception:
                            prob = None
                    if prob_thr is not None and prob is not None and prob < float(prob_thr):
                        continue
                    if min_raw is not None and intensity < float(min_raw):
                        continue
                    filtered.append({"y": y, "x": x, "intensity": intensity, "probability": prob})

                # Expensive per-cell ratio refinement: cache by full settings signature so
                # probability/intensity prefilters cannot reuse a stale refined peak set.
                per_cell_ratio = (
                    float(fs.get("foci_per_cell_ratio", 2.0)) if bool(fs.get("foci_per_cell_ratio_enabled", False)) else None
                )
                if per_cell_ratio is not None and bg is not None:
                    cache = getattr(dlg, "_foci_refine_cache", None)
                    if not isinstance(cache, dict):
                        cache = {}
                        dlg._foci_refine_cache = cache
                    refine_key = (tuple(source_key), int(fid), str(request_settings_sig))
                    cached = cache.get(refine_key)
                    if cached is None:
                        from bactimate.modules.foci_utils import refine_spotiflow_peaks

                        cached = refine_spotiflow_peaks(
                            np.asarray(bg),
                            filtered,
                            labeled_mask=np.asarray(mask_2d, dtype=np.int32),
                            per_cell_ratio=float(per_cell_ratio),
                        )
                        cache[refine_key] = list(cached)
                    filtered = list(cached)
                pass_peaks = list(filtered)
                pass_keys = {(int(p.get("y", 0)), int(p.get("x", 0))) for p in pass_peaks}
                fail_peaks = []
                for r in frame_rows:
                    try:
                        key = (int(float(r.get("y", 0) or 0)), int(float(r.get("x", 0) or 0)))
                        if key in pass_keys:
                            continue
                        fail_peaks.append(
                            {
                                "y": key[0],
                                "x": key[1],
                                "intensity": float(r.get("intensity", 0) or 0.0),
                                "probability": r.get("probability", None),
                            }
                        )
                    except Exception:
                        continue
            except Exception:
                pass_peaks = []
                fail_peaks = []

            raw_n = len(frame_rows)
            pass_n = len(pass_peaks)
            fail_n = max(0, raw_n - pass_n)

            dlg._last_preview = {
                "source_key": source_key,
                "frame_idx": int(fid),
                "settings_sig": request_settings_sig,
                "mask_2d": np.asarray(mask_2d, dtype=np.int32),
                "accepted": accepted,
                "rejected": rejected,
                "n_obj": int(n_obj) if n_obj is not None else None,
                "foci_txt": foci_txt,
                "pass_peaks": pass_peaks,
                "fail_peaks": fail_peaks,
                "bg": bg,
                "info_text": (
                    f"Accepted: {len(accepted)}  Rejected: {len(rejected)}  Total: {int(n_obj)}"
                    f"  |  Foci(frame): raw {raw_n} / pass {pass_n} / fail {fail_n}{foci_txt}"
                ),
            }
            dlg.refresh()

        def on_thread_finished() -> None:
            try:
                if getattr(dlg, "_progress_dialog", None) is prog:
                    _safely_dismiss_progress()
                    dlg._progress_dialog = None
                elif getattr(dlg, "_progress_dialog", None) is not None:
                    try:
                        dlg._progress_dialog.canceled.disconnect()
                    except Exception:
                        pass
                    dlg._progress_dialog.close()
                    dlg._progress_dialog = None
            except Exception:
                pass
            _safe_release_preview_worker(dlg)
            _set_primary_idle()
            try:
                dlg._preview_worker_slot = None
                dlg._preview_worker_done_slot = None
            except Exception:
                pass
            if getattr(dlg, "_close_pending", False):
                try:
                    dlg.close()
                except Exception:
                    pass
                return

            # If the user queued a re-apply while the worker was running, execute it now.
            try:
                if bool(getattr(dlg, "_pending_apply", False)):
                    dlg._pending_apply = False
                    fi2 = int(getattr(dlg, "_pending_apply_frame_idx", 0) or 0)
                    fs2 = dict(getattr(dlg, "_pending_apply_settings", {}) or {})
                    sig2 = str(getattr(dlg, "_pending_apply_settings_sig", "") or _settings_signature_for(fs2))
                    src2 = tuple(getattr(dlg, "_pending_apply_source_key", _selected_source()))
                    _start_worker(fi2, settings_snapshot=fs2, settings_sig=sig2, source_key=src2)
                    return
            except Exception:
                pass

        dlg._preview_worker_slot = on_result_ready
        dlg._preview_worker_done_slot = on_thread_finished
        worker.result_ready.connect(on_result_ready)
        worker.result_ready.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(on_thread_finished)
        thread.start()

    def apply_current() -> None:
        try:
            _ui_to_settings()
        except Exception:
            pass
        idx = int(frame_spin.value()) - 1
        try:
            fs = tab.get_filter_settings() if hasattr(tab, "get_filter_settings") else {}
        except Exception:
            fs = {}
        sig = _settings_signature_for(fs)
        src = tuple(_selected_source())
        try:
            dlg._last_preview = {}
            dlg.set_frame_idx(idx)
            dlg.set_source_key(src)
            dlg.refresh()
        except Exception:
            pass
        _start_worker(idx, settings_snapshot=dict(fs), settings_sig=sig, source_key=src)

    def _render(state) -> RenderResult:
        last = getattr(dlg, "_last_preview", None) or {}
        # LUT frame is always the current background (if any), else zeros so the LUT section still renders.
        bg = _get_bg_frame(state.frame_idx)
        lut_frame = np.asarray(bg) if bg is not None else np.zeros(masks.shape[1:3], dtype=np.float32)
        bg_u8 = _apply_lut_to_bg(bg, state.vmin, state.vmax)
        current_source_key = tuple(_selected_source())

        # See is_preview_cache_hit() for the cache-validity rules and rationale.
        if is_preview_cache_hit(last, int(state.frame_idx), current_source_key):
            mask_2d = np.asarray(last.get("mask_2d"), dtype=np.int32)
            accepted = set(last.get("accepted") or [])
            rejected = set(last.get("rejected") or [])
            pass_peaks = list(last.get("pass_peaks") or [])
            fail_peaks = list(last.get("fail_peaks") or [])
            info_text = str(last.get("info_text") or "")
        else:
            mask_2d = np.asarray(masks[int(state.frame_idx)], dtype=np.int32)
            # Initial preview state is intentionally UNFILTERED:
            # - show all segmented cells as "accepted"
            # - show all raw foci points (no probability/intensity/ratio filtering)
            all_labels = set(np.unique(mask_2d).astype(np.int32).tolist()) - {0}
            accepted = set(all_labels)
            rejected = set()
            pass_peaks = []
            fail_peaks = []
            try:
                foci_src0 = _foci_source_for_selected_image()
                rows_all0 = _load_rows_for_source(foci_src0)
                frame_rows0 = _rows_for_frame(rows_all0, int(state.frame_idx))
                pass_peaks = []
                for r in frame_rows0:
                    try:
                        y = int(float(r.get("y", 0) or 0))
                        x = int(float(r.get("x", 0) or 0))
                    except Exception:
                        continue
                    pass_peaks.append({"y": y, "x": x, "intensity": r.get("intensity", None), "probability": r.get("probability", None)})
            except Exception:
                pass_peaks = []
            raw_n0 = len(pass_peaks)
            info_text = (
                f"Unfiltered preview: cells {len(all_labels)} shown, foci(frame) raw {raw_n0}. "
                f"Click Re-apply to apply filters."
            )

        overlay = _render_overlay(
            mask_2d,
            accepted,
            rejected,
            bg_u8,
            pass_peaks,
            fail_peaks,
            show_cells=bool(state.extra.get("show_cells", True)),
            show_foci=bool(state.extra.get("show_foci", bool(foci_overlay_available))),
        )
        pm = to_qpixmap(overlay)
        return RenderResult(
            pixmap=pm,
            image_size=(overlay.shape[1], overlay.shape[0]),
            lut_frame=lut_frame,
            title_text=f"Filter Effects Preview — {mask_path.name}",
            info_text=info_text,
        )

    def _disconnect_worker_finished() -> None:
        w = getattr(dlg, "_preview_worker", None)
        thread = getattr(dlg, "_preview_thread", None)
        s = getattr(dlg, "_preview_worker_slot", None)
        if w is not None and s is not None:
            try:
                w.result_ready.disconnect(s)
            except Exception:
                pass
            try:
                if thread is not None:
                    w.result_ready.disconnect(thread.quit)
            except Exception:
                pass
        ds = getattr(dlg, "_preview_worker_done_slot", None)
        if thread is not None and ds is not None:
            try:
                thread.finished.disconnect(ds)
            except Exception:
                pass
        _safe_release_preview_worker(dlg)

    def _is_running() -> bool:
        thread = getattr(dlg, "_preview_thread", None)
        return bool(thread is not None and thread.isRunning())

    def _wait(ms: int) -> bool:
        thread = getattr(dlg, "_preview_thread", None)
        if thread is None:
            return True
        try:
            return bool(thread.wait(int(ms)))
        except Exception:
            return True

    def _on_close_defer() -> None:
        write_gui_log_line("Preview", "Filter preview: worker running; deferring close until finished.")
        dlg._close_pending = True

    def _request_stop() -> None:
        try:
            dlg._preview_request_id += 1
        except Exception:
            pass
        thread = getattr(dlg, "_preview_thread", None)
        if thread is not None:
            try:
                thread.requestInterruption()
                thread.quit()
            except Exception:
                pass
        try:
            if getattr(dlg, "_progress_dialog", None) is not None:
                try:
                    dlg._progress_dialog.canceled.disconnect()
                except Exception:
                    pass
                dlg._progress_dialog.close()
                dlg._progress_dialog = None
        except Exception:
            pass
        _on_close_defer()

    def _disconnect_and_restore() -> None:
        _disconnect_worker_finished()

    try:
        main_win = tab.window() if hasattr(tab, "window") and callable(tab.window) else tab
        dlg = PreviewDialogBase(
            main_win,
            title=f"Filter Effects Preview — {mask_path.name}",
            top_controls=settings_host,
            top_controls_title="Filter parameters",
            top_controls_expanded=True,
            bottom_controls=bottom,
            primary_action_label="Re-apply",
            primary_action_callback=apply_current,
            render_callback=_render,
            worker_guard=WorkerGuard(
                is_running=_is_running,
                wait=_wait,
                request_stop=_request_stop,
                disconnect=_disconnect_and_restore,
                on_close_while_running="defer_hide",
                warning_title="Filtering in progress",
                warning_text="Please wait for filtering to finish before closing the preview.",
            ),
            minimum_size=(1100, 760),
        )
        # Keep reference on the stage tab so app shutdown waits for worker safely.
        try:
            tab._filter_preview_dialog = dlg
        except Exception:
            pass
        dlg._close_pending = False
        dlg._preview_request_id = 0
        dlg._bg_paths = bg_paths
        dlg._bg_cache = {}  # idx -> TYX stack
        dlg._foci_rows_cache = {}  # resolved foci-source path -> csv rows
        dlg._last_preview = {}
        dlg.set_extra("show_cells", True)
        dlg.set_extra("show_foci", bool(foci_overlay_available))

        def _set_state_from_ui_and_apply(*, reset_lut: bool = False) -> None:
            dlg.set_frame_idx(int(frame_spin.value()) - 1)
            src = _selected_source()
            dlg.set_source_key(src)
            if reset_lut:
                try:
                    dlg.state().vmin = None
                    dlg.state().vmax = None
                    dlg.lut_widget.set_lut(None, None)
                except Exception:
                    pass
            apply_current()

        frame_spin.valueChanged.connect(lambda _v: _set_state_from_ui_and_apply(reset_lut=False))
        image_combo.currentIndexChanged.connect(lambda *_: _set_state_from_ui_and_apply(reset_lut=True))

        def _on_nav_changed(st: PreviewNavState, reason: str) -> None:
            # Mirror shared controls back into legacy widgets so all existing helpers keep working.
            try:
                if reason in ("frame", "init"):
                    frame_spin.blockSignals(True)
                    frame_spin.setValue(int(st.frame_idx) + 1)
                    frame_spin.blockSignals(False)
                if reason in ("source", "init"):
                    key = st.source_key
                    idx = 0
                    for ii in range(int(image_combo.count())):
                        if image_combo.itemData(ii) == key:
                            idx = int(ii)
                            break
                    image_combo.blockSignals(True)
                    image_combo.setCurrentIndex(idx)
                    image_combo.blockSignals(False)
            except Exception:
                return
            # Navigation should be lightweight: update state + refresh, but do NOT auto-run filtering.
            # Filtering is applied explicitly via Re-apply.
            try:
                dlg.set_frame_idx(int(frame_spin.value()) - 1)
                dlg.set_source_key(_selected_source())
                if reason == "source":
                    try:
                        dlg.state().vmin = None
                        dlg.state().vmax = None
                        dlg.lut_widget.set_lut(None, None)
                    except Exception:
                        pass
                dlg.refresh()
            except Exception:
                pass

        nav_ctrl.nav_changed.connect(_on_nav_changed)

        def _on_overlay_changed(key: str, checked: bool) -> None:
            dlg.set_extra(str(key), bool(checked))
            dlg.refresh()

        nav_ctrl.overlay_changed.connect(_on_overlay_changed)

        dlg.set_frame_idx(0)
        dlg.set_source_key(_selected_source())
        dlg.refresh()  # draw initial (pre-worker) overlay and initialize histogram
        # ---- Wire parameter panel to tab filter settings --------------
        def _ui_to_settings() -> None:
            if not hasattr(tab, "set_filter_settings"):
                return

            # Defensive: spinboxes use setKeyboardTracking(False), which means
            # value() can lag behind the line-edit text until editingFinished
            # fires. If the user typed a new value and clicked "Re-apply"
            # without first tabbing/clicking off the spinbox, value() may
            # otherwise return the previous value. interpretText() forces the
            # spinbox to commit whatever is in its line edit right now.
            for w in (
                w_pixel,
                w_min_ar,
                w_max_ar,
                w_min_ecc,
                w_min_sol,
                w_min_area,
                w_max_area,
                w_min_w,
                w_max_w,
                w_min_len,
                w_prob_thr,
                w_minint,
                w_ratio,
            ):
                try:
                    w.interpretText()
                except Exception:
                    pass

            def _opt0(w: QDoubleSpinBox) -> Optional[float]:
                try:
                    v = float(w.value())
                    return v if v > 0 else None
                except Exception:
                    return None

            tab.set_filter_settings(
                {
                    "pixel_size": float(w_pixel.value()),
                    "min_aspect_ratio": float(w_min_ar.value()),
                    "max_aspect_ratio": float(w_max_ar.value()),
                    "min_eccentricity": float(w_min_ecc.value()),
                    "min_solidity": float(w_min_sol.value()),
                    "min_area_um2": _opt0(w_min_area),
                    "max_area_um2": _opt0(w_max_area),
                    "min_width_um": _opt0(w_min_w),
                    "max_width_um": _opt0(w_max_w),
                    "min_length_um": _opt0(w_min_len),
                    "exclude_border": bool(w_excl_border.isChecked()),
                    "foci_prob_enabled": bool(w_prob_en.isChecked()),
                    "foci_prob_thresh": float(w_prob_thr.value()),
                    "foci_min_int_enabled": bool(w_minint_en.isChecked()),
                    "foci_min_int": float(w_minint.value()),
                    "foci_per_cell_ratio_enabled": bool(w_ratio_en.isChecked()),
                    "foci_per_cell_ratio": float(w_ratio.value()),
                }
            )

        def _settings_to_ui() -> None:
            fs = tab.get_filter_settings() if hasattr(tab, "get_filter_settings") else {}
            # Block signals to avoid accidental re-apply.
            for w in (
                w_pixel,
                w_min_ar,
                w_max_ar,
                w_min_ecc,
                w_min_sol,
                w_min_area,
                w_max_area,
                w_min_w,
                w_max_w,
                w_min_len,
                w_prob_thr,
                w_minint,
                w_ratio,
            ):
                try:
                    w.blockSignals(True)
                except Exception:
                    pass
            for cb in (w_excl_border, w_prob_en, w_minint_en, w_ratio_en):
                try:
                    cb.blockSignals(True)
                except Exception:
                    pass
            try:
                w_pixel.setValue(float(fs.get("pixel_size", 0.108) or 0.108))
                w_min_ar.setValue(float(fs.get("min_aspect_ratio", 1.5) or 1.5))
                w_max_ar.setValue(float(fs.get("max_aspect_ratio", 10.0) or 10.0))
                w_min_ecc.setValue(float(fs.get("min_eccentricity", 0.5) or 0.5))
                w_min_sol.setValue(float(fs.get("min_solidity", 0.7) or 0.7))
                w_min_area.setValue(float(fs.get("min_area_um2") or 0.0))
                w_max_area.setValue(float(fs.get("max_area_um2") or 0.0))
                w_min_w.setValue(float(fs.get("min_width_um") or 0.0))
                w_max_w.setValue(float(fs.get("max_width_um") or 0.0))
                w_min_len.setValue(float(fs.get("min_length_um") or 0.0))
                w_excl_border.setChecked(bool(fs.get("exclude_border", False)))
                w_prob_en.setChecked(bool(fs.get("foci_prob_enabled", False)))
                w_prob_thr.setValue(float(fs.get("foci_prob_thresh", 0.5) or 0.5))
                w_minint_en.setChecked(bool(fs.get("foci_min_int_enabled", False)))
                w_minint.setValue(float(fs.get("foci_min_int", 0.0) or 0.0))
                w_ratio_en.setChecked(bool(fs.get("foci_per_cell_ratio_enabled", False)))
                w_ratio.setValue(float(fs.get("foci_per_cell_ratio", 2.0) or 2.0))
            finally:
                for w in (
                    w_pixel,
                    w_min_ar,
                    w_max_ar,
                    w_min_ecc,
                    w_min_sol,
                    w_min_area,
                    w_max_area,
                    w_min_w,
                    w_max_w,
                    w_min_len,
                    w_prob_thr,
                    w_minint,
                    w_ratio,
                ):
                    try:
                        w.blockSignals(False)
                    except Exception:
                        pass
                for cb in (w_excl_border, w_prob_en, w_minint_en, w_ratio_en):
                    try:
                        cb.blockSignals(False)
                    except Exception:
                        pass

        # Always keep tab state in sync with UI edits (no auto-run).
        for w in (
            w_pixel,
            w_min_ar,
            w_max_ar,
            w_min_ecc,
            w_min_sol,
            w_min_area,
            w_max_area,
            w_min_w,
            w_max_w,
            w_min_len,
            w_prob_thr,
            w_minint,
            w_ratio,
        ):
            try:
                w.valueChanged.connect(lambda *_: _ui_to_settings())
            except Exception:
                pass
        for cb in (w_excl_border, w_prob_en, w_minint_en, w_ratio_en):
            try:
                cb.toggled.connect(lambda *_: _ui_to_settings())
            except Exception:
                pass

        _settings_to_ui()
        # Test seam: expose enough internals so headless tests can drive the
        # full apply flow without scraping the QObject tree. Production code
        # never reads these handles.
        try:
            dlg._pf_apply_current = apply_current
            dlg._pf_ui_to_settings = _ui_to_settings
            dlg._pf_settings_to_ui = _settings_to_ui
            dlg._pf_spinboxes = {
                "pixel_size": w_pixel,
                "min_aspect_ratio": w_min_ar,
                "max_aspect_ratio": w_max_ar,
                "min_eccentricity": w_min_ecc,
                "min_solidity": w_min_sol,
                "min_area_um2": w_min_area,
                "max_area_um2": w_max_area,
                "min_width_um": w_min_w,
                "max_width_um": w_max_w,
                "min_length_um": w_min_len,
                "foci_prob_thresh": w_prob_thr,
                "foci_min_int": w_minint,
                "foci_per_cell_ratio": w_ratio,
            }
            dlg._pf_checkboxes = {
                "exclude_border": w_excl_border,
                "foci_prob_enabled": w_prob_en,
                "foci_min_int_enabled": w_minint_en,
                "foci_per_cell_ratio_enabled": w_ratio_en,
            }
        except Exception:
            pass
        # Do NOT auto-apply filtering on open; show the unfiltered baseline first.
        # User can click Re-apply to compute accepted vs rejected.
        try:
            dlg.state().vmin = None
            dlg.state().vmax = None
            dlg.lut_widget.set_lut(None, None)
        except Exception:
            pass
        dlg.refresh()
        dlg.show()
    except Exception as e:
        _disconnect_and_restore()
        QMessageBox.warning(tab, "Preview Error", f"Could not open filtering preview:\n{e}")
