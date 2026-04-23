from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactimate.gui.analyze_masks_preview_worker import AnalyzeMasksPreviewWorker
from bactimate.gui.image_canvas import ImageCanvas
from bactimate.gui.image_utils import scale_for_preview, to_qpixmap
from bactimate.gui.log_config import write_gui_log_line
from bactimate.gui.utils import safe_imread


def _safe_release_preview_worker(owner, attr: str = "_preview_worker", wait_ms: int = 10_000) -> None:
    w = getattr(owner, attr, None)
    if w is None:
        return
    try:
        if w.isRunning():
            w.wait(wait_ms)
    except Exception:
        pass
    try:
        w.deleteLater()
    except Exception:
        pass
    try:
        setattr(owner, attr, None)
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

    # Optional: load raw foci rows for CSV-only peak-filter summary
    raw_foci_rows = []
    try:
        foci_sources = tab._resolve_foci_paths_for_mask(mask_path)
        if foci_sources:
            from bactimate.gui.artifact_paths import foci_csv_path, raw_foci_csv_path
            import csv as _csv

            csv_new = foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=foci_sources[0])
            csv_legacy = raw_foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=foci_sources[0])
            csv_path = csv_new if csv_new.exists() else (csv_legacy if csv_legacy.exists() else None)
            if csv_path is not None and csv_path.exists():
                with open(csv_path, "r", newline="") as f:
                    reader = _csv.DictReader(f)
                    for r in reader:
                        raw_foci_rows.append(r)
    except Exception:
        raw_foci_rows = []

    num_frames = int(masks.shape[0])

    class PreviewDialog(QDialog):
        def closeEvent(self, event):
            w = getattr(self, "_preview_worker", None)
            s = getattr(self, "_preview_worker_slot", None)
            if w is not None and w.isRunning():
                write_gui_log_line("Preview", "Filter preview: worker running; deferring close until finished.")
                self._close_pending = True
                self.hide()
                event.ignore()
                return
            if w is not None and s is not None:
                try:
                    w.finished.disconnect(s)
                except Exception:
                    pass
            _safe_release_preview_worker(self)
            try:
                tab._restore_analysis_settings_to_stash()
            except Exception:
                pass
            try:
                tab._restore_peak_filters_to_stash()
            except Exception:
                pass
            super().closeEvent(event)

    dlg = PreviewDialog(tab)
    dlg._close_pending = False
    dlg._preview_request_id = 0
    dlg.setWindowTitle(f"Filter Effects Preview — {mask_path.name}")
    dlg.setMinimumSize(1100, 760)
    layout = QVBoxLayout(dlg)

    # Host the shared analysis settings in this dialog
    try:
        settings_host = QWidget()
        settings_host_layout = QVBoxLayout(settings_host)
        settings_host_layout.setContentsMargins(0, 0, 0, 0)
        settings_host_layout.setSpacing(0)
        tab._place_analysis_settings_in_preview(settings_host_layout)
        try:
            tab._place_peak_filters_in_preview(settings_host_layout)
        except Exception:
            pass
        layout.addWidget(settings_host)
    except Exception:
        pass

    canvas = ImageCanvas()
    canvas.setMinimumSize(400, 300)
    canvas.setStyleSheet("background-color: #1e1e1e;")
    layout.addWidget(canvas, 1)

    nav = QHBoxLayout()
    nav.addWidget(QLabel("Image:"))
    image_combo = QComboBox()
    nav.addWidget(image_combo)
    norm_cb = QCheckBox("Normalized")
    norm_cb.setChecked(True)
    norm_cb.setToolTip("Percentile normalization for display (pLow/pHigh).")
    nav.addWidget(norm_cb)
    nav.addWidget(QLabel("pLow"))
    p_low = QDoubleSpinBox()
    p_low.setRange(0.0, 49.0)
    p_low.setDecimals(1)
    p_low.setSingleStep(0.5)
    p_low.setValue(1.0)
    p_low.setFixedWidth(70)
    nav.addWidget(p_low)
    nav.addWidget(QLabel("pHigh"))
    p_high = QDoubleSpinBox()
    p_high.setRange(51.0, 100.0)
    p_high.setDecimals(1)
    p_high.setSingleStep(0.5)
    p_high.setValue(99.8)
    p_high.setFixedWidth(74)
    nav.addWidget(p_high)

    nav.addWidget(QLabel("Frame:"))
    frame_spin = QSpinBox()
    frame_spin.setRange(1, max(1, num_frames))
    frame_spin.setValue(1)
    nav.addWidget(frame_spin)
    nav.addWidget(QLabel(f"of {num_frames}"))
    nav.addStretch()
    info_label = QLabel("")
    info_label.setStyleSheet("font-weight: bold; color: #2196F3;")
    nav.addWidget(info_label)
    reapply_btn = QPushButton("Re-apply")
    nav.addWidget(reapply_btn)
    layout.addLayout(nav)

    # ---- background image discovery + cache --------------------------
    from bactimate.modules.image_stack_utils import coerce_stack_to_tyx

    def _list_background_paths() -> list[Path]:
        folder = mask_path.parent
        out: list[Path] = []
        seen: set[str] = set()

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

        try:
            for fp in tab._resolve_foci_paths_for_mask(mask_path) or []:
                fp = Path(fp)
                try:
                    rp = str(fp.resolve())
                except Exception:
                    rp = str(fp)
                if rp in seen:
                    continue
                seen.add(rp)
                out.append(fp)
        except Exception:
            pass

        out.sort(key=lambda x: x.name.lower())
        return out

    bg_paths = _list_background_paths()
    image_combo.addItem("Mask only", ("none", -1))
    for i, fp in enumerate(bg_paths):
        image_combo.addItem(fp.name, ("file", i))

    dlg._bg_paths = bg_paths
    dlg._bg_cache = {}  # idx -> TYX stack

    def _get_bg_frame(frame_idx: int) -> Optional[np.ndarray]:
        sel = image_combo.currentData()
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

    def _render_overlay(mask_2d: np.ndarray, accepted: set[int], rejected: set[int], bg_2d: Optional[np.ndarray], peaks: list[dict]) -> np.ndarray:
        from skimage.segmentation import find_boundaries
        from bactimate.modules.image_stack_utils import percentile_normalize_2d

        if bg_2d is None:
            bg_u8 = np.zeros(mask_2d.shape[:2], dtype=np.uint8)
        else:
            bg_f = np.asarray(bg_2d, dtype=np.float32)
            if norm_cb.isChecked():
                bg_f = percentile_normalize_2d(bg_f, pmin=float(p_low.value()), pmax=float(p_high.value()))
                bg_u8 = (bg_f * 255.0).clip(0, 255).astype(np.uint8)
            else:
                mn = float(np.nanmin(bg_f)) if bg_f.size else 0.0
                mx = float(np.nanmax(bg_f)) if bg_f.size else 1.0
                if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                    bg_u8 = np.zeros(mask_2d.shape[:2], dtype=np.uint8)
                else:
                    bg_u8 = ((bg_f - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

        scaled_img, scaled_mask = scale_for_preview(bg_u8, mask_2d)
        rgb = np.stack([scaled_img, scaled_img, scaled_img], axis=-1).astype(np.uint8)

        if accepted:
            am = np.isin(scaled_mask, np.array(list(accepted), dtype=np.int32))
            if np.any(am):
                alpha = 0.55
                rgb[am, 0] = (rgb[am, 0].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                rgb[am, 1] = (rgb[am, 1].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                rgb[am, 2] = (rgb[am, 2].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                mask_acc = np.where(am, scaled_mask, 0).astype(np.int32)
                b = find_boundaries(mask_acc, mode="outer")
                rgb[b] = 0
        if rejected:
            rm = np.isin(scaled_mask, np.array(list(rejected), dtype=np.int32))
            if np.any(rm):
                alpha = 0.55
                rgb[rm, 0] = (rgb[rm, 0].astype(np.float32) * (1 - alpha) + 255 * alpha).astype(np.uint8)
                rgb[rm, 1] = (rgb[rm, 1].astype(np.float32) * (1 - alpha)).astype(np.uint8)
                rgb[rm, 2] = (rgb[rm, 2].astype(np.float32) * (1 - alpha)).astype(np.uint8)

        # Foci markers (orange rings)
        try:
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
                                rgb[y2, x2, 0] = 255
                                rgb[y2, x2, 1] = 165
                                rgb[y2, x2, 2] = 0
        except Exception:
            pass

        return rgb

    def get_current_parameters() -> dict:
        # Mirror the worker expectations in `AnalyzeMasksPreviewWorker.run()`
        def _val(x, default):
            try:
                return float(x.value())
            except Exception:
                return float(default)

        def _opt_um2(w) -> Optional[float]:
            try:
                v = float(w.value())
                return v if v > 0 else None
            except Exception:
                return None

        return {
            "pixel_size": _val(getattr(tab, "pixel_size", None), 0.108),
            "min_aspect_ratio": _val(getattr(tab, "min_aspect_ratio", None), 1.5),
            "max_aspect_ratio": _val(getattr(tab, "max_aspect_ratio", None), 10.0),
            "min_eccentricity": _val(getattr(tab, "min_eccentricity", None), 0.5),
            "min_solidity": _val(getattr(tab, "min_solidity", None), 0.7),
            "min_area": _opt_um2(getattr(tab, "min_area", None)),
            "max_area": _opt_um2(getattr(tab, "max_area", None)),
            "min_width": _opt_um2(getattr(tab, "min_width", None)),
            "max_width": _opt_um2(getattr(tab, "max_width", None)),
        }

    def _start_worker(frame_idx: int) -> None:
        if frame_idx < 0 or frame_idx >= num_frames:
            return

        dlg._preview_request_id += 1
        rid = int(dlg._preview_request_id)
        mask_2d = np.asarray(masks[frame_idx], dtype=np.int32)

        prog = QProgressDialog("Applying filters...", "Cancel", 0, 100, dlg)
        prog.setWindowTitle("Processing")
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.setValue(0)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.show()
        QApplication.processEvents()

        params = get_current_parameters()
        display_params = {"do_fi_contour": False, "fi_contour_thickness_val": 1, "run_foci": False, "foci_snapshot": None}
        display_params["raw_foci_rows"] = raw_foci_rows
        # Filtering preview filters (CSV-only summary; per-cell ratio is handled in batch run)
        try:
            display_params["filter_peaks"] = {
                "prob_thresh": (float(tab.filter_prob_thresh.value()) if tab.filter_prob_enabled.isChecked() else None),
                "min_peak_raw": (float(tab.filter_min_int.value()) if tab.filter_min_int_enabled.isChecked() else None),
                "per_cell_ratio": (float(tab.filter_per_cell_ratio.value()) if tab.filter_per_cell_ratio_enabled.isChecked() else None),
            }
        except Exception:
            display_params["filter_peaks"] = {}

        existing = getattr(dlg, "_preview_worker", None)
        if existing is not None and existing.isRunning():
            return

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
        dlg._preview_worker = worker

        def on_progress(pct: int, msg: str) -> None:
            try:
                prog.setLabelText(str(msg))
                prog.setValue(int(pct))
            except Exception:
                pass

        worker.progress.connect(on_progress)

        def on_finished(
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
            try:
                prog.close()
            except Exception:
                pass
            _safe_release_preview_worker(dlg)
            if getattr(dlg, "_close_pending", False):
                dlg.close()
                return
            if err:
                info_label.setText(f"Error: {str(err)[:80]}")
                return
            accepted = set(acc_list or [])
            rejected = set(rej_list or [])
            foci_txt = ""
            try:
                if int(_n_foci_total) > 0:
                    foci_txt = f"  |  Foci CSV: {int(_n_foci_total)} raw → {int(_n_foci_drawn)} pass"
            except Exception:
                foci_txt = ""
            info_label.setText(f"Accepted: {len(accepted)}  Rejected: {len(rejected)}  Total: {int(n_obj)}{foci_txt}")
            bg = _get_bg_frame(int(fid))

            # Build filtered peaks for this frame to draw.
            peaks_to_draw: list[dict] = []
            try:
                frame_rows = []
                for r in (raw_foci_rows or []):
                    try:
                        fr = int(float(r.get("frame", 0) or 0)) if "frame" in r else 0
                    except Exception:
                        fr = 0
                    if fr != int(fid):
                        continue
                    frame_rows.append(r)

                prob_thr = float(tab.filter_prob_thresh.value()) if tab.filter_prob_enabled.isChecked() else None
                min_raw = float(tab.filter_min_int.value()) if tab.filter_min_int_enabled.isChecked() else None
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

                per_cell_ratio = float(tab.filter_per_cell_ratio.value()) if tab.filter_per_cell_ratio_enabled.isChecked() else None
                if per_cell_ratio is not None and bg is not None:
                    from bactimate.modules.foci_utils import refine_spotiflow_peaks

                    filtered = refine_spotiflow_peaks(
                        np.asarray(bg),
                        filtered,
                        labeled_mask=np.asarray(mask_2d, dtype=np.int32),
                        per_cell_ratio=float(per_cell_ratio),
                    )
                peaks_to_draw = list(filtered)
            except Exception:
                peaks_to_draw = []

            overlay = _render_overlay(
                np.asarray(mask_2d, dtype=np.int32),
                accepted,
                rejected,
                bg,
                peaks_to_draw,
            )
            pm = to_qpixmap(overlay)
            try:
                canvas.set_pixmap(pm, (overlay.shape[1], overlay.shape[0]), reset_view=False)
            except Exception:
                try:
                    canvas.setPixmap(pm)  # type: ignore[attr-defined]
                except Exception:
                    pass

        dlg._preview_worker_slot = on_finished
        worker.finished.connect(on_finished)
        worker.start()

    def apply_current() -> None:
        idx = int(frame_spin.value()) - 1
        _start_worker(idx)

    frame_spin.valueChanged.connect(lambda _v: apply_current())
    reapply_btn.clicked.connect(apply_current)
    image_combo.currentIndexChanged.connect(lambda *_: apply_current())
    norm_cb.toggled.connect(lambda *_: apply_current())
    p_low.valueChanged.connect(lambda *_: apply_current())
    p_high.valueChanged.connect(lambda *_: apply_current())

    apply_current()
    dlg.show()

