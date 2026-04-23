from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QWidget,
    QVBoxLayout,
)

from bactimate.gui.session_params import apply_to_widgets
from bactimate.gui.utils import safe_imread
from bactimate.gui.tabs._analyze_masks_base import AnalyzeMasksBase


class FilteringTab(AnalyzeMasksBase):
    """Section 2: Filtering (mask morphology/size + foci CSV peak filtering)."""

    # ---- base hooks ---------------------------------------------------
    def _stage_key(self) -> str:
        return "filtering"

    def _file_group_title(self) -> str:
        return "Filter masks"

    def _preview_button_label(self) -> str:
        return "Select Filter"

    def _preview_button_tooltip(self) -> str:
        return "Open preview and adjust filtering settings in the preview window"

    def _primary_button_label(self) -> str:
        return "Filter File/Folder"

    def _primary_button_tooltip(self) -> str:
        return "Write filtered mask and foci CSV artifacts for the selected path"

    def _stage_note_text(self) -> str:
        return "Filtering settings are adjusted in Preview. Open preview to tune parameters, then run filtering from this tab."

    def _build_stage_controls(self, scroll_layout: QVBoxLayout) -> None:
        # Peak filters are adjusted in the Filter Preview window (Select Filter),
        # not on the main tab. We keep the widgets alive here so batch filtering and
        # session params can read them, but stash them in a hidden container.
        peak_group = self._build_filtering_peak_filters_widget()
        self._peak_filters_stash = QWidget(self)
        self._peak_filters_stash.hide()
        stash_layout = QVBoxLayout(self._peak_filters_stash)
        stash_layout.setContentsMargins(0, 0, 0, 0)
        stash_layout.setSpacing(0)
        stash_layout.addWidget(peak_group)
        self._peak_filters_stash_layout = stash_layout

    def _wire_stage_buttons(self) -> None:
        self.preview_btn.clicked.connect(self.preview_filter_effects)
        self.analyze_btn.clicked.connect(self.filter_masks)

    # ---- filtering-only UI -------------------------------------------
    def _build_filtering_peak_filters_widget(self) -> QGroupBox:
        peak_group = QGroupBox("Peak filtering (from foci CSV)")
        peak_layout = QVBoxLayout()
        peak_layout.setSpacing(6)

        row = QHBoxLayout()
        self.filter_prob_enabled = QCheckBox("Filter by probability ≥")
        self.filter_prob_enabled.setChecked(False)
        self.filter_prob_thresh = QDoubleSpinBox()
        self.filter_prob_thresh.setRange(0.0, 1.0)
        self.filter_prob_thresh.setDecimals(3)
        self.filter_prob_thresh.setSingleStep(0.05)
        self.filter_prob_thresh.setValue(0.5)
        row.addWidget(self.filter_prob_enabled)
        row.addWidget(self.filter_prob_thresh)
        row.addStretch()
        peak_layout.addLayout(row)

        row2 = QHBoxLayout()
        self.filter_min_int_enabled = QCheckBox("Min raw intensity ≥")
        self.filter_min_int_enabled.setChecked(False)
        self.filter_min_int = QDoubleSpinBox()
        self.filter_min_int.setRange(0, 100000)
        self.filter_min_int.setDecimals(0)
        self.filter_min_int.setSingleStep(100)
        self.filter_min_int.setValue(0)
        row2.addWidget(self.filter_min_int_enabled)
        row2.addWidget(self.filter_min_int)
        row2.addStretch()
        peak_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.filter_per_cell_ratio_enabled = QCheckBox("Per-cell ratio ≥")
        self.filter_per_cell_ratio_enabled.setChecked(False)
        self.filter_per_cell_ratio = QDoubleSpinBox()
        self.filter_per_cell_ratio.setRange(0.5, 100.0)
        self.filter_per_cell_ratio.setDecimals(2)
        self.filter_per_cell_ratio.setSingleStep(0.1)
        self.filter_per_cell_ratio.setValue(2.0)
        row3.addWidget(self.filter_per_cell_ratio_enabled)
        row3.addWidget(self.filter_per_cell_ratio)
        row3.addWidget(QLabel("(focus mean / rest-of-cell mean)"))
        row3.addStretch()
        peak_layout.addLayout(row3)

        peak_group.setLayout(peak_layout)
        self.peak_filters_group = peak_group
        return peak_group

    def _place_peak_filters_in_preview(self, target_layout) -> None:
        section = getattr(self, "peak_filters_group", None)
        if section is None:
            return
        try:
            self._move_widget_between_layouts(section, target_layout)
        except Exception:
            pass

    def _restore_peak_filters_to_stash(self) -> None:
        section = getattr(self, "peak_filters_group", None)
        stash_layout = getattr(self, "_peak_filters_stash_layout", None)
        if section is None or stash_layout is None:
            return
        try:
            self._move_widget_between_layouts(section, stash_layout)
        except Exception:
            pass

    # ---- runtime entry points ----------------------------------------
    def preview_filter_effects(self) -> None:
        if not self.file_path.text():
            QMessageBox.warning(self, "No Selection", "Please select a mask file or folder first.")
            return
        if not Path(self.file_path.text()).exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return

        # Defer implementation to the dedicated preview module.
        from bactimate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

        show_filter_preview(self, self.file_path.text().strip())

    def filter_masks(self) -> None:
        raw_path = (self.file_path.text() or "").strip() if getattr(self, "file_path", None) else ""
        if not raw_path:
            QMessageBox.warning(self, "No Selection", "Please select a mask file or folder first.")
            return
        p = Path(raw_path)
        if not p.exists():
            QMessageBox.warning(self, "Not Found", "The selected file or folder does not exist.")
            return

        if p.is_file():
            mask_files = [p]
        else:
            mask_files = self._get_mask_files_for_feedback()
        if not mask_files:
            QMessageBox.warning(self, "No Masks Found", "No mask files were found in the selected path.")
            return

        def _safe_checked(attr: str) -> bool:
            w = getattr(self, attr, None)
            try:
                return bool(w is not None and w.isChecked())
            except RuntimeError:
                return False

        def _safe_value(attr: str, default: float) -> float:
            w = getattr(self, attr, None)
            try:
                return float(w.value()) if w is not None else float(default)
            except RuntimeError:
                return float(default)

        prob_thresh = _safe_value("filter_prob_thresh", 0.0) if _safe_checked("filter_prob_enabled") else None
        min_peak_raw = _safe_value("filter_min_int", 0.0) if _safe_checked("filter_min_int_enabled") else None
        per_cell_ratio = (
            _safe_value("filter_per_cell_ratio", 0.0) if _safe_checked("filter_per_cell_ratio_enabled") else None
        )

        progress = QProgressDialog("Filtering masks...", "Cancel", 0, len(mask_files), self)
        progress.setWindowTitle("Filtering")
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        from bactimate.gui.artifact_paths import filtered_mask_path_for_mask, filtered_foci_csv_path, raw_foci_csv_path

        try:
            from bactimate.scripts.analyze_cell_areas import calculate_cell_properties, filter_rod_shaped_cells, filter_by_size
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Filtering Error", f"Could not import filtering pipeline:\n{e}")
            return

        try:
            from bactimate.modules.foci_utils import refine_spotiflow_peaks
            from bactimate.modules.image_stack_utils import coerce_stack_to_tyx, coerce_plane_to_2d
        except Exception:
            refine_spotiflow_peaks = None
            coerce_stack_to_tyx = None
            coerce_plane_to_2d = None

        def _read_raw_foci_csv(csv_path: Path) -> list[dict]:
            if not csv_path.exists():
                return []
            rows: list[dict] = []
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

        def _write_filtered_foci_csv(out_path: Path, rows: list[dict], include_frame: bool) -> None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["y", "x", "intensity", "probability"]
            if include_frame:
                fieldnames = ["frame"] + fieldnames
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    out = {
                        "y": int(r.get("y", 0)),
                        "x": int(r.get("x", 0)),
                        "intensity": float(r.get("intensity", 0.0) or 0.0),
                        "probability": ("" if r.get("probability") is None else float(r.get("probability"))),
                    }
                    if include_frame:
                        out["frame"] = int(r.get("frame", 0))
                    w.writerow(out)

        for i, mask_path in enumerate(mask_files):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"Filtering {mask_path.name} ({i+1}/{len(mask_files)})...")
            progress.setValue(i)
            QApplication.processEvents()

            try:
                masks = safe_imread(str(mask_path))
                masks = np.asarray(masks)
                if masks.ndim == 2:
                    masks = masks[np.newaxis, :, :]
            except Exception as e:
                if hasattr(self.parent_window, "log_output"):
                    self.parent_window.log_output(f"[Filtering] Failed to load mask {mask_path.name}: {e}\n")
                continue

            out = np.zeros_like(masks, dtype=np.int32)
            for fi in range(masks.shape[0]):
                frame_mask = np.asarray(masks[fi], dtype=np.int32)
                try:
                    props = calculate_cell_properties(
                        frame_mask,
                        pixel_size=float(self.pixel_size.value()) if hasattr(self, "pixel_size") else 0.108,
                        exclude_border=bool(getattr(self, "exclude_border_cb", None) and self.exclude_border_cb.isChecked()),
                    )
                    rod = filter_rod_shaped_cells(
                        props,
                        min_aspect_ratio=float(self.min_aspect_ratio.value()),
                        max_aspect_ratio=float(self.max_aspect_ratio.value()),
                        min_eccentricity=float(self.min_eccentricity.value()),
                        min_solidity=float(self.min_solidity.value()),
                    )
                    filtered = filter_by_size(
                        rod,
                        min_area_um2=(float(self.min_area.value()) if getattr(self, "min_area", None) and self.min_area.value() > 0 else None),
                        max_area_um2=(float(self.max_area.value()) if getattr(self, "max_area", None) and self.max_area.value() > 0 else None),
                        min_width_um=(float(self.min_width.value()) if getattr(self, "min_width", None) and self.min_width.value() > 0 else None),
                        max_width_um=(float(self.max_width.value()) if getattr(self, "max_width", None) and self.max_width.value() > 0 else None),
                    )
                    accepted = {int(c["label"]) for c in filtered}
                    if accepted:
                        out_frame = np.where(np.isin(frame_mask, np.array(list(accepted), dtype=np.int32)), frame_mask, 0)
                    else:
                        out_frame = np.zeros_like(frame_mask, dtype=np.int32)
                    out[fi] = out_frame
                except Exception:
                    out[fi] = frame_mask

            out_path = filtered_mask_path_for_mask(mask_path)
            try:
                tifffile.imwrite(str(out_path), out.astype(np.int32))
                if hasattr(self.parent_window, "log_output"):
                    self.parent_window.log_output(f"[Filtering] Wrote filtered mask: {out_path}\n")
            except Exception as e:
                if hasattr(self.parent_window, "log_output"):
                    self.parent_window.log_output(f"[Filtering] Failed to write filtered mask {out_path.name}: {e}\n")

            # Filter foci when possible:
            try:
                foci_sources = self._resolve_foci_paths_for_mask(Path(mask_path))
            except Exception:
                foci_sources = []
            if not foci_sources:
                continue
            if refine_spotiflow_peaks is None or coerce_stack_to_tyx is None or coerce_plane_to_2d is None:
                continue
            for src in foci_sources:
                try:
                    from bactimate.gui.artifact_paths import foci_csv_path

                    csv_new = foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
                    csv_legacy = raw_foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
                    raw_rows = _read_raw_foci_csv(csv_new)
                    if not raw_rows:
                        raw_rows = _read_raw_foci_csv(csv_legacy)
                    if not raw_rows:
                        continue
                    foci_stack = np.asarray(safe_imread(str(src)))
                    foci_tyx = coerce_stack_to_tyx(foci_stack)
                    if foci_tyx is None or foci_tyx.size == 0:
                        continue
                    include_frame = bool(foci_tyx.shape[0] > 1 or any(int(r.get("frame", 0)) != 0 for r in raw_rows))

                    # Apply CSV-only filters first (probability/min raw)
                    filtered_rows = []
                    for r in raw_rows:
                        if prob_thresh is not None and r.get("probability") is not None and float(r["probability"]) < float(prob_thresh):
                            continue
                        if min_peak_raw is not None and float(r.get("intensity", 0.0) or 0.0) < float(min_peak_raw):
                            continue
                        filtered_rows.append(r)

                    # Per-cell ratio filter requires access to the image + segmentation; re-use refine_spotiflow_peaks helper
                    if per_cell_ratio is not None:
                        # refine_spotiflow_peaks expects peaks list; we pass (y,x,intensity,prob,frame)-like dict rows
                        filtered_rows = refine_spotiflow_peaks(
                            peaks=filtered_rows,
                            fi_stack=foci_tyx,
                            mask_stack=out if out.ndim == 3 else out[np.newaxis, :, :],
                            min_rel=None,
                            min_sep_px=None,
                            min_size_px=None,
                            max_size_px=None,
                            blob_threshold=None,
                            min_peak_raw=None,
                            min_focus_mean=None,
                            per_cell_ratio=float(per_cell_ratio),
                        )

                    out_csv = filtered_foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
                    _write_filtered_foci_csv(out_csv, filtered_rows, include_frame=include_frame)
                    if hasattr(self.parent_window, "log_output"):
                        self.parent_window.log_output(f"[Filtering] Wrote filtered foci CSV: {out_csv}\n")
                except Exception as e:
                    if hasattr(self.parent_window, "log_output"):
                        self.parent_window.log_output(f"[Filtering] Foci filtering failed for {src.name}: {e}\n")
                    continue

        try:
            progress.setValue(len(mask_files))
            progress.close()
        except Exception:
            pass
        QMessageBox.information(self, "Filtering complete", "Filtering finished. Filtered artifacts were written next to the masks.")

    # ---- session params extension ------------------------------------
    def get_session_params(self) -> dict[str, Any]:
        d = super().get_session_params()
        d.update(
            {
                "filter_prob_enabled": bool(self.filter_prob_enabled.isChecked()),
                "filter_prob_thresh": float(self.filter_prob_thresh.value()),
                "filter_min_int_enabled": bool(self.filter_min_int_enabled.isChecked()),
                "filter_min_int": float(self.filter_min_int.value()),
                "filter_per_cell_ratio_enabled": bool(self.filter_per_cell_ratio_enabled.isChecked()),
                "filter_per_cell_ratio": float(self.filter_per_cell_ratio.value()),
            }
        )
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        super().set_session_params(data)
        apply_to_widgets(
            data,
            [
                ("filter_prob_enabled", self.filter_prob_enabled),
                ("filter_prob_thresh", self.filter_prob_thresh),
                ("filter_min_int_enabled", self.filter_min_int_enabled),
                ("filter_min_int", self.filter_min_int),
                ("filter_per_cell_ratio_enabled", self.filter_per_cell_ratio_enabled),
                ("filter_per_cell_ratio", self.filter_per_cell_ratio),
            ],
        )

