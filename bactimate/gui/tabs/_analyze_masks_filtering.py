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
        return "Open Preview to tune filtering parameters, then run filtering from this tab."

    def _build_stage_controls(self, scroll_layout: QVBoxLayout) -> None:
        # Intentionally keep the main tab minimal: all parameter controls live in the Preview.
        return

    def _wire_stage_buttons(self) -> None:
        self._filtering_in_progress = False
        self._cancel_filtering_requested = False
        self.preview_action_btn.clicked.connect(self.preview_filter_effects)
        self.primary_action_btn.clicked.connect(self._on_filter_or_cancel_clicked)

    def _on_filter_or_cancel_clicked(self) -> None:
        # Stateful action button: acts as Cancel while filtering is running.
        if bool(getattr(self, "_filtering_in_progress", False)):
            self._cancel_filtering_requested = True
            try:
                self.primary_action_btn.setEnabled(False)
                self.primary_action_btn.setText("Cancelling…")
            except Exception:
                pass
            return
        self.filter_masks()

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
        # Legacy hook (preview used to reparent widgets). Intentionally no-op:
        # filtering controls remain visible in the main Filtering section.
        return

    def _restore_peak_filters_to_stash(self) -> None:
        # Legacy hook (preview used to reparent widgets). Intentionally no-op.
        return

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

    # ---- settings snapshot (owned state) -----------------------------
    def _default_filter_settings(self) -> dict[str, Any]:
        return {
            # Cell / mask filters
            "pixel_size": 0.108,
            "min_aspect_ratio": 1.5,
            "max_aspect_ratio": 10.0,
            "min_eccentricity": 0.5,
            "min_solidity": 0.7,
            "min_area_um2": None,
            "max_area_um2": None,
            "min_width_um": None,
            "max_width_um": None,
            "min_length_um": None,
            "exclude_border": False,
            # Foci CSV filters
            "foci_prob_enabled": False,
            "foci_prob_thresh": 0.5,
            "foci_min_int_enabled": False,
            "foci_min_int": 0.0,
            "foci_per_cell_ratio_enabled": False,
            "foci_per_cell_ratio": 2.0,
            "foci_exclude_outside_cells": False,
        }

    def _ensure_filter_settings(self) -> dict[str, Any]:
        fs = getattr(self, "_filter_settings", None)
        if not isinstance(fs, dict):
            fs = self._default_filter_settings()
            self._filter_settings = dict(fs)
        # Backfill missing keys for forward compatibility.
        for k, v in self._default_filter_settings().items():
            if k not in fs:
                fs[k] = v
        return fs

    def get_filter_settings(self) -> dict[str, Any]:
        """Return a stable snapshot used by both preview and batch filtering."""
        fs = dict(self._ensure_filter_settings())
        # Normalize numeric optionals.
        for k in ("min_area_um2", "max_area_um2", "min_width_um", "max_width_um", "min_length_um"):
            v = fs.get(k, None)
            try:
                if v is None:
                    fs[k] = None
                else:
                    fv = float(v)
                    fs[k] = fv if fv > 0 else None
            except Exception:
                fs[k] = None
        try:
            fs["exclude_border"] = bool(fs.get("exclude_border", False))
        except Exception:
            fs["exclude_border"] = False
        return fs

    def set_filter_settings(self, data: dict[str, Any]) -> None:
        fs = self._ensure_filter_settings()
        if not isinstance(data, dict):
            return
        for k in self._default_filter_settings().keys():
            if k in data:
                fs[k] = data.get(k)

    def reset_filter_settings_defaults(self) -> None:
        self._filter_settings = self._default_filter_settings()

    def filter_masks(self) -> None:
        if bool(getattr(self, "_filtering_in_progress", False)):
            return
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

        fs = self.get_filter_settings()
        prob_thresh = float(fs.get("foci_prob_thresh", 0.5)) if bool(fs.get("foci_prob_enabled", False)) else None
        min_peak_raw = float(fs.get("foci_min_int", 0.0)) if bool(fs.get("foci_min_int_enabled", False)) else None
        per_cell_ratio = (
            float(fs.get("foci_per_cell_ratio", 2.0)) if bool(fs.get("foci_per_cell_ratio_enabled", False)) else None
        )
        exclude_outside_cells = bool(fs.get("foci_exclude_outside_cells", False))

        self._filtering_in_progress = True
        self._cancel_filtering_requested = False
        try:
            self.preview_action_btn.setEnabled(False)
        except Exception:
            pass
        try:
            self.primary_action_btn.setText("Cancel Filtering")
            self.primary_action_btn.setToolTip("Stop the current filtering job.")
        except Exception:
            pass

        progress = QProgressDialog("Filtering masks...", None, 0, len(mask_files), self)
        progress.setWindowTitle("Filtering")
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        from bactimate.gui.artifact_paths import (
            filtered_mask_path_for_mask,
            filtered_foci_csv_path,
            legacy_foci_csv_candidates,
            raw_foci_csv_path,
        )

        try:
            from bactimate.scripts.analyze_cell_areas import calculate_cell_properties, filter_rod_shaped_cells, filter_by_size
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Filtering Error", f"Could not import filtering pipeline:\n{e}")
            self._filtering_in_progress = False
            try:
                self.primary_action_btn.setText(self._primary_button_label())
                self.primary_action_btn.setToolTip(self._primary_button_tooltip())
                self.primary_action_btn.setEnabled(True)
                self.preview_action_btn.setEnabled(True)
            except Exception:
                pass
            return

        try:
            from bactimate.modules.foci_utils import refine_spotiflow_peaks
            from bactimate.modules.image_stack_utils import coerce_stack_to_tyx, coerce_plane_to_2d
        except Exception:
            refine_spotiflow_peaks = None
            coerce_stack_to_tyx = None
            coerce_plane_to_2d = None
        helper_skip_logged = False

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
            if bool(getattr(self, "_cancel_filtering_requested", False)):
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
                        pixel_size=float(fs.get("pixel_size", 0.108) or 0.108),
                        exclude_border=bool(fs.get("exclude_border", False)),
                    )
                    rod = filter_rod_shaped_cells(
                        props,
                        min_aspect_ratio=float(fs.get("min_aspect_ratio", 1.5) or 1.5),
                        max_aspect_ratio=float(fs.get("max_aspect_ratio", 10.0) or 10.0),
                        min_eccentricity=float(fs.get("min_eccentricity", 0.5) or 0.5),
                        min_solidity=float(fs.get("min_solidity", 0.7) or 0.7),
                    )
                    filtered = filter_by_size(
                        rod,
                        min_area_um2=fs.get("min_area_um2", None),
                        max_area_um2=fs.get("max_area_um2", None),
                        min_width_um=fs.get("min_width_um", None),
                        max_width_um=fs.get("max_width_um", None),
                        min_length_um=fs.get("min_length_um", None),
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
                if not helper_skip_logged and hasattr(self.parent_window, "log_output"):
                    self.parent_window.log_output(
                        "[Filtering] Skipping foci CSV filtering: required helper imports are unavailable "
                        "(refine_spotiflow_peaks/coerce_stack_to_tyx/coerce_plane_to_2d).\n"
                    )
                    helper_skip_logged = True
                continue
            for src in foci_sources:
                try:
                    from bactimate.gui.artifact_paths import foci_csv_path

                    csv_new = foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
                    csv_legacy = raw_foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
                    raw_rows = _read_raw_foci_csv(csv_new)
                    chosen_csv = csv_new if raw_rows else None
                    if not raw_rows:
                        raw_rows = _read_raw_foci_csv(csv_legacy)
                        if raw_rows:
                            chosen_csv = csv_legacy
                    if not raw_rows:
                        for cand in legacy_foci_csv_candidates(mask_path.parent, mask_path=mask_path, foci_source_path=src):
                            raw_rows = _read_raw_foci_csv(cand)
                            if raw_rows:
                                chosen_csv = cand
                                break
                    if not raw_rows:
                        continue
                    if chosen_csv is not None and chosen_csv not in (csv_new, csv_legacy):
                        if hasattr(self.parent_window, "log_output"):
                            self.parent_window.log_output(
                                f"[Filtering] Using legacy foci CSV fallback: {chosen_csv.name}\n"
                            )
                    foci_stack = np.asarray(safe_imread(str(src)))
                    foci_tyx = coerce_stack_to_tyx(foci_stack)
                    if foci_tyx is None or foci_tyx.size == 0:
                        continue
                    include_frame = bool(foci_tyx.shape[0] > 1 or any(int(r.get("frame", 0)) != 0 for r in raw_rows))

                    # Apply CSV-only filters first (probability/min raw)
                    filtered_rows = []
                    for r in raw_rows:
                        try:
                            fi_idx = int(float(r.get("frame", 0) or 0))
                        except Exception:
                            fi_idx = 0
                        if exclude_outside_cells:
                            try:
                                mask_idx = max(0, min(int(fi_idx), int(out.shape[0]) - 1))
                                y = int(float(r.get("y", 0) or 0))
                                x = int(float(r.get("x", 0) or 0))
                                h, w = out[mask_idx].shape[:2]
                                if y < 0 or y >= h or x < 0 or x >= w:
                                    continue
                                if int(out[mask_idx, y, x]) <= 0:
                                    continue
                            except Exception:
                                continue
                        if prob_thresh is not None and r.get("probability") is not None and float(r["probability"]) < float(prob_thresh):
                            continue
                        if min_peak_raw is not None and float(r.get("intensity", 0.0) or 0.0) < float(min_peak_raw):
                            continue
                        filtered_rows.append(r)

                    # Per-cell ratio filter requires access to the image + segmentation; re-use refine_spotiflow_peaks helper.
                    # NOTE: refine_spotiflow_peaks operates on a single 2D plane + 2D label image, so we apply it
                    # frame-by-frame to match the per-frame semantics used by the live preview window.
                    if per_cell_ratio is not None:
                        mask_stack_3d = out if out.ndim == 3 else out[np.newaxis, :, :]
                        per_frame_groups: dict[int, list[dict]] = {}
                        for r in filtered_rows:
                            try:
                                fi_idx = int(float(r.get("frame", 0) or 0))
                            except Exception:
                                fi_idx = 0
                            per_frame_groups.setdefault(fi_idx, []).append(r)

                        refined_rows: list[dict] = []
                        n_frames = int(foci_tyx.shape[0])
                        for fi_idx, group in per_frame_groups.items():
                            f_clamped = max(0, min(int(fi_idx), n_frames - 1))
                            try:
                                plane = coerce_plane_to_2d(foci_tyx[f_clamped])
                            except Exception:
                                refined_rows.extend(group)
                                continue
                            mask_idx = max(0, min(int(fi_idx), int(mask_stack_3d.shape[0] - 1)))
                            label_plane = np.asarray(mask_stack_3d[mask_idx], dtype=np.int32)
                            try:
                                refined = refine_spotiflow_peaks(
                                    np.asarray(plane),
                                    list(group),
                                    labeled_mask=label_plane,
                                    per_cell_ratio=float(per_cell_ratio),
                                )
                            except Exception:
                                refined = list(group)
                            for r in refined:
                                rr = dict(r)
                                rr.setdefault("frame", fi_idx)
                                refined_rows.append(rr)
                        filtered_rows = refined_rows

                    out_csv = filtered_foci_csv_path(mask_path.parent, mask_path=mask_path, foci_source_path=src)
                    _write_filtered_foci_csv(out_csv, filtered_rows, include_frame=include_frame)
                    if hasattr(self.parent_window, "log_output"):
                        self.parent_window.log_output(f"[Filtering] Wrote filtered foci CSV: {out_csv}\n")
                    # Best-effort: also write a filtered overlay artifact for visual inspection.
                    try:
                        from bactimate.gui.foci_overlay_artifacts import (
                            filtered_overlay_path_for_filtered_csv,
                            write_foci_overlay_tif,
                        )

                        overlay_path = filtered_overlay_path_for_filtered_csv(out_csv)
                        ok = write_foci_overlay_tif(
                            foci_image_path=Path(src),
                            output_path=overlay_path,
                            rows=filtered_rows,
                            mask_path=out_path if Path(out_path).exists() else None,
                        )
                        if ok and hasattr(self.parent_window, "log_output"):
                            self.parent_window.log_output(f"[Filtering] Wrote filtered overlay: {overlay_path}\n")
                    except Exception:
                        pass
                except Exception as e:
                    if hasattr(self.parent_window, "log_output"):
                        self.parent_window.log_output(f"[Filtering] Foci filtering failed for {src.name}: {e}\n")
                    continue

        try:
            progress.setValue(len(mask_files))
            progress.close()
        except Exception:
            pass
        cancelled = bool(getattr(self, "_cancel_filtering_requested", False))
        self._filtering_in_progress = False
        try:
            self.primary_action_btn.setText(self._primary_button_label())
            self.primary_action_btn.setToolTip(self._primary_button_tooltip())
            self.primary_action_btn.setEnabled(True)
            self.preview_action_btn.setEnabled(True)
        except Exception:
            pass
        if cancelled:
            if hasattr(self.parent_window, "log_output"):
                self.parent_window.log_output("[Filtering] Cancelled by user.\n")
            QMessageBox.information(self, "Filtering cancelled", "Filtering was cancelled.")
        else:
            output_folder = None
            try:
                output_dirs = {Path(m).parent.resolve() for m in mask_files}
                if len(output_dirs) == 1:
                    output_folder = next(iter(output_dirs))
                else:
                    selected_path = Path(raw_path)
                    output_folder = selected_path if selected_path.is_dir() else selected_path.parent
            except Exception:
                output_folder = None
            self._show_finished_dialog_with_output_folder(
                title="Filtering complete",
                message="Filtering finished. Filtered artifacts were written next to the masks.",
                output_folder=output_folder,
            )

    # ---- session params extension ------------------------------------
    def get_session_params(self) -> dict[str, Any]:
        d = super().get_session_params()
        fs = self.get_filter_settings()
        # Persist filter settings explicitly so they survive moving controls into preview.
        d["filter_settings"] = dict(fs)
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        super().set_session_params(data)
        # Preferred: unified snapshot dict.
        fs = data.get("filter_settings", None)
        if isinstance(fs, dict):
            self.set_filter_settings(fs)
            return
        # Backward-compat for older sessions.
        legacy = {}
        for k in (
            "filter_prob_enabled",
            "filter_prob_thresh",
            "filter_min_int_enabled",
            "filter_min_int",
            "filter_per_cell_ratio_enabled",
            "filter_per_cell_ratio",
        ):
            if k in data:
                legacy[k] = data.get(k)
        if legacy:
            self.set_filter_settings(
                {
                    "foci_prob_enabled": bool(legacy.get("filter_prob_enabled", False)),
                    "foci_prob_thresh": float(legacy.get("filter_prob_thresh", 0.5) or 0.5),
                    "foci_min_int_enabled": bool(legacy.get("filter_min_int_enabled", False)),
                    "foci_min_int": float(legacy.get("filter_min_int", 0.0) or 0.0),
                    "foci_per_cell_ratio_enabled": bool(legacy.get("filter_per_cell_ratio_enabled", False)),
                    "foci_per_cell_ratio": float(legacy.get("filter_per_cell_ratio", 2.0) or 2.0),
                }
            )
