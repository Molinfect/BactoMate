"""
Background thread for Analyze Masks → Filter Effects Preview.

Runs the same filter pipeline as the main analysis plus build_preview_vis off the GUI thread.
Supports cooperative cancellation: if the user requests a newer preview while this worker runs,
it exits early (skips heavy overlay work) so CPU is not wasted on stale results.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from bactimate.gui.analyze_masks_preview_vis import build_preview_vis


class AnalyzeMasksPreviewWorker(QThread):
    progress = pyqtSignal(int, str)  # percent 0-100, message for progress dialog
    # finished: rid, fid, acc, rej, all, n_obj, err, vis, peaks, n_foci, raw_median_fi, min_rel, mode_txt
    finished = pyqtSignal(int, int, list, list, list, int, str, object, object, int, object, object, str)

    def __init__(
        self,
        request_id: int,
        frame_idx: int,
        mask_cleaned: np.ndarray,
        params: dict[str, Any],
        fi_image: Optional[np.ndarray],
        display_params: dict[str, Any],
        skip_vis: bool,
        multi_foci_fi_2d: Optional[list[np.ndarray]] = None,
        foci_channel_labels: Optional[list[str]] = None,
        latest_request_id: Optional[Callable[[], int]] = None,
    ):
        super().__init__()
        self._request_id = request_id
        self._frame_idx = frame_idx
        # References only; copies happen in run() after supersede checks (saves work when cancelled).
        self._mask_cleaned = mask_cleaned
        self._params = dict(params)
        self._fi_image = fi_image
        self._display_params = dict(display_params) if display_params else {}
        self._skip_vis = bool(skip_vis)
        self._multi_foci_fi_2d = multi_foci_fi_2d
        self._foci_channel_labels = foci_channel_labels
        self._latest_request_id = latest_request_id

    def _superseded(self) -> bool:
        if self._latest_request_id is None:
            return False
        try:
            return int(self._latest_request_id()) != int(self._request_id)
        except Exception:
            return False

    def _emit_superseded(self) -> None:
        """Emit a finished signal the UI will ignore (rid != current), triggering a fresh preview."""
        self.finished.emit(
            self._request_id,
            self._frame_idx,
            [],
            [],
            [],
            0,
            "",
            None,
            [],
            0,
            None,
            None,
            "",
        )

    def run(self) -> None:
        err = ""
        accepted_list: list = []
        rejected_list: list = []
        all_list: list = []
        num_objects = 0
        vis = None
        peaks_for_hover: list = []
        n_foci_total = 0
        raw_median_fi: Optional[float] = None
        min_rel: Optional[float] = None
        mode_txt = ""
        try:
            if self._superseded():
                self._emit_superseded()
                return

            # Own a copy for thread safety (parent may reuse buffers on next preview request)
            mask_work = np.asarray(self._mask_cleaned).copy()

            params = self._params
            self.progress.emit(20, "Computing cell properties... (20%)")
            from bactimate.scripts.analyze_cell_areas import (
                calculate_cell_properties,
                filter_rod_shaped_cells,
                filter_by_size,
            )

            if self._superseded():
                self._emit_superseded()
                return

            properties = calculate_cell_properties(
                mask_work,
                pixel_size=params["pixel_size"],
                exclude_border=params.get("exclude_border", False),
            )
            self.progress.emit(45, "Filtering shape... (45%)")
            rod_shaped = filter_rod_shaped_cells(
                properties,
                min_aspect_ratio=params["min_aspect_ratio"],
                max_aspect_ratio=params["max_aspect_ratio"],
                min_eccentricity=params["min_eccentricity"],
                min_solidity=params["min_solidity"],
            )
            self.progress.emit(70, "Applying size filter... (70%)")
            filtered = filter_by_size(
                rod_shaped,
                min_area_um2=params.get("min_area"),
                max_area_um2=params.get("max_area"),
                min_width_um=params.get("min_width"),
                max_width_um=params.get("max_width"),
            )
            accepted_labels = {cell["label"] for cell in filtered}
            all_labels = set(np.unique(mask_work)) - {0}
            rejected_labels = all_labels - accepted_labels
            accepted_list = list(accepted_labels)
            rejected_list = list(rejected_labels)
            all_list = list(all_labels)
            num_objects = len(all_labels)

            if self._superseded():
                self._emit_superseded()
                return

            self.progress.emit(85, "Building overlay... (85%)")
            if not self._skip_vis:
                dp = self._display_params
                progress_cb = lambda p, m: self.progress.emit(p, m)

                fi_copy: Optional[np.ndarray] = None
                if self._fi_image is not None:
                    fi_copy = np.asarray(self._fi_image).copy()
                multi_copy: Optional[list[np.ndarray]] = None
                if self._multi_foci_fi_2d:
                    multi_copy = [np.asarray(x).copy() for x in self._multi_foci_fi_2d]

                if self._superseded():
                    self._emit_superseded()
                    return

                vis, peaks_for_hover, n_foci_total, raw_median_fi, min_rel, mode_txt = build_preview_vis(
                    mask_work,
                    set(accepted_list),
                    set(rejected_list),
                    fi_copy,
                    dp.get("do_fi_contour", False),
                    dp.get("fi_contour_thickness_val", 1),
                    dp.get("run_foci", False),
                    dp.get("foci_snapshot"),
                    progress_callback=progress_cb,
                    multi_foci_fi_2d=multi_copy,
                    foci_channel_labels=self._foci_channel_labels,
                )
            self.progress.emit(90, "Ready to draw... (90%)")
        except Exception as e:
            err = str(e)
        self.finished.emit(
            self._request_id,
            self._frame_idx,
            accepted_list,
            rejected_list,
            all_list,
            num_objects,
            err,
            vis,
            peaks_for_hover if peaks_for_hover is not None else [],
            n_foci_total,
            raw_median_fi,
            min_rel,
            mode_txt,
        )
