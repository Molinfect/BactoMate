"""
Background thread for Analyze Masks → Filter Effects Preview.

Runs the same filter pipeline as the main analysis plus build_preview_vis off the GUI thread.
Supports cooperative cancellation: if the user requests a newer preview while this worker runs,
it exits early (skips heavy overlay work) so CPU is not wasted on stale results.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from bactomate.gui.analyze_masks_preview_vis import build_preview_vis


def _friendly_foci_phase_label(msg: str) -> str:
    """Map internal pipeline phase names to user-facing step labels."""
    key = str(msg or "").strip()
    aliases = {
        "Applying CSV filters": "Applying peak filters",
        "Applying peak filters": "Applying peak filters",
        "Growing foci blobs (passing)": "Growing passing foci blobs",
        "Growing passing foci blobs": "Growing passing foci blobs",
        "Measuring foci blobs": "Measuring foci metrics",
        "Measuring foci metrics": "Measuring foci metrics",
        "Scoring failed foci": "Classifying failed foci",
        "Classifying failed foci": "Classifying failed foci",
    }
    return aliases.get(key, key)


class AnalyzeMasksPreviewWorker(QObject):
    progress = pyqtSignal(int, str, str)  # percent, step title, detail line
    # result_ready: rid, fid, acc, rej, all, n_obj, err, vis, peaks, n_foci, n_foci_drawn,
    #   raw_median_fi, min_rel, mode_txt, pass_peaks, fail_peaks, foci_channels
    result_ready = pyqtSignal(
        int, int, list, list, list, int, str, object, object, int, int, object, object, str, list, list, object
    )

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
        foci_apply: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        # Named so any Qt fatal message ("QThread: Destroyed while thread '<name>' is still running")
        # points at the right class instead of an empty string.
        self.setObjectName(type(self).__name__)
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
        self._foci_apply = dict(foci_apply) if foci_apply else None
        self._last_progress_pct = 0
        self._last_emitted_progress_pct = -1
        self._last_progress_emit_ts = 0.0
        self._min_progress_emit_s = 0.05
        self._pending_progress: Optional[tuple[int, str, str]] = None

    def _superseded(self) -> bool:
        if self._latest_request_id is None:
            return False
        try:
            return int(self._latest_request_id()) != int(self._request_id)
        except Exception:
            return False

    def _emit_superseded(self) -> None:
        """Emit a result the UI will ignore (rid != current), triggering a fresh preview."""
        self.result_ready.emit(
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
            0,
            None,
            None,
            "",
            [],
            [],
            {},
        )

    def _emit_progress(self, pct: int, step: str, detail: str = "") -> None:
        """Emit overall progress; never decrease percent (avoids 92% → 88% regressions)."""
        try:
            pct_i = max(int(self._last_progress_pct), max(0, min(100, int(pct))))
        except Exception:
            pct_i = max(0, min(100, int(pct)))
        self._last_progress_pct = pct_i
        step_txt = str(step or "Working…")
        detail_txt = str(detail or "")
        now = time.monotonic()
        force_emit = pct_i >= 100 or pct_i != int(self._last_emitted_progress_pct)
        if not force_emit and (now - self._last_progress_emit_ts) < self._min_progress_emit_s:
            self._pending_progress = (pct_i, step_txt, detail_txt)
            return
        self._pending_progress = None
        self._last_progress_emit_ts = now
        self._last_emitted_progress_pct = pct_i
        self.progress.emit(pct_i, step_txt, detail_txt)

    def flush_progress(self) -> None:
        """Emit the most recent throttled progress update (e.g. before worker exit)."""
        pending = self._pending_progress
        if pending is None:
            return
        pct_i, step_txt, detail_txt = pending
        self._pending_progress = None
        self._last_progress_emit_ts = time.monotonic()
        self._last_emitted_progress_pct = int(pct_i)
        self.progress.emit(int(pct_i), str(step_txt), str(detail_txt))

    def _step(self, idx: int, total: int, title: str) -> str:
        try:
            i = max(1, int(idx))
            t = max(1, int(total))
        except Exception:
            i, t = 1, 1
        return f"Step {i}/{t} · {title}"

    def _pct_in_step(self, step: int, sub_frac: float = 0.0) -> int:
        """Map step 1–4 and a 0–1 fraction within that step to an overall 0–100%."""
        bounds = {1: (2, 8), 2: (8, 55), 3: (55, 92), 4: (92, 100)}
        lo, hi = bounds.get(int(step), (0, 100))
        frac = max(0.0, min(1.0, float(sub_frac)))
        return int(round(float(lo) + (float(hi) - float(lo)) * frac))

    def _score_foci_channel_jobs(
        self,
        jobs: list[dict[str, Any]],
        mask_work: np.ndarray,
        accepted_labels: set[int],
    ) -> dict[str, dict[str, list]]:
        from bactomate.gui.foci_filter_pipeline import compute_foci_pass_fail_for_frame

        out: dict[str, dict[str, list]] = {}
        n_jobs = max(1, len(jobs))
        # Parallelize when multiple channels are present. We avoid emitting fine-grained
        # per-row progress from worker threads; instead we emit channel-level milestones
        # from this worker thread only (Qt signal emission stays thread-consistent).
        if len(jobs) > 1:
            acc = {int(lbl) for lbl in accepted_labels if int(lbl) > 0}
            ch_names = [str(j.get("channel_key") or "_default") for j in jobs]
            # Keys are casefolded settings keys; progress text shows the readable channel tag.
            ch_labels = {str(j.get("channel_key") or "_default"): str(j.get("channel_label") or j.get("channel_key") or "_default") for j in jobs}
            pending_ch: set[str] = set(ch_names)
            finished_ch: list[str] = []
            par_lock = threading.Lock()

            def _emit_parallel_status(
                pct: int,
                *,
                just_finished: str | None = None,
                pass_n: int = 0,
                fail_n: int = 0,
            ) -> None:
                with par_lock:
                    if just_finished:
                        pending_ch.discard(just_finished)
                        if just_finished not in finished_ch:
                            finished_ch.append(just_finished)
                    running = sorted(ch_labels.get(k, k) for k in pending_ch)
                    done = [ch_labels.get(k, k) for k in finished_ch]
                    n_done = len(done)
                    if running:
                        step = self._step(
                            3,
                            4,
                            f"Foci · scoring {', '.join(running)}",
                        )
                    elif n_done >= n_jobs:
                        step = self._step(3, 4, "Foci · all channels complete")
                    else:
                        step = self._step(3, 4, "Foci · scoring channels")
                    parts: list[str] = []
                    if just_finished:
                        parts.append(
                            f"Finished {ch_labels.get(just_finished, just_finished)}: {pass_n:,} pass, {fail_n:,} fail"
                        )
                    if running:
                        parts.append(f"Running: {', '.join(running)}")
                    if done:
                        parts.append(f"Done: {', '.join(done)}")
                    if not running and n_done < n_jobs:
                        parts.append(f"{n_done} of {n_jobs} channels complete")
                    pct_out = int(pct)
                    if not running and n_done >= n_jobs:
                        pct_out = max(pct_out, self._pct_in_step(3, 1.0))
                    self._emit_progress(
                        pct_out,
                        step,
                        " · ".join(parts) if parts else f"{n_jobs} channels",
                    )

            _emit_parallel_status(
                self._pct_in_step(3, 0.05),
                pass_n=0,
                fail_n=0,
            )

            def _run_one(job0: dict[str, Any]) -> tuple[str, list, list]:
                ch_key0 = str(job0.get("channel_key") or "_default")
                rows0 = list(job0.get("frame_rows") or [])
                plane0 = job0.get("foci_plane")
                fs0 = dict(job0.get("settings") or {})
                if plane0 is None or not rows0:
                    return ch_key0, [], []
                try:
                    ch_pass0, ch_fail0 = compute_foci_pass_fail_for_frame(
                        rows0,
                        fs0,
                        mask_2d=mask_work,
                        foci_plane=np.asarray(plane0),
                        accepted_cell_labels=acc,
                        progress_callback=None,
                    )
                except Exception:
                    ch_pass0, ch_fail0 = [], []
                return ch_key0, list(ch_pass0), list(ch_fail0)

            max_workers = min(4, max(2, len(jobs)))
            done_jobs = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_run_one, dict(j)): j for j in jobs}
                for fut in as_completed(list(futs.keys())):
                    if self._superseded():
                        break
                    try:
                        ch_key, ch_pass, ch_fail = fut.result()
                    except Exception:
                        continue
                    out[str(ch_key)] = {"pass_peaks": list(ch_pass), "fail_peaks": list(ch_fail)}
                    done_jobs += 1
                    if done_jobs >= n_jobs:
                        pct = self._pct_in_step(3, 1.0)
                    else:
                        pct = self._pct_in_step(
                            3, 0.05 + 0.9 * float(done_jobs) / float(n_jobs)
                        )
                    _emit_parallel_status(
                        pct,
                        just_finished=str(ch_key),
                        pass_n=len(ch_pass),
                        fail_n=len(ch_fail),
                    )
            _emit_parallel_status(self._pct_in_step(3, 1.0))
            return out

        for ji, job in enumerate(jobs):
            if self._superseded():
                break
            ch_key = str(job.get("channel_key") or "_default")
            ch_lbl = str(job.get("channel_label") or ch_key)
            rows = list(job.get("frame_rows") or [])
            foci_plane = job.get("foci_plane")
            fs = dict(job.get("settings") or {})
            n_rows = len(rows)
            base_pct = self._pct_in_step(3, float(ji) / float(n_jobs))
            end_band = self._pct_in_step(3, float(ji + 1) / float(n_jobs))
            span = max(1, end_band - base_pct)
            if foci_plane is None or not rows:
                out[ch_key] = {"pass_peaks": [], "fail_peaks": []}
                self._emit_progress(
                    base_pct,
                    f"Foci · {ch_lbl}",
                    f"Channel {ji + 1} of {n_jobs} — no peaks on this frame",
                )
                continue
            plane = np.asarray(foci_plane)
            acc = {int(lbl) for lbl in accepted_labels if int(lbl) > 0}
            last_pct = [-1]
            last_done = [-1]

            def _cb(
                done: int,
                total: int,
                msg: str,
                *,
                _base=base_pct,
                _span=span,
                _key=ch_key,
                _ji=ji,
                _n_jobs=n_jobs,
            ) -> None:
                if total <= 0:
                    pct = _base
                    sub_pct = 0
                else:
                    frac = float(done) / float(total)
                    pct = min(self._pct_in_step(3, 1.0), max(_base, _base + round(_span * frac)))
                    sub_pct = int(round(100.0 * frac))
                if done <= last_done[0] and pct == last_pct[0]:
                    return
                last_done[0] = int(done)
                last_pct[0] = pct
                phase = _friendly_foci_phase_label(msg)
                step = self._step(3, 4, f"Foci · {_key} · {phase}")
                if total > 0:
                    detail = (
                        f"Channel {_ji + 1} of {_n_jobs} — "
                        f"{done:,} / {total:,} ({sub_pct}%)"
                    )
                else:
                    detail = f"Channel {_ji + 1} of {_n_jobs}"
                self._emit_progress(pct, step, detail)

            self._emit_progress(
                base_pct,
                self._step(3, 4, f"Foci · scoring {ch_lbl}"),
                f"Channel {ji + 1} of {n_jobs} — {n_rows:,} peaks on frame",
            )
            try:
                ch_pass, ch_fail = compute_foci_pass_fail_for_frame(
                    rows,
                    fs,
                    mask_2d=mask_work,
                    foci_plane=plane,
                    accepted_cell_labels=acc,
                    progress_callback=_cb,
                )
            except Exception:
                ch_pass, ch_fail = [], []
            out[ch_key] = {"pass_peaks": list(ch_pass), "fail_peaks": list(ch_fail)}
            end_pct = min(self._pct_in_step(3, 1.0), base_pct + span)
            self._emit_progress(
                end_pct,
                self._step(3, 4, f"Foci · {ch_lbl} complete"),
                f"{len(ch_pass):,} pass, {len(ch_fail):,} fail",
            )
        return out

    def run(self) -> None:
        err = ""
        accepted_list: list = []
        rejected_list: list = []
        all_list: list = []
        num_objects = 0
        vis = None
        peaks_for_hover: list = []
        n_foci_total = 0
        n_foci_drawn = 0
        raw_median_fi: Optional[float] = None
        min_rel: Optional[float] = None
        mode_txt = ""
        pass_peaks: list = []
        fail_peaks: list = []
        foci_channels: dict[str, dict[str, list]] = {}
        try:
            if self._superseded():
                self._emit_superseded()
                return

            # Own a copy for thread safety (parent may reuse buffers on next preview request)
            mask_work = np.asarray(self._mask_cleaned).copy()
            self._emit_progress(self._pct_in_step(1, 0.5), self._step(1, 4, "Preparing filter apply"), "")

            params = self._params
            reuse_cells = None
            if self._foci_apply:
                reuse_cells = self._foci_apply.get("reuse_cell_filter")
            if isinstance(reuse_cells, dict) and reuse_cells.get("accepted") is not None:
                accepted_list = [int(x) for x in (reuse_cells.get("accepted") or [])]
                rejected_list = [int(x) for x in (reuse_cells.get("rejected") or [])]
                all_list = [int(x) for x in (reuse_cells.get("all") or [])]
                num_objects = int(reuse_cells.get("n_obj") or len(all_list))
                accepted_labels = set(accepted_list)
                self._emit_progress(
                    self._pct_in_step(2, 0.35),
                    self._step(2, 4, "Cells · Reusing previous filter"),
                    (
                        "Cell settings unchanged · "
                        f"{len(accepted_list):,} accepted, {len(rejected_list):,} rejected"
                    ),
                )
            else:
                self._emit_progress(
                    self._pct_in_step(2, 0.0),
                    self._step(2, 4, "Cells · Measuring properties"),
                    "Scanning mask…",
                )
                from bactomate.scripts.analyze_cell_areas import (
                    calculate_cell_properties,
                    filter_rod_shaped_cells,
                    filter_by_size,
                )

                if self._superseded():
                    self._emit_superseded()
                    return

                def _cell_props_progress(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    # Step 2 uses ~85% of its band for per-cell measurement (8–55% overall).
                    sub = 0.85 * float(done + 1) / float(total)
                    step_pct = int(round(100.0 * float(done + 1) / float(total)))
                    self._emit_progress(
                        self._pct_in_step(2, sub),
                        self._step(2, 4, "Cells · Measuring properties"),
                        f"{done + 1:,} / {total:,} cells ({step_pct}% of step)",
                    )

                properties = calculate_cell_properties(
                    mask_work,
                    pixel_size=params["pixel_size"],
                    exclude_border=params.get("exclude_border", False),
                    progress_callback=_cell_props_progress,
                )
                self._emit_progress(
                    self._pct_in_step(2, 0.88),
                    self._step(2, 4, "Cells · Filtering morphology"),
                    "Aspect ratio, eccentricity, solidity",
                )
                rod_shaped = filter_rod_shaped_cells(
                    properties,
                    min_aspect_ratio=params["min_aspect_ratio"],
                    max_aspect_ratio=params["max_aspect_ratio"],
                    min_eccentricity=params["min_eccentricity"],
                    min_solidity=params["min_solidity"],
                )
                self._emit_progress(
                    self._pct_in_step(2, 0.94),
                    self._step(2, 4, "Cells · Applying size filter"),
                    "",
                )
                filtered = filter_by_size(
                    rod_shaped,
                    min_area_um2=params.get("min_area"),
                    max_area_um2=params.get("max_area"),
                    min_width_um=params.get("min_width"),
                    max_width_um=params.get("max_width"),
                    min_length_um=params.get("min_length"),
                    max_length_um=params.get("max_length"),
                )
                accepted_labels = {cell["label"] for cell in filtered}
                if bool(params.get("focus_filter_enabled")) and accepted_labels:
                    phase_plane = params.get("focus_phase_plane")
                    if phase_plane is not None:
                        self._emit_progress(
                            self._pct_in_step(2, 0.97),
                            self._step(2, 4, "Cells · Focus filter"),
                            "Reject out-of-focus cells",
                        )
                        try:
                            from bactomate.modules.cell_focus_score import (
                                filter_labels_by_focus,
                                score_cells_in_frame,
                            )

                            n_focus = max(1, len(accepted_labels))

                            def _focus_progress(done: int, total: int) -> None:
                                tot = max(1, int(total))
                                frac = min(1.0, float(done) / float(tot))
                                self._emit_progress(
                                    self._pct_in_step(2, 0.94 + 0.05 * frac),
                                    self._step(2, 4, "Cells · Focus filter"),
                                    f"Scoring focus {int(done):,} / {tot:,}",
                                )

                            self._emit_progress(
                                self._pct_in_step(2, 0.94),
                                self._step(2, 4, "Cells · Focus filter"),
                                f"Scoring focus 0 / {n_focus:,}",
                            )
                            scores = score_cells_in_frame(
                                np.asarray(phase_plane, dtype=np.float64),
                                mask_work,
                                accepted_labels,
                                modality=params.get("focus_modality", "phase"),
                                progress_callback=_focus_progress,
                            )
                            frac = float(params.get("focus_min_fraction_of_median", 0.5) or 0.5)
                            accepted_labels = filter_labels_by_focus(
                                scores,
                                min_fraction_of_median=frac if frac > 0 else None,
                            )
                        except Exception:
                            pass
                all_labels = set(np.unique(mask_work)) - {0}
                rejected_labels = all_labels - accepted_labels
                accepted_list = list(accepted_labels)
                rejected_list = list(rejected_labels)
                all_list = list(all_labels)
                num_objects = len(all_labels)
                complete_detail = (
                    f"{len(accepted_list):,} accepted, {len(rejected_list):,} rejected"
                )
                if self._foci_apply:
                    jobs = list(self._foci_apply.get("channels") or [])
                    if jobs:
                        ch_names = ", ".join(
                            str(j.get("channel_key") or "_default") for j in jobs
                        )
                        complete_detail += (
                            f" · Next: rescore foci ({ch_names}) — "
                            "peaks in rejected cells will fail"
                        )
                    elif self._foci_apply.get("prefilled_channels"):
                        cached = ", ".join(
                            str(k)
                            for k in (self._foci_apply.get("prefilled_channels") or {}).keys()
                        )
                        complete_detail += f" · Foci unchanged ({cached})"
                self._emit_progress(
                    self._pct_in_step(2, 1.0),
                    self._step(2, 4, "Cells · Size filter complete"),
                    complete_detail,
                )

            if self._superseded():
                self._emit_superseded()
                return

            # Optional: CSV-driven peak filtering summary for the Filtering preview.
            # This does not draw peaks; it only reports raw vs filtered counts so the
            # UI can show “foci raw→filtered” matching `Filter File/Folder`.
            try:
                dp0 = self._display_params or {}
                raw_rows = dp0.get("raw_foci_rows")
                fp = dp0.get("filter_peaks") or {}
                if isinstance(raw_rows, list) and raw_rows:
                    n_foci_total = int(len(raw_rows))
                    prob_thresh = fp.get("prob_thresh")
                    min_peak_raw = fp.get("min_peak_raw")
                    filtered_rows = []
                    for r in raw_rows:
                        try:
                            if prob_thresh is not None:
                                pr = r.get("probability")
                                if pr is not None and pr != "" and float(pr) < float(prob_thresh):
                                    continue
                            if min_peak_raw is not None:
                                if float(r.get("intensity", 0.0) or 0.0) < float(min_peak_raw):
                                    continue
                        except Exception:
                            # If a row is malformed, treat it as filtered-out.
                            continue
                        filtered_rows.append(r)
                    n_foci_drawn = int(len(filtered_rows))
            except Exception:
                pass

            if self._foci_apply:
                if self._superseded():
                    self._emit_superseded()
                    return
                fa = self._foci_apply
                prefilled = fa.get("prefilled_channels") or {}
                if isinstance(prefilled, dict):
                    for ch_key, payload in prefilled.items():
                        if isinstance(payload, dict):
                            foci_channels[str(ch_key)] = {
                                "pass_peaks": list(payload.get("pass_peaks") or []),
                                "fail_peaks": list(payload.get("fail_peaks") or []),
                            }
                jobs = list(fa.get("channels") or [])
                if not jobs and fa.get("frame_rows"):
                    jobs = [
                        {
                            "channel_key": str(fa.get("channel_key") or "_default"),
                            "frame_rows": list(fa.get("frame_rows") or []),
                            "foci_plane": fa.get("foci_plane"),
                            "settings": dict(fa.get("settings") or {}),
                        }
                    ]
                if jobs:
                    ch_list = ", ".join(
                        str(j.get("channel_key") or "_default") for j in jobs
                    )
                    reused = [
                        str(k)
                        for k, v in (prefilled or {}).items()
                        if isinstance(v, dict)
                    ]
                    detail_parts: list[str] = []
                    if reused:
                        detail_parts.append(
                            f"Reusing cached: {', '.join(reused)}"
                        )
                    detail_parts.append(f"Scoring: {ch_list}")
                    overview = " · ".join(detail_parts)
                    if reused and jobs:
                        overview = (
                            f"Reusing: {', '.join(reused)} · "
                            f"Scoring now: {ch_list}"
                        )
                    elif jobs:
                        overview = f"Scoring now: {ch_list}"
                    self._emit_progress(
                        self._pct_in_step(3, 0.0),
                        self._step(3, 4, "Foci · starting channel scoring"),
                        overview,
                    )
                    scored = self._score_foci_channel_jobs(jobs, mask_work, accepted_labels)
                    self._emit_progress(
                        self._pct_in_step(4, 0.35),
                        self._step(4, 4, "Finalizing filter results"),
                        "Packaging channel results…",
                    )
                    foci_channels.update(scored)
                    n_foci_total = sum(
                        len(list(job.get("frame_rows") or [])) for job in jobs
                    )
                    n_foci_drawn = sum(
                        len((ch or {}).get("pass_peaks") or [])
                        for ch in foci_channels.values()
                    )
                    active_key = str(fa.get("active_channel_key") or "")
                    if active_key and active_key in foci_channels:
                        pass_peaks = list(foci_channels[active_key].get("pass_peaks") or [])
                        fail_peaks = list(foci_channels[active_key].get("fail_peaks") or [])
                    elif foci_channels:
                        first = next(iter(foci_channels.values()))
                        pass_peaks = list(first.get("pass_peaks") or [])
                        fail_peaks = list(first.get("fail_peaks") or [])
                    self._emit_progress(
                        self._pct_in_step(4, 0.5),
                        self._step(4, 4, "Finalizing filter results"),
                        "All foci channels scored",
                    )
                elif isinstance(prefilled, dict) and prefilled:
                    cached = ", ".join(str(k) for k in prefilled.keys())
                    self._emit_progress(
                        self._pct_in_step(3, 0.0),
                        self._step(3, 4, "Foci · Using cached results"),
                        f"Channels unchanged: {cached}",
                    )

            if not self._skip_vis:
                self._emit_progress(
                    self._pct_in_step(4, 0.65),
                    self._step(4, 4, "Building preview overlay"),
                    "",
                )
                dp = self._display_params
                progress_cb = lambda p, m, d="": self._emit_progress(int(p), str(m), str(d))

                fi_copy: Optional[np.ndarray] = None
                if self._fi_image is not None:
                    fi_copy = np.asarray(self._fi_image).copy()
                multi_copy: Optional[list[np.ndarray]] = None
                if self._multi_foci_fi_2d:
                    multi_copy = [np.asarray(x).copy() for x in self._multi_foci_fi_2d]

                if self._superseded():
                    self._emit_superseded()
                    return

                vis, peaks_for_hover, n_foci_total, n_foci_drawn, raw_median_fi, min_rel, mode_txt = build_preview_vis(
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
            self._emit_progress(self._pct_in_step(4, 0.85), self._step(4, 4, "Finalizing results"), "")
        except Exception as e:
            err = str(e)
        self.flush_progress()
        self._emit_progress(self._pct_in_step(4, 1.0), self._step(4, 4, "Complete"), "")
        self.result_ready.emit(
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
            n_foci_drawn,
            raw_median_fi,
            min_rel,
            mode_txt,
            pass_peaks if pass_peaks is not None else [],
            fail_peaks if fail_peaks is not None else [],
            foci_channels if foci_channels is not None else {},
        )
