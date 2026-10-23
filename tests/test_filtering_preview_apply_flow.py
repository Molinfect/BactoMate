"""End-to-end Qt integration tests for the Filtering Preview apply flow.

Regression coverage for the bug where the Filtering Preview overlay never
visibly updated after clicking Apply / Re-apply. Root cause:
``QProgressDialog.canceled`` was connected to ``_cancel_current``, and
``prog.close()`` inside ``on_result_ready`` programmatically fires that
signal -- which then bumped ``_preview_request_id`` and made the just-arrived
worker result look "superseded", so ``_last_preview`` stayed empty and
``_render`` always fell back to the unfiltered baseline.

Spec under test:
1. Opening the preview auto-applies current filters once (pass/fail overlay + hover).
2. Clicking Apply preview must populate ``_last_preview`` with worker output
   (accepted/rejected) -- the result must NOT be silently discarded.
3. The cached result must survive a subsequent ``refresh()``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QCoreApplication
    from PyQt6.QtWidgets import QApplication, QWidget
except Exception as e:  # pragma: no cover
    pytest.skip(f"PyQt6 not available: {e}", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _qapp_module():
    """Module-scoped QApplication; without this Qt aborts on Cocoa thread checks."""

    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _make_mixed_mask() -> np.ndarray:
    """A few rod cells (high aspect ratio) + a few square cells (aspect ~1)."""

    mask = np.zeros((1, 256, 256), dtype=np.int32)
    # Long thin rods (aspect ratio ~12) — should pass strict morphology filter.
    mask[0, 30:38, 30:130] = 1
    mask[0, 60:68, 30:130] = 2
    # Square blobs (aspect ratio ~1) — should fail any min_aspect_ratio > ~1.
    mask[0, 100:140, 30:70] = 3
    mask[0, 100:140, 90:130] = 4
    return mask


def _wait_until(predicate, timeout_s: float = 10.0, step_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(step_s)
    return predicate()


def _build_tab(tmp_dir: Path):
    from bactomate.gui.tabs._analyze_masks_filtering import FilteringTab

    parent = QWidget()
    parent.show()
    tab = FilteringTab(parent=parent)
    tab.file_path.setText(str(tmp_dir))
    return tab, parent


def _build_hosted_tab(tmp_dir: Path):
    from PyQt6.QtWidgets import QHBoxLayout
    from bactomate.gui.module_shell import PreviewSurface
    from bactomate.gui.tabs.analyze_masks import AnalyzeTab, FilteringTab
    from bactomate.gui.tabs.segmentation import SegmentationTab
    from bactomate.gui.tabs.unet_seg import UNetSegmentTab
    from bactomate.gui.tabs._analyze_masks_preview_filtering import _segmentation_tab_host

    parent = QWidget()
    tab = FilteringTab(parent)
    host = SegmentationTab(parent, filtering_tab=tab, analyze_tab=AnalyzeTab(parent), unet_tab=UNetSegmentTab(parent))
    surface = PreviewSurface(parent)
    layout = QHBoxLayout(parent)
    layout.addWidget(host)
    layout.addWidget(surface)
    host.attach_module_preview_mount(surface.mount_widget())
    tab.file_path.setText(str(tmp_dir))
    parent.show()
    assert _segmentation_tab_host(tab) is host
    return tab, parent, host, surface


def _write_mask_tif(tmp_dir: Path) -> Path:
    seg_dir = tmp_dir / "_segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    mask_path = seg_dir / "sample_masks.tif"
    tifffile.imwrite(str(mask_path), _make_mixed_mask().astype(np.int32))
    return mask_path


def _write_phase_tif(tmp_dir: Path, *, pixel_size_um: float, nonzero: bool = False) -> Path:
    phase_path = tmp_dir / "sample.tif"
    # Match the synthetic mask size (256x256) so preview overlay compositing is valid.
    if nonzero:
        yy, xx = np.mgrid[0:256, 0:256]
        stack = (1200 + (yy + xx) * 10).astype(np.uint16)[None, ...]
    else:
        stack = np.zeros((1, 256, 256), dtype=np.uint16)
    tifffile.imwrite(
        str(phase_path),
        stack,
        imagej=True,
        metadata={"images": 1, "frames": 1, "unit": "micron"},
        resolution=(1.0 / float(pixel_size_um), 1.0 / float(pixel_size_um)),
        resolutionunit=1,
    )
    return phase_path


def _wait_for_filter_apply(dlg, timeout_s: float = 20.0) -> bool:
    """Wait until the filter preview worker has committed ``_last_preview``."""
    return _wait_until(
        lambda: "mask_2d" in (getattr(dlg, "_last_preview", None) or {}),
        timeout_s=timeout_s,
    )


def _preview_thread_idle(dlg) -> bool:
    return dlg._filter_apply.idle


def _filter_apply_busy(dlg) -> bool:
    try:
        return bool((getattr(dlg.state(), "extra", None) or {}).get("filter_apply_busy"))
    except Exception:
        return False


def _wait_for_apply_generation(dlg, *, rid_before: int, timeout_s: float = 20.0) -> bool:
    """Wait until a new apply finished after ``rid_before`` (not just a stale cache hit)."""

    def _done() -> bool:
        rid = dlg._filter_apply.generation
        last = getattr(dlg, "_last_preview", None) or {}
        return (
            rid > int(rid_before)
            and "mask_2d" in last
            and _preview_thread_idle(dlg)
            and not _filter_apply_busy(dlg)
        )

    return _wait_until(_done, timeout_s=timeout_s)


def _open_preview(tmp_dir: Path):
    from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

    mask_path = _write_mask_tif(tmp_dir)
    tab, parent = _build_tab(tmp_dir)
    show_filter_preview(tab, str(mask_path))
    QCoreApplication.processEvents()
    dlg = getattr(tab, "_filter_preview_dialog", None)
    return tab, parent, dlg


def _open_preview_with_calibrated_phase(tmp_dir: Path, *, pixel_size_um: float):
    from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

    _write_phase_tif(tmp_dir, pixel_size_um=pixel_size_um)
    mask_path = _write_mask_tif(tmp_dir)
    tab, parent = _build_tab(tmp_dir)
    show_filter_preview(tab, str(mask_path))
    QCoreApplication.processEvents()
    dlg = getattr(tab, "_filter_preview_dialog", None)
    return tab, parent, dlg


def _shutdown(tab, parent):
    try:
        dlg = getattr(tab, "_filter_preview_dialog", None)
        if dlg is not None:
            guard = getattr(dlg, "_worker_guard", None)
            try:
                if guard is not None and guard.is_running():
                    guard.request_stop()
                    guard.wait(10000)
                if guard is not None:
                    guard.disconnect()
            except Exception:
                pass
            try:
                dlg.close()
            except Exception:
                pass
            try:
                tab._filter_preview_dialog = None
            except Exception:
                pass
    except Exception:
        pass
    try:
        parent.close()
        parent.deleteLater()
    except Exception:
        pass
    QCoreApplication.processEvents()


@pytest.fixture
def held_preview_workers(monkeypatch):
    """Hold real calculations so superseded results arrive after the next action."""
    from bactomate.gui.tabs import _analyze_masks_preview_filtering as preview

    workers = []
    real_worker = preview.AnalyzeMasksPreviewWorker

    class HeldWorker(real_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.started = threading.Event()
            self.proceed = threading.Event()
            workers.append(self)

        def run(self):
            self.started.set()
            self.proceed.wait(10)
            # Deliberately deliver obsolete results; lifecycle must reject them
            # even when cooperative worker cancellation misses its last check.
            self._latest_request_id = None
            super().run()

    monkeypatch.setattr(preview, 'AnalyzeMasksPreviewWorker', HeldWorker)
    yield workers
    for worker in workers:
        worker.proceed.set()


def test_rapid_applies_keep_only_latest_snapshot(tmp_path, held_preview_workers):
    _write_phase_tif(tmp_path, pixel_size_um=0.108)
    foci_path = tmp_path / 'sample_foci_GFP.tif'
    tifffile.imwrite(foci_path, np.ones((1, 256, 256), dtype=np.uint16))
    tab, parent, dlg = _open_preview(tmp_path)
    workers = held_preview_workers
    try:
        assert _wait_until(lambda: len(workers) == 1 and workers[0].started.is_set())
        foci_source = ('file', list(dlg._bg_paths).index(foci_path))
        for min_ar in (2.0, 3.0, 5.0):
            dlg._pf_spinboxes['min_aspect_ratio'].setValue(min_ar)
            dlg._pf_apply_current(source_key=foci_source if min_ar == 5.0 else None)
        # A later unsubmitted edit must not change the queued Apply snapshot.
        dlg._pf_spinboxes['min_aspect_ratio'].setValue(1.0)
        workers[0].proceed.set()
        assert _wait_until(lambda: len(workers) == 2 and workers[1].started.is_set())
        assert not dlg._last_preview, 'Obsolete initial result must not be committed'
        assert workers[1]._params['min_aspect_ratio'] == 5.0
        workers[1].proceed.set()
        assert _wait_until(lambda: dlg._filter_apply.idle)
        assert len(workers) == 2
        assert {3, 4}.issubset(dlg._last_preview['rejected'])
        assert dlg._last_preview['source_key'] == foci_source
    finally:
        for worker in workers:
            worker.proceed.set()
        _shutdown(tab, parent)


def test_cancel_drops_already_queued_apply(tmp_path, held_preview_workers):
    tab, parent, dlg = _open_preview(tmp_path)
    workers = held_preview_workers
    try:
        assert _wait_until(lambda: len(workers) == 1 and workers[0].started.is_set())
        dlg._pf_spinboxes['min_aspect_ratio'].setValue(5.0)
        dlg._pf_apply_current()
        dlg._progress_dialog.canceled.emit()
        workers[0].proceed.set()
        assert _wait_until(lambda: dlg._filter_apply.idle)
        assert len(workers) == 1
        assert not dlg._last_preview
        assert not _filter_apply_busy(dlg)
    finally:
        for worker in workers:
            worker.proceed.set()
        _shutdown(tab, parent)


def test_queued_apply_keeps_foci_settings_snapshot(tmp_path, held_preview_workers):
    from bactomate.gui.filter_preview_utils import foci_channel_settings_signature

    mask_path = _write_mask_tif(tmp_path)
    foci = np.ones((1, 256, 256), dtype=np.uint16)
    foci[0, 33, 40] = 1000
    tifffile.imwrite(tmp_path / 'sample_foci_GFP.tif', foci)
    (mask_path.parent / 'sample_foci_GFP.csv').write_text(
        'frame,y,x,intensity,probability,cell_id\n0,33,40,1000,0.6,1\n'
    )
    tab, parent, dlg = _open_preview(tmp_path)
    workers = held_preview_workers
    try:
        assert _wait_until(lambda: len(workers) == 1 and workers[0].started.is_set())
        dlg._pf_checkboxes['foci_prob_enabled'].setChecked(True)
        dlg._pf_spinboxes['foci_prob_thresh'].setValue(0.9)
        dlg._pf_apply_current()
        # Model later channel edits/synchronization without another Apply.
        tab.set_foci_channel_settings('GFP', {'foci_prob_thresh': 0.1})
        workers[0].proceed.set()
        assert _wait_until(lambda: len(workers) == 2 and workers[1].started.is_set())
        settings = workers[1]._foci_apply['channels'][0]['settings']
        assert settings['foci_prob_thresh'] == 0.9
        workers[1].proceed.set()
        assert _wait_until(lambda: dlg._filter_apply.idle)
        channel = dlg._last_preview['foci_channels']['GFP']
        assert not channel['pass_peaks']
        assert len(channel['fail_peaks']) == 1
        assert dlg._last_preview['foci_channel_sigs']['GFP'] == foci_channel_settings_signature(settings)
    finally:
        for worker in workers:
            worker.proceed.set()
        _shutdown(tab, parent)


def test_failed_apply_start_returns_to_idle():
    from PyQt6.QtWidgets import QPushButton
    from bactomate.gui.filter_preview_apply import FilterPreviewApplyLifecycle
    from bactomate.gui.preview.preview_window import PreviewDialogBase

    dlg = PreviewDialogBase(title='Apply failure')
    button = QPushButton()

    def fail_to_start(*_args):
        raise RuntimeError('Synthetic preview preparation failure')

    lifecycle = FilterPreviewApplyLifecycle(dlg, button, lambda: None, fail_to_start)
    try:
        lifecycle.request(0, {}, 'settings', ('none', -1))
        assert lifecycle.idle
        assert not _filter_apply_busy(dlg)
        assert button.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_cancel_then_apply_waits_for_teardown(tmp_path, held_preview_workers):
    from bactomate.gui.filter_preview_utils import wait_for_filter_preview_apply_idle

    tab, parent, dlg = _open_preview(tmp_path)
    workers = held_preview_workers
    try:
        assert _wait_until(lambda: len(workers) == 1 and workers[0].started.is_set())
        dlg._progress_dialog.canceled.emit()
        assert not wait_for_filter_preview_apply_idle(dlg, timeout_ms=0)
        dlg._pf_spinboxes['min_aspect_ratio'].setValue(5.0)
        dlg._pf_apply_current()
        workers[0].proceed.set()
        assert _wait_until(lambda: len(workers) == 2 and workers[1].started.is_set())
        assert not dlg._last_preview
        workers[1].proceed.set()
        assert wait_for_filter_preview_apply_idle(dlg, timeout_ms=20000)
        assert {3, 4}.issubset(dlg._last_preview['rejected'])
        assert not _filter_apply_busy(dlg)
    finally:
        for worker in workers:
            worker.proceed.set()
        _shutdown(tab, parent)


def test_close_discards_stale_result_and_queued_apply(tmp_path, held_preview_workers):
    tab, parent, dlg = _open_preview(tmp_path)
    workers = held_preview_workers
    closed = []
    dlg.finished.connect(closed.append)
    try:
        assert _wait_until(lambda: len(workers) == 1 and workers[0].started.is_set())
        dlg._pf_spinboxes['min_aspect_ratio'].setValue(5.0)
        dlg._pf_apply_current()
        dlg.close()
        assert not dlg.isVisible()
        assert not closed, 'Close must wait for the worker teardown'
        workers[0].proceed.set()
        assert _wait_until(lambda: bool(closed) and dlg._filter_apply.idle)
        assert not dlg._last_preview
        assert len(workers) == 1, 'Closing must drop queued Apply'
    finally:
        for worker in workers:
            worker.proceed.set()
        _shutdown(tab, parent)


@pytest.mark.parametrize('via_reload', [True, False])
def test_mask_switch_keeps_replacement_when_old_result_arrives(tmp_path, held_preview_workers, monkeypatch, via_reload):
    from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

    mask_path = _write_mask_tif(tmp_path)
    replacement_path = mask_path.with_name('replacement_masks.tif')
    replacement = _make_mixed_mask()
    replacement[replacement > 0] += 100
    tifffile.imwrite(replacement_path, replacement)
    tab, parent, host, surface = _build_hosted_tab(tmp_path)
    show_filter_preview(tab, str(mask_path))
    old = tab._filter_preview_dialog
    closed = []
    old.finished.connect(closed.append)

    def open_selected(path):
        tab._filter_preview_mask_path = Path(path)
        show_filter_preview(tab, str(path))

    monkeypatch.setattr(tab, '_open_filter_preview_for_mask', open_selected)
    workers = held_preview_workers
    try:
        assert _wait_until(lambda: len(workers) == 1 and workers[0].started.is_set())
        if via_reload:
            tab.reload_filter_preview_for_path(replacement_path)
        else:
            open_selected(replacement_path)
        new = tab._filter_preview_dialog
        assert new is not old
        assert _wait_until(lambda: len(workers) == 2 and workers[1].started.is_set())
        workers[0].proceed.set()
        assert _wait_until(lambda: bool(closed))
        assert not old._last_preview
        assert tab._filter_preview_dialog is new
        assert tab._filter_preview_mask_path == replacement_path
        assert new.isVisible()
        assert surface._preview_context == 'filtering'
        assert surface.mount_widget().layout().itemAt(0).widget() is new
        workers[1].proceed.set()
        assert _wait_until(lambda: new._filter_apply.idle)
        assert set(new._last_preview['accepted']) | set(new._last_preview['rejected']) == {101, 102, 103, 104}
    finally:
        for worker in workers:
            worker.proceed.set()
        _shutdown(tab, parent)


@pytest.mark.parametrize('dismissal', ['_dismiss_filtering_embedded_preview', '_dismiss_embedded_previews', 'backend'])
@pytest.mark.parametrize('idle', [False, True])
def test_real_host_dismissal_retires_filter_preview(tmp_path, held_preview_workers, dismissal, idle):
    from PyQt6 import sip
    from PyQt6.QtCore import QEvent
    from bactomate.gui.module_shell import mount_embedded_preview
    from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

    mask_path = _write_mask_tif(tmp_path)
    tab, parent, host, surface = _build_hosted_tab(tmp_path)
    show_filter_preview(tab, str(mask_path))
    dlg = tab._filter_preview_dialog
    lifecycle = dlg._filter_apply
    workers = held_preview_workers
    closed = []
    dlg.finished.connect(closed.append)
    try:
        assert _wait_until(lambda: len(workers) == 1 and workers[0].started.is_set())
        if idle:
            workers[0].proceed.set()
            assert _wait_until(lambda: lifecycle.idle)
        else:
            dlg._pf_spinboxes['min_aspect_ratio'].setValue(5.0)
            dlg._pf_apply_current()
        if dismissal == 'backend':
            host.backend_combo.setCurrentIndex(1)
        else:
            getattr(host, dismissal)()
        assert tab._filter_preview_dialog is None
        assert tab._filter_preview_mask_path is None
        assert dlg.parent() is tab, 'The host must remove a retired preview from the mount'
        assert not dlg.isVisible()
        if not idle:
            assert not closed
            assert dlg._preview_thread.isRunning()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert not sip.isdeleted(dlg), 'A running preview must survive deferred deletions'
        # A subsequent backend/analysis display must survive the old worker's finish.
        replacement = QWidget()
        mount_embedded_preview(surface.mount_widget(), replacement, preview_surface=surface)
        workers[0].proceed.set()
        assert _wait_until(lambda: bool(closed) and lifecycle.idle)
        assert len(workers) == 1, 'Host dismissal must discard queued Apply'
        if not idle:
            assert not dlg._last_preview
        assert surface.mount_widget().layout().itemAt(0).widget() is replacement
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(dlg), 'Retired previews must be deleted after teardown'
    finally:
        for worker in workers:
            worker.proceed.set()
        if not sip.isdeleted(dlg):
            dlg._worker_guard.request_stop()
            dlg._worker_guard.wait(10000)
            dlg._worker_guard.disconnect()
        _shutdown(tab, parent)


def test_preview_reads_legacy_tiff_and_foci_csv_without_rewriting(tmp_path):
    from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

    mask_path = _write_mask_tif(tmp_path)
    mask = _make_mixed_mask()
    mask[mask > 0] *= 17
    tifffile.imwrite(mask_path, mask)
    foci_path = tmp_path / 'sample_foci_GFP.tif'
    foci = np.ones_like(mask, dtype=np.uint16)
    foci[0, 33, 40] = 1000
    tifffile.imwrite(foci_path, foci)
    # Legacy flat CSV location and zero-based frame, with original cell labels.
    csv_path = mask_path.parent / 'sample_foci_GFP.csv'
    csv_path.write_text('frame,y,x,intensity,probability,cell_id\n0,33,40,1000,0.9,17\n1,33,40,2000,0.9,34\n')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (mask_path, foci_path, csv_path)}
    tab, parent = _build_tab(tmp_path)
    show_filter_preview(tab, str(mask_path))
    dlg = tab._filter_preview_dialog
    try:
        assert _wait_until(lambda: dlg._filter_apply.idle)
        assert set(dlg._last_preview['accepted']) | set(dlg._last_preview['rejected']) == {17, 34, 51, 68}
        channels = dlg._last_preview['foci_channels']
        peaks = [p for c in channels.values() for p in c['pass_peaks'] + c['fail_peaks']]
        assert len(peaks) == 1
        assert int(peaks[0]['cell_id']) == 17
        assert int(peaks[0]['frame']) == 0
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before
    finally:
        _shutdown(tab, parent)


def test_failed_mask_replacement_keeps_current_preview(tmp_path, monkeypatch):
    from bactomate.gui.tabs import _analyze_masks_filtering as filtering
    from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

    mask_path = _write_mask_tif(tmp_path)
    tab, parent, host, surface = _build_hosted_tab(tmp_path)
    show_filter_preview(tab, str(mask_path))
    dlg = tab._filter_preview_dialog
    invalid_mask = tmp_path / 'invalid_masks.tif'
    invalid_mask.write_bytes(b'Not a TIFF')
    errors = []
    monkeypatch.setattr(filtering, 'show_message_dialog', lambda *args, **kwargs: errors.append(args[1]))
    try:
        assert _wait_until(lambda: dlg._filter_apply.idle)
        current_path = tab._filter_preview_mask_path
        tab.reload_filter_preview_for_path(invalid_mask)
        assert errors == ['Load Error']
        assert tab._filter_preview_dialog is dlg
        assert tab._filter_preview_mask_path == current_path
        assert dlg.isVisible()
    finally:
        _shutdown(tab, parent)


def test_preview_open_refresh_renders_baseline_pixmap():
    """Opening the preview must paint the unfiltered baseline (no NameError in _render)."""

    _qapp()
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        tab, parent, dlg = _open_preview(tmp_dir)
        try:
            assert dlg is not None
            dlg.refresh()
            QCoreApplication.processEvents()
            pm = dlg.canvas.pixmap()
            assert pm is not None and not pm.isNull(), (
                "Initial refresh must set a canvas pixmap (regression: undefined "
                "name in unfiltered _render info_text)"
            )
        finally:
            _shutdown(tab, parent)


def test_preview_apply_populates_last_preview_with_filtered_result():
    """End-to-end: changing min_aspect_ratio in preview + Apply must produce
    a partition (some accepted, some rejected) in ``_last_preview``."""

    _qapp()
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        tab, parent, dlg = _open_preview(tmp_dir)
        try:
            assert dlg is not None
            apply_current = getattr(dlg, "_pf_apply_current", None)
            spinboxes = getattr(dlg, "_pf_spinboxes", None)
            assert apply_current is not None
            assert spinboxes is not None and "min_aspect_ratio" in spinboxes

            ok = _wait_for_filter_apply(dlg, timeout_s=20.0)
            assert ok, "Opening the preview should auto-apply current filters once"

            last_before = dict(getattr(dlg, "_last_preview", {}))
            accepted_before = set(int(x) for x in (last_before.get("accepted") or []))
            rejected_before = set(int(x) for x in (last_before.get("rejected") or []))
            assert accepted_before or rejected_before, (
                "Auto-apply should partition cells into accepted/rejected sets"
            )

            # Tighten min_aspect_ratio so the square cells fall outside the filter.
            spinboxes["min_aspect_ratio"].setValue(5.0)
            QCoreApplication.processEvents()

            apply_current()

            ok = _wait_until(
                lambda: "mask_2d" in (getattr(dlg, "_last_preview", None) or {}),
                timeout_s=20.0,
            )
            assert ok, (
                "Apply preview must populate _last_preview with the worker result. "
                "If this fails, the worker's result is being discarded before _render "
                "can use it -- typically because _preview_request_id was bumped "
                "between worker start and result_ready."
            )

            last = dict(getattr(dlg, "_last_preview", {}))
            accepted = set(int(x) for x in (last.get("accepted") or []))
            rejected = set(int(x) for x in (last.get("rejected") or []))

            assert len(rejected) > 0, (
                f"Strict filter must reject at least one cell; "
                f"accepted={sorted(accepted)} rejected={sorted(rejected)}"
            )
            assert {3, 4}.issubset(rejected), (
                f"Square cells (labels 3,4) should be rejected by min_aspect_ratio=5; "
                f"rejected={sorted(rejected)} accepted={sorted(accepted)}"
            )

            # The cached result must survive subsequent renders -- _render's cache
            # check should NOT discard a freshly-applied filter result.
            dlg.refresh()
            QCoreApplication.processEvents()
            still_last = dict(getattr(dlg, "_last_preview", {}) or {})
            assert "mask_2d" in still_last
            assert set(int(x) for x in (still_last.get("rejected") or [])) == rejected
        finally:
            _shutdown(tab, parent)


def test_preview_pixel_size_uses_metadata_from_original_phase_tiff():
    """Preview pixel-size spinbox should not stay on the hard-coded default."""

    _qapp()
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        tab, parent, dlg = _open_preview_with_calibrated_phase(tmp_dir, pixel_size_um=0.25)
        try:
            assert dlg is not None
            spinboxes = getattr(dlg, "_pf_spinboxes", None)
            assert spinboxes is not None and "pixel_size" in spinboxes
            assert spinboxes["pixel_size"].value() == pytest.approx(0.25, abs=0.001)
            assert tab.get_filter_settings()["pixel_size"] == pytest.approx(0.25, abs=0.001)
        finally:
            _shutdown(tab, parent)


def _write_foci_tif(tmp_dir: Path, *, pixel_size_um: float) -> Path:
    foci_path = tmp_dir / "sample_foci_GFP.tif"
    stack = np.zeros((1, 64, 64), dtype=np.uint16)
    tifffile.imwrite(
        str(foci_path),
        stack,
        imagej=True,
        metadata={"images": 1, "frames": 1, "unit": "micron"},
        resolution=(1.0 / float(pixel_size_um), 1.0 / float(pixel_size_um)),
        resolutionunit=1,
    )
    return foci_path


def test_preview_image_switch_keeps_pixel_size_and_reapplies_last_settings():
    """Switching View must keep last filter params and applied overlays (no metadata overwrite)."""

    _qapp()
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        _write_phase_tif(tmp_dir, pixel_size_um=0.25)
        _write_foci_tif(tmp_dir, pixel_size_um=0.50)
        mask_path = _write_mask_tif(tmp_dir)
        tab, parent = _build_tab(tmp_dir)
        from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

        show_filter_preview(tab, str(mask_path))
        QCoreApplication.processEvents()
        dlg = getattr(tab, "_filter_preview_dialog", None)
        try:
            assert dlg is not None
            ok = _wait_for_filter_apply(dlg, timeout_s=20.0)
            assert ok, "Opening the preview should auto-apply once"

            spinboxes = getattr(dlg, "_pf_spinboxes", None)
            assert spinboxes is not None and "pixel_size" in spinboxes
            assert spinboxes["pixel_size"].value() == pytest.approx(0.25, abs=0.001)

            # User-tuned params on the first preview image.
            spinboxes["pixel_size"].setValue(0.33)
            spinboxes["min_aspect_ratio"].setValue(5.0)
            QCoreApplication.processEvents()
            apply_current = getattr(dlg, "_pf_apply_current", None)
            assert apply_current is not None
            rid_before_tune = dlg._filter_apply.generation
            apply_current()
            ok = _wait_for_apply_generation(dlg, rid_before=rid_before_tune, timeout_s=20.0)
            assert ok

            last_before = dict(getattr(dlg, "_last_preview", {}) or {})
            rejected_before = set(int(x) for x in (last_before.get("rejected") or []))
            assert rejected_before, "Tight aspect filter should reject square cells"

            nav = getattr(dlg, "_nav_ctrl", None)
            assert nav is not None
            combo = nav._image_combo
            assert combo.count() >= 2, "Need phase + foci View entries to switch images"

            dlg_src_before = tuple(dlg.state().source_key or ())
            combo.setCurrentIndex(1 if combo.currentIndex() == 0 else 0)
            QCoreApplication.processEvents()
            ok = _wait_until(
                lambda: tuple(dlg.state().source_key or ()) != dlg_src_before,
                timeout_s=5.0,
            )
            assert ok, "Switching View must update the dialog source_key"

            assert spinboxes["pixel_size"].value() == pytest.approx(0.33, abs=0.001), (
                "Pixel size must stay locked when switching preview images "
                "(must not pick up the other TIFF's metadata)"
            )
            assert tab.get_filter_settings()["pixel_size"] == pytest.approx(0.33, abs=0.001)
            assert spinboxes["min_aspect_ratio"].value() == pytest.approx(5.0, abs=0.01)

            last_after = dict(getattr(dlg, "_last_preview", {}) or {})
            rejected_after = set(int(x) for x in (last_after.get("rejected") or []))
            assert rejected_after == rejected_before, (
                "View switch should keep the last applied cell-filter outcome "
                "(background swap only; no silent unfilter)"
            )
        finally:
            _shutdown(tab, parent)


def test_progress_dialog_close_does_not_bump_request_id():
    """Regression: dismissing the in-flight progress dialog after the worker
    finishes must not bump ``_preview_request_id``. The previous bug had
    ``prog.canceled -> _cancel_current`` connected, and ``prog.close()`` in
    ``on_result_ready`` auto-fired that signal, bumping the request id and
    making the just-received result look superseded."""

    _qapp()
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        tab, parent, dlg = _open_preview(tmp_dir)
        try:
            assert dlg is not None
            apply_current = getattr(dlg, "_pf_apply_current", None)
            assert apply_current is not None

            ok = _wait_for_filter_apply(dlg, timeout_s=20.0)
            assert ok, "Opening the preview should auto-apply current filters once"

            generation = dlg._filter_apply.generation
            dlg._pf_spinboxes["min_aspect_ratio"].setValue(6.0)
            QCoreApplication.processEvents()
            apply_current()
            assert _wait_for_apply_generation(dlg, rid_before=generation)
            assert dlg._filter_apply.generation == generation + 1, (
                "Programmatic progress dismissal must not cancel a completed Apply"
            )
        finally:
            _shutdown(tab, parent)


def test_apply_keeps_phase_visible_after_crushed_lut() -> None:
    """Apply must not leave Phase/input as a solid black canvas under a stale LUT."""

    _qapp()
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        _write_phase_tif(tmp_dir, pixel_size_um=0.108, nonzero=True)
        mask_path = _write_mask_tif(tmp_dir)
        tab, parent = _build_tab(tmp_dir)
        from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview

        show_filter_preview(tab, str(mask_path))
        QCoreApplication.processEvents()
        dlg = getattr(tab, "_filter_preview_dialog", None)
        try:
            assert dlg is not None
            assert _wait_for_filter_apply(dlg, timeout_s=20.0)

            # Simulate a leftover high-gain LUT from a previous foci View.
            dlg.state().vmin = 20000.0
            dlg.state().vmax = 34000.0
            apply_current = getattr(dlg, "_pf_apply_current", None)
            assert callable(apply_current)
            apply_current()
            assert _wait_for_filter_apply(dlg, timeout_s=20.0)
            assert _wait_until(
                lambda: _preview_thread_idle(dlg) and not _filter_apply_busy(dlg),
                timeout_s=10.0,
            )
            dlg.refresh()
            QCoreApplication.processEvents()

            base = np.asarray(getattr(dlg, "_preview_overlay_base", None))
            assert isinstance(base, np.ndarray) and base.ndim == 3 and base.shape[-1] == 3
            dark = float(((base[..., 0] < 8) & (base[..., 1] < 8) & (base[..., 2] < 8)).mean())
            assert dark < 0.5, (
                f"Phase background should remain visible after Apply; "
                f"frac near-black={dark:.3f} vmin={dlg.state().vmin} vmax={dlg.state().vmax}"
            )
            assert int(base.max()) >= 40
        finally:
            _shutdown(tab, parent)


def test_nav_sync_keeps_list_phase_key_not_first_entry() -> None:
    """Qt list keys must match tuple PreviewSource keys (no snap to index 0)."""
    _qapp()
    from bactomate.gui.preview.navigation_controls import (
        PreviewNavSpec,
        PreviewNavState,
        PreviewNavigationControls,
        PreviewSource,
    )

    sources = [
        PreviewSource(key=("file", 0), label="Foci: mNG"),
        PreviewSource(key=("file", 1), label="FI: mNG"),
        PreviewSource(key=("file", 2), label="Phase / input"),
    ]
    nav = PreviewNavigationControls(
        spec=PreviewNavSpec(frame_count=1, image_sources=sources, show_image=True),
    )
    nav.set_state(PreviewNavState(source_key=["file", 2], frame_idx=0), emit=False)
    assert nav.state().source_key == ("file", 2)
    assert nav._image_combo.currentText() == "Phase / input"


def test_view_foci_not_painted_as_phase_when_dialog_state_desynced() -> None:
    """Regression: Qt list keys must not leave View=Foci while painting Phase."""

    _qapp()
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        _write_phase_tif(tmp_dir, pixel_size_um=0.108, nonzero=True)
        # Bright-on-black foci plane, same geometry as the mask (256x256).
        foci = np.zeros((1, 256, 256), dtype=np.uint16)
        foci[0, 40:48, 40:48] = 50000
        tifffile.imwrite(str(tmp_dir / "sample_foci_mNG.tif"), foci)
        mask_path = _write_mask_tif(tmp_dir)
        tab, parent = _build_tab(tmp_dir)
        from bactomate.gui.tabs._analyze_masks_preview_filtering import show_filter_preview
        from bactomate.gui.preview.navigation_controls import PreviewNavState

        show_filter_preview(tab, str(mask_path))
        QCoreApplication.processEvents()
        dlg = getattr(tab, "_filter_preview_dialog", None)
        try:
            assert dlg is not None
            assert _wait_for_filter_apply(dlg, timeout_s=20.0)
            bg_paths = list(getattr(dlg, "_bg_paths", []) or [])
            foci_i = next(i for i, p in enumerate(bg_paths) if "_foci" in Path(p).name.lower())
            phase_i = next(
                i
                for i, p in enumerate(bg_paths)
                if "_foci" not in Path(p).name.lower()
            )
            nav = dlg._nav_ctrl
            # Historic bug path: Phase key as a Qt list failed nav equality when Foci
            # was first in the dropdown, snapping the label to Foci while dlg stayed Phase.
            dlg.set_source_key(["file", phase_i])
            nav.set_state(PreviewNavState(source_key=["file", phase_i], frame_idx=0), emit=False)
            # With the fix, list Phase keys stay on Phase (not snapped to index 0 / Foci).
            assert nav.state().source_key == ("file", phase_i)
            assert nav.state().source_key != ("file", foci_i)
            assert "Foci" not in str(nav._image_combo.currentText())

            # Switch View to Foci via list key (as Qt may supply) and confirm paint.
            nav.set_state(PreviewNavState(source_key=["file", foci_i], frame_idx=0), emit=True)
            for _ in range(25):
                QCoreApplication.processEvents()
                time.sleep(0.02)
            assert nav.state().source_key == ("file", foci_i)
            assert dlg.state().source_key == ("file", foci_i)
            assert "Foci" in str(nav._image_combo.currentText())
            dlg.refresh()
            QCoreApplication.processEvents()
            base = np.asarray(getattr(dlg, "_preview_overlay_base", None))
            assert isinstance(base, np.ndarray)
            # Foci plane is mostly zeros → dark canvas; Phase ramp mean is much higher.
            assert float(base.mean()) < 40.0, (
                f"Foci View must not paint Phase background; mean={float(base.mean()):.1f}"
            )
        finally:
            _shutdown(tab, parent)
