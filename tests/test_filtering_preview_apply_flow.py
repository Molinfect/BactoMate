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


def _write_mask_tif(tmp_dir: Path) -> Path:
    seg_dir = tmp_dir / "_segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    mask_path = seg_dir / "sample_masks.tif"
    tifffile.imwrite(str(mask_path), _make_mixed_mask().astype(np.int32))
    return mask_path


def _write_phase_tif(tmp_dir: Path, *, pixel_size_um: float) -> Path:
    phase_path = tmp_dir / "sample.tif"
    stack = np.zeros((1, 64, 64), dtype=np.uint16)
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
            dlg._close_pending = True
            dlg.close()
    except Exception:
        pass
    QCoreApplication.processEvents()
    try:
        parent.close()
        parent.deleteLater()
    except Exception:
        pass
    QCoreApplication.processEvents()


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

            rid_changes: list[tuple[int, int]] = []
            real_setattr = type(dlg).__setattr__

            def traced_setattr(self, name, value):
                if name == "_preview_request_id":
                    old = getattr(self, "_preview_request_id", None)
                    if value != old:
                        rid_changes.append((int(old or 0), int(value)))
                real_setattr(self, name, value)

            type(dlg).__setattr__ = traced_setattr
            try:
                spinboxes = getattr(dlg, "_pf_spinboxes", None)
                assert spinboxes is not None and "min_aspect_ratio" in spinboxes
                spinboxes["min_aspect_ratio"].setValue(6.0)
                QCoreApplication.processEvents()
                apply_current()
                ok = _wait_for_filter_apply(dlg, timeout_s=20.0)
                assert ok, "Worker result should reach _last_preview"

                # Exactly one rid bump expected from this manual apply.
                start_rid = int(getattr(dlg, "_preview_request_id", 0)) - 1
                assert rid_changes == [(start_rid, start_rid + 1)], (
                    f"Expected a single rid increment from _start_worker; "
                    f"got {rid_changes}. An extra increment indicates the "
                    f"progress-dialog 'canceled' signal is still firing during "
                    f"programmatic dismissal."
                )
            finally:
                type(dlg).__setattr__ = real_setattr
        finally:
            _shutdown(tab, parent)
