"""Thresholds outside the usual output range must reach the model unchanged."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    # Isolate inference from optional GUI, GPU and model imports.
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('threshold', [-30.0, -7.5, 2.5, 8.0])
def test_sam_batch_forwards_unclipped_threshold(monkeypatch, threshold):
    model = Mock()
    model.eval.return_value = ([np.zeros((2, 2), dtype=np.uint16)],)
    monkeypatch.setitem(__import__('sys').modules, 'cellpose', SimpleNamespace(
        models=SimpleNamespace(CellposeModel=Mock(return_value=model))))
    namespace = dict(np=np, os=os, safe_imread=lambda _: np.zeros((2, 2)),
                     _to_three_channel=lambda image: image)
    run = load_function('bactomate/scripts/batch_process_tiff_sequence.py',
                        'process_frames_with_cellpose_sam', namespace)
    run(['frame.tif'], diameter=1500.0, flow_threshold=3.0, mask_threshold=threshold)
    assert model.eval.call_args.kwargs['cellprob_threshold'] == threshold
    assert model.eval.call_args.kwargs['diameter'] == 1500.0
    assert model.eval.call_args.kwargs['flow_threshold'] == 3.0


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -float('inf')])
def test_sam_rejects_nonfinite_threshold(threshold):
    run = load_function('bactomate/scripts/batch_process_tiff_sequence.py',
                        'process_frames_with_cellpose_sam', dict(np=np))
    with pytest.raises(ValueError, match='finite'):
        run([], mask_threshold=threshold)


def test_gui_controls_and_preview_keep_extended_values(qapp):
    from PyQt6.QtWidgets import QDoubleSpinBox
    import sys

    path = 'bactomate/gui/tabs/omnipose_seg.py'
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    owner = SimpleNamespace(diameter=QDoubleSpinBox(), flow_threshold=QDoubleSpinBox(),
                            mask_threshold=QDoubleSpinBox())
    # Run the actual range assignments without constructing the full tab.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'setRange' and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr in vars(owner)):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(node)], type_ignores=[])), path, 'exec'),
                 dict(self=owner, sys=sys))
    link = load_function(path, '_link_spinboxes', dict(QDoubleSpinBox=QDoubleSpinBox))
    for name, value in [('diameter', 1500), ('flow_threshold', 3), ('mask_threshold', -30)]:
        main = getattr(owner, name)
        preview = QDoubleSpinBox()
        link(preview, main)
        preview.setValue(value)
        assert main.value() == value
        main.setValue(value + 1)
        assert preview.value() == value + 1
    owner.mask_threshold.setValue(8)
    get_threshold = load_function(path, '_get_effective_mask_threshold', {})
    assert get_threshold(owner) == 8
