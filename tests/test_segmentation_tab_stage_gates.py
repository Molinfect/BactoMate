"""Regression tests for SegmentationTab stage-gate refresh wiring."""

from __future__ import annotations

import ast
from pathlib import Path


def test_backend_failure_path_uses_existing_stage_gate_updater():
    source_path = (
        Path(__file__).resolve().parent.parent
        / "bactomate"
        / "gui"
        / "tabs"
        / "segmentation.py"
    )
    tree = ast.parse(source_path.read_text())

    method_calls = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }

    assert "_update_stage_gates" in method_calls
    assert "_update_analyze_gate" not in method_calls
