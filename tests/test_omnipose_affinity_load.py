"""Tests for bact_phase_affinity loader path (3-class, no _omni symlink)."""
from __future__ import annotations

from pathlib import Path

import pytest

from bactomate.gui.omnipose_utils import (
    affinity_loader_path_without_omni_in_stem,
    cellpose_omni_load_flag,
    eval_uses_affinity_segmentation,
    inspect_omnipose_checkpoint,
    is_affinity_builtin_model,
    is_affinity_checkpoint_path,
    load_cellpose_segmentation_model,
    prepare_builtin_pretrained_loader_path,
)


def test_is_affinity_builtin_model():
    assert is_affinity_builtin_model("bact_phase_affinity")
    assert not is_affinity_builtin_model("bact_phase_omni")
    assert not is_affinity_builtin_model(None)


def test_is_affinity_checkpoint_path():
    assert is_affinity_checkpoint_path("/tmp/bact_phase_affinitytorch_0")
    assert is_affinity_checkpoint_path("/tmp/bact_phase_affinitytorch_0_omni.pth")
    assert not is_affinity_checkpoint_path("/tmp/my_omni_model.pth")
    assert not is_affinity_checkpoint_path(None)


def test_affinity_alias_strips_omni_from_stem(tmp_path):
    ckpt = tmp_path / "bact_phase_affinitytorch_0_omni.pth"
    ckpt.write_bytes(b"stub")
    alias = affinity_loader_path_without_omni_in_stem(ckpt)
    assert "omni" not in Path(alias).stem.lower()


def test_cellpose_omni_load_flag():
    assert cellpose_omni_load_flag(model_type="bact_phase_affinity") is False
    assert cellpose_omni_load_flag(model_type="bact_phase_omni") is True
    assert cellpose_omni_load_flag(pretrained_path="/x/custom_affinity.pth") is False
    assert cellpose_omni_load_flag(pretrained_path="/x/custom_omni.pth") is True


def test_eval_uses_affinity_segmentation():
    assert eval_uses_affinity_segmentation(model_type="bact_phase_affinity")
    assert not eval_uses_affinity_segmentation(model_type="bact_phase_omni")


def test_prepare_builtin_skips_omni_symlink_for_affinity(tmp_path):
    ckpt = tmp_path / "bact_phase_affinitytorch_0"
    ckpt.write_bytes(b"stub")
    out = prepare_builtin_pretrained_loader_path(ckpt, "bact_phase_affinity")
    assert Path(out).is_file()
    assert "omni" not in Path(out).stem.lower()


def test_prepare_builtin_adds_omni_symlink_for_flow_model(tmp_path):
    ckpt = tmp_path / "bact_phase_omnitorch_0"
    ckpt.write_bytes(b"stub")
    out = prepare_builtin_pretrained_loader_path(ckpt, "bact_phase_omni")
    assert Path(out).stem.endswith("_omni") or "_omni" in Path(out).stem


def test_bundled_affinity_checkpoint_has_three_output_channels():
    ckpt = Path(__file__).resolve().parents[1] / "omnipose_pretrained" / "bact_phase_affinitytorch_0"
    if not ckpt.is_file():
        pytest.skip("bundled bact_phase_affinity checkpoint not present")
    torch = pytest.importorskip("torch")
    info = inspect_omnipose_checkpoint(ckpt)
    assert info.get("output_layer_out_channels") == 3
    assert cellpose_omni_load_flag(model_type="bact_phase_affinity") is False
    del torch


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath("omnipose_pretrained", "bact_phase_affinitytorch_0").is_file(),
    reason="bundled affinity checkpoint missing",
)
def test_load_bact_phase_affinity_cellpose_nclasses():
    """Integration: load affinity built-in; expect nclasses=3 even if cache had _omni alias."""
    ckpt = Path(__file__).resolve().parents[1] / "omnipose_pretrained" / "bact_phase_affinitytorch_0"
    pytest.importorskip("cellpose_omni")

    model = load_cellpose_segmentation_model(
        gpu=False,
        loader_path=ckpt,
        model_type="bact_phase_affinity",
    )
    nclasses = getattr(model, "nclasses", None) or getattr(getattr(model, "cp", None), "nclasses", None)
    assert nclasses == 3
