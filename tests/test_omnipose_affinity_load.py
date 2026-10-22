"""Tests for bact_phase_affinity loader path (3-class, no _omni symlink)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bactomate.gui.omnipose_utils import (
    affinity_loader_path_without_omni_in_stem,
    cellpose_omni_load_flag,
    eval_uses_affinity_segmentation,
    inspect_omnipose_checkpoint,
    is_affinity_builtin_model,
    is_affinity_checkpoint_path,
    load_cellpose_segmentation_model,
    omnipose_eval_first_mask,
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


def test_omnipose_eval_first_mask_accepts_three_or_four_tuple():
    import numpy as np

    mask = np.zeros((4, 4), dtype=np.uint16)
    assert omnipose_eval_first_mask((mask, [], [])) is mask
    assert omnipose_eval_first_mask((mask, [], [], None)) is mask
    assert omnipose_eval_first_mask(mask) is mask


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


@pytest.mark.parametrize("gpu, load_fails", [(False, False), (True, False), (True, True)])
def test_affinity_fallback_loads_wrapped_network(monkeypatch, tmp_path, gpu, load_fails):
    torch = pytest.importorskip("torch")
    from bactomate.gui import omnipose_utils as loader

    first_error = RuntimeError("initial model load failed")
    fallback_error = RuntimeError("checkpoint load failed")
    checkpoint = tmp_path / "affinity.pth"
    loaded = []

    class Network(torch.nn.Module):
        def load_model(self, path, cpu=False):
            loaded.append((path, cpu))
            if load_fails:
                raise fallback_error

    network = Network()
    model = SimpleNamespace(
        net=torch.nn.DataParallel(network) if gpu else network,
        nclasses=3,
    )

    def constructor(**kwargs):
        if kwargs["pretrained_model"] is not False:
            raise first_error
        return model

    package = ModuleType("cellpose_omni")
    models = ModuleType("cellpose_omni.models")
    models.CellposeModel = constructor
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose_omni", package)
    monkeypatch.setitem(sys.modules, "cellpose_omni.models", models)
    monkeypatch.setattr(loader, "bootstrap_cellpose_omni_for_bactomate", lambda: None)
    monkeypatch.setattr(loader, "materialize_bact_phase_affinity_checkpoint", lambda _: checkpoint)

    if load_fails:
        with pytest.raises(RuntimeError, match="checkpoint load failed") as error:
            loader.load_bact_phase_affinity_model(gpu=gpu, checkpoint_path=checkpoint)
        assert error.value.__context__ is first_error
    else:
        assert loader.load_bact_phase_affinity_model(gpu=gpu, checkpoint_path=checkpoint) is model
    assert loaded == [(str(checkpoint), not gpu)]


@pytest.mark.parametrize(
    "model_type, wrapped",
    [("bact_phase_affinity", False), ("bact_phase_omni", True), ("bact_phase_omni", False)],
)
def test_batch_accepts_direct_and_wrapped_models(monkeypatch, tmp_path, model_type, wrapped):
    import numpy as np
    import tifffile

    from bactomate.gui import omnipose_utils as loader
    from bactomate.gui import omnipose_pretrained_cache as cache

    loader.bootstrap_cellpose_omni_for_bactomate()
    pytest.importorskip("cellpose_omni")
    from bactomate.scripts import batch_process_tiff_sequence as batch

    affinity = model_type == "bact_phase_affinity"

    def evaluate(images, **kwargs):
        assert kwargs["affinity_seg"] is affinity
        return [np.ones_like(image, dtype=np.uint16) for image in images], [], []

    model = SimpleNamespace(nchan=1 if affinity else 2, nclasses=3 if affinity else 4, eval=evaluate)
    if wrapped:
        model = SimpleNamespace(cp=model, eval=evaluate)
    monkeypatch.setattr(loader, "load_bact_phase_affinity_model", lambda **_: model)
    monkeypatch.setattr(loader, "load_cellpose_segmentation_model", lambda **_: model)
    monkeypatch.setattr(cache, "ensure_builtin_in_bactomate_cache", lambda _: tmp_path / "affinity.pth")
    image = tmp_path / "frame.tif"
    tifffile.imwrite(image, np.zeros((16, 16), dtype=np.uint16))

    masks = batch.process_frames_with_omnipose(
        [str(image), str(image)], model_type=model_type,
        use_gpu=False, batch_size=1, diameter=30,
    )
    assert len(masks) == 2
    assert all(mask.shape == (16, 16) and np.all(mask == 1) for mask in masks)


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


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath("omnipose_pretrained", "bact_phase_affinitytorch_0").is_file(),
    reason="bundled affinity checkpoint missing",
)
def test_bact_phase_affinity_eval_returns_three_tuple():
    """CellposeModel.eval returns 3 values; preview must not unpack 4."""
    import numpy as np

    from bactomate.gui.omnipose_utils import load_bact_phase_affinity_model, omnipose_eval_first_mask

    ckpt = Path(__file__).resolve().parents[1] / "omnipose_pretrained" / "bact_phase_affinitytorch_0"
    pytest.importorskip("cellpose_omni")

    model = load_bact_phase_affinity_model(gpu=False, checkpoint_path=ckpt)
    img = np.random.randint(0, 255, (128, 128), dtype=np.uint16).astype(np.float32)
    result = model.eval(
        img,
        channels=[0, 0],
        diameter=30.0,
        flow_threshold=0.4,
        mask_threshold=0.0,
        omni=True,
        affinity_seg=True,
        cluster=False,
        net_avg=False,
        resample=False,
        verbose=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    mask = omnipose_eval_first_mask(result)
    assert mask.shape == (128, 128)
