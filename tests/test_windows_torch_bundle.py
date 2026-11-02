"""Exercise the installer gate without importing the heavyweight torch stack."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "bundle_check", Path(__file__).resolve().parents[1] / "setup/windows/check_torch_cuda_bundle.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize(
    "versions,available,required,expected",
    [
        (("2.11.0+cpu", "0.26.0+cpu", "2.11.0+cpu"), False, False, True),
        (("2.11.0+cpu", "0.26.0+cpu", "2.11.0+cpu"), False, True, False),
        (("2.11.0+cu128", "0.26.0+cu128", "2.11.0+cu128"), True, True, True),
        (("2.11.0+cu128", "0.26.0+cu126", "2.11.0+cu128"), True, True, False),
        (("2.11.0+cu128", "0.26.0+cu128", "2.11.0+cu128"), False, True, False),
        (("2.12.0+cpu", "0.27.0+cpu", "2.12.0+cpu"), False, False, False),
    ],
)
def test_bundle_gate(monkeypatch, versions, available, required, expected):
    for name, version in zip(("torch", "torchvision", "torchaudio"), versions):
        module = ModuleType(name)
        module.__version__ = version
        monkeypatch.setitem(sys.modules, name, module)
    torch = sys.modules["torch"]
    torch.version = SimpleNamespace(cuda="12.8" if "+cu" in versions[0] else None)
    torch.cuda = SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torchvision.ops", SimpleNamespace(nms=lambda: None))
    assert check.bundle_ok(require_cuda=required) is expected
