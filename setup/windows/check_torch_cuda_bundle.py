"""
Exit 0 when torch, torchvision, and torchaudio are mutually compatible.
Exit 1 when a reinstall is needed (mixed CUDA builds, or torchvision ops fail to load).

Used by setup/windows/install.bat step 6 before skipping the CUDA wheel install.
"""
from __future__ import annotations

import sys


def _cuda_tag(version: str) -> str:
    if "+" in version:
        return version.split("+", 1)[1]
    return "cpu"


def bundle_ok() -> bool:
    try:
        import torch
        import torchvision
        import torchaudio  # noqa: F401
    except (ImportError, RuntimeError, OSError):
        return False

    tags = [
        _cuda_tag(torch.__version__),
        _cuda_tag(torchvision.__version__),
        _cuda_tag(torchaudio.__version__),
    ]
    cu_tags = [t for t in tags if t.startswith("cu")]
    if cu_tags and len(set(cu_tags)) != 1:
        return False
    if cu_tags and any(not t.startswith("cu") for t in tags):
        return False

    try:
        from torchvision.ops import nms  # noqa: F401
    except Exception:
        return False

    return True


def main() -> int:
    return 0 if bundle_ok() else 1


if __name__ == "__main__":
    sys.exit(main())
