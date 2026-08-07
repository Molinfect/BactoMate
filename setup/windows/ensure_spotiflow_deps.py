"""
Ensure wandb/protobuf/tensorboard stay compatible with Spotiflow *and* TensorFlow.

Spotiflow imports wandb at package load. Too-old wandb + too-new protobuf raises:

  ImportError: cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'

But blindly upgrading wandb/protobuf to latest breaks U-Net / tensorflow-intel
(protobuf must stay <5; tensorboard must stay <2.17 for tensorflow<2.17).

Called from setup\\windows\\install.bat before the smoke test. Safe to re-run:

  venv\\Scripts\\python.exe setup\\windows\\ensure_spotiflow_deps.py
"""
from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys
from typing import Optional

# Keep aligned with requirements.txt and setup/windows/constraints.txt.
PIP_SPECS = (
    "wandb>=0.22.0,!=0.23.1,<0.28.0",
    "protobuf>=4.25.0,<5.0.0",
    "tensorboard>=2.12.0,<2.17.0",
)


def _pkg_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _major(version: str) -> int:
    return int(version.split(".")[0])


def _wandb_import_ok() -> tuple[bool, str]:
    try:
        import wandb  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised on broken envs
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _deps_ok() -> tuple[bool, str]:
    ok, err = _wandb_import_ok()
    if not ok:
        return False, err

    pb = _pkg_version("protobuf")
    if pb is None:
        return False, "protobuf is not installed"
    if _major(pb) >= 5:
        return False, f"protobuf {pb} is too new for tensorflow<2.17 (need <5.0.0)"

    tb = _pkg_version("tensorboard")
    if tb is not None and _major(tb) >= 3:
        return False, f"tensorboard {tb} is too new for tensorflow<2.17 (need <2.17)"
    if tb is not None:
        # tensorflow 2.16 wants tensorboard>=2.16,<2.17; keep any 2.12–2.16.x
        parts = [int(x) for x in tb.split(".")[:2]]
        minor = parts[1] if len(parts) > 1 else 0
        if parts[0] > 2 or (parts[0] == 2 and minor >= 17):
            return False, f"tensorboard {tb} is too new for tensorflow<2.17 (need <2.17)"

    return True, ""


def _repair() -> int:
    cmd = [sys.executable, "-m", "pip", "install", *PIP_SPECS]
    print("Repairing Spotiflow/TensorFlow deps:", " ".join(PIP_SPECS))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    ok, err = _deps_ok()
    if ok:
        pb = _pkg_version("protobuf") or "?"
        tb = _pkg_version("tensorboard") or "?"
        wb = _pkg_version("wandb") or "?"
        print(f"Spotiflow/TensorFlow deps OK (wandb={wb}, protobuf={pb}, tensorboard={tb}).")
        return 0

    print(f"Deps check failed ({err}); repairing...", file=sys.stderr)
    rc = _repair()
    if rc != 0:
        print(
            "[BREAK] pip repair for wandb/protobuf/tensorboard failed.\n"
            "  Fix: python -m pip install "
            + " ".join(f'"{s}"' for s in PIP_SPECS)
            + "\n  See docs/INSTALL_AND_LAUNCH_windows.md",
            file=sys.stderr,
        )
        return rc

    ok2, err2 = _deps_ok()
    if not ok2:
        print(
            f"[BREAK] deps still broken after repair ({err2}).\n"
            "  See docs/INSTALL_AND_LAUNCH_windows.md (Spotiflow / wandb section).",
            file=sys.stderr,
        )
        return 1

    print("Spotiflow/TensorFlow deps OK after repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
