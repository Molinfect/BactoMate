"""Point bundled venv pyvenv.cfg at runtime/Python311 under the bundle root."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def patch_pyvenv_cfg(root: Path) -> None:
    runtime = (root / "runtime" / "Python311").resolve()
    cfg = root / "venv" / "pyvenv.cfg"
    text = cfg.read_text(encoding="utf-8")
    text = re.sub(
        r"^home = .*$",
        lambda _: f"home = {runtime}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^executable = .*$",
        lambda _: f"executable = {runtime}\\python.exe",
        text,
        flags=re.MULTILINE,
    )
    cfg.write_text(text, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: patch_bundle_pyvenv.py <bundle-root>", file=sys.stderr)
        return 1
    root = Path(sys.argv[1]).resolve()
    cfg = root / "venv" / "pyvenv.cfg"
    if not cfg.is_file():
        print(f"Error: missing {cfg}", file=sys.stderr)
        return 1
    patch_pyvenv_cfg(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
