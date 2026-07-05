"""
Build a portable Windows folder bundle at dist/BactoMate/.

Analogous to build/build_app_bundle.sh on macOS: copies project source, the project
venv, and the Python 3.11 runtime that created the venv so the bundle runs without
a separate Python install on target PCs.

Run on Windows after setup\\install_windows.bat:
    python build\\build_windows_bundle.py

Optional zip: build\\create_windows_zip.bat
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _query_base_prefix(venv_python: Path) -> Path:
    out = subprocess.check_output(
        [str(venv_python), "-c", "import sys; print(sys.base_prefix)"],
        text=True,
    ).strip()
    return Path(out).resolve()


def _should_skip_dir(name: str) -> bool:
    return name in {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "dist",
        "BactoMate.app",
        "venv.old",
        "Logs",
    }


def _copy_project_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if _should_skip_dir(item.name) or item.name.endswith(".pyc"):
            continue
        if item.name == "venv":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    ".DS_Store",
                    ".pytest_cache",
                ),
            )
        else:
            shutil.copy2(item, target)


def _copy_python_runtime(base_prefix: Path, runtime_dir: Path) -> None:
    if (base_prefix / "conda-meta").is_dir():
        raise SystemExit(
            "Error: venv was created from Conda Python. Rebuild with python.org 3.11:\n"
            "  rmdir /s /q venv\n"
            "  setup\\windows\\install.bat"
        )
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying Python runtime from {base_prefix} ...")
    shutil.copytree(
        base_prefix,
        runtime_dir,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "Doc",
            "include",
            "libs",
            "tcl",
            "Tools",
        ),
    )


def _rewrite_pyvenv_cfg(pyvenv_cfg: Path, runtime_dir: Path) -> None:
    runtime = runtime_dir.resolve()
    text = pyvenv_cfg.read_text(encoding="utf-8")
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
    pyvenv_cfg.write_text(text, encoding="utf-8")


def _write_bundle_launcher(dist: Path) -> None:
    launcher = dist / "BactoMate.bat"
    shutil.copy2(_project_root() / "setup" / "windows" / "launch_bundle.bat", launcher)


def main() -> int:
    if sys.platform != "win32":
        print("Error: build_windows_bundle.py must run on Windows.", file=sys.stderr)
        return 1

    root = _project_root()
    venv_python = root / "venv" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        print("Error: venv not found. Run setup\\windows\\install.bat first.", file=sys.stderr)
        return 1

    dist = root / "dist" / "BactoMate"
    runtime_dir = dist / "runtime" / "Python311"

    if dist.exists():
        print(f"Removing existing {dist} ...")
        shutil.rmtree(dist)

    print("Copying project files ...")
    _copy_project_tree(root, dist)

    base_prefix = _query_base_prefix(venv_python)
    _copy_python_runtime(base_prefix, runtime_dir)

    print("Copying virtual environment ...")
    shutil.copytree(root / "venv", dist / "venv")

    pyvenv_cfg = dist / "venv" / "pyvenv.cfg"
    if not pyvenv_cfg.is_file():
        print(f"Error: missing {pyvenv_cfg}", file=sys.stderr)
        return 1
    _rewrite_pyvenv_cfg(pyvenv_cfg, runtime_dir)

    _write_bundle_launcher(dist)
    shutil.copy2(root / "setup" / "windows" / "launch_bundle.bat", dist / "launch_windows.bat")

    bundle_py = dist / "venv" / "Scripts" / "python.exe"
    print("Verifying bundle (--version) ...")
    subprocess.run([str(bundle_py), str(dist / "main.py"), "--version"], cwd=dist, check=True)
    print("Verifying bundle (--smoke-test) ...")
    subprocess.run([str(bundle_py), str(dist / "main.py"), "--smoke-test"], cwd=dist, check=True)

    print()
    print(f"Created: {dist}")
    print("Launch: double-click BactoMate.bat or launch_windows.bat inside the folder.")
    print("Zip for distribution: build\\windows\\create_windows_zip.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
