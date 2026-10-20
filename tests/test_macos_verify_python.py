"""Tests for the macOS source-install Python gate (setup/macos/verify_python.py)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

VERIFY_PATH = Path(__file__).resolve().parents[1] / "setup" / "macos" / "verify_python.py"


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("macos_verify_python", VERIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load_verify_module()
check_python = verify.check_python

NORMAL_EXE = "/opt/homebrew/opt/python@3.11/bin/python3.11"
NORMAL_PREFIX = "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11"


def test_accepts_python_311_with_normal_path(tmp_path):
    assert check_python((3, 11), NORMAL_EXE, str(tmp_path)) is None
    assert check_python((3, 11), NORMAL_EXE, NORMAL_PREFIX) is None


@pytest.mark.parametrize("version", [(3, 10), (3, 12), (3, 14)])
def test_rejects_other_versions(version):
    error = check_python(version, NORMAL_EXE, NORMAL_PREFIX)
    assert error is not None
    assert f"{version[0]}.{version[1]}" in error
    assert "3.11" in error
    assert "https://www.python.org/downloads/" in error
    assert "brew install python@3.11" in error
    assert "BactoMate.dmg" in error


@pytest.mark.parametrize("name", ["miniconda3", "Miniconda3", "anaconda3", "Anaconda3"])
def test_rejects_conda_prefix(name):
    error = check_python((3, 11), NORMAL_EXE, f"/Users/lab/{name}")
    assert error is not None
    assert "BactoMate.dmg" in error


@pytest.mark.parametrize("name", ["miniconda3", "anaconda3"])
def test_rejects_conda_executable(name):
    error = check_python((3, 11), f"/Users/lab/{name}/bin/python3.11", NORMAL_PREFIX)
    assert error is not None
    assert "Conda" in error


def test_rejects_conda_meta_directory(tmp_path):
    (tmp_path / "conda-meta").mkdir()
    error = check_python((3, 11), NORMAL_EXE, str(tmp_path))
    assert error is not None
    assert "Conda" in error


def test_main_prints_executable_and_exits_zero_or_one():
    result = subprocess.run(
        [sys.executable, str(VERIFY_PATH)], capture_output=True, text=True
    )
    if result.returncode == 0:
        assert result.stdout.strip() == str(Path(sys.executable).absolute())
        assert result.stderr == ""
        return
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip()
