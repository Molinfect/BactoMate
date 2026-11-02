"""Step 1 (find Python 3.11) of setup/windows/install.bat, run under real cmd.exe.

`py` and `winget` are replaced by .cmd shims on a minimal PATH; LOCALAPPDATA and
ProgramFiles point at empty temp folders, so the runner's own Pythons are invisible.
BACTOMATE_INSTALL_STEP1_ONLY makes the installer exit right after Step 1.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or sys.version_info[:2] != (3, 11),
    reason="runs install.bat under cmd.exe; needs Windows and Python 3.11",
)

WIN_SETUP = Path(__file__).resolve().parents[1] / "setup" / "windows"

PY_SHIM = r"""@echo off
if not defined MOCK_PY_311 (
    echo No suitable Python runtime found 1>&2
    exit /b 1
)
echo %MOCK_PY_311%
"""

# Mimics a per-user python.org install by junctioning a real Python home into place.
WINGET_SHIM = r"""@echo off
echo %*>> "%MOCK_WINGET_LOG%"
if defined MOCK_WINGET_INSTALL_FROM (
    mkdir "%LOCALAPPDATA%\Programs\Python" 2>nul
    mklink /J "%LOCALAPPDATA%\Programs\Python\Python311" "%MOCK_WINGET_INSTALL_FROM%" >nul
)
exit /b 0
"""


class Installer:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "BactoMate"
        win_dir = self.root / "setup" / "windows"
        win_dir.mkdir(parents=True)
        for name in ("install.bat", "verify_python.py"):
            shutil.copy2(WIN_SETUP / name, win_dir / name)
        (self.root / "requirements.txt").write_text("")
        (self.root / "bactomate").mkdir()
        (self.root / "bactomate" / "__init__.py").write_text('__version__ = "0.0.0"\n')
        self.bat = win_dir / "install.bat"

        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.winget_log = tmp_path / "winget_args.txt"
        local_appdata = tmp_path / "LocalAppData"
        program_files = tmp_path / "ProgramFiles"
        local_appdata.mkdir()
        program_files.mkdir()

        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        self.env = dict(os.environ)
        for key in ("MOCK_PY_311", "MOCK_WINGET_INSTALL_FROM", "PYTHONHOME", "PYTHONPATH"):
            self.env.pop(key, None)
        self.env.update(
            PATH=os.pathsep.join(
                [str(self.bin), str(system32), str(system32 / "WindowsPowerShell" / "v1.0")]
            ),
            LOCALAPPDATA=str(local_appdata),
            ProgramFiles=str(program_files),
            MOCK_WINGET_LOG=str(self.winget_log),
            BACTOMATE_INSTALL_STEP1_ONLY="1",
        )

    def with_py(self, python_311: Path | None) -> Installer:
        (self.bin / "py.cmd").write_text(PY_SHIM, newline="\r\n")
        if python_311 is not None:
            self.env["MOCK_PY_311"] = str(python_311)
        return self

    def with_winget(self, install_from: Path | None = None) -> Installer:
        (self.bin / "winget.cmd").write_text(WINGET_SHIM, newline="\r\n")
        if install_from is not None:
            self.env["MOCK_WINGET_INSTALL_FROM"] = str(install_from)
        return self

    def run(self, stdin: str = "\n") -> subprocess.CompletedProcess[str]:
        comspec = self.env.get("ComSpec", "cmd.exe")
        result = subprocess.run(
            [comspec, "/d", "/c", str(self.bat)],
            input=stdin,
            capture_output=True,
            text=True,
            errors="replace",
            env=self.env,
            cwd=self.root,
            timeout=180,
        )
        result.output = result.stdout + result.stderr
        print(result.output)
        return result

    @property
    def log(self) -> str:
        return (self.root / "Logs" / "install_last.log").read_text(errors="replace")


@pytest.fixture
def installer(tmp_path):
    return Installer(tmp_path)


def test_uses_python_311_from_py_launcher(installer):
    result = installer.with_py(Path(sys.executable)).with_winget().run()

    assert result.returncode == 0
    assert "Using host Python:" in result.output
    assert "Install Python 3.11 now" not in result.output
    assert not installer.winget_log.exists()


def test_missing_python_without_winget_explains_and_stops(installer):
    result = installer.run()

    assert result.returncode == 1
    assert "Python 3.11 (64-bit) was not found on this PC" in result.output
    assert "Anaconda/Conda" in result.output
    assert "winget is not available" in result.output
    assert "BactoMate-win64.zip" in result.output
    assert "py launcher" in result.output
    # The bare "py" call must not leak cmd's "not recognized" error any more.
    assert "not recognized" not in result.output
    assert "winget: not available" in installer.log


def test_rejected_python_311_does_not_offer_winget(installer, tmp_path):
    conda_like = tmp_path / "conda_env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(conda_like)], check=True)
    (conda_like / "conda-meta").mkdir()

    result = installer.with_py(conda_like / "Scripts" / "python.exe").with_winget().run()

    assert result.returncode == 1
    assert "Python 3.11 was found but cannot be used" in result.output
    assert "Conda Python is not supported" in result.output
    assert "Install Python 3.11 now" not in result.output
    assert not installer.winget_log.exists()


def test_declined_winget_install_stops_without_installing(installer):
    result = installer.with_py(None).with_winget().run(stdin="N\n")

    assert result.returncode == 1
    assert "Install Python 3.11 now" in result.output
    assert "Python 3.11 (64-bit) was not found on this PC" in result.output
    assert not installer.winget_log.exists()
    assert "winget: declined by user" in installer.log


def test_accepted_winget_install_is_verified_and_used(installer):
    result = installer.with_winget(install_from=Path(sys.base_prefix)).run(stdin="Y\n")

    assert result.returncode == 0
    assert "Python 3.11 installed." in result.output
    assert "Using host Python:" in result.output
    winget_args = installer.winget_log.read_text()
    assert "--id Python.Python.3.11" in winget_args
    assert "InstallAllUsers=0" in winget_args
    assert "PrependPath=0" in winget_args
    assert "winget: install exit code 0" in installer.log
