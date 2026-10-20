"""Exercise the launcher's actual cmd-to-Python path argument on Windows."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "win32", "requires Windows cmd.exe")
class BundleLauncherTests(unittest.TestCase):
    def test_relocate_environment(self):
        project = Path(__file__).resolve().parents[1]
        source = project / "setup" / "windows"
        launcher = (source / "launch_bundle.bat").read_text(encoding="utf-8")
        # Stop after environment repair; the GUI and installed packages aren't needed.
        launcher = launcher.split('set "PYEXE=', 1)[0] + "exit /b 0\n"
        launcher = launcher.replace(
            r"%ROOT%runtime\Python311\python.exe", sys.executable
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in ("BactoMate", "Bacto Mate (portable)"):
                with self.subTest(folder=name):
                    root = Path(directory) / name
                    scripts = root / "setup" / "windows"
                    scripts.mkdir(parents=True)
                    shutil.copy2(source / "patch_bundle_pyvenv.py", scripts)
                    cfg = root / "venv" / "pyvenv.cfg"
                    cfg.parent.mkdir()
                    cfg.write_text("home = old\nexecutable = old\n", encoding="utf-8")
                    (root / "BactoMate.bat").write_text(launcher, encoding="utf-8")
                    result = subprocess.run(
                        ["cmd", "/d", "/c", "BactoMate.bat"], cwd=root,
                        input="\n", text=True, capture_output=True, timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    runtime = (root / "runtime" / "Python311").resolve()
                    self.assertIn(f"home = {runtime}\n", cfg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
