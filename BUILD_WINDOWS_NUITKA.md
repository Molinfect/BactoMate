# Building BactiMate for Windows (Standalone via Nuitka)

This document describes how to build a **Windows standalone** executable for BactiMate using **Nuitka**, and how to run that build on **GitHub** (e.g. GitHub Actions).

## Files involved (no surprises after a 2h build)

| File | Purpose |
|------|--------|
| **build_windows.bat** | Wrapper: calls `build_windows_bundle.bat`. Use this or the bundle script directly. |
| **build_windows_bundle.bat** | **Primary Windows build script.** Runs Nuitka, moves output to `BactiMate-Windows`, copies nofollow packages, verifies the build, then **runs the exe with `--smoke-test`** so the job only passes if the exe actually starts and imports numpy/PyQt6/tifffile. |
| **BactiMate.spec** | Optional **PyInstaller** spec. The recommended Windows build is Nuitka via `build_windows.bat` / `build_windows_bundle.bat`. Use `pyinstaller BactiMate.spec` only if you prefer PyInstaller. |
| **omnipose_batch_gui.py** | Entry point (no .spec required for Nuitka). |

After a successful run you get **BactiMate-Windows\** with **BactiMate.exe** (or omnipose_batch_gui.exe), required DLLs, and copied packages. The script verifies the exe and key packages, then **runs the exe once with `--smoke-test`**; if that fails (exe won't start or can't import numpy/PyQt6/tifffile), the build fails. So a green build means the exe has been proven to run.

## Why Nuitka for Windows?

- **Standalone exe + DLLs**: One folder users can copy anywhere; no system Python needed.
- **PyQt6 support**: Nuitka’s `--plugin-enable=pyqt6` handles Qt plugins and dependencies on Windows.
- **Performance**: Compiled code can start and run faster than a pure Python + venv bundle.
- **macOS**: The project uses a venv-bundle approach on macOS due to past PyQt6/Nuitka issues there; Nuitka is used for **Windows** only.

## Strategy Overview

| Aspect | Choice | Notes |
|--------|--------|------|
| **Mode** | `--standalone` | Folder with `.exe` + dependencies; easier to debug than `--onefile`. |
| **Entry point** | `omnipose_batch_gui.py` | Single script, `if __name__ == "__main__": main()`. |
| **Qt** | `--plugin-enable=pyqt6` | Required for PyQt6 apps on Windows. |
| **Data files** | `--include-data-files=logo2.png=logo2.png` | Logo next to exe; GUI uses `base_path` when frozen. |
| **Compiler** | MSVC (default on `windows-latest`) or MinGW64 | Nuitka uses MSVC when available; otherwise it can download MinGW64 (`--assume-yes-for-downloads` in CI). |
| **PyTorch** | CPU-only in CI | Use default `torch` from pip on Windows runner to keep build time and size manageable. For GPU, users can build locally with CUDA wheels. |

## What the build produces

1. **Nuitka output**: A directory (e.g. `omnipose_batch_gui.dist`) containing:
   - `omnipose_batch_gui.exe` (or `BactiMate.exe` if overridden),
   - Qt DLLs and plugins,
   - Python runtime and compiled extensions,
   - Other dependencies (NumPy, PyTorch, etc.).
2. **Bundling**: That directory is renamed to `BactiMate-Windows`, `logo2.png` is ensured next to the exe, then the folder is zipped as `dist/BactiMate-Windows.zip` for release/artifacts.

## GitHub Actions setup

The existing workflow (`.github/workflows/build.yml`) has a **build-windows** job that:

1. Runs on `windows-latest`.
2. Checks out the repo, sets up Python 3.11, creates a venv, installs `requirements.txt`.
3. Runs `build_windows_bundle.bat` to perform the Nuitka build and produce `BactiMate-Windows`.
4. Zips `BactiMate-Windows` → `dist/BactiMate-Windows.zip` and uploads it as an artifact (and optionally to a release).

So the **Windows build is fully driven by `build_windows_bundle.bat`**, which calls Nuitka and then arranges the folder and zip.

### Optional: Nuitka-Action

You can later switch to [Nuitka/Nuitka-Action](https://github.com/Nuitka/Nuitka-Action) for a more declarative YAML-based build. The action accepts `script-name`, `mode: standalone`, `enable-plugins: pyqt6`, and `include-data-files`. For a first iteration, the inline `build_windows_bundle.bat` keeps everything in-repo and easy to tune.

## Local Windows build (developer)

1. **Prerequisites**: Python 3.11, Visual Studio Build Tools (or MinGW64), and a venv with dependencies:

   ```batch
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   pip install nuitka
   ```

2. **Run the same script**:

   ```batch
   build_windows_bundle.bat
   ```

   This uses the active Python/venv and produces `BactiMate-Windows` and `dist\BactiMate-Windows.zip`.

3. **One-off Nuitka command** (for debugging):

   ```batch
   python -m nuitka --standalone --plugin-enable=pyqt6 --include-data-files=logo2.png=logo2.png --assume-yes-for-downloads --output-dir=build omnipose_batch_gui.py
   ```

   Then rename `build\omnipose_batch_gui.dist` and add `logo2.png` if needed.

## Code changes for frozen/Nuitka builds

- **Logo path**: The GUI loads `logo2.png` via `Path(__file__).parent / "logo2.png"`. When running as a Nuitka standalone exe, `__file__` may not point at the real app dir. The app already defines `base_path = os.path.dirname(sys.executable)` when `sys.frozen` or `__compiled__` is set. The logo is loaded using `base_path` in that case so it finds `logo2.png` next to the exe.
- **Plugins/base_path**: Existing logic already uses `base_path` for PyQt6 plugins when frozen; Nuitka’s PyQt6 plugin puts Qt next to the exe, so `base_path` remains correct.
- **Models**: Omnipose/cellpose_omni models are downloaded at runtime (e.g. from GitHub). No need to ship them in the Nuitka bundle.

## Confidence and known risks (before you build again)

**How likely is "build once, run on Windows"?**  
**High** if the build passes: we run the exe with `--smoke-test` (imports numpy, PyQt6, tifffile and exits). If that succeeds, the exe has been proven to start and load the main dependencies. Remaining risk is only in features we don't exercise in the smoke test (e.g. segmentation, tracking).

| Risk | Why | What we do / what you can do |
|------|-----|------------------------------|
| **Job timeout** | Nuitka C compile (even with nofollow) can take 2–4+ hours. | Workflow sets `timeout-minutes: 360` for the Windows job. If it still times out, run with "Windows only" or reduce compiled surface. |
| **Dependency Walker download** | dependencywalker.com can be down or slow. | We pre-download in CI; if it fails, the build step fails with a clear error. |
| **site-packages path wrong** | In CI, `python -c "site.getsitepackages()"` might not match the venv. | We fall back to `%VIRTUAL_ENV%\Lib\site-packages` when the env is set. If copy still skips, verification fails the build. |
| **Package copy skipped** | If SITEPACKAGES is empty, no packages are copied. | We fail the build if SITEPACKAGES is missing, or if numpy is missing in site-packages, or if `BactiMate-Windows\numpy` is missing after copy. |
| **Exe starts, then crashes** | Missing DLL or import error in code not hit by smoke test. | We run `BactiMate.exe --smoke-test` after build; if that fails, the build fails. So a green build means the exe at least starts and imports numpy/PyQt6/tifffile. If it later crashes in the GUI or segmentation, run from cmd to see the traceback. |

**Before you build again:** Workflow has a 6h timeout; the build script fails if the exe is missing, site-packages/numpy copy failed, or **the exe fails the smoke test** (we run `BactiMate.exe --smoke-test`). A green build means the exe has been run once and successfully imported numpy, PyQt6, and tifffile.

## Troubleshooting

| Issue | What to try |
|-------|--------------|
| **Missing DLL** | Run the exe from a shell and check the error; add the reported DLL’s package to `--include-package` or include the DLL explicitly if it’s from a vendored binary. |
| **Qt platform plugin** | Ensure `--plugin-enable=pyqt6` is set and that the built folder contains `PyQt6\Qt6\plugins\platforms\qwindows.dll` (or equivalent). |
| **Logo not found** | Confirm `logo2.png` is in the same directory as the exe and that the GUI uses `base_path` when `sys.frozen` or `__compiled__` is true. |
| **Build timeouts on GitHub** | Job has `timeout-minutes: 360`. Use "Windows only" in workflow_dispatch if needed; or add more `--nofollow-import-to` to shorten compile. |
| **Out of memory** | Use `--low-memory` and/or reduce parallel jobs in Nuitka. |
| **"numpy not found" / copy failed** | Build fails if `BactiMate-Windows\numpy` is missing. Check log: SITEPACKAGES path and that venv has numpy; run from repo root with venv activated. |

## Onefile (optional)

For a single `.exe` that unpacks at runtime:

```batch
python -m nuitka --onefile --plugin-enable=pyqt6 --include-data-files=logo2.png=logo2.png --assume-yes-for-downloads --output-dir=build omnipose_batch_gui.py
```

Startup is slower and debugging is harder; the **standalone folder** is recommended first.
