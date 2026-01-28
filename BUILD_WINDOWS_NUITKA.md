# Building BactiMate for Windows (Standalone via Nuitka)

This document describes how to build a **Windows standalone** executable for BactiMate using **Nuitka**, and how to run that build on **GitHub** (e.g. GitHub Actions).

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

## Troubleshooting

| Issue | What to try |
|-------|--------------|
| **Missing DLL** | Run the exe from a shell and check the error; add the reported DLL’s package to `--include-package` or include the DLL explicitly if it’s from a vendored binary. |
| **Qt platform plugin** | Ensure `--plugin-enable=pyqt6` is set and that the built folder contains `PyQt6\Qt6\plugins\platforms\qwindows.dll` (or equivalent). |
| **Logo not found** | Confirm `logo2.png` is in the same directory as the exe and that the GUI uses `base_path` when `sys.frozen` or `__compiled__` is true. |
| **Build timeouts on GitHub** | Use CPU-only PyTorch, enable Nuitka/cache if using Nuitka-Action, or split into a “prepare env” job and a “compile” job with a longer timeout. |
| **Out of memory** | Use `--low-memory` and/or reduce parallel jobs in Nuitka. |

## Onefile (optional)

For a single `.exe` that unpacks at runtime:

```batch
python -m nuitka --onefile --plugin-enable=pyqt6 --include-data-files=logo2.png=logo2.png --assume-yes-for-downloads --output-dir=build omnipose_batch_gui.py
```

Startup is slower and debugging is harder; the **standalone folder** is recommended first.
