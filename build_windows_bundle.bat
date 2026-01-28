@echo off
REM Build BactiMate Windows standalone via Nuitka.
REM Run from repo root with venv activated (e.g. by GitHub Actions).
REM Produces BactiMate-Windows\ folder containing the exe and dependencies.

setlocal
cd /d "%~dp0"

if not exist "omnipose_batch_gui.py" (
    echo ERROR: omnipose_batch_gui.py not found. Run from repo root.
    exit /b 1
)

echo [1/4] Ensuring Nuitka...
pip install nuitka ordered-set --quiet

echo [2/4] Running Nuitka standalone build (target: under 6h on CI)...
REM Include heavy/slow packages as bytecode so C compilation finishes within job timeout
REM Aggressive optimizations: max parallel jobs, low-memory mode, no LTO, disable ccache, exclude heavy packages
python -m nuitka ^
  --standalone ^
  --plugin-enable=pyqt6 ^
  --include-data-files=logo2.png=logo2.png ^
  --nofollow-import-to=torch ^
  --nofollow-import-to=colour ^
  --nofollow-import-to=numpy ^
  --nofollow-import-to=scipy ^
  --nofollow-import-to=matplotlib ^
  --nofollow-import-to=pandas ^
  --nofollow-import-to=skimage ^
  --nofollow-import-to=dask ^
  --nofollow-import-to=numba ^
  --nofollow-import-to=xarray ^
  --nofollow-import-to=wandb ^
  --nofollow-import-to=IPython ^
  --nofollow-import-to=ipywidgets ^
  --nofollow-import-to=jupyter ^
  --nofollow-import-to=notebook ^
  --nofollow-import-to=networkx ^
  --nofollow-import-to=trackastra ^
  --nofollow-import-to=tifffile ^
  --nofollow-import-to=nd2 ^
  --nofollow-import-to=cellpose ^
  --nofollow-import-to=omnipose ^
  --jobs=auto ^
  --lto=no ^
  --low-memory ^
  --no-pyi-file ^
  --disable-ccache ^
  --no-progressbar ^
  --assume-yes-for-downloads ^
  --output-dir=build ^
  omnipose_batch_gui.py

if errorlevel 1 (
    echo ERROR: Nuitka build failed.
    exit /b 1
)

REM Nuitka creates build\omnipose_batch_gui.dist
set DIST=build\omnipose_batch_gui.dist
if not exist "%DIST%\omnipose_batch_gui.exe" (
    echo ERROR: Expected %DIST%\omnipose_batch_gui.exe not found.
    exit /b 1
)

echo [3/4] Preparing BactiMate-Windows folder...
if exist "BactiMate-Windows" rmdir /s /q "BactiMate-Windows"
move "%DIST%" "BactiMate-Windows"

REM Ensure logo is next to exe (include-data-files should already put it there)
if not exist "BactiMate-Windows\logo2.png" (
    if exist "logo2.png" copy "logo2.png" "BactiMate-Windows\logo2.png"
)

REM Optional: rename exe to BactiMate.exe for clearer UX
if exist "BactiMate-Windows\omnipose_batch_gui.exe" (
    move "BactiMate-Windows\omnipose_batch_gui.exe" "BactiMate-Windows\BactiMate.exe"
)

echo [4/4] Done. BactiMate-Windows\ is ready.
echo Artifacts: BactiMate-Windows\BactiMate.exe (or omnipose_batch_gui.exe) and dependencies.
endlocal
