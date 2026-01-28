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

echo [2/4] Running Nuitka standalone build (this can take 15-45 min)...
REM Include torch as bytecode (do not compile) to avoid Nuitka crash in torch._dynamo.pgo
python -m nuitka ^
  --standalone ^
  --plugin-enable=pyqt6 ^
  --include-data-files=logo2.png=logo2.png ^
  --nofollow-import-to=torch ^
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
