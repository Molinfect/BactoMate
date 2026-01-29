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
  --jobs=4 ^
  --lto=no ^
  --low-memory ^
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

echo [3/5] Preparing BactiMate-Windows folder...
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

REM [Option A] Copy nofollow packages from site-packages into dist so exe can import them at runtime.
REM Build stays fast (no compile of numpy/torch/etc); packages are present at runtime.
echo [4/5] Copying heavy packages from site-packages into BactiMate-Windows...
for /f "delims=" %%i in ('python -c "import site; print(site.getsitepackages()[0])"') do set SITEPACKAGES=%%i
if not defined SITEPACKAGES (
    echo WARNING: Could not get site-packages path. Skipping package copy.
) else (
    echo Using site-packages: %SITEPACKAGES%
    REM Copy each nofollow package (and .libs if present) into BactiMate-Windows
    if exist "%SITEPACKAGES%\numpy" xcopy /E /I /Y /Q "%SITEPACKAGES%\numpy" "BactiMate-Windows\numpy" >nul
    if exist "%SITEPACKAGES%\numpy.libs" xcopy /E /I /Y /Q "%SITEPACKAGES%\numpy.libs" "BactiMate-Windows\numpy.libs" >nul
    if exist "%SITEPACKAGES%\scipy" xcopy /E /I /Y /Q "%SITEPACKAGES%\scipy" "BactiMate-Windows\scipy" >nul
    if exist "%SITEPACKAGES%\scipy.libs" xcopy /E /I /Y /Q "%SITEPACKAGES%\scipy.libs" "BactiMate-Windows\scipy.libs" >nul
    if exist "%SITEPACKAGES%\torch" xcopy /E /I /Y /Q "%SITEPACKAGES%\torch" "BactiMate-Windows\torch" >nul
    if exist "%SITEPACKAGES%\colour" xcopy /E /I /Y /Q "%SITEPACKAGES%\colour" "BactiMate-Windows\colour" >nul
    if exist "%SITEPACKAGES%\matplotlib" xcopy /E /I /Y /Q "%SITEPACKAGES%\matplotlib" "BactiMate-Windows\matplotlib" >nul
    if exist "%SITEPACKAGES%\pandas" xcopy /E /I /Y /Q "%SITEPACKAGES%\pandas" "BactiMate-Windows\pandas" >nul
    if exist "%SITEPACKAGES%\skimage" xcopy /E /I /Y /Q "%SITEPACKAGES%\skimage" "BactiMate-Windows\skimage" >nul
    if exist "%SITEPACKAGES%\tifffile" xcopy /E /I /Y /Q "%SITEPACKAGES%\tifffile" "BactiMate-Windows\tifffile" >nul
    if exist "%SITEPACKAGES%\nd2" xcopy /E /I /Y /Q "%SITEPACKAGES%\nd2" "BactiMate-Windows\nd2" >nul
    if exist "%SITEPACKAGES%\cellpose" xcopy /E /I /Y /Q "%SITEPACKAGES%\cellpose" "BactiMate-Windows\cellpose" >nul
    if exist "%SITEPACKAGES%\cellpose_omni" xcopy /E /I /Y /Q "%SITEPACKAGES%\cellpose_omni" "BactiMate-Windows\cellpose_omni" >nul
    if exist "%SITEPACKAGES%\omnipose" xcopy /E /I /Y /Q "%SITEPACKAGES%\omnipose" "BactiMate-Windows\omnipose" >nul
    if exist "%SITEPACKAGES%\networkx" xcopy /E /I /Y /Q "%SITEPACKAGES%\networkx" "BactiMate-Windows\networkx" >nul
    if exist "%SITEPACKAGES%\trackastra" xcopy /E /I /Y /Q "%SITEPACKAGES%\trackastra" "BactiMate-Windows\trackastra" >nul
    REM Optional deps (only if present)
    if exist "%SITEPACKAGES%\dask" xcopy /E /I /Y /Q "%SITEPACKAGES%\dask" "BactiMate-Windows\dask" >nul
    if exist "%SITEPACKAGES%\numba" xcopy /E /I /Y /Q "%SITEPACKAGES%\numba" "BactiMate-Windows\numba" >nul
    if exist "%SITEPACKAGES%\xarray" xcopy /E /I /Y /Q "%SITEPACKAGES%\xarray" "BactiMate-Windows\xarray" >nul
    if exist "%SITEPACKAGES%\wandb" xcopy /E /I /Y /Q "%SITEPACKAGES%\wandb" "BactiMate-Windows\wandb" >nul
    echo Heavy packages copied.
)

echo [5/5] Done. BactiMate-Windows\ is ready.
echo Artifacts: BactiMate-Windows\BactiMate.exe (or omnipose_batch_gui.exe) and dependencies.
endlocal
