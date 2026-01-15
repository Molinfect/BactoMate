@echo off
REM Build script for Windows using Nuitka
REM This creates a standalone executable that should avoid PyInstaller issues

echo ==========================================
echo Building BactiMate for Windows with Nuitka
echo ==========================================
echo.

REM Check if Nuitka is installed
python -c "import nuitka" 2>nul
if errorlevel 1 (
    echo Installing Nuitka...
    pip install nuitka
)

REM Apply all patches before building
echo Applying compatibility patches...
if exist apply_all_patches.py (
    python apply_all_patches.py
)

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist BactiMate.exe del /q BactiMate.exe

REM Create output directory
if not exist dist mkdir dist

REM Build with Nuitka
echo.
echo Building with Nuitka (this may take a while)...
echo.

python -m nuitka ^
    --standalone ^
    --onefile ^
    --windows-icon-from-ico=BactiMate.ico ^
    --output-dir=dist ^
    --output-filename=BactiMate.exe ^
    --follow-imports ^
    --include-package=omnipose ^
    --include-package=cellpose_omni ^
    --include-package=PyQt6 ^
    --include-package=numpy ^
    --include-package=scipy ^
    --include-package=skimage ^
    --include-package=torch ^
    --include-package=matplotlib ^
    --include-package-data=omnipose ^
    --include-package-data=cellpose_omni ^
    --include-package-data=PyQt6 ^
    --enable-plugin=pyqt6 ^
    --nofollow-import-to=tkinter ^
    --nofollow-import-to=matplotlib.tests ^
    --nofollow-import-to=numpy.tests ^
    --nofollow-import-to=scipy.tests ^
    --nofollow-import-to=pandas.tests ^
    omnipose_batch_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Nuitka build failed!
    exit /b 1
)

echo.
echo ==========================================
echo Build complete!
echo ==========================================
echo.
echo The executable is located in: dist\BactiMate.exe
echo.
echo NOTE: Nuitka builds may require testing to ensure all dependencies
echo       are properly included. If you encounter issues, you may need
echo       to add additional --include-package flags.
echo.

pause
