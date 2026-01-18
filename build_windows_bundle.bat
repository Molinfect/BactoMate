@echo off
REM Build script for Windows bundle
REM This creates a folder with the entire project and a launcher .bat file
REM Similar to the macOS .app bundle approach - no compilation needed

echo ==========================================
echo Building BactiMate Windows Bundle
echo ==========================================
echo.

REM Get the script directory (remove trailing backslash)
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

REM Clean previous builds
echo Cleaning previous builds...
if exist "BactiMate-Windows" rmdir /s /q "BactiMate-Windows"
if exist "dist\BactiMate-Windows" rmdir /s /q "dist\BactiMate-Windows"

REM Create bundle directory
echo Creating bundle structure...
set BUNDLE_DIR=BactiMate-Windows
mkdir "%BUNDLE_DIR%"

REM Create launcher batch file
echo Creating launcher...
(
echo @echo off
echo REM BactiMate launcher for Windows
echo.
echo REM Get the directory where this script is located
echo set SCRIPT_DIR=%%~dp0
echo cd /d "%%SCRIPT_DIR%"
echo.
echo REM Check if venv exists
echo if not exist "venv\Scripts\python.exe" ^(
echo     echo ERROR: Python virtual environment not found!
echo     echo Please ensure venv directory exists in the bundle.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Activate venv and run the GUI
echo echo Launching BactiMate...
echo call "venv\Scripts\activate.bat"
echo python omnipose_batch_gui.py
echo.
echo REM If there's an error, keep window open
echo if errorlevel 1 ^(
echo     echo.
echo     echo ERROR: BactiMate failed to launch.
echo     echo Check the error messages above.
echo     pause
echo ^)
) > "%BUNDLE_DIR%\BactiMate.bat"

REM Copy all project files (excluding build artifacts)
echo Copying project files to bundle...
echo   (this may take several minutes for large venv...)

REM Use robocopy for better progress and error handling
REM Exclude patterns: .git, __pycache__, build artifacts, etc.
REM Note: ROBOCOPY requires absolute paths and proper quoting
set "SOURCE_DIR=%SCRIPT_DIR%"
set "DEST_DIR=%SCRIPT_DIR%\%BUNDLE_DIR%"
robocopy "%SOURCE_DIR%" "%DEST_DIR%" /E /XD .git __pycache__ dist build BactiMate.app BactiMate-Windows venv_backup .venv /XF *.pyc *.pyo *.pyd .DS_Store *.log *.dmg nuitka-crash-report.xml /NFL /NDL /NP /NJH /NJS
REM ROBOCOPY returns non-zero exit codes even on success, so check errorlevel properly
if errorlevel 8 (
    echo ERROR: ROBOCOPY failed with error level %errorlevel%
    exit /b 1
)

REM Verify critical files were copied
if not exist "%BUNDLE_DIR%\venv" (
    echo ERROR: venv directory not found in bundle!
    exit /b 1
)

if not exist "%BUNDLE_DIR%\omnipose_batch_gui.py" (
    echo ERROR: omnipose_batch_gui.py not found in bundle!
    exit /b 1
)

REM Verify Python executable exists and is self-contained
echo Verifying Python executable...
if exist "%BUNDLE_DIR%\venv\Scripts\python.exe" (
    echo   Python executable found
) else (
    echo ERROR: Python executable not found in bundle!
    exit /b 1
)

REM Note: Windows Python is typically self-contained with DLLs in the venv
REM The --copies flag in venv creation ensures all files are copied, not symlinked
echo.
echo NOTE: Windows bundle includes the entire Python virtual environment.
echo       All Python DLLs and dependencies should be in venv\DLLs and venv\Scripts.
echo       The bundle should work on Windows machines without Python installed.

echo.
echo ==========================================
echo Build complete!
echo ==========================================
echo.
echo The Windows bundle is located at: %BUNDLE_DIR%
echo.
echo To test the bundle, run:
echo   %BUNDLE_DIR%\BactiMate.bat
echo.
echo To distribute, zip the entire "%BUNDLE_DIR%" folder.
echo Users can extract and run BactiMate.bat to launch the application.
echo.
REM Don't pause in CI environments (GitHub Actions)
if not "%CI%"=="true" (
    pause
)
