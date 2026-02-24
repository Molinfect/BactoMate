@echo off
setlocal EnableDelayedExpansion
:: BactiMate Windows one-shot install. Run from anywhere; script dir is setup/, project root is parent.
set "SETUP_DIR=%~dp0"
set "ROOT=%SETUP_DIR%.."
cd /d "%ROOT%"
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found. This script must live in the BactiMate project.
    echo Current dir: %CD%
    echo Setup dir: %SETUP_DIR%
    pause
    exit /b 1
)

echo.
echo ========== BactiMate Windows Install ==========
echo Project root: %ROOT%
echo.

:: Step 1: Python
echo [Step 1/7] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [BREAK] Python 3.11 not found.
        echo   Fix: Install Python 3.11 from https://www.python.org/downloads/
        echo   Tick "Add python.exe to PATH" in the installer.
        pause
        exit /b 1
    )
    set "PY=python"
) else (
    set "PY=py -3.11"
)
echo   Using: %PY%
echo.

:: Step 2: venv
echo [Step 2/7] Creating virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo   venv already exists; skipping create.
) else (
    %PY% -m venv venv
    if errorlevel 1 (
        echo [BREAK] Failed to create venv.
        echo   Fix: Run "%PY% -m venv venv" manually and check for errors.
        pause
        exit /b 1
    )
)
echo.

:: Step 3: activate and upgrade pip
echo [Step 3/7] Activating venv and upgrading pip...
call venv\Scripts\activate.bat
pip install --upgrade pip
if errorlevel 1 (
    echo [BREAK] pip upgrade failed.
    echo   Fix: Run "venv\Scripts\activate" then "pip install --upgrade pip".
    pause
    exit /b 1
)
echo.

:: Step 4: requirements
echo [Step 4/7] Installing dependencies (this may take several minutes)...
pip install --use-deprecated=legacy-resolver -r requirements.txt
if errorlevel 1 (
    echo [BREAK] pip install -r requirements.txt failed.
    echo   Fix: With venv active, try: pip install --use-deprecated=legacy-resolver -r requirements.txt
    echo   If you see resolution errors, check docs/INSTALL_AND_LAUNCH_windows.md
    pause
    exit /b 1
)
echo.

:: Step 5: editable install
echo [Step 5/7] Installing BactiMate in editable mode...
pip install -e .
if errorlevel 1 (
    echo [BREAK] pip install -e . failed.
    echo   Fix: With venv active, run "pip install -e ." from the project root.
    pause
    exit /b 1
)
echo.

:: Step 6: PyTorch CUDA (optional)
echo [Step 6/7] Detecting NVIDIA GPU for PyTorch CUDA wheel...
set "CU_INDEX="
for /f "delims=" %%a in ('python "%SETUP_DIR%get_cuda_wheel_index.py" 2^>nul') do set "CU_INDEX=%%a"
if defined CU_INDEX (
    echo   Detected CUDA index: %CU_INDEX%
    echo   Installing PyTorch with GPU support...
    pip uninstall torch torchvision torchaudio -y >nul 2>&1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CU_INDEX%
    if errorlevel 1 (
        echo [WARNING] PyTorch CUDA install failed. You can retry later with:
        echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CU_INDEX%
        echo   Or install VC++ Redistributable (64-bit) from Microsoft, then retry.
    ) else (
        echo   PyTorch + %CU_INDEX% installed.
    )
) else (
    echo   No nvidia-smi or no GPU; skipping PyTorch CUDA. BactiMate will use CPU.
    echo   To add GPU later: see docs/INSTALL_AND_LAUNCH_windows.md
)
echo.

:: Step 7: create launch script
echo [Step 7/7] Creating launch_windows.bat...
(
    echo @echo off
    echo set "ROOT=%%~dp0"
    echo cd /d "%%ROOT%%"
    echo if not exist "venv\Scripts\activate.bat" ^(
    echo     echo [ERROR] venv not found. Run setup\install_windows.bat first.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo call venv\Scripts\activate.bat
    echo bactimate
    echo if errorlevel 1 pause
) > "%ROOT%\launch_windows.bat"
if errorlevel 1 (
    echo [WARNING] Could not write launch_windows.bat. You can launch with: venv\Scripts\activate then bactimate
) else (
    echo   Created: %ROOT%\launch_windows.bat
)
echo.
echo ========== Install complete ==========
echo Run BactiMate by double-clicking launch_windows.bat or:
echo   venv\Scripts\activate
echo   bactimate
echo.
pause
