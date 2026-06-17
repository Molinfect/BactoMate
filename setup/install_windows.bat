@echo off
setlocal EnableDelayedExpansion
:: BactoMate Windows one-shot install. Run from anywhere; script dir is setup/, project root is parent.
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "SETUP_DIR=%~dp0"
set "ROOT=%SETUP_DIR%.."
cd /d "%ROOT%"
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found. This script must live in the BactoMate project.
    echo Current dir: %CD%
    echo Setup dir: %SETUP_DIR%
    pause
    exit /b 1
)

echo.
echo ========== BactoMate Windows Install ==========
echo Project root: %ROOT%
echo.

:: Step 1: Python
echo [Step 1/7] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [BREAK] Python 3.11 not found.
        echo   Fix: Install Python 3.11 ^from https://www.python.org/downloads/
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

:: Step 3: activate and ensure a stable pip (24.x; avoid 26.x upgrade churn on Windows)
echo [Step 3/7] Activating venv and ensuring pip 24.x...
call venv\Scripts\activate.bat
python -m pip install "pip>=24.2,<26"
if errorlevel 1 (
    echo [BREAK] pip install failed.
    echo   Fix: Run "venv\Scripts\activate" then "python -m pip install pip^>=24.2,^<26".
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
echo [Step 5/7] Installing BactoMate in editable mode...
pip install -e .
if errorlevel 1 (
    echo [BREAK] pip install -e . failed.
    echo   Fix: With venv active, run "pip install -e ." ^from the project root.
    pause
    exit /b 1
)
echo.

:: Step 6: PyTorch CUDA (optional)
echo [Step 6/7] Detecting NVIDIA GPU for PyTorch CUDA wheel...
set "PYEXE=venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [WARNING] venv python not found; skipping PyTorch CUDA step.
    goto :step6_done
)

:: 6a) Skip only when torch, torchvision, and torchaudio are a matched bundle (not just cuda.is_available).
"%PYEXE%" "%SETUP_DIR%check_torch_cuda_bundle.py" >nul 2>&1
if not errorlevel 1 (
    echo   Matched torch/torchvision/torchaudio; skipping PyTorch reinstall.
    goto :step6_done
)
echo   Realigning PyTorch (version mismatch detected)...
set "TORCH_REALIGN=1"

:: 6b) Detect CUDA wheel index (unique temp name per run; for /f avoids set /p parse bugs).
set "CU_INDEX="
set "CU_TMP=%TEMP%\bactomate_cuda_%RANDOM%.txt"
del /q "%CU_TMP%" >nul 2>&1
"%PYEXE%" "%SETUP_DIR%get_cuda_wheel_index.py" > "%CU_TMP%" 2>nul
if exist "%CU_TMP%" for /f "usebackq delims=" %%I in ("%CU_TMP%") do set "CU_INDEX=%%I"
if exist "%CU_TMP%" del /q "%CU_TMP%" >nul 2>&1
if not defined CU_INDEX (
    echo   No NVIDIA GPU detected; installing matched CPU PyTorch wheels...
    pip install --force-reinstall --no-deps "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
    if errorlevel 1 (
        echo [WARNING] PyTorch CPU reinstall failed. Try manually:
        echo   pip install --force-reinstall "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
    ) else (
        "%PYEXE%" "%SETUP_DIR%check_torch_cuda_bundle.py" >nul 2>&1
        if not errorlevel 1 (
            echo   Matched CPU PyTorch installed.
        ) else (
            echo [WARNING] PyTorch still mismatched after CPU reinstall.
        )
    )
    goto :step6_done
)

echo   Detected CUDA index: !CU_INDEX!
echo   Installing PyTorch with GPU support...
set "CU_INSTALLED="
set "CU_USED="
for %%I in (!CU_INDEX! cu128 cu126 cu124 cu121 cu118) do (
    if not defined CU_INSTALLED (
        echo   Trying wheel index %%I ...
        pip install --upgrade --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%%I
        if not errorlevel 1 (
            "%PYEXE%" "%SETUP_DIR%torch_cuda_available.py" >nul 2>&1
            if not errorlevel 1 (
                set "CU_INSTALLED=1"
                set "CU_USED=%%I"
            )
        )
    )
)
if defined CU_INSTALLED (
    echo   PyTorch + !CU_USED! installed.
) else (
    echo [WARNING] PyTorch CUDA install failed. You can retry later with:
    echo   pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!CU_INDEX!
    echo   Or install VC++ Redistributable ^(64-bit^) from Microsoft, then retry.
    goto :step6_done
)
"%PYEXE%" "%SETUP_DIR%check_torch_cuda_bundle.py" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] torch/torchvision/torchaudio still incompatible after CUDA install.
    echo   Run: pip install --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!CU_USED!
)
:step6_done
if defined TORCH_REALIGN (
    echo   Restoring numpy/tensorflow-compatible pins after PyTorch reinstall...
    pip install "numpy>=1.26.0,<2.0.0" -q
    set "TORCH_REALIGN="
)
echo.

:: Step 6.5: Apply BactoMate compatibility patches (cellpose_omni, omnipose, etc.)
echo [Step 6.5] Applying BactoMate compatibility patches...
set "PYTHONIOENCODING=utf-8"
python "%SETUP_DIR%apply_all_patches.py"
if errorlevel 1 (
    echo [WARNING] Some patches failed to apply. Segmentation may fail with errors like
    echo   "Cellpose.__init__^(^) got an unexpected keyword argument 'pretrained_model'".
    echo   You can re-run the patches manually with:
    echo     venv\Scripts\activate ^&^& python setup\apply_all_patches.py
)
echo.

:: Step 6.9: verify install
echo [Step 6.9] Verifying install...
python main.py --version
if errorlevel 1 (
    echo [BREAK] python main.py --version failed.
    pause
    exit /b 1
)
python main.py --smoke-test
if errorlevel 1 (
    echo [BREAK] Smoke test failed. See docs/INSTALL_AND_LAUNCH_windows.md
    pause
    exit /b 1
)
echo   Smoke test OK.
echo.

:: Step 7: create launch script + BactoMate shortcut (custom icon)
echo [Step 7/7] Creating launch_windows.bat and BactoMate.lnk...
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
    echo powershell -NoProfile -ExecutionPolicy Bypass -File "%%ROOT%%setup\minimize_console.ps1" ^>nul 2^>^&1
    echo bactomate
    echo if errorlevel 1 pause
) > "%ROOT%\launch_windows.bat"
if exist "%ROOT%\launch_windows.bat" (
    echo   Created: %ROOT%\launch_windows.bat
) else (
    echo [WARNING] Could not write launch_windows.bat. You can launch with: venv\Scripts\activate then bactomate
)
"%PYEXE%" "%SETUP_DIR%create_windows_launcher_assets.py"
if exist "%ROOT%\BactoMate.lnk" (
    echo   Created: %ROOT%\BactoMate.lnk ^(BactoMate logo — use this to launch^)
) else (
    echo [WARNING] Could not create BactoMate.lnk shortcut. Use launch_windows.bat to start.
)
if exist "%USERPROFILE%\Desktop\BactoMate.lnk" (
    echo   Created desktop shortcut: %USERPROFILE%\Desktop\BactoMate.lnk
) else (
    echo [INFO] Desktop shortcut not created ^(Desktop folder may be redirected^).
)
echo.
echo ========== Install complete ==========
echo Run BactoMate by double-clicking BactoMate.lnk ^(logo icon^) or launch_windows.bat:
echo   %ROOT%\BactoMate.lnk
echo   %ROOT%\launch_windows.bat
echo A BactoMate.lnk shortcut is also placed on your Desktop when possible.
echo.
pause
