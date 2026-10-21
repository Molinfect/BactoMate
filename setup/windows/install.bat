@echo off
setlocal EnableDelayedExpansion
:: BactoMate Windows one-shot install. Strict: python.org 3.11 x64 only; all pip via venv\Scripts\python.exe.
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "WIN_DIR=%~dp0"
pushd "%WIN_DIR%..\.."
if errorlevel 1 (
    echo [ERROR] Could not resolve the BactoMate project root from:
    echo   %WIN_DIR%
    echo Extract the full ZIP and run setup\windows\install.bat or install_windows.bat at the project root.
    pause
    exit /b 1
)
set "ROOT=%CD%"
popd
cd /d "%ROOT%"
if not exist "%ROOT%\requirements.txt" (
    echo [ERROR] requirements.txt not found in the project root.
    echo Expected root: %ROOT%
    echo Setup script: %WIN_DIR%
    echo.
    echo Extract the full GitHub ZIP ^(folder must contain main.py, requirements.txt, setup\^).
    pause
    exit /b 1
)

echo.%ROOT%| findstr /i /c:"\Program Files" /c:"\Program Files (x86)" /c:"\Windows\" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Do not install under Program Files or Windows.
    echo Use a user-writable folder, e.g. C:\Users\%USERNAME%\BactoMate
    echo Current: %ROOT%
    pause
    exit /b 1
)

:: Read the release version without requiring a working Python installation.
set "BACTOMATE_VERSION=unknown"
for /f "tokens=2 delims== " %%V in ('findstr /b /c:"__version__ = " "%ROOT%\bactomate\__init__.py" 2^>nul') do set "BACTOMATE_VERSION=%%~V"

if not exist "%ROOT%\Logs" mkdir "%ROOT%\Logs" 2>nul
set "INSTALL_LOG=%ROOT%\Logs\install_last.log"
echo ========== BactoMate %BACTOMATE_VERSION% Windows source installer %DATE% %TIME% ========== > "%INSTALL_LOG%"
echo ROOT=%ROOT%>> "%INSTALL_LOG%"

:: ZIP downloads are often marked "blocked" outside Desktop; unblock scripts before pip/venv.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-ChildItem -LiteralPath '%ROOT%' -Recurse -Include *.bat,*.cmd,*.ps1,*.py -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue" ^
  >nul 2>&1

set "WRITE_TEST=%ROOT%\Logs\.install_write_test"
echo test>"%WRITE_TEST%" 2>nul
if not exist "%WRITE_TEST%" (
    echo [ERROR] Cannot write to the install folder:
    echo   %ROOT%
    echo Choose a folder you own ^(Documents, D:\LabTools, etc.^), not Program Files or a read-only share.
    echo Log: %INSTALL_LOG%
    pause
    exit /b 1
)
del /q "%WRITE_TEST%" >nul 2>&1

:: Ignore stray Python env vars from Conda/Homebrew/system during install.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"

echo.
echo ========== BactoMate %BACTOMATE_VERSION% Windows source installer ==========
echo Project root: %ROOT%
echo.

:: Step 1: Python 3.11 x64 (python.org / py launcher only — no generic python fallback)
echo [Step 1/8] Verifying Python 3.11 ^(64-bit, non-Conda^)...
set "HOST_PY="
set "HOST_PY_ERR=%TEMP%\bactomate_host_py_err_%RANDOM%.txt"
py -3.11 "%WIN_DIR%verify_python.py" > "%TEMP%\bactomate_host_py.txt" 2> "%HOST_PY_ERR%"
if errorlevel 1 (
    echo [BREAK] Python 3.11 verification failed.
    if exist "%HOST_PY_ERR%" type "%HOST_PY_ERR%"
    echo.
    echo   Fix: Install Python 3.11 ^(64-bit^) from https://www.python.org/downloads/
    echo   Tick "Add python.exe to PATH". Do not use Conda or Microsoft Store Python.
    if exist "%HOST_PY_ERR%" del /q "%HOST_PY_ERR%" >nul 2>&1
    pause
    exit /b 1
)
if exist "%HOST_PY_ERR%" del /q "%HOST_PY_ERR%" >nul 2>&1
set "HOST_PY="
for /f "usebackq delims=" %%I in ("%TEMP%\bactomate_host_py.txt") do set "HOST_PY=%%I"
del /q "%TEMP%\bactomate_host_py.txt" >nul 2>&1
if not defined HOST_PY (
    echo [BREAK] Could not read Python 3.11 path from verify script.
    pause
    exit /b 1
)
if not exist "!HOST_PY!" (
    echo [BREAK] Could not resolve Python 3.11 executable.
    pause
    exit /b 1
)
echo   Using host Python: !HOST_PY!
echo.

:: Step 2: venv
echo [Step 2/8] Creating virtual environment...
set "PYEXE=%ROOT%\venv\Scripts\python.exe"
set "PIPEXE=%ROOT%\venv\Scripts\pip.exe"
if exist "%PYEXE%" (
    echo   venv already exists; skipping create.
) else (
    "!HOST_PY!" -m venv "%ROOT%\venv"
    if errorlevel 1 (
        echo [BREAK] Failed to create venv.
        echo   Fix: Run "!HOST_PY!" -m venv venv manually and check for errors.
        pause
        exit /b 1
    )
)
if not exist "%PYEXE%" (
    echo [BREAK] venv python not found at %PYEXE%
    pause
    exit /b 1
)
echo.

:: Step 3: pip 24.x (stable on Windows; avoid 26.x churn)
echo [Step 3/8] Ensuring pip 24.x in BactoMate venv...
"%PYEXE%" -m pip install "pip>=24.2,<26"
if errorlevel 1 (
    echo [BREAK] pip install failed.
    echo   Fix: "%PYEXE%" -m pip install "pip>=24.2,<26"
    pause
    exit /b 1
)
echo.

:: Step 4: dbscan prebuilt wheel (omnipose dep; 1.0.0 is sdist-only on Windows)
echo [Step 4/8] Installing dbscan 0.0.10 ^(prebuilt Windows wheel^)...
"%PYEXE%" -m pip install "dbscan==0.0.10" --only-binary dbscan
if errorlevel 1 (
    echo [BREAK] dbscan prebuilt wheel install failed.
    echo   See docs/INSTALL_AND_LAUNCH_windows.md - Troubleshooting: Microsoft Visual C++ 14 / Failed building wheel for dbscan
    pause
    exit /b 1
)
echo.

:: Step 5: requirements
echo [Step 5/8] Installing dependencies ^(this may take several minutes^)...
"%PYEXE%" -m pip install --use-deprecated=legacy-resolver -r requirements.txt -c "%WIN_DIR%constraints.txt"
if errorlevel 1 (
    echo [BREAK] pip install -r requirements.txt failed.
    echo   Fix: "%PYEXE%" -m pip install --use-deprecated=legacy-resolver -r requirements.txt -c setup\windows\constraints.txt
    echo   If you see resolution errors, check docs/INSTALL_AND_LAUNCH_windows.md
    pause
    exit /b 1
)
echo.

:: Step 6: editable install
echo [Step 6/8] Installing BactoMate in editable mode...
"%PYEXE%" -m pip install -e .
if errorlevel 1 (
    echo [BREAK] pip install -e . failed.
    echo   Fix: "%PYEXE%" -m pip install -e .
    pause
    exit /b 1
)
echo.

:: Step 7: PyTorch CUDA (optional)
echo [Step 7/8] Detecting NVIDIA GPU for PyTorch CUDA wheel...

:: 7a) Skip only when torch, torchvision, and torchaudio are a matched bundle.
"%PYEXE%" "%WIN_DIR%check_torch_cuda_bundle.py" >nul 2>&1
if not errorlevel 1 (
    echo   Matched torch/torchvision/torchaudio; skipping PyTorch reinstall.
    goto :step7_done
)
echo   Realigning PyTorch ^(version mismatch detected^)...
set "TORCH_REALIGN=1"

:: 7b) Detect CUDA wheel index
set "CU_INDEX="
set "CU_TMP=%TEMP%\bactomate_cuda_%RANDOM%.txt"
del /q "%CU_TMP%" >nul 2>&1
"%PYEXE%" "%WIN_DIR%get_cuda_wheel_index.py" > "%CU_TMP%" 2>nul
if exist "%CU_TMP%" for /f "usebackq delims=" %%I in ("%CU_TMP%") do set "CU_INDEX=%%I"
if exist "%CU_TMP%" del /q "%CU_TMP%" >nul 2>&1
if not defined CU_INDEX (
    echo   No NVIDIA GPU detected; installing matched CPU PyTorch wheels...
    "%PYEXE%" -m pip install --force-reinstall --no-deps "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
    if errorlevel 1 (
        echo [WARNING] PyTorch CPU reinstall failed. Try manually:
        echo   "%PYEXE%" -m pip install --force-reinstall "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
    ) else (
        "%PYEXE%" "%WIN_DIR%check_torch_cuda_bundle.py" >nul 2>&1
        if not errorlevel 1 (
            echo   Matched CPU PyTorch installed.
        ) else (
            echo [WARNING] PyTorch still mismatched after CPU reinstall.
        )
    )
    goto :step7_done
)

echo   Detected CUDA index: !CU_INDEX!
echo   Installing PyTorch with GPU support...
set "CU_INSTALLED="
set "CU_USED="
for %%I in (!CU_INDEX! cu128 cu126 cu124 cu121 cu118) do (
    if not defined CU_INSTALLED (
        echo   Trying wheel index %%I ...
        "%PYEXE%" -m pip install --upgrade --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%%I
        if not errorlevel 1 (
            "%PYEXE%" "%WIN_DIR%torch_cuda_available.py" >nul 2>&1
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
    echo   "%PYEXE%" -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!CU_INDEX!
    echo   Or install VC++ Redistributable ^(64-bit^) from Microsoft, then retry.
    goto :step7_done
)
"%PYEXE%" "%WIN_DIR%check_torch_cuda_bundle.py" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] torch/torchvision/torchaudio still incompatible after CUDA install.
    echo   Run: "%PYEXE%" -m pip install --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!CU_USED!
)
:step7_done
if defined TORCH_REALIGN (
    echo   Restoring numpy/tensorflow-compatible pins after PyTorch reinstall...
    "%PYEXE%" -m pip install "numpy>=1.26.0,<2.0.0" -q
    set "TORCH_REALIGN="
)
echo.

:: Step 7.5: Apply BactoMate compatibility patches
echo [Step 7.5] Applying BactoMate compatibility patches...
set "PYTHONIOENCODING=utf-8"
"%PYEXE%" "%ROOT%\setup\apply_all_patches.py"
if errorlevel 1 (
    echo [WARNING] Some patches failed to apply. Segmentation may fail with errors like
    echo   "Cellpose.__init__^(^) got an unexpected keyword argument 'pretrained_model'".
    echo   You can re-run the patches manually with:
    echo     "%PYEXE%" setup\apply_all_patches.py
)
echo.

:: Step 7.9: verify install
echo [Step 7.9] Verifying install...
"%PYEXE%" "%WIN_DIR%ensure_spotiflow_deps.py"
if errorlevel 1 (
    echo [BREAK] Spotiflow dependency check failed ^(wandb/protobuf^).
    echo   Fix: "%PYEXE%" setup\windows\ensure_spotiflow_deps.py
    echo   See docs/INSTALL_AND_LAUNCH_windows.md
    pause
    exit /b 1
)
"%PYEXE%" "%ROOT%\main.py" --version
if errorlevel 1 (
    echo [BREAK] main.py --version failed.
    pause
    exit /b 1
)
"%PYEXE%" "%ROOT%\main.py" --smoke-test
if errorlevel 1 (
    echo [BREAK] Smoke test failed. See docs/INSTALL_AND_LAUNCH_windows.md
    pause
    exit /b 1
)
echo   Smoke test OK.
echo.

:: Step 8: strict launchers + shortcut
echo [Step 8/8] Creating launch_windows.bat and BactoMate.lnk...
copy /y "%WIN_DIR%launch_strict.bat" "%ROOT%\launch_windows.bat" >nul
if exist "%ROOT%\launch_windows.bat" (
    echo   Created: %ROOT%\launch_windows.bat ^(strict venv-only launcher^)
) else (
    echo [WARNING] Could not write launch_windows.bat.
)
"%PYEXE%" "%WIN_DIR%create_launcher_assets.py"
if exist "%ROOT%\BactoMate.lnk" (
    echo   Created: %ROOT%\BactoMate.lnk ^(BactoMate logo - use this to launch^)
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
echo BactoMate uses only: %PYEXE%
echo Install log: %INSTALL_LOG%
echo Run by double-clicking BactoMate.lnk or launch_windows.bat:
echo   %ROOT%\BactoMate.lnk
echo   %ROOT%\launch_windows.bat
echo.
echo To build a portable folder for other PCs ^(no separate Python install^):
echo   build\windows\build_windows_bundle.bat
echo.
pause
