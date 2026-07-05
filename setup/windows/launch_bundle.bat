@echo off
setlocal EnableExtensions
:: Portable bundle launcher: patch venv paths, then run via bundled venv Python.
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "RUNTIME_PY=%ROOT%runtime\Python311\python.exe"
if not exist "%RUNTIME_PY%" (
    echo [ERROR] Bundled Python runtime not found at:
    echo   %RUNTIME_PY%
    pause
    exit /b 1
)

"%RUNTIME_PY%" "%ROOT%setup\windows\patch_bundle_pyvenv.py" "%ROOT%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to configure bundled virtual environment.
    pause
    exit /b 1
)

set "PYEXE=%ROOT%venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [ERROR] BactoMate venv not found at:
    echo   %PYEXE%
    pause
    exit /b 1
)

set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "VIRTUAL_ENV=%ROOT%venv"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%setup\windows\minimize_console.ps1" >nul 2>&1
"%PYEXE%" -m bactomate %*
set "EC=%ERRORLEVEL%"
if %EC% neq 0 (
    echo.
    echo BactoMate exited with error code %EC%.
    pause
)
exit /b %EC%
