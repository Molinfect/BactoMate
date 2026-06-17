@echo off
setlocal EnableExtensions
:: Strict launcher when invoked from setup\ (project root is parent).
set "SETUP_DIR=%~dp0"
set "ROOT=%SETUP_DIR%.."
cd /d "%ROOT%"

set "PYEXE=%ROOT%\venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [ERROR] BactoMate venv not found.
    echo Run setup\install_windows.bat first.
    echo See docs/INSTALL_AND_LAUNCH_windows.md for manual steps.
    pause
    exit /b 1
)

set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "VIRTUAL_ENV=%ROOT%\venv"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\setup\minimize_console.ps1" >nul 2>&1
"%PYEXE%" -m bactomate %*
set "EC=%ERRORLEVEL%"
if %EC% neq 0 (
    echo.
    echo BactoMate exited with error code %EC%.
    pause
)
exit /b %EC%
