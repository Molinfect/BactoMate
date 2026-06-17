@echo off
:: BactoMate launcher for Windows. Prefer BactoMate.lnk (logo icon) in the project root.
set "SETUP_DIR=%~dp0"
set "ROOT=%SETUP_DIR%.."
cd /d "%ROOT%"
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found.
    echo Run setup\install_windows.bat first to install BactoMate.
    echo See docs/INSTALL_AND_LAUNCH_windows.md for manual steps.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\setup\minimize_console.ps1" >nul 2>&1
bactomate
if errorlevel 1 (
    echo.
    echo BactoMate exited with an error. Check the message above.
    pause
)
exit /b 0
