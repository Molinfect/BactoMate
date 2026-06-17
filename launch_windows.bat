@echo off
set "ROOT=%~dp0"
cd /d "%ROOT%"
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Run setup\install_windows.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%setup\minimize_console.ps1" >nul 2>&1
bactomate
if errorlevel 1 pause
