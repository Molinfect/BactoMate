@echo off
setlocal EnableExtensions
:: Build portable Windows bundle (dist\BactoMate\). Run from project root after install_windows.bat.
set "BUILD_DIR=%~dp0"
set "ROOT=%BUILD_DIR%.."
cd /d "%ROOT%"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run setup\install_windows.bat first.
    pause
    exit /b 1
)

echo ========== BactoMate Windows Bundle Build ==========
"venv\Scripts\python.exe" "%BUILD_DIR%build_windows_bundle.py"
set "EC=%ERRORLEVEL%"
if %EC% neq 0 pause
exit /b %EC%
