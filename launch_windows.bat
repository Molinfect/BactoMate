@echo off
setlocal EnableExtensions
:: Strict BactoMate launcher: only venv\Scripts\python.exe; ignores system/Homebrew/Conda Python.
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYEXE=%ROOT%venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [ERROR] BactoMate venv not found at:
    echo   %PYEXE%
    echo Run setup\windows\install.bat first.
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
