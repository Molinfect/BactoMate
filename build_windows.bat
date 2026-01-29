@echo off
REM Wrapper for Windows build. Calls the actual Nuitka build script.
REM Use this or build_windows_bundle.bat directly - both produce BactiMate-Windows\.
cd /d "%~dp0"
if not exist "build_windows_bundle.bat" (
    echo ERROR: build_windows_bundle.bat not found. Run from repo root.
    exit /b 1
)
call "build_windows_bundle.bat"
exit /b %errorlevel%
