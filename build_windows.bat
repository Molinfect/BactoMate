@echo off
REM Build script for Windows executable
REM This creates a standalone .exe file that can run on any Windows machine

echo ==========================================
echo Building BactiMate for Windows
echo ==========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist BactiMate.spec echo Spec file exists, using it...

REM Build the executable
echo.
echo Building executable...
pyinstaller BactiMate.spec

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Build complete!
echo ==========================================
echo.
echo The executable is located in: dist\BactiMate.exe
echo.
echo You can now distribute this .exe file to users.
echo They can run it by double-clicking - no Python installation needed!
echo.
pause
