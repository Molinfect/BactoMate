@echo off
setlocal EnableExtensions
set "BUILD_DIR=%~dp0"
set "ROOT=%BUILD_DIR%.."
set "BUNDLE=%ROOT%\dist\BactoMate"
set "ZIP=%ROOT%\dist\BactoMate-win64.zip"

if not exist "%BUNDLE%\BactoMate.bat" (
    echo [ERROR] Bundle not found. Run build\build_windows_bundle.bat first.
    pause
    exit /b 1
)

if exist "%ZIP%" del /q "%ZIP%"

echo Creating %ZIP% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path '%BUNDLE%\*' -DestinationPath '%ZIP%' -Force"
if errorlevel 1 (
    echo [ERROR] Failed to create zip.
    pause
    exit /b 1
)

echo Created: %ZIP%
pause
