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
echo Large bundles can take several minutes. Please wait.

set "SEVENZIP="
if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles%\7-Zip\7z.exe"
if not defined SEVENZIP if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"

if defined SEVENZIP (
    echo Using 7-Zip ^(fast^) ...
    "%SEVENZIP%" a -tzip -mx=1 -bsp1 "%ZIP%" "%BUNDLE%\*"
    if errorlevel 2 goto zipfail
    goto zipdone
)

where tar >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No zip tool found. Install 7-Zip or use Windows 10+ ^(includes tar^).
    pause
    exit /b 1
)

echo Using built-in tar ^(faster than PowerShell Compress-Archive^) ...
tar -acf "%ZIP%" -C "%BUNDLE%" .
if errorlevel 1 goto zipfail

:zipdone
echo.
echo Created: %ZIP%
for %%A in ("%ZIP%") do echo Size: %%~zA bytes
pause
exit /b 0

:zipfail
echo [ERROR] Failed to create zip.
pause
exit /b 1
