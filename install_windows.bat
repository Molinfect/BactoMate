@echo off
setlocal EnableExtensions
:: Run from project root after extracting the ZIP anywhere (not only Desktop).
cd /d "%~dp0"
call "%~dp0setup\install_windows.bat" %*
