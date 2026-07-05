@echo off
:: Back-compat stub — use setup\windows\install.bat
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0windows\install.bat" %*
