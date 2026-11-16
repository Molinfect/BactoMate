$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $root 'Logs\install_last.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File | Unblock-File -ErrorAction SilentlyContinue
$env:BACTOMATE_INSTALL_LOGGED = '1'
# Keep stderr in the transcript without treating native diagnostics as PowerShell errors.
$ErrorActionPreference = 'Continue'
# Pass the path via an env var: Windows PowerShell re-quotes native arguments that contain spaces,
# which broke paths like "BactoMate (1)!". /s strips only the outer quote pair, leaving "<path>".
$env:BACTOMATE_INSTALL_BAT = Join-Path $PSScriptRoot 'install.bat'
& $env:ComSpec /d /v:off /s /c '""%BACTOMATE_INSTALL_BAT%""' 2>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
