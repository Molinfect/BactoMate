$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$log = Join-Path $root 'Logs\install_last.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File | Unblock-File -ErrorAction SilentlyContinue
$env:BACTOMATE_INSTALL_LOGGED = '1'
# Keep stderr in the transcript without treating native diagnostics as PowerShell errors.
$ErrorActionPreference = 'Continue'
& $env:ComSpec /d /v:off /c ('""{0}""' -f (Join-Path $PSScriptRoot 'install.bat')) 2>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
