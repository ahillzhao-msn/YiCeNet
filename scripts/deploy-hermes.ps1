# Deploy yicenet into the Hermes Agent venv and refresh the Claude Code plugin (hooks).
# Run from anywhere inside the YiCeNet tree (or override paths with params).
#
#   .\scripts\deploy-hermes.ps1               # build wheel, install, refresh hooks, restart daemon
#   .\scripts\deploy-hermes.ps1 -SkipBuild    # re-install the newest wheel in dist\
#   .\scripts\deploy-hermes.ps1 -Editable     # dev machine: pip install -e the source tree
#   .\scripts\deploy-hermes.ps1 -ProjectDir C:\path\to\YiCeNet
param(
    [string]$ProjectDir = (Split-Path $PSScriptRoot -Parent),
    [string]$HermesDir  = "$env:LOCALAPPDATA\hermes\hermes-agent",
    [switch]$SkipBuild,
    [switch]$Editable
)

$ErrorActionPreference = "Stop"

# Native tools (uv, python) write progress to stderr; Windows PowerShell 5.1 turns any native
# stderr line into a terminating error under "Stop". Run them with "Continue" and judge by exit code.
function Invoke-Native([string]$What, [scriptblock]$Cmd) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Cmd 2>&1 | ForEach-Object { "$_" } | Out-Host; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "$What failed (exit $code)" }
}

# 1. Hermes venv python (uv projects use .venv, older installs use venv)
$py = @("$HermesDir\.venv\Scripts\python.exe", "$HermesDir\venv\Scripts\python.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) { throw "Hermes venv python not found under $HermesDir" }
Write-Host "Hermes python: $py"

# 2. Install yicenet (dependencies resolved, yicenet itself always reinstalled)
if ($Editable) {
    Write-Host "Installing editable from $ProjectDir ..."
    Invoke-Native "uv pip install -e" { uv pip install --python $py --reinstall-package yicenet -e $ProjectDir }
} else {
    if (-not $SkipBuild) {
        Write-Host "Building yicenet from $ProjectDir ..."
        Invoke-Native "uv build" { uv --project $ProjectDir build }
    }
    $whl = Get-ChildItem -Path "$ProjectDir\dist\*.whl" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $whl) { throw "No .whl found in $ProjectDir\dist -- run without -SkipBuild first." }
    Write-Host "Wheel: $($whl.Name)"
    Invoke-Native "uv pip install" { uv pip install --python $py --reinstall-package yicenet $whl.FullName }
}

# 3. Post-deploy: local tokenizer (no network), Claude Code hooks, daemon restart
$post = @'
import pathlib
from yicenet import __version__
from yicenet.tokenizer import install_tokenizer
from yicenet.daemon.launcher import stop_daemon
install_tokenizer()
if (pathlib.Path.home() / ".claude").exists():
    from yicenet.install.claude import ClaudeCodeInstaller
    ClaudeCodeInstaller().register_hooks()   # rewrites ~/.claude/hooks runner + settings.json hooks
    print("Claude Code hooks refreshed")
stop_daemon()                                # next hook call respawns it on the new code
print("Installed: yicenet", __version__)
'@
# via a temp file: Windows PowerShell strips embedded double quotes from native arguments
$postFile = Join-Path $env:TEMP "yicenet-post-deploy.py"
Set-Content -Path $postFile -Value $post -Encoding utf8
$env:PYTHONIOENCODING = "utf-8"
try { Invoke-Native "post-deploy" { & $py $postFile } } finally { Remove-Item $postFile -ErrorAction SilentlyContinue }
