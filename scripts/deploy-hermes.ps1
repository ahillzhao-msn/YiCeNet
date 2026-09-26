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

# 1. Hermes venv python (uv projects use .venv, older installs use venv)
$py = @("$HermesDir\.venv\Scripts\python.exe", "$HermesDir\venv\Scripts\python.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) { throw "Hermes venv python not found under $HermesDir" }
Write-Host "Hermes python: $py"

# 2. Install yicenet (dependencies resolved, yicenet itself always reinstalled)
if ($Editable) {
    Write-Host "Installing editable from $ProjectDir ..."
    uv pip install --python $py --reinstall-package yicenet -e $ProjectDir
    if ($LASTEXITCODE -ne 0) { throw "uv pip install -e failed" }
} else {
    if (-not $SkipBuild) {
        Write-Host "Building yicenet from $ProjectDir ..."
        uv --project $ProjectDir build
        if ($LASTEXITCODE -ne 0) { throw "uv build failed" }
    }
    $whl = Get-ChildItem -Path "$ProjectDir\dist\*.whl" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $whl) { throw "No .whl found in $ProjectDir\dist -- run without -SkipBuild first." }
    Write-Host "Wheel: $($whl.Name)"
    uv pip install --python $py --reinstall-package yicenet $whl.FullName
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }
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
$env:PYTHONIOENCODING = "utf-8"
& $py -c $post
if ($LASTEXITCODE -ne 0) { throw "post-deploy step failed" }
