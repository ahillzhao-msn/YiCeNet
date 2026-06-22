# Deploy yicenet into the Hermes Agent venv.
# Run from anywhere inside the YiCeNet tree (or override paths with params).
#
#   .\scripts\deploy-hermes.ps1
#   .\scripts\deploy-hermes.ps1 -SkipBuild       # re-install last wheel, skip rebuild
#   .\scripts\deploy-hermes.ps1 -ProjectDir C:\path\to\YiCeNet
param(
    [string]$ProjectDir = (Split-Path $PSScriptRoot -Parent),
    [string]$HermesDir  = "$env:LOCALAPPDATA\hermes\hermes-agent",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# 1. Build
if (-not $SkipBuild) {
    Write-Host "Building yicenet from $ProjectDir ..."
    uv --project $ProjectDir build
    if ($LASTEXITCODE -ne 0) { throw "uv build failed" }
}

# 2. Auto-detect newest wheel in dist/
$distDir = Join-Path $ProjectDir "dist"
$whl = Get-ChildItem -Path "$distDir\*.whl" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $whl) {
    throw "No .whl found in $distDir -- run without -SkipBuild first."
}
Write-Host "Wheel: $($whl.Name)"

# 3. Install into Hermes venv
Write-Host "Installing into $HermesDir ..."
uv --project $HermesDir pip install --force-reinstall --no-deps $whl.FullName
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }

# 4. Verify
$py = Join-Path $HermesDir "venv\Scripts\python.exe"
if (Test-Path $py) {
    $ver = & $py -c "import yicenet; print(yicenet.__version__)"
    Write-Host "Installed: yicenet $ver"
}
