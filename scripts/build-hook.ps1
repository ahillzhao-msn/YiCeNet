# Build the native hook client (native/yicenet-hook) and install it to ~/.yicenet/bin.
# Needs CMake + a C++ compiler; uses the CMake bundled with Visual Studio when none is on PATH.
#
#   .\scripts\build-hook.ps1            # build + install
#   .\scripts\build-hook.ps1 -OutDir X  # install somewhere else
param(
    [string]$ProjectDir = (Split-Path $PSScriptRoot -Parent),
    [string]$OutDir = "$env:USERPROFILE\.yicenet\bin"
)
$ErrorActionPreference = "Stop"

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $cand = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if ($vs -and (Test-Path $cand)) { $cmake = $cand }
    }
}
if (-not $cmake) { throw "CMake not found (install Visual Studio C++ tools or CMake)" }

$src = Join-Path $ProjectDir "native\yicenet-hook"
$build = Join-Path $ProjectDir "build\yicenet-hook"
$old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
try {
    & $cmake -S $src -B $build 2>&1 | ForEach-Object { "$_" } | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
    & $cmake --build $build --config Release 2>&1 | ForEach-Object { "$_" } | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
} finally { $ErrorActionPreference = $old }

$exe = Get-ChildItem -Path $build -Recurse -Filter "yicenet-hook.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $exe) { throw "yicenet-hook.exe not produced" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
# A running hook holds the exe open; rename it aside instead of failing the copy.
$dst = Join-Path $OutDir "yicenet-hook.exe"
if (Test-Path $dst) {
    Remove-Item "$dst.old" -ErrorAction SilentlyContinue
    try { Remove-Item $dst } catch { Rename-Item $dst "$dst.old" }
}
Copy-Item $exe.FullName $dst
Write-Host "Installed $dst"
