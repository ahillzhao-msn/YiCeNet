@echo off
REM YiCeNet Flywheel — environment-adaptive launcher (Windows)
REM
REM Auto-detects the Python environment:
REM   1. Hermes Agent venv  (%LOCALAPPDATA%\hermes\hermes-agent\venv)
REM   2. System python (via PATH)
REM
REM Logs to %%USERPROFILE%%\.yicenet\logs\flywheel.log

setlocal enabledelayedexpansion

set "YICENET_HOME=%USERPROFILE%\.yicenet"
set "LOG=%YICENET_HOME%\logs\flywheel.log"

if not exist "%YICENET_HOME%\logs\" mkdir "%YICENET_HOME%\logs"

echo [%DATE% %TIME%] Flywheel run starting >> "%LOG%"

REM ── 1) Hermes Agent venv ──
set "HERMES_PY=%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe"
if exist "!HERMES_PY!" (
    "!HERMES_PY!" -m yicenet.flywheel >> "%LOG%" 2>&1
    set "EC=!ERRORLEVEL!"
    if !EC! equ 0 (
        echo [%DATE% %TIME%] Flywheel run finished (exit=0, hermes-venv) >> "%LOG%"
        exit /b 0
    )
)

REM ── 2) System Python ──
python -m yicenet.flywheel >> "%LOG%" 2>&1
set "EC=!ERRORLEVEL!"
if !EC! equ 0 (
    echo [%DATE% %TIME%] Flywheel run finished (exit=0, system-python) >> "%LOG%"
    exit /b 0
)

echo [%DATE% %TIME%] Flywheel run failed (exit=!EC!) — no usable python >> "%LOG%"
exit /b 1
