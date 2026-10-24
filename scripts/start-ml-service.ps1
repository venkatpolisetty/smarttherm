$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$mlRoot = Join-Path $projectRoot "ml-integration"
$pythonEnvironment = Join-Path $mlRoot ".venv"
$python = Join-Path $pythonEnvironment "Scripts\python.exe"
$requirements = Join-Path $mlRoot "requirements.txt"

if (-not (Test-Path $python)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python was not found. Install 64-bit Python 3.12+ and reopen PowerShell."
    }
    python -m venv $pythonEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the SmartTherm ML virtual environment."
    }
}

$versionText = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($LASTEXITCODE -ne 0 -or [version]$versionText -lt [version]"3.12") {
    throw "Python 3.12+ is required for SmartTherm ML inference."
}

& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Could not install SmartTherm ML dependencies."
}

$env:SMARTTHERM_MODEL_DIR = Join-Path $mlRoot "models"
Push-Location $mlRoot
try {
    & $python -m uvicorn smarttherm_ml.service:app --app-dir $mlRoot --host 127.0.0.1 --port 8090
    if ($LASTEXITCODE -ne 0) {
        throw "SmartTherm Python inference service exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
