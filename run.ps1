param(
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$VenvPy = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $VenvPy)) { $VenvPy = Join-Path $PSScriptRoot '.venv/bin/python' }
if (-not (Test-Path $VenvPy)) {
  $py = Get-Command python -ErrorAction SilentlyContinue
  if ($py) { $VenvPy = $py.Path }
}

if (-not (Test-Path $VenvPy)) {
  Write-Host 'ERROR: Python not found and .venv is missing.' -ForegroundColor Red
  Write-Host '       Run install.ps1 or install.bat first to set up the environment.'
  exit 1
}

if (-not (Test-Path (Join-Path $PSScriptRoot 'backend/app.py'))) {
  Write-Host 'ERROR: backend/app.py not found. Run from project root.' -ForegroundColor Red
  exit 1
}

Write-Host 'Starting backend on http://localhost:8000 ...'
Start-Process -FilePath $VenvPy -ArgumentList @('-m','uvicorn','backend.app:app','--port','8000','--reload') -WorkingDirectory $PSScriptRoot -WindowStyle Normal

Start-Sleep -Seconds 2
if (-not $NoBrowser) { Start-Process 'http://localhost:8000' }

if (-not (Test-Path (Join-Path $PSScriptRoot 'web/dist'))) {
  Write-Host 'Note: UI build not found at web/dist. Backend will still run APIs.' -ForegroundColor Yellow
  Write-Host '      Build UI with:  cd web; npm install; npm run build'
}

Write-Host 'Done.'
exit 0
