param(
  [switch]$Full,
  [switch]$NoUI
)

$ErrorActionPreference = 'Stop'

# Accept alt flags (e.g., --full) if passed positionally
if (-not $Full) { if ($args -contains '--full' -or $args -contains '/full') { $Full = $true } }
if (-not $NoUI) { if ($args -contains '--no-ui' -or $args -contains '/no-ui') { $NoUI = $true } }

# Move to repo root (this script's directory)
Set-Location -LiteralPath $PSScriptRoot

$LogDir = Join-Path $PSScriptRoot 'logs'
$LogFile = Join-Path $LogDir 'install.log'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

function Write-Log {
  param([Parameter(Mandatory)][string]$Message,[switch]$NoConsole)
  $ts = (Get-Date).ToString('s')
  $line = "[$ts] $Message"
  if (-not $NoConsole) { Write-Host $Message }
  Add-Content -Path $LogFile -Value $line
}

function Exec {
  param(
    [Parameter(Mandatory)][string]$File,
    [string[]]$Args,
    [string]$Cwd
  )
  if ($Cwd) { Push-Location $Cwd }
  try {
    Write-Log "Running: $File $($Args -join ' ')" -NoConsole
    & $File @Args 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
      throw "Command failed with exit code ${LASTEXITCODE}: $File"
    }
  } finally {
    if ($Cwd) { Pop-Location }
  }
}

Write-Host '============================================================='
Write-Host ' Stunning Local Image Generator - One-click Installer'
Write-Host '============================================================='
Write-Host '  This will:'
Write-Host '    - Create a Python virtual environment (.venv)'
Write-Host '    - Install backend dependencies (minimal by default)'
Write-Host '    - Install and build the React web UI'
Write-Host '    - Prepare models/ and outputs/ folders'
Write-Host ''
Write-Host "  A log will be saved to $LogFile"
Write-Host ''

# 1) Python discovery
Write-Log '[1/8] Checking Python...'
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$pyExe = Get-Command python -ErrorAction SilentlyContinue
$pyCmd = $null
if ($pyLauncher) { $pyCmd = @('py','-3') }
elseif ($pyExe) { $pyCmd = @($pyExe.Path) }
else {
  Write-Log 'ERROR: Python 3.10+ not found on PATH.'
  Write-Host 'ERROR: Python 3.10+ not found on PATH.' -ForegroundColor Red
  Write-Host '       Install from https://www.python.org/downloads/ and re-run.'
  exit 1
}

# 1b) Version check 3.10–3.12
$verOut = & $pyCmd[0] $pyCmd[1..10] -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2>$null
try { $ver = [version]$verOut } catch { $ver = $null }
if (-not $ver -or $ver.Major -ne 3 -or $ver.Minor -lt 10 -or $ver.Minor -gt 12) {
  Write-Log "ERROR: Requires Python 3.10–3.12. Detected: $verOut"
  Write-Host "ERROR: Requires Python 3.10–3.12. Detected: $verOut" -ForegroundColor Red
  exit 1
}

# 2) Create venv if needed
Write-Log '[2/8] Creating virtual environment (.venv)...'
if (-not (Test-Path '.venv')) {
  Exec -File $pyCmd[0] -Args ($pyCmd[1..10] + @('-m','venv','.venv'))
} else {
  Write-Host '  Using existing .venv'
}

# Use venv python directly to avoid ExecutionPolicy issues
$VenvPy = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $VenvPy)) { $VenvPy = Join-Path $PSScriptRoot '.venv/bin/python' }
if (-not (Test-Path $VenvPy)) {
  Write-Log 'ERROR: venv python not found after creation.'
  Write-Host 'ERROR: venv python not found after creation.' -ForegroundColor Red
  exit 1
}

# 3) Upgrade pip
Write-Log '[3/8] Upgrading pip...'
Exec -File $VenvPy -Args @('-m','pip','install','--upgrade','pip')

# 4) Backend dependencies
if ($Full) {
  Write-Log '[4/8] Installing FULL backend requirements (PyTorch/Diffusers)...'
  Exec -File $VenvPy -Args @('-m','pip','install','-r','backend/requirements.txt')
} else {
  Write-Log '[4/8] Installing MINIMAL backend requirements...'
  $req = Get-Content -LiteralPath 'backend/requirements.txt'
  $exclude = 'torch','diffusers','transformers','safetensors','accelerate','numpy'
  $min = foreach ($line in $req) {
    if ($null -eq $line -or $line.Trim().StartsWith('#')) { $line }
    elseif ($exclude | Where-Object { $line -match [regex]::Escape($_) }) { continue }
    else { $line }
  }
  $tmpReq = [System.IO.Path]::GetTempFileName()
  Set-Content -LiteralPath $tmpReq -Value ($min -join [Environment]::NewLine)
  try {
    Exec -File $VenvPy -Args @('-m','pip','install','-r',$tmpReq)
  } finally {
    Remove-Item -LiteralPath $tmpReq -ErrorAction SilentlyContinue
  }
}

# 5) Web UI (optional)
Write-Log '[5/8] Checking Node.js and npm...'
$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm -ErrorAction SilentlyContinue
if ($NoUI) {
  Write-Host '  Skipping UI by request (--no-ui)'
} elseif (-not $node -or -not $npm) {
  Write-Log 'WARN: Node/npm not found. Skipping UI install/build.'
  Write-Host '  Node/npm not found; skipping UI build. Install Node 18+ and re-run to build UI.' -ForegroundColor Yellow
} else {
  Write-Log '[6/8] Installing web dependencies (npm install)...'
  Exec -File $npm.Path -Args @('install') -Cwd (Join-Path $PSScriptRoot 'web')
  Write-Log '[7/8] Building web UI (npm run build)...'
  Exec -File $npm.Path -Args @('run','build') -Cwd (Join-Path $PSScriptRoot 'web')
}

# 8) Folders
Write-Log '[8/8] Ensuring folder layout...'
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'outputs') | Out-Null
foreach ($d in 'models','models/checkpoints','models/loras','models/embeddings') {
  New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot $d) | Out-Null
}

Write-Host ''
Write-Host 'Install complete.' -ForegroundColor Green
Write-Host ''
Write-Host 'Next steps:'
Write-Host '  1) Start backend:'
Write-Host '     .\.venv\Scripts\python -m uvicorn backend.app:app --reload --port 8000'
if ($NoUI -or -not $node -or -not $npm) {
  Write-Host '  2) (Optional) Build UI later:'
  Write-Host '     cd web; npm install; npm run build'
  Write-Host '     Then open: http://localhost:8000'
} else {
  Write-Host '  2) Open the app at:'
  Write-Host '     http://localhost:8000'
}
Write-Host ''
Write-Host 'Tips:'
Write-Host '  - Full GPU deps later:  .\install.ps1 -Full'
Write-Host '  - Configure paths via /settings; models go in models\checkpoints'

exit 0

trap {
  Write-Host ''
  Write-Host "Install failed. See log: $LogFile" -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
}
