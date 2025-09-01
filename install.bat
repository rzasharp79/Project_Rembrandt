@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Stunning Local Image Generator - Installer

rem Change to repo root (this script's directory)
pushd "%~dp0" >nul 2>&1

set "LOGDIR=logs"
set "LOG=%LOGDIR%\install.log"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1

echo =============================================================
echo  Stunning Local Image Generator - One-click Installer
echo =============================================================
echo   This will:
echo     - Create a Python virtual environment (.venv)
echo     - Install backend dependencies (minimal by default)
echo     - Install and build the React web UI
echo     - Prepare models/ and outputs/ folders
echo.
echo   A log will be saved to %LOG%
echo.

rem Parse flags (optional): --full installs PyTorch/Diffusers
set "FULL=0"
:args
if "%~1"=="" goto after_args
if /I "%~1"=="--full" set "FULL=1"
if /I "%~1"=="/full" set "FULL=1"
shift
goto args
:after_args

call :log [1/8] Checking Python...

rem Try to locate Python launcher first, then python
where py >nul 2>&1
if not errorlevel 1 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>&1
  if not errorlevel 1 (
    set "PY_CMD=python"
  ) else (
    echo [ERROR] Python 3.10+ not found. ^(Install from https://www.python.org/downloads/^ and add to PATH^)>>"%LOG%"
    echo [ERROR] Python 3.10+ not found on PATH.
    echo         Please install Python 3.10-3.12 and re-run install.bat
    goto :fail
  )
)

rem Validate version is between 3.10 and 3.12
%PY_CMD% -c "import sys;exit(0 if (3,10)<=sys.version_info[:2]<(3,13) else 1)" >>"%LOG%" 2>&1
if errorlevel 1 (
  for /f "usebackq delims=" %%v in (`%PY_CMD% -c "import sys;print(sys.version)" 2^>nul`) do set "PYVER=%%v"
  echo [ERROR] Python version detected: !PYVER! >>"%LOG%"
  echo [ERROR] Requires Python 3.10-3.12. Detected: !PYVER!
  goto :fail
)

call :log [2/8] Creating virtual environment (.venv)...
if not exist .venv (
  %PY_CMD% -m venv .venv >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment. See %LOG%
    goto :fail
  )
) else (
  echo   Using existing .venv
)

call :log [3/8] Activating virtual environment and upgrading pip...
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo [ERROR] Failed to activate .venv. See %LOG%
  goto :fail
)
python -m pip install --upgrade pip >>"%LOG%" 2>&1
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip. See %LOG%
  goto :fail
)

if "%FULL%"=="1" (
  call :log [4/8] Installing FULL backend requirements ^(PyTorch/Diffusers^)... ^(this can take a while^)
  python -m pip install -r backend\requirements.txt >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Backend dependency install failed. See %LOG%
    echo         Tip: If PyTorch fails, try FULL install later using --full and follow PyTorch site instructions.
    goto :fail
  )
) else (
  call :log [4/8] Installing MINIMAL backend requirements...
  if exist backend\requirements-min.txt del /q backend\requirements-min.txt >nul 2>&1
  rem Build a minimal requirements file on the fly (exclude heavy optional libs)
  cmd /c type backend\requirements.txt ^| findstr /V /I "torch diffusers transformers safetensors accelerate numpy" > backend\requirements-min.txt
  python -m pip install -r backend\requirements-min.txt >>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Minimal backend dependency install failed. See %LOG%
    goto :fail
  )
)

call :log [5/8] Checking Node.js and npm...
where node >nul 2>&1
if errorlevel 1 (
  echo [WARN] Node.js not found. Skipping web UI install/build. >>"%LOG%"
  echo   Node.js not found; skipping UI build. Install Node 18+ from https://nodejs.org/ and re-run to build UI.
  set "SKIP_WEB=1"
) else (
  where npm >nul 2>&1
  if errorlevel 1 (
    echo [WARN] npm not found though node exists; skipping UI. >>"%LOG%"
    echo   npm not found; skipping UI build. Ensure Node/npm are on PATH.
    set "SKIP_WEB=1"
  ) else (
    for /f "tokens=*" %%v in ('node -v') do set "NODEVER=%%v"
    echo   Node version: !NODEVER!
    set "SKIP_WEB=0"
  )
)

if not "%SKIP_WEB%"=="1" (
  call :log [6/8] Installing web dependencies ^(npm install^)...
  pushd web >nul 2>&1
  call npm install >>"%LOG%" 2>&1
  if errorlevel 1 (
    popd >nul 2>&1
    echo [ERROR] npm install failed. See %LOG%
    goto :fail
  )
  call :log [7/8] Building web UI ^(npm run build^)...
  call npm run build >>"%LOG%" 2>&1
  if errorlevel 1 (
    popd >nul 2>&1
    echo [ERROR] UI build failed. See %LOG%
    goto :fail
  )
  popd >nul 2>&1
) else (
  echo [6/8] Skipping web install/build.
)

call :log [8/8] Ensuring folder layout...
if not exist outputs mkdir outputs >nul 2>&1
if not exist models mkdir models >nul 2>&1
if not exist models\checkpoints mkdir models\checkpoints >nul 2>&1
if not exist models\loras mkdir models\loras >nul 2>&1
if not exist models\embeddings mkdir models\embeddings >nul 2>&1

echo.
echo Install complete.
echo.
echo Next steps:
echo   1) Start backend:  .\.venv\Scripts\activate ^&^& python -m uvicorn backend.app:app --reload --port 8000
if "%SKIP_WEB%"=="1" (
  echo   2^) ^(Optional^) Build UI later:  cd web ^&^& npm install ^&^& npm run build
  echo      Then open: http://localhost:8000
) else (
  echo   2^) Open the app at:  http://localhost:8000
)
echo.
echo Tips:
echo   - To install full GPU dependencies later, run: install.bat --full
echo   - Configure paths in-app or via /settings; models go in models\checkpoints
echo.
popd >nul 2>&1
goto :eof

:fail
echo.
echo Install failed. Please review: %LOG%
echo If you need help, share the log content when asking for support.
popd >nul 2>&1
exit /b 1

rem Simple logger: prints to console and appends to log file
:log
set _msg=%*
echo %_msg%
echo %_msg%>>"%LOG%"
exit /b 0
