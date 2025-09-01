@echo off
setlocal EnableExtensions
title Start Backend and Open App

rem Change to repo root (this script's directory)
pushd "%~dp0" >nul 2>&1

set "VENV_PY=.venv\Scripts\python.exe"
set "PY_CMD="

if exist "%VENV_PY%" (
  set "PY_CMD=%VENV_PY%"
) else (
  where python >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Python not found and .venv is missing.
    echo         Run install.bat first to set up the environment.
    goto :end
  ) else (
    for /f "delims=" %%P in ('where python') do (
      set "PY_CMD=%%P"
      goto :gotpy
    )
  )
)

:gotpy
if not exist backend\app.py (
  echo [ERROR] backend\app.py not found. Run this from the project root.
  goto :end
)

echo [1/2] Starting backend on http://localhost:8000 ...
if exist "%PY_CMD%" (
  start "Backend - Uvicorn" "%PY_CMD%" -m uvicorn backend.app:app --port 8000 --reload
) else (
  start "Backend - Uvicorn" %PY_CMD% -m uvicorn backend.app:app --port 8000 --reload
)

rem Brief wait before opening the browser
ping -n 3 127.0.0.1 >nul 2>&1

echo [2/2] Opening app in your browser ...
start "" "http://localhost:8000"

if not exist "web\dist" (
  echo [Note] UI build not found at web\dist. Backend will still run APIs.
  echo        Build UI with:  cd web ^&^& npm install ^&^& npm run build
)

:end
popd >nul 2>&1
endlocal
exit /b 0

