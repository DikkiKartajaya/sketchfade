@echo off
REM ShadowDraw demo launcher for Windows. Double-click to start the demo.
cd /d "%~dp0"

REM pick a Python 3 launcher
set PY=
where py >nul 2>&1 && set PY=py -3
if "%PY%"=="" ( where python >nul 2>&1 && set PY=python )
if "%PY%"=="" (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/ and try again.
  pause
  exit /b 1
)

REM install dependencies only if missing
%PY% -c "import numpy, PIL" >nul 2>&1
if errorlevel 1 (
  echo Installing dependencies ^(numpy, Pillow^) ...
  %PY% -m pip install --quiet numpy Pillow
)

set PORT=9000
REM open the browser a moment after the server comes up
start "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:%PORT%/"

echo Starting ShadowDraw demo on http://127.0.0.1:%PORT%/
echo Close this window (or press Ctrl+C) to stop.
%PY% server.py %PORT%
