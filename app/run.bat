@echo off
REM SketchFade launcher for Windows. Double-click to start.
cd /d "%~dp0"

set PY=
where py >nul 2>&1 && set PY=py -3
if "%PY%"=="" ( where python >nul 2>&1 && set PY=python )
if "%PY%"=="" (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)

%PY% -c "import numpy, PIL, diffusers, peft" >nul 2>&1
if errorlevel 1 %PY% -m pip install --quiet -r requirements.txt

set PORT=9100
start "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:%PORT%/"
%PY% server.py %PORT%
