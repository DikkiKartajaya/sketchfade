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

%PY% -c "import numpy, PIL, diffusers, peft, mediapipe" >nul 2>&1
if errorlevel 1 %PY% -m pip install --quiet -r requirements.txt

REM NVIDIA GPU but a CPU-only torch: switch to the CUDA build once (~2.5 GB).
REM server.py then uses the GPU when it has enough memory, else the CPU.
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  %PY% -c "import sys, torch; sys.exit(torch.version.cuda is None)" >nul 2>&1
  if errorlevel 1 (
    echo Installing PyTorch with CUDA for the NVIDIA GPU, this takes a while once ...
    %PY% -m pip uninstall --quiet -y torch
    %PY% -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu130
  )
)

set PORT=9100
start "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:%PORT%/"
%PY% server.py %PORT%
