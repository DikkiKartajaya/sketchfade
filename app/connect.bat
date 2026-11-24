@echo off
setlocal EnableDelayedExpansion
REM SketchFade on the lab GPU server, used from this laptop. Double-click to start.
REM   1. connects to the server over SSH and lists its GPUs; you pick one
REM   2. runs the app on that GPU (serve_gpu.sh; it listens only on 127.0.0.1
REM      of the server, nothing is opened to the network)
REM   3. tunnels it to this laptop and opens http://127.0.0.1:PORT/ when ready
REM Ctrl+C or closing this window stops the app and releases the GPU. If the
REM connection drops, the app stops by itself after 2 minutes without a ping.
REM The first run asks for the server and saves it in connect.cfg (not in git).
cd /d "%~dp0"

where ssh >nul 2>&1
if errorlevel 1 (
  echo The OpenSSH client was not found. Windows: Settings ^> Apps ^> Optional features ^> OpenSSH Client.
  pause
  exit /b 1
)

set CFG=connect.cfg
if not exist "%CFG%" (
  echo First run: where does SketchFade live on the lab server?
  set /p SERVER=SSH login, e.g. name@gpu-server:
  set RDIR=
  set /p RDIR=Folder of the app on the server [~/sketchfade/app]:
  if "!RDIR!"=="" set RDIR=~/sketchfade/app
  set PORT=
  set /p PORT=Port [9100]:
  if "!PORT!"=="" set PORT=9100
  > "%CFG%" echo SERVER=!SERVER!
  >> "%CFG%" echo RDIR=!RDIR!
  >> "%CFG%" echo PORT=!PORT!
  echo Saved in %CFG%. Edit or delete it to change these.
  echo.
)
for /f "usebackq tokens=1,* delims==" %%a in ("%CFG%") do set "%%a=%%b"

echo SketchFade on %SERVER% (port %PORT%)
echo The browser opens by itself once the app is ready.
echo.

REM Keep-alive in the background: waits until the app answers, opens the
REM browser, then pings every 20 s so the server knows the laptop is still
REM there; it ends after 3 failed pings (app stopped).
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$u='http://127.0.0.1:%PORT%'; $ok=$false;" ^
  "for ($i=0; $i -lt 900 -and -not $ok; $i++) { try { Invoke-WebRequest -UseBasicParsing -Method Post -Body '{}' -TimeoutSec 3 ($u+'/api/ping') | Out-Null; $ok=$true } catch { Start-Sleep 2 } };" ^
  "if (-not $ok) { exit };" ^
  "Start-Process ($u+'/');" ^
  "$fail=0; while ($fail -lt 3) { Start-Sleep 20; try { Invoke-WebRequest -UseBasicParsing -Method Post -Body '{}' -TimeoutSec 10 ($u+'/api/ping') | Out-Null; $fail=0 } catch { $fail++ } }"

REM -t: a terminal on the server, so closing this window hangs up the app
REM LogLevel=QUIET: hides "connect failed" noise while the app is still loading
ssh -t -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o LogLevel=QUIET ^
    -L %PORT%:127.0.0.1:%PORT% %SERVER% "bash %RDIR%/serve_gpu.sh %PORT%"
set CODE=%errorlevel%

echo.
if "%CODE%"=="255" (
  echo Could not connect, or port %PORT% is already in use on this laptop.
  echo To see the SSH error, run:  ssh %SERVER%
  echo To change the server or port, edit or delete %~dp0%CFG%
) else (
  echo SketchFade stopped; the GPU on the server is released.
)
pause
