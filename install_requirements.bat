@echo off
REM Installs every package this project needs, then runs the self-test.
cd /d "%~dp0"
set "PY=%LOCALAPPDATA%\Microsoft\WindowsApps\python3.13.exe"
if not exist "%PY%" set "PY=python"
echo Using Python: %PY%
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
echo.
echo ===== Running self-test =====
"%PY%" selftest.py
echo.
pause
