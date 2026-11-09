@echo off
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  python run_local.py
) else (
  py -3.11 run_local.py
)
pause
