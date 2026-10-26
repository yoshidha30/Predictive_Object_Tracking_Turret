@echo off
cd /d "%~dp0"
echo Launching Predictive Turret Simulation Lab...
".venv\Scripts\python.exe" main.py %*
pause
