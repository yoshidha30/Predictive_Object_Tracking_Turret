@echo off
cd /d "%~dp0"
echo Running Predictive Turret Unit Tests...
".venv\Scripts\python.exe" -m pytest %*
pause
