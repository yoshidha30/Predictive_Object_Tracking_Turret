Set-Location -Path $PSScriptRoot
Write-Host "Launching Predictive Turret Simulation Lab..." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\main.py" $args
