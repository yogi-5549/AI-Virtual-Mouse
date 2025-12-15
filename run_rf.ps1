# run_rf.ps1 - Runs Random Forest version
Write-Host "Activating Random Forest environment..."
& ".\.venv_rf\Scripts\Activate.ps1"

Write-Host "Running Random Forest gesture control..."
python step6b_realtime_test.py

deactivate
