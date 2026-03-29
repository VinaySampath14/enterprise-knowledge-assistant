@echo off
setlocal

REM Tiny launcher for the standard Phase 0 experiment workflow.
set "ROOT=%~dp0..\.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python venv not found at "%PY%"
  echo Create it first: python -m venv .venv ^&^& .\.venv\Scripts\activate ^&^& pip install -r requirements.txt
  exit /b 1
)

set "SCRIPT=%ROOT%\scripts\experiments\run_phase_gate.py"
if not exist "%SCRIPT%" (
  echo [ERROR] Script not found: "%SCRIPT%"
  exit /b 1
)

echo [INFO] Running experiment phase gate...
"%PY%" "%SCRIPT%" --dataset eval_v2/synthetic_scaffold_dataset_refined.jsonl --category-field refined_category --expected-type-field expected_type_refined %*

endlocal
