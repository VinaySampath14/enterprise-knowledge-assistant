@echo off
setlocal

if "%~1"=="" (
  echo Usage: scripts\run_bundle.bat ^<ablation_version^>
  echo Example: scripts\run_bundle.bat v4b_bm25_alpha06
  exit /b 1
)

set VERSION=%~1

python scripts/experiments/run_bundle.py --ablation-version %VERSION%
if errorlevel 1 exit /b %errorlevel%

echo.
echo [OK] Bundle completed for version %VERSION%.
exit /b 0
