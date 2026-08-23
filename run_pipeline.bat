@echo off
REM ============================================================
REM  NextPitchAI v5 — one-click pipeline runner for Windows
REM
REM  Usage (from a Command Prompt in the repo folder, or just
REM  double-click this file in Explorer):
REM
REM      run_pipeline.bat            Run everything (skips steps
REM                                  whose output already exists)
REM      run_pipeline.bat --fresh    Force re-run of every step
REM
REM  Steps:
REM    1. Create/activate a virtual environment (.venv)
REM    2. Install dependencies
REM    3. 01_scrape_statcast.py  (~30-90 min, needs internet)
REM    4. 02_preprocess.py       (~5-15 min)
REM    5. 03_train.py            (minutes-hours depending on CPU/GPU)
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set FRESH=0
if /i "%~1"=="--fresh" set FRESH=1

echo.
echo ============================================
echo  NextPitchAI v5 pipeline
echo ============================================

REM ---- 1. Find Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH" during install.
    goto :fail
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or newer is required.
    python --version
    goto :fail
)

REM ---- 2. Virtual environment ----
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create virtual environment.
        goto :fail
    )
)
set PY=.venv\Scripts\python.exe

REM ---- 3. Dependencies ----
echo Installing/updating dependencies ...
"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -m pip install --quiet pybaseball pandas numpy pyarrow scikit-learn joblib imbalanced-learn matplotlib tensorflow
if errorlevel 1 (
    echo ERROR: dependency install failed. Check your internet connection.
    goto :fail
)
echo Dependencies OK.

REM ---- 4. Step 1: scrape ----
if exist "statcast_raw_v5.parquet" if %FRESH%==0 (
    echo.
    echo [1/3] SKIP scrape — statcast_raw_v5.parquet already exists.
    echo       ^(use "run_pipeline.bat --fresh" to re-scrape^)
    goto :preprocess
)
echo.
echo [1/3] Scraping Statcast data — this takes 30-90 minutes ...
"%PY%" 01_scrape_statcast.py
if errorlevel 1 (
    echo ERROR: scrape failed. Check your internet connection and re-run;
    echo Baseball Savant can be flaky — a re-run resumes from scratch.
    goto :fail
)
if not exist "statcast_raw_v5.parquet" (
    echo ERROR: scrape finished but statcast_raw_v5.parquet was not created.
    goto :fail
)

:preprocess
REM ---- 5. Step 2: preprocess ----
if exist "data_v5\meta_v5.json" if %FRESH%==0 (
    echo.
    echo [2/3] SKIP preprocess — data_v5\ already exists.
    echo       ^(use "run_pipeline.bat --fresh" to rebuild^)
    goto :train
)
echo.
echo [2/3] Preprocessing into training arrays ...
"%PY%" 02_preprocess.py
if errorlevel 1 (
    echo ERROR: preprocessing failed. Scroll up for the Python traceback.
    goto :fail
)

:train
REM ---- 6. Step 3: train ----
echo.
echo [3/3] Training model — grab a coffee ...
"%PY%" 03_train.py
if errorlevel 1 (
    echo ERROR: training failed. Scroll up for the Python traceback.
    goto :fail
)

echo.
echo ============================================
echo  DONE!
echo  Model:            data_v5\best_model_v5.keras
echo  Training curves:  data_v5\training_curves_v5.png
echo  Copy the classification report printed above
echo  back into Claude Code to tune hyperparameters.
echo ============================================
pause
exit /b 0

:fail
echo.
echo Pipeline stopped due to an error (see message above).
pause
exit /b 1
