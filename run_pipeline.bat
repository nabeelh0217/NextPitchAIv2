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

REM ---- 1. Find a TensorFlow-compatible Python (3.10 - 3.13) ----
REM TensorFlow does not publish wheels for every Python release — if your
REM default "python" is newer than TensorFlow supports (e.g. 3.14+), pip
REM install fails with "No matching distribution found for tensorflow".
REM We use the Windows "py" launcher (installed by default with python.org
REM installers) to find an already-installed compatible version, so you
REM don't need to change your default python/PATH.
set PYEXE=

where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.13 3.12 3.11 3.10) do (
        if not defined PYEXE (
            py -%%V -c "exit()" >nul 2>nul
            if not errorlevel 1 set PYEXE=py -%%V
        )
    )
)

if not defined PYEXE (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12),(3,13)] else 1)" >nul 2>nul
        if not errorlevel 1 set PYEXE=python
    )
)

if not defined PYEXE (
    echo ERROR: No TensorFlow-compatible Python found ^(need 3.10, 3.11, 3.12, or 3.13^).
    echo.
    where python >nul 2>nul
    if not errorlevel 1 (
        echo Your default "python" is:
        python --version
        echo.
    ) else (
        echo No "python" was found on PATH either.
        echo.
    )
    echo TensorFlow does not support newer Python versions yet on Windows.
    echo Install Python 3.12 from:
    echo   https://www.python.org/downloads/release/python-3120/
    echo During install, keep "Install launcher for all users" / "py launcher"
    echo checked ^(it's on by default^) — you do NOT need to change your
    echo default python or PATH. Re-run this script afterward and it will
    echo find and use 3.12 automatically via the py launcher.
    goto :fail
)

echo Using Python:
%PYEXE% --version

REM ---- 2. Virtual environment ----
REM A .venv left over from an earlier failed run (created before this
REM script picked a compatible Python) would otherwise be silently reused.
REM Validate it and rebuild if it's missing or incompatible.
set REBUILD_VENV=0
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12),(3,13)] else 1)" >nul 2>nul
    if errorlevel 1 set REBUILD_VENV=1
) else (
    set REBUILD_VENV=1
)

if %REBUILD_VENV%==1 (
    if exist ".venv" (
        echo Existing .venv is missing or incompatible — rebuilding...
        rmdir /s /q .venv
    )
    echo Creating virtual environment .venv ...
    %PYEXE% -m venv .venv
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
REM A real multi-season scrape is hundreds of MB+. Treat anything smaller
REM as corrupt/incomplete (e.g. left over from an interrupted run) rather
REM than trusting mere file existence.
set MIN_PARQUET_BYTES=1000000
set SCRAPE_VALID=0
if exist "statcast_raw_v5.parquet" (
    for %%A in ("statcast_raw_v5.parquet") do set SCRAPE_SIZE=%%~zA
    if !SCRAPE_SIZE! GEQ %MIN_PARQUET_BYTES% set SCRAPE_VALID=1
)

if %SCRAPE_VALID%==1 if %FRESH%==0 (
    echo.
    echo [1/3] SKIP scrape — statcast_raw_v5.parquet already exists ^(!SCRAPE_SIZE! bytes^).
    echo       ^(use "run_pipeline.bat --fresh" to re-scrape^)
    goto :preprocess
)
if exist "statcast_raw_v5.parquet" if %SCRAPE_VALID%==0 (
    echo statcast_raw_v5.parquet exists but looks incomplete/corrupt ^(!SCRAPE_SIZE! bytes^) — deleting and re-scraping.
    del /f /q "statcast_raw_v5.parquet"
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
for %%A in ("statcast_raw_v5.parquet") do set SCRAPE_SIZE=%%~zA
if !SCRAPE_SIZE! LSS %MIN_PARQUET_BYTES% (
    echo ERROR: scrape finished but statcast_raw_v5.parquet is only !SCRAPE_SIZE! bytes — something went wrong.
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
