#!/usr/bin/env bash
# ============================================================
#  NextPitchAI — one-command pipeline runner for macOS / Linux
#  (Windows: use run_pipeline.bat)
#
#  Usage (from the repo folder):
#      ./run_pipeline.sh                     Run everything; steps whose
#                                            output already exists are skipped
#      ./run_pipeline.sh --fresh-preprocess  Keep the scraped parquet, rebuild
#                                            data_v5/ and retrain
#      ./run_pipeline.sh --fresh             Redo every step, INCLUDING the
#                                            30-90 min scrape
#      ./run_pipeline.sh --setup-only        Just create .venv + install deps
#
#  Steps:
#    1. Find a TensorFlow-compatible Python (3.10 - 3.13)
#    2. Create/validate a virtual environment (.venv)
#    3. Install dependencies
#    4. 01_scrape_statcast.py  (~30-90 min, needs internet)
#    5. 02_preprocess.py       (~5-15 min)
#    6. 03_train.py            (30-90+ min on CPU)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

usage() { sed -n '2,21p' "$0" | sed 's/^#//'; }

FRESH=0
FRESH_PREPROCESS=0
SETUP_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --fresh)            FRESH=1 ;;
        --fresh-preprocess) FRESH_PREPROCESS=1 ;;
        --setup-only)       SETUP_ONLY=1 ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

PARQUET="statcast_raw_v5.parquet"
MIN_PARQUET_BYTES=1000000
SUPPORTED_PY='import sys; sys.exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12),(3,13)] else 1)'

fail()   { echo; echo "ERROR: $*"; echo; echo "Pipeline stopped."; exit 1; }
banner() { echo; echo "============================================"; echo " $*"; echo "============================================"; }
# wc -c is portable; `stat` flags differ between macOS and Linux.
file_size() { wc -c < "$1" | tr -d ' '; }

banner "NextPitchAI v6 pipeline"

# ---- 1. Find a TensorFlow-compatible Python (3.10 - 3.13) ----
# TensorFlow publishes wheels only for these versions; a newer system
# python (e.g. 3.14) fails with "No matching distribution found for
# tensorflow". Probe versioned binaries first — Homebrew, python.org and
# pyenv all put these on PATH — newest supported first, then fall back to
# plain python3 only if its own version is in range.
PYEXE=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "$SUPPORTED_PY" >/dev/null 2>&1; then
        PYEXE="$cand"
        break
    fi
done

if [ -z "$PYEXE" ]; then
    echo "ERROR: No TensorFlow-compatible Python found (need 3.10, 3.11, 3.12, or 3.13)."
    echo
    if command -v python3 >/dev/null 2>&1; then
        echo "Your default python3 is: $(python3 --version 2>&1)"
        echo "TensorFlow does not support that version yet."
    else
        echo "No python3 was found on PATH."
    fi
    echo
    echo "Install Python 3.12 with Homebrew:"
    echo "    brew install python@3.12"
    echo "(or from https://www.python.org/downloads/macos/)"
    echo "Then re-run this script — it finds python3.12 automatically."
    exit 1
fi

echo "Using Python: $("$PYEXE" --version 2>&1)  [$(command -v "$PYEXE")]"

# ---- 2. Virtual environment ----
# A .venv left over from an earlier run with an incompatible interpreter
# would otherwise be silently reused. Validate it; rebuild if needed.
REBUILD_VENV=0
if [ -x ".venv/bin/python" ]; then
    .venv/bin/python -c "$SUPPORTED_PY" >/dev/null 2>&1 || REBUILD_VENV=1
else
    REBUILD_VENV=1
fi

if [ "$REBUILD_VENV" = 1 ]; then
    if [ -d ".venv" ]; then
        echo "Existing .venv is missing or incompatible — rebuilding..."
        rm -rf .venv
    fi
    echo "Creating virtual environment .venv ..."
    "$PYEXE" -m venv .venv || fail "could not create virtual environment."
fi
PY=".venv/bin/python"

# ---- 3. Dependencies ----
# imbalanced-learn is intentionally absent: v6 replaced oversampling with
# class-weighted focal loss. tensorflow-metal (Apple GPU plugin) is also
# deliberately not installed — it has a history of LSTM/RNN bugs, and a
# correct model matters more than a faster one. Add it yourself if you
# want to experiment: .venv/bin/pip install tensorflow-metal
echo "Installing/updating dependencies ..."
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install --quiet pybaseball pandas numpy pyarrow scikit-learn joblib matplotlib tensorflow \
    || fail "dependency install failed. Check your internet connection."
echo "Dependencies OK."

if [ "$SETUP_ONLY" = 1 ]; then
    banner "Environment ready"
    echo " Interpreter: $(pwd)/.venv/bin/python"
    echo " Run ./run_pipeline.sh (no flags) to scrape, preprocess and train."
    exit 0
fi

# ---- 4. Step 1: scrape ----
# A real multi-season scrape is hundreds of MB+. Treat anything smaller
# as corrupt/incomplete (e.g. left over from an interrupted run) rather
# than trusting mere file existence.
SIZE=0
SCRAPE_VALID=0
if [ -f "$PARQUET" ]; then
    SIZE=$(file_size "$PARQUET")
    [ "$SIZE" -ge "$MIN_PARQUET_BYTES" ] && SCRAPE_VALID=1
fi

if [ "$SCRAPE_VALID" = 1 ] && [ "$FRESH" = 0 ]; then
    echo
    echo "[1/3] SKIP scrape — $PARQUET already exists ($SIZE bytes)."
    echo "      (use ./run_pipeline.sh --fresh to re-scrape)"
else
    if [ -f "$PARQUET" ] && [ "$SCRAPE_VALID" = 0 ]; then
        echo "$PARQUET exists but looks incomplete/corrupt ($SIZE bytes) — deleting and re-scraping."
        rm -f "$PARQUET"
    fi
    echo
    echo "[1/3] Scraping Statcast data — this takes 30-90 minutes ..."
    echo "      Keep the laptop plugged in and awake (System Settings > Battery)."
    "$PY" 01_scrape_statcast.py \
        || fail "scrape failed. Check your internet connection and re-run; Baseball Savant can be flaky."
    [ -f "$PARQUET" ] || fail "scrape finished but $PARQUET was not created."
    SIZE=$(file_size "$PARQUET")
    [ "$SIZE" -ge "$MIN_PARQUET_BYTES" ] \
        || fail "scrape finished but $PARQUET is only $SIZE bytes — something went wrong."
fi

# ---- 5. Step 2: preprocess ----
if [ -f "data_v5/meta_v5.json" ] && [ "$FRESH" = 0 ] && [ "$FRESH_PREPROCESS" = 0 ]; then
    echo
    echo "[2/3] SKIP preprocess — data_v5/ already exists."
    echo "      (use ./run_pipeline.sh --fresh-preprocess to rebuild it)"
else
    if [ -d "data_v5" ] && { [ "$FRESH" = 1 ] || [ "$FRESH_PREPROCESS" = 1 ]; }; then
        echo "Removing existing data_v5/ (arrays + previous model) ..."
        rm -rf data_v5
    fi
    echo
    echo "[2/3] Preprocessing into training arrays ..."
    "$PY" 02_preprocess.py || fail "preprocessing failed. Scroll up for the Python traceback."
fi

# ---- 6. Step 3: train ----
echo
echo "[3/3] Training model — grab a coffee ..."
"$PY" 03_train.py || fail "training failed. Scroll up for the Python traceback."

banner "DONE"
echo " Model:            data_v5/best_model_v5.keras"
echo " Eval report:      data_v5/eval_report_v5.txt   (safe to close the terminal)"
echo " Training curves:  data_v5/training_curves_v5.png"
echo " Paste the report back into Claude Code / Cursor and log the run"
echo " in docs/EXPERIMENTS.md. Regenerate it any time with:"
echo "     .venv/bin/python evaluate_model.py"
echo
