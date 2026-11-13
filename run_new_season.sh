#!/usr/bin/env bash
# ============================================================
#  NextPitchAI — test the shipped model on a new season
#
#      ./run_new_season.sh            (tests 2026)
#      ./run_new_season.sh 2027
#
#  No retraining. Four steps, and it STOPS at the first one that
#  fails: running them by hand let three later steps run after the
#  scrape had failed, and the last screen of output was about
#  something else entirely.
#
#    1. Scrape    fetch only the weeks the file does not have
#    2. Extend    build data_ext/ with nothing refitted; refuses unless
#                 the seasons the model knew come out identical
#    3. Gate      lock temperature, threshold and bar on the validation
#                 season, score the new one once
#    4. Bundle    site/serving/, from the dataset the gate measured
#
#  Every step can be re-run. The scrape keeps what it has downloaded,
#  and the gate keeps the seasons it has already scored.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

SEASON="${1:-2026}"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "No $PY here. Run ./run_pipeline.sh --setup-only first."; exit 1; }
[ -f data_v5/best_model_v5.keras ] || { echo "No trained model in data_v5/. This script tests an existing model."; exit 1; }

step() { echo; echo "============================================"; echo " $1"; echo "============================================"; }
stop() {
    echo
    echo "STOPPED at step $1. Nothing after it was run."
    echo "Send the output from the '$1' banner down to here."
    exit 1
}

step "1/4  Scrape"
"$PY" 01_scrape_statcast.py || stop "1/4  Scrape"

step "2/4  Extend the dataset (nothing refitted)"
"$PY" 02_preprocess.py --freeze-from data_v5 --out-dir data_ext || stop "2/4  Extend the dataset"

step "3/4  Gate: score $SEASON once"
"$PY" analyze_actionability.py --data-dir data_ext --test-season "$SEASON" || stop "3/4  Gate"

step "4/4  Serving bundle"
"$PY" site/build_serving_artifacts.py --data-dir data_ext || stop "4/4  Serving bundle"

step "DONE"
echo " The report:  data_ext/actionability_report_v5.txt"
echo " Send sections A, 6 and 7 and the two verdicts."
echo
echo " To put the result on the site:"
echo "     git add site/serving"
echo "     git commit -m \"Rebuild serving bundle with $SEASON\""
echo "     git push"
