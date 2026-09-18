# NextPitchAI v2 — agent context

Read this before touching anything. It is the memory of record for the
project: what it does, how it is wired, the rules that must not be broken,
and where we are. `docs/EXPERIMENTS.md` holds the run-by-run history.

## What this is

A model that predicts the **next pitch type** an MLB pitcher will throw
(10 canonical Statcast classes: FF, SI, FC, SL, ST, CU, KC, CH, FS, KN)
from game state, matchup history, and the pitcher's own recent sequence.
The softmax is **masked to the pitcher's real arsenal inside the model**,
so it can only ever predict pitches that pitcher actually throws. A Flask
website (`site/`, not yet built) will serve these predictions.

## File map

| Path | Role |
|---|---|
| `01_scrape_statcast.py` | Pulls 2022–2024 Statcast pitches via pybaseball → `statcast_raw_v5.parquet` (atomic write) |
| `02_preprocess.py` | Leakage-free feature engineering → `data_v5/` arrays, scalers, ID maps, website artifacts |
| `03_train.py` | BiLSTM + embeddings + arsenal-masked softmax, class-weighted focal loss → `data_v5/best_model_v5.keras` + `eval_report_v5.txt` |
| `evaluate_model.py` | Regenerates the evaluation report from the saved model (same split, no retraining). `03_train.py` imports its `build_report` so the two can't drift |
| `run_pipeline.sh` / `run_pipeline.bat` | One-command runners (macOS/Linux, Windows). Skip completed steps; validate artifacts |
| `docs/EXPERIMENTS.md` | Every training run, its numbers, and its verdict. **Append a row after every run.** |
| `.cursor/rules/` | Cursor-scoped rules (point back here) |

`data_v5/` and `*.parquet` are gitignored (GBs). The `_v5` file naming is
kept for runner continuity even though the schema is v6 — `meta_v5.json`
carries `"version": 6`.

## How to run

```bash
./run_pipeline.sh                    # macOS/Linux: everything, skipping done steps
./run_pipeline.sh --fresh-preprocess # rebuild data_v5/ + retrain, keep parquet
./run_pipeline.sh --fresh            # redo all, incl. the 30-90 min scrape
run_pipeline.bat                     # Windows equivalent (--fresh only)
```

Needs Python 3.10–3.13 (TensorFlow has no wheels for newer). The runners
find one automatically and manage `.venv`. The scrape hits Baseball
Savant directly — it cannot run from sandboxed/proxied environments.

## Hard invariants — do not break these

1. **No label leakage in priors.** Every historical feature (arsenal
   prior, matchup history, batter seen-profile, whiff rates) is an
   *expanding* statistic computed over PRIOR rows only, excluding the
   current pitch (`expanding_prior_dist` / `expanding_prior_rate` in
   `02_preprocess.py` subtract the current row's one-hot after the
   cumsum). Any new historical feature must follow the same pattern.
2. **Chronological sort first.** `02_preprocess.py` sorts by
   `game_date, game_pk, at_bat_number, pitch_number` before any prior or
   sequence is built. Priors and sequences are meaningless otherwise.
3. **Sequences are per (game_pk, pitcher).** The lookback window is the
   *same pitcher's* previous pitches. The v4 approach (previous N rows of
   the game) mostly captured the opposing pitcher — never regress to it.
4. **Arsenal mask semantics.** `X_arsenal_mask` is 1 for every pitch type
   the pitcher threw ≥1 time in the full dataset. It is a model INPUT;
   `03_train.py` adds `(1 - mask) * -1e9` to the logits before softmax.
   Every training label is unmasked by construction. Unknown pitchers at
   inference get an all-ones mask.
5. **Split before any resampling; evaluate on the natural distribution.**
   Never oversample/duplicate rows before `train_test_split`. (v6 has no
   resampling at all — class balance is per-class alpha in the focal
   loss.) Reported metrics are always on the untouched validation split.
6. **One change per training round, logged.** Runs take an hour+.
   Change one thing, retrain, append the result to
   `docs/EXPERIMENTS.md` with the verdict, then decide the next change.
7. **Atomic artifact writes.** `01_scrape_statcast.py` writes to a `.tmp`
   and `os.replace`s into place; the runners refuse a parquet < 1 MB. A
   corrupt/empty artifact silently reused cost a full re-scrape once.

## Evaluation vocabulary

`03_train.py` and `evaluate_model.py` print the model against the
**pitcher-prior baseline** — that pitcher's training pitch mix used as a
full probability distribution — on top-1, top-3 and log-loss. The
baseline is what you can predict knowing only who is on the mound, so
**log-loss improvement over it is the test of whether the game-context
features do anything at all**. 10-class top-1 reads lower than the old
4-bucket numbers; that is expected and not a regression.

## Current status (2026-09-18)

- **Run 5 (first real v6 training): top-1 44.0%, top-3 91.3%, baseline
  42.8%, lift +1.2%.** Training dynamics were healthy (28 epochs, best
  val_loss at 20, no overfitting). Rescored against the prior
  *distribution*: **log-loss 1.1864 vs 1.3221 (+0.136), top-3 +5.5** —
  the context features carry real signal. The flat top-1 was a bug:
  `ALPHA_POWER=0.5` stacked a 17:1 class-weight ratio on focal gamma=2
  and wrecked the majority class (FF recall 30% at precision 64%).
- **Run 6 is queued**: `ALPHA_POWER` set to 0.0 (uniform). Training-only
  change — `./run_pipeline.sh` suffices, no preprocessing rebuild.
  Expect the hidden signal to surface as top-1 lift.
- After run 6: if top-1 lift is now several points with log-loss holding
  or improving, lock the model and build the Flask site. The product is
  the calibrated distribution (top-3 already 91%), not the argmax.

## Conventions

- Scripts are flat, numbered, config-at-top; keep that style.
- Comments explain constraints the code can't show (leakage reasoning,
  why a value was chosen), not what the next line does.
- Never commit `.venv/`, `data_v5/`, `*.parquet`, `*.keras`, `*.npy`,
  `*.pkl`.
