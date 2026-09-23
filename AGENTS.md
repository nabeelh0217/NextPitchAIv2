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

## The goal is a hitter's edge, NOT top-1 accuracy

Read this before proposing any modeling change. The project's purpose is
to give a batter a real advantage in the box. That makes most of the
obvious metrics misleading:

- A hitter has ~400ms and decides **gear up vs stay back**, and **where**.
  He cannot act on a 10-way distribution. The binary fastball-family call
  is the decision; the 10-class argmax is not.
- **Flat average accuracy is the wrong statistic.** What matters is: in
  what fraction of pitches is the read strong enough to COMMIT, and how
  right are we there? 47% everywhere is useless; 47% overall but 80%
  inside a confident 15% is a product.
- **Accuracy without edge is worthless.** The bar is not "is the model
  right" but "does it beat the scouting report the hitter already has" —
  i.e. that pitcher's own base rate for the situation. A model at 77%
  where the base rate is 78% has negative value. `analyze_actionability.py`
  exists because this trap is easy to walk into.

`analyze_actionability.py` is the gate: it requires >=10% of pitches
advised, >=70% accuracy there, AND >=+2 points over the pitcher's base
rate. Run it before any further modeling work.

## File map

| Path | Role |
|---|---|
| `01_scrape_statcast.py` | Pulls 2022–2024 Statcast pitches via pybaseball → `statcast_raw_v5.parquet` (atomic write) |
| `02_preprocess.py` | Leakage-free feature engineering → `data_v5/` arrays, scalers, ID maps, website artifacts |
| `03_train.py` | BiLSTM + embeddings + arsenal-masked softmax, class-weighted focal loss → `data_v5/best_model_v5.keras` + `eval_report_v5.txt` |
| `evaluate_model.py` | Regenerates the evaluation report from the saved model (same split, no retraining). `03_train.py` imports its `build_report` so the two can't drift |
| `analyze_actionability.py` | **The product gate.** Confidence stratification, calibration, and the commit-slice edge over the pitcher's base rate. No retraining. Run this before any modeling change |
| `diagnose_arsenal.py` | Measures how much the arsenal mask actually constrains, and whether the baseline comparison is fair |
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

## Current status (2026-09-23)

- **Alpha search is CLOSED.** Runs 5/6/7 (ALPHA_POWER 0.5 / 0.0 / 0.25) moved
  log-loss only 0.033 nats while moving top-1 4.9 points — a decision-threshold
  frontier, not an information gain. Run 6 (top-1 48.9%, log-loss 1.1531) is
  best on calibration; run 7 (47.4%, 1.1650, macro-F1 0.457) is best balanced.
  Do not reopen it.
- **A 5-dimension audit refuted the "model guesses from all pitch types"
  hypothesis** — the mask is applied at evaluation, and the prior baseline is
  itself arsenal-constrained, so the reported lift was always within-arsenal.
  See the post-run-7 section of `docs/EXPERIMENTS.md` for the full findings.
- **Open defects, in priority order** (also in EXPERIMENTS.md):
  1. CRITICAL — the random per-pitch split leaves validation rows dependent on
     training rows (same at-bat, same outing). Est. 1-4 points of top-1
     inflation. Fixing means a grouped/temporal split and a retrain, and the
     honest number will read LOWER.
  2. HIGH — the arsenal mask is built over train+val, so the true validation
     label can never be zeroed; contaminates log-loss.
  3. MEDIUM — mask threshold is a raw count, not a usage share, so starters'
     masks approach all-ones.
- **Reality check**: realistic ceiling is 52-56% top-1; top-3 at ~92% is within
  2-3 points of any achievable ceiling. Lead with top-3 and the fastball-family
  binary number, not 10-class top-1.
- `diagnose_arsenal.py` measures items 2-3 empirically from `data_v5/` in
  seconds. Run it before acting on them.

## Conventions

- Scripts are flat, numbered, config-at-top; keep that style.
- Comments explain constraints the code can't show (leakage reasoning,
  why a value was chosen), not what the next line does.
- Never commit `.venv/`, `data_v5/`, `*.parquet`, `*.keras`, `*.npy`,
  `*.pkl`.
