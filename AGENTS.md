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

**The gate PASSED (2026-09-23).** At a 65% commit threshold the model
advises on 31.0% of pitches, is right 75.6% of the time there, and beats
the pitcher's base rate by +5.2 points. The model is good enough; the
remaining work is product, not modeling.

Two rules that follow, and must not be softened:
- **A card names the likeliest pitch WITHIN the family it tells the hitter
  to sit on.** "Sit soft, likeliest fastball" is incoherent advice.
- **The bar is the count-split scouting report**, not the pitcher's
  overall mix. Any advance scout already knows he throws fastballs 3-1;
  beating that is the only edge worth claiming.

## File map

| Path | Role |
|---|---|
| `01_scrape_statcast.py` | Pulls 2022–2024 Statcast pitches via pybaseball → `statcast_raw_v5.parquet` (atomic write) |
| `02_preprocess.py` | Leakage-free feature engineering → `data_v5/` arrays, scalers, ID maps, website artifacts |
| `03_train.py` | BiLSTM + embeddings + arsenal-masked softmax, class-weighted focal loss → `data_v5/best_model_v5.keras` + `eval_report_v5.txt` |
| `evaluate_model.py` | Regenerates the evaluation report from the saved model (same split, no retraining). `03_train.py` imports its `build_report` so the two can't drift |
| `analyze_actionability.py` | **The product gate.** Confidence stratification, calibration, and the commit-slice edge over the pitcher's base rate. No retraining. Run this before any modeling change |
| `build_commit_cards.py` | **The product.** Mines per-pitcher, per-count "sit hard / sit soft" rules from held-out predictions. Every rule must beat the COUNT-SPLIT scouting report, not just the pitcher's overall mix |
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

`03_train.py` takes flags that override the config constants, and prints
a RUN CONFIG banner showing what is actually in effect. **Prefer flags to
editing the file** — two runs in a row were invalidated by an edit that
never reached the executed copy:

```bash
python 03_train.py --split random --no-location-head   # run 10
python 03_train.py --split temporal --holdout-season 2025
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
   Each timestep carries the pitch TYPE one-hot, the OUTCOME one-hot
   (ball / called strike / whiff / foul / in-play), physics, and a
   same-at-bat flag. The outcome is safe only because `build_sequences`
   slices strictly before the current index — the outcome of the pitch
   being predicted would leak its type directly. Never widen the slice.
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
7. **A run records the split it actually ran.** `03_train.py` writes
   `data_v5/split_v5.json`; `evaluate_model.py` reads it rather than
   keeping its own copy of `SPLIT_MODE`, and refuses to score if the row
   count disagrees. Never duplicate split config across the two files —
   run 9b was scored on the wrong rows for exactly that reason, and the
   report looked completely normal.
8. **Judge a head on a metric that can move.** Where one class is the
   plurality in nearly every conditioning cell (the zone head: `shadow`
   at 40.8%), argmax accuracy is pinned to the baseline no matter what
   the head learned. Pre-register log-loss against the same baseline, and
   the binary collapse the user can actually act on, alongside accuracy.
9. **Atomic artifact writes.** `01_scrape_statcast.py` writes to a `.tmp`
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

## Current status (2026-09-26)

- **Run 10 is the current model**: 4 seasons, random split, no location
  head. top-1 47.8%, top-3 91.4%, log-loss 1.1719, FB-family 62.8%.
- **The location head is OFF and stays off.** It cost ~0.3 points top-1 /
  0.003 nats and returned a location read actionable on only 5.8% of
  pitches, under the >=10% bar. Location is mostly execution variance; no
  architecture change fixes that. Do not spend more runs on it.
- **Only compare runs on the same data.** Run 8 (3 seasons) vs run 10 (4
  seasons) is invalid — different validation sets, different difficulty.
  Adding 2025 moved top-1 -0.4 and log-loss +0.0153 with config otherwise
  identical, which is the test set changing, not the model regressing.
- **Seed variance is +-0.3 points top-1 / +-0.001 nats** (9a vs 9b, the
  same experiment twice). Anything smaller is noise. Runs 5-8 were
  compared without knowing this.
- **THE BLOCKER: there is still no temporal number.** Every figure so far
  comes from a random split whose validation rows share games and at-bats
  with training rows. The site cannot quote those. Next run:
  `python 03_train.py --split temporal --holdout-season 2025
  --no-location-head`, then `python analyze_actionability.py`.
- All four scoring scripts now route through `evaluate_model.load_split()`.
  They previously hardcoded the random split, which would have scored a
  temporally-trained model on its own training rows.

## Earlier status (2026-09-24)

- **MODELING IS CLOSED.** Run 8 added previous-pitch outcomes to the
  sequence and cleared all three pre-registered section-5 criteria: cell
  win rate 45% -> **51%**, oracle gap -0.1% -> **+0.3%**, lift over the
  count-split scout +0.7% -> **+1.1%**. The model now exceeds the best
  constant rule fitted on the validation rows themselves, so it is
  discriminating within situations — something no static table can
  reproduce. Headline: top-1 48.2%, top-3 92.0%, log-loss 1.1566.
- **The product is a LIVE lookup, not a memorizable card.**
  `build_commit_cards.py` correctly returns ~nothing, because the edge is
  within-situation. Do not resurrect the card format.
- **Section 6 of `analyze_actionability.py` is the number for the site**:
  on the ~36% of pitches where the tool speaks at a 65% threshold, it is
  right ~76%. Lead with that and the binary hard/soft framing — never
  with 10-class top-1.
- Temperature calibration (T = 0.800) is applied in the gate and the card
  miner. The model is under-confident out of the box; calibrating buys
  ~+3 points of coverage for free.
- **Next: build `site/`.** Remaining known defects (random split, mask
  built over train+val, count-based mask threshold) are recorded in
  EXPERIMENTS.md and are measurement-hygiene, not blockers.
- The location head exists and the scrape now covers `zone`, `sz_top`,
  `sz_bot` plus the 2025 season.

## Conventions

- Scripts are flat, numbered, config-at-top; keep that style.
- Comments explain constraints the code can't show (leakage reasoning,
  why a value was chosen), not what the next line does.
- Never commit `.venv/`, `data_v5/`, `*.parquet`, `*.keras`, `*.npy`,
  `*.pkl`.
