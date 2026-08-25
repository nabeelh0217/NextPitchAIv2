# NextPitchAI v2

Predicts the next **pitch type** (FF, SI, FC, SL, ST, CU, KC, CH, FS, KN)
from MLB Statcast data using a BiLSTM + embeddings model whose softmax is
**masked to each pitcher's actual arsenal** — the model can only predict
pitches the pitcher really throws. Continuation of NextPitchAI, revamped
for full-scale release.

## Quick start (Windows)

Double-click `run_pipeline.bat` (or run it from a Command Prompt in the
repo folder). It creates a virtual environment, installs dependencies,
and runs all three steps in order. Steps whose output already exists are
skipped, so a re-run after a failure picks up where it left off:

```bat
run_pipeline.bat            :: run everything (skip completed steps)
run_pipeline.bat --fresh    :: force re-run of every step
```

## Pipeline (v6)

Or run the three scripts manually, in order, **on a machine with
unrestricted internet** (Baseball Savant is scraped directly):

```bash
pip install pybaseball pandas numpy pyarrow scikit-learn joblib matplotlib tensorflow

python 01_scrape_statcast.py   # ~30-90 min -> statcast_raw_v5.parquet (~2-4 GB)
python 02_preprocess.py        # -> data_v5/ arrays + inference artifacts
python 03_train.py             # -> data_v5/best_model_v5.keras + report
```

### 1. `01_scrape_statcast.py`
Pulls pitch-level data for the 2022-2024 seasons in weekly chunks.
Keeps identifiers (pitcher/batter/catcher MLBAM IDs), game state, pitch
physics, ballpark, times-through-order, and pitcher rest days.

### 2. `02_preprocess.py`
Builds training arrays. The target is one of **10 canonical pitch types**
(legacy Statcast codes merged in). Every historical statistic is an
**expanding prior that excludes the current pitch** — the model never
sees the future or the label it is predicting:

| Feature group | What it captures |
|---|---|
| Pitcher arsenal prior | Expanding distribution of pitch types this pitcher throws |
| Matchup history | Pitcher-vs-batter pitch-type distribution + familiarity, shrunk toward the pitcher's arsenal for small samples |
| Batter seen-profile | What pitch types pitchers historically feed this batter |
| Batter whiff rates | Which pitch types this batter swings through, per type |
| Game state | Count, outs, inning, score, baserunners, RISP, pitch of AB |
| Pitcher workload | Times through the order, in-game pitch count, days rest |
| Identity embeddings | Pitcher, batter, catcher, ballpark, handedness |
| Pitch sequences | This pitcher's last 8 pitches: type, velo, location, spin, movement (pfx), same-at-bat flag |
| Arsenal mask | Binary (N,10): which pitch types this pitcher throws at all |

Also emits website inference artifacts: `pitcher_arsenal_v5.json`,
`pitcher_priors_v5.json`, `pitcher_arsenal_mask_v5.json` (the mask the
site feeds the model), `matchup_table_v5.parquet`, and
`batter_profiles_v5.parquet`.

### 3. `03_train.py`
Stacked BiLSTM over the pitcher's own pitch sequence + dense context
branch + six embedding branches. The pitcher's **arsenal mask is a model
input**: non-arsenal pitch logits are forced to -inf before the softmax,
in training and at inference, so all capacity goes into discriminating
within the real repertoire. Class imbalance is handled by
**class-weighted focal loss** (per-class alpha from inverse-sqrt
frequency; no row duplication). Evaluation reports top-1/top-3 accuracy,
a per-type classification report and confusion matrix, and a
"pitcher's most common pitch" baseline for honest lift measurement —
all on the natural class distribution.

## Notes

- Large artifacts (`*.parquet`, `*.npy`, `*.keras`, `*.pkl`, `data_v5/`)
  are gitignored — keep them local or use Git LFS.
- Junk pitch rows (pitchouts, intentional balls, unknowns) are dropped:
  not real pitch-selection decisions.
- `data_v5/` naming is kept for pipeline continuity even though the
  schema is v6 (pitch-type targets); `meta_v5.json` carries
  `"version": 6`.
- Next up: Flask website (`site/`) serving arsenal-masked predictions.
