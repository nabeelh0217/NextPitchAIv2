# NextPitchAI v2

Predicts the next pitch type (fastball / breaking / offspeed / special) from
MLB Statcast data using a BiLSTM + embeddings model with rich, leakage-free
engineered features. Continuation of NextPitchAI, revamped for full-scale release.

## Pipeline (v5)

Run the three scripts in order **on a machine with unrestricted internet**
(Baseball Savant is scraped directly):

```bash
pip install pybaseball pandas numpy pyarrow scikit-learn joblib imbalanced-learn matplotlib tensorflow

python 01_scrape_statcast.py   # ~30-90 min -> statcast_raw_v5.parquet (~2-4 GB)
python 02_preprocess.py        # -> data_v5/ arrays + inference artifacts
python 03_train.py             # -> data_v5/best_model_v5.keras + report
```

### 1. `01_scrape_statcast.py`
Pulls pitch-level data for the 2022-2024 seasons in weekly chunks.
Keeps identifiers (pitcher/batter/catcher MLBAM IDs), game state, pitch
physics, ballpark, times-through-order, and pitcher rest days.

### 2. `02_preprocess.py`
Builds training arrays. Every historical statistic is an **expanding prior
that excludes the current pitch** — the model never sees the future or the
label it is predicting:

| Feature group | What it captures |
|---|---|
| Pitcher arsenal prior | Expanding distribution of buckets this pitcher throws |
| Matchup history | Pitcher-vs-batter bucket distribution + familiarity, shrunk toward the pitcher's arsenal for small samples |
| Batter seen-profile | What buckets pitchers historically feed this batter |
| Batter whiff rates | Where this batter swings and misses, per bucket |
| Game state | Count, outs, inning, score, baserunners, RISP, pitch of AB |
| Pitcher workload | Times through the order, in-game pitch count, days rest |
| Identity embeddings | Pitcher, batter, catcher, ballpark, handedness |
| Pitch sequences | Last 8 pitches: bucket, velo, location, spin, movement (pfx), same-at-bat flag |

Also emits website inference artifacts: `pitcher_arsenal_v5.json`,
`pitcher_priors_v5.json` (for arsenal masking of predictions),
`matchup_table_v5.parquet`, and `batter_profiles_v5.parquet`.

### 3. `03_train.py`
Stacked BiLSTM over the pitch sequence + dense context branch + six
embedding branches, trained with focal loss (gamma=2). Train/val is split
**before** oversampling (so no duplicated rows leak into validation), the
oversampling of rare classes is capped, and evaluation reports
per-class accuracy, a confusion matrix, and top-2 accuracy on the natural
class distribution.

## Notes

- Large artifacts (`*.parquet`, `*.npy`, `*.keras`, `*.pkl`, `data_v5/`)
  are gitignored — keep them local or use Git LFS.
- The "other" bucket from v4 (pitchouts, intentional balls) was dropped:
  those are not real pitch-selection decisions.
- Next up: Flask website (`site/`) serving predictions with arsenal masking.
