# site/ — the hitter-facing app

```bash
python site/build_serving_artifacts.py   # once per retrain
python site/app.py                       # http://127.0.0.1:5000
```

Deploying is **[DEPLOY.md](DEPLOY.md)**. The short version: Render builds
from the GitHub repo, so `site/serving/` is committed and must be
rebuilt and committed after every retrain.

## What it claims, and what it must not

The product is a **selective overlay**. It stays silent unless the read
clears the confidence threshold, because a wrong commit costs a hitter
more than no advice at all.

The claim, from run 12 on a season the model never trained on:

> On 35% of pitches it gives you a read. It is right 72.5% of the time
> there. A count-split scouting report would be right 68.3% on those
> same pitches.

Three rules the code enforces rather than trusting anyone to remember:

1. **Coverage always appears beside accuracy.** "72.5% accurate" alone is
   misleading — it is 72.5% on a third of pitches.
2. **Never "better than a scouting report".** Across all situations a
   fixed per-count rule is as good or better (+0.7%, and the model wins
   in only 43% of cells). Its value is knowing *when* to deviate.
3. **Never lead with 10-class top-1** (43.2%). A hitter cannot act on a
   ten-way distribution in 400ms. The binary call is the product.

Every number on the page comes from `product_claim_v5.json`, written by
`analyze_actionability.py` from held-out rows. Nothing is hardcoded, so a
retrain that weakens the model updates the site's claim instead of
leaving a stale boast in the HTML. Missing that file just hides the
panel.

## Layout

| file | role |
|---|---|
| `build_serving_artifacts.py` | Derives `serving/` from `data_v5/`: priors, arsenal mask, physics medians, pruned matchups, names, exported weights |
| `export_model.py` | Keras → `model_weights.npz`, verifying NumPy parity before writing |
| `numpy_model.py` | The forward pass, numpy only — no TensorFlow at serve time |
| `player_names.py` | MLB id → name, cached; works offline |
| `predictor.py` | Assembles the serve-time feature vector and makes the call |
| `app.py` | `/`, `/api/players`, `/api/arsenal/<id>`, `/api/predict`, `/healthz` |
| `wsgi.py` | gunicorn entrypoint; loads the model at import so `--preload` shares it |

## What breaks, based on what already has

**Serve-time features drifting from training features.** Three defences,
kept because this failure is silent — the site keeps returning confident
calls while the model reads garbage:

- the game-state block comes from `pitch_features.build_context_features`,
  the same function training uses — imported, never reimplemented;
- the assembled column order is checked against `ctx_feature_names` and
  raises on mismatch instead of predicting;
- the arsenal mask comes from the table the training run saved.

**Encoded vs raw ids.** Player tables are keyed by **raw MLB id** with
the embedding row in an `enc` column. These were mixed up once and every
lookup silently missed, falling back to league averages while the site
still returned confident calls. `known_pitcher` / `known_batter` in the
API now report any fallback and the UI warns.

**A stale bundle.** Anything a build did not write is swept from
`serving/`, because a leftover file is served as though it were current.

**TensorFlow creeping back in.** It costs 663 MB against a 512 MB box.
`requirements.txt` is the contract.
