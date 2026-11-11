# site/ — the hitter-facing app

```bash
.venv/bin/python site/build_serving_artifacts.py
.venv/bin/python site/app.py
```

The first is run once per retrain or new season (add `--data-dir
data_ext` for the latter); the second serves http://127.0.0.1:5000.

Deploying is **[DEPLOY.md](DEPLOY.md)**. The short version: Render builds
from the GitHub repo, so `site/serving/` is committed and must be
rebuilt and committed after every retrain.

## What it claims, and what it must not

The product is a **selective overlay**. It stays silent unless the read
clears the confidence threshold, because a wrong commit costs a hitter
more than no advice at all.

The claim is not written here, on purpose. It is whatever
`analyze_actionability.py` last wrote to `product_claim_v5.json`: how
often the tool speaks, how often it is right there, and how often a
scouting report kept current through the previous pitch would have been
right on those same pitches. An earlier figure (+4.2 points on 35% of
pitches) was measured against a report frozen before the season and is
withdrawn — see `docs/EXPERIMENTS.md`.

Four rules the code enforces rather than trusting anyone to remember:

1. **Coverage always appears beside accuracy.** An accuracy figure alone
   is misleading when it covers a third of pitches.
2. **Never "better than a scouting report" unless the gate says so.**
   The words on the page are chosen by the gate's verdict
   (`claim_view` in `app.py`), not written into the template. If the
   test does not show an edge, the page says that.
3. **Never lead with 10-class top-1.** A hitter cannot act on a ten-way
   distribution in 400ms. The binary call is the product.
4. **The bundle serves what the gate measured.** Arsenals are every
   pitch type thrown through the newest data, keyed by raw MLB id, and a
   player the model never trained on is served from the unknown bucket
   with his own priors.

Every number on the page comes from `product_claim_v5.json`. Nothing is
hardcoded, so a retrain or a harder test updates the site's claim
instead of leaving a stale boast in the HTML.

## Layout

| file | role |
|---|---|
| `build_serving_artifacts.py` | Derives `serving/` from a dataset directory (`--data-dir`, default `data_v5/`): priors, arsenals current through the newest pitch, physics medians, pruned matchups, names, exported weights |
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
