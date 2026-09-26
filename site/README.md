# site/ — the hitter-facing app

## Run it

```bash
python site/build_serving_artifacts.py   # once per retrain
python site/app.py                       # http://127.0.0.1:5000
```

Needs `flask` on top of the training requirements. `site/serving/` is
derived from `data_v5/` and is gitignored — rebuild it after every
training run, or the app serves lookups from the previous model.

## What it claims, and what it must not

The product is a **selective overlay**. It stays silent unless the read
clears the confidence threshold, because a wrong commit costs a hitter
more than no advice at all.

The claim, verbatim from run 12 on a season the model never trained on:

> On 35% of pitches it gives you a read. It is right 72.5% of the time
> there. A count-split scouting report would be right 68.3% on those
> same pitches.

Three rules, all of which the code enforces rather than trusting anyone
to remember:

1. **Coverage always appears beside accuracy.** "72.5% accurate" on its
   own is a misleading claim — it is 72.5% on a third of pitches.
2. **Never "better than a scouting report".** Across all situations a
   fixed per-count rule is as good or better (section 5: +0.7%, and the
   model wins in only 43% of cells). Its value is knowing *when* to
   deviate, not being better everywhere.
3. **Never lead with 10-class top-1** (43.2%). A hitter cannot act on a
   ten-way distribution in 400ms. The binary call is the product.

Every number on the page is read from `data_v5/product_claim_v5.json`,
written by `analyze_actionability.py` from held-out rows. Nothing is
hardcoded, so a retrain that weakens the model updates the site's claim
instead of leaving a stale boast in the HTML. If that file is missing
the claim panel simply does not render.

## Layout

| file | role |
|---|---|
| `build_serving_artifacts.py` | Derives `serving/` from `data_v5/`: per-pitcher and per-batter priors, the arsenal mask the run trained with, physics medians, matchups |
| `predictor.py` | Assembles the serve-time feature vector and makes the call |
| `app.py` | Flask routes: `/`, `/api/arsenal/<id>`, `/api/predict` |

## The thing most likely to break

Serve-time features drifting from training features. Three defences,
kept because this class of bug is silent — the site keeps returning
confident calls while the model reads garbage:

- the game-state block is built by `02_preprocess.build_context_features`,
  imported rather than reimplemented;
- the assembled column order is checked against `ctx_feature_names` in
  `meta_v5.json`, and a mismatch raises instead of predicting;
- the arsenal mask comes from the table the training run saved, so the
  model is served the mask it was trained with.

Player tables are keyed by **raw MLB id**, with the encoded embedding
row in an `enc` column. These were briefly mixed up during development
and every lookup silently missed, falling back to league averages while
the site still returned confident-looking calls. `known_pitcher` /
`known_batter` in the API response now say when a fallback happened, and
the UI warns.
