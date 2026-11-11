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
app in `site/` serves these predictions; it is built and deploys to
Render (see **Deployment**).

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

**That 2026-09-23 "gate PASSED" was measured on a random split and is
withdrawn.** On a temporal split (run 11, hold out 2025) the model does
NOT beat a count-split scouting table across situations: lift -0.6%, it
wins in 38% of cells, and it trails the best constant per-cell rule by
2.8 points. All three pre-registered criteria fail. The conclusion that
the edge is within-situation was a random-split artifact.

What run 12 reported (2026-09-26): on the **35%** of pitches where it is
confident, it is right **72.5%** where the count-split table is right
**68.3%** — **+4.2 points**. **That figure is superseded and must not be
quoted** (2026-11-11). It was measured against a table frozen at the end
of training while the model's inputs kept updating, with an arsenal mask
built over every row, on the season the threshold was tuned on. All
three flatter the model. The number of record is whatever
`analyze_actionability.py --test-season 2026` writes: the 2026 season,
used for nothing else, against a scouting table kept current through the
previous pitch. Whatever it says, the claim is a selective overlay, not
a replacement, and is always stated with its coverage.

Two rules that follow, and must not be softened:
- **A card names the likeliest pitch WITHIN the family it tells the hitter
  to sit on.** "Sit soft, likeliest fastball" is incoherent advice.
- **The bar is the count-split scouting report**, not the pitcher's
  overall mix. Any advance scout already knows he throws fastballs 3-1;
  beating that is the only edge worth claiming.
- **The report is kept current.** A real one in August includes April to
  July. Comparing the model against a table frozen before the season
  credits it with knowing what happened since — which is not insight.

## File map

| Path | Role |
|---|---|
| `01_scrape_statcast.py` | Pulls 2022–2026 Statcast pitches via pybaseball → `statcast_raw_v5.parquet` (atomic write). **Incremental**: with a parquet present it fetches only the weeks that have no rows and appends, never rewriting what a model was built on. `--fresh` re-scrapes everything |
| `02_preprocess.py` | Leakage-free feature engineering → `data_v5/` arrays, scalers, ID maps, website artifacts. `--freeze-from data_v5 --out-dir data_ext` extends a dataset forward with the ID maps, scalers and league fallbacks FROZEN, and refuses to finish unless the original seasons reproduce exactly — that is what makes a later season a test set without retraining |
| `03_train.py` | BiLSTM + embeddings + arsenal-masked softmax, class-weighted focal loss → `data_v5/best_model_v5.keras` + `eval_report_v5.txt` |
| `evaluate_model.py` | Regenerates the evaluation report from the saved model (same split, no retraining, no TensorFlow). **Also the one home of everything a scoring script needs**: `scoring_mask`, `known_ids`, `live_scout`, `predict_rows`, `cluster_bootstrap_ci`. `03_train.py` imports its `build_report` so the two can't drift. `--as-trained` reproduces the training-time report exactly |
| `analyze_actionability.py` | **The product gate, and the only source of the site's numbers.** Scores with a causal mask against a live scouting table; section A re-derives the previously published number and shows what each correction costs it. `--test-season 2026` locks temperature, threshold and bar on the validation season and scores the test season once. Run this before any modeling change |
| `test_leakfree.py` | Checks that the mask and the scouting table never read the pitch being judged or anything after it, each against a brute-force loop. Seconds, no data needed. **Run it after touching `pitch_features.py` or `evaluate_model.py`** |
| `build_commit_cards.py` | Mines per-pitcher, per-count "sit hard / sit soft" rules from held-out predictions. A rule must contradict the pre-season report AND beat the live table. Correctly returns ~nothing; the card format is dead |
| `diagnose_unknown_pitcher.py` | **Gates the custom-arsenal mode.** Ablates the pitcher embedding to index 0 on held-out rows at the gate's own locked temperature, threshold and bar, so its rows are comparable to section 6 |
| `diagnose_arsenal.py` | Measures how much the arsenal mask actually constrains, and whether the baseline comparison is fair |
| `run_pipeline.sh` / `run_pipeline.bat` | One-command runners (macOS/Linux, Windows). Skip completed steps; validate artifacts |
| `docs/EXPERIMENTS.md` | Every training run, its numbers, and its verdict. **Append a row after every run.** |
| `pitch_features.py` | **Shared feature vocabulary.** Pitch/outcome/zone classes and every pure feature builder. Imported by BOTH `02_preprocess.py` and the site so training and serving cannot drift. numpy + pandas ONLY — keep it that way |
| `site/` | **The product.** Flask app serving the selective overlay. `build_serving_artifacts.py` -> `export_model.py` -> `predictor.py` -> `app.py` |
| `site/numpy_model.py` | TensorFlow-free inference. `export_model.py` converts the Keras graph to `model_weights.npz` and verifies parity before writing |
| `site/player_names.py` | MLB Stats API id→name lookup, cached to `site/serving/player_names.json`. Network-optional; the site falls back to numeric ids |
| `site/DEPLOY.md` | The Render runbook: build, commit the bundle, custom domain, memory, troubleshooting |
| `render.yaml` | Render blueprint. gunicorn `--workers 2 --preload`, health check `/healthz` |
| `.cursor/rules/` | Cursor-scoped rules (point back here) |

`data_v5/`, `data_ext/` and `*.parquet` are gitignored (GBs), with one
deliberate exception: `site/serving/` is committed so Render can serve
it. `data_ext/` is `data_v5/` extended forward by `--freeze-from`; it
holds the same model and, for the seasons the model knew, the same
arrays bit for bit. The
`_v5` file naming is
kept for runner continuity even though the schema is v6 — `meta_v5.json`
carries `"version": 6`.

## How to run

```bash
./run_pipeline.sh                    # macOS/Linux: everything, skipping done steps
./run_pipeline.sh --fresh-preprocess # rebuild data_v5/ + retrain, keep parquet
./run_pipeline.sh --fresh            # redo all, incl. the 30-90 min scrape
run_pipeline.bat                     # Windows equivalent (--fresh only)
```

**`--fresh-preprocess` deletes `data_v5/`, model included.** And once the
parquet holds 2026, a plain rebuild fits the scalers and ID maps on 2026
too, which spends it as a test season. To retrain and keep one:

```bash
./run_pipeline.sh --fresh-preprocess --fit-through=2025
```

`03_train.py` takes flags that override the config constants, and prints
a RUN CONFIG banner showing what is actually in effect. Its defaults are
the shipping configuration (temporal split, hold out 2025, location head
OFF), because the runners call it with no flags. **Prefer flags to
editing the file** — two runs in a row were invalidated by an edit that
never reached the executed copy:

```bash
.venv/bin/python 03_train.py --split random --no-location-head   # run 10
.venv/bin/python 03_train.py --split temporal --holdout-season 2025
```

Needs Python 3.10–3.13 (TensorFlow has no wheels for newer). The runners
find one automatically and manage `.venv`. The scrape hits Baseball
Savant directly — it cannot run from sandboxed/proxied environments.

**Testing the shipped model on a new season, no retraining** — the flow
that produces the site's number. One command at a time; each prints what
to run next if something is missing:

```bash
.venv/bin/python 01_scrape_statcast.py
.venv/bin/python 02_preprocess.py --freeze-from data_v5 --out-dir data_ext
.venv/bin/python analyze_actionability.py --data-dir data_ext --test-season 2026
.venv/bin/python site/build_serving_artifacts.py --data-dir data_ext
```

Then commit `site/serving` and push. TensorFlow is needed for none of
it. `.venv/bin/python test_leakfree.py` checks the measurement code
itself in a few seconds.

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
4. **The arsenal mask never knows the future.** `02_preprocess.py`
   still emits a full-data mask, and nothing may be scored with it.
   `03_train.py` rebuilds it from `idx_tr` for TRAINING. Every scoring
   script gets its mask from `evaluate_model.scoring_mask`, which is
   CAUSAL: the types a pitcher had thrown before each pitch, keyed by
   his raw MLB id, all ones until `MASK_MIN_HISTORY` (100) career
   pitches. The serving bundle carries the same thing as of the newest
   pitch. A full-data mask leaks the future on a temporal split — it
   reveals pitches a pitcher only added in the held-out season — and
   for two months every scoring script used exactly that while training
   used the honest one. The logit penalty is `ARSENAL_MASK_PENALTY = -12`,
   not -1e9: ~2.4% of held-out pitches are a type the pitcher had never
   thrown, and infinite penalty is wrong for an event that happened.
   Pitchers with no training rows get an all-ones mask, as do unknown
   pitchers at inference. Never leave a row all-zero — every logit would
   hit the floor and the softmax would be meaningless.
5. **Split before any resampling; evaluate on the natural distribution.**
   Never oversample/duplicate rows before `train_test_split`. (v6 has no
   resampling at all — class balance is per-class alpha in the focal
   loss.) Reported metrics are always on the untouched validation split.
6. **Judge against the count-split table, kept current, never the
   pitcher's overall mix.** The overall mix ignores the count, which
   every hitter reads off a scouting report. Section 3 of
   `analyze_actionability.py` uses it and is a diagnostic only; sections
   5 and 6 carry the real bar and the final verdict gates on those. The
   bar is `evaluate_model.live_scout`: built only from pitches before
   the one judged, so both sides of the comparison know the same things.
   Never compare against a table frozen at the end of training. Report
   coverage beside accuracy always — an accuracy figure without the
   share of pitches it covers is a misleading claim — and report the
   interval beside the edge.
7. **One change per training round, logged.** Runs take an hour+.
   Change one thing, retrain, append the result to
   `docs/EXPERIMENTS.md` with the verdict, then decide the next change.
8. **A run records the split it actually ran.** `03_train.py` writes
   `data_v5/split_v5.json`; `evaluate_model.py` reads it rather than
   keeping its own copy of `SPLIT_MODE`, and refuses to score if the row
   count disagrees. Never duplicate split config across the two files —
   run 9b was scored on the wrong rows for exactly that reason, and the
   report looked completely normal.
9. **Judge a head on a metric that can move.** Where one class is the
   plurality in nearly every conditioning cell (the zone head: `shadow`
   at 40.8%), argmax accuracy is pinned to the baseline no matter what
   the head learned. Pre-register log-loss against the same baseline, and
   the binary collapse the user can actually act on, alongside accuracy.
10. **Atomic artifact writes.** `01_scrape_statcast.py` writes to a `.tmp`
   and `os.replace`s into place; the runners refuse a parquet < 1 MB. A
   corrupt/empty artifact silently reused cost a full re-scrape once.
11. **A test season is scored once, with everything locked.** The
   temperature, the commit threshold and the choice of bar are fixed on
   the validation season; the test season is then scored and the result
   stands. The validation season is where the checkpoint, the
   temperature and the threshold were chosen, so its number is a tuning
   result and is never the claim once a test number exists. Criteria go
   into `docs/EXPERIMENTS.md` BEFORE the test is run. A season the model
   has trained or tuned on is never a test season again — and that
   includes preprocessing: `02_preprocess.py` fits its scalers, ID maps
   and league fallbacks on every season it is given unless
   `--fit-through` holds the later ones out.
12. **An id the model never trained on is the unknown bucket.**
   `02_preprocess.py` numbers players over all rows, so anyone who
   debuted in the held-out season has an embedding row that was never
   updated and still holds random values. `evaluate_model.known_ids`
   sends those ids to 0 at scoring time and in the serving bundle.
   Keyed lookups use the RAW MLB id: the encoded id pools every such
   player into one row.
13. **One implementation of the measurement.** The mask, the scouting
   table, the id rule and the interval live in `evaluate_model.py` and
   `pitch_features.py`. A scoring script that builds its own copy will
   drift, and the last time four of them did, none of the four agreed
   with what training used.

## Evaluation vocabulary

`03_train.py` and `evaluate_model.py` print the model against the
**pitcher-prior baseline** — that pitcher's training pitch mix used as a
full probability distribution — on top-1, top-3 and log-loss. The
baseline is what you can predict knowing only who is on the mound, so
**log-loss improvement over it is the test of whether the game-context
features do anything at all**. 10-class top-1 reads lower than the old
4-bucket numbers; that is expected and not a regression.

## Current status (2026-11-11)

- **The measurement was rebuilt; the model was not.** Run 12 is still
  the shipping model. Four defects were found in how its "+4.2 points on
  35%" was measured — a frozen scouting table, a mask from the future in
  every scoring script, a "held-out" season that was also the tuning
  season, and never-trained embedding rows fed to the model. All four
  are fixed in the code. `docs/EXPERIMENTS.md` has the detail and the
  criteria, which were written before any 2026 pitch was scored.
- **The 2026 result is PENDING.** It needs the scrape, which cannot run
  from a sandbox. Until the four commands under "How to run" have been
  run and the bundle committed, the live site shows the run-12 figure
  with a note that it was measured against a frozen report.
- **Do not quote +4.2, and do not guess the new number.** On a synthetic
  league the same corrections took a +2.0 edge to zero. That shows the
  code works; it says nothing about the size of the effect on real
  pitching.
- **The site's words follow the verdict.** `site/app.py:claim_view`
  picks the sentences from the gate's status (confirmed / reduced /
  not_confirmed / validation_only / legacy), including the last
  paragraph of the About tab. Never write "beats a scouting report"
  into a template; make it conditional on `cv`.
- **The bundle now lists every pitcher with 100+ pitches** by raw MLB id,
  including those who debuted after training. They are served from the
  unknown-pitcher embedding with their OWN priors and arsenal — which is
  how the gate scores them.
- **Next modeling step, when there is one:** feed the model the live
  count x side rate as a feature. See the end of the measurement entry
  in `docs/EXPERIMENTS.md` for the ordered list. One change per round.

## Earlier status (2026-09-26) — superseded where it conflicts with the above

- **Run 12 is the shipping model.** Temporal split (hold out 2025),
  no location head, honest training-only arsenal mask. top-1 43.2%,
  log-loss 1.5765.
- ~~The pre-registered ship criterion was met: section 6 edge +4.2
  points on 35.2% of pitches.~~ **SUPERSEDED 2026-11-11** — measured
  against a frozen table with a mask from the future, on the tuning
  season. The criterion (>= +2.0 on >= 10%) stands; the measurement of
  it is the 2026 test.
- ~~The product claim: "On 35% of pitches ... right 72.5% ... table
  68.3%."~~ **SUPERSEDED** — the claim is whatever the gate writes for
  2026. Coverage ALWAYS beside accuracy still holds.
- **It is a selective overlay, NOT a replacement for a scouting report.**
  Never claim "better than a scouting report". (The +0.7% / 43%-of-cells
  figures were against the frozen table too; read section 5 of the
  current report instead.)
- **Do not quote section 3's +8.3%.** It scores against the pitcher's
  overall mix, which ignores the count. Diagnostic only.
- Log-loss lift over the smoothed baseline is only **+0.0435** now. On
  full-distribution terms the model is barely better than a smoothed
  pitcher prior; its value is concentrated in the confident slice.
- **Never quote a random-split number.** Runs 1-10 share games and
  at-bats between train and validation.
- The location head is OFF and stays off. Seed variance is +-0.3 points
  top-1 / +-0.001 nats.
- **`site/` is built.** `.venv/bin/python site/build_serving_artifacts.py` then
  `.venv/bin/python site/app.py`. Every figure it shows is read from
  `product_claim_v5.json` in the bundle, which `analyze_actionability.py`
  writes from held-out rows — nothing is hardcoded, so a retrain updates
  the site's claim instead of leaving a stale boast in the HTML.
- **Serving-bundle rule:** `site/serving/` is derived but **committed** —
  Render builds from the repo and cannot see `data_v5/`. Rebuild AND
  commit it after EVERY training run, or the app serves the previous
  model's lookups against the new weights.
- **The batter is optional.** Whoever uses this IS the batter and is not
  in an MLB table, so an absent batter falls back to LEAGUE averages (not
  uniform — an expanding prior backs off to the league on a player's
  first pitch, so that is the state the model was trained to see).
  Handedness is still asked for; platoon splits are large.
- **Custom arsenal mode** lets a hand-entered pitch mix stand in for a
  pitcher who is not in the data (college, high school, a call-up). The
  embedding goes to index 0 — the unknown-pitcher bucket the model was
  TRAINED with — and the mask comes from the listed pitches.
  **Measured (2026-09-27): it costs essentially nothing.** Ablating the
  embedding, zone priors and matchup history and adding noise to the mix
  moves the edge +5.6% -> +6.1% and top-1 by -0.2 points. (Both edges
  were against the frozen table, so only the COMPARISON between them
  holds. Re-run `diagnose_unknown_pitcher.py --data-dir data_ext` for
  numbers on the current bar.) What the model
  needs from a pitcher's identity is mostly WHICH PITCHES ARE POSSIBLE,
  and that is the arsenal mask, which custom mode supplies by hand. This
  also explains the +0.0435 log-loss lift: past the arsenal, the edge is
  count and sequence, both of which survive.
  **It does NOT validate college or high-school use.** That was MLB
  pitching with identity hidden, not pitchers whose sequencing the model
  has never seen. Claim "supply the arsenal instead of the name", never
  "it works outside affiliated baseball" — that needs non-MLB data.
- Serving tables are keyed by **raw MLB id** with the encoded embedding
  row in an `enc` column. Mixing the two makes every lookup miss
  silently and fall back to league averages while the site still returns
  confident calls. `known_pitcher`/`known_batter` in the API flag it.

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

## Deployment (Render, public repo)

**Render builds from the GitHub repo and nothing else.** `data_v5/`, the
raw parquet and `best_model_v5.keras` are all gitignored, so none of them
exist on the server. `site/serving/` is therefore COMMITTED — it is the
only path by which the model reaches production.

The release loop, in order, every time the model changes:

```bash
.venv/bin/python 03_train.py ...                     # retrain
.venv/bin/python analyze_actionability.py            # writes product_claim_v5.json
.venv/bin/python site/build_serving_artifacts.py     # bundle + names + weight export
git add site/serving && git commit && git push
```

When the dataset is `data_ext/` (new season, same model), pass
`--data-dir data_ext` to the last two and skip the retrain. The builder
refuses to run without a claim in that directory, and refuses a claim
whose `model_sha256` is not the model it is about to export.

**Skipping the rebuild is the dangerous failure.** The app would serve the
previous model's priors and arsenals against the new weights and return
confident, plausible, wrong calls. The claim and the weights are now
tied by hash at build time, but nothing ties the committed bundle to the
model on your disk — the discipline is still the safeguard.

Serving is TensorFlow-free by necessity, not preference. Measured in
clean subprocesses: TensorFlow loading the model is **663 MB** RSS, the
whole serving stack is **128 MB**. Render's free tier is 512 MB.
TensorFlow is needed ONLY for the offline `export_model.py` step.

- `export_model.py` reads the `.keras` archive directly (zipfile + h5py)
  and needs NO TensorFlow. TF is used only for the optional parity
  cross-check, which runs in a CHILD process — a broken TF install calls
  abort(), which no try/except can catch, and that would otherwise take
  the export down after the weights were already written. Verified
  bit-identical to the TF path: all 34 arrays equal, same arch JSON.
- The h5 weight groups are keyed by Keras's AUTO-GENERATED names, not the
  layer names in config.json (`logits` is stored under `dense_3`).
  `vars/@name` records the true name — index by that. Matching
  positionally would load real weights into the wrong layer and still
  produce a model that runs.
- Nothing in the serving path may import tensorflow, keras, torch or
  scikit-learn. `site/requirements.txt` is the contract; adding one of
  them silently reintroduces the OOM.
- The bundle is capped at `BUNDLE_MAX_MB` (40). If a rebuild trips it,
  raise `MIN_MATCHUP_MEETINGS` rather than the cap.
- The free tier sleeps after ~15 min idle and cold-starts in 30-60s.
  A paid instance removes that; pricing changes, so check it.

## Conventions

- Scripts are flat, numbered, config-at-top; keep that style.
- Comments explain constraints the code can't show (leakage reasoning,
  why a value was chosen), not what the next line does.
- Never commit `.venv/`, `data_v5/`, `*.parquet`, `*.keras`, `*.npy`,
  `*.pkl` — except inside `site/serving/`, which is the deploy bundle and
  is committed on purpose. `.gitignore` encodes that carve-out; the
  bundle is size-capped so it cannot quietly grow.
