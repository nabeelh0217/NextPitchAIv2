# Experiment log

Every training run, what changed, what happened, and the verdict. Append
a row after each run. Runs take an hour or more — **change one thing at a
time** so the result is attributable.

All metrics are on the natural-distribution validation split (20%,
stratified, split *before* any resampling). Runs 1–4 share identical
data and labels except where noted, so they are directly comparable.

## Pre-history (v4)

The original v4 pipeline reported **66% accuracy** on 5 buckets. That
number was inflated: it oversampled the whole dataset *before* the
train/val split, so duplicated minority rows appeared in both sets. Not a
valid baseline.

## v5 — 4 buckets (fastball / breaking / offspeed / special)

| Run | Change vs. previous | Acc | Top-2 | Macro-F1 | Verdict |
|---|---|---|---|---|---|
| 1 | v5 baseline: leakage-free expanding priors, park+catcher embeddings, split-before-oversample, capped oversampling (`MAX_MINORITY_FRACTION=0.4`, `MAX_DUPLICATION=20`) | **59.7%** | **91.7%** | 0.553 | First honest number. breaking↔fastball is the dominant confusion (~102K of ~172K errors). `special` over-fires (95.7% recall, 43.2% precision). |
| 2 | Cutter `FC` moved fastball→breaking; `MAX_DUPLICATION` 20→8 | 57.2% | 90.7% | 0.557 | **Worse.** Confusion pair grew to ~115K. Cutters are genuinely in-between; moving them relocated the boundary problem and shrank the easy majority class. `MAX_DUPLICATION` change ~neutral. FC reverted. |
| 3 | FC reverted; **sequences rebuilt per (game, pitcher)** instead of previous N rows of the game | 59.5% | 91.6% | — | Flat overall, but breaking recall collapsed 42.4%→28.9% while fastball rose to 78%: model leaned into the majority. Sequence fix is structurally correct (kept) — it removed the opposing pitcher's pitches from the lookback. |
| 4 | `MAX_MINORITY_FRACTION` 0.4→0.75 (breaking had been getting zero oversampling: cap sat below its natural count) | 54.3% | 89.9% | 0.540 | Breaking recall recovered to 50.7% but offspeed over-fired (precision 0.324, 61K fastballs called offspeed), fastball recall fell to 50.7%. **Oversampling knobs are zero-sum** — three rounds just moved recall between classes. Training curves: val loss bottoms at epoch ~4–7 then rises; duplicated rows drive early memorization. |

### Conclusions that drove v6
- The bucket target is the ceiling, not the tuning. "Breaking" means a
  different pitch for every pitcher; the model was being asked to blur
  distinct decisions.
- Physical row duplication (RandomOverSampler) accelerates overfitting.
  Use loss weighting instead.
- ~60% top-1 / ~91% top-2 on 4 buckets is in the published range for this
  problem; don't expect big jumps from knob-turning.

## v6 — 10 real pitch types, arsenal-masked softmax

Changes: target = canonical pitch type (FF, SI, FC, SL, ST, CU, KC, CH,
FS, KN; legacy codes merged); all priors at pitch-type granularity
(context 31→55 features); per-pitcher binary arsenal mask fed to the model
and applied to the logits before softmax; RandomOverSampler removed in
favor of per-class alpha (inverse-sqrt frequency, mean-normalized, capped
at 4) inside the focal loss; evaluation adds top-3 and a
"pitcher's most common pitch" baseline.

| Run | Change vs. previous | Top-1 | Top-3 | Baseline | Lift | Verdict |
|---|---|---|---|---|---|---|
| 5 | v6 first real run (MacBook Air, Python 3.13) | **44.0%** | **91.3%** | 42.8% | **+1.2%** | 28 epochs, best val_loss 0.2278 at epoch 20. **No early overfitting** — val loss plateaus instead of rising, so the class-weighted loss did fix v5 run 4's memorization. Argmax lift is nearly nil, but rescoring with the prior-*distribution* baseline showed the features carry real signal: **log-loss 1.1864 vs prior 1.3221 (+0.136 nats), top-3 91.3% vs 85.9% (+5.5%)**. The top-1 was suppressed by the weighting: `ALPHA_POWER=0.5` produced a **17:1** weight ratio (FF 0.233, KN 4.0) *on top of* focal gamma=2. FF recall 30.2% at precision 0.637 (knows fastballs, penalized for calling them); KC recall 85.0% at precision 0.270, KN recall 98.1% at precision 0.336. Imbalance corrected twice — v5's error in a different form. |
| 6 | `ALPHA_POWER` 0.5 → 0.0 (uniform alpha; focal gamma alone handles imbalance). Evaluation now compares against the pitcher-prior **distribution** on top-1/top-3/log-loss. | _pending_ | | | | |

## Do not retry

- **Cutter (FC) as breaking** — run 2, hurt on every metric.
- **Tuning `MAX_MINORITY_FRACTION` / `MAX_DUPLICATION`** — runs 2–4,
  zero-sum. (Both constants no longer exist in v6.)
- **Game-level sequence lookback** — the v4 bug; per-pitcher is correct.
- **Oversampling before the split** — the v4 66% was fake.
- **Stacking two imbalance corrections.** Oversampling + focal loss (v5)
  and alpha weighting + focal loss (run 5) both wrecked the majority
  class. Focal gamma=2 is the correction; add class alpha only with
  evidence that a class is genuinely ignored, and keep the ratio small.

## The bar to beat

The honest baseline is **the pitcher's own pitch mix, with no game
context** — `evaluate_model.py` scores it as a full probability
distribution (top-1, top-3, log-loss), not just its argmax. Any claim
that sequence/count/matchup features matter has to show up as a log-loss
improvement over that.

**Settled by rescoring run 5: the features have real signal.** Model
log-loss 1.1864 vs prior 1.3221 (+0.136 nats ≈ 14% higher likelihood on
the true pitch), top-3 +5.5 points over the prior. Run 5's flat top-1
was the class-weighting bug, not an information ceiling. Reference
numbers for the prior baseline on this split: top-1 42.8%, top-3 85.9%,
log-loss 1.3221.

## Pipeline bug fixes worth remembering (so they aren't regressed)

| Commit | Fix |
|---|---|
| `56f6303` | Runner auto-detects a TensorFlow-compatible Python (3.10–3.13); newer system Pythons have no TF wheels. |
| `56ec22a` | Runner validates an existing `.venv`'s interpreter and rebuilds it if incompatible — a stale venv from a failed run was being silently reused. |
| `5bd31a6` | Scrape writes to `.tmp` then `os.replace`; runner treats a parquet < 1 MB as corrupt and re-scrapes. An interrupted write left a 0-byte file that was trusted as complete. |
| `f7ecd13` | Sequences grouped by `(game_pk, pitcher)`. |
