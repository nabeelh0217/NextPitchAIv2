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
| 6 | `ALPHA_POWER` 0.5 → 0.0 (uniform alpha; focal gamma alone handles imbalance). Evaluation now compares against the pitcher-prior **distribution** on top-1/top-3/log-loss. | **48.9%** | **92.1%** | 42.8% | **+6.1%** | 28 epochs, best val_loss 0.5984 at epoch 20. **Every headline metric best-so-far**: top-1 lift 5x better than run 5, top-3 +6.2 over prior, log-loss 1.1531 vs prior 1.3221 (+0.169). Confirms the run-5 diagnosis — the double imbalance correction was suppressing real signal. **But it is a frontier move, not a free win**: the model now leans on FF/SI when uncertain, so minority recall collapsed (CU 51.4%→13.5%, KC 85.0%→21.0%, FS 84.2%→48.8%, ST 67.5%→41.1%) while FF rose 30.2%→69.5% and SI 45.4%→62.7%. macro-F1 fell 0.449→0.422 even as weighted-F1 rose 0.438→0.468. A curveball is now the argmax only 13.5% of the times it is actually thrown. |
| 7 | `ALPHA_POWER` 0.0 → 0.25 (midpoint probe; weight ratio 3.6:1 vs run 5's 17.2:1 and run 6's 1:1) | **47.4%** | **91.9%** | 42.8% | **+4.5%** | 27 epochs, best val_loss 0.4281 at epoch 19. log-loss 1.1650. **Best macro-F1 of any run (0.457** vs run 6's 0.422 and run 5's 0.449). Decision rule partially met: KC recovered 21.0%→53.5% and FS 48.8%→74.8%, but CU reached only 30.6% (rule wanted ~35-40%); top-1 lift +4.5% > +1.2% and log-loss 1.1650 < 1.1864, both pass. **Verdict: the alpha frontier is confirmed and closed.** Runs 5/6/7 moved log-loss only 0.033 nats while moving top-1 4.9 points — the information content is pinned; alpha only moves the decision threshold. Pick by product need: run 6 for best calibration/top-1, run 7 for balance across pitch types. |

## Post-run-7 audit (5 parallel auditors + adversarial verification)

Triggered by the question "is the evaluation flawed because the model is
guessing from all pitch types rather than the pitcher's arsenal?"

**That hypothesis is REFUTED**, unanimously and at high confidence, four ways:
the mask op is baked into the saved graph (03_train.py:246) and reproduced by
`load_model`; `evaluate_model.py` passes the mask as the 9th input (a missing
mask would be a shape error, not a silent pass-through); Keras softmax makes a
masked logit exactly 0.0; and the confusion matrix shows only 707/426,731
knuckleball predictions. **Critically, the pitcher-prior baseline is ALSO
arsenal-constrained** — a pitcher's own pitch mix assigns zero to types they
never threw — so the reported lift was already a within-arsenal comparison.
One auditor went further: the mask is structurally a *superset* of the
baseline's support, so it contributes ~zero to the reported lift.

**The one place the intuition lands:** at deployment, an unseen pitcher gets an
all-ones mask, so the served model really does choose among all 10 types.
Expect materially worse than 47.4% for debut/low-volume pitchers.

### Real defects found, ranked by effect on the reported numbers

| Severity | Defect | Effect |
|---|---|---|
| **CRITICAL** | Random per-pitch split puts same-at-bat and same-outing pitches in both train and val (`03_train.py` `train_test_split`). Validation rows are not independent of training rows. | Inflates top-1/top-3/log-loss AND the lift, since the baseline is split-invariant. Est. **1-4 points of top-1** — a large fraction of the +4.5/+6.1 lift. Magnitude unmeasured. |
| **HIGH** | Arsenal mask is built over train+val (`02_preprocess.py:373`), so the true validation label can never be zeroed. | Contaminates log-loss specifically. Estimates ranged 5-18%, 20-70%, and 0.06-0.13 of the 0.157 lift — auditors disagreed; needs measurement. Barely touches top-1/top-3. |
| **HIGH** | Baseline is unsmoothed, so it can assign exactly 0 and eat 27.6 nats after clipping; the masked softmax structurally never can. | Inflates the log-loss lift only. Est. +0.005 to +0.028 nats of the reported +0.157. **Fixed** — evaluator now reports a smoothed baseline alongside the raw one. |
| **MEDIUM** | Mask threshold is a raw count (>=1 ever), not a usage share, so permissiveness scales with pitch volume — starters' masks approach all-ones. | 25,684 val rows (6.0%) are predictions a true-arsenal mask would forbid. Degrades the calibrated distribution, which is the product. |
| **MEDIUM** | macro-F1 / per-class recall were used to steer three hour-long runs, but the product ships a probability distribution. Focal loss is not a proper scoring rule. | No effect on the numbers; it misdirected the tuning. On log-loss alone, run 6 wins outright and the alpha search should have ended there. |
| LOW | Baseline keyed on UNK-collapsed pitcher id while the model's mask uses the raw id. | ~0.5% of rows, ~3-4% of the lift. |
| LOW | No held-out test set — early stopping, checkpointing and the 3-run alpha search all scored on the same rows. | ~2% of the lift; does not change run ordering. |

### The ceiling (the answer to "something isn't working")

- Pitcher identity carries **78% of all extractable information**. The entire
  game-context contribution is worth ~0.58 "effective pitch types".
- **Realistic ceiling is 52-56% top-1.** At 47-49% the model has captured
  roughly half the available headroom.
- **Top-3 at 91.9-92.1% is within 2-3 points of any achievable ceiling** — and
  top-3 is what the product ships. The 10-class top-1 is the wrong headline.
- Published numbers that look better are **binary** (fastball vs rest) or use
  the predicted pitch's own velocity/spin as a feature, i.e. leakage. Measured
  as lift over its own honest baseline, this project is at or above the
  non-leaky literature.
- Collapsed to fastball-family vs rest, run 7 scores ~62% vs a ~56% baseline —
  now reported automatically by `evaluate_model.py`.
- Highest-leverage next move is **shipping with the right headline metric**.
  More data ranks LAST and may be net negative.

### Run 7 decision rule (fixed in advance)

Both endpoints of the alpha frontier are now measured, so this is a
single bounded probe, not a sweep. Keep `ALPHA_POWER = 0.25` **only if**
it recovers minority recall (CU and KC back above ~35–40%) **and** holds
top-1 lift above run 5's +1.2% with log-loss at or below run 5's 1.1864.
Otherwise revert to `0.0` and accept run 6 as the final model. Either
outcome ends the tuning loop; next work is the Flask site.

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
- **Sweeping `ALPHA_POWER` further.** Runs 5/6/7 measured 0.5, 0.0 and
  0.25. This is a precision/recall frontier — moving the constant trades
  minority recall against top-1/log-loss, it does not unlock a better
  model. Pick the point that fits the product and stop.

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

## Reframe (2026-09-23): the goal is a hitter's edge, not top-1

Runs 1-7 optimized accuracy. Against the project's actual purpose — a
batter gaining an advantage — accuracy was never the right target, and the
entropy analysis shows it is now exhausted anyway (run 6 sits within 0.02
nats of what a pitcher throwing an ordinary 4-pitch mix leaves on the
table; pitcher identity alone carries 78% of all extractable information).

**What changed:**

- The decision a hitter makes is **gear up vs stay back**, and **where** —
  not which of 10 pitch types. The binary fastball-family call is the
  product surface; the 10-class argmax is supporting detail.
- The headline metric is the **commit slice**: of the pitches where the
  read is strong enough to act on, how many are there and how right are we?
- **The bar is edge over the pitcher's own base rate**, not raw accuracy.
  A model at 77% where the scouting report already gets 78% has negative
  value. `analyze_actionability.py` enforces this: >=10% of pitches
  advised, >=70% accuracy, >=+2pts edge. It was validated against the
  synthetic dataset (where pitch types are drawn independently of context)
  and correctly returns NOT ACTIONABLE there, diagnosing "77.2% accuracy
  but the base rate already gets 78.4%".

**Known gap found during the reframe:** `build_pitch_features` puts the
type, velocity, location and movement of the previous 8 pitches into the
sequence but **not their outcomes**. `description` is used only for the
batter's season-long whiff profile. The model cannot distinguish "just
swung through a slider" from "fouled off three fastballs" — a first-order
sequencing signal. Adding a per-timestep outcome one-hot (ball / called
strike / swinging strike / foul / in-play) is the highest-prior feature
change available, and the next thing to try if the gate says the current
model has no edge.

## Gate result (2026-09-23): ACTIONABLE

`analyze_actionability.py` on the run-7 model:

> At a **65% commit threshold** the model advises on **31.0% of pitches**,
> is right **75.6%** of the time there, **+5.2 points** better than the
> pitcher's own base rate.

That clears all three bars. The model is good enough to build on; the
remaining work is product, not modeling. `build_commit_cards.py` turns
this into the deliverable — per-pitcher, per-count "sit hard / sit soft"
rules, mined from held-out predictions only.

Two correctness rules found while building it, both caught by running
against the synthetic negative control:

1. **A card must name the likeliest pitch within the family it recommends.**
   The first version reported the global argmax, producing "SIT SOFT /
   likeliest FF" — self-contradictory advice a hitter cannot act on.
2. **The baseline must be the count-split scouting report**, not the
   pitcher's overall mix. Every advance scout knows he goes fastball 3-1.
   Against the overall rate the cards showed large fake edges; against the
   count-split rate the synthetic model's edges correctly collapse to +0%.

A rule additionally requires `MIN_CONSISTENCY` (70% of the confident calls
in that cell agreeing). Without it a single "sit" label can be pasted over
a situation where the model is confidently split, which is worse than
silence.

## Full gate report + first card run (2026-09-23)

**Where the edge actually is.** Section 4 originally compared the model to
the pitcher's *overall* hard/soft rate. Against the bar that matters —
"just sit whichever side is commoner in this count", which every hitter
already knows — the picture inverts:

| count | model | naive | edge |
|---|---|---|---|
| 2-2 | 59.5% | 50.5% | **+9.0** |
| 0-1 | 58.6% | 51.1% | **+7.5** |
| 1-1 | 58.6% | 51.4% | **+7.2** |
| 3-1 | 78.9% | 78.1% | +0.8 |
| 3-0 | 94.5% | 95.1% | **-0.6** |

The model adds nothing on 3-0 (everyone knows it is a fastball) and 7-9
points in the ambiguous counts where the hitter is genuinely guessing.
That is the ideal shape for a decision aid, and it is the opposite of what
the pitcher-overall baseline suggested.

**Calibration is badly off.** The model is systematically UNDER-confident —
says 64%, right 84%; says 74%, right 93%. This is the known signature of
focal loss, and it silently suppresses coverage by pushing pitches the
model knows below the commit threshold. Fixed with one-parameter
temperature scaling (`fit_temperature` / `apply_temperature` in
`evaluate_model.py`), fitted on half the validation rows and applied to
all. No retraining.

**The first card run returned zero rules, and that was a bug, not a
finding.** `MIN_N = 120` required 120 held-out pitches per
(pitcher x count x batter-hand) cell, but validation is 20% of the data,
so the average cell holds ~9 pitches and even a workhorse starter's
largest cell lands near 150 — of which only ~31% clear the confidence
threshold. No cell could ever qualify. Replaced with:

- `MIN_SPOKE = 25` floor plus a **one-sided binomial test** (z >= 1.645)
  against the count-split scouting report, so a small cell can still earn
  a rule when the effect is large and a big cell cannot coast on noise.
- Batter-hand split is now **off by default** (`--by-hand` to enable); it
  thins every cell 3x and only survives for the highest-volume starters.

## First real card run (2026-09-23) — and the selection bug it exposed

Temperature calibration on real data fitted **T = 0.800** (sharpened),
confirming the under-confidence the calibration table showed.

The run produced 7 rules, but every one reported accuracies summing to
exactly 100% (82/18, 76/24, 72/28, ...). That was a **selection
artifact**, not signal: the rule was scored only on pitches where the
model was confident, so the model was choosing the very sample its
baseline got judged on. A hitter in the box cannot know which pitches
those are — the card tells him to sit on *every* pitch in the count.

**Fix:** a rule is now scored over the entire (pitcher, count) cell.
Consequences:

- A card only prints when its call **contradicts** the count-split
  scouting report. If the two agree, the hitter already had it and the
  card adds nothing.
- Because the card and the report are then opposing constant calls on the
  same pitches, their accuracies necessarily sum to 100%. That is
  arithmetic; the card's claim is that the report is on the wrong side.
- Significance is now a test against a coin flip (is the card's side
  really the majority?), over the full cell, with `MIN_CELL = 60`.
- The confident-subset accuracy is retained as a separate `(conf)` column
  — meaningful for a live tool, not for a memorizable card.

Also fixed: `MIN_N` survived in the JSON payload after the rename and
crashed the run after the text report had been written; and the
count-level index maps for training and validation are now the same
object, since two independently derived maps could silently index
different counts.
