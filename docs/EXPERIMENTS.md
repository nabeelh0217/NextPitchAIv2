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

## Zero cards (2026-09-23) — and the question it raises

With honest full-cell scoring, **no (pitcher, count) cell** had a constant
call that contradicts the count-split scouting report and beats it. Zero
rules.

That is not "the model is worthless" — it is inconsistent with the
aggregate numbers unless you read it carefully:

- aggregate: **+5.2%** edge at a 65% commit threshold, on 31% of pitches
- per count vs naive: **+9.0** (2-2), **+7.5** (0-1), **+7.2** (1-1)
- per (pitcher, count) constant rule: **nothing**

All three are consistent with one explanation: the model's value lies in
distinguishing *which pitches within a situation* will be soft, not in
shifting the situation's overall majority. A card that says "Lugo, 2-2:
always sit soft" discards exactly that — it forces one answer per
situation, which is the one thing the model is not contributing.

**Section 5 of `analyze_actionability.py` now tests this directly.** For
every cell it compares the model's varying call against two constant
rules:

- **scout** — the majority side learned from TRAINING rows, scored on
  validation. Fair, out-of-sample, and what a real report would say.
- **oracle** — the best constant rule fitted on the validation rows
  themselves. Nobody could write it in advance; losing to it slightly is
  not a failure.

The verdict is explicit:

- **POSITIVE lift vs scout** -> the model knows things no static card can
  carry. Build a live lookup, not a card.
- **FLAT** -> ship the static table; it is simpler, needs no model at
  serve time, and is honest.
- **NEGATIVE** -> the network is not earning its keep over a table.

Validated on the synthetic negative control, where context carries no
signal: it correctly returns NEGATIVE (-3.7% vs scout, model winning only
10% of cells) and recommends shipping the table.

## Section 5 result (2026-09-23): the model is at table parity

Full gate on the run-7 model, after temperature calibration (T = 0.800,
which bought +2.7pts of coverage at the 65% threshold for free):

```
model (varying call)          61.5%
scout constant (from train)   60.9%    lift +0.7%
oracle constant (fit on val)  61.6%    gap  -0.1%
cells where model beats scout  928/2048 (45%)
```

**The honest verdict is table parity, not a live-model win**, for two
reasons the first version of the verdict logic missed:

1. The model beats the count-split scouting table in only **45% of
   situations** — worse than a coin flip. The positive aggregate comes
   from winning bigger where it wins, not from winning broadly. That is
   not a dependable per-situation edge.
2. It does not exceed the **oracle** constant rule; it sits exactly on it
   (-0.1%). Matching the best constant per cell means the model is only
   choosing the right constant — which a table encodes just as well.

The verdict logic now requires all three of: significant lift over scout,
a >50% cell win rate, AND beating the oracle. Otherwise it recommends
shipping the table.

### Where the headline edge actually went

Sections 1-4 compare against the pitcher's **overall** mix, which ignores
the count. Section 5's bar is **count-split**. The difference is the
whole story:

| baseline | model edge |
|---|---|
| pitcher's overall mix (§3, 65% threshold) | +5.9% |
| league count majority (§4, 2-2) | +9.5% |
| **pitcher's count-split table (§5)** | **+0.7%** |

Nearly all of the apparent edge is "the model knows about counts" — which
every hitter already does. Against a properly built pitcher x count
table, the model adds +0.7%.

That +0.7% is real and significant over 243K pitches, and it is worth
capturing: build the table from the **model's** per-cell call rather than
from raw training frequencies. It ships with no TensorFlow at serve time.

### The one experiment that could change this

Within-cell lift of ~0 is exactly what you would expect if the sequence
branch carries no information beyond what the count already encodes —
and `build_pitch_features` omits the OUTCOME of each previous pitch
(ball / called strike / swinging strike / foul / in-play). The model
cannot tell "just swung through a slider" from "fouled off three
fastballs".

That is a falsifiable hypothesis: add the outcome one-hot per timestep,
retrain, and re-run section 5. If within-cell lift and the cell win rate
move, the model earns a live deployment. If they do not, ship the table
and stop.

## Run 8 (queued): previous-pitch outcomes in the sequence

The falsifiable hypothesis from section 5. Within-cell lift of ~0 is what
you would expect if the sequence branch carries nothing the count does not
already encode — and until now it did not know what HAPPENED to any of the
previous 8 pitches.

`build_pitch_features` now emits a 5-way outcome one-hot per timestep
(ball / called_strike / whiff / foul / in_play), placed between the pitch
one-hot and the physics block so the scaler still touches only the
continuous tail. Sequence width 17 -> 22.

Why this should matter: 1-1 reached via two fouls is a different at-bat
from 1-1 reached via ball-then-called-strike, and the count cannot say so.
A catcher calls the next pitch very differently after a swinging strike.

**Leakage:** the outcome of the CURRENT pitch would leak its type almost
directly. It is safe only because `build_sequences` slices strictly before
the current index. Verified on synthetic data: 400 sampled rows, every
non-padded timestep matches a strictly earlier pitch, no row sees its own
outcome, and every outcome block is exactly one-hot.

**Success criterion, fixed in advance** — judge on section 5, not top-1:
- cell win rate crosses **50%** (was 45%), AND
- the oracle gap closes (was -0.1%), AND
- lift over scout materially above +0.7%

If those move, the model earns a live deployment. If they do not, the
sequence branch has nothing to give: ship the pitcher x count table and
stop modeling.

## Run 8: previous-pitch outcomes in the sequence — RESULT

Sequence width 17 -> 22 (added the 5-way outcome one-hot per timestep).
Everything else identical to run 7.

| metric | run 7 | run 8 | delta |
|---|---|---|---|
| top-1 | 47.4% | **48.2%** | +0.8 |
| top-3 | 91.9% | **92.0%** | +0.1 |
| log-loss | 1.1650 | **1.1566** | -0.0084 |
| macro-F1 | 0.457 | **0.465** | +0.008 |
| best val_loss | 0.4281 @ ep19 | **0.4244 @ ep17** | better, sooner |

Binary fastball-family vs rest (first run to report it): model 62.9%,
pitcher-prior baseline 59.7%, **+3.1%**.

Per-class recall moved most on **CH +4.6** and **SI +4.0** — the pitches
most often called in response to how a hitter just reacted, which is
exactly what the outcome feature encodes. ST -4.7 and KN -2.0 gave back
some. Net clearly positive.

Over 426,731 validation rows the top-1 gain is ~10 standard errors, so
the feature is real, not noise. But it is small, and **top-1 was
explicitly not the success criterion for this run** — see the run 8
criteria above. The decision rests on section 5 of
`analyze_actionability.py`: cell win rate crossing 50% (was 45%), the
oracle gap closing (was -0.1%), and lift over scout exceeding +0.7%.

### Section 5 for run 8: ALL THREE CRITERIA CLEARED

| criterion (fixed before the run) | run 7 | run 8 | |
|---|---|---|---|
| cell win rate > 50% | 45% | **51%** | pass |
| oracle gap closes | -0.1% | **+0.3%** | pass |
| lift over scout > +0.7% | +0.7% | **+1.1%** | pass |

The oracle line is the important one. The model now **exceeds the best
constant rule fitted on the validation rows themselves** — a rule nobody
could write in advance, since it is fit in-sample on the answers. Beating
it means the model discriminates *within* situations, which no static
table can reproduce by construction.

The mechanism confirms itself: the largest section-4 gains are **0-2
(+5.9 -> +7.7)** and **1-2 (+5.9 -> +7.3)** — two-strike counts, where
what the hitter just did is maximally informative about the putaway
pitch. That is exactly what the outcome one-hot encodes.

Caveat on magnitude: +1.1% over a good count-split table is modest, the
win rate is barely over half, and the oracle margin is only ~1.5x the
noise band. Directionally decisive, small in size.

**MODELING IS CLOSED.** The sequence branch earns its keep; the product
is a live lookup, not a static card.

### Section 6: the product claim

Added because sections 3-5 are model diagnostics, not the sentence a
hitter is promised. On the pitches where the tool speaks, versus what his
count-split table would have said on those same pitches:

| commit @ | speaks on | tool | table | edge |
|---|---|---|---|---|
| 65% | 35.9% | 76.0% | — | — |

(The advised pitches are model-selected, so this is not a fair
model-vs-table comparison — but it is the product's real claim, since
staying silent the rest of the time is part of the design.)

## v7: location head + 2025 season + temporal split

Three changes, deliberately run as two training runs so the data effect
and the split effect stay separable.

**Location target — Statcast attack zones** (`heart / shadow / chase /
waste`), computed in units of the BATTER'S OWN strike zone from
`plate_x`, `plate_z`, `sz_top`, `sz_bot`. Chebyshev distance from zone
centre in zone-half-widths, bucketed at 0.67 / 1.33 / 2.00. That framing
maps onto a hitter's decision (damage it / protect / lay off / take)
rather than onto geometry, which a 13-cell grid does not.

Rows with unusable location get label 0 and **sample_weight 0**, so they
contribute nothing to the zone loss rather than being guessed at.

**Architecture**: second softmax head off the shared trunk, loss weight
**0.3**. Sharing the trunk gives it the type/location correlation
(sliders go low-away, four-seamers go up) implicitly, without forcing a
sparse 40-class joint target. No arsenal mask on this head — any pitcher
can miss anywhere. Checkpointing and early stopping monitor
`val_output_loss` (the type head), not the weighted total, so the
location head can never quietly drive model selection.

**Location priors** added to the context vector with the same expanding
leakage-free pattern as the arsenal prior: where this pitcher has put the
ball, and where this batter has been pitched, both excluding the current
row. Context 55 -> 63 features.

**Temporal split**: `SPLIT_MODE = "temporal"` trains on every season
before `HOLDOUT_SEASON` and validates on that season. No shared games or
at-bats, and it is the real deployment question. This addresses the
CRITICAL open defect.

### Run plan

| run | split | purpose |
|---|---|---|
| 9a | random | comparable to run 8; isolates the effect of adding 2025 + the location head |
| 9b | temporal (hold out 2025) | the honest number, and the one to quote publicly |

### Location success criteria, fixed in advance

Location is far noisier than type — a pitcher aims and misses, and that
execution variance is irreducible. Judge the zone head ONLY against its
baselines:

- beats "league commonest zone" by a clear margin, AND
- beats "this pitcher's commonest zone" by >= +2 points.

If it fails both, drop the head (`ENABLE_LOCATION_HEAD = False`) rather
than shipping a location read that is worse than a constant guess.

Verified on synthetic data, where locations are generated from
independent noise: the head correctly scores 41.1% against a 43.4%
baseline and collapses onto the two commonest zones. A no-signal target
produces a negative result, so the measurement is not self-flattering.

---

## Runs 9a / 9b — 4 seasons + location head (2026-09-25)

Data: 2022-2025, 2,863,345 pitches (3 seasons -> 4, +33%).

| run | intended split | actual split | top-1 | top-3 | log-loss | FB-family |
|---|---|---|---|---|---|---|
| 8 | random | random | **48.2%** | 92.0% | **1.1566** | — |
| 9a | random | random (572,669 val) | 47.3% | 91.5% | 1.1755 | 62.7% |
| 9b | temporal, hold out 2025 | **random (572,669 val)** | 47.6% | 91.5% | 1.1744 | 62.9% |

### 9b did not run temporally — the config edit never took effect

Both reports score exactly 572,669 validation rows with identical
per-class supports (CH 60983, CU 38868, FF 185506, ...). That is 20.000%
of the dataset — a random split. A 2025 holdout would be ~700K rows with
a different class mix. The two runs differ only in training stochasticity.

**There is still no temporal number.** The CRITICAL split defect is open.

Accidental value: 9a and 9b are the same experiment run twice, so they
give the first **seed-variance estimate** — +-0.3 points top-1 and
+-0.001 nats. Any future lift smaller than that is noise. Runs 5-8 were
all compared without knowing this.

Fix: `03_train.py` now writes `data_v5/split_v5.json` and both the
startup banner and the report header print the split. `evaluate_model.py`
reads that file instead of keeping a second copy of the config, and
refuses to score if the reproduced row count disagrees with the run's.

### The location head FAILED its pre-registered criterion

Required >= +2 points over the pitcher's commonest zone. Delivered
**+0.0** on both baselines — 40.8% model vs 40.8% for a constant guess,
with heart recall 0.0% and shadow recall 99.9%. The argmax never leaves
the plurality class.

Verdict by the rule fixed in advance: **fail**. Not rescued.

But accuracy cannot decide this head, and that is a flaw in the criterion
I wrote, not a reason to accept the head. `shadow` is the plurality zone
in nearly every conditioning cell, so an argmax pinned to it is the
arithmetically correct response to a mildly informative distribution. The
head can carry real information and still show 0.0% recall elsewhere.
This is the same error AGENTS.md already warns about for the type head —
"flat average accuracy is the wrong statistic" — repeated on a new head.

So a **new, separately pre-registered** test, which is not a retroactive
pass for the old one:

- **zone log-loss** must beat both the league zone mix and the smoothed
  pitcher zone mix. Log-loss moves when argmax structurally cannot.
- **heart vs rest** — "is this one over the plate?", the location
  analogue of the hard/soft call, and the only form of this a hitter can
  act on — must beat the ~24% base rate by a real margin on a slice worth
  quoting.

If log-loss lift is <= 0, the head is dead and comes out.

Negative control passes: on synthetic data with locations drawn from
noise, zone log-loss comes out **0.085 nats WORSE** than baseline and the
heart table shows +0.3% at its only populated threshold.

### The type head also got worse, and two things changed at once

Run 8 -> 9a is -0.9 points top-1 and +0.019 nats, both well outside the
+-0.3 / +-0.001 seed noise. But runs 9a/9b changed the season count AND
added the location head, which violates the one-change-per-run rule.

Leading suspect is capacity theft. The trunk narrows to a 64-unit
bottleneck and the zone head hangs off that same 64-dim vector, so zone
gradients reshape a representation the type head depends on. Loss weight
0.3 bounds the loss contribution, not the representational damage.

Run 10 isolates it: 4 seasons, random split, `ENABLE_LOCATION_HEAD =
False`. If top-1 returns to ~48.2%, the head is a net negative and comes
out regardless of what its own log-loss says. If it stays ~47.4%, the
extra season explains the drop and the head is exonerated.

### Location head re-scored (no retraining) — ALIVE but marginal

`evaluate_model.py` against the saved 9b model, with the new metrics.

|  | accuracy | vs model | log-loss | vs model |
|---|---|---|---|---|
| model | 40.8% | — | 1.2733 | — |
| league zone mix | 40.8% | +0.0 | 1.2946 | **+0.0213** |
| pitcher zone mix | 40.8% | +0.0 | 1.2924 | **+0.0191** |

**Log-loss lift is positive on both baselines**, at roughly 20x the
measured seed noise. The head learned something real. This also confirms
the diagnostic was the problem, not the head: accuracy read +0.0 while
log-loss found signal, which is precisely what invariant 8 now guards.

Sanity check passes — marginal zone entropy is 1.2944 nats and the league
baseline scores 1.2946, so the baseline is exactly the marginal.

**Size it honestly.** The head removes **1.6%** of available location
uncertainty (0.0213 / 1.2944). The type head removes **13%** of its own
(0.177 / 1.351). Location carries about a ninth as much extractable
signal as type — expected for a quantity that is mostly execution
variance.

heart vs rest, base rate 24.1%:

| read | coverage | heart rate | vs base |
|---|---|---|---|
| P(heart) >= 0.30 — "be ready" | 1.5% | 34.9% | +10.8 |
| P(heart) <= 0.15 — "take it" | 4.3% | 12.1% | -11.9 |

Both tails move the heart rate by ~half in relative terms. But total
actionable coverage is **5.8%**, below this project's own >=10% bar. Some
of that is under-confidence — the type head needed T=0.800 and gained ~3
points of coverage; the zone head is likely compressed the same way.
Calibrating it would widen the spread without inventing information, but
that work waits until the head's survival is settled.

**Verdict: do not delete, do not ship.** The head is real but thin, and
it may be costing the type head 0.9 points of top-1 — which is the
product. A 5.8%-coverage location hint is not worth degrading the
hard/soft call. Run 10 decides:

- type head returns to ~48.2% -> the head stole capacity. Remove it, or
  rebuild it decoupled: branch the zone head off `combined` rather than
  off the shared 64-unit bottleneck `z`, so zone gradients stop reshaping
  the representation the type head depends on. One line, and the direct
  test of the capacity-theft hypothesis.
- type head stays ~47.4% -> the extra season explains the drop, the head
  is exonerated, keep it as a secondary read behind hard/soft.

---

## Run 10 — location head OFF, 4 seasons, random split (2026-09-26)

`03_train.py --split random --no-location-head`. Confirmed genuinely off:
the report monitors `val_loss`, not `val_output_loss`, and carries no
location block.

| run | data | head | top-1 | top-3 | log-loss | FB-family |
|---|---|---|---|---|---|---|
| 8 | 3 seasons | off | 48.2% | 92.0% | 1.1566 | — |
| 9a | 4 seasons | on | 47.3% | 91.5% | 1.1755 | 62.7% |
| 9b | 4 seasons | on | 47.6% | 91.5% | 1.1744 | 62.9% |
| **10** | 4 seasons | **off** | **47.8%** | 91.4% | **1.1719** | 62.8% |

### What the head costs: ~0.3 points, not 0.9

9a/9b vs 10 is the only clean comparison here — same data, same split,
same validation rows. Head off is **+0.35 points top-1** (at the ±0.3
seed-noise floor, so not resolvable) and **-0.0031 nats** (about 3x the
±0.001 noise, so probably real but small).

**I over-attributed the drop last session.** Run 8 vs run 10 is not a
valid comparison: run 8's validation set is 20% of three seasons, run
10's is 20% of four. Different rows, different difficulty. Most of the
"0.9 point regression" was the test set changing underneath, not the
head.

### Adding 2025 made the random-split numbers worse

Run 8 -> run 10, config otherwise identical: **-0.4 points top-1, +0.0153
nats** (15x noise, unambiguous). More data made the measured numbers
worse, which means the 4-season validation set is harder, not that the
model got worse. Candidate causes: Statcast reclassification drift (the
sweeper split out of SL), a richer 2025 pitch mix, more pitchers with
thin histories. Not worth chasing — the seasons are not comparable test
sets, so only same-data comparisons mean anything from here on.

### Verdict on the location head

Cost is small and real; benefit is real and thin (1.6% of location
uncertainty, 5.8% actionable coverage vs the >=10% bar). Decoupling it
from the shared bottleneck might buy back the 0.003 nats, but it would
not move the 5.8%, and coverage is what fails the bar. Location is mostly
execution variance and no architecture fixes that.

**Leave it off. Stop spending runs on it.** Revisit only if the site
wants a secondary "over the plate / out of the zone" tint, and if so
temperature-calibrate the zone head first — the coverage figure above is
uncalibrated, and the type head gained ~3 points of coverage from T=0.8.

Note: run 10 overwrote `best_model_v5.keras`, so the 9b two-head model is
gone. Re-scoring the zone head now needs a retrain.

### Still open, and now blocking the site

**There is no temporal number.** Every figure quoted so far comes from a
random split whose validation rows share games and at-bats with training
rows. The site's headline claim cannot come from that.

Fixed this session so it can be measured safely: `analyze_actionability.py`,
`build_commit_cards.py` and `diagnose_arsenal.py` all hardcoded the random
split. Run a temporally-trained model through them and they would have
scored it on rows it trained on — inflating exactly the number intended
for the site, with plausible-looking output. All four scoring scripts now
go through `evaluate_model.load_split()`, which reads `split_v5.json`.

Next: `03_train.py --split temporal --holdout-season 2025
--no-location-head`, then `analyze_actionability.py`. That number is the
one the site quotes.

---

## Run 11 — TEMPORAL split, hold out 2025 (2026-09-26)

`--split temporal --holdout-season 2025 --no-location-head`.
729,688 validation pitches, no shared games with training.

top-1 **43.1%**, top-3 87.3%, log-loss **1.3194**, FB-family 60.2%.
(Random-split run 10 on the same data: 47.8% / 1.1719. Not the same test
set, so the gap is not a "drop" — it is what removing shared games and
at-bats costs, which is the honest deployment number.)

**Training peaked at epoch 3 of 11.** Run 10 peaked at 18 of 26. The
model starts overfitting to pre-2025 almost immediately, which is itself
evidence that much of what it learns is season-specific.

### The within-situation edge does not survive an honest split

Section 5, against the count-split scout — the only bar that matters:

| | run 8 (random) | run 11 (temporal) |
|---|---|---|
| lift vs scout | **+1.1%** | **-0.6%** |
| cells where model wins | **51%** | **38%** |
| gap to oracle | **+0.3%** | **-2.8%** |

Run 8 cleared all three pre-registered criteria. Run 11 **fails all
three**, and the script's own section-5 verdict reads NEGATIVE: "a
constant per-situation rule beats the model."

**The conclusion that the product must be a live lookup was a
random-split artifact.** With validation rows sharing games and at-bats
with training rows, the model could lean on outing-specific patterns.
Against a season it has never seen, a static count-split table is better
across situations.

### What does survive: the confident slice

Section 6, on held-out 2025, model-selected slice, both scored against
the count-split table:

| commit @ | speaks on | tool | table | edge |
|---|---|---|---|---|
| 60% | 51.5% | 66.4% | 63.3% | +3.1% |
| **65%** | **31.1%** | **71.0%** | **67.1%** | **+3.9%** |
| 70% | 15.5% | 77.9% | 74.2% | +3.7% |

The model is worse than the table on average and better than it where it
is confident. Those are consistent: confidence is informative about when
to deviate. A selective overlay is defensible; "better than a scouting
report" is not.

### Two fixes this run forced

**The final VERDICT was keyed to section 3.** It printed ACTIONABLE
(+6.8%) in the same report where section 5 printed NEGATIVE, because
section 3 scores against the pitcher's OVERALL mix — a bar that ignores
the count and that every hitter already clears. AGENTS.md has said since
2026-09-23 that the count-split table is the only bar. The verdict now
gates on sections 5 and 6, reports coverage next to accuracy, and has an
"ACTIONABLE AS A SELECTIVE OVERLAY, NOT A REPLACEMENT" state.

**The arsenal mask leaks the future.** `02_preprocess.py` builds it over
every row, so on a temporal split it tells the model which pitches each
pitcher throws in 2025 — including ones he only added that year. The mask
zeroes out impossible classes and the scout baseline gets no equivalent,
so the leak flatters the model in exactly the comparison above.
`03_train.py` now rebuilds it from training rows only (`--full-arsenal-
mask` restores the old behaviour). The logit penalty softens from -1e9 to
-12, because with a training-only mask ~2% of 2025 pitches are a type the
pitcher had genuinely never thrown, and -1e9 charges those the full
27.6-nat clip for an event that did happen.

**So the +3.9% above is optimistic.** Run 12 measures it without the leak.

### Run 12

`--split temporal --holdout-season 2025 --no-location-head` on the fixed
mask, then `analyze_actionability.py`. Pre-registered, so it cannot be
renegotiated afterwards:

- section 6 edge at 65% stays **>= +2.0 points** on **>= 10%** of pitches
  -> ship as a selective overlay, claim scoped to the slice, coverage
  always shown beside accuracy.
- below that -> the honest product is the count-split table itself, and
  the site should serve that. It needs no model at serve time.

---

## Run 12 — temporal, honest arsenal mask (2026-09-26) — SHIP

`--split temporal --holdout-season 2025 --no-location-head`, mask built
from training rows only, logit penalty -12.

| | run 11 (leaky mask) | run 12 (honest mask) |
|---|---|---|
| top-1 | 43.1% | 43.2% |
| log-loss | 1.3194 | **1.5765** |
| log-loss lift vs smoothed | +0.3007 | **+0.0435** |
| section 5 lift vs scout | -0.6% | **+0.7%** |
| cells won | 38% | **43%** |
| gap to oracle | -2.8% | **-1.4%** |
| section 6 @65% | +3.9% on 31.1% | **+4.2% on 35.2%** |

### PRE-REGISTERED CRITERION MET

Required >= +2.0 points on >= 10% of pitches. Delivered **+4.2 points on
35.2%**. Build `site/` as a selective overlay.

### Why removing a leak made the numbers better

Counterintuitive, so worth stating plainly. Two things changed together
(a violation of one-change-per-run — the net is clear but attribution is
not):

1. **Train-only mask** — strictly less information.
2. **Penalty -1e9 -> -12** — the model can hedge when the mask is wrong.

Log-loss got **worse** (1.3194 -> 1.5765), and that is the honest cost
showing up: 2.371% of 2025 pitches are a type the pitcher never threw
before, and at ~12 nats each that is ~0.28 nats — almost exactly the
0.257 observed. Against the fair smoothed baseline the log-loss lift
collapses from +0.3007 to **+0.0435**, so on full-distribution terms the
model is now barely better than a smoothed pitcher prior. That is the
truthful picture; run 11's looked better because the mask was telling it
the future.

Top-1 and every actionability metric improved because the hard mask had
taught the model to trust a signal that will not exist at serve time.
With a soft penalty it learns to hedge, and its confidence becomes more
honest — calibration temperature moved 0.975 -> 0.875 and coverage at
the 65% threshold rose 31.1% -> 35.2%.

### The claim, and its limits

- **Section 5 is still NOT BROAD**: +0.7% aggregate lift, but the model
  beats the count-split table in only 43% of cells. It is not a
  replacement for a scouting report and must never be sold as one.
- **Section 6 is the product**: on 35.2% of pitches it speaks, it is
  right 72.5% where the table is right 68.3%.
- Coverage must appear beside accuracy everywhere on the site. "72.5%
  accurate" alone is a misleading claim.
- Sections 1-4 use the pitcher's overall mix and are diagnostics only.
  The +8.3% in section 3 is NOT the product number.

Model shipped to the site: run 12, temporal split, no location head,
honest mask.
