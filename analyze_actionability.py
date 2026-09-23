"""
NextPitchAI — is this model actually usable by a hitter?
=========================================================
Average accuracy is the wrong question. A hitter has ~400ms and decides
"gear up or stay back". He never acts on a 10-way distribution, and he
never sits on a 50/50 read. What matters is:

    In what fraction of pitches is the read strong enough to COMMIT,
    and how right are we in exactly those pitches?

A model that is 47% overall but 80% inside a confident 15% is a real
product. A model that is 47% everywhere is not. This measures which one
we have. No retraining — it scores the saved model's existing
probabilities.

Usage:
    .venv/bin/python analyze_actionability.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate_model import (
    DATA_DIR, SPLIT_SEED, SPLIT_TEST_SIZE, FASTBALL_FAMILY,
    PRIOR_SMOOTHING, _smoothed, fit_temperature, apply_temperature,
)

REPORT_PATH = DATA_DIR / "actionability_report_v5.txt"

# A hitter only benefits if committing beats his default. Below this the
# advice is noise and the correct output is "react".
COMMIT_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)

# The bar for "this is worth shipping to a hitter", all three required:
MIN_SHARE = 0.10   # speaks often enough to matter
MIN_ACC = 0.70     # right often enough to commit the swing
MIN_EDGE = 0.02    # and BEATS the scouting report he already has.
# The edge is the load-bearing one. High accuracy on 3-0 counts is
# worthless if the base rate already gives it to him for free.
CONF_BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def hdr(out, title):
    out("")
    out("=" * 72)
    out(title)
    out("=" * 72)


def main():
    import tensorflow as tf

    lines = []
    out = lines.append

    with open(DATA_DIR / "meta_v5.json") as f:
        meta = json.load(f)
    classes = meta["pitch_classes"]
    K = len(classes)
    fam = np.array([1 if c in FASTBALL_FAMILY else 0 for c in classes])

    print("Loading arrays...")
    X = {k: np.load(DATA_DIR / f"X_{k}.npy") for k in
         ["seq", "ctx", "pitcher_id", "batter_id", "pitcher_hand",
          "batter_hand", "park_id", "catcher_id", "arsenal_mask"]}
    y = np.load(DATA_DIR / "y_labels.npy")

    idx = np.arange(len(y))
    idx_tr, idx_val = train_test_split(
        idx, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, stratify=y)

    print("Loading model and predicting...")
    model = tf.keras.models.load_model(
        DATA_DIR / "best_model_v5.keras", compile=False)
    probs = model.predict(
        [X[k][idx_val] for k in
         ["seq", "ctx", "pitcher_id", "batter_id", "pitcher_hand",
          "batter_hand", "park_id", "catcher_id", "arsenal_mask"]],
        batch_size=2048, verbose=0)

    yv = y[idx_val]

    # Focal loss leaves the model under-confident, which pushes pitches it
    # actually knows below the commit threshold. Fit T on the first half,
    # apply to all; one parameter, but keep the split honest anyway.
    raw_probs = probs
    half = len(yv) // 2
    temp, _ = fit_temperature(probs[:half], yv[:half])
    probs = apply_temperature(probs, temp)

    pred = probs.argmax(1)
    conf = probs.max(1)
    N = len(yv)

    # Binary "gear up vs stay back" — the decision a hitter actually makes.
    p_hard = probs[:, fam == 1].sum(1)
    hard_true = fam[yv]
    hard_pred = (p_hard >= 0.5).astype(int)
    hard_conf = np.maximum(p_hard, 1 - p_hard)

    # Pitcher's own base rate, from training rows only — what a scouting
    # report already tells the hitter. Any value we add must beat this.
    n_pid = int(max(X["pitcher_id"].max(), 0)) + 1
    counts = np.zeros((n_pid, K))
    np.add.at(counts, (X["pitcher_id"][idx_tr], y[idx_tr]), 1.0)
    league = np.bincount(y[idx_tr], minlength=K).astype(float)
    league /= league.sum()
    totals = counts.sum(1, keepdims=True)
    prior = _smoothed(counts, totals, league)[X["pitcher_id"][idx_val]]
    prior_pred = prior.argmax(1)
    prior_hard = (prior[:, fam == 1].sum(1) >= 0.5).astype(int)

    hdr(out, "0. CALIBRATION FIX")
    out(f"Fitted temperature T = {temp:.3f} "
        f"({'sharpened' if temp < 1 else 'softened'}).")
    raw_hard = raw_probs[:, fam == 1].sum(1)
    raw_conf = np.maximum(raw_hard, 1 - raw_hard)
    cal_hard = probs[:, fam == 1].sum(1)
    cal_conf = np.maximum(cal_hard, 1 - cal_hard)
    out("")
    out(f"{'commit @':>10}{'raw':>12}{'calibrated':>14}{'change':>10}")
    for t in COMMIT_THRESHOLDS:
        out(f"{t:>10.0%}{(raw_conf >= t).mean():>12.1%}"
            f"{(cal_conf >= t).mean():>14.1%}"
            f"{(cal_conf >= t).mean() - (raw_conf >= t).mean():>+10.1%}")
    out("")
    out("More pitches clearing the bar means more situations where the")
    out("hitter gets advice — for free, with no retraining.")

    hdr(out, "1. CONFIDENCE STRATIFICATION — where is the read strong?")
    out("Binned by the model's top probability. 'hard/soft' is the binary")
    out("gear-up decision; that is the one a hitter can actually act on.")
    out("")
    out(f"{'confidence':>12}{'pitches':>10}{'% of all':>10}"
        f"{'10-cls acc':>12}{'hard/soft':>11}{'prior h/s':>11}{'edge':>8}")
    for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        a10 = (pred[m] == yv[m]).mean()
        ahs = (hard_pred[m] == hard_true[m]).mean()
        phs = (prior_hard[m] == hard_true[m]).mean()
        out(f"{f'{lo:.1f}-{hi:.1f}':>12}{m.sum():>10,}{m.sum()/N:>10.1%}"
            f"{a10:>12.1%}{ahs:>11.1%}{phs:>11.1%}{ahs - phs:>+8.1%}")

    hdr(out, "2. CALIBRATION — is the confidence honest?")
    out("A hitter can only act on a number that means what it says.")
    out("'predicted' vs 'actual' should track closely.")
    out("")
    out(f"{'bin':>12}{'n':>10}{'predicted':>12}{'actual':>10}{'gap':>9}")
    for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        out(f"{f'{lo:.1f}-{hi:.1f}':>12}{m.sum():>10,}{conf[m].mean():>12.1%}"
            f"{(pred[m] == yv[m]).mean():>10.1%}"
            f"{(pred[m] == yv[m]).mean() - conf[m].mean():>+9.1%}")

    hdr(out, "3. THE COMMIT SLICE — the headline number")
    out("If we only advise when the binary read clears a threshold, how")
    out("often do we speak, and how right are we when we do?")
    out("")
    out(f"{'threshold':>11}{'% advised':>12}{'accuracy':>11}"
        f"{'prior acc':>11}{'edge':>8}{'verdict':>22}")
    best = None
    for t in COMMIT_THRESHOLDS:
        m = hard_conf >= t
        if m.sum() == 0:
            out(f"{t:>11.0%}{0:>12.1%}{'—':>11}{'—':>11}{'—':>8}")
            continue
        acc = (hard_pred[m] == hard_true[m]).mean()
        pacc = (prior_hard[m] == hard_true[m]).mean()
        share = m.mean()
        edge = acc - pacc
        ok = share >= MIN_SHARE and acc >= MIN_ACC and edge >= MIN_EDGE
        why = "ACTIONABLE" if ok else (
            "no edge vs scouting" if edge < MIN_EDGE else
            "too rare" if share < MIN_SHARE else "not accurate enough")
        # Keep the widest qualifying slice, then the best edge among those.
        if ok and (best is None or edge > best[3]):
            best = (t, share, acc, edge)
        out(f"{t:>11.0%}{share:>12.1%}{acc:>11.1%}{pacc:>11.1%}"
            f"{edge:>+8.1%}{why:>22}")

    hdr(out, "4. COUNT-CONDITIONAL EDGE — where does it live?")
    # balls/strikes are ctx columns 0 and 1, standardized; rank-order is
    # preserved by StandardScaler so unique values recover the levels.
    balls = X["ctx"][idx_val, 0]
    strikes = X["ctx"][idx_val, 1]
    b_lv = {v: i for i, v in enumerate(sorted(np.unique(balls)))}
    s_lv = {v: i for i, v in enumerate(sorted(np.unique(strikes)))}
    out("'naive' = just sit whichever side is commoner in that count —")
    out("what every hitter already knows. THAT is the bar that matters.")
    out("")
    out(f"{'count':>8}{'pitches':>10}{'model':>9}{'pitcher':>9}{'naive':>8}"
        f"{'vs naive':>10}{'%hard':>8}")
    rows = []
    for bv, bi in b_lv.items():
        for sv, si in s_lv.items():
            m = (balls == bv) & (strikes == sv)
            if m.sum() < 500:
                continue
            acc = (hard_pred[m] == hard_true[m]).mean()
            pacc = (prior_hard[m] == hard_true[m]).mean()
            share = hard_true[m].mean()
            naive = max(share, 1 - share)   # always sit the count majority
            rows.append((f"{bi}-{si}", int(m.sum()), acc, pacc, naive,
                         acc - naive, share))
    for r in sorted(rows, key=lambda r: -r[5]):
        out(f"{r[0]:>8}{r[1]:>10,}{r[2]:>9.1%}{r[3]:>9.1%}{r[4]:>8.1%}"
            f"{r[5]:>+10.1%}{r[6]:>8.1%}")
    out("")
    best_counts = [r for r in sorted(rows, key=lambda r: -r[5])[:3]]
    out("Strongest counts vs what the hitter already knows: "
        + ", ".join(f"{r[0]} ({r[5]:+.1%})" for r in best_counts))

    hdr(out, "VERDICT")
    if best:
        t, share, acc, edge = best
        out(f"ACTIONABLE. At a {t:.0%} commit threshold the model advises on")
        out(f"{share:.1%} of pitches and is right {acc:.1%} of the time there,")
        out(f"{edge:+.1%} better than the pitcher's own base rate.")
        out("")
        out("Build the product around this slice: stay silent by default,")
        out("speak only when the read clears the threshold.")
    else:
        # Separate "can't predict" from "predicts fine but adds nothing".
        m70 = hard_conf >= 0.70
        if m70.sum():
            acc70 = (hard_pred[m70] == hard_true[m70]).mean()
            pacc70 = (prior_hard[m70] == hard_true[m70]).mean()
            out(f"NOT ACTIONABLE. At a 70% threshold the model speaks on "
                f"{m70.mean():.1%} of")
            out(f"pitches at {acc70:.1%} accuracy — but the pitcher's own base "
                f"rate already")
            out(f"gets {pacc70:.1%} there, an edge of {acc70 - pacc70:+.1%}.")
            out("")
            if acc70 - pacc70 < MIN_EDGE:
                out("The bottleneck is NOT accuracy — it is that a scouting")
                out("report already tells the hitter everything the model does.")
                out("Game context must add something the base rates do not.")
        else:
            out("NOT ACTIONABLE — the model is never confident enough to "
                "commit.")
        out("")
        out("Pivot options, in order:")
        out("  1. Pre-game scouting summaries (count-conditional tendencies)")
        out("     rather than live per-pitch calls — see section 4 for where")
        out("     the tendencies are actually sharp.")
        out("  2. Add previous-pitch OUTCOMES to the sequence (Step 2 of the")
        out("     plan) and re-check; that signal is currently absent.")
        out("  3. Add a location head — 'fastball up' is a different swing.")

    text = "\n".join(lines)
    print(text)
    REPORT_PATH.write_text(text + "\n")
    print(f"\nWritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
