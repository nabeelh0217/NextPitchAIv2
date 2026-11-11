"""
NextPitchAI — is this model actually usable by a hitter?
=========================================================
Average accuracy is the wrong question. A hitter has ~400ms and decides
"gear up or stay back". He never acts on a 10-way distribution, and he
never sits on a 50/50 read. What matters is:

    In what fraction of pitches is the read strong enough to COMMIT,
    how right are we in exactly those pitches, and is that any better
    than the scouting report he already has?

This is the gate that answers it, and the only source of the numbers the
website shows. No retraining and no TensorFlow: it scores the saved
model with the NumPy forward pass.

What "fair" means here — both changed in 2026-11, see docs/EXPERIMENTS.md:

  * The model is scored with a CAUSAL arsenal mask: only the pitch types
    a pitcher had thrown before each pitch. It used to be handed a mask
    built over every row, which knew what he would add next season.
  * The bar is a LIVE scouting table: count-split (and split by batter
    side), built from every pitch before the one being judged. It used
    to be a table frozen at the end of training, while the model's own
    inputs kept updating through the season being scored.

Both sides now see exactly what was known before the pitch, and nothing
after it.

Usage:
    .venv/bin/python analyze_actionability.py
        Validation only: the season the model was tuned on.

    .venv/bin/python analyze_actionability.py --data-dir data_ext --test-season 2026
        Validation, then a CLEAN TEST. The calibration temperature, the
        commit threshold and the choice of bar are fixed on the
        validation season; the test season is then scored once with all
        three locked. That is the number worth publishing.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_model import (
    DATA_DIR, FASTBALL_FAMILY, SCOUT_PLAIN, SCOUT_RUNGS, apply_temperature,
    build_arsenal_mask, fit_temperature, full_mask_through, known_ids,
    live_scout, load_inputs, load_row_meta, load_scoring_model, load_split,
    predict_rows, read_split, scoring_mask, widest_ci,
)
from pitch_features import dense_codes

# A hitter only benefits if committing beats his default. Below this the
# advice is noise and the correct output is "react".
COMMIT_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)

# The bar for "this is worth shipping to a hitter":
MIN_COVERAGE = 0.10   # a read on fewer pitches than this is not a product
MIN_EDGE = 0.02       # and it BEATS the scouting table by this much there.
# The edge is the load-bearing one. High accuracy on 3-0 counts is
# worthless if the table already gives it to him for free.
CONF_BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

# Minimum held-out pitches in a (pitcher, count) cell before its
# within-situation lift means anything.
MIN_CELL_LIFT = 60

# Resamples for the clustered intervals. Whole games, and whole pitchers,
# are resampled; the wider of the two intervals is reported.
N_BOOT = 2000

# The frozen table's own rule, kept only to reproduce the old number.
OLD_BAR_MIN_TRAIN = 40


def hdr(out, title):
    out("")
    out("=" * 72)
    out(title)
    out("=" * 72)


def old_bar_lean(pid_enc, balls, strikes, hard, idx_tr, rows):
    """
    The bar the gate used before 2026-11, reproduced exactly: a pitcher x
    count table built from TRAINING rows only and never updated, keyed by
    the encoded pitcher id, falling back to the count's own hard rate on
    the rows being judged when a cell is thin.

    Not a fair bar — it is frozen while the model's inputs are not — and
    kept for one purpose: showing what the published number was measured
    against, and how far each correction moves it.
    """
    n_pid = int(pid_enc.max()) + 1
    ch = np.zeros((n_pid, 4, 3))
    ct = np.zeros((n_pid, 4, 3))
    np.add.at(ch, (pid_enc[idx_tr], balls[idx_tr], strikes[idx_tr]), hard[idx_tr])
    np.add.at(ct, (pid_enc[idx_tr], balls[idx_tr], strikes[idx_tr]), 1.0)
    lean = np.zeros(len(rows), dtype=bool)
    p, b, s, h = pid_enc[rows], balls[rows], strikes[rows], hard[rows]
    for bb in range(4):
        for ss in range(3):
            sel = (b == bb) & (s == ss)
            if not sel.any():
                continue
            tt = ct[p[sel], bb, ss]
            rate = np.where(tt >= OLD_BAR_MIN_TRAIN,
                            ch[p[sel], bb, ss] / np.maximum(tt, 1.0), h[sel].mean())
            lean[sel] = rate >= 0.5
    return lean


def binary_read(probs, fam):
    """(P(hard), the call, its confidence) from class probabilities."""
    p_hard = probs[:, fam == 1].sum(1)
    return p_hard, p_hard >= 0.5, np.maximum(p_hard, 1.0 - p_hard)


def slice_rows(conf, tool_ok, tables):
    """One dict per commit threshold: coverage, tool accuracy, and each
    table's accuracy on the SAME advised pitches."""
    rows = []
    for t in COMMIT_THRESHOLDS:
        sel = conf >= t
        if sel.sum() == 0:
            continue
        rows.append({
            "threshold": float(t), "coverage": float(sel.mean()),
            "n": int(sel.sum()), "tool": float(tool_ok[sel].mean()),
            **{name: float(ok[sel].mean()) for name, ok in tables.items()},
        })
    return rows


def cell_summary(pitcher, balls, strikes, hard, tool_ok, bar_ok):
    """
    Model vs bar across every (pitcher, count) situation with enough
    held-out pitches — the "is it better everywhere, or only where it
    speaks" question. Also the best CONSTANT call per cell fitted on these
    very rows, which nobody could write in advance: an upper bound for
    any count card.
    """
    cell = dense_codes(pitcher, balls, strikes)
    g = int(cell.max()) + 1
    n = np.bincount(cell, minlength=g).astype(np.float64)
    hard_n = np.bincount(cell, weights=hard.astype(np.float64), minlength=g)
    tool_n = np.bincount(cell, weights=tool_ok.astype(np.float64), minlength=g)
    bar_n = np.bincount(cell, weights=bar_ok.astype(np.float64), minlength=g)
    keep = n >= MIN_CELL_LIFT
    in_cells = keep[cell]
    if not keep.any():
        return None
    oracle_n = np.maximum(hard_n, n - hard_n)
    tot = n[keep].sum()

    # Per count, over the qualifying cells only.
    first = np.zeros(g, dtype=np.int64)
    first[cell[::-1]] = np.arange(len(cell))[::-1]        # a row of each cell
    cnt = balls[first] * 3 + strikes[first]
    per_count = []
    for c in np.unique(cnt[keep]):
        m = keep & (cnt == c)
        per_count.append({
            "count": f"{int(c) // 3}-{int(c) % 3}", "n": int(n[m].sum()),
            "model": float(tool_n[m].sum() / n[m].sum()),
            "bar": float(bar_n[m].sum() / n[m].sum()),
            "oracle": float(oracle_n[m].sum() / n[m].sum()),
        })
    return {
        "cells": int(keep.sum()), "n": int(tot), "in_cells": in_cells,
        "model": float(tool_n[keep].sum() / tot),
        "bar": float(bar_n[keep].sum() / tot),
        "oracle": float(oracle_n[keep].sum() / tot),
        "cells_won": float((tool_n[keep] / n[keep] > bar_n[keep] / n[keep]).mean()),
        "per_count": per_count,
    }


def main():
    ap = argparse.ArgumentParser(description="The product gate.")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="dataset to score (default: data_v5)")
    ap.add_argument("--test-season", type=int, default=None,
                    help="score this season ONCE with temperature, threshold "
                         "and bar locked on the validation season")
    ap.add_argument("--skip-reproduce", action="store_true",
                    help="skip re-deriving the previously published number "
                         "(saves one prediction pass)")
    args = ap.parse_args()
    data_dir = args.data_dir.resolve()

    lines = []
    out = lines.append

    meta = json.loads((data_dir / "meta_v5.json").read_text())
    classes = meta["pitch_classes"]
    K = len(classes)
    fam = np.array([1 if c in FASTBALL_FAMILY else 0 for c in classes])

    print("Loading arrays...")
    X = load_inputs(data_dir)
    y = np.load(data_dir / "y_labels.npy")
    seasons = np.load(data_dir / "X_season.npy")
    rm = load_row_meta(data_dir)
    if rm is None:
        raise SystemExit(
            f"{data_dir.name}/ has no row_meta_v5.npz, so a live scouting table "
            f"cannot be built for it: the raw pitcher id and integer count per "
            f"row are missing. Extend it without refitting anything:\n\n"
            f"    .venv/bin/python 02_preprocess.py --freeze-from {data_dir.name} "
            f"--out-dir data_ext\n"
            f"    .venv/bin/python analyze_actionability.py --data-dir data_ext\n")
    if meta.get("verified_against_frozen") is False:
        print("WARNING: this dataset did NOT reproduce the one it was extended "
              "from. Treat every number below as unverified.")

    split = read_split(data_dir)
    idx_tr, idx_val, split_desc = load_split(y, data_dir)
    holdout = int(split.get("holdout_season", 0)) if split.get("mode") == "temporal" else None
    if args.test_season is not None:
        if holdout is None:
            raise SystemExit("--test-season needs a model trained with a temporal "
                             "split: on a random split every season is part training data.")
        if args.test_season <= holdout:
            raise SystemExit(f"--test-season must be later than the validation "
                             f"season ({holdout}); {args.test_season} was used to "
                             f"train or tune this model.")
        idx_test = np.flatnonzero(seasons == args.test_season)
        if len(idx_test) == 0:
            raise SystemExit(f"no rows for season {args.test_season} in {data_dir.name}/ — "
                             f"seasons present: {sorted(set(seasons.tolist()))}")
    print(f"Split: {split_desc}   train {len(idx_tr):,}   validation {len(idx_val):,}"
          + (f"   test {len(idx_test):,}" if args.test_season else ""))

    pitcher = rm["pitcher"]
    balls = rm["balls"].astype(np.int64)
    strikes = rm["strikes"].astype(np.int64)
    game = rm["game_pk"]
    side = np.asarray(X["batter_hand"])
    pid_enc = np.asarray(X["pitcher_id"])
    hard = fam[y]

    print("Building the causal arsenal mask and the live scouting tables...")
    mask, mask_desc = scoring_mask(pid_enc, y, idx_tr, K, rm)
    scout = live_scout(pitcher, balls, strikes, side, seasons, hard)
    train_only = build_arsenal_mask(pid_enc, y, idx_tr, K)[pid_enc]

    print("Loading the model (NumPy, no TensorFlow)...")
    model = load_scoring_model(data_dir)
    # Ids with an embedding row but no training rows behind it (anyone who
    # debuted in the held-out season) go to the unknown bucket.
    known = known_ids(X, idx_tr)
    id_cols = {k: np.asarray(X[k]) for k in known}

    def never_trained(rows):
        """Share of `rows` carrying an id the model never trained on."""
        return {k: float(((lut[id_cols[k][rows]] == 0) & (id_cols[k][rows] != 0)).mean())
                for k, lut in known.items()}

    def evaluate(rows, label, temp=None):
        """Score one season. Fits the temperature on its first half unless
        one is passed in, in which case nothing at all is fitted."""
        raw = predict_rows(model, X, rows, mask[rows], label=label, known=known)
        yv = y[rows]
        if temp is None:
            half = len(yv) // 2
            temp, _ = fit_temperature(raw[:half], yv[:half])
        probs = apply_temperature(raw, temp)
        p_hard, call, conf = binary_read(probs, fam)
        ht = hard[rows].astype(bool)
        tables = {name: (scout[name][rows] >= 0.5) == ht for name, _ in SCOUT_RUNGS}
        tables["frozen"] = old_bar_lean(pid_enc, balls, strikes, hard, idx_tr, rows) == ht
        return {
            "rows": rows, "raw": raw, "probs": probs, "temp": float(temp),
            "yv": yv, "hard_true": ht, "call": call, "conf": conf,
            "tool_ok": call == ht, "tables": tables,
            "mix_ok": (scout["pitcher"][rows] >= 0.5) == ht,
            "clusters": [game[rows], pitcher[rows]],
        }

    val = evaluate(idx_val, "validation")

    # -----------------------------------------------------------------
    hdr(out, "0. HOW THIS WAS MEASURED")
    out(f"Model split        : {split_desc}")
    out(f"Validation season  : {len(idx_val):,} pitches"
        + (f"   |   test season {args.test_season}: {len(idx_test):,} pitches"
           if args.test_season else ""))
    out(f"Arsenal mask       : {mask_desc}")
    out("Scouting tables    : live — built only from pitches before the one judged")
    out("")
    lbl = y[idx_val]
    out("Pitches of a type outside the pitcher's mask (the model cannot call them):")
    out(f"  mask frozen at the end of training : "
        f"{(train_only[idx_val, lbl] == 0).mean():.2%}")
    out(f"  causal mask (used here)            : "
        f"{(mask[idx_val, lbl] == 0).mean():.2%}")
    out("")
    out("Pitches carrying an id the model never trained on. Each has an")
    out("embedding row that was never updated, so it is sent to the unknown")
    out("bucket the model WAS trained with:")
    nt_val = never_trained(idx_val)
    out("  " + "   ".join(f"{k.replace('_id', '')} {v:.1%}" for k, v in nt_val.items()))

    # -----------------------------------------------------------------
    # The bar is the strongest table ON THE VALIDATION SEASON, and is
    # then locked, exactly as the model's threshold is. Choosing it on the
    # test season would be selection too, even in the table's favour.
    overall = {name: float(ok.mean()) for name, ok in val["tables"].items()}
    bar = max((name for name, _ in SCOUT_RUNGS), key=lambda k: overall[k])
    bar_label = dict(SCOUT_RUNGS)[bar]

    val_slice = slice_rows(val["conf"], val["tool_ok"], val["tables"])
    eligible = [r for r in val_slice if r["coverage"] >= MIN_COVERAGE]
    chosen = max(eligible, key=lambda r: r["tool"] - r[bar]) if eligible else None

    # -----------------------------------------------------------------
    hdr(out, "A. WHAT THE CORRECTIONS CHANGE")
    published, already_fair = None, False
    pub_path = data_dir / "product_claim_v5.published.json"
    if pub_path.exists():
        pub = json.loads(pub_path.read_text())
        # A schema-2 claim was written by THIS gate: it is already measured
        # the fair way and there is no old number to explain.
        already_fair = int(pub.get("schema", 1)) >= 2
        published = pub.get("recommended") or None
    old = None
    if not args.skip_reproduce and holdout is not None and not already_fair:
        print("Re-deriving the previously published number (old mask, frozen table)...")
        half = len(idx_val) // 2

        def as_before(mask_rows, label):
            """One pass the way the old gate fed the model: ids untouched."""
            raw = predict_rows(model, X, idx_val, mask_rows, label=label)
            t, _ = fit_temperature(raw[:half], y[idx_val][:half])
            _, call, conf = binary_read(apply_temperature(raw, t), fam)
            return float(t), conf, call == val["hard_true"]

        leaky = full_mask_through(pitcher, y, seasons, holdout, K)
        t_old, conf_old, ok_old = as_before(leaky[idx_val], "old mask")
        rows_old = [r for r in slice_rows(conf_old, ok_old, {"frozen": val["tables"]["frozen"]})
                    if r["coverage"] >= MIN_COVERAGE]
        if rows_old:
            best_old = max(rows_old, key=lambda r: r["tool"] - r["frozen"])
            old = {"temp": t_old, "conf": conf_old, "ok": ok_old, **best_old}
            # The causal mask alone, ids still untouched, so the two
            # corrections to the model's inputs can be told apart.
            _, conf_mid, ok_mid = as_before(mask[idx_val], "causal mask only")
    if old:
        out("The number on the site was produced by the old gate. Re-derived here")
        out("from the same model, to check nothing else moved:")
        out("")
        out(f"{'':<22}{'speaks on':>11}{'tool':>8}{'table':>8}{'edge':>8}")
        if published:
            out(f"{'published':<22}{published['coverage']:>11.1%}"
                f"{published['tool_accuracy']:>8.1%}{published['table_accuracy']:>8.1%}"
                f"{published['edge']:>+8.1%}")
        out(f"{'re-derived':<22}{old['coverage']:>11.1%}{old['tool']:>8.1%}"
            f"{old['frozen']:>8.1%}{old['tool'] - old['frozen']:>+8.1%}"
            f"   (threshold {old['threshold']:.0%}, T {old['temp']:.3f})")
        out("")
        out(f"Each correction in turn, all at that same {old['threshold']:.0%} threshold:")
        out("")
        out(f"{'':<50}{'speaks on':>10}{'tool':>8}{'table':>8}{'edge':>8}")

        def step(label, conf, ok, table_ok):
            sel = conf >= old["threshold"]
            if not sel.any():
                return
            out(f"{label:<50}{sel.mean():>10.1%}{ok[sel].mean():>8.1%}"
                f"{table_ok[sel].mean():>8.1%}{ok[sel].mean() - table_ok[sel].mean():>+8.1%}")

        frozen_ok = val["tables"]["frozen"]
        step("old gate: mask from the future, frozen table", old["conf"], old["ok"], frozen_ok)
        step("+ causal mask (model stops seeing the future)", conf_mid, ok_mid, frozen_ok)
        step("+ never-trained ids sent to the unknown bucket", val["conf"], val["tool_ok"], frozen_ok)
        for name, label in SCOUT_RUNGS:
            step("+ " + label, val["conf"], val["tool_ok"], val["tables"][name])
        out("")
        out("The first three rows change what the MODEL is given; the rest change")
        out("what the TABLE is given. Read down the 'edge' column: the drop across")
        out("the table rows is how much of the old number was the model holding")
        out("information the table had been denied.")
    elif already_fair:
        out("(nothing to reproduce — the claim this dataset was extended from was")
        out(" already measured with a causal mask against a live table)")
    elif args.skip_reproduce:
        out("(skipped: --skip-reproduce)")
    else:
        out("(skipped — there is no earlier number to reproduce for a random split)")

    # -----------------------------------------------------------------
    hdr(out, "B. CALIBRATION")
    out(f"Fitted temperature T = {val['temp']:.3f} "
        f"({'sharpened' if val['temp'] < 1 else 'softened'}), on the first half of "
        f"the validation season.")
    _, _, raw_conf = binary_read(val["raw"], fam)
    out("")
    out(f"{'commit @':>10}{'raw':>12}{'calibrated':>14}{'change':>10}")
    for t in COMMIT_THRESHOLDS:
        out(f"{t:>10.0%}{(raw_conf >= t).mean():>12.1%}"
            f"{(val['conf'] >= t).mean():>14.1%}"
            f"{(val['conf'] >= t).mean() - (raw_conf >= t).mean():>+10.1%}")

    probs, yv = val["probs"], val["yv"]
    pred, top = probs.argmax(1), probs.max(1)
    N = len(yv)

    hdr(out, "1. CONFIDENCE STRATIFICATION — where is the read strong?")
    out("Binned by the model's top probability. 'hard/soft' is the binary")
    out("gear-up decision; 'his mix' is just sitting on whichever side this")
    out("pitcher throws more, ignoring the count.")
    out("")
    out(f"{'confidence':>12}{'pitches':>10}{'% of all':>10}"
        f"{'10-cls acc':>12}{'hard/soft':>11}{'his mix':>10}{'edge':>8}")
    for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
        m = (top >= lo) & (top < hi)
        if m.sum() == 0:
            continue
        ahs, phs = val["tool_ok"][m].mean(), val["mix_ok"][m].mean()
        out(f"{f'{lo:.1f}-{hi:.1f}':>12}{m.sum():>10,}{m.sum() / N:>10.1%}"
            f"{(pred[m] == yv[m]).mean():>12.1%}{ahs:>11.1%}{phs:>10.1%}{ahs - phs:>+8.1%}")

    hdr(out, "2. CALIBRATION — is the confidence honest?")
    out("A hitter can only act on a number that means what it says.")
    out("")
    out(f"{'bin':>12}{'n':>10}{'predicted':>12}{'actual':>10}{'gap':>9}")
    for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
        m = (top >= lo) & (top < hi)
        if m.sum() == 0:
            continue
        act = (pred[m] == yv[m]).mean()
        out(f"{f'{lo:.1f}-{hi:.1f}':>12}{m.sum():>10,}{top[m].mean():>12.1%}"
            f"{act:>10.1%}{act - top[m].mean():>+9.1%}")

    hdr(out, "3. THE COMMIT SLICE vs HIS OVERALL MIX — a diagnostic, not the claim")
    out("'His mix' ignores the count, which every hitter reads off a scouting")
    out("report. Beating it proves little. Section 6 carries the real bar.")
    out("")
    out(f"{'threshold':>11}{'% advised':>12}{'accuracy':>11}{'his mix':>10}{'edge':>8}")
    for t in COMMIT_THRESHOLDS:
        m = val["conf"] >= t
        if m.sum() == 0:
            continue
        acc, pacc = val["tool_ok"][m].mean(), val["mix_ok"][m].mean()
        out(f"{t:>11.0%}{m.mean():>12.1%}{acc:>11.1%}{pacc:>10.1%}{acc - pacc:>+8.1%}")

    hdr(out, "4. BY COUNT — where does it live?")
    out("'naive' = sit whichever side is commoner in that count across the")
    out("league, which every hitter already knows.")
    out("")
    out(f"{'count':>8}{'pitches':>10}{'model':>9}{'his mix':>9}{'naive':>8}"
        f"{'vs naive':>10}{'%hard':>8}")
    bv, sv = balls[idx_val], strikes[idx_val]
    by_count = []
    for b in range(4):
        for s in range(3):
            m = (bv == b) & (sv == s)
            if m.sum() < 500:
                continue
            share = val["hard_true"][m].mean()
            acc = val["tool_ok"][m].mean()
            by_count.append((f"{b}-{s}", int(m.sum()), acc, val["mix_ok"][m].mean(),
                             max(share, 1 - share), share))
    for c, n, acc, mix, naive, share in sorted(by_count, key=lambda r: -(r[2] - r[4])):
        out(f"{c:>8}{n:>10,}{acc:>9.1%}{mix:>9.1%}{naive:>8.1%}"
            f"{acc - naive:>+10.1%}{share:>8.1%}")

    def across_all(ev):
        """Section 5 for one season: model vs the locked bar in every
        situation, with a clustered interval on the lift."""
        r = ev["rows"]
        cs = cell_summary(pitcher[r], balls[r], strikes[r], ev["hard_true"],
                          ev["tool_ok"], ev["tables"][bar])
        if cs is None:
            return None
        lo, hi = widest_ci(ev["clusters"], cs["in_cells"], ev["tool_ok"],
                           ev["tables"][bar], n_boot=N_BOOT)
        cs["lift"], cs["ci"] = cs["model"] - cs["bar"], [lo, hi]
        cs["verdict"] = "ahead" if lo > 0 else "behind" if hi < 0 else "level"
        return cs

    hdr(out, "5. ACROSS ALL SITUATIONS — better everywhere, or only when it speaks?")
    out(f"The bar: {bar_label}.")
    out("It was the strongest table across the whole validation season:")
    for name, label in SCOUT_RUNGS:
        out(f"  {overall[name]:.1%}  {label}" + ("   <- the bar" if name == bar else ""))
    out(f"  {overall['frozen']:.1%}  count table frozen at the end of training (the old bar)")
    out(f"  {val['tool_ok'].mean():.1%}  the model, calling every pitch")
    out("")
    cs_val = across_all(val)
    if cs_val:
        out(f"Every (pitcher, count) situation with >={MIN_CELL_LIFT} held-out pitches: "
            f"{cs_val['cells']:,} situations, {cs_val['n']:,} pitches")
        out(f"  model                         {cs_val['model']:.1%}")
        out(f"  the bar                       {cs_val['bar']:.1%}")
        out(f"  best fixed call per situation {cs_val['oracle']:.1%}   <- hindsight; an upper bound for a count card")
        out(f"  LIFT vs the bar               {cs_val['lift']:+.1%}   "
            f"95% CI [{cs_val['ci'][0]:+.1%}, {cs_val['ci'][1]:+.1%}]")
        out(f"  situations where the model wins  {cs_val['cells_won']:.0%}")
        out("")
        out(f"{'count':>8}{'pitches':>10}{'model':>9}{'bar':>9}{'lift':>8}{'hindsight':>11}")
        for pc in sorted(cs_val["per_count"], key=lambda d: -(d["model"] - d["bar"])):
            out(f"{pc['count']:>8}{pc['n']:>10,}{pc['model']:>9.1%}{pc['bar']:>9.1%}"
                f"{pc['model'] - pc['bar']:>+8.1%}{pc['oracle']:>11.1%}")
        out("")
        out({"ahead": "The model is AHEAD of the table across all situations, not only on its confident slice.",
             "level": "Across all situations the model and the table cannot be told apart.",
             "behind": "Across all situations the TABLE is ahead. Any value is confined to the confident slice."}
            [cs_val["verdict"]])
    else:
        out(f"No situation had >={MIN_CELL_LIFT} held-out pitches — cannot measure.")

    hdr(out, "6. THE PRODUCT CLAIM")
    out("When the tool SPEAKS, how often is it right — and how often would each")
    out("scouting table have been right on those same pitches?")
    out("")
    out("The advised pitches are chosen by the model, so this is not a fair")
    out("fight across all pitches (section 5 is). It IS the product's claim,")
    out("because staying silent the rest of the time is the design.")
    out("")
    head = f"{'commit @':>9}{'speaks on':>11}{'tool':>8}{'frozen':>9}"
    head += "".join(f"{n.replace('live_side_season', 'season').replace('live_side', 'side'):>8}"
                    for n, _ in SCOUT_RUNGS)
    out(head + f"{'edge':>9}")
    for r in val_slice:
        line = (f"{r['threshold']:>9.0%}{r['coverage']:>11.1%}{r['tool']:>8.1%}"
                f"{r['frozen']:>9.1%}")
        line += "".join(f"{r[n]:>8.1%}" for n, _ in SCOUT_RUNGS)
        out(line + f"{r['tool'] - r[bar]:>+9.1%}"
            + ("   <- chosen" if chosen and r["threshold"] == chosen["threshold"] else ""))
    out("")
    out("  frozen = count table frozen at the end of training (the old bar)")
    out("  live   = count table, kept current      side = also split by batter side")
    out("  season = also weighted to this season   edge = tool minus the bar "
        f"('{bar.replace('live_side_season', 'season').replace('live_side', 'side')}')")

    def claim_at(ev, threshold):
        """The claim numbers for one season at a fixed threshold."""
        sel = ev["conf"] >= threshold
        if sel.sum() == 0:
            return None
        lo, hi = widest_ci(ev["clusters"], sel, ev["tool_ok"], ev["tables"][bar],
                           n_boot=N_BOOT)
        tool, tab = float(ev["tool_ok"][sel].mean()), float(ev["tables"][bar][sel].mean())
        return {"threshold": float(threshold), "coverage": float(sel.mean()),
                "n_advised": int(sel.sum()), "tool_accuracy": tool,
                "table_accuracy": tab, "edge": tool - tab, "edge_ci": [lo, hi],
                "ladder": {name: float(ok[sel].mean()) for name, ok in ev["tables"].items()}}

    val_claim = claim_at(val, chosen["threshold"]) if chosen else None

    hdr(out, f"VERDICT ON THE VALIDATION SEASON{f' ({holdout})' if holdout else ''}")
    if val_claim:
        out(f"When it speaks (>={val_claim['threshold']:.0%} confidence): "
            f"{val_claim['coverage']:.1%} of pitches, {val_claim['tool_accuracy']:.1%} right,")
        out(f"vs {val_claim['table_accuracy']:.1%} for the bar — "
            f"{val_claim['edge']:+.1%}  (95% CI [{val_claim['edge_ci'][0]:+.1%}, "
            f"{val_claim['edge_ci'][1]:+.1%}]).")
        out("")
        out("This threshold was CHOSEN on these pitches as the one with the best")
        out("edge, so the figure flatters itself a little. It is a tuning result.")
        if args.test_season is None:
            out("Score a later season with --test-season for a number that is not.")
    else:
        out(f"No threshold advises on >={MIN_COVERAGE:.0%} of pitches. Nothing to claim.")

    # -----------------------------------------------------------------
    test_claim = cs_test = status = None
    if args.test_season is not None and not val_claim:
        # Nothing could be locked, so there is nothing to test. That is a
        # failed test, not an absent one.
        status = "not_confirmed"
        hdr(out, "VERDICT ON THE CLEAN TEST")
        out(f"NOT CONFIRMED: on the validation season no threshold advises on "
            f">={MIN_COVERAGE:.0%} of pitches,")
        out("so there was no setting to carry to the test season.")
    if args.test_season is not None and val_claim:
        test = evaluate(idx_test, f"test {args.test_season}", temp=val["temp"])
        test_claim = claim_at(test, val_claim["threshold"])
        cs_test = across_all(test)

        hdr(out, f"7. CLEAN TEST — the {args.test_season} season")
        out("Nothing was fitted or chosen on these pitches. Locked beforehand, on")
        out(f"the {holdout} season:")
        out(f"  temperature  {val['temp']:.3f}")
        out(f"  threshold    {val_claim['threshold']:.0%}")
        out(f"  the bar      {bar_label}")
        out("")
        lbl = y[idx_test]
        out(f"Pitches of a type outside the causal mask: "
            f"{(mask[idx_test, lbl] == 0).mean():.2%}   "
            f"(frozen mask: {(train_only[idx_test, lbl] == 0).mean():.2%})")
        nt_test = never_trained(idx_test)
        out(f"Pitches from pitchers the model never trained on (no embedding, or one "
            f"never updated): {(known['pitcher_id'][pid_enc[idx_test]] == 0).mean():.1%}")
        out("  never-trained ids sent to the unknown bucket: "
            + "   ".join(f"{k.replace('_id', '')} {v:.1%}" for k, v in nt_test.items()))
        out("")
        if test_claim:
            out(f"{'':<14}{'speaks on':>11}{'tool':>8}{'bar':>8}{'edge':>8}   95% CI")
            out(f"{'validation':<14}{val_claim['coverage']:>11.1%}"
                f"{val_claim['tool_accuracy']:>8.1%}{val_claim['table_accuracy']:>8.1%}"
                f"{val_claim['edge']:>+8.1%}   [{val_claim['edge_ci'][0]:+.1%}, "
                f"{val_claim['edge_ci'][1]:+.1%}]")
            out(f"{'TEST':<14}{test_claim['coverage']:>11.1%}"
                f"{test_claim['tool_accuracy']:>8.1%}{test_claim['table_accuracy']:>8.1%}"
                f"{test_claim['edge']:>+8.1%}   [{test_claim['edge_ci'][0]:+.1%}, "
                f"{test_claim['edge_ci'][1]:+.1%}]")
            out("")
            out("On the advised pitches, each table in turn:")
            out(f"  {test_claim['ladder']['frozen']:.1%}  frozen count table (the old bar)")
            for name, label in SCOUT_RUNGS:
                out(f"  {test_claim['ladder'][name]:.1%}  {label}"
                    + ("   <- the bar" if name == bar else ""))
            out(f"  {test_claim['tool_accuracy']:.1%}  the tool")
        if cs_test:
            out("")
            out(f"Across all situations ({cs_test['cells']:,} with >={MIN_CELL_LIFT} pitches): "
                f"model {cs_test['model']:.1%}, bar {cs_test['bar']:.1%}, "
                f"lift {cs_test['lift']:+.1%}  [{cs_test['ci'][0]:+.1%}, {cs_test['ci'][1]:+.1%}], "
                f"model wins {cs_test['cells_won']:.0%} of them.")

        # Pre-registered in docs/EXPERIMENTS.md before any real test
        # number existed. Do not adjust these after seeing one.
        if not test_claim or test_claim["coverage"] < MIN_COVERAGE:
            status = "not_confirmed"
        elif test_claim["edge_ci"][0] > 0 and test_claim["edge"] >= MIN_EDGE:
            status = "confirmed"
        elif test_claim["edge_ci"][0] > 0:
            status = "reduced"
        else:
            status = "not_confirmed"

        hdr(out, "VERDICT ON THE CLEAN TEST")
        if status == "confirmed":
            out("CONFIRMED. On a season used for nothing else, the tool beats an")
            out(f"up-to-date scouting table by {test_claim['edge']:+.1%} on the "
                f"{test_claim['coverage']:.0%} of pitches where it speaks,")
            out("and the interval excludes zero. That is the claim the site may make.")
        elif status == "reduced":
            out("REDUCED. The edge is real — the interval excludes zero — but at")
            out(f"{test_claim['edge']:+.1%} it is under the {MIN_EDGE:.0%} bar set in advance.")
            out("The site must show this smaller number, not the validation one.")
        else:
            why = ("it speaks on too few pitches" if not test_claim
                   or test_claim["coverage"] < MIN_COVERAGE else
                   "the interval on the edge includes zero")
            out(f"NOT CONFIRMED: {why}.")
            out("On a season used for nothing else, the tool is not shown to beat a")
            out("scouting table that is kept up to date. The site must say so; it")
            out("may not claim an edge.")

    text = "\n".join(lines)
    print(text)
    report_path = data_dir / "actionability_report_v5.txt"
    report_path.write_text(text + "\n")

    # -----------------------------------------------------------------
    # The site reads these rather than carrying hand-copied figures, so a
    # stale or flattering number cannot survive a retrain. When a test
    # season was scored, ITS numbers are the headline: they are the only
    # ones nothing was tuned on.
    headline, head_cells = (test_claim, cs_test) if args.test_season else (val_claim, cs_val)
    if args.test_season:
        measured = (f"trained through {holdout - 1}, tuned on {holdout}, "
                    f"tested on {args.test_season}")
        n_measured = int(len(idx_test))
    else:
        measured, n_measured = split_desc, int(len(idx_val))
        status = "validation_only"
    if holdout is None:
        # A random split shares games and at-bats between train and
        # validation. Its numbers are printed above as a diagnostic and
        # are never handed to the site.
        status, headline = "random_split", None
        print("\nNOTE: this model was trained on a random split. No claim is "
              "written for the site;\nretrain with --split temporal for a "
              "number that can be published.")

    def cells_json(cs):
        return None if cs is None else {
            "lift": cs["lift"], "ci": cs["ci"], "cells_won": cs["cells_won"],
            "verdict": cs["verdict"], "model": cs["model"], "bar": cs["bar"]}

    claim = {
        "schema": 2,
        "status": status,
        "split": measured,
        "model_split": split_desc,
        "evaluated_season": args.test_season or holdout,
        "n_val": n_measured,
        "n_train": int(len(idx_tr)),
        "train_through": (holdout - 1) if holdout else None,
        "tuned_on": holdout,
        "temperature": val["temp"],
        # The confidence at which the site speaks: chosen on the validation
        # season, and the one the headline numbers were measured at.
        "threshold": val_claim["threshold"] if val_claim else None,
        # The bundle builder refuses to pair this claim with any other
        # weights: a claim for one model beside another's is a number
        # nobody measured.
        "model_sha256": hashlib.sha256(
            (data_dir / "best_model_v5.keras").read_bytes()).hexdigest(),
        "mask": mask_desc,
        "baseline": bar,
        "baseline_label": bar_label,
        "baseline_plain": SCOUT_PLAIN[bar],
        "min_edge": MIN_EDGE,
        "min_coverage": MIN_COVERAGE,
        "recommended": headline,
        "overall": cells_json(head_cells),
        "section5_lift_vs_scout": head_cells["lift"] if head_cells else 0.0,
        "section5_verdict": head_cells["verdict"] if head_cells else "unmeasured",
        "slice": [{"threshold": r["threshold"], "coverage": r["coverage"],
                   "tool_accuracy": r["tool"], "table_accuracy": r[bar],
                   "edge": r["tool"] - r[bar]} for r in val_slice],
        "validation": {"season": holdout, "n": int(len(idx_val)),
                       "recommended": val_claim, "overall": cells_json(cs_val)},
        "shippable": bool(headline and headline["edge"] >= MIN_EDGE
                          and headline["coverage"] >= MIN_COVERAGE
                          and status in ("confirmed", "validation_only")),
    }
    claim_path = data_dir / "product_claim_v5.json"
    # Once a test number exists for this model, a tuning-season number
    # must not replace it as the claim. Re-running the gate without
    # --test-season is a look at the validation season, not a new result.
    if status == "validation_only" and claim_path.exists():
        prev = json.loads(claim_path.read_text())
        if (prev.get("status") in ("confirmed", "reduced", "not_confirmed")
                and prev.get("model_sha256") == claim["model_sha256"]):
            print(f"\nKept the existing claim in {claim_path.name}: it was measured on "
                  f"{prev.get('evaluated_season')}, a season nothing\nwas tuned on, and "
                  f"a validation-season number does not replace that. (Pass "
                  f"--test-season {prev.get('evaluated_season')} to re-measure it.)")
            print(f"Written to {report_path}")
            return
    claim_path.write_text(json.dumps(claim, indent=2))
    print(f"\nProduct claim written to {claim_path}")
    print(f"Written to {report_path}")


if __name__ == "__main__":
    main()
