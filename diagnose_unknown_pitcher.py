"""
NextPitchAI v6 — how much is the tool worth for a pitcher it does not know?
===========================================================================
The site can be given a hand-entered pitch mix, so it works for a college
or high-school arm who is not in the data. The question that decides
whether that mode is worth shipping is how much of the edge survives
losing the pitcher's identity.

Measured by ABLATION on real held-out rows:

  full     — inference as the gate scores a known MLB arm
  no-id    — identical, except the pitcher embedding is forced to index
             0, the "unknown pitcher" bucket the model was trained with.
             Diagnostic only: it isolates the embedding.
  custom   — everything the custom-arsenal mode does NOT have: no
             embedding, no zone tendencies, no matchup history, and a
             pitch mix typed from memory rather than the exact statistic.

All three are scored the way the gate scores: same causal arsenal mask,
same calibration temperature and commit threshold (read from the gate's
claim, never re-fitted here), and the same live scouting table as the
bar. So the rows ARE comparable to section 6 of analyze_actionability.py.

Needs the gate to have been run on the dataset. No TensorFlow, no
retraining.

    .venv/bin/python diagnose_unknown_pitcher.py
    .venv/bin/python diagnose_unknown_pitcher.py --data-dir data_ext --season 2026
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_model import (
    DATA_DIR, FASTBALL_FAMILY, SCOUT_RUNGS, apply_temperature, known_ids,
    live_scout, load_inputs, load_row_meta, load_scoring_model, load_split,
    predict_rows, scoring_mask, widest_ci,
)

# Standard deviation of the error a user makes typing a pitch mix from
# memory, in SCALED units. Roughly a few percentage points per pitch.
MIX_NOISE_SD = 0.25


def ablation(mode, cols, rng):
    """The edit applied to each batch for one condition, or None."""
    if mode == "full":
        return None

    def edit(inputs):
        inputs["pitcher_id"][:] = 0
        if mode != "custom":
            return
        ctx = inputs["ctx"]
        # A hand-entered mix is a rounded guess, not the exact expanding
        # prior. Perturb it so the measurement reflects the accuracy a
        # real user would get, not a best case they cannot reach.
        ars = ctx[:, cols["ars"]] + rng.normal(0.0, MIX_NOISE_SD,
                                               (len(ctx), len(cols["ars"])))
        ctx[:, cols["ars"]] = ars
        # No matchup history: it backs off to the pitcher's own mix, with
        # zero prior meetings.
        ctx[:, cols["match"]] = ars
        # League values in SCALED space: the training mean is, by
        # definition, zero after standardisation.
        ctx[:, cols["fam"]] = 0.0
        ctx[:, cols["zone"]] = 0.0
    return edit


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--season", type=int, default=None,
                    help="score this season instead of the validation season")
    args = ap.parse_args()
    data_dir = args.data_dir.resolve()

    meta = json.loads((data_dir / "meta_v5.json").read_text())
    classes, names = meta["pitch_classes"], meta["ctx_feature_names"]
    fam = np.array([1 if c in FASTBALL_FAMILY else 0 for c in classes])

    claim_path = data_dir / "product_claim_v5.json"
    claim = json.loads(claim_path.read_text()) if claim_path.exists() else {}
    rm = load_row_meta(data_dir)
    if rm is None or int(claim.get("schema", 1)) < 2 or not claim.get("recommended"):
        raise SystemExit(
            f"Run the gate on {data_dir.name}/ first — this script scores at the "
            f"temperature, threshold and bar it locked:\n\n"
            f"    .venv/bin/python analyze_actionability.py --data-dir {data_dir.name}\n")
    temp = float(claim["temperature"])
    threshold = float(claim["recommended"]["threshold"])
    bar = claim["baseline"]

    X = load_inputs(data_dir)
    y = np.load(data_dir / "y_labels.npy")
    seasons = np.load(data_dir / "X_season.npy")
    idx_tr, idx_val, split_desc = load_split(y, data_dir)
    rows = idx_val if args.season is None else np.flatnonzero(seasons == args.season)
    if len(rows) == 0:
        raise SystemExit(f"no rows for season {args.season} in {data_dir.name}/")
    print(f"Split: {split_desc}   rows scored: {len(rows):,}"
          + (f" (season {args.season})" if args.season else " (validation season)"))
    print(f"Locked by the gate: temperature {temp:.3f}, threshold {threshold:.0%}, "
          f"bar = {dict(SCOUT_RUNGS)[bar]}\n")

    mask, _ = scoring_mask(np.asarray(X["pitcher_id"]), y, idx_tr, len(classes), rm)
    hard = fam[y]
    scout = live_scout(rm["pitcher"], rm["balls"].astype(np.int64),
                       rm["strikes"].astype(np.int64), np.asarray(X["batter_hand"]),
                       seasons, hard)
    hard_true = hard[rows].astype(bool)
    bar_ok = (scout[bar][rows] >= 0.5) == hard_true
    clusters = [rm["game_pk"][rows], rm["pitcher"][rows]]

    model = load_scoring_model(data_dir)
    known = known_ids(X, idx_tr)
    cols = {
        "ars": [names.index(f"arsenal_{c}") for c in classes],
        "match": [names.index(f"matchup_{c}") for c in classes],
        "fam": names.index("matchup_familiarity"),
        "zone": [names.index(f"zoneprior_{z}") for z in meta["zone_classes"]],
    }
    rng = np.random.default_rng(0)

    print("  full   = as the gate scores a known MLB pitcher")
    print("  no-id  = pitcher embedding blanked (diagnostic only)")
    print("  custom = what the custom-arsenal mode actually has:")
    print("           no embedding, no zone tendencies, no matchup history,")
    print(f"           and a typed mix (noise sd {MIX_NOISE_SD} scaled)\n")
    print(f"{'':10s}{'top-1':>8}{'hard/soft':>11}{'speaks':>9}{'tool':>8}"
          f"{'bar':>8}{'edge':>8}   95% CI")
    res = {}
    for label in ("full", "no-id", "custom"):
        raw = predict_rows(model, X, rows, mask[rows], known=known,
                           edit=ablation(label, cols, rng))
        probs = apply_temperature(raw, temp)
        p_hard = probs[:, fam == 1].sum(1)
        ok = (p_hard >= 0.5) == hard_true
        sel = np.maximum(p_hard, 1.0 - p_hard) >= threshold
        r = {"top1": float((probs.argmax(1) == y[rows]).mean()),
             "acc": float(ok.mean()), "cov": float(sel.mean())}
        if sel.any():
            r["tool"], r["bar"] = float(ok[sel].mean()), float(bar_ok[sel].mean())
            r["ci"] = widest_ci(clusters, sel, ok, bar_ok)
            print(f"{label:10s}{r['top1']:>8.1%}{r['acc']:>11.1%}{r['cov']:>9.1%}"
                  f"{r['tool']:>8.1%}{r['bar']:>8.1%}{r['tool'] - r['bar']:>+8.1%}"
                  f"   [{r['ci'][0]:+.1%}, {r['ci'][1]:+.1%}]")
        else:
            print(f"{label:10s}{r['top1']:>8.1%}{r['acc']:>11.1%}{'never speaks':>41}")
        res[label] = r

    f, n, c = res["full"], res["no-id"], res["custom"]
    print()
    print(f"Blanking the embedding alone costs {f['top1'] - n['top1']:+.1%} top-1. If that")
    print("is near zero, the embedding is redundant with arsenal_prior — the")
    print("pitcher's mix is in the context vector too, and the model reads it")
    print("there. That is a finding about the model, NOT a result about the")
    print("custom mode.\n")
    if "tool" in f and "tool" in c:
        fe, ce = f["tool"] - f["bar"], c["tool"] - c["bar"]
        print(f"What the custom mode costs: {c['top1'] - f['top1']:+.1%} top-1, "
              f"edge {fe:+.1%} -> {ce:+.1%}\n")
        if c["ci"][0] > 0:
            print("VERDICT: the custom mode beats the bar on the pitches where it")
            print("speaks. It may make that claim with ITS OWN numbers — never the")
            print("known-pitcher ones.")
        else:
            print("VERDICT: the custom mode is not shown to beat a scouting table")
            print("kept up to date. Ship it as a pitch-mix explorer and attach no")
            print("accuracy claim.")
        print()
        print("Either way this is MLB pitching with identity hidden. It says")
        print("nothing about pitchers whose sequencing the model has never seen.")


if __name__ == "__main__":
    main()
