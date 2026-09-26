"""
NextPitchAI v6 — how much is the tool worth for a pitcher it does not know?
===========================================================================
The site can now be given a hand-entered pitch mix, so it works for a
college or high-school arm who is not in the data. The question that
decides whether that mode is worth shipping is how much of the edge
survives losing the pitcher's identity.

Pitcher identity is the single largest signal in this model. This script
measures what is left without it, by ABLATION on the real held-out rows:

  full     — inference exactly as the site does it for a known MLB arm
  no-id    — identical, except the pitcher embedding is forced to index
             0, the "unknown pitcher" bucket the model was trained with.
             The arsenal prior and mask stay, which is precisely the
             custom-arsenal case: we know what he throws, not who he is.

Both are scored against the same bar the product uses — the count-split
scouting table — so the numbers are comparable to section 6 of
analyze_actionability.py.

Needs data_v5/ and site/serving/. No TensorFlow, no retraining.

    python diagnose_unknown_pitcher.py
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_v5"
SERVING = BASE_DIR / "site" / "serving"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "site"))

from evaluate_model import load_split, FASTBALL_FAMILY  # noqa: E402
from numpy_model import NumpyModel  # noqa: E402

BATCH = 4096
# Standard deviation of the error a user makes typing a pitch mix from
# memory, in SCALED units. Roughly a few percentage points per pitch.
MIX_NOISE_SD = 0.25
THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)


def run(model, X, idx, mode, cols=None, league=None, rng=None):
    """
    Predicted probabilities over idx under one of three conditions.

      full      — as the site serves a known MLB pitcher
      no-id     — pitcher embedding only is blanked
      custom    — everything the site does NOT have for a hand-entered
                  pitcher: no embedding, no zone tendencies, no matchup
                  history, and an arsenal prior the user typed from
                  memory rather than the exact expanding statistic

    Only `custom` answers the question the custom-arsenal mode poses.
    `no-id` is a diagnostic: it isolates how much the embedding alone is
    worth, which turns out to be very little because the pitcher's mix
    is also in the context vector.
    """
    out = []
    for i in range(0, len(idx), BATCH):
        sl = idx[i:i + BATCH]
        ctx = np.array(X["ctx"][sl], dtype=np.float32)
        pid = np.array(X["pitcher_id"][sl], np.int32)

        if mode in ("no-id", "custom"):
            pid = np.zeros(len(sl), np.int32)
        if mode == "custom":
            a, m, f, z = cols["ars"], cols["match"], cols["fam"], cols["zone"]
            # A hand-entered mix is a rounded guess, not the exact
            # expanding prior. Perturb it so the measurement reflects the
            # accuracy a real user would get, not a best case they cannot
            # reach.
            ars = ctx[:, a]
            if rng is not None:
                ars = ars + rng.normal(0.0, MIX_NOISE_SD, ars.shape)
            ctx[:, a] = ars
            # No matchup history: it backs off to the pitcher's own mix,
            # with zero prior meetings.
            ctx[:, m] = ars
            ctx[:, f] = league["fam"]
            # No per-pitcher location tendencies, only the league's.
            ctx[:, z] = league["zone"]

        out.append(model.predict({
            "seq": X["seq"][sl], "ctx": ctx,
            "pitcher_id": pid, "batter_id": X["batter_id"][sl],
            "pitcher_hand": X["pitcher_hand"][sl],
            "batter_hand": X["batter_hand"][sl],
            "park_id": X["park_id"][sl], "catcher_id": X["catcher_id"][sl],
            "arsenal_mask": X["arsenal_mask"][sl]}))
    return np.concatenate(out)


def main():
    meta = json.loads((DATA_DIR / "meta_v5.json").read_text())
    classes = meta["pitch_classes"]
    hard_i = [i for i, c in enumerate(classes) if c in FASTBALL_FAMILY]

    y = np.load(DATA_DIR / "y_labels.npy")
    idx_tr, idx_val, split_desc = load_split(y)
    print(f"Split: {split_desc}   validation rows: {len(idx_val):,}\n")

    keys = ["seq", "ctx", "pitcher_id", "batter_id", "pitcher_hand",
            "batter_hand", "park_id", "catcher_id", "arsenal_mask"]
    X = {k: np.load(DATA_DIR / f"X_{k}.npy", mmap_mode="r") for k in keys}

    model = NumpyModel(SERVING / "model_weights.npz", SERVING / "model_arch.json")

    # Counts, for the scouting-table bar. ctx columns 0 and 1 are scaled,
    # so recover the integer levels by ranking the distinct values.
    names = meta["ctx_feature_names"]
    ctx_val = np.asarray(X["ctx"][idx_val][:, [names.index("balls"),
                                               names.index("strikes")]])
    balls = np.unique(ctx_val[:, 0], return_inverse=True)[1]
    strikes = np.unique(ctx_val[:, 1], return_inverse=True)[1]

    yv = y[idx_val]
    hard_true = np.isin(yv, hard_i).astype(int)

    # The bar: for each (pitcher, count) cell, the majority hard/soft call
    # from TRAINING rows. This is the scouting report a hitter already has.
    ptr = np.asarray(X["pitcher_id"])
    ctx_tr = np.asarray(X["ctx"][idx_tr][:, [names.index("balls"),
                                             names.index("strikes")]])
    b_tr = np.unique(ctx_tr[:, 0], return_inverse=True)[1]
    s_tr = np.unique(ctx_tr[:, 1], return_inverse=True)[1]
    hard_tr = np.isin(y[idx_tr], hard_i).astype(int)
    npid = int(ptr.max()) + 1
    cnt = np.zeros((npid, 4, 3)); hit = np.zeros((npid, 4, 3))
    np.add.at(cnt, (ptr[idx_tr], b_tr, s_tr), 1.0)
    np.add.at(hit, (ptr[idx_tr], b_tr, s_tr), hard_tr)
    rate = np.divide(hit, np.maximum(cnt, 1))
    league_rate = hard_tr.mean()
    scout = np.where(cnt[ptr[idx_val], balls, strikes] >= 40,
                     rate[ptr[idx_val], balls, strikes], league_rate) >= 0.5

    # Column groups the custom mode has to give up.
    cols = {
        "ars": [names.index(f"arsenal_{c}") for c in classes],
        "match": [names.index(f"matchup_{c}") for c in classes],
        "fam": names.index("matchup_familiarity"),
        "zone": [names.index(f"zoneprior_{z}") for z in meta["zone_classes"]],
    }
    # League values in SCALED space: the training mean is, by definition,
    # zero after standardisation.
    league = {"fam": 0.0, "zone": 0.0}
    rng = np.random.default_rng(0)

    print("  full   = as served for a known MLB pitcher")
    print("  no-id  = pitcher embedding blanked (diagnostic only)")
    print("  custom = what the custom-arsenal mode actually has:")
    print("           no embedding, no zone tendencies, no matchup history,")
    print(f"           and a typed mix (noise sd {MIX_NOISE_SD} scaled)\n")
    print(f"{'':16s}{'top-1':>9}{'hard/soft':>11}"
          f"{'speaks':>9}{'tool':>8}{'table':>8}{'edge':>8}")
    results = {}
    for label in ("full", "no-id", "custom"):
        probs = run(model, X, idx_val, label, cols, league, rng)
        top1 = float((probs.argmax(1) == yv).mean())
        p_hard = probs[:, hard_i].sum(1)
        pred_hard = (p_hard >= 0.5).astype(int)
        acc = float((pred_hard == hard_true).mean())

        best = None
        for t in THRESHOLDS:
            sel = np.maximum(p_hard, 1 - p_hard) >= t
            if sel.mean() < 0.10:
                continue
            tool = float((pred_hard[sel] == hard_true[sel]).mean())
            tab = float((scout[sel] == hard_true[sel]).mean())
            if best is None or tool - tab > best[3] - best[2]:
                best = (float(sel.mean()), t, tab, tool)
        results[label] = (top1, acc, best)
        if best:
            cov, t, tab, tool = best
            print(f"{label:16s}{top1:>8.1%}{acc:>11.1%}"
                  f"{cov:>9.1%}{tool:>8.1%}{tab:>8.1%}{tool - tab:>+8.1%}")
        else:
            print(f"{label:16s}{top1:>8.1%}{acc:>11.1%}"
                  f"{'never speaks on >=10%':>33}")

    print()
    f, n, c = results["full"], results["no-id"], results["custom"]
    print(f"Blanking the embedding alone costs {f[0] - n[0]:+.1%} top-1. If that")
    print("is near zero, the embedding is redundant with arsenal_prior — the")
    print("pitcher's mix is in the context vector too, and the model reads it")
    print("there. That is a finding about the model, NOT a result about the")
    print("custom mode.\n")
    if f[2] and c[2]:
        fe, ce = f[2][3] - f[2][2], c[2][3] - c[2][2]
        print(f"What the custom mode actually costs: {f[0] - c[0]:+.1%} top-1, "
              f"edge {fe:+.1%} -> {ce:+.1%}")
        print()
        print("NOTE: these absolute edges are NOT section 6's. There is no")
        print("temperature calibration here and the scout table is built")
        print("slightly differently. Compare the rows to each other, and quote")
        print("analyze_actionability.py for the product claim.")
        print()
        if ce >= 0.02:
            print("VERDICT: the custom mode keeps a real edge. It can be shipped")
            print("with ITS OWN numbers — never the MLB ones.")
        else:
            print("VERDICT: the custom mode does not beat a table a hitter could")
            print("write himself. Ship it as a pitch-mix explorer and attach no")
            print("accuracy claim.")


if __name__ == "__main__":
    main()
