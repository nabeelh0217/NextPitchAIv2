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
THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)


def run(model, X, idx, force_unknown_pitcher):
    """Predicted probabilities over idx, optionally blanking pitcher id."""
    out = []
    for i in range(0, len(idx), BATCH):
        sl = idx[i:i + BATCH]
        pid = (np.zeros(len(sl), np.int32) if force_unknown_pitcher
               else X["pitcher_id"][sl])
        out.append(model.predict({
            "seq": X["seq"][sl], "ctx": X["ctx"][sl],
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

    print(f"{'':16s}{'top-1':>9}{'hard/soft':>11}"
          f"{'speaks':>9}{'tool':>8}{'table':>8}{'edge':>8}")
    results = {}
    for label, blank in (("full", False), ("no-id", True)):
        probs = run(model, X, idx_val, blank)
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
    f, n = results["full"], results["no-id"]
    print(f"Losing the pitcher's identity costs {f[0] - n[0]:+.1%} top-1 "
          f"and {f[1] - n[1]:+.1%} on the hard/soft call.")
    if f[2] and n[2]:
        fe, ne = f[2][3] - f[2][2], n[2][3] - n[2][2]
        print(f"Edge over the count-split table: {fe:+.1%} -> {ne:+.1%}")
        print()
        if ne >= 0.02:
            print("The custom-arsenal mode carries a real edge. Ship it, with")
            print("its own coverage and accuracy numbers — not the MLB ones.")
        else:
            print("The custom-arsenal mode does NOT beat the table a hitter")
            print("could write himself. Offer it as a pitch-mix explorer, and")
            print("do not attach the +4.2 point claim to it — that number was")
            print("measured with pitcher identity, which this mode does not have.")


if __name__ == "__main__":
    main()
