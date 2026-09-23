"""
NextPitchAI — commit cards
===========================
Turns the model into something a hitter can actually use.

A batter cannot read a screen mid-at-bat. What he can do is walk up
knowing three or four rules: "3-1 against this guy, sit fastball." This
finds those rules — the (pitcher, count, batter-hand) situations where
the model's read is BOTH strong enough to commit AND better than the
scouting report the hitter already has.

Everything here is measured on held-out validation rows, never on
synthetic situations, so a card is a claim that survived out-of-sample.

Usage:
    .venv/bin/python build_commit_cards.py
    .venv/bin/python build_commit_cards.py --names   # resolve MLBAM ids

Output:
    data_v5/commit_cards_v5.txt   human-readable cards
    data_v5/commit_cards_v5.json  for the website
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from evaluate_model import (
    DATA_DIR, SPLIT_SEED, SPLIT_TEST_SIZE, FASTBALL_FAMILY, _smoothed,
)

# A rule only earns a place on a card if it clears all three. These match
# analyze_actionability.py — the gate that validated the model.
MIN_N = 120          # enough held-out pitches for the cell to mean anything
MIN_CONF = 0.65      # the commit threshold the gate found actionable
MIN_ACC = 0.70       # a hitter following the rule is right this often
MIN_EDGE = 0.02      # and beats the count-split scouting report
MIN_CONSISTENCY = 0.70   # the confident calls must agree with each other,
# or it is not a rule anyone can memorize — it is situation-dependent
# noise inside the count, and a single "sit" label would mislead.

HAND = {0: "RHB", 1: "LHB", 2: "SWITCH"}


def resolve_names(mlbam_ids):
    """Best-effort MLBAM -> name. Needs network; degrades to ids."""
    try:
        from pybaseball import playerid_reverse_lookup
        df = playerid_reverse_lookup(list(mlbam_ids), key_type="mlbam")
        return {
            int(r.key_mlbam): f"{r.name_first.title()} {r.name_last.title()}"
            for r in df.itertuples()
        }
    except Exception as e:
        print(f"(name lookup unavailable: {e})")
        return {}


def main():
    import tensorflow as tf

    with open(DATA_DIR / "meta_v5.json") as f:
        meta = json.load(f)
    classes = meta["pitch_classes"]
    K = len(classes)
    fam = np.array([1 if c in FASTBALL_FAMILY else 0 for c in classes])

    print("Loading arrays...")
    keys = ["seq", "ctx", "pitcher_id", "batter_id", "pitcher_hand",
            "batter_hand", "park_id", "catcher_id", "arsenal_mask"]
    X = {k: np.load(DATA_DIR / f"X_{k}.npy") for k in keys}
    y = np.load(DATA_DIR / "y_labels.npy")

    idx = np.arange(len(y))
    idx_tr, idx_val = train_test_split(
        idx, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, stratify=y)

    print("Predicting on held-out rows...")
    model = tf.keras.models.load_model(
        DATA_DIR / "best_model_v5.keras", compile=False)
    probs = model.predict([X[k][idx_val] for k in keys],
                          batch_size=2048, verbose=0)

    yv = y[idx_val]
    pid = X["pitcher_id"][idx_val]
    bhand = X["batter_hand"][idx_val]

    # Binary gear-up decision — the one a hitter actually makes.
    p_hard = probs[:, fam == 1].sum(1)
    hard_true = fam[yv]
    hard_pred = (p_hard >= 0.5).astype(int)
    hard_conf = np.maximum(p_hard, 1 - p_hard)

    # The pitcher's own base rate, training rows only: the scouting report.
    n_pid = int(X["pitcher_id"].max()) + 1
    counts = np.zeros((n_pid, K))
    np.add.at(counts, (X["pitcher_id"][idx_tr], y[idx_tr]), 1.0)
    league = np.bincount(y[idx_tr], minlength=K).astype(float)
    league /= league.sum()
    totals = counts.sum(1, keepdims=True)
    prior_tbl = _smoothed(counts, totals, league)
    overall_hard_rate = prior_tbl[:, fam == 1].sum(1)

    # The honest bar is the COUNT-SPLIT scouting report, not the pitcher's
    # overall mix — any advance scout already knows he goes fastball 3-1.
    # Built from training rows only.
    def levels_on(a):
        return {v: i for i, v in enumerate(sorted(np.unique(a)))}
    b_all = levels_on(X["ctx"][:, 0])
    s_all = levels_on(X["ctx"][:, 1])
    balls_tr = np.array([b_all[v] for v in X["ctx"][idx_tr, 0]])
    strikes_tr = np.array([s_all[v] for v in X["ctx"][idx_tr, 1]])
    hard_tr = fam[y[idx_tr]]
    pid_tr_a = X["pitcher_id"][idx_tr]
    nb, ns = max(b_all.values()) + 1, max(s_all.values()) + 1
    cnt_hard = np.zeros((n_pid, nb, ns))
    cnt_tot = np.zeros((n_pid, nb, ns))
    np.add.at(cnt_hard, (pid_tr_a, balls_tr, strikes_tr), hard_tr)
    np.add.at(cnt_tot, (pid_tr_a, balls_tr, strikes_tr), 1.0)

    # Counts: ctx cols 0/1 are standardized balls/strikes; StandardScaler
    # preserves rank, so sorted unique values recover the original levels.
    def levels(col):
        u = sorted(np.unique(X["ctx"][idx_val, col]))
        return {v: i for i, v in enumerate(u)}
    b_lv, s_lv = levels(0), levels(1)
    balls = np.array([b_lv[v] for v in X["ctx"][idx_val, 0]])
    strikes = np.array([s_lv[v] for v in X["ctx"][idx_val, 1]])

    print("Mining situations...")
    cards = defaultdict(list)
    for p in np.unique(pid):
        if p == 0:          # <UNK> bucket is many pitchers pooled; skip
            continue
        pm = pid == p
        for b in range(max(b_lv.values()) + 1):
            for s in range(max(s_lv.values()) + 1):
                for h in np.unique(bhand[pm]):
                    m = pm & (balls == b) & (strikes == s) & (bhand == h)
                    n = int(m.sum())
                    if n < MIN_N:
                        continue
                    # Only the pitches we'd actually speak on.
                    spoke = m & (hard_conf >= MIN_CONF)
                    if spoke.sum() < MIN_N // 2:
                        continue

                    # The rule a hitter memorizes: "in this spot, sit X."
                    lean = int(p_hard[spoke].mean() >= 0.5)
                    # It is only a rule if the confident calls agree.
                    consistency = float((hard_pred[spoke] == lean).mean())
                    if consistency < MIN_CONSISTENCY:
                        continue

                    # Accuracy of FOLLOWING THE RULE, not of the model's
                    # per-pitch call — that is what the hitter experiences.
                    acc = float((hard_true[spoke] == lean).mean())

                    # Bar: the count-split scouting report for this pitcher.
                    tot = cnt_tot[p, b, s]
                    rate = (cnt_hard[p, b, s] / tot if tot >= 20
                            else overall_hard_rate[p])
                    prior_lean = int(rate >= 0.5)
                    pacc = float((hard_true[spoke] == prior_lean).mean())
                    edge = acc - pacc
                    if acc < MIN_ACC or edge < MIN_EDGE:
                        continue

                    # Name the likeliest pitch WITHIN the family we are
                    # telling him to sit on, or the advice contradicts
                    # itself ("sit soft, likeliest fastball").
                    mean_p = probs[spoke].mean(0)
                    side = np.where(fam == lean)[0]
                    top = int(side[mean_p[side].argmax()])
                    cards[int(p)].append({
                        "count": f"{b}-{s}",
                        "batter_hand": HAND.get(int(h), str(h)),
                        "n": n,
                        "advised_pct": float(spoke.sum() / n),
                        "sit": "HARD" if lean == 1 else "SOFT",
                        "likeliest_pitch": classes[top],
                        "accuracy": acc,
                        "base_rate": pacc,
                        "edge": edge,
                        "consistency": consistency,
                    })

    # Rank pitchers by how much total edge their card carries.
    inv = {v: k for k, v in
           joblib.load(DATA_DIR / "pitcher_id_map_v5.pkl").items()}
    names = {}
    if "--names" in sys.argv:
        names = resolve_names([int(inv[p]) for p in cards if p in inv])

    ranked = sorted(cards.items(),
                    key=lambda kv: -sum(r["edge"] * r["n"] for r in kv[1]))

    lines = []
    out = lines.append
    out("=" * 74)
    out("COMMIT CARDS — situations worth sitting on")
    out("=" * 74)
    out("")
    out(f"Every rule below cleared all three bars on HELD-OUT pitches:")
    out(f"  >= {MIN_N} pitches in the situation")
    out(f"  >= {MIN_ACC:.0%} right for a hitter who FOLLOWS THE RULE, on")
    out(f"     pitches where the model is {MIN_CONF:.0%}+ confident")
    out(f"  >= +{MIN_EDGE:.0%} better than this pitcher's COUNT-SPLIT scouting")
    out(f"     report (not his overall mix — scouts already have that)")
    out(f"  >= {MIN_CONSISTENCY:.0%} of the confident calls agreeing, so it is")
    out(f"     a rule and not noise inside the count")
    out("")
    out("SIT HARD = gear up for velocity (four-seam / sinker / cutter).")
    out("SIT SOFT = stay back (slider / curve / change / split).")
    out("If a count is not listed, the model has no edge there — react.")
    out("")
    total_rules = sum(len(v) for v in cards.values())
    out(f"{len(cards)} pitchers carry at least one rule; "
        f"{total_rules} rules total.")

    for p, rules in ranked:
        mlbam = int(inv.get(p, -1))
        label = names.get(mlbam, f"MLBAM {mlbam}")
        out("")
        out("-" * 74)
        out(f"{label}")
        out("-" * 74)
        out(f"{'count':>7}{'vs':>8}{'sit':>7}{'likeliest':>11}"
            f"{'acc':>8}{'base':>8}{'edge':>8}{'pitches':>9}")
        for r in sorted(rules, key=lambda r: -r["edge"]):
            out(f"{r['count']:>7}{r['batter_hand']:>8}{r['sit']:>7}"
                f"{r['likeliest_pitch']:>11}{r['accuracy']:>8.0%}"
                f"{r['base_rate']:>8.0%}{r['edge']:>+8.0%}{r['n']:>9,}")

    if not cards:
        out("")
        out("No situation cleared the bar. The model's edge is real in")
        out("aggregate but too diffuse to localize into per-pitcher rules —")
        out("try loosening MIN_N, or present the aggregate gate instead.")

    text = "\n".join(lines)
    print(text)
    (DATA_DIR / "commit_cards_v5.txt").write_text(text + "\n")
    payload = {
        "thresholds": {"min_n": MIN_N, "min_conf": MIN_CONF,
                       "min_acc": MIN_ACC, "min_edge": MIN_EDGE},
        "pitchers": [
            {"pitcher_id": p, "mlbam": int(inv.get(p, -1)),
             "name": names.get(int(inv.get(p, -1))), "rules": r}
            for p, r in ranked
        ],
    }
    (DATA_DIR / "commit_cards_v5.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {DATA_DIR / 'commit_cards_v5.txt'} and .json")


if __name__ == "__main__":
    main()
