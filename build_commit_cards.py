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
    DATA_DIR, SPLIT_SEED, SPLIT_TEST_SIZE, FASTBALL_FAMILY, load_split, _smoothed,
    fit_temperature, apply_temperature,
)

# A rule only earns a place on a card if it clears all three. These match
# analyze_actionability.py — the gate that validated the model.
# Validation is 20% of the data, so a (pitcher x count x hand) cell holds
# ~9 pitches on average. A fixed MIN_N large enough to trust is therefore
# larger than any cell that exists — the first version used 120 and
# returned zero rules for that reason alone. Use a significance test
# instead: a small cell can still earn a rule if the effect is large.
MIN_CELL = 60        # held-out pitches in the situation; the rule is
                     # scored over ALL of them, not a confident subset
MIN_SPOKE = 25       # floor below which no test is meaningful
MIN_TRAIN = 40       # training pitches needed to trust a count-split rate
Z_CRIT = 1.645       # one-sided 95%: the rule really beats the scouting report
MIN_CONF = 0.65      # the commit threshold the gate found actionable
MIN_ACC = 0.70       # a hitter following the rule is right this often
MIN_EDGE = 0.02      # and beats the count-split scouting report
MIN_CONSISTENCY = 0.70   # the confident calls must agree with each other,
# or it is not a rule anyone can memorize — it is situation-dependent
# noise inside the count, and a single "sit" label would mislead.

# Splitting by batter hand thins every cell 3x. Off by default; the hand
# split only survives for the highest-volume starters.
BY_HAND = "--by-hand" in sys.argv
CALIBRATE = "--raw" not in sys.argv

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

    idx_tr, idx_val, split_desc = load_split(y)
    print(f"Split: {split_desc}   Validation rows: {len(idx_val):,}")

    print("Predicting on held-out rows...")
    model = tf.keras.models.load_model(
        DATA_DIR / "best_model_v5.keras", compile=False)
    probs = model.predict([X[k][idx_val] for k in keys],
                          batch_size=2048, verbose=0)

    yv = y[idx_val]
    if CALIBRATE:
        # Fit T on the first half, apply to all: one parameter over
        # hundreds of thousands of rows, but keep the split honest anyway.
        half = len(yv) // 2
        temp, _ = fit_temperature(probs[:half], yv[:half])
        probs = apply_temperature(probs, temp)
        print(f"Temperature calibration: T={temp:.3f} "
              f"({'sharpened' if temp < 1 else 'softened'})")
    else:
        temp = 1.0
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
    # Reuse the SAME level maps the training counts were built with, or a
    # cell lookup can silently index the wrong count.
    balls = np.array([b_all[v] for v in X["ctx"][idx_val, 0]])
    strikes = np.array([s_all[v] for v in X["ctx"][idx_val, 1]])
    b_lv, s_lv = b_all, s_all

    print("Mining situations...")
    cards = defaultdict(list)
    for p in np.unique(pid):
        if p == 0:          # <UNK> bucket is many pitchers pooled; skip
            continue
        pm = pid == p
        for b in range(max(b_lv.values()) + 1):
            for s in range(max(s_lv.values()) + 1):
                for h in (np.unique(bhand[pm]) if BY_HAND else [None]):
                    m = pm & (balls == b) & (strikes == s)
                    if h is not None:
                        m = m & (bhand == h)
                    n = int(m.sum())
                    if n < MIN_CELL:
                        continue

                    # A card instructs the hitter to sit on EVERY pitch in
                    # this situation — he cannot know in the box which ones
                    # the model would have flagged. So the rule must be
                    # scored over the whole cell. Scoring it on the
                    # model-confident subset instead lets the model pick
                    # the sample its own baseline is judged on, which
                    # manufactured +65% "edges" in the first card run.
                    lean = int(p_hard[m].mean() >= 0.5)
                    consistency = float((hard_pred[m] == lean).mean())
                    if consistency < MIN_CONSISTENCY:
                        continue

                    # The scouting report's call for the same situation.
                    tot = cnt_tot[p, b, s]
                    rate = (cnt_hard[p, b, s] / tot if tot >= MIN_TRAIN
                            else overall_hard_rate[p])
                    prior_lean = int(rate >= 0.5)

                    # A card is only worth printing if it CONTRADICTS the
                    # report. If it agrees, the hitter already had it.
                    if lean == prior_lean:
                        continue

                    acc = float((hard_true[m] == lean).mean())
                    pacc = 1.0 - acc     # they disagree, so this is exact
                    edge = acc - pacc
                    if acc < MIN_ACC or edge < MIN_EDGE:
                        continue

                    # Since the two sides disagree, the real question is
                    # whether the card's side is genuinely the majority.
                    z = (acc - 0.5) / np.sqrt(0.25 / n)
                    if z < Z_CRIT:
                        continue

                    # Supplementary: how it does on the confident subset.
                    spoke = m & (hard_conf >= MIN_CONF)
                    ns = int(spoke.sum())
                    acc_conf = (float((hard_true[spoke] == lean).mean())
                                if ns else float("nan"))

                    # Name the likeliest pitch WITHIN the family we are
                    # telling him to sit on, or the advice contradicts
                    # itself ("sit soft, likeliest fastball").
                    mean_p = probs[m].mean(0)
                    side = np.where(fam == lean)[0]
                    top = int(side[mean_p[side].argmax()])
                    cards[int(p)].append({
                        "count": f"{b}-{s}",
                        "batter_hand": HAND.get(int(h), str(h)) if h is not None else "ALL",
                        "n": n,
                        "n_confident": ns,
                        "acc_confident": acc_conf,
                        "sit": "HARD" if lean == 1 else "SOFT",
                        "likeliest_pitch": classes[top],
                        "accuracy": acc,
                        "base_rate": pacc,
                        "edge": edge,
                        "consistency": consistency,
                        "z": float(z),
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
    out("Every rule below is a spot where the model CONTRADICTS the")
    out("count-split scouting report — and is right to. Scored over every")
    out("pitch in the situation, not a subset the model picked.")
    out("")
    out("  'card'    how often sitting as the card says is right")
    out("  'report'  how often the scouting report is right in that spot")
    out("  '(conf)'  same rule, restricted to pitches the model flags —")
    out("            for a live tool; a hitter in the box cannot use it")
    out("")
    out("card + report always sum to 100%: they are opposing constant calls")
    out("on the same pitches. That is arithmetic, not a coincidence — the")
    out("card's claim is simply that the report is on the wrong side here.")
    out("")
    out(f"Bars: >= {MIN_CELL} held-out pitches in the situation, card right")
    out(f"  >= {MIN_ACC:.0%} of the time, significant at one-sided 95% "
        f"(z >= {Z_CRIT})")
    out(f"  and >= {MIN_CONSISTENCY:.0%} of the model's calls in the cell agree,")
    out(f"  so it is a rule and not noise inside the count.")
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
            f"{'card':>7}{'report':>8}{'pitches':>9}{'(conf)':>9}")
        for r in sorted(rules, key=lambda r: -r["accuracy"]):
            cf = (f"{r['acc_confident']:.0%}" if r['acc_confident'] == r['acc_confident']
                  else "-")
            out(f"{r['count']:>7}{r['batter_hand']:>8}{r['sit']:>7}"
                f"{r['likeliest_pitch']:>11}{r['accuracy']:>7.0%}"
                f"{r['base_rate']:>8.0%}{r['n']:>9,}{cf:>9}")

    if not cards:
        out("")
        out("No situation cleared the bar. The aggregate edge is real but")
        out("too diffuse to localize per pitcher. Try --by-hand off (already")
        out("default), or fall back to the LEAGUE-WIDE count rules, which the")
        out("actionability report shows are strongest in 2-2, 0-1 and 1-1.")

    text = "\n".join(lines)
    print(text)
    (DATA_DIR / "commit_cards_v5.txt").write_text(text + "\n")
    payload = {
        "thresholds": {"min_cell": MIN_CELL, "min_conf": MIN_CONF,
                       "min_acc": MIN_ACC, "min_edge": MIN_EDGE,
                       "z_crit": Z_CRIT, "temperature": temp},
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
