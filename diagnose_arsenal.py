"""
NextPitchAI — arsenal / evaluation diagnostics
===============================================
Answers, empirically, the questions the reported metrics cannot:

  A. How much does the arsenal mask actually constrain the model?
  B. Is the model-vs-baseline comparison apples-to-apples?
  C. What is the ceiling, given how much pitchers randomize?
  D. Where does the model beat the prior, and where does it not?

Parts A-C are pure numpy over data_v5/ and run in seconds.
Part D loads the saved model and takes a few minutes.

Usage:
    .venv/bin/python diagnose_arsenal.py          # A-C only (fast)
    .venv/bin/python diagnose_arsenal.py --model  # also D
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_v5"
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42


def hdr(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main():
    with open(DATA_DIR / "meta_v5.json") as f:
        meta = json.load(f)
    classes = meta["pitch_classes"]
    K = len(classes)

    mask = np.load(DATA_DIR / "X_arsenal_mask.npy")
    pid = np.load(DATA_DIR / "X_pitcher_id.npy")
    y = np.load(DATA_DIR / "y_labels.npy")
    N = len(y)

    idx = np.arange(N)
    idx_tr, idx_val = train_test_split(
        idx, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, stratify=y)

    # Rows share a mask per pitcher; collapse to unique pitchers via the mask
    # bit-pattern so "per pitcher" stats are not weighted by workload.
    packed = (mask > 0).dot(1 << np.arange(K))
    _, first_idx, inv = np.unique(packed, return_index=True, return_inverse=True)
    n_repertoires = len(first_idx)

    # ---------------------------------------------------------------
    hdr("A. Does the arsenal mask actually constrain anything?")
    sizes_per_row = mask.sum(axis=1)
    sizes_per_rep = mask[first_idx].sum(axis=1)
    print(f"Pitch classes available          : {K}")
    print(f"Distinct arsenal patterns        : {n_repertoires:,}")
    print(f"Mean arsenal size (per pitch)    : {sizes_per_row.mean():.2f} of {K}")
    print(f"Mean arsenal size (per repertoire): {sizes_per_rep.mean():.2f} of {K}")
    print(f"Median / p90 / max (per pitch)   : {np.median(sizes_per_row):.0f}"
          f" / {np.percentile(sizes_per_row, 90):.0f} / {sizes_per_row.max():.0f}")
    print("\nDistribution of arsenal size, weighted by pitches thrown:")
    for s in range(1, K + 1):
        n = int((sizes_per_row == s).sum())
        if n:
            print(f"  {s:2d} types: {n / N:6.1%} of pitches")
    print(f"\n-> A uniform guess inside the mask would score "
          f"{np.mean(1.0 / sizes_per_row):.1%} top-1.")
    print(f"-> Masking removes {1 - sizes_per_row.mean() / K:.0%} of the raw "
          f"{K}-way search space on average.")

    # ---------------------------------------------------------------
    hdr("B. How much of each mask is real vs. statistical noise?")
    # Per-pitcher usage shares, over TRAINING rows only (no leakage).
    n_pid = int(pid.max()) + 1
    counts = np.zeros((n_pid, K))
    np.add.at(counts, (pid[idx_tr], y[idx_tr]), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, np.maximum(totals, 1.0))

    print("A pitch type is in the mask if thrown >=1 time EVER. How many of")
    print("those are pitches the pitcher actually goes to?\n")
    print(f"{'threshold':>12}{'mean types':>13}{'vs mask':>10}")
    row_shares = shares[pid]
    row_mask = mask > 0
    base = row_mask.sum(axis=1).mean()
    print(f"{'in mask':>12}{base:>13.2f}{'—':>10}")
    for thr in (0.005, 0.01, 0.02, 0.05, 0.10):
        eff = ((row_shares >= thr) & row_mask).sum(axis=1).mean()
        print(f"{thr:>11.1%}{eff:>13.2f}{eff - base:>+10.2f}")

    junk = ((row_shares < 0.01) & row_mask).sum(axis=1).mean()
    print(f"\n-> {junk:.2f} of the {base:.2f} masked types per pitch are used "
          f"<1% of the time ({junk / base:.0%} of the mask).")
    print("   Those are mostly Statcast misclassifications, not real weapons.")

    # Rare-class sanity probe: who is allowed to throw a knuckleball?
    if "KN" in classes:
        kn = classes.index("KN")
        allowed = row_mask[:, kn]
        real = (row_shares[:, kn] >= 0.01) & allowed
        print(f"\nKN probe: {allowed.sum():,} pitches come from pitchers whose "
              f"mask allows KN,")
        print(f"          but only {real.sum():,} from pitchers who throw it "
              f">=1% of the time.")
        print(f"          Actual KN pitches in the data: {(y == kn).sum():,}")

    # ---------------------------------------------------------------
    hdr("C. Is the model-vs-baseline comparison fair?")
    unk_rows = (pid[idx_val] == 0).sum()
    print(f"Validation rows whose pitcher is <UNK> (pitcher_id 0): "
          f"{unk_rows:,} ({unk_rows / len(idx_val):.2%})")
    print("  The BASELINE pools all these pitchers into one blended mix,")
    print("  while the MODEL still gets each one's exact arsenal mask.")
    print("  If this share is large, some of the model's lift is structural.")

    # Baseline zeros: types the pitcher never threw in training but does in val
    prior = np.divide(counts, np.maximum(totals, 1.0))
    prior_val = prior[pid[idx_val]]
    hit = prior_val[np.arange(len(idx_val)), y[idx_val]]
    zeros = (hit == 0).sum()
    print(f"\nValidation pitches the baseline assigns probability EXACTLY 0: "
          f"{zeros:,} ({zeros / len(idx_val):.2%})")
    print("  Each costs the baseline -log(1e-12) = 27.6 nats after clipping.")
    if zeros:
        infl = zeros / len(idx_val) * (27.6 - 2.0)
        print(f"  Rough inflation of the baseline's log-loss: ~{infl:.3f} nats")
        print(f"  Reported gap was 0.157-0.169 nats — so this alone could")
        print(f"  account for {'MOST' if infl > 0.10 else 'part'} of the model's log-loss win.")

    # ---------------------------------------------------------------
    hdr("D. What is the ceiling?")
    # Entropy of each pitcher's own mix = what a perfect prior-only model gets.
    p = prior_val
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    print(f"Mean conditional entropy H(pitch | pitcher) : {ent.mean():.3f} nats")
    print(f"  -> perplexity {np.exp(ent.mean()):.2f}: knowing only who is")
    print(f"     pitching, it is like guessing among "
          f"{np.exp(ent.mean()):.1f} equally likely pitches.")
    print(f"Best possible top-1 from the prior alone   : "
          f"{p.max(axis=1).mean():.1%}")
    print("  (That is the ceiling for ANY model that ignores game context.)")
    print("\nReported model log-loss 1.1531-1.1650 vs prior 1.3221.")
    print(f"Model perplexity ~{np.exp(1.1531):.2f} vs prior "
          f"~{np.exp(1.3221):.2f}.")

    # ---------------------------------------------------------------
    if "--model" in sys.argv:
        hdr("E. Per-pitcher: where does the model beat the prior?")
        import tensorflow as tf
        print("Loading model + arrays...")
        X = {k: np.load(DATA_DIR / f"X_{k}.npy") for k in
             ["seq", "ctx", "pitcher_id", "batter_id", "pitcher_hand",
              "batter_hand", "park_id", "catcher_id", "arsenal_mask"]}
        model = tf.keras.models.load_model(
            DATA_DIR / "best_model_v5.keras", compile=False)
        probs = model.predict(
            [X["seq"][idx_val], X["ctx"][idx_val], X["pitcher_id"][idx_val],
             X["batter_id"][idx_val], X["pitcher_hand"][idx_val],
             X["batter_hand"][idx_val], X["park_id"][idx_val],
             X["catcher_id"][idx_val], X["arsenal_mask"][idx_val]],
            batch_size=2048, verbose=0)

        yv = y[idx_val]
        pv = pid[idx_val]
        m_hit = probs[np.arange(len(yv)), yv]
        b_hit = np.clip(prior_val[np.arange(len(yv)), yv], 1e-12, 1.0)
        m_correct = probs.argmax(1) == yv
        b_correct = prior_val.argmax(1) == yv

        # Does the model beat the prior for most pitchers, or a few?
        better = wins = total = 0
        for u in np.unique(pv):
            sel = pv == u
            if sel.sum() < 200:
                continue
            total += 1
            if m_correct[sel].mean() > b_correct[sel].mean():
                wins += 1
            if (-np.log(m_hit[sel])).mean() < (-np.log(b_hit[sel])).mean():
                better += 1
        print(f"Pitchers with >=200 validation pitches: {total}")
        print(f"  model beats prior on top-1   : {wins}/{total} ({wins/max(total,1):.0%})")
        print(f"  model beats prior on log-loss: {better}/{total} ({better/max(total,1):.0%})")
        print("\n-> If these are near 100%, the gain is broad and real.")
        print("   If near 50%, the model is winning on a subset and losing elsewhere.")

        # Excluding the baseline's zero-probability rows: the fair log-loss
        ok = prior_val[np.arange(len(yv)), yv] > 0
        print(f"\nFair log-loss on the {ok.sum():,} rows where the baseline is")
        print(f"not structurally zero ({ok.mean():.1%} of validation):")
        print(f"  model    : {-np.log(np.clip(m_hit[ok], 1e-12, 1)).mean():.4f}")
        print(f"  baseline : {-np.log(np.clip(b_hit[ok], 1e-12, 1)).mean():.4f}")
        print(f"  top-1  model {m_correct[ok].mean():.1%} vs baseline "
              f"{b_correct[ok].mean():.1%}")
        print("\n-> THIS is the honest headline comparison.")

    print()


if __name__ == "__main__":
    main()
