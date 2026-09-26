"""
NextPitchAI v6 — Evaluate a saved model
========================================
Regenerates the full evaluation report from the saved best model and the
data_v5/ arrays, WITHOUT retraining. Reproduces the exact train/val split
used by 03_train.py (same seed, same stratification), so the numbers are
identical to what training printed.

Use it when the terminal output is gone, or to re-score a model later.
03_train.py imports `build_report` from here so the two never drift.

Usage:
    python evaluate_model.py            # -> prints + data_v5/eval_report_v5.txt

Takes a few minutes (one predict pass over ~430K validation rows).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_v5"
REPORT_PATH = DATA_DIR / "eval_report_v5.txt"

# Must match 03_train.py exactly or the split (and the numbers) change.
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42


def load_split(y_labels, data_dir: Path = DATA_DIR):
    """
    Reproduce the split the saved model was actually trained with.

    03_train.py records it in split_v5.json. Every script that scores a
    saved model must come through here — a hardcoded random split applied
    to a temporally-trained model silently scores it on rows it trained
    on, which inflates every number downstream and looks entirely
    plausible in the output.

    Returns (idx_tr, idx_val, split_desc).
    """
    split_path = data_dir / "split_v5.json"
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    mode = split.get("mode", "random")
    idx = np.arange(len(y_labels))
    if mode == "temporal":
        seasons = np.load(data_dir / "X_season.npy")
        holdout = int(split["holdout_season"])
        idx_tr, idx_val = idx[seasons < holdout], idx[seasons == holdout]
        desc = f"temporal, validate on {holdout}"
    else:
        idx_tr, idx_val = train_test_split(
            idx, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED,
            stratify=y_labels)
        desc = f"random {1 - SPLIT_TEST_SIZE:.0%}/{SPLIT_TEST_SIZE:.0%}, seed {SPLIT_SEED}"
    if split and len(idx_val) != split.get("n_val", len(idx_val)):
        raise SystemExit(
            f"split mismatch: reproduced {len(idx_val):,} val rows but the run "
            f"recorded {split['n_val']:,}. data_v5/ changed since training — "
            f"re-run 03_train.py.")
    if not split:
        print("WARNING: no split_v5.json — assuming the random split. If this "
              "model was trained temporally the numbers below are invalid.")
    return idx_tr, idx_val, desc


def _topk_acc(probs, y, k):
    """Fraction of rows whose true class is in the top-k of probs."""
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.any(y[:, None] == topk, axis=1).mean())


def _log_loss(probs, y):
    """Mean negative log-likelihood of the true class."""
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def _pitcher_prior_table(pid_tr, y_tr, n_classes, n_pitchers):
    """
    Each pitcher's pitch-type distribution over TRAINING rows, as a
    probability table. Pitchers unseen in training fall back to the
    league distribution. This is the honest bar for the model: what you
    can predict knowing only who is pitching, with no game context.
    """
    counts = np.zeros((n_pitchers, n_classes), dtype=np.float64)
    np.add.at(counts, (pid_tr, y_tr), 1.0)
    league = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    league /= league.sum()
    totals = counts.sum(axis=1, keepdims=True)
    return np.where(totals > 0, counts / np.maximum(totals, 1.0), league)


# Fastball family, for the literature-comparable binary collapse. Most
# published next-pitch numbers are fastball-vs-rest, NOT 10-class, so the
# 10-class top-1 cannot be compared to them directly.
FASTBALL_FAMILY = {"FF", "SI", "FC"}

# Pseudo-counts for the smoothed baseline, matching ARSENAL_SMOOTHING in
# 02_preprocess.py. The model's own arsenal_prior feature is smoothed this
# way; scoring the baseline unsmoothed while the masked softmax can never
# output a zero makes the log-loss comparison asymmetric in the model's
# favour. Both are reported so runs 5-7 stay comparable.
PRIOR_SMOOTHING = 25.0


def _smoothed(counts, totals, league, sm=PRIOR_SMOOTHING):
    return (counts + sm * league) / (totals + sm)


def fit_temperature(probs, y, grid=None):
    """
    One-parameter calibration. Focal loss leaves this model badly
    UNDER-confident (it says 64% and is right 84%), which pushes pitches
    it actually knows below any commit threshold. Rescaling the logits by
    T<1 sharpens them back to honest.

    Probabilities in, probabilities out: log(p) recovers the logits up to
    a constant, which softmax is invariant to. Masked classes have p=0 ->
    -inf, so they stay masked.
    """
    if grid is None:
        grid = np.linspace(0.30, 2.00, 69)
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    rows = np.arange(len(y))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        z = logp / t
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        e /= e.sum(1, keepdims=True)
        nll = float(-np.log(np.clip(e[rows, y], 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t, best_nll


def apply_temperature(probs, t):
    z = np.log(np.clip(probs, 1e-12, 1.0)) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _binary_collapse(probs, y, classes):
    """Fastball-family vs rest: sum probability within each group."""
    fam = np.array([1 if c in FASTBALL_FAMILY else 0 for c in classes])
    p_fb = probs[:, fam == 1].sum(axis=1)
    return (p_fb >= 0.5).astype(int), fam[y]


def build_report(y_val, y_pred_probs, pid_tr, y_tr, pid_val, pitch_classes,
                 history=None, split_desc=None) -> str:
    """
    The evaluation block, as one string: the model vs the pitcher-prior
    baseline on top-1/top-3/log-loss, a per-type classification report,
    confusion matrix, per-class accuracy, and (if given) a training
    summary from the Keras history dict.
    """
    lines = []
    out = lines.append

    y_pred = np.argmax(y_pred_probs, axis=1)
    n_classes = y_pred_probs.shape[1]

    # Baseline: the pitcher's own training distribution, used as a full
    # probabilistic predictor rather than just its argmax. Comparing
    # log-loss against it answers the question that matters — does game
    # context add anything beyond knowing who is on the mound?
    n_pitchers = int(max(pid_tr.max(), pid_val.max())) + 1
    counts = np.zeros((n_pitchers, n_classes), dtype=np.float64)
    np.add.at(counts, (pid_tr, y_tr), 1.0)
    league = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    league /= league.sum()
    totals = counts.sum(axis=1, keepdims=True)

    prior_probs = np.where(totals > 0, counts / np.maximum(totals, 1.0), league)[pid_val]
    smooth_probs = _smoothed(counts, totals, league)[pid_val]

    m_top1, m_top3 = _topk_acc(y_pred_probs, y_val, 1), _topk_acc(y_pred_probs, y_val, 3)
    b_top1, b_top3 = _topk_acc(prior_probs, y_val, 1), _topk_acc(prior_probs, y_val, 3)
    s_top1, s_top3 = _topk_acc(smooth_probs, y_val, 1), _topk_acc(smooth_probs, y_val, 3)
    m_ll = _log_loss(y_pred_probs, y_val)
    b_ll = _log_loss(prior_probs, y_val)
    s_ll = _log_loss(smooth_probs, y_val)

    # How many rows is the raw baseline structurally unable to score? Each
    # is clipped to 1e-12 (27.6 nats) while the masked model can never be.
    n_zero = int((prior_probs[np.arange(len(y_val)), y_val] == 0).sum())

    out("=" * 50)
    out("EVALUATION (natural distribution)")
    out("=" * 50)
    # Stamp the split into the report. Two runs that differ only by split
    # produce reports that are otherwise near-identical in shape, and a
    # config edit that silently fails to take is invisible without this.
    out(f"\nSplit: {split_desc or 'UNRECORDED'}")
    out(f"Validation rows: {len(y_val):,}")
    if history:
        # With a second head Keras renames these: val_output_loss is the
        # type head, val_loss becomes the weighted total.
        lk = "val_output_loss" if "val_output_loss" in history else "val_loss"
        ak = ("val_output_accuracy" if "val_output_accuracy" in history
              else "val_accuracy")
        n_epochs = len(history[lk])
        best = int(np.argmin(history[lk]))
        acc = f", val_acc there {history[ak][best]:.1%}" if ak in history else ""
        out(f"\nTraining: {n_epochs} epochs run, best {lk} "
            f"{history[lk][best]:.4f} at epoch {best + 1}{acc}")

    out("\nModel vs pitcher-prior baseline")
    out("(baseline = this pitcher's training pitch mix, no game context.")
    out(" 'smoothed' backs off toward the league mix so the baseline cannot")
    out(" be charged 27.6 nats for a pitch it never saw; the masked model")
    out(" structurally never can be. Smoothed is the fair log-loss bar.)")
    out(f"{'':<12}{'model':>10}{'prior':>10}{'lift':>9}"
        f"{'smoothed':>11}{'lift':>9}")
    out(f"{'top-1':<12}{m_top1:>9.1%}{b_top1:>10.1%}{m_top1 - b_top1:>+9.1%}"
        f"{s_top1:>11.1%}{m_top1 - s_top1:>+9.1%}")
    out(f"{'top-3':<12}{m_top3:>9.1%}{b_top3:>10.1%}{m_top3 - b_top3:>+9.1%}"
        f"{s_top3:>11.1%}{m_top3 - s_top3:>+9.1%}")
    out(f"{'log-loss':<12}{m_ll:>10.4f}{b_ll:>10.4f}{b_ll - m_ll:>+9.4f}"
        f"{s_ll:>11.4f}{s_ll - m_ll:>+9.4f}   (positive = model better)")
    out(f"\nRows the raw baseline scores as exactly 0 (clipped to 27.6 nats): "
        f"{n_zero:,} ({n_zero / len(y_val):.3%})")
    out(f"  Those alone move the raw log-loss lift by "
        f"~{n_zero * 27.631 / len(y_val):.4f} nats.")

    # Fastball-family vs rest — what most published work actually measures.
    mb, yb = _binary_collapse(y_pred_probs, y_val, pitch_classes)
    bb, _ = _binary_collapse(prior_probs, y_val, pitch_classes)
    out("\nFastball-family (FF/SI/FC) vs rest — comparable to published work,")
    out("most of which reports this binary task rather than 10-class:")
    out(f"  model {(mb == yb).mean():.1%}   baseline {(bb == yb).mean():.1%}"
        f"   lift {(mb == yb).mean() - (bb == yb).mean():+.1%}")

    present = sorted(set(y_val.tolist()) | set(y_pred.tolist()))
    present_names = [pitch_classes[i] for i in present]

    out("\nClassification Report:")
    out(classification_report(
        y_val, y_pred, labels=present, target_names=present_names,
        digits=3, zero_division=0,
    ))

    out("\nConfusion Matrix (rows = actual, cols = predicted):")
    cm = confusion_matrix(y_val, y_pred, labels=present)
    out(pd.DataFrame(cm, index=present_names, columns=present_names).to_string())

    out("\nPer-class accuracy:")
    for i in present:
        mask = y_val == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            out(f"  {pitch_classes[i]}: {acc:.1%} ({mask.sum():,} samples)")

    return "\n".join(lines)


def build_location_report(z_val, zone_probs, w_val, pid_tr, z_tr, w_tr,
                          pid_val, zone_classes) -> str:
    """
    The attack-zone head, held to the same bar as the type head.

    Accuracy alone cannot judge this head. `shadow` is the plurality zone
    in almost every conditioning cell, so an argmax that never leaves it
    is the arithmetically correct response to a mildly informative
    distribution — the head can carry real information and still show
    0.0% recall on the other three zones. Log-loss moves when accuracy
    structurally cannot, so it is the test of whether anything was
    learned; the heart-vs-rest collapse is the test of whether what was
    learned is worth telling a hitter.
    """
    lines = []
    out = lines.append
    n_zone = zone_probs.shape[1]
    names = list(zone_classes) if zone_classes else [str(i) for i in range(n_zone)]

    m = w_val > 0
    mt = w_tr > 0
    zp_probs = zone_probs[m]
    zp = zp_probs.argmax(1)
    zt = z_val[m]
    ztr = z_tr[mt]
    pid_v = pid_val[m]

    out("=" * 50)
    out("LOCATION HEAD (attack zone)")
    out("=" * 50)
    out(f"\nScored on the {m.sum():,} validation pitches with a usable zone.")
    out("Location is far noisier than type — a pitcher aims and misses — so")
    out("judge this against the baselines, not in absolute terms.\n")

    # --- Baselines, as full distributions so log-loss is comparable ---
    league = np.bincount(ztr, minlength=n_zone).astype(np.float64)
    league /= league.sum()
    npid = int(max(pid_tr.max(), pid_val.max())) + 1
    counts = np.zeros((npid, n_zone), dtype=np.float64)
    np.add.at(counts, (pid_tr[mt], ztr), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    pit_probs = _smoothed(counts, totals, league)[pid_v]
    lg_probs = np.repeat(league[None, :], len(zt), axis=0)

    league_lean = int(league.argmax())
    pit_lean = np.where(totals[:, 0] > 0, counts.argmax(1), league_lean).astype(int)

    acc = float((zp == zt).mean())
    b_league = float((zt == league_lean).mean())
    b_pitcher = float((zt == pit_lean[pid_v]).mean())
    m_ll = _log_loss(zp_probs, zt)
    l_ll = _log_loss(lg_probs, zt)
    p_ll = _log_loss(pit_probs, zt)

    out(f"{'':<26}{'accuracy':>10}{'vs model':>10}{'log-loss':>11}{'vs model':>10}")
    out(f"{'model':<26}{acc:>10.1%}{'':>10}{m_ll:>11.4f}{'':>10}")
    out(f"{'league zone mix':<26}{b_league:>10.1%}{acc - b_league:>+10.1%}"
        f"{l_ll:>11.4f}{l_ll - m_ll:>+10.4f}")
    out(f"{'pitcher zone mix':<26}{b_pitcher:>10.1%}{acc - b_pitcher:>+10.1%}"
        f"{p_ll:>11.4f}{p_ll - m_ll:>+10.4f}")
    out("  (log-loss: positive = model better. Accuracy can sit exactly on")
    out("   the baseline while log-loss moves — that means real but weak")
    out("   signal, not a dead head.)")

    # --- The hitter's actual question: is this one hittable? ---
    # heart == over the plate. Everything else is protect or take. This is
    # the location analogue of the hard/soft collapse on the type head.
    hi = names.index("heart") if "heart" in names else 0
    p_heart = zp_probs[:, hi]
    is_heart = (zt == hi).astype(np.int64)
    base = float(is_heart.mean())
    out(f"\nheart vs rest — 'is this one over the plate?', the location")
    out(f"analogue of the hard/soft call. Base rate {base:.1%}.")
    out(f"{'threshold':<14}{'advised':>10}{'P(heart)':>11}{'actual':>10}{'vs base':>10}")
    for thr in (0.30, 0.35, 0.40, 0.50):
        sel = p_heart >= thr
        if sel.sum() == 0:
            out(f"{'>= ' + format(thr, '.2f'):<14}{0.0:>10.1%}{'-':>11}{'-':>10}{'-':>10}")
            continue
        hit = float(is_heart[sel].mean())
        out(f"{'>= ' + format(thr, '.2f'):<14}{sel.mean():>10.1%}"
            f"{p_heart[sel].mean():>11.1%}{hit:>10.1%}{hit - base:>+10.1%}")
    # The other tail is advice too: a pitch the model says is very unlikely
    # to be over the plate is a take.
    for thr in (0.15, 0.10):
        sel = p_heart <= thr
        if sel.sum() == 0:
            continue
        hit = float(is_heart[sel].mean())
        out(f"{'<= ' + format(thr, '.2f'):<14}{sel.mean():>10.1%}"
            f"{p_heart[sel].mean():>11.1%}{hit:>10.1%}{hit - base:>+10.1%}")

    out("\nPer-zone recall:")
    for i, zc in enumerate(names):
        sel = zt == i
        if sel.sum():
            out(f"  {str(zc):<8}{(zp[sel] == i).mean():>7.1%} "
                f"({sel.sum():,} pitches, {sel.mean():.1%} of all)")

    out("\nConfusion (rows actual, cols predicted):")
    cmz = np.zeros((n_zone, n_zone), int)
    np.add.at(cmz, (zt, zp), 1)
    out(pd.DataFrame(cmz, index=names, columns=names).to_string())

    return "\n".join(lines)


def write_report(text: str, path: Path = REPORT_PATH) -> None:
    path.write_text(text + "\n")
    print(f"\nReport written to {path}")


def main():
    import tensorflow as tf

    with open(DATA_DIR / "meta_v5.json") as f:
        meta = json.load(f)
    pitch_classes = meta["pitch_classes"]

    print("Loading arrays...")
    X = {
        "seq": np.load(DATA_DIR / "X_seq.npy"),
        "ctx": np.load(DATA_DIR / "X_ctx.npy"),
        "pitcher_id": np.load(DATA_DIR / "X_pitcher_id.npy"),
        "batter_id": np.load(DATA_DIR / "X_batter_id.npy"),
        "pitcher_hand": np.load(DATA_DIR / "X_pitcher_hand.npy"),
        "batter_hand": np.load(DATA_DIR / "X_batter_hand.npy"),
        "park_id": np.load(DATA_DIR / "X_park_id.npy"),
        "catcher_id": np.load(DATA_DIR / "X_catcher_id.npy"),
        "arsenal_mask": np.load(DATA_DIR / "X_arsenal_mask.npy"),
    }
    y_labels = np.load(DATA_DIR / "y_labels.npy")

    idx_tr, idx_val, split_desc = load_split(y_labels)
    print(f"Split: {split_desc}   Validation rows: {len(idx_val):,}")

    model_path = DATA_DIR / "best_model_v5.keras"
    print(f"Loading {model_path.name} ...")
    # compile=False: the custom focal loss isn't needed for inference.
    model = tf.keras.models.load_model(model_path, compile=False)

    val_inputs = [
        X["seq"][idx_val], X["ctx"][idx_val],
        X["pitcher_id"][idx_val], X["batter_id"][idx_val],
        X["pitcher_hand"][idx_val], X["batter_hand"][idx_val],
        X["park_id"][idx_val], X["catcher_id"][idx_val],
        X["arsenal_mask"][idx_val],
    ]
    print("Predicting...")
    pred = model.predict(val_inputs, batch_size=2048, verbose=0)
    # A two-head model returns a dict; a single-head one returns an array.
    if isinstance(pred, dict):
        y_pred_probs, zone_probs = pred["output"], pred.get("zone_output")
    else:
        y_pred_probs, zone_probs = pred, None

    history = None
    hist_path = DATA_DIR / "training_history_v5.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)

    report = build_report(
        y_labels[idx_val], y_pred_probs,
        X["pitcher_id"][idx_tr], y_labels[idx_tr],
        X["pitcher_id"][idx_val], pitch_classes, history, split_desc,
    )

    zone_path = DATA_DIR / "y_zone.npy"
    if zone_probs is not None and zone_path.exists():
        y_zone = np.load(zone_path)
        zr, zv = y_zone[idx_tr], y_zone[idx_val]
        report += "\n\n" + build_location_report(
            np.where(zv >= 0, zv, 0).astype(np.int64), zone_probs,
            (zv >= 0).astype(np.float32),
            X["pitcher_id"][idx_tr],
            np.where(zr >= 0, zr, 0).astype(np.int64),
            (zr >= 0).astype(np.float32),
            X["pitcher_id"][idx_val],
            meta.get("zone_classes", []),
        )
    print("\n" + report)
    write_report(report)


if __name__ == "__main__":
    main()
