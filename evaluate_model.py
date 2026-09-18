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


def build_report(y_val, y_pred_probs, pid_tr, y_tr, pid_val, pitch_classes,
                 history=None) -> str:
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
    prior_table = _pitcher_prior_table(pid_tr, y_tr, n_classes, n_pitchers)
    prior_probs = prior_table[pid_val]

    m_top1, m_top3 = _topk_acc(y_pred_probs, y_val, 1), _topk_acc(y_pred_probs, y_val, 3)
    b_top1, b_top3 = _topk_acc(prior_probs, y_val, 1), _topk_acc(prior_probs, y_val, 3)
    m_ll, b_ll = _log_loss(y_pred_probs, y_val), _log_loss(prior_probs, y_val)

    out("=" * 50)
    out("EVALUATION (natural distribution)")
    out("=" * 50)
    if history:
        n_epochs = len(history["val_loss"])
        best = int(np.argmin(history["val_loss"]))
        out(f"\nTraining: {n_epochs} epochs run, best val_loss "
            f"{history['val_loss'][best]:.4f} at epoch {best + 1}, "
            f"val_acc there {history['val_accuracy'][best]:.1%}")

    out("\nModel vs pitcher-prior baseline")
    out("(baseline = this pitcher's training pitch mix, no game context)")
    out(f"{'':<12}{'model':>10}{'baseline':>12}{'lift':>10}")
    out(f"{'top-1':<12}{m_top1:>9.1%}{b_top1:>12.1%}{m_top1 - b_top1:>+10.1%}")
    out(f"{'top-3':<12}{m_top3:>9.1%}{b_top3:>12.1%}{m_top3 - b_top3:>+10.1%}")
    out(f"{'log-loss':<12}{m_ll:>10.4f}{b_ll:>12.4f}{b_ll - m_ll:>+10.4f}"
        "   (positive = model better)")

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

    # Same split as 03_train.py -> identical validation rows.
    indices = np.arange(len(y_labels))
    idx_tr, idx_val = train_test_split(
        indices, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED,
        stratify=y_labels)
    print(f"Validation rows: {len(idx_val):,}")

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
    y_pred_probs = model.predict(val_inputs, batch_size=2048, verbose=0)

    history = None
    hist_path = DATA_DIR / "training_history_v5.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)

    report = build_report(
        y_labels[idx_val], y_pred_probs,
        X["pitcher_id"][idx_tr], y_labels[idx_tr],
        X["pitcher_id"][idx_val], pitch_classes, history,
    )
    print("\n" + report)
    write_report(report)


if __name__ == "__main__":
    main()
