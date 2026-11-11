"""
NextPitchAI v6 — Evaluate a saved model
========================================
Regenerates the full evaluation report from the saved best model and the
arrays beside it, WITHOUT retraining and without TensorFlow, on the split
03_train.py recorded. 03_train.py imports `build_report` from here so the
two never drift.

By default the model is scored the way the gate scores it and the site
serves it: the arsenal mask as of each pitch, and ids it never trained on
sent to the unknown bucket. Those numbers differ a little from what
training printed, because training validates with the mask frozen at the
end of training and passes every id through. `--as-trained` reproduces
the training-time report exactly, for when the terminal output is gone.

This module is also where every script that scores a model gets its
arsenal mask, its scouting table and its intervals — see "Scoring a
saved model" below. Do not reimplement any of the three elsewhere.

Usage:
    .venv/bin/python evaluate_model.py
        -> prints + data_v5/eval_report_v5.txt

    .venv/bin/python evaluate_model.py --as-trained
        -> the report 03_train.py printed, digit for digit

    .venv/bin/python evaluate_model.py --data-dir data_ext --eval-season 2026
        -> the same report for a season the model was never tuned on

Takes a few minutes per season (one NumPy predict pass).
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from pitch_features import (
    MASK_MIN_HISTORY, causal_arsenal_mask, dense_codes, prior_group_cumsum,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_v5"
REPORT_PATH = DATA_DIR / "eval_report_v5.txt"

# Must match 03_train.py exactly or the split (and the numbers) change.
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42


def build_arsenal_mask(pitcher_ids, labels, idx_tr, n_pitch):
    """
    The arsenal mask, built from TRAINING rows only.

    02_preprocess.py emits a full-data mask, which leaks on a temporal
    split: it reveals pitch types a pitcher only started throwing in the
    held-out season. 03_train.py and the serving bundle both call this so
    the model is served the same mask it was trained with.

    A pitcher with no training rows gets all ones — an unknown arsenal,
    same as an unseen pitcher at inference. A row must never be all
    zeros: every logit would hit the floor and the softmax would be
    meaningless.
    """
    n_pid = int(pitcher_ids.max()) + 1
    counts = np.zeros((n_pid, n_pitch), dtype=np.float32)
    np.add.at(counts, (pitcher_ids[idx_tr], labels[idx_tr]), 1.0)
    tbl = (counts >= 1).astype(np.float32)
    tbl[tbl.sum(1) == 0] = 1.0
    return tbl


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
    # Both halves are checked. A dataset extended forward in frozen mode
    # holds later seasons as well; they fall in neither half, and the
    # counts for the seasons the model knew must still match exactly.
    for name, got in (("n_val", len(idx_val)), ("n_train", len(idx_tr))):
        if split and got != split.get(name, got):
            raise SystemExit(
                f"split mismatch: reproduced {got:,} {name[2:]} rows but the run "
                f"recorded {split[name]:,}. {data_dir.name}/ changed since "
                f"training — re-run 03_train.py.")
    if not split:
        print("WARNING: no split_v5.json — assuming the random split. If this "
              "model was trained temporally the numbers below are invalid.")
    return idx_tr, idx_val, desc


def read_split(data_dir: Path = DATA_DIR) -> dict:
    """The split a model was trained with, as 03_train.py recorded it."""
    path = data_dir / "split_v5.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------
# Scoring a saved model
# ---------------------------------------------------------------------
# Every script that scores a model comes through these, for two reasons.
#
# The arsenal mask. X_arsenal_mask.npy is built over EVERY row, so it
# knows which pitches a pitcher throws in the held-out season. 03_train.py
# replaces it in memory before training, but until 2026-11 every scoring
# script fed the on-disk copy to the model — so the gate that produced
# the site's claim was scored with a mask from the future, on a model
# trained with an honest one. scoring_mask() is now the only source.
#
# TensorFlow. A .keras file is a zip of JSON and HDF5; reading it needs
# neither TensorFlow nor Keras, and a broken TF install aborts the whole
# process rather than raising. Scoring runs on the NumPy forward pass,
# which is verified against Keras to 1e-7.

INPUT_KEYS = ["seq", "ctx", "pitcher_id", "batter_id", "pitcher_hand",
              "batter_hand", "park_id", "catcher_id"]


def load_inputs(data_dir: Path = DATA_DIR) -> dict:
    """Model input arrays, memory-mapped: scoring touches one season."""
    return {k: np.load(data_dir / f"X_{k}.npy", mmap_mode="r") for k in INPUT_KEYS}


def load_row_meta(data_dir: Path = DATA_DIR):
    """Raw pitcher/batter ids, game and integer count per row, or None for
    a dataset built before 02_preprocess.py wrote them."""
    path = data_dir / "row_meta_v5.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_scoring_model(data_dir: Path = DATA_DIR):
    """The saved model as a NumPy forward pass, exported on the spot.

    Exported from data_dir's own .keras every time rather than read from
    site/serving/, so the weights scored are provably the weights in that
    directory and not a bundle left over from an earlier model.
    """
    model_path = data_dir / "best_model_v5.keras"
    if not model_path.exists():
        raise SystemExit(f"no model at {model_path} — train first")
    meta = json.loads((data_dir / "meta_v5.json").read_text())
    want = meta.get("model_sha256")
    if want:
        got = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(
                f"{model_path} is not the model this dataset was extended "
                f"for (sha256 differs). Rebuild it with 02_preprocess.py "
                f"--freeze-from.")
    site = BASE_DIR / "site"
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))
    from export_model import ARCH_NAME, WEIGHTS_NAME, KerasArchive, export
    from numpy_model import NumpyModel
    tmp = Path(tempfile.mkdtemp(prefix="nextpitch_score_"))
    export(KerasArchive(model_path), tmp, source=model_path.name)
    return NumpyModel(tmp / WEIGHTS_NAME, tmp / ARCH_NAME)


ID_KEYS = ["pitcher_id", "batter_id", "park_id", "catcher_id"]


def known_ids(X, idx_tr, keys=ID_KEYS) -> dict:
    """
    Per embedding input, a lookup that keeps an id the model TRAINED on
    and sends every other id to 0, the unknown bucket.

    02_preprocess.py numbers every frequent player over all rows, the
    held-out season included. A pitcher who debuted in that season
    therefore owns an embedding row with no training rows behind it,
    still holding its random initial values. Passing that id through
    feeds the model noise exactly where it was trained to expect the
    unknown bucket. An id is known to the model only if it trained on it.

    The identity on a random split, where every id has training rows.
    """
    out = {}
    for k in keys:
        col = np.asarray(X[k])
        seen = np.zeros(int(col.max()) + 1, dtype=bool)
        seen[col[idx_tr]] = True
        out[k] = np.where(seen, np.arange(len(seen)), 0).astype(col.dtype)
    return out


def predict_rows(model, X, rows, mask_rows, batch: int = 4096, label: str = "",
                 all_heads: bool = False, known: dict = None, edit=None):
    """Class probabilities for `rows`, with `mask_rows` as the arsenal mask.

    `known` is known_ids(): ids the model never trained on go to the
    unknown bucket. Omit it only to reproduce a number measured without.
    `edit(inputs)` may change a batch in place before it is scored, for
    ablations; the arrays it receives are copies.

    With all_heads, a {head name: array} dict of every output the model
    has (the optional zone head included) from the same single pass.
    """
    rows = np.asarray(rows)
    outs = {}
    for i in range(0, len(rows), batch):
        sl = rows[i:i + batch]
        # A season is one contiguous block; a slice reads it without the
        # copy a fancy index makes of every array.
        take = (slice(int(sl[0]), int(sl[-1]) + 1)
                if len(sl) and sl[-1] - sl[0] + 1 == len(sl) else sl)
        inputs = {k: np.asarray(X[k][take]) for k in INPUT_KEYS}
        for k, lut in (known or {}).items():
            inputs[k] = lut[inputs[k]]
        inputs["arsenal_mask"] = np.asarray(mask_rows[i:i + batch], dtype=np.float32)
        if edit is not None:
            inputs = {k: np.array(v) for k, v in inputs.items()}
            edit(inputs)
        heads = model.predict_all(inputs) if all_heads else {"": model.predict(inputs)}
        for name, p in heads.items():
            if name not in outs:
                outs[name] = np.empty((len(rows), p.shape[1]), dtype=np.float32)
            outs[name][i:i + len(sl)] = p
        if label and (i // batch) % 40 == 0:
            print(f"  {label}: {min(i + batch, len(rows)):,}/{len(rows):,}", flush=True)
    if all_heads:
        return outs
    return outs.get("", np.empty((0, 0), dtype=np.float32))


def scoring_mask(pitcher_ids, labels, idx_tr, n_pitch, row_meta=None):
    """
    The arsenal mask a model may honestly be scored with. Returns
    (mask (N, n_pitch) uint8, one-line description).

    With row metadata: the CAUSAL mask — what each pitcher had thrown
    before each pitch. That is what a site refreshed during the season
    knows, and it uses nothing from the future.

    Without it (a dataset older than row_meta_v5.npz): the training-rows
    mask the model was trained with. Honest, but frozen: a pitch added
    after training can never be predicted.

    Never the on-disk X_arsenal_mask.npy.
    """
    if row_meta is not None:
        mask = causal_arsenal_mask(row_meta["pitcher"], labels, n_pitch)
        return mask, (f"causal — pitches thrown before each pitch "
                      f"(all ones until {MASK_MIN_HISTORY} career pitches)")
    tbl = build_arsenal_mask(pitcher_ids, labels, idx_tr, n_pitch)
    return (tbl[pitcher_ids].astype(np.uint8),
            "training rows only (no row metadata in this dataset)")


def full_mask_through(pitcher_raw, labels, seasons, last_season, n_pitch):
    """The LEAKY mask the gate used before 2026-11: every type a pitcher
    threw in any season up to `last_season`, the held-out one included.
    Kept only so the previously published number can be reproduced and
    the size of the correction shown."""
    codes = dense_codes(pitcher_raw)
    rows = seasons <= last_season
    tbl = np.zeros((int(codes.max()) + 1, n_pitch), dtype=np.uint8)
    tbl[codes[rows], labels[rows]] = 1
    tbl[tbl.sum(1) == 0] = 1
    return tbl[codes]


# ---------------------------------------------------------------------
# The scouting table the model has to beat
# ---------------------------------------------------------------------
# Pseudo-counts for the live tables. They are the constants the model's
# OWN priors are smoothed with in 02_preprocess.py (ARSENAL_SMOOTHING and
# MATCHUP_SMOOTHING), so both sides of the comparison shrink thin samples
# the same way. They are fixed in advance, not tuned on any held-out row.
SCOUT_M_PITCHER = 25.0   # a pitcher's overall / per-side rate, toward the level above
SCOUT_M_CELL = 10.0      # a pitcher x count (x side) cell, toward its backoff
SCOUT_M_SEASON = 25.0    # this season's cell, toward the all-seasons cell

SCOUT_RUNGS = [
    ("live", "count table, kept current"),
    ("live_side", "count x batter-side table, kept current"),
    ("live_side_season", "the same, weighted to this season"),
]

# What each rung is, in a hitter's words, completing the sentence "the
# scouting report it is measured against is ...". The site prints whichever
# one the gate used, so its description of the bar cannot drift from the bar.
SCOUT_PLAIN = {
    "live": "split by count and updated after every pitch",
    "live_side": "split by count and batter side and updated after every pitch",
    "live_side_season": ("split by count and batter side, weighted toward the "
                         "current season, and updated after every pitch"),
}


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def live_scout(pitcher, balls, strikes, side, season, hard) -> dict:
    """
    P(fastball family) for every row, from scouting tables that have seen
    ONLY the pitches thrown before it. Returns {rung: (N,) array}.

    Rows must be in chronological order. Nothing here is fitted to a
    split: each row's estimate is a running count over earlier rows.

    Why this exists. The gate used to compare the model against a
    pitcher x count table frozen at the end of training, while the
    model's own inputs (arsenal_prior, matchup history) kept updating
    through the held-out season. For a rookie, or a veteran with a new
    pitch, the model knew and the table did not, and the model was
    credited with "beating the scouting report" for holding newer
    information. A real report in August includes April to July. Both
    sides now see exactly what was known before the pitch.

    The rungs get progressively harder to beat:
      pitcher           his overall rate (the bar AGENTS.md says not to use)
      live              pitcher x count
      live_side         pitcher x count x batter side — real reports are
                        always split by handedness
      live_side_season  the same cell, shrunk toward this season's usage

    Thin cells back off instead of guessing: a pitcher's rate in a count
    he has rarely been in starts from his overall rate moved by how much
    the LEAGUE shifts in that count, on the logit scale. That is what an
    analyst does with 3 pitches of evidence, and it is what makes the
    table usable for a rookie in his second start.
    """
    # One (hard, total) pair per row; a single stable sort per key serves both.
    both = np.stack([np.asarray(hard).astype(np.int64),
                     np.ones(len(hard), dtype=np.int64)], axis=1)

    def shrunk(cols, m, toward):
        """Running hard rate within `cols`, shrunk toward `toward` by m
        pseudo-pitches. Computed and released one level at a time: on 4M
        rows, holding every level's counts at once costs over a gigabyte."""
        key = dense_codes(*cols) if cols else np.zeros(len(both), dtype=np.int64)
        kn = prior_group_cumsum(key, both)
        return (kn[:, 0] + m * toward) / (kn[:, 1] + m)

    r_lg = shrunk((), 1.0, 0.5)
    r_p = shrunk((pitcher,), SCOUT_M_PITCHER, r_lg)
    count_shift = _logit(shrunk((balls, strikes), SCOUT_M_PITCHER, r_lg)) - _logit(r_lg)
    del r_lg

    r_pc = shrunk((pitcher, balls, strikes), SCOUT_M_CELL,
                  _expit(_logit(r_p) + count_shift))
    del count_shift

    side_shift = _logit(shrunk((pitcher, side), SCOUT_M_PITCHER, r_p)) - _logit(r_p)
    r_pch = shrunk((pitcher, balls, strikes, side), SCOUT_M_CELL,
                   _expit(_logit(r_pc) + side_shift))
    del side_shift

    r_season = shrunk((pitcher, balls, strikes, side, season), SCOUT_M_SEASON, r_pch)
    return {"pitcher": r_p, "live": r_pc, "live_side": r_pch,
            "live_side_season": r_season}


def cluster_bootstrap_ci(clusters, sel, a_ok, b_ok, n_boot: int = 2000, seed: int = 0):
    """
    95% interval for accuracy(a) - accuracy(b) on the rows in `sel`,
    resampling whole clusters with replacement.

    Pitches are not independent draws: a game has one starter, one plan
    and one umpire, and a pitcher's habits run across all his games. A
    per-pitch standard error treats 700,000 pitches as 700,000 facts and
    comes out far too narrow. Pass game ids or pitcher ids as `clusters`.
    """
    sel = np.asarray(sel, dtype=bool)
    codes = dense_codes(np.asarray(clusters))
    g = int(codes.max()) + 1 if len(codes) else 0
    n = np.bincount(codes, weights=sel.astype(np.float64), minlength=g)
    d = np.bincount(codes, weights=(sel & np.asarray(a_ok, bool)).astype(np.float64)
                    - (sel & np.asarray(b_ok, bool)).astype(np.float64), minlength=g)
    if g == 0 or n.sum() == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(0, n_boot, 250):                     # chunks keep memory flat
        w = rng.multinomial(g, np.full(g, 1.0 / g), size=min(250, n_boot - i))
        tot = w @ n
        draws[i:i + len(w)] = np.where(tot > 0, (w @ d) / np.maximum(tot, 1.0), np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def widest_ci(cluster_sets, sel, a_ok, b_ok, **kw):
    """The widest interval over several ways of clustering — conservative
    when it is not obvious which dependence dominates."""
    best = None
    for clusters in cluster_sets:
        lo, hi = cluster_bootstrap_ci(clusters, sel, a_ok, b_ok, **kw)
        if best is None or (hi - lo) > (best[1] - best[0]):
            best = (lo, hi)
    return best


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
# Defined in pitch_features so the site can use it without
# importing this module, which pulls in scikit-learn.
from pitch_features import FASTBALL_FAMILY  # noqa: E402,F401

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
                 history=None, split_desc=None, mask_desc=None) -> str:
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
    # The mask moves top-1 by half a point on its own. A report that does
    # not say which one it was scored with cannot be compared to another.
    out(f"Arsenal mask: {mask_desc or 'UNRECORDED'}")
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
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nReport written to {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate the evaluation report from the saved model.")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="dataset to score (default: data_v5)")
    ap.add_argument("--eval-season", type=int, default=None,
                    help="score this season instead of the validation split; "
                         "it must be later than anything the model was tuned on")
    ap.add_argument("--as-trained", action="store_true",
                    help="reproduce the training-time report: the mask frozen at "
                         "the end of training, and every id passed through")
    args = ap.parse_args()
    data_dir = args.data_dir.resolve()

    meta = json.loads((data_dir / "meta_v5.json").read_text())
    pitch_classes = meta["pitch_classes"]

    print("Loading arrays...")
    X = load_inputs(data_dir)
    y_labels = np.load(data_dir / "y_labels.npy")
    pid = np.asarray(X["pitcher_id"])

    idx_tr, idx_val, split_desc = load_split(y_labels, data_dir)
    rows, report_name = idx_val, "eval_report_v5.txt"
    if args.eval_season is not None:
        split = read_split(data_dir)
        if split.get("mode") != "temporal":
            raise SystemExit("--eval-season needs a model trained with a temporal "
                             "split: on a random split every season is part training data.")
        if args.eval_season <= int(split["holdout_season"]):
            raise SystemExit(f"--eval-season must be later than the validation season "
                             f"({split['holdout_season']}); {args.eval_season} was used "
                             f"to train or tune this model.")
        seasons = np.load(data_dir / "X_season.npy")
        rows = np.flatnonzero(seasons == args.eval_season)
        if len(rows) == 0:
            raise SystemExit(f"no rows for season {args.eval_season} in {data_dir.name}/ — "
                             f"seasons present: {sorted(set(seasons.tolist()))}")
        split_desc += f" — SCORED ON {args.eval_season}, a season used for nothing else"
        report_name = f"eval_report_v5_{args.eval_season}.txt"
    if args.as_trained and args.eval_season is None:
        report_name = "eval_report_v5_as_trained.txt"
    print(f"Split: {split_desc}   Rows scored: {len(rows):,}")

    if args.as_trained:
        # What 03_train.py validates with. Kept so its report can be
        # regenerated; not how the model is measured or served.
        mask = build_arsenal_mask(pid, y_labels, idx_tr, len(pitch_classes))[pid]
        mask_desc, known = "training rows only", None
    else:
        mask, mask_desc = scoring_mask(pid, y_labels, idx_tr, len(pitch_classes),
                                       load_row_meta(data_dir))
        known = known_ids(X, idx_tr)
        mask_desc += "; ids never trained on -> unknown bucket"
    print(f"Arsenal mask: {mask_desc}")

    print("Loading the model (NumPy, no TensorFlow)...")
    model = load_scoring_model(data_dir)
    for k, lut in (known or {}).items():
        ids = np.asarray(X[k])[rows]
        share = float(((lut[ids] == 0) & (ids != 0)).mean())
        if share > 0:
            print(f"  {k}: {share:.1%} of scored rows carry an id the model never "
                  f"trained on -> unknown bucket")
    heads = predict_rows(model, X, rows, mask[rows], label="scoring",
                         all_heads=True, known=known)
    y_pred_probs = heads[model.primary_output]
    zone_probs = heads.get("zone_output")

    history = None
    hist_path = data_dir / "training_history_v5.json"
    if hist_path.exists() and args.eval_season is None:
        history = json.loads(hist_path.read_text())

    report = build_report(
        y_labels[rows], y_pred_probs, pid[idx_tr], y_labels[idx_tr], pid[rows],
        pitch_classes, history, split_desc, mask_desc,
    )

    zone_path = data_dir / "y_zone.npy"
    if zone_probs is not None and zone_path.exists():
        y_zone = np.load(zone_path)
        zr, zv = y_zone[idx_tr], y_zone[rows]
        report += "\n\n" + build_location_report(
            np.where(zv >= 0, zv, 0).astype(np.int64), zone_probs,
            (zv >= 0).astype(np.float32),
            pid[idx_tr],
            np.where(zr >= 0, zr, 0).astype(np.int64),
            (zr >= 0).astype(np.float32),
            pid[rows],
            meta.get("zone_classes", []),
        )
    print("\n" + report)
    write_report(report, data_dir / report_name)


if __name__ == "__main__":
    main()
