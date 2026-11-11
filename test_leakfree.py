"""
NextPitchAI — the measurement must not see the future
======================================================
Every number the site shows comes from two things built row by row: the
arsenal mask the model is scored with and the scouting table it is scored
against. If either reads a pitch at or after the one being judged, the
published edge is partly an artefact — and the output looks completely
normal. That has happened here twice (a mask built over every row; a
table frozen while the model's inputs were not), so it is tested.

Each fast routine is checked against the slowest, most obvious loop that
could compute the same thing, and then against the property that matters:
changing the present or the future must not change the past.

No data and no model needed; runs in a few seconds.

    .venv/bin/python test_leakfree.py
"""

import sys
from pathlib import Path

import numpy as np

import analyze_actionability as gate
import evaluate_model as em
from pitch_features import (
    MASK_MIN_HISTORY, causal_arsenal_mask, dense_codes, prior_group_cumsum,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "site"))
import build_serving_artifacts as bundle  # noqa: E402


def make_rows(n=1500, n_pitchers=9, seed=0):
    """A small chronological log with the columns the tables are keyed on."""
    rng = np.random.default_rng(seed)
    return {
        # Raw ids are large and sparse, like MLBAM's.
        "pitcher": rng.choice(np.arange(600000, 600000 + 37 * n_pitchers, 37), size=n),
        "balls": rng.integers(0, 4, n),
        "strikes": rng.integers(0, 3, n),
        "side": rng.integers(0, 2, n),
        "season": np.sort(rng.integers(2024, 2027, n)),
        "hard": rng.integers(0, 2, n),
        "label": rng.integers(0, 6, n),
    }


def scout_of(r, hard=None):
    return em.live_scout(r["pitcher"], r["balls"], r["strikes"], r["side"],
                         r["season"], r["hard"] if hard is None else hard)


# ---------------------------------------------------------------------
# prior_group_cumsum
# ---------------------------------------------------------------------
def test_prior_group_cumsum_matches_loop():
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 7, 400)
    for values in (rng.integers(0, 5, 400), rng.random(400),
                   rng.integers(0, 3, (400, 4))):
        want = np.zeros(values.shape, dtype=np.float64)
        for i in range(len(keys)):
            earlier = np.flatnonzero(keys[:i] == keys[i])
            want[i] = values[earlier].sum(axis=0)
        got = prior_group_cumsum(keys, values)
        assert got.shape == values.shape
        assert np.allclose(got, want, atol=1e-9)


def test_prior_group_cumsum_integer_input_is_exact():
    keys = np.zeros(50_000, dtype=np.int64)
    got = prior_group_cumsum(keys, np.ones(50_000, dtype=np.int32))
    assert got.dtype == np.int64
    assert np.array_equal(got, np.arange(50_000))


def test_dense_codes_equal_exactly_when_every_column_is():
    r = make_rows(600)
    code = dense_codes(r["pitcher"], r["balls"], r["strikes"])
    tup = list(zip(r["pitcher"].tolist(), r["balls"].tolist(), r["strikes"].tolist()))
    seen = {}
    for c, t in zip(code.tolist(), tup):
        assert seen.setdefault(c, t) == t          # one code -> one tuple
    assert len(set(tup)) == len(set(code.tolist()))  # one tuple -> one code


# ---------------------------------------------------------------------
# The arsenal mask
# ---------------------------------------------------------------------
def test_causal_mask_matches_loop():
    r = make_rows(900, n_pitchers=5)
    k = int(r["label"].max()) + 1
    for min_history in (0, 40):
        got = causal_arsenal_mask(r["pitcher"], r["label"], k, min_history)
        for i in range(len(r["label"])):
            mine = r["label"][:i][r["pitcher"][:i] == r["pitcher"][i]]
            # No history at all is an unknown arsenal, never an empty one.
            want = (np.ones(k, dtype=np.uint8) if len(mine) < max(min_history, 1)
                    else np.isin(np.arange(k), mine).astype(np.uint8))
            assert np.array_equal(got[i], want), i


def test_causal_mask_ignores_the_present_and_the_future():
    r = make_rows(900, n_pitchers=5)
    k = int(r["label"].max()) + 1
    base = causal_arsenal_mask(r["pitcher"], r["label"], k, 30)
    rng = np.random.default_rng(2)
    for cut in (1, 250, 600, 899):
        changed = r["label"].copy()
        changed[cut:] = rng.integers(0, k, len(changed) - cut)
        again = causal_arsenal_mask(r["pitcher"], changed, k, 30)
        # Row `cut` itself is unchanged too: a pitch is never in its own mask.
        assert np.array_equal(base[:cut + 1], again[:cut + 1]), cut


def test_causal_mask_never_leaves_a_row_empty():
    r = make_rows(900, n_pitchers=5)
    k = int(r["label"].max()) + 1
    for min_history in (0, 1, MASK_MIN_HISTORY):
        mask = causal_arsenal_mask(r["pitcher"], r["label"], k, min_history)
        assert mask.sum(1).min() >= 1


def test_causal_mask_is_keyed_by_the_raw_pitcher():
    # Two pitchers, disjoint arsenals. Pooling them (as the encoded id does
    # for every rare pitcher) would give each the other's pitches.
    pitcher = np.array([11, 22] * 200)
    label = np.array([0, 3] * 200)
    mask = causal_arsenal_mask(pitcher, label, 4, min_history=10)
    assert np.array_equal(mask[-2], [1, 0, 0, 0])
    assert np.array_equal(mask[-1], [0, 0, 0, 1])


def test_old_full_mask_knows_the_future_and_the_causal_one_does_not():
    # A pitcher who adds class 2 only in the held-out season.
    pitcher = np.full(600, 7)
    season = np.repeat([2024, 2025], 300)
    label = np.where(season == 2025, np.tile([0, 2], 300), 0)
    leaky = em.full_mask_through(pitcher, label, season, 2025, 3)
    causal = causal_arsenal_mask(pitcher, label, 3, min_history=50)
    first_new = int(np.flatnonzero(label == 2)[0])
    assert leaky[0, 2] == 1                      # known two seasons early
    assert causal[first_new, 2] == 0             # not known until thrown
    assert causal[first_new + 1, 2] == 1         # and known immediately after


# ---------------------------------------------------------------------
# The scouting table
# ---------------------------------------------------------------------
def _scout_loop(r):
    """live_scout written the slow way, straight from its definition."""
    logit, expit = em._logit, em._expit
    mp, mc, ms = em.SCOUT_M_PITCHER, em.SCOUT_M_CELL, em.SCOUT_M_SEASON
    n = len(r["hard"])
    out = {k: np.zeros(n) for k in ("pitcher", "live", "live_side", "live_side_season")}
    for i in range(n):
        h = r["hard"][:i]
        same = {c: r[c][:i] == r[c][i] for c in ("pitcher", "balls", "strikes", "side", "season")}
        count = same["balls"] & same["strikes"]

        def rate(sel, m, toward):
            return (h[sel].sum() + m * toward) / (sel.sum() + m)

        r_lg = (h.sum() + 0.5) / (i + 1.0)
        r_lc = rate(count, mp, r_lg)
        r_p = rate(same["pitcher"], mp, r_lg)
        r_pc = rate(same["pitcher"] & count, mc,
                    expit(logit(r_p) + logit(r_lc) - logit(r_lg)))
        r_ph = rate(same["pitcher"] & same["side"], mp, r_p)
        r_pch = rate(same["pitcher"] & count & same["side"], mc,
                     expit(logit(r_pc) + logit(r_ph) - logit(r_p)))
        r_s = rate(same["pitcher"] & count & same["side"] & same["season"], ms, r_pch)
        out["pitcher"][i], out["live"][i] = r_p, r_pc
        out["live_side"][i], out["live_side_season"][i] = r_pch, r_s
    return out


def test_live_scout_matches_loop():
    r = make_rows(1200)
    got, want = scout_of(r), _scout_loop(r)
    assert set(got) == set(want) == {"pitcher"} | {name for name, _ in em.SCOUT_RUNGS}
    for name in want:
        assert np.allclose(got[name], want[name], atol=1e-10), name


def test_live_scout_ignores_the_present_and_the_future():
    r = make_rows(1500)
    base = scout_of(r)
    rng = np.random.default_rng(3)
    for cut in (0, 1, 400, 1100, 1499):
        changed = r["hard"].copy()
        changed[cut:] = rng.integers(0, 2, len(changed) - cut)
        again = scout_of(r, changed)
        for name in base:
            # Up to AND INCLUDING `cut`: the table's call on a pitch must
            # not move when that pitch's own outcome is changed.
            assert np.array_equal(base[name][:cut + 1], again[name][:cut + 1]), (name, cut)


def test_live_scout_is_a_probability_and_starts_uninformed():
    r = make_rows(1500)
    s = scout_of(r)
    for name, v in s.items():
        assert np.all((v > 0) & (v < 1)), name
        assert v[0] == 0.5, name                 # nothing known before pitch one


def test_live_scout_learns_a_rookie_and_the_frozen_table_cannot():
    # The defect this replaces. A pitcher who debuts in the held-out season
    # and throws hard on every 0-0 pitch: the live table knows within a
    # start; a table frozen at the end of training never learns it.
    n_old, n_new = 600, 300
    pitcher = np.r_[np.full(n_old, 1), np.full(n_new, 2)]
    season = np.r_[np.full(n_old, 2024), np.full(n_new, 2025)]
    zeros = np.zeros(n_old + n_new, dtype=np.int64)
    hard = np.r_[np.zeros(n_old, dtype=np.int64), np.ones(n_new, dtype=np.int64)]
    live = em.live_scout(pitcher, zeros, zeros, zeros, season, hard)
    rookie = np.flatnonzero(pitcher == 2)
    for name, _ in em.SCOUT_RUNGS:
        assert (live[name][rookie[60:]] >= 0.5).all(), name

    pid_enc = np.r_[np.full(n_old, 1), np.zeros(n_new, dtype=np.int64)]   # rookie -> unknown
    frozen = gate.old_bar_lean(pid_enc, zeros, zeros, hard, np.arange(n_old), rookie)
    assert frozen.all()      # it only "knows" via the answers of the rows it is judged on
    # ...which is the eval-truth fallback, not knowledge: with a mixed
    # season that fallback follows the league, not the pitcher.
    mixed = hard.copy()
    mixed[n_old::3] = 0
    frozen = gate.old_bar_lean(pid_enc, zeros, zeros, mixed, np.arange(n_old), rookie)
    assert frozen.all() or not frozen.any()      # one constant call for every pitch


# ---------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------
def test_bootstrap_matches_an_explicit_resample():
    rng = np.random.default_rng(4)
    n = 4000
    clusters = rng.integers(0, 60, n) * 13 + 5
    sel = rng.random(n) < 0.4
    a_ok = rng.random(n) < 0.72
    b_ok = rng.random(n) < 0.68
    lo, hi = em.cluster_bootstrap_ci(clusters, sel, a_ok, b_ok, n_boot=500, seed=9)

    ids = np.unique(clusters)
    num = np.array([(sel & a_ok & (clusters == c)).sum() - (sel & b_ok & (clusters == c)).sum()
                    for c in ids], dtype=float)
    den = np.array([(sel & (clusters == c)).sum() for c in ids], dtype=float)
    g = len(ids)
    rng2 = np.random.default_rng(9)
    draws = []
    for i in range(0, 500, 250):
        for w in rng2.multinomial(g, np.full(g, 1.0 / g), size=250):
            draws.append((w * num).sum() / (w * den).sum())
    want = np.percentile(draws, [2.5, 97.5])
    assert np.allclose([lo, hi], want, atol=1e-12)
    point = (a_ok[sel].mean() - b_ok[sel].mean())
    assert lo < point < hi


def test_bootstrap_is_zero_width_when_the_two_agree():
    rng = np.random.default_rng(5)
    ok = rng.random(2000) < 0.7
    lo, hi = em.cluster_bootstrap_ci(rng.integers(0, 40, 2000), np.ones(2000, bool), ok, ok)
    assert lo == hi == 0.0


def test_clustered_interval_is_wider_than_a_per_pitch_one():
    # Whole games go one way or the other. Treating pitches as independent
    # would call this difference certain; resampling games must not.
    rng = np.random.default_rng(6)
    games = np.repeat(np.arange(30), 300)
    game_effect = rng.random(30) < 0.5
    a_ok = game_effect[games]
    b_ok = ~a_ok
    sel = np.ones(len(games), bool)
    lo, hi = em.cluster_bootstrap_ci(games, sel, a_ok, b_ok, n_boot=800)
    lo_iid, hi_iid = em.cluster_bootstrap_ci(np.arange(len(games)), sel, a_ok, b_ok, n_boot=800)
    assert (hi - lo) > 5 * (hi_iid - lo_iid)
    assert em.widest_ci([np.arange(len(games)), games], sel, a_ok, b_ok, n_boot=800) == (lo, hi)


# ---------------------------------------------------------------------
# The gate's own arithmetic
# ---------------------------------------------------------------------
def test_cell_summary_matches_a_plain_loop():
    rng = np.random.default_rng(7)
    n = 6000
    pitcher = rng.choice([101, 202, 303], n)
    balls, strikes = rng.integers(0, 2, n), rng.integers(0, 2, n)
    hard = rng.integers(0, 2, n)
    tool_ok, bar_ok = rng.random(n) < 0.7, rng.random(n) < 0.66
    cs = gate.cell_summary(pitcher, balls, strikes, hard, tool_ok, bar_ok)

    tot = tool = bar = oracle = cells = won = 0
    inside = np.zeros(n, dtype=bool)
    for p in np.unique(pitcher):
        for b in range(4):
            for s in range(3):
                m = (pitcher == p) & (balls == b) & (strikes == s)
                if m.sum() < gate.MIN_CELL_LIFT:
                    continue
                inside |= m
                cells += 1
                tot += m.sum()
                tool += tool_ok[m].sum()
                bar += bar_ok[m].sum()
                oracle += max(hard[m].sum(), m.sum() - hard[m].sum())
                won += tool_ok[m].mean() > bar_ok[m].mean()
    assert cs["cells"] == cells and cs["n"] == tot
    assert np.array_equal(cs["in_cells"], inside)
    assert np.isclose(cs["model"], tool / tot) and np.isclose(cs["bar"], bar / tot)
    assert np.isclose(cs["oracle"], oracle / tot)
    assert np.isclose(cs["cells_won"], won / cells)
    assert sum(pc["n"] for pc in cs["per_count"]) == tot
    for pc in cs["per_count"]:
        b, s = map(int, pc["count"].split("-"))
        m = inside & (balls == b) & (strikes == s)
        assert pc["n"] == m.sum() and np.isclose(pc["model"], tool_ok[m].mean())


def test_slice_rows_reports_every_table_on_the_same_pitches():
    rng = np.random.default_rng(8)
    conf = rng.random(5000) * 0.5 + 0.5
    tool_ok = rng.random(5000) < 0.7
    tables = {"a": rng.random(5000) < 0.6, "b": rng.random(5000) < 0.65}
    for row in gate.slice_rows(conf, tool_ok, tables):
        sel = conf >= row["threshold"]
        assert row["n"] == sel.sum() and np.isclose(row["coverage"], sel.mean())
        assert np.isclose(row["tool"], tool_ok[sel].mean())
        for name, ok in tables.items():
            assert np.isclose(row[name], ok[sel].mean())


# ---------------------------------------------------------------------
# Scoring plumbing
# ---------------------------------------------------------------------
class _EchoModel:
    """Returns, per row, numbers that identify the row and the mask it got."""

    def predict(self, inputs):
        return np.stack([inputs["pitcher_id"].astype(np.float32),
                         inputs["ctx"][:, 0],
                         inputs["arsenal_mask"].sum(1)], axis=1)


def test_predict_rows_keeps_rows_and_masks_aligned():
    n = 1000
    X = {k: np.arange(n, dtype=np.int32) for k in em.INPUT_KEYS}
    X["ctx"] = np.arange(n, dtype=np.float32)[:, None] * 2.0
    X["seq"] = np.zeros((n, 2, 2), dtype=np.float32)
    mask = (np.arange(n)[:, None] % 5 >= np.arange(4)[None, :]).astype(np.uint8)
    rng = np.random.default_rng(10)
    for rows in (np.arange(200, 700),                     # one contiguous season
                 np.sort(rng.choice(n, 300, replace=False)),   # scattered
                 rng.permutation(n)[:300],                # unsorted
                 np.arange(5)):                           # smaller than a batch
        got = em.predict_rows(_EchoModel(), X, rows, mask[rows], batch=64)
        assert np.array_equal(got[:, 0], rows)
        assert np.array_equal(got[:, 1], rows * 2.0)
        assert np.array_equal(got[:, 2], mask[rows].sum(1))


def test_predict_rows_sends_untrained_ids_to_the_unknown_bucket():
    n = 300
    X = {k: np.arange(n, dtype=np.int32) % 7 for k in em.INPUT_KEYS}
    X["ctx"] = np.zeros((n, 1), dtype=np.float32)
    X["seq"] = np.zeros((n, 2, 2), dtype=np.float32)
    X["pitcher_id"] = np.r_[np.tile([1, 2, 3], 50), np.tile([3, 4, 5], 50)].astype(np.int32)
    idx_tr = np.arange(150)                      # ids 4 and 5 appear only after training
    known = em.known_ids(X, idx_tr)
    assert known["pitcher_id"].tolist() == [0, 1, 2, 3, 0, 0]
    rows = np.arange(150, 300)
    mask = np.ones((n, 3), dtype=np.uint8)
    got = em.predict_rows(_EchoModel(), X, rows, mask[rows], batch=64, known=known)
    assert np.array_equal(got[:, 0], np.tile([3, 0, 0], 50))
    # Without it the model is handed the never-trained rows as they are.
    raw = em.predict_rows(_EchoModel(), X, rows, mask[rows], batch=64)
    assert np.array_equal(raw[:, 0], np.tile([3, 4, 5], 50))
    # The caller's arrays are never modified.
    assert X["pitcher_id"][150:153].tolist() == [3, 4, 5]


def test_known_ids_is_the_identity_when_every_id_was_trained():
    X = {"pitcher_id": np.array([0, 1, 2, 3, 1, 2, 3, 0], dtype=np.int32)}
    lut = em.known_ids(X, np.arange(8), keys=["pitcher_id"])["pitcher_id"]
    assert lut.tolist() == [0, 1, 2, 3] and lut.dtype == np.int32


def test_an_ablation_edit_cannot_leak_into_the_source_arrays():
    n = 200
    X = {k: np.arange(n, dtype=np.int32) for k in em.INPUT_KEYS}
    X["ctx"] = np.ones((n, 1), dtype=np.float32)
    X["seq"] = np.zeros((n, 2, 2), dtype=np.float32)
    mask = np.ones((n, 3), dtype=np.uint8)

    def blank(inputs):
        inputs["pitcher_id"][:] = 0
        inputs["ctx"][:] = 5.0

    got = em.predict_rows(_EchoModel(), X, np.arange(n), mask, batch=64, edit=blank)
    assert (got[:, 0] == 0).all() and (got[:, 1] == 5.0).all()
    assert np.array_equal(X["pitcher_id"], np.arange(n)) and (X["ctx"] == 1.0).all()


# ---------------------------------------------------------------------
# The serving bundle's roster
# ---------------------------------------------------------------------
def test_roster_is_keyed_by_raw_id_and_takes_each_players_last_row():
    #              row: 0    1    2    3    4    5    6    7
    raw = np.array([501, 777, 501, -1, 777, 501, 909, 777])
    ids, last, n, keep, inv = bundle.roster(raw, min_rows=2)
    assert ids.tolist() == [501, 777]            # 909 has one row; -1 is nobody
    assert last.tolist() == [5, 7]               # the LAST row, i.e. full history
    assert n.tolist() == [3, 3]
    # `inv` and `keep` index the same per-player table the mask is built on.
    assert np.array_equal(np.unique(raw)[inv], raw)
    assert keep.sum() == 2


def test_roster_does_not_pool_players_the_encoder_pooled():
    # Three pitchers the model has no embedding for all encode to 0. Keyed
    # by that, "the last row of id 0" is one of them served as all three.
    raw = np.repeat([11, 22, 33], 40)
    enc = np.zeros(120, dtype=np.int32)
    ids, last, n, _, _ = bundle.roster(raw, min_rows=10)
    assert ids.tolist() == [11, 22, 33] and last.tolist() == [39, 79, 119]
    assert (enc[last] == 0).all()                # all served from the unknown bucket


def test_scoring_mask_never_falls_back_to_all_zero():
    r = make_rows(900, n_pitchers=5)
    k = int(r["label"].max()) + 1
    enc = dense_codes(r["pitcher"]).astype(np.int32)
    idx_tr = np.arange(300)
    causal, desc = em.scoring_mask(enc, r["label"], idx_tr, k, {"pitcher": r["pitcher"]})
    assert "causal" in desc and causal.sum(1).min() >= 1
    frozen, desc = em.scoring_mask(enc, r["label"], idx_tr, k, None)
    assert "training rows" in desc and frozen.sum(1).min() >= 1
    # The frozen fallback is built from training rows and nothing else.
    again, _ = em.scoring_mask(enc, np.where(np.arange(900) >= 300, 0, r["label"]),
                               idx_tr, k, None)
    assert np.array_equal(frozen, again)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}  {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
