"""
NextPitchAI v6 — shared pitch-feature vocabulary
================================================
The canonical pitch classes, outcome classes, attack zones, and the pure
functions that turn a Statcast frame into model features.

This lives apart from 02_preprocess.py for two reasons:

  * 02_preprocess.py imports scikit-learn and joblib at module level, and
    the website imports these builders at serve time. Pulling in
    scikit-learn (~100MB) so the server can compute a 14-element game
    state vector is not a trade worth making, and the deployed app runs
    on a 512MB box.
  * Training and serving MUST build features identically. A serve-time
    reimplementation that drifts by one column produces confident,
    plausible, wrong predictions — the exact failure this project keeps
    running into. One definition, imported by both, makes that
    impossible rather than merely unlikely.

numpy and pandas only. Keep it that way.
"""
import numpy as np
import pandas as pd

# =========================
# Pitch type -> canonical class mapping
# =========================
# 10 canonical classes; rare/legacy Statcast codes merge into their
# modern equivalents. Junk rows (PO pitchout, IN intentional ball,
# UN unknown, AB automatic ball) are dropped entirely — they are not
# real pitch-selection decisions.
PITCH_TYPE_CANON = {
    "FF": "FF",   # four-seam fastball
    "FA": "FF",   # generic fastball (legacy)
    "SI": "SI",   # sinker
    "FT": "SI",   # two-seam (legacy code, same family)
    "FC": "FC",   # cutter
    "SL": "SL",   # slider
    "ST": "ST",   # sweeper
    "SV": "ST",   # slurve (closest modern family)
    "CU": "CU",   # curveball
    "CS": "CU",   # slow curve
    "EP": "CU",   # eephus (big slow curve family)
    "KC": "KC",   # knuckle-curve
    "CH": "CH",   # changeup
    "SC": "CH",   # screwball (fades like a change)
    "FS": "FS",   # splitter
    "FO": "FS",   # forkball
    "KN": "KN",   # knuckleball
}

PITCH_CLASSES = sorted(set(PITCH_TYPE_CANON.values()))  # alphabetical
PITCH_TO_ID = {p: i for i, p in enumerate(PITCH_CLASSES)}
N_PITCH = len(PITCH_CLASSES)
print(f"Pitch classes ({N_PITCH}): {PITCH_CLASSES}")

# Swing / whiff classification from Statcast `description`
SWING_DESCRIPTIONS = {
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "foul", "foul_tip", "foul_bunt", "bunt_foul_tip",
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}

# What HAPPENED to each previous pitch. A catcher calls the next pitch
# very differently after a swinging strike than after a foul, and the
# count alone cannot express that: 1-1 reached via two fouls is a
# different at-bat from 1-1 reached via ball-then-called-strike.
# Used ONLY in the lookback window (see build_sequences), never for the
# pitch being predicted — the outcome of the current pitch would leak
# its type directly.
OUTCOME_CLASSES = ["ball", "called_strike", "whiff", "foul", "in_play"]
N_OUTCOME = len(OUTCOME_CLASSES)
OUTCOME_OF = {}
for _d in ("ball", "blocked_ball", "pitchout", "hit_by_pitch"):
    OUTCOME_OF[_d] = 0
OUTCOME_OF["called_strike"] = 1
for _d in WHIFF_DESCRIPTIONS:
    OUTCOME_OF[_d] = 2
for _d in ("foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "foul_pitchout"):
    OUTCOME_OF[_d] = 3
for _d in ("hit_into_play", "hit_into_play_score", "hit_into_play_no_out"):
    OUTCOME_OF[_d] = 4


# =========================
# Location target — Statcast attack zones
# =========================
# heart  : middle of the plate, damage it
# shadow : straddling the edge, protect
# chase  : off the plate but reachable, lay off
# waste  : nowhere near
#
# Computed in units of the BATTER'S OWN strike zone, which is why
# sz_top/sz_bot are scraped: "up" is a different pitch to a 5'6" hitter
# than a 6'7" one, and the heart/shadow boundary is exactly where that
# difference decides the label.
ZONE_CLASSES = ["heart", "shadow", "chase", "waste"]
N_ZONE = len(ZONE_CLASSES)
PLATE_HALF_FT = 0.708 + 0.121   # half plate (8.5in) + ball radius
ZONE_EDGES = (0.67, 1.33, 2.00)  # heart | shadow | chase | waste


def compute_attack_zone(df: pd.DataFrame) -> np.ndarray:
    """
    Chebyshev distance from zone centre, in zone-half-widths, bucketed.
    r <= 0.67 heart, <= 1.33 shadow, <= 2.0 chase, else waste.
    Rows missing location or zone bounds return -1 and are excluded from
    the location loss rather than guessed at.
    """
    px = df["plate_x"].values.astype(np.float64)
    pz = df["plate_z"].values.astype(np.float64)
    top = df["sz_top"].values.astype(np.float64) if "sz_top" in df.columns else np.full(len(df), np.nan)
    bot = df["sz_bot"].values.astype(np.float64) if "sz_bot" in df.columns else np.full(len(df), np.nan)

    # League-average fallback so a missing zone bound costs one row's
    # precision, not the whole row.
    top = np.where(np.isnan(top), 3.40, top)
    bot = np.where(np.isnan(bot), 1.60, bot)
    half_h = np.maximum((top - bot) / 2.0, 0.1)
    mid = (top + bot) / 2.0

    zx = np.abs(px) / PLATE_HALF_FT
    zz = np.abs(pz - mid) / half_h
    r = np.maximum(zx, zz)

    out = np.full(len(df), N_ZONE - 1, dtype=np.int64)   # default waste
    out[r <= ZONE_EDGES[2]] = 2
    out[r <= ZONE_EDGES[1]] = 1
    out[r <= ZONE_EDGES[0]] = 0
    out[np.isnan(px) | np.isnan(pz)] = -1
    return out


def build_id_mapping(series: pd.Series, min_count: int) -> dict:
    """Frequent players get unique IDs; rare players map to 0 (<UNK>)."""
    counts = series.value_counts()
    frequent = counts[counts >= min_count].index.tolist()
    return {pid: (i + 1) for i, pid in enumerate(sorted(frequent))}


def encode_ids(series: pd.Series, mapping: dict) -> np.ndarray:
    return np.array([mapping.get(v, 0) for v in series], dtype=np.int32)


def expanding_prior_dist(df: pd.DataFrame, group_cols, onehot: np.ndarray,
                         smoothing: float, fallback: np.ndarray) -> tuple:
    """
    For each row, the distribution of pitch types seen in PRIOR rows of
    the same group, smoothed toward `fallback` (shape (N, K) or (K,)).
    Excludes the current row, so there is no label leakage.
    Returns (prior_dist (N,K), prior_total (N,)).
    """
    oh = pd.DataFrame(onehot, index=df.index)
    csum = oh.groupby([df[c] for c in group_cols], sort=False).cumsum().values
    prior_counts = csum - onehot  # exclude current row
    prior_total = prior_counts.sum(axis=1, keepdims=True)

    if fallback.ndim == 1:
        fallback = np.broadcast_to(fallback, prior_counts.shape)

    dist = (prior_counts + smoothing * fallback) / (prior_total + smoothing)
    return dist.astype(np.float32), prior_total.squeeze(1).astype(np.float32)


def expanding_prior_rate(df: pd.DataFrame, group_cols, num_onehot: np.ndarray,
                         den_onehot: np.ndarray, smoothing: float,
                         fallback_rate: np.ndarray) -> np.ndarray:
    """
    Per-row, per-class prior rate = prior_num / prior_den within the group
    (both excluding the current row), smoothed toward the league rate.
    Used for batter whiff rate per pitch type.
    """
    num = pd.DataFrame(num_onehot, index=df.index)
    den = pd.DataFrame(den_onehot, index=df.index)
    gb = [df[c] for c in group_cols]
    prior_num = num.groupby(gb, sort=False).cumsum().values - num_onehot
    prior_den = den.groupby(gb, sort=False).cumsum().values - den_onehot

    rate = (prior_num + smoothing * fallback_rate) / (prior_den + smoothing)
    return rate.astype(np.float32)


def compute_times_through_order(df: pd.DataFrame) -> np.ndarray:
    """TTO = how many times this batter has faced this pitcher this game."""
    ab_keys = df[["game_pk", "pitcher", "batter", "at_bat_number"]].drop_duplicates()
    ab_keys = ab_keys.sort_values(["game_pk", "pitcher", "batter", "at_bat_number"])
    ab_keys["tto"] = ab_keys.groupby(["game_pk", "pitcher", "batter"]).cumcount() + 1
    merged = df.merge(ab_keys, on=["game_pk", "pitcher", "batter", "at_bat_number"],
                      how="left")
    return merged["tto"].fillna(1).values.astype(np.float32)


def compute_days_rest(df: pd.DataFrame) -> np.ndarray:
    """Days since the pitcher's previous game (from game_date)."""
    pg = df[["pitcher", "game_pk", "game_date"]].drop_duplicates(
        subset=["pitcher", "game_pk"]).sort_values(["pitcher", "game_date"])
    pg["days_rest"] = pg.groupby("pitcher")["game_date"].diff().dt.days
    merged = df.merge(pg[["pitcher", "game_pk", "days_rest"]],
                      on=["pitcher", "game_pk"], how="left")
    return merged["days_rest"].fillna(5).values.astype(np.float32)


def build_context_features(df: pd.DataFrame) -> tuple:
    """
    Numeric game-state context vector. Returns (ctx, names).
    """
    names = [
        "balls", "strikes", "outs", "inning", "is_top", "score_diff",
        "on_1b", "on_2b", "on_3b", "risp", "pitch_of_ab",
        "times_through_order", "pitch_count_game", "days_rest",
    ]
    ctx = np.zeros((len(df), len(names)), dtype=np.float32)

    ctx[:, 0] = df["balls"].fillna(0).values.astype(np.float32)
    ctx[:, 1] = df["strikes"].fillna(0).values.astype(np.float32)
    ctx[:, 2] = df["outs_when_up"].fillna(0).values.astype(np.float32)
    ctx[:, 3] = np.minimum(df["inning"].fillna(1).values, 9).astype(np.float32)
    ctx[:, 4] = (df["inning_topbot"] == "Top").astype(np.float32)

    score_diff = (df["fld_score"].fillna(0) - df["bat_score"].fillna(0)).values
    ctx[:, 5] = np.clip(score_diff, -6, 6).astype(np.float32)

    on_2b = df["on_2b"].notna().values
    on_3b = df["on_3b"].notna().values
    ctx[:, 6] = df["on_1b"].notna().astype(np.float32)
    ctx[:, 7] = on_2b.astype(np.float32)
    ctx[:, 8] = on_3b.astype(np.float32)
    ctx[:, 9] = (on_2b | on_3b).astype(np.float32)  # runners in scoring position

    ctx[:, 10] = np.minimum(df["pitch_number"].fillna(1).values, 15).astype(np.float32)

    # Times through order: prefer Statcast's column, else compute
    if "n_thruorder_pitcher" in df.columns and df["n_thruorder_pitcher"].notna().mean() > 0.5:
        tto = df["n_thruorder_pitcher"].values.astype(np.float64)
        computed = compute_times_through_order(df)
        tto = np.where(np.isnan(tto), computed, tto)
    else:
        tto = compute_times_through_order(df)
    ctx[:, 11] = np.minimum(tto, 4).astype(np.float32)

    ctx[:, 12] = np.minimum(
        df.groupby(["game_pk", "pitcher"]).cumcount().values + 1, 120
    ).astype(np.float32)

    # Days rest: prefer Statcast's column, else compute from game_date
    if ("pitcher_days_since_prev_game" in df.columns
            and df["pitcher_days_since_prev_game"].notna().mean() > 0.5):
        rest = df["pitcher_days_since_prev_game"].fillna(5).values.astype(np.float64)
    else:
        rest = compute_days_rest(df)
    ctx[:, 13] = np.clip(rest, 0, 15).astype(np.float32)

    return ctx, names


def build_pitch_features(df: pd.DataFrame, pitch_idx: np.ndarray) -> np.ndarray:
    """
    Per-pitch feature vector used as sequence timesteps:
      0..K-1       : one-hot pitch type
      K..K+O-1     : one-hot outcome (ball/called/whiff/foul/in-play)
      K+O          : release_speed
      K+O+1        : plate_x
      K+O+2        : plate_z
      K+O+3        : release_spin_rate
      K+O+4        : pfx_x (horizontal movement)
      K+O+5        : pfx_z (vertical movement)

    Both one-hots come FIRST so the scaler can be applied to the
    continuous tail only. An unrecognised description leaves the outcome
    block all-zero, which is a valid "unknown" state.
    """
    n_cont = 6
    base = N_PITCH + N_OUTCOME
    feats = np.zeros((len(df), base + n_cont), dtype=np.float32)
    for i in range(N_PITCH):
        feats[pitch_idx == i, i] = 1.0

    out_idx = df["description"].map(OUTCOME_OF).values
    known = ~pd.isna(out_idx)
    rows = np.flatnonzero(known)
    feats[rows, N_PITCH + out_idx[known].astype(int)] = 1.0

    feats[:, base + 0] = df["release_speed"].fillna(0).values
    feats[:, base + 1] = df["plate_x"].fillna(0).values
    feats[:, base + 2] = df["plate_z"].fillna(0).values
    feats[:, base + 3] = df["release_spin_rate"].fillna(0).values
    feats[:, base + 4] = df["pfx_x"].fillna(0).values
    feats[:, base + 5] = df["pfx_z"].fillna(0).values
    return feats


def build_sequences(pitch_feats: np.ndarray, df: pd.DataFrame,
                    seq_len: int) -> np.ndarray:
    """
    (N, seq_len, F+1) sequences of the previous pitches thrown by the
    SAME PITCHER in the same game. The extra final feature per timestep
    is a same-at-bat flag so the model can tell "earlier this at-bat"
    from "earlier this game". A pitcher's first pitches of a game are
    zero-padded.
    """
    N, F = pitch_feats.shape
    sequences = np.zeros((N, seq_len, F + 1), dtype=np.float32)
    ab_vals = df["at_bat_number"].values

    groups = df.groupby(["game_pk", "pitcher"], sort=False).indices
    for idxs in groups.values():
        # idxs are ascending positions (df is chronologically sorted)
        for pos, i in enumerate(idxs):
            prev = idxs[max(0, pos - seq_len):pos]
            if len(prev) == 0:
                continue
            start = seq_len - len(prev)
            sequences[i, start:, :F] = pitch_feats[prev]
            sequences[i, start:, F] = (ab_vals[prev] == ab_vals[i]).astype(np.float32)

    return sequences
