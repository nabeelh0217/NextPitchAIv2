"""
NextPitchAI v6 — Step 2: Preprocess into training arrays
==========================================================
Reads the raw Statcast parquet and produces training arrays with
leakage-free engineered features. Every "prior" statistic (arsenal,
matchup history, batter profiles) is computed as an EXPANDING mean
that excludes the current pitch — the model never sees information
from the future or from the pitch it is predicting.

v6: the target is the ACTUAL pitch type (10 canonical Statcast codes),
not a coarse bucket. Every pitcher's repertoire is different — buckets
conflated one pitcher's slider with another's curveball. A per-pitcher
binary arsenal mask is emitted so the model (and the website) can
restrict predictions to pitches the pitcher actually throws.

Features:
  - Pitcher arsenal prior: expanding distribution of pitch types thrown
  - Matchup history: pitcher-vs-batter pitch-type distribution +
    familiarity, shrunk toward the pitcher's own arsenal
  - Batter seen-profile: what pitch types this batter gets fed
  - Batter whiff rates per pitch type
  - Times through the order, in-game pitch count, pitcher rest days
  - RISP flag, pitch number within at-bat
  - Ballpark ID (home team) + catcher ID for embeddings
  - Sequences: this pitcher's previous 8 pitches (type one-hot, physics,
    same-at-bat flag)
  - Arsenal mask: (N, 10) binary — which pitch types this pitcher throws

Requirements:
    pip install pandas numpy scikit-learn joblib pyarrow

Usage:
    python 02_preprocess.py

Output:
    data_v5/ folder with .npy arrays and .pkl/.json artifacts
    (v5 file naming kept so run_pipeline.bat's skip checks still work)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "statcast_raw_v5.parquet"
RAW_PATH_FALLBACK = BASE_DIR / "statcast_raw_v4.parquet"
OUT_DIR = BASE_DIR / "data_v5"
OUT_DIR.mkdir(exist_ok=True)

SEQ_LEN = 8  # lookback window (this pitcher's previous pitches)

# Minimum appearances for a player to get their own embedding.
# Below threshold -> <UNK> token (id 0).
MIN_PITCHER_APPEARANCES = 100
MIN_BATTER_APPEARANCES = 50
MIN_CATCHER_APPEARANCES = 200

# Smoothing strengths (pseudo-counts) for expanding priors
ARSENAL_SMOOTHING = 25.0    # toward league pitch-type distribution
MATCHUP_SMOOTHING = 10.0    # toward pitcher's own arsenal prior
SEEN_SMOOTHING = 25.0       # toward league pitch-type distribution
WHIFF_SMOOTHING = 20.0      # toward league whiff rate per pitch type

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
      0..K-1 : one-hot pitch type
      K      : release_speed
      K+1    : plate_x
      K+2    : plate_z
      K+3    : release_spin_rate
      K+4    : pfx_x (horizontal movement)
      K+5    : pfx_z (vertical movement)
    """
    n_cont = 6
    feats = np.zeros((len(df), N_PITCH + n_cont), dtype=np.float32)
    for i in range(N_PITCH):
        feats[pitch_idx == i, i] = 1.0
    feats[:, N_PITCH + 0] = df["release_speed"].fillna(0).values
    feats[:, N_PITCH + 1] = df["plate_x"].fillna(0).values
    feats[:, N_PITCH + 2] = df["plate_z"].fillna(0).values
    feats[:, N_PITCH + 3] = df["release_spin_rate"].fillna(0).values
    feats[:, N_PITCH + 4] = df["pfx_x"].fillna(0).values
    feats[:, N_PITCH + 5] = df["pfx_z"].fillna(0).values
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


def main():
    print("Loading raw data...")
    path = RAW_PATH if RAW_PATH.exists() else RAW_PATH_FALLBACK
    print(f"  Using {path.name}")
    df = pd.read_parquet(path)
    print(f"Raw rows: {len(df):,}")

    # ---------------------------
    # Filter unusable rows
    # ---------------------------
    df = df[df["pitch_type"].isin(PITCH_TYPE_CANON.keys())].copy()
    print(f"After pitch_type filter: {len(df):,}")

    df = df.dropna(subset=["balls", "strikes", "outs_when_up", "inning",
                           "pitcher", "batter", "stand", "p_throws"])
    print(f"After dropping NaN critical fields: {len(df):,}")

    # Chronological sort — REQUIRED for expanding priors and sequences
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    else:
        print("WARNING: no game_date column (old v4 scrape). "
              "Priors will use game_pk order; days_rest falls back to default.")
        df["game_date"] = pd.Timestamp("2000-01-01")
        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    df = df.reset_index(drop=True)

    # ---------------------------
    # Target labels (canonical pitch type)
    # ---------------------------
    df["canon"] = df["pitch_type"].map(PITCH_TYPE_CANON)
    y_labels = df["canon"].map(PITCH_TO_ID).values.astype(np.int64)
    pitch_onehot = np.eye(N_PITCH, dtype=np.float32)[y_labels]

    print("\nTarget distribution:")
    for p, i in sorted(PITCH_TO_ID.items(), key=lambda x: x[1]):
        count = int((y_labels == i).sum())
        print(f"  {p}: {count:,} ({count/len(y_labels)*100:.1f}%)")

    league_dist = pitch_onehot.mean(axis=0)

    # ---------------------------
    # Player / park / catcher ID mappings
    # ---------------------------
    pitcher_map = build_id_mapping(df["pitcher"], MIN_PITCHER_APPEARANCES)
    batter_map = build_id_mapping(df["batter"], MIN_BATTER_APPEARANCES)
    X_pitcher_id = encode_ids(df["pitcher"], pitcher_map)
    X_batter_id = encode_ids(df["batter"], batter_map)
    print(f"\nPitcher embeddings: {len(pitcher_map):,} unique + 1 UNK")
    print(f"Batter embeddings:  {len(batter_map):,} unique + 1 UNK")

    if "home_team" in df.columns:
        park_map = {t: i + 1 for i, t in enumerate(sorted(df["home_team"].dropna().unique()))}
        X_park_id = encode_ids(df["home_team"], park_map)
    else:
        park_map = {}
        X_park_id = np.zeros(len(df), dtype=np.int32)
    print(f"Parks: {len(park_map)} + 1 UNK")

    if "fielder_2" in df.columns:
        catcher_map = build_id_mapping(df["fielder_2"].dropna(), MIN_CATCHER_APPEARANCES)
        X_catcher_id = encode_ids(df["fielder_2"], catcher_map)
    else:
        catcher_map = {}
        X_catcher_id = np.zeros(len(df), dtype=np.int32)
    print(f"Catcher embeddings: {len(catcher_map):,} unique + 1 UNK")

    # ---------------------------
    # Handedness
    # ---------------------------
    hand_map_batter = {"R": 0, "L": 1, "S": 2}
    hand_map_pitcher = {"R": 0, "L": 1}
    X_batter_hand = np.array([hand_map_batter.get(h, 0) for h in df["stand"]], dtype=np.int32)
    X_pitcher_hand = np.array([hand_map_pitcher.get(h, 0) for h in df["p_throws"]], dtype=np.int32)

    # ---------------------------
    # Arsenal mask (full-data repertoire membership)
    # ---------------------------
    # 1 where this pitcher threw that pitch type at least once in the
    # dataset. Repertoire membership is stable descriptive info (the
    # website supplies it at inference too), NOT label leakage — and by
    # construction every training label is unmasked. The model's softmax
    # is restricted to these classes, so it never wastes capacity on
    # pitches this pitcher cannot throw.
    print("\nBuilding arsenal masks...")
    arsenal_sets = df.groupby("pitcher")["canon"].agg(set)
    mask_by_pitcher = {
        pid: np.array([1.0 if p in s else 0.0 for p in PITCH_CLASSES],
                      dtype=np.float32)
        for pid, s in arsenal_sets.items()
    }
    X_arsenal_mask = np.stack([mask_by_pitcher[pid] for pid in df["pitcher"]])
    avg_arsenal = X_arsenal_mask.sum(axis=1).mean()
    print(f"Average arsenal size: {avg_arsenal:.2f} pitch types")

    # ---------------------------
    # Expanding priors (leakage-free, pitch-type granularity)
    # ---------------------------
    print("\nComputing pitcher arsenal priors (expanding, leak-free)...")
    arsenal_prior, arsenal_total = expanding_prior_dist(
        df, ["pitcher"], pitch_onehot, ARSENAL_SMOOTHING, league_dist)

    print("Computing pitcher-vs-batter matchup history...")
    matchup_dist, matchup_total = expanding_prior_dist(
        df, ["pitcher", "batter"], pitch_onehot, MATCHUP_SMOOTHING, arsenal_prior)
    matchup_familiarity = np.log1p(matchup_total)[:, None].astype(np.float32)

    print("Computing batter seen-pitch profiles...")
    seen_prior, _ = expanding_prior_dist(
        df, ["batter"], pitch_onehot, SEEN_SMOOTHING, league_dist)

    print("Computing batter whiff rates per pitch type...")
    desc = df["description"].fillna("")
    swing_flag = desc.isin(SWING_DESCRIPTIONS).values.astype(np.float32)
    whiff_flag = desc.isin(WHIFF_DESCRIPTIONS).values.astype(np.float32)
    swing_onehot = pitch_onehot * swing_flag[:, None]
    whiff_onehot = pitch_onehot * whiff_flag[:, None]
    league_whiff_rate = (whiff_onehot.sum(axis=0)
                         / np.maximum(swing_onehot.sum(axis=0), 1.0))
    batter_whiff = expanding_prior_rate(
        df, ["batter"], whiff_onehot, swing_onehot, WHIFF_SMOOTHING, league_whiff_rate)

    # ---------------------------
    # Context features
    # ---------------------------
    print("\nBuilding game-state context features...")
    ctx_state, ctx_state_names = build_context_features(df)

    prior_names = (
        [f"arsenal_{p}" for p in PITCH_CLASSES]
        + [f"matchup_{p}" for p in PITCH_CLASSES] + ["matchup_familiarity"]
        + [f"seen_{p}" for p in PITCH_CLASSES]
        + [f"whiff_{p}" for p in PITCH_CLASSES]
    )
    X_ctx = np.concatenate([
        ctx_state, arsenal_prior, matchup_dist, matchup_familiarity,
        seen_prior, batter_whiff,
    ], axis=1)
    ctx_feature_names = ctx_state_names + prior_names
    print(f"Context shape: {X_ctx.shape} ({len(ctx_feature_names)} features)")

    ctx_scaler = StandardScaler()
    X_ctx_scaled = ctx_scaler.fit_transform(X_ctx).astype(np.float32)

    # ---------------------------
    # Sequence features
    # ---------------------------
    print("\nBuilding per-pitch feature vectors...")
    pitch_feats = build_pitch_features(df, y_labels.astype(np.int32))

    seq_scaler = StandardScaler()
    continuous = pitch_feats[:, N_PITCH:]
    pitch_feats[:, N_PITCH:] = seq_scaler.fit_transform(continuous).astype(np.float32)

    print(f"Building per-pitcher sequences of length {SEQ_LEN} "
          "(this may take a few minutes)...")
    X_seq = build_sequences(pitch_feats, df, SEQ_LEN)
    print(f"Sequence shape: {X_seq.shape}")

    # ---------------------------
    # Website inference artifacts (FULL-data aggregates)
    # ---------------------------
    print("\nBuilding inference artifacts for the website...")
    arsenal, priors, masks_json = {}, {}, {}
    for pid, group in df.groupby("pitcher"):
        pid_str = str(int(pid))
        counts = group["canon"].value_counts()
        total = counts.sum()
        arsenal[pid_str] = counts.index.tolist()
        priors[pid_str] = {p: float(counts.get(p, 0) / total) for p in PITCH_CLASSES}
        masks_json[pid_str] = [int(v) for v in mask_by_pitcher[pid]]

    matchup_table = (df.groupby(["pitcher", "batter", "canon"])
                     .size().unstack(fill_value=0).reset_index())
    matchup_table.to_parquet(OUT_DIR / "matchup_table_v5.parquet", index=False)

    batter_profiles = pd.DataFrame({
        "batter": df["batter"].values,
        **{f"seen_{p}": seen_prior[:, i] for i, p in enumerate(PITCH_CLASSES)},
        **{f"whiff_{p}": batter_whiff[:, i] for i, p in enumerate(PITCH_CLASSES)},
    }).groupby("batter").last().reset_index()
    batter_profiles.to_parquet(OUT_DIR / "batter_profiles_v5.parquet", index=False)

    # ---------------------------
    # Save everything
    # ---------------------------
    print(f"\nSaving arrays to {OUT_DIR}...")
    np.save(OUT_DIR / "X_seq.npy", X_seq)
    np.save(OUT_DIR / "X_ctx.npy", X_ctx_scaled)
    np.save(OUT_DIR / "X_pitcher_id.npy", X_pitcher_id)
    np.save(OUT_DIR / "X_batter_id.npy", X_batter_id)
    np.save(OUT_DIR / "X_pitcher_hand.npy", X_pitcher_hand)
    np.save(OUT_DIR / "X_batter_hand.npy", X_batter_hand)
    np.save(OUT_DIR / "X_park_id.npy", X_park_id)
    np.save(OUT_DIR / "X_catcher_id.npy", X_catcher_id)
    np.save(OUT_DIR / "X_arsenal_mask.npy", X_arsenal_mask)
    np.save(OUT_DIR / "y_labels.npy", y_labels)

    joblib.dump(ctx_scaler, OUT_DIR / "context_scaler_v5.pkl")
    joblib.dump(seq_scaler, OUT_DIR / "seq_scaler_v5.pkl")
    joblib.dump(PITCH_TO_ID, OUT_DIR / "pitch_map_v5.pkl")
    joblib.dump(pitcher_map, OUT_DIR / "pitcher_id_map_v5.pkl")
    joblib.dump(batter_map, OUT_DIR / "batter_id_map_v5.pkl")
    joblib.dump(park_map, OUT_DIR / "park_id_map_v5.pkl")
    joblib.dump(catcher_map, OUT_DIR / "catcher_id_map_v5.pkl")

    with open(OUT_DIR / "pitcher_arsenal_v5.json", "w") as f:
        json.dump(arsenal, f)
    with open(OUT_DIR / "pitcher_priors_v5.json", "w") as f:
        json.dump(priors, f)
    with open(OUT_DIR / "pitcher_arsenal_mask_v5.json", "w") as f:
        json.dump({"pitch_classes": PITCH_CLASSES, "masks": masks_json}, f)
    with open(OUT_DIR / "league_stats_v5.json", "w") as f:
        json.dump({
            "league_pitch_dist": {p: float(league_dist[i]) for i, p in enumerate(PITCH_CLASSES)},
            "league_whiff_rate": {p: float(league_whiff_rate[i]) for i, p in enumerate(PITCH_CLASSES)},
            "smoothing": {
                "arsenal": ARSENAL_SMOOTHING, "matchup": MATCHUP_SMOOTHING,
                "seen": SEEN_SMOOTHING, "whiff": WHIFF_SMOOTHING,
            },
        }, f, indent=2)

    meta = {
        "version": 6,
        "n_samples": len(y_labels),
        "seq_len": SEQ_LEN,
        "seq_feats": X_seq.shape[2],
        "ctx_feats": X_ctx_scaled.shape[1],
        "ctx_feature_names": ctx_feature_names,
        "n_pitch_types": N_PITCH,
        "pitch_classes": PITCH_CLASSES,
        "n_pitcher_ids": len(pitcher_map) + 1,
        "n_batter_ids": len(batter_map) + 1,
        "n_park_ids": len(park_map) + 1,
        "n_catcher_ids": len(catcher_map) + 1,
        "min_pitcher_appearances": MIN_PITCHER_APPEARANCES,
        "min_batter_appearances": MIN_BATTER_APPEARANCES,
        "min_catcher_appearances": MIN_CATCHER_APPEARANCES,
    }
    with open(OUT_DIR / "meta_v5.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved {len(y_labels):,} samples.")
    print(f"Metadata: {json.dumps({k: v for k, v in meta.items() if k != 'ctx_feature_names'}, indent=2)}")


if __name__ == "__main__":
    main()
