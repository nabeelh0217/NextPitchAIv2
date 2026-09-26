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
  - Sequences: this pitcher's previous 8 pitches (type one-hot, OUTCOME
    one-hot, physics, same-at-bat flag)
  - Arsenal mask: (N, 10) binary — which pitch types this pitcher throws
  - Location: attack-zone target (heart/shadow/chase/waste) plus
    leakage-free pitcher and batter location priors

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
# Feature vocabulary + builders
# =========================
# Defined in pitch_features.py so the website builds features with
# exactly this code rather than a copy that can drift. See that
# module's docstring.
from pitch_features import (  # noqa: E402
    PITCH_TYPE_CANON, PITCH_CLASSES, PITCH_TO_ID, N_PITCH,
    SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS,
    OUTCOME_CLASSES, N_OUTCOME, OUTCOME_OF,
    ZONE_CLASSES, N_ZONE, PLATE_HALF_FT, ZONE_EDGES,
    compute_attack_zone, build_id_mapping, encode_ids,
    expanding_prior_dist, expanding_prior_rate,
    compute_times_through_order, compute_days_rest,
    build_context_features, build_pitch_features, build_sequences,
)


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

    # ---------------------------
    # Location target + leakage-free location priors
    # ---------------------------
    print("\nComputing attack zones...")
    y_zone = compute_attack_zone(df)
    valid = y_zone >= 0
    print(f"  usable location rows: {valid.sum():,} "
          f"({valid.mean():.1%}); unusable are excluded from the zone loss")
    for i, z in enumerate(ZONE_CLASSES):
        n = int((y_zone == i).sum())
        print(f"  {z}: {n:,} ({n/max(valid.sum(),1)*100:.1f}%)")

    # Same expanding pattern as the arsenal prior: where has this pitcher
    # put the ball BEFORE this pitch. Rows with no usable zone contribute
    # nothing to the running counts.
    zone_onehot = np.zeros((len(df), N_ZONE), dtype=np.float32)
    zone_onehot[valid, y_zone[valid]] = 1.0
    league_zone = (zone_onehot.sum(0) / max(zone_onehot.sum(), 1.0)).astype(np.float32)
    print("Computing pitcher location priors (expanding, leak-free)...")
    zone_prior, _ = expanding_prior_dist(
        df, ["pitcher"], zone_onehot, ARSENAL_SMOOTHING, league_zone)

    print("Computing batter location-seen profiles...")
    zone_seen, _ = expanding_prior_dist(
        df, ["batter"], zone_onehot, SEEN_SMOOTHING, league_zone)

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
        seen_prior, batter_whiff, zone_prior, zone_seen,
    ], axis=1)
    ctx_feature_names = (ctx_state_names + prior_names
                         + [f"zoneprior_{z}" for z in ZONE_CLASSES]
                         + [f"zoneseen_{z}" for z in ZONE_CLASSES])
    print(f"Context shape: {X_ctx.shape} ({len(ctx_feature_names)} features)")

    ctx_scaler = StandardScaler()
    X_ctx_scaled = ctx_scaler.fit_transform(X_ctx).astype(np.float32)

    # ---------------------------
    # Sequence features
    # ---------------------------
    print("\nBuilding per-pitch feature vectors...")
    pitch_feats = build_pitch_features(df, y_labels.astype(np.int32))

    seq_scaler = StandardScaler()
    cont_start = N_PITCH + N_OUTCOME
    continuous = pitch_feats[:, cont_start:]
    pitch_feats[:, cont_start:] = seq_scaler.fit_transform(continuous).astype(np.float32)

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
    np.save(OUT_DIR / "y_zone.npy", y_zone)
    # Season enables a temporal split (train on past years, validate on
    # the most recent) without re-reading the parquet.
    np.save(OUT_DIR / "X_season.npy",
            df["game_date"].dt.year.values.astype(np.int32))

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
        "outcome_classes": OUTCOME_CLASSES,
        "zone_classes": ZONE_CLASSES,
        "n_zones": N_ZONE,
        "seasons": sorted(int(s) for s in df["game_date"].dt.year.unique()),
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
