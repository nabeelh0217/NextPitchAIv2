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
        Fit mode: builds data_v5/ from scratch. ID maps and scalers are
        FITTED to the data, so any model trained on an earlier data_v5/
        no longer matches it and must be retrained.

    python 02_preprocess.py --freeze-from data_v5 --out-dir data_ext
        Frozen mode: extends an existing dataset forward in time WITHOUT
        refitting anything. ID maps, scalers and league fallbacks come
        from --freeze-from, so the model trained there can be scored on
        seasons it has never seen. Rows from the original seasons are
        checked to reproduce the frozen arrays exactly; if they do not,
        the run fails rather than hand over features the model was not
        trained on.

Output:
    data_v5/ (or --out-dir) with .npy arrays and .pkl/.json artifacts
    (v5 file naming kept so run_pipeline.bat's skip checks still work)
"""

import argparse
import hashlib
import json
import shutil
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
EXT_DIR = BASE_DIR / "data_ext"     # default --out-dir in frozen mode

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

# Arrays that must come out identical for the original seasons when a
# dataset is extended in frozen mode. X_arsenal_mask is deliberately not
# here: it is built over every row, so a pitcher who adds a pitch in the
# new season legitimately changes his old rows. Nothing scores with it.
FROZEN_CHECK = [
    "X_ctx", "X_seq", "X_pitcher_id", "X_batter_id", "X_pitcher_hand",
    "X_batter_hand", "X_park_id", "X_catcher_id", "X_season",
    "y_labels", "y_zone",
]
# Carried across so the extended directory is self-contained: the model,
# and the record of how it was split and trained.
FROZEN_COPY = ["best_model_v5.keras", "split_v5.json",
               "training_history_v5.json", "arsenal_mask_table_v5.npy"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_against_frozen(frozen: Path, out_dir: Path, n_old: int) -> list:
    """
    Compare the first n_old rows of every FROZEN_CHECK array in out_dir
    with the frozen array. Returns a list of human-readable problems;
    empty means the original seasons reproduced.

    This is the proof that frozen mode builds features the way the model
    was trained on them. Without it a silent difference in one column
    gives confident, plausible, wrong scores on the new season.
    """
    problems = []
    for name in FROZEN_CHECK:
        a_path, b_path = frozen / f"{name}.npy", out_dir / f"{name}.npy"
        if not a_path.exists():
            continue                      # older dataset without this array
        a = np.load(a_path, mmap_mode="r")
        b = np.load(b_path, mmap_mode="r")
        if len(a) != n_old or a.shape[1:] != b.shape[1:]:
            problems.append(f"{name}: frozen shape {a.shape} vs rebuilt "
                            f"{(n_old,) + b.shape[1:]}")
            continue
        worst, bad_rows = 0.0, 0
        for i in range(0, n_old, 200_000):
            j = min(i + 200_000, n_old)       # b is longer: it has the new rows
            x, y = np.asarray(a[i:j]), np.asarray(b[i:j])
            if x.dtype.kind == "f":
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                d = d.reshape(len(d), -1).max(axis=1) if d.ndim > 1 else d
                worst = max(worst, float(d.max(initial=0.0)))
                bad_rows += int((d > 1e-5).sum())
            else:
                neq = x != y
                bad_rows += int(neq.reshape(len(neq), -1).any(axis=1).sum()
                                if neq.ndim > 1 else neq.sum())
        status = "ok" if bad_rows == 0 else "MISMATCH"
        print(f"    {name:<16} {status:<9}"
              + (f"max |diff| {worst:.2e}" if a.dtype.kind == "f" else "exact")
              + (f"   {bad_rows:,} rows differ" if bad_rows else ""))
        if bad_rows:
            problems.append(f"{name}: {bad_rows:,} of {n_old:,} original rows differ")
    return problems


def main():
    ap = argparse.ArgumentParser(description="Build model arrays from the raw parquet.")
    ap.add_argument("--freeze-from", type=Path, default=None, metavar="DIR",
                    help="reuse this dataset's ID maps, scalers and league "
                         "fallbacks instead of fitting new ones")
    ap.add_argument("--out-dir", type=Path, default=None, metavar="DIR",
                    help="where to write (default: data_v5, or data_ext with --freeze-from)")
    ap.add_argument("--raw", type=Path, default=None, metavar="PARQUET")
    ap.add_argument("--fit-through", type=int, default=None, metavar="SEASON",
                    help="fit the ID maps, scalers and league fallbacks on "
                         "seasons up to this one only; later seasons are "
                         "still encoded and written, as held-out data")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="frozen mode: keep the output even if the original "
                         "seasons do not reproduce (it is marked unverified)")
    args = ap.parse_args()

    frozen = args.freeze_from.resolve() if args.freeze_from else None
    out_dir = (args.out_dir or (EXT_DIR if frozen else OUT_DIR)).resolve()
    if frozen is not None and args.fit_through is not None:
        raise SystemExit("--fit-through fits new artifacts and --freeze-from "
                         "reuses old ones; pass one or the other")
    if frozen is not None:
        if not (frozen / "meta_v5.json").exists():
            raise SystemExit(f"--freeze-from {frozen}: no meta_v5.json there")
        if out_dir == frozen:
            raise SystemExit("--out-dir must differ from --freeze-from: frozen "
                             "mode never overwrites the dataset it extends")
        frozen_meta = json.loads((frozen / "meta_v5.json").read_text())
        # The seasons the frozen artifacts were FITTED on. A dataset that
        # was itself an extension keeps pointing at the original fit.
        fit_seasons = set(frozen_meta.get("fit_seasons") or frozen_meta.get("seasons")
                          or np.unique(np.load(frozen / "X_season.npy")).tolist())
        print(f"FROZEN mode: artifacts from {frozen}")
        print(f"  fitted on seasons {sorted(fit_seasons)}; nothing is refitted")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    path = args.raw or (RAW_PATH if RAW_PATH.exists() else RAW_PATH_FALLBACK)
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

    season_of_row = df["game_date"].dt.year.values.astype(np.int32)
    all_seasons = sorted(int(s) for s in np.unique(season_of_row))
    if frozen is None:
        # Everything fitted below — ID maps, scalers, league fallbacks —
        # is a statistic of the rows it is fitted on. A season that goes
        # into the fit has been seen, however lightly, and cannot be a
        # clean test season afterwards. --fit-through holds later seasons
        # out of the fit while still encoding them.
        if args.fit_through is None:
            fit_rows = np.ones(len(df), dtype=bool)
        else:
            fit_rows = season_of_row <= args.fit_through
            if not fit_rows.any():
                raise SystemExit(f"--fit-through {args.fit_through}: the raw data "
                                 f"starts later (seasons {all_seasons})")
        fit_seasons = set(int(s) for s in np.unique(season_of_row[fit_rows]))
        held = [s for s in all_seasons if s not in fit_seasons]
        print(f"\nFitting ID maps, scalers and league fallbacks on {sorted(fit_seasons)}"
              + (f"; held out of the fit: {held}" if held else ""))
        if not held and len(all_seasons) > 1:
            print(f"  Every season is in the fit, so none of them can later be a clean "
                  f"test season.\n  To keep {all_seasons[-1]} for that: "
                  f"--fit-through {all_seasons[-2]}")
    else:
        # League fallbacks are FIT statistics too. Computed over the new
        # season as well they would shift, slightly, every prior that is
        # smoothed toward them — including in rows the model trained on.
        fit_rows = np.isin(season_of_row, sorted(fit_seasons))
        if not fit_rows.any():
            raise SystemExit("frozen mode: the raw data holds none of the "
                             f"seasons the artifacts were fitted on {sorted(fit_seasons)}")
    league_dist = pitch_onehot[fit_rows].mean(axis=0)

    # ---------------------------
    # Player / park / catcher ID mappings
    # ---------------------------
    if frozen is None:
        # From fit rows only: a player first seen in a held-out season
        # must encode to 0, the unknown bucket, not get a row of his own
        # that training will never touch.
        pitcher_map = build_id_mapping(df["pitcher"][fit_rows], MIN_PITCHER_APPEARANCES)
        batter_map = build_id_mapping(df["batter"][fit_rows], MIN_BATTER_APPEARANCES)
    else:
        # The embedding rows a trained model learned are indexed by these
        # maps. A player who is not in them — a rookie from the new
        # season — encodes to 0, the unknown bucket the model was trained
        # with, exactly as he would at serve time.
        pitcher_map = joblib.load(frozen / "pitcher_id_map_v5.pkl")
        batter_map = joblib.load(frozen / "batter_id_map_v5.pkl")
    X_pitcher_id = encode_ids(df["pitcher"], pitcher_map)
    X_batter_id = encode_ids(df["batter"], batter_map)
    print(f"\nPitcher embeddings: {len(pitcher_map):,} unique + 1 UNK")
    print(f"Batter embeddings:  {len(batter_map):,} unique + 1 UNK")

    if frozen is not None:
        park_map = joblib.load(frozen / "park_id_map_v5.pkl")
        X_park_id = (encode_ids(df["home_team"], park_map) if "home_team" in df.columns
                     else np.zeros(len(df), dtype=np.int32))
    elif "home_team" in df.columns:
        park_map = {t: i + 1 for i, t in
                    enumerate(sorted(df["home_team"][fit_rows].dropna().unique()))}
        X_park_id = encode_ids(df["home_team"], park_map)
    else:
        park_map = {}
        X_park_id = np.zeros(len(df), dtype=np.int32)
    print(f"Parks: {len(park_map)} + 1 UNK")

    if frozen is not None:
        catcher_map = joblib.load(frozen / "catcher_id_map_v5.pkl")
        X_catcher_id = (encode_ids(df["fielder_2"], catcher_map) if "fielder_2" in df.columns
                        else np.zeros(len(df), dtype=np.int32))
    elif "fielder_2" in df.columns:
        catcher_map = build_id_mapping(df["fielder_2"][fit_rows].dropna(),
                                       MIN_CATCHER_APPEARANCES)
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
    league_zone = (zone_onehot[fit_rows].sum(0)
                   / max(zone_onehot[fit_rows].sum(), 1.0)).astype(np.float32)
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
    league_whiff_rate = (whiff_onehot[fit_rows].sum(axis=0)
                         / np.maximum(swing_onehot[fit_rows].sum(axis=0), 1.0))
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

    if frozen is None:
        ctx_scaler = StandardScaler()
        if fit_rows.all():
            X_ctx_scaled = ctx_scaler.fit_transform(X_ctx).astype(np.float32)
        else:
            ctx_scaler.fit(X_ctx[fit_rows])
            X_ctx_scaled = ctx_scaler.transform(X_ctx).astype(np.float32)
    else:
        if frozen_meta["ctx_feature_names"] != ctx_feature_names:
            raise SystemExit("frozen mode: the context features built by this "
                             "code are not the ones the frozen dataset has. "
                             "The code has changed since it was built.")
        ctx_scaler = joblib.load(frozen / "context_scaler_v5.pkl")
        X_ctx_scaled = ctx_scaler.transform(X_ctx).astype(np.float32)

    # ---------------------------
    # Sequence features
    # ---------------------------
    print("\nBuilding per-pitch feature vectors...")
    pitch_feats = build_pitch_features(df, y_labels.astype(np.int32))

    cont_start = N_PITCH + N_OUTCOME
    continuous = pitch_feats[:, cont_start:]
    if frozen is None:
        seq_scaler = StandardScaler()
        if fit_rows.all():
            pitch_feats[:, cont_start:] = seq_scaler.fit_transform(continuous).astype(np.float32)
        else:
            seq_scaler.fit(continuous[fit_rows])
            pitch_feats[:, cont_start:] = seq_scaler.transform(continuous).astype(np.float32)
    else:
        seq_scaler = joblib.load(frozen / "seq_scaler_v5.pkl")
        pitch_feats[:, cont_start:] = seq_scaler.transform(continuous).astype(np.float32)

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
    matchup_table.to_parquet(out_dir / "matchup_table_v5.parquet", index=False)

    batter_profiles = pd.DataFrame({
        "batter": df["batter"].values,
        **{f"seen_{p}": seen_prior[:, i] for i, p in enumerate(PITCH_CLASSES)},
        **{f"whiff_{p}": batter_whiff[:, i] for i, p in enumerate(PITCH_CLASSES)},
    }).groupby("batter").last().reset_index()
    batter_profiles.to_parquet(out_dir / "batter_profiles_v5.parquet", index=False)

    # ---------------------------
    # Save everything
    # ---------------------------
    print(f"\nSaving arrays to {out_dir}...")
    np.save(out_dir / "X_seq.npy", X_seq)
    np.save(out_dir / "X_ctx.npy", X_ctx_scaled)
    np.save(out_dir / "X_pitcher_id.npy", X_pitcher_id)
    np.save(out_dir / "X_batter_id.npy", X_batter_id)
    np.save(out_dir / "X_pitcher_hand.npy", X_pitcher_hand)
    np.save(out_dir / "X_batter_hand.npy", X_batter_hand)
    np.save(out_dir / "X_park_id.npy", X_park_id)
    np.save(out_dir / "X_catcher_id.npy", X_catcher_id)
    np.save(out_dir / "X_arsenal_mask.npy", X_arsenal_mask)
    np.save(out_dir / "y_labels.npy", y_labels)
    np.save(out_dir / "y_zone.npy", y_zone)
    # Season enables a temporal split (train on past years, validate on
    # the most recent) without re-reading the parquet.
    np.save(out_dir / "X_season.npy", season_of_row)
    # Per-row facts that are NOT model inputs but that honest scoring
    # needs: the raw pitcher id (encoded ids pool every rare pitcher into
    # one bucket, so a per-pitcher baseline or arsenal built on them is
    # wrong for exactly the pitchers a frozen model knows least about),
    # the game (for clustered confidence intervals) and the integer count.
    np.savez_compressed(
        out_dir / "row_meta_v5.npz",
        pitcher=df["pitcher"].values.astype(np.int64),
        batter=df["batter"].values.astype(np.int64),
        game_pk=df["game_pk"].values.astype(np.int64),
        game_date=(df["game_date"].dt.year * 10000 + df["game_date"].dt.month * 100
                   + df["game_date"].dt.day).values.astype(np.int32),
        balls=df["balls"].values.astype(np.int8),
        strikes=df["strikes"].values.astype(np.int8),
    )

    joblib.dump(ctx_scaler, out_dir / "context_scaler_v5.pkl")
    joblib.dump(seq_scaler, out_dir / "seq_scaler_v5.pkl")
    joblib.dump(PITCH_TO_ID, out_dir / "pitch_map_v5.pkl")
    joblib.dump(pitcher_map, out_dir / "pitcher_id_map_v5.pkl")
    joblib.dump(batter_map, out_dir / "batter_id_map_v5.pkl")
    joblib.dump(park_map, out_dir / "park_id_map_v5.pkl")
    joblib.dump(catcher_map, out_dir / "catcher_id_map_v5.pkl")

    with open(out_dir / "pitcher_arsenal_v5.json", "w") as f:
        json.dump(arsenal, f)
    with open(out_dir / "pitcher_priors_v5.json", "w") as f:
        json.dump(priors, f)
    with open(out_dir / "pitcher_arsenal_mask_v5.json", "w") as f:
        json.dump({"pitch_classes": PITCH_CLASSES, "masks": masks_json}, f)
    with open(out_dir / "league_stats_v5.json", "w") as f:
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
        # The seasons the ID maps, scalers and league fallbacks were
        # fitted on. Differs from "seasons" in a frozen extension, or when
        # --fit-through held later seasons out.
        "fit_seasons": sorted(int(s) for s in fit_seasons),
        "frozen_from": str(frozen) if frozen is not None else None,
        "n_pitcher_ids": len(pitcher_map) + 1,
        "n_batter_ids": len(batter_map) + 1,
        "n_park_ids": len(park_map) + 1,
        "n_catcher_ids": len(catcher_map) + 1,
        "min_pitcher_appearances": MIN_PITCHER_APPEARANCES,
        "min_batter_appearances": MIN_BATTER_APPEARANCES,
        "min_catcher_appearances": MIN_CATCHER_APPEARANCES,
    }
    if frozen is not None:
        # The original seasons must be a prefix of the new arrays: same
        # rows, same order, with every new row strictly later.
        n_old = int(fit_rows.sum())
        prefix_ok = bool(fit_rows[:n_old].all())
        print(f"\nChecking the {n_old:,} original-season rows against {frozen.name}/ ...")
        problems = ([] if prefix_ok else
                    ["rows from the original seasons are not a contiguous prefix: "
                     "the new data is interleaved with the old in time"])
        if prefix_ok:
            problems = verify_against_frozen(frozen, out_dir, n_old)
        meta["verified_against_frozen"] = not problems
        if problems:
            print("\n" + "!" * 68)
            print("FROZEN CHECK FAILED — the original seasons did not reproduce:")
            for prob in problems:
                print(f"  - {prob}")
            print("Either the raw rows for those seasons have changed since the")
            print("dataset was built, or the feature code has. Scores on the new")
            print("season would not be for the features the model was trained on.")
            print("!" * 68)
            if not args.allow_mismatch:
                raise SystemExit(4)
        else:
            print("  every original row reproduced: the new season is built with "
                  "exactly the features the model was trained on.")

        for name in FROZEN_COPY:
            if (frozen / name).exists():
                shutil.copy2(frozen / name, out_dir / name)
        # What the site was showing before this dataset existed, kept so
        # the gate can reproduce it and show what the corrections changed.
        if (frozen / "product_claim_v5.json").exists():
            shutil.copy2(frozen / "product_claim_v5.json",
                         out_dir / "product_claim_v5.published.json")
        model = out_dir / "best_model_v5.keras"
        if model.exists():
            meta["model_sha256"] = sha256_of(model)

    with open(out_dir / "meta_v5.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved {len(y_labels):,} samples to {out_dir}.")
    print(f"Metadata: {json.dumps({k: v for k, v in meta.items() if k != 'ctx_feature_names'}, indent=2)}")


if __name__ == "__main__":
    main()
