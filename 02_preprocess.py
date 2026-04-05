"""
NextPitchAI v4 — Step 2: Preprocess into training arrays
==========================================================
Reads the raw Statcast parquet and produces:
  - X_seq:         (N, 5, F_seq)  pitch sequence features
  - X_ctx:         (N, F_ctx)     numeric context (count, inning, bases, score)
  - X_pitcher_id:  (N,)           MLBAM pitcher IDs (int-encoded)
  - X_batter_id:   (N,)           MLBAM batter IDs (int-encoded)
  - X_pitcher_hand:(N,)           pitcher hand (0=R, 1=L)
  - X_batter_hand: (N,)           batter hand (0=R, 1=L, 2=S)
  - y_labels:      (N,)           bucket-encoded target

Requirements:
    pip install pandas numpy scikit-learn joblib

Usage:
    python 02_preprocess.py

Output:
    data_v4/ folder with .npy arrays and .pkl artifacts
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================
BASE_DIR = Path(r"C:\Users\nabzt\OneDrive\Desktop\PitchGPT")
RAW_PATH = BASE_DIR / "statcast_raw_v4.parquet"
OUT_DIR  = BASE_DIR / "data_v4"
OUT_DIR.mkdir(exist_ok=True)

SEQ_LEN = 5  # lookback window

# Minimum appearances for a pitcher/batter to get their own embedding.
# Anyone below this threshold gets mapped to an <UNK> token.
# This prevents the embedding table from being huge with rarely-seen players.
MIN_PITCHER_APPEARANCES = 100  # ~100 pitches ≈ a few games
MIN_BATTER_APPEARANCES  = 50   # ~50 plate appearances

# =========================
# Pitch type → bucket mapping
# =========================
# Based on standard Statcast pitch codes
PITCH_TO_BUCKET = {
    # Fastballs
    "FF": "fastball",   # four-seam
    "FT": "fastball",   # two-seam (older code)
    "SI": "fastball",   # sinker
    "FC": "fastball",   # cutter
    "FA": "fastball",   # generic fastball

    # Breaking
    "SL": "breaking",   # slider
    "CU": "breaking",   # curveball
    "KC": "breaking",   # knuckle-curve
    "SV": "breaking",   # sweeper
    "CS": "breaking",   # slow curve
    "ST": "breaking",   # sweeping curve (newer code)

    # Offspeed
    "CH": "offspeed",   # changeup
    "FS": "offspeed",   # splitter
    "FO": "offspeed",   # forkball
    "SC": "offspeed",   # screwball

    # Special
    "KN": "special",    # knuckleball
    "EP": "special",    # eephus

    # Other / unknown
    "PO": "other",      # pitchout
    "IN": "other",      # intentional ball
    "UN": "other",      # unknown
    "AB": "other",      # automatic ball
}

BUCKET_CLASSES = sorted(list(set(PITCH_TO_BUCKET.values())))  # alphabetical
BUCKET_TO_ID = {b: i for i, b in enumerate(BUCKET_CLASSES)}
print(f"Bucket classes: {BUCKET_CLASSES}")
print(f"Bucket IDs: {BUCKET_TO_ID}")


def build_id_mapping(series: pd.Series, min_count: int) -> dict:
    """
    Build an ID mapping where frequent players get unique IDs
    and rare players map to 0 (the <UNK> token).
    Returns: {mlbam_id: encoded_int}, where 0 = UNK.
    """
    counts = series.value_counts()
    frequent = counts[counts >= min_count].index.tolist()

    mapping = {pid: (i + 1) for i, pid in enumerate(sorted(frequent))}
    # 0 is reserved for UNK
    return mapping


def encode_ids(series: pd.Series, mapping: dict) -> np.ndarray:
    """Encode MLBAM IDs using the mapping. Unknown → 0."""
    return np.array([mapping.get(v, 0) for v in series], dtype=np.int32)


def build_context_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the numeric context vector for each pitch.
    Features (in order):
      0: balls (0-3)
      1: strikes (0-2)
      2: outs (0-2)
      3: inning (capped at 9)
      4: is_top (1 if top of inning, 0 if bottom)
      5: score_diff (fld_score - bat_score, clipped to [-6, 6])
      6: on_1b (0 or 1)
      7: on_2b (0 or 1)
      8: on_3b (0 or 1)
    """
    ctx = np.zeros((len(df), 9), dtype=np.float32)

    ctx[:, 0] = df["balls"].fillna(0).values.astype(np.float32)
    ctx[:, 1] = df["strikes"].fillna(0).values.astype(np.float32)
    ctx[:, 2] = df["outs_when_up"].fillna(0).values.astype(np.float32)
    ctx[:, 3] = np.minimum(df["inning"].fillna(1).values, 9).astype(np.float32)
    ctx[:, 4] = (df["inning_topbot"] == "Top").astype(np.float32)

    score_diff = (df["fld_score"].fillna(0) - df["bat_score"].fillna(0)).values
    ctx[:, 5] = np.clip(score_diff, -6, 6).astype(np.float32)

    ctx[:, 6] = df["on_1b"].notna().astype(np.float32)
    ctx[:, 7] = df["on_2b"].notna().astype(np.float32)
    ctx[:, 8] = df["on_3b"].notna().astype(np.float32)

    return ctx


def build_sequence_features(df: pd.DataFrame) -> np.ndarray:
    """
    For each pitch in the at-bat, build a feature vector from the PREVIOUS pitch.
    Sequence features per timestep:
      0-4:  one-hot bucket of previous pitch (5 dims)
      5:    release_speed (normalized later)
      6:    plate_x
      7:    plate_z
      8:    release_spin_rate (normalized later)
    Total: 9 features per timestep (same as your v3)
    """
    n_bucket = len(BUCKET_CLASSES)
    n_feats = n_bucket + 4  # 5 bucket dims + speed + plate_x + plate_z + spin

    # Map each row's pitch_type to bucket index
    bucket_idx = np.array([
        BUCKET_TO_ID.get(PITCH_TO_BUCKET.get(pt, "other"), BUCKET_TO_ID["other"])
        for pt in df["pitch_type"]
    ], dtype=np.int32)

    speed = df["release_speed"].fillna(0).values.astype(np.float32)
    px = df["plate_x"].fillna(0).values.astype(np.float32)
    pz = df["plate_z"].fillna(0).values.astype(np.float32)
    spin = df["release_spin_rate"].fillna(0).values.astype(np.float32)

    # Build per-pitch feature vectors
    pitch_feats = np.zeros((len(df), n_feats), dtype=np.float32)
    for i in range(n_bucket):
        pitch_feats[bucket_idx == i, i] = 1.0
    pitch_feats[:, n_bucket]     = speed
    pitch_feats[:, n_bucket + 1] = px
    pitch_feats[:, n_bucket + 2] = pz
    pitch_feats[:, n_bucket + 3] = spin

    return pitch_feats


def build_sequences(pitch_feats: np.ndarray, game_ids: np.ndarray,
                    ab_ids: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build (N, seq_len, F) sequence arrays from per-pitch features.
    For each pitch, look back up to seq_len previous pitches in the SAME game.
    Cross-game boundaries are NOT bridged (pad with zeros).
    """
    N, F = pitch_feats.shape
    sequences = np.zeros((N, seq_len, F), dtype=np.float32)

    for i in range(N):
        # Look back within the same game
        lookback = []
        for j in range(1, seq_len + 1):
            idx = i - j
            if idx < 0:
                break
            if game_ids[idx] != game_ids[i]:
                break  # different game — don't cross boundary
            lookback.append(pitch_feats[idx])

        # Fill from the end (most recent = last position)
        lookback.reverse()
        start = seq_len - len(lookback)
        for k, feat in enumerate(lookback):
            sequences[i, start + k] = feat

    return sequences


def main():
    print("Loading raw data...")
    df = pd.read_parquet(RAW_PATH)
    print(f"Raw rows: {len(df):,}")

    # ---------------------------
    # Filter out unusable rows
    # ---------------------------
    # Drop pitches with no type or rare/junk types
    df = df[df["pitch_type"].isin(PITCH_TO_BUCKET.keys())].copy()
    print(f"After pitch_type filter: {len(df):,}")

    # Drop rows missing critical fields
    df = df.dropna(subset=["balls", "strikes", "outs_when_up", "inning",
                           "pitcher", "batter", "stand", "p_throws"])
    print(f"After dropping NaN critical fields: {len(df):,}")

    # Sort chronologically (critical for sequence building)
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    # ---------------------------
    # Target labels
    # ---------------------------
    df["bucket"] = df["pitch_type"].map(PITCH_TO_BUCKET)
    y_labels = np.array([BUCKET_TO_ID[b] for b in df["bucket"]], dtype=np.int64)
    print(f"\nTarget distribution:")
    for b, i in sorted(BUCKET_TO_ID.items(), key=lambda x: x[1]):
        count = (y_labels == i).sum()
        print(f"  {b}: {count:,} ({count/len(y_labels)*100:.1f}%)")

    # ---------------------------
    # Player ID mappings
    # ---------------------------
    pitcher_map = build_id_mapping(df["pitcher"], MIN_PITCHER_APPEARANCES)
    batter_map  = build_id_mapping(df["batter"],  MIN_BATTER_APPEARANCES)

    print(f"\nPitcher embeddings: {len(pitcher_map):,} unique + 1 UNK")
    print(f"Batter embeddings:  {len(batter_map):,} unique + 1 UNK")

    X_pitcher_id = encode_ids(df["pitcher"], pitcher_map)
    X_batter_id  = encode_ids(df["batter"],  batter_map)

    # ---------------------------
    # Handedness
    # ---------------------------
    hand_map_batter  = {"R": 0, "L": 1, "S": 2}
    hand_map_pitcher = {"R": 0, "L": 1}

    X_batter_hand  = np.array([hand_map_batter.get(h, 0) for h in df["stand"]], dtype=np.int32)
    X_pitcher_hand = np.array([hand_map_pitcher.get(h, 0) for h in df["p_throws"]], dtype=np.int32)

    # ---------------------------
    # Context features
    # ---------------------------
    X_ctx = build_context_features(df)
    print(f"\nContext shape: {X_ctx.shape}")

    # Scale context
    ctx_scaler = StandardScaler()
    X_ctx_scaled = ctx_scaler.fit_transform(X_ctx).astype(np.float32)

    # ---------------------------
    # Sequence features
    # ---------------------------
    print("\nBuilding per-pitch feature vectors...")
    pitch_feats = build_sequence_features(df)
    print(f"Per-pitch feature shape: {pitch_feats.shape}")

    # Scale continuous sequence features (columns 5-8: speed, px, pz, spin)
    # Don't scale the one-hot bucket columns (0-4)
    seq_scaler = StandardScaler()
    continuous_cols = pitch_feats[:, len(BUCKET_CLASSES):]
    continuous_scaled = seq_scaler.fit_transform(continuous_cols).astype(np.float32)
    pitch_feats[:, len(BUCKET_CLASSES):] = continuous_scaled

    print("Building sequences (this may take a few minutes)...")
    game_ids = df["game_pk"].values
    ab_ids = df["at_bat_number"].values
    X_seq = build_sequences(pitch_feats, game_ids, ab_ids, SEQ_LEN)
    print(f"Sequence shape: {X_seq.shape}")

    # ---------------------------
    # Build pitcher arsenal priors (for inference-time masking)
    # ---------------------------
    print("\nBuilding pitcher arsenal priors...")
    arsenal = {}
    priors = {}
    for pid, group in df.groupby("pitcher"):
        pid_str = str(int(pid))
        bucket_counts = group["bucket"].value_counts()
        total = bucket_counts.sum()

        # Arsenal: which buckets this pitcher actually throws
        arsenal[pid_str] = bucket_counts.index.tolist()

        # Priors: probability distribution over buckets
        priors[pid_str] = {
            b: float(bucket_counts.get(b, 0) / total)
            for b in BUCKET_CLASSES
        }

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
    np.save(OUT_DIR / "y_labels.npy", y_labels)

    joblib.dump(ctx_scaler, OUT_DIR / "context_scaler_v4.pkl")
    joblib.dump(seq_scaler, OUT_DIR / "seq_scaler_v4.pkl")
    joblib.dump(BUCKET_TO_ID, OUT_DIR / "bucket_map_v4.pkl")
    joblib.dump(pitcher_map, OUT_DIR / "pitcher_id_map_v4.pkl")
    joblib.dump(batter_map, OUT_DIR / "batter_id_map_v4.pkl")

    with open(OUT_DIR / "pitcher_arsenal_v4.json", "w") as f:
        json.dump(arsenal, f)
    with open(OUT_DIR / "pitcher_priors_v4.json", "w") as f:
        json.dump(priors, f)

    # Save metadata for the training script
    meta = {
        "n_samples": len(y_labels),
        "seq_len": SEQ_LEN,
        "seq_feats": X_seq.shape[2],
        "ctx_feats": X_ctx_scaled.shape[1],
        "n_buckets": len(BUCKET_CLASSES),
        "bucket_classes": BUCKET_CLASSES,
        "n_pitcher_ids": len(pitcher_map) + 1,  # +1 for UNK
        "n_batter_ids": len(batter_map) + 1,
        "min_pitcher_appearances": MIN_PITCHER_APPEARANCES,
        "min_batter_appearances": MIN_BATTER_APPEARANCES,
    }
    with open(OUT_DIR / "meta_v4.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved {len(y_labels):,} samples.")
    print(f"Metadata: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
