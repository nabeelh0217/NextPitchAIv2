"""
NextPitchAI v6 — build the serving bundle
=========================================
Derives the small lookup tables the website needs from data_v5/, so the
Flask app never has to load the multi-GB training arrays.

Run once after training:

    python site/build_serving_artifacts.py

Writes site/serving/. Everything in it is derived, so it is gitignored
and rebuilt whenever the model is retrained.

Why the lookups are taken as a pitcher's LAST row: the priors in X_ctx
are *expanding* statistics, so the final row for a pitcher is that
pitcher's full history as of the end of the data — exactly the state a
live prediction should start from. This is the same rule
02_preprocess.py already uses for batter_profiles_v5.parquet.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data_v5"
OUT_DIR = Path(__file__).resolve().parent / "serving"
PARQUET = BASE_DIR / "statcast_raw_v5.parquet"

from evaluate_model import load_split, build_arsenal_mask  # noqa: E402


def _cols(names, prefix, n):
    """Indices of the n context columns starting with `prefix`."""
    idx = [i for i, nm in enumerate(names) if nm.startswith(prefix)]
    if len(idx) != n:
        raise SystemExit(
            f"expected {n} context columns matching {prefix!r}, found "
            f"{len(idx)}. meta_v5.json and the serving build disagree — "
            f"rebuild data_v5/.")
    return idx


def main():
    if not (DATA_DIR / "meta_v5.json").exists():
        raise SystemExit(f"no {DATA_DIR}/meta_v5.json — run the pipeline first")
    OUT_DIR.mkdir(exist_ok=True)
    meta = json.loads((DATA_DIR / "meta_v5.json").read_text())
    names = meta["ctx_feature_names"]
    classes = meta["pitch_classes"]
    zones = meta["zone_classes"]
    n_pitch = len(classes)

    print("Loading arrays...")
    y = np.load(DATA_DIR / "y_labels.npy")
    pid = np.load(DATA_DIR / "X_pitcher_id.npy")
    bid = np.load(DATA_DIR / "X_batter_id.npy")
    p_hand = np.load(DATA_DIR / "X_pitcher_hand.npy")
    b_hand = np.load(DATA_DIR / "X_batter_hand.npy")
    X_ctx = np.load(DATA_DIR / "X_ctx.npy")
    ctx_scaler = joblib.load(DATA_DIR / "context_scaler_v5.pkl")

    idx_tr, idx_val, split_desc = load_split(y)
    print(f"Split: {split_desc}")

    # The mask the model was actually trained with. Serving it a
    # different mask would change its behaviour from what was evaluated.
    mask_path = DATA_DIR / "arsenal_mask_table_v5.npy"
    if mask_path.exists():
        mask_tbl = np.load(mask_path)
        print("Arsenal mask: loaded the table the run saved")
    else:
        mask_tbl = build_arsenal_mask(pid, y, idx_tr, n_pitch)
        print("Arsenal mask: rebuilt from the recorded split")

    # Undo the scaler so the priors are readable probabilities again;
    # the app re-applies the same scaler after assembling a row.
    print("Recovering raw context priors...")
    raw = ctx_scaler.inverse_transform(X_ctx.astype(np.float64))

    ars_c = _cols(names, "arsenal_", n_pitch)
    zp_c = _cols(names, "zoneprior_", len(zones))
    seen_c = _cols(names, "seen_", n_pitch)
    whiff_c = _cols(names, "whiff_", n_pitch)
    zs_c = _cols(names, "zoneseen_", len(zones))

    # Tables are keyed by RAW MLB player id, because that is what the
    # matchup table and the physics medians use, and what a caller can
    # reasonably supply. The ENCODED id (the embedding row) rides along
    # in a column. Mixing the two silently turns every lookup into
    # "unknown player" and the model quietly predicts from a league
    # average, which looks like a working site.
    pid_map = joblib.load(DATA_DIR / "pitcher_id_map_v5.pkl")
    bid_map = joblib.load(DATA_DIR / "batter_id_map_v5.pkl")
    enc2raw_p = {int(v): int(k) for k, v in pid_map.items()}
    enc2raw_b = {int(v): int(k) for k, v in bid_map.items()}

    def last_rows(enc_ids):
        """Final row index per encoded id — the full expanding history."""
        order = np.argsort(enc_ids, kind="stable")
        srt = enc_ids[order]
        last = np.flatnonzero(np.r_[srt[1:] != srt[:-1], True])
        return srt[last], order[last]

    print("Building pitcher table...")
    enc, rows = last_rows(pid)
    keep = [i for i, e in enumerate(enc) if int(e) in enc2raw_p]
    enc, rows = enc[keep], rows[keep]
    pit = pd.DataFrame({"pitcher_id": [enc2raw_p[int(e)] for e in enc],
                        "enc": enc.astype(int)})
    for i, c in enumerate(classes):
        pit[f"arsenal_{c}"] = raw[rows, ars_c[i]]
    for i, z in enumerate(zones):
        pit[f"zoneprior_{z}"] = raw[rows, zp_c[i]]
    pit["hand"] = p_hand[rows]
    for i, c in enumerate(classes):
        pit[f"mask_{c}"] = mask_tbl[enc, i]
    pit.to_parquet(OUT_DIR / "pitchers.parquet", index=False)

    print("Building batter table...")
    enc, rows = last_rows(bid)
    keep = [i for i, e in enumerate(enc) if int(e) in enc2raw_b]
    enc, rows = enc[keep], rows[keep]
    bat = pd.DataFrame({"batter_id": [enc2raw_b[int(e)] for e in enc],
                        "enc": enc.astype(int)})
    for i, c in enumerate(classes):
        bat[f"seen_{c}"] = raw[rows, seen_c[i]]
    for i, c in enumerate(classes):
        bat[f"whiff_{c}"] = raw[rows, whiff_c[i]]
    for i, z in enumerate(zones):
        bat[f"zoneseen_{z}"] = raw[rows, zs_c[i]]
    bat["hand"] = b_hand[rows]
    bat.to_parquet(OUT_DIR / "batters.parquet", index=False)

    # Per-(pitcher, type) physics, for filling the sequence when the user
    # gives us pitch types but obviously not spin rates. Median, not
    # mean: velocity and movement both have long tails.
    print("Building physics medians from the parquet...")
    phys_cols = ["release_speed", "plate_x", "plate_z",
                 "release_spin_rate", "pfx_x", "pfx_z"]
    if PARQUET.exists():
        df = pd.read_parquet(PARQUET, columns=["pitcher", "pitch_type"] + phys_cols)
        from importlib.machinery import SourceFileLoader
        pre = SourceFileLoader("pre", str(BASE_DIR / "02_preprocess.py")).load_module()
        df["canon"] = df["pitch_type"].map(pre.PITCH_TYPE_CANON)
        df = df.dropna(subset=["canon"])
        phys = df.groupby(["pitcher", "canon"])[phys_cols].median().reset_index()
        league = df[phys_cols].median().to_dict()
    else:
        print(f"  WARNING: {PARQUET.name} not found — the app will fall back to")
        print("  league-median physics for every sequence timestep.")
        phys = pd.DataFrame(columns=["pitcher", "canon"] + phys_cols)
        league = {c: 0.0 for c in phys_cols}
    phys.to_parquet(OUT_DIR / "physics.parquet", index=False)

    print("Copying matchups and maps...")
    for f in ["matchup_table_v5.parquet", "pitcher_id_map_v5.pkl",
              "batter_id_map_v5.pkl", "context_scaler_v5.pkl",
              "seq_scaler_v5.pkl", "meta_v5.json"]:
        src = DATA_DIR / f
        if src.exists():
            (OUT_DIR / f).write_bytes(src.read_bytes())

    (OUT_DIR / "serving_meta.json").write_text(json.dumps({
        "split": split_desc,
        "league_physics": league,
        "n_pitchers": int(len(pit)),
        "n_batters": int(len(bat)),
    }, indent=2))
    print(f"\nServing bundle written to {OUT_DIR}")
    print(f"  {len(pit):,} pitchers, {len(bat):,} batters")


if __name__ == "__main__":
    main()
