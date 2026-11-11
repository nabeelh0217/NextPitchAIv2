"""
NextPitchAI v6 — build the serving bundle
=========================================
Derives the small lookup tables the website needs from a dataset
directory, so the Flask app never has to load the multi-GB training
arrays.

    .venv/bin/python site/build_serving_artifacts.py
    .venv/bin/python site/build_serving_artifacts.py --data-dir data_ext

Writes site/serving/, which is committed: Render builds from the repo and
can see nothing else. Rebuild it AND commit it whenever the model, the
data or the gate's claim changes.

Why the lookups are taken as a player's LAST row: the priors in X_ctx
are *expanding* statistics, so the final row for a pitcher is that
pitcher's full history as of the end of the data — exactly the state a
live prediction should start from. This is the same rule
02_preprocess.py already uses for batter_profiles_v5.parquet.

The bundle must serve what the gate measured. The gate scores the model
with each pitcher's arsenal AS OF THE PITCH — every type he had thrown
so far — so the arsenal served here is every type he has thrown through
the newest data, keyed by his real MLB id. Serving the mask frozen at
the end of training instead would be a different product from the one
the claim describes: it can never call a pitch added since.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA_DIR = BASE_DIR / "data_v5"
OUT_DIR = Path(__file__).resolve().parent / "serving"

# Below this many meetings a matchup prior carries no information the
# pitcher's own mix does not already have.
MIN_MATCHUP_MEETINGS = 5

# The bundle is committed so Render can read it, so it has to stay small.
# GitHub warns at 50MB per file and refuses at 100MB; well before that a
# bloated bundle makes every clone and deploy slow.
BUNDLE_MAX_MB = 40
PARQUET = BASE_DIR / "statcast_raw_v5.parquet"

from evaluate_model import (  # noqa: E402
    build_arsenal_mask, known_ids, load_row_meta, load_split,
)
from pitch_features import MASK_MIN_HISTORY  # noqa: E402


def _cols(names, prefix, n):
    """Indices of the n context columns starting with `prefix`."""
    idx = [i for i, nm in enumerate(names) if nm.startswith(prefix)]
    if len(idx) != n:
        raise SystemExit(
            f"expected {n} context columns matching {prefix!r}, found "
            f"{len(idx)}. meta_v5.json and the serving build disagree — "
            f"rebuild data_v5/.")
    return idx


def roster(raw_ids, min_rows):
    """
    One entry per player with at least `min_rows` rows:
    (raw id, index of the player's last row, row count, code per row).

    Keyed by the RAW MLB id. The encoded id pools every player without an
    embedding into index 0, so "the last row of id 0" is one arbitrary
    stranger's history served under everyone's name. Negative ids mark
    rows with no known player and are never listed.
    """
    uniq, inv = np.unique(raw_ids, return_inverse=True)
    n = np.bincount(inv, minlength=len(uniq))
    last = np.zeros(len(uniq), dtype=np.int64)
    last[inv] = np.arange(len(raw_ids))        # repeated index: the last write wins
    keep = (uniq >= 0) & (n >= min_rows)
    return uniq[keep], last[keep], n[keep], keep, inv


def main():
    ap = argparse.ArgumentParser(
        description="Build site/serving/ from a dataset directory.")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help="dataset to build from (default: data_v5)")
    ap.add_argument("--raw", type=Path, default=PARQUET,
                    help="raw Statcast parquet, for the physics medians")
    ap.add_argument("--out", type=Path, default=OUT_DIR,
                    help="where to write the bundle (default: site/serving)")
    ap.add_argument("--no-names", action="store_true",
                    help="skip the MLB name lookup (no network)")
    ap.add_argument("--no-model", action="store_true",
                    help="skip the weight export")
    ap.add_argument("--allow-no-claim", action="store_true",
                    help="build even though the gate has not been run on this "
                         "dataset (the site then has no claim panel)")
    a = ap.parse_args()
    data_dir = a.data_dir.resolve()
    out_dir = a.out.resolve()

    if not (data_dir / "meta_v5.json").exists():
        raise SystemExit(f"no {data_dir}/meta_v5.json — run the pipeline first")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta_v5.json").read_text())
    names = meta["ctx_feature_names"]
    classes = meta["pitch_classes"]
    zones = meta["zone_classes"]
    n_pitch = len(classes)

    # The claim and the weights must describe the same model. The gate
    # stamps the claim with the model file's hash; a claim measured on an
    # earlier model, bundled with newer weights, is a figure nobody ever
    # measured, and it would look exactly like a real one.
    model_path = data_dir / "best_model_v5.keras"
    model_sha = (hashlib.sha256(model_path.read_bytes()).hexdigest()
                 if model_path.exists() else None)
    claim_path = data_dir / "product_claim_v5.json"
    claim = json.loads(claim_path.read_text()) if claim_path.exists() else None
    if claim is None and not a.allow_no_claim:
        # The claim panel is the one thing the site is required to show.
        # A bundle built before the gate has run would deploy without it,
        # or beside a number measured on something else.
        raise SystemExit(
            f"no {claim_path.name} in {data_dir.name}/ — measure first:\n\n"
            f"    .venv/bin/python analyze_actionability.py --data-dir {data_dir.name}\n")
    if claim and claim.get("model_sha256") and model_sha \
            and claim["model_sha256"] != model_sha:
        raise SystemExit(
            f"{claim_path.name} was measured on a different model than "
            f"{model_path.name} (sha256 differs). Re-run "
            f"analyze_actionability.py --data-dir {data_dir.name} first.")

    print(f"Loading arrays from {data_dir.name}/ ...")
    y = np.load(data_dir / "y_labels.npy")
    pid = np.load(data_dir / "X_pitcher_id.npy")
    bid = np.load(data_dir / "X_batter_id.npy")
    p_hand = np.load(data_dir / "X_pitcher_hand.npy")
    b_hand = np.load(data_dir / "X_batter_hand.npy")
    # Memory-mapped: X_ctx is ~0.7 GB on four seasons and we need about
    # 4,500 rows of it. Reading it whole, widening to float64 and
    # inverse-transforming the lot peaked near 3.6 GB and the OS killed
    # the process on a laptop.
    X_ctx = np.load(data_dir / "X_ctx.npy", mmap_mode="r")
    ctx_scaler = joblib.load(data_dir / "context_scaler_v5.pkl")

    idx_tr, idx_val, split_desc = load_split(y, data_dir)
    print(f"Split: {split_desc}")
    # An embedding row with no training rows behind it still holds its
    # random initial values. Those players are served from the unknown
    # bucket, which is how the gate scores them.
    known = known_ids({"pitcher_id": pid, "batter_id": bid}, idx_tr,
                      keys=["pitcher_id", "batter_id"])

    # Tables are keyed by RAW MLB player id, because that is what the
    # matchup table and the physics medians use, and what a caller can
    # reasonably supply. The ENCODED id (the embedding row) rides along
    # in a column. Mixing the two silently turns every lookup into
    # "unknown player" and the model quietly predicts from a league
    # average, which looks like a working site.
    rm = load_row_meta(data_dir)
    if rm is not None:
        p_raw, b_raw = rm["pitcher"].astype(np.int64), rm["batter"].astype(np.int64)
        data_through = str(int(rm["game_date"].max()))
        data_through = f"{data_through[:4]}-{data_through[4:6]}-{data_through[6:]}"
        min_p = MASK_MIN_HISTORY
        min_b = int(meta.get("min_batter_appearances", 50))
        mask_desc = (f"every pitch type thrown through {data_through} "
                     f"(pitchers with {MASK_MIN_HISTORY}+ pitches)")
    else:
        # A dataset from before row_meta_v5.npz: the raw id is only
        # recoverable for players who have an embedding.
        def raw_of(enc_ids, id_map):
            tbl = np.full(int(enc_ids.max()) + 1, -1, dtype=np.int64)
            for raw, enc in id_map.items():
                if 0 < int(enc) < len(tbl):
                    tbl[int(enc)] = int(raw)
            return tbl[enc_ids]
        p_raw = raw_of(pid, joblib.load(data_dir / "pitcher_id_map_v5.pkl"))
        b_raw = raw_of(bid, joblib.load(data_dir / "batter_id_map_v5.pkl"))
        data_through, min_p, min_b = None, 1, 1
        mask_desc = "training rows only (no row metadata in this dataset)"
    print(f"Arsenal mask: {mask_desc}")

    # Undo the scaler so the priors are readable probabilities again;
    # the app re-applies the same scaler after assembling a row.
    ars_c = _cols(names, "arsenal_", n_pitch)
    zp_c = _cols(names, "zoneprior_", len(zones))
    seen_c = _cols(names, "seen_", n_pitch)
    whiff_c = _cols(names, "whiff_", n_pitch)
    zs_c = _cols(names, "zoneseen_", len(zones))

    def unscale(rows):
        """Undo the scaler for just these rows, straight off the mmap."""
        rows = np.asarray(rows)
        order = np.argsort(rows)                 # a sorted read off the mmap
        out = np.empty((len(rows), X_ctx.shape[1]), dtype=np.float64)
        out[order] = np.asarray(X_ctx[rows[order]], dtype=np.float64)
        return ctx_scaler.inverse_transform(out)

    print("Building pitcher table...")
    ids, rows, n_rows, kept, inv = roster(p_raw, min_p)
    pit = pd.DataFrame({"pitcher_id": ids.astype(np.int64),
                        # 0 = no embedding. The model was trained with that
                        # bucket, and measured with it for exactly these
                        # pitchers: his priors and arsenal are still his own.
                        "enc": known["pitcher_id"][pid[rows]].astype(int),
                        # Pitch counts rank the pickers so an empty search
                        # box offers the players someone is likely to want.
                        "n_pitches": n_rows.astype(int)})
    raw = unscale(rows)
    for i, c in enumerate(classes):
        pit[f"arsenal_{c}"] = raw[:, ars_c[i]]
    for i, z in enumerate(zones):
        pit[f"zoneprior_{z}"] = raw[:, zp_c[i]]
    pit["hand"] = p_hand[rows]
    if rm is not None:
        thrown = np.zeros((len(kept), n_pitch), dtype=np.float32)
        thrown[inv, y] = 1.0
        mask_rows = thrown[kept]
    else:
        # The mask the model was trained with, as the run saved it.
        mask_path = data_dir / "arsenal_mask_table_v5.npy"
        mask_tbl = (np.load(mask_path) if mask_path.exists()
                    else build_arsenal_mask(pid, y, idx_tr, n_pitch))
        mask_rows = mask_tbl[pid[rows]]
    if (mask_rows.sum(1) == 0).any():
        raise SystemExit("a pitcher has an empty arsenal mask — every logit "
                         "would hit the floor. The dataset is inconsistent.")
    for i, c in enumerate(classes):
        pit[f"mask_{c}"] = mask_rows[:, i]
    pit.to_parquet(out_dir / "pitchers.parquet", index=False)
    print(f"  {len(pit):,} pitchers, {int((pit['enc'] == 0).sum()):,} of them "
          f"without an embedding (served from their own priors and arsenal)")

    print("Building batter table...")
    ids, rows, n_rows, _, _ = roster(b_raw, min_b)
    bat = pd.DataFrame({"batter_id": ids.astype(np.int64),
                        "enc": known["batter_id"][bid[rows]].astype(int),
                        "n_pitches": n_rows.astype(int)})
    raw = unscale(rows)
    for i, c in enumerate(classes):
        bat[f"seen_{c}"] = raw[:, seen_c[i]]
    for i, c in enumerate(classes):
        bat[f"whiff_{c}"] = raw[:, whiff_c[i]]
    for i, z in enumerate(zones):
        bat[f"zoneseen_{z}"] = raw[:, zs_c[i]]
    bat["hand"] = b_hand[rows]
    bat.to_parquet(out_dir / "batters.parquet", index=False)

    # Per-(pitcher, type) physics, for filling the sequence when the user
    # gives us pitch types but obviously not spin rates. Median, not
    # mean: velocity and movement both have long tails.
    print("Building physics medians from the parquet...")
    phys_cols = ["release_speed", "plate_x", "plate_z",
                 "release_spin_rate", "pfx_x", "pfx_z"]
    if a.raw.exists():
        df = pd.read_parquet(a.raw, columns=["pitcher", "pitch_type"] + phys_cols)
        from pitch_features import PITCH_TYPE_CANON
        # Categorical, and float32: 2.8M raw strings plus float64 physics
        # is most of a gigabyte for what ends up a few thousand medians.
        df["canon"] = df["pitch_type"].map(PITCH_TYPE_CANON).astype("category")
        df = df.drop(columns=["pitch_type"]).dropna(subset=["canon"])
        for c in phys_cols:
            df[c] = df[c].astype(np.float32)
        phys = (df.groupby(["pitcher", "canon"], observed=True)[phys_cols]
                  .median().reset_index())
        league = df[phys_cols].median().to_dict()
    else:
        print(f"  WARNING: {a.raw.name} not found — the app will fall back to")
        print("  league-median physics for every sequence timestep.")
        phys = pd.DataFrame(columns=["pitcher", "canon"] + phys_cols)
        league = {c: 0.0 for c in phys_cols}
    phys.to_parquet(out_dir / "physics.parquet", index=False)

    # Matchups, pruned. Every pitcher x batter pair that ever met is tens
    # of millions of rows and far too big to commit; below a handful of
    # meetings the matchup prior is indistinguishable from the pitcher's
    # own mix, which is exactly what the predictor falls back to.
    src = data_dir / "matchup_table_v5.parquet"
    if src.exists():
        mt = pd.read_parquet(src)
        cnt = mt[[c for c in mt.columns if c in classes]].sum(axis=1)
        kept_m = mt[cnt >= MIN_MATCHUP_MEETINGS]
        kept_m.to_parquet(out_dir / "matchup_table_v5.parquet", index=False)
        print(f"Matchups: {len(kept_m):,} of {len(mt):,} pairs kept "
              f"(>={MIN_MATCHUP_MEETINGS} meetings)")

    # Scalers as plain arrays. Unpickling a StandardScaler drags
    # scikit-learn (~100MB) into the deploy image for two vectors of
    # floats; the transform is (x - mean) / scale either way.
    seq_s = joblib.load(data_dir / "seq_scaler_v5.pkl")
    np.savez_compressed(
        out_dir / "scalers.npz",
        ctx_mean=ctx_scaler.mean_.astype(np.float32),
        ctx_scale=ctx_scaler.scale_.astype(np.float32),
        seq_mean=seq_s.mean_.astype(np.float32),
        seq_scale=seq_s.scale_.astype(np.float32))

    (out_dir / "meta_v5.json").write_bytes((data_dir / "meta_v5.json").read_bytes())
    if claim is not None:
        # Copied in, so the deployed site carries its own claim instead of
        # depending on the dataset directory being present on the server.
        (out_dir / "product_claim_v5.json").write_bytes(claim_path.read_bytes())
        if rm is not None and int(claim.get("schema", 1)) < 2:
            print("  WARNING: this claim was written by the old gate (frozen table,")
            print("  full-data mask). Re-run analyze_actionability.py and rebuild.")
    else:
        print(f"  WARNING: no {claim_path.name} — the site will render without")
        print("  its claim panel. Do not deploy this bundle.")

    # Player names. Needs the internet, so it is best-effort and the
    # cache is never clobbered on failure — run it on a machine with
    # network access before deploying.
    if not a.no_names:
        try:
            from player_names import fetch_names
            ids = ([int(i) for i in pit["pitcher_id"]]
                   + [int(i) for i in bat["batter_id"]])
            got = fetch_names(sorted(set(ids)), out_dir / "player_names.json")
            print(f"Player names: {len(got):,} cached")
        except Exception as e:
            print(f"  WARNING: name lookup unavailable ({e}).")
            print("  The site falls back to numeric ids. Re-run with network.")

    # TensorFlow-free weights for serving.
    if not a.no_model:
        # A SUBPROCESS, deliberately. The optional TensorFlow cross-check
        # inside export_model.py can abort() the interpreter on a broken
        # install, and importing TF after pandas and numpy have started
        # their thread pools deadlocks on macOS. Keeping the export in its
        # own process means neither can take the rest of the bundle down.
        script = Path(__file__).resolve().parent / "export_model.py"
        print("\nExporting model weights (separate process)...")
        r = subprocess.run(
            [sys.executable, str(script),
             "--model", str(model_path),
             "--out", str(out_dir)],
            env={**os.environ,
                 # TF's own thread pools on top of Accelerate's are what
                 # deadlock; one interop thread avoids the pile-up and
                 # costs nothing for a single forward pass.
                 "TF_NUM_INTEROP_THREADS": "1",
                 "TF_CPP_MIN_LOG_LEVEL": "2",
                 "OMP_NUM_THREADS": "1"})
        if r.returncode != 0:
            print(f"  WARNING: model export failed (exit {r.returncode}).")
            print(f"  Run `{sys.executable} site/export_model.py` before deploying.")

    # League averages, for a batter or pitcher we know nothing about.
    # An expanding prior falls back to exactly these on a player's first
    # pitch, so they are what the model was trained to see for an unknown
    # one — not a uniform distribution, which it has never seen. They are
    # the values 02_preprocess.py itself used: on a dataset extended
    # forward they come from the seasons the model was fitted on, not from
    # every row here.
    fit_seasons = meta.get("fit_seasons")
    season_path = data_dir / "X_season.npy"
    fit_rows = (np.isin(np.load(season_path), fit_seasons)
                if fit_seasons and season_path.exists()
                else np.ones(len(y), dtype=bool))
    lpath = data_dir / "league_stats_v5.json"
    lstats = json.loads(lpath.read_text()) if lpath.exists() else {}
    lp = lstats.get("league_pitch_dist")
    league_pitch = ([float(lp.get(c, 0.0)) for c in classes] if lp else
                    (np.bincount(y[fit_rows], minlength=n_pitch)
                     / max(int(fit_rows.sum()), 1)).tolist())
    zpath = data_dir / "y_zone.npy"
    if zpath.exists():
        yz = np.load(zpath)[fit_rows]
        yz = yz[yz >= 0]
        league_zone = (np.bincount(yz, minlength=len(zones)) / max(len(yz), 1)).tolist()
    else:
        league_zone = [1.0 / len(zones)] * len(zones)
    lw = lstats.get("league_whiff_rate", {})
    league_whiff = [float(lw.get(c, 0.25)) for c in classes]

    (out_dir / "serving_meta.json").write_text(json.dumps({
        "split": split_desc,
        "built_from": data_dir.name,
        "data_through": data_through,
        "arsenal_mask": mask_desc,
        "model_sha256": model_sha,
        "league_pitch_dist": league_pitch,
        "league_zone_dist": league_zone,
        "league_whiff_rate": league_whiff,
        "league_physics": league,
        "n_pitchers": int(len(pit)),
        "n_batters": int(len(bat)),
    }, indent=2))
    # Sweep anything this build did not write. A file left behind by an
    # earlier build is served as though it were current, which is how a
    # stale artifact quietly becomes a wrong answer.
    expected = {
        "pitchers.parquet", "batters.parquet", "physics.parquet",
        "matchup_table_v5.parquet", "scalers.npz", "meta_v5.json",
        "serving_meta.json", "product_claim_v5.json", "player_names.json",
        "model_weights.npz", "model_arch.json",
    }
    if claim is None:
        # No claim in this dataset: one left over from an earlier build
        # would be a number for a different model or a different gate.
        expected.discard("product_claim_v5.json")
    for f in out_dir.iterdir():
        if f.is_file() and f.name not in expected:
            print(f"  removing stale {f.name}")
            f.unlink()

    total_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"\nServing bundle written to {out_dir}")
    print(f"  {len(pit):,} pitchers, {len(bat):,} batters, {total_mb:.1f} MB"
          + (f", data through {data_through}" if data_through else ""))
    for f in sorted(out_dir.rglob("*"), key=lambda f: -f.stat().st_size)[:5]:
        if f.is_file():
            print(f"    {f.stat().st_size / 1e6:>7.2f} MB  {f.name}")
    if total_mb > BUNDLE_MAX_MB:
        raise SystemExit(
            f"\nBundle is {total_mb:.1f} MB, over the {BUNDLE_MAX_MB} MB limit. "
            f"It is committed to git, so this would bloat every clone and "
            f"deploy. Raise MIN_MATCHUP_MEETINGS (now {MIN_MATCHUP_MEETINGS}) "
            f"or drop the largest file above.")
    if out_dir == OUT_DIR.resolve():
        print("\nNow commit it, or the site keeps serving the previous bundle:")
        print("    git add site/serving")
        print('    git commit -m "Rebuild serving bundle"')
        print("    git push")


if __name__ == "__main__":
    main()
