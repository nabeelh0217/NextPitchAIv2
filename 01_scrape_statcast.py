"""
NextPitchAI v5 — Step 1: Scrape Statcast data
================================================
Pulls pitch-level data from Baseball Savant via pybaseball.
Includes MLBAM IDs for pitcher/batter personalization, plus v5
columns: game date, ballpark (home team), catcher, times-through-order,
and pitcher rest days.

Requirements:
    pip install pybaseball pandas pyarrow

Usage:
    python 01_scrape_statcast.py            # fetch whatever is missing
    python 01_scrape_statcast.py --fresh    # discard the file, re-scrape all

Output:
    statcast_raw_v5.parquet  (~2-4 GB for multiple seasons)

When the parquet already exists the default is INCREMENTAL: only the
weekly windows with no rows in the file are fetched, and they are
appended. Existing rows are never re-downloaded or rewritten. That is
what lets a new season be added without disturbing the data a trained
model was built on, and it makes a re-run after a flaky connection fetch
only the weeks that failed.

v7 adds zone / sz_top / sz_bot for the location head. Everything else
the model uses was already being scraped.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
from pybaseball import statcast

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "statcast_raw_v5.parquet"

# Seasons to pull — more data = better embeddings for pitcher/batter IDs.
# Each full season is ~700k-750k pitches. 4 seasons gives ~2.9M rows.
SEASONS = [
    ("2022-04-07", "2022-10-05"),
    ("2023-03-30", "2023-10-01"),
    ("2024-03-28", "2024-09-29"),
    ("2025-03-18", "2025-09-28"),   # opens with the Tokyo Series
    ("2026-03-25", "2026-09-27"),
]

# From this season on, keep REGULAR-SEASON games only. pybaseball's
# statcast() also returns spring training and the postseason for any
# dates it is given, so a date window alone does not keep exhibition
# games out — and exhibition pitch selection is not what a hitter faces.
# Earlier seasons are deliberately left exactly as first scraped: the
# trained model's priors were built on those rows, and changing them
# would break the frozen-artifact check in 02_preprocess.py. (The 2025
# window starts at the Tokyo Series on 03-18, so it may hold a week of
# spring-training games. Fix that on the next full re-scrape by lowering
# this to 2022 and retraining.)
REGULAR_ONLY_FROM = 2026

# Adding seasons FORWARD is safe; adding them backward is not. The
# sweeper (ST) barely existed as a Statcast label before 2023, so 2021
# and earlier would inject a labelling shift that looks like a real
# change in pitcher behaviour. 2025 is the closest season to deployment
# and the most representative data available.

# Columns we need for v5
# (pybaseball returns ~90 columns; we keep only what matters)
KEEP_COLUMNS = [
    # Identifiers
    "game_pk", "game_date", "at_bat_number", "pitch_number",
    "pitcher", "batter",                        # <-- MLBAM IDs

    # Target
    "pitch_type",                               # e.g. FF, SL, CH, CU, etc.

    # Count & situation
    "balls", "strikes", "outs_when_up",
    "inning", "inning_topbot",

    # Baserunners (NaN = nobody on; a number = runner's MLBAM ID)
    "on_1b", "on_2b", "on_3b",

    # Score
    "bat_score", "fld_score",

    # Handedness
    "stand",                                    # batter hand: R/L
    "p_throws",                                 # pitcher hand: R/L

    # Pitch physics (for sequence features + location prediction)
    "release_speed", "release_spin_rate",
    "plate_x", "plate_z",
    "pfx_x", "pfx_z",                          # horizontal/vertical movement

    # v7: location target. sz_top/sz_bot are the BATTER-SPECIFIC strike
    # zone for that pitch — "up" means something different to a 5'6"
    # hitter than a 6'7" one, so a league-average zone would mislabel a
    # lot of borderline pitches. `zone` is Statcast's own 1-14 label,
    # kept as a cross-check on our derived attack zones.
    "zone", "sz_top", "sz_bot",

    # v5: environment & personnel
    "home_team",                                # ballpark proxy (park factors)
    "fielder_2",                                # catcher MLBAM ID (game-calling)

    # v5: pitcher workload (Statcast provides these precomputed)
    "n_thruorder_pitcher",                      # times through the order
    "pitcher_days_since_prev_game",             # rest days

    # Outcome (used for batter swing/whiff profiles + future extensions)
    "description", "events", "type",

    # R / S / F / D / L / W. Kept so the regular-season filter is
    # auditable in the file; rows scraped before this was added hold NaN.
    "game_type",
]


def season_windows(start_dt: str, end_dt: str, chunk_days: int = 7):
    """The weekly (start, end) windows a season is fetched in."""
    out = []
    current, end = pd.Timestamp(start_dt), pd.Timestamp(end_dt)
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        out.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return out


def fetch_window(start, end):
    """One window, with one retry. Returns a frame, or None if it FAILED.

    An empty frame and a failure are different things and must not be
    confused: the first means Savant has nothing for those dates, the
    second leaves a hole that should be retried.
    """
    for attempt in (1, 2):
        try:
            df = statcast(start_dt=str(start.date()), end_dt=str(end.date()))
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            if attempt == 1:
                print(f"ERROR: {e} — retrying in 10s...", end=" ", flush=True)
                time.sleep(10)
            else:
                print(f"retry FAILED: {e}")
    return None


def clean(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Column subset, usable pitch types, and the regular-season filter."""
    if len(df) == 0:
        return df
    if season >= REGULAR_ONLY_FROM:
        if "game_type" in df.columns:
            df = df[df["game_type"] == "R"]
        else:
            print("\n  WARNING: no game_type column in this pull — cannot drop "
                  "spring-training or postseason games for this window.")
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()
    df = df.dropna(subset=["pitch_type"])
    return df[df["pitch_type"] != ""]


def main():
    ap = argparse.ArgumentParser(description="Scrape Statcast pitches.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the existing parquet and re-scrape every season")
    args = ap.parse_args()

    # What is already on disk, by calendar day. Reading one column keeps
    # this cheap even when the file is several GB.
    have_days = set()
    incremental = OUTPUT_PATH.exists() and not args.fresh
    if incremental:
        days = pd.read_parquet(OUTPUT_PATH, columns=["game_date"])["game_date"]
        # Plain dates, so membership cannot depend on timestamp resolution.
        have_days = {d.date() for d in pd.to_datetime(days).dt.normalize().unique()}
        print(f"Existing file: {len(days):,} rows over {len(have_days):,} game days.")
        print("Fetching only the weekly windows with no rows in it "
              "(--fresh re-scrapes everything).")

    todo = []
    for start_dt, end_dt in SEASONS:
        for w0, w1 in season_windows(start_dt, end_dt):
            if not any(d.date() in have_days for d in pd.date_range(w0, w1)):
                todo.append((w0, w1))
    if incremental and not todo:
        print("Nothing to fetch — every window of every season is present.")
        return

    new_parts, failed = [], []
    season = None
    for w0, w1 in todo:
        if w0.year != season:
            season = w0.year
            print(f"\n{'='*50}\nSeason {season}\n{'='*50}")
        print(f"  Pulling {w0.date()} → {w1.date()} ...", end=" ", flush=True)
        df = fetch_window(w0, w1)
        if df is None:
            failed.append((w0, w1))
        else:
            raw = len(df)
            df = clean(df, w0.year)
            if len(df):
                new_parts.append(df)
            print(f"{len(df):,} pitches" + (f" (of {raw:,} pulled)" if raw != len(df) else ""))
        time.sleep(2)  # be nice to Baseball Savant servers

    if not new_parts and not incremental:
        print("No data collected! Check your internet connection and pybaseball.")
        sys.exit(1)

    if new_parts:
        added = pd.concat(new_parts, ignore_index=True)
        missing = [c for c in KEEP_COLUMNS if c not in added.columns]
        if missing:
            print(f"WARNING: These columns were not in the data: {missing}")
        if incremental:
            # Existing rows go through untouched: no re-clean, no dedupe.
            existing = pd.read_parquet(OUTPUT_PATH)
            combined = pd.concat([existing, added], ignore_index=True)
            print(f"\nAppending {len(added):,} new rows to {len(existing):,} existing.")
        else:
            combined = added

        # Sort chronologically (important for sequence building later)
        combined = combined.sort_values(
            ["game_pk", "at_bat_number", "pitch_number"], kind="stable"
        ).reset_index(drop=True)

        if len(combined) == 0:
            print("ERROR: no rows survived cleanup — refusing to write an empty file.")
            sys.exit(1)

        # Save as parquet (much faster + smaller than CSV). Write to a temp
        # file first and atomically replace the final path only once the
        # write has fully succeeded — so a closed window, sleep, or crash
        # mid-write can never leave a corrupt/empty file sitting at
        # OUTPUT_PATH (which a later run would otherwise mistake for a
        # completed scrape).
        tmp_path = OUTPUT_PATH.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, OUTPUT_PATH)
        print(f"\nSaved to: {OUTPUT_PATH}")
        print(f"Final size: {len(combined):,} rows, {len(combined.columns)} columns")

        per_season = pd.to_datetime(combined["game_date"]).dt.year.value_counts().sort_index()
        print("\nRows per season (a full season is roughly 700-750K):")
        for yr, n in per_season.items():
            print(f"  {yr}: {n:,}")
        print(f"\nUnique pitchers: {combined['pitcher'].nunique():,}")
        print(f"Unique batters:  {combined['batter'].nunique():,}")
    else:
        print("\nNo new rows were fetched; the existing file is unchanged.")

    # A hole in the data must never pass silently: everything downstream
    # would run, and simply be built on less than it claims.
    if failed:
        print(f"\n{'!'*60}")
        print(f"{len(failed)} window(s) FAILED and are missing from the file:")
        for w0, w1 in failed:
            print(f"    {w0.date()} → {w1.date()}")
        print("Re-run this script: it will fetch only those windows.")
        print(f"{'!'*60}")
        sys.exit(3)


if __name__ == "__main__":
    main()
