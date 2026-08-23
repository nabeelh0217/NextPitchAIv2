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
    python 01_scrape_statcast.py

Output:
    statcast_raw_v5.parquet  (~2-4 GB for multiple seasons)
"""

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
# Each full season is ~700k-750k pitches. 3 seasons gives ~2.2M rows.
# Start with 2022-2024; add 2021 later if you want more.
SEASONS = [
    ("2022-04-07", "2022-10-05"),
    ("2023-03-30", "2023-10-01"),
    ("2024-03-28", "2024-09-29"),
]

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

    # Pitch physics (for sequence features + future location prediction)
    "release_speed", "release_spin_rate",
    "plate_x", "plate_z",
    "pfx_x", "pfx_z",                          # horizontal/vertical movement

    # v5: environment & personnel
    "home_team",                                # ballpark proxy (park factors)
    "fielder_2",                                # catcher MLBAM ID (game-calling)

    # v5: pitcher workload (Statcast provides these precomputed)
    "n_thruorder_pitcher",                      # times through the order
    "pitcher_days_since_prev_game",             # rest days

    # Outcome (used for batter swing/whiff profiles + future extensions)
    "description", "events", "type",
]


def scrape_season(start_dt: str, end_dt: str, chunk_days: int = 7) -> pd.DataFrame:
    """
    Pull one season in weekly chunks to avoid timeouts.
    Baseball Savant can be flaky with large date ranges.
    """
    chunks = []
    current = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)

    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        print(f"  Pulling {current.date()} → {chunk_end.date()} ...", end=" ", flush=True)

        try:
            df = statcast(
                start_dt=str(current.date()),
                end_dt=str(chunk_end.date())
            )
            if df is not None and len(df) > 0:
                chunks.append(df)
                print(f"{len(df):,} pitches")
            else:
                print("0 pitches (off-day or no data)")
        except Exception as e:
            print(f"ERROR: {e} — retrying in 10s...")
            time.sleep(10)
            try:
                df = statcast(
                    start_dt=str(current.date()),
                    end_dt=str(chunk_end.date())
                )
                if df is not None and len(df) > 0:
                    chunks.append(df)
                    print(f"  Retry OK: {len(df):,} pitches")
            except Exception as e2:
                print(f"  Retry FAILED: {e2} — skipping this chunk")

        current = chunk_end + timedelta(days=1)
        time.sleep(2)  # be nice to Baseball Savant servers

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def main():
    all_data = []

    for start_dt, end_dt in SEASONS:
        print(f"\n{'='*50}")
        print(f"Season: {start_dt} to {end_dt}")
        print(f"{'='*50}")
        df = scrape_season(start_dt, end_dt)
        if len(df) > 0:
            all_data.append(df)
            print(f"  Season total: {len(df):,} pitches")

    if not all_data:
        print("No data collected! Check your internet connection and pybaseball.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*50}")
    print(f"TOTAL RAW: {len(combined):,} pitches across {len(SEASONS)} seasons")

    # Keep only the columns we need
    available = [c for c in KEEP_COLUMNS if c in combined.columns]
    missing = [c for c in KEEP_COLUMNS if c not in combined.columns]
    if missing:
        print(f"WARNING: These columns were not in the data: {missing}")

    combined = combined[available].copy()

    # Basic cleanup before saving
    # Drop rows with no pitch_type (intentional balls, etc.)
    before = len(combined)
    combined = combined.dropna(subset=["pitch_type"])
    combined = combined[combined["pitch_type"] != ""]
    print(f"Dropped {before - len(combined):,} rows with missing pitch_type")

    # Sort chronologically (important for sequence building later)
    combined = combined.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    # Save as parquet (much faster + smaller than CSV)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Final size: {len(combined):,} rows, {len(combined.columns)} columns")

    # Quick stats
    print(f"\nPitch type distribution:")
    print(combined["pitch_type"].value_counts().head(15).to_string())
    print(f"\nUnique pitchers: {combined['pitcher'].nunique():,}")
    print(f"Unique batters:  {combined['batter'].nunique():,}")


if __name__ == "__main__":
    main()
