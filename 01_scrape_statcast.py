"""
NextPitchAI v5 — Step 1: Scrape Statcast data
================================================
Pulls pitch-level data from Baseball Savant via pybaseball.
Includes MLBAM IDs for pitcher/batter personalization, plus v5
columns: game date, ballpark (home team), catcher, times-through-order,
and pitcher rest days.

Usage:
    .venv/bin/python 01_scrape_statcast.py              # fetch whatever is missing
    .venv/bin/python 01_scrape_statcast.py --fresh      # discard the file, re-scrape all
    .venv/bin/python 01_scrape_statcast.py --fill-gaps  # also fetch empty weeks inside
                                                        # seasons a model was built on

Output:
    statcast_raw_v5.parquet  (~2-4 GB for multiple seasons)

When the parquet already exists the default is INCREMENTAL: only the
weekly windows with no rows in the file are fetched, and they are
appended. Existing rows are never re-downloaded or rewritten — they are
copied through as stored and then compared, column by column, with the
file they came from before it is replaced. That is what lets a new season
be added without disturbing the data a trained model was built on.

Three things here exist because of a run that downloaded a whole season
and then lost it:

  * The new rows are made to MATCH THE EXISTING FILE, column by column.
    The same pybaseball returns game_date as text under pandas 3 and as
    timestamps under pandas 2 (it only parses columns whose dtype prints
    as "object" or "string"; pandas 3 calls it "str"). A file written
    under one and extended under the other held both, and could not be
    saved. The existing file's stored type and format always win.
  * Every window is saved to statcast_raw_v5.parts/ as it arrives, and a
    re-run picks those up instead of downloading again. A failure while
    merging costs seconds, not the download.
  * Gaps inside a season that data_v5/ was built on are left alone unless
    --fill-gaps is given: filling one changes the rows a trained model's
    features came from.

v7 adds zone / sz_top / sz_bot for the location head. Everything else
the model uses was already being scraped.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from datetime import timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

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


# Windows already downloaded but not yet merged into the file.
PARTS_DIR = OUTPUT_PATH.with_suffix(".parts")
# The dataset a trained model was built from, if there is one.
DATASET_META = BASE_DIR / "data_v5" / "meta_v5.json"
SORT_KEYS = ["game_pk", "at_bat_number", "pitch_number"]

# How a date can be written as text. Savant's CSV uses the first.
TEXT_DATE_FORMATS = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"), "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"), "%Y-%m-%dT%H:%M:%S"),
]


class SchemaMismatch(Exception):
    """New rows cannot be stored the way the existing file stores a column."""


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
    # Imported here so the merge logic can be imported, and tested, on a
    # machine with neither pybaseball nor a network.
    from pybaseball import statcast
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


def as_dates(col: pd.Series) -> pd.Series:
    """game_date as datetime64, whatever this pandas/pybaseball pair returned.

    Text under pandas 3, timestamps under pandas 2. Normalising here means
    every downloaded window has one representation; how the finished file
    stores it is decided later, by the file.
    """
    if not pd.api.types.is_datetime64_any_dtype(col):
        try:
            col = pd.to_datetime(col, format="%Y-%m-%d")
        except (ValueError, TypeError):
            col = pd.to_datetime(col)       # not Savant's usual format: parse or raise
    if getattr(col.dt, "tz", None) is not None:
        col = col.dt.tz_localize(None)
    return col


def clean(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Column subset, usable pitch types, the regular-season filter, typed
    dates, and a stable order within the window."""
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
    df = df[df["pitch_type"] != ""]
    if len(df) == 0:
        return df
    df["game_date"] = as_dates(df["game_date"])
    df = df.dropna(subset=["game_date"])
    # Stable, so two rows with the same key keep the order Savant gave them.
    return df.sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------
# Downloaded windows, kept on disk until they are safely in the file
# ---------------------------------------------------------------------
def part_path(w0, w1) -> Path:
    return PARTS_DIR / f"{w0.date()}_{w1.date()}.parquet"


def save_part(df: pd.DataFrame, path: Path) -> None:
    PARTS_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_part(path: Path, season: int):
    """A window an earlier run downloaded, or None if it cannot be trusted."""
    if not path.exists():
        return None
    try:
        tbl = pq.read_table(path)
    except Exception:
        return None
    if tbl.num_rows == 0 or not {"game_pk", "game_date", "pitch_type"} <= set(tbl.column_names):
        return None
    if not pa.types.is_timestamp(tbl.schema.field("game_date").type):
        return None                      # written by a version that did not type the dates
    if season >= REGULAR_ONLY_FROM:
        if "game_type" not in tbl.column_names:
            return None
        if set(pc.unique(tbl["game_type"]).to_pylist()) - {"R"}:
            return None
    return tbl


# ---------------------------------------------------------------------
# Making new rows fit the file
# ---------------------------------------------------------------------
def _is_text(t) -> bool:
    return (pa.types.is_string(t) or pa.types.is_large_string(t)
            or getattr(pa.types, "is_string_view", lambda _t: False)(t))


def text_date_format(values, what: str) -> str:
    """The strftime format a text date column is written in, read off its
    own values. Never guessed: an unrecognised format is an error."""
    values = [v for v in values if v is not None]
    for regex, fmt in TEXT_DATE_FORMATS:
        if values and all(regex.match(v) for v in values):
            return fmt
    raise SchemaMismatch(
        f"{what} holds dates as text in a format this script does not know, "
        f"for example {values[:3]!r}.")


def conform(col, target, name: str, text_format: str = None):
    """`col` (an Arrow column of new rows) as type `target`, the type the
    existing file stores `name` in. Values are converted, never reinterpreted,
    and a conversion that would lose anything raises."""
    src = col.type
    if src.equals(target):
        return col
    if pa.types.is_null(src):
        return pa.chunked_array([pa.nulls(len(col), target)])
    if pa.types.is_dictionary(src):
        col, src = col.cast(src.value_type), src.value_type
    try:
        if _is_text(target) and (pa.types.is_timestamp(src) or pa.types.is_date(src)):
            if text_format is None:
                raise SchemaMismatch(f"no text format known for column {name!r}")
            stamps = col.cast(pa.timestamp("s"))            # raises if finer than a second
            out = pc.strftime(stamps, format=text_format).cast(target)
            # Read it back: the text must mean exactly the timestamps it came from.
            back = pc.strptime(out.cast(pa.string()), format=text_format, unit="s")
            same = pc.equal(back, stamps)
            if pc.sum(pc.fill_null(pc.invert(same), False)).as_py():
                raise SchemaMismatch(
                    f"column {name!r}: dates such as {stamps.slice(0, 1).to_pylist()} do "
                    f"not survive being written as {text_format!r}")
            return out
        if pa.types.is_timestamp(target) and _is_text(src):
            if target.tz is not None:
                raise SchemaMismatch(f"column {name!r} is stored with a time zone")
            fmt = text_date_format(pc.unique(col.drop_null()).to_pylist(), f"column {name!r}")
            return pc.strptime(col.cast(pa.string()), format=fmt, unit="s").cast(target)
        if pa.types.is_date(target) and _is_text(src):
            fmt = text_date_format(pc.unique(col.drop_null()).to_pylist(), f"column {name!r}")
            return pc.strptime(col.cast(pa.string()), format=fmt, unit="s").cast(target)
        return col.cast(target)            # safe cast: numbers, text widths, time units
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
        raise SchemaMismatch(
            f"column {name!r}: the file stores {target}, the new rows are {src}, and "
            f"they cannot be converted without loss ({str(e).splitlines()[0]})") from e


def merged_schema(base, parts):
    """The schema of the finished file.

    With an existing file (`base`), each of its columns keeps the type it
    is stored in and columns it lacks are appended. Without one, the first
    downloaded window plays that role. A column that is entirely empty has
    no type of its own and takes the first real one found.
    """
    schemas = ([base] if base is not None else []) + list(parts)
    names = []
    for s in schemas:
        names += [n for n in s.names if n not in names]
    fields = []
    for name in names:
        types = [s.field(name).type for s in schemas if name in s.names]
        real = [t for t in types if not pa.types.is_null(t)]
        t = real[0] if real else pa.null()
        if base is None and pa.types.is_integer(t) and any(pa.types.is_floating(x) for x in real):
            t = pa.float64()             # a fresh file may widen; an existing one is never changed
        fields.append(pa.field(name, t))
    return pa.schema(fields)


def pandas_metadata(base, parts, schema, total_rows: int):
    """The `pandas` entry of the file's metadata, which is what makes
    pd.read_parquet give back the same dtypes as before. Kept from the
    existing file, with the row count corrected and new columns added."""
    donors = ([base] if base is not None else []) + list(parts)
    donors = [d for d in donors if d.metadata and b"pandas" in d.metadata]
    if not donors:
        return None
    meta = json.loads(donors[0].metadata[b"pandas"])
    for d in meta.get("index_columns", []):
        if isinstance(d, dict) and d.get("kind") == "range":
            d["stop"] = d.get("start", 0) + total_rows * d.get("step", 1)
    have = {c.get("field_name") for c in meta.get("columns", [])}
    for name in schema.names:
        if name in have:
            continue
        for d in donors[1:]:
            entry = next((c for c in json.loads(d.metadata[b"pandas"]).get("columns", [])
                          if c.get("field_name") == name), None)
            if entry:
                meta["columns"].append(entry)
                break
    return {b"pandas": json.dumps(meta).encode()}


def conform_table(tbl, schema, text_formats):
    """`tbl` with exactly the columns of `schema`, in its order and types."""
    cols = []
    for f in schema:
        if f.name in tbl.column_names:
            cols.append(conform(tbl[f.name], f.type, f.name, text_formats.get(f.name)))
        else:
            cols.append(pa.chunked_array([pa.nulls(tbl.num_rows, f.type)]))
    return pa.Table.from_arrays(cols, schema=schema)


def same_values(a, b) -> bool:
    """Are two columns the same, value for value?

    Arrow's own equality calls NaN unequal to NaN, so a float column that
    holds one would "differ" from an exact copy of itself and stop a
    perfectly good merge. Here NaN matches NaN and null matches null.
    """
    a, b = a.combine_chunks(), b.combine_chunks()
    if len(a) != len(b) or a.null_count != b.null_count:
        return False
    if not pa.types.is_floating(a.type):
        return a.equals(b)
    if a.null_count and not pc.all(pc.equal(pc.is_null(a), pc.is_null(b))).as_py():
        return False
    eq = pc.or_(pc.equal(a, b), pc.and_(pc.is_nan(a), pc.is_nan(b)))
    return bool(pc.all(pc.fill_null(eq, True)).as_py())


def merge(part_paths, existing: bool) -> dict:
    """
    Write existing rows + the downloaded windows to OUTPUT_PATH, atomically.

    Existing rows are streamed through as stored. Nothing passes through
    pandas, so no dtype is re-inferred on the way: the only values that
    are converted are the NEW ones, to the types the file already uses.
    The finished file is checked before it replaces anything.
    """
    parts = [pq.read_table(p) for p in part_paths]
    part_schemas = [t.schema for t in parts]
    old = pq.ParquetFile(OUTPUT_PATH) if existing else None
    base = old.schema_arrow if old is not None else None
    n_old = old.metadata.num_rows if old is not None else 0
    n_new = sum(t.num_rows for t in parts)

    schema = merged_schema(base, part_schemas)
    schema = schema.with_metadata(pandas_metadata(base, part_schemas, schema, n_old + n_new) or {})

    # Where the file holds a date as text, new dates must be written in the
    # very same format, or pd.to_datetime would later meet two formats in
    # one column.
    text_formats = {}
    if base is not None:
        for f in base:
            news = [s.field(f.name).type for s in part_schemas if f.name in s.names]
            if _is_text(f.type) and any(pa.types.is_timestamp(t) or pa.types.is_date(t) for t in news):
                stored = pc.unique(pq.read_table(OUTPUT_PATH, columns=[f.name])[f.name].drop_null())
                text_formats[f.name] = text_date_format(stored.to_pylist(), f"{OUTPUT_PATH.name}: {f.name}")
                print(f"  {f.name}: the file stores text ({text_formats[f.name]}); "
                      f"new dates are written the same way")
    changed = [(f.name, str(f.type), sorted({str(s.field(f.name).type) for s in part_schemas
                                               if f.name in s.names and not s.field(f.name).type.equals(f.type)}))
               for f in schema]
    changed = [c for c in changed if c[2] and c[0] not in text_formats]
    if existing and changed:
        print("  converted to the file's stored type: "
              + ", ".join(f"{n} ({'/'.join(src)} -> {dst})" for n, dst, src in changed))

    tmp = OUTPUT_PATH.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp, schema, compression="snappy") as writer:
        if old is not None:
            # Bounded batches: the file may have been written as one huge row group.
            for batch in old.iter_batches(batch_size=250_000):
                writer.write_table(conform_table(pa.Table.from_batches([batch]), schema, text_formats))
        # Weekly windows are small; write them as row groups of a sensible size.
        pending, pending_rows = [], 0
        for i, tbl in enumerate(parts):
            pending.append(conform_table(tbl, schema, text_formats))
            pending_rows += tbl.num_rows
            if pending_rows >= 250_000 or i == len(parts) - 1:
                writer.write_table(pa.concat_tables(pending))
                pending, pending_rows = [], 0

    # ---- check the result before it replaces anything ----
    new = pq.ParquetFile(tmp)
    if new.metadata.num_rows != n_old + n_new:
        raise RuntimeError(f"wrote {new.metadata.num_rows:,} rows, expected {n_old + n_new:,}")
    if old is not None:
        for name in base.names:
            was = pq.read_table(OUTPUT_PATH, columns=[name])[name]
            now = pq.read_table(tmp, columns=[name])[name].slice(0, n_old)
            if pa.types.is_null(was.type):
                same = now.null_count == n_old
            else:
                same = same_values(now, was)
            if not same:
                raise RuntimeError(f"existing rows changed in column {name!r} — not replacing the file")
    # And it must still open the way 02_preprocess.py opens it.
    dates = pd.to_datetime(pd.read_parquet(tmp, columns=["game_date"])["game_date"])
    os.replace(tmp, OUTPUT_PATH)
    return {"rows": n_old + n_new, "old": n_old, "new": n_new, "columns": len(schema.names),
            "per_season": dates.dt.year.value_counts().sort_index(),
            "new_columns": [n for n in schema.names if base is not None and n not in base.names]}


def existing_days():
    """(calendar days the file already has rows for, its row count)."""
    col = pq.read_table(OUTPUT_PATH, columns=["game_date"])["game_date"]
    days = pc.unique(col.drop_null()).to_pandas()
    # Plain dates, so membership cannot depend on text vs timestamp or on resolution.
    return {d.date() for d in pd.to_datetime(days)}, len(col)


def protected_seasons() -> set:
    """Seasons the dataset in data_v5/ was built on.

    Filling a gap in one of them changes the rows a trained model's
    features came from, and `02_preprocess.py --freeze-from` would then,
    correctly, refuse to say the new season was built the same way.
    """
    try:
        return {int(s) for s in json.loads(DATASET_META.read_text()).get("seasons") or []}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser(description="Scrape Statcast pitches.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the existing parquet and re-scrape every season")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="also fetch weeks with no rows inside a season that "
                         "data_v5/ was built on (changes that season's data)")
    args = ap.parse_args()

    # One line that says which library pairing produced whatever follows.
    try:
        from importlib.metadata import version
        pyb = version("pybaseball")
    except Exception:
        pyb = "?"
    print(f"pandas {pd.__version__} | pyarrow {pa.__version__} | pybaseball {pyb}")

    incremental = OUTPUT_PATH.exists() and not args.fresh
    have_days, protected = set(), set()
    if incremental:
        have_days, n_rows = existing_days()
        stored = pq.read_schema(OUTPUT_PATH).field("game_date").type
        print(f"Existing file: {n_rows:,} rows over {len(have_days):,} game days "
              f"(game_date stored as {stored}).")
        print("Fetching only the weekly windows with no rows in it "
              "(--fresh re-scrapes everything).")
        if not args.fill_gaps:
            protected = protected_seasons()

    todo, gaps = [], []
    for start_dt, end_dt in SEASONS:
        for w0, w1 in season_windows(start_dt, end_dt):
            if any(d.date() in have_days for d in pd.date_range(w0, w1)):
                continue
            (gaps if w0.year in protected else todo).append((w0, w1))
    if gaps:
        print(f"\nLeft alone: {len(gaps)} week(s) with no rows inside a season "
              f"data_v5/ was built on:")
        for w0, w1 in gaps:
            print(f"    {w0.date()} → {w1.date()}")
        print("  They may simply have had no games. If they did, fetching them now would "
              "change\n  the data a trained model came from. (--fill-gaps fetches them anyway.)")
    if incremental and not todo:
        print("\nNothing to fetch — every window of every season is present.")
        return

    part_paths, failed, resumed = [], [], 0
    season = None
    for w0, w1 in todo:
        if w0.year != season:
            season = w0.year
            print(f"\n{'='*50}\nSeason {season}\n{'='*50}")
        path = part_path(w0, w1)
        tbl = load_part(path, w0.year)
        if tbl is not None:
            print(f"  {w0.date()} → {w1.date()} already downloaded ({tbl.num_rows:,} pitches)")
            part_paths.append(path)
            resumed += 1
            continue
        print(f"  Pulling {w0.date()} → {w1.date()} ...", end=" ", flush=True)
        df = fetch_window(w0, w1)
        if df is None:
            failed.append((w0, w1))
        else:
            raw = len(df)
            df = clean(df, w0.year)
            if len(df):
                save_part(df, path)
                part_paths.append(path)
            print(f"{len(df):,} pitches" + (f" (of {raw:,} pulled)" if raw != len(df) else ""))
        time.sleep(2)  # be nice to Baseball Savant servers
    if resumed:
        print(f"\n{resumed} window(s) came from {PARTS_DIR.name}/, saved by an earlier run.")

    if not part_paths and not incremental:
        print("No data collected! Check your internet connection and pybaseball.")
        sys.exit(1)

    if part_paths:
        got = set().union(*(pq.read_schema(p).names for p in part_paths))
        missing = [c for c in KEEP_COLUMNS if c not in got]
        if missing:
            print(f"WARNING: These columns were not in the data: {missing}")
        print(f"\n{'Appending to' if incremental else 'Writing'} {OUTPUT_PATH.name} ...")
        try:
            done = merge(part_paths, existing=incremental)
        except Exception as e:
            # A half-written temporary file is not a parquet and must not
            # be left where a later step, or git, could pick it up.
            OUTPUT_PATH.with_suffix(".parquet.tmp").unlink(missing_ok=True)
            print(f"\n{'!'*60}")
            print("The download is safe; the merge is what failed.")
            print(f"  {OUTPUT_PATH.name} has not been changed.")
            print(f"  The {len(part_paths)} downloaded window(s) are kept in {PARTS_DIR.name}/,")
            print("  so running this again will not download them a second time.")
            print(f"  Reason: {e}")
            print(f"{'!'*60}")
            if isinstance(e, SchemaMismatch):
                sys.exit(2)
            raise
        if incremental:
            print(f"Appended {done['new']:,} new rows to {done['old']:,} existing; "
                  f"every existing row checked unchanged.")
            if done["new_columns"]:
                print(f"  New column(s), empty for the existing rows: {done['new_columns']}")
        print(f"\nSaved to: {OUTPUT_PATH}")
        print(f"Final size: {done['rows']:,} rows, {done['columns']} columns")
        print("\nRows per season (a full season is roughly 700-750K):")
        for yr, n in done["per_season"].items():
            print(f"  {yr}: {n:,}")
        # Merged windows are in the file now; the saved copies have done their job.
        for p in part_paths:
            p.unlink(missing_ok=True)
        if PARTS_DIR.exists() and not any(PARTS_DIR.iterdir()):
            PARTS_DIR.rmdir()
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
