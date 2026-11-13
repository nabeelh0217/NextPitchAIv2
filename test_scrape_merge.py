"""
NextPitchAI — adding rows to the scrape must not disturb what is there
=======================================================================
The first attempt to add a season to an existing file downloaded 714,115
pitches and then could not save them: the file held dates as text, the
new pull held them as timestamps, and the two were stacked in one column.
The test that should have caught it replaced pybaseball with a stub that
returned one dtype for everything, so it never mixed the two.

These build the two kinds of file by hand and append to each, and check
the three things that matter: the existing rows come through unchanged,
the new rows are stored the way the file already stores that column, and
a merge that cannot be done leaves the file exactly as it was.

No network, no pybaseball, no data. A few seconds.

    .venv/bin/python test_scrape_merge.py
"""
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_spec = importlib.util.spec_from_file_location(
    "scrape", Path(__file__).resolve().parent / "01_scrape_statcast.py")
scrape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scrape)


def rows(first_day, n_days, start_pk, per_day=40):
    """A small Statcast-shaped frame: one game a day, `per_day` pitches."""
    days = pd.date_range(first_day, periods=n_days)
    n = n_days * per_day
    rng = np.random.default_rng(start_pk)
    return pd.DataFrame({
        "game_pk": np.repeat(np.arange(start_pk, start_pk + n_days), per_day),
        "game_date": np.repeat(days.values, per_day),
        "at_bat_number": np.tile(np.arange(per_day) // 4 + 1, n_days),
        "pitch_number": np.tile(np.arange(per_day) % 4 + 1, n_days),
        "pitcher": pd.array(rng.integers(500000, 700000, n), dtype="Int64"),
        "pitch_type": rng.choice(["FF", "SL", "CH"], n),
        "release_speed": rng.normal(92, 5, n).round(1),
        "on_1b": pd.array(np.where(rng.random(n) < 0.7, pd.NA, 600001), dtype="Int64"),
        "home_team": rng.choice(["NYY", "LAD"], n),
    })


class Workspace:
    """A scratch directory standing in for the repo root."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.saved = (scrape.OUTPUT_PATH, scrape.PARTS_DIR)
        scrape.OUTPUT_PATH = self.dir / "statcast_raw_v5.parquet"
        scrape.PARTS_DIR = self.dir / "statcast_raw_v5.parts"
        return self

    def __exit__(self, *exc):
        scrape.OUTPUT_PATH, scrape.PARTS_DIR = self.saved
        self.tmp.cleanup()

    def existing(self, table: pa.Table):
        pq.write_table(table, scrape.OUTPUT_PATH)
        return pq.read_table(scrape.OUTPUT_PATH)

    def parts(self, *frames):
        """Windows as the scrape saves them: cleaned, dates typed."""
        out = []
        for i, df in enumerate(frames):
            df = df.copy()
            df["game_date"] = scrape.as_dates(df["game_date"])
            path = scrape.PARTS_DIR / f"part{i}.parquet"
            scrape.save_part(df, path)
            out.append(path)
        return out


def as_written_under_pandas3(df):
    """An existing file as the ORIGINAL scrape wrote it under pandas 3:
    pybaseball left the dates as text, and text columns are large_string."""
    t = pa.Table.from_pandas(df.assign(game_date=df["game_date"].dt.strftime("%Y-%m-%d")),
                             preserve_index=False)
    for name in ("game_date", "pitch_type", "home_team"):
        i = t.schema.get_field_index(name)
        t = t.set_column(i, name, t[name].cast(pa.large_string()))
    return t


def as_written_under_pandas2(df):
    """...and under pandas 2: dates parsed to timestamps, text as string."""
    return pa.Table.from_pandas(df, preserve_index=False)


# ---------------------------------------------------------------------
# The failure that happened
# ---------------------------------------------------------------------
def test_new_timestamps_are_stored_as_the_files_text():
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    with Workspace() as w:
        before = w.existing(as_written_under_pandas3(old_df))
        done = scrape.merge(w.parts(new_df), existing=True)
        after = pq.read_table(scrape.OUTPUT_PATH)

        assert done["old"] == len(old_df) and done["new"] == len(new_df)
        # The file's own types win, column for column.
        for f in before.schema:
            assert after.schema.field(f.name).type.equals(f.type), f.name
        # Existing rows: identical, value for value.
        assert after.slice(0, len(old_df)).equals(before)
        # New dates: text, in the SAME format, so one to_datetime reads the column.
        new_dates = after["game_date"].slice(len(old_df)).to_pylist()
        assert new_dates[0] == "2026-03-25" and new_dates[-1] == "2026-03-30"
        parsed = pd.to_datetime(pd.read_parquet(scrape.OUTPUT_PATH)["game_date"])
        assert parsed.dt.year.value_counts().to_dict() == {2025: len(old_df), 2026: len(new_df)}


def test_reading_it_back_with_pandas_gives_the_same_dtypes_as_before():
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    for writer in (as_written_under_pandas3, as_written_under_pandas2):
        with Workspace() as w:
            w.existing(writer(old_df))
            was = pd.read_parquet(scrape.OUTPUT_PATH)
            scrape.merge(w.parts(new_df), existing=True)
            now = pd.read_parquet(scrape.OUTPUT_PATH)
            assert list(now.columns) == list(was.columns)
            assert (now.dtypes == was.dtypes).all(), (writer.__name__, now.dtypes, was.dtypes)
            pd.testing.assert_frame_equal(now.iloc[:len(was)], was)
            # The new values are the ones that were downloaded.
            got = now.iloc[len(was):].reset_index(drop=True)
            assert np.array_equal(got["game_pk"].to_numpy(), new_df["game_pk"].to_numpy())
            assert np.allclose(got["release_speed"].astype(float), new_df["release_speed"])
            assert (pd.to_datetime(got["game_date"]).values == new_df["game_date"].values).all()
            assert got["on_1b"].isna().equals(new_df["on_1b"].isna())


def test_a_file_with_timestamps_keeps_timestamps_whatever_the_resolution():
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    with Workspace() as w:
        before = w.existing(as_written_under_pandas2(old_df))
        stored = before.schema.field("game_date").type
        new_df["game_date"] = new_df["game_date"].astype("datetime64[s]")     # another resolution
        scrape.merge(w.parts(new_df), existing=True)
        after = pq.read_table(scrape.OUTPUT_PATH)
        assert after.schema.field("game_date").type.equals(stored)
        assert after.slice(0, len(old_df)).equals(before)


def test_text_dates_can_be_fitted_to_a_timestamp_column():
    col = pa.chunked_array([pa.array(["2026-04-01", None, "2026-04-02"])])
    out = scrape.conform(col, pa.timestamp("ns"), "game_date")
    assert out.type == pa.timestamp("ns")
    assert out.to_pylist() == [pd.Timestamp("2026-04-01"), None, pd.Timestamp("2026-04-02")]


def test_as_dates_accepts_whatever_pybaseball_returned():
    want = pd.Series(pd.to_datetime(["2026-04-01", "2026-04-02"]))
    text = pd.Series(["2026-04-01", "2026-04-02"])
    for given in (text, text.astype(object), want, want.dt.tz_localize("UTC"),
                  pd.Series([pd.Timestamp("2026-04-01"), "2026-04-02"], dtype=object)):
        got = scrape.as_dates(given)
        assert pd.api.types.is_datetime64_any_dtype(got) and got.dt.tz is None
        assert (got.values == want.values).all()


# ---------------------------------------------------------------------
# A merge that cannot be done must change nothing
# ---------------------------------------------------------------------
def _untouched(path, before_bytes):
    return path.read_bytes() == before_bytes and not path.with_suffix(".parquet.tmp").exists()


def test_an_unknown_text_date_format_is_refused_not_guessed():
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    with Workspace() as w:
        odd = pa.Table.from_pandas(
            old_df.assign(game_date=old_df["game_date"].dt.strftime("%m/%d/%Y")), preserve_index=False)
        w.existing(odd)
        raw = scrape.OUTPUT_PATH.read_bytes()
        parts = w.parts(new_df)
        try:
            scrape.merge(parts, existing=True)
            raise AssertionError("merged into a date format it could not have known")
        except scrape.SchemaMismatch as e:
            assert "09/20/2025" in str(e)
        scrape.OUTPUT_PATH.with_suffix(".parquet.tmp").unlink(missing_ok=True)   # main() does this
        assert _untouched(scrape.OUTPUT_PATH, raw)
        assert all(p.exists() for p in parts)        # the download is still there


def test_a_date_with_a_time_is_not_squeezed_into_a_date_only_column():
    col = pa.chunked_array([pa.array([pd.Timestamp("2026-04-01 19:05:00")], type=pa.timestamp("ns"))])
    try:
        scrape.conform(col, pa.large_string(), "game_date", "%Y-%m-%d")
        raise AssertionError("dropped the time of day")
    except scrape.SchemaMismatch:
        pass


def test_numbers_convert_only_when_nothing_is_lost():
    whole = pa.chunked_array([pa.array([3.0, None, 5.0])])
    assert scrape.conform(whole, pa.int64(), "zone").to_pylist() == [3, None, 5]
    ints = pa.chunked_array([pa.array([3, None, 5])])
    assert scrape.conform(ints, pa.float64(), "zone").to_pylist() == [3.0, None, 5.0]
    try:
        scrape.conform(pa.chunked_array([pa.array([3.5])]), pa.int64(), "zone")
        raise AssertionError("truncated 3.5 to an integer")
    except scrape.SchemaMismatch as e:
        assert "zone" in str(e)


def test_a_nan_in_the_existing_file_does_not_look_like_a_changed_row():
    # NaN is not equal to NaN, so a naive "did the old rows change?" check
    # would refuse to extend any file that holds one.
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    with Workspace() as w:
        t = as_written_under_pandas3(old_df)
        speed = t["release_speed"].to_numpy().copy()
        speed[[3, 50]] = np.nan
        i = t.schema.get_field_index("release_speed")
        t = t.set_column(i, "release_speed", pa.array(speed, from_pandas=False))    # real NaNs, not nulls
        before = w.existing(t)
        assert before["release_speed"].null_count == 0
        scrape.merge(w.parts(new_df), existing=True)
        after = pq.read_table(scrape.OUTPUT_PATH)["release_speed"].slice(0, len(old_df))
        assert np.array_equal(after.to_numpy(), speed, equal_nan=True)
    x = pa.chunked_array([pa.array([1.0, float("nan"), None])])
    assert scrape.same_values(x, x)
    assert not scrape.same_values(x, pa.chunked_array([pa.array([1.0, 2.0, None])]))
    assert not scrape.same_values(x, pa.chunked_array([pa.array([1.0, None, float("nan")])]))


# ---------------------------------------------------------------------
# Columns that come and go
# ---------------------------------------------------------------------
def test_a_new_column_is_added_and_is_empty_for_existing_rows():
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    new_df["game_type"] = "R"
    with Workspace() as w:
        before = w.existing(as_written_under_pandas3(old_df))
        done = scrape.merge(w.parts(new_df), existing=True)
        after = pq.read_table(scrape.OUTPUT_PATH)
        assert done["new_columns"] == ["game_type"]
        assert after.column_names == before.column_names + ["game_type"]
        assert after["game_type"].slice(0, len(old_df)).null_count == len(old_df)
        assert set(after["game_type"].slice(len(old_df)).to_pylist()) == {"R"}
        assert after.select(before.column_names).slice(0, len(old_df)).equals(before)
        # pandas can still tell what every column is, the new one included.
        meta = json.loads(after.schema.metadata[b"pandas"])
        assert "game_type" in {c["field_name"] for c in meta["columns"]}
        assert pd.read_parquet(scrape.OUTPUT_PATH)["game_type"].iloc[-1] == "R"


def test_a_column_missing_from_the_new_rows_is_empty_not_invented():
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    with Workspace() as w:
        w.existing(as_written_under_pandas3(old_df))
        scrape.merge(w.parts(new_df.drop(columns=["on_1b"])), existing=True)
        after = pq.read_table(scrape.OUTPUT_PATH)
        assert after["on_1b"].slice(len(old_df)).null_count == len(new_df)
        assert after.schema.field("on_1b").type == pa.int64()


def test_a_column_that_was_entirely_empty_takes_the_new_rows_type():
    old_df, new_df = rows("2025-09-20", 5, 776000), rows("2026-03-25", 6, 823000)
    with Workspace() as w:
        t = as_written_under_pandas3(old_df)
        i = t.schema.get_field_index("on_1b")
        w.existing(t.set_column(i, "on_1b", pa.nulls(len(old_df))))           # Arrow's null type
        scrape.merge(w.parts(new_df), existing=True)
        after = pq.read_table(scrape.OUTPUT_PATH)
        assert after.schema.field("on_1b").type == pa.int64()
        assert after["on_1b"].slice(0, len(old_df)).null_count == len(old_df)
        assert after["on_1b"].slice(len(old_df)).null_count == int(new_df["on_1b"].isna().sum())


# ---------------------------------------------------------------------
# A first scrape, with no file to conform to
# ---------------------------------------------------------------------
def test_a_fresh_scrape_stores_dates_as_dates_and_widens_numbers():
    a, b = rows("2026-03-25", 3, 823000), rows("2026-04-01", 3, 823100)
    a["release_speed"] = a["release_speed"].round().astype("int64")   # a window of whole numbers
    with Workspace() as w:
        done = scrape.merge(w.parts(a, b), existing=False)
        after = pq.read_table(scrape.OUTPUT_PATH)
        assert done["rows"] == len(a) + len(b) and done["old"] == 0
        assert pa.types.is_timestamp(after.schema.field("game_date").type)
        assert after.schema.field("release_speed").type == pa.float64()
        got = pd.read_parquet(scrape.OUTPUT_PATH)
        assert np.allclose(got["release_speed"].astype(float),
                           np.r_[a["release_speed"].to_numpy(float), b["release_speed"].to_numpy()])


def test_windows_are_resumed_only_when_they_can_be_trusted():
    with Workspace() as w:
        good = rows("2026-03-25", 2, 823000).assign(game_type="R")
        (path,) = w.parts(good)
        assert scrape.load_part(path, 2026).num_rows == len(good)
        # A regular-season-only season must not resume a window holding exhibition games.
        (mixed,) = w.parts(good.assign(game_type=["R", "S"] * (len(good) // 2)))
        assert scrape.load_part(mixed, 2026) is None
        assert scrape.load_part(mixed, 2025) is not None        # 2025 is kept as first scraped
        assert scrape.load_part(scrape.PARTS_DIR / "absent.parquet", 2026) is None
        broken = scrape.PARTS_DIR / "broken.parquet"
        broken.write_bytes(b"not a parquet file")
        assert scrape.load_part(broken, 2026) is None


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:                     # a crash is a failure too
            failed += 1
            print(f"  FAIL  {name}  {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
