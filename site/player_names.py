#!/usr/bin/env python3
"""
NextPitchAI v6 — MLB player id -> human name
============================================
The serving tables are keyed by raw MLB player id, so the site's
dropdowns show numbers. This turns those ids into names, handedness and
position so a hitter can pick "Gerrit Cole (RHP)" instead of 543037.

    python site/player_names.py --ids 605483,592789
    python site/player_names.py --from-serving     # every id in site/serving/

Three rules this module is built around:

  * `resolve()` NEVER touches the network. It is the call the deployed
    site makes on every request; a name lookup must not be able to hang
    a prediction or to fail because MLB is down. Only `fetch_names()`
    goes out, and only from the CLI or an explicit refresh.
  * `fetch_names()` never raises and never shrinks the cache. It merges
    what it got into what it had, and if the network is unreachable it
    logs a warning and returns the existing cache untouched. A name file
    is a convenience; losing it must never break serving.
  * The write is atomic (tmp file in the same directory + os.replace).
    A half-written artifact silently reused has already cost this
    project a full re-scrape once — see invariant 10 in AGENTS.md.

Standard library only, deliberately: the deploy image already carries
TensorFlow and Flask, and this must not add a dependency to it. pandas
is imported lazily inside `--from-serving`, which is a developer path
that runs on a machine that already has it.

The cache lives in site/serving/, which is gitignored and rebuilt per
training run. `build_serving_artifacts.py` only adds files to that
directory, so the name cache survives a rebuild — but it will not exist
on a fresh clone or a fresh deploy, and the fallback labels below are
what the site shows until someone runs this script.
"""
import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVING_DIR = Path(__file__).resolve().parent / "serving"
CACHE_PATH = SERVING_DIR / "player_names.json"

API_URL = "https://statsapi.mlb.com/api/v1/people"
# 100 six-digit ids + separators is ~750 characters of query string; the
# assembled URL is checked against MAX_URL_LEN anyway and the chunk is
# split further if a future id scheme makes it longer.
CHUNK_SIZE = 100
MAX_URL_LEN = 1800
TIMEOUT_S = 15
RETRIES = 3
BACKOFF_S = 1.5          # multiplied by 2**attempt, plus jitter
PAUSE_BETWEEN_CHUNKS_S = 0.2
USER_AGENT = "NextPitchAI/6 (+https://github.com/; contact: site admin)"

log = logging.getLogger("player_names")

# load_names() is called per request by the site; re-reading and
# re-parsing the JSON every time is wasteful, so keep the parsed map
# until the file's mtime/size changes. The returned dict is shared —
# treat it as read-only.
_MEMO: dict = {}


# --------------------------------------------------------------- cache

def _norm_id(value):
    """MLB ids as canonical strings; None for anything that isn't one."""
    try:
        return str(int(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _norm_record(raw):
    """Accept a full record or a bare name string (hand-written caches)."""
    if isinstance(raw, str):
        return {"name": raw, "hand": None, "bats": None, "pos": None}
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return {
        "name": name.strip(),
        "hand": raw.get("hand"),
        "bats": raw.get("bats"),
        "pos": raw.get("pos"),
    }


def load_names(cache_path=CACHE_PATH):
    """The cache as {"605483": {"name", "hand", "bats", "pos"}}.

    Missing, unreadable or corrupt file -> {}. A name cache is never
    worth raising over; the caller falls back to id labels.
    """
    path = Path(cache_path)
    try:
        stat = path.stat()
    except OSError:
        return {}

    stamp = (stat.st_mtime_ns, stat.st_size)
    hit = _MEMO.get(str(path))
    if hit is not None and hit[0] == stamp:
        return hit[1]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("player name cache %s is unreadable (%s); "
                    "continuing with id labels", path, exc)
        return {}
    if not isinstance(payload, dict):
        log.warning("player name cache %s is not a JSON object; ignoring", path)
        return {}

    names = {}
    for key, raw in payload.items():
        pid = _norm_id(key)
        rec = _norm_record(raw)
        if pid and rec:
            names[pid] = rec
    _MEMO[str(path)] = (stamp, names)
    return names


def _atomic_write(path, payload):
    """tmp file in the same directory, fsync, os.replace. See invariant 10."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # The destination is still whatever it was; only the tmp is lost.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ------------------------------------------------------------- labels

def label(pid, rec=None, kind="Player"):
    """Dropdown text. Unknown ids get a usable label, never a blank."""
    pid = _norm_id(pid) or str(pid)
    if not rec or not rec.get("name"):
        return f"{kind} {pid}"
    pos, hand = rec.get("pos"), rec.get("hand")
    bats = rec.get("bats")
    if pos == "P" and hand:
        return f"{rec['name']} ({hand}HP)"
    tail = "/".join(x for x in [pos, f"B:{bats}" if bats else None] if x)
    return f"{rec['name']} ({tail})" if tail else rec["name"]


def resolve(ids, cache_path=CACHE_PATH, kind="Player"):
    """Cache-only lookup for every requested id. NEVER hits the network.

    Returns {"605483": {"name", "hand", "bats", "pos", "known", "label"}}
    with an entry for every id asked for, so callers cannot KeyError and
    a dropdown cannot render a blank row.

    `name` is None when the id is not in the cache, and `label` is
    always a usable string ("Pitcher 605483"). Display `label`; test
    `name` (or `known`) to ask whether a real name is known. Filling
    `name` with the fallback would make `bool(rec["name"])` report every
    unresolved id as named, which is how a dropdown quietly starts
    presenting id strings as player names.
    """
    names = load_names(cache_path)
    out = {}
    for raw in ids:
        pid = _norm_id(raw)
        if pid is None:
            continue
        rec = names.get(pid)
        out[pid] = {
            "name": rec["name"] if rec else None,
            "hand": rec.get("hand") if rec else None,
            "bats": rec.get("bats") if rec else None,
            "pos": rec.get("pos") if rec else None,
            "known": rec is not None,
            "label": label(pid, rec, kind),
        }
    return out


def options(ids, cache_path=CACHE_PATH, kind="Player"):
    """[{"id": 605483, "label": ...}] sorted by name, unknowns last.

    What a by-name <select> wants. Named players sort alphabetically;
    ids with no name fall to the bottom rather than being dropped.
    """
    resolved = resolve(ids, cache_path, kind)
    rows = [{"id": int(pid), "label": rec["label"], "known": rec["known"]}
            for pid, rec in resolved.items()]
    rows.sort(key=lambda r: (not r["known"], r["label"].lower()))
    return rows


# ------------------------------------------------------------ fetching

def _chunks(ids):
    """CHUNK_SIZE ids per request, split further if the URL gets long."""
    batch = []
    for pid in ids:
        batch.append(pid)
        too_long = len(API_URL) + len("?personIds=") + len(",".join(batch)) > MAX_URL_LEN
        if len(batch) >= CHUNK_SIZE or too_long:
            if too_long and len(batch) > 1:
                yield batch[:-1]
                batch = [batch[-1]]
            else:
                yield batch
                batch = []
    if batch:
        yield batch


def _http_get_json(url):
    """One GET. Raises on failure; retry policy lives in _fetch_chunk."""
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_people(payload):
    """MLB /people response -> {id: record}. Tolerant of missing fields."""
    out = {}
    if not isinstance(payload, dict):
        return out
    for person in payload.get("people") or []:
        if not isinstance(person, dict):
            continue
        pid = _norm_id(person.get("id"))
        name = person.get("fullName")
        if not pid or not isinstance(name, str) or not name.strip():
            continue
        # Both hands are kept: the same id is a pitcher in one table and
        # a batter in another, and two-way players make that literal.
        out[pid] = {
            "name": name.strip(),
            "hand": (person.get("pitchHand") or {}).get("code"),
            "bats": (person.get("batSide") or {}).get("code"),
            "pos": (person.get("primaryPosition") or {}).get("abbreviation"),
        }
    return out


def _fetch_chunk(batch):
    """One chunk with retry/backoff. Returns {} rather than raising."""
    url = f"{API_URL}?personIds={','.join(batch)}"
    if len(url) > MAX_URL_LEN:
        log.warning("chunk URL is %d chars, over MAX_URL_LEN=%d; skipping",
                    len(url), MAX_URL_LEN)
        return {}

    for attempt in range(RETRIES):
        try:
            return _parse_people(_http_get_json(url))
        except urllib.error.HTTPError as exc:
            # 4xx other than rate-limiting will not fix itself on retry.
            transient = exc.code == 429 or exc.code >= 500
            log.warning("MLB API HTTP %s for %d ids (attempt %d/%d)%s",
                        exc.code, len(batch), attempt + 1, RETRIES,
                        "" if transient else " — not retrying")
            if not transient:
                return {}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.warning("MLB API request failed for %d ids (attempt %d/%d): %s",
                        len(batch), attempt + 1, RETRIES, exc)
        if attempt < RETRIES - 1:
            time.sleep(BACKOFF_S * (2 ** attempt) + random.uniform(0, 0.3))
    return {}


def fetch_names(ids, cache_path=CACHE_PATH, force=False):
    """Look up the ids we do not have yet and merge them into the cache.

    Returns the full merged map. Never raises and never writes a cache
    smaller than the one on disk: the merge is additive, and if nothing
    new arrived the file is left exactly as it was. An unreachable
    network is a warning, not an error — the site keeps serving with
    whatever names it already had.
    """
    merged = dict(load_names(cache_path))
    wanted, seen = [], set()
    for raw in ids:
        pid = _norm_id(raw)
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        if force or pid not in merged:
            wanted.append(pid)

    if not wanted:
        log.info("nothing to fetch: %d of %d ids already cached",
                 len(seen), len(seen))
        return merged

    log.info("fetching %d player names from the MLB Stats API", len(wanted))
    gained, failures, chunks = 0, 0, 0
    for batch in _chunks(wanted):
        chunks += 1
        got = _fetch_chunk(batch)
        if not got:
            failures += 1
        for pid, rec in got.items():
            if force or pid not in merged or merged[pid] != rec:
                merged[pid] = rec
                gained += 1
        if PAUSE_BETWEEN_CHUNKS_S:
            time.sleep(PAUSE_BETWEEN_CHUNKS_S)

    if failures:
        log.warning("%d of %d request batches failed — keeping the existing "
                    "cache for those ids (the site falls back to id labels)",
                    failures, chunks)
    if not gained:
        # Writing here could only replace a good file with an identical
        # or emptier one. Leave it alone.
        log.warning("no new names retrieved; %s left untouched (%d entries)",
                    cache_path, len(merged))
        return merged

    try:
        _atomic_write(cache_path, merged)
    except OSError as exc:
        log.warning("could not write %s (%s); returning names in memory only",
                    cache_path, exc)
        return merged
    log.info("wrote %s — %d entries (+%d new)", cache_path, len(merged), gained)
    return merged


# ----------------------------------------------------------------- CLI

def ids_from_serving(serving_dir=SERVING_DIR):
    """Every player id in the serving bundle. pandas is imported here
    only — the serving path must stay standard-library."""
    serving_dir = Path(serving_dir)
    sources = [("pitchers.parquet", ["pitcher_id"]),
               ("batters.parquet", ["batter_id"]),
               ("physics.parquet", ["pitcher"]),
               ("matchup_table_v5.parquet", ["pitcher", "batter"])]
    found = set()
    try:
        import pandas as pd
    except ImportError:
        log.error("--from-serving needs pandas; pass --ids instead")
        return []
    for fname, cols in sources:
        path = serving_dir / fname
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=cols)
        except Exception as exc:  # a stale bundle must not kill the CLI
            log.warning("could not read %s: %s", path, exc)
            continue
        for col in cols:
            found.update(_norm_id(v) for v in df[col].dropna().unique())
        log.info("%s -> %d ids so far", fname, len(found - {None}))
    found.discard(None)
    return sorted(found, key=int)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map raw MLB player ids to names for the site's dropdowns.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", help="comma-separated MLB player ids")
    src.add_argument("--from-serving", action="store_true",
                     help="every id in site/serving/*.parquet")
    ap.add_argument("--cache", default=str(CACHE_PATH), help="cache file path")
    ap.add_argument("--serving-dir", default=str(SERVING_DIR))
    ap.add_argument("--force", action="store_true",
                    help="refetch ids already in the cache")
    ap.add_argument("--offline", action="store_true",
                    help="no network; just print what the cache resolves")
    ap.add_argument("--kind", default="Player",
                    help='fallback label prefix, e.g. "Pitcher"')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.from_serving:
        ids = ids_from_serving(args.serving_dir)
    else:
        ids = [i for i in (_norm_id(x) for x in args.ids.split(",")) if i]
    if not ids:
        log.error("no usable ids")
        return 2

    before = len(load_names(args.cache))
    if not args.offline:
        fetch_names(ids, args.cache, force=args.force)
    resolved = resolve(ids, args.cache, args.kind)
    known = sum(1 for r in resolved.values() if r["known"])

    for row in options(ids, args.cache, args.kind)[:20]:
        print(f"  {row['id']:>9}  {row['label']}")
    if len(ids) > 20:
        print(f"  ... {len(ids) - 20} more")
    print(f"\n{known}/{len(ids)} ids resolved to names "
          f"(cache {args.cache}: {before} -> {len(load_names(args.cache))})")
    # Non-zero when the fetch could not fill anything in, so a cron or a
    # deploy script notices instead of shipping a dropdown full of ids.
    return 0 if (known or args.offline) else 1


if __name__ == "__main__":
    sys.exit(main())
