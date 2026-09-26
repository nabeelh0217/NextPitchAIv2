"""
Gunicorn entrypoint: `gunicorn --chdir site wsgi:app`.

Importing this module builds the Predictor eagerly rather than on the
first request. With --preload that happens once in the master process
before workers fork, so the weights and lookup tables are shared
copy-on-write instead of duplicated per worker, and the first visitor
does not pay the load cost on top of a cold start.

A failure here must NOT kill the process: app.py surfaces the reason on
the page and /healthz returns 503, which is far easier to diagnose from
Render's dashboard than a boot loop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app, get_predictor  # noqa: E402

try:
    get_predictor()
except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
    print(f"wsgi: predictor unavailable at boot: {exc}", file=sys.stderr)

if __name__ == "__main__":
    app.run()
