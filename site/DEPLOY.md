# Deploying to Render

Render builds from your GitHub repo and nothing else. It cannot see your
laptop, so **anything gitignored does not exist on the server** — that
includes `data_v5/`, `statcast_raw_v5.parquet` and
`best_model_v5.keras`. `site/serving/` is committed for exactly this
reason: it is the only path by which the model reaches production.

## 1. Build the bundle locally

Run these on the machine that has `data_v5/`, in this order. The first
two need your training environment; only the third matters for deploy.

```bash
.venv/bin/python analyze_actionability.py          # writes data_v5/product_claim_v5.json
.venv/bin/python site/player_names.py --from-serving   # MLB names; needs internet
.venv/bin/python site/build_serving_artifacts.py   # bundle + weight export
```

`build_serving_artifacts.py` does the name fetch and the weight export
itself, so the middle step is only needed if you want to refresh names
without rebuilding. `--no-names` and `--no-model` skip those stages.

Expect output ending like:

```
Serving bundle written to site/serving
  1,483 pitchers, 3,102 batters, 8.4 MB
```

It refuses to write a bundle over `BUNDLE_MAX_MB` (40). If you trip
that, raise `MIN_MATCHUP_MEETINGS` — do not raise the cap.

## 2. Check it before you push

```bash
.venv/bin/python site/app.py          # http://127.0.0.1:5000
```

Confirm the claim panel shows a coverage figure next to the accuracy
figure, and that the pickers show names rather than `Pitcher 605483`.
Numeric ids mean the name fetch did not run.

To rehearse the real thing:

```bash
pip install -r site/requirements.txt
gunicorn --chdir site wsgi:app --bind 0.0.0.0:8000 --workers 2 --preload
```

## 3. Commit the bundle

**This is the step people skip, and skipping it is worse than a broken
deploy.** The site would serve the previous model's priors and arsenals
against the new weights and return confident, plausible, wrong calls.
Nothing in the bundle carries a version stamp that would catch it.

```bash
git add site/serving && git commit -m "Rebuild serving bundle" && git push
```

`.gitignore` bans `*.parquet`, `*.npy` and friends everywhere except
`site/serving/`. That carve-out is deliberate; leave the rest alone.

## 4. Create the service

1. Render dashboard → **New → Blueprint**, point it at the repo.
2. It reads `render.yaml` and creates the `nextpitchai` web service.
3. First build takes a few minutes. Watch the log for
   `Booting worker with pid`.
4. Check `https://<service>.onrender.com/healthz` → `{"status": "ok"}`.

`/healthz` returns 200 only once the model can actually serve, so a
process that is up but cannot predict is correctly reported unhealthy
rather than being handed traffic.

Without a blueprint: New → Web Service, Python runtime, build
`pip install -r site/requirements.txt`, start command copied from
`render.yaml`, health check path `/healthz`.

## 5. Custom domain

1. Service → **Settings → Custom Domains → Add**, enter e.g.
   `nextpitch.example.com`.
2. Render shows the DNS record to create at your registrar:
   - subdomain → `CNAME` to `<service>.onrender.com`
   - apex/root → the `A` record Render gives you (most registrars cannot
     CNAME an apex; some offer ALIAS/ANAME, which also works)
3. Wait for propagation (minutes to a few hours). Render issues a TLS
   certificate automatically once it resolves — do not buy one.

## Know this about the free tier

It **sleeps after ~15 minutes idle** and takes roughly **30–60 seconds**
to wake. For a link you are sending people, that first impression is
bad. Their paid Starter tier (around $7/month at the time of writing —
check, pricing moves) keeps it warm.

A cron ping to keep it awake is against the spirit of the free tier and
Render may act on it; if the site matters, pay for the instance.

## Memory

Measured on this app, in clean subprocesses:

| | peak RSS |
|---|---|
| TensorFlow loading the model | 663 MB |
| `numpy_model` alone | 29 MB |
| Full serving stack | 128 MB |
| 2 gunicorn workers, preloaded (PSS) | 175 MB |

A free instance has 512 MB. **TensorFlow will not fit** — that is why
serving runs on exported NumPy weights and `site/requirements.txt` has
no `tensorflow`, `keras`, `torch` or `scikit-learn` in it. Adding any of
them silently reintroduces an OOM crash.

`--preload` matters: it loads the model once in the master before
forking, so workers share those pages copy-on-write. Summed RSS looks
like ~320 MB because it double-counts the shared pages; PSS (175 MB) is
the real figure.

## Troubleshooting

**Deploy succeeds, `/healthz` returns 503.** The bundle is missing or
incomplete. The JSON body carries the reason. Almost always
`site/serving/` was not committed — check `git ls-files site/serving`.

**"Out of memory" / worker killed.** Something pulled a heavy import in.
Check `pip list` in the build log for tensorflow or scikit-learn, and
that `--workers 2` was not raised.

**Pickers show `Pitcher 605483`.** `player_names.json` is missing or
empty. Run `.venv/bin/python site/player_names.py --from-serving` with
internet, rebuild, commit. The site works fine this way — names are
cosmetic.

**`CERTIFICATE_VERIFY_FAILED` during the name fetch.** Python cannot
verify HTTPS certificates. On macOS this is normal for python.org
builds, which ship their own OpenSSL and ignore the system keychain:

```bash
.venv/bin/pip install certifi     # the script prefers it automatically
# or: open "/Applications/Python 3.11/Install Certificates.command"
```

Then re-run the name fetch and rebuild. The bundle itself builds fine
without names.

**No claim panel.** `product_claim_v5.json` is not in the bundle. Run
`analyze_actionability.py`, then rebuild. Do not hand-write the numbers
into the template: the panel is wired to the gate so a retrain updates
the claim instead of leaving a stale boast on the page.

**The site answers, but the numbers feel wrong after a retrain.** You
almost certainly skipped step 1 or 3. Rebuild the bundle and commit it.

**Build hangs on `[mutex.cc] RAW: Lock blocking`.** A TensorFlow
deadlock, seen on macOS: numpy and pandas start Accelerate's thread pool,
then TF starts its own on top and the two lock up. The weight export
therefore runs in its own process, which prevents it. If you still hit
it, run the stages separately:

```bash
.venv/bin/python site/build_serving_artifacts.py --no-model
.venv/bin/python site/export_model.py
```

**First request after idle times out.** That is the free-tier cold start,
not a bug. `--timeout 120` already allows for it.

**404 on `/static/style.css`.** `--chdir site` is missing from the start
command; Flask resolves `static/` relative to the app root.
