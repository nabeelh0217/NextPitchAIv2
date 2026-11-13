# Deploying to Render

Render builds from your GitHub repo and nothing else. It cannot see your
laptop, so **anything gitignored does not exist on the server** — that
includes `data_v5/`, `statcast_raw_v5.parquet` and
`best_model_v5.keras`. `site/serving/` is committed for exactly this
reason: it is the only path by which the model reaches production.

## 0. Which machine

Run these wherever `data_v5/` lives. It and `statcast_raw_v5.parquet`
are gitignored, so a fresh clone on another computer does NOT have them
— copy them across rather than retraining. Retraining produces a
different model, which invalidates the numbers in
`product_claim_v5.json` and the claim the site makes; you would have to
re-run `analyze_actionability.py` and re-check the ship criterion first.

Commands below use the macOS/Linux venv path. On Windows use
`.venv\Scripts\python` throughout, e.g.

```
.venv\Scripts\python site\build_serving_artifacts.py
```

## 1. Build the bundle locally

Run these on the machine that has `data_v5/`, one line at a time. Do not
paste them as a block with comments on the end; zsh treats the `#` as
part of the command.

**A new season, same model** (no retraining; this is what puts a clean
test number on the site):

```bash
./run_new_season.sh
```

That runs the four steps below in order and stops at the first one that
fails, telling you which. They can also be run one at a time:

```bash
.venv/bin/python 01_scrape_statcast.py
.venv/bin/python 02_preprocess.py --freeze-from data_v5 --out-dir data_ext
.venv/bin/python analyze_actionability.py --data-dir data_ext --test-season 2026
.venv/bin/python site/build_serving_artifacts.py --data-dir data_ext
```

What each one does:

1. Fetches only the weeks the parquet does not already have and appends
   them. The rows the model was built on are not touched.
2. Builds `data_ext/` with the ID maps and scalers frozen, and stops with
   an error unless the 2022-2025 rows come out identical to `data_v5/`.
   Needs about as much memory and disk again as the original preprocess.
3. The gate. Ten to fifteen minutes: four passes over a season, in
   NumPy, with progress printed. Read sections A, 6 and 7 and the two
   verdicts. It is deterministic, so running it twice gives the same
   answer; what you may not do is change a setting after seeing it.
4. The bundle, from the same directory the gate just measured.

**After a retrain** (the model changed; `data_v5/` is the dataset):

```bash
.venv/bin/python analyze_actionability.py
.venv/bin/python site/build_serving_artifacts.py
```

The builder fetches player names and exports the weights itself.
`--no-names` and `--no-model` skip those stages. It will not run without
a claim in the dataset directory, and it refuses a claim that was
measured on a different model file than the one it is exporting.

Expect output ending like:

```
Serving bundle written to site/serving
  1,612 pitchers, 3,240 batters, 9.1 MB, data through 2026-09-27
```

It refuses to write a bundle over `BUNDLE_MAX_MB` (40). If you trip
that, raise `MIN_MATCHUP_MEETINGS` — do not raise the cap.

## 2. Check it before you push

```bash
.venv/bin/python site/app.py          # http://127.0.0.1:5000
```

Confirm the claim panel shows a coverage figure next to the accuracy
figure, that it names the season it was tested on, and that the pickers
show names rather than `Pitcher 605483`. Numeric ids mean the name fetch
did not run.

**Read the About tab too.** Its last paragraph is chosen by the gate's
verdict (`ABOUT_CLOSING` in `app.py`), so it changes when the result
does. It is written in the first person on your behalf — edit the
wording there if it does not sound like you, but keep one per verdict.

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

## 4b. Which branch Render watches

`render.yaml` sets `autoDeploy: true` but names no branch, so Render
watches the repo's DEFAULT branch. If your work is on a feature branch,
nothing deploys and the dashboard looks fine.

Either merge to the default branch, or pick the branch in the Blueprint
screen, or add `branch: <name>` to `render.yaml`. Merging is better —
a feature branch as a permanent deploy target is something you will
forget about.

After that, every push to that branch rebuilds and redeploys. Pushing
code is all it takes.

**Except for the model.** A push does not retrain anything and does not
rebuild the bundle. New weights reach Render only if you run
`build_serving_artifacts.py` locally and commit `site/serving/` — which
is the whole reason that directory is committed.

## 5. Custom domain

Add the domain in Render FIRST — DNS cannot verify against a domain the
service does not know about.

Service → **Settings → Custom Domains → Add Custom Domain**. Add both
the apex (`yourdomain.com`) and `www.yourdomain.com`. Render then shows
the records to create: an **A record value** for the apex and a **CNAME
target** for www.

**Use the values Render displays.** Their IPs change, and a stale one
copied from a tutorial fails silently — the domain simply never
resolves and nothing explains why.

### At the registrar (GoDaddy shown; others differ only in menus)

My Products → the domain → **DNS → Manage Zones**.

| Type | Name | Value | TTL |
|---|---|---|---|
| A | `@` | the IP Render showed | 600 |
| CNAME | `www` | `<service>.onrender.com` | 600 |

Two GoDaddy-specific traps, both of which produce a site that half
works or does not load at all:

- **Edit the existing `@` A record, do not add a second one.** GoDaddy
  ships a parking A record on `@`. Two A records make DNS round-robin
  between Render and the parking page, so the site loads intermittently
  and looks like a Render problem. Same for the `www` CNAME, which
  usually already exists pointing at `@`.
- **Delete any Domain Forwarding.** It is often on by default and
  overrides DNS entirely, so the domain never reaches Render no matter
  how correct the records are.

TTL 600 while setting up: the 1-hour default makes every mistake take an
hour to correct.

### Verify

```bash
dig +short yourdomain.com          # Render's IP
dig +short www.yourdomain.com      # <service>.onrender.com
```

Once those resolve, Render marks the domain Verified and issues TLS
itself — minutes, occasionally an hour. Do not buy a certificate. If it
sits on "Verifying" beyond that, it is almost always one of the two
traps above.

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

**The scrape says "The download is safe; the merge is what failed".**
Your existing file has not been touched, and the weeks it downloaded are
kept in `statcast_raw_v5.parts/`, so the next run will not download them
again. The "Reason" line names the column and the two types that would
not fit together; send that line.

**The scrape lists weeks it "left alone".** They are weeks with no rows
inside a season your trained model was built on. Usually they simply had
no games. Fetching one would change that season's data, and step 2 would
then refuse, so it does not unless you pass `--fill-gaps`.

**Step 2 says "Nothing to extend".** The parquet holds no season beyond
the ones `data_v5/` was built on: the scrape did not finish. Run it
again and check it ends with "Saved to:".

**Step 2 stops on a `STOP` line about a column being empty.** Baseball
Savant has stopped sending, or renamed, a column the model uses, for the
new season only. Nothing was written. This needs a code change in the
scrape; send the STOP lines.

**"The arrays were rebuilt after this model was trained".** `data_v5/`
was re-preprocessed and the model in it is from before. Retrain.

**`02_preprocess.py --freeze-from` stops with "does not reproduce".**
The 2022-2025 rows no longer produce the arrays the model was trained
on — usually because the parquet's old rows changed (a `--fresh`
re-scrape) or a library was upgraded. Nothing was written. Do not pass
`--allow-mismatch` to get past it for a number you intend to publish:
the model would be scored on inputs it was not built with.

**The gate says the dataset "has no row_meta_v5.npz".** It was built
before the gate needed raw pitcher ids and integer counts. Run the
`--freeze-from` line above; it does not refit anything.

**The claim panel says the edge is "too small to tell from zero".**
That is the result, not a bug. See `docs/EXPERIMENTS.md` for the
criteria, which were fixed before the test was run.

**No claim panel.** `product_claim_v5.json` is not in the bundle. Run
`analyze_actionability.py`, then rebuild. Do not hand-write the numbers
into the template: the panel is wired to the gate so a retrain updates
the claim instead of leaving a stale boast on the page.

**The site answers, but the numbers feel wrong after a retrain.** You
almost certainly skipped step 1 or 3. Rebuild the bundle and commit it.

**`mutex lock failed` / `libc++abi: terminating` during export.** A
broken TensorFlow install. It no longer matters: reading a `.keras` file
needs only `zipfile` and `h5py`, so the export runs without TensorFlow
and the weights are written regardless. You will see "Verification DID
NOT RUN" — that is the Keras-vs-NumPy cross-check being skipped, not the
export failing. Check `site/serving/model_weights.npz` exists and carry
on; run `site/test_numpy_parity.py` wherever TF works if you want the
check.

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
