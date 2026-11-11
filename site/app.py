"""
NextPitchAI v6 — Flask app
==========================
    python site/build_serving_artifacts.py   # once, after training
    python site/app.py                       # http://127.0.0.1:5000

The product is a SELECTIVE OVERLAY. It stays silent unless the read
clears the confidence threshold the gate measured, because a wrong
commit costs a hitter more than no advice. Every accuracy figure shown
is read from product_claim_v5.json, written by analyze_actionability.py
on held-out rows — the app never hardcodes a number, so it cannot keep
quoting a figure a retrain invalidated.

The same goes for the WORDS. "Beats a scouting report" is a claim about
a number; claim_view() decides which sentences the page may use from the
gate's verdict, so the copy cannot outlive the result either.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request

from predictor import NotBuilt, Predictor
from pitch_features import FASTBALL_FAMILY, OUTCOME_NAMES, PITCH_NAMES

# Everything on the About tab, in one place so it can be edited without
# touching markup. Deliberately sparse: fill in what you want public,
# delete what you do not.
ABOUT = {
    "name": "Nabeel Hussain",
    "tagline": "I built NextPitchAI to find out whether a model could give a "
               "hitter a real advantage in the box — not a better accuracy "
               "number, an actual edge.",
    "body": [
        "Most of the work turned out to be measurement rather than modelling. "
        "The first version looked good: 48% accuracy across ten pitch types, "
        "well above the 42% you get from knowing nothing but who is on the "
        "mound. Then I scored it against the thing a hitter already has — a "
        "count-based scouting report — and most of the apparent edge "
        "evaporated. The model mostly knew that pitchers throw fastballs on "
        "3-0, which every hitter also knows.",

        "Rebuilding the evaluation was harder than rebuilding the model. "
        "Testing on randomly held-out pitches let it see other pitches from "
        "the same at-bat, which flattered it badly; the honest test is a "
        "whole season it has never seen. Several results I was pleased with "
        "did not survive that change.",
    ],
    "links": [
        {"label": "GitHub", "url": "https://github.com/nabeelh0217/NextPitchAIv2"},
        # {"label": "LinkedIn", "url": "https://www.linkedin.com/in/..."},
        # {"label": "Email", "url": "mailto:you@example.com"},
    ],
}

# The closing paragraph of the bio states a result, so it is written per
# verdict rather than once. Edit the wording freely; keep one per verdict.
ABOUT_CLOSING = {
    # Before the fair re-measurement: the claim as first published.
    "legacy": (
        "What is here is what survived. It learns from about 2.9 million "
        "pitches thrown between 2022 and 2024 and is measured on all of 2025, "
        "a season held out entirely. It speaks on roughly a third of pitches "
        "and beats a count-split scouting report by a few points there, and "
        "stays quiet the rest of the time. That is a smaller claim than I set "
        "out hoping for. It is the one I can defend."),
    "validation_only": (
        "What is here is what survived so far. It learns from pitches thrown "
        "through {train_through} and is measured on {season}, against "
        "{bar}. On about {coverage} of pitches it gives a read, and is right "
        "{tool} of the time there to the report's {table}. Those settings "
        "were tuned on that same season, so the real test is the next one."),
    "confirmed": (
        "What is here is what survived. It learns from pitches thrown through "
        "{train_through}, was tuned on {tuned_on}, and is tested on all of "
        "{season} — a season it never saw, against {bar}. It speaks on about "
        "{coverage} of pitches and beats that report by {edge} points there, "
        "and stays quiet the rest of the time. That is a smaller claim than I "
        "set out hoping for. It is the one I can defend."),
    "reduced": (
        "What is here is what survived. It learns from pitches thrown through "
        "{train_through}, was tuned on {tuned_on}, and is tested on all of "
        "{season} — a season it never saw, against {bar}. It speaks on about "
        "{coverage} of pitches and beats that report by {edge} points there. "
        "That is real, but it is under the two points I set as the bar before "
        "I looked, and I would rather show the small number than a flattering "
        "one."),
    "not_confirmed": (
        "The last test was the hardest. I let the scouting report keep up "
        "with the season, the way a real one does, and scored all of "
        "{season} — a year the model had never seen and nothing had been "
        "tuned on. There it speaks on about {coverage} of pitches and is "
        "right {tool} of the time; the scouting report is right {table} on "
        "those same pitches. That is not an edge. I have left the tool up "
        "with the real number on it, because the measurement is the result, "
        "and it says exactly what the next version has to beat."),
}


def claim_view(claim):
    """
    Everything the page says about accuracy, derived from the gate's claim.

    Returns None when there is no claim to show. `status` is the gate's
    verdict: confirmed / reduced / not_confirmed for a season nothing was
    tuned on, validation_only for the tuning season, and legacy for a
    claim written before the gate compared against an up-to-date table.
    """
    rec = (claim or {}).get("recommended")
    if not rec:
        return None
    status = claim.get("status") or "legacy"
    overall = claim.get("overall") or {}
    ci = rec.get("edge_ci")
    pct = lambda x, d=1: f"{x * 100:.{d}f}%"        # noqa: E731
    v = {
        "status": status,
        "tested": status in ("confirmed", "reduced", "not_confirmed"),
        # May the page say the tool BEATS the report? Only on the gate's word.
        "beats": status in ("confirmed", "reduced", "legacy")
                 or (status == "validation_only" and rec["edge"] > 0),
        "coverage": pct(rec["coverage"], 0),
        "silent": pct(1.0 - rec["coverage"], 0),
        "tool": pct(rec["tool_accuracy"]),
        "table": pct(rec["table_accuracy"]),
        "edge": f"{rec['edge'] * 100:+.1f}",
        "ci": (f"{ci[0] * 100:+.1f} to {ci[1] * 100:+.1f}" if ci else None),
        "n": f"{int(claim.get('n_val', 0)):,}",
        # A claim from before the gate kept the table current was measured
        # against a frozen one; the page must not call that "up to date".
        "bar": ("an up-to-date scouting report" if status != "legacy"
                else "a count-split scouting report"),
        "bar_detail": claim.get("baseline_plain"),
        # Where the difference sits relative to zero, by its interval.
        "sign": ("ahead" if ci and ci[0] > 0 else
                 "behind" if ci and ci[1] < 0 else "zero"),
        "season": claim.get("evaluated_season"),
        "tuned_on": claim.get("tuned_on"),
        "train_through": claim.get("train_through"),
        "split": claim.get("split"),
        "min_edge": f"{float(claim.get('min_edge', 0.02)) * 100:.0f}",
        # Across EVERY situation, not only where it speaks.
        "overall": overall.get("verdict") or "legacy",
        "overall_lift": f"{float(overall.get('lift', claim.get('section5_lift_vs_scout', 0.0))) * 100:+.1f}",
    }
    v["bar_cap"] = v["bar"][0].upper() + v["bar"][1:]
    return v


def about_for(view):
    """The About tab, with a closing paragraph that matches the verdict.

    With no claim there is no result to state, so there is no closing
    paragraph: a sentence about beating a scouting report must not appear
    on a page that carries no measurement.
    """
    if not view:
        return ABOUT
    closing = ABOUT_CLOSING.get(view["status"], ABOUT_CLOSING["legacy"])
    return {**ABOUT, "body": ABOUT["body"] + [closing.format(**view)]}


app = Flask(__name__)
_P = None
_ERR = None


def get_predictor():
    global _P, _ERR
    if _P is None and _ERR is None:
        try:
            _P = Predictor()
        except Exception as e:  # surfaced in the UI, not swallowed
            _ERR = str(e)
    return _P


@app.route("/")
def index():
    p = get_predictor()
    view = claim_view(p.claim) if p else None
    return render_template(
        "index.html", error=_ERR,
        pitchers=p.pitcher_ids()[:800] if p else [],
        batters=p.batter_ids()[:800] if p else [],
        cv=view,
        data_through=(p.smeta.get("data_through") if p else None),
        outcomes=["ball", "called_strike", "whiff", "foul", "in_play"],
        pitches=p.classes if p else [],
        pitch_names=PITCH_NAMES,
        hard_family=sorted(FASTBALL_FAMILY),
        outcome_names=OUTCOME_NAMES,
        about=about_for(view))


@app.route("/api/arsenal/<int:pitcher>")
def arsenal(pitcher):
    p = get_predictor()
    if not p:
        return jsonify({"error": _ERR}), 503
    return jsonify({"arsenal": p.arsenal(pitcher)})


@app.route("/api/players")
def players():
    p = get_predictor()
    if not p:
        return jsonify({"error": _ERR}), 503
    role = request.args.get("role", "pitcher")
    if role not in ("pitcher", "batter"):
        return jsonify({"error": "role must be pitcher or batter"}), 400
    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 100))
    except ValueError:
        limit = 20
    return jsonify({"players": p.players(role, request.args.get("q", ""), limit),
                    "named": p.names_loaded()})


@app.route("/healthz")
def healthz():
    """Render's health check. 200 only once the model can actually serve —
    a process that is up but cannot predict is not healthy."""
    p = get_predictor()
    if not p:
        return jsonify({"status": "unavailable", "error": _ERR}), 503
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    p = get_predictor()
    if not p:
        return jsonify({"error": _ERR}), 503
    s = request.get_json(force=True)
    try:
        return jsonify(p.predict(s))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True)
