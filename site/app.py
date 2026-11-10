"""
NextPitchAI v6 — Flask app
==========================
    python site/build_serving_artifacts.py   # once, after training
    python site/app.py                       # http://127.0.0.1:5000

The product is a SELECTIVE OVERLAY. It stays silent unless the read
clears the confidence threshold the gate measured, because a wrong
commit costs a hitter more than no advice. Every accuracy figure shown
is read from data_v5/product_claim_v5.json, written by
analyze_actionability.py on held-out rows — the app never hardcodes a
number, so it cannot keep quoting a figure a retrain invalidated.
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

        "What is here is what survived. It learns from about 2.9 million "
        "pitches thrown between 2022 and 2024 and is measured on all of 2025, "
        "a season held out entirely. It speaks on roughly a third of pitches "
        "and beats a count-split scouting report by a few points there, and "
        "stays quiet the rest of the time. That is a smaller claim than I set "
        "out hoping for. It is the one I can defend.",
    ],
    "links": [
        {"label": "GitHub", "url": "https://github.com/nabeelh0217/NextPitchAIv2"},
        # {"label": "LinkedIn", "url": "https://www.linkedin.com/in/..."},
        # {"label": "Email", "url": "mailto:you@example.com"},
    ],
}

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
    return render_template(
        "index.html", error=_ERR,
        pitchers=p.pitcher_ids()[:800] if p else [],
        batters=p.batter_ids()[:800] if p else [],
        claim=p.claim if p else None,
        outcomes=["ball", "called_strike", "whiff", "foul", "in_play"],
        pitches=p.classes if p else [],
        pitch_names=PITCH_NAMES,
        hard_family=sorted(FASTBALL_FAMILY),
        outcome_names=OUTCOME_NAMES,
        about=ABOUT)


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
