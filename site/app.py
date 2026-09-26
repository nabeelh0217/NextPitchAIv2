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
from flask import Flask, jsonify, render_template, request

from predictor import NotBuilt, Predictor

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
        pitches=p.classes if p else [])


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
