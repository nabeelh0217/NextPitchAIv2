"""
NextPitchAI v6 — serve-time prediction
======================================
Turns a game situation into the one call a hitter can act on:
GEAR UP (fastball family) / STAY BACK (offspeed), or NO READ.

The single biggest risk here is the serve-time feature vector drifting
from the one the model trained on. Three defences:

  * the game-state block is built by pitch_features.build_context_features,
    the same function 02_preprocess.py uses at training time — imported,
    never reimplemented;
  * the assembled column order is checked against `ctx_feature_names` in
    meta_v5.json, and a mismatch raises rather than predicting quietly;
  * the arsenal mask comes from the table the training run saved.

Sequence timesteps carry the pitch types and outcomes the caller
supplies. Physics (velocity, spin, movement) is filled with that
pitcher's median for that pitch type, because a user knows what was
thrown but not how hard. Missing timesteps stay zero in SCALED space,
which is how 02_preprocess pads a pitcher's first pitches of a game.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from numpy_model import NumpyModel  # noqa: E402
from pitch_features import build_context_features  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SERVING = Path(__file__).resolve().parent / "serving"
DATA_DIR = BASE_DIR / "data_v5"

FASTBALL_FAMILY = {"FF", "SI", "FC"}
PHYS_COLS = ["release_speed", "plate_x", "plate_z",
             "release_spin_rate", "pfx_x", "pfx_z"]
# Outcome one-hot order, must match OUTCOME_CLASSES in 02_preprocess.py.
OUTCOMES = ["ball", "called_strike", "whiff", "foul", "in_play"]


class NotBuilt(RuntimeError):
    pass


class Predictor:
    def __init__(self):
        if not SERVING.exists():
            raise NotBuilt(
                "site/serving/ is missing — run "
                "`python site/build_serving_artifacts.py` after training.")

        self.meta = json.loads((SERVING / "meta_v5.json").read_text())
        self.smeta = json.loads((SERVING / "serving_meta.json").read_text())
        self.classes = self.meta["pitch_classes"]
        self.zones = self.meta["zone_classes"]
        self.ctx_names = self.meta["ctx_feature_names"]
        self.seq_len = self.meta["seq_len"]
        self.n_pitch = len(self.classes)

        # Plain arrays, not pickled StandardScalers: unpickling one drags
        # scikit-learn (~100MB) into the deploy image for two vectors of
        # floats, and the transform is (x - mean) / scale regardless.
        sc = np.load(SERVING / "scalers.npz")
        self.ctx_mean, self.ctx_scale = sc["ctx_mean"], sc["ctx_scale"]
        self.seq_mean, self.seq_scale = sc["seq_mean"], sc["seq_scale"]
        self.pit = pd.read_parquet(SERVING / "pitchers.parquet").set_index("pitcher_id")
        self.bat = pd.read_parquet(SERVING / "batters.parquet").set_index("batter_id")
        self.phys = pd.read_parquet(SERVING / "physics.parquet")
        self.phys_idx = {(int(r.pitcher), r.canon): r for r in self.phys.itertuples()}
        self.league_phys = self.smeta["league_physics"]

        mt = SERVING / "matchup_table_v5.parquet"
        self.matchup = pd.read_parquet(mt) if mt.exists() else None
        if self.matchup is not None:
            self.matchup = self.matchup.set_index(["pitcher", "batter"])

        # The claim the site is allowed to make, straight from the gate.
        claim_path = DATA_DIR / "product_claim_v5.json"
        self.claim = (json.loads(claim_path.read_text())
                      if claim_path.exists() else None)
        rec = (self.claim or {}).get("recommended") or {}
        self.temperature = float((self.claim or {}).get("temperature", 1.0))
        self.threshold = float(rec.get("threshold", 0.65))

        # NumPy, not TensorFlow. Importing TF costs ~660MB RSS against a
        # 512MB box; the exported weights run the same graph in ~30MB and
        # export_model.py refuses to write them unless they match Keras to
        # better than 1e-4.
        w, a = SERVING / "model_weights.npz", SERVING / "model_arch.json"
        if not (w.exists() and a.exists()):
            raise NotBuilt(
                f"{w.name} / {a.name} missing — run "
                f"`python site/export_model.py` (needs TensorFlow, so run it "
                f"locally, not on the server).")
        self.model = NumpyModel(w, a)

    # ---------- player search ----------
    def _index(self, table, role):
        """(id, name, hand, n_pitches) rows, commonest first."""
        try:
            from player_names import resolve
            names = resolve([int(i) for i in table.index],
                            SERVING / "player_names.json")
        except Exception:
            names = {}
        label = "Pitcher" if role == "pitcher" else "Batter"
        rows = []
        for pid_, r in table.iterrows():
            info = names.get(str(int(pid_))) or {}
            rows.append({
                "id": int(pid_),
                # Falling back to the id keeps the picker usable before
                # anyone has run the name fetch; it never renders blank.
                "name": info.get("name") or f"{label} {int(pid_)}",
                "named": bool(info.get("name")),
                # A pitcher's throwing hand and a batter's side are
                # different fields; the same id can appear in both tables.
                "hand": (info.get("hand") if role == "pitcher"
                         else info.get("bats")) or (
                    "R" if int(r.get("hand", 0)) == 0 else "L"),
                "n": int(r.get("n_pitches", 0)),
            })
        rows.sort(key=lambda d: -d["n"])
        return rows

    def players(self, role="pitcher", q="", limit=20):
        if not hasattr(self, "_pidx"):
            self._pidx = {"pitcher": self._index(self.pit, "pitcher"),
                          "batter": self._index(self.bat, "batter")}
        rows = self._pidx.get(role, [])
        q = (q or "").strip().lower()
        if q:
            # Prefix hits first: typing "deg" should surface deGrom above
            # someone merely containing the letters.
            starts = [r for r in rows if r["name"].lower().startswith(q)]
            subs = [r for r in rows
                    if q in r["name"].lower() and r not in starts]
            rows = starts + subs
        return rows[:limit]

    def names_loaded(self):
        """True only if real names are actually resolving. The cache file
        merely existing is not enough — an empty or stale one still leaves
        the picker showing numeric ids, and the UI prompts on this."""
        if not hasattr(self, "_pidx"):
            self.players("pitcher", "", 1)
        return any(r["named"] for r in self._pidx.get("pitcher", [])[:50])

    # ---------- lookups ----------
    def pitcher_ids(self):
        return [int(i) for i in self.pit.index]

    def batter_ids(self):
        return [int(i) for i in self.bat.index]

    def arsenal(self, pitcher):
        """Pitch types this pitcher actually throws, commonest first."""
        if pitcher not in self.pit.index:
            return list(self.classes)
        row = self.pit.loc[pitcher]
        got = [(c, float(row[f"arsenal_{c}"])) for c in self.classes
               if row.get(f"mask_{c}", 1.0) > 0]
        return [c for c, _ in sorted(got, key=lambda kv: -kv[1])]

    # ---------- feature assembly ----------
    def _state_block(self, s):
        """Game state, built by the preprocessing module itself."""
        df = pd.DataFrame([{
            "balls": s["balls"], "strikes": s["strikes"],
            "outs_when_up": s["outs"], "inning": s["inning"],
            "inning_topbot": "Top" if s.get("top", True) else "Bot",
            "fld_score": s.get("fld_score", 0), "bat_score": s.get("bat_score", 0),
            "on_1b": 1.0 if s.get("on_1b") else np.nan,
            "on_2b": 1.0 if s.get("on_2b") else np.nan,
            "on_3b": 1.0 if s.get("on_3b") else np.nan,
            "pitch_number": s.get("pitch_of_ab", 1),
            "game_pk": 1, "pitcher": s["pitcher"], "batter": s["batter"],
            "at_bat_number": 1, "game_date": pd.Timestamp("2025-06-01"),
            "n_thruorder_pitcher": float(s.get("times_through_order", 1)),
            "pitcher_days_since_prev_game": float(s.get("days_rest", 5)),
        }])
        ctx, names = build_context_features(df)
        if names != self.ctx_names[:len(names)]:
            raise RuntimeError(
                "game-state feature names no longer match meta_v5.json; "
                "data_v5/ and 02_preprocess.py are out of step — rebuild.")
        ctx = ctx.astype(np.float64)
        # cumcount() over a one-row frame cannot know the real workload.
        ctx[0, names.index("pitch_count_game")] = min(
            float(s.get("pitch_count_game", 1)), 120.0)
        return ctx, names

    def _priors(self, pitcher, batter):
        prow = self.pit.loc[pitcher] if pitcher in self.pit.index else None
        brow = self.bat.loc[batter] if batter in self.bat.index else None
        uni = 1.0 / self.n_pitch

        arsenal = np.array([float(prow[f"arsenal_{c}"]) if prow is not None else uni
                            for c in self.classes])
        seen = np.array([float(brow[f"seen_{c}"]) if brow is not None else uni
                         for c in self.classes])
        whiff = np.array([float(brow[f"whiff_{c}"]) if brow is not None else 0.25
                          for c in self.classes])
        zp = np.array([float(prow[f"zoneprior_{z}"]) if prow is not None
                       else 1.0 / len(self.zones) for z in self.zones])
        zs = np.array([float(brow[f"zoneseen_{z}"]) if brow is not None
                       else 1.0 / len(self.zones) for z in self.zones])

        # Matchup history; with none, fall back to the pitcher's own mix,
        # which is what an expanding prior holds before any meetings.
        fam, mdist = 0.0, arsenal.copy()
        if self.matchup is not None and (pitcher, batter) in self.matchup.index:
            row = self.matchup.loc[(pitcher, batter)]
            counts = np.array([float(row.get(c, 0.0)) for c in self.classes])
            if counts.sum() > 0:
                mdist, fam = counts / counts.sum(), float(counts.sum())
        return arsenal, mdist, fam, seen, whiff, zp, zs

    def _sequence(self, pitcher, recent):
        """(1, seq_len, F+1) — right-aligned, zero-padded in scaled space."""
        n_out = len(OUTCOMES)
        base = self.n_pitch + n_out
        cont_start = base
        feats = np.zeros((max(len(recent), 1), base + len(PHYS_COLS)), np.float32)
        for i, p in enumerate(recent):
            if p.get("pitch") in self.classes:
                feats[i, self.classes.index(p["pitch"])] = 1.0
            if p.get("outcome") in OUTCOMES:
                feats[i, self.n_pitch + OUTCOMES.index(p["outcome"])] = 1.0
            rec = self.phys_idx.get((int(pitcher), p.get("pitch")))
            for j, col in enumerate(PHYS_COLS):
                v = getattr(rec, col, None) if rec is not None else None
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = self.league_phys.get(col, 0.0)
                feats[i, cont_start + j] = float(v)
        if recent:
            feats[:len(recent), cont_start:] = (
                (feats[:len(recent), cont_start:] - self.seq_mean)
                / self.seq_scale)

        seq = np.zeros((1, self.seq_len, feats.shape[1] + 1), np.float32)
        take = recent[-self.seq_len:] if recent else []
        if take:
            block = feats[len(recent) - len(take):len(recent)]
            start = self.seq_len - len(block)
            seq[0, start:, :feats.shape[1]] = block
            # same-at-bat flag
            seq[0, start:, -1] = np.array(
                [1.0 if p.get("same_ab") else 0.0 for p in take], np.float32)
        return seq

    # ---------- the call ----------
    def predict(self, s):
        pitcher, batter = int(s["pitcher"]), int(s["batter"])
        state, names = self._state_block(s)
        ars, mdist, fam, seen, whiff, zp, zs = self._priors(pitcher, batter)
        ctx = np.concatenate([
            state[0], ars, mdist, [fam], seen, whiff, zp, zs]).astype(np.float64)
        if len(ctx) != len(self.ctx_names):
            raise RuntimeError(
                f"assembled {len(ctx)} context features, model expects "
                f"{len(self.ctx_names)} — serving bundle is stale.")
        ctx = ((ctx[None, :] - self.ctx_mean) / self.ctx_scale).astype(np.float32)

        prow = self.pit.loc[pitcher] if pitcher in self.pit.index else None
        mask = np.array([[float(prow[f"mask_{c}"]) if prow is not None else 1.0
                          for c in self.classes]], np.float32)

        brow = self.bat.loc[batter] if batter in self.bat.index else None
        # Encoded ids come off the row, never from a separate map — the
        # two keyings drifting apart is how every lookup silently became
        # "unknown player" while the site still returned a confident call.
        pi = np.array([int(prow["enc"]) if prow is not None else 0], np.int32)
        bi = np.array([int(brow["enc"]) if brow is not None else 0], np.int32)
        ph = np.array([int(prow["hand"]) if prow is not None else 0], np.int32)
        bh = np.array([int(brow["hand"]) if brow is not None else 0], np.int32)
        park = np.array([s.get("park_id", 0)], np.int32)
        catcher = np.array([s.get("catcher_id", 0)], np.int32)

        probs = self.model.predict({
            "seq": self._sequence(pitcher, s.get("recent", [])), "ctx": ctx,
            "pitcher_id": pi, "batter_id": bi,
            "pitcher_hand": ph, "batter_hand": bh,
            "park_id": park, "catcher_id": catcher,
            "arsenal_mask": mask})[0]

        # Same temperature the gate measured the claim at.
        if self.temperature and self.temperature != 1.0:
            lg = np.log(np.clip(probs, 1e-12, 1.0)) / self.temperature
            lg -= lg.max()
            probs = np.exp(lg) / np.exp(lg).sum()

        hard_i = [i for i, c in enumerate(self.classes) if c in FASTBALL_FAMILY]
        p_hard = float(probs[hard_i].sum())
        conf = max(p_hard, 1.0 - p_hard)
        gear_up = p_hard >= 0.5

        # Name the likeliest pitch WITHIN the family being advised —
        # "sit soft, likeliest fastball" is incoherent advice.
        fam_i = hard_i if gear_up else [i for i in range(self.n_pitch)
                                        if i not in hard_i]
        best = max(fam_i, key=lambda i: probs[i])

        order = np.argsort(probs)[::-1]
        return {
            "known_pitcher": prow is not None,
            "known_batter": brow is not None,
            "speaks": conf >= self.threshold,
            "call": "GEAR UP" if gear_up else "STAY BACK",
            "family": "fastball" if gear_up else "offspeed",
            "confidence": conf,
            "threshold": self.threshold,
            "likeliest": self.classes[best],
            "likeliest_p": float(probs[best]),
            "distribution": [(self.classes[i], float(probs[i])) for i in order
                             if probs[i] >= 0.005],
            "claim": self.claim,
        }
