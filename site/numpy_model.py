"""
NextPitchAI v6 — serve-time forward pass, NumPy only
=====================================================
Runs the trained pitch-type model without TensorFlow. Importing TF costs
300-500MB RSS and the free host has 512MB; this module needs NumPy and
the two files site/export_model.py writes, and the whole app then fits in
roughly 150MB.

    from numpy_model import NumpyModel
    m = NumpyModel("site/serving/model_weights.npz",
                   "site/serving/model_arch.json")
    probs = m.predict({"seq": ..., "ctx": ..., "pitcher_id": ..., ...})

`predict` returns (B, n_pitch) softmax rows — the arsenal-masked
probabilities, identical to what Keras returns for the same inputs
(site/test_numpy_parity.py asserts max |diff| < 1e-4 on a fixed batch).

IMPORT NOTHING HEAVY HERE. No tensorflow, no torch, no sklearn, no
scipy. That is the entire point of the file, and the parity test checks
sys.modules to make sure a future import does not creep back in.

The arithmetic below is not free-form. A subtly wrong LSTM does not
crash; it returns confident, plausible, wrong pitch calls. The details
that must stay exactly as they are:

  * Keras packs the LSTM kernels in gate order i, f, c, o.
  * unit_forget_bias is already baked into the SAVED bias; do not add 1
    to the forget gate again.
  * Bidirectional runs the backward branch over the reversed time axis
    and, with return_sequences, flips the result back BEFORE merging, so
    index t means time t in both halves. Skip the flip and the two
    halves are misaligned by a reversal and every later number is quietly
    wrong.
  * BatchNormalization at inference uses the moving statistics and the
    layer's own epsilon, which the exporter records (do not assume 1e-3).
  * Dropout is the identity.
"""

import json
from pathlib import Path

import numpy as np

# Must match site/export_model.py. A mismatch means the serving bundle was
# written by a different exporter than this reader expects.
ARCH_FORMAT = "nextpitch-numpy-arch"
ARCH_VERSION = 1


# =========================
# Activations
# =========================
def _sigmoid(x):
    # Split by sign: exp(+large) overflows to inf and the naive form then
    # returns nan rather than saturating at 1.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "sigmoid": _sigmoid,
    "softmax": _softmax,
}


# =========================
# Layers
# =========================
def _lstm(x, kernel, recurrent, bias, units, go_backwards, return_sequences,
          activation, recurrent_activation):
    """One Keras LSTM direction. x is (B, T, F)."""
    n_batch, n_steps, _ = x.shape
    # The input projection does not depend on h, so it is one matmul for
    # the whole sequence; only the recurrent term has to run in a loop.
    z_in = x @ kernel
    if bias is not None:
        z_in = z_in + bias
    h = np.zeros((n_batch, units), np.float32)
    c = np.zeros((n_batch, units), np.float32)

    steps = range(n_steps - 1, -1, -1) if go_backwards else range(n_steps)
    outs = []
    for t in steps:
        z = z_in[:, t, :] + h @ recurrent
        i = recurrent_activation(z[:, :units])
        f = recurrent_activation(z[:, units:2 * units])
        g = activation(z[:, 2 * units:3 * units])
        o = recurrent_activation(z[:, 3 * units:])
        c = f * c + i * g
        h = o * activation(c)
        if return_sequences:
            outs.append(h)
    if not return_sequences:
        return h
    return np.stack(outs, axis=1)


def _bidirectional(node, weights, x):
    halves = []
    for tag in ("forward", "backward"):
        spec = node[tag]
        y = _lstm(
            x,
            weights[f"{node['name']}.{tag}.kernel"],
            weights[f"{node['name']}.{tag}.recurrent_kernel"],
            weights.get(f"{node['name']}.{tag}.bias"),
            spec["units"],
            spec["go_backwards"],
            spec["return_sequences"],
            ACTIVATIONS[spec["activation"]],
            ACTIVATIONS[spec["recurrent_activation"]],
        )
        if spec["go_backwards"] and spec["return_sequences"]:
            # Keras' Bidirectional re-reverses the backward outputs so both
            # halves are in forward time order before the merge.
            y = y[:, ::-1, :]
        halves.append(y)
    return np.concatenate(halves, axis=-1)


# =========================
# Model
# =========================
class NumpyModel:
    def __init__(self, npz_path, arch_path):
        arch_path, npz_path = Path(arch_path), Path(npz_path)
        self.arch = json.loads(arch_path.read_text())
        if self.arch.get("format") != ARCH_FORMAT:
            raise ValueError(f"{arch_path} is not a {ARCH_FORMAT} file")
        if int(self.arch.get("version", -1)) != ARCH_VERSION:
            raise ValueError(
                f"{arch_path} is arch version {self.arch.get('version')}, this "
                f"reader speaks {ARCH_VERSION} — re-run site/export_model.py")

        with np.load(npz_path) as z:
            self.weights = {k: np.asarray(z[k], np.float32) for k in z.files}

        self.nodes = self.arch["nodes"]
        self.input_specs = self.arch["inputs"]
        self.outputs = self.arch["outputs"]
        self.primary_output = self.arch["primary_output"]

        self._alias = {}
        for spec in self.input_specs:
            self._alias[spec["name"]] = spec["name"]
            self._alias[spec["alias"]] = spec["name"]
        self._int_inputs = {n["inputs"][0] for n in self.nodes
                            if n["op"] == "embedding"}
        self._embed_limit = {n["inputs"][0]: n["input_dim"] for n in self.nodes
                             if n["op"] == "embedding"}

    # ---------- input handling ----------
    def _prepare(self, inputs):
        supplied = {}
        for key, value in inputs.items():
            name = self._alias.get(key)
            if name is None:
                raise KeyError(
                    f"unknown input {key!r}; expected any of "
                    f"{sorted(self._alias)}")
            supplied[name] = value

        vals, sizes = {}, set()
        for spec in self.input_specs:
            name, shape = spec["name"], spec["shape"]
            if name not in supplied:
                raise KeyError(f"missing input {name!r} "
                               f"(alias {spec['alias']!r})")
            arr = np.asarray(supplied[name])
            if name in self._int_inputs:
                # Keras takes these as (B, 1); (B,) is the friendlier
                # serving shape, so accept both and normalise.
                arr = arr.reshape(arr.shape[0], 1).astype(np.int64, copy=False)
                limit = self._embed_limit[name]
                if arr.size and (arr.min() < 0 or arr.max() >= limit):
                    raise ValueError(
                        f"{name}: id out of range for an embedding of "
                        f"{limit} rows (min {arr.min()}, max {arr.max()}) — "
                        "map unknown players to a known row before calling")
            else:
                arr = arr.astype(np.float32, copy=False)
                want = tuple(d for d in shape[1:])
                if arr.shape[1:] != want:
                    raise ValueError(
                        f"{name}: got shape {arr.shape}, model wants "
                        f"(batch, {', '.join(str(d) for d in want)})")
            vals[name] = arr
            sizes.add(arr.shape[0])
        if len(sizes) != 1:
            raise ValueError(f"inputs disagree on batch size: {sorted(sizes)}")
        return vals

    # ---------- forward ----------
    def predict_all(self, inputs):
        """Every model head, as {output_name: (B, n) array}."""
        vals = self._prepare(inputs)
        w = self.weights

        for node in self.nodes:
            op, name = node["op"], node["name"]
            if op == "input":
                continue                       # already in vals
            if op == "elementwise":
                a, b = (vals[s["node"]] if "node" in s else np.float32(s["const"])
                        for s in node["args"])
                vals[name] = {"add": np.add, "subtract": np.subtract,
                              "multiply": np.multiply}[node["fn"]](a, b)
                continue

            x = [vals[i] for i in node["inputs"]]
            if op == "dense":
                y = x[0] @ w[f"{name}.kernel"]
                if node["use_bias"]:
                    y = y + w[f"{name}.bias"]
                vals[name] = ACTIVATIONS[node["activation"]](y)
            elif op == "batch_normalization":
                inv = 1.0 / np.sqrt(w[f"{name}.moving_variance"] + node["epsilon"])
                if node["scale"]:
                    inv = inv * w[f"{name}.gamma"]
                y = (x[0] - w[f"{name}.moving_mean"]) * inv
                vals[name] = y + w[f"{name}.beta"] if node["center"] else y
            elif op == "dropout":
                vals[name] = x[0]              # identity at inference
            elif op == "embedding":
                vals[name] = w[f"{name}.embeddings"][x[0]]
            elif op == "flatten":
                vals[name] = x[0].reshape(x[0].shape[0], -1)
            elif op == "concatenate":
                vals[name] = np.concatenate(x, axis=node["axis"])
            elif op == "activation":
                vals[name] = ACTIVATIONS[node["activation"]](x[0])
            elif op == "bidirectional_lstm":
                vals[name] = _bidirectional(node, w, x[0])
            else:
                raise ValueError(f"{name}: no NumPy implementation for op {op!r}")

        return {nm: np.asarray(vals[src], np.float32)
                for nm, src in self.outputs.items()}

    def predict(self, inputs):
        """(B, n_pitch) arsenal-masked softmax; rows sum to 1."""
        return self.predict_all(inputs)[self.primary_output]
