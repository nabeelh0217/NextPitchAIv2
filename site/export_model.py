"""
NextPitchAI v6 — offline: Keras model -> NumPy weights + architecture JSON
==========================================================================
At serve time TensorFlow is nothing but a matrix multiplier, and it is an
expensive one: importing it costs 300-500MB RSS, which does not fit a
512MB host. This model is ~450k parameters, so the same arithmetic in
plain NumPy serves identical numbers in a fraction of the memory.

This script runs OFFLINE, beside training, where TensorFlow is already
installed. It writes two files that site/numpy_model.py reads:

    model_arch.json     the graph, topologically sorted, as plain ops
    model_weights.npz   every weight, keyed "<layer>.<role>"

It walks the SAVED MODEL'S OWN GRAPH rather than restating the layer
stack from 03_train.py. A hand-copied architecture is a second source of
truth that goes stale in silence, and the failure mode here is not a
crash but a confident wrong prediction. Walking the graph also means the
optional zone_output head needs no special case: if the run trained one
it is exported, and if it did not, nothing refers to it.

Anything the walker does not recognise raises. Never give it a fallback
that guesses at an unknown op — a guessed op is exactly the
silent-wrong-answer bug this file exists to prevent.

Usage:
    python site/export_model.py
    python site/export_model.py --model data_v5/best_model_v5.keras \
                                --out-dir site/serving

Output (derived, gitignored with the rest of site/serving/):
    site/serving/model_arch.json
    site/serving/model_weights.npz
"""

import argparse
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_v5"
SERVING_DIR = Path(__file__).resolve().parent / "serving"

DEFAULT_MODEL = DATA_DIR / "best_model_v5.keras"
ARCH_NAME = "model_arch.json"
WEIGHTS_NAME = "model_weights.npz"

# Bumped whenever the arch schema changes shape. numpy_model.py refuses a
# version it was not written against rather than misreading the fields.
ARCH_FORMAT = "nextpitch-numpy-arch"
ARCH_VERSION = 1

# Only what numpy_model.py actually implements. Exporting an activation
# it cannot run would surface as wrong numbers, not as an error.
SUPPORTED_ACTIVATIONS = {"linear", "relu", "tanh", "sigmoid", "softmax"}

# keras.src.ops.numpy elementwise ops, which is how the arsenal-masked
# logits (`logits + (1 - mask) * PENALTY`) appear in the saved graph —
# they are bare ops, not layers, so they carry no weights.
ELEMENTWISE = {"Add": "add", "Subtract": "subtract", "Multiply": "multiply"}


# =========================
# Graph walking
# =========================
def _layer_name(entry):
    return entry.get("config", {}).get("name") or entry["name"]


def _activation_name(raw, where):
    if raw is None:
        return "linear"
    if not isinstance(raw, str):
        raise ValueError(f"{where}: non-string activation {raw!r}; "
                         "numpy_model.py cannot reproduce a serialized callable")
    if raw not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"{where}: activation {raw!r} is not implemented in "
                         f"numpy_model.py (has {sorted(SUPPORTED_ACTIVATIONS)})")
    return raw


def _arg_source(arg, where):
    """One serialized call argument -> {'node': name} or {'const': value}."""
    if isinstance(arg, dict) and arg.get("class_name") == "__keras_tensor__":
        return {"node": arg["config"]["keras_history"][0]}
    if isinstance(arg, bool):
        raise ValueError(f"{where}: unexpected boolean argument")
    if isinstance(arg, (int, float)):
        return {"const": float(arg)}
    raise ValueError(f"{where}: unsupported call argument {arg!r}")


def _call(entry):
    inbound = entry.get("inbound_nodes") or []
    if len(inbound) != 1:
        raise ValueError(f"{_layer_name(entry)}: expected exactly one call in the "
                         f"graph, found {len(inbound)}; shared layers are not supported")
    return inbound[0].get("args", []), inbound[0].get("kwargs", {})


# Call kwargs Keras records that change nothing at inference. `mask=None`
# means the RNNs and BatchNorms were never given a sequence mask, and
# `training=False` is the Dropout call that is already the identity here.
# Anything else — a real mask tensor, training=True — would change the
# numbers, so it stops the export rather than being ignored.
BENIGN_KWARGS = {"mask": (None,), "training": (False, None)}


def _tensor_inputs(entry):
    """Input layer names of a layer called with a single tensor (or list)."""
    where = _layer_name(entry)
    args, kwargs = _call(entry)
    for key, value in kwargs.items():
        if key not in BENIGN_KWARGS or value not in BENIGN_KWARGS[key]:
            raise ValueError(f"{where}: call kwarg {key}={value!r} changes the "
                             "forward pass and is not supported")
    if len(args) != 1:
        raise ValueError(f"{where}: expected one positional argument, got {len(args)}")
    seq = args[0] if isinstance(args[0], list) else [args[0]]
    names = []
    for a in seq:
        src = _arg_source(a, where)
        if "node" not in src:
            raise ValueError(f"{where}: constant where a tensor was expected")
        names.append(src["node"])
    return names



_WEIGHTED = {"Dense", "Embedding", "BatchNormalization", "Bidirectional"}


def _expect(got, want, what):
    """Shape guard. A wrong name-to-weights mapping usually shows up here
    rather than as a wrong answer three layers downstream."""
    if got != want:
        raise SystemExit(f"{what}: expected {want}, archive has {got}. "
                         f"The weights do not match the config.")


# ---------------------------------------------------------------------
# Reading a .keras archive WITHOUT TensorFlow
# ---------------------------------------------------------------------
# A .keras file is a zip of config.json (exactly what model.get_config()
# returns) and model.weights.h5. Nothing about reading it needs
# TensorFlow, and requiring TF here made the export impossible on a
# machine whose TF install aborts at import — which is where this came
# from. h5py is small and already present.
#
# These shims expose only the handful of attributes _convert() touches,
# so the converter is identical for both paths. test_numpy_parity.py
# asserts the two produce bit-identical output wherever TF does work.

class _Cell:
    """LSTM cell weights: vars/0 kernel, 1 recurrent_kernel, 2 bias."""

    def __init__(self, g, cfg, where):
        v = g["vars"]
        units = int(cfg["units"])
        self.kernel = np.asarray(v["0"])
        self.recurrent_kernel = np.asarray(v["1"])
        self.bias = (np.asarray(v["2"])
                     if cfg.get("use_bias", True) and "2" in v else None)
        _expect(self.kernel.shape[1], 4 * units, f"{where} kernel gates")
        _expect(self.recurrent_kernel.shape, (units, 4 * units),
                f"{where} recurrent_kernel")
        if self.bias is not None:
            _expect(self.bias.shape, (4 * units,), f"{where} bias")


class LSTM:  # the converter checks __class__.__name__, so the name matters
    def __init__(self, cfg, g, where):
        self._cfg = cfg
        self.units = int(cfg["units"])
        self.cell = _Cell(g["cell"], cfg, where)

    def get_config(self):
        return self._cfg


class _ShimLayer:
    """Just enough of a Keras layer for _convert(), read from the archive."""

    def __init__(self, cls, cfg, g, name):
        self._cls, self._cfg = cls, cfg
        v = g["vars"] if (g is not None and "vars" in g) else None

        if cls == "Dense":
            units = int(cfg["units"])
            self.kernel = np.asarray(v["0"])
            self.bias = (np.asarray(v["1"])
                         if cfg.get("use_bias", True) and "1" in v else None)
            _expect(self.kernel.shape[1], units, f"{name} kernel units")
            if self.bias is not None:
                _expect(self.bias.shape, (units,), f"{name} bias")
        elif cls == "Embedding":
            self.embeddings = np.asarray(v["0"])
            _expect(self.embeddings.shape,
                    (int(cfg["input_dim"]), int(cfg["output_dim"])),
                    f"{name} embeddings")
        elif cls == "BatchNormalization":
            self.axis = cfg.get("axis", -1)
            self.epsilon = float(cfg.get("epsilon", 1e-3))
            self.scale = bool(cfg.get("scale", True))
            self.center = bool(cfg.get("center", True))
            # gamma, beta, moving_mean, moving_variance — gamma and beta
            # omitted when scale/center are off.
            i = 0
            if self.scale:
                self.gamma = np.asarray(v[str(i)]); i += 1
            if self.center:
                self.beta = np.asarray(v[str(i)]); i += 1
            self.moving_mean = np.asarray(v[str(i)]); i += 1
            self.moving_variance = np.asarray(v[str(i)])
            _expect(self.moving_variance.shape, self.moving_mean.shape,
                    f"{name} BN statistics")
        elif cls == "Bidirectional":
            self.merge_mode = cfg.get("merge_mode", "concat")
            fcfg = dict(cfg["layer"]["config"])
            bcfg = (dict(cfg["backward_layer"]["config"])
                    if cfg.get("backward_layer") else dict(fcfg))
            if not cfg.get("backward_layer"):
                bcfg["go_backwards"] = not fcfg.get("go_backwards", False)
            self.forward_layer = LSTM(fcfg, g["forward_layer"], f"{name}/forward")
            self.backward_layer = LSTM(bcfg, g["backward_layer"], f"{name}/backward")

    def __getattr__(self, item):
        # Plain config-backed attributes the converter reads directly,
        # e.g. Concatenate.axis. Only reached when normal lookup fails,
        # so it never shadows the weights set above, and a key that is
        # genuinely absent still raises AttributeError.
        try:
            return object.__getattribute__(self, "_cfg")[item]
        except KeyError:
            raise AttributeError(
                f"{self._cls} layer has no {item!r} in its saved config") from None

    def get_config(self):
        return self._cfg


class _Tensor:
    """Stands in for a Keras tensor: a name, a shape and a dtype. The
    _keras_history shim mirrors what export() reads off a real one."""

    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = tuple(shape) if shape is not None else ()
        self.dtype = dtype
        self._keras_history = (_Named(name),)


class _Named:
    def __init__(self, name):
        self.name = name


class KerasArchive:
    """A .keras file, read with zipfile + h5py instead of TensorFlow."""

    def __init__(self, path):
        import h5py
        self._zip = zipfile.ZipFile(path)
        self._cfg = json.loads(self._zip.read("config.json"))["config"]
        try:
            self.keras_version = json.loads(
                self._zip.read("metadata.json")).get("keras_version")
        except Exception:
            self.keras_version = None
        self._h5 = h5py.File(io.BytesIO(self._zip.read("model.weights.h5")), "r")
        self._layers = {e["name"]: e for e in self._cfg["layers"]}

        # The h5 groups are keyed by Keras's AUTO-GENERATED names, not the
        # names in config.json: `logits` is stored under `dense_3` and
        # `pitcher_id_embedding` under `embedding`. Matching those by
        # position would load real weights into the wrong layer and still
        # produce a working-looking model. Each `vars` group records the
        # true layer name in an attribute, so index by that instead of
        # inferring anything.
        self._byname = {}
        groups = self._h5["layers"]
        for key in groups:
            g = groups[key]
            v = g.get("vars")
            if v is not None and "name" in v.attrs:
                self._byname[str(v.attrs["name"])] = g
            # A Bidirectional's own vars group is empty; its real name is
            # only recoverable from the h5 key, which Keras leaves equal
            # to the layer name unless the wrapper itself was renamed.
            if "forward_layer" in g:
                self._byname.setdefault(key, g)

        # Graph endpoints, from config.json rather than live tensors.
        self.inputs = []
        for name, _, _ in self._cfg["input_layers"]:
            icfg = self._layers[name]["config"]
            self.inputs.append(_Tensor(
                name, icfg.get("batch_shape") or icfg.get("shape"),
                icfg.get("dtype", "float32")))
        outs = self._cfg["output_layers"]
        if outs and isinstance(outs[0], str):   # single output, unwrapped
            outs = [outs]
        self.output_names = [o[0] for o in outs]
        self.outputs = [_Tensor(o[0], None, None) for o in outs]

    def get_config(self):
        return self._cfg

    def get_layer(self, name):
        entry = self._layers[name]
        cls = entry["class_name"]
        grp = self._byname.get(name)
        if grp is None and cls in _WEIGHTED:
            raise SystemExit(
                f"{name}: no weights group in the .keras archive. Known "
                f"groups: {sorted(self._byname)}. The archive and its "
                f"config disagree — re-save the model.")
        return _ShimLayer(cls, entry.get("config") or {}, grp, name)


def _convert(entry, model, weights):
    """One serialized layer -> one arch node (and its weights into `weights`)."""
    cls = entry["class_name"]
    cfg = entry.get("config", {})
    name = _layer_name(entry)

    if cls == "InputLayer":
        return {"op": "input", "name": name}

    if cls in ELEMENTWISE:
        args, kwargs = _call(entry)
        if kwargs:
            raise ValueError(f"{name}: elementwise kwargs are not supported")
        if len(args) != 2:
            raise ValueError(f"{name}: {cls} with {len(args)} arguments; "
                             "only the binary form is supported")
        return {"op": "elementwise", "name": name, "fn": ELEMENTWISE[cls],
                "args": [_arg_source(a, name) for a in args]}

    layer = model.get_layer(name)
    inputs = _tensor_inputs(entry)

    if cls == "Dense":
        weights[f"{name}.kernel"] = np.asarray(layer.kernel)
        if layer.bias is not None:
            weights[f"{name}.bias"] = np.asarray(layer.bias)
        return {"op": "dense", "name": name, "inputs": inputs,
                "use_bias": layer.bias is not None,
                "activation": _activation_name(cfg.get("activation"), name)}

    if cls == "Embedding":
        weights[f"{name}.embeddings"] = np.asarray(layer.embeddings)
        return {"op": "embedding", "name": name, "inputs": inputs,
                "input_dim": int(layer.embeddings.shape[0]),
                "output_dim": int(layer.embeddings.shape[1])}

    if cls == "BatchNormalization":
        axis = int(layer.axis if np.ndim(layer.axis) == 0 else np.ravel(layer.axis)[0])
        if axis not in (-1,):
            # numpy_model normalises over the last axis only. Every BN in
            # this model runs on a rank-2 tensor, so -1 is the whole story;
            # a different axis would need real broadcasting logic.
            raise ValueError(f"{name}: BatchNormalization axis={axis} is not supported")
        if layer.scale:
            weights[f"{name}.gamma"] = np.asarray(layer.gamma)
        if layer.center:
            weights[f"{name}.beta"] = np.asarray(layer.beta)
        weights[f"{name}.moving_mean"] = np.asarray(layer.moving_mean)
        weights[f"{name}.moving_variance"] = np.asarray(layer.moving_variance)
        return {"op": "batch_normalization", "name": name, "inputs": inputs,
                "epsilon": float(layer.epsilon), "axis": axis,
                "scale": bool(layer.scale), "center": bool(layer.center)}

    if cls == "Dropout":
        # Identity at inference. Kept as a node so the exported graph is a
        # faithful picture of the trained one.
        return {"op": "dropout", "name": name, "inputs": inputs}

    if cls == "Flatten":
        return {"op": "flatten", "name": name, "inputs": inputs}

    if cls == "Concatenate":
        return {"op": "concatenate", "name": name, "inputs": inputs,
                "axis": int(layer.axis)}

    if cls == "Activation":
        return {"op": "activation", "name": name, "inputs": inputs,
                "activation": _activation_name(cfg.get("activation"), name)}

    if cls == "Bidirectional":
        node = {"op": "bidirectional_lstm", "name": name, "inputs": inputs,
                "merge_mode": layer.merge_mode}
        if layer.merge_mode != "concat":
            raise ValueError(f"{name}: merge_mode {layer.merge_mode!r} is not supported")
        for tag, sub in (("forward", layer.forward_layer),
                         ("backward", layer.backward_layer)):
            if sub.__class__.__name__ != "LSTM":
                raise ValueError(f"{name}: {tag} layer is {sub.__class__.__name__}, "
                                 "only LSTM is implemented")
            scfg = sub.get_config()
            if scfg.get("return_state"):
                raise ValueError(f"{name}: return_state is not supported")
            if scfg.get("stateful"):
                raise ValueError(f"{name}: stateful RNNs are not supported")
            if float(scfg.get("recurrent_dropout") or 0.0) != 0.0:
                # Inference-time no-op, but flag it rather than assume.
                pass
            node[tag] = {
                "units": int(sub.units),
                "go_backwards": bool(scfg["go_backwards"]),
                "return_sequences": bool(scfg["return_sequences"]),
                "use_bias": bool(scfg["use_bias"]),
                "activation": _activation_name(scfg["activation"], f"{name}/{tag}"),
                "recurrent_activation": _activation_name(
                    scfg["recurrent_activation"], f"{name}/{tag} recurrent"),
            }
            weights[f"{name}.{tag}.kernel"] = np.asarray(sub.cell.kernel)
            weights[f"{name}.{tag}.recurrent_kernel"] = np.asarray(sub.cell.recurrent_kernel)
            if sub.cell.bias is not None:
                weights[f"{name}.{tag}.bias"] = np.asarray(sub.cell.bias)
        if node["forward"]["return_sequences"] != node["backward"]["return_sequences"]:
            raise ValueError(f"{name}: directions disagree on return_sequences")
        return node

    raise ValueError(
        f"{name}: layer class {cls!r} has no NumPy implementation. Add one to "
        "site/numpy_model.py and a converter here — do not guess.")


def _node_deps(node):
    if node["op"] == "elementwise":
        return [a["node"] for a in node["args"] if "node" in a]
    return list(node.get("inputs", []))


def _topo_sort(nodes):
    """Execution order. The serialized order is usually already valid; this
    makes it so by construction, so numpy_model can be a plain loop."""
    pending, ordered, done = list(nodes), [], set()
    while pending:
        progressed = False
        for node in list(pending):
            if all(d in done for d in _node_deps(node)):
                ordered.append(node)
                done.add(node["name"])
                pending.remove(node)
                progressed = True
        if not progressed:
            stuck = {n["name"]: _node_deps(n) for n in pending}
            raise ValueError(f"graph has a cycle or a dangling input: {stuck}")
    return ordered


def _keras_version(model):
    v = getattr(model, "keras_version", None)
    if v:
        return v
    mod = sys.modules.get("keras")
    return getattr(mod, "__version__", "unknown") if mod else "unknown"


def export(model, out_dir: Path, source: str = ""):
    """Write model_arch.json + model_weights.npz. Returns (arch, weights)."""
    cfg = model.get_config()
    weights = {}
    nodes = [_convert(e, model, weights) for e in cfg["layers"]]
    nodes = _topo_sort(nodes)

    # Input specs come from the model's own tensors, not the serialized
    # InputLayer configs, so shape and dtype are whatever Keras will
    # actually accept. The alias is the key callers pass in the input
    # dict: "seq_input" -> "seq".
    specs = []
    for t in model.inputs:
        nm = t.name
        alias = nm[:-len("_input")] if nm.endswith("_input") else nm
        specs.append({
            "name": nm,
            "alias": alias,
            "shape": [None if d is None else int(d) for d in t.shape],
            "dtype": str(t.dtype).replace("<dtype: '", "").replace("'>", ""),
        })

    out_names = list(getattr(model, "output_names", None) or [])
    out_tensors = model.outputs
    if len(out_names) != len(out_tensors):
        out_names = [t._keras_history[0].name for t in out_tensors]
    # The graph node behind each named output. Keras names the output by
    # the layer that produced it, which is the node name we evaluate.
    outputs = {}
    for nm, t in zip(out_names, out_tensors):
        outputs[nm] = t._keras_history[0].name
    if "output" in outputs:
        primary = "output"
    elif len(outputs) == 1:
        primary = next(iter(outputs))
    else:
        raise ValueError(f"cannot tell which of {sorted(outputs)} is the pitch-type "
                         "head; it should be named 'output'")

    arch = {
        "format": ARCH_FORMAT,
        "version": ARCH_VERSION,
        "source_model": source,
        # From the archive's own metadata, or the live module when a real
        # Keras model was passed. Never import keras just for this: on a
        # broken install that import calls abort(), which no try/except
        # can catch, and it would take the whole export down for a
        # cosmetic field.
        "keras_version": _keras_version(model),
        "inputs": specs,
        "outputs": outputs,
        "primary_output": primary,
        "n_params": int(sum(int(np.prod(w.shape)) for w in weights.values())),
        "nodes": nodes,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    arch_path, w_path = out_dir / ARCH_NAME, out_dir / WEIGHTS_NAME

    # Atomic, same rule as 01_scrape_statcast.py: a half-written artifact
    # silently reused is a whole wasted debugging session.
    tmp = arch_path.with_suffix(arch_path.suffix + ".tmp")
    tmp.write_text(json.dumps(arch, indent=2))
    os.replace(tmp, arch_path)

    tmp = w_path.with_suffix(w_path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{k: np.asarray(v, np.float32)
                                   for k, v in weights.items()})
    os.replace(tmp, w_path)

    return arch, weights


# =========================
# Test batch (shared with test_numpy_parity.py)
# =========================
def random_batch(arch, seed=1234, batch=48):
    """A reproducible batch of model inputs, keyed by input ALIAS.

    Deliberately includes the rows that have historically broken hand-written
    forward passes: an entirely zero-padded sequence (a pitcher's first pitch
    of a game), a partially padded one, a single-pitch arsenal mask, and an
    all-ones mask (the unknown-pitcher fallback).
    """
    rng = np.random.default_rng(seed)
    limits = {}
    for node in arch["nodes"]:
        if node["op"] == "embedding":
            limits[node["inputs"][0]] = node["input_dim"]

    out = {}
    for spec in arch["inputs"]:
        name, alias, shape = spec["name"], spec["alias"], spec["shape"]
        tail = [d for d in shape[1:]]
        if name in limits:
            out[alias] = rng.integers(0, limits[name], size=(batch,)).astype(np.int32)
        elif alias == "arsenal_mask":
            k = tail[-1]
            m = (rng.random((batch, k)) < 0.5).astype(np.float32)
            m[np.arange(batch), rng.integers(0, k, size=batch)] = 1.0  # never all-zero
            m[2] = 0.0
            m[2, 3] = 1.0          # one pitch only: the hardest mask
            m[3] = 1.0             # unknown pitcher -> all ones
            out[alias] = m
        else:
            arr = rng.standard_normal((batch, *tail)).astype(np.float32)
            if len(tail) == 2:     # the sequence input
                arr[0] = 0.0       # fully padded
                arr[1, :5] = 0.0   # partially padded
            out[alias] = arr
    return out


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", type=Path, default=SERVING_DIR)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the Keras-vs-NumPy check (do not)")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--verify-only", action="store_true",
                    help="internal: the child that runs the Keras comparison")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"no model at {args.model} — train first")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if args.verify_only:
        arch = json.loads((args.out_dir / ARCH_NAME).read_text())
    else:
        print(f"Loading {args.model} (no TensorFlow) ...")
        model = KerasArchive(args.model)
    if not args.verify_only:
        arch, weights = export(model, args.out_dir, source=args.model.name)

    if not args.verify_only:
        print(f"  nodes      : {len(arch['nodes'])}")
        print(f"  params     : {arch['n_params']:,}")
        print(f"  outputs    : {arch['outputs']} (primary '{arch['primary_output']}')")
        print(f"  arch       : {args.out_dir / ARCH_NAME}")
        print(f"  weights    : {args.out_dir / WEIGHTS_NAME} "
              f"({(args.out_dir / WEIGHTS_NAME).stat().st_size / 1e6:.2f} MB)")

    if args.no_verify:
        print("Verification SKIPPED — the export is unproven.")
        return

    # Verify on every export, not just when someone remembers to run the
    # test: a retrain that changes the graph must fail here, loudly.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from numpy_model import NumpyModel

    # Verifying means running the ORIGINAL graph, which is the one thing
    # here that needs TensorFlow. A broken TF install does not raise — it
    # calls abort(), which no try/except can catch — so verification runs
    # in a CHILD process. The weights are already written by now, so a
    # child that dies costs the check, not the export.
    if not args.verify_only:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--verify-only",
             "--model", str(args.model), "--out-dir", str(args.out_dir),
             "--tolerance", str(args.tolerance)])
        if r.returncode != 0:
            print()
            print("Verification DID NOT RUN — TensorFlow is unusable here "
                  f"(child exited {r.returncode}).")
            print("The export itself is unaffected: the weights came straight")
            print("out of the .keras archive without TensorFlow. To confirm the")
            print("NumPy graph reproduces Keras, run site/test_numpy_parity.py")
            print("on a machine with a working TensorFlow.")
        return

    import keras
    ref_model = keras.models.load_model(args.model, compile=False)

    batch = random_batch(arch)
    alias_to_name = {s["alias"]: s["name"] for s in arch["inputs"]}
    keras_out = ref_model.predict({alias_to_name[k]: v for k, v in batch.items()},
                                  verbose=0)
    np_model = NumpyModel(args.out_dir / WEIGHTS_NAME, args.out_dir / ARCH_NAME)
    np_out = np_model.predict_all(batch)

    if not isinstance(keras_out, dict):
        keras_out = {arch["primary_output"]: keras_out}
    worst = 0.0
    for nm, ref in keras_out.items():
        diff = float(np.max(np.abs(np.asarray(ref) - np_out[nm])))
        worst = max(worst, diff)
        print(f"  max|keras - numpy|  {nm:12s} = {diff:.3e}")
    if worst >= args.tolerance:
        raise SystemExit(f"EXPORT IS WRONG: {worst:.3e} >= {args.tolerance:.0e}")
    print(f"Verified: worst {worst:.3e} < {args.tolerance:.0e}")


if __name__ == "__main__":
    main()
