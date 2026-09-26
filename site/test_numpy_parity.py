"""
NextPitchAI v6 — Keras vs NumPy numeric parity
===============================================
site/numpy_model.py exists so the app can serve without TensorFlow. It is
worth having only if it computes the SAME numbers, so this is the gate on
it: export the real trained model, push one fixed batch through Keras and
through NumPy, and assert the largest absolute difference on every head is
under 1e-4.

A near-miss here is not cosmetic. A transposed gate or an unreversed
backward pass produces probabilities that look entirely reasonable and are
wrong, and nothing downstream would ever notice.

The batch is site/export_model.random_batch, which deliberately contains a
fully zero-padded sequence, a partially padded one, a single-pitch arsenal
mask and an all-ones mask.

Also reports the reason the whole exercise exists: peak RSS of a
NumPy-only serving process against a TensorFlow one, measured in child
processes so each number is that path's real cost, plus the size of the
exported weights.

Usage:
    python site/test_numpy_parity.py
    python site/test_numpy_parity.py --model /path/to/best_model_v5.keras
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SITE_DIR = Path(__file__).resolve().parent
BASE_DIR = SITE_DIR.parent
DEFAULT_MODEL = BASE_DIR / "data_v5" / "best_model_v5.keras"

TOLERANCE = 1e-4
BATCH = 48
SEED = 1234

# numpy_model.py earns its place by importing none of these.
HEAVY = ("tensorflow", "keras", "torch", "sklearn", "scipy", "pandas", "jax")

# Peak RSS, read the only way that is trustworthy here: Linux
# /proc/self/status VmHWM. ru_maxrss is INHERITED across fork/exec on this
# kernel, so a child spawned from this TensorFlow-loaded process reports
# the parent's peak and every path looks identical. macOS has no /proc, so
# fall back to ru_maxrss, which reports bytes there and kilobytes on Linux.
PEAK_RSS_FN = r"""
def peak_rss_kb():
    import resource, sys
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw // 1024 if sys.platform == "darwin" else raw
"""

# Run in a child so the measurement is that path's own cost and not this
# process's, which has already paid for TensorFlow.
RSS_PROBE = PEAK_RSS_FN + r"""
import sys
sys.path.insert(0, {site!r})
import numpy as np
from numpy_model import NumpyModel
m = NumpyModel({npz!r}, {arch!r})
b = np.load({batch!r})
probs = m.predict({{k: b[k] for k in b.files}})
heavy = [h for h in {heavy!r} if h in sys.modules]
print("ROWSUM", float(np.abs(probs.sum(axis=1) - 1.0).max()))
print("HEAVY", ",".join(heavy) if heavy else "none")
print("RSS_KB", peak_rss_kb())
"""

TF_RSS_PROBE = PEAK_RSS_FN + r"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras
m = keras.models.load_model({model!r}, compile=False)
import numpy as np
b = np.load({batch!r})
names = {names!r}
m.predict({{names[k]: b[k] for k in b.files}}, verbose=0)
print("RSS_KB", peak_rss_kb())
"""


def main():
    ap = argparse.ArgumentParser(description="Keras vs NumPy parity check.")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="the .keras model to export and check")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the export (default: a temp dir)")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"no model at {args.model}")

    sys.path.insert(0, str(SITE_DIR))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import export_model
    from numpy_model import NumpyModel

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="numpy_parity_"))
    print(f"model    : {args.model}")
    print(f"export to: {out_dir}")

    import keras
    model = keras.models.load_model(args.model, compile=False)
    arch, _ = export_model.export(model, out_dir, source=args.model.name)
    npz = out_dir / export_model.WEIGHTS_NAME
    arch_path = out_dir / export_model.ARCH_NAME
    print(f"params   : {arch['n_params']:,}")
    print(f"heads    : {arch['outputs']}  primary '{arch['primary_output']}'")

    batch = export_model.random_batch(arch, seed=SEED, batch=BATCH)
    alias_to_name = {s["alias"]: s["name"] for s in arch["inputs"]}
    print(f"batch    : {BATCH} rows "
          f"(row 0 fully padded, row 1 partly padded, row 2 one-pitch mask, "
          f"row 3 all-ones mask)")

    keras_out = model.predict({alias_to_name[k]: v for k, v in batch.items()},
                              verbose=0)
    if not isinstance(keras_out, dict):
        keras_out = {arch["primary_output"]: keras_out}

    np_model = NumpyModel(npz, arch_path)
    np_out = np_model.predict_all(batch)

    print("\n--- parity ---")
    worst = 0.0
    for name, ref in keras_out.items():
        ref = np.asarray(ref, np.float32)
        got = np_out[name]
        diff = float(np.max(np.abs(ref - got)))
        agree = float(np.mean(ref.argmax(1) == got.argmax(1)))
        worst = max(worst, diff)
        print(f"  {name:12s} max|diff| = {diff:.3e}   "
              f"mean|diff| = {float(np.mean(np.abs(ref - got))):.3e}   "
              f"argmax agreement = {agree:.1%}")

    probs = np_out[arch["primary_output"]]
    print(f"  row sums deviate from 1 by at most "
          f"{float(np.abs(probs.sum(1) - 1.0).max()):.3e}")
    # The masked classes are the point of the model, so check the mask is
    # still doing its job after the round trip rather than only the norm.
    mask = batch["arsenal_mask"]
    print(f"  max probability on an out-of-arsenal class: "
          f"{float((probs * (1 - mask)).max()):.3e}")

    # --- memory, the reason this module exists ---
    print("\n--- memory ---")
    batch_file = out_dir / "parity_batch.npz"
    np.savez(batch_file, **batch)

    probe = RSS_PROBE.format(site=str(SITE_DIR), npz=str(npz),
                             arch=str(arch_path), batch=str(batch_file),
                             heavy=HEAVY)
    res = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"numpy-only probe failed:\n{res.stderr}")
    info = dict(line.split(" ", 1) for line in res.stdout.strip().splitlines())
    np_rss = int(info["RSS_KB"]) / 1024.0
    print(f"  numpy-only serving process peak RSS : {np_rss:7.1f} MB")
    print(f"  heavy imports pulled in by numpy_model: {info['HEAVY']}")
    print(f"  exported weights  {npz.name}: {npz.stat().st_size / 1e6:.2f} MB"
          f"  (source .keras {args.model.stat().st_size / 1e6:.2f} MB)")

    tf_probe = TF_RSS_PROBE.format(model=str(args.model), batch=str(batch_file),
                                   names=alias_to_name)
    res = subprocess.run([sys.executable, "-c", tf_probe],
                         capture_output=True, text=True)
    if res.returncode == 0:
        tf_rss = int(res.stdout.strip().splitlines()[-1].split()[-1]) / 1024.0
        print(f"  tensorflow serving process peak RSS : {tf_rss:7.1f} MB"
              f"   ({tf_rss - np_rss:+.1f} MB)")
    else:
        print("  tensorflow probe failed; skipping the comparison")

    print()
    if info["HEAVY"] != "none":
        raise SystemExit(f"FAIL: numpy_model.py imported {info['HEAVY']}")
    if float(info["ROWSUM"]) > 1e-5:
        raise SystemExit(f"FAIL: softmax rows do not sum to 1 ({info['ROWSUM']})")
    assert worst < args.tolerance, (
        f"FAIL: max abs diff {worst:.3e} >= {args.tolerance:.0e} — the NumPy "
        "forward pass does not match Keras")
    print(f"PASS: worst max abs diff {worst:.3e} < {args.tolerance:.0e}")


if __name__ == "__main__":
    main()
