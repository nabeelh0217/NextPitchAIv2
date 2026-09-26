"""
NextPitchAI v6 — Step 3: Train model
======================================
Predicts the ACTUAL pitch type (10 canonical Statcast classes) with the
pitcher's arsenal enforced INSIDE the model: a binary mask input forces
the logits of pitch types this pitcher never throws to -inf before the
softmax, in training and at inference. The model spends all of its
capacity discriminating within each pitcher's real repertoire.

Training strategy:
  - Class-weighted focal loss (gamma=2, per-class alpha from natural
    frequencies). Replaces physical row duplication (RandomOverSampler),
    which the run-4 training curves showed drives early memorization.
  - Stacked BiLSTM over this pitcher's last 8 pitches
  - Embeddings: pitcher, batter, catcher, park, handedness
  - Train/val split BEFORE any resampling; evaluation on the natural
    distribution.

Requirements:
    pip install tensorflow numpy pandas scikit-learn joblib matplotlib

Usage:
    python 03_train.py

Output:
    data_v5/final_model_v5.keras
    data_v5/best_model_v5.keras
    data_v5/training_curves_v5.png
    data_v5/training_history_v5.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense,
    Dropout, Concatenate, Flatten, BatchNormalization, Activation,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras import backend as K

from evaluate_model import (build_report, build_location_report,
                            build_arsenal_mask, write_report)

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_v5"

# Hyperparameters
EPOCHS = 60
BATCH_SIZE = 512
PATIENCE = 8
LEARNING_RATE = 1e-3

# Embedding dimensions
PITCHER_EMBED_DIM = 32
BATTER_EMBED_DIM = 16
HAND_EMBED_DIM = 4
PARK_EMBED_DIM = 4
CATCHER_EMBED_DIM = 8

# Model dimensions
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

# Focal loss
FOCAL_GAMMA = 2.0
# Per-class alpha = (N / (K * count)) ** ALPHA_POWER, mean-normalized and
# capped. ALPHA_POWER = 0 means uniform weights.
#
# Both endpoints have been measured; this is a precision/recall frontier,
# not a bug to fix:
#   0.5 (run 5): 17:1 weight ratio (FF 0.233, KN 4.0) stacked on focal
#       gamma=2. Wrecked the majority class — FF recall 30.2% at
#       precision 0.637 — for minority recall (KC 85.0%, FS 84.2%).
#       top-1 lift +1.2%, log-loss 1.1864, macro-F1 0.449.
#   0.0 (run 6): uniform. Every headline metric best-so-far — top-1 lift
#       +6.1%, top-3 +6.2%, log-loss 1.1531 — but the model leans on
#       FF/SI when uncertain, so minority recall collapsed (CU 51.4% ->
#       13.5%, KC 85.0% -> 21.0%). macro-F1 fell to 0.422.
#
# 0.25 is the midpoint probe. DECISION RULE, fixed in advance so this
# does not become an open-ended sweep: keep 0.25 only if it recovers
# minority recall (CU/KC above ~35-40%) AND holds top-1 lift above +1.2%
# with log-loss at or below 1.1864. Otherwise revert to 0.0 and accept
# run 6. Either way this is the last alpha round — see docs/EXPERIMENTS.md.
ALPHA_POWER = 0.25
ALPHA_CAP = 2.0

# Split mode.
#   "random"   - stratified 80/20 over all pitches. Comparable to runs
#                1-8, but validation rows share at-bats and outings with
#                training rows, so the edge is optimistic.
#   "temporal" - train on every season before HOLDOUT_SEASON, validate
#                on that season. No shared games, and it is the actual
#                deployment question: predict a year you have not seen.
#                Numbers read LOWER; they are the honest ones.
SPLIT_MODE = "temporal"
HOLDOUT_SEASON = 2025

# Second head: where the pitch goes (heart/shadow/chase/waste).
# Location is far noisier than type - a pitcher aims and misses - so it
# is weighted well below the type head and must never drag it down.
ENABLE_LOCATION_HEAD = True
LOCATION_LOSS_WEIGHT = 0.3

# Logit penalty for pitch types outside the pitcher's arsenal.
ARSENAL_MASK_PENALTY = -12.0

# The values above are defaults; every one can be overridden on the
# command line. Two runs in a row were invalidated by a config edit that
# never reached the file being executed (run 9b went out as a random
# split after SPLIT_MODE was set to "temporal"). A flag cannot silently
# fail to apply, and the banner below prints what is actually in effect.
_ap = argparse.ArgumentParser(description="Train the v6 pitch model.")
_ap.add_argument("--split", choices=["random", "temporal"], default=SPLIT_MODE)
_ap.add_argument("--holdout-season", type=int, default=HOLDOUT_SEASON)
_ap.add_argument("--location-head", dest="loc", action="store_true", default=None,
                 help="force the attack-zone head ON")
_ap.add_argument("--no-location-head", dest="loc", action="store_false",
                 help="force the attack-zone head OFF")
_ap.add_argument("--location-weight", type=float, default=LOCATION_LOSS_WEIGHT)
_ap.add_argument("--epochs", type=int, default=EPOCHS)
_ap.add_argument("--full-arsenal-mask", action="store_true",
                 help="build the mask over ALL rows (leaky; reproduces runs 1-11)")
_args = _ap.parse_args()

SPLIT_MODE = _args.split
HOLDOUT_SEASON = _args.holdout_season
ENABLE_LOCATION_HEAD = ENABLE_LOCATION_HEAD if _args.loc is None else _args.loc
LOCATION_LOSS_WEIGHT = _args.location_weight
EPOCHS = _args.epochs
FULL_ARSENAL_MASK = _args.full_arsenal_mask

print("=" * 60)
print("RUN CONFIG — check this matches what you intended")
print("=" * 60)
print(f"  split          : {SPLIT_MODE}"
      + (f" (hold out {HOLDOUT_SEASON})" if SPLIT_MODE == "temporal" else ""))
print(f"  location head  : {'ON  (weight %.2f)' % LOCATION_LOSS_WEIGHT if ENABLE_LOCATION_HEAD else 'OFF'}")
print(f"  epochs         : {EPOCHS}")
print(f"  arsenal mask   : {'ALL rows (LEAKY)' if FULL_ARSENAL_MASK else 'training rows only'}")
print("=" * 60)

# =========================
# 0) Load metadata + arrays
# =========================
print("Loading data...")
with open(DATA_DIR / "meta_v5.json") as f:
    meta = json.load(f)

X_seq = np.load(DATA_DIR / "X_seq.npy")
X_ctx = np.load(DATA_DIR / "X_ctx.npy")
X_pitcher_id = np.load(DATA_DIR / "X_pitcher_id.npy")
X_batter_id = np.load(DATA_DIR / "X_batter_id.npy")
X_pitcher_hand = np.load(DATA_DIR / "X_pitcher_hand.npy")
X_batter_hand = np.load(DATA_DIR / "X_batter_hand.npy")
X_park_id = np.load(DATA_DIR / "X_park_id.npy")
X_catcher_id = np.load(DATA_DIR / "X_catcher_id.npy")
X_arsenal_mask = np.load(DATA_DIR / "X_arsenal_mask.npy")
y_labels = np.load(DATA_DIR / "y_labels.npy")
y_zone = (np.load(DATA_DIR / "y_zone.npy")
          if (DATA_DIR / "y_zone.npy").exists() else None)
seasons = (np.load(DATA_DIR / "X_season.npy")
           if (DATA_DIR / "X_season.npy").exists() else None)
HAS_LOCATION = ENABLE_LOCATION_HEAD and y_zone is not None
n_zone = int(meta.get("n_zones", 4))
zone_classes = meta.get("zone_classes", [])

print(f"Samples: {len(y_labels):,}")
print(f"Seq shape: {X_seq.shape}")
print(f"Ctx shape: {X_ctx.shape}")
print(f"Arsenal mask shape: {X_arsenal_mask.shape} "
      f"(avg arsenal size {X_arsenal_mask.sum(axis=1).mean():.2f})")
print(f"Pitch classes: {meta['pitch_classes']}")

seq_len = meta["seq_len"]
seq_feats = meta["seq_feats"]
ctx_feats = meta["ctx_feats"]
n_pitch = meta["n_pitch_types"]
pitch_classes = meta["pitch_classes"]


# =========================
# 1) Train/val split (natural distribution; no resampling anywhere)
# =========================
indices = np.arange(len(y_labels))
if SPLIT_MODE == "temporal":
    if seasons is None:
        raise SystemExit("temporal split needs X_season.npy — re-run 02_preprocess.py")
    idx_tr = indices[seasons < HOLDOUT_SEASON]
    idx_val = indices[seasons == HOLDOUT_SEASON]
    if len(idx_val) == 0:
        raise SystemExit(f"no rows for season {HOLDOUT_SEASON}; check the scrape")
    print(f"\nTEMPORAL split: train on {sorted(set(seasons[idx_tr].tolist()))}, "
          f"validate on {HOLDOUT_SEASON}")
    print("These numbers are the honest ones and will read lower than the")
    print("random-split runs — no shared games between train and val.")
else:
    print("\nSplitting train/val (80/20, stratified)...")
    idx_tr, idx_val = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_labels)
print(f"Train: {len(idx_tr):,}  Val: {len(idx_val):,}")

# Record the split that actually ran. evaluate_model.py reads this to
# reproduce the same rows, and the eval report prints it — a SPLIT_MODE
# edit that fails to take is otherwise invisible in the output.
SPLIT_DESC = (f"temporal, validate on {HOLDOUT_SEASON}" if SPLIT_MODE == "temporal"
              else "random 80%/20%, seed 42")
with open(DATA_DIR / "split_v5.json", "w") as f:
    json.dump({"mode": SPLIT_MODE, "holdout_season": HOLDOUT_SEASON,
               "n_train": int(len(idx_tr)), "n_val": int(len(idx_val))}, f, indent=2)
print(f"Split recorded: {SPLIT_DESC}")

# Rebuild the arsenal mask from TRAINING rows only.
#
# 02_preprocess.py builds it over every row, which leaks: on a temporal
# split it tells the model which pitches a pitcher will throw in the
# held-out season, including ones he only added that year. The mask is a
# real advantage (it zeroes out impossible classes) and the scout
# baseline gets no equivalent, so the leak flatters the model in exactly
# the comparison that decides the product.
#
# The split is only known here, so the mask is rebuilt here.
if not FULL_ARSENAL_MASK:
    _tbl = build_arsenal_mask(X_pitcher_id, y_labels, idx_tr, n_pitch)
    _new = _tbl[X_pitcher_id]
    _lost = float(((X_arsenal_mask > 0) & (_new == 0)).sum()) / len(X_arsenal_mask)
    print(f"Arsenal mask rebuilt from training rows: "
          f"{X_arsenal_mask.sum(1).mean():.2f} -> {_new.sum(1).mean():.2f} "
          f"types/pitch ({_lost:.2%} of rows lost a type the leaky mask had)")
    _unmaskable = float((_new[np.arange(len(y_labels)), y_labels] == 0)[idx_val].mean())
    print(f"  {_unmaskable:.2%} of validation rows are a type the pitcher "
          f"never threw in training")
    X_arsenal_mask = _new
    np.save(DATA_DIR / "arsenal_mask_table_v5.npy", _tbl)
else:
    print("Arsenal mask: ALL rows (leaky) — reproducing runs 1-11")


def gather(idx):
    return [
        X_seq[idx], X_ctx[idx],
        X_pitcher_id[idx], X_batter_id[idx],
        X_pitcher_hand[idx], X_batter_hand[idx],
        X_park_id[idx], X_catcher_id[idx],
        X_arsenal_mask[idx],
    ]


train_inputs = gather(idx_tr)
val_inputs = gather(idx_val)
y_tr = y_labels[idx_tr]
y_val = y_labels[idx_val]

if HAS_LOCATION:
    # Rows with no usable location get label 0 and sample_weight 0, so
    # they contribute nothing to the zone loss instead of being guessed.
    z_tr_raw, z_val_raw = y_zone[idx_tr], y_zone[idx_val]
    z_tr = np.where(z_tr_raw >= 0, z_tr_raw, 0).astype(np.int64)
    z_val = np.where(z_val_raw >= 0, z_val_raw, 0).astype(np.int64)
    w_tr = (z_tr_raw >= 0).astype(np.float32)
    w_val = (z_val_raw >= 0).astype(np.float32)
    print(f"Location head ON (weight {LOCATION_LOSS_WEIGHT}): "
          f"{w_tr.mean():.1%} of training rows have a usable zone")
    y_tr_out = {"output": y_tr, "zone_output": z_tr}
    y_val_out = {"output": y_val, "zone_output": z_val}
    sw_tr = {"output": np.ones(len(y_tr), np.float32), "zone_output": w_tr}
    sw_val = {"output": np.ones(len(y_val), np.float32), "zone_output": w_val}
else:
    y_tr_out, y_val_out, sw_tr, sw_val = y_tr, y_val, None, None


# =========================
# 2) Class-weighted focal loss
# =========================
# Focal loss with gamma=2 already concentrates gradient on hard/rare
# examples. ALPHA_POWER adds optional extra per-class weighting on top;
# it defaults to 0 (uniform) because run 5 showed the two corrections
# compound badly. See the constant's comment above.
counts = np.maximum(np.bincount(y_tr, minlength=n_pitch), 1)
raw_alpha = (len(y_tr) / (n_pitch * counts)) ** ALPHA_POWER
class_alpha = np.minimum(raw_alpha / raw_alpha.mean(), ALPHA_CAP).astype(np.float32)

print(f"\nClass weights (focal alpha, ALPHA_POWER={ALPHA_POWER}):")
for i, cls in enumerate(pitch_classes):
    print(f"  {cls}: n={counts[i]:,}  alpha={class_alpha[i]:.3f}")


def focal_loss(gamma=FOCAL_GAMMA, alpha_vec=class_alpha):
    alpha_const = tf.constant(alpha_vec, dtype=tf.float32)

    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        cross_entropy = -y_true * tf.math.log(y_pred)
        weights = alpha_const * tf.pow(1 - y_pred, gamma)
        loss = tf.reduce_sum(weights * cross_entropy, axis=-1)
        return tf.reduce_mean(loss)
    return focal_loss_fn


# =========================
# 3) Build model
# =========================
print("\nBuilding model...")

# --- Sequence branch (stacked BiLSTM over this pitcher's last pitches) ---
seq_input = Input(shape=(seq_len, seq_feats), name="seq_input")
s = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True))(seq_input)
s = Dropout(DROPOUT_RATE)(s)
s = Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False))(s)
s = Dropout(DROPOUT_RATE)(s)

# --- Context branch (game state + arsenal/matchup/batter priors) ---
ctx_input = Input(shape=(ctx_feats,), name="ctx_input")
c = Dense(96, activation="relu")(ctx_input)
c = BatchNormalization()(c)
c = Dropout(DROPOUT_RATE)(c)

# --- Identity embeddings ---
def embed_branch(name, n_ids, dim):
    inp = Input(shape=(1,), name=name, dtype="int32")
    emb = Embedding(input_dim=n_ids, output_dim=dim,
                    name=f"{name}_embedding")(inp)
    return inp, Flatten()(emb)

pitcher_id_input, pitcher_emb = embed_branch("pitcher_id", meta["n_pitcher_ids"], PITCHER_EMBED_DIM)
batter_id_input, batter_emb = embed_branch("batter_id", meta["n_batter_ids"], BATTER_EMBED_DIM)
pitcher_hand_input, pitcher_hand_emb = embed_branch("pitcher_hand", 2, HAND_EMBED_DIM)
batter_hand_input, batter_hand_emb = embed_branch("batter_hand", 3, HAND_EMBED_DIM)
park_id_input, park_emb = embed_branch("park_id", meta["n_park_ids"], PARK_EMBED_DIM)
catcher_id_input, catcher_emb = embed_branch("catcher_id", meta["n_catcher_ids"], CATCHER_EMBED_DIM)

# --- Arsenal mask input ---
mask_input = Input(shape=(n_pitch,), name="arsenal_mask")

# --- Merge everything ---
combined = Concatenate()([
    s, c,
    pitcher_emb, batter_emb,
    pitcher_hand_emb, batter_hand_emb,
    park_emb, catcher_emb,
])

z = Dense(DENSE_UNITS, activation="relu")(combined)
z = BatchNormalization()(z)
z = Dropout(DROPOUT_RATE)(z)
z = Dense(64, activation="relu")(z)
z = Dropout(0.2)(z)

# Arsenal-masked softmax: pitch types outside this pitcher's repertoire
# get their logits pushed to -inf, so they receive ~zero probability and
# contribute nothing to the loss. All discrimination happens within the
# pitcher's actual arsenal.
logits = Dense(n_pitch, name="logits")(z)
# Soft, not infinite. With a training-only mask a held-out pitcher can
# genuinely throw a type he had never thrown before (~2% of 2025 rows),
# and -1e9 charges those the full 27.6-nat clip. exp(-12) ~ 6e-6 still
# excludes the class for every practical purpose without pretending the
# event is impossible.
masked_logits = logits + (1.0 - mask_input) * ARSENAL_MASK_PENALTY
output = Activation("softmax", name="output")(masked_logits)

# Second head: where the pitch ends up. Shares the trunk, so the
# type/location correlation (sliders go low-away, four-seamers go up) is
# available to it implicitly without forcing a sparse 40-class joint
# target. No arsenal mask here — every pitcher can miss anywhere.
outputs = output
if HAS_LOCATION:
    zh = Dense(64, activation="relu", name="zone_hidden")(z)
    zh = Dropout(0.2)(zh)
    zone_output = Dense(n_zone, activation="softmax", name="zone_output")(zh)
    # Dict, not list: targets, losses and sample_weights are all keyed by
    # name, and a list here makes Keras index a dict by position.
    outputs = {"output": output, "zone_output": zone_output}

model = Model(
    inputs=[
        seq_input, ctx_input,
        pitcher_id_input, batter_id_input,
        pitcher_hand_input, batter_hand_input,
        park_id_input, catcher_id_input,
        mask_input,
    ],
    outputs=outputs
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
if HAS_LOCATION:
    model.compile(
        optimizer=optimizer,
        loss={"output": focal_loss(),
              "zone_output": "sparse_categorical_crossentropy"},
        loss_weights={"output": 1.0, "zone_output": LOCATION_LOSS_WEIGHT},
        weighted_metrics={"output": ["accuracy"], "zone_output": ["accuracy"]},
    )
else:
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=["accuracy"])
model.summary()


# =========================
# 4) Callbacks
# =========================
# With two heads val_loss is the weighted total; the type head is what
# we select on.
MONITOR = "val_output_loss" if HAS_LOCATION else "val_loss"

callbacks = [
    ModelCheckpoint(
        str(DATA_DIR / "best_model_v5.keras"),
        monitor=MONITOR,
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        str(DATA_DIR / "final_model_v5.keras"),
        save_best_only=False
    ),
    EarlyStopping(
        monitor=MONITOR,
        mode="min",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor=MONITOR,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]


# =========================
# 5) Train
# =========================
print(f"\nTraining for up to {EPOCHS} epochs...")
print(f"Focal loss: gamma={FOCAL_GAMMA}, per-class alpha (printed above)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Early stopping patience: {PATIENCE}")

history = model.fit(
    train_inputs, y_tr_out,
    sample_weight=sw_tr,
    validation_data=((val_inputs, y_val_out, sw_val) if HAS_LOCATION
                     else (val_inputs, y_val)),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# =========================
# 6) Evaluate — natural distribution
# =========================
# The report is also written to data_v5/eval_report_v5.txt so a closed
# terminal doesn't lose it. `python evaluate_model.py` regenerates it from
# the saved model at any time.
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

pred = model.predict(val_inputs, batch_size=2048, verbose=0)
y_pred_probs, zone_probs = ((pred["output"], pred["zone_output"])
                            if HAS_LOCATION else (pred, None))

# --- Location head, held to the same bar as the type head ---
# Built by evaluate_model.build_location_report so training and a later
# re-score can never disagree, same rule as build_report.
if HAS_LOCATION:
    LOCATION_BLOCK = build_location_report(
        z_val, zone_probs, w_val,
        X_pitcher_id[idx_tr], z_tr, w_tr,
        X_pitcher_id[idx_val], zone_classes,
    )
else:
    LOCATION_BLOCK = ""

report = build_report(
    y_val, y_pred_probs,
    X_pitcher_id[idx_tr], y_tr,
    X_pitcher_id[idx_val], pitch_classes, hist_dict, SPLIT_DESC,
)
if LOCATION_BLOCK:
    report = report + "\n\n" + LOCATION_BLOCK
print("\n" + report)
write_report(report)


# =========================
# 7) Save training curves + history
# =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

_ak = "output_accuracy" if "output_accuracy" in history.history else "accuracy"
ax1.plot(history.history[_ak], label="train acc")
ax1.plot(history.history["val_" + _ak], label="val acc")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy")
ax1.legend()

_lk = "output_loss" if "output_loss" in history.history else "loss"
ax2.plot(history.history[_lk], label="train loss (type head)")
ax2.plot(history.history["val_" + _lk], label="val loss (type head)")
if "val_zone_output_loss" in history.history:
    ax2.plot(history.history["val_zone_output_loss"], label="val loss (zone head)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig(str(DATA_DIR / "training_curves_v5.png"), dpi=300)
print(f"Training curves saved to {DATA_DIR / 'training_curves_v5.png'}")

hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(DATA_DIR / "training_history_v5.json", "w") as f:
    json.dump(hist_dict, f, indent=2)

print("\nDone!")
print(f"Best model:   {DATA_DIR / 'best_model_v5.keras'}")
print(f"Final model:  {DATA_DIR / 'final_model_v5.keras'}")
print(f"Eval report:  {DATA_DIR / 'eval_report_v5.txt'}")
