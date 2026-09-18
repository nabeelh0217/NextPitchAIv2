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

from evaluate_model import build_report, write_report

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
# Run 5 used 0.5, which produced a 17:1 weight ratio (FF 0.233, KN 4.0)
# ON TOP OF focal gamma=2, which already upweights rare/hard examples.
# The result was a model that under-called four-seamers (recall 30% at
# precision 64%) and over-called every rare type (KC recall 85% at
# precision 27%) — imbalance corrected twice, the same error v5 made with
# oversampling. Gamma alone is the imbalance correction; raise this only
# with evidence that a class is genuinely being ignored.
ALPHA_POWER = 0.0
ALPHA_CAP = 2.0

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
print("\nSplitting train/val (80/20, stratified)...")
indices = np.arange(len(y_labels))
idx_tr, idx_val = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y_labels)
print(f"Train: {len(idx_tr):,}  Val: {len(idx_val):,}")


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
masked_logits = logits + (1.0 - mask_input) * -1e9
output = Activation("softmax", name="output")(masked_logits)

model = Model(
    inputs=[
        seq_input, ctx_input,
        pitcher_id_input, batter_id_input,
        pitcher_hand_input, batter_hand_input,
        park_id_input, catcher_id_input,
        mask_input,
    ],
    outputs=output
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=focal_loss(), metrics=["accuracy"])
model.summary()


# =========================
# 4) Callbacks
# =========================
callbacks = [
    ModelCheckpoint(
        str(DATA_DIR / "best_model_v5.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        str(DATA_DIR / "final_model_v5.keras"),
        save_best_only=False
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
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
    train_inputs, y_tr,
    validation_data=(val_inputs, y_val),
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

y_pred_probs = model.predict(val_inputs, batch_size=2048, verbose=0)
report = build_report(
    y_val, y_pred_probs,
    X_pitcher_id[idx_tr], y_tr,
    X_pitcher_id[idx_val], pitch_classes, hist_dict,
)
print("\n" + report)
write_report(report)


# =========================
# 7) Save training curves + history
# =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"], label="train acc")
ax1.plot(history.history["val_accuracy"], label="val acc")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy")
ax1.legend()

ax2.plot(history.history["loss"], label="train loss")
ax2.plot(history.history["val_loss"], label="val loss")
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
