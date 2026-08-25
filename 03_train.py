"""
NextPitchAI v5 — Step 3: Train model
======================================
Keeps the proven training strategy from the 66% run:
  - Focal loss (gamma=2)
  - Random oversampling for class balance
  - Stacked BiLSTM
  - EarlyStopping + checkpoints

v5 upgrades:
  - New inputs: park + catcher embeddings; context now carries arsenal
    priors, matchup history, batter profiles, TTO, rest days, etc.
  - METHODOLOGY FIX: split train/val FIRST, then oversample only the
    training set. (v4 oversampled before splitting, which put duplicate
    rows in both train and val and inflated validation accuracy.)
  - Capped oversampling so tiny classes aren't duplicated 100x.
  - Evaluation runs on the natural (un-oversampled) validation
    distribution — the honest number.

Requirements:
    pip install tensorflow numpy pandas scikit-learn joblib imbalanced-learn matplotlib

Usage:
    python 03_train_v4.py

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
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense,
    Dropout, Concatenate, Flatten, BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras import backend as K

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

# Focal loss params (proven from 66% run)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 1.0

# Oversampling caps: minority classes are boosted to at most
# MAX_MINORITY_FRACTION of the majority class, and never duplicated
# more than MAX_DUPLICATION times. Prevents 100x-duplicated rare
# classes (e.g. knuckleballs) from teaching the model false confidence.
# 0.75 (not 0.4): at 0.4 the cap sat BELOW breaking's natural count, so
# the second-largest class got zero oversampling and its recall collapsed
# to 28.9% (run 3) while fastball recall climbed — the model just leaned
# into the 1.8:1 majority imbalance.
MAX_MINORITY_FRACTION = 0.75
MAX_DUPLICATION = 8

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
y_labels = np.load(DATA_DIR / "y_labels.npy")

print(f"Samples: {len(y_labels):,}")
print(f"Seq shape: {X_seq.shape}")
print(f"Ctx shape: {X_ctx.shape}")
print(f"Pitcher IDs: {meta['n_pitcher_ids']}, Batter IDs: {meta['n_batter_ids']}")
print(f"Park IDs: {meta['n_park_ids']}, Catcher IDs: {meta['n_catcher_ids']}")
print(f"Buckets: {meta['bucket_classes']}")

seq_len = meta["seq_len"]
seq_feats = meta["seq_feats"]
ctx_feats = meta["ctx_feats"]
n_buckets = meta["n_buckets"]

INT_INPUTS = ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
              "park_id", "catcher_id"]


# =========================
# 1) Train/val split FIRST (no oversampled duplicates leak into val)
# =========================
print("\nSplitting train/val (80/20, stratified) BEFORE oversampling...")
indices = np.arange(len(y_labels))
idx_tr, idx_val = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y_labels)

print(f"Train: {len(idx_tr):,}  Val: {len(idx_val):,}")


# =========================
# 2) Oversample the TRAINING set only (capped)
# =========================
print("\nOversampling minority classes in the training set...")
y_tr_orig = y_labels[idx_tr]
class_counts = np.bincount(y_tr_orig, minlength=n_buckets)
majority = class_counts.max()

sampling_strategy = {}
for i in range(n_buckets):
    if class_counts[i] == 0:
        continue
    target = min(int(MAX_MINORITY_FRACTION * majority),
                 class_counts[i] * MAX_DUPLICATION)
    sampling_strategy[i] = max(class_counts[i], target)

print("Oversampling targets:")
for i, cls in enumerate(meta["bucket_classes"]):
    print(f"  {cls}: {class_counts[i]:,} -> {sampling_strategy.get(i, 0):,}")

ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
idx_tr_res, y_tr = ros.fit_resample(idx_tr.reshape(-1, 1), y_tr_orig)
idx_tr_res = idx_tr_res.ravel()
print(f"Training set after oversampling: {len(y_tr):,}")


def gather(idx):
    return [
        X_seq[idx], X_ctx[idx],
        X_pitcher_id[idx], X_batter_id[idx],
        X_pitcher_hand[idx], X_batter_hand[idx],
        X_park_id[idx], X_catcher_id[idx],
    ]


train_inputs = gather(idx_tr_res)
val_inputs = gather(idx_val)
y_val = y_labels[idx_val]


# =========================
# 3) Focal loss (proven from 66% run)
# =========================
def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        cross_entropy = -y_true * tf.math.log(y_pred)
        weights = alpha * tf.pow(1 - y_pred, gamma)
        loss = tf.reduce_sum(weights * cross_entropy, axis=-1)
        return tf.reduce_mean(loss)
    return focal_loss_fn


# =========================
# 4) Build model
# =========================
print("\nBuilding model...")

# --- Sequence branch (stacked BiLSTM) ---
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
output = Dense(n_buckets, activation="softmax", name="output")(z)

model = Model(
    inputs=[
        seq_input, ctx_input,
        pitcher_id_input, batter_id_input,
        pitcher_hand_input, batter_hand_input,
        park_id_input, catcher_id_input,
    ],
    outputs=output
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=focal_loss(), metrics=["accuracy"])
model.summary()


# =========================
# 5) Callbacks
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
# 6) Train
# =========================
print(f"\nTraining for up to {EPOCHS} epochs...")
print(f"Focal loss: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")
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
# 7) Evaluate — on the NATURAL validation distribution
# =========================
print("\n" + "=" * 50)
print("EVALUATION (natural distribution, no oversampled duplicates)")
print("=" * 50)

y_pred_probs = model.predict(val_inputs, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_val, y_pred,
    labels=list(range(n_buckets)),
    target_names=meta["bucket_classes"],
    digits=3,
    zero_division=0,
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred, labels=list(range(n_buckets)))
print(pd.DataFrame(cm, index=meta["bucket_classes"], columns=meta["bucket_classes"]))

print("\nPer-class accuracy:")
for i, cls in enumerate(meta["bucket_classes"]):
    mask = y_val == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).mean()
        print(f"  {cls}: {acc:.1%} ({mask.sum():,} samples)")

# Top-2 accuracy — useful for the website ("most likely / second guess")
top2 = np.argsort(y_pred_probs, axis=1)[:, -2:]
top2_acc = np.mean([(y_val[i] in top2[i]) for i in range(len(y_val))])
print(f"\nTop-2 accuracy: {top2_acc:.1%}")


# =========================
# 8) Save training curves + history
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
print(f"Best model: {DATA_DIR / 'best_model_v5.keras'}")
print(f"Final model: {DATA_DIR / 'final_model_v5.keras'}")
