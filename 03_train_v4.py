"""
NextPitchAI v4 — Step 3: Train model
======================================
Combines the PROVEN training strategy from the 66% run:
  - Focal loss (gamma=2)
  - RandomOverSampler for class balance
  - Stacked BiLSTM
  - EarlyStopping + checkpoints

WITH new v4 features:
  - Pitcher MLBAM ID embeddings (learned representations)
  - Batter MLBAM ID embeddings
  - Richer sequence features (pitch physics)

Requirements:
    pip install tensorflow numpy pandas scikit-learn joblib imbalanced-learn matplotlib

Usage:
    python 03_train_v4.py

Output:
    data_v4/final_model_v4.keras
    data_v4/best_model_v4.keras
    data_v4/training_curves_v4.png
    data_v4/training_history_v4.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense,
    Dropout, Concatenate, Flatten, BatchNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras import backend as K

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_v4"

# Hyperparameters
EPOCHS = 60
BATCH_SIZE = 512
PATIENCE = 8
LEARNING_RATE = 1e-3

# Embedding dimensions
PITCHER_EMBED_DIM = 32   # learned representation per pitcher
BATTER_EMBED_DIM = 16    # learned representation per batter
HAND_EMBED_DIM = 4       # handedness embedding

# Model dimensions
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

# Focal loss params (proven from 66% run)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 1.0

# =========================
# 0) Load metadata + arrays
# =========================
print("Loading data...")
with open(DATA_DIR / "meta_v4.json") as f:
    meta = json.load(f)

X_seq         = np.load(DATA_DIR / "X_seq.npy")
X_ctx         = np.load(DATA_DIR / "X_ctx.npy")
X_pitcher_id  = np.load(DATA_DIR / "X_pitcher_id.npy")
X_batter_id   = np.load(DATA_DIR / "X_batter_id.npy")
X_pitcher_hand= np.load(DATA_DIR / "X_pitcher_hand.npy")
X_batter_hand = np.load(DATA_DIR / "X_batter_hand.npy")
y_labels      = np.load(DATA_DIR / "y_labels.npy")

print(f"Samples: {len(y_labels):,}")
print(f"Seq shape: {X_seq.shape}")
print(f"Ctx shape: {X_ctx.shape}")
print(f"Pitcher IDs: {meta['n_pitcher_ids']}, Batter IDs: {meta['n_batter_ids']}")
print(f"Buckets: {meta['bucket_classes']}")

seq_len = meta["seq_len"]
seq_feats = meta["seq_feats"]
ctx_feats = meta["ctx_feats"]
n_buckets = meta["n_buckets"]
n_pitcher_ids = meta["n_pitcher_ids"]
n_batter_ids = meta["n_batter_ids"]


# =========================
# 1) Oversample rare classes
# =========================
print("\nOversampling minority classes...")

# Flatten all inputs into one array for oversampling
X_combined = np.concatenate([
    X_seq.reshape(len(y_labels), -1),       # (N, seq_len * seq_feats)
    X_ctx,                                   # (N, ctx_feats)
    X_pitcher_id.reshape(-1, 1),            # (N, 1)
    X_batter_id.reshape(-1, 1),             # (N, 1)
    X_pitcher_hand.reshape(-1, 1),          # (N, 1)
    X_batter_hand.reshape(-1, 1),           # (N, 1)
], axis=1)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_combined, y_labels)
print(f"After oversampling: {len(y_resampled):,} samples")

# Print new class distribution
for i, cls in enumerate(meta["bucket_classes"]):
    count = (y_resampled == i).sum()
    print(f"  {cls}: {count:,}")

# Restore individual arrays from the combined matrix
idx = 0
seq_size = seq_len * seq_feats
X_seq_res = X_resampled[:, idx:idx+seq_size].reshape(-1, seq_len, seq_feats).astype(np.float32)
idx += seq_size

X_ctx_res = X_resampled[:, idx:idx+ctx_feats].astype(np.float32)
idx += ctx_feats

X_pitcher_id_res = X_resampled[:, idx].astype(np.int32)
idx += 1

X_batter_id_res = X_resampled[:, idx].astype(np.int32)
idx += 1

X_pitcher_hand_res = X_resampled[:, idx].astype(np.int32)
idx += 1

X_batter_hand_res = X_resampled[:, idx].astype(np.int32)
idx += 1


# =========================
# 2) Train/val split
# =========================
print("\nSplitting train/val (80/20, stratified)...")
(X_seq_tr, X_seq_val,
 X_ctx_tr, X_ctx_val,
 pid_tr, pid_val,
 bid_tr, bid_val,
 ph_tr, ph_val,
 bh_tr, bh_val,
 y_tr, y_val) = train_test_split(
    X_seq_res, X_ctx_res,
    X_pitcher_id_res, X_batter_id_res,
    X_pitcher_hand_res, X_batter_hand_res,
    y_resampled,
    test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"Train: {len(y_tr):,}  Val: {len(y_val):,}")


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

# --- Sequence branch (stacked BiLSTM, same as 66% run) ---
seq_input = Input(shape=(seq_len, seq_feats), name="seq_input")
s = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True))(seq_input)
s = Dropout(DROPOUT_RATE)(s)
s = Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False))(s)
s = Dropout(DROPOUT_RATE)(s)

# --- Context branch ---
ctx_input = Input(shape=(ctx_feats,), name="ctx_input")
c = Dense(64, activation="relu")(ctx_input)
c = BatchNormalization()(c)
c = Dropout(DROPOUT_RATE)(c)

# --- Pitcher ID embedding (NEW in v4) ---
pitcher_id_input = Input(shape=(1,), name="pitcher_id", dtype="int32")
pitcher_emb = Embedding(
    input_dim=n_pitcher_ids,
    output_dim=PITCHER_EMBED_DIM,
    name="pitcher_embedding"
)(pitcher_id_input)
pitcher_emb = Flatten()(pitcher_emb)

# --- Batter ID embedding (NEW in v4) ---
batter_id_input = Input(shape=(1,), name="batter_id", dtype="int32")
batter_emb = Embedding(
    input_dim=n_batter_ids,
    output_dim=BATTER_EMBED_DIM,
    name="batter_embedding"
)(batter_id_input)
batter_emb = Flatten()(batter_emb)

# --- Handedness embeddings ---
pitcher_hand_input = Input(shape=(1,), name="pitcher_hand", dtype="int32")
pitcher_hand_emb = Embedding(input_dim=2, output_dim=HAND_EMBED_DIM)(pitcher_hand_input)
pitcher_hand_emb = Flatten()(pitcher_hand_emb)

batter_hand_input = Input(shape=(1,), name="batter_hand", dtype="int32")
batter_hand_emb = Embedding(input_dim=3, output_dim=HAND_EMBED_DIM)(batter_hand_input)
batter_hand_emb = Flatten()(batter_hand_emb)

# --- Merge everything ---
combined = Concatenate()([
    s,                  # sequence encoding
    c,                  # context
    pitcher_emb,        # who is pitching
    batter_emb,         # who is batting
    pitcher_hand_emb,   # pitcher handedness
    batter_hand_emb,    # batter handedness
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
        pitcher_hand_input, batter_hand_input
    ],
    outputs=output
)

# Compile with focal loss
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=focal_loss(), metrics=["accuracy"])
model.summary()


# =========================
# 5) Callbacks
# =========================
callbacks = [
    ModelCheckpoint(
        str(DATA_DIR / "best_model_v4.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        str(DATA_DIR / "final_model_v4.keras"),
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
print(f"Oversampled: yes")
print(f"Batch size: {BATCH_SIZE}")
print(f"Early stopping patience: {PATIENCE}")

train_inputs = [X_seq_tr, X_ctx_tr, pid_tr, bid_tr, ph_tr, bh_tr]
val_inputs   = [X_seq_val, X_ctx_val, pid_val, bid_val, ph_val, bh_val]

history = model.fit(
    train_inputs, y_tr,
    validation_data=(val_inputs, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# =========================
# 7) Evaluate
# =========================
print("\n" + "="*50)
print("EVALUATION")
print("="*50)

# Predictions on validation set
y_pred_probs = model.predict(val_inputs, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_val, y_pred,
    target_names=meta["bucket_classes"],
    digits=3
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(pd.DataFrame(
    cm,
    index=meta["bucket_classes"],
    columns=meta["bucket_classes"]
))

# Per-class accuracy (critical check: make sure we're not just predicting one class!)
print("\nPer-class accuracy:")
for i, cls in enumerate(meta["bucket_classes"]):
    mask = y_val == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).mean()
        print(f"  {cls}: {acc:.1%} ({mask.sum():,} samples)")


# =========================
# 8) Save training curves
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
plt.savefig(str(DATA_DIR / "training_curves_v4.png"), dpi=300)
plt.show()
print(f"Training curves saved to {DATA_DIR / 'training_curves_v4.png'}")

# Save history
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(DATA_DIR / "training_history_v4.json", "w") as f:
    json.dump(hist_dict, f, indent=2)

print("\nDone!")
print(f"Best model: {DATA_DIR / 'best_model_v4.keras'}")
print(f"Final model: {DATA_DIR / 'final_model_v4.keras'}")
