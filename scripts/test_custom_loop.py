import time
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
import keras

X = np.random.randn(320, 60, 1).astype(np.float32)
y = np.random.randint(0, 7, size=(320,)).astype(np.int32)
X_val = np.random.randn(100, 60, 1).astype(np.float32)
y_val = np.random.randint(0, 7, size=(100,)).astype(np.int32)

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.test_fit_speed import get_model

m = get_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
opt = keras.optimizers.Adam(0.001)

train_ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)

@tf.function
def train_step(x_b, y_b):
    with tf.GradientTape() as tape:
        logits = m(x_b, training=True)
        loss = loss_fn(y_b, logits)
    grads = tape.gradient(loss, m.trainable_variables)
    opt.apply_gradients(zip(grads, m.trainable_variables))
    return loss

@tf.function
def val_step(x_b, y_b):
    logits = m(x_b, training=False)
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_b), tf.float32))
    return acc

# Warmup
for xb, yb in train_ds:
    train_step(xb, yb)

t0 = time.time()
best_val = 0.0
for ep in range(100):
    for xb, yb in train_ds:
        train_step(xb, yb)
    # val
    acc = val_step(X_val, y_val)
    if acc > best_val:
        best_val = acc

print(f"Custom tf.function training loop (100 epochs): {time.time()-t0:.2f}s!")
