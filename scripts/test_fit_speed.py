import time
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# Build model
def get_model():
    m = keras.Sequential([
        layers.Input(shape=(60, 1)),
        layers.Conv1D(8, 15, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 15, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 15, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 15, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(7, activation='softmax')
    ])
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

X = np.random.randn(320, 60, 1).astype(np.float32)
y = np.random.randint(0, 7, size=(320,)).astype(np.int32)
X_val = np.random.randn(100, 60, 1).astype(np.float32)
y_val = np.random.randint(0, 7, size=(100,)).astype(np.int32)

# Benchmark standard fit with default batch 32
m1 = get_model()
t0 = time.time()
m1.fit(X, y, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)
print(f"Standard batch=32, epochs=100: {time.time()-t0:.2f}s")

# Benchmark with batch_size=len(X) or batch_size=64
m2 = get_model()
t0 = time.time()
m2.fit(X, y, validation_data=(X_val, y_val), epochs=100, batch_size=64, verbose=0)
print(f"Batch=64, epochs=100: {time.time()-t0:.2f}s")
