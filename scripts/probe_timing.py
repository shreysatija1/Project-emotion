import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers

def probe_model(n_samples, n_dim=48, epochs=100, batch_size=32):
    X = np.random.randn(n_samples, n_dim, 1).astype(np.float32)
    y = np.random.randint(0, 7, size=(n_samples,)).astype(np.int32)
    X_val = np.random.randn(100, n_dim, 1).astype(np.float32)
    y_val = np.random.randint(0, 7, size=(100,)).astype(np.int32)
    
    model = keras.Sequential([
        layers.Input(shape=(n_dim, 1)),
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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    t0 = time.time()
    model.fit(X, y, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    elapsed = time.time() - t0
    return elapsed

t_emodb_unaug = probe_model(320)
print(f"EmoDB unaugmented (320 samples, 100 ep): {t_emodb_unaug:.2f}s")
t_emodb_aug = probe_model(1600)
print(f"EmoDB 5x augmented (1600 samples, 100 ep): {t_emodb_aug:.2f}s")
t_cafe_aug = probe_model(3120)
print(f"CAFE 5x augmented (3120 samples, 100 ep): {t_cafe_aug:.2f}s")
