import time
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import concurrent.futures
import numpy as np

def train_dummy(worker_id):
    import tensorflow as tf
    import keras
    from keras import layers
    
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    X = np.random.randn(320, 60, 1).astype(np.float32)
    y = np.random.randint(0, 7, size=(320,)).astype(np.int32)
    X_val = np.random.randn(100, 60, 1).astype(np.float32)
    y_val = np.random.randint(0, 7, size=(100,)).astype(np.int32)
    
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
    t0 = time.time()
    m.fit(X, y, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)
    dur = time.time() - t0
    return worker_id, dur

if __name__ == "__main__":
    t_start = time.time()
    n_workers = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(train_dummy, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            wid, dur = f.result()
            print(f"Worker {wid} finished in {dur:.2f}s")
    total = time.time() - t_start
    print(f"8 runs with 4 workers finished in {total:.2f}s (throughput: {total/8:.2f}s per run)")
